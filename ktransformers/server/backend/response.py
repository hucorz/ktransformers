import os
import json
import torch
from transformers import DynamicCache
from .utils.utils import parse_user_format_fields, data_dumps
from .utils.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    USER_PROMPT_SUFFIX,
    USER_PROMPT_SUFFIX_WITH_DATA,
)
from .utils.cache import merge_kv_caches
from .trie import build_trie_from_format
from ktransformers.server.config.log import logger


def is_bool_token(token, tokenizer):
    text = tokenizer.decode(token)
    return text.strip().lower() in ["true", "false"], text


def extend_input_ids(tokenizer, cur_generate_token, finish_fields_cnt, user_format, data_cnt):
    format_keys = list(user_format.keys())
    fields_cnt = len(format_keys)
    cur_data_idx = int(finish_fields_cnt // fields_cnt)
    cur_field_idx = int(finish_fields_cnt % fields_cnt)
    cur_field = format_keys[cur_field_idx]
    last_token_id = cur_generate_token[-1]

    extend_str = ""
    cur_generate_str = tokenizer.decode(cur_generate_token)
    logger.info(f"cur_generate_str:{cur_generate_str}, cur_field: {user_format[cur_field]}")
    if user_format[cur_field]["type"] != "Literal":
        # not Literal type
        if "\n" not in tokenizer.decode([last_token_id]):
            return False, None, None  # continue decode
    else:
        # Literal type or bool type
        target_category = user_format[cur_field]["trie"].search_unique_category(cur_generate_str)
        if not target_category:
            return False, None, None  # continue decode

        logger.info(f"cur_generate_str:{cur_generate_str}, trie: {target_category}")
        extend_str += target_category[len(cur_generate_str.strip()) :] + '"'
        if cur_field_idx < fields_cnt - 1:
            extend_str += ",\n"
        else:
            extend_str += "\n"

    input_ids = torch.tensor(tokenizer.encode(extend_str)).unsqueeze(0)

    format_content = ""
    if cur_field_idx < fields_cnt - 1:
        format_content = f'\t\t"{format_keys[cur_field_idx + 1]}":'
    elif cur_data_idx < data_cnt - 1:
        format_content = f'\t}},\n\t{{\n\t\t"{format_keys[0]}":'
    else:
        format_content = f"\t}}\n]\n[[## COMPLETE ##]]"
    format_content_ids = tokenizer.encode(
        format_content, return_tensors="pt", add_special_tokens=False
    )
    input_ids = torch.cat([input_ids, format_content_ids], dim=1)
    extend_str += format_content

    return True, input_ids, extend_str


def _generate_turbo(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    kv_cache: tuple,
    user_format: dict,
    data_cnt: int = 4,
    max_new_tokens: int = 200,
    use_turbo: bool = True,
):
    response = []
    cur_generate_token = []

    input_ids = input_ids.to(model.device)
    past_key_values = DynamicCache.from_legacy_cache(kv_cache)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1))
            if next_token_id == tokenizer.eos_token_id:
                break
            if use_turbo:
                cur_generate_token.append(next_token_id)
                field_finish, input_ids, generate_str = extend_input_ids(
                    tokenizer, cur_generate_token, len(response), user_format, data_cnt
                )
                if not field_finish:
                    input_ids = torch.tensor([next_token_id]).unsqueeze(0)
                else:
                    response.append(generate_str)
            else:
                input_ids = torch.tensor([next_token_id]).unsqueeze(0)
                # if "\n" in tokenizer.decode(next_token_id) or "\t" in tokenizer.decode(next_token_id):
                #     print(repr(tokenizer.decode(next_token_id)))
                response.append(tokenizer.decode(next_token_id))
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
    return "".join(response)


def generate_turbo(
    model,
    tokenizer,
    input_str: str,
    user_format: str,
    kv_cache: tuple = None,
    data_cnt: int = 4,
    max_new_tokens: int = 200,
    add_generation_prompt: bool = False,
):
    """
    - case1: without cache
        input_str need to contain both system prompt and user prompt
    - case2: with cache (only system prompt cache, exclude data info)
        the input_str need to contain the data info, like:
        ==================
        [[## DATA ##]]
        {data_entries}
        [[## QUERY ##]]
        "{query}"
        [[## FORMAT ##]]
        {output_format}
        ==================
    - case3: with cache (both system prompt cache and data info)
        the input_str is like:
        ==================
        [[## QUERY ##]]
        "{query}"
        [[## FORMAT ##]]
        {output_format}
        ==================

    """
    user_format = parse_user_format_fields(user_format)

    for k, v in user_format.items():
        if v["type"] == "Literal":
            user_format[k]["trie"] = build_trie_from_format(v["values"])
        elif v["type"] == "bool":
            user_format[k]["trie"] = build_trie_from_format("Literal[true, false]")
        else:
            user_format[k]["trie"] = None

    if add_generation_prompt:
        input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    pre_extend_text = f'[[## RESULT ##]]\n[\n\t{{\n\t\t"{list(user_format.keys())[0]}":'
    input_str += pre_extend_text
    inputs = tokenizer(input_str, return_tensors="pt").to(model.device)
    return pre_extend_text + _generate_turbo(
        model,
        tokenizer,
        inputs["input_ids"],
        kv_cache,
        user_format,
        data_cnt=data_cnt,
        max_new_tokens=max_new_tokens,
    )


def response_normal(
    model, tokenizer, data: list[dict], query: str, user_format: str, max_new_tokens: int = 200
):
    data_entries = data_dumps(data)

    user_prompt = USER_PROMPT.format(
        data_entries=data_entries, data_length=len(data), query=query, output_format=user_format
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            top_p=None,
            top_k=None,
            temperature=None,
        )
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[:, input_length:]
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


def response_turbo_without_cache(
    model, tokenizer, data: list[dict], query: str, user_format: str, max_new_tokens: int = 200
):
    data_entries = data_dumps(data)

    user_prompt = USER_PROMPT.format(
        data_entries=data_entries, data_length=len(data), query=query, output_format=user_format
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate_turbo(
        model,
        tokenizer,
        input_str,
        user_format,
        None,
        data_cnt=len(data),
        max_new_tokens=max_new_tokens,
    )


def response_turbo_with_system_cache(
    model,
    tokenizer,
    data: list[dict],
    query: str,
    user_format: str,
    system_cache: tuple,
    max_new_tokens: int = 200,
):
    """
    cache content may like:
    <|im_start|>system
    ....
    <|im_end|>
    <|im_start|>user
    [[## DATA ##]]
    """
    data_entries = data_dumps(data)

    input_str = USER_PROMPT_SUFFIX_WITH_DATA.format(
        data_entries=data_entries, data_length=len(data), query=query, output_format=user_format
    )
    input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    return generate_turbo(
        model,
        tokenizer,
        input_str,
        user_format,
        system_cache,
        data_cnt=len(data),
        max_new_tokens=max_new_tokens,
    )


def response_turbo_with_all_cache(
    model,
    tokenizer,
    query: str,
    user_format: str,
    system_cache: tuple,
    data_cache: tuple,
    data_cnt: int = 4,
    max_new_tokens: int = 200,
):
    """
    kv_cache content may like:
    <|im_start|>system
    ....
    <|im_end|>
    <|im_start|>user
    [[## DATA ##]]
    ...
    ...
    """

    user_prompt = USER_PROMPT_SUFFIX.format(
        data_length=data_cnt, query=query, output_format=user_format
    )

    input_str = "\n" + user_prompt + f"<|im_end|>\n<|im_start|>assistant\n"
    return generate_turbo(
        model,
        tokenizer,
        input_str,
        user_format,
        merge_kv_caches([system_cache, data_cache]),
        data_cnt=data_cnt,
        max_new_tokens=max_new_tokens,
    )
