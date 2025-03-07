import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import DynamicCache
from .utils import load_data
from .prompts import SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_SUFFIX, USER_PROMPT_SUFFIX_WITH_DATA


def save_cache(cache: tuple, cache_dir: str, cache_name: str):
    torch.save(cache, os.path.join(cache_dir, cache_name))


def get_kv_cache_slice(kv_cache, st, ed):
    """
    kv_cache shape: Tuple(layer, 2) +  Tensor(batch_size, num_heads, seq_len, head_dim)
    """
    res = []
    for i in range(len(kv_cache)):
        k = kv_cache[i][0]
        v = kv_cache[i][1]
        k_slice = k[:, :, st:ed, :].detach()
        v_slice = v[:, :, st:ed, :].detach()
        res.append((k_slice, v_slice))
    return res


def merge_kv_caches(cache_list):
    if not cache_list:
        return []

    merged = []
    num_layers = len(cache_list[0])

    for layer_idx in range(num_layers):
        k_list = []
        v_list = []
        for cache in cache_list:
            k, v = cache[layer_idx]
            k_list.append(k)
            v_list.append(v)

        merged_k = torch.cat(k_list, dim=2)
        merged_v = torch.cat(v_list, dim=2)
        merged.append((merged_k, merged_v))

    return merged


def _sys_cache_prep(model, tokenizer, cache_path: str = None):
    assert cache_path is not None, "The cache_dir must be specified"
    system_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n<|im_start|>user\n[[## DATA ##]]\n"
    )
    system_inputs = tokenizer(system_prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    output = model(**system_inputs)
    system_cache = output.past_key_values
    torch.save(system_cache, cache_path)
    return system_cache


def _cache_prep(
    model,
    tokenizer,
    data_path: str,
    fields: list[str],
    stride: int = 4,
    cache_dir: str = None,
    system_cache: tuple = None,
):
    assert cache_dir is not None, "The cache_dir must be specified"

    data = load_data(data_path, fields)

    meta = {
        "fields": fields,
        "stride": stride,
        "data_path": data_path,
        "cache_dir": cache_dir,
        "system_length": None,
        "data_length": {},
    }

    meta["system_length"] = system_cache[0][0].shape[2]

    def data_dumps_list(data: list[dict]):
        data_entries = []
        for i, entry in enumerate(data):
            entry = json.dumps(entry, ensure_ascii=False)
            data_entries.append(f"Data {i+1}: {entry}\n")
        return data_entries

    # process data cache
    for start_idx in tqdm(range(0, len(data), stride)):
        data_entries_list = data_dumps_list(data[start_idx : start_idx + stride])
        token_ids = []
        for idx, entry in enumerate(data_entries_list):
            entry_ids = tokenizer.encode(entry, add_special_tokens=False)
            meta["data_length"][start_idx + idx] = len(entry_ids)
            token_ids.extend(entry_ids)

        token_ids = torch.tensor(token_ids, device=model.device).unsqueeze(0)
        try:
            past_key_values = DynamicCache.from_legacy_cache(system_cache)

            with torch.inference_mode():
                output = model(token_ids, past_key_values=past_key_values, use_cache=True)

            # remove system cache from past_key_values
            data_cache = get_kv_cache_slice(
                output.past_key_values, system_cache[0][0].shape[2], None
            )

            save_cache(data_cache, cache_dir, f"data_cache_stride{stride}_st{start_idx}.pt")

            del output
            del past_key_values
            del data_cache

        finally:
            del token_ids

    # 保存元数据
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
