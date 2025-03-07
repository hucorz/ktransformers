import json
import re
import traceback
from time import time
from uuid import uuid4
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import (
    ChatCompletionCreate,
    ChatCompletionChunk,
    ChatCompletionObject,
    CachePrepCreate,
    DataQueryCreate,
)
from ktransformers.server.backend.base import BackendInterfaceBase

from ktransformers.server.config.log import logger

router = APIRouter()

models = [
    {"id": "0", "name": "ktranformers-model"},
]


@router.get("/models", tags=["openai"])
async def list_models():
    return models


@router.post("/chat/completions", tags=["openai"])
async def chat_completion(request: Request, create: ChatCompletionCreate):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()
    # input_ids = interface.format_and_tokenize_input_ids(id,messages=create.get_tokenizer_messages())

    input_message = [json.loads(m.model_dump_json()) for m in create.messages]

    if create.stream:

        async def inner():
            chunk = ChatCompletionChunk(id=id, object="chat.completion.chunk", created=int(time()))
            async for token in interface.inference(input_message, id):
                chunk.set_token(token)
                yield chunk

        return chat_stream_response(request, inner())
    else:
        comp = ChatCompletionObject(id=id, object="chat.completion.chunk", created=int(time()))
        async for token in interface.inference(input_message, id):
            comp.append_token(token)
        return comp


@router.post("/cache/prep", tags=["openai"])
def cache_prep(request: Request, create: CachePrepCreate):
    interface: BackendInterfaceBase = get_interface()
    data_path = create.data_path
    stride = create.stride
    fields = create.fields
    force_prep = create.force_prep

    try:
        interface.cache_prep(
            data_path=data_path, stride=stride, fields=fields, force_prep=force_prep
        )
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception as e:
        stack_info = traceback.format_exc()
        logger.error(stack_info)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/data/query", tags=["openai"])
def data_query(request: Request, create: DataQueryCreate):
    interface: BackendInterfaceBase = get_interface()

    try:
        response_list = interface.data_query(**create.dict())
        response_list = [
            re.search(r"\[\[## RESULT ##\]\](.*)\[\[## COMPLETE ##\]\]", m, re.DOTALL)
            .group(1)
            .strip()
            for m in response_list
        ]
        response_list = fix_common_response_error(response_list, valid=True)
        response_list = [json.loads(m) for m in response_list]
        response_list = [item for sub in response_list for item in sub]
        return JSONResponse(status_code=200, content={"status": "ok", "result": response_list})
    except Exception as e:
        stack_info = traceback.format_exc()
        logger.error(stack_info)
        return JSONResponse(status_code=500, content={"error": str(e)})


def fix_common_response_error(response_list: list, valid: bool = False):
    for idx, response in enumerate(response_list):
        response = re.sub(r": False", r": false", response)
        response = re.sub(r": True", r": true", response)
        response_list[idx] = response
        if valid:
            try:
                json.loads(response)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.error(f"\nResponse{idx}: {response}\n{stack_info}")
    return response_list
