import re
import json
import pandas as pd
import torch
import ast
from ktransformers.server.config.log import logger


def parse_user_format_fields(user_format: str):
    # 解析 format，比如 USER_FORMAT = r"""list[{relatedAI: bool}, {relatedMath: bool}]"""

    user_format = user_format.strip()

    # 确认最外层必须要是一个 list
    # assert user_format.lower().startswith("list["), "The user format must start with 'list['"
    # assert user_format.endswith("]"), "The user format must end with ']'"

    # 去掉最外层的 list, 解析出每个字段, key 是字段名，value 是字段类型
    # user_format = user_format[5:-1]
    fields = {}
    # for field in user_format.split(","):
    #     field = field.strip()
    #     field = field[1:-1] if field.startswith("{") and field.endswith("}") else field
    #     field_name, field_type = field.split(":")
    #     fields[field_name.strip()] = {"type": field_type.strip()}
    matches = re.finditer(r"\{([^:]+):\s*([^}]+)\}", user_format)
    for match in matches:
        field_name = match.group(1).strip()
        field_type = match.group(2).strip()
        # logger.info(
        #     f"parse_user_format_fields\nfield_name: {field_name}\nfield_type: {field_type}\n"
        # )
        fields[field_name] = {"type": field_type}

        if field_type.startswith("Literal[") and field_type.endswith("]"):
            try:
                literal_values = ast.literal_eval(field_type[len("Literal") :].strip())
                field_type = {"type": "Literal", "values": literal_values}
            except Exception as e:
                logger.error(f"Error parsing Literal: {e}")
                field_type = {"type": field_type}
        else:
            field_type = {"type": field_type}

        fields[field_name] = field_type
    return fields


def data_dumps(data: list[dict]):
    data_entries = json.dumps(data[0], ensure_ascii=False)
    return data_entries


def load_data(data_path: str, fields: list[str] = None):
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".json"):
        data = pd.read_json(data_path)
    if fields is not None:
        data = data[fields]
    return data.to_dict(orient="records")
