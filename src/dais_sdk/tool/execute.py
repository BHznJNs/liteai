import json
import inspect
import xml.etree.ElementTree as ET
from dataclasses import asdict, is_dataclass
from functools import singledispatch
from typing import Any, Callable, NamedTuple, assert_never, cast
from types import FunctionType, MethodType
from pydantic import BaseModel
from .types import ToolDef, ToolLike
from ..types import ContentBlock, TextBlock, ImageBlock, DocumentBlock, AudioBlock, VideoBlock


def _arguments_normalizer(arguments: str | dict) -> dict:
    if isinstance(arguments, str):
        if len(arguments.strip()) == 0:
            return {}
        parsed = json.loads(arguments)
        return cast(dict, parsed)
    elif isinstance(arguments, dict):
        return arguments
    else:
        assert_never(arguments)

def _result_normalizer(result: Any) -> str | list[ContentBlock]:
    match result:
        case str(): return result
        case list() if all(isinstance(item, TextBlock | ImageBlock | DocumentBlock | AudioBlock | VideoBlock) for item in result):
            return result
        case ET.Element():
            return ET.tostring(result, encoding="unicode", method="xml")
        case _ if is_dataclass(result) and not isinstance(result, type):
            result = asdict(result)
        case BaseModel():
            result = result.model_dump()
        case _: pass
    return json.dumps(result, ensure_ascii=False)

class ToolResult(NamedTuple):
    serialized: str | list[ContentBlock]
    raw: Any

@singledispatch
async def execute_tool(tool: ToolLike, arguments: str | dict) -> ToolResult:
    """
    Raises:
        ValueError: If the tool type is not supported.
        JSONDecodeError: If the arguments is a string but not valid JSON.
    """
    raise ValueError(f"Invalid tool type: {type(tool)}")

@execute_tool.register(FunctionType)
@execute_tool.register(MethodType)
async def _(toolfn: Callable, arguments: str | dict) -> ToolResult:
    arguments = _arguments_normalizer(arguments)
    result = (await toolfn(**arguments)
             if inspect.iscoroutinefunction(toolfn)
             else toolfn(**arguments))
    return ToolResult(_result_normalizer(result), result)

@execute_tool.register(ToolDef)
async def _(tooldef: ToolDef, arguments: str | dict) -> ToolResult:
    arguments = _arguments_normalizer(arguments)
    result = (await tooldef.execute(**arguments)
             if inspect.iscoroutinefunction(tooldef.execute)
             else tooldef.execute(**arguments))
    return ToolResult(_result_normalizer(result), result)
