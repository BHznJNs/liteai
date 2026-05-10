"""
Modified from: https://github.com/mozilla-ai/any-llm/blob/main/src/any_llm/tools.py
"""

import dataclasses
import enum
import inspect
import types as _types
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import (Annotated as _Annotated, Literal as _Literal,
                    NotRequired as _NotRequired, Required as _Required,
                    TypeAliasType as _TypeAliasType,
                    is_typeddict as _is_typeddict, Any, get_args,
                    get_origin, get_type_hints)
from pydantic import BaseModel as PydanticBaseModel
from .types import ToolFn, ToolDef, RawToolDef, ToolLike, ToolSchema

def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert Python type annotation to a JSON Schema for a parameter.

    Supported mappings (subset tailored for LLM tool schemas):
    - Primitives: str/int/float/bool -> string/integer/number/boolean
    - bytes -> string with contentEncoding base64
    - datetime/date/time -> string with format date-time/date/time
    - list[T] / Sequence[T] / set[T] / frozenset[T] -> array with items=schema(T)
      - set/frozenset include uniqueItems=true
      - list without type args defaults items to string
    - dict[K,V] / Mapping[K,V] -> object with additionalProperties=schema(V)
      - dict without type args defaults additionalProperties to string
    - tuple[T1, T2, ...] -> array with prefixItems per element and min/maxItems
    - tuple[T, ...] -> array with items=schema(T)
    - Union[X, Y] and X | Y -> oneOf=[schema(X), schema(Y)] (without top-level type)
    - Optional[T] (Union[T, None]) -> oneOf=[schema(T), {"type": "null"}]
    - Literal[...]/Enum -> enum with appropriate type inference when uniform
    - TypedDict -> object with properties/required per annotations
    - dataclass/Pydantic BaseModel -> object with nested properties inferred from fields
    """
    origin = get_origin(python_type)
    args = get_args(python_type)

    if _Annotated is not None and origin is _Annotated and len(args) >= 1:
        python_type = args[0]
        origin = get_origin(python_type)
        args = get_args(python_type)

    if isinstance(python_type, _TypeAliasType):
        python_type = python_type.__value__
        origin = get_origin(python_type)
        args = get_args(python_type)

    if origin in (_NotRequired, _Required) and len(args) == 1:
        python_type = args[0]
        origin = get_origin(python_type)
        args = get_args(python_type)

    if python_type is Any:
        return {"type": "string"}

    primitive_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if python_type in primitive_map:
        return {"type": primitive_map[python_type]}

    if python_type is bytes:
        return {"type": "string", "contentEncoding": "base64"}
    if python_type is datetime:
        return {"type": "string", "format": "date-time"}
    if python_type is date:
        return {"type": "string", "format": "date"}
    if python_type is time:
        return {"type": "string", "format": "time"}

    if python_type is list:
        return {"type": "array", "items": {"type": "string"}}
    if python_type is dict:
        return {"type": "object", "additionalProperties": {"type": "string"}}

    if origin is _Literal:
        literal_values = list(args)
        schema_lit: dict[str, Any] = {"enum": literal_values}
        if all(isinstance(v, bool) for v in literal_values):
            schema_lit["type"] = "boolean"
        elif all(isinstance(v, str) for v in literal_values):
            schema_lit["type"] = "string"
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in literal_values):
            schema_lit["type"] = "integer"
        elif all(isinstance(v, int | float) and not isinstance(v, bool) for v in literal_values):
            schema_lit["type"] = "number"
        return schema_lit

    if inspect.isclass(python_type) and issubclass(python_type, enum.Enum):
        enum_values = [e.value for e in python_type]
        value_types = {type(v) for v in enum_values}
        schema: dict[str, Any] = {"enum": enum_values}
        if value_types == {str}:
            schema["type"] = "string"
        elif value_types == {int}:
            schema["type"] = "integer"
        elif value_types <= {int, float}:
            schema["type"] = "number"
        elif value_types == {bool}:
            schema["type"] = "boolean"
        return schema

    if _is_typeddict(python_type):
        annotations: dict[str, Any] = getattr(python_type, "__annotations__", {}) or {}
        required_keys = set(getattr(python_type, "__required_keys__", set()))
        td_properties: dict[str, Any] = {}
        td_required: list[str] = []
        for field_name, field_type in annotations.items():
            td_properties[field_name] = _python_type_to_json_schema(field_type)
            if field_name in required_keys:
                td_required.append(field_name)
        schema_td: dict[str, Any] = {
            "type": "object",
            "properties": td_properties,
        }
        if td_required:
            schema_td["required"] = td_required
        return schema_td

    if inspect.isclass(python_type) and dataclasses.is_dataclass(python_type):
        type_hints = get_type_hints(python_type)
        dc_properties: dict[str, Any] = {}
        dc_required: list[str] = []
        for field in dataclasses.fields(python_type):
            field_type = type_hints.get(field.name, Any)
            dc_properties[field.name] = _python_type_to_json_schema(field_type)
            if (
                field.default is dataclasses.MISSING
                and getattr(field, "default_factory", dataclasses.MISSING) is dataclasses.MISSING
            ):
                dc_required.append(field.name)
        schema_dc: dict[str, Any] = {"type": "object", "properties": dc_properties}
        if dc_required:
            schema_dc["required"] = dc_required
        return schema_dc

    if inspect.isclass(python_type) and issubclass(python_type, PydanticBaseModel):
        model_type_hints = get_type_hints(python_type)
        pd_properties: dict[str, Any] = {}
        pd_required: list[str] = []
        model_fields = getattr(python_type, "model_fields", {})
        for name, field_info in model_fields.items():
            pd_properties[name] = _python_type_to_json_schema(model_type_hints.get(name, Any))
            is_required = getattr(field_info, "is_required", None)
            if callable(is_required) and is_required():
                pd_required.append(name)
        schema_pd: dict[str, Any] = {"type": "object", "properties": pd_properties}
        if pd_required:
            schema_pd["required"] = pd_required
        return schema_pd

    if origin in (list, Sequence, set, frozenset):
        item_type = args[0] if args else Any
        item_schema = _python_type_to_json_schema(item_type)
        schema_arr: dict[str, Any] = {"type": "array", "items": item_schema or {"type": "string"}}
        if origin in (set, frozenset):
            schema_arr["uniqueItems"] = True
        return schema_arr
    if origin is tuple:
        if not args:
            return {"type": "array", "items": {"type": "string"}}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        prefix_items = [_python_type_to_json_schema(a) for a in args]
        return {
            "type": "array",
            "prefixItems": prefix_items,
            "minItems": len(prefix_items),
            "maxItems": len(prefix_items),
        }

    if origin in (dict, Mapping):
        value_type = args[1] if len(args) >= 2 else Any
        value_schema = _python_type_to_json_schema(value_type)
        return {"type": "object", "additionalProperties": value_schema or {"type": "string"}}

    typing_union = getattr(__import__("typing"), "Union", None)
    if origin in (typing_union, _types.UnionType):
        non_none_args = [a for a in args if a is not type(None)]
        has_none = len(non_none_args) < len(args)

        if len(non_none_args) > 1:
            schemas = [_python_type_to_json_schema(arg) for arg in non_none_args]
            if has_none:
                schemas.append({"type": "null"})
            return {"oneOf": schemas}

        if non_none_args:
            inner = _python_type_to_json_schema(non_none_args[0])
            if has_none:
                # if inner is `oneOf`, append null to it and prevent nested oneOf
                if "oneOf" in inner:
                    inner["oneOf"].append({"type": "null"})
                    return inner
                return {"oneOf": [inner, {"type": "null"}]}
            return inner

        return {"type": "null"} # for `Union[None]`

    return {"type": "string"}

def _parse_description(python_type: Any) -> str | None:
    origin = get_origin(python_type)
    if origin is not _Annotated: return None

    args = get_args(python_type)
    metadata = args[1:]

    if len(metadata) == 0: return None
    if not isinstance(metadata[0], str): return None
    return inspect.cleandoc(metadata[0])

def _parse_callable_properties(func: ToolFn) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    annotated_type_hints = get_type_hints(func, include_extras=True)

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        type_hint = type_hints.get(param_name, str)
        param_schema = _python_type_to_json_schema(type_hint)

        annotated_type = annotated_type_hints.get(param_name)
        param_description = _parse_description(annotated_type)

        type_name = getattr(type_hint, "__name__", str(type_hint))
        properties[param_name] = {
            **param_schema,
            "description": param_description or f"Parameter {param_name} of type {type_name}",
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default

    return properties, required

def generate_tool_definition_from_callable(func: ToolFn) -> ToolSchema:
    """Convert a Python callable to OpenAI tools format.

    Args:
        func: A Python callable (function) to convert to a tool

    Returns:
        Dictionary in OpenAI tools format

    Raises:
        ValueError: If the function doesn't have proper docstring or type annotations

    Example:
        ```
        >>> def get_weather(location: str, unit: str = "celsius") -> str:
        ...     '''Get weather information for a location.'''
        ...     return f"Weather in {location} is sunny, 25°{unit[0].upper()}"
        >>>
        >>> tool = generate_tool_definition_from_callable(get_weather)
        >>> # Returns OpenAI tools format dict
        ```

    """
    if not func.__doc__:
        msg = f"Function {func.__name__} must have a docstring"
        raise ValueError(msg)

    properties, required = _parse_callable_properties(func)
    return ToolSchema(
        name=func.__name__,
        description=inspect.cleandoc(func.__doc__),
        parameters={"type": "object", "properties": properties, "required": required},
    )

def generate_tool_definition_from_tool_def(tool_def: ToolDef) -> ToolSchema:
    """Convert a ToolDef to OpenAI tools format.

    Args:
        tool_def: A ToolDef to convert to a tool

    Returns:
        Dictionary in OpenAI tools format

    Example:
        >>> tool_def = ToolDef(
        ...     name="get_weather",
        ...     description="Get weather information for a location.",
        ...     execute=SomeFunction(),
        ... )
        >>> tool = generate_tool_definition_from_tool_def(tool_def)
        >>> # Returns OpenAI tools format dict
    """
    properties, required = _parse_callable_properties(tool_def.execute)
    return ToolSchema(
        name=tool_def.name,
        description=tool_def.description,
        parameters=tool_def.parameters or {"type": "object", "properties": properties, "required": required},
    )

def generate_tool_definition_from_raw_tool_def(raw_tool_def: RawToolDef) -> ToolSchema:
    return raw_tool_def

def prepare_tools(tools: Sequence[ToolLike]) -> list[ToolSchema]:
    tool_likes = []
    for tool in tools:
        if callable(tool):
            tool_likes.append(generate_tool_definition_from_callable(tool))
        elif isinstance(tool, ToolDef):
            tool_likes.append(generate_tool_definition_from_tool_def(tool))
        else:
            tool_likes.append(generate_tool_definition_from_raw_tool_def(tool))
    return tool_likes
