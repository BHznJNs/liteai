import dataclasses
import enum
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import Annotated, Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel as PydanticBaseModel

from dais_sdk.tool.prepare import _python_type_to_json_schema


class TestPythonTypeToJsonSchema:
    # ------------------------------------------------------------------------
    # 1.1 primitive types
    # ------------------------------------------------------------------------

    def test_primitive_types(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}
        assert _python_type_to_json_schema(int) == {"type": "integer"}
        assert _python_type_to_json_schema(float) == {"type": "number"}
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}
        assert _python_type_to_json_schema(Any) == {"type": "string"}

    # ------------------------------------------------------------------------
    # 1.2 bytes and datetime types
    # ------------------------------------------------------------------------

    def test_bytes_type(self):
        assert _python_type_to_json_schema(bytes) == {"type": "string", "contentEncoding": "base64"}

    def test_datetime_types(self):
        assert _python_type_to_json_schema(datetime) == {"type": "string", "format": "date-time"}
        assert _python_type_to_json_schema(date) == {"type": "string", "format": "date"}
        assert _python_type_to_json_schema(time) == {"type": "string", "format": "time"}

    # ------------------------------------------------------------------------
    # 1.3 container types
    # ------------------------------------------------------------------------

    def test_list_without_type_args(self):
        assert _python_type_to_json_schema(list) == {"type": "array", "items": {"type": "string"}}

    def test_list_with_type(self):
        assert _python_type_to_json_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}
        assert _python_type_to_json_schema(list[str]) == {"type": "array", "items": {"type": "string"}}

    def test_sequence_type(self):
        assert _python_type_to_json_schema(Sequence[str]) == {"type": "array", "items": {"type": "string"}}
        assert _python_type_to_json_schema(Sequence[int]) == {"type": "array", "items": {"type": "integer"}}

    def test_set_types(self):
        assert _python_type_to_json_schema(set[str]) == {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        }
        assert _python_type_to_json_schema(frozenset[int]) == {
            "type": "array",
            "items": {"type": "integer"},
            "uniqueItems": True,
        }

    def test_tuple_variable_length(self):
        assert _python_type_to_json_schema(tuple[int, ...]) == {"type": "array", "items": {"type": "integer"}}

    def test_tuple_fixed_length(self):
        result = _python_type_to_json_schema(tuple[str, int, bool])
        assert result == {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}],
            "minItems": 3,
            "maxItems": 3,
        }

    def test_dict_without_type_args(self):
        assert _python_type_to_json_schema(dict) == {"type": "object", "additionalProperties": {"type": "string"}}

    def test_dict_with_types(self):
        assert _python_type_to_json_schema(dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }
        assert _python_type_to_json_schema(dict[str, str]) == {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }

    def test_mapping_type(self):
        assert _python_type_to_json_schema(Mapping[str, float]) == {
            "type": "object",
            "additionalProperties": {"type": "number"},
        }

    # ------------------------------------------------------------------------
    # 1.6 Union types
    # ------------------------------------------------------------------------

    def test_union_two_types(self):
        result = _python_type_to_json_schema(Union[str, int])
        assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    def test_union_pipe_syntax(self):
        result = _python_type_to_json_schema(str | int)
        assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    def test_optional_type(self):
        assert _python_type_to_json_schema(Optional[str]) == {"oneOf": [{"type": "string"}, {"type": "null"}]}
        assert _python_type_to_json_schema(Optional[int]) == {"oneOf": [{"type": "integer"}, {"type": "null"}]}

    def test_union_with_none_only(self):
        result = _python_type_to_json_schema(type(None))
        assert result == {"type": "string"}

    def test_union_multiple_types(self):
        result = _python_type_to_json_schema(Union[str, int, float])
        assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}, {"type": "number"}]}

    # ------------------------------------------------------------------------
    # 1.7 Literal types
    # ------------------------------------------------------------------------

    def test_literal_bool(self):
        assert _python_type_to_json_schema(Literal[True, False]) == {"type": "boolean", "enum": [True, False]}

    def test_literal_string(self):
        assert _python_type_to_json_schema(Literal["a", "b", "c"]) == {"type": "string", "enum": ["a", "b", "c"]}

    def test_literal_integer(self):
        assert _python_type_to_json_schema(Literal[1, 2, 3]) == {"type": "integer", "enum": [1, 2, 3]}

    def test_literal_mixed_number(self):
        assert _python_type_to_json_schema(Literal[1, 2.5, 3]) == {"type": "number", "enum": [1, 2.5, 3]}

    # ------------------------------------------------------------------------
    # 1.8 Enum types
    # ------------------------------------------------------------------------

    def test_string_enum(self):
        class Color(enum.Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        assert _python_type_to_json_schema(Color) == {"type": "string", "enum": ["red", "green", "blue"]}

    def test_integer_enum(self):
        class Status(enum.Enum):
            PENDING = 0
            ACTIVE = 1
            COMPLETED = 2

        assert _python_type_to_json_schema(Status) == {"type": "integer", "enum": [0, 1, 2]}

    def test_mixed_number_enum(self):
        class Priority(enum.Enum):
            LOW = 1
            MEDIUM = 2.5
            HIGH = 5

        assert _python_type_to_json_schema(Priority) == {"type": "number", "enum": [1, 2.5, 5]}

    def test_boolean_enum(self):
        class Toggle(enum.Enum):
            ON = True
            OFF = False

        assert _python_type_to_json_schema(Toggle) == {"type": "boolean", "enum": [True, False]}

    # ------------------------------------------------------------------------
    # 1.9 TypedDict tests
    # ------------------------------------------------------------------------

    def test_typeddict_all_required(self):
        from typing import TypedDict

        class Person(TypedDict):
            name: str
            age: int

        result = _python_type_to_json_schema(Person)
        assert result == {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

    def test_typeddict_partial_required(self):
        from typing import TypedDict

        class User(TypedDict, total=False):
            username: str
            email: str

        result = _python_type_to_json_schema(User)
        assert result["type"] == "object"
        assert "username" in result["properties"]
        assert "email" in result["properties"]
        assert "required" not in result or result.get("required") == []

    # ------------------------------------------------------------------------
    # 1.10 dataclass tests
    # ------------------------------------------------------------------------

    def test_dataclass_with_defaults(self):
        @dataclasses.dataclass
        class Config:
            host: str
            port: int = 8080
            debug: bool = False

        result = _python_type_to_json_schema(Config)
        assert result == {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "debug": {"type": "boolean"},
            },
            "required": ["host"],
        }

    def test_dataclass_with_factory(self):
        @dataclasses.dataclass
        class Container:
            name: str
            items: list[str] = dataclasses.field(default_factory=list)

        result = _python_type_to_json_schema(Container)
        assert "name" in result["required"]
        assert "items" not in result.get("required", [])
        assert result["properties"]["items"] == {"type": "array", "items": {"type": "string"}}

    # ------------------------------------------------------------------------
    # 1.11 Pydantic model tests
    # ------------------------------------------------------------------------

    def test_pydantic_model(self):
        class User(PydanticBaseModel):
            username: str
            email: str
            age: int = 18

        result = _python_type_to_json_schema(User)
        assert result["type"] == "object"
        assert "username" in result["properties"]
        assert "email" in result["properties"]
        assert "age" in result["properties"]
        assert "username" in result["required"]
        assert "email" in result["required"]
        assert "age" not in result["required"]

    # ------------------------------------------------------------------------
    # 1.12 Annotated type tests
    # ------------------------------------------------------------------------

    def test_annotated_type(self):
        assert _python_type_to_json_schema(Annotated[str, "metadata"]) == {"type": "string"}
        assert _python_type_to_json_schema(Annotated[int, "positive"]) == {"type": "integer"}
        assert _python_type_to_json_schema(Annotated[list[str], "non-empty"]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    # ------------------------------------------------------------------------
    # 1.13 Nested types
    # ------------------------------------------------------------------------

    def test_nested_list_dict(self):
        result = _python_type_to_json_schema(list[dict[str, int]])
        assert result == {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "integer"}}}

    def test_complex_nested_structure(self):
        result = _python_type_to_json_schema(dict[str, list[tuple[int, str]]])
        assert result["type"] == "object"
        assert result["additionalProperties"]["type"] == "array"
        assert result["additionalProperties"]["items"]["type"] == "array"
        assert result["additionalProperties"]["items"]["prefixItems"] == [
            {"type": "integer"},
            {"type": "string"},
        ]

    def test_nested_optional(self):
        result = _python_type_to_json_schema(list[str | None])
        assert result == {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "null"}]}}
