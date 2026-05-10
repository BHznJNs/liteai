import dataclasses
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel

from dais_sdk.types import ToolDef
from dais_sdk.tool.prepare import prepare_tools, _python_type_to_json_schema


class _NestedTypedDict(TypedDict):
    flag: bool
    tags: NotRequired[list[str]]


@dataclasses.dataclass
class _NestedDataclass:
    active: bool
    score: float | None = None


class _NestedPydanticModel(BaseModel):
    name: str
    count: int | None = None


class TestPythonTypeToJsonSchema:
    def test_optional_type(self):
        assert _python_type_to_json_schema(str | None) == {
            "oneOf": [{"type": "string"}, {"type": "null"}]
        }

    def test_union_type(self):
        assert _python_type_to_json_schema(str | int | None) == {
            "oneOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ]
        }

    def test_literal_type(self):
        assert _python_type_to_json_schema(Literal["open", "closed"]) == {
            "enum": ["open", "closed"],
            "type": "string",
        }

    def test_nested_typed_dict(self):
        class Payload(TypedDict):
            meta: _NestedTypedDict
            title: NotRequired[str]

        assert _python_type_to_json_schema(Payload) == {
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {
                        "flag": {"type": "boolean"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["flag"],
                },
                "title": {"type": "string"},
            },
            "required": ["meta"],
        }

    def test_nested_dataclass(self):
        @dataclasses.dataclass
        class Payload:
            meta: _NestedDataclass
            label: str = "default"

        assert _python_type_to_json_schema(Payload) == {
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {
                        "active": {"type": "boolean"},
                        "score": {
                            "oneOf": [{"type": "number"}, {"type": "null"}]
                        },
                    },
                    "required": ["active"],
                },
                "label": {"type": "string"},
            },
            "required": ["meta"],
        }

    def test_nested_pydantic_model(self):
        class Payload(BaseModel):
            meta: _NestedPydanticModel
            enabled: bool = True

        assert _python_type_to_json_schema(Payload) == {
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {
                            "oneOf": [{"type": "integer"}, {"type": "null"}]
                        },
                    },
                    "required": ["name"],
                },
                "enabled": {"type": "boolean"},
            },
            "required": ["meta"],
        }

    def test_union_of_pydantic_models(self):
        class WeatherRequest(BaseModel):
            city: str

        class WeatherSuccess(BaseModel):
            temperature_celsius: int
            condition: str

        class WeatherFailure(BaseModel):
            error_code: Literal["not_found", "rate_limited"]
            retryable: bool

        WeatherMessage = WeatherRequest | WeatherSuccess | WeatherFailure

        assert _python_type_to_json_schema(WeatherMessage) == {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
                {
                    "type": "object",
                    "properties": {
                        "temperature_celsius": {"type": "integer"},
                        "condition": {"type": "string"},
                    },
                    "required": ["temperature_celsius", "condition"],
                },
                {
                    "type": "object",
                    "properties": {
                        "error_code": {
                            "enum": ["not_found", "rate_limited"],
                            "type": "string",
                        },
                        "retryable": {"type": "boolean"},
                    },
                    "required": ["error_code", "retryable"],
                },
            ]
        }

    def test_type_alias_of_pydantic_model_union(self):
        class CityLookup(BaseModel):
            city: str

        class ForecastSnapshot(BaseModel):
            high_celsius: int
            low_celsius: int

        class ForecastError(BaseModel):
            reason: Literal["not_found", "unavailable"]

        type WeatherPayload = CityLookup | ForecastSnapshot | ForecastError # type: ignore

        assert _python_type_to_json_schema(WeatherPayload) == {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
                {
                    "type": "object",
                    "properties": {
                        "high_celsius": {"type": "integer"},
                        "low_celsius": {"type": "integer"},
                    },
                    "required": ["high_celsius", "low_celsius"],
                },
                {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "enum": ["not_found", "unavailable"],
                            "type": "string",
                        },
                    },
                    "required": ["reason"],
                },
            ]
        }


class TestPrepareTools:
    # ------------------------------------------------------------------------
    # 3.1 callable list
    # ------------------------------------------------------------------------

    def test_prepare_tools_with_callables(self):

        def tool1(x: int) -> int:
            """Tool 1"""
            return x * 2

        def tool2(y: str) -> str:
            """Tool 2"""
            return y.upper()

        result = prepare_tools([tool1, tool2])

        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"

    def test_prepare_tools_with_complex_callable(self):
        class WeatherRequest(BaseModel):
            city: str

        class WeatherSuccess(BaseModel):
            temperature_celsius: int
            condition: str

        class WeatherFailure(BaseModel):
            error_code: Literal["not_found", "rate_limited"]
            retryable: bool

        def tool(x: WeatherRequest | WeatherSuccess | WeatherFailure) -> int:
            """Tool"""
            return 2

        result = prepare_tools([tool])

        assert len(result) == 1
        assert result[0] == {
            "name": "tool",
            "description": "Tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "temperature_celsius": {"type": "integer"},
                                    "condition": {"type": "string"},
                                },
                                "required": ["temperature_celsius", "condition"],
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "error_code": {
                                        "enum": ["not_found", "rate_limited"],
                                        "type": "string",
                                    },
                                    "retryable": {"type": "boolean"},
                                },
                                "required": ["error_code", "retryable"],
                            },
                        ],
                        "description": "Parameter x of type Union",
                    }
                },
                "required": ["x"],
            },
        }

    # ------------------------------------------------------------------------
    # 3.2 raw definition list
    # ------------------------------------------------------------------------

    def test_prepare_tools_with_raw_definitions(self):
        raw_tool = {
            "name": "custom_tool",
            "description": "A custom tool",
            "parameters": {"type": "object", "properties": {}},
        }

        result = prepare_tools([raw_tool])

        assert len(result) == 1
        assert result[0] == raw_tool

    # ------------------------------------------------------------------------
    # 3.3 mixed input
    # ------------------------------------------------------------------------

    def test_prepare_tools_mixed(self):

        def func_tool(a: str) -> str:
            """Function tool"""
            return a

        tool_def = ToolDef(
            name="tool_def",
            description="Tool definition",
            execute=lambda x: x,
        )

        dict_tool = {
            "name": "dict_tool",
            "description": "Dict tool",
            "parameters": {"type": "object", "properties": {}},
        }

        result = prepare_tools([func_tool, tool_def, dict_tool])

        assert len(result) == 3
        assert result[0]["name"] == "func_tool"
        assert result[0]["description"] == "Function tool"
        assert result[1]["name"] == "tool_def"
        assert result[2]["name"] == "dict_tool"

    # ------------------------------------------------------------------------
    # 3.4 empty list
    # ------------------------------------------------------------------------

    def test_prepare_tools_empty_list(self):
        result = prepare_tools([])
        assert result == []
