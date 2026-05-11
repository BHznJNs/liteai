import asyncio
import json

import pytest

from dais_sdk.types import ToolDef
from dais_sdk.tool.execute import execute_tool


class TestToolExecution:
    # ------------------------------------------------------------------------
    # 4.1 execute_tool (async) with regular functions
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_simple_function(self):
        def divide(a: int, b: int) -> float:
            """Divide two numbers"""
            return a / b

        result = await execute_tool(divide, '{"a": 10, "b": 2}')
        assert result.serialized == "5.0"
        assert result.raw == 5.0
        assert json.loads(result.serialized) == 5.0

    @pytest.mark.asyncio
    async def test_execute_tool_async_function_with_list(self):
        def sum_list(numbers: list) -> int:
            """Sum a list of numbers"""
            return sum(numbers)

        result = await execute_tool(sum_list, '{"numbers": [1, 2, 3, 4, 5]}')
        assert result.serialized == "15"
        assert result.raw == 15
        assert json.loads(result.serialized) == 15

    # ------------------------------------------------------------------------
    # 4.6 execute_tool (async) with async functions
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_function(self):
        async def async_power(base: int, exp: int) -> int:
            """Async power calculation"""
            await asyncio.sleep(0.01)
            return base ** exp

        result = await execute_tool(async_power, '{"base": 2, "exp": 10}')
        assert result.serialized == "1024"
        assert result.raw == 1024
        assert json.loads(result.serialized) == 1024

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_function_complex(self):
        async def async_process(items: list[str], prefix: str) -> list[str]:
            """Async process list"""
            await asyncio.sleep(0.01)
            return [prefix + item for item in items]

        result = await execute_tool(async_process, '{"items": ["a", "b", "c"], "prefix": "x_"}')
        assert isinstance(result.serialized, str)
        assert result.raw == ["x_a", "x_b", "x_c"]
        assert json.loads(result.serialized) == ["x_a", "x_b", "x_c"]

    # ------------------------------------------------------------------------
    # 4.7 execute_tool (async) with ToolDef
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_tooldef_sync(self):
        def multiply_by_two(n: int) -> int:
            """Multiply by two"""
            return n * 2

        tool_def = ToolDef(
            name="multiply_tool",
            description="Multiply number by two",
            execute=multiply_by_two,
        )

        result = await execute_tool(tool_def, '{"n": 21}')
        assert result.serialized == "42"
        assert result.raw == 42
        assert json.loads(result.serialized) == 42

    @pytest.mark.asyncio
    async def test_execute_tool_async_tooldef_async(self):
        async def async_reverse(text: str) -> str:
            """Async reverse string"""
            await asyncio.sleep(0.01)
            return text[::-1]

        tool_def = ToolDef(
            name="reverse_tool",
            description="Reverse a string",
            execute=async_reverse,
        )

        result = await execute_tool(tool_def, '{"text": "hello"}')
        # String results should pass through unchanged
        assert result.serialized == "olleh"
        assert result.raw == "olleh"

    # ------------------------------------------------------------------------
    # 4.4 error handling
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_json(self):
        def dummy(x: int) -> int:
            """Dummy function"""
            return x

        with pytest.raises(json.JSONDecodeError):
            await execute_tool(dummy, 'not valid json')

    # ------------------------------------------------------------------------
    # 4.5 class method tests
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_instance_method(self):
        class Calculator:
            def subtract(self, x: int, y: int) -> int:
                """Subtract two numbers"""
                return x - y

        calc = Calculator()
        result = await execute_tool(calc.subtract, '{"x": 20, "y": 8}')
        assert result.serialized == "12"
        assert result.raw == 12
        assert json.loads(result.serialized) == 12

    @pytest.mark.asyncio
    async def test_execute_tool_async_classmethod(self):
        class MathUtils:
            @classmethod
            def multiply(cls, a: int, b: int) -> int:
                """Multiply numbers"""
                return a * b

        result = await execute_tool(MathUtils.multiply, '{"a": 9, "b": 6}')
        assert result.serialized == "54"
        assert result.raw == 54
        assert json.loads(result.serialized) == 54

    @pytest.mark.asyncio
    async def test_execute_tool_async_staticmethod(self):
        class StringUtils:
            @staticmethod
            def uppercase(text: str) -> str:
                """Convert to uppercase"""
                return text.upper()

        result = await execute_tool(StringUtils.uppercase, '{"text": "world"}')
        assert result.serialized == "WORLD"
        assert result.raw == "WORLD"

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_instance_method(self):
        class AsyncCalculator:
            async def divide(self, x: float, y: float) -> float:
                """Async divide"""
                await asyncio.sleep(0.01)
                return x / y

        calc = AsyncCalculator()
        result = await execute_tool(calc.divide, '{"x": 100, "y": 4}')
        assert result.serialized == "25.0"
        assert result.raw == 25.0
        assert json.loads(result.serialized) == 25.0

    # ------------------------------------------------------------------------
    # 4.6 invalid tool type error tests
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_int(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool(456, '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_string(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool("invalid", '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_list(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool([4, 5, 6], '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_dict(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool({"foo": "bar"}, '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_none(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool(None, '{"x": 1}')  # type: ignore
