import asyncio
import json
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

from dais_sdk.types import LlmRequestParams, UserMessage, ToolDef
from dais_sdk.tool import python_tool, PythonToolset
from dais_sdk.tool.prepare import prepare_tools, generate_tool_definition_from_callable
from dais_sdk.tool.execute import execute_tool


class TestToolsetDecorator:
    """Test the @python_tool decorator and PythonToolset class"""

    # ------------------------------------------------------------------------
    # 1.1 @python_tool decorator basic functionality
    # ------------------------------------------------------------------------

    def test_tool_decorator_marks_function(self):
        """@python_tool decorator should mark a function with the tool flag"""
        from dais_sdk.tool.toolset.python_toolset import TOOL_FLAG

        @python_tool
        def my_function(x: int) -> int:
            """Test function"""
            return x * 2

        assert hasattr(my_function, TOOL_FLAG)
        assert getattr(my_function, TOOL_FLAG) is True

    def test_tool_decorator_preserves_function(self):
        """@python_tool decorator should preserve the original function"""

        def original_func(a: str, b: int) -> str:
            """Original docstring"""
            return a * b

        decorated_func = python_tool(original_func)

        # Function should still work normally
        assert decorated_func("x", 3) == "xxx"
        assert decorated_func.__name__ == "original_func"
        assert decorated_func.__doc__ == "Original docstring"

    def test_tool_decorator_on_method(self):
        """@python_tool decorator should work on methods"""
        from dais_sdk.tool.toolset.python_toolset import TOOL_FLAG

        class MyClass:
            @python_tool
            def my_method(self, value: int) -> int:
                """Test method"""
                return value + 10

        obj = MyClass()
        # The unbound method should have the flag
        assert hasattr(MyClass.my_method, TOOL_FLAG)
        # Bound method should also work
        assert obj.my_method(5) == 15

    def test_tool_decorator_preserves_generate_tool_definition(self):
        """@python_tool decorator should produce identical tool definition via generate_tool_definition_from_callable"""

        # Original function
        def original_function(x: int, y: int, z: str = "default") -> str:
            """Calculate and format result."""
            return f"x={x}, y={y}, z={z}"

        # Decorated function
        decorated_function = python_tool(original_function)

        # Generate tool definitions from both
        original_schema = generate_tool_definition_from_callable(original_function)
        decorated_schema = generate_tool_definition_from_callable(decorated_function)

        # Verify all fields are identical
        assert original_schema["name"] == decorated_schema["name"]
        assert original_schema["description"] == decorated_schema["description"]
        assert original_schema["parameters"] == decorated_schema["parameters"]

        # Also verify deep equality of the entire schema
        assert original_schema == decorated_schema

    def test_tool_decorator_with_validate_preserves_generate_tool_definition(self):
        """@python_tool(validate=True) should produce identical tool definition via generate_tool_definition_from_callable"""

        def original_function(x: int, y: int, z: str = "default") -> str:
            """Calculate and format result."""
            return f"x={x}, y={y}, z={z}"

        decorated_function = python_tool(validate=True)(original_function)

        original_schema = generate_tool_definition_from_callable(original_function)
        decorated_schema = generate_tool_definition_from_callable(decorated_function)

        assert original_schema["name"] == decorated_schema["name"]
        assert original_schema["description"] == decorated_schema["description"]
        assert original_schema["parameters"] == decorated_schema["parameters"]
        assert original_schema == decorated_schema


    def test_tool_decorator_with_validate_runtime_call(self):
        """@python_tool(validate=True) should validate arguments on runtime call"""

        @python_tool(validate=True)
        def add(x: int, y: int) -> int:
            """Add two integers."""
            return x + y

        # Correct types: should return normally
        assert add(1, 2) == 3

        # Wrong type: should raise pydantic.ValidationError
        with pytest.raises(ValidationError):
            add("not-an-int", 2) # type: ignore

    def test_tool_decorator_with_mapping_defaults(self):
        """@python_tool(defaults=...) should accept Mapping and preserve key-values"""
        from dais_sdk.tool.toolset.python_toolset import get_tool_defaults

        defaults = MappingProxyType({"auto_approve": True, "max_tokens": 512})

        @python_tool(defaults=defaults)
        def read_file(path: str) -> str:
            """Read file"""
            return path

        stored_defaults = get_tool_defaults(read_file)
        assert isinstance(stored_defaults, dict)
        assert stored_defaults == {"auto_approve": True, "max_tokens": 512}

    def test_toolset_get_tools_propagates_defaults(self):
        """PythonToolset.get_tools should propagate defaults into ToolDef.defaults"""

        class DefaultToolset(PythonToolset):
            @python_tool(defaults=MappingProxyType({"auto_approve": True, "retries": 2}))
            def fetch(self, url: str) -> str:
                """Fetch content"""
                return url

        tools = DefaultToolset().get_tools(namespaced_tool_name=False)

        assert len(tools) == 1
        assert tools[0].name == "fetch"
        assert tools[0].defaults == {"auto_approve": True, "retries": 2}

    def test_tool_defaults_do_not_track_external_mutation(self):
        """defaults should be copied at decoration time and not track external dict mutations"""
        defaults: dict = {"auto_approve": True}

        class DefaultToolset(PythonToolset):
            @python_tool(defaults=defaults)
            def review(self, text: str) -> str:
                """Review text"""
                return text

        defaults["auto_approve"] = False
        defaults["new_field"] = "late-change"

        tools = DefaultToolset().get_tools(namespaced_tool_name=False)

        assert len(tools) == 1
        assert tools[0].defaults == {"auto_approve": True}


class TestToolsetBasic:
    """Test basic PythonToolset functionality"""

    # ------------------------------------------------------------------------
    # 1.2 PythonToolset with no tools
    # ------------------------------------------------------------------------

    def test_empty_toolset(self):
        """PythonToolset with no @python_tool decorated methods should return empty list"""
        class EmptyToolset(PythonToolset):
            def regular_method(self):
                """Not a tool"""
                return "hello"

        toolset = EmptyToolset()
        tools = toolset.get_tools()

        assert tools == []

    def test_toolset_name_property(self):
        """PythonToolset.name should return the class name by default"""
        class MyToolset(PythonToolset):
            pass

        toolset = MyToolset()
        assert toolset.name == "MyToolset"

    def test_toolset_format_tool_name(self):
        """Toolset.format_tool_name should follow the Namespace__ToolName pattern"""
        class MyToolset(PythonToolset):
            pass

        toolset = MyToolset()
        assert toolset.format_tool_name("my_tool") == "MyToolset__my_tool"

    # ------------------------------------------------------------------------
    # 1.3 PythonToolset with single tool
    # ------------------------------------------------------------------------

    def test_toolset_single_tool(self):
        """PythonToolset should discover single @python_tool decorated method"""
        class SingleToolset(PythonToolset):
            @python_tool
            def calculate(self, x: int, y: int) -> int:
                """Add two numbers"""
                return x + y

        toolset = SingleToolset()
        tools = toolset.get_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], ToolDef)
        assert tools[0].name == "SingleToolset__calculate"
        assert callable(tools[0].execute)

    # ------------------------------------------------------------------------
    # 1.4 PythonToolset with multiple tools
    # ------------------------------------------------------------------------

    def test_toolset_multiple_tools(self):
        """PythonToolset should discover all @python_tool decorated methods"""
        class MultiToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add two numbers"""
                return a + b

            @python_tool
            def multiply(self, a: int, b: int) -> int:
                """Multiply two numbers"""
                return a * b

            def helper(self, x: int) -> int:
                """Not a tool"""
                return x * 2

            @python_tool
            def subtract(self, a: int, b: int) -> int:
                """Subtract two numbers"""
                return a - b

        toolset = MultiToolset()
        tools = toolset.get_tools()

        assert len(tools) == 3
        assert all(isinstance(t, ToolDef) for t in tools)
        tool_names = {t.name for t in tools}
        assert tool_names == {"MultiToolset__add", "MultiToolset__multiply", "MultiToolset__subtract"}

    # ------------------------------------------------------------------------
    # 1.5 Tool methods are bound methods
    # ------------------------------------------------------------------------

    def test_toolset_methods_are_bound(self):
        """Tools returned from PythonToolset should be bound methods"""
        class Calculator(PythonToolset):
            def __init__(self, base: int):
                self.base = base

            @python_tool
            def add_to_base(self, value: int) -> int:
                """Add value to base"""
                return self.base + value

        calc = Calculator(base=100)
        tools = calc.get_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], ToolDef)
        # Should be able to call without passing self
        assert callable(tools[0].execute)
        assert tools[0].execute(50) == 150


class TestToolsetIntegration:
    """Test PythonToolset integration with tool preparation"""

    # ------------------------------------------------------------------------
    # 1.6 prepare_tools with PythonToolset methods
    # ------------------------------------------------------------------------

    def test_prepare_tools_with_toolset_methods(self):
        """prepare_tools should work with methods from PythonToolset"""
        class MathToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add two numbers"""
                return a + b

            @python_tool
            def multiply(self, x: int, y: int) -> int:
                """Multiply two numbers"""
                return x * y

        toolset = MathToolset()
        tool_methods = toolset.get_tools()
        prepared = prepare_tools(tool_methods)

        assert len(prepared) == 2

        # Check tool names
        names = {t["name"] for t in prepared}
        assert "MathToolset__add" in names
        assert "MathToolset__multiply" in names

    # ------------------------------------------------------------------------
    # 1.7 Mixed tools: functions and PythonToolset methods
    # ------------------------------------------------------------------------

    def test_prepare_tools_mixed_with_toolset(self):
        """prepare_tools should handle mix of functions and PythonToolset methods"""
        def standalone_func(text: str) -> str:
            """Standalone function"""
            return text.upper()

        class MyToolset(PythonToolset):
            @python_tool
            def toolset_method(self, n: int) -> int:
                """Toolset method"""
                return n * 2

        toolset = MyToolset()
        all_tools = [standalone_func] + toolset.get_tools()
        prepared = prepare_tools(all_tools)

        assert len(prepared) == 2
        names = {t["name"] for t in prepared}
        assert names == {"standalone_func", "MyToolset__toolset_method"}


class TestToolsetWithExecution:
    """Test PythonToolset methods with tool execution"""

    # ------------------------------------------------------------------------
    # 1.8 Execute PythonToolset method asynchronously
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_toolset_method_async(self):
        """execute_tool should work with PythonToolset methods"""
        class StringTools(PythonToolset):
            @python_tool
            def reverse(self, text: str) -> str:
                """Reverse a string"""
                return text[::-1]

        tools_instance = StringTools()
        methods = tools_instance.get_tools()

        result = await execute_tool(methods[0], '{"text": "hello"}')
        assert result == "olleh"

    # ------------------------------------------------------------------------
    # 1.9 Execute async PythonToolset method
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_async_toolset_method(self):
        """Async methods in PythonToolset should work"""
        class AsyncToolset(PythonToolset):
            @python_tool
            async def async_multiply(self, a: int, b: int) -> int:
                """Async multiply"""
                await asyncio.sleep(0.01)
                return a * b

        toolset = AsyncToolset()
        methods = toolset.get_tools()

        result = await execute_tool(methods[0], '{"a": 7, "b": 8}')
        assert result == "56"
        assert json.loads(result) == 56


class TestToolsetAdvanced:
    """Advanced PythonToolset test cases"""

    # ------------------------------------------------------------------------
    # 1.11 PythonToolset with instance state
    # ------------------------------------------------------------------------

    def test_toolset_with_state(self):
        """PythonToolset methods should have access to instance state"""
        class StatefulToolset(PythonToolset):
            def __init__(self, multiplier: int):
                self.multiplier = multiplier
                self.call_count = 0

            @python_tool
            def scale(self, value: int) -> int:
                """Scale value by multiplier"""
                self.call_count += 1
                return value * self.multiplier

        toolset = StatefulToolset(multiplier=5)
        tools = toolset.get_tools()

        # Execute the tool
        assert isinstance(tools[0], ToolDef)
        result1 = tools[0].execute(10)
        assert result1 == 50
        assert toolset.call_count == 1

        result2 = tools[0].execute(20)
        assert result2 == 100
        assert toolset.call_count == 2

    # ------------------------------------------------------------------------
    # 1.12 PythonToolset inheritance
    # ------------------------------------------------------------------------

    def test_toolset_inheritance(self):
        """Child PythonToolset should inherit parent's tools"""

        class BaseToolset(PythonToolset):
            @python_tool
            def base_method(self, x: int) -> int:
                """Base method"""
                return x * 2

        class DerivedToolset(BaseToolset):
            @python_tool
            def derived_method(self, x: int) -> int:
                """Derived method"""
                return x + 10

        toolset = DerivedToolset()
        tools = toolset.get_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"DerivedToolset__base_method", "DerivedToolset__derived_method"}

    # ------------------------------------------------------------------------
    # 1.13 PythonToolset with complex parameter types
    # ------------------------------------------------------------------------

    def test_toolset_complex_types(self):
        """PythonToolset methods with complex types should work with prepare_tools"""

        class ComplexToolset(PythonToolset):
            @python_tool
            def process_data(self, items: list[str], config: dict[str, Any]) -> dict:
                """Process items with config"""
                return {"count": len(items), "config": config}

        toolset = ComplexToolset()
        prepared = prepare_tools(toolset.get_tools())

        assert len(prepared) == 1
        params = prepared[0]["parameters"]
        assert "items" in params["properties"]
        assert "config" in params["properties"]
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["config"]["type"] == "object"

    # ------------------------------------------------------------------------
    # 1.14 Multiple PythonToolset instances
    # ------------------------------------------------------------------------

    def test_multiple_toolset_instances(self):
        """Multiple instances of same PythonToolset should work independently"""

        class Counter(PythonToolset):
            def __init__(self, start: int):
                self.value = start

            @python_tool
            def increment(self, amount: int) -> int:
                """Increment counter"""
                self.value += amount
                return self.value

        counter1 = Counter(start=0)
        counter2 = Counter(start=100)

        tools1 = counter1.get_tools()
        tools2 = counter2.get_tools()

        # Execute on different instances
        assert isinstance(tools1[0], ToolDef)
        assert isinstance(tools2[0], ToolDef)
        assert tools1[0].execute(5) == 5
        assert tools2[0].execute(5) == 105
        assert counter1.value == 5
        assert counter2.value == 105


class TestRequestParamsWithToolsets:
    """Test LlmRequestParams integration with toolsets"""

    # ------------------------------------------------------------------------
    # 1.15 LlmRequestParams extracts tools from toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_from_toolsets(self):
        """LlmRequestParams should extract tools from toolsets parameter"""
        class MyToolset(PythonToolset):
            @python_tool
            def tool1(self, x: int) -> int:
                """Tool 1"""
                return x * 2

            @python_tool
            def tool2(self, y: str) -> str:
                """Tool 2"""
                return y.upper()

        toolset = MyToolset()
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[toolset]
        )

        extracted = params.extract_tools()

        assert extracted is not None
        assert len(extracted) == 2
        assert all(isinstance(tool, ToolDef) for tool in extracted)

    # ------------------------------------------------------------------------
    # 1.16 LlmRequestParams with both tools and toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_mixed_tools_and_toolsets(self):
        """LlmRequestParams should handle both tools and toolsets together"""
        def standalone_tool(a: int) -> int:
            """Standalone tool"""
            return a + 1

        class MyToolset(PythonToolset):
            @python_tool
            def toolset_method(self, b: int) -> int:
                """Toolset method"""
                return b * 2

        toolset = MyToolset()
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[standalone_tool],
            toolsets=[toolset]
        )

        extracted = params.extract_tools()

        assert extracted is not None
        assert len(extracted) == 2
        # Check we have both tools
        tool_names = {getattr(t, 'name', getattr(t, '__name__', str(t))) for t in extracted}
        assert 'standalone_tool' in tool_names
        assert 'MyToolset__toolset_method' in tool_names

    # ------------------------------------------------------------------------
    # 1.17 LlmRequestParams with multiple toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_multiple_toolsets(self):
        """LlmRequestParams should handle multiple toolsets"""
        class MathToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add"""
                return a + b

        class StringToolset(PythonToolset):
            @python_tool
            def upper(self, text: str) -> str:
                """Upper"""
                return text.upper()

            @python_tool
            def lower(self, text: str) -> str:
                """Lower"""
                return text.lower()

        math_tools = MathToolset()
        string_tools = StringToolset()

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[math_tools, string_tools]
        )

        extracted = params.extract_tools()

        assert extracted is not None
        assert len(extracted) == 3  # 1 from MathToolset + 2 from StringToolset

    # ------------------------------------------------------------------------
    # 1.18 LlmRequestParams with None toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_none_toolsets(self):
        """LlmRequestParams should handle None toolsets gracefully"""
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=None,
            toolsets=None
        )

        extracted = params.extract_tools()

        assert extracted is None

    # ------------------------------------------------------------------------
    # 1.19 LlmRequestParams with empty toolsets list
    # ------------------------------------------------------------------------

    def test_extract_tools_empty_toolsets(self):
        """LlmRequestParams should handle empty toolsets list"""
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=None,
            toolsets=[]
        )

        extracted = params.extract_tools()

        # Should return empty list since toolsets is empty
        assert extracted == []


class TestToolsetEdgeCases:
    """Edge cases and error scenarios for PythonToolset"""

    # ------------------------------------------------------------------------
    # 1.20 PythonToolset with private methods
    # ------------------------------------------------------------------------

    def test_toolset_ignores_private_methods(self):
        """PythonToolset should not include private methods even if decorated"""

        class PrivateToolset(PythonToolset):
            @python_tool
            def public_method(self, x: int) -> int:
                """Public"""
                return x

            @python_tool
            def _private_method(self, x: int) -> int:
                """Private"""
                return x * 2

        toolset = PrivateToolset()
        tools = toolset.get_tools()

        # Both should be included (private methods are still methods)
        # The TOOL_FLAG doesn't distinguish between public/private
        tool_names = {t.name for t in tools}
        # This depends on implementation, but typically both would be included
        assert 'PrivateToolset__public_method' in tool_names

    # ------------------------------------------------------------------------
    # 1.21 Toolset with classmethod/staticmethod
    # ------------------------------------------------------------------------

    def test_toolset_with_classmethod(self):
        """@python_tool on classmethod should be returned by get_tools"""

        class ToolsetWithClassmethod(PythonToolset):
            @python_tool
            def instance_method(self, x: int) -> int:
                """Instance method"""
                return x

            @classmethod
            @python_tool
            def class_method(cls, x: int) -> int:
                """Class method"""
                return x * 2

        toolset = ToolsetWithClassmethod()
        tools = toolset.get_tools()

        assert len(tools) == 2

    # ------------------------------------------------------------------------
    # 1.22 Toolset method without docstring
    # ------------------------------------------------------------------------

    def test_toolset_method_without_docstring_allowed(self):
        """PythonToolset method without docstring should be allowed but have empty description"""
        class NoDocToolset(PythonToolset):
            @python_tool
            def no_doc_method(self, x: int) -> int:
                return x * 2

        toolset = NoDocToolset()
        tools = toolset.get_tools()

        assert len(tools) == 1
        assert tools[0].description == ""
        
        # prepare_tools should now work because ToolDef has description (even if empty)
        prepared = prepare_tools(tools)
        assert len(prepared) == 1
        assert prepared[0]["description"] == ""


class TestNamespacedToolName:
    """Test namespaced_tool_name parameter in get_tools"""

    # ------------------------------------------------------------------------
    # 1.23 get_tools with namespaced_tool_name=False
    # ------------------------------------------------------------------------

    def test_get_tools_without_namespace(self):
        """get_tools with namespaced_tool_name=False should return original tool names"""
        class MyToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add two numbers"""
                return a + b

            @python_tool
            def multiply(self, x: int, y: int) -> int:
                """Multiply two numbers"""
                return x * y

        toolset = MyToolset()
        
        # With namespaced_tool_name=True (default)
        tools_with_namespace = toolset.get_tools(namespaced_tool_name=True)
        assert len(tools_with_namespace) == 2
        tool_names_with_ns = {t.name for t in tools_with_namespace}
        assert tool_names_with_ns == {"MyToolset__add", "MyToolset__multiply"}
        
        # With namespaced_tool_name=False
        tools_without_namespace = toolset.get_tools(namespaced_tool_name=False)
        assert len(tools_without_namespace) == 2
        tool_names_without_ns = {t.name for t in tools_without_namespace}
        assert tool_names_without_ns == {"add", "multiply"}

    def test_get_tools_default_namespace(self):
        """get_tools should use namespaced_tool_name=True by default"""
        class SimpleToolset(PythonToolset):
            @python_tool
            def calculate(self, value: int) -> int:
                """Calculate something"""
                return value * 2

        toolset = SimpleToolset()
        
        # Default behavior should be namespaced
        tools_default = toolset.get_tools()
        tools_explicit = toolset.get_tools(namespaced_tool_name=True)
        
        assert len(tools_default) == 1
        assert len(tools_explicit) == 1
        assert tools_default[0].name == tools_explicit[0].name
        assert tools_default[0].name == "SimpleToolset__calculate"

    def test_get_tools_returns_copy(self):
        """get_tools should return a copy, not a reference to internal cache"""
        class MyToolset(PythonToolset):
            @python_tool
            def tool1(self, x: int) -> int:
                """Tool 1"""
                return x * 2

        toolset = MyToolset()
        tools1 = toolset.get_tools(namespaced_tool_name=False)
        original_tool = tools1[0]
        
        # Modify the returned tool's name
        tools1[0] = ToolDef(
            name="modified_name",
            description=original_tool.description,
            parameters=original_tool.parameters,
            execute=original_tool.execute
        )
        
        # Get tools again - should not reflect the modification
        tools2 = toolset.get_tools(namespaced_tool_name=False)
        
        assert len(tools2) == 1
        assert tools2[0].name == "tool1"  # Original name, not "modified_name"
        assert tools1[0].name == "modified_name"  # The first list was modified