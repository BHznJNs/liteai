"""
Test suite for LlmRequestParams in dais_sdk.types.request_params
"""

import pytest

from dais_sdk.types import (
    LlmRequestParams,
    UserMessage,
    AssistantMessage,
    MessageGroup,
    ToolDef, RawToolDef,
)
from dais_sdk.tool.toolset import python_tool, PythonToolset


class TestLlmRequestParamsInit:
    """Test LlmRequestParams initialization and default values"""

    # ------------------------------------------------------------------------
    # 1.1 Test required fields
    # ------------------------------------------------------------------------

    def test_init_required_fields(self):
        """Test that required fields (model, messages) are correctly assigned"""
        messages = [UserMessage(content="Hello")]
        params = LlmRequestParams(model="gpt-4", messages=messages)

        assert params.model == "gpt-4"
        assert params.messages == messages

    def test_init_single_message(self):
        """Test that a single message object is accepted (if supported) or list is required"""
        msg = UserMessage(content="Hello")
        params = LlmRequestParams(model="gpt-4", messages=[msg])

        assert params.model == "gpt-4"
        assert len(params.messages) == 1

    # ------------------------------------------------------------------------
    # 1.2 Test default values
    # ------------------------------------------------------------------------

    def test_default_values(self):
        """Test fields have correct default values when not provided"""
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")]
        )

        assert params.tools is None
        assert params.toolsets is None
        assert params.tool_choice == "auto"
        assert params.timeout_sec is None
        assert params.temperature is None
        assert params.max_tokens is None
        assert params.headers is None
        assert params.extra_args is None
        assert params.instructions is None

    # ------------------------------------------------------------------------
    # 1.3 Test optional fields setting
    # ------------------------------------------------------------------------

    def test_optional_fields(self):
        """Test that optional fields can be set correctly"""
        messages = [UserMessage(content="Hello")]
        params = LlmRequestParams(
            model="gpt-4",
            messages=messages,
            instructions="Follow policy",
            temperature=0.7,
            max_tokens=1000,
            tool_choice="required",
            headers={"Authorization": "Bearer key"},
            extra_args={"custom_param": "value"}
        )

        assert params.instructions == "Follow policy"
        assert params.temperature == 0.7
        assert params.max_tokens == 1000
        assert params.tool_choice == "required"
        assert params.headers == {"Authorization": "Bearer key"}
        assert params.extra_args == {"custom_param": "value"}

    def test_init_with_mixed_messages(self):
        """Test initialization with mixed message types"""
        user_msg = UserMessage(content="Hello")
        params = LlmRequestParams(model="test", messages=[user_msg])
        assert len(params.messages) == 1


class TestExtractTools:
    """Test extract_tools method"""

    # ------------------------------------------------------------------------
    # 2.1 Test extract_tools with no tools defined
    # ------------------------------------------------------------------------

    def test_extract_tools_none(self):
        """Test extract_tools returns None when both tools and toolsets are None"""
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")]
        )

        assert params.extract_tools() is None

    def test_extract_tools_empty_toolsets(self):
        """Test extract_tools returns empty list when tools=None and toolsets=[]"""
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[]
        )

        # Per current implementation logic, if toolsets is empty list, it extends with nothing,
        # so tools list ends up being empty but not None (since list check passes)
        # Wait, current code:
        # if self.toolsets: (Empty list is falsy, so skipped)
        # if self.tools: (None is falsy, so skipped)
        # if self.tools is None and self.toolsets is None: return None
        # So with tools=None, toolsets=[], it should return [].
        result = params.extract_tools()
        assert result == []

    # ------------------------------------------------------------------------
    # 2.2 Test extract_tools with callable functions
    # ------------------------------------------------------------------------

    def test_extract_tools_functions(self):
        """Test extract_tools with callable functions in tools list"""
        def tool_func1(x: int) -> int:
            """Function 1"""
            return x

        def tool_func2(y: str) -> str:
            """Function 2"""
            return y

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[tool_func1, tool_func2]
        )

        extracted = params.extract_tools()
        assert extracted is not None
        assert len(extracted) == 2
        assert tool_func1 in extracted
        assert tool_func2 in extracted

    # ------------------------------------------------------------------------
    # 2.3 Test extract_tools with toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_from_toolsets(self):
        """Test extract_tools properly calls get_tools on toolsets"""
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
        # Check types are ToolDef
        assert all(isinstance(t, ToolDef) for t in extracted)

    # ------------------------------------------------------------------------
    # 2.4 Test extract_tools with mixed tools and toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_mixed_tools_and_toolsets(self):
        """Test extract_tools combines tools from both lists"""
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

        # Helper to safely get tool names regardless of type
        def get_tool_name(tool):
            if isinstance(tool, ToolDef):
                return tool.name
            elif hasattr(tool, '__name__'):
                return tool.__name__
            elif isinstance(tool, dict):
                return tool.get("name")
            return str(tool)

        tool_names = {get_tool_name(t) for t in extracted}
        assert 'standalone_tool' in tool_names
        assert 'MyToolset__toolset_method' in tool_names

    # ------------------------------------------------------------------------
    # 2.5 Test extract_tools caching
    # ------------------------------------------------------------------------

    def test_extract_tools_caching(self):
        """Test that extract_tools returns the cached list to avoid recomputation"""
        def my_tool(x: int) -> int:
            """My tool"""
            return x

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[my_tool]
        )

        first_call = params.extract_tools()
        second_call = params.extract_tools()

        assert first_call is second_call

    # ------------------------------------------------------------------------
    # 2.6 Test extract_tools with multiple toolsets
    # ------------------------------------------------------------------------

    def test_extract_tools_multiple_toolsets(self):
        """Test extract_tools aggregates tools from multiple toolsets"""
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

        math_tools = MathToolset()
        string_tools = StringToolset()

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[math_tools, string_tools]
        )

        extracted = params.extract_tools()
        assert extracted is not None
        assert len(extracted) == 2  # 1 from Math + 1 from String
        # All tools from PythonToolset are ToolDef, so .name is safe here
        tool_names = {t.name for t in extracted}  # type: ignore
        assert "MathToolset__add" in tool_names
        assert "StringToolset__upper" in tool_names


class TestMessageGroups:
    def test_message_group_has_message(self):
        user_message = UserMessage(content="hello")
        assistant_message = AssistantMessage(content="world")
        group = MessageGroup(messages=[user_message, assistant_message])

        assert group.has(user_message.id)
        assert group.has(assistant_message.id)
        assert not group.has("missing")

    def test_message_group_find_returns_first_matching_message(self):
        first = UserMessage(content="first")
        second = AssistantMessage(content="second")
        third = UserMessage(content="third")
        group = MessageGroup(messages=[first, second, third])

        found = group.find(lambda message: message.role == "user")

        assert found is first

    def test_message_group_rejects_nested_group(self):
        inner_group = MessageGroup(messages=[UserMessage(content="inner")])

        with pytest.raises(ValueError, match="MessageGroup cannot contain another MessageGroup"):
            MessageGroup(messages=[inner_group])

    def test_expand_messages_flattens_message_groups(self):
        first = UserMessage(content="first")
        second = AssistantMessage(content="second")
        third = UserMessage(content="third")
        params = LlmRequestParams(
            model="test-model",
            messages=[first, MessageGroup(messages=[second, third])],
        )

        expanded = params.expand_messages()
        assert expanded == [first, second, third]


class TestFindTool:
    """Test find_tool method"""

    # ------------------------------------------------------------------------
    # 3.1 Test find_tool with no tools defined
    # ------------------------------------------------------------------------

    def test_find_tool_none_defined(self):
        """Test find_tool returns None when no tools are defined"""
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")]
        )

        result = params.find_tool("any_tool")
        assert result is None

    # ------------------------------------------------------------------------
    # 3.2 Test find_tool in tools list
    # ------------------------------------------------------------------------

    def test_find_tool_in_tools(self):
        """Test find_tool locates tool in the tools list"""
        def my_function(x: int) -> int:
            """My function"""
            return x

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[my_function]
        )

        result = params.find_tool("my_function")
        assert result is my_function

    def test_find_tool_in_tools_dict(self):
        """Test find_tool locates tool in dict-based tools list"""
        raw_tool: RawToolDef = {
            "name": "search_data",
            "description": "Search data",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[raw_tool]
        )

        result = params.find_tool("search_data")
        assert result is raw_tool

    # ------------------------------------------------------------------------
    # 3.3 Test find_tool in toolsets (namespaced)
    # ------------------------------------------------------------------------

    def test_find_tool_in_toolsets(self):
        """Test find_tool locates tool in toolsets using namespaced name"""
        class Calculator(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add"""
                return a + b

        calc = Calculator()
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[calc]
        )

        result = params.find_tool("Calculator__add")
        assert result is not None
        assert isinstance(result, ToolDef)
        assert result.name == "Calculator__add"

    # ------------------------------------------------------------------------
    # 3.4 Test find_tool not found
    # ------------------------------------------------------------------------

    def test_find_tool_not_found(self):
        """Test find_tool returns None for non-existent tool"""
        def my_tool(x: int) -> int:
            """My tool"""
            return x

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[my_tool]
        )

        result = params.find_tool("non_existent_tool")
        assert result is None

    # ------------------------------------------------------------------------
    # 3.5 Test find_tool in mixed tools and toolsets
    # ------------------------------------------------------------------------

    def test_find_tool_in_mixed(self):
        """Test find_tool searches both tools and toolsets"""
        def standalone_tool() -> str:
            return "standalone"

        class MyToolset(PythonToolset):
            @python_tool
            def toolset_tool(self) -> str:
                return "toolset"

        ts = MyToolset()
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[standalone_tool],
            toolsets=[ts]
        )

        assert params.find_tool("standalone_tool") is standalone_tool
        assert params.find_tool("MyToolset__toolset_tool") is not None
