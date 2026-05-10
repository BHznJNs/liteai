from dais_sdk.types import ToolDef
from dais_sdk.tool.prepare import prepare_tools


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
