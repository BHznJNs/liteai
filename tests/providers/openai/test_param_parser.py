from typing import Any, cast

import pytest

from dais_sdk.providers.openai import OpenAIProviderMessageParser, OpenAIProviderParamParser
from dais_sdk.types.content_block import TextBlock
from dais_sdk.types.message import AssistantMessage, ResolvedToolMessage, ResolvedUserMessage, ToolMessage
from dais_sdk.types.request_params import LlmRequestParams


def _build_parser() -> OpenAIProviderParamParser:
    return OpenAIProviderParamParser(OpenAIProviderMessageParser())


def test_parse_nonstream_maps_core_fields_and_extra_args() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[ResolvedUserMessage(content="hello")],
        tool_choice="required",
        temperature=0.3,
        max_tokens=128,
        extra_args={"top_p": 0.9, "presence_penalty": 0.1},
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    assert parsed["model"] == "gpt-4o-mini"
    assert parsed["tool_choice"] == "required"
    assert parsed["temperature"] == 0.3
    assert parsed["max_tokens"] == 128
    assert parsed["top_p"] == 0.9
    assert parsed["presence_penalty"] == 0.1

    messages = cast(list[dict[str, Any]], parsed["messages"])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hello"


def test_parse_nonstream_includes_instructions_as_system_message() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[ResolvedUserMessage(content="hello")],
        instructions="Follow policy",
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    messages = cast(list[dict[str, Any]], parsed["messages"])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Follow policy"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "hello"


def test_parse_nonstream_without_tools_does_not_inject_tools_key() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[ResolvedUserMessage(content="hello")],
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    assert "tools" not in parsed


def test_parse_nonstream_with_tools_injects_tools_key() -> None:
    parser = _build_parser()

    def search_docs(query: str) -> str:
        """Search docs by query."""
        return query

    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[ResolvedUserMessage(content="hello")],
        tools=[search_docs],
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    assert "tools" in parsed
    tools = cast(list[dict[str, Any]], parsed["tools"])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "search_docs"


def test_parse_stream_sets_stream_flags_and_include_usage() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[ResolvedUserMessage(content="hello")],
    )

    parsed = cast(dict[str, Any], parser.parse_stream(params))

    assert parsed["stream"] is True
    assert parsed["stream_options"] == {"include_usage": True}
    assert parsed["model"] == "gpt-4o-mini"


def test_preparse_messages_delays_tool_generated_user_messages_until_after_tool_results() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[
            AssistantMessage(tool_calls=[
                AssistantMessage.ToolCall(id="call_1", name="make_file", arguments={}),
            ]),
            ResolvedToolMessage(
                call_id="call_1",
                name="make_file",
                arguments={},
                content=[TextBlock(text="generated text")],
                is_error=False,
            ),
            ResolvedUserMessage(content="next user message"),
        ],
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    messages = cast(list[dict[str, Any]], parsed["messages"])
    assert [message["role"] for message in messages] == ["assistant", "tool", "user", "user"]
    assert messages[1]["tool_call_id"] == "call_1"
    assert "call_1" in messages[2]["content"][0]["text"]
    assert messages[2]["content"][1] == {"type": "text", "text": "generated text"}
    assert messages[3]["content"] == "next user message"


def test_preparse_messages_flushes_multiple_tool_generated_user_messages_in_order() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[
            AssistantMessage(tool_calls=[
                AssistantMessage.ToolCall(id="call_1", name="make_file", arguments={}),
                AssistantMessage.ToolCall(id="call_2", name="make_file", arguments={}),
            ]),
            ResolvedToolMessage(
                call_id="call_1",
                name="make_file",
                arguments={},
                content=[TextBlock(text="first")],
                is_error=False,
            ),
            ResolvedToolMessage(
                call_id="call_2",
                name="make_file",
                arguments={},
                content=[TextBlock(text="second")],
                is_error=False,
            ),
            ResolvedUserMessage(content="continue"),
        ],
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    messages = cast(list[dict[str, Any]], parsed["messages"])
    assert [message["role"] for message in messages] == ["assistant", "tool", "tool", "user", "user", "user"]
    assert messages[1]["tool_call_id"] == "call_1"
    assert messages[2]["tool_call_id"] == "call_2"
    assert messages[3]["content"][1] == {"type": "text", "text": "first"}
    assert messages[4]["content"][1] == {"type": "text", "text": "second"}
    assert messages[5]["content"] == "continue"


def test_preparse_messages_flushes_tool_generated_user_message_at_end() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[
            AssistantMessage(tool_calls=[
                AssistantMessage.ToolCall(id="call_1", name="make_file", arguments={}),
            ]),
            ResolvedToolMessage(
                call_id="call_1",
                name="make_file",
                arguments={},
                content=[TextBlock(text="generated text")],
                is_error=False,
            ),
        ],
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    messages = cast(list[dict[str, Any]], parsed["messages"])
    assert [message["role"] for message in messages] == ["assistant", "tool", "user"]
    assert messages[1]["tool_call_id"] == "call_1"
    assert messages[2]["content"][1] == {"type": "text", "text": "generated text"}


def test_parse_nonstream_rejects_unresolved_tool_message() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[ToolMessage(call_id="call_1", name="sum", arguments={}, result="ok")],
    )

    with pytest.raises(ValueError, match="Encountered unresolved tool message"):
        parser.parse_nonstream(params)
