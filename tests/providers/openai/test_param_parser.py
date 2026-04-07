from typing import Any, cast

from dais_sdk.providers.openai import OpenAIProviderMessageParser, OpenAIProviderParamParser
from dais_sdk.types.message import ResolvedUserMessage
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
