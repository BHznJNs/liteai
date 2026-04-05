from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from dais_sdk.providers.openai import OpenAIProviderMessageParser
from dais_sdk.types import Base64Source, ImageBlock, TextBlock
from dais_sdk.types.event import TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent
from dais_sdk.types.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: tuple[int, int, int] | None = None,
) -> ChatCompletionChunk:
    usage_obj = None
    if usage is not None:
        usage_obj = SimpleNamespace(
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            total_tokens=usage[2],
        )

    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta)
    return cast(ChatCompletionChunk, SimpleNamespace(choices=[choice], usage=usage_obj))


def _empty_choice_chunk() -> ChatCompletionChunk:
    return cast(ChatCompletionChunk, SimpleNamespace(choices=[], usage=None))


def _tool_call_chunk_item(
    *,
    call_id: str,
    name: str | None,
    arguments: str | None,
    index: int,
) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        index=index,
    )


def _chat_completion(
    *,
    content: str | None,
    tool_calls: list[Any] | None = None,
    usage: tuple[int, int, int] | None = None,
) -> ChatCompletion:
    usage_obj = None
    if usage is not None:
        usage_obj = SimpleNamespace(
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            total_tokens=usage[2],
        )

    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return cast(ChatCompletion, SimpleNamespace(choices=[choice], usage=usage_obj))


def test_normalize_chunk_text_only() -> None:
    chunk = _chunk(content="hello world")

    events = OpenAIProviderMessageParser.normalize_chunk(chunk)

    assert events is not None
    assert len(events) == 1
    text_event = events[0]
    assert isinstance(text_event, TextChunkEvent)
    assert text_event.content == "hello world"


def test_normalize_chunk_tool_call_only() -> None:
    chunk = _chunk(
        tool_calls=[
            _tool_call_chunk_item(
                call_id="call_1",
                name="search",
                arguments='{"q":"python"}',
                index=0,
            )
        ]
    )

    events = OpenAIProviderMessageParser.normalize_chunk(chunk)

    assert events is not None
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ToolCallChunkEvent)
    assert event.id == "call_1"
    assert event.name == "search"
    assert event.arguments == '{"q":"python"}'
    assert event.index == 0


def test_normalize_chunk_usage_only() -> None:
    chunk = _chunk(usage=(10, 20, 30))

    events = OpenAIProviderMessageParser.normalize_chunk(chunk)

    assert events is not None
    assert len(events) == 1
    usage_event = events[0]
    assert isinstance(usage_event, UsageChunkEvent)
    assert usage_event.input_tokens == 10
    assert usage_event.output_tokens == 20
    assert usage_event.total_tokens == 30


def test_normalize_chunk_empty_choices_returns_none() -> None:
    chunk = _empty_choice_chunk()

    events = OpenAIProviderMessageParser.normalize_chunk(chunk)

    assert events is None


def test_to_message_maps_content_usage_and_tool_calls() -> None:
    response = _chat_completion(
        content="assistant output",
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="math", arguments='{"a":1,"b":2}'),
            )
        ],
        usage=(1, 2, 3),
    )

    result = OpenAIProviderMessageParser.to_message(response)

    assert result.content == "assistant output"
    assert result.usage is not None
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 2
    assert result.usage.total_tokens == 3
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "math"
    assert result.tool_calls[0].arguments == {"a": 1, "b": 2}


def test_from_message_system_and_user() -> None:
    system_msg = SystemMessage(content="system prompt")
    user_msg = UserMessage(content="hello")

    parsed_system = OpenAIProviderMessageParser.from_message(system_msg)
    parsed_user = OpenAIProviderMessageParser.from_message(user_msg)

    assert parsed_system["role"] == "system"
    assert parsed_system["content"] == "system prompt"
    assert parsed_user["role"] == "user"
    assert parsed_user["content"] == "hello"


def test_from_message_user_with_text_and_image_blocks() -> None:
    user_msg = UserMessage(
        content="hello",
        attachments=[
            TextBlock(text="extra text"),
            ImageBlock(source=Base64Source(mime_type="image/png", data="abc")),
        ],
    )

    parsed_user = cast(dict[str, Any], OpenAIProviderMessageParser.from_message(user_msg))

    assert parsed_user["role"] == "user"
    content = cast(list[dict[str, Any]], parsed_user["content"])
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "extra text"
    assert content[2]["type"] == "image_url"


def test_from_message_assistant_with_tool_calls() -> None:
    assistant = AssistantMessage()
    assistant.content = "tool request"
    assistant.tool_calls = [
        AssistantMessage.ToolCall(
            id="call_1",
            name="sum",
            arguments={"x": 1, "y": 2},
        )
    ]

    parsed = cast(dict[str, Any], OpenAIProviderMessageParser.from_message(assistant))

    assert parsed["role"] == "assistant"
    assert parsed["content"] == "tool request"
    assert "tool_calls" in parsed
    tool_calls = cast(list[dict[str, Any]], parsed["tool_calls"])
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["function"]["name"] == "sum"
    assert tool_calls[0]["function"]["arguments"] == '{"x": 1, "y": 2}'


def test_from_message_tool_complete_and_incomplete() -> None:
    complete_msg = ToolMessage(
        call_id="call_1",
        name="sum",
        arguments={"x": 1},
        result="ok",
    )

    parsed = OpenAIProviderMessageParser.from_message(complete_msg)
    assert parsed["role"] == "tool"
    assert parsed["tool_call_id"] == "call_1"
    assert parsed["content"] == "ok"

    incomplete_msg = ToolMessage(
        call_id="call_2",
        name="sum",
        arguments={"x": 2},
        result=None,
        error=None,
    )

    with pytest.raises(ValueError):
        OpenAIProviderMessageParser.from_message(incomplete_msg)
