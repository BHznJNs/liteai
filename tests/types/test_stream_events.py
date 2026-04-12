from typing import cast

import pytest

from dais_sdk.types import (
    AssistantMessage,
    AssistantMessageEvent,
    StreamMessageEvent,
    TextChunkEvent,
    ToolCallChunkEvent,
    UsageChunkEvent,
)


def test_text_chunk_event_fields() -> None:
    event = TextChunkEvent(content="hello")

    assert event.content == "hello"


def test_usage_chunk_event_fields() -> None:
    event = UsageChunkEvent(input_tokens=10, output_tokens=20, total_tokens=30)

    assert event.input_tokens == 10
    assert event.output_tokens == 20
    assert event.total_tokens == 30


def test_tool_call_chunk_event_fields() -> None:
    event = ToolCallChunkEvent(
        id="call_1",
        name="search_docs",
        arguments='{"query":"python"}',
        index=0,
    )

    assert event.id == "call_1"
    assert event.name == "search_docs"
    assert event.arguments == '{"query":"python"}'
    assert event.index == 0


def test_assistant_message_event_fields() -> None:
    message = AssistantMessage(content="final answer")
    event = AssistantMessageEvent(message=message)

    assert event.message is message
    assert event.message.content == "final answer"


def _consume_stream_event(event: StreamMessageEvent) -> str:
    match event:
        case TextChunkEvent(content=content):
            return f"text:{content}"
        case UsageChunkEvent(input_tokens=inp, output_tokens=out, total_tokens=total):
            return f"usage:{inp}/{out}/{total}"
        case ToolCallChunkEvent(id=tool_id, name=name, arguments=args, index=index):
            return f"tool:{tool_id}:{name}:{args}:{index}"
        case AssistantMessageEvent(message=message):
            return f"assistant:{message.content}"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (TextChunkEvent(content="hi"), "text:hi"),
        (UsageChunkEvent(input_tokens=1, output_tokens=2, total_tokens=3), "usage:1/2/3"),
        (
            ToolCallChunkEvent(
                id="call_1",
                name="sum",
                arguments='{"x":1}',
                index=0,
            ),
            'tool:call_1:sum:{"x":1}:0',
        ),
        (AssistantMessageEvent(message=AssistantMessage(content="done")), "assistant:done"),
    ],
)
def test_stream_message_event_union_is_branchable(event: StreamMessageEvent, expected: str) -> None:
    consumed = _consume_stream_event(cast(StreamMessageEvent, event))

    assert consumed == expected
