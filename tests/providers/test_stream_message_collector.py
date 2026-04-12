from dais_sdk.providers.utils import StreamMessageCollector, ToolCallCollector
from dais_sdk.types import TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent


def test_tool_call_collector_collect_merges_chunks_by_index() -> None:
    collector = ToolCallCollector()

    collector.collect(ToolCallChunkEvent(id="call_1", name="sea", arguments='{"q":', index=0))
    collector.collect(ToolCallChunkEvent(id=None, name="rch", arguments='"python"}', index=0))

    tool_calls = collector.get_tool_calls()

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "search"
    assert tool_calls[0].arguments == {"q": "python"}


def test_tool_call_collector_get_tool_calls_restores_json_arguments() -> None:
    collector = ToolCallCollector()

    collector.collect(ToolCallChunkEvent(id="call_1", name="sum", arguments='{"x":1}', index=0))
    collector.collect(ToolCallChunkEvent(id="call_2", name="echo", arguments='{"msg":"ok"}', index=1))

    tool_calls = collector.get_tool_calls()

    assert len(tool_calls) == 2
    assert tool_calls[0].arguments == {"x": 1}
    assert tool_calls[1].arguments == {"msg": "ok"}


def test_stream_message_collector_collect_accumulates_text_and_usage() -> None:
    collector = StreamMessageCollector()

    collector.collect(TextChunkEvent(content="hello "))
    collector.collect(TextChunkEvent(content="world"))
    collector.collect(UsageChunkEvent(input_tokens=3, output_tokens=4, total_tokens=7))

    message = collector.get_message()

    assert message.content == "hello world"
    assert message.usage is not None
    assert message.usage.input_tokens == 3
    assert message.usage.output_tokens == 4
    assert message.usage.total_tokens == 7


def test_stream_message_collector_get_message_returns_result_and_resets_state() -> None:
    collector = StreamMessageCollector()

    collector.collect(TextChunkEvent(content="first"))
    collector.collect(ToolCallChunkEvent(id="call_1", name="sum", arguments='{"x":1}', index=0))
    collector.collect(UsageChunkEvent(input_tokens=1, output_tokens=2, total_tokens=3))

    first = collector.get_message()

    assert first.content == "first"
    assert first.tool_calls is not None
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].id == "call_1"
    assert first.tool_calls[0].name == "sum"
    assert first.tool_calls[0].arguments == {"x": 1}
    assert first.usage is not None
    assert first.usage.total_tokens == 3

    collector.collect(TextChunkEvent(content="second"))

    second = collector.get_message()

    assert second.content == "second"
    assert second.tool_calls == []
    assert second.usage is None
