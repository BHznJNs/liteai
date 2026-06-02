from typing import Any, cast

import pytest

from dais_sdk.core.llm import LLM
from dais_sdk.types.content_block import ContentBlockMetadata, TextBlock
from dais_sdk.types.message import MessageGroup, ResolvedToolMessage, ResolvedUserMessage, ToolMessage, UserMessage
from dais_sdk.types.request_params import LlmRequestParams


class StubContentBlockResolver:
    def __init__(self, results: dict[str, Any]):
        self._results = results

    async def resolve(self, metadata: ContentBlockMetadata):
        return self._results[cast(str, metadata["id"])]


def _build_llm(resolver: StubContentBlockResolver | None = None) -> LLM:
    return LLM(name="test-model", provider=cast(Any, object()), content_block_resolver=resolver)


@pytest.mark.asyncio
async def test_resolve_params_resolves_tool_message_string_result() -> None:
    tool_message = ToolMessage(
        id="msg_1",
        call_id="call_1",
        name="sum",
        arguments={"x": 1},
        result="ok",
        metadata={"source": "tool"},
    )
    params = LlmRequestParams(messages=[tool_message])

    resolved = await _build_llm()._resolve_params(params)

    assert resolved.model == "test-model"
    assert len(resolved.messages) == 1
    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.id == "msg_1"
    assert message.call_id == "call_1"
    assert message.name == "sum"
    assert message.arguments == {"x": 1}
    assert message.content == "ok"
    assert message.is_error is False
    assert message.metadata == {"source": "tool"}


@pytest.mark.asyncio
async def test_resolve_params_resolves_tool_message_error() -> None:
    params = LlmRequestParams(messages=[
        ToolMessage(
            call_id="call_1",
            name="sum",
            arguments={"x": 1},
            error="boom",
        )
    ])

    resolved = await _build_llm()._resolve_params(params)

    assert len(resolved.messages) == 1
    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.content == '{"error": "boom"}'
    assert message.is_error is True


@pytest.mark.asyncio
async def test_resolve_params_skips_incomplete_tool_message() -> None:
    params = LlmRequestParams(messages=[
        UserMessage(content="hi"),
        ToolMessage(call_id="call_1", name="sum", arguments={"x": 1}),
    ])

    resolved = await _build_llm()._resolve_params(params)

    assert len(resolved.messages) == 1
    assert isinstance(resolved.messages[0], ResolvedUserMessage)
    assert resolved.messages[0].content == "hi"


@pytest.mark.asyncio
async def test_resolve_params_resolves_tool_message_empty_content_blocks_with_resolver() -> None:
    params = LlmRequestParams(messages=[
        ToolMessage(call_id="call_1", name="generate_file", arguments={}, result=[])
    ])

    resolved = await _build_llm(StubContentBlockResolver({}))._resolve_params(params)

    assert len(resolved.messages) == 1
    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.content == []


@pytest.mark.asyncio
async def test_resolve_params_resolves_tool_message_empty_content_blocks_without_resolver() -> None:
    params = LlmRequestParams(messages=[
        ToolMessage(call_id="call_1", name="generate_file", arguments={}, result=[])
    ])

    resolved = await _build_llm()._resolve_params(params)

    assert len(resolved.messages) == 1
    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.content == []


@pytest.mark.asyncio
async def test_resolve_params_rejects_tool_message_content_blocks_without_resolver() -> None:
    params = LlmRequestParams(messages=[
        ToolMessage(
            call_id="call_1",
            name="generate_file",
            arguments={},
            result=[{"id": "file_1"}],
        )
    ])

    with pytest.raises(ValueError, match="LLM.content_block_resolver not set"):
        await _build_llm()._resolve_params(params)


@pytest.mark.asyncio
async def test_resolve_params_resolves_tool_message_content_blocks_with_resolver() -> None:
    metadata = {"id": "file_1"}
    resolver = StubContentBlockResolver({"file_1": TextBlock(text="file content")})
    params = LlmRequestParams(messages=[
        ToolMessage(
            id="msg_1",
            call_id="call_1",
            name="generate_file",
            arguments={},
            result=[metadata],
            metadata={"source": "tool"},
        )
    ])

    resolved = await _build_llm(resolver)._resolve_params(params)

    assert len(resolved.messages) == 1
    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.id == "msg_1"
    assert message.call_id == "call_1"
    assert message.name == "generate_file"
    assert message.arguments == {}
    assert message.content == [TextBlock(text="file content")]
    assert message.is_error is False
    assert message.metadata == {"source": "tool"}


@pytest.mark.asyncio
async def test_resolve_params_flattens_resolver_content_block_lists() -> None:
    resolver = StubContentBlockResolver({
        "file_1": [TextBlock(text="a"), TextBlock(text="b")],
    })
    params = LlmRequestParams(messages=[
        ToolMessage(call_id="call_1", name="generate_file", arguments={}, result=[{"id": "file_1"}])
    ])

    resolved = await _build_llm(resolver)._resolve_params(params)

    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.content == [TextBlock(text="a"), TextBlock(text="b")]


@pytest.mark.asyncio
async def test_resolve_params_skips_none_resolver_results() -> None:
    resolver = StubContentBlockResolver({
        "file_1": None,
        "file_2": TextBlock(text="ok"),
    })
    params = LlmRequestParams(messages=[
        ToolMessage(
            call_id="call_1",
            name="generate_file",
            arguments={},
            result=[{"id": "file_1"}, {"id": "file_2"}],
        )
    ])

    resolved = await _build_llm(resolver)._resolve_params(params)

    message = resolved.messages[0]
    assert isinstance(message, ResolvedToolMessage)
    assert message.content == [TextBlock(text="ok")]


@pytest.mark.asyncio
async def test_resolve_params_skips_tool_message_when_all_resolver_results_are_none() -> None:
    resolver = StubContentBlockResolver({"file_1": None, "file_2": None})
    params = LlmRequestParams(messages=[
        ToolMessage(
            call_id="call_1",
            name="generate_file",
            arguments={},
            result=[{"id": "file_1"}, {"id": "file_2"}],
        )
    ])

    resolved = await _build_llm(resolver)._resolve_params(params)

    assert resolved.messages == []


@pytest.mark.asyncio
async def test_resolve_params_preserves_user_and_tool_message_order() -> None:
    resolver = StubContentBlockResolver({"file_1": TextBlock(text="attachment")})
    params = LlmRequestParams(messages=[
        UserMessage(content="hi", attachments=[{"id": "file_1"}]),
        ToolMessage(call_id="call_1", name="sum", arguments={"x": 1}, result="ok"),
    ])

    resolved = await _build_llm(resolver)._resolve_params(params)

    assert len(resolved.messages) == 2
    assert isinstance(resolved.messages[0], ResolvedUserMessage)
    assert isinstance(resolved.messages[1], ResolvedToolMessage)
    assert resolved.messages[0].content == "hi"
    assert resolved.messages[1].content == "ok"


@pytest.mark.asyncio
async def test_resolve_params_resolves_tool_message_inside_message_group() -> None:
    params = LlmRequestParams(messages=[
        MessageGroup(messages=[
            ToolMessage(call_id="call_1", name="sum", arguments={"x": 1}, result="ok"),
        ])
    ])

    resolved = await _build_llm()._resolve_params(params)

    assert len(resolved.messages) == 1
    assert isinstance(resolved.messages[0], ResolvedToolMessage)
    assert resolved.messages[0].content == "ok"
