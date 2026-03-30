from types import SimpleNamespace
from typing import Any, cast

import pytest

from dais_sdk.providers import openai as openai_module
from dais_sdk.providers.openai import OpenAIProvider
from dais_sdk.types.event import AssistantMessageEvent, TextChunkEvent, UsageChunkEvent
from dais_sdk.types.message import AssistantMessage, UserMessage
from dais_sdk.types.request_params import LlmRequestParams


class FakeAsyncStream:
    def __init__(self, chunks: list[Any]):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        value = self._chunks[self._index]
        self._index += 1
        return value

    async def close(self):
        pass


class FakeModelsAPI:
    def __init__(self, model_ids: list[str]):
        self._model_ids = model_ids
        self.called = False

    async def list(self):
        self.called = True
        return SimpleNamespace(data=[SimpleNamespace(id=model_id) for model_id in self._model_ids])


class FakeCompletionsAPI:
    def __init__(self, response: Any):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, *, model_ids: list[str], completion_response: Any):
        self.models = FakeModelsAPI(model_ids)
        self.chat = SimpleNamespace(completions=FakeCompletionsAPI(completion_response))

    def with_options(self, **kwargs: Any):
        return self

    async def close(self):
        pass

class StubParamParser:
    def __init__(self, *, nonstream: dict[str, Any], stream: dict[str, Any]):
        self._nonstream = nonstream
        self._stream = stream
        self.nonstream_called_with: LlmRequestParams | None = None
        self.stream_called_with: LlmRequestParams | None = None

    def parse_nonstream(self, params: LlmRequestParams) -> dict[str, Any]:
        self.nonstream_called_with = params
        return self._nonstream

    def parse_stream(self, params: LlmRequestParams) -> dict[str, Any]:
        self.stream_called_with = params
        return self._stream


class StubMessageParser:
    def __init__(self, *, nonstream_message: AssistantMessage):
        self._nonstream_message = nonstream_message
        self.to_message_called_with: Any = None

    def to_message(self, response: Any) -> AssistantMessage:
        self.to_message_called_with = response
        return self._nonstream_message

    def normalize_chunk(self, chunk: Any):
        if chunk is None:
            return None
        return cast(list[TextChunkEvent | UsageChunkEvent], chunk)


@pytest.mark.asyncio
async def test_openai_provider_list_models_uses_client_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeClient(model_ids=["gpt-4o", "gpt-4o-mini"], completion_response=None)
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **_: fake_client)

    provider = OpenAIProvider(base_url="https://example.com/v1", api_key="test-key")

    result = await provider.list_models()

    assert result == ["gpt-4o", "gpt-4o-mini"]
    assert fake_client.models.called is True


@pytest.mark.asyncio
async def test_openai_provider_request_nonstream_calls_client_and_extracts_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_response = SimpleNamespace(tag="nonstream-response")
    fake_client = FakeClient(model_ids=[], completion_response=raw_response)
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **_: fake_client)

    provider = OpenAIProvider(base_url="https://example.com/v1", api_key="test-key")

    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[UserMessage(content="hello")],
        headers={"x-test": "1"},
    )

    base_message = AssistantMessage()
    base_message.content = "<thinking>reasoning</thinking>final answer"

    stub_param_parser = StubParamParser(
        nonstream={"model": "mock-model", "messages": []},
        stream={"model": "mock-model", "messages": [], "stream": True},
    )
    stub_message_parser = StubMessageParser(nonstream_message=base_message)

    provider._param_parser = cast(Any, stub_param_parser)
    provider._message_parser = cast(Any, stub_message_parser)

    result = await provider.request_nonstream(params)

    assert stub_param_parser.nonstream_called_with is params
    assert stub_message_parser.to_message_called_with is raw_response
    assert result.content == "final answer"
    assert result.reasoning_content == "reasoning"

    create_calls = fake_client.chat.completions.calls
    assert len(create_calls) == 1
    assert create_calls[0]["model"] == "mock-model"
    assert create_calls[0]["messages"] == []
    assert create_calls[0]["extra_headers"] == {"x-test": "1"}


@pytest.mark.asyncio
async def test_openai_provider_request_stream_yields_chunks_and_final_message_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_chunks = [
        [TextChunkEvent(content="hello ")],
        None,
        [TextChunkEvent(content="world")],
        [UsageChunkEvent(input_tokens=3, output_tokens=4, total_tokens=7)],
    ]
    fake_stream = FakeAsyncStream(stream_chunks)

    fake_client = FakeClient(model_ids=[], completion_response=fake_stream)
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **_: fake_client)

    provider = OpenAIProvider(base_url="https://example.com/v1", api_key="test-key")

    params = LlmRequestParams(
        model="gpt-4o-mini",
        messages=[UserMessage(content="stream")],
        headers={"x-test": "stream"},
    )

    base_message = AssistantMessage()
    base_message.content = "unused"

    stub_param_parser = StubParamParser(
        nonstream={"model": "mock-model", "messages": []},
        stream={"model": "mock-model", "messages": [], "stream": True},
    )
    stub_message_parser = StubMessageParser(nonstream_message=base_message)

    provider._param_parser = cast(Any, stub_param_parser)
    provider._message_parser = cast(Any, stub_message_parser)

    events = [event async for event in provider.request_stream(params)]

    assert stub_param_parser.stream_called_with is params

    create_calls = fake_client.chat.completions.calls
    assert len(create_calls) == 1
    assert create_calls[0]["model"] == "mock-model"
    assert create_calls[0]["messages"] == []
    assert create_calls[0]["stream"] is True
    assert create_calls[0]["extra_headers"] == {"x-test": "stream"}

    assert len(events) == 4
    assert isinstance(events[0], TextChunkEvent)
    assert isinstance(events[1], TextChunkEvent)
    assert isinstance(events[2], UsageChunkEvent)
    assert isinstance(events[3], AssistantMessageEvent)

    final_event = cast(AssistantMessageEvent, events[3])
    assert final_event.message.content == "hello world"
    assert final_event.message.usage is not None
    assert final_event.message.usage.input_tokens == 3
    assert final_event.message.usage.output_tokens == 4
    assert final_event.message.usage.total_tokens == 7
