from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from dais_sdk.core.llm import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.providers import openai_responses as openai_responses_module
from dais_sdk.providers.openai_responses import (
    OpenAIResponsesProvider,
    OpenAIResponsesProviderMessageParser,
    OpenAIResponsesProviderParamParser,
)
from dais_sdk.providers.exception import ContentBlockTypeNotSupportedError
from dais_sdk.types.event import AssistantMessageEvent, TextChunkEvent, UsageChunkEvent
from dais_sdk.types.message import AssistantMessage, ResolvedToolMessage, ResolvedUserMessage, ToolMessage, UserMessage
from dais_sdk.types.content_block import AudioBlock, Base64Source, DocumentBlock, ImageBlock, TextBlock, UrlSource
from dais_sdk.types.request_params import LlmRequestParams


class AnswerSchema(BaseModel):
    answer: str


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


class FakeResponsesAPI:
    def __init__(self, response: Any):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, *, model_ids: list[str], response: Any):
        self.models = FakeModelsAPI(model_ids)
        self.responses = FakeResponsesAPI(response)

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


def _build_parser() -> OpenAIResponsesProviderParamParser:
    return OpenAIResponsesProviderParamParser(OpenAIResponsesProviderMessageParser())


def test_parse_nonstream_maps_core_fields_for_responses_api() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4.1-mini",
        instructions="Follow policy",
        messages=[ResolvedUserMessage(content="hello")],
        tool_choice="required",
        temperature=0.3,
        max_tokens=128,
        extra_args={"top_p": 0.9},
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    assert parsed["model"] == "gpt-4.1-mini"
    assert parsed["instructions"] == "Follow policy"
    assert parsed["tool_choice"] == "required"
    assert parsed["temperature"] == 0.3
    assert parsed["max_output_tokens"] == 128
    assert parsed["top_p"] == 0.9

    input_items = cast(list[dict[str, Any]], parsed["input"])
    assert len(input_items) == 1
    assert input_items[0]["role"] == "user"
    assert input_items[0]["content"] == [{"type": "input_text", "text": "hello"}]


def test_from_message_resolved_tool_message_string_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="sum",
        arguments={"x": 1},
        content="ok",
        is_error=False,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    assert parsed["type"] == "function_call_output"
    assert parsed["call_id"] == "call_1"
    assert parsed["output"] == "ok"
    assert parsed["status"] == "completed"


def test_from_message_resolved_tool_message_content_block_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="make_file",
        arguments={},
        content=[TextBlock(text="hello")],
        is_error=False,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    assert parsed["type"] == "function_call_output"
    assert parsed["call_id"] == "call_1"
    assert parsed["status"] == "completed"
    output = cast(list[dict[str, Any]], parsed["output"])
    assert output[0]["type"] == "input_text"
    assert output[0]["text"] == "hello"


def test_from_message_resolved_tool_message_maps_image_url_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="make_file",
        arguments={},
        content=[ImageBlock(source=UrlSource(url="https://example.com/image.png"))],
        is_error=False,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    output = cast(list[dict[str, Any]], parsed["output"])
    assert output[0]["type"] == "input_image"
    assert output[0]["image_url"] == "https://example.com/image.png"
    assert output[0]["detail"] == "auto"


def test_from_message_resolved_tool_message_maps_image_base64_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="make_file",
        arguments={},
        content=[ImageBlock(source=Base64Source(mime_type="image/png", data="abc"))],
        is_error=False,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    output = cast(list[dict[str, Any]], parsed["output"])
    assert output[0]["type"] == "input_image"
    assert output[0]["image_url"] == "data:image/png;base64,abc"


def test_from_message_resolved_tool_message_maps_document_url_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="make_file",
        arguments={},
        content=[DocumentBlock(source=UrlSource(url="https://example.com/file.pdf"))],
        is_error=False,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    output = cast(list[dict[str, Any]], parsed["output"])
    assert output[0]["type"] == "input_file"
    assert output[0]["file_url"] == "https://example.com/file.pdf"


def test_from_message_resolved_tool_message_maps_document_base64_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="make_file",
        arguments={},
        content=[DocumentBlock(source=Base64Source(mime_type="application/pdf", data="abc"))],
        is_error=False,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    output = cast(list[dict[str, Any]], parsed["output"])
    assert output[0]["type"] == "input_file"
    assert output[0]["file_data"] == "data:application/pdf;base64,abc"


def test_from_message_resolved_tool_message_rejects_audio_output() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="make_file",
        arguments={},
        content=[AudioBlock(source=Base64Source(mime_type="audio/wav", data="abc"))],
        is_error=False,
    )

    with pytest.raises(ContentBlockTypeNotSupportedError):
        OpenAIResponsesProviderMessageParser.from_message(tool_msg)


def test_from_message_resolved_tool_message_error_output_status_stays_completed() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="call_1",
        name="sum",
        arguments={},
        content='{"error": "boom"}',
        is_error=True,
    )

    parsed = cast(dict[str, Any], OpenAIResponsesProviderMessageParser.from_message(tool_msg))

    assert parsed["status"] == "completed"
    assert parsed["output"] == '{"error": "boom"}'


def test_from_message_rejects_unresolved_tool_message_for_responses_api() -> None:
    with pytest.raises(ValueError, match="Encountered unresolved tool message"):
        OpenAIResponsesProviderMessageParser.from_message(
            ToolMessage(call_id="call_1", name="sum", arguments={}, result="ok")
        )


def test_parse_nonstream_with_tools_injects_function_tools() -> None:
    parser = _build_parser()

    def search_docs(query: str) -> str:
        """Search docs by query."""
        return query

    params = LlmRequestParams(
        model="gpt-4.1-mini",
        messages=[ResolvedUserMessage(content="hello")],
        tools=[search_docs],
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    assert "tools" in parsed
    tools = cast(list[dict[str, Any]], parsed["tools"])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "search_docs"
    assert tools[0]["description"] == "Search docs by query."


def test_parse_nonstream_with_pydantic_output_maps_to_text_json_schema() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4.1-mini",
        messages=[ResolvedUserMessage(content="hello")],
        output=AnswerSchema,
    )

    parsed = cast(dict[str, Any], parser.parse_nonstream(params))

    text_config = cast(dict[str, Any], parsed["text"])
    assert text_config["format"]["type"] == "json_schema"
    assert text_config["format"]["name"] == "AnswerSchema"
    assert text_config["format"]["strict"] is True
    assert text_config["format"]["schema"]["title"] == "AnswerSchema"


def test_parse_stream_sets_stream_flag() -> None:
    parser = _build_parser()
    params = LlmRequestParams(
        model="gpt-4.1-mini",
        messages=[ResolvedUserMessage(content="hello")],
    )

    parsed = cast(dict[str, Any], parser.parse_stream(params))

    assert parsed["stream"] is True
    assert parsed["model"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_openai_responses_provider_list_models_uses_client_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeClient(model_ids=["gpt-4.1", "gpt-4.1-mini"], response=None)
    monkeypatch.setattr(openai_responses_module, "AsyncOpenAI", lambda **_: fake_client)

    provider = OpenAIResponsesProvider(base_url="https://example.com/v1", api_key="test-key")

    result = await provider.list_models()

    assert result == ["gpt-4.1", "gpt-4.1-mini"]
    assert fake_client.models.called is True


@pytest.mark.asyncio
async def test_openai_responses_provider_request_nonstream_calls_responses_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_response = SimpleNamespace(tag="nonstream-response")
    fake_client = FakeClient(model_ids=[], response=raw_response)
    monkeypatch.setattr(openai_responses_module, "AsyncOpenAI", lambda **_: fake_client)

    provider = OpenAIResponsesProvider(base_url="https://example.com/v1", api_key="test-key")
    params = LlmRequestParams(
        model="gpt-4.1-mini",
        messages=[UserMessage(content="hello")],
        headers={"x-test": "1"},
    )

    stub_param_parser = StubParamParser(
        nonstream={"model": "mock-model", "input": []},
        stream={"model": "mock-model", "input": [], "stream": True},
    )
    stub_message_parser = StubMessageParser(nonstream_message=AssistantMessage(content="final answer"))

    provider._param_parser = cast(Any, stub_param_parser)
    provider._message_parser = cast(Any, stub_message_parser)

    result = await provider.request_nonstream(params)

    assert stub_param_parser.nonstream_called_with is params
    assert stub_message_parser.to_message_called_with is raw_response
    assert result.content == "final answer"

    create_calls = fake_client.responses.calls
    assert len(create_calls) == 1
    assert create_calls[0]["model"] == "mock-model"
    assert create_calls[0]["input"] == []
    assert create_calls[0]["extra_headers"] == {"x-test": "1"}


@pytest.mark.asyncio
async def test_openai_responses_provider_request_stream_yields_chunks_and_final_message_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_chunks = [
        [TextChunkEvent(content="hello ")],
        None,
        [TextChunkEvent(content="world")],
        [UsageChunkEvent(input_tokens=3, output_tokens=4, total_tokens=7)],
    ]
    fake_stream = FakeAsyncStream(stream_chunks)
    fake_client = FakeClient(model_ids=[], response=fake_stream)
    monkeypatch.setattr(openai_responses_module, "AsyncOpenAI", lambda **_: fake_client)

    provider = OpenAIResponsesProvider(base_url="https://example.com/v1", api_key="test-key")
    params = LlmRequestParams(
        model="gpt-4.1-mini",
        messages=[UserMessage(content="stream")],
        headers={"x-test": "stream"},
    )

    stub_param_parser = StubParamParser(
        nonstream={"model": "mock-model", "input": []},
        stream={"model": "mock-model", "input": [], "stream": True},
    )
    stub_message_parser = StubMessageParser(nonstream_message=AssistantMessage(content="unused"))

    provider._param_parser = cast(Any, stub_param_parser)
    provider._message_parser = cast(Any, stub_message_parser)

    events = [event async for event in provider.request_stream(params)]

    assert stub_param_parser.stream_called_with is params

    create_calls = fake_client.responses.calls
    assert len(create_calls) == 1
    assert create_calls[0]["model"] == "mock-model"
    assert create_calls[0]["input"] == []
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


def test_llm_create_provider_supports_openai_responses() -> None:
    provider = LLM.create_provider(
        LlmProviders.OPENAI_RESPONSES,
        base_url="https://example.com/v1",
        api_key="test-key",
    )

    assert isinstance(provider, OpenAIResponsesProvider)
