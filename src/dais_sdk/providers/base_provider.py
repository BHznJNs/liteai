from abc import ABC, abstractmethod
from ..types.message import AssistantMessage, BaseMessage
from ..types.event import ReasoningChunkEvent, StreamMessageGenerator, TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent
from ..types.request_params import LlmRequestParams


class BaseMessageParser[TChunk, TNonStreamResponse, TProviderMessage](ABC):
    @staticmethod
    @abstractmethod
    def normalize_chunk(chunk: TChunk) -> list[TextChunkEvent | ReasoningChunkEvent | ToolCallChunkEvent | UsageChunkEvent] | None: ...

    @staticmethod
    @abstractmethod
    def to_message(response: TNonStreamResponse) -> AssistantMessage: ...

    @staticmethod
    @abstractmethod
    def from_message(message: BaseMessage) -> TProviderMessage | list[TProviderMessage]: ...

class BaseParamParser[TNonStreamParams, TStreamParams](ABC):
    def __init__(self, message_parser: BaseMessageParser):
        self._message_parser = message_parser

    @abstractmethod
    def parse_nonstream(self, params: LlmRequestParams) -> TNonStreamParams: ...

    @abstractmethod
    def parse_stream(self, params: LlmRequestParams) -> TStreamParams: ...

class BaseProvider(ABC):
    @abstractmethod
    def __init__(self, base_url: str, api_key: str):
        ...

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    @abstractmethod
    async def request_nonstream(self, params: LlmRequestParams) -> AssistantMessage: ...

    @abstractmethod
    def request_stream(self, params: LlmRequestParams) -> StreamMessageGenerator: ...

    @abstractmethod
    async def close(self): ...
