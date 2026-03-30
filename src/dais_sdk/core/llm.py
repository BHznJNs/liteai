import asyncio
from typing import TYPE_CHECKING
from collections.abc import Generator
from ..providers import LlmProviders

if TYPE_CHECKING:
    from ..providers import BaseProvider
    from ..types import (
        LlmRequestParams, StreamMessageGenerator,
        StreamMessageEvent, AssistantMessage,
    )

class LLM:
    def __init__(self, name: str, provider: BaseProvider):
        self._name = name
        self._provider = provider

    @staticmethod
    def create_provider(provider_type: LlmProviders, base_url: str, api_key: str) -> BaseProvider:
        match provider_type:
            case LlmProviders.OPENAI:
                from ..providers.openai import OpenAIProvider
                return OpenAIProvider(base_url, api_key)
            case LlmProviders.ANTHROPIC:
                from ..providers.anthropic import AnthropicProvider
                return AnthropicProvider(base_url, api_key)
            case _:
                raise ValueError(f"Unsupported provider type: {provider_type}")

    async def generate_text(self, params: LlmRequestParams) -> AssistantMessage:
        params.model = params.model or self._name
        return await self._provider.request_nonstream(params)

    def generate_text_sync(self, params: LlmRequestParams) -> AssistantMessage:
        return asyncio.run(self.generate_text(params))

    async def stream_text(self, params: LlmRequestParams) -> StreamMessageGenerator:
        params.model = params.model or self._name
        async for chunk in self._provider.request_stream(params):
            yield chunk

    def stream_text_sync(self, params: LlmRequestParams) -> Generator[StreamMessageEvent, None, None]:
        with asyncio.Runner() as runner:
            gen = self.stream_text(params)
            while True:
                try:
                    chunk = runner.run(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

    async def close(self):
        await self._provider.close()
