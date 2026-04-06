import asyncio
from typing import TYPE_CHECKING
from collections.abc import Generator
from dataclasses import replace

from dais_sdk.types.content_block import ContentBlockResolver
from ..providers import LlmProviders
from ..types.message import ResolvedUserMessage, UserMessage

if TYPE_CHECKING:
    from ..providers import BaseProvider
    from ..types import (
        LlmRequestParams, StreamMessageGenerator,
        StreamMessageEvent, AssistantMessage,
    )

class LLM:
    def __init__(self,
                 name: str,
                 provider: BaseProvider,
                 content_block_resolver: ContentBlockResolver
                 ):
        self._name = name
        self._provider = provider
        self._content_block_resolver = content_block_resolver

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

    async def _resolve_params(self, params: LlmRequestParams) -> LlmRequestParams:
        resolved_messages = []
        for message in params.messages:
            if isinstance(message, UserMessage):
                if message.attachments is not None:
                    content_block_resolve_tasks = [self._content_block_resolver.resolve(attachment) for attachment in message.attachments]
                    resolved_content_blocks = await asyncio.gather(*content_block_resolve_tasks)
                else:
                    resolved_content_blocks = None
                resolved_messages.append(ResolvedUserMessage(
                    id=message.id,
                    content=message.content,
                    attachments=resolved_content_blocks,
                ))
                continue
            resolved_messages.append(message)
        new_params = replace(params, messages=resolved_messages)
        new_params.model = new_params.model or self._name
        return new_params

    async def generate_text(self, params: LlmRequestParams) -> AssistantMessage:
        resolved_params = await self._resolve_params(params)
        return await self._provider.request_nonstream(resolved_params)

    def generate_text_sync(self, params: LlmRequestParams) -> AssistantMessage:
        return asyncio.run(self.generate_text(params))

    async def stream_text(self, params: LlmRequestParams) -> StreamMessageGenerator:
        resolved_params = await self._resolve_params(params)
        async for chunk in self._provider.request_stream(resolved_params):
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
