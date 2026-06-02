import asyncio
from typing import TYPE_CHECKING, Sequence
from collections.abc import Generator
from dataclasses import replace

from ..logger import logger
from ..providers import LlmProviders
from ..types.message import ResolvedToolMessage, ResolvedUserMessage, ToolMessage, UserMessage
from ..types.content_block import ContentBlock, ContentBlockMetadata

if TYPE_CHECKING:
    from ..providers import BaseProvider
    from ..types import (
        ContentBlockResolver,
        BaseMessage,
        LlmRequestParams, StreamMessageGenerator,
        StreamMessageEvent, AssistantMessage,
    )


class LLM:
    def __init__(self,
                 name: str,
                 provider: BaseProvider,
                 content_block_resolver: ContentBlockResolver | None = None,
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
            case LlmProviders.OPENAI_RESPONSES:
                from ..providers.openai_responses import OpenAIResponsesProvider
                return OpenAIResponsesProvider(base_url, api_key)
            case LlmProviders.ANTHROPIC:
                from ..providers.anthropic import AnthropicProvider
                return AnthropicProvider(base_url, api_key)
            case _:
                raise ValueError(f"Unsupported provider type: {provider_type}")

    async def _resolve_params(self, params: LlmRequestParams) -> LlmRequestParams:
        async def resolve_content_blocks(content_block_resolver: ContentBlockResolver, metadata_list: Sequence[ContentBlockMetadata]) -> list[ContentBlock] | None:
            resolved_content_blocks: list[ContentBlock] = []
            content_block_resolve_tasks = [content_block_resolver.resolve(metadata) for metadata in metadata_list]
            content_block_resolve_results = await asyncio.gather(*content_block_resolve_tasks)
            for result in content_block_resolve_results:
                if result is None:
                    logger.warning("`None` appeared in content_block_resolver results, which will be skipped")
                    continue
                elif isinstance(result, list):
                    resolved_content_blocks.extend(result)
                else:
                    resolved_content_blocks.append(result)
            if len(metadata_list) > 0 and len(resolved_content_blocks) == 0:
                return None
            return resolved_content_blocks

        async def resolve_messages(content_block_resolver: ContentBlockResolver, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
            resolved_messages = []
            for message in messages:
                match message:
                    case UserMessage() as message:
                        resolved_content_blocks = None
                        if message.attachments is not None:
                            resolved_content_blocks = await resolve_content_blocks(content_block_resolver, message.attachments)

                        resolved_messages.append(ResolvedUserMessage(
                            id=message.id,
                            content=message.content,
                            attachments=resolved_content_blocks,
                        ))
                    case ToolMessage() as message if message.content is None:
                        pass # ignore the incomplete tool message
                    case ToolMessage() as message if message.content is not None:
                        resolved_content: str | list[ContentBlock] | None
                        if isinstance(message.content, list):
                            resolved_content = await resolve_content_blocks(content_block_resolver, message.content)
                        else:
                            resolved_content = message.content
                        if resolved_content is None: continue
                        resolved_messages.append(ResolvedToolMessage(
                            id=message.id,
                            call_id=message.call_id,
                            name=message.name,
                            arguments=message.arguments,
                            content=resolved_content,
                            is_error=message.error is not None,
                            metadata=message.metadata,
                        ))
                    case _ as message:
                        resolved_messages.append(message)
            return resolved_messages

        expanded_messages = params.expand_messages()

        if self._content_block_resolver is not None:
            resolved_messages = await resolve_messages(self._content_block_resolver, expanded_messages)
        else:
            logger.warning("LLM.content_block_resolver not set, message resources will not be uploaded.")
            resolved_messages = []
            for message in expanded_messages:
                match message:
                    case UserMessage() as message:
                        resolved_messages.append(ResolvedUserMessage(id=message.id, content=message.content))
                    case ToolMessage() as message if message.content is None:
                        pass # ignore the incomplete tool message
                    case ToolMessage() as message if message.content is not None:
                        if isinstance(message.content, list):
                            if len(message.content) > 0:
                                raise ValueError("LLM.content_block_resolver not set, not able to resolve tool message resources.")
                            resolved_content = []
                        else:
                            resolved_content = message.content
                        resolved_messages.append(ResolvedToolMessage(
                            id=message.id,
                            call_id=message.call_id,
                            name=message.name,
                            arguments=message.arguments,
                            content=resolved_content,
                            is_error=message.error is not None,
                            metadata=message.metadata,
                        ))
                    case _ as message:
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
