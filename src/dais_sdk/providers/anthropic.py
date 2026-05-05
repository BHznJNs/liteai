import json
from typing import Literal, cast, override
from anthropic import (
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from anthropic.types import ImageBlockParam, Message, MessageParam, TextBlockParam, ToolChoiceAnyParam, ToolChoiceAutoParam, ToolChoiceNoneParam, ToolParam, ToolResultBlockParam, ToolUseBlock, ToolUseBlockParam
from anthropic.types.message_create_params import MessageCreateParamsBase, MessageCreateParamsNonStreaming
from anthropic.types.tool_param import InputSchema
from anthropic.lib.streaming import ParsedMessageStreamEvent
from pydantic import BaseModel
from .base_provider import BaseMessageParser, BaseParamParser, BaseProvider
from .exception import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderNetworkError,
    ProviderTimeoutError,
    ProviderServerError,
    ContentBlockTypeNotSupportedError,
)
from .utils import StreamMessageCollector, StrictInlineJsonSchema
from ..tool.prepare import prepare_tools
from ..types import (
    LlmRequestParams,
    ContentBlock, ImageBlock, TextBlock,
    BaseMessage, SystemMessage, UserMessage, ToolMessage, AssistantMessage,
    TextChunkEvent, ReasoningChunkEvent, ToolCallChunkEvent, UsageChunkEvent, AssistantMessageEvent,
)
from ..types.message import ResolvedUserMessage


class AnthropicProviderMessageParser(BaseMessageParser[
    ParsedMessageStreamEvent,
    Message,
    MessageParam,
]):
    @staticmethod
    def _content_block_to_content_part(content_block: ContentBlock) -> TextBlockParam | ImageBlockParam:
        match content_block:
            case TextBlock():
                return TextBlockParam(type="text", text=content_block.text)
            case ImageBlock() if content_block.source.type == "url":
                # Anthropic supports URL sources for images
                return ImageBlockParam(
                    type="image",
                    source={"type": "url", "url": content_block.source.url},
                )
            case ImageBlock() if content_block.source.type == "base64":
                if content_block.source.mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                    raise ContentBlockTypeNotSupportedError(content_block.source.mime_type)
                media_type = cast(Literal["image/jpeg", "image/png", "image/gif", "image/webp"], content_block.source.mime_type)
                return ImageBlockParam(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": media_type,
                        "data": content_block.source.data
                    },
                )
            case _:
                # Anthropic does not support audio input via the messages API
                raise ContentBlockTypeNotSupportedError(content_block.type)

    @staticmethod
    def normalize_chunk(chunk: ParsedMessageStreamEvent) -> list[TextChunkEvent | ReasoningChunkEvent | ToolCallChunkEvent | UsageChunkEvent]:
        result = []
        match chunk.type:
            case "text":
                result.append(TextChunkEvent(chunk.text))
            case "thinking":
                result.append(ReasoningChunkEvent(chunk.thinking))
            case "content_block_stop" if chunk.content_block.type == "tool_use":
                result.append(ToolCallChunkEvent(
                    id=chunk.content_block.id,
                    name=chunk.content_block.name,
                    arguments=json.dumps(chunk.content_block.input),
                    index=chunk.index))
        return result

    @override
    @staticmethod
    def to_message(response: Message) -> AssistantMessage:
        content_text: str | None = None
        reasoning_content: str | None = None
        tool_calls: list[AssistantMessage.ToolCall] | None = None

        for block in response.content:
            if block.type == "text":
                if content_text is None: content_text = ""
                content_text += block.text
            elif block.type == "thinking":
                if reasoning_content is None: reasoning_content = ""
                reasoning_content += block.thinking
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(AssistantMessage.ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        usage = response.usage
        return AssistantMessage(
            content=content_text,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=AssistantMessage.Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ) if usage else None,
        )

    @override
    @staticmethod
    def from_message(message: BaseMessage) -> MessageParam:
        match message:
            case ResolvedUserMessage() if message.attachments is None:
                return MessageParam(role="user", content=message.content)

            case ResolvedUserMessage() if message.attachments is not None:
                parts: list[TextBlockParam | ImageBlockParam] = [
                    TextBlockParam(type="text", text=message.content)
                ]
                for content_block in message.attachments:
                    parts.append(
                        AnthropicProviderMessageParser._content_block_to_content_part(content_block)
                    )
                return MessageParam(role="user", content=parts)

            case AssistantMessage():
                content_blocks = []
                if message.content is not None:
                    content_blocks.append(
                        TextBlockParam(type="text", text=message.content)
                    )
                if message.tool_calls is not None:
                    for tc in message.tool_calls:
                        content_blocks.append(ToolUseBlockParam(
                            type="tool_use",
                            id=tc.id,
                            name=tc.name,
                            input=tc.arguments,
                        ))

                # Anthropic rejects an assistant message with an empty content array,
                # fall back to an empty string if there is nothing.
                if len(content_blocks) == 0:
                    content_blocks.append(TextBlockParam(type="text", text=""))

                return MessageParam(role="assistant", content=content_blocks)

            case ToolMessage() as message:
                # Anthropic tool results travel as a *user* message containing a tool_result block.
                return MessageParam(
                    role="user",
                    content=[ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=message.call_id,
                        content=message.content,
                    )],
                )

            case SystemMessage():
                raise ValueError(
                    "SystemMessage must be passed as the `system` parameter, "
                    "not in the messages array."
                )

            case UserMessage() as message:
                raise ValueError(f"Encountered unresolved user message: {message}")
            case _:
                raise NotImplementedError(
                    f"Unsupported message type: {type(message)}"
                )

class AnthropicProviderParamParser(BaseParamParser[
    MessageCreateParamsNonStreaming,
    MessageCreateParamsBase,
]):
    def _preparse_tools(self, params: LlmRequestParams) -> list[ToolParam] | None:
        extracted_tool_likes = params.extract_tools()
        if extracted_tool_likes is None: return None

        tool_schemas = prepare_tools(extracted_tool_likes)
        return [ToolParam(
            name=tool_schema["name"],
            description=tool_schema["description"],
            input_schema=cast(InputSchema, tool_schema["parameters"]),
        ) for tool_schema in tool_schemas]

    def _preparse_messages(self, params: LlmRequestParams) -> list[MessageParam]:
        transformed_messages: list[MessageParam] = []
        for message in params.messages:
            if (type(message) is ToolMessage and not message.is_complete):
                continue
            parsed_message = self._message_parser.from_message(message)
            transformed_messages.append(parsed_message)
        return transformed_messages

    @override
    def parse_nonstream(self, params: LlmRequestParams) -> MessageCreateParamsNonStreaming:
        base_params = cast(MessageCreateParamsNonStreaming, self.parse_stream(params))
        base_params["stream"] = False
        return base_params

    @override
    def parse_stream(self, params: LlmRequestParams) -> MessageCreateParamsBase:
        assert params.model is not None
        result  = MessageCreateParamsBase(
            model=params.model,
            max_tokens=32767,
            messages=self._preparse_messages(params),
            **(params.extra_args or {})
        )
        if params.instructions is not None:
            result["system"] = params.instructions
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.max_tokens is not None:
            result["max_tokens"] = params.max_tokens
        if (tools := self._preparse_tools(params)) is not None:
            result["tools"] = tools
        if params.tool_choice is not None:
            if "tools" not in result:
                # Anthropic SDK requires tools to be set if tool_choice is not "none"
                params.tool_choice = "none"
            match params.tool_choice:
                case "auto":     result["tool_choice"] = ToolChoiceAutoParam(type="auto")
                case "required": result["tool_choice"] = ToolChoiceAnyParam(type="any")
                case "none":     result["tool_choice"] = ToolChoiceNoneParam(type="none")
        if params.output is not None and params.output != "text":
            if params.output == "json":
                result["output_config"] = {"format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True
                    }
                }}
            if isinstance(params.output, BaseModel):
                result["output_config"] = {"format": {
                    "type": "json_schema",
                    "schema": params.output.model_json_schema(schema_generator=StrictInlineJsonSchema),
                }}
        return result

class AnthropicProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str):
        self._client = AsyncAnthropic(
            base_url=AnthropicProvider._base_url_preprocess(base_url),
            api_key=api_key,
        )
        self._message_parser = AnthropicProviderMessageParser()
        self._param_parser = AnthropicProviderParamParser(self._message_parser)

    @staticmethod
    def _base_url_preprocess(base_url: str) -> str:
        return base_url.rstrip("/").removesuffix("/v1")

    @override
    async def list_models(self) -> list[str]:
        models = await self._client.models.list()
        return [model.id for model in models.data]

    @override
    async def request_nonstream(self, params: LlmRequestParams):
        parsed = self._param_parser.parse_nonstream(params)
        try:
            response = await self._client.messages.create(
                **parsed,
                timeout=params.timeout_sec,
                extra_headers=params.headers,
            )
        except AuthenticationError as e:
            raise ProviderAuthenticationError(e.message) from e
        except BadRequestError as e:
            raise ProviderBadRequestError(e.message) from e
        except RateLimitError as e:
            raise ProviderRateLimitError(e.message) from e
        except APITimeoutError as e:
            raise ProviderTimeoutError(e.message) from e
        except APIConnectionError as e:
            raise ProviderNetworkError(e.message) from e
        except APIError as e:
            raise ProviderServerError(e.message) from e
        return self._message_parser.to_message(response)

    @override
    async def request_stream(self, params: LlmRequestParams):
        parsed = self._param_parser.parse_stream(params)
        message_collector = StreamMessageCollector()

        try:
            async with self._client.messages.stream(
                **parsed,
                timeout=params.timeout_sec,
                extra_headers=params.headers,
            ) as stream:
                async for event in stream:
                    normalized_chunks = self._message_parser.normalize_chunk(event)
                    for chunk in normalized_chunks:
                        yield chunk
                        message_collector.collect(chunk)

                message = await stream.get_final_message()
                yield UsageChunkEvent(
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                    total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                )

                full_message = message_collector.get_message()
                full_message.usage = AssistantMessage.Usage(
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                    total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                )
                yield AssistantMessageEvent(message=full_message)
        except AuthenticationError as e:
            raise ProviderAuthenticationError(e.message) from e
        except BadRequestError as e:
            raise ProviderBadRequestError(e.message) from e
        except RateLimitError as e:
            raise ProviderRateLimitError(e.message) from e
        except APITimeoutError as e:
            raise ProviderTimeoutError(e.message) from e
        except APIConnectionError as e:
            raise ProviderNetworkError(e.message) from e
        except APIError as e:
            raise ProviderServerError(e.message) from e

    @override
    async def close(self):
        await self._client.close()
