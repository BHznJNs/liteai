import json
import re
from typing import Any, Literal, cast, override
from openai import (
    AsyncOpenAI,
    AsyncStream,
    APIError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from openai.types.shared_params import FunctionDefinition
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming, CompletionCreateParamsStreaming
from pydantic import BaseModel
from .base_provider import BaseProvider, BaseMessageParser, BaseParamParser
from .exception import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderNetworkError,
    ProviderTimeoutError,
    ProviderBadRequestError,
    ContentBlockTypeNotSupportedError,
)
from .utils import StreamMessageCollector, StrictInlineJsonSchema
from ..tool.prepare import prepare_tools
from ..types import (
    LlmRequestParams,
    ContentBlock, AudioBlock, ImageBlock, TextBlock,
    BaseMessage, SystemMessage, UserMessage, AssistantMessage, ToolMessage,
    AssistantMessageEvent, StreamMessageGenerator, TextChunkEvent, ReasoningChunkEvent, ToolCallChunkEvent, UsageChunkEvent
)
from ..types.message import ResolvedToolMessage, ResolvedUserMessage


THIRD_PARTY_REASONING_CONTENT_KEY = "reasoning_content"

class OpenAIProviderMessageParser(BaseMessageParser[
    ChatCompletionChunk,
    ChatCompletion,
    ChatCompletionMessageParam,
]):
    @staticmethod
    def _content_block_to_content_part(content_block: ContentBlock) -> ChatCompletionContentPartParam:
        match content_block:
            case TextBlock():
                return ChatCompletionContentPartTextParam(
                    type="text",
                    text=content_block.text,
                )
            case ImageBlock(source=source) if source.type == "url":
                return ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": source.url,
                               "detail": "auto"})
            case ImageBlock(source=source) if source.type == "base64":
                return ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:{source.mime_type};base64,{source.data}",
                               "detail": "auto"})
            case AudioBlock(source=source) if source.type == "base64":
                extname = source.mime_type.split("/")[-1]
                if extname not in ["mp3", "wav"]:
                    raise ContentBlockTypeNotSupportedError(f"audio/{extname}")
                return ChatCompletionContentPartInputAudioParam(
                    type="input_audio",
                    input_audio={"data": source.data,
                                 "format": cast(Literal["mp3", "wav"], extname)})
            case _:
                raise ContentBlockTypeNotSupportedError(content_block.type)

    @override
    @staticmethod
    def normalize_chunk(chunk: ChatCompletionChunk) -> list[TextChunkEvent | ReasoningChunkEvent | ToolCallChunkEvent | UsageChunkEvent] | None:
        if len(chunk.choices) == 0: return None

        result = []

        if chunk.usage:
            result.append(UsageChunkEvent(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens))

        delta: ChoiceDelta | None = chunk.choices[0].delta
        if delta is None:
            # The delta will be None for some cases
            # when the provider returns empty data.
            return None
        if delta.content:
            result.append(TextChunkEvent(delta.content))
        if (reasoning := getattr(delta, THIRD_PARTY_REASONING_CONTENT_KEY, None)) is not None:
            result.append(ReasoningChunkEvent(reasoning))
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                result.append(ToolCallChunkEvent(
                    tool_call.id,
                    name=tool_call.function and tool_call.function.name,
                    arguments=tool_call.function and tool_call.function.arguments,
                    index=tool_call.index))
        return result

    @override
    @staticmethod
    def to_message(response: ChatCompletion) -> AssistantMessage:
        if (response.choices is None or # some providers may return choises as None
            len(response.choices) == 0):
            raise ValueError("Empty response")

        usage = response.usage
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        reasoning_content = getattr(message, THIRD_PARTY_REASONING_CONTENT_KEY, None)

        if message.tool_calls:
            tool_calls = [AssistantMessage.ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            ) for tool_call in message.tool_calls
              if tool_call.type == "function"]

        return AssistantMessage(
            content=message.content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=AssistantMessage.Usage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ) if usage else None,
        )

    @override
    @staticmethod
    def from_message(message: BaseMessage) -> ChatCompletionMessageParam | list[ChatCompletionMessageParam]:
        match message:
            case SystemMessage():
                return ChatCompletionSystemMessageParam(
                    role=message.role,
                    content=message.content,
                )
            case ResolvedUserMessage() if message.attachments is None:
                return ChatCompletionUserMessageParam(
                    role=message.role,
                    content=message.content,
                )
            case ResolvedUserMessage() if message.attachments is not None:
                attachment_contents = [OpenAIProviderMessageParser._content_block_to_content_part(content_block)
                                      for content_block in message.attachments]
                return ChatCompletionUserMessageParam(
                    role=message.role,
                    content=[
                        ChatCompletionContentPartTextParam(text=message.content, type="text"),
                        *attachment_contents
                    ],
                )
            case AssistantMessage():
                message_param = ChatCompletionAssistantMessageParam(
                    role=message.role,
                    content=message.content,
                )
                if message.reasoning_content is not None:
                    message_param[THIRD_PARTY_REASONING_CONTENT_KEY] = message.reasoning_content # type: ignore[typeddict-unknown-key]
                if message.tool_calls is not None:
                    tool_calls = [ChatCompletionMessageFunctionToolCallParam(
                        type="function",
                        id=tool_call.id,
                        function={
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                        },
                    ) for tool_call in message.tool_calls]
                    message_param["tool_calls"] = tool_calls
                return message_param
            case ResolvedToolMessage() as message:
                content = message.content
                if isinstance(content, list):
                    return cast(list[ChatCompletionMessageParam], [
                        ChatCompletionToolMessageParam(
                            role=message.role,
                            content="[System] Tool executed successfully. The resulting files are attached in the next user message due to API constraints. Please refer to them.",
                            tool_call_id=message.call_id,
                        ),
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=[
                                ChatCompletionContentPartTextParam(text=f"[System] Files generated by tool call (id: {message.call_id}):", type="text"),
                                *[OpenAIProviderMessageParser._content_block_to_content_part(item) for item in content]
                            ]
                        )
                    ])
                if content is None: return []
                return ChatCompletionToolMessageParam(
                    role=message.role,
                    content=content,
                    tool_call_id=message.call_id,
                )
            case ToolMessage() as message:
                raise ValueError(f"Encountered unresolved tool message: {message}")
            case UserMessage() as message:
                raise ValueError(f"Encountered unresolved user message: {message}")
            case _:
                raise NotImplementedError(f"Unsupported message type: {type(message)}")

class OpenAIProviderParamParser(BaseParamParser[
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming
]):
    def _preparse_tools(self, params: LlmRequestParams) -> list[ChatCompletionFunctionToolParam] | None:
        extracted_tool_likes = params.extract_tools()
        if extracted_tool_likes is None: return None

        tool_schemas = prepare_tools(extracted_tool_likes)
        return [ChatCompletionFunctionToolParam(
            type="function",
            function=cast(FunctionDefinition, tool_schema),
        ) for tool_schema in tool_schemas]

    def _preparse_messages(self, params: LlmRequestParams) -> list[ChatCompletionMessageParam]:
        transformed_messages: list[ChatCompletionMessageParam] = []
        pending_user_messages: list[ChatCompletionMessageParam] = []

        def flush_pending_user_messages() -> None:
            transformed_messages.extend(pending_user_messages)
            pending_user_messages.clear()

        if params.instructions is not None:
            transformed_messages.append(ChatCompletionSystemMessageParam(
                role="system",
                content=params.instructions,
            ))
        for message in params.messages:
            parsed_message = self._message_parser.from_message(message)
            if isinstance(parsed_message, list):
                if isinstance(message, ResolvedToolMessage):
                    for item in parsed_message:
                        if cast(dict[str, Any], item)["role"] == "user":
                            pending_user_messages.append(item)
                        else:
                            transformed_messages.append(item)
                else:
                    flush_pending_user_messages()
                    transformed_messages.extend(parsed_message)
            else:
                if cast(dict[str, Any], parsed_message)["role"] != "tool":
                    flush_pending_user_messages()
                transformed_messages.append(parsed_message)
        flush_pending_user_messages()
        return transformed_messages

    @override
    def parse_nonstream(self, params: LlmRequestParams) -> CompletionCreateParamsNonStreaming:
        assert params.model is not None
        result_params = CompletionCreateParamsNonStreaming(
            model=params.model,
            messages=self._preparse_messages(params),
            tool_choice=params.tool_choice,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            **(params.extra_args or {})
        )
        if (tools := self._preparse_tools(params)) is not None:
            result_params["tools"] = cast(list[ChatCompletionFunctionToolParam], tools)
        match params.output:
            case "text":
                # since text is the default response format,
                # we don't need to set it explicitly to avoid conflict with tools
                pass
            case "json":
                result_params["response_format"] = {"type": "json_object"}
            case model if issubclass(model, BaseModel):
                name = model.__name__
                description = model.__doc__ or ""
                result_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "description": description,
                        "strict": True,
                        "schema": model.model_json_schema(schema_generator=StrictInlineJsonSchema),
                    }
                }
            case _:
                raise NotImplementedError(f"Unsupported output format: {params.output}")
        return result_params

    @override
    def parse_stream(self, params: LlmRequestParams) -> CompletionCreateParamsStreaming:
        base_params = cast(CompletionCreateParamsStreaming, self.parse_nonstream(params))
        base_params["stream"] = True
        base_params["stream_options"] = {"include_usage": True}
        return base_params

class OpenAIProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str):
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._message_parser = OpenAIProviderMessageParser()
        self._param_parser = OpenAIProviderParamParser(self._message_parser)

    @staticmethod
    def _parse_thinking_content(message: AssistantMessage) -> AssistantMessage:
        def match_thinking_content(text: str) -> tuple[str, str | None]:
            pattern = r"<(think|thinking)>(.*?)</\1>"
            match = re.match(pattern, text, re.DOTALL)
            if match:
                start, end = match.span()
                thinking_content = match.group(2)
                remaining_content = text[:start] + text[end:]
                return remaining_content.strip(), thinking_content
            return text, None

        if message.content is None or message.reasoning_content is not None:
            return message.model_copy()
        content, reasoning_content = match_thinking_content(message.content)
        return message.model_copy(
            update={
                "content": content,
                "reasoning_content": reasoning_content,
            })

    @override
    async def list_models(self) -> list[str]:
        models = await self._client.models.list()
        return [model.id for model in models.data]

    @override
    async def request_nonstream(self, params: LlmRequestParams):
        parsed = self._param_parser.parse_nonstream(params)
        timeout_client = self._client.with_options(timeout=params.timeout_sec)
        try:
            response = await timeout_client.chat.completions.create(
                **parsed,
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

        message = self._message_parser.to_message(response)
        return self._parse_thinking_content(message)

    @override
    async def request_stream(self, params: LlmRequestParams) -> StreamMessageGenerator:
        parsed = self._param_parser.parse_stream(params)
        timeout_client = self._client.with_options(timeout=params.timeout_sec)

        response: AsyncStream | None = None
        try:
            response = await timeout_client.chat.completions.create(
                **parsed,
                extra_headers=params.headers,
            )
            message_collector = StreamMessageCollector()
            async for chunk in response:
                normalized_chunks = self._message_parser.normalize_chunk(chunk)
                if normalized_chunks is None: continue
                for normalized in normalized_chunks:
                    yield normalized
                    message_collector.collect(normalized)
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
        finally:
            if response:
                await response.close()

        full_message = message_collector.get_message()
        full_message = self._parse_thinking_content(full_message)
        yield AssistantMessageEvent(message=full_message)

    @override
    async def close(self):
        await self._client.close()
