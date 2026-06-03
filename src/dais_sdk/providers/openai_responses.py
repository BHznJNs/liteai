import json
from typing import cast, override
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseCompletedEvent,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseReasoningTextDeltaEvent,
    ResponseFunctionCallOutputItemParam,
    ResponseInputTextContentParam,
    ResponseInputImageContentParam,
    ResponseInputFileContentParam,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming, ResponseCreateParamsStreaming
from openai.types.responses.response_input_item_param import FunctionCallOutput
from pydantic import BaseModel
from .base_provider import BaseMessageParser, BaseParamParser, BaseProvider
from .exception import (
    ContentBlockTypeNotSupportedError,
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from .utils import StreamMessageCollector, StrictInlineJsonSchema
from ..tool.prepare import prepare_tools
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    AudioBlock,
    BaseMessage,
    ContentBlock,
    ImageBlock,
    DocumentBlock,
    LlmRequestParams,
    ReasoningChunkEvent,
    StreamMessageGenerator,
    SystemMessage,
    TextBlock,
    TextChunkEvent,
    ToolCallChunkEvent,
    ToolMessage,
    UsageChunkEvent,
    UserMessage,
)
from ..types.message import ResolvedToolMessage, ResolvedUserMessage


class OpenAIResponsesProviderMessageParser(BaseMessageParser[
    ResponseStreamEvent,
    Response,
    ResponseInputItemParam,
]):
    @staticmethod
    def _content_block_to_user_content_part(content_block: ContentBlock) -> ResponseInputContentParam:
        match content_block:
            case TextBlock():
                return ResponseInputTextParam(type="input_text", text=content_block.text)
            case ImageBlock(source=source) if source.type == "url":
                return ResponseInputImageParam(type="input_image", image_url=source.url, detail="auto")
            case ImageBlock(source=source) if source.type == "base64":
                return ResponseInputImageParam(
                    type="input_image",
                    image_url=f"data:{source.mime_type};base64,{source.data}",
                    detail="auto",
                )
            case DocumentBlock(source=source) if source.type == "url":
                return ResponseInputFileParam(
                    type="input_file",
                    file_url=source.url,
                )
            case DocumentBlock(source=source) if source.type == "base64":
                return ResponseInputFileParam(
                    type="input_file",
                    file_data=f"data:{source.mime_type};base64,{source.data}",
                )
            case _:
                raise ContentBlockTypeNotSupportedError(content_block.type)

    @staticmethod
    def _content_block_to_tool_content_part(content_block: ContentBlock) -> ResponseFunctionCallOutputItemParam:
        match content_block:
            case TextBlock():
                return ResponseInputTextContentParam(type="input_text", text=content_block.text)
            case ImageBlock(source=source) if source.type == "url":
                return ResponseInputImageContentParam(type="input_image", image_url=source.url, detail="auto")
            case ImageBlock(source=source) if source.type == "base64":
                return ResponseInputImageContentParam(
                    type="input_image",
                    image_url=f"data:{source.mime_type};base64,{source.data}",
                    detail="auto",
                )
            case DocumentBlock(source=source) if source.type == "url":
                return ResponseInputFileContentParam(
                    type="input_file",
                    file_url=source.url,
                )
            case DocumentBlock(source=source) if source.type == "base64":
                return ResponseInputFileContentParam(
                    type="input_file",
                    file_data=f"data:{source.mime_type};base64,{source.data}",
                )
            case AudioBlock():
                raise ContentBlockTypeNotSupportedError(content_block.type)
            case _:
                raise ContentBlockTypeNotSupportedError(content_block.type)

    @override
    @staticmethod
    def normalize_chunk(
        chunk: ResponseStreamEvent,
    ) -> list[TextChunkEvent | ReasoningChunkEvent | ToolCallChunkEvent | UsageChunkEvent] | None:
        result: list[TextChunkEvent | ReasoningChunkEvent | ToolCallChunkEvent | UsageChunkEvent] = []

        match chunk.type:
            case "response.output_text.delta":
                chunk = cast(ResponseTextDeltaEvent, chunk)
                result.append(TextChunkEvent(chunk.delta))
            case "response.function_call_arguments.delta":
                chunk = cast(ResponseFunctionCallArgumentsDeltaEvent, chunk)
                result.append(ToolCallChunkEvent(
                    id=chunk.item_id,
                    name=None,
                    arguments=chunk.delta,
                    index=chunk.output_index,
                ))
            case "response.function_call_arguments.done":
                chunk = cast(ResponseFunctionCallArgumentsDoneEvent, chunk)
                result.append(ToolCallChunkEvent(
                    id=chunk.item_id,
                    name=chunk.name,
                    arguments=chunk.arguments,
                    index=chunk.output_index,
                ))
            case "response.reasoning_text.delta":
                chunk = cast(ResponseReasoningTextDeltaEvent, chunk)
                result.append(ReasoningChunkEvent(chunk.delta))
            case "response.completed":
                chunk = cast(ResponseCompletedEvent, chunk)
                usage = chunk.response.usage
                if usage is not None:
                    result.append(UsageChunkEvent(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens,
                    ))

        return result or None

    @override
    @staticmethod
    def to_message(response: Response) -> AssistantMessage:
        content: str | None = None
        reasoning_content: str | None = None
        tool_calls: list[AssistantMessage.ToolCall] | None = None

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if isinstance(part, ResponseOutputText):
                        if content is None: content = ""
                        content += part.text
            elif isinstance(item, ResponseReasoningItem):
                reasoning_blocks = item.content or item.summary
                if reasoning_content is None: reasoning_content = ""
                reasoning_content += "\n".join(content.text for content in reasoning_blocks)
            elif isinstance(item, ResponseFunctionToolCall):
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(AssistantMessage.ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=json.loads(item.arguments),
                ))

        usage = response.usage
        return AssistantMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=AssistantMessage.Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ) if usage else None,
        )

    @override
    @staticmethod
    def from_message(message: BaseMessage) -> ResponseInputItemParam | list[ResponseInputItemParam]:
        match message:
            case SystemMessage():
                raise ValueError(
                    "SystemMessage must be passed as the `instructions` parameter, not in the input array."
                )
            case ResolvedUserMessage() if message.attachments is None:
                return EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=message.content)],
                )
            case ResolvedUserMessage() if message.attachments is not None:
                attachment_contents = [
                    OpenAIResponsesProviderMessageParser._content_block_to_user_content_part(content_block)
                    for content_block in message.attachments
                ]
                return EasyInputMessageParam(
                    type="message",
                    role="user",
                    content=[
                        ResponseInputTextParam(type="input_text", text=message.content),
                    ] + attachment_contents
                )
            case AssistantMessage() as message:
                messages = []
                if message.content is not None:
                    messages.append(EasyInputMessageParam(
                        type="message",
                        role="assistant",
                        content=message.content,
                    ))
                if message.reasoning_content is not None:
                    messages.append(ResponseReasoningItemParam(
                        type="reasoning",
                        status="completed",
                        id=message.id,
                        summary=[{
                            "type": "summary_text",
                            "text": message.reasoning_content,
                        }],
                    ))
                if message.tool_calls is not None:
                    messages.extend(
                        ResponseFunctionToolCallParam(
                            type="function_call",
                            status="completed",
                            call_id=tool_call.id,
                            name=tool_call.name,
                            arguments=json.dumps(tool_call.arguments, ensure_ascii=False)
                        )
                        for tool_call in message.tool_calls
                    )
                return messages
            case ResolvedToolMessage() as message:
                if isinstance(message.content, list):
                    tool_output = [
                        OpenAIResponsesProviderMessageParser._content_block_to_tool_content_part(content_block)
                        for content_block in message.content
                    ]
                else:
                    tool_output = message.content
                return FunctionCallOutput(
                    type="function_call_output",
                    status="completed",
                    call_id=message.call_id,
                    output=tool_output,
                )
            case ToolMessage() as message:
                raise ValueError(f"Encountered unresolved tool message: {message}")
            case UserMessage() as message:
                raise ValueError(f"Encountered unresolved user message: {message}")
            case _:
                raise NotImplementedError(f"Unsupported message type: {type(message)}")


class OpenAIResponsesProviderParamParser(BaseParamParser[ResponseCreateParamsNonStreaming, ResponseCreateParamsStreaming]):
    def _preparse_tools(self, params: LlmRequestParams) -> list[FunctionToolParam] | None:
        extracted_tool_likes = params.extract_tools()
        if extracted_tool_likes is None:
            return None

        tool_schemas = prepare_tools(extracted_tool_likes)
        return [FunctionToolParam(
            type="function",
            strict=True,
            name=tool_schema["name"],
            description=tool_schema["description"],
            parameters=cast(dict, tool_schema["parameters"]),
        ) for tool_schema in tool_schemas]

    def _preparse_messages(self, params: LlmRequestParams) -> ResponseInputParam:
        transformed_messages: ResponseInputParam = []
        for message in params.messages:
            parsed_messages = self._message_parser.from_message(message)
            if isinstance(parsed_messages, list):
                transformed_messages.extend(parsed_messages)
            else:
                transformed_messages.append(parsed_messages)
        return transformed_messages

    @override
    def parse_nonstream(self, params: LlmRequestParams) -> ResponseCreateParamsNonStreaming:
        assert params.model is not None
        result_params = ResponseCreateParamsNonStreaming(
            model=params.model,
            input=self._preparse_messages(params),
            tool_choice=params.tool_choice,
            **(params.extra_args or {}),
        )
        if params.instructions is not None:
            result_params["instructions"] = params.instructions
        if params.temperature is not None:
            result_params["temperature"] = params.temperature
        if params.max_tokens is not None:
            result_params["max_output_tokens"] = params.max_tokens
        if (tools := self._preparse_tools(params)) is not None:
            result_params["tools"] = tools

        match params.output:
            case "text":
                pass
            case "json":
                result_params["text"] = ResponseTextConfigParam(format={"type": "json_object"})
            case model if issubclass(model, BaseModel):
                result_params["text"] = ResponseTextConfigParam(
                    format=ResponseFormatTextJSONSchemaConfigParam(
                        type="json_schema",
                        name=model.__name__,
                        description=model.__doc__ or "",
                        strict=True,
                        schema=model.model_json_schema(schema_generator=StrictInlineJsonSchema),
                    )
                )
            case _:
                raise NotImplementedError(f"Unsupported output format: {params.output}")

        return result_params

    @override
    def parse_stream(self, params: LlmRequestParams) -> ResponseCreateParamsStreaming:
        base_params = cast(ResponseCreateParamsStreaming, self.parse_nonstream(params))
        base_params["stream"] = True
        return base_params


class OpenAIResponsesProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._message_parser = OpenAIResponsesProviderMessageParser()
        self._param_parser = OpenAIResponsesProviderParamParser(self._message_parser)

    @override
    async def list_models(self) -> list[str]:
        models = await self._client.models.list()
        return [model.id for model in models.data]

    @override
    async def request_nonstream(self, params: LlmRequestParams):
        parsed = self._param_parser.parse_nonstream(params)
        timeout_client = self._client.with_options(timeout=params.timeout_sec)
        try:
            response = await timeout_client.responses.create(
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

        return self._message_parser.to_message(cast(Response, response))

    @override
    async def request_stream(self, params: LlmRequestParams) -> StreamMessageGenerator:
        parsed = self._param_parser.parse_stream(params)
        timeout_client = self._client.with_options(timeout=params.timeout_sec)

        response: AsyncStream[ResponseStreamEvent] | None = None
        try:
            response = cast(AsyncStream[ResponseStreamEvent], await timeout_client.responses.create(
                **parsed,
                extra_headers=params.headers,
            ))
            message_collector = StreamMessageCollector()
            async for chunk in response:
                normalized_chunks = self._message_parser.normalize_chunk(chunk)
                if normalized_chunks is None:
                    continue
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

        yield AssistantMessageEvent(message=message_collector.get_message())

    @override
    async def close(self):
        await self._client.close()
