import copy
import dataclasses
import json
from typing import Any, override
from pydantic.json_schema import GenerateJsonSchema
from ..types.event import ReasoningChunkEvent, TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent
from ..types.message import AssistantMessage


class StrictInlineJsonSchema(GenerateJsonSchema):
    @override
    def model_schema(self, schema):
        result = super().model_schema(schema)
        result["additionalProperties"] = False
        return result

    @override
    def generate(self, schema, mode="validation"):
        result = super().generate(schema, mode)
        return self._resolve_refs(result)

    def _resolve_refs(self, schema: dict[str, Any]) -> dict[str, Any]:
        defs = schema.get("$defs", {})
        def resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_name = node["$ref"].split("/")[-1]
                    return resolve(copy.deepcopy(defs[ref_name]))
                return {k: resolve(v) for k, v in node.items() if k != "$defs"}
            if isinstance(node, list):
                return [resolve(item) for item in node]
            return node
        return resolve(schema)


class ToolCallCollector:
    @dataclasses.dataclass
    class ToolCallTemp:
        id: str = ""
        name: str = ""
        arguments: str = ""

    def __init__(self):
        self.tool_call_map: dict[int, ToolCallCollector.ToolCallTemp] = {}

    def collect(self, tool_call_chunk: ToolCallChunkEvent):
        if tool_call_chunk.index not in self.tool_call_map:
            self.tool_call_map[tool_call_chunk.index] = ToolCallCollector.ToolCallTemp()

        temp_tool_call = self.tool_call_map[tool_call_chunk.index]
        if tool_call_chunk.id:
            temp_tool_call.id = tool_call_chunk.id
        if tool_call_chunk.name:
            temp_tool_call.name += tool_call_chunk.name
        if tool_call_chunk.arguments:
            temp_tool_call.arguments += tool_call_chunk.arguments

    def get_tool_calls(self) -> list[AssistantMessage.ToolCall]:
        return [AssistantMessage.ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            arguments=json.loads(tool_call.arguments),
        ) for tool_call in self.tool_call_map.values()]

class StreamMessageCollector:
    def __init__(self):
        self._message_buf = AssistantMessage(content=None)
        self._tool_call_collector = ToolCallCollector()

    def collect(self, chunk: TextChunkEvent | ReasoningChunkEvent | ToolCallChunkEvent | UsageChunkEvent):
        match chunk:
            case ToolCallChunkEvent():
                self._tool_call_collector.collect(chunk)
            case TextChunkEvent():
                if self._message_buf.content is None:
                    self._message_buf.content = ""
                self._message_buf.content += chunk.content
            case ReasoningChunkEvent():
                if self._message_buf.reasoning_content is None:
                    self._message_buf.reasoning_content = ""
                self._message_buf.reasoning_content += chunk.content
            case UsageChunkEvent():
                if self._message_buf.usage is None:
                    self._message_buf.usage = AssistantMessage.Usage.default()
                # since the usage chunk is the last chunk, we can directly assign the values here
                self._message_buf.usage.input_tokens = chunk.input_tokens
                self._message_buf.usage.output_tokens = chunk.output_tokens
                self._message_buf.usage.total_tokens = chunk.total_tokens

    def get_message(self) -> AssistantMessage:
        self._message_buf.tool_calls = self._tool_call_collector.get_tool_calls()
        result = self._message_buf

        # reset the message buffer
        self._message_buf = AssistantMessage()
        self._tool_call_collector = ToolCallCollector()
        return result
