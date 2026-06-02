import uuid
import json
from abc import ABC
from collections.abc import Callable
from typing import Annotated, Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Discriminator, Field, field_validator
from .content_block import ContentBlock, ContentBlockMetadata


class BaseMessage(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SystemMessage(BaseMessage):
    model_config = ConfigDict(json_schema_extra={
        "required": ["content", "role"]
    })

    content: str
    role: Literal["system"] = "system"


class ToolMessage(BaseMessage):
    model_config = ConfigDict(json_schema_extra={
        "required": ["call_id", "name", "arguments", "result", "error", "role", "metadata"]
    })

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str | list[ContentBlockMetadata] | None = None
    error: str | None = None
    role: Literal["tool"] = "tool"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.result is not None or self.error is not None

    @property
    def content(self) -> str | list[ContentBlockMetadata] | None:
        if self.error is not None:
            return json.dumps({"error": self.error}, ensure_ascii=False)
        return self.result

class ResolvedToolMessage(BaseMessage):
    call_id: str
    name: str
    arguments: dict[str, Any]
    content: str | list[ContentBlock]
    is_error: bool
    role: Literal["tool"] = "tool"
    metadata: dict[str, Any] = Field(default_factory=dict)

class AssistantMessage(BaseMessage):
    class ToolCall(BaseModel):
        id: str
        name: str
        arguments: dict[str, Any]

    class Usage(BaseModel):
        input_tokens: int
        output_tokens: int
        total_tokens: int

        @classmethod
        def default(cls) -> Self:
            return cls(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0)

    model_config = ConfigDict(json_schema_extra={
        "required": ["content", "reasoning_content", "tool_calls", "audio", "images", "usage", "role"]
    })

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    role: Literal["assistant"] = "assistant"

    def get_incomplete_tool_messages(self) -> list[ToolMessage] | None:
        """
        Get a incomplete tool message from the assistant message.
        The returned tool message is incomplete,
        which means it only contains the tool call id, name and arguments.
        Returns None if there is no tool call in the assistant message.
        """
        if self.tool_calls is None: return None
        results: list[ToolMessage] = []
        for tool_call in self.tool_calls:
            results.append(ToolMessage(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=None,
                error=None))
        return results

class UserMessage(BaseMessage):
    model_config = ConfigDict(json_schema_extra={
        "required": ["content", "role"]
    })

    content: str
    attachments: list[ContentBlockMetadata] | None = None
    role: Literal["user"] = "user"

class ResolvedUserMessage(BaseMessage):
    content: str
    attachments: list[ContentBlock] | None = None
    role: Literal["user"] = "user"

type Message = Annotated[
    UserMessage | AssistantMessage | SystemMessage | ToolMessage,
    Discriminator("role")
]

class MessageGroup[M: BaseMessage](BaseMessage):
    messages: list[M]

    @field_validator("messages")
    def validate_messages(cls, messages: list[M]) -> list[M]:
        for message in messages:
            if isinstance(message, MessageGroup):
                raise ValueError("MessageGroup cannot contain another MessageGroup")
        return messages

    def has(self, id: str) -> bool:
        return any(message.id == id for message in self.messages)

    def find(self, predicator: Callable[[M], bool]) -> M | None:
        for message in self.messages:
            if predicator(message):
                return message
        return None

__all__ = [
    "BaseMessage",
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "ResolvedToolMessage",
    "MessageGroup",
]
