from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from .message import AssistantMessage

@dataclass(frozen=True)
class TextChunkEvent:
    content: str

@dataclass(frozen=True)
class ReasoningChunkEvent:
    content: str

@dataclass(frozen=True)
class UsageChunkEvent:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass(frozen=True)
class ToolCallChunkEvent:
    id: str | None
    name: str | None
    arguments: str | None
    index: int

@dataclass(frozen=True)
class AssistantMessageEvent:
    """
    This event is sent when the assistant message is complete.
    """
    message: AssistantMessage

type StreamMessageEvent = TextChunkEvent | ReasoningChunkEvent | UsageChunkEvent | ToolCallChunkEvent | AssistantMessageEvent
type StreamMessageGenerator = AsyncGenerator[StreamMessageEvent, None]

__all__ = [
    "StreamMessageEvent",
    "StreamMessageGenerator",
    "TextChunkEvent",
    "ReasoningChunkEvent",
    "UsageChunkEvent",
    "ToolCallChunkEvent",
    "AssistantMessageEvent",
]
