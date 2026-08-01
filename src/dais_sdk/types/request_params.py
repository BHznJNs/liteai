from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.shared_params import ReasoningEffort as OpenAiReasoningEffort
    from ..providers.anthropic import ReasoningEffort as AnthropicReasoningEffort
    from . import ToolLike
    from ..tool.toolset import Toolset
    from .message import BaseMessage

@dataclass
class LlmRequestParams:
    messages: Sequence[BaseMessage]
    model: str | None = None
    instructions: str | None = None
    tools: list[ToolLike] | None = None
    toolsets: list[Toolset] | None = None
    tool_choice: Literal["auto", "required", "none"] = "auto"
    output: Literal["text", "json"] | type[BaseModel] = "text"
    reasoning: OpenAiReasoningEffort | AnthropicReasoningEffort | None = None

    timeout_sec: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None

    extra_args: dict[str, Any] | None = None

    _extract_tools_cache: list[ToolLike] | None = field(default=None, init=False, repr=False)

    def expand_messages(self) -> list[BaseMessage]:
        from .message import MessageGroup

        expanded_messages: list[BaseMessage] = []
        for message in self.messages:
            if isinstance(message, MessageGroup):
                expanded_messages.extend(message.messages)
            else:
                expanded_messages.append(message)
        return expanded_messages

    def extract_tools(self) -> list[ToolLike] | None:
        if self._extract_tools_cache is not None:
            return self._extract_tools_cache

        if self.tools is None and self.toolsets is None:
            return None
        tools = []
        if self.tools:
            tools = self.tools
        if self.toolsets:
            for toolset in self.toolsets:
                tools.extend(toolset.get_tools())

        self._extract_tools_cache = tools
        return tools

    def find_tool(self, tool_name: str) -> ToolLike | None:
        from ..tool.utils import find_tool_by_name

        has_tool = ((self.tools is not None and len(self.tools) > 0) or
                        (self.toolsets is not None and len(self.toolsets) > 0))
        if not has_tool: return None

        if (tools := self.extract_tools()) is None:
            return None
        return find_tool_by_name(tools, tool_name)

__all__ = [
    "LlmRequestParams",
]
