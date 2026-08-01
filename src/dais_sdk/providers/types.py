from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.shared_params import ReasoningEffort as OpenAiReasoningEffort
    from .anthropic import ReasoningEffort as AnthropicReasoningEffort

class LlmProviders(str, Enum):
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"

type ReasoningEffort = OpenAiReasoningEffort | AnthropicReasoningEffort
