from enum import Enum
from openai.types.shared_params import ReasoningEffort as OpenAiReasoningEffort
from .anthropic import ReasoningEffort as AnthropicReasoningEffort

class LlmProviders(str, Enum):
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"

type ReasoningEffort = OpenAiReasoningEffort | AnthropicReasoningEffort
