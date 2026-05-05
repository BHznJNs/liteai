from enum import Enum
from .base_provider import BaseProvider

class LlmProviders(str, Enum):
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"

__all__ = [
    "LlmProviders",
    "BaseProvider",
]
