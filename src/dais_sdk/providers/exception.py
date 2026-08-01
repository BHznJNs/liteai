class ProviderError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class ProviderServerError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider server error: {message}")

class ProviderBadRequestError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider bad request error: {message}")

class ContentBlockTypeNotSupportedError(ProviderBadRequestError):
    def __init__(self, content_block_type: str):
        super().__init__(f"Content block type not supported: {content_block_type}")
        self.content_block_type = content_block_type

class ProviderAuthenticationError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider authentication error: {message}")

class ProviderRateLimitError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider rate limit error: {message}")

class ProviderNetworkError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider network error: {message}")

class ProviderTimeoutError(ProviderNetworkError):
    def __init__(self, message: str):
        super().__init__(f"Provider timeout error: {message}")

class UnsupportedReasoningEffort(ProviderError):
    def __init__(self, provider: str, effort: str):
        super().__init__(f'Provider "{provider}" does not support reasoning effort "{effort}"')

__all__ = [
    "ProviderError",
    "ProviderAuthenticationError",
    "ProviderBadRequestError",
    "ProviderRateLimitError",
    "ProviderServerError",
    "ProviderNetworkError",
    "ProviderTimeoutError",
    "ContentBlockTypeNotSupportedError",
    "UnsupportedReasoningEffort",
]
