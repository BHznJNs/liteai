class ProviderError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class ProviderServerError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider server error: {message}")

class ProviderBadRequestError(ProviderError):
    def __init__(self, message: str):
        super().__init__(f"Provider bad request error: {message}")

class AttachmentTypeNotSupportedError(ProviderBadRequestError):
    def __init__(self, attachment_type: str):
        super().__init__(f"Attachment type not supported: {attachment_type}")
        self.attachment_type = attachment_type

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
