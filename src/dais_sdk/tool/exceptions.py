import json
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .types import ToolLike

class McpConnectionErrorCode(StrEnum):
    CONNECTION_TIMEOUT = "MCP_CONNECTION_TIMEOUT"

    # remote
    CONNECTION_FAILED = "MCP_CONNECTION_FAILED"
    AUTH_FAILED = "MCP_AUTH_FAILED"
    PROTOCOL_ERROR = "MCP_PROTOCOL_ERROR"

    # local
    COMMAND_NOT_FOUND = "MCP_COMMAND_NOT_FOUND"
    PERMISSION_DENIED = "MCP_PERMISSION_DENIED"
    PROCESS_START_FAILED = "MCP_PROCESS_START_FAILED"
    PROCESS_CRASHED = "MCP_PROCESS_CRASHED"

    @classmethod
    def from_exception(cls, e: BaseException) -> McpConnectionErrorCode:
        import asyncio
        import subprocess
        import anyio
        import httpx
        import mcp
        match e:
            case TimeoutError() | asyncio.TimeoutError() | httpx.ConnectTimeout():
                return cls.CONNECTION_TIMEOUT
            case httpx.ConnectError():
                return cls.CONNECTION_FAILED
            case httpx.HTTPStatusError() as e if e.response.status_code == 401:
                return cls.AUTH_FAILED
            case mcp.McpError():
                return cls.PROTOCOL_ERROR
            case FileNotFoundError():
                return cls.COMMAND_NOT_FOUND
            case PermissionError():
                return cls.PERMISSION_DENIED
            case subprocess.SubprocessError():
                return cls.PROCESS_START_FAILED
            case BrokenPipeError() | anyio.EndOfStream():
                return cls.PROCESS_CRASHED
            case _:
                return cls.CONNECTION_FAILED

class McpConnectionError(Exception):
    def __init__(self, code: McpConnectionErrorCode):
        super().__init__(code)
        self.error_code = code


class LlmToolException(Exception): ...

class ToolDoesNotExistError(LlmToolException):
    def __init__(self, tool_name: str):
        super().__init__(f"Tool does not exist: '{tool_name}'")
        self.tool_name = tool_name

class ToolArgumentDecodeError(LlmToolException):
    def __init__(self, tool_name: str, arguments: str, raw_error: json.JSONDecodeError):
        super().__init__("Tool argument decode error: ", raw_error)
        self.tool_name = tool_name
        self.arguments = arguments
        self.raw_error = raw_error

class ToolExecutionError(LlmToolException):
    def __init__(self, tool: ToolLike, arguments: str | dict, raw_error: Exception):
        super().__init__("Tool execution error: ", raw_error)
        self.tool = tool
        self.arguments = arguments
        self.raw_error = raw_error
