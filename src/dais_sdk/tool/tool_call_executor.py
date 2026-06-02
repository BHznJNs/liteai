import asyncio
import json
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from .execute import execute_tool
from .utils import get_tool_name
from .exceptions import LlmToolException, ToolArgumentDecodeError, ToolExecutionError
from ..logger import logger

if TYPE_CHECKING:
    from ..types import ToolLike, ContentBlock


type ExceptionHandler[E: LlmToolException] = Callable[[E], str]

class ToolExceptionHandlerManager:
    def __init__(self):
        self._handlers: dict[type[LlmToolException], ExceptionHandler[Any]] = {}

    def register[E: LlmToolException](self, exception_type: type[E]):
        def decorator(handler: ExceptionHandler[E]) -> ExceptionHandler[E]:
            self.set_handler(exception_type, handler)
            return handler
        return decorator

    def set_handler[E: LlmToolException](self, exception_type: type[E], handler: ExceptionHandler[E]):
        self._handlers[exception_type] = handler

    def get_handler[E: LlmToolException](self, exception_type: type[E]) -> ExceptionHandler[E] | None:
        return self._handlers.get(exception_type)

    def handle(self, e: LlmToolException) -> str:
        def find_best_handler[E: LlmToolException](exc_type: type[E]) -> ExceptionHandler[E] | None:
            for cls in exc_type.__mro__:
                if cls in self._handlers:
                    return self._handlers[cls]
            return None

        # Searches the MRO of the exception type to make sure the subclasses of
        # the registered exception type can also be handled.
        handler = find_best_handler(type(e))
        if handler is None:
            logger.warning(f"Unhandled tool exception: {type(e).__name__}", exc_info=e)
            return f"Unhandled tool exception | {type(e).__name__}: {e}"
        return handler(e)

class ToolCallOutcome(NamedTuple):
    result: str | list[ContentBlock] | None
    error: str | None
    raw_result: Any | None

class ToolCallExecutor:
    def __init__(self):
        self._exception_handler = ToolExceptionHandlerManager()

    @property
    def exception_handler(self) -> ToolExceptionHandlerManager:
        return self._exception_handler

    async def execute(self,
                      tool: ToolLike,
                      arguments: str | dict) -> ToolCallOutcome:
        """
        Returns:
            A tuple of (result, error, raw_result)
        """
        result, error, raw_result = None, None, None
        try:
            result, raw_result = await execute_tool(tool, arguments)
        except json.JSONDecodeError as e:
            assert type(arguments) is str
            _error = ToolArgumentDecodeError(get_tool_name(tool), arguments, e)
            error = self._exception_handler.handle(_error)
        except Exception as e:
            _error = ToolExecutionError(tool, arguments, e)
            error = self._exception_handler.handle(_error)
        return ToolCallOutcome(result, error, raw_result)

    def execute_sync(self,
                     tool: ToolLike,
                     arguments: str | dict
                     ) -> ToolCallOutcome:
        """
        Synchronous wrapper of `execute`.
        """
        return asyncio.run(self.execute(tool, arguments))

__all__ = [
    "ToolCallExecutor"
]
