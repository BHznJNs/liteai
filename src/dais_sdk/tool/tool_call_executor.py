import asyncio
from typing import TYPE_CHECKING, Any, Callable, NamedTuple
from .execute import execute_tool
from .exceptions import LlmToolException, ToolArgumentParsingError, ToolResultSerializationError, ToolExecutionError
from ..logger import logger

if TYPE_CHECKING:
    from ..types import ToolLike, ContentBlock, ContentBlockMetadata, ContentBlockPersister


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
    result: str | list[ContentBlockMetadata] | None
    error: str | None
    raw_result: Any | None

class ToolCallExecutor:
    def __init__(self, content_block_persister: ContentBlockPersister | None = None):
        self._exception_handler = ToolExceptionHandlerManager()
        self._content_block_persister = content_block_persister

    @property
    def exception_handler(self) -> ToolExceptionHandlerManager:
        return self._exception_handler

    async def _persist_content_blocks(self, content_blocks: list[ContentBlock]) -> list[ContentBlockMetadata]:
        if len(content_blocks) == 0: return []
        if self._content_block_persister is None:
            raise ValueError("ToolCallExecutor.content_block_persister not set, not able to persist tool message resources.")
        return [
            await self._content_block_persister.persist(content_block)
            for content_block in content_blocks
        ]

    async def execute(self,
                      tool: ToolLike,
                      arguments: str | dict) -> ToolCallOutcome:
        """
        Returns:
            A tuple of (result, error, raw_result)
        """
        try:
            result, raw_result = await execute_tool(tool, arguments)
        except (ToolArgumentParsingError, ToolResultSerializationError) as e:
            raise e
        except Exception as e:
            _error = ToolExecutionError(tool, arguments, e)
            error = self._exception_handler.handle(_error)
            return ToolCallOutcome(None, error, None)

        if isinstance(result, str):
            return ToolCallOutcome(result, None, raw_result)
        try:
            persisted = await self._persist_content_blocks(result)
            return ToolCallOutcome(persisted, None, raw_result)
        except Exception as e:
            logger.exception("Failed to persist tool call contents")
            _error = ToolExecutionError(tool, arguments, e)
            error = self._exception_handler.handle(_error)
            return ToolCallOutcome(None, error, None)

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
