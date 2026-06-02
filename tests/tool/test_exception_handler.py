import json
import pytest
from unittest.mock import patch, MagicMock

from dais_sdk.tool.tool_call_executor import ToolExceptionHandlerManager
from dais_sdk.types import (
    ToolDef,
    ToolDoesNotExistError,
    ToolArgumentParsingError,
    ToolExecutionError,
)


class TestToolExceptionHandlerManager:
    """Tests for ToolExceptionHandlerManager class"""

    # ------------------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------------------

    def test_init_creates_empty_handlers_dict(self):
        """Test that __init__ creates an empty handlers dictionary"""
        manager = ToolExceptionHandlerManager()
        assert manager._handlers == {}
        assert isinstance(manager._handlers, dict)

    # ------------------------------------------------------------------------
    # set_handler tests
    # ------------------------------------------------------------------------

    def test_set_handler_tool_does_not_exist_error(self):
        """Test setting handler for ToolDoesNotExistError"""
        manager = ToolExceptionHandlerManager()

        def handler(e) -> str:
            return f"Tool not found: {e.tool_name}"

        manager.set_handler(ToolDoesNotExistError, handler)
        assert ToolDoesNotExistError in manager._handlers
        assert manager._handlers[ToolDoesNotExistError] == handler

    def test_set_handler_tool_argument_decode_error(self):
        """Test setting handler for ToolArgumentParsingError"""
        manager = ToolExceptionHandlerManager()

        def handler(e) -> str:
            return f"Invalid arguments for tool {e.tool_name}"

        manager.set_handler(ToolArgumentParsingError, handler)
        assert ToolArgumentParsingError in manager._handlers
        assert manager._handlers[ToolArgumentParsingError] == handler

    def test_set_handler_tool_execution_error(self):
        """Test setting handler for ToolExecutionError"""
        manager = ToolExceptionHandlerManager()

        def handler(e) -> str:
            return f"Execution failed for tool {e.tool}"

        manager.set_handler(ToolExecutionError, handler)
        assert ToolExecutionError in manager._handlers
        assert manager._handlers[ToolExecutionError] == handler

    def test_set_handler_overwrites_existing_handler(self):
        """Test that set_handler overwrites existing handler"""
        manager = ToolExceptionHandlerManager()

        def first_handler(e) -> str:
            return "First handler"

        def second_handler(e) -> str:
            return "Second handler"

        manager.set_handler(ToolDoesNotExistError, first_handler)
        manager.set_handler(ToolDoesNotExistError, second_handler)

        assert manager._handlers[ToolDoesNotExistError] == second_handler

    # ------------------------------------------------------------------------
    # register decorator tests
    # ------------------------------------------------------------------------

    def test_register_decorator_tool_does_not_exist_error(self):
        """Test register decorator for ToolDoesNotExistError"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolDoesNotExistError)
        def handler(e) -> str:
            return f"Tool '{e.tool_name}' does not exist"

        assert ToolDoesNotExistError in manager._handlers
        assert manager._handlers[ToolDoesNotExistError] == handler

    def test_register_decorator_tool_argument_decode_error(self):
        """Test register decorator for ToolArgumentParsingError"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolArgumentParsingError)
        def handler(e) -> str:
            return f"Failed to decode arguments: {e.arguments}"

        assert ToolArgumentParsingError in manager._handlers
        assert manager._handlers[ToolArgumentParsingError] == handler

    def test_register_decorator_tool_execution_error(self):
        """Test register decorator for ToolExecutionError"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolExecutionError)
        def handler(e) -> str:
            return f"Tool execution failed: {e.raw_error}"

        assert ToolExecutionError in manager._handlers
        assert manager._handlers[ToolExecutionError] == handler

    def test_register_decorator_returns_handler(self):
        """Test that register decorator returns the handler function"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolDoesNotExistError)
        def handler(e) -> str:
            return "Handled"

        # The decorator should return the handler so it can be used directly
        assert callable(handler)
        assert handler(ToolDoesNotExistError("test_tool")) == "Handled"

    # ------------------------------------------------------------------------
    # get_handler tests
    # ------------------------------------------------------------------------

    def test_get_handler_returns_handler_when_set(self):
        """Test get_handler returns the handler when it exists"""
        manager = ToolExceptionHandlerManager()

        def handler(e) -> str:
            return "Handled"

        manager.set_handler(ToolDoesNotExistError, handler)
        retrieved_handler = manager.get_handler(ToolDoesNotExistError)

        assert retrieved_handler == handler

    def test_get_handler_returns_none_when_not_set(self):
        """Test get_handler returns None when handler doesn't exist"""
        manager = ToolExceptionHandlerManager()

        result = manager.get_handler(ToolDoesNotExistError)
        assert result is None

    def test_get_handler_different_exception_types(self):
        """Test get_handler for different exception types"""
        manager = ToolExceptionHandlerManager()

        def not_exist_handler(e) -> str:
            return "Tool not found"

        def decode_handler(e) -> str:
            return "Decode error"

        manager.set_handler(ToolDoesNotExistError, not_exist_handler)
        manager.set_handler(ToolArgumentParsingError, decode_handler)

        assert manager.get_handler(ToolDoesNotExistError) == not_exist_handler
        assert manager.get_handler(ToolArgumentParsingError) == decode_handler
        assert manager.get_handler(ToolExecutionError) is None

    # ------------------------------------------------------------------------
    # handle tests
    # ------------------------------------------------------------------------

    def test_handle_with_registered_handler(self):
        """Test handle method with a registered handler"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolDoesNotExistError)
        def handler(e) -> str:
            return f"Custom: Tool {e.tool_name} not found"

        exception = ToolDoesNotExistError("my_tool")
        result = manager.handle(exception)

        assert result == "Custom: Tool my_tool not found"

    def test_handle_with_tool_argument_decode_error(self):
        """Test handle method with ToolArgumentParsingError"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolArgumentParsingError)
        def handler(e) -> str:
            return f"Bad args for {e.tool_name}: {e.arguments}"

        raw_error = json.JSONDecodeError("test", "invalid", 0)
        exception = ToolArgumentParsingError("test_tool", "invalid json", raw_error)
        result = manager.handle(exception)

        assert result == "Bad args for test_tool: invalid json"

    def test_handle_with_tool_execution_error(self):
        """Test handle method with ToolExecutionError"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolExecutionError)
        def handler(e) -> str:
            return f"Execution error: {type(e.raw_error).__name__}"

        # Create a mock tool for ToolExecutionError
        mock_tool = ToolDef(
            name="test_tool",
            description="A test tool",
            execute=lambda x: x,
        )
        raw_error = ValueError("Something went wrong")
        exception = ToolExecutionError(mock_tool, "{}", raw_error)
        result = manager.handle(exception)

        assert result == "Execution error: ValueError"

    def test_handle_without_registered_handler_logs_warning(self):
        """Test handle method logs warning when no handler is registered"""
        manager = ToolExceptionHandlerManager()

        exception = ToolDoesNotExistError("unknown_tool")

        with patch("dais_sdk.tool.tool_call_executor.logger") as mock_logger:
            result = manager.handle(exception)

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "Unhandled tool exception" in call_args[0][0]
            assert "ToolDoesNotExistError" in call_args[0][0]
            assert call_args[1]["exc_info"] == exception

    def test_handle_without_registered_handler_returns_default_message(self):
        """Test handle method returns default message when no handler is registered"""
        manager = ToolExceptionHandlerManager()

        exception = ToolDoesNotExistError("unknown_tool")

        with patch("dais_sdk.tool.tool_call_executor.logger"):
            result = manager.handle(exception)

        assert "Unhandled tool exception" in result
        assert "ToolDoesNotExistError" in result
        assert "unknown_tool" in result

    def test_handle_all_exception_types_without_handlers(self):
        """Test handle method with all exception types without handlers"""
        manager = ToolExceptionHandlerManager()

        mock_tool = ToolDef(
            name="test_tool",
            description="A test tool",
            execute=lambda x: x,
        )

        with patch("dais_sdk.tool.tool_call_executor.logger"):
            # ToolDoesNotExistError
            not_exist = ToolDoesNotExistError("tool1")
            result1 = manager.handle(not_exist)
            assert "ToolDoesNotExistError" in result1

            # ToolArgumentParsingError
            raw_error = json.JSONDecodeError("test", "bad", 0)
            decode_error = ToolArgumentParsingError("tool2", "bad json", raw_error)
            result2 = manager.handle(decode_error)
            assert "ToolArgumentParsingError" in result2

            # ToolExecutionError
            exec_error = ToolExecutionError(mock_tool, "{}", RuntimeError("fail"))
            result3 = manager.handle(exec_error)
            assert "ToolExecutionError" in result3

    # ------------------------------------------------------------------------
    # Integration tests
    # ------------------------------------------------------------------------

    def test_multiple_handlers_work_independently(self):
        """Test that multiple handlers work independently"""
        manager = ToolExceptionHandlerManager()

        mock_tool = ToolDef(
            name="test_tool",
            description="A test tool",
            execute=lambda x: x,
        )

        @manager.register(ToolDoesNotExistError)
        def not_exist_handler(e) -> str:
            return f"NotExist: {e.tool_name}"

        @manager.register(ToolArgumentParsingError)
        def decode_handler(e) -> str:
            return f"Decode: {e.tool_name}"

        @manager.register(ToolExecutionError)
        def exec_handler(e) -> str:
            return f"Exec: {e.tool.name}"

        # Test each handler
        assert manager.handle(ToolDoesNotExistError("t1")) == "NotExist: t1"

        raw_error = json.JSONDecodeError("test", "bad", 0)
        assert manager.handle(ToolArgumentParsingError("t2", "bad", raw_error)) == "Decode: t2"

        assert manager.handle(ToolExecutionError(mock_tool, "{}", Exception())) == "Exec: test_tool"

    def test_handler_receives_correct_exception_instance(self):
        """Test that handler receives the correct exception instance"""
        manager = ToolExceptionHandlerManager()
        received_exceptions = []

        @manager.register(ToolDoesNotExistError)
        def handler(e) -> str:
            received_exceptions.append(e)
            return f"Got: {e.tool_name}"

        exception = ToolDoesNotExistError("specific_tool")
        manager.handle(exception)

        assert len(received_exceptions) == 1
        assert received_exceptions[0] is exception
        assert received_exceptions[0].tool_name == "specific_tool"

    def test_handler_return_value_is_returned(self):
        """Test that handler's return value is returned by handle"""
        manager = ToolExceptionHandlerManager()

        @manager.register(ToolDoesNotExistError)
        def handler(e) -> str:
            return "custom return value"

        result = manager.handle(ToolDoesNotExistError("any"))
        assert result == "custom return value"

    def test_set_handler_and_register_are_equivalent(self):
        """Test that set_handler and register decorator achieve the same result"""
        manager1 = ToolExceptionHandlerManager()
        manager2 = ToolExceptionHandlerManager()

        # Using set_handler
        def handler1(e) -> str:
            return "Handler 1"

        manager1.set_handler(ToolDoesNotExistError, handler1)

        # Using register decorator
        @manager2.register(ToolDoesNotExistError)
        def handler2(e) -> str:
            return "Handler 2"

        # Both should have handlers registered
        assert manager1.get_handler(ToolDoesNotExistError) is not None
        assert manager2.get_handler(ToolDoesNotExistError) is not None

        # Both handlers should be callable
        assert manager1.handle(ToolDoesNotExistError("test")) == "Handler 1"
        assert manager2.handle(ToolDoesNotExistError("test")) == "Handler 2"
