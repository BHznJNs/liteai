from typing import Any, cast

from anthropic.types import ImageBlockParam, Message, MessageParam, TextBlockParam
import pytest

from dais_sdk.providers.anthropic import AnthropicProviderMessageParser
from dais_sdk.types import Base64Source, DocumentBlock, ImageBlock, TextBlock, UrlSource, UserMessage
from dais_sdk.types.message import ResolvedToolMessage, ResolvedUserMessage, ToolMessage


def test_from_message_rejects_unresolved_user_message() -> None:
    with pytest.raises(ValueError, match="Encountered unresolved user message"):
        AnthropicProviderMessageParser.from_message(UserMessage(content="hello"))


def test_from_message_resolved_user_with_text_and_image_blocks() -> None:
    user_msg = ResolvedUserMessage(
        content="hello",
        attachments=[
            TextBlock(text="extra text"),
            ImageBlock(source=Base64Source(mime_type="image/png", data="abc")),
        ],
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(user_msg))

    assert parsed["role"] == "user"
    content = cast(list[TextBlockParam | ImageBlockParam], parsed["content"])
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "extra text"
    assert content[2]["type"] == "image"


def test_from_message_resolved_tool_message_string_result() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="toolu_1",
        name="sum",
        arguments={"x": 1},
        content="ok",
        is_error=False,
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(tool_msg))

    assert parsed["role"] == "user"
    content = cast(list[dict[str, Any]], parsed["content"])
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "toolu_1"
    assert content[0]["content"] == "ok"
    assert content[0]["is_error"] is False


def test_from_message_resolved_tool_message_error_result() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="toolu_1",
        name="sum",
        arguments={"x": 1},
        content='{"error": "boom"}',
        is_error=True,
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(tool_msg))

    content = cast(list[dict[str, Any]], parsed["content"])
    assert content[0]["content"] == '{"error": "boom"}'
    assert content[0]["is_error"] is True


def test_from_message_resolved_tool_message_content_blocks() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="toolu_1",
        name="make_file",
        arguments={},
        content=[
            TextBlock(text="hello"),
            ImageBlock(source=Base64Source(mime_type="image/png", data="abc")),
        ],
        is_error=False,
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(tool_msg))

    assert parsed["role"] == "user"
    content = cast(list[dict[str, Any]], parsed["content"])
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "toolu_1"
    assert content[0]["is_error"] is False
    tool_result_content = cast(list[dict[str, Any]], content[0]["content"])
    assert tool_result_content[0]["type"] == "text"
    assert tool_result_content[0]["text"] == "hello"
    assert tool_result_content[1]["type"] == "image"


def test_from_message_resolved_tool_message_document_url_block() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="toolu_1",
        name="make_file",
        arguments={},
        content=[DocumentBlock(source=UrlSource(url="https://example.com/file.pdf"))],
        is_error=False,
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(tool_msg))

    content = cast(list[dict[str, Any]], parsed["content"])
    tool_result_content = cast(list[dict[str, Any]], content[0]["content"])
    part = tool_result_content[0]
    assert part["type"] == "document"
    assert part["source"]["type"] == "url"
    assert part["source"]["url"] == "https://example.com/file.pdf"


def test_from_message_resolved_tool_message_document_base64_pdf_block() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="toolu_1",
        name="make_file",
        arguments={},
        content=[DocumentBlock(source=Base64Source(mime_type="application/pdf", data="abc"))],
        is_error=False,
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(tool_msg))

    content = cast(list[dict[str, Any]], parsed["content"])
    tool_result_content = cast(list[dict[str, Any]], content[0]["content"])
    part = tool_result_content[0]
    assert part["type"] == "document"
    assert part["source"]["type"] == "base64"
    assert part["source"]["media_type"] == "application/pdf"
    assert part["source"]["data"] == "abc"


def test_from_message_resolved_tool_message_document_base64_text_block() -> None:
    tool_msg = ResolvedToolMessage(
        call_id="toolu_1",
        name="make_file",
        arguments={},
        content=[DocumentBlock(source=Base64Source(mime_type="text/plain", data="abc"))],
        is_error=False,
    )

    parsed = cast(MessageParam, AnthropicProviderMessageParser.from_message(tool_msg))

    content = cast(list[dict[str, Any]], parsed["content"])
    tool_result_content = cast(list[dict[str, Any]], content[0]["content"])
    part = tool_result_content[0]
    assert part["type"] == "document"
    assert part["source"]["type"] == "text"
    assert part["source"]["media_type"] == "text/plain"
    assert part["source"]["data"] == "abc"


def test_from_message_rejects_unresolved_tool_message() -> None:
    with pytest.raises(ValueError, match="Encountered unresolved tool message"):
        AnthropicProviderMessageParser.from_message(
            ToolMessage(call_id="call_1", name="sum", arguments={}, result="ok")
        )


def test_to_message_accumulates_multiple_text_blocks() -> None:
    response = Message.model_validate({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    })

    message = AnthropicProviderMessageParser.to_message(response)

    assert message.content == "hello world"
    assert message.reasoning_content is None


def test_to_message_accumulates_multiple_thinking_blocks() -> None:
    response = Message.model_validate({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [
            {"type": "thinking", "thinking": "first ", "signature": "sig_1"},
            {"type": "thinking", "thinking": "second", "signature": "sig_2"},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    })

    message = AnthropicProviderMessageParser.to_message(response)

    assert message.content is None
    assert message.reasoning_content == "first second"

