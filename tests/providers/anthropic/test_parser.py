from typing import Any, cast

from anthropic.types import ImageBlockParam, Message, MessageParam, TextBlockParam
import pytest

from dais_sdk.providers.anthropic import AnthropicProviderMessageParser
from dais_sdk.types import Base64Source, ImageBlock, TextBlock, UserMessage
from dais_sdk.types.message import ResolvedUserMessage


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

