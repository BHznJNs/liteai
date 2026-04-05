from typing import Any, cast

from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam

from dais_sdk.providers.anthropic import AnthropicProviderMessageParser
from dais_sdk.types import Base64Source, ImageBlock, TextBlock, UserMessage


def test_from_message_user_with_text_and_image_blocks() -> None:
    user_msg = UserMessage(
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
