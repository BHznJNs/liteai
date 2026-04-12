import pytest

from dais_sdk.types import (
    Base64Source,
    ContentBlock,
    ContentBlockMetadata,
    ImageBlock,
    TextBlock,
    UrlSource,
    UserMessage,
)


def test_text_block_has_text_type_and_text_field():
    block = TextBlock(text="hello")

    assert block.type == "text"
    assert block.text == "hello"


def test_content_block_accepts_text_and_image_blocks():
    text_block: ContentBlock = TextBlock(text="hello")
    image_block: ContentBlock = ImageBlock(
        source=UrlSource(url="https://example.com/image.png")
    )

    assert text_block.type == "text"
    assert image_block.type == "image"


def test_user_message_attachments_accept_content_block_metadata():
    metadata: ContentBlockMetadata = {
        "kind": "image",
        "mime_type": "image/png",
        "data": "abc",
    }
    message = UserMessage(
        content="hello",
        attachments=[metadata],
    )

    assert message.attachments is not None
    assert message.attachments[0] == metadata
