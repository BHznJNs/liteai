from typing import Any, Literal, Mapping, Protocol
from pydantic import BaseModel


class Base64Source(BaseModel):
    type: Literal["base64"] = "base64"
    mime_type: str
    data: str

class UrlSource(BaseModel):
    type: Literal["url"] = "url"
    url: str

type FileSource = Base64Source | UrlSource

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: FileSource

class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    source: FileSource

class AudioBlock(BaseModel):
    type: Literal["audio"] = "audio"
    source: FileSource

class VideoBlock(BaseModel):
    type: Literal["video"] = "video"
    source: FileSource

type ContentBlock = TextBlock | ImageBlock | DocumentBlock | AudioBlock | VideoBlock

type ContentBlockType = Literal["text", "image", "document", "audio", "video"]
type ContentBlockMetadata = Mapping[str, Any]

class ContentBlockResolver(Protocol):
    async def resolve(self, metadata: ContentBlockMetadata) -> list[ContentBlock] | ContentBlock | None: ...

class ContentBlockPersister(Protocol):
    async def persist(self, content_block: ContentBlock) -> ContentBlockMetadata: ...

__all__ = [
    "ContentBlock",
    "ContentBlockType",
    "ContentBlockMetadata",
    "ContentBlockResolver",
    "ContentBlockPersister",

    "FileSource",
    "Base64Source",
    "UrlSource",
    "TextBlock",
    "ImageBlock",
    "DocumentBlock",
    "AudioBlock",
    "VideoBlock",
]
