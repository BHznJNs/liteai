from dataclasses import replace
from typing import cast, override
from mcp.types import TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource, TextResourceContents, BlobResourceContents
from .toolset import Toolset
from ..exceptions import McpConnectionErrorCode, McpConnectionError
from ..types import ToolDef, ToolFunctionParameterSchema
from ...mcp_client.base_mcp_client import McpClient, Tool, ToolResult
from ...mcp_client.local_mcp_client import LocalMcpClient, LocalServerParams
from ...mcp_client.remote_mcp_client import RemoteMcpClient, RemoteServerParams, OAuthParams
from ...types import AudioBlock, Base64Source, ContentBlock, DocumentBlock, ImageBlock, TextBlock, UrlSource, VideoBlock
from ...logger import logger


class McpToolset(Toolset):
    def __init__(self, client: McpClient):
        self._client = client
        self._tools_cache: list[ToolDef] | None = None

    def _mcp_tool_to_tool_def(self, mcp_tool: Tool) -> ToolDef:
        async def wrapper(**kwargs) -> list[ContentBlock]:
            result = await self._client.call_tool(mcp_tool.name, kwargs)
            return self._format_tool_result(result)

        tool_def = ToolDef(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            parameters=cast(ToolFunctionParameterSchema, mcp_tool.inputSchema),
            execute=wrapper
        )
        return tool_def

    def _format_tool_result(self, result: ToolResult) -> list[ContentBlock]:
        is_error, content = result.is_error, result.content
        content_blocks: list[ContentBlock] = []

        if is_error:
            content_blocks.append(TextBlock(text="Error executing tool:"))

        for block in content:
            match block:
                case TextContent():
                    content_blocks.append(TextBlock(text=block.text))
                case ImageContent():
                    content_blocks.append(ImageBlock(source=Base64Source(
                        mime_type=block.mimeType,
                        data=block.data,
                    )))
                case AudioContent():
                    content_blocks.append(AudioBlock(source=Base64Source(
                        mime_type=block.mimeType,
                        data=block.data,
                    )))
                case ResourceLink():
                    details = [f"Resource Reference: {block.uri}"]
                    if block.mimeType: details.append(f"Type: {block.mimeType}")
                    if block.size: details.append(f"Size: {block.size} bytes")
                    if block.description: details.append(f"Description: {block.description}")
                    content_blocks.append(TextBlock(text="\n".join(details)))
                case EmbeddedResource() if isinstance(block.resource, TextResourceContents):
                    content_blocks.append(TextBlock(text=block.resource.text))
                case EmbeddedResource() if isinstance(block.resource, BlobResourceContents):
                    resource = block.resource
                    mime_type = resource.mimeType or "application/octet-stream"
                    source = Base64Source(mime_type=mime_type,
                                          data=resource.blob)
                    if mime_type.startswith("image/"): content_blocks.append(ImageBlock(source=source))
                    elif mime_type.startswith("audio/"): content_blocks.append(AudioBlock(source=source))
                    elif mime_type.startswith("video/"): content_blocks.append(VideoBlock(source=source))
                    else: content_blocks.append(DocumentBlock(source=source))
                case _:
                    logger.warning(f"Unknown tool result block type: {type(block)}")

        return content_blocks

    async def connect(self) -> None:
        """
        Raises:
            - McpConnectionError
        """
        def get_first_leaf_exception(e: BaseException) -> BaseException:
            if isinstance(e, BaseExceptionGroup):
                return get_first_leaf_exception(e=e.exceptions[0])
            return e

        try:
            await self._client.connect()
            await self.refresh_tools()
        except Exception as e:
            leaf = get_first_leaf_exception(e)
            logger.exception(f"MCP server connect error: {type(leaf).__name__}", exc_info=leaf)
            error_code = McpConnectionErrorCode.from_exception(leaf)
            raise McpConnectionError(error_code)

    async def disconnect(self) -> None:
        await self._client.disconnect()
        self._tools_cache = None

    async def refresh_tools(self) -> None:
        mcp_tools = await self._client.list_tools()
        self._tools_cache = [self._mcp_tool_to_tool_def(tool)
                             for tool in mcp_tools]

    @property
    @override
    def name(self) -> str:
        return self._client.name

    @property
    def connected(self) -> bool:
        return self._tools_cache is not None

    @override
    def get_tools(self, namespaced_tool_name: bool = True) -> list[ToolDef]:
        if not self.connected:
            raise RuntimeError(f"Not connected to MCP server. Call await {self.__class__.__name__}(...).connect() first")
        assert self._tools_cache is not None
        if not namespaced_tool_name:
            return list(self._tools_cache)
        return [replace(tool, name=self.format_tool_name(tool.name)) for tool in self._tools_cache]


class LocalMcpToolset(McpToolset):
    def __init__(self, name: str, params: LocalServerParams):
        client = LocalMcpClient(name, params)
        super().__init__(client)

class RemoteMcpToolset(McpToolset):
    def __init__(self, name: str, params: RemoteServerParams):
        client = RemoteMcpClient(name, params)
        super().__init__(client)
