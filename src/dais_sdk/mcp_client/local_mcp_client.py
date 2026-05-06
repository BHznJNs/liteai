import asyncio
from typing import Any, override
from mcp import ClientSession, StdioServerParameters as StdioServerParams
from mcp.client.stdio import stdio_client
from .base_mcp_client import McpClient, Tool, ToolResult, McpSessionNotEstablishedError

class LocalServerParams(StdioServerParams): ...

class LocalMcpClient(McpClient):
    def __init__(self, name: str, params: LocalServerParams):
        self._name: str = name
        self._params: LocalServerParams = params
        self._session: ClientSession | None = None
        self._run_task: asyncio.Task | None = None

        self._connect_error: BaseException | None = None
        self._ready_event = asyncio.Event()
        self._disconnect_event = asyncio.Event()

    async def _run(self):
        try:
            async with stdio_client(self._params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._ready_event.set()
                    await self._disconnect_event.wait()
        except BaseException as e:
            self._connect_error = e
            self._ready_event.set()
        finally:
            self._session = None

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    async def connect(self):
        self._run_task = asyncio.create_task(self._run())
        await self._ready_event.wait()
        if self._connect_error:
            raise self._connect_error

    @override
    async def list_tools(self) -> list[Tool]:
        if not self._session:
            raise McpSessionNotEstablishedError()

        result = await self._session.list_tools()
        return result.tools

    @override
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        if not self._session:
            raise McpSessionNotEstablishedError()

        response = await self._session.call_tool(tool_name, arguments)
        return ToolResult(response.isError, response.content)

    @override
    async def disconnect(self) -> None:
        try:
            if self._disconnect_event:
                self._disconnect_event.set()

            if self._run_task and not self._run_task.done():
                try:
                    await self._run_task
                except Exception: pass
        finally:
            self._ready_event.clear()
            self._disconnect_event.clear()
            self._connect_error = None
            self._run_task = None
