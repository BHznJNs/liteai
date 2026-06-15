import asyncio
import httpx
import webbrowser
from typing import Any, NamedTuple, override
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl, BaseModel, ConfigDict, PrivateAttr
from .oauth_server import LocalOAuthServer, OAuthCode, TokenStorage, InMemoryTokenStorage
from .base_mcp_client import McpClient, Tool, ToolResult, McpSessionNotEstablishedError
from ..logger import logger

class OAuthParams(BaseModel):
    oauth_scopes: list[str] | None
    oauth_timeout: int = 120
    _oauth_token_storage: TokenStorage = PrivateAttr(default_factory=InMemoryTokenStorage)

class RemoteServerParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    bearer_token: str | None
    oauth_params: OAuthParams | None
    http_headers: dict[str, str] | None

# --- --- --- --- --- ---

class OAuthContext(NamedTuple):
    client: httpx.AsyncClient
    server: LocalOAuthServer

    async def aclose(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.client.aclose())
            tg.create_task(self.server.stop())

class RemoteMcpClient(McpClient):
    def __init__(self,
                 name: str,
                 params: RemoteServerParams,
                 storage: TokenStorage | None = None):
        self._name = name
        self._description: str | None = None
        self._params = params
        self._session: ClientSession | None = None
        self._oauth_context = self._init_oauth()
        if self._params.oauth_params is not None and storage is not None:
            self._params.oauth_params._oauth_token_storage = storage

        self._run_task: asyncio.Task | None = None
        self._connect_error: BaseException | None = None
        self._ready_event = asyncio.Event()
        self._disconnect_event = asyncio.Event()

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def description(self) -> str | None:
        return self._description

    def _init_http_headers(self) -> dict[str, str] | None:
        if self._params.http_headers is None and self._params.bearer_token is None:
            return None
        headers = {}
        if self._params.http_headers is not None:
            headers.update(self._params.http_headers)
        if self._params.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._params.bearer_token}"
        return headers

    def _init_oauth(self) -> OAuthContext | None:
        if self._params.oauth_params is None:
            return None

        oauth_params = self._params.oauth_params

        server = LocalOAuthServer(timeout=oauth_params.oauth_timeout)
        scopes = None
        if oauth_params.oauth_scopes is not None:
            scopes = " ".join(oauth_params.oauth_scopes)

        client_provider = OAuthClientProvider(
            server_url=self._params.url,
            client_metadata=OAuthClientMetadata(
                client_name=self._name,
                redirect_uris=[AnyUrl(server.callback_url)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope=scopes,
                token_endpoint_auth_method="none",
            ),
            storage=oauth_params._oauth_token_storage,
            redirect_handler=self._handle_redirect,
            callback_handler=self._handle_oauth_callback,
        )
        client = httpx.AsyncClient(auth=client_provider,
                                   headers=self._init_http_headers(),
                                   follow_redirects=True)
        return OAuthContext(client, server)

    async def _handle_redirect(self, url: str) -> None:
        logger.info("[OAuth] Authentication required, opening browser...")
        logger.info(f"[OAuth] If browser does not open automatically, copy and open the following link: \n{url}\n")
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.error("[OAuth] Not able to open browser", exc_info=e)

    async def _handle_oauth_callback(self) -> OAuthCode:
        if self._oauth_context is None:
            raise ValueError("OAuth context not initialized")
        return await self._oauth_context.server.wait_for_code()

    async def _run(self):
        custum_http_client: httpx.AsyncClient | None = None
        if self._oauth_context is not None:
            http_client = self._oauth_context.client
            await self._oauth_context.server.start()
        else:
            http_client = httpx.AsyncClient(headers=self._init_http_headers(), follow_redirects=True)
            custum_http_client = http_client

        try:
            async with streamable_http_client(self._params.url, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    init_result = await session.initialize()
                    self._session = session
                    self._description = init_result.instructions
                    self._ready_event.set()
                    await self._disconnect_event.wait()
        except BaseException as e:
            self._connect_error = e
            self._ready_event.set()
        finally:
            self._session = None
            self._description = None
            if custum_http_client:
                await custum_http_client.aclose()

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
    async def disconnect(self):
        try:
            if self._disconnect_event:
                self._disconnect_event.set()

            if self._run_task and not self._run_task.done():
                try:
                    await self._run_task
                except Exception: pass

            if self._oauth_context:
                try:
                    await self._oauth_context.aclose()
                except* Exception: pass
        finally:
            self._ready_event.clear()
            self._disconnect_event.clear()
            self._run_task = None
            self._connect_error = None
            self._oauth_context = self._init_oauth()
