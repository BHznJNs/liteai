from __future__ import annotations

import asyncio
import httpx
import shutil
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any

import pytest
import pytest_asyncio

from dais_sdk.mcp_client import (
    LocalMcpClient,
    LocalServerParams,
    McpClient,
    RemoteMcpClient,
    RemoteServerParams,
)
from dais_sdk.tool.toolset.mcp_toolset import McpToolset

STREAMABLE_HTTP_HOST = "127.0.0.1"
STREAMABLE_HTTP_PORT = 3001
STREAMABLE_HTTP_URL = f"http://{STREAMABLE_HTTP_HOST}:{STREAMABLE_HTTP_PORT}/mcp"
STREAMABLE_HTTP_STARTUP_TIMEOUT = 60.0
STREAMABLE_HTTP_SHUTDOWN_TIMEOUT = 10.0

pytestmark = pytest.mark.integration


def _skip_if_no_npx() -> None:
    if shutil.which("npx") is None:
        pytest.skip("npx not available; skipping MCP everything integration tests")


def _wait_for_server(url: str, timeout: float, interval: float) -> None:
    base_url = url.rsplit("/mcp", 1)[0]  # e.g. http://localhost:3000
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(base_url, timeout=1.0)
            return
        except httpx.ConnectError, httpx.ConnectTimeout:
            time.sleep(interval)
    raise RuntimeError(
        f"MCP everything server did not become ready within {timeout}s "
        f"(checked: {base_url})"
    )

@pytest.fixture(scope="session")
def streamable_http_server():
    """
    Launch `@modelcontextprotocol/server-everything` (streamableHttp mode),
    will automatically terminate the process at the end of the test suite.
    """
    proc = subprocess.Popen(
        [
            "npx", "-y",
            "@modelcontextprotocol/server-everything",
            "streamableHttp",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    try:
        _wait_for_server(STREAMABLE_HTTP_URL, STREAMABLE_HTTP_STARTUP_TIMEOUT, 3.0)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _unwrap_schema(schema: dict[str, Any]) -> dict[str, Any]:
    for key in ("anyOf", "oneOf", "allOf"):
        candidates = schema.get(key)
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                return candidate
    return schema


def _normalized_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for entry in schema_type:
            if entry != "null":
                return entry
        return schema_type[0] if schema_type else None
    if isinstance(schema_type, str):
        return schema_type
    return None


def _sample_value(schema: dict[str, Any]) -> Any:
    schema = _unwrap_schema(schema)
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]

    schema_type = _normalized_type(schema)
    if schema_type == "string":
        return "test"
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        items = schema.get("items", {})
        min_items = int(schema.get("minItems", 0) or 0)
        if min_items > 0:
            return [_sample_value(items) for _ in range(min_items)]
        return []
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(properties, dict) and isinstance(required, list):
            return {
                name: _sample_value(properties.get(name, {}))
                for name in required
                if isinstance(name, str)
            }
        return {}
    return "test"


def _build_tool_args(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {}
    schema = _unwrap_schema(schema)
    if _normalized_type(schema) != "object":
        return {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return {}
    return {
        name: _sample_value(properties.get(name, {}))
        for name in required
        if isinstance(name, str)
    }

async def _exercise_client(client: McpClient) -> None:
    await client.connect()
    try:
        # Verify description is accessible and has correct type after connect
        assert isinstance(client.description, (str, type(None)))

        tools = await client.list_tools()
        assert tools, "Expected at least one MCP tool"
        tool = tools[0]
        print(tool)
        arguments = _build_tool_args(tool.inputSchema)
        result = await client.call_tool(tool.name, arguments or None)
        assert result.is_error is False
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_local_mcp_client_stdio_everything() -> None:
    _skip_if_no_npx()
    params = LocalServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything", "stdio"],
    )
    client = LocalMcpClient(name="everything-stdio", params=params)
    await _exercise_client(client)


@pytest.mark.asyncio
async def test_remote_mcp_client_streamable_http_everything(streamable_http_server) -> None:
    params = RemoteServerParams(
        url=STREAMABLE_HTTP_URL,
        bearer_token=None,
        oauth_params=None,
        http_headers=None,
    )
    client = RemoteMcpClient(name="everything-http", params=params)
    await _exercise_client(client)


@pytest.mark.asyncio
async def test_local_mcp_toolset_description() -> None:
    """McpToolset.description should proxy client.description after connect."""
    _skip_if_no_npx()
    params = LocalServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything", "stdio"],
    )
    client = LocalMcpClient(name="everything-stdio-ts", params=params)
    toolset = McpToolset(client)
    # Before connect, description should still be accessible (but None since not connected)
    assert toolset.description is None
    await toolset.connect()
    try:
        # After connect, description should be either a string or None
        assert isinstance(toolset.description, (str, type(None)))
        assert toolset.description == client.description
    finally:
        await toolset.disconnect()
