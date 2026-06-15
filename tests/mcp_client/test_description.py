"""Tests for McpClient.description and McpToolset.description properties."""

from typing import override

import pytest

from dais_sdk.mcp_client import McpClient
from dais_sdk.tool.toolset.mcp_toolset import McpToolset


class TestMcpClientDescription:
    """Test McpClient base class description property."""

    def test_base_mcp_client_description_defaults_to_none(self):
        """McpClient.description should default to None in the base class."""

        class ConcreteMcpClient(McpClient):
            @property
            def name(self) -> str:
                return "test"

            @override
            async def connect(self): ...

            @override
            async def disconnect(self): ...

            @override
            async def list_tools(self): return []

            @override
            async def call_tool(self, tool_name, arguments=None): ...

        client = ConcreteMcpClient()
        assert client.description is None

    def test_mcp_client_description_can_be_overridden(self):
        """Subclasses of McpClient can override description."""

        class DescribedMcpClient(McpClient):
            @property
            def name(self) -> str:
                return "described"

            @property
            @override
            def description(self) -> str | None:
                return "Custom server description"

            @override
            async def connect(self): ...

            @override
            async def disconnect(self): ...

            @override
            async def list_tools(self): return []

            @override
            async def call_tool(self, tool_name, arguments=None): ...

        client = DescribedMcpClient()
        assert client.description == "Custom server description"


class TestMcpToolsetDescription:
    """Test McpToolset.description delegates to McpClient.description."""

    def test_mcp_toolset_description_proxies_client_description(self):
        """McpToolset.description should return the client's description."""

        class StubClient(McpClient):
            @property
            def name(self) -> str:
                return "stub"

            @property
            @override
            def description(self) -> str | None:
                return "Stub server instructions"

            @override
            async def connect(self): ...

            @override
            async def disconnect(self): ...

            @override
            async def list_tools(self): return []

            @override
            async def call_tool(self, tool_name, arguments=None): ...

        client = StubClient()
        toolset = McpToolset(client)
        assert toolset.description == "Stub server instructions"

    def test_mcp_toolset_description_none_when_client_has_none(self):
        """McpToolset.description should return None when client description is None."""

        class NoDescClient(McpClient):
            @property
            def name(self) -> str:
                return "no-desc"

            @override
            async def connect(self): ...

            @override
            async def disconnect(self): ...

            @override
            async def list_tools(self): return []

            @override
            async def call_tool(self, tool_name, arguments=None): ...

        client = NoDescClient()
        toolset = McpToolset(client)
        assert toolset.description is None