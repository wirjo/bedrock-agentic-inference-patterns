"""Local MCP policy server and a synchronous client for the LangGraph notebook.

The client runs MCP's asynchronous stdio session in an AnyIO portal so notebook
nodes can keep using the existing synchronous Run.call retry/measurement wrapper.
The server exposes only a read-only policy search tool.
"""

from contextlib import asynccontextmanager, ExitStack
from datetime import timedelta
from functools import partial
import json
import os
import sys

from anyio import BrokenResourceError
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.server.mcpserver import MCPServer
    _MCP_V2 = True
except ModuleNotFoundError as error:
    if error.name != "mcp.server.mcpserver":
        raise
    from mcp.server.fastmcp import FastMCP as MCPServer
    _MCP_V2 = False

from .workshop import ROOT, Settings


def _read_timeout(seconds):
    """Use numeric seconds in MCP 2.x and timedelta in the MCP 1.x client."""
    return seconds if _MCP_V2 else timedelta(seconds=seconds)


@asynccontextmanager
async def _session(parameters):
    """Own subprocess streams and the MCP session within one asynchronous task."""
    session_closed = False
    try:
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(
                read, write, read_timeout_seconds=_read_timeout(180),
            ) as session:
                await session.initialize()
                yield session
            session_closed = True
    except* BrokenResourceError:
        # MCP 1.x can forward a late reply to its already-closed receive stream.
        # Ignore only that transport cleanup race after a clean session exit;
        # startup, RPC errors, and other exceptions must still propagate.
        if _MCP_V2 or not session_closed:
            raise


class PolicyMCPClient:
    """Start a local policy server and expose its MCP tools as ordinary callables."""

    def __init__(self, settings, server_parameters=None):
        """Select the local server command; tests can supply a fixture server."""
        self.parameters = server_parameters or StdioServerParameters(
            command=sys.executable, args=["-m", "utils.rag_mcp"], cwd=str(ROOT),
            env={
                **os.environ,
                "AWS_REGION": settings.region,
                "KNOWLEDGE_BASE_ID": settings.knowledge_base_id,
                # A separate index keeps the server's storage independent of notebook 03.
                "BEDROCK_CHROMA_PATH": os.getenv(
                    "BEDROCK_MCP_CHROMA_PATH", str(ROOT / ".chroma/mcp")
                ),
            },
        )
        self.stack = ExitStack()

    def __enter__(self):
        """Initialize the MCP connection before the timed agent workflow begins."""
        try:
            self.portal = self.stack.enter_context(start_blocking_portal())
            self.session = self.stack.enter_context(
                self.portal.wrap_async_context_manager(_session(self.parameters))
            )
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, exc_type, exc, traceback):
        """Close the session, subprocess, and portal even when the graph fails."""
        self.stack.close()
        return False

    def tools(self):
        """Discover schemas using MCP wire-field names, shared by SDK 1.x and 2.x."""
        response = self.portal.call(self.session.list_tools)
        return [
            {
                "name": tool["name"],
                "description": tool.get("description") or tool["name"],
                "parameters": {**tool["inputSchema"], "additionalProperties": False},
            }
            for tool in response.model_dump(by_alias=True)["tools"]
        ]

    def call_tool(self, name, arguments, timeout=60):
        """Call an MCP tool, preserving the caller's RPC timeout and surfacing errors."""
        result = self.portal.call(partial(
            self.session.call_tool, name, arguments,
            read_timeout_seconds=_read_timeout(timeout),
        )).model_dump(by_alias=True)
        if result.get("isError", False):
            text = " ".join(item.get("text", "") for item in result["content"])
            raise RuntimeError(f"MCP tool failed: {text}")
        structured = result.get("structuredContent")
        if structured is not None:
            return structured
        text = "".join(item.get("text", "") for item in result["content"])
        return json.loads(text)


def make_server(retriever):
    """Expose the same retrieval backend used in notebook 03 as one MCP tool."""
    server = MCPServer("Bedrock support policies", log_level="WARNING")

    @server.tool()
    def search_support_policy(query: str) -> dict:
        """Search support policies for evidence needed to answer a customer question."""
        response = retriever.retrieve(query, result_limit=3)
        sources = []
        for item in response["retrievalResults"]:
            text = item["content"].get("text")
            if text:
                sources.append({"source_id": f"S{len(sources) + 1}", "text": text})
        return {"backend": retriever.backend, "sources": sources}

    return server


if __name__ == "__main__":
    from .retrieval import create_retriever

    make_server(create_retriever(Settings.from_env())).run(transport="stdio")
