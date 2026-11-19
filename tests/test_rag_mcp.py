"""Real local MCP transport checks; runnable under either supported SDK version."""

import sys
from time import perf_counter

import pytest
from mcp import StdioServerParameters

from utils.rag_mcp import PolicyMCPClient
from utils.workshop import ROOT, Settings


@pytest.fixture
def policy_client():
    """Use the current interpreter for both client and fixture-server SDK imports."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "tests/mcp_fixture_server.py"), str(ROOT)],
        cwd=str(ROOT),
    )
    with PolicyMCPClient(Settings(), server_parameters=parameters) as client:
        yield client


def test_discovers_tool_schema_and_returns_policy_evidence(policy_client):
    tool, = policy_client.tools()
    assert tool["name"] == "search_support_policy"
    assert tool["parameters"]["properties"]["query"]["type"] == "string"
    assert tool["parameters"]["required"] == ["query"]
    assert tool["parameters"]["additionalProperties"] is False
    result = policy_client.call_tool(tool["name"], {"query": "replacement eligibility"})
    assert result["backend"] == "OFFLINE FIXTURE"
    assert result["sources"][0]["source_id"] == "S1"
    assert "a replacement is allowed" in result["sources"][0]["text"]


def test_mcp_error_result_is_not_treated_as_policy_evidence(policy_client):
    with pytest.raises(RuntimeError, match="MCP tool failed:.*search_support_policy"):
        policy_client.call_tool("search_support_policy", {"query": "__fixture_error__"})


def test_rpc_timeout_stops_waiting_for_a_slow_tool(policy_client):
    started = perf_counter()
    with pytest.raises(Exception, match=r"(?i)timed?\s*out|timeout"):
        policy_client.call_tool(
            "search_support_policy", {"query": "__fixture_slow__"}, timeout=0.05,
        )
    assert perf_counter() - started < 1  # The fixture takes two seconds to finish.
