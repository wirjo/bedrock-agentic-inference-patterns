"""Offline policy-tool service; used only by MCP transport tests."""

from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from utils.rag_mcp import make_server


class FixtureRetriever:
    """Exercise the real policy server with deterministic data and test failures."""

    backend = "OFFLINE FIXTURE"

    def retrieve(self, query, result_limit=3):
        """Return policy evidence, or trigger a requested transport-test condition."""
        if query == "__fixture_error__":
            raise RuntimeError("Synthetic policy retrieval failure")
        if query == "__fixture_slow__":
            time.sleep(2)
        policy = (root / "data/knowledge-base/delivery-policy.txt").read_text()
        return {"retrievalResults": [{"content": {"text": policy}}]}


if __name__ == "__main__":
    make_server(FixtureRetriever()).run(transport="stdio")
