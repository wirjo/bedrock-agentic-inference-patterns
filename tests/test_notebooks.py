import nbformat
import pytest
import sys
from mcp import StdioServerParameters

from utils import workshop as w
from utils import rag_mcp
from offline_backend import OfflineBackend
from native_offline_backend import NativeOfflineBackend


@pytest.fixture(params=["openai", "claude"])
def offline(monkeypatch, tmp_path, request):
    native = request.param == "claude"
    backend = NativeOfflineBackend() if native else OfflineBackend()
    model_id = "us.anthropic.claude-sonnet-4-6" if native else w.DEFAULT_MODEL
    settings = w.Settings(
        model_id=model_id, knowledge_base_id="KBTEST1234", backoff_base_s=0, backoff_cap_s=0,
    )
    monkeypatch.setattr(w.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(w.Rates, "load", classmethod(
        lambda cls, model_id=None: w.Rates(
            settings.model_id, 1, 2, cache_read_per_million=0.5,
            cache_write_per_million=1.25, cache_write_ttl="5m" if native else "30m",
            source="OFFLINE TEST",
        )
    ))
    monkeypatch.setattr(w, "make_model_client", backend.model_client)
    if native:
        monkeypatch.setattr(w, "make_converse_client", backend.native_client)
    else:
        def unexpected_native(*args, **kwargs):
            raise AssertionError("OpenAI models must use Responses automatically")
        monkeypatch.setattr(w, "make_converse_client", unexpected_native)
    monkeypatch.setattr(w, "make_retrieval_client", backend.retrieval_client)
    monkeypatch.setenv("BEDROCK_ARTIFACT_DIR", str(tmp_path))
    real_mcp_client = rag_mcp.PolicyMCPClient
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(w.ROOT / "tests/mcp_fixture_server.py"), str(w.ROOT)],
        cwd=str(w.ROOT),
    )
    monkeypatch.setattr(
        rag_mcp, "PolicyMCPClient",
        lambda settings: real_mcp_client(settings, server_parameters=parameters),
    )
    yield backend
    backend.close()


def execute_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            # Dependencies are already installed in the test environment.
            if "install-dependencies" in cell.metadata.get("tags", []):
                continue
            # Rehearsal outputs are user evidence; execute source in a fresh
            # namespace without requiring those saved outputs to be removed.
            exec(compile(cell.source, f"{path.name}:cell-{index}", "exec"), namespace)
    return namespace


def test_all_notebooks_and_recovery_boundaries(offline):
    namespaces = {}
    for path in sorted(w.ROOT.glob("0*.ipynb")):
        namespaces[path.name[:2]] = execute_notebook(path)
    assert set(namespaces) == {"00", "01", "02", "03", "04", "05"}
    for index in ("01", "02", "03"):
        ns = namespaces[index]
        for name in ("baseline", "recovery", "tuned"):
            run = ns[f"{name}_run"]
            assert not run.error
            # The talk reviews prose quality with the audience; execution isn't a quality score.
            assert run.task_passed is None
            assert ns[f"{name}_answer"]
        assert ns["recovery_run"].summary()["synthetic_failures"] == 1
        assert ns["baseline_run"].summary()["synthetic_failures"] == 0
        assert ns["tuned_run"].summary()["synthetic_failures"] == 0
    tool = namespaces["01"]["recovery_run"]
    assert tool.summary()["logical_model_calls"] == 2
    assert tool.summary()["tool_attempts"] == 2
    chain = namespaces["02"]["recovery_run"]
    assert [a.step for a in chain.attempts] == ["classify", "assess", "assess", "draft"]
    rag = namespaces["03"]["recovery_run"]
    assert rag.summary()["retrieval_attempts"] == 1
    assert rag.summary()["model_attempts"] == 2
    assert len(offline.retrieval_requests) == 3
    optimization = namespaces["04"]
    assert optimization["baseline_run"].summary()["logical_model_calls"] == 3
    assert optimization["fused_run"].summary()["logical_model_calls"] == 1
    assert optimization["stable_run"].summary()["cached_input_tokens"] > 0
    dynamic_prefixes = optimization["dynamic_run"].frame()["cache_prefix_sha256"]
    stable_prefixes = optimization["stable_run"].frame()["cache_prefix_sha256"]
    assert dynamic_prefixes.nunique() == len(optimization["questions"])
    assert stable_prefixes.nunique() == 1
    assert optimization["stable_run"].summary()["inference_apis"] == (
        "responses_stream" if optimization["settings"].resolved_api == "responses"
        else "converse_stream"
    )
    agent = namespaces["05"]["agent_run"]
    assert agent.summary()["logical_model_calls"] == 3
    assert agent.summary()["tool_attempts"] == 1
    assert agent.summary()["retrieval_attempts"] == 1
    assert "[S1]" in namespaces["05"]["result"]["answer"]
    # Serialized attempts contain no input prompts or encrypted reasoning payloads.
    serialized = (w.artifact_directory() / "01_tools_baseline.json").read_text()
    assert "synthetic-reasoning-for-local-test" not in serialized
    assert "audit_events" not in serialized
    assert "native-signature" not in serialized


def test_empty_retrieval_skips_generation(offline):
    ns = execute_notebook(w.ROOT / "03_knowledge_bases_rag.ipynb")
    requests_before = len(offline.requests)
    offline.empty_retrieval = True
    run, answer, passages = ns["run_rag"](
        "empty", system_prompt=ns["system_prompt"], instruction=ns["instruction"],
    )
    assert not passages and answer == "No policy evidence was found."
    assert run.summary()["logical_model_calls"] == 0
    assert len(offline.requests) == requests_before
    assert run.task_passed is False


def test_invalid_tool_arguments_are_not_retried(offline):
    ns = execute_notebook(w.ROOT / "01_function_calling.ipynb")
    offline.invalid_tool_arguments = True
    requests_before = len(offline.requests)
    with pytest.raises(ValueError, match="known order ID"):
        ns["run_tool_demo"](
            "invalid", system_prompt=ns["system_prompt"], instruction=ns["instruction"],
        )
    assert len(offline.requests) == requests_before + 1


def request_text_and_system(request):
    if "modelId" in request:
        system = " ".join(block.get("text", "") for block in request.get("system", []))
        text = " ".join(
            block.get("text", "") for message in request["messages"]
            for block in message["content"]
        )
    else:
        system = request.get("instructions", "")
        text = " ".join(
            item["content"] for item in request["input"]
            if isinstance(item.get("content"), str)
        )
    return text, system


def test_rag_prompt_edits_reach_retrieval_and_generation(offline):
    ns = execute_notebook(w.ROOT / "03_knowledge_bases_rag.ipynb")
    new_system = ns["system_prompt"] + " Use a friendly tone."
    new_instruction = "Explain the replacement options for ORDER-1042, four days late."
    ns["run_rag"]("edited-rag", new_system, new_instruction)
    assert offline.retrieval_requests[-1]["retrievalQuery"]["text"] == new_instruction
    text, system = request_text_and_system(offline.requests[-1])
    assert system == new_system
    assert new_instruction in text


def test_fused_prompt_edits_are_not_frozen_in_function_defaults(offline):
    ns = execute_notebook(w.ROOT / "04_results_and_tuning.ipynb")
    new_system = ns["system_prompt"] + " Use plain language."
    new_instruction = "Explain ORDER-1042 eligibility in a short update."
    ns["run_fused"]("edited-fused", new_system, new_instruction, ns["evidence"])
    text, system = request_text_and_system(offline.requests[-1])
    assert system == new_system
    assert new_instruction in text


@pytest.mark.parametrize("implicit_caching", [False, True])
def test_cache_layout_reuse_does_not_depend_on_implicit_caching(offline, implicit_caching):
    """Exercise the demo under both cache policies; synthetic timings are not benchmarks."""
    offline.implicit_caching = implicit_caching
    ns = execute_notebook(w.ROOT / "04_results_and_tuning.ipynb")
    dynamic = ns["dynamic_run"].frame()
    stable = ns["stable_run"].frame()
    unmarked = ns["unmarked_run"].frame()
    assert dynamic["requested_cache_mode"].eq("on").all()
    assert stable["requested_cache_mode"].eq("on").all()
    assert dynamic["cache_prefix_sha256"].nunique() == len(ns["questions"])
    assert stable["cache_prefix_sha256"].nunique() == 1
    assert dynamic["cache_write_tokens"].gt(0).all()
    assert stable.iloc[0]["cache_write_tokens"] > 0
    assert stable.iloc[1:]["cache_write_tokens"].eq(0).all()
    assert stable.iloc[1:]["cached_input_tokens"].gt(0).all()
    assert stable["time_to_first_text_s"].notna().all()
    assert ns["cache_measurements"](ns["stable_run"])["input_usd"].notna().all()
    assert bool(unmarked.iloc[1:]["cached_input_tokens"].gt(0).all()) == implicit_caching
    cache_requests = [
        request for request in offline.requests
        if "prompt_cache_options" in request
        or any("cachePoint" in block for block in request.get("system", []))
    ]
    assert cache_requests
    for request in cache_requests:
        if "modelId" not in request:
            assert request["stream"] is True
            assert isinstance(request["instructions"], str)


def test_agent_uses_prompt_from_current_graph_state(offline):
    ns = execute_notebook(w.ROOT / "05_langgraph_agent.ipynb")
    new_system = ns["system_prompt"] + " Explain the next action clearly."
    new_instruction = "Investigate ORDER-1042 again."
    with w.Run("edited-agent", w.Settings.from_env()) as run:
        ns["agent_run"] = run
        ns["agent_node"]({
            "system_prompt": new_system,
            "messages": [{"role": "user", "content": new_instruction}],
            "tool_calls": [], "turns": 0, "answer": "",
        })
    request = offline.requests[-1]
    text, system = request_text_and_system(request)
    if "modelId" in request:
        # Converse keeps instructions and the optional cached reference in
        # separate system blocks. Both must survive an edited graph prompt.
        assert request["system"][0] == {"text": new_system}
        if ns["cache_reference"] is not None:
            assert request["system"][1] == {"text": ns["cache_reference"]}
    else:
        assert system == new_system
    assert new_instruction in text
