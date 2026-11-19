# Live-code inference patterns on Amazon Bedrock

These notebooks support this code talk. The inference layer is prepared;
workflow and optimization code stays visible in the notebooks. For a 60-minute
delivery, treat notebook 05 as a prepared closing walkthrough and select which
notebook 04 comparisons to run live; all experiments remain runnable afterward.

| Notebook                                               | What we build                                    | One tuning change                                             |
| ------------------------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------- |
| [00 · Inference](00_setup_and_inference.ipynb)         | A request, its metrics, and streaming            | Establish the measurement baseline                            |
| [01 · Function calling](01_function_calling.ipynb)     | A plain tool definition and one lookup round     | Trim the tool result                                          |
| [02 · Reasoning chain](02_reasoning_chains.ipynb)      | Three calls passing ordinary text                | Lower classification effort                                   |
| [03 · RAG](03_knowledge_bases_rag.ipynb)               | Retrieve from Knowledge Bases or Chroma          | Request fewer passages                                        |
| [04 · Optimize inference](04_results_and_tuning.ipynb) | Implement and measure optimizations              | Consolidate calls, shorten answers, cache a stable prefix     |
| [05 · LangGraph agent](05_langgraph_agent.ipynb)       | Model-directed tool loop with an MCP policy tool | Compact tool results, bounded output, optional prefix caching |

Each pattern runs a baseline, one labeled recovery example, and a tuned version.
The functions execute their complete workflows in one timed block so pauses while
presenting notebook cells do not become inference latency.

## Edit prompts without editing the workflow

Every notebook has an **Edit the prompts** section before the model calls.
`system_prompt` defines behavior and constraints; `instruction` supplies the task
or question. Multi-stage examples use a plain `system_prompts` dictionary so each
stage can be adjusted independently.

The demo functions receive these values explicitly. Edit and run the prompt cell,
then rerun the example; function definitions do not need to be rerun. In notebook 03:

```python
recovery_run, recovery_answer, _ = run_rag(
    "03_rag_recovery",
    system_prompt=system_prompt,
    instruction=instruction,
    before_generation=FailOnce("RAG generation request"),
)
```

The RAG instruction is used for both retrieval and generation. Notebook 04 also
exposes its concise-answer and caching prompts, and notebook 05 passes the prompt
into fresh graph state at invocation. Rerun both baseline and tuned runs after prompt
edits so their comparisons remain meaningful; changing a cached prefix can cause a cache miss.

## Prepare and launch

`requirements.txt` lists the notebook dependencies directly. Pip or uv resolves
their dependencies automatically; there is no separate input file or compilation
step. Core libraries retain their version pins, while indirect dependencies are
resolved from those libraries' requirements.

Run from the repository root using Python 3.12 or a compatible environment:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name bedrock-inference --display-name "Python (Bedrock inference)"

export AWS_PROFILE="YOUR_PROFILE"
export AWS_REGION="us-east-1"
# Optional: omit this variable to use local Chroma.
# export KNOWLEDGE_BASE_ID="YOUR_KNOWLEDGE_BASE_ID"
.venv/bin/jupyter lab
```

Select **Python (Bedrock inference)**. Notebooks 03 and 05 use Knowledge Bases when
`KNOWLEDGE_BASE_ID` is set, and local Chroma otherwise.
If you moved the repository, recreate `.venv` and register the kernel using the
commands above; an older environment or kernel may still reference the previous path.
Model access, IAM permissions, and the selected retrieval backend should be ready
before the session. Model generation still makes real Bedrock requests when Chroma
is used for retrieval.

The order and policies are fictional workshop fixtures. `ORDER-1042` is in transit
and four days late in the fixed scenario. `data/order.json` also contains verbose
internal fields for the tool-response tuning comparison.

For an existing S3 Knowledge Base data source, upload the sample policies and sync:

```bash
aws s3 sync data/knowledge-base/ s3://YOUR_BUCKET/bedrock-inference/policies/
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id "$KNOWLEDGE_BASE_ID" \
  --data-source-id "YOUR_DATA_SOURCE_ID"
```

Keep the delivery-eligibility sentence intact during chunking. The notebook identity
needs retrieval permission on the Knowledge Base and inference/streaming permissions
for its chosen model or inference profile.

### Local Chroma

Without a Knowledge Base ID, `create_retriever(settings)` indexes the short `.txt`
policies using Chroma's local ONNX MiniLM embeddings. The embedding model may
download on first use; warm it and the index before presenting. Index construction
happens in setup, outside the measured retrieval.

The local index persists under `.chroma/notebook/`. The MCP server uses
`.chroma/mcp/` so each process owns its index. Changed documents and metadata are
updated; removed documents are removed from the generated index. The short demo
policies use one chunk per document; production documents need a chunking strategy.

`BEDROCK_CHROMA_PATH` and `BEDROCK_MCP_CHROMA_PATH` override those locations. A
configured Knowledge Base failure is always surfaced rather than silently switching
the evidence source. Chroma is a configuration fallback, not an error fallback.

## The prepared inference interface

The same methods work for OpenAI and Claude:

```python
from utils.workshop import Rates, Run, Settings, compare

settings = Settings.from_env()
rates = Rates.load(model_id=settings.model_id)

with Run("example", settings, rates) as run:
    response = run.generate("answer", "Explain agent inference in two sentences.")

print(response.text)
display(run.metrics())
display(compare(run))
```

Use `run.stream(..., on_text=callback)` to display text as it arrives. The complete
stream is consumed to collect final usage; interrupted output is never silently replayed.

`BEDROCK_MODEL_ID` selects the model. OpenAI IDs use Responses; Claude and other native
Bedrock models use Converse/ConverseStream. The default remains
`global.openai.gpt-5.6-terra`. For an opaque application-profile ARN, explicitly set
`BEDROCK_INFERENCE_API` to `responses` or `converse` because its name cannot identify
the underlying model.

Tools are plain dictionaries containing `name`, `description`, and `parameters`.
Append `response.message` to preserve the model's tool request, then a plain tool-result
dictionary containing `role`, `tool_call_id`, and `content`. The inference layer handles
provider formats and reasoning state. Existing `Tool` objects and structured-output
options remain supported for other applications, but the talk does not require them.

Notebook 02 uses `reasoning_effort="high"` or `"medium"`. The wrapper maps this to
OpenAI effort, Claude adaptive effort, or explicit budgets on older thinking-capable
Claude models. Models without a known mapping raise a clear error. Other demos leave
thinking at the selected model's default. Converse does not report a separate
reasoning-token count.

## Notebook 04: actual optimization experiments

The first experiment replaces three sequential calls with one when the same
evidence is already available. The next adds a concise-answer instruction and
reduces the output ceiling. Both retain a manual quality review; lowering a ceiling
does not guarantee fewer tokens and may cause incomplete output.

The main caching experiment moves changing request metadata from before the long
reference to after it, keeping an explicit checkpoint in both variants. This is a
deliberately cache-unfriendly baseline compared with a reusable-prefix layout,
using the same questions, reference, and request IDs. If request metadata is not
useful to the model, omit it rather than sending it.

The experiment repeats questions and streams every variant to capture time to first
visible text as well as total duration. An optional third run removes the checkpoint
from the stable layout to check whether the model benefits from an explicit policy.
Set `RUN_CHECKPOINT_COMPARISON = False` to skip that comparison during the talk.

```python
response = run.stream(
    "answer", question,
    cache_prefix=reference_text,
    cache_mode="on",
    max_output_tokens=2048,
)
```

For GPT-5.6 Responses, the adapter adds a developer content block with
`prompt_cache_breakpoint`, uses `prompt_cache_options.mode="explicit"` and a
30-minute TTL, and supplies a stable cache key. `cache_mode="off"` requests explicit
mode without our breakpoint; it must not be treated as an empirically guaranteed
cache-off control. The observed rehearsal included cache reads without a developer
checkpoint, so the notebook bases its interpretation on returned usage counters.

For compatible Claude models, it adds a Converse cache point after the stable
system prefix with a five-minute TTL. `off` removes our checkpoint; any model-level
implicit caching may still apply. Other model/API combinations currently raise a
clear unsupported-option error for this explicit cache interface.

The trace records the requested cache mode and a prefix digest for diagnostics
without storing the reference itself. Restart the kernel before running the revised
notebook so the current helper definitions are loaded.

The prepared `data/cache_reference.md` is a fictional workflow handbook with worked
cases, not authoritative eligibility policy. Its length makes cache eligibility
observable; minimum prefix sizes vary by model. Do not add unnecessary context just
to make caching look useful. A repeated request is not assumed to hit the cache.

Inspect cache reads/writes and input-only cost as well as the full request cost;
different output lengths can otherwise obscure the cache effect. Include initial
write costs and compare later-request medians. The optional break-even calculation
is a projection from observed input costs under stated reuse assumptions, not a
guarantee. An explicit checkpoint can remain more expensive than implicit reuse.
Earlier runs can leave the prefix warm, so first-request usage must be inspected
before interpreting a write premium.

See [AWS prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
for model support, minimum sizes, TTLs, and regional behavior.

## Notebook 05: LangGraph and MCP

The graph has a model node, a tool-execution node, and a conditional edge. The model
chooses an order lookup, policy search, or a final answer; its messages and tool
results remain in graph state. Model-call and graph-recursion budgets bound the loop.

`PolicyMCPClient` starts a local stdio MCP server, discovers the policy tool's schema,
and closes the server after the run. The read-only `search_support_policy` tool uses
the same Knowledge Base/Chroma selection as notebook 03. Startup and indexing happen
before the timed agent task. The synchronous client bridge keeps the notebook on
the existing `Run.call` measurement/retry interface.

The notebook requirements specify MCP 2.2.0 (`MCPServer`). The helper also supports
MCP 1.28 (`FastMCP`), translating client timeout values and using the shared MCP
protocol field names for schemas, structured results, and error flags. The local
server runs with the notebook's Python interpreter.

If notebook 05 reports `No module named 'mcp.server.mcpserver'`, select
**Python (Bedrock inference)**, or this repository's `.venv/bin/python` in VS Code, and
restart the kernel. A generic `python312` kernel may use different packages.
To inspect the current kernel, run `import sys; print(sys.executable)` and
`from importlib.metadata import version; print(version("mcp"))`.

Its compact order result and answer budget carry the earlier tuning lessons into
the graph. `USE_PROMPT_CACHE` is optional because a long reference should only be
sent when it is useful.

## Measurements and recovery

- `run.metrics()` shows attempts, reported tokens, elapsed time, and retry waits.
- `compare(baseline, tuned)` shows compact task totals and reported inference cost.
- `run.frame()` and `compare(..., detailed=True)` expose the full measurements,
  including cache counts, model identity, and service timing.
- `run.save()` writes raw measurements to `artifacts/` for rehearsal and later analysis.

The prepared retry policy allows three total attempts, with timeouts, backoff, jitter,
and server-requested waits. Both SDKs have automatic retries disabled so attempts are
visible. `FailOnce` simulates a failure before an operation; it does not induce a real
AWS outage. A real failed request can have unknown token usage.

The tool example performs one read-only lookup and two model calls. A production write
tool needs idempotency before repeating side effects. RAG retries generation using
already-retrieved passages; empty evidence returns an explicit no-answer outcome.

Review answer quality directly in each notebook using its short review criteria.
There are no generated output schemas or grading-framework walkthroughs. A nonempty
response or successful HTTP call is not treated as proof of task correctness.

## Prices and rehearsal

`data/prices.bedrock-global-terra.json` contains the Bedrock rates checked on
2026-09-25, with a source link and input-token bracket. Recheck prices before the event.
Use `data/prices.local.json` or `BEDROCK_PRICE_FILE` for another model or updated rates.
`Rates.load(model_id=...)` never applies a different model's prices automatically.

Unknown prices or usage stay unavailable rather than becoming zero. Cost estimates
cover reported model usage and exclude retrieval, vector-store, and other service
charges. Cache-write prices require a matching `cache_write_ttl`; unknown or mixed
TTLs remain unpriced. The bundled Global Terra rate includes its 30-minute write
price. Reasoning tokens are already included in output tokens and are not added again.

Rerun baseline and tuned pairs after editing prompts or changing models. Inspect cached
usage and repeat measurements during rehearsal; one live run is not a stable benchmark.
The notebooks are saved without generated outputs. Existing artifacts are not rewritten
until their corresponding notebook cells run.

## Implementation and verification

Two Python files implement the shared inference layer:

- `utils/workshop.py`: the shared inference calls, streaming, retries, and measurements.
- `utils/inference.py`: model routing and translation of messages, tools, and responses.

Two integration modules support the new retrieval/agent examples:

- `utils/retrieval.py`: shared Knowledge Base and persistent Chroma retrieval.
- `utils/rag_mcp.py`: the local MCP server and its client connection.

Both are prepared infrastructure with documented interfaces. `utils/__init__.py`
marks the utility folder as a Python package. Import from `utils.workshop`; data
and generated artifacts live in `data/` and `artifacts/` at the repository root. The live
walkthrough stays in the notebooks. After moving the files, restart any existing
notebook kernel and rerun its setup cell so it loads the package consistently.

The files under `tests/` provide offline checks and simulated service
responses. Keep them for development and rehearsal verification; they are not
required to run the notebooks or include in an attendee package. That package needs
the notebooks, the `utils/` folder, `data/`, `requirements.txt`, and this README.

For development, install `requirements-dev.txt`. It includes the notebook
requirements and adds the tools used for validation, execution, and testing:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The MCP transport regression tests can also be run in an existing MCP 1.28
environment with `python -m pytest -q tests/test_rag_mcp.py`; they launch the
actual local server and check discovery, policy results, error propagation,
and request timeouts.

Tests execute the same notebook code against OpenAI and Claude fixtures and check API
routing, tool-result formats, streaming, cache controls/pricing, the graph, and MCP
transport. Chroma tests use deterministic test embeddings without downloading a model.
They make no AWS inference requests. Real model answer quality, cache hits, account
access, and measured performance still need a rehearsal in the session account.
