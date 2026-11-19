"""Client setup, usage accounting, and bounded retries for the Bedrock notebooks.

OpenAI Responses, Bedrock Converse, and Bedrock ConverseStream share the same
measurement tables. The notebooks define the agent workflows. This module supplies common
infrastructure: Settings selects the model and execution limits, Usage and Rates
interpret reported tokens, and Run records each operation and its retry attempts.

Durations are measured in seconds. Prices are USD per million tokens; calculated
costs are USD. Missing usage or unavailable pricing remains None rather than zero.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import boto3
import openai
import pandas as pd
from .inference import (
    InferenceResponse,
    Message,
    Tool,
    ToolCall,
    UnsupportedInferenceOption,
    build_request,
    normalize_response,
    normalize_tools,
    object_schema,
    resolve_api,
    tool_result,
)
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

# Notebooks, data, and artifacts live at the repository root, above this package.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "global.openai.gpt-5.6-terra"


@dataclass(frozen=True)
class Settings:
    """Immutable connection settings and execution limits shared by a task.

    Attributes:
        model_id: Bedrock model or inference-profile identifier.
        region: AWS Region used to construct both service clients.
        knowledge_base_id: Knowledge Base identifier; only the RAG notebook needs it.
        attempt_timeout_s: Maximum timeout passed to an individual operation.
        workflow_budget_s: Elapsed-time budget for the complete Run.
        max_attempts: Maximum attempts per operation, including the initial attempt.
        backoff_base_s: Initial ceiling for exponential backoff with full jitter.
        backoff_cap_s: Maximum jitter ceiling; a server-requested wait can exceed it.
        inference_api: auto routes identifiable OpenAI IDs to Responses and other
            models to Converse. An explicit override handles opaque profile ARNs.

    Run checks the workflow budget between operations and attempts. The callback
    must apply its supplied timeout; these settings do not interrupt Python code.
    """

    model_id: str = DEFAULT_MODEL
    region: str = "us-east-1"
    knowledge_base_id: str = ""
    attempt_timeout_s: float = 60
    workflow_budget_s: float = 240
    max_attempts: int = 3  # Includes the first attempt.
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 4
    inference_api: str = "auto"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load model, Region, and Knowledge Base settings from the environment.

        Read BEDROCK_MODEL_ID, KNOWLEDGE_BASE_ID, and BEDROCK_INFERENCE_API. Resolve the Region from
        AWS_REGION, AWS_DEFAULT_REGION, the Boto3 session, then us-east-1, in that
        order. Retry limits and timeouts retain the dataclass defaults.

        Returns:
            A Settings instance; the Knowledge Base identifier may be empty.
        """
        region = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or boto3.Session().region_name
            or "us-east-1"
        )
        return cls(
            model_id=os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL),
            region=region,
            knowledge_base_id=os.getenv("KNOWLEDGE_BASE_ID", ""),
            inference_api=os.getenv("BEDROCK_INFERENCE_API", "auto"),
        )

    @property
    def resolved_api(self) -> str:
        """Return the automatically selected backend or the explicit override."""
        return resolve_api(self.model_id, self.inference_api)

    def __post_init__(self):
        """Reject attempt limits below one, nonpositive budgets, or negative backoff.

        Raises:
            ValueError: An execution limit is outside its allowed range.
        """
        if self.max_attempts < 1:
            raise ValueError("max_attempts must include at least the initial attempt")
        if self.attempt_timeout_s <= 0 or self.workflow_budget_s <= 0:
            raise ValueError("Timeouts and budgets must be positive")
        if self.backoff_base_s < 0 or self.backoff_cap_s < 0:
            raise ValueError("Backoff must be nonnegative")
        if self.inference_api not in {"auto", "responses", "converse"}:
            raise ValueError("inference_api must be auto, responses, or converse")


def make_model_client(settings: Settings):
    """Create an OpenAI SDK client that sends inference requests through Bedrock.

    Args:
        settings: Region and default request timeout for the Runtime endpoint.

    Returns:
        An OpenAI client using the AWS credential chain for request signing.
        Automatic SDK retries are disabled because Run.call owns retry accounting.
        The caller can reuse the client and should close it when finished.

    Raises:
        RuntimeError: The installed SDK does not expose the Bedrock provider.
    """
    try:
        from openai.providers import bedrock
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt in this notebook's kernel. "
            "The Bedrock provider requires the pinned OpenAI SDK."
        ) from exc
    return openai.OpenAI(
        provider=bedrock(endpoint="runtime", region=settings.region, api_key=None),
        max_retries=0,  # Run.call owns retry policy and attempt accounting.
        timeout=settings.attempt_timeout_s,
    )


def make_converse_client(settings: Settings, timeout_s: float | None = None):
    """Create a native Bedrock Runtime client for Converse and ConverseStream.

    Args:
        settings: AWS Region and default per-attempt timeout.
        timeout_s: Override supplied by Run when its remaining budget is shorter.

    Returns:
        A Boto3 client exposing converse() and converse_stream(), using the AWS
        credential chain. SDK retries are disabled so Run can count attempts.
        The caller owns the client and must close it after consuming any stream.

    Request fields stay in Bedrock's native format: messages, system, toolConfig,
    inferenceConfig, and model-specific additionalModelRequestFields. OpenAI
    tools, reasoning, and text-format arguments are not translated.
    """
    timeout_s = settings.attempt_timeout_s if timeout_s is None else timeout_s
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.region,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=min(5, timeout_s),
            read_timeout=timeout_s,
        ),
    )


def make_retrieval_client(settings: Settings, timeout_s: float):
    """Create a Boto3 Knowledge Base retrieval client for one operation attempt.

    Args:
        settings: Supplies the AWS Region.
        timeout_s: Remaining per-attempt timeout provided by Run.call. The read
            timeout uses this value; the connect timeout is capped at five seconds.

    Returns:
        A bedrock-agent-runtime client with SDK retries disabled. The caller must
        close it after use. Connect/read timeouts are not a single total deadline.
    """
    return boto3.client(
        "bedrock-agent-runtime",
        region_name=settings.region,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=min(5, timeout_s),
            read_timeout=timeout_s,
        ),
    )


def as_dict(value: Any) -> dict:
    """Return a dictionary unchanged or serialize an SDK/Pydantic response.

    Model serialization uses JSON-compatible values, API field aliases, and omits
    None fields. An existing dictionary is returned as-is, without copying or
    filtering it. Other input objects must support the same model_dump interface.
    """
    if isinstance(value, InferenceResponse):
        return value.raw_response
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


@dataclass(frozen=True)
class Usage:
    """Token counts reported for one model response, including incomplete output.

    Attributes:
        input_tokens: Total input tokens, including cache reads and writes.
        output_tokens: Total reported output tokens, including reasoning tokens.
        cached_input_tokens: Cache-read subset of input tokens, or None if absent.
        cache_write_tokens: Endpoint-specific cache-write count; defaults to zero
            when the response does not report this category.
        reasoning_tokens: Reasoning subset of output tokens, or None if absent.
        cache_write_ttl: TTL reported by cache details or attached by the explicit
            cache request, used to select the matching write price.

    Subset counts explain the totals; they must not be added to those totals again.
    Converse inputTokens counts uncached input only. Its cacheReadInputTokens and
    cacheWriteInputTokens are added to produce the common input_tokens total.
    Native Converse does not report a separate reasoning-token count; it stays None.
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None
    cache_write_tokens: int
    reasoning_tokens: int | None
    cache_write_ttl: str | None = None

    @classmethod
    def from_response(cls, response) -> "Usage | None":
        """Extract usage from Responses, Converse, or a consumed ConverseStream.

        Returns:
            Usage when the response contains a usage object, otherwise None.
            Missing OpenAI cache-read/reasoning details remain None. Absent
            Converse cache counters are treated as zero.

        Raises:
            ValueError: The response contains an unconsumed event stream.
            KeyError: A present usage object lacks a required input/output total.
        """
        data = as_dict(response)
        if "stream" in data:
            raise ValueError("Consume ConverseStream before reading its final usage")
        raw = data.get("usage")
        if raw is None:
            return None
        ttl = data.get("_workshop", {}).get("cache_write_ttl")
        ttls = {detail["ttl"] for detail in raw.get("cacheDetails", []) if detail.get("inputTokens", 0)}
        if ttls:
            ttl = next(iter(ttls)) if len(ttls) == 1 else "mixed"
        if ttl is None and "openai.gpt-5.6" in data.get("model", ""):
            ttl = "30m"
        if "inputTokens" in raw:
            cached = raw.get("cacheReadInputTokens", 0)
            written = raw.get("cacheWriteInputTokens", 0)
            return cls(
                input_tokens=raw["inputTokens"] + cached + written,
                output_tokens=raw["outputTokens"],
                cached_input_tokens=cached,
                cache_write_tokens=written,
                reasoning_tokens=None,
                cache_write_ttl=ttl if written else None,
            )
        input_details = raw.get("input_tokens_details") or {}
        output_details = raw.get("output_tokens_details") or {}
        return cls(
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            cached_input_tokens=input_details.get("cached_tokens"),
            cache_write_tokens=input_details.get("cache_write_tokens", 0),
            reasoning_tokens=output_details.get("reasoning_tokens"),
            cache_write_ttl=ttl if input_details.get("cache_write_tokens", 0) else None,
        )


@dataclass(frozen=True)
class Rates:
    """Bedrock prices for one model/profile and input-token price bracket.

    Attributes:
        model_id: Exact identifier these rates apply to.
        input_per_million: USD per million uncached input tokens.
        output_per_million: USD per million total output tokens.
        cache_read_per_million: USD per million cache-read tokens, when known.
        source: Pricing reference URL or description.
        checked_on: Date the prices were checked; recorded as metadata.
        max_input_tokens: Inclusive request input limit for this price bracket,
            or None to omit the bracket check.
        cache_write_per_million: USD per million newly cached tokens, when configured.
        cache_write_ttl: TTL to which that write price applies (5m, 30m, or 1h).

    Prices come from local configuration. This class does not fetch or refresh
    them, and its estimates exclude retrieval and other service charges.
    """

    model_id: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None = None
    source: str = ""
    checked_on: str = ""
    max_input_tokens: int | None = None
    cache_write_per_million: float | None = None
    cache_write_ttl: str | None = None

    def __post_init__(self):
        """Reject any supplied token price that is negative.

        Raises:
            ValueError: An input, output, or cache-read price is below zero.
        """
        for name in ("input_per_million", "output_per_million", "cache_read_per_million", "cache_write_per_million"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.cache_write_ttl not in {None, "5m", "30m", "1h"}:
            raise ValueError("Unsupported cache-write pricing TTL")

    @classmethod
    def load(cls, model_id: str | None = None) -> "Rates | None":
        """Read the selected pricing JSON file without making a network request.

        BEDROCK_PRICE_FILE takes precedence. Otherwise, use data/prices.local.json
        if it exists, then the bundled Global Terra pricing file. A missing
        explicitly selected file does not trigger another fallback.

        Args:
            model_id: If supplied, return None when the file belongs to another
                model/profile. This lets other models run with token/latency
                metrics while their prices remain unconfigured.

        Returns:
            Rates, or None if the selected file is missing or either required
            input/output price is absent or null. Malformed JSON and invalid
            constructor arguments propagate to the caller.
        """
        local = ROOT / "data/prices.local.json"
        default = local if local.exists() else ROOT / "data/prices.bedrock-global-terra.json"
        path = Path(os.getenv("BEDROCK_PRICE_FILE", default))
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if model_id is not None and data.get("model_id") != model_id:
            return None
        if data.get("input_per_million") is None or data.get("output_per_million") is None:
            return None
        return cls(**data)

    def estimate(self, usage: Usage, model_id: str) -> float | None:
        """Estimate USD for one response's reported input and output tokens.

        Charge uncached input, cached input, and total output at their respective
        rates. Reasoning tokens are already part of output_tokens.

        Args:
            usage: Reported token counts for one response.
            model_id: Identifier used by the request; must match these rates.

        Returns:
            Estimated USD, or None when the request exceeds this price bracket,
            required cache prices are unavailable, write TTLs do not match the
            configured price, or the cache-read count is unknown.

        Raises:
            ValueError: The model differs or a cache-read count being priced is
                outside the range from zero to input_tokens.
        """
        if self.model_id != model_id:
            raise ValueError("Pricing model_id differs from the model being measured")
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            return None
        if usage.cached_input_tokens is None:
            return None
        cached = usage.cached_input_tokens
        written = usage.cache_write_tokens
        if cached < 0 or written < 0 or cached + written > usage.input_tokens:
            raise ValueError("Cache reads and writes must fit within input_tokens")
        if cached and self.cache_read_per_million is None:
            return None
        if written and (
            self.cache_write_per_million is None
            or not self.cache_write_ttl
            or usage.cache_write_ttl != self.cache_write_ttl
        ):
            return None
        return (
            (usage.input_tokens - cached - written) * self.input_per_million
            + cached * (self.cache_read_per_million or 0)
            + written * (self.cache_write_per_million or 0)
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


class BudgetExceeded(TimeoutError):
    """Signal that the task budget is exhausted or cannot accommodate a retry wait.

    Run.call can raise this before an attempt or after recording a returned result
    that exceeded the budget. It does not cancel an operation already in flight.
    """

    pass


class IncompleteResponse(ValueError):
    """Signal incomplete, blocked, malformed, or unsuccessful model output.

    This is an application-level outcome check, separate from HTTP success.
    Returned token usage can still be recorded before the exception is raised.
    """

    pass


class SyntheticTransientError(ConnectionError):
    """Represent a simulated failure injected before a tool or service operation.

    FailOnce raises this exception so the notebooks can demonstrate recovery.
    Run marks the attempt as synthetic and excludes it from unknown model usage.
    """


class ModelStreamError(RuntimeError):
    """A consumed model stream failed; partial output is retained and never replayed."""

    def __init__(self, message: str, partial_response: dict):
        """Attach partial content and reported usage for diagnostics and accounting."""
        super().__init__(message)
        self.partial_response = partial_response


class ConverseStreamError(ModelStreamError):
    """Signal a failed or truncated stream and retain its partial response.

    Errors encountered while consuming a stream are not automatically retried:
    text or tool-call fragments may already have reached the caller. The
    partial_response can contain reported usage and unfinished content blocks.
    It is diagnostic output, not a completed turn to execute or replay.
    """

    def __init__(self, message: str, partial_response: dict):
        """Attach the accumulated response so Run can retain available measurements."""
        super().__init__(message, partial_response)


def _append_converse_delta(block: dict, delta: dict):
    """Accumulate one text, tool-input, or reasoning delta into its content block.

    Tool input remains a JSON string until contentBlockStop. Reasoning text,
    signatures, and redacted bytes are preserved for subsequent conversation turns.
    Unsupported content types raise rather than being silently discarded.
    """
    if set(delta) == {"text"}:
        if block and "text" not in block:
            raise ValueError("Content block changed type")
        block["text"] = block.get("text", "") + delta["text"]
    elif set(delta) == {"toolUse"}:
        if "toolUse" not in block:
            raise ValueError("Tool input arrived without a tool-use start event")
        block["toolUse"]["input"] += delta["toolUse"]["input"]
    elif set(delta) == {"reasoningContent"}:
        if block and "reasoningContent" not in block:
            raise ValueError("Content block changed type")
        content = block.setdefault("reasoningContent", {})
        reasoning = delta["reasoningContent"]
        if not reasoning or set(reasoning) - {"text", "signature", "redactedContent"}:
            raise ValueError("Unsupported reasoning delta")
        if "redactedContent" in reasoning:
            if "reasoningText" in content or len(reasoning) != 1:
                raise ValueError("Mixed redacted and visible reasoning in one block")
            content["redactedContent"] = content.get("redactedContent", b"") + reasoning["redactedContent"]
        else:
            if "redactedContent" in content:
                raise ValueError("Mixed redacted and visible reasoning in one block")
            text = content.setdefault("reasoningText", {})
            for key, value in reasoning.items():
                text[key] = text.get(key, "") + value
    else:
        raise ValueError(f"Unsupported ConverseStream content delta: {sorted(delta)}")


def consume_converse_stream(
    response: dict,
    *,
    on_text: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    started_at: float | None = None,
    deadline: float | None = None,
) -> dict:
    """Consume a Bedrock event stream and return a Converse-shaped response.

    Args:
        response: Raw result of a Boto3 converse_stream() call.
        on_text: Optional callback receiving visible text fragments as they arrive.
            Reasoning text and tool arguments are accumulated but not sent here.
        clock: Monotonic clock for local stream measurements.
        started_at: Timestamp before the HTTP request for accurate time to first
            visible text. If omitted, timing starts when consumption begins.
        deadline: Optional absolute timestamp checked between incoming events.

    Returns:
        A dictionary with output.message, stopReason, final usage/metrics, and
        ResponseMetadata. _workshop contains the local API label and first-text
        duration. Text, tool-use, and reasoning content blocks retain their order.
        The event stream is closed on success and failure.

    Raises:
        ConverseStreamError: Iteration/callback errors, unsupported content, invalid
            tool JSON, or missing message-stop/metadata events. Partial content and
            available usage are attached; Run will not automatically replay it.
        BudgetExceeded: The supplied deadline is reached between events. The
            partial response is attached to this exception as well.

    Image, citation, and server-tool-result stream blocks are not assembled by
    this text/tool/reasoning collector. Native nonstreaming Converse can return
    those blocks unchanged. A read timeout still bounds waiting for the next event.
    """
    started_at = clock() if started_at is None else started_at
    stream = response["stream"]
    partial = {
        "output": {"message": {"role": "assistant", "content": []}},
        "ResponseMetadata": response.get("ResponseMetadata", {}),
        "_workshop": {"inference_api": "converse_stream", "time_to_first_text_s": None},
    }
    blocks: dict[int, dict] = {}
    closed: set[int] = set()
    seen_start = seen_stop = seen_metadata = False
    try:
        for event in stream:
            if deadline is not None and clock() >= deadline:
                raise BudgetExceeded("Task budget exceeded while consuming ConverseStream")
            if len(event) != 1:
                raise ValueError("Expected one ConverseStream event per message")
            kind, payload = next(iter(event.items()))
            if kind.endswith("Exception"):
                raise RuntimeError(f"{kind}: {payload.get('message', 'Stream error')}")
            if kind == "messageStart":
                if seen_start:
                    raise ValueError("Duplicate messageStart event")
                partial["output"]["message"]["role"] = payload["role"]
                seen_start = True
            elif kind in {"contentBlockStart", "contentBlockDelta", "contentBlockStop"}:
                index = payload["contentBlockIndex"]
                if seen_stop or index in closed:
                    raise ValueError("Content arrived after its block or message stopped")
                block = blocks.setdefault(index, {})
                # Keep the partial message current even if parsing or a callback fails.
                partial["output"]["message"]["content"] = [blocks[i] for i in sorted(blocks)]
                if kind == "contentBlockStart":
                    start = payload["start"]
                    if block:
                        raise ValueError("Duplicate content-block start")
                    if set(start) == {"toolUse"}:
                        tool = start["toolUse"]
                        block["toolUse"] = {
                            "toolUseId": tool["toolUseId"], "name": tool["name"], "input": "",
                        }
                    elif start:
                        raise ValueError(f"Unsupported ConverseStream content start: {sorted(start)}")
                elif kind == "contentBlockDelta":
                    delta = payload["delta"]
                    _append_converse_delta(block, delta)
                    if delta.get("text"):
                        if partial["_workshop"]["time_to_first_text_s"] is None:
                            partial["_workshop"]["time_to_first_text_s"] = clock() - started_at
                        if on_text is not None:
                            on_text(delta["text"])
                else:
                    if "toolUse" in block:
                        block["toolUse"]["input"] = json.loads(block["toolUse"]["input"])
                    closed.add(index)
            elif kind == "messageStop":
                if seen_stop:
                    raise ValueError("Duplicate messageStop event")
                partial["stopReason"] = payload["stopReason"]
                if "additionalModelResponseFields" in payload:
                    partial["additionalModelResponseFields"] = payload["additionalModelResponseFields"]
                seen_stop = True
            elif kind == "metadata":
                if seen_metadata:
                    raise ValueError("Duplicate metadata event")
                for key in ("usage", "metrics", "trace", "performanceConfig", "serviceTier"):
                    if key in payload:
                        partial[key] = payload[key]
                seen_metadata = True
            else:
                raise ValueError(f"Unsupported ConverseStream event: {kind}")
        if not (seen_start and seen_stop and seen_metadata) or set(blocks) != closed:
            raise ValueError("Truncated ConverseStream: missing message, block-stop, or metadata event")
        return partial
    except BudgetExceeded as exc:
        exc.partial_response = partial
        raise
    except Exception as exc:
        raise ConverseStreamError(f"ConverseStream interrupted: {exc}", partial) from exc
    finally:
        try:
            stream.close()
        except Exception as exc:
            raise ConverseStreamError(f"Could not close ConverseStream: {exc}", partial) from exc


def _response_measurements(response, kind: str) -> dict:
    """Extract service request IDs and optional model timing without saving content."""
    data = as_dict(response)
    metadata = data.get("ResponseMetadata", {})
    details = {
        "request_id": getattr(response, "_request_id", None) or metadata.get("RequestId"),
    }
    if kind == "model":
        local = data.get("_workshop", {})
        native = isinstance(data.get("output"), dict) or "stopReason" in data or bool(local)
        details["inference_api"] = local.get("inference_api", "converse" if native else "responses")
        if native:
            latency_ms = data.get("metrics", {}).get("latencyMs")
            details["model_latency_s"] = latency_ms / 1000 if latency_ms is not None else None
            details["time_to_first_text_s"] = local.get("time_to_first_text_s")
        for key in ("requested_cache_mode", "cache_prefix_sha256", "cache_prefix_characters"):
            if key in local:
                details[key] = local[key]
    return details


def _consume_responses_stream(client, request, *, on_text, clock, started_at, deadline):
    """Consume OpenAI Responses events with the same timing and recovery contract.

    HTTP establishment failures remain retryable. Once the stream opens, all
    consumption and callback errors retain partial usage and become terminal.
    Completed, incomplete, and failed terminal responses are returned for the
    common completion check after Run has recorded their usage.
    """
    local = {"inference_api": "responses_stream", "time_to_first_text_s": None}
    partial = {"status": "in_progress", "output": [], "_workshop": local}
    text = ""
    opened = False
    terminal = None
    try:
        with client.responses.stream(**request) as stream:
            opened = True
            for event in stream:
                event_response = getattr(event, "response", None)
                if event_response is not None:
                    partial = {**as_dict(event_response), "_workshop": local}
                if clock() >= deadline:
                    raise BudgetExceeded("Task budget exceeded while consuming Responses stream")
                if event.type == "response.output_text.delta":
                    text += event.delta
                    if event.delta:
                        if local["time_to_first_text_s"] is None:
                            local["time_to_first_text_s"] = clock() - started_at
                        partial["output"] = [{
                            "type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }]
                        if on_text is not None:
                            on_text(event.delta)
                if event.type in {"response.completed", "response.incomplete", "response.failed"}:
                    terminal = partial
            if terminal is None:
                raise RuntimeError("Responses stream ended without a terminal response event")
        return terminal
    except BudgetExceeded as exc:
        exc.partial_response = partial
        raise
    except Exception as exc:
        if not opened:
            raise
        raise ModelStreamError(f"Responses stream interrupted: {exc}", partial) from exc


class FailOnce:
    """Callable fixture that fails once, then permits subsequent attempts.

    Call this immediately before the operation being demonstrated. It raises on
    its first invocation and returns normally thereafter; it does not execute the
    underlying operation. Create a new instance for each recovery demonstration.

    Attributes:
        label: Human-readable operation name included in the exception.
        fired: Whether this instance has already injected its failure.
    """

    def __init__(self, label: str):
        """Initialize an unused failure fixture with a descriptive operation label."""
        self.label = label
        self.fired = False

    def __call__(self):
        """Raise SyntheticTransientError on the first call; return None thereafter."""
        if not self.fired:
            self.fired = True
            raise SyntheticTransientError(f"SIMULATED transient failure: {self.label}")


def retryable(exc: Exception) -> bool:
    """Decide whether the demo policy allows another attempt for an exception.

    Recognize the synthetic fixture, SDK connection/timeouts, selected transient
    HTTP statuses, and specific transient Bedrock error codes. Known OpenAI quota
    or billing exhaustion codes are excluded. Unrecognized errors return False.

    This function only classifies the error; Run.call applies attempt limits,
    waiting, and budget checks before retrying.
    """
    if isinstance(exc, SyntheticTransientError):
        return True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        code = error.get("code", "") if isinstance(error, dict) else ""
        if code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return False
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceUnavailableException",
            "InternalServerException",
            "InternalFailure",
            "RequestTimeout",
            "ModelTimeoutException",
            "ModelNotReadyException",
        }
    return isinstance(
        exc,
        (ConnectionClosedError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError),
    )


def retry_after_seconds(exc: Exception) -> float | None:
    """Extract a server-requested retry delay from SDK exception response headers.

    Accept HTTP response headers or Botocore ResponseMetadata.HTTPHeaders.
    Prefer retry-after-ms, then parse retry-after as seconds or an HTTP date.
    Absolute dates are compared with the current UTC time.

    Returns:
        A nonnegative delay in seconds, or None for absent/unparseable headers.
        Negative delays and dates already in the past are clamped to zero.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    else:
        headers = getattr(response, "headers", {})
    headers = {k.lower(): v for k, v in headers.items()}
    try:
        if "retry-after-ms" in headers:
            return max(0, float(headers["retry-after-ms"]) / 1000)
        value = headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0, float(value))
        except ValueError:
            return max(
                0,
                (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds(),
            )
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass
class Attempt:
    """One recorded invocation of an operation, including failed retry attempts.

    Attributes:
        operation: Logical operation ID shared by its retry attempts.
        step: Notebook-supplied stage name, such as assess or retrieve_policy.
        kind: Operation category: model, tool, or retrieval.
        attempt: One-based attempt number within the operation.
        status: returned or failed; returned does not imply task correctness.
        elapsed_s: Measured attempt duration, excluding subsequent retry backoff.
        usage: Model token usage when reported; None for other operation types.
        reported_usage_cost_usd: Estimated model cost, or None when unavailable.
        error: Exception type and message for a failed invocation.
        synthetic: Whether a failure came from the deliberate local fixture.
        request_id: SDK/service request identifier when available.
        backoff_s: Retry delay passed to the sleep function after this failure.
        inference_api: responses, converse, or converse_stream when identified.
        model_latency_s: Bedrock-reported latency in seconds, separate from elapsed_s.
        time_to_first_text_s: Time from a stream attempt's request start to its first
            visible text fragment. This is not the full workflow's first-text time.
        requested_cache_mode: Requested explicit-checkpoint setting, not proof of a cache miss/hit.
        cache_prefix_sha256: Digest of the supplied prefix, without storing its content.
        cache_prefix_characters: Prefix length in characters; not a token count.
    """

    operation: int
    step: str
    kind: str
    attempt: int
    status: str
    elapsed_s: float
    usage: Usage | None = None
    reported_usage_cost_usd: float | None = None
    error: str = ""
    synthetic: bool = False
    request_id: str | None = None
    backoff_s: float = 0
    inference_api: str | None = None
    model_latency_s: float | None = None
    time_to_first_text_s: float | None = None
    requested_cache_mode: str | None = None
    cache_prefix_sha256: str | None = None
    cache_prefix_characters: int | None = None


class Run:
    """Record one complete task and apply bounded retries to its operations.

    Use as a context manager, call each measured operation through call(), and
    record the task's correctness with check(). frame() exposes individual attempts;
    summary() aggregates measurements for notebook comparisons.

    The clock starts at construction. Time budgets are checked between operations
    and attempts. Callbacks receive a timeout to apply to their own clients; Run
    does not forcibly interrupt network requests or arbitrary Python code.
    """

    def __init__(
        self,
        label: str,
        settings: Settings,
        rates: Rates | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ):
        """Start a task trace with its settings and optional pricing.

        Args:
            label: Name shown in comparisons and used as the default artifact name.
            settings: Connection identifiers, retry limits, and time budgets.
            rates: Matching token prices, or None to omit cost estimates.
            clock: Monotonic clock returning seconds.
            sleep: Function that waits for a retry delay in seconds.
            jitter: Function selecting a delay between a lower and upper bound.
                Clock, sleep, and jitter can be replaced for deterministic tests.

        Raises:
            ValueError: Supplied prices belong to a different model/profile.
        """
        self.label, self.settings, self.rates = label, settings, rates
        if rates is not None and rates.model_id != settings.model_id:
            raise ValueError("Pricing model_id differs from the model being measured")
        self.clock, self.sleep, self.jitter = clock, sleep, jitter
        self.started = clock()
        self.finished: float | None = None
        self.attempts: list[Attempt] = []
        self.operations = 0
        self.task_passed: bool | None = None
        self.error = ""
        self._responses_client = None

    def __enter__(self):
        """Return this trace for a with block without resetting its start time."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Finish timing and record an escaping exception as a failed task.

        The arguments are supplied by Python's context-manager protocol. Returning
        False lets any exception propagate; a normal exit leaves task_passed at
        the value set by check(), or None if no check was performed.
        """
        if self._responses_client is not None:
            self._responses_client.close()
        self.finished = self.clock()
        if exc is not None:
            self.error = f"{type(exc).__name__}: {exc}"
            self.task_passed = False
        return False

    @property
    def remaining_s(self):
        """Return budget minus elapsed seconds; zero or negative means exhausted."""
        return self.settings.workflow_budget_s - (self.clock() - self.started)

    def call(
        self, step: str, kind: str, invoke: Callable[[float], Any],
        *, on_text: Callable[[str], None] | None = None,
    ):
        """Execute one logical operation, recording each invocation and retry.

        Args:
            step: Descriptive name used in the attempt trace.
            kind: model, tool, or retrieval. Model results are inspected for usage;
                retrieval results are inspected for a service request identifier.
            invoke: Callback accepting a timeout in seconds and returning the
                operation's result. It must apply that timeout itself. Disable
                SDK retries so this method observes every application attempt.
            on_text: Visible-text callback when invoke returns a raw ConverseStream.
                The stream is consumed inside the measured attempt before usage
                is extracted. Leave the owning client open until call() returns.

        Returns:
            The callback's result after a successful return within the budget.
            A returned model response still needs require_completed() or read_json()
            and task-specific correctness checks.

        Raises:
            ValueError: kind is not a supported operation category.
            BudgetExceeded: The budget is exhausted or cannot accommodate the
                server-requested or jittered retry delay.
            Exception: The callback error when it is nonretryable or attempts
                are exhausted. Usage parsing and pricing errors also propagate.

        Failed model invocations may have unknown usage. Returned usage is recorded
        before checking whether the operation overran the workflow budget.
        Errors during stream consumption retain available partial measurements
        and are terminal; errors establishing the request use the normal policy.
        """
        if kind not in {"model", "tool", "retrieval"}:
            raise ValueError(f"Unknown operation kind: {kind}")
        self.operations += 1
        operation = self.operations
        for attempt_number in range(1, self.settings.max_attempts + 1):
            if self.remaining_s <= 0:
                raise BudgetExceeded(f"Task budget exhausted before {step}")
            timeout_s = min(self.settings.attempt_timeout_s, self.remaining_s)
            started = self.clock()
            try:
                result = invoke(timeout_s)
                if kind == "model" and isinstance(result, dict) and "stream" in result:
                    result = consume_converse_stream(
                        result, on_text=on_text, clock=self.clock, started_at=started,
                        deadline=self.started + self.settings.workflow_budget_s,
                    )
            except Exception as exc:
                partial = getattr(exc, "partial_response", None)
                usage = Usage.from_response(partial) if kind == "model" and partial is not None else None
                cost = self.rates.estimate(usage, self.settings.model_id) if self.rates and usage else None
                entry = Attempt(
                    operation, step, kind, attempt_number, "failed", self.clock() - started,
                    usage=usage, reported_usage_cost_usd=cost,
                    error=f"{type(exc).__name__}: {exc}",
                    synthetic=isinstance(exc, SyntheticTransientError),
                    **(_response_measurements(partial, kind) if partial is not None else {}),
                )
                self.attempts.append(entry)
                if not retryable(exc) or attempt_number == self.settings.max_attempts:
                    raise
                delay = retry_after_seconds(exc)
                if delay is None:
                    ceiling = min(
                        self.settings.backoff_cap_s,
                        self.settings.backoff_base_s * 2 ** (attempt_number - 1),
                    )
                    delay = self.jitter(0, ceiling)
                if delay >= self.remaining_s:
                    raise BudgetExceeded(f"Retry delay would exceed the budget for {step}") from exc
                entry.backoff_s = delay
                self.sleep(delay)
                continue

            usage = Usage.from_response(result) if kind == "model" else None
            cost = self.rates.estimate(usage, self.settings.model_id) if self.rates and usage else None
            self.attempts.append(
                Attempt(
                    operation, step, kind, attempt_number, "returned",
                    self.clock() - started, usage=usage,
                    reported_usage_cost_usd=cost,
                    **(_response_measurements(result, kind) if kind != "tool" else {}),
                )
            )
            if self.remaining_s <= 0:
                raise BudgetExceeded(f"Task budget exceeded while running {step}")
            return result
        raise AssertionError("Unreachable")

    def generate(
        self, step: str, messages, *, instructions: str = "", tools=(),
        tool_choice: str = "auto", max_output_tokens: int = 4096,
        temperature: float | None = None, reasoning_effort: str | None = None,
        response_schema: dict | None = None, before_attempt=None,
        cache_prefix: str | None = None, cache_mode: str = "on",
    ) -> InferenceResponse:
        """Generate a complete model turn through the automatically selected API.

        Args:
            step: Stage name used in the shared attempt table.
            messages: A plain text prompt, or a conversation containing role/content
                dictionaries and response.message objects. Tool results can be
                dictionaries with role="tool", tool_call_id, and string content.
            instructions: System instructions, translated to the provider's field.
            tools: Plain name/description/parameters dictionaries or Tool instances.
            tool_choice: auto, required, or the name of one supplied tool.
            max_output_tokens: Total output limit, including any reasoning tokens.
            temperature: Optional sampling control, when supported by the model.
            reasoning_effort: Optional low/medium/high control. OpenAI uses effort;
                supported Claude models use adaptive effort or documented budget mapping.
                None leaves the provider's default thinking behavior unchanged.
            response_schema: Native structured-output schema, also validated locally
                for final answers. The model must support structured outputs.
            before_attempt: Optional demo fault hook, invoked before the SDK request.
            cache_prefix: Stable reference text to place before the changing request.
                Explicit caching is mapped for GPT-5.6 Responses and compatible Claude models.
            cache_mode: on adds a breakpoint. off omits it and requests explicit-only
                mode on GPT-5.6. It is not an empirically guaranteed cache-off switch;
                use returned cache counters to determine what the service actually reused.

        Returns:
            InferenceResponse with the same text, tool_calls, message, and usage
            fields for both APIs. Incomplete output or invalid schemas/arguments
            raises after returned usage has been recorded.

        Routing uses Settings.model_id; API choice never appears in notebook
        orchestration. Model-specific capabilities still apply, and unsupported
        options are surfaced rather than silently removed.
        """
        return self._generate(
            step, messages, stream=False, instructions=instructions, tools=tools,
            tool_choice=tool_choice, max_output_tokens=max_output_tokens,
            temperature=temperature, reasoning_effort=reasoning_effort,
            response_schema=response_schema, before_attempt=before_attempt,
            cache_prefix=cache_prefix, cache_mode=cache_mode,
        )

    def stream(
        self, step: str, messages, *, on_text=None, instructions: str = "", tools=(),
        tool_choice: str = "auto", max_output_tokens: int = 4096,
        temperature: float | None = None, reasoning_effort: str | None = None,
        response_schema: dict | None = None, before_attempt=None,
        cache_prefix: str | None = None, cache_mode: str = "on",
    ) -> InferenceResponse:
        """Stream visible text through the selected API and return the common result.

        Arguments match generate(); on_text receives visible text fragments as
        they arrive. OpenAI uses Responses streaming; other models use ConverseStream.
        The complete stream is consumed inside the measured attempt. Tool calls
        are exposed only after assembly and validation. Consumption errors raise
        ModelStreamError (or BudgetExceeded) without replaying partial output.
        """
        return self._generate(
            step, messages, stream=True, on_text=on_text, instructions=instructions,
            tools=tools, tool_choice=tool_choice, max_output_tokens=max_output_tokens,
            temperature=temperature, reasoning_effort=reasoning_effort,
            response_schema=response_schema, before_attempt=before_attempt,
            cache_prefix=cache_prefix, cache_mode=cache_mode,
        )

    def _generate(self, step, messages, *, stream, on_text=None, before_attempt=None, **options):
        """Translate common arguments, execute one measured request, and normalize it."""
        api = self.settings.resolved_api
        options["tools"] = normalize_tools(options.get("tools", ()))
        request = build_request(api, self.settings.model_id, messages, **options)

        def invoke(timeout):
            """Apply the demo hook and keep each selected client alive through consumption."""
            if before_attempt is not None:
                before_attempt()
            started = self.clock()
            if api == "responses":
                if self._responses_client is None:
                    self._responses_client = make_model_client(self.settings)
                client = self._responses_client.with_options(timeout=timeout)
                if stream:
                    return _consume_responses_stream(
                        client, request, on_text=on_text, clock=self.clock,
                        started_at=started, deadline=self.started + self.settings.workflow_budget_s,
                    )
                return client.responses.create(**request)
            client = make_converse_client(self.settings, timeout)
            try:
                if stream:
                    return consume_converse_stream(
                        client.converse_stream(**request), on_text=on_text, clock=self.clock,
                        started_at=started, deadline=self.started + self.settings.workflow_budget_s,
                    )
                return client.converse(**request)
            finally:
                client.close()

        def measured_invoke(timeout):
            """Annotate the requested cache TTL before usage is priced."""
            raw = invoke(timeout)
            if options.get("cache_prefix") is not None:
                data = dict(as_dict(raw))
                local = dict(data.get("_workshop", {}))
                local.update(
                    inference_api=f"{api}_stream" if stream else api,
                    cache_write_ttl="30m" if api == "responses" else "5m",
                    requested_cache_mode=options.get("cache_mode", "on"),
                    cache_prefix_sha256=sha256(options["cache_prefix"].encode()).hexdigest(),
                    cache_prefix_characters=len(options["cache_prefix"]),
                )
                data["_workshop"] = local
                request_id = getattr(raw, "_request_id", None)
                if request_id:
                    data["ResponseMetadata"] = {"RequestId": request_id}
                return data
            return raw

        raw = self.call(step, "model", measured_invoke)
        require_completed(raw)
        result = normalize_response(
            as_dict(raw), api=f"{api}_stream" if stream else api, model_id=self.settings.model_id,
            usage=Usage.from_response(raw), tools=options.get("tools", ()),
            response_schema=options.get("response_schema"),
        )
        if result.schema is not None and not result.tool_calls:
            read_json(result)
        return result

    def converse(self, step: str, **request) -> dict:
        """Make a measured native Converse call with this Run's model and retries.

        Pass Bedrock request fields such as messages, system, inferenceConfig, and
        toolConfig. modelId defaults to settings.model_id; a different value is
        rejected to keep pricing and trace identity accurate. The helper creates
        and closes the Runtime client for each attempt and returns the full native
        response. Apply require_completed() and task-specific checks afterward.
        """
        return self._converse_request(step, request, stream=False)

    def converse_stream(
        self, step: str, *, on_text: Callable[[str], None] | None = None, **request,
    ) -> dict:
        """Make a measured ConverseStream call and consume it before returning.

        Request fields are identical to the native Boto3 method. on_text receives
        visible text while it arrives. The returned Converse-shaped dictionary
        contains assembled text/tool/reasoning content, stop reason, final usage,
        service metrics, and local first-text timing.

        Client and stream are closed after consumption. Request-establishment
        failures can retry; a failure while consuming the stream is surfaced with
        partial_response and is not silently replayed.
        """
        return self._converse_request(step, request, stream=True, on_text=on_text)

    def _converse_request(self, step: str, request: dict, *, stream: bool, on_text=None):
        """Bind the model, per-attempt client lifetime, and native inference operation."""
        request = dict(request)
        if request.get("modelId", self.settings.model_id) != self.settings.model_id:
            raise ValueError("Request modelId differs from the model being measured")
        request["modelId"] = self.settings.model_id

        def invoke(timeout):
            """Keep the Runtime client open through response or stream consumption."""
            started = self.clock()
            client = make_converse_client(self.settings, timeout)
            try:
                if not stream:
                    return client.converse(**request)
                response = client.converse_stream(**request)
                return consume_converse_stream(
                    response, on_text=on_text, clock=self.clock, started_at=started,
                    deadline=self.started + self.settings.workflow_budget_s,
                )
            finally:
                client.close()

        return self.call(step, "model", invoke)

    def check(self, passed: bool):
        """Store and return the task rubric's Boolean result.

        Each call replaces the previous value rather than combining checks. The
        notebook should combine its criteria before calling this method.
        """
        self.task_passed = bool(passed)
        return self.task_passed

    def frame(self) -> pd.DataFrame:
        """Return one DataFrame row per attempt with token fields flattened.

        Missing usage remains None (which Pandas may display as NaN). Attempts
        stay in execution order, including failures and synthetic fixtures.
        With no recorded attempts, return an empty DataFrame without columns.
        """
        rows = []
        for attempt in self.attempts:
            row = asdict(attempt)
            usage = row.pop("usage")
            row.update(usage or {
                "input_tokens": None, "output_tokens": None,
                "cached_input_tokens": None, "cache_write_tokens": None,
                "reasoning_tokens": None,
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def metrics(self) -> pd.DataFrame:
        """Return a compact attempt table for a code talk.

        Full measurements, including cache and service-latency details, remain
        available from frame(). Missing token counts remain missing, not zero.
        """
        columns = {
            "step": "Step", "attempt": "Attempt", "status": "Result",
            "input_tokens": "Input", "output_tokens": "Output",
            "reasoning_tokens": "Reasoning", "elapsed_s": "Seconds",
            "backoff_s": "Retry wait",
        }
        return self.frame().reindex(columns=columns).rename(columns=columns).round(3)

    def summary(self) -> dict:
        """Aggregate attempt counts, reported usage, durations, cost, and task result.

        Logical model calls are distinct operation IDs with recorded attempts.
        Attempt counts include retries and synthetic failures; synthetic failures
        are also counted separately. Durations by kind exclude retry backoff,
        while elapsed_s covers the entire task through exit or the current time.
        reported_model_latency_s sums only available Bedrock service metrics.
        First-text timings stay in frame(), because each belongs to one stream
        attempt rather than the complete multi-step task.

        Token totals include only reported usage. A token category remains None
        if any reported response lacks that category. Cost is summed only when
        every response with usage can be priced; it can still exclude real failed
        attempts whose usage is unknown. usage_complete flags that missing usage
        and is also False when no model usage was reported.
        """
        model = [a for a in self.attempts if a.kind == "model"]
        reported = [a for a in model if a.usage is not None]
        unknown = [a for a in model if a.usage is None and not a.synthetic]
        priced = [a.reported_usage_cost_usd for a in reported]
        service_latencies = [a.model_latency_s for a in model if a.model_latency_s is not None]
        elapsed = (self.finished if self.finished is not None else self.clock()) - self.started
        totals = {
            name: sum(getattr(a.usage, name) for a in reported)
            if reported and all(getattr(a.usage, name) is not None for a in reported) else None
            for name in (
                "input_tokens", "output_tokens", "cached_input_tokens",
                "cache_write_tokens", "reasoning_tokens",
            )
        }
        return {
            "run": self.label,
            "model_id": self.settings.model_id,
            "region": self.settings.region,
            "inference_apis": ",".join(sorted({a.inference_api for a in model if a.inference_api})),
            "logical_model_calls": len({a.operation for a in model}),
            "model_attempts": len(model),
            "retrieval_attempts": sum(a.kind == "retrieval" for a in self.attempts),
            "tool_attempts": sum(a.kind == "tool" for a in self.attempts),
            "retries": sum(a.attempt > 1 for a in self.attempts),
            "synthetic_failures": sum(a.synthetic for a in self.attempts),
            "model_attempts_without_usage": len(unknown),
            **totals,
            "model_s": sum(a.elapsed_s for a in model),
            "reported_model_latency_s": sum(service_latencies) if service_latencies else None,
            "retrieval_s": sum(a.elapsed_s for a in self.attempts if a.kind == "retrieval"),
            "tool_s": sum(a.elapsed_s for a in self.attempts if a.kind == "tool"),
            "backoff_s": sum(a.backoff_s for a in self.attempts),
            "elapsed_s": elapsed,
            "reported_usage_cost_usd": sum(priced)
            if priced and all(p is not None for p in priced) else None,
            "usage_complete": not unknown and bool(reported),
            "task_passed": self.task_passed,
        }

    def save(self, name: str | None = None):
        """Write the current summary, attempts, and terminal error to a JSON file.

        Args:
            name: Artifact filename stem; defaults to the run label.

        Returns:
            The written Path under artifact_directory(). Parent directories are
            created, and an existing file with the same name is overwritten.

        Request/response payloads are not saved, although recorded exception
        messages are included. Call after the with block to save final timing.
        """
        directory = artifact_directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name or self.label}.json"
        path.write_text(json.dumps({
            "summary": self.summary(),
            "attempts": [asdict(a) for a in self.attempts],
            "error": self.error,
        }, indent=2))
        return path


def compare(*runs: Run, detailed: bool = False) -> pd.DataFrame:
    """Compare task totals; request detailed=True for all recorded summary fields.

    Pass baseline and tuned runs for the same task. The function presents their
    measurements without deciding whether a difference is a reliable improvement.
    The compact view does not imply that answer quality has been checked.
    """
    frame = pd.DataFrame([r.summary() for r in runs]).set_index("run")
    if detailed:
        return frame
    columns = {
        "logical_model_calls": "Calls", "input_tokens": "Input",
        "output_tokens": "Output", "elapsed_s": "Seconds",
        "retries": "Retries", "reported_usage_cost_usd": "Reported USD",
        "usage_complete": "Usage complete",
    }
    if frame["model_id"].nunique() > 1:
        columns = {"model_id": "Model", **columns}
    return frame[list(columns)].rename(columns=columns)


def artifact_directory() -> Path:
    """Return BEDROCK_ARTIFACT_DIR or the repository's artifacts/ directory."""
    return Path(os.getenv("BEDROCK_ARTIFACT_DIR", ROOT / "artifacts"))


def require_completed(response):
    """Require a completed Responses result or a successful native Converse turn.

    Call after Run.call so usage from incomplete or failed model output is still
    recorded. HTTP success alone does not establish response completion.

    Raises:
        IncompleteResponse: Responses status is not completed, or Converse
            stopReason is not end_turn, stop_sequence, or tool_use. A tool_use
            turn is complete but still requires application tool execution.
    """
    data = as_dict(response)
    if "stopReason" in data or isinstance(data.get("output"), dict) or "stream" in data:
        if data.get("stopReason") not in {"end_turn", "stop_sequence", "tool_use"}:
            raise IncompleteResponse(f"Converse stopReason={data.get('stopReason')}")
    elif data.get("status") != "completed":
        raise IncompleteResponse(
            f"Response status={data.get('status')}; "
            f"details={data.get('incomplete_details')}; error={data.get('error')}"
        )
    return response


def output_text(response) -> str:
    """Extract visible answer text from Responses or an assembled Converse result.

    Reasoning, tool calls, and other content blocks are preserved in the original
    result but omitted here. Use require_completed() separately before treating
    this text as a completed answer.
    """
    if isinstance(response, InferenceResponse):
        return response.text
    data = as_dict(response)
    if "stream" in data:
        raise ValueError("Consume ConverseStream before extracting its answer")
    output = data.get("output", [])
    if isinstance(output, dict):
        return "".join(
            block["text"] for block in output.get("message", {}).get("content", [])
            if "text" in block
        )
    return "".join(
        block["text"] for item in output if item.get("type") == "message"
        for block in item.get("content", []) if block.get("type") == "output_text"
    )


def json_format(name: str, properties: dict, required: list[str] | None = None):
    """Build the Responses text configuration for a strict JSON object schema.

    Args:
        name: Schema name sent to the API.
        properties: Mapping of field names to their JSON Schema definitions.
        required: Required field names; None defaults to every property.

    Returns:
        A dictionary suitable for responses.create(text=...), with strict mode
        enabled and additional object properties forbidden. Nested schemas are
        passed through as supplied; this helper does not validate API support.
    """
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required if required is not None else list(properties),
                "additionalProperties": False,
            },
        }
    }


def read_json(response, schema: dict | None = None) -> dict:
    """Parse visible JSON from Responses or Converse and optionally validate it.

    The notebooks request object schemas. An explicit schema is validated for
    either API. Otherwise, if Responses echoes a json_schema
    format with a schema, validate the parsed value locally against that schema.
    Otherwise, only completion and JSON parsing are checked.

    Raises:
        IncompleteResponse: The response status is not completed.
        json.JSONDecodeError: The model's visible output is not valid JSON.
        jsonschema.ValidationError: Parsed output violates the echoed schema.
        jsonschema.SchemaError: The echoed schema itself is invalid.
    """
    import jsonschema

    require_completed(response)
    if schema is None and isinstance(response, InferenceResponse):
        schema = response.schema
    result = json.loads(output_text(response))
    # Validate against the requested schema locally as well when echoed by the API.
    fmt = (as_dict(response).get("text") or {}).get("format", {})
    if schema is not None:
        jsonschema.validate(result, schema)
    elif fmt.get("type") == "json_schema" and "schema" in fmt:
        jsonschema.validate(result, fmt["schema"])
    return result


def load_order() -> dict:
    """Read a fresh copy of the fixed demo order snapshot from data/order.json.

    The fixture's as_of_date is intentional: comparisons use the same facts
    regardless of when the notebook runs.
    """
    return json.loads((ROOT / "data/order.json").read_text())


def order_facts(order: dict) -> dict:
    """Select the seven order fields needed by the support-task demonstrations.

    Return a new dictionary containing identity, status, dates, delay, carrier,
    and next step. Internal fulfillment and audit fields are omitted to create
    the trimmed tool result used in the cost comparison.

    Raises:
        KeyError: The supplied order lacks one of the required fixture fields.
    """
    return {key: order[key] for key in (
        "order_id", "status", "promised_delivery_date", "as_of_date",
        "days_late", "carrier", "next_step",
    )}
