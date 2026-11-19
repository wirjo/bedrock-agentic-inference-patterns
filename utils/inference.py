"""Portable text/tool contracts and request translation for workshop.Run.

This module performs no network requests. Run.generate() and Run.stream() own
execution, retries, timing, and usage collection. Native continuation data remains
inside Message so callers never have to translate tool IDs or reasoning blocks.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import jsonschema


class UnsupportedInferenceOption(ValueError):
    """An explicitly requested control cannot be mapped to the selected model."""


def resolve_api(model_id: str, override: str = "auto") -> str:
    """Route OpenAI model/profile IDs to Responses and other models to Converse.

    An opaque application-profile or prompt ARN cannot identify its model. Set
    override to responses or converse for those resources instead of guessing.
    """
    if override not in {"auto", "responses", "converse"}:
        raise ValueError("inference_api must be auto, responses, or converse")
    if override != "auto":
        return override
    if any(part in model_id for part in ("/application-inference-profile/", ":application-inference-profile/", ":prompt/")):
        raise ValueError("Set inference_api explicitly for an opaque application-profile or prompt ARN")
    return "responses" if re.search(r"(?:^|[./])openai\.", model_id) else "converse"


@dataclass(frozen=True)
class Tool:
    """A callable tool's provider-independent name, description, and JSON schema."""

    name: str
    description: str
    parameters: dict

    def __post_init__(self):
        """Validate the declared schema before sending it to a model."""
        if not self.name:
            raise ValueError("Tool name must not be empty")
        jsonschema.Draft202012Validator.check_schema(self.parameters)


def normalize_tools(tools) -> tuple[Tool, ...]:
    """Accept plain tool dictionaries while retaining the existing Tool API.

    Each dictionary contains name, description, and parameters. Validation and
    provider-specific packaging remain inside the inference layer.
    """
    normalized = []
    for tool in tools:
        if isinstance(tool, dict):
            tool = Tool(**tool)
        if not isinstance(tool, Tool):
            raise TypeError("Tools must be dictionaries or Tool instances")
        normalized.append(tool)
    return tuple(normalized)


@dataclass(frozen=True)
class ToolCall:
    """One requested tool invocation with a stable ID and parsed JSON arguments."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Message:
    """A portable conversation item, including opaque assistant continuation data.

    Ordinary input can use a text prompt or dictionaries with role and content.
    Append response.message for assistant turns. Tool results can use plain
    role/tool_call_id/content dictionaries or tool_result(call, value).
    The private native data preserves signatures and reasoning items.
    Generated messages belong to one model; start fresh history when switching it.
    """

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    is_error: bool = False
    _api: str | None = field(default=None, repr=False)
    _model_id: str | None = field(default=None, repr=False)
    _native: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class InferenceResponse:
    """Common result from generate() and stream(), regardless of the chosen API.

    Attributes:
        text: Visible answer text, excluding internal reasoning.
        tool_calls: Validated tool calls with parsed argument dictionaries.
        message: Assistant continuation item to append to the input history.
        usage: Normalized workshop.Usage, or None when unreported.
        stop_reason: Common end_turn, tool_use, max_tokens, or provider error reason.
        api: Actual request API, including the streaming suffix when applicable.
        schema: Requested JSON response schema, when supplied.
        raw_response: Native result for diagnostics; excluded from repr.
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    message: Message
    usage: Any
    stop_reason: str
    api: str
    schema: dict | None = field(default=None, repr=False)
    raw_response: dict = field(default_factory=dict, repr=False)


def tool_result(call: ToolCall, value: Any, *, is_error: bool = False) -> Message:
    """Create a portable tool-result message linked to the original tool call.

    Strings are sent as-is; other JSON-serializable values are encoded as JSON.
    The adapter supplies function_call_output or toolResult on the appropriate API.
    """
    content = value if isinstance(value, str) else json.dumps(value)
    return Message("tool", content, tool_call_id=call.id, is_error=is_error)


def object_schema(name: str, properties: dict, required: list[str] | None = None) -> dict:
    """Build a named, closed JSON object schema shared by both inference APIs."""
    return {
        "title": name,
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _messages(messages, api: str, model_id: str) -> list[Message]:
    """Accept a text prompt or conversation and preserve native assistant state.

    Plain tool-result dictionaries use role, tool_call_id, and string content.
    Assistant tool-call turns still use response.message to preserve reasoning.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    result = []
    for value in messages:
        if isinstance(value, dict):
            if set(value) - {"role", "content", "tool_call_id", "is_error"}:
                raise ValueError("Use text messages or plain tool-result dictionaries")
            value = Message(**value)
        if not isinstance(value, Message):
            raise TypeError("messages must contain Message objects or role/content dictionaries")
        if value.role not in {"user", "assistant", "tool"} or not isinstance(value.content, str):
            raise ValueError("Use text user/assistant messages; pass system text through instructions")
        if value.role == "tool" and not value.tool_call_id:
            raise ValueError("A tool result requires its call ID")
        if value.role != "tool" and (value.tool_call_id is not None or value.is_error):
            raise ValueError("Only tool-result messages can carry tool_call_id or is_error")
        if value.tool_calls and value._native is None:
            raise ValueError("Append response.message to preserve the original tool-call turn")
        if value._native is not None and (value._api != api or value._model_id != model_id):
            raise ValueError("Start fresh conversation history when switching model or inference API")
        result.append(value)
    if not result:
        raise ValueError("messages must not be empty")
    return result


def _openai_messages(messages: list[Message]) -> list[dict]:
    """Translate portable messages, preserving every native Responses output item."""
    result = []
    for message in messages:
        if message._native is not None:
            result.extend(deepcopy(message._native))
        elif message.role == "tool":
            output = json.dumps({"error": message.content}) if message.is_error else message.content
            result.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": output})
        else:
            result.append({"role": message.role, "content": message.content})
    return result


def _converse_messages(messages: list[Message]) -> list[dict]:
    """Translate messages and group adjacent tool results into one native user turn."""
    result = []
    for message in messages:
        if message._native is not None:
            native = deepcopy(message._native)
        elif message.role == "tool":
            native = {"role": "user", "content": [{"toolResult": {
                "toolUseId": message.tool_call_id,
                "content": [{"text": message.content}],
                "status": "error" if message.is_error else "success",
            }}]}
        else:
            native = {"role": message.role, "content": [{"text": message.content}]}
        if result and result[-1]["role"] == native["role"]:
            result[-1]["content"].extend(native["content"])
        else:
            result.append(native)
    return result


def _converse_reasoning(model_id: str, effort: str, max_tokens: int) -> dict:
    """Map common effort levels to documented Claude thinking controls.

    Adaptive Claude models use output_config.effort. Older thinking-capable Claude
    models use explicit workshop budgets: low=1024, medium=2048, high=3072 tokens.
    These budgets are a workshop policy, not provider-defined equivalent effort.
    Unknown or non-thinking models fail rather than silently ignoring the option.
    """
    model = model_id.lower()
    adaptive = re.search(
        r"anthropic\.claude-(?:(?:opus|sonnet)-4-[67]|(?:opus|sonnet|fable|mythos)-5|mythos-preview)",
        model,
    )
    if adaptive:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    extended = re.search(r"anthropic\.claude-(?:3-7-|(?:opus|sonnet)-4(?:-|$)|haiku-4-5)", model)
    if extended:
        budget = {"low": 1024, "medium": 2048, "high": 3072}[effort]
        if budget >= max_tokens:
            raise UnsupportedInferenceOption(
                f"{effort} effort needs max_output_tokens above the {budget}-token thinking budget"
            )
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    raise UnsupportedInferenceOption(
        f"No reasoning_effort mapping is known for {model_id}; omit the option or select a supported model"
    )


def build_request(
    api: str,
    model_id: str,
    messages,
    *,
    instructions: str = "",
    tools: list[Tool | dict] | tuple[Tool | dict, ...] = (),
    tool_choice: str = "auto",
    max_output_tokens: int = 4096,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    response_schema: dict | None = None,
    cache_prefix: str | None = None,
    cache_mode: str = "on",
) -> dict:
    """Translate the shared inference arguments into one provider's request schema.

    JSON output uses native structured-output fields on both APIs. Models must
    support the requested feature; there is no silent prompt-only fallback.
    Tool arguments and final JSON are also validated locally after inference.
    """
    if api not in {"responses", "converse"}:
        raise ValueError(f"Unsupported inference API: {api}")
    if cache_mode not in {"on", "off"}:
        raise ValueError("cache_mode must be on or off")
    if cache_mode == "off" and cache_prefix is None:
        raise ValueError("cache_mode requires a cache_prefix")
    if cache_prefix is not None and not cache_prefix.strip():
        raise ValueError("cache_prefix must contain reference text")
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    if reasoning_effort not in {None, "low", "medium", "high"}:
        raise UnsupportedInferenceOption("reasoning_effort must be low, medium, high, or None")
    tools = normalize_tools(tools)
    tool_names = [tool.name for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        raise ValueError("Tool names must be unique")
    if tool_choice not in {"auto", "required", *tool_names} or (not tools and tool_choice != "auto"):
        raise ValueError("tool_choice must be auto, required, or the name of a supplied tool")
    if response_schema is not None:
        jsonschema.Draft202012Validator.check_schema(response_schema)
    messages = _messages(messages, api, model_id)

    if api == "responses":
        request = {
            "model": model_id, "input": _openai_messages(messages),
            "max_output_tokens": max_output_tokens, "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            request["instructions"] = instructions
        if tools:
            request["tools"] = [
                {"type": "function", "name": t.name, "description": t.description,
                 "parameters": t.parameters, "strict": True} for t in tools
            ]
            request["tool_choice"] = (
                tool_choice if tool_choice in {"auto", "required"}
                else {"type": "function", "name": tool_choice}
            )
            request["parallel_tool_calls"] = False
        if response_schema is not None:
            request["text"] = {"format": {
                "type": "json_schema", "name": response_schema.get("title", "answer"),
                "strict": True, "schema": response_schema,
            }}
        if reasoning_effort is not None:
            request["reasoning"] = {"effort": reasoning_effort}
        if temperature is not None:
            request["temperature"] = temperature
        if cache_prefix is not None:
            if not re.search(r"openai\.gpt-5\.6(?:-|$)", model_id):
                raise UnsupportedInferenceOption(
                    "The explicit Responses cache mapping currently supports GPT-5.6 models"
                )
            block = {"type": "input_text", "text": cache_prefix}
            if cache_mode == "on":
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            request["input"].insert(0, {
                "type": "message", "role": "developer", "content": [block],
            })
            request["extra_body"] = {
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
                "prompt_cache_key": "bedrock-inference-" + sha256(cache_prefix.encode()).hexdigest()[:20],
            }
        return request

    request = {
        "modelId": model_id, "messages": _converse_messages(messages),
        "inferenceConfig": {"maxTokens": max_output_tokens},
    }
    if instructions:
        request["system"] = [{"text": instructions}]
    if tools:
        choice = (
            {"auto": {}} if tool_choice == "auto" else
            {"any": {}} if tool_choice == "required" else {"tool": {"name": tool_choice}}
        )
        request["toolConfig"] = {
            "tools": [{"toolSpec": {
                "name": t.name, "description": t.description, "inputSchema": {"json": t.parameters},
            }} for t in tools],
            "toolChoice": choice,
        }
    if response_schema is not None:
        request["outputConfig"] = {"textFormat": {
            "type": "json_schema",
            "structure": {"jsonSchema": {
                "name": response_schema.get("title", "answer"), "schema": json.dumps(response_schema),
            }},
        }}
    if reasoning_effort is not None:
        if temperature is not None or tool_choice != "auto":
            raise UnsupportedInferenceOption("Claude thinking requires default temperature and automatic tool choice")
        request["additionalModelRequestFields"] = _converse_reasoning(model_id, reasoning_effort, max_output_tokens)
    if temperature is not None:
        request["inferenceConfig"]["temperature"] = temperature
    if cache_prefix is not None:
        if "anthropic.claude-" not in model_id:
            raise UnsupportedInferenceOption(
                "The native cache-prefix mapping currently supports Claude; check model cache support"
            )
        request.setdefault("system", []).append({"text": cache_prefix})
        if cache_mode == "on":
            request["system"].append({"cachePoint": {"type": "default", "ttl": "5m"}})
    return request


def normalize_response(
    data: dict, *, api: str, model_id: str, usage, tools=(), response_schema=None,
) -> InferenceResponse:
    """Expose the same result fields while retaining native assistant state privately."""
    backend = "responses" if api.startswith("responses") else "converse"
    calls = []
    if backend == "responses":
        native = deepcopy(data.get("output", []))
        text = "".join(
            part["text"] for item in native if item.get("type") == "message"
            for part in item.get("content", []) if part.get("type") == "output_text"
        )
        for item in native:
            if item.get("type") == "function_call":
                calls.append(ToolCall(item["call_id"], item["name"], json.loads(item["arguments"])))
        stop_reason = "tool_use" if calls else "end_turn"
    else:
        native = deepcopy(data["output"]["message"])
        text = "".join(block["text"] for block in native["content"] if "text" in block)
        for block in native["content"]:
            if "toolUse" in block:
                item = block["toolUse"]
                calls.append(ToolCall(item["toolUseId"], item["name"], deepcopy(item["input"])))
        stop_reason = data["stopReason"]
    schemas = {tool.name: tool.parameters for tool in normalize_tools(tools)}
    for call in calls:
        if call.name not in schemas:
            raise ValueError(f"Model requested an undeclared tool: {call.name}")
        jsonschema.validate(call.arguments, schemas[call.name])
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("Model returned duplicate tool-call IDs")
    message = Message(
        "assistant", text, tuple(calls), _api=backend, _model_id=model_id, _native=native,
    )
    return InferenceResponse(
        text, tuple(calls), message, usage, stop_reason, api, response_schema, data,
    )
