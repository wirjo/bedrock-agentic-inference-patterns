import json
from copy import deepcopy

import boto3
import pytest
from botocore.exceptions import ClientError, EventStreamError, ParamValidationError
from botocore.stub import Stubber
from botocore.validate import validate_parameters

from utils import workshop as w
from test_workshop import Clock


MODEL = "us.amazon.nova-pro-v1:0"
SETTINGS = w.Settings(model_id=MODEL, backoff_base_s=0, backoff_cap_s=0)
MESSAGES = [{"role": "user", "content": [{"text": "Hello"}]}]
NATIVE_USAGE = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
NATIVE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Hello world"}]}},
    "stopReason": "end_turn",
    "usage": NATIVE_USAGE,
    "metrics": {"latencyMs": 450},
    "ResponseMetadata": {"RequestId": "native-request", "HTTPStatusCode": 200},
}


class Events:
    def __init__(self, events, clock=None):
        self.events = events
        self.clock = clock
        self.closed = False

    def __iter__(self):
        for event in self.events:
            if self.clock:
                self.clock.sleep(0.1)
            if isinstance(event, Exception):
                raise event
            yield event

    def close(self):
        self.closed = True


def text_events(stop_reason="end_turn"):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello "}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "world"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": stop_reason}},
        {"metadata": {"usage": NATIVE_USAGE, "metrics": {"latencyMs": 450}}},
    ]


class StreamClient:
    def __init__(self, stream, clock=None, opening_error=None):
        self.stream = stream
        self.clock = clock
        self.opening_error = opening_error
        self.closed = False
        self.requests = []

    def converse_stream(self, **request):
        service = boto3.Session()._session.get_service_model("bedrock-runtime")
        validate_parameters(request, service.operation_model("ConverseStream").input_shape)
        self.requests.append(request)
        if self.clock:
            self.clock.sleep(0.3)
        if self.opening_error:
            raise self.opening_error
        return {"stream": self.stream, "ResponseMetadata": {"RequestId": "stream-request"}}

    def close(self):
        self.closed = True


def test_runtime_factory_disables_hidden_retries(monkeypatch):
    captured = {}

    def factory(name, **kwargs):
        captured.update(service=name, **kwargs)
        return "client"

    monkeypatch.setattr(w.boto3, "client", factory)
    assert w.make_converse_client(SETTINGS, 2) == "client"
    assert captured["service"] == "bedrock-runtime"
    assert captured["config"].retries == {"mode": "standard", "total_max_attempts": 1}
    assert captured["config"].read_timeout == 2
    assert captured["config"].connect_timeout == 2


def test_converse_cache_usage_normalizes_disjoint_token_categories():
    response = deepcopy(NATIVE_RESPONSE)
    response["usage"].update(cacheReadInputTokens=1000, cacheWriteInputTokens=200)
    usage = w.Usage.from_response(response)
    assert usage.input_tokens == 1210
    assert usage.cached_input_tokens == 1000
    assert usage.cache_write_tokens == 200
    assert usage.reasoning_tokens is None
    response["usage"]["cacheWriteInputTokens"] = 0
    usage = w.Usage.from_response(response)
    rates = w.Rates(MODEL, 2, 8, cache_read_per_million=0.5)
    assert rates.estimate(usage, MODEL) == pytest.approx((10 * 2 + 1000 * 0.5 + 5 * 8) / 1e6)


def test_converse_absent_cache_fields_mean_no_cache_and_no_reasoning_breakdown():
    usage = w.Usage.from_response(NATIVE_RESPONSE)
    assert usage == w.Usage(10, 5, 0, 0, None)


def test_rates_for_other_model_can_remain_unconfigured(monkeypatch, tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"model_id": w.DEFAULT_MODEL, "input_per_million": 2, "output_per_million": 12}))
    monkeypatch.setenv("BEDROCK_PRICE_FILE", str(path))
    assert w.Rates.load(model_id=MODEL) is None
    assert w.Rates.load().model_id == w.DEFAULT_MODEL


def test_native_converse_request_retry_and_measurements(monkeypatch):
    created = []
    stubbers = []

    def factory(settings, timeout):
        client = boto3.client(
            "bedrock-runtime", region_name=settings.region,
            aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="dummy",
        )
        stubber = Stubber(client)
        expected = {"modelId": MODEL, "messages": MESSAGES, "inferenceConfig": {"maxTokens": 128}}
        if not created:
            stubber.add_client_error(
                "converse", service_error_code="ThrottlingException", http_status_code=429,
                expected_params=expected,
            )
        else:
            stubber.add_response("converse", deepcopy(NATIVE_RESPONSE), expected)
        stubber.activate()
        created.append(client)
        stubbers.append(stubber)
        return client

    monkeypatch.setattr(w, "make_converse_client", factory)
    with w.Run("native", SETTINGS) as run:
        response = run.converse("answer", messages=MESSAGES, inferenceConfig={"maxTokens": 128})
        w.require_completed(response)
        run.check(w.output_text(response) == "Hello world")
    for stubber in stubbers:
        stubber.assert_no_pending_responses()
        stubber.deactivate()
    assert len(created) == 2
    assert run.summary()["logical_model_calls"] == 1
    assert run.summary()["retries"] == 1
    assert run.summary()["input_tokens"] == 10
    assert run.summary()["usage_complete"] is False
    assert run.attempts[-1].request_id == "native-request"
    assert run.attempts[-1].model_latency_s == 0.45
    assert run.summary()["model_id"] == MODEL
    assert run.summary()["inference_apis"] == "converse"


def test_stream_consumed_in_measured_attempt_and_closed(monkeypatch):
    clock = Clock()
    stream = Events(text_events(), clock)
    client = StreamClient(stream, clock)
    monkeypatch.setattr(w, "make_converse_client", lambda settings, timeout: client)
    fragments = []
    with w.Run("stream", SETTINGS, clock=clock, sleep=clock.sleep) as run:
        result = run.converse_stream(
            "answer", messages=MESSAGES, inferenceConfig={"maxTokens": 128},
            on_text=fragments.append,
        )
        w.require_completed(result)
        run.check(w.output_text(result) == "Hello world")
    assert fragments == ["Hello ", "world"]
    assert stream.closed and client.closed
    assert run.attempts[0].time_to_first_text_s == pytest.approx(0.5)
    assert run.attempts[0].elapsed_s == pytest.approx(0.9)
    assert run.attempts[0].model_latency_s == 0.45
    assert run.attempts[0].request_id == "stream-request"
    assert run.attempts[0].inference_api == "converse_stream"
    assert run.summary()["output_tokens"] == 5
    assert run.summary()["reported_model_latency_s"] == 0.45


def test_generic_run_call_also_consumes_raw_stream():
    stream = Events(text_events())
    with w.Run("raw-stream", SETTINGS) as run:
        result = run.call("answer", "model", lambda timeout: {"stream": stream})
    assert stream.closed
    assert w.output_text(result) == "Hello world"
    assert run.summary()["input_tokens"] == 10


def test_raw_stream_cannot_be_mistaken_for_final_usage():
    with pytest.raises(ValueError, match="Consume"):
        w.Usage.from_response({"stream": Events([])})


def test_tool_input_and_signed_reasoning_survive_stream_assembly():
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "reason"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {
            "toolUse": {"toolUseId": "tool-1", "name": "get_order_status"},
        }}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"order_id":'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"ORDER-1042"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": NATIVE_USAGE, "metrics": {"latencyMs": 1}}},
    ]
    visible = []
    result = w.consume_converse_stream({"stream": Events(events)}, on_text=visible.append)
    assert w.require_completed(result) is result
    content = result["output"]["message"]["content"]
    assert content[0] == {"reasoningContent": {"reasoningText": {"text": "reason", "signature": "sig"}}}
    assert content[1]["toolUse"]["input"] == {"order_id": "ORDER-1042"}
    assert content[1]["toolUse"]["toolUseId"] == "tool-1"
    assert not visible and w.output_text(result) == ""


def test_redacted_reasoning_bytes_preserved():
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": b"one"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": b"two"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": NATIVE_USAGE}},
    ]
    result = w.consume_converse_stream({"stream": Events(events)})
    assert result["output"]["message"]["content"][0]["reasoningContent"]["redactedContent"] == b"onetwo"


@pytest.mark.parametrize("error", [
    {"modelStreamErrorException": {"message": "interrupted"}},
    EventStreamError({"Error": {"Code": "ModelStreamErrorException", "Message": "interrupted"}}, "ConverseStream"),
])
def test_midstream_errors_never_replay_partial_output(monkeypatch, error):
    stream = Events(text_events()[:2] + [error])
    client = StreamClient(stream)
    monkeypatch.setattr(w, "make_converse_client", lambda settings, timeout: client)
    delivered = []
    run = w.Run("partial", SETTINGS)
    with pytest.raises(w.ConverseStreamError) as caught, run:
        run.converse_stream("answer", messages=MESSAGES, on_text=delivered.append)
    assert delivered == ["Hello "]
    assert len(client.requests) == 1
    assert len(run.attempts) == 1
    assert stream.closed and client.closed
    assert run.summary()["usage_complete"] is False
    assert w.output_text(caught.value.partial_response) == "Hello "


def test_opening_throttle_can_retry_before_stream_consumption(monkeypatch):
    error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "wait"}}, "ConverseStream")
    clients = [StreamClient(None, opening_error=error), StreamClient(Events(text_events()))]
    pending = iter(clients)
    monkeypatch.setattr(w, "make_converse_client", lambda settings, timeout: next(pending))
    with w.Run("opening", SETTINGS) as run:
        run.converse_stream("answer", messages=MESSAGES)
    assert all(client.closed for client in clients)
    assert run.summary()["retries"] == 1
    assert run.summary()["output_tokens"] == 5


def test_late_stream_failure_keeps_final_usage_when_already_received():
    events = Events(text_events() + [RuntimeError("connection failed after metadata")])
    run = w.Run("late", SETTINGS)
    with pytest.raises(w.ConverseStreamError), run:
        run.call("answer", "model", lambda timeout: {"stream": events})
    assert run.summary()["output_tokens"] == 5
    assert run.summary()["usage_complete"] is True
    assert run.task_passed is False


def test_missing_final_metadata_is_an_explicit_truncated_stream():
    events = Events(text_events()[:-1])
    with pytest.raises(w.ConverseStreamError, match="Truncated"):
        w.consume_converse_stream({"stream": events})
    assert events.closed


def test_callback_failure_is_terminal_and_closes_stream():
    stream = Events(text_events())

    def broken_callback(text):
        raise ValueError("display failed")

    run = w.Run("callback", SETTINGS)
    with pytest.raises(w.ConverseStreamError, match="display failed"), run:
        run.call("answer", "model", lambda timeout: {"stream": stream}, on_text=broken_callback)
    assert len(run.attempts) == 1
    assert stream.closed


def test_stream_budget_exhaustion_closes_stream():
    clock = Clock()
    settings = w.Settings(model_id=MODEL, workflow_budget_s=0.25)
    events = Events(text_events(), clock)
    run = w.Run("deadline", settings, clock=clock, sleep=clock.sleep)
    with pytest.raises(w.BudgetExceeded), run:
        run.call("answer", "model", lambda timeout: {"stream": events})
    assert len(run.attempts) == 1
    assert events.closed
    assert run.task_passed is False


@pytest.mark.parametrize("stop", ["max_tokens", "guardrail_intervened", "content_filtered", "malformed_tool_use"])
def test_unsuccessful_native_stop_reason_preserves_usage(stop):
    stream = Events(text_events(stop_reason=stop))
    run = w.Run("incomplete", SETTINGS)
    with pytest.raises(w.IncompleteResponse), run:
        result = run.call("answer", "model", lambda timeout: {"stream": stream})
        w.require_completed(result)
    assert run.summary()["output_tokens"] == 5
    assert len(run.attempts) == 1


def test_converse_json_can_be_validated_against_explicit_schema():
    result = deepcopy(NATIVE_RESPONSE)
    result["output"]["message"]["content"] = [{"text": '{"answer": 42}'}]
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
    assert w.read_json(result, schema=schema) == {"answer": 42}


def test_request_model_mismatch_is_rejected_before_a_call():
    run = w.Run("wrong-model", SETTINGS)
    with pytest.raises(ValueError, match="modelId"):
        run.converse("answer", modelId="different-model", messages=MESSAGES)
    assert not run.attempts


def test_unsupported_stream_content_is_not_silently_dropped():
    stream = Events([
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"image": {"source": {}}}}},
    ])
    with pytest.raises(w.ConverseStreamError, match="Unsupported"):
        w.consume_converse_stream({"stream": stream})
    assert stream.closed
