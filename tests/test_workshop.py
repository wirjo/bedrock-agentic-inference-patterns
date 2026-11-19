from dataclasses import replace
from types import SimpleNamespace

import httpx2
import openai
import pytest
from botocore.exceptions import ClientError

from utils import workshop as w


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def response(usage=None, status="completed"):
    return {
        "usage": usage or {
            "input_tokens": 100, "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30},
        },
        "status": status,
    }


def test_reasoning_not_double_counted_and_cached_input_priced_separately():
    usage = w.Usage.from_response(response())
    rates = w.Rates(w.DEFAULT_MODEL, 2, 8, cache_read_per_million=0.5)
    assert rates.estimate(usage, w.DEFAULT_MODEL) == pytest.approx((80 * 2 + 20 * 0.5 + 50 * 8) / 1e6)
    with w.Run("usage", w.Settings(), rates) as run:
        run.call("answer", "model", lambda _: response())
    assert run.summary()["output_tokens"] == 50
    assert run.summary()["reasoning_tokens"] == 30


def test_unknown_cost_and_unsupported_cache_write_are_not_zero():
    rates = w.Rates(w.DEFAULT_MODEL, 2, 8)
    assert rates.estimate(w.Usage(100, 50, 20, 0, 30), w.DEFAULT_MODEL) is None
    assert rates.estimate(w.Usage(100, 50, 0, 100, 30), w.DEFAULT_MODEL) is None
    with pytest.raises(ValueError, match="model_id"):
        rates.estimate(w.Usage(100, 50, 0, 0, 30), "another-model")


def test_short_context_rates_do_not_silently_price_long_context():
    rates = w.Rates(w.DEFAULT_MODEL, 2, 12, max_input_tokens=272000)
    assert rates.estimate(w.Usage(272001, 50, 0, 0, 30), w.DEFAULT_MODEL) is None
    assert rates.estimate(w.Usage(272000, 50, 0, 0, 30), w.DEFAULT_MODEL) is not None


def test_retry_limit_and_unknown_usage():
    clock = Clock()
    settings = w.Settings(max_attempts=3, backoff_base_s=1)
    request = httpx2.Request("POST", "https://example.invalid")
    count = 0

    def fail(_):
        nonlocal count
        count += 1
        raise openai.APIConnectionError(request=request)

    run = w.Run("failure", settings, clock=clock, sleep=clock.sleep, jitter=lambda low, high: high)
    with pytest.raises(openai.APIConnectionError), run:
        run.call("model", "model", fail)
    assert count == 3
    assert run.summary()["retries"] == 2
    assert run.summary()["logical_model_calls"] == 1
    assert run.summary()["model_attempts_without_usage"] == 3
    assert run.summary()["backoff_s"] == 3
    assert run.summary()["reported_usage_cost_usd"] is None
    assert run.task_passed is False


def test_retry_after_stops_instead_of_ignoring_server_wait():
    clock = Clock()
    settings = w.Settings(workflow_budget_s=5)
    raw = httpx2.Response(
        429, headers={"Retry-After": "10"},
        request=httpx2.Request("POST", "https://example.invalid"),
    )
    exc = openai.RateLimitError("retry later", response=raw, body={"code": "rate_limit_exceeded"})
    run = w.Run("budget", settings, clock=clock, sleep=clock.sleep)
    with pytest.raises(w.BudgetExceeded), run:
        run.call("model", "model", lambda _: (_ for _ in ()).throw(exc))
    assert len(run.attempts) == 1
    assert clock.now == 0


def test_terminal_errors_not_retried():
    exc = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "Retrieve")
    run = w.Run("denied", w.Settings())
    with pytest.raises(ClientError), run:
        run.call("retrieve", "retrieval", lambda _: (_ for _ in ()).throw(exc))
    assert len(run.attempts) == 1
    assert not w.retryable(ValueError("invalid tool arguments"))


def test_success_after_transport_failure_retains_unknown_usage():
    request = httpx2.Request("POST", "https://example.invalid")
    count = 0

    def invoke(_):
        nonlocal count
        count += 1
        if count == 1:
            raise openai.APIConnectionError(request=request)
        return response()

    with w.Run("recover", w.Settings(backoff_base_s=0)) as run:
        run.call("model", "model", invoke)
    assert run.summary()["input_tokens"] == 100
    assert run.summary()["model_attempts_without_usage"] == 1
    assert run.summary()["usage_complete"] is False


def test_incomplete_output_usage_recorded_before_application_error():
    result = SimpleNamespace(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"}, error=None,
        model_dump=lambda **kwargs: response(status="incomplete"),
    )
    run = w.Run("incomplete", w.Settings())
    with pytest.raises(w.IncompleteResponse), run:
        returned = run.call("model", "model", lambda _: result)
        w.require_completed(returned)
    assert len(run.attempts) == 1
    assert run.summary()["output_tokens"] == 50
    assert run.task_passed is False
