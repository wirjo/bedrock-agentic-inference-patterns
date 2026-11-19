import pytest

from utils.inference import build_request, UnsupportedInferenceOption
from utils.workshop import DEFAULT_MODEL, Rates, Usage
from utils import workshop
from test_notebooks import offline


def test_responses_cache_controls_request_explicit_mode_with_or_without_a_breakpoint():
    request = build_request(
        "responses", DEFAULT_MODEL, "question",
        cache_prefix="reference", cache_mode="on",
    )
    block = request["input"][0]["content"][0]
    assert request["input"][0]["role"] == "developer"
    assert block["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert request["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    without_checkpoint = build_request(
        "responses", DEFAULT_MODEL, "question", cache_prefix="reference", cache_mode="off",
    )
    assert "prompt_cache_breakpoint" not in without_checkpoint["input"][0]["content"][0]
    assert without_checkpoint["extra_body"]["prompt_cache_options"]["mode"] == "explicit"
    assert request["extra_body"]["prompt_cache_key"] == without_checkpoint["extra_body"]["prompt_cache_key"]


def test_converse_checkpoint_follows_stable_prefix():
    request = build_request(
        "converse", "us.anthropic.claude-sonnet-4-6", "question",
        instructions="instructions", cache_prefix="reference",
    )
    assert request["system"] == [
        {"text": "instructions"}, {"text": "reference"},
        {"cachePoint": {"type": "default", "ttl": "5m"}},
    ]
    assert request["messages"][-1]["content"] == [{"text": "question"}]


def test_unsupported_cache_controls_fail_before_inference():
    with pytest.raises(UnsupportedInferenceOption):
        build_request("responses", "openai.gpt-5.5", "question", cache_prefix="reference")
    with pytest.raises(UnsupportedInferenceOption):
        build_request("converse", "amazon.nova-pro-v1:0", "question", cache_prefix="reference")


def test_writes_reads_and_reasoning_are_not_double_charged():
    usage = Usage(
        input_tokens=1000, output_tokens=100, cached_input_tokens=200,
        cache_write_tokens=500, reasoning_tokens=80, cache_write_ttl="30m",
    )
    rates = Rates(
        DEFAULT_MODEL, 2, 12, cache_read_per_million=0.2,
        cache_write_per_million=2.5, cache_write_ttl="30m",
    )
    expected = (300 * 2 + 200 * 0.2 + 500 * 2.5 + 100 * 12) / 1e6
    assert rates.estimate(usage, DEFAULT_MODEL) == pytest.approx(expected)
    wrong_ttl = Rates(
        DEFAULT_MODEL, 2, 12, cache_read_per_million=0.2,
        cache_write_per_million=2.5, cache_write_ttl="5m",
    )
    assert wrong_ttl.estimate(usage, DEFAULT_MODEL) is None


def test_actual_converse_ttl_overrides_requested_default_and_mixed_ttls_stay_unpriced():
    data = {
        "_workshop": {"cache_write_ttl": "5m"},
        "usage": {
            "inputTokens": 10, "outputTokens": 5, "totalTokens": 15,
            "cacheWriteInputTokens": 500,
            "cacheDetails": [{"inputTokens": 500, "ttl": "1h"}],
        },
    }
    assert Usage.from_response(data).cache_write_ttl == "1h"
    data["usage"]["cacheDetails"].append({"inputTokens": 100, "ttl": "5m"})
    assert Usage.from_response(data).cache_write_ttl == "mixed"
    rates = Rates(DEFAULT_MODEL, 2, 12, cache_write_per_million=2.5, cache_write_ttl="5m")
    assert rates.estimate(Usage.from_response(data), DEFAULT_MODEL) is None


def test_reported_implicit_reads_are_retained_without_a_checkpoint(offline):
    """A requested 'off' mode must not overwrite service-reported cache usage."""
    offline.implicit_caching = True
    prefix = "Stable support reference. " * 300
    settings = workshop.Settings.from_env()
    with workshop.Run("implicit-read-observation", settings) as run:
        run.generate(
            "first", "Explain delivery status.",
            cache_prefix=prefix, cache_mode="off",
        )
        run.generate(
            "second", "Explain the next support action.",
            cache_prefix=prefix, cache_mode="off",
        )
    assert run.attempts[1].usage.cached_input_tokens > 0
    assert run.attempts[1].requested_cache_mode == "off"
    assert run.attempts[0].cache_prefix_sha256 == run.attempts[1].cache_prefix_sha256
    assert run.summary()["cached_input_tokens"] > 0
