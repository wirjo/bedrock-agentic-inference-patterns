import json
from dataclasses import asdict

import pytest

from utils import workshop as w
from utils.inference import build_request, resolve_api
from test_notebooks import offline


CLAUDE = "us.anthropic.claude-sonnet-4-6"
MESSAGES = [{"role": "user", "content": "Look up ORDER-1042."}]
TOOLS = [w.Tool("get_order_status", "Look up the order.", w.object_schema("lookup", {
    "order_id": {"type": "string"},
}))]
SCHEMA = w.object_schema("order_answer", {
    "order_id": {"type": "string"}, "status": {"type": "string"},
    "days_late": {"type": "integer"}, "next_step": {"type": "string"},
    "answer": {"type": "string"},
})


@pytest.mark.parametrize("model, api", [
    ("global.openai.gpt-5.6-terra", "responses"),
    ("openai.gpt-5.6-terra", "responses"),
    ("arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-5.6-terra", "responses"),
    (CLAUDE, "converse"),
    ("us.amazon.nova-pro-v1:0", "converse"),
])
def test_model_id_selects_api_automatically(model, api):
    assert w.Settings(model_id=model).resolved_api == api


def test_opaque_profile_needs_explicit_api_instead_of_guessing():
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/opaque"
    with pytest.raises(ValueError, match="explicitly"):
        resolve_api(arn)
    assert resolve_api(arn, "converse") == "converse"


@pytest.mark.parametrize("methods", [
    ("generate", "generate"), ("stream", "stream"),
    ("generate", "stream"), ("stream", "generate"),
])
def test_identical_tool_loop_with_both_methods_and_both_models(offline, methods):
    settings = w.Settings.from_env()
    history = list(MESSAGES)
    with w.Run("portable-tool-loop", settings) as run:
        first = getattr(run, methods[0])("lookup", messages=history, tools=TOOLS, response_schema=SCHEMA)
        assert isinstance(first, w.InferenceResponse)
        assert first.tool_calls[0].arguments == {"order_id": "ORDER-1042"}
        history.append(first.message)
        history.append(w.tool_result(first.tool_calls[0], w.order_facts(w.load_order())))
        final = getattr(run, methods[1])("answer", messages=history, tools=TOOLS, response_schema=SCHEMA)
        answer = w.read_json(final)
        run.check(answer["order_id"] == "ORDER-1042" and not final.tool_calls)
    assert run.task_passed is True
    assert run.summary()["logical_model_calls"] == 2
    assert len(offline.requests) == 2
    assert first.usage.input_tokens > 0
    assert "signature" not in repr(first.message)


@pytest.mark.parametrize("method", ["generate", "stream"])
def test_plain_tool_definitions_and_result_messages(offline, method):
    settings = w.Settings.from_env()
    tools = [asdict(tool) for tool in TOOLS]
    with w.Run("plain-tools", settings) as run:
        first = getattr(run, method)("lookup", MESSAGES, tools=tools)
        call = first.tool_calls[0]
        conversation = [
            *MESSAGES,
            first.message,
            {
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(w.order_facts(w.load_order())),
            },
        ]
        final = getattr(run, method)("answer", conversation, tools=tools)
    assert "ORDER-1042" in final.text
    assert not final.tool_calls
    assert len(offline.requests) == 2


@pytest.mark.parametrize("api,model_id", [
    ("responses", w.DEFAULT_MODEL), ("converse", CLAUDE),
])
def test_plain_prompt_matches_explicit_user_message(api, model_id):
    text = "Explain agent inference."
    assert build_request(api, model_id, text) == build_request(
        api, model_id, [{"role": "user", "content": text}]
    )


def test_claude_effort_is_mapped_to_adaptive_controls():
    request = build_request(
        "converse", CLAUDE, MESSAGES, reasoning_effort="medium", response_schema=SCHEMA,
    )
    assert request["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"},
    }
    assert json.loads(request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]) == SCHEMA
    assert "reasoning" not in request


def test_older_thinking_model_uses_an_explicit_budget():
    request = build_request(
        "converse", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", MESSAGES,
        reasoning_effort="high", max_output_tokens=4096,
    )
    assert request["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 3072
    with pytest.raises(w.UnsupportedInferenceOption, match="above"):
        build_request(
            "converse", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", MESSAGES,
            reasoning_effort="high", max_output_tokens=2048,
        )


def test_unsupported_reasoning_is_never_silently_dropped():
    with pytest.raises(w.UnsupportedInferenceOption, match="No reasoning_effort mapping"):
        build_request("converse", "us.amazon.nova-pro-v1:0", MESSAGES, reasoning_effort="medium")
    with pytest.raises(w.UnsupportedInferenceOption, match="automatic tool choice"):
        build_request(
            "converse", CLAUDE, MESSAGES, tools=TOOLS,
            tool_choice="required", reasoning_effort="high",
        )


def test_signed_continuation_cannot_be_replayed_into_another_model(offline):
    settings = w.Settings.from_env()
    with w.Run("one-model", settings) as run:
        result = run.generate("lookup", messages=MESSAGES, tools=TOOLS, response_schema=SCHEMA)
    with pytest.raises(ValueError, match="fresh conversation"):
        build_request("converse", "another-model", [*MESSAGES, result.message])


def test_generate_validates_json_after_recording_usage(offline, monkeypatch):
    settings = w.Settings.from_env()
    run = w.Run("invalid-schema", settings)
    # Reuse a real backend fixture result, then impose an incompatible local rubric.
    incompatible = {**SCHEMA, "required": ["a_field_the_model_did_not_return"]}
    original = w.normalize_response

    def require_extra_field(*args, **kwargs):
        return original(*args, **{**kwargs, "response_schema": incompatible})

    monkeypatch.setattr(w, "normalize_response", require_extra_field)
    import jsonschema
    with pytest.raises(jsonschema.ValidationError), run:
        run.generate(
            "answer", messages=[{"role": "user", "content": json.dumps({
                "order_id": "ORDER-1042", "recommended_action": "contact_carrier",
            })}],
            response_schema=w.object_schema("draft", {
                "order_id": {"type": "string"}, "next_step": {"type": "string"},
                "answer": {"type": "string"},
            }),
        )
    assert len(run.attempts) == 1
    assert run.summary()["output_tokens"] > 0
    assert run.task_passed is False
