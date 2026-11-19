"""Native Claude fixtures for executing the unchanged portable notebooks offline."""

import json

import boto3
from botocore.stub import Stubber
from botocore.validate import validate_parameters

from utils import workshop as w
from offline_backend import OfflineBackend
from test_converse import Events


class NativeOfflineBackend(OfflineBackend):
    implicit_caching = False

    def model_client(self, settings):
        raise AssertionError("Claude must never create an OpenAI SDK client")

    def answer(self, request):
        definition = request.get("outputConfig", {}).get("textFormat", {}).get("structure", {}).get("jsonSchema", {})
        name = definition.get("name")
        messages = request["messages"]
        facts = w.order_facts(w.load_order())
        output = None
        if name == "order_answer" or request.get("toolConfig"):
            agent_tools = any(
                t["toolSpec"]["name"] == "search_support_policy"
                for t in request.get("toolConfig", {}).get("tools", [])
            )
            tool_results = [
                b["toolResult"] for m in messages for b in m["content"] if "toolResult" in b
            ]
            if not tool_results:
                output = [
                    {"reasoningContent": {"reasoningText": {"text": "Use the order tool.", "signature": "native-signature"}}},
                    {"toolUse": {
                        "toolUseId": "native-call-1", "name": "get_order_status",
                        "input": {"order_id": "UNKNOWN" if self.invalid_tool_arguments else facts["order_id"]},
                    }},
                ]
            else:
                assistant = next(m for m in messages if m["role"] == "assistant")
                assert assistant["content"][0]["reasoningContent"]["reasoningText"]["signature"] == "native-signature"
                data = json.loads(tool_results[-1]["content"][0]["text"])
                if agent_tools and "sources" not in data:
                    output = [{"toolUse": {
                        "toolUseId": "native-policy-1", "name": "search_support_policy",
                        "input": {"query": "replacement eligibility for an in-transit order four days late"},
                    }}]
                elif agent_tools:
                    answer = "ORDER-1042 meets the replacement conditions in the delivery policy [S1]."
                else:
                    assert tool_results[-1]["toolUseId"] == "native-call-1"
                    answer = {k: data[k] for k in ("order_id", "status", "days_late", "next_step")}
                    answer["answer"] = "The order is delayed; contact the carrier."
                    if name is None:
                        answer = "ORDER-1042 is in transit and four days late; contact the carrier."
        elif name == "classification":
            answer = {"issue_type": "delayed_delivery", "order_id": facts["order_id"]}
        elif name == "assessment":
            answer = {
                "order_id": facts["order_id"], "carrier_status": facts["status"],
                "days_late": facts["days_late"], "recommended_action": facts["next_step"],
            }
        elif name == "draft":
            answer = {
                "order_id": facts["order_id"], "next_step": facts["next_step"],
                "answer": "The order is delayed; contact the carrier.",
            }
        elif name == "policy_answer":
            payload = json.loads(messages[-1]["content"][0]["text"])
            source = next(
                (s for s in payload["sources"] if "a replacement is allowed" in s["text"]), None
            )
            if source:
                quote = next(line for line in source["text"].splitlines() if "a replacement is allowed" in line)
                answer = {
                    "replacement_allowed": True,
                    "answer": "A replacement is allowed under the demo policy.",
                    "evidence": [{"source_id": source["source_id"], "quote": quote}],
                }
            else:
                answer = {"replacement_allowed": None, "answer": "Insufficient evidence.", "evidence": []}
        else:
            instructions = " ".join(block.get("text", "") for block in request.get("system", []))
            prompt = " ".join(block.get("text", "") for block in messages[-1]["content"])
            answer = self.plain_answer(instructions, prompt)
        if output is None:
            text = json.dumps(answer) if isinstance(answer, dict) else answer
            output = [{"text": text}]
        input_tokens = max(1, len(json.dumps(request, default=str)) // 4)
        system = request.get("system", [])
        prefix = "".join(block.get("text", "") for block in system)
        cached, written = self.cache_usage(
            prefix, bool(prefix), explicit=any("cachePoint" in b for b in system),
        )
        return {
            "output": {"message": {"role": "assistant", "content": output}},
            "stopReason": "tool_use" if any("toolUse" in b for b in output) else "end_turn",
            "usage": {
                "inputTokens": input_tokens - cached - written,
                "cacheReadInputTokens": cached, "cacheWriteInputTokens": written,
                "cacheDetails": [{"inputTokens": written, "ttl": "5m"}] if written else [],
                "outputTokens": 60, "totalTokens": input_tokens + 60,
            },
            "metrics": {"latencyMs": 1},
        }

    def native_client(self, settings, timeout):
        backend = self

        class Client:
            def converse(self, **request):
                backend.requests.append(request)
                client = boto3.client(
                    "bedrock-runtime", region_name=settings.region,
                    aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="dummy",
                )
                response = backend.answer(request)
                with Stubber(client) as stubber:
                    stubber.add_response("converse", response, request)
                    response = client.converse(**request)
                    stubber.assert_no_pending_responses()
                client.close()
                return response

            def converse_stream(self, **request):
                backend.requests.append(request)
                service = boto3.Session()._session.get_service_model("bedrock-runtime")
                validate_parameters(request, service.operation_model("ConverseStream").input_shape)
                response = backend.answer(request)
                events = [{"messageStart": {"role": "assistant"}}]
                for index, block in enumerate(response["output"]["message"]["content"]):
                    if "text" in block:
                        events.append({"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": block["text"]}}})
                    elif "reasoningContent" in block:
                        reason = block["reasoningContent"]["reasoningText"]
                        for key, value in reason.items():
                            events.append({"contentBlockDelta": {
                                "contentBlockIndex": index, "delta": {"reasoningContent": {key: value}},
                            }})
                    else:
                        tool = block["toolUse"]
                        events.extend([
                            {"contentBlockStart": {"contentBlockIndex": index, "start": {
                                "toolUse": {"name": tool["name"], "toolUseId": tool["toolUseId"]},
                            }}},
                            {"contentBlockDelta": {"contentBlockIndex": index, "delta": {
                                "toolUse": {"input": json.dumps(tool["input"])},
                            }}},
                        ])
                    events.append({"contentBlockStop": {"contentBlockIndex": index}})
                events.extend([
                    {"messageStop": {"stopReason": response["stopReason"]}},
                    {"metadata": {"usage": response["usage"], "metrics": response["metrics"]}},
                ])
                return {"stream": Events(events), "ResponseMetadata": {"RequestId": "native-offline"}}

            def close(self):
                pass
        return Client()
