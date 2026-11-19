"""Synthetic test responses only. This module is never used by the live notebooks."""

import json
from pathlib import Path

import boto3
import httpx2
from botocore.config import Config
from botocore.stub import ANY, Stubber
from openai import OpenAI
from openai.providers import bedrock

from utils import workshop as w


class OfflineBackend:
    implicit_caching = True

    def __init__(self):
        self.requests = []
        self.retrieval_requests = []
        self.clients = []
        self.empty_retrieval = False
        self.invalid_tool_arguments = False
        self.cache = set()

    def cache_usage(self, prefix, enabled, explicit=True):
        """Simulate prefix reuse, including implicit reads without a checkpoint."""
        if not enabled or len(prefix) // 4 < 1024 or (not explicit and not self.implicit_caching):
            return 0, 0
        tokens = len(prefix) // 4
        key = ("explicit" if explicit else "implicit", prefix)
        if key in self.cache:
            return tokens, 0
        self.cache.add(key)
        return 0, tokens if explicit else 0

    def plain_answer(self, instructions, prompt):
        """Return deterministic prose for the simplified, schema-free notebook prompts."""
        if instructions.startswith("Classify"):
            return "delayed_delivery"
        if instructions.startswith("Assess"):
            return "ORDER-1042 is in transit and four days late. Contact the carrier."
        if instructions.startswith("Write"):
            return "ORDER-1042 is four days late. Contact the carrier for an update."
        if instructions.startswith(("Use the supplied order and policy", "Answer from the supplied evidence")):
            return "ORDER-1042 is in transit and four days late. The policy permits replacement after three days [S1]."
        if instructions.startswith("Use only the supplied passages"):
            if "a replacement is allowed" in prompt:
                return (
                    "ORDER-1042 is eligible for a replacement [S1]. "
                    '"For an order whose carrier status is in_transit and which is '
                    "at least 3 calendar days past its promised delivery date, "
                    'a replacement is allowed."'
                )
            return "The supplied passages do not establish replacement eligibility."
        return "ORDER-1042 is in transit and four days late; contact the carrier."

    def model_client(self, settings):
        # Test the actual Bedrock provider and signing path with dummy credentials.
        client = OpenAI(
            provider=bedrock(
                endpoint="runtime", region=settings.region, api_key=None,
                access_key_id="AKIDEXAMPLE",
                secret_access_key="dummy-secret-for-local-tests-only",
            ),
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
            max_retries=0,
        )
        self.clients.append(client)
        return client

    def handle(self, request):
        assert request.url.host == "bedrock-runtime.us-east-1.amazonaws.com"
        assert request.url.path == "/openai/v1/responses"
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        body = json.loads(request.content)
        self.requests.append(body)
        assert body["model"] == w.DEFAULT_MODEL
        assert body["store"] is False
        output_format = body.get("text", {}).get("format", {})
        name = output_format.get("name")
        inputs = body["input"]
        payload = json.loads(inputs[-1]["content"]) if name in {
            "classification", "assessment", "draft", "policy_answer"
        } else None
        output = None
        if name == "order_answer" or body.get("tools"):
            results = [i for i in inputs if i.get("type") == "function_call_output"]
            agent_tools = any(t["name"] == "search_support_policy" for t in body.get("tools", []))
            if not results:
                arguments = {"order_id": "UNKNOWN" if self.invalid_tool_arguments else "ORDER-1042"}
                output = [
                    {"id": "rs_test", "type": "reasoning", "summary": [],
                     "encrypted_content": "synthetic-reasoning-for-local-test"},
                    {"id": "fc_test", "type": "function_call", "name": "get_order_status",
                     "call_id": "call_test", "arguments": json.dumps(arguments), "status": "completed"},
                ]
            else:
                assert any(i.get("type") == "reasoning" for i in inputs)
                assert any(i.get("type") == "function_call" for i in inputs)
                data = json.loads(results[-1]["output"])
                if agent_tools and "sources" not in data:
                    output = [{
                        "id": "fc_policy", "type": "function_call",
                        "name": "search_support_policy", "call_id": "call_policy",
                        "arguments": json.dumps({"query": "replacement eligibility for an in-transit order four days late"}),
                        "status": "completed",
                    }]
                elif agent_tools:
                    answer = "ORDER-1042 meets the replacement conditions in the delivery policy [S1]."
                else:
                    assert results[-1]["call_id"] == "call_test"
                    answer = {k: data[k] for k in ("order_id", "status", "days_late", "next_step")}
                    answer["answer"] = "ORDER-1042 is in transit and four days late; contact the carrier."
                    if name is None:
                        answer = answer["answer"]
        elif name == "classification":
            answer = {"issue_type": "delayed_delivery", "order_id": payload["order"]["order_id"]}
        elif name == "assessment":
            order = payload["order"]
            answer = {
                "order_id": order["order_id"], "carrier_status": order["status"],
                "days_late": order["days_late"], "recommended_action": order["next_step"],
            }
        elif name == "draft":
            answer = {
                "order_id": payload["order_id"], "next_step": payload["recommended_action"],
                "answer": "The order is delayed. Contact the carrier for an update.",
            }
        elif name == "policy_answer":
            source = next(
                (s for s in payload["sources"] if "a replacement is allowed" in s["text"]), None
            )
            if source:
                quote = next(
                    line for line in source["text"].splitlines() if "a replacement is allowed" in line
                )
                answer = {
                    "replacement_allowed": True,
                    "answer": "The order meets the demo policy's replacement conditions.",
                    "evidence": [{"source_id": source["source_id"], "quote": quote}],
                }
            else:
                answer = {"replacement_allowed": None, "answer": "Insufficient evidence.", "evidence": []}
        else:
            prompt = inputs[-1].get("content", "") if inputs else ""
            answer = self.plain_answer(body.get("instructions", ""), prompt)
        if output is None:
            text = json.dumps(answer) if isinstance(answer, dict) else answer
            output = [{
                "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }]
        # Deliberately synthetic token counts: never used as real performance evidence.
        input_tokens = max(1, len(json.dumps(body)) // 4)
        prefix = ""
        explicit_checkpoint = False
        for item in inputs:
            content = item.get("content")
            if item.get("role") == "developer" and isinstance(content, list):
                for block in content:
                    if block.get("type") == "input_text":
                        prefix += block["text"]
                        explicit_checkpoint |= bool(block.get("prompt_cache_breakpoint"))
        cached, written = self.cache_usage(prefix, bool(prefix), explicit_checkpoint)
        reasoning = {"high": 11, "medium": 7, "low": 3}.get(body.get("reasoning", {}).get("effort"), 0)
        response = {
            "id": f"resp_test_{len(self.requests)}", "object": "response",
            "created_at": 1, "status": "completed", "model": body["model"],
            "output": output, "parallel_tool_calls": False, "error": None,
            "incomplete_details": None, "text": body.get("text", {"format": {"type": "text"}}),
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
                "output_tokens": 60 + reasoning,
                "output_tokens_details": {"reasoning_tokens": reasoning},
                "total_tokens": input_tokens + 60 + reasoning,
            },
        }
        if body.get("stream"):
            events = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}]
            for index, item in enumerate(output):
                pending = dict(item)
                if item["type"] == "message":
                    pending.update(status="in_progress", content=[])
                elif item["type"] == "function_call":
                    pending.update(status="in_progress", arguments="")
                events.append({"type": "response.output_item.added", "output_index": index, "item": pending})
                if item["type"] == "message":
                    part = item["content"][0]
                    events.extend([
                        {"type": "response.content_part.added", "item_id": item["id"],
                         "output_index": index, "content_index": 0, "part": {**part, "text": ""}},
                        {"type": "response.output_text.delta", "item_id": item["id"],
                         "output_index": index, "content_index": 0, "delta": part["text"], "logprobs": []},
                        {"type": "response.output_text.done", "item_id": item["id"],
                         "output_index": index, "content_index": 0, "text": part["text"], "logprobs": []},
                        {"type": "response.content_part.done", "item_id": item["id"],
                         "output_index": index, "content_index": 0, "part": part},
                    ])
                elif item["type"] == "function_call":
                    events.extend([
                        {"type": "response.function_call_arguments.delta", "item_id": item["id"],
                         "output_index": index, "delta": item["arguments"]},
                        {"type": "response.function_call_arguments.done", "item_id": item["id"],
                         "output_index": index, "arguments": item["arguments"]},
                    ])
                events.append({"type": "response.output_item.done", "output_index": index, "item": item})
            events.append({"type": "response.completed", "response": response})
            payload = "".join(
                f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': i})}\n\n"
                for i, event in enumerate(events)
            )
            return httpx2.Response(
                200, content=payload,
                headers={"content-type": "text/event-stream", "x-request-id": "offline-stream-test"},
            )
        return httpx2.Response(200, json=response, headers={"x-request-id": "offline-test"})

    def retrieval_client(self, settings, timeout_s):
        backend = self

        class Retrieval:
            def close(self):
                pass

            def retrieve(self, **kwargs):
                backend.retrieval_requests.append(kwargs)
                count = kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"]
                paths = sorted((w.ROOT / "data/knowledge-base").glob("*.txt"))
                paths.sort(key=lambda p: (p.name != "delivery-policy.txt", p.name))
                results = [] if backend.empty_retrieval else [
                    {
                        "content": {"type": "TEXT", "text": path.read_text()},
                        "location": {"type": "S3", "s3Location": {"uri": f"s3://example/{path.name}"}},
                        "score": 0.95,
                    }
                    for path in paths[:count]
                ]
                client = boto3.client(
                    "bedrock-agent-runtime", region_name=settings.region,
                    aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="dummy",
                    config=Config(retries={"mode": "standard", "total_max_attempts": 1}),
                )
                with Stubber(client) as stubber:
                    stubber.add_response("retrieve", {"retrievalResults": results}, {
                        "knowledgeBaseId": settings.knowledge_base_id,
                        "retrievalQuery": {"text": ANY},
                        "retrievalConfiguration": {"vectorSearchConfiguration": ANY},
                    })
                    response = client.retrieve(**kwargs)
                    stubber.assert_no_pending_responses()
                client.close()
                return response
        return Retrieval()

    def close(self):
        for client in self.clients:
            client.close()
