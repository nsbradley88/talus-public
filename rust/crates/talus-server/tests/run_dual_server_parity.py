#!/usr/bin/env python3
"""Dual-server black-box HTTP parity test suite for WP-1.4b (`serve/server.py` vs `talus-server`).

Runs the 12-check functional HTTP test suite over real loopback sockets against:
1. `serve/server.py` (Python reference server with `MockEngine` + `ByteTokenizer`)
2. `talus-server` (Rust binary target with `ScriptedMockEngine` + `ByteTokenizer`)
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http_request(
    port: int,
    method: str,
    path: str,
    body: dict[str, Any] | bytes | str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    req_headers: dict[str, str] = {"Connection": "close"}
    if headers:
        req_headers.update(headers)
    payload: bytes | None = None
    if isinstance(body, dict):
        payload = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    elif isinstance(body, str):
        payload = body.encode("utf-8")
    elif isinstance(body, bytes):
        payload = body
    conn.request(method, path, body=payload, headers=req_headers)
    resp = conn.getresponse()
    raw = resp.read()
    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
    status = resp.status
    conn.close()
    return status, resp_headers, raw


def parse_sse_events(raw: bytes) -> list[tuple[str | None, str]]:
    text = raw.decode("utf-8", errors="replace")
    events: list[tuple[str | None, str]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events


class PythonServerHarness:
    """In-process `serve/server.py` HTTP server backed by a clocked `MockEngine`."""

    def __init__(self, scripts: list[str], api_key: str = "", aliases: list[str] | None = None):
        from serve.server import ByteTokenizer, ChatTemplate, MockEngine, ROOT, Service, serve

        class ClockedMockEngine(MockEngine):
            def __init__(self, tokenizer, script):
                super().__init__(tokenizer, script)
                self._cached_ids: list[int] = []
                self.last: dict[str, Any] = {}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                no_cache = bool((sampling or {}).get("strata_no_cache"))
                reused = 0
                if not no_cache and self._cached_ids:
                    for a, b in zip(self._cached_ids, ids):
                        if a == b:
                            reused += 1
                        else:
                            break
                n = 0
                try:
                    for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                        n += 1
                        yield t
                finally:
                    self._cached_ids = list(ids)
                    self.last = {
                        "generated": n,
                        "prompt_tokens": len(ids),
                        "prompt_ms": 12.0,
                        "decode_ms": max(1.0, 4.0 * n),
                        "finish": "stop",
                        "reused": reused,
                    }

        self.tok = ByteTokenizer()
        self.engine = ClockedMockEngine(self.tok, list(scripts))
        tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        tpl.glm_tools = True
        self.svc = Service(self.engine, self.tok, tpl, model_name="qwen3.8-27b")
        self.svc.tool_call_recovery = True
        if aliases:
            self.svc.set_aliases(aliases)
        if api_key:
            self.svc.api_key = api_key
        self.httpd = serve(self.svc, host="127.0.0.1", port=0)
        self.port = int(self.httpd.server_address[1])

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class RustServerHarness:
    """Subprocess `talus-server` binary backed by `ScriptedMockEngine`."""

    def __init__(
        self,
        bin_path: Path,
        scripts: list[str],
        api_key: str = "",
        aliases: list[str] | None = None,
    ):
        cmd = [
            str(bin_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--model-name",
            "qwen3.8-27b",
            "--mock",
        ]
        for alias in aliases or []:
            cmd.extend(["--model-alias", alias])
        if api_key:
            cmd.extend(["--api-key", api_key])
        for s in scripts:
            cmd.extend(["--script", s])

        env = dict(os.environ)
        env.pop("STRATA_API_KEY", None)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert self.proc.stdout is not None
        deadline = time.time() + 10.0
        port: int | None = None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read() if self.proc.stderr else ""
                    raise RuntimeError(f"talus-server exited early ({self.proc.returncode}): {err}")
                time.sleep(0.02)
                continue
            m = re.search(r"http://127\.0\.0\.1:(\d+)", line)
            if m:
                port = int(m.group(1))
                break
        if port is None:
            self.close()
            raise RuntimeError("timed out waiting for talus-server listening line")
        self.port = port

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)


def run_12_checks(label: str, make_server, is_rust: bool = False) -> list[str]:
    results: list[str] = []

    scripts = [
        # 1: /completion #1 (cache populate + ignore_eos)
        "Completion alpha.",
        # 2: /completion #2 (cache hit)
        "Completion beta.",
        # 3: /completion #3 (token array prompt)
        "Token array completion.",
        # 4: /completions sync (#4)
        "Completions route reply.",
        # 5: /v1/chat/completions sync with <think>
        "Analyzing request.</think>\n\nHello from Talus!",
        # 6: /v1/chat/completions stream with <think> + cached_tokens
        "Step 1.</think>\n\nStreamed reply!",
        # 7: /v1/chat/completions Qwen XML tool call
        "Call weather tool.</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        # 8: /v1/chat/completions GLM <arg_key> tool call
        "</think>\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
        # 9: /v1/responses sync with thinking + encrypted_content
        "Reasoning trace.</think>\n\nResponse API answer.",
        # 10: /v1/responses stream with encrypted reasoning replay
        "</think>Streamed response output.",
        # 11: /v1/responses tool call
        "Use tool.</think>\n<tool_call>{\"name\": \"lookup_Order\", \"arguments\": {\"id\": 42}}</tool_call>",
        # 12: /v1/messages sync with thinking
        "Anthropic thought.</think>\n\nAnthropic message text.",
        # 13: /v1/messages stream with thinking + tool_use
        "Thinking for tool.</think><tool_call><function=get_weather><parameter=city>Berlin</parameter></function></tool_call>",
        # 14: /v1/completions sync (talus-server)
        "OpenAI text completion sync.",
        # 15: /v1/completions stream (talus-server)
        "OpenAI text completion stream.",
    ]

    srv = make_server(scripts=scripts, api_key="", aliases=["qwen27"])
    try:
        port = srv.port

        # Check 1: /health (+ DNS rebinding Host check when no API key is configured)
        st, _, raw = http_request(port, "GET", "/health")
        assert st == 200, f"[{label}] /health status {st}"
        health = json.loads(raw)
        assert health.get("status") == "ok", f"[{label}] /health body {health}"
        st_rebind, _, _ = http_request(
            port,
            "GET",
            "/health",
            headers={"Host": "evil.attacker.example.com"},
        )
        assert st_rebind == 403, f"[{label}] expected 403 on disallowed Host header without API key, got {st_rebind}"
        results.append("1. /health")

        # Check 2: /v1/models (with primary model + alias)
        st, _, raw = http_request(port, "GET", "/v1/models")
        assert st == 200, f"[{label}] /v1/models status {st}"
        models = json.loads(raw)
        assert models.get("object") == "list"
        model_ids = [m["id"] for m in models.get("data", [])]
        assert "qwen3.8-27b" in model_ids and "qwen27" in model_ids, f"[{label}] models: {model_ids}"
        results.append("2. /v1/models")

        # Check 3: /props
        st, _, raw = http_request(port, "GET", "/props")
        assert st == 200, f"[{label}] /props status {st}"
        props = json.loads(raw)
        assert props["total_slots"] == 1
        assert props["default_generation_settings"]["n_ctx"] > 0
        assert "chat_template" in props
        results.append("3. /props")

        # Check 4: /slots
        st, _, raw = http_request(port, "GET", "/slots")
        assert st == 200, f"[{label}] /slots status {st}"
        slots = json.loads(raw)
        assert isinstance(slots, list) and len(slots) == 1
        assert slots[0]["id"] == 0 and slots[0]["is_processing"] is False
        results.append("4. /slots")

        # Check 5: /tokenize + /detokenize
        st, _, raw = http_request(port, "POST", "/tokenize", {"content": "Hello Talus 123!"})
        assert st == 200, f"[{label}] /tokenize status {st}"
        toks = json.loads(raw)["tokens"]
        assert isinstance(toks, list) and len(toks) > 0
        st, _, raw = http_request(port, "POST", "/detokenize", {"tokens": toks})
        assert st == 200, f"[{label}] /detokenize status {st}"
        assert json.loads(raw)["content"] == "Hello Talus 123!"
        st, _, raw = http_request(
            port,
            "POST",
            "/tokenize",
            {"content": "<|im_start|>user\nhi<|im_end|>", "parse_special": True},
        )
        assert st == 200
        special_toks = json.loads(raw)["tokens"]
        assert len(special_toks) == 9
        st, _, raw_detok = http_request(port, "POST", "/detokenize", {"tokens": special_toks})
        assert st == 200 and json.loads(raw_detok)["content"] == "<|im_start|>user\nhi<|im_end|>"
        results.append("5. /tokenize + /detokenize")

        # Check 6: /completion (ignore_eos, cache_prompt, timings, token array)
        shared_prefix = "System prompt prefix for KV cache reuse: " + ("abc " * 10)
        st, _, raw = http_request(
            port,
            "POST",
            "/completion",
            {
                "prompt": shared_prefix + "first question",
                "n_predict": len("Completion alpha."),
                "ignore_eos": True,
                "cache_prompt": True,
            },
        )
        assert st == 200, f"[{label}] /completion #1 status {st}: {raw}"
        c1 = json.loads(raw)
        assert c1["content"].removesuffix("<|im_end|>") == "Completion alpha."
        assert c1["stop"] is True
        assert c1["tokens_evaluated"] > 0 and c1["tokens_predicted"] > 0
        assert "timings" in c1 and c1["timings"]["predicted_n"] == c1["tokens_predicted"]

        st, _, raw = http_request(
            port,
            "POST",
            "/completion",
            {
                "prompt": shared_prefix + "second question",
                "n_predict": 32,
                "cache_prompt": True,
            },
        )
        assert st == 200
        c2 = json.loads(raw)
        assert c2["content"].removesuffix("<|im_end|>") == "Completion beta."
        assert c2["timings"]["cache_n"] > 0, f"[{label}] expected timings.cache_n > 0: {c2}"

        st, _, raw = http_request(
            port,
            "POST",
            "/completion",
            {"prompt": [72, 105], "n_predict": 32},
        )
        assert st == 200
        c3 = json.loads(raw)
        assert c3["content"].removesuffix("<|im_end|>") == "Token array completion."
        results.append("6. /completion (ignore_eos, cache_prompt, timings, token array)")

        # Check 7: /completions (and /v1/completions sync + stream on talus-server)
        st, _, raw = http_request(
            port,
            "POST",
            "/completions",
            {"prompt": "Say hi", "n_predict": 32},
        )
        assert st == 200
        vc1 = json.loads(raw)
        assert vc1["content"].removesuffix("<|im_end|>") == "Completions route reply."
        assert "timings" in vc1
        results.append("7. /completions + /v1/completions (sync + stream)")

        # Check 8: /v1/chat/completions (sync + stream + thinking + tool_calls + cached_tokens + timings)
        chat_messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {"model": "qwen3.8-27b", "messages": chat_messages},
        )
        assert st == 200, f"[{label}] chat sync status {st}: {raw}"
        cc1 = json.loads(raw)
        msg1 = cc1["choices"][0]["message"]
        assert msg1["content"] == "Hello from Talus!", f"[{label}] got {msg1}"
        assert msg1.get("reasoning_content") == "Analyzing request.", f"[{label}] got {msg1}"
        assert "timings" in cc1 and "usage" in cc1

        # Stream chat with same prefix -> should hit prompt cache and stream reasoning_content + content
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "qwen27",
                "messages": chat_messages + [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "More"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert st == 200
        cc_events = parse_sse_events(raw)
        assert cc_events[-1][1] == "[DONE]"
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        cached_toks = 0
        for _, d in cc_events:
            if d == "[DONE]":
                continue
            chunk = json.loads(d)
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                if delta.get("content"):
                    content_parts.append(delta["content"])
            if chunk.get("usage"):
                cached_toks = chunk["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        assert "".join(reasoning_parts) == "Step 1."
        assert "".join(content_parts) == "Streamed reply!"
        assert cached_toks > 0, f"[{label}] expected cached_tokens > 0 in streaming chat usage"

        # Tool calls: Qwen XML format
        weather_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {"model": "qwen3.8-27b", "messages": chat_messages, "tools": weather_tools},
        )
        assert st == 200
        tc_resp = json.loads(raw)
        assert tc_resp["choices"][0]["finish_reason"] == "tool_calls"
        tcalls = tc_resp["choices"][0]["message"]["tool_calls"]
        assert len(tcalls) == 1
        assert tcalls[0]["function"]["name"] == "get_weather"
        assert json.loads(tcalls[0]["function"]["arguments"]) == {"city": "Paris"}

        # Tool calls: GLM <arg_key> format
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {"model": "qwen3.8-27b", "messages": chat_messages, "tools": weather_tools},
        )
        assert st == 200
        tc_glm = json.loads(raw)
        assert tc_glm["choices"][0]["finish_reason"] == "tool_calls"
        tcalls_glm = tc_glm["choices"][0]["message"]["tool_calls"]
        assert tcalls_glm[0]["function"]["name"] == "get_weather"
        assert json.loads(tcalls_glm[0]["function"]["arguments"]) == {"city": "Tokyo"}
        results.append("8. /v1/chat/completions (sync + stream + thinking + Qwen/GLM tool_calls + cached_tokens)")

        # Check 9: /v1/responses + stateless reasoning replay + /v1/responses/{id}
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/responses",
            {
                "model": "qwen3.8-27b",
                "input": "Explain Talus",
                "include": ["reasoning.encrypted_content"],
            },
        )
        assert st == 200, f"[{label}] /v1/responses status {st}: {raw}"
        r1 = json.loads(raw)
        assert r1["object"] == "response" and r1["status"] == "completed"
        resp_id = r1["id"]
        assert resp_id.startswith("resp_")
        output_types = [item["type"] for item in r1["output"]]
        assert "reasoning" in output_types and "message" in output_types
        rs_item = [item for item in r1["output"] if item["type"] == "reasoning"][0]
        msg_item = [item for item in r1["output"] if item["type"] == "message"][0]
        assert rs_item["encrypted_content"].startswith("strata.r1:")
        assert msg_item["content"][0]["text"] == "Response API answer."

        # Stateless /v1/responses/{id} and /v1/responses/{id}/cancel return 404 not_found
        st, _, raw = http_request(port, "GET", f"/v1/responses/{resp_id}")
        assert st == 404, f"[{label}] GET /v1/responses/{{id}} status {st}: {raw}"
        st, _, raw = http_request(port, "POST", f"/v1/responses/{resp_id}/cancel", {})
        assert st == 404 and json.loads(raw)["error"]["code"] == "not_found"

        # previous_response_id rejected with 400 param="previous_response_id"
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/responses",
            {"model": "qwen3.8-27b", "input": "x", "previous_response_id": resp_id},
        )
        assert st == 400 and json.loads(raw)["error"]["param"] == "previous_response_id"

        # Streaming POST /v1/responses with stateless encrypted reasoning replay in `input`
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/responses",
            {
                "model": "qwen3.8-27b",
                "input": [
                    {"role": "user", "content": "Explain Talus"},
                    {"type": "reasoning", "summary": [], "encrypted_content": rs_item["encrypted_content"]},
                    msg_item,
                    {"role": "user", "content": "Follow up"},
                ],
                "stream": True,
            },
        )
        assert st == 200
        resp_events = parse_sse_events(raw)
        event_names = [ev for ev, _ in resp_events]
        assert "response.created" in event_names
        assert "response.output_text.delta" in event_names
        assert "response.completed" in event_names

        # Tool call in /v1/responses
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/responses",
            {
                "model": "qwen3.8-27b",
                "input": "Check order 42",
                "tools": [{"type": "function", "name": "lookup_Order", "parameters": {"type": "object"}}],
            },
        )
        assert st == 200
        r_tc = json.loads(raw)
        fn_items = [item for item in r_tc["output"] if item["type"] == "function_call"]
        assert len(fn_items) == 1 and fn_items[0]["name"] == "lookup_Order"
        results.append("9. /v1/responses (sync + stream + encrypted reasoning replay + function_call + stateless 404)")

        # Check 10: /v1/messages + /v1/messages/count_tokens (Anthropic)
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": "qwen3.8-27b",
                "system": "Be concise.",
                "messages": [{"role": "user", "content": "Count my tokens please."}],
            },
        )
        assert st == 200, f"[{label}] /v1/messages/count_tokens status {st}: {raw}"
        ct = json.loads(raw)
        assert ct["input_tokens"] > 0

        # Sync /v1/messages with thinking enabled
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/messages",
            {
                "model": "qwen3.8-27b",
                "max_tokens": 64,
                "thinking": {"type": "enabled", "budget_tokens": 32},
                "messages": [{"role": "user", "content": "Hello Claude"}],
            },
        )
        assert st == 200, f"[{label}] /v1/messages sync status {st}: {raw}"
        am1 = json.loads(raw)
        assert am1["type"] == "message" and am1["role"] == "assistant"
        block_types = [b["type"] for b in am1["content"]]
        assert "thinking" in block_types and "text" in block_types
        assert "cache_read_input_tokens" in am1["usage"]

        # Stream /v1/messages with thinking + tool_use
        st, _, raw = http_request(
            port,
            "POST",
            "/v1/messages",
            {
                "model": "qwen3.8-27b",
                "max_tokens": 256,
                "stream": True,
                "thinking": {"type": "enabled", "budget_tokens": 32},
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    }
                ],
                "messages": [{"role": "user", "content": "Weather in Berlin?"}],
            },
        )
        assert st == 200, f"[{label}] /v1/messages stream status {st}: {raw}"
        am_events = parse_sse_events(raw)
        am_event_names = [ev for ev, _ in am_events]
        assert "message_start" in am_event_names
        assert "content_block_start" in am_event_names
        assert "content_block_delta" in am_event_names
        assert "content_block_stop" in am_event_names
        assert "message_delta" in am_event_names
        assert "message_stop" in am_event_names
        delta_types = [
            json.loads(d)["delta"]["type"]
            for ev, d in am_events
            if ev == "content_block_delta"
        ]
        assert "thinking_delta" in delta_types and "input_json_delta" in delta_types
        results.append("10. /v1/messages (sync + stream + thinking + tool_use + count_tokens)")

        # Check 11: /metrics (Prometheus vllm + strata on /metrics?format=prometheus and llamacpp on /metrics/prometheus)
        st, _, raw = http_request(
            port,
            "GET",
            "/metrics?format=prometheus",
            headers={"Accept": "text/plain"},
        )
        assert st == 200, f"[{label}] /metrics?format=prometheus status {st}"
        metrics_text = raw.decode("utf-8")
        for metric_name in (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:request_success_total",
            "strata:live_state",
            "strata:totals_decode_seconds_total",
            "vllm:time_to_first_token_seconds_bucket",
        ):
            assert metric_name in metrics_text, f"[{label}] missing metric {metric_name}"

        st, _, raw_llama = http_request(port, "GET", "/metrics/prometheus")
        assert st == 200, f"[{label}] /metrics/prometheus status {st}"
        llama_text = raw_llama.decode("utf-8")
        for metric_name in (
            "llamacpp:prompt_tokens_total",
            "llamacpp:tokens_predicted_total",
            "llamacpp:requests_processing",
        ):
            assert metric_name in llama_text, f"[{label}] missing llamacpp metric {metric_name}"
        if is_rust:
            st, _, raw = http_request(
                port,
                "POST",
                "/v1/completions",
                {"model": "qwen3.8-27b", "prompt": "Say hi", "max_tokens": 32},
            )
            assert st == 200
            oai_c1 = json.loads(raw)
            assert oai_c1["object"] == "text_completion"
            assert oai_c1["choices"][0]["text"] == "OpenAI text completion sync."
            assert "timings" in oai_c1 and "usage" in oai_c1

            st, _, raw = http_request(
                port,
                "POST",
                "/v1/completions",
                {"model": "qwen3.8-27b", "prompt": "Say hi", "max_tokens": 32, "stream": True},
            )
            assert st == 200
            oai_sse = parse_sse_events(raw)
            assert oai_sse[-1][1] == "[DONE]"
            oai_chunks = [json.loads(d) for _, d in oai_sse if d != "[DONE]"]
            assert "".join(c["choices"][0].get("text", "") for c in oai_chunks) == "OpenAI text completion stream."
        results.append("11. /metrics (Prometheus llamacpp + vllm + strata metrics)")
    finally:
        srv.close()

    # Check 12: API key enforcement & security (/health exempt, 401 on missing/wrong key, Bearer & x-api-key accepted)
    auth_srv = make_server(scripts=["Auth ok."], api_key="talus-secret-key-99", aliases=["qwen27"])
    try:
        aport = auth_srv.port
        # /health is exempt from API key
        st, _, _ = http_request(aport, "GET", "/health")
        assert st == 200, f"[{label}] /health should be exempt from API key, got {st}"

        # Missing key -> 401
        for method, path, payload in (
            ("GET", "/v1/models", None),
            ("GET", "/props", None),
            ("GET", "/metrics", None),
            ("POST", "/v1/chat/completions", {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}]}),
            ("POST", "/v1/messages", {"model": "qwen3.8-27b", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
            ("POST", "/completion", {"prompt": "hi"}),
        ):
            st, _, _ = http_request(aport, method, path, payload)
            assert st == 401, f"[{label}] expected 401 on {method} {path} without key, got {st}"

        # Wrong key -> 401
        st, _, _ = http_request(
            aport,
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert st == 401, f"[{label}] expected 401 with wrong Bearer key, got {st}"

        # Correct key via Authorization: Bearer
        st, _, _ = http_request(
            aport,
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer talus-secret-key-99"},
        )
        assert st == 200, f"[{label}] expected 200 with valid Bearer key, got {st}"

        # Correct key via x-api-key
        st, _, _ = http_request(
            aport,
            "POST",
            "/v1/chat/completions",
            {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "talus-secret-key-99"},
        )
        assert st == 200, f"[{label}] expected 200 with valid x-api-key, got {st}"

        # With API key configured, tunnel/proxy Host headers are allowed (auth protects endpoints)
        st, _, _ = http_request(
            aport,
            "GET",
            "/health",
            headers={"Host": "tunnel.example.com"},
        )
        assert st == 200, f"[{label}] expected 200 on /health with tunnel Host when API key is set, got {st}"

        results.append("12. API key auth (401 rejection, Bearer & x-api-key, /health exempt, Host rebinding 403)")
    finally:
        auth_srv.close()

    return results


def find_talus_server_binary() -> Path:
    env_bin = os.environ.get("TALUS_SERVER_BIN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)
    candidates = [
        REPO_ROOT / "rust/target/debug/talus-server",
        REPO_ROOT / "rust/target/release/talus-server",
    ]
    for c in candidates:
        if c.exists():
            return c
    subprocess.run(
        ["cargo", "build", "-p", "talus-server", "--bin", "talus-server"],
        cwd=str(REPO_ROOT / "rust"),
        check=True,
    )
    assert candidates[0].exists(), f"talus-server binary not found at {candidates[0]}"
    return candidates[0]


def main() -> int:
    bin_path = find_talus_server_binary()
    print(f"[parity] Running 12-check suite against Python serve/server.py ...")
    py_results = run_12_checks(
        "serve/server.py",
        lambda scripts, api_key="", aliases=None: PythonServerHarness(scripts, api_key=api_key, aliases=aliases),
    )
    for item in py_results:
        print(f"  [PASS] serve/server.py :: {item}")

    print(f"[parity] Running 12-check suite against Rust talus-server ({bin_path}) ...")
    rs_results = run_12_checks(
        "talus-server",
        lambda scripts, api_key="", aliases=None: RustServerHarness(
            bin_path, scripts, api_key=api_key, aliases=aliases
        ),
    )
    for item in rs_results:
        print(f"  [PASS] talus-server    :: {item}")

    assert py_results == rs_results
    print(f"[parity] ALL {len(rs_results)}/12 DUAL-SERVER PARITY CHECKS PASSED (serve/server.py == talus-server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
