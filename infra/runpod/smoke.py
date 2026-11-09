"""Pre-flight checks against a running vLLM server, before spending a real run.

    VLLM_BASE_URL=... VLLM_API_KEY=... python infra/runpod/smoke.py Qwen/Qwen3-8B

1. Tool call: offers a `web_search_raw` tool with the same schema the tasks
   use and asks something only a search can answer. A `tool_calls` entry in
   the response means the chat template rendered the tool and the parser
   extracted the call. Prose about searching, or a `<tool_call>` tag left in
   `content`, means one of the two is wrong.
2. Decode speed: one streamed request, timed from the first token to the last,
   so prefill and network round-trips through the Runpod proxy are excluded.
   This is the per-stream number; aggregate throughput under concurrency is in
   the server log ("Avg generation throughput", "Running: N reqs").
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search_raw",
        "description": "Search the web. Returns a numbered list of results with their URLs.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
        },
    },
}


def main(model: str) -> int:
    base = os.environ["VLLM_BASE_URL"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'inspectai')}"}
    client = httpx.Client(base_url=base, headers=headers, timeout=300)

    r = client.post("/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Find the TAP endpoint URL for the MAST Gaia DR3 service. Use your tools."}],
        "tools": [SEARCH_TOOL],
        "tool_choice": "auto",
        "max_tokens": 2048,
    })
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print("tool_calls:", json.dumps([c["function"] for c in calls]) if calls else "NONE")
    if not calls:
        print("content:", (msg.get("content") or "")[:800])
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    print("reasoning field:", "present" if reasoning else "absent (thinking, if any, is in content)")

    n, first, last = 0, None, None
    with client.stream("POST", "/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Write a 600-word history of the Gaia mission."}],
        "max_tokens": 1024,
        "min_tokens": 1024,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }) as s:
        for line in s.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                n = chunk["usage"]["completion_tokens"]
            elif chunk.get("choices"):
                now = time.perf_counter()
                first = first or now
                last = now
    if first and last and n > 1:
        print(f"decode: {n} tokens, {(n - 1) / (last - first):.1f} tok/s single-stream")
    return 0 if calls else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"))
