"""Custom agent tools.

``web_search_raw`` exists because Inspect's built-in ``web_search()`` (Tavily
provider) forces ``include_answer=True`` and returns the synthesized answer as
the tool text; source URLs are attached only as citations, which not every
model surface renders. In practice the model is told *what* a page says and
never *where* it is. For a task whose whole difficulty is "find the service",
that quietly removes the service from the map. This tool returns the raw hits:
title, URL, snippet. Same provider for every model under test.
"""

from __future__ import annotations

import os

import httpx
from inspect_ai.tool import Tool, ToolError, tool

TAVILY_URL = "https://api.tavily.com/search"


def format_results(results: list[dict], *, snippet_chars: int = 400) -> str:
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = " ".join((r.get("content") or "").split())[:snippet_chars]
        lines.append(f"{i}. {title}\n   {url}\n   {content}")
    return "\n\n".join(lines)


@tool
def web_search_raw(max_results: int = 8, search_depth: str = "basic", timeout: float = 30.0) -> Tool:
    """Web search returning raw results (title, URL, snippet), not a summary."""

    async def execute(query: str) -> str:
        """Search the web. Returns a numbered list of results with their URLs.

        Args:
            query: Search query.
        """
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ToolError("web search is not configured (TAVILY_API_KEY unset)")
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(TAVILY_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"})
            except httpx.HTTPError as exc:
                raise ToolError(f"search request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ToolError(f"search provider returned HTTP {resp.status_code}: {resp.text[:200]}")
        return format_results(resp.json().get("results", []))

    return execute
