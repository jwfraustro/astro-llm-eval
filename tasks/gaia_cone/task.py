"""Inspect task: fetch N Gaia DR3 sources inside a random cone.

Run (one frontier model, 5 trials per variant, search tier)::

    inspect eval tasks/gaia_cone/task.py@gaia_cone \
        --model anthropic/claude-sonnet-5-5 -T tool_tier=search --epochs 5

Tool tiers (the ablation axis):

* ``none``   - bash + python only. Measures recall of astro infrastructure.
* ``search`` - adds ``web_search``. Measures ability to look things up.
* ``docs``   - no search, but curated reference notes are in the prompt.
"""

from __future__ import annotations

import os
import dotenv
from dataclasses import asdict
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash, python, web_search

from harness.tools import web_search_raw
from tasks.gaia_cone.prompts import SYSTEM_PROMPT, render_user_prompt
from tasks.gaia_cone.scorer import tap_verify
from tasks.gaia_cone.variants import DEFAULT_VARIANTS, load_variants

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "sandbox" / "compose.yaml"

TOOL_TIERS = ("none", "search", "docs")

dotenv.load_dotenv()

@task
def gaia_cone(
    tool_tier: str = "search",
    variants: str = str(DEFAULT_VARIANTS),
    limit: int | None = None,
    message_limit: int = 30,
    tool_timeout: int = 180,
    time_limit: int = 900,
    search_tool: str = "raw",
    search_provider: str | None = None,
) -> Task:
    if tool_tier not in TOOL_TIERS:
        raise ValueError(f"tool_tier must be one of {TOOL_TIERS}, got {tool_tier!r}")

    tools = [bash(timeout=tool_timeout), python(timeout=tool_timeout)]
    if tool_tier == "search":
        # Same external provider for every model, so the comparison is about
        # the model, not whose built-in search is better. "raw" returns URLs;
        # see harness.tools for why that matters.
        if search_tool == "raw":
            tools.append(web_search_raw())
        elif search_tool == "inspect":
            provider = search_provider or os.environ.get("ASTRO_EVAL_SEARCH_PROVIDER", "tavily")
            tools.append(web_search(provider))
        else:
            raise ValueError(f"search_tool must be 'raw' or 'inspect', got {search_tool!r}")

    all_variants = load_variants(Path(variants))
    if limit is not None:
        all_variants = all_variants[:limit]

    samples = [
        Sample(
            id=v.id,
            input=render_user_prompt(v, tool_tier),
            target="",  # verification is live; there is no fixed answer
            metadata=asdict(v),
        )
        for v in all_variants
    ]

    return Task(
        name=f"gaia_cone_{tool_tier}",
        dataset=MemoryDataset(samples, name=f"gaia_cone[{tool_tier}]"),
        solver=basic_agent(
            init=system_message(SYSTEM_PROMPT),
            tools=tools,
            message_limit=message_limit,
            submit_description="Call this once /workspace/result.csv has been written, with a one-sentence summary.",
        ),
        scorer=tap_verify(),
        sandbox=("docker", str(COMPOSE_FILE)),
        time_limit=time_limit,
        metadata={"tool_tier": tool_tier, "variants_file": str(variants), "search_tool": search_tool if tool_tier == "search" else None},
        tags=["astro", "gaia", f"tier:{tool_tier}"],
    )
