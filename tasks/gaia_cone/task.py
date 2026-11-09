"""Inspect task: fetch N Gaia DR3 sources inside a random cone.

Run (one frontier model, 5 trials per variant, search tier)::

    inspect eval tasks/gaia_cone/task.py@gaia_cone \
        --model anthropic/claude-sonnet-5-5 -T tool_tier=search --epochs 5

Tool tiers (the ablation axis):

* ``none``   - bash + python only. Measures recall of astro infrastructure.
* ``search`` - adds ``web_search``. Measures ability to look things up.
* ``docs``   - no search, but curated reference notes are in the prompt.

Construction is shared with every other agent task; see ``harness.agent_task``.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from harness.agent_task import build_task
from tasks.gaia_cone.prompts import render_user_prompt
from tasks.gaia_cone.scorer import tap_verify
from tasks.gaia_cone.variants import DEFAULT_VARIANTS, load_variants


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
    return build_task(
        "gaia_cone",
        load_variants(Path(variants)),
        render_user_prompt,
        tap_verify(),
        tool_tier=tool_tier,
        variants_file=variants,
        limit=limit,
        message_limit=message_limit,
        tool_timeout=tool_timeout,
        time_limit=time_limit,
        search_tool=search_tool,
        search_provider=search_provider,
        tags=["gaia"],
    )
