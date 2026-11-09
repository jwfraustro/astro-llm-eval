"""Shared construction of agent tasks: tools per tier, samples, Task wiring.

Every task in ``tasks/`` is the same shape: a seeded variant becomes a
``Sample``, a ``basic_agent`` with bash + python (+ search, by tier) works in
the docker sandbox, and one or more scorers read ``/workspace/result.csv``.
The ``@task`` function in each package keeps its own signature (Inspect reads
``-T`` options from it) and its own defaults; everything else lives here.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import dotenv
from inspect_ai import Task
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import Tool, bash, python, web_search

from harness.prompts import RESULT_PATH, check_tool_tier, render_system_prompt, submit_description
from harness.tools import web_search_raw

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "sandbox" / "compose.yaml"

dotenv.load_dotenv()


def build_tools(
    tool_tier: str,
    *,
    tool_timeout: int = 180,
    search_tool: str = "raw",
    search_provider: str | None = None,
) -> list[Tool]:
    """bash + python for every tier; a search tool for the ``search`` tier.

    ``search_tool``: ``raw`` (default) gives the model title/URL/snippet results
    via :func:`harness.tools.web_search_raw`. ``inspect`` uses Inspect's
    built-in ``web_search``, which returns a synthesized answer whose source
    URLs the model may never see; a Sonnet run with it was told what
    vo-labs.dev does but never where it was. Either way the provider is the
    same external one for every model, so the comparison is about the model,
    not whose built-in search is better.
    """
    check_tool_tier(tool_tier)
    tools: list[Tool] = [bash(timeout=tool_timeout), python(timeout=tool_timeout)]
    if tool_tier == "search":
        if search_tool == "raw":
            tools.append(web_search_raw())
        elif search_tool == "inspect":
            provider = search_provider or os.environ.get("ASTRO_EVAL_SEARCH_PROVIDER", "tavily")
            tools.append(web_search(provider))
        else:
            raise ValueError(f"search_tool must be 'raw' or 'inspect', got {search_tool!r}")
    return tools


def build_samples(variants: Sequence, render_user_prompt: Callable[[object, str], str], tool_tier: str) -> list[Sample]:
    """One sample per variant. The variant's fields become the sample metadata,
    which is how the scorer gets them back (``Variant.from_dict(state.metadata)``)."""
    return [
        Sample(
            id=v.id,
            input=render_user_prompt(v, tool_tier),
            target="",  # verification is live; there is no fixed answer
            metadata=asdict(v),
        )
        for v in variants
    ]


def build_task(
    name: str,
    variants: Sequence,
    render_user_prompt: Callable[[object, str], str],
    scorer: Scorer | list[Scorer],
    *,
    tool_tier: str = "search",
    variants_file: str | Path | None = None,
    limit: int | None = None,
    message_limit: int = 30,
    tool_timeout: int = 180,
    time_limit: int = 900,
    search_tool: str = "raw",
    search_provider: str | None = None,
    approval: list[ApprovalPolicy] | None = None,
    tags: Sequence[str] = (),
    result_path: str = RESULT_PATH,
) -> Task:
    """Assemble the Inspect ``Task`` every agent task shares.

    ``name`` is the bare task name (``gaia_cone``); the tier is appended so logs
    from different tiers never collide. ``tags`` are task-specific and are
    added to the common ``astro`` and ``tier:<tier>`` tags. ``result_path``
    is where the task tells the model to write its answer; it only feeds the
    ``submit`` tool's description here, the prompts and scorer get it directly.
    """
    check_tool_tier(tool_tier)
    all_variants = list(variants)
    if limit is not None:
        all_variants = all_variants[:limit]

    return Task(
        name=f"{name}_{tool_tier}",
        dataset=MemoryDataset(build_samples(all_variants, render_user_prompt, tool_tier), name=f"{name}[{tool_tier}]"),
        solver=basic_agent(
            init=system_message(render_system_prompt(tool_tier)),
            tools=build_tools(tool_tier, tool_timeout=tool_timeout, search_tool=search_tool, search_provider=search_provider),
            message_limit=message_limit,
            submit_description=submit_description(result_path),
        ),
        scorer=scorer,
        sandbox=("docker", str(COMPOSE_FILE)),
        time_limit=time_limit,
        approval=approval,
        metadata={
            "tool_tier": tool_tier,
            "variants_file": str(variants_file) if variants_file is not None else None,
            "search_tool": search_tool if tool_tier == "search" else None,
        },
        tags=["astro", *tags, f"tier:{tool_tier}"],
    )
