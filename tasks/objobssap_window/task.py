"""Inspect task: when is a position observable from Baltimore, via a VO protocol?

Intentionally hard. The right answer requires knowing (or discovering) that
IVOA ObjObsSAP exists, finding one of its two implementations (not registered,
so this means web search), reading its docs page to connect "an example
observatory in Baltimore, MD" to the service's default facility, and driving a
Working-Draft protocol correctly (MJD range syntax, MIN_OBS, VOTable output).

Run::

    inspect eval tasks/objobssap_window/task.py@objobssap_window \
        --model anthropic/claude-sonnet-5-5 -T tool_tier=search --epochs 3

Scores: ``interval_verify`` (outcome) and ``protocol_use`` (process). Read them
together: outcome-pass / protocol-fail means it computed rise and set locally.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash, python, web_search

from tasks.objobssap_window.prompts import SYSTEM_PROMPT, render_user_prompt
from tasks.objobssap_window.scorer import interval_verify, protocol_use
from tasks.objobssap_window.variants import DEFAULT_VARIANTS, load_variants

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "sandbox" / "compose.yaml"

TOOL_TIERS = ("none", "search", "docs")

dotenv.load_dotenv()

# Judge labels specific to this task; passed to harness.judges.failure_taxonomy(extra_labels=...).
JUDGE_LABELS = {
    "protocol_unknown": "Never identified ObjObsSAP (or any observability protocol) as the relevant VO standard.",
    "found_spec_no_service": "Identified ObjObsSAP but could not locate a running implementation.",
    "wrong_facility_empty_result": "Queried the service with an unsupported FACILITY, got an empty table, and treated that as the answer.",
    "misread_time_format": "Passed TIME in the wrong scale or syntax (ISO or JD instead of MJD, wrong range form).",
    "computed_locally": "Computed observability with astropy/astroplan or similar instead of using a VO service.",
}


@task
def objobssap_window(
    tool_tier: str = "search",
    variants: str = str(DEFAULT_VARIANTS),
    limit: int | None = None,
    message_limit: int = 40,
    tool_timeout: int = 180,
    time_limit: int = 1200,
    search_provider: str | None = None,
) -> Task:
    if tool_tier not in TOOL_TIERS:
        raise ValueError(f"tool_tier must be one of {TOOL_TIERS}, got {tool_tier!r}")

    tools = [bash(timeout=tool_timeout), python(timeout=tool_timeout)]
    if tool_tier == "search":
        provider = search_provider or os.environ.get("ASTRO_EVAL_SEARCH_PROVIDER", "tavily")
        tools.append(web_search(provider))

    all_variants = load_variants(Path(variants))
    if limit is not None:
        all_variants = all_variants[:limit]

    samples = [
        Sample(id=v.id, input=render_user_prompt(v, tool_tier), target="", metadata=asdict(v))
        for v in all_variants
    ]

    return Task(
        name=f"objobssap_window_{tool_tier}",
        dataset=MemoryDataset(samples, name=f"objobssap_window[{tool_tier}]"),
        solver=basic_agent(
            init=system_message(SYSTEM_PROMPT),
            tools=tools,
            message_limit=message_limit,
            submit_description="Call this once /workspace/result.csv has been written, with a one-sentence summary.",
        ),
        scorer=[interval_verify(), protocol_use()],
        sandbox=("docker", str(COMPOSE_FILE)),
        time_limit=time_limit,
        metadata={"tool_tier": tool_tier, "variants_file": str(variants)},
        tags=["astro", "objobssap", "vo", f"tier:{tool_tier}"],
    )
