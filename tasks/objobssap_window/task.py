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

Construction is shared with every other agent task; see ``harness.agent_task``.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.approval import ApprovalPolicy

from harness.agent_task import build_task
from tasks.objobssap_window.approvers import rename_nudge
from tasks.objobssap_window.prompts import render_user_prompt
from tasks.objobssap_window.scorer import interval_verify, protocol_use
from tasks.objobssap_window.variants import DEFAULT_VARIANTS, load_variants

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
    message_limit: int = 60,
    tool_timeout: int = 180,
    time_limit: int = 1800,
    search_tool: str = "raw",
    search_provider: str | None = None,
) -> Task:
    return build_task(
        "objobssap_window",
        load_variants(Path(variants)),
        render_user_prompt,
        [interval_verify(), protocol_use()],
        tool_tier=tool_tier,
        variants_file=variants,
        limit=limit,
        message_limit=message_limit,
        tool_timeout=tool_timeout,
        time_limit=time_limit,
        search_tool=search_tool,
        search_provider=search_provider,
        approval=[ApprovalPolicy(rename_nudge(), "*")],
        tags=["objobssap", "vo"],
    )
