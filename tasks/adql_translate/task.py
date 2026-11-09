"""Inspect task: translate a SQL-dialect query into ADQL that a TAP service accepts.

The simplest possible ADQL task: ``LIMIT n`` has to become ``TOP n``. It is a
placeholder for the templated natural-language-to-ADQL task set described in
``docs/task-ideas.md``, and exists mainly so the task registry has a third,
differently shaped member (output is a query, not a CSV).

Run::

    inspect eval tasks/adql_translate/task.py@adql_translate \
        --model anthropic/claude-sonnet-5-5 -T tool_tier=none --epochs 3

Construction is shared with every other agent task; see ``harness.agent_task``.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from harness.agent_task import build_task
from tasks.adql_translate.prompts import RESULT_PATH, render_user_prompt
from tasks.adql_translate.scorer import adql_verify
from tasks.adql_translate.variants import DEFAULT_VARIANTS, load_variants


@task
def adql_translate(
    tool_tier: str = "none",
    variants: str = str(DEFAULT_VARIANTS),
    limit: int | None = None,
    message_limit: int = 20,
    tool_timeout: int = 120,
    time_limit: int = 600,
    search_tool: str = "raw",
    search_provider: str | None = None,
) -> Task:
    return build_task(
        "adql_translate",
        load_variants(Path(variants)),
        render_user_prompt,
        adql_verify(),
        tool_tier=tool_tier,
        variants_file=variants,
        limit=limit,
        message_limit=message_limit,
        tool_timeout=tool_timeout,
        time_limit=time_limit,
        search_tool=search_tool,
        search_provider=search_provider,
        tags=["adql", "tap"],
        result_path=RESULT_PATH,
    )
