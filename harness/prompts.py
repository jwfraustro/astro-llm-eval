"""Prompt text shared by every agent task.

The system prompt, the tool-tier wording and the ``docs``-tier appendix are
identical across tasks on purpose: the ablation is about the task and the
model, never about prompt phrasing. Each task supplies only its user-prompt
template. Changes here are reviewable diffs that apply to every task at once.
"""

from __future__ import annotations

from pathlib import Path

# Where every task asks the model to write its answer. The system prompt, the
# user prompts and the scorers all refer to this one constant.
RESULT_PATH = "/workspace/result.csv"

TOOL_TIERS = ("none", "search", "docs")

SYSTEM_TEMPLATE = """You are an expert astronomer working in a Linux sandbox.

You have tools to run bash commands and Python code{search_clause}. The sandbox
has internet access. A Python environment with astropy, astroquery, pyvo,
pandas and requests is already installed; you may pip install anything else
you need.

Write output files under /workspace. When the task is complete, call the
`submit` tool with a one-sentence summary of what you did and which service or
library you used. If you were successful in your task, start the summary with
"SUCCESS:". If you were not successful, start the summary with "FAIL:". In both
cases, provide a clear and concise summary of what happened. Do not call
`submit` before the required output file exists.
""".strip()

# The SUCCESS:/FAIL: prefix is not decoration: the scorer compares the model's
# self-report against the verifier's verdict (see harness.selfreport) to
# measure whether a model knows when it has failed.


def submit_description(result_path: str = RESULT_PATH) -> str:
    return f"Call this once {result_path} has been written, with a one-sentence summary."


# The system prompt must name every tool the tier actually provides. Saying
# "bash and Python" while also handing over a search tool is a contradiction
# that strong models resolve by reading the tool schemas and small open-weight
# models resolve by believing the prose, which turns the search tier into a
# test of prompt-vs-schema arbitration instead of search use.
SEARCH_CLAUSE = {
    "none": "",
    "search": ", and a web search tool",
    "docs": "",
}

TIER_APPENDIX = {
    "none": "",
    "search": "",
    "docs": "\n\nReference notes that may help:\n\n{docs}",
}


def check_tool_tier(tool_tier: str) -> str:
    if tool_tier not in TOOL_TIERS:
        raise ValueError(f"tool_tier must be one of {TOOL_TIERS}, got {tool_tier!r}")
    return tool_tier


def render_system_prompt(tool_tier: str) -> str:
    return SYSTEM_TEMPLATE.format(search_clause=SEARCH_CLAUSE[check_tool_tier(tool_tier)])


def load_docs(docs_dir: Path) -> str:
    """Concatenate ``docs_dir/*.md`` for the ``docs`` tier.

    Everything in there goes to the model verbatim. Keep those files short,
    correct, and vendor-neutral: the tier measures "can it use good docs", not
    "did we hand it the answer". Maintainer notes belong in the task module or
    the README, never in the docs files.
    """
    parts = [p.read_text().strip() for p in sorted(Path(docs_dir).glob("*.md"))]
    return "\n\n---\n\n".join(parts)


def with_tier_appendix(text: str, tool_tier: str, docs_dir: Path) -> str:
    """Append the tier-specific appendix (currently only the ``docs`` notes)."""
    appendix = TIER_APPENDIX[check_tool_tier(tool_tier)]
    if "{docs}" in appendix:
        appendix = appendix.format(docs=load_docs(docs_dir))
    return text + appendix
