"""Prompt text for the Gaia cone task.

Kept in one place so prompt changes are reviewable diffs, and so the three
tool tiers share identical wording apart from the tier-specific appendix.
"""

from __future__ import annotations

from pathlib import Path

from .variants import ConeVariant

RESULT_PATH = "/workspace/result.csv"
DOCS_DIR = Path(__file__).resolve().parent / "docs"

SYSTEM_PROMPT = f"""You are an expert astronomer working in a Linux sandbox.

You have tools to run bash commands and Python code. The sandbox has internet
access. A Python environment with astropy, astroquery, pyvo, pandas and requests
is already installed; you may pip install anything else you need.

Write output files under /workspace. When the task is complete, call the
`submit` tool with a one-sentence summary of what you did and which service or
library you used. Do not call `submit` before the required output file exists.
""".strip()

USER_TEMPLATE = f"""Fetch {{n_rows}} rows of Gaia DR3 sources that lie within {{radius}} degrees of
RA = {{ra}}, Dec = {{dec}} (ICRS, decimal degrees). Any {{n_rows}} such sources are
acceptable.

Write the result to {RESULT_PATH} as a CSV file with a header row. It must
contain at least the columns `source_id`, `ra` and `dec`; extra columns are
fine. `source_id` must be the Gaia DR3 source identifier written as a plain
integer, and `ra`/`dec` must be ICRS decimal degrees.

When the file is written, call `submit`.
""".strip()

TIER_APPENDIX = {
    "none": "",
    "search": "",
    "docs": "\n\nReference notes that may help:\n\n{docs}",
}


def load_docs() -> str:
    # Everything in docs/*.md goes to the model verbatim in the `docs` tier.
    # Keep those files short, correct, and vendor-neutral: the tier measures
    # "can it use good docs", not "did we hand it the answer". Maintainer
    # notes belong here or in the README, never in the docs files.
    parts = [p.read_text().strip() for p in sorted(DOCS_DIR.glob("*.md"))]
    return "\n\n---\n\n".join(parts)


def render_user_prompt(v: ConeVariant, tool_tier: str) -> str:
    text = USER_TEMPLATE.format(n_rows=v.n_rows, radius=v.radius, ra=v.ra, dec=v.dec)
    appendix = TIER_APPENDIX[tool_tier]
    if "{docs}" in appendix:
        appendix = appendix.format(docs=load_docs())
    return text + appendix
