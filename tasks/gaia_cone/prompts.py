"""User prompt for the Gaia cone task.

The system prompt and the tool-tier wording are shared with every task in
``harness.prompts``; only the task statement lives here, so prompt changes are
reviewable diffs and the three tool tiers share identical wording apart from
the tier-specific appendix.
"""

from __future__ import annotations

from pathlib import Path

from harness.prompts import RESULT_PATH, with_tier_appendix
from tasks.gaia_cone.variants import ConeVariant

DOCS_DIR = Path(__file__).resolve().parent / "docs"

USER_TEMPLATE = f"""Fetch {{n_rows}} rows of Gaia DR3 sources that lie within {{radius}} degrees of
RA = {{ra}}, Dec = {{dec}} (ICRS, decimal degrees). Any {{n_rows}} such sources are
acceptable.

Write the result to {RESULT_PATH} as a CSV file with a header row. It must
contain at least the columns `source_id`, `ra` and `dec`; extra columns are
fine. `source_id` must be the Gaia DR3 source identifier written as a plain
integer, and `ra`/`dec` must be ICRS decimal degrees.

When the file is written, call `submit`.
""".strip()


def render_user_prompt(v: ConeVariant, tool_tier: str) -> str:
    text = USER_TEMPLATE.format(n_rows=v.n_rows, radius=v.radius, ra=v.ra, dec=v.dec)
    return with_tier_appendix(text, tool_tier, DOCS_DIR)
