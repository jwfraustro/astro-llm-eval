"""User prompt for the ObjObsSAP observability task.

The system prompt and the tool-tier wording are shared with every task in
``harness.prompts``; only the task statement lives here.
"""

from __future__ import annotations

from pathlib import Path

from harness.prompts import RESULT_PATH, with_tier_appendix
from tasks.objobssap_window.variants import WindowVariant

# For this task the docs/ notes are deliberately thin: a pointer to the
# service, not its parameters.
DOCS_DIR = Path(__file__).resolve().parent / "docs"

# "an example observatory in Baltimore, MD" is deliberate: the model has to
# connect that phrase to whatever facility a service it finds actually offers.
USER_TEMPLATE = f"""Using a Virtual Observatory (VO) protocol, determine the time intervals during
which the sky position RA = {{ra}}, Dec = {{dec}} (ICRS, decimal degrees) is
observable for at least {{min_obs}} seconds from an example observatory in
Baltimore, MD, within the period MJD {{t0}} to MJD {{t1}}.

Write the result to {RESULT_PATH} as a CSV file with a header row and the
columns `t_start_mjd` and `t_stop_mjd` (Modified Julian Date, decimal days),
one row per observability interval, in chronological order. If there are no
such intervals, write only the header row.

When the file is written, call `submit`.
""".strip()


def render_user_prompt(v: WindowVariant, tool_tier: str) -> str:
    text = USER_TEMPLATE.format(ra=v.ra, dec=v.dec, min_obs=v.min_obs, t0=v.t0, t1=v.t1)
    return with_tier_appendix(text, tool_tier, DOCS_DIR)
