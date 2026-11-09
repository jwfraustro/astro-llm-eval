"""Prompt text for the ObjObsSAP observability task."""

from __future__ import annotations

from pathlib import Path

from tasks.objobssap_window.variants import WindowVariant

RESULT_PATH = "/workspace/result.csv"
DOCS_DIR = Path(__file__).resolve().parent / "docs"

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

TIER_APPENDIX = {
    "none": "",
    "search": "",
    "docs": "\n\nReference notes that may help:\n\n{docs}",
}


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


def render_system_prompt(tool_tier: str) -> str:
    return SYSTEM_TEMPLATE.format(search_clause=SEARCH_CLAUSE[tool_tier])


def load_docs() -> str:
    # Handed to the model verbatim in the `docs` tier. For this task the notes
    # are deliberately thin: a pointer to the service, not its parameters.
    parts = [p.read_text().strip() for p in sorted(DOCS_DIR.glob("*.md"))]
    return "\n\n---\n\n".join(parts)


def render_user_prompt(v: WindowVariant, tool_tier: str) -> str:
    text = USER_TEMPLATE.format(ra=v.ra, dec=v.dec, min_obs=v.min_obs, t0=v.t0, t1=v.t1)
    appendix = TIER_APPENDIX[tool_tier]
    if "{docs}" in appendix:
        appendix = appendix.format(docs=load_docs())
    return text + appendix
