"""User prompt for the ADQL translation task.

The system prompt and the tool-tier wording are shared with every task in
``harness.prompts``; only the task statement lives here.
"""

from __future__ import annotations

from pathlib import Path

from harness.prompts import with_tier_appendix
from harness.tap import MAST_GAIADR3_TAP
from tasks.adql_translate.variants import TranslateVariant

# A query, not a table, so this task has its own output path.
RESULT_PATH = "/workspace/result.adql"
DOCS_DIR = Path(__file__).resolve().parent / "docs"

# The service is named so the model can test its query; the task is the
# dialect, not service discovery.
USER_TEMPLATE = f"""The following query is written in a SQL dialect that the TAP service at
{MAST_GAIADR3_TAP} does not accept:

    {{source_sql}}

Translate it into valid ADQL that returns the same result from that service.

Write the ADQL query, and nothing else (no code fences, no commentary), to
{RESULT_PATH}.

When the file is written, call `submit`.
""".strip()


def render_user_prompt(v: TranslateVariant, tool_tier: str) -> str:
    text = USER_TEMPLATE.format(source_sql=v.source_sql)
    return with_tier_appendix(text, tool_tier, DOCS_DIR)
