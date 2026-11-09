"""Pure verification logic for the ADQL translation task.

The model's query and the gold query are both run against the same TAP
service; the verdict compares what came back, never the query text. Column
names are compared as sets (``SELECT a, b`` and ``SELECT b, a`` are the same
answer) and row counts are compared exactly (``TOP n`` is deterministic in
count even when the rows themselves are not ordered).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness import scoring as _shared
from harness.tap import GaiaVerifier, QueryRejected
from tasks.adql_translate.variants import TranslateVariant

# Mechanical failure reasons.
NO_OUTPUT_FILE = _shared.NO_OUTPUT_FILE
UNREADABLE_FILE = _shared.UNREADABLE_FILE
EMPTY_QUERY = "empty_query"
QUERY_REJECTED = "query_rejected"  # the service returned 4xx for the model's ADQL
COLUMNS_MISMATCH = "columns_mismatch"
ROW_COUNT_MISMATCH = "row_count_mismatch"
INFRA_ERROR = _shared.INFRA_ERROR

_FENCE_CHARS = "`"


@dataclass
class Verdict:
    passed: bool
    reason: str | None = None
    detail: str = ""
    n_rows: int = 0
    n_expected: int = 0
    infra_error: bool = False
    columns: list[str] = field(default_factory=list)
    expected_columns: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason or "pass",
            "detail": self.detail,
            "n_rows": self.n_rows,
            "n_expected": self.n_expected,
            "infra_error": self.infra_error,
            "columns": self.columns,
            "expected_columns": self.expected_columns,
        }


def parse_query_text(text: str) -> tuple[str | None, Verdict | None]:
    """Normalise the file: strip whitespace, a trailing semicolon, and code fences
    the prompt said not to write (we are testing ADQL, not instruction-following)."""
    lines = [ln for ln in text.strip().splitlines() if not ln.strip().startswith(_FENCE_CHARS * 3)]
    query = " ".join(" ".join(lines).split()).rstrip(";").strip()
    if not query:
        return None, Verdict(False, EMPTY_QUERY, "file contains no query")
    return query, None


def verify_query(query: str, variant: TranslateVariant, client: GaiaVerifier) -> Verdict:
    """Run the model's query and the gold query; compare columns and row count.

    Raises :class:`harness.tap.TAPError` if the service cannot be reached for
    the *gold* query. A rejected *model* query is a model failure.
    """
    exp_cols, exp_rows = client.sync_query_full(variant.gold_adql)  # may raise TAPError
    exp_names = [c.lower() for c in exp_cols]
    try:
        got_cols, got_rows = client.sync_query_full(query)
    except QueryRejected as exc:
        return Verdict(False, QUERY_REJECTED, str(exc), n_expected=len(exp_rows), expected_columns=exp_names)
    got_names = [c.lower() for c in got_cols]

    common = dict(n_rows=len(got_rows), n_expected=len(exp_rows), columns=got_names, expected_columns=exp_names)
    if set(got_names) != set(exp_names):
        return Verdict(False, COLUMNS_MISMATCH, f"got columns {got_names}, expected {exp_names}", **common)
    if len(got_rows) != len(exp_rows):
        return Verdict(False, ROW_COUNT_MISMATCH, f"{len(got_rows)} rows, expected {len(exp_rows)}", **common)
    return Verdict(True, None, f"{len(got_rows)} rows, {len(got_names)} columns match", **common)


def verify_query_text(text: str, variant: TranslateVariant, client: GaiaVerifier) -> Verdict:
    """Convenience: parse + verify. Still raises TAPError on service failure."""
    query, failure = parse_query_text(text)
    if failure is not None:
        return failure
    assert query is not None
    return verify_query(query, variant, client)
