"""Reading the CSV the agent wrote, up to the point where the task takes over.

Header handling is the same for every CSV task: an empty or malformed file is
``unreadable_file``, header names are normalised so ``Source_ID`` or `` ra ``
still count, and required columns may be spelled a few ways. Row-level typing
is task-specific and stays in each task's ``verify.py``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from harness.verdict import UNREADABLE_FILE

MISSING_COLUMNS = "missing_columns"


class CsvError(Exception):
    """A pre-verification failure; ``reason`` is a failure-reason constant."""

    def __init__(self, reason: str, detail: str, n_rows: int = 0) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.n_rows = n_rows


@dataclass
class CsvTable:
    fieldnames: list[str]  # normalised: stripped and lower-cased
    rows: list[dict[str, str]]  # keyed by normalised field name

    def __len__(self) -> int:
        return len(self.rows)


def read_result_csv(text: str) -> CsvTable:
    """Parse the file. Raises :class:`CsvError` (``unreadable_file``) on failure."""
    try:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise CsvError(UNREADABLE_FILE, "empty file")
        reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]
        rows = list(reader)
    except csv.Error as exc:
        raise CsvError(UNREADABLE_FILE, f"csv parse error: {exc}") from exc
    return CsvTable(list(reader.fieldnames), rows)


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    """First of ``candidates`` present in ``fieldnames`` (both normalised)."""
    return next((c for c in candidates if c in fieldnames), None)


def require_columns(table: CsvTable, required: tuple[str, ...]) -> None:
    """Raise :class:`CsvError` (``missing_columns``) unless every name is present."""
    missing = [c for c in required if c not in table.fieldnames]
    if missing:
        raise CsvError(MISSING_COLUMNS, f"missing {missing}; got {table.fieldnames}", n_rows=len(table))


def cell(row: dict[str, str], column: str | None) -> str:
    """A stripped cell value; empty when the column or value is absent."""
    if column is None:
        return ""
    return (row.get(column) or "").strip()
