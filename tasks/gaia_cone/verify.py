"""Pure verification logic for the Gaia cone task.

Everything here is synchronous and side-effect free apart from the TAP lookup,
so it can be unit-tested with a fake verifier and re-run offline from a stashed
CSV (see ``scripts/rescore.py``) if MAST was unreachable during the eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness import verdict as _base
from harness import csvio as _csvio
from harness.csvio import CsvError, cell, read_result_csv, require_columns
from harness.geometry import angular_separation_deg
from harness.tap import GaiaVerifier
from tasks.gaia_cone.variants import ConeVariant

# Mechanical failure reasons. These are *not* the LLM-judged taxonomy in
# harness/judges.py; they are what the verifier can establish on its own.
NO_OUTPUT_FILE = _base.NO_OUTPUT_FILE
UNREADABLE_FILE = _base.UNREADABLE_FILE
MISSING_COLUMNS = _csvio.MISSING_COLUMNS
TOO_FEW_ROWS = "too_few_rows"
SOURCE_ID_NOT_INTEGER = "source_id_not_integer"
COORDS_NOT_NUMERIC = "coordinates_not_numeric"
DUPLICATE_SOURCE_ID = "duplicate_source_id"
UNKNOWN_SOURCE_ID = "unknown_source_id"  # fabricated, or wrong catalog/release
OUTSIDE_CONE = "outside_cone"
COORDS_MISMATCH = "coordinates_mismatch"  # real id, made-up position
INFRA_ERROR = _base.INFRA_ERROR

REQUIRED_COLUMNS = ("source_id", "ra", "dec")


@dataclass
class ResultRow:
    source_id: int
    ra: float
    dec: float


@dataclass
class Verdict(_base.Verdict):
    n_found: int = 0
    n_in_cone: int = 0
    bad_ids: list[int] = field(default_factory=list)


def parse_result_csv(text: str) -> tuple[list[ResultRow] | None, Verdict | None]:
    """Parse the model's CSV. Returns (rows, None) or (None, failing Verdict)."""
    try:
        table = read_result_csv(text)
        require_columns(table, REQUIRED_COLUMNS)
    except CsvError as exc:
        return None, Verdict(False, exc.reason, exc.detail, n_rows=exc.n_rows)

    rows: list[ResultRow] = []
    for i, r in enumerate(table.rows, start=1):
        sid_text = cell(r, "source_id")
        # int() must succeed on the literal text: "1.23e18" or "123.0" means the
        # id went through a float somewhere and cannot be trusted.
        if not sid_text.lstrip("-").isdigit():
            return None, Verdict(False, SOURCE_ID_NOT_INTEGER, f"row {i}: source_id={sid_text!r}", n_rows=len(table))
        try:
            ra, dec = float(cell(r, "ra")), float(cell(r, "dec"))
        except ValueError:
            return None, Verdict(
                False, COORDS_NOT_NUMERIC, f"row {i}: ra={r.get('ra')!r} dec={r.get('dec')!r}", n_rows=len(table)
            )
        rows.append(ResultRow(int(sid_text), ra, dec))
    return rows, None


def verify_rows(
    rows: list[ResultRow],
    variant: ConeVariant,
    verifier: GaiaVerifier,
    *,
    cone_tolerance_deg: float = 1e-3,
    coord_match_tolerance_deg: float = 1.0 / 3600.0,
) -> Verdict:
    """Check parsed rows against the catalog and the cone geometry.

    Raises :class:`harness.tap.TAPError` if the service cannot be reached; the
    caller decides how to record that (it is never INCORRECT).
    """
    n = len(rows)
    if n < variant.n_rows:
        return Verdict(False, TOO_FEW_ROWS, f"{n} rows, need {variant.n_rows}", n_rows=n)

    ids = [r.source_id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        return Verdict(False, DUPLICATE_SOURCE_ID, f"{len(dupes)} duplicated ids", n_rows=n, bad_ids=dupes)

    found = verifier.lookup(ids)  # may raise TAPError
    unknown = [i for i in ids if i not in found]
    if unknown:
        return Verdict(
            False,
            UNKNOWN_SOURCE_ID,
            f"{len(unknown)}/{n} ids not in Gaia DR3",
            n_rows=n,
            n_found=n - len(unknown),
            bad_ids=unknown,
        )

    outside: list[int] = []
    mismatched: list[int] = []
    for r in rows:
        cat = found[r.source_id]
        sep_center = angular_separation_deg(cat.ra, cat.dec, variant.ra, variant.dec)
        if sep_center > variant.radius + cone_tolerance_deg:
            outside.append(r.source_id)
        if angular_separation_deg(cat.ra, cat.dec, r.ra, r.dec) > coord_match_tolerance_deg:
            mismatched.append(r.source_id)

    n_in_cone = n - len(outside)
    if outside:
        return Verdict(
            False, OUTSIDE_CONE, f"{len(outside)}/{n} sources outside the cone", n_rows=n, n_found=n, n_in_cone=n_in_cone, bad_ids=outside
        )
    if mismatched:
        return Verdict(
            False,
            COORDS_MISMATCH,
            f"{len(mismatched)}/{n} rows report coordinates that do not match the catalog",
            n_rows=n,
            n_found=n,
            n_in_cone=n_in_cone,
            bad_ids=mismatched,
        )
    return Verdict(True, None, f"{n} sources verified in cone", n_rows=n, n_found=n, n_in_cone=n_in_cone)


def verify_csv_text(text: str, variant: ConeVariant, verifier: GaiaVerifier) -> Verdict:
    """Convenience: parse + verify. Still raises TAPError on service failure."""
    rows, failure = parse_result_csv(text)
    if failure is not None:
        return failure
    assert rows is not None
    return verify_rows(rows, variant, verifier)

