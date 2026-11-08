"""Pure verification logic for the ObjObsSAP observability task.

Two independent questions are answered about each trial:

1. **Outcome** - do the intervals in the model's CSV match what the service
   says? (``verify_csv_text``)
2. **Process** - did the model actually drive an ObjObsSAP service, or did it
   compute rise/set locally? (``detect_protocol_use``)

They are scored separately because a frontier model can plausibly get (1)
right with astropy alone if the facility is horizon-only, and that is a
different finding from "used the protocol".
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

from harness.objobssap import Interval, ObjObsSAPClient
from harness.selfreport import parse_self_report, self_report_agreement  # noqa: F401  (re-exported)
from tasks.objobssap_window.variants import WindowVariant

# Mechanical failure reasons.
NO_OUTPUT_FILE = "no_output_file"
UNREADABLE_FILE = "unreadable_file"
MISSING_COLUMNS = "missing_columns"
TIMES_NOT_NUMERIC = "times_not_numeric"
TIMES_OUTSIDE_WINDOW = "times_outside_window"  # parsed, but nowhere near the requested MJDs: wrong time scale?
INVERTED_INTERVAL = "inverted_interval"
EXPECTED_EMPTY_GOT_ROWS = "expected_empty_got_rows"
EXPECTED_ROWS_GOT_EMPTY = "expected_rows_got_empty"  # classic "bad FACILITY -> empty -> done"
INTERVAL_COUNT_MISMATCH = "interval_count_mismatch"
INTERVAL_BOUNDARY_MISMATCH = "interval_boundary_mismatch"
INFRA_ERROR = "infra_error"

START_COLUMNS = ("t_start_mjd", "t_start", "start_mjd", "start")
STOP_COLUMNS = ("t_stop_mjd", "t_stop", "stop_mjd", "stop", "t_end_mjd", "t_end", "end_mjd", "end")

DEFAULT_TOLERANCE_S = 90.0


@dataclass
class Verdict:
    passed: bool
    reason: str | None = None
    detail: str = ""
    n_rows: int = 0
    n_expected: int = 0
    max_boundary_error_s: float | None = None
    truth_drift: bool = False  # live service answer differs from the one stored at generation
    infra_error: bool = False

    def as_metadata(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason or "pass",
            "detail": self.detail,
            "n_rows": self.n_rows,
            "n_expected": self.n_expected,
            "max_boundary_error_s": self.max_boundary_error_s,
            "truth_drift": self.truth_drift,
            "infra_error": self.infra_error,
        }


# ------------------------------------------------------------------ parsing


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def parse_result_csv(text: str) -> tuple[list[Interval] | None, Verdict | None]:
    try:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return None, Verdict(False, UNREADABLE_FILE, "empty file")
        reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]
        raw = list(reader)
    except csv.Error as exc:
        return None, Verdict(False, UNREADABLE_FILE, f"csv parse error: {exc}")

    c_start = _find_column(reader.fieldnames, START_COLUMNS)
    c_stop = _find_column(reader.fieldnames, STOP_COLUMNS)
    if c_start is None or c_stop is None:
        return None, Verdict(False, MISSING_COLUMNS, f"need t_start_mjd/t_stop_mjd; got {reader.fieldnames}")

    out: list[Interval] = []
    for i, r in enumerate(raw, start=1):
        try:
            a, b = float((r.get(c_start) or "").strip()), float((r.get(c_stop) or "").strip())
        except ValueError:
            return None, Verdict(False, TIMES_NOT_NUMERIC, f"row {i}: {r.get(c_start)!r}, {r.get(c_stop)!r}", n_rows=len(raw))
        if b < a:
            return None, Verdict(False, INVERTED_INTERVAL, f"row {i}: stop {b} < start {a}", n_rows=len(raw))
        out.append(Interval(a, b))
    return sorted(out, key=lambda iv: iv.t_start), None


# --------------------------------------------------------------- comparison


def normalize(intervals: list[Interval], t0: float, t1: float, min_obs_s: float) -> list[Interval]:
    """Clip to the requested window and drop anything shorter than MIN_OBS.

    The service returns overlapping intervals whole; a model may clip them to
    the window. Both are reasonable readings of the task, so compare on the
    clipped form.
    """
    out: list[Interval] = []
    for iv in intervals:
        a, b = max(iv.t_start, t0), min(iv.t_stop, t1)
        if b > a and (b - a) * 86400.0 >= min_obs_s:
            out.append(Interval(a, b))
    return out


def compare_intervals(
    got: list[Interval],
    expected: list[Interval],
    variant: WindowVariant,
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> Verdict:
    n_rows, n_exp = len(got), len(expected)

    # Time-scale sanity: anything a year away from the window is not MJD.
    for iv in got:
        if iv.t_stop < variant.t0 - 365 or iv.t_start > variant.t1 + 365:
            return Verdict(False, TIMES_OUTSIDE_WINDOW, f"interval {iv.t_start}-{iv.t_stop} is far from MJD {variant.t0}-{variant.t1}", n_rows=n_rows, n_expected=n_exp)

    g = normalize(got, variant.t0, variant.t1, variant.min_obs)
    e = normalize(expected, variant.t0, variant.t1, variant.min_obs)

    # Rows that exist but never intersect the window must not vanish quietly:
    # on a negative control they are invented observability; otherwise they are
    # in the wrong place (or the wrong time scale).
    if got and not g:
        reason = EXPECTED_EMPTY_GOT_ROWS if not e else TIMES_OUTSIDE_WINDOW
        return Verdict(False, reason, f"{n_rows} row(s) reported, none intersect MJD {variant.t0}-{variant.t1}", n_rows=n_rows, n_expected=n_exp)

    if not e and g:
        return Verdict(False, EXPECTED_EMPTY_GOT_ROWS, f"never observable, but {len(g)} interval(s) reported", n_rows=n_rows, n_expected=n_exp)
    if e and not g:
        return Verdict(False, EXPECTED_ROWS_GOT_EMPTY, f"{len(e)} interval(s) expected, none reported", n_rows=n_rows, n_expected=n_exp)
    if len(g) != len(e):
        return Verdict(False, INTERVAL_COUNT_MISMATCH, f"{len(g)} interval(s) reported, {len(e)} expected", n_rows=n_rows, n_expected=n_exp)

    worst = 0.0
    for a, b in zip(g, e):
        worst = max(worst, abs(a.t_start - b.t_start) * 86400.0, abs(a.t_stop - b.t_stop) * 86400.0)
    if worst > tolerance_s:
        return Verdict(False, INTERVAL_BOUNDARY_MISMATCH, f"worst boundary error {worst:.0f}s > {tolerance_s:.0f}s", n_rows=n_rows, n_expected=n_exp, max_boundary_error_s=worst)
    return Verdict(True, None, f"{len(e)} interval(s) matched within {worst:.0f}s", n_rows=n_rows, n_expected=n_exp, max_boundary_error_s=worst)


def fetch_truth(variant: WindowVariant, client: ObjObsSAPClient) -> tuple[list[Interval], bool]:
    """Live answer from the service, plus whether it drifted from the stored one."""
    live = client.query(variant.ra, variant.dec, variant.t0, variant.t1, min_obs=variant.min_obs)  # may raise ObjObsSAPError
    drift = False
    if variant.expected is not None:
        stored = [Interval(a, b) for a, b in variant.expected]
        drift = len(stored) != len(live) or any(
            abs(s.t_start - l.t_start) * 86400 > 1 or abs(s.t_stop - l.t_stop) * 86400 > 1 for s, l in zip(stored, live)
        )
    return live, drift


def verify_csv_text(text: str, variant: WindowVariant, client: ObjObsSAPClient, *, tolerance_s: float = DEFAULT_TOLERANCE_S) -> Verdict:
    """Parse + fetch truth + compare. Raises ObjObsSAPError on service failure."""
    got, failure = parse_result_csv(text)
    if failure is not None:
        return failure
    assert got is not None
    truth, drift = fetch_truth(variant, client)
    verdict = compare_intervals(got, truth, variant, tolerance_s=tolerance_s)
    verdict.truth_drift = drift
    return verdict


# ------------------------------------------------------- process detection

SERVICE_HOST_RE = re.compile(r"vo-labs\.dev/objobssap", re.I)
PROTOCOL_NAME_RE = re.compile(r"objobssap|ivo://ivoa\.net/std/ObjObsSAP", re.I)
VOTABLE_FIELD_RE = re.compile(r"t_observability|time\.validity", re.I)
LOCAL_COMPUTE_RE = re.compile(r"EarthLocation|AltAz|astroplan|Observer\(|get_sun\(|target_rise_time|target_set_time", re.I)


@dataclass
class ProtocolSignals:
    service_called: bool = False  # service URL appears in a tool call
    service_response_seen: bool = False  # ObjObsSAP VOTable fields appear in a tool result
    protocol_named: bool = False  # the protocol is named anywhere in the transcript
    local_computation: bool = False  # astropy/astroplan rise-set machinery in tool calls
    evidence: list[str] = field(default_factory=list)

    @property
    def used_protocol(self) -> bool:
        return self.service_called or self.service_response_seen

    def as_metadata(self) -> dict:
        return {
            "used_protocol": self.used_protocol,
            "service_called": self.service_called,
            "service_response_seen": self.service_response_seen,
            "protocol_named": self.protocol_named,
            "local_computation": self.local_computation,
            "evidence": self.evidence[:10],
        }


def _snippet(text: str, m: re.Match) -> str:
    lo, hi = max(0, m.start() - 40), min(len(text), m.end() + 40)
    return text[lo:hi].replace("\n", " ")


def detect_protocol_use(tool_call_texts: list[str], tool_result_texts: list[str], other_texts: list[str] = ()) -> ProtocolSignals:
    sig = ProtocolSignals()
    for t in tool_call_texts:
        if m := SERVICE_HOST_RE.search(t):
            sig.service_called = True
            sig.evidence.append(f"call: ...{_snippet(t, m)}...")
        if m := LOCAL_COMPUTE_RE.search(t):
            sig.local_computation = True
            sig.evidence.append(f"local: ...{_snippet(t, m)}...")
    for t in tool_result_texts:
        if m := VOTABLE_FIELD_RE.search(t):
            sig.service_response_seen = True
            sig.evidence.append(f"result: ...{_snippet(t, m)}...")
    for t in [*tool_call_texts, *tool_result_texts, *other_texts]:
        if PROTOCOL_NAME_RE.search(t):
            sig.protocol_named = True
            break
    return sig
