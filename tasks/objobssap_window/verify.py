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
#
# Detection works on (tool call, tool result) PAIRS. The earlier version looked
# for VOTable field names anywhere in tool results and tripped on the model
# curl-ing the ObjObsSAP spec; it also flagged `pip install astroplan` as local
# computation. Lesson: evidence of *use* must come from a request-shaped call
# and the response to that same call.

# An ObjObsSAP request: a POS=ra,dec pair together with TIME or MIN_OBS, either
# as URL query syntax or as Python dict/kwarg syntax.
REQUEST_RE = re.compile(
    r"""(POS\s*=\s*[-+\d.]+\s*,\s*[-+\d.]+ | ["']POS["']\s*:\s*f?["'][-+\d.{}\w]+\s*,)""",
    re.I | re.X,
)
REQUEST_EXTRA_RE = re.compile(r"TIME\s*[=:]|MIN_OBS", re.I)
# A VOTable came back (any VOTable: we only ask whether the request got a response).
RESPONSE_RE = re.compile(r"<VOTABLE|QUERY_STATUS|\bt_start\b", re.I)
URL_RE = re.compile(r"https?://[^\s\"'\\]+", re.I)
URL_HOST_RE = re.compile(r"https?://([A-Za-z0-9.\-]+(?::\d+)?)", re.I)


def _request_hosts(call: str) -> list[str]:
    """Hosts of the URLs in a call, preferring ones that carry the query itself."""
    urls = URL_RE.findall(call)
    query_urls = [u for u in urls if re.search(r"POS=|/query\b", u, re.I)]
    hosts = [m.group(1) for u in (query_urls or urls) if (m := URL_HOST_RE.match(u))]
    return list(dict.fromkeys(hosts)) or ["<no-url>"]

PROTOCOL_NAME_RE = re.compile(r"objobssap|ivo://ivoa\.net/std/ObjObsSAP", re.I)
SPEC_RE = re.compile(r"ivoa\.net/documents/ObjObsSAP", re.I)
SOURCE_RE = re.compile(r"github\.com/[\w.\-]+/[\w.\-]*objobssap|fastapi-objobssap|swift-vo", re.I)
LOCALHOST_RE = re.compile(r"^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(:\d+)?$", re.I)
# Actual rise/set code, not the word "astroplan" in a pip install line.
LOCAL_COMPUTE_RE = re.compile(
    r"from\s+astroplan|import\s+astroplan|EarthLocation\(|AltAz\(|Observer\(|get_sun\(|target_rise_time|target_set_time|\.rise_time\(|\.set_time\(",
)

KNOWN_SERVICE_HOSTS = {
    "vo-labs.dev": "vo-labs",
    "xmmvischeck.esac.esa.int:8080": "esa-xmm",
    "xmmvischeck.esac.esa.int": "esa-xmm",
}


@dataclass
class ServiceQuery:
    host: str
    responded: bool  # a VOTable came back for this request
    snippet: str


@dataclass
class ProtocolSignals:
    queries: list[ServiceQuery] = field(default_factory=list)
    protocol_named: bool = False  # the protocol is named anywhere in the transcript
    spec_read: bool = False  # fetched the IVOA Working Draft
    source_found: bool = False  # found an implementation's source code
    local_computation: bool = False  # astropy/astroplan rise-set code in tool calls
    evidence: list[str] = field(default_factory=list)

    @property
    def service_queried(self) -> bool:
        return any(q.responded for q in self.queries)

    @property
    def used_protocol(self) -> bool:
        return self.service_queried

    @property
    def self_hosted(self) -> bool:
        return any(LOCALHOST_RE.match(q.host) for q in self.queries)

    @property
    def hosts(self) -> list[str]:
        seen: list[str] = []
        for q in self.queries:
            label = KNOWN_SERVICE_HOSTS.get(q.host.lower(), "localhost" if LOCALHOST_RE.match(q.host) else q.host)
            if label not in seen:
                seen.append(label)
        return seen

    def as_metadata(self) -> dict:
        return {
            "used_protocol": self.used_protocol,
            "service_queried": self.service_queried,
            "query_attempts": len(self.queries),
            "hosts": self.hosts,
            "self_hosted": self.self_hosted,
            "protocol_named": self.protocol_named,
            "spec_read": self.spec_read,
            "source_found": self.source_found,
            "local_computation": self.local_computation,
            "evidence": self.evidence[:12],
        }


def _snippet(text: str, m: re.Match, pad: int = 50) -> str:
    lo, hi = max(0, m.start() - pad), min(len(text), m.end() + pad)
    return text[lo:hi].replace("\n", " ")


def detect_protocol_use(tool_pairs: list[tuple[str, str]], other_texts: list[str] = ()) -> ProtocolSignals:
    """Classify the transcript from (tool call text, tool result text) pairs."""
    sig = ProtocolSignals()
    for call, result in tool_pairs:
        if (m := REQUEST_RE.search(call)) and REQUEST_EXTRA_RE.search(call):
            hosts = _request_hosts(call)
            responded = bool(RESPONSE_RE.search(result or ""))
            for host in hosts:
                sig.queries.append(ServiceQuery(host=host, responded=responded, snippet=_snippet(call, m)))
            sig.evidence.append(f"request{'+response' if responded else ' (no VOTable back)'} -> {hosts}: ...{_snippet(call, m)}...")
        if m := SPEC_RE.search(call):
            sig.spec_read = True
        if m := SOURCE_RE.search(call):
            if not sig.source_found:
                sig.evidence.append(f"source: ...{_snippet(call, m)}...")
            sig.source_found = True
        if m := LOCAL_COMPUTE_RE.search(call):
            if not sig.local_computation:
                sig.evidence.append(f"local: ...{_snippet(call, m)}...")
            sig.local_computation = True
    for t in [*(c for c, _ in tool_pairs), *(r for _, r in tool_pairs), *other_texts]:
        if PROTOCOL_NAME_RE.search(t or ""):
            sig.protocol_named = True
            break
    return sig
