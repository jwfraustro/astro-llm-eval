"""Offline tests for the ObjObsSAP task: VOTable parsing, comparison, process detection."""

from __future__ import annotations

import pytest

from harness.objobssap import Interval, ObjObsSAPError, parse_votable_intervals
from tasks.objobssap_window import verify as V
from tasks.objobssap_window.variants import WindowVariant, generate

VOTABLE = """<?xml version="1.0"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
 <RESOURCE type="results">
  <INFO ID="QUERY_STATUS" name="QUERY_STATUS" value="OK"/>
  <TABLE>
   <FIELD ID="t_validity" datatype="float" name="t_validity"/>
   <FIELD ID="validity_accuracy" datatype="char" name="validity_accuracy" arraysize="*"/>
   <FIELD ID="validity_predictor" datatype="char" name="validity_predictor" arraysize="*"/>
   <FIELD ID="t_start" datatype="float" name="t_start"/>
   <FIELD ID="t_stop" datatype="float" name="t_stop"/>
   <FIELD ID="t_observability" datatype="float" name="t_observability"/>
   <FIELD ID="facility" datatype="char" name="facility" arraysize="*"/>
   <DATA><TABLEDATA>
    <TR><TD>61011.3</TD><TD>HIGH</TD><TD>Example Observatory</TD><TD>61001.30846697977</TD><TD>61001.81576698786</TD><TD>43830.72</TD><TD>Example Observatory</TD></TR>
    <TR><TD>61010.3</TD><TD>HIGH</TD><TD>Example Observatory</TD><TD>61000.31119652558</TD><TD>61000.81849727314</TD><TD>43830.78</TD><TD>Example Observatory</TD></TR>
   </TABLEDATA></DATA>
  </TABLE>
 </RESOURCE>
</VOTABLE>"""

ERROR_VOTABLE = """<?xml version="1.0"?><VOTABLE version="1.1" xmlns="http://www.ivoa.net/xml/VOTable/v1.1">
<INFO ID="Error" name="Error" value="Error in query RESPONSEFORMAT: Input should be 'votable'"/></VOTABLE>"""

TRUTH = [Interval(61000.31119652558, 61000.81849727314), Interval(61001.30846697977, 61001.81576698786)]
VARIANT = WindowVariant(id="t", ra=187.2, dec=2.05, t0=61000.0, t1=61001.0, min_obs=30, seed=0, negative_control=False)
NEGATIVE = WindowVariant(id="n", ra=10.0, dec=-80.0, t0=61000.0, t1=61002.0, min_obs=30, seed=0, negative_control=True)


class FakeClient:
    def __init__(self, truth):
        self.truth = truth

    def query(self, ra, dec, t0, t1, *, min_obs=30, facility=None):
        return list(self.truth)


def csv_of(intervals, header="t_start_mjd,t_stop_mjd"):
    return header + "\n" + "".join(f"{a},{b}\n" for a, b in intervals)


# --------------------------------------------------------------- VOTable


def test_parse_votable_sorted_and_typed():
    ivs = parse_votable_intervals(VOTABLE)
    assert [iv.t_start for iv in ivs] == [61000.31119652558, 61001.30846697977]
    assert ivs[0].facility == "Example Observatory" and ivs[0].t_observability == 43830.78


def test_parse_votable_error_info_raises():
    with pytest.raises(ObjObsSAPError, match="RESPONSEFORMAT"):
        parse_votable_intervals(ERROR_VOTABLE)


# ------------------------------------------------------------ comparison


def test_pass_whole_intervals():
    text = csv_of([(iv.t_start, iv.t_stop) for iv in TRUTH])
    v = V.verify_csv_text(text, VARIANT, FakeClient(TRUTH))
    assert v.passed and v.max_boundary_error_s < 1


def test_pass_when_model_clips_to_window():
    # Service returns overlapping intervals whole; a model that clips to [t0, t1] is also right.
    text = csv_of([(61000.31119652558, 61000.81849727314)])  # second interval lies past t1 entirely
    assert V.verify_csv_text(text, VARIANT, FakeClient(TRUTH)).passed


def test_pass_with_rounding_within_tolerance():
    text = csv_of([(61000.3112, 61000.8185), (61001.3085, 61001.8158)])
    assert V.verify_csv_text(text, VARIANT, FakeClient(TRUTH)).passed


def test_boundary_mismatch():
    text = csv_of([(61000.30, 61000.82), (61001.3085, 61001.8158)])  # ~16 min off
    v = V.verify_csv_text(text, VARIANT, FakeClient(TRUTH))
    assert v.reason == V.INTERVAL_BOUNDARY_MISMATCH


def test_alternate_headers_ok():
    text = csv_of([(iv.t_start, iv.t_stop) for iv in TRUTH], header="T_START, T_STOP")
    assert V.verify_csv_text(text, VARIANT, FakeClient(TRUTH)).passed


def test_missing_columns():
    assert V.verify_csv_text("start_time,end_time\n1,2\n", VARIANT, FakeClient(TRUTH)).reason == V.MISSING_COLUMNS


def test_iso_times_rejected():
    text = "t_start_mjd,t_stop_mjd\n2025-11-20T07:28:00,2025-11-20T19:38:00\n"
    assert V.verify_csv_text(text, VARIANT, FakeClient(TRUTH)).reason == V.TIMES_NOT_NUMERIC


def test_jd_instead_of_mjd_flagged():
    text = csv_of([(2461000.81, 2461001.32)])
    assert V.verify_csv_text(text, VARIANT, FakeClient(TRUTH)).reason == V.TIMES_OUTSIDE_WINDOW


def test_guessed_facility_empty_result():
    v = V.verify_csv_text("t_start_mjd,t_stop_mjd\n", VARIANT, FakeClient(TRUTH))
    assert v.reason == V.EXPECTED_ROWS_GOT_EMPTY and v.n_expected == 2


def test_negative_control_pass_and_fail():
    assert V.verify_csv_text("t_start_mjd,t_stop_mjd\n", NEGATIVE, FakeClient([])).passed
    v = V.verify_csv_text(csv_of([(61000.4, 61000.9)]), NEGATIVE, FakeClient([]))
    assert v.reason == V.EXPECTED_EMPTY_GOT_ROWS


def test_invented_rows_outside_window_still_fail():
    # Regression: clipping to [t0, t1] must not silently erase rows that never
    # intersect the window. On a negative control that is invented observability...
    v = V.verify_csv_text(csv_of([(61010.4, 61010.9)]), NEGATIVE, FakeClient([]))
    assert v.reason == V.EXPECTED_EMPTY_GOT_ROWS
    # ...and on a positive variant it means the times are in the wrong place.
    v = V.verify_csv_text(csv_of([(61010.4, 61010.9), (61011.4, 61011.9)]), VARIANT, FakeClient(TRUTH))
    assert v.reason == V.TIMES_OUTSIDE_WINDOW


def test_faithful_echo_with_spillover_interval_passes():
    # The service returns an interval that starts after t1 (61001.308 > 61001.0);
    # a model that copies the service output verbatim must still pass.
    text = csv_of([(iv.t_start, iv.t_stop) for iv in TRUTH])
    assert TRUTH[1].t_start > VARIANT.t1
    assert V.verify_csv_text(text, VARIANT, FakeClient(TRUTH)).passed


# Haiku's real run: computed locally, reported the interval already in progress
# at t0 (61358.3-61358.44) which the service omits, plus three that match.
HAIKU_VARIANT = WindowVariant(id="h", ra=42.8832, dec=3.0763, t0=61358.3, t1=61361.3, min_obs=30, seed=0, negative_control=False)
HAIKU_TRUTH = [Interval(61358.92561524315, 61359.43875143258), Interval(61359.922885739245, 61360.43602104671), Interval(61360.92015539482, 61361.43329065014)]
HAIKU_CSV = "t_start_mjd,t_stop_mjd\n61358.3,61358.441550923424\n61358.925694433376,61359.438773128\n61359.92291663795,61360.43599533258\n61360.92013884253,61361.29988420618\n"


def test_in_progress_at_start_is_optional_by_default():
    v = V.verify_csv_text(HAIKU_CSV, HAIKU_VARIANT, FakeClient(HAIKU_TRUTH))
    assert v.passed and v.in_progress_at_start == "reported_only" and v.max_boundary_error_s < 30


def test_in_progress_at_start_strict():
    got, _ = V.parse_result_csv(HAIKU_CSV)
    v = V.compare_intervals(got, HAIKU_TRUTH, HAIKU_VARIANT, strict_window_start=True)
    assert v.reason == V.INTERVAL_COUNT_MISMATCH


def test_in_progress_expected_only_is_also_optional():
    # If the service is later fixed to return the in-progress interval, a model that omits it still passes.
    truth = [Interval(61357.92, 61358.441550923424), *HAIKU_TRUTH]
    text = csv_of([(iv.t_start, iv.t_stop) for iv in HAIKU_TRUTH])
    v = V.verify_csv_text(text, HAIKU_VARIANT, FakeClient(truth))
    assert v.passed and v.in_progress_at_start == "expected_only"


def test_only_an_in_progress_interval_when_more_expected_fails():
    v = V.verify_csv_text("t_start_mjd,t_stop_mjd\n61358.3,61358.44\n", HAIKU_VARIANT, FakeClient(HAIKU_TRUTH))
    assert v.reason == V.EXPECTED_ROWS_GOT_EMPTY


def test_truth_drift_detected():
    stored = WindowVariant(**{**VARIANT.__dict__, "expected": [[61000.0, 61000.5], [61001.0, 61001.5]]})
    text = csv_of([(iv.t_start, iv.t_stop) for iv in TRUTH])
    v = V.verify_csv_text(text, stored, FakeClient(TRUTH))
    assert v.passed and v.truth_drift


# ----------------------------------------------------- process detection


VOTABLE_RESULT = '<VOTABLE><INFO name="QUERY_STATUS" value="OK"/><FIELD name="t_start"/><TR><TD>61000.3</TD></TR></VOTABLE>'


def test_detect_curl_request_with_votable_response():
    call = '{"command": "curl -s \'https://vo-labs.dev/objobssap/api/query?POS=42.88,3.07&TIME=61358.3/61361.3&MIN_OBS=30\'"}'
    sig = V.detect_protocol_use([(call, VOTABLE_RESULT)])
    assert sig.used_protocol and sig.hosts == ["vo-labs"] and not sig.local_computation


def test_detect_python_params_request():
    call = '{"code": "r = requests.get(\'https://vo-labs.dev/objobssap/api/query\', params={\'POS\': f\'{ra},{dec}\', \'TIME\': \'61358.3/61361.3\', \'MIN_OBS\': 30})\\nprint(r.text)"}'
    sig = V.detect_protocol_use([(call, VOTABLE_RESULT)])
    assert sig.used_protocol and sig.hosts == ["vo-labs"]


def test_request_without_response_is_an_attempt_not_use():
    # Sonnet's actual xmmvischeck attempt: request-shaped, nothing came back.
    call = '{"command": "curl -s -m 20 \\"http://xmmvischeck.esac.esa.int:8080/objobssap/query?POS=42.8832,3.0763&TIME=61358.3/61361.3&MIN_OBS=30\\" | head -50"}'
    sig = V.detect_protocol_use([(call, "HTTP/2 200\ndate: ...\n")])
    assert not sig.used_protocol and sig.queries and sig.hosts == ["esa-xmm"] and not sig.queries[0].responded


def test_spec_text_in_result_is_not_a_response():
    # Regression: curl-ing the Working Draft puts t_observability / VOTABLE text
    # in a tool result. That is reading, not querying.
    call = '{"command": "curl -s https://www.ivoa.net/documents/ObjObsSAP/20241118/WD-ObjObsSAP-1.0-20241118.html | sed ..."}'
    result = 'name="t_observability" ... &lt;VOTABLE xmlns:xsi=... QUERY_STATUS ... t_start'
    sig = V.detect_protocol_use([(call, result)])
    assert not sig.used_protocol and not sig.queries and sig.spec_read and sig.protocol_named


def test_pip_install_astroplan_is_not_local_computation():
    sig = V.detect_protocol_use([('{"command": "pip install -q fastapi uvicorn astroplan pydantic-settings"}', "ok")])
    assert not sig.local_computation
    sig = V.detect_protocol_use([('{"code": "from astroplan import Observer\\nobs = Observer(latitude=39.3)"}', "ok")])
    assert sig.local_computation


def test_source_found_and_self_hosted():
    pairs = [
        ('{"command": "git clone -q https://github.com/jwfraustro/fastapi-objobssap"}', "done"),
        ('{"command": "curl -s \'http://localhost:8000/objobssap/api/query?POS=42.88,3.07&TIME=61358.3/61361.3&MIN_OBS=30\'"}', VOTABLE_RESULT),
    ]
    sig = V.detect_protocol_use(pairs)
    assert sig.source_found and sig.self_hosted and sig.used_protocol and sig.hosts == ["localhost"]


def test_request_shaped_prose_without_url_or_response_is_ignored():
    # Sonnet's submit message recapped "POS=42.8832,3.0763, TIME=... MIN_OBS=30"; the scorer
    # excludes submit calls, but any other URL-less, response-less match must not count either.
    call = '{"answer": "Queried the service with POS=42.8832,3.0763, TIME=61358.3/61361.3 and MIN_OBS=30."}'
    sig = V.detect_protocol_use([(call, "Queried the service with POS=42.8832,3.0763 ...")])
    assert not sig.queries


def test_request_without_url_but_with_response_counts():
    # URL built in an earlier call; this call only carries the params and gets a VOTable back.
    sig = V.detect_protocol_use([('{"code": "r = s.get(url, params={\'POS\': \'42.88,3.07\', \'MIN_OBS\': 30})\\nprint(r.text)"}', VOTABLE_RESULT)])
    assert sig.used_protocol and sig.hosts == ["<no-url>"]


def test_admits_no_vo_regex():
    assert V.ADMITS_NO_VO_RE.search("SUCCESS: ...; I did not use a VO protocol service and did not impose a night/Sun constraint.")
    assert V.ADMITS_NO_VO_RE.search("SUCCESS: computed locally with astropy")
    assert not V.ADMITS_NO_VO_RE.search("SUCCESS: Queried the VO-Labs ObjObsSAP service")


def test_mention_alone_is_not_use():
    sig = V.detect_protocol_use([], ["ObjObsSAP would be the right protocol here."])
    assert sig.protocol_named and not sig.used_protocol and not sig.queries


# ----------------------------------------------------------------- variants


def test_variants_negative_controls_and_windows():
    vs = generate(10, seed=3)
    assert generate(10, seed=3) == vs
    negs = [v for v in vs if v.negative_control]
    assert len(negs) == 2 and all(v.dec <= -60 for v in negs)
    for v in vs:
        assert 3 <= v.t1 - v.t0 <= 7 and (v.negative_control or -40 <= v.dec <= 48)
