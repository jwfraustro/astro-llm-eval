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


def test_truth_drift_detected():
    stored = WindowVariant(**{**VARIANT.__dict__, "expected": [[61000.0, 61000.5], [61001.0, 61001.5]]})
    text = csv_of([(iv.t_start, iv.t_stop) for iv in TRUTH])
    v = V.verify_csv_text(text, stored, FakeClient(TRUTH))
    assert v.passed and v.truth_drift


# ----------------------------------------------------- process detection


def test_detect_service_call_and_response():
    sig = V.detect_protocol_use(
        ['{"code": "requests.get(\'https://vo-labs.dev/objobssap/api/query\', params=...)"}'],
        ["<FIELD name=\"t_observability\" .../>"],
    )
    assert sig.used_protocol and sig.service_called and sig.service_response_seen and not sig.local_computation


def test_detect_local_computation_only():
    sig = V.detect_protocol_use(
        ['{"code": "from astropy.coordinates import EarthLocation, AltAz\\nloc = EarthLocation(lat=39.3, lon=-76.6)"}'],
        ["rise 07:28 set 19:38"],
        ["I will use ObjObsSAP concepts but compute locally"],
    )
    assert not sig.used_protocol and sig.local_computation and sig.protocol_named


def test_mention_alone_is_not_use():
    sig = V.detect_protocol_use([], [], ["ObjObsSAP would be the right protocol here."])
    assert sig.protocol_named and not sig.used_protocol


# ----------------------------------------------------------------- variants


def test_variants_negative_controls_and_windows():
    vs = generate(10, seed=3)
    assert generate(10, seed=3) == vs
    negs = [v for v in vs if v.negative_control]
    assert len(negs) == 2 and all(v.dec <= -60 for v in negs)
    for v in vs:
        assert 3 <= v.t1 - v.t0 <= 7 and (v.negative_control or -40 <= v.dec <= 48)
