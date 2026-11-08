"""Live tests against vo-labs.dev. Run with `pytest -m live`."""

from __future__ import annotations

import pytest

from harness.objobssap import ObjObsSAPClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    c = ObjObsSAPClient()
    yield c
    c.close()


def test_healthcheck(client):
    assert client.healthcheck()


def test_query_returns_daily_windows(client):
    ivs = client.query(187.2, 2.05, 61000.0, 61001.0, min_obs=30)
    assert len(ivs) == 2  # overlapping intervals are returned whole
    assert ivs[0].t_start == pytest.approx(61000.3112, abs=1e-3)
    assert ivs[0].facility == "Example Observatory"
    assert ivs[0].t_observability == pytest.approx(ivs[0].duration_s, rel=1e-3)


def test_never_rises_is_empty(client):
    assert client.query(10.0, -80.0, 61000.0, 61002.0) == []


def test_unknown_facility_is_silently_empty(client):
    # The trap the task relies on: not an error, just no rows.
    assert client.query(187.2, 2.05, 61000.0, 61001.0, facility="HST") == []
