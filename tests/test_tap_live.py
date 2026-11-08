"""Live tests against MAST. Run with `pytest -m live`; skipped by default via
`-m 'not live'` in CI. These make three small queries."""

from __future__ import annotations

import pytest

from harness.tap import GaiaVerifier

pytestmark = pytest.mark.live

KNOWN_ID = 3700386145394808064  # near RA=187.2, Dec=+2.05
BOGUS_ID = 1


@pytest.fixture(scope="module")
def verifier():
    v = GaiaVerifier()
    yield v
    v.close()


def test_healthcheck(verifier):
    assert verifier.healthcheck()


def test_lookup_mixed(verifier):
    found = verifier.lookup([KNOWN_ID, BOGUS_ID])
    assert KNOWN_ID in found and BOGUS_ID not in found
    assert abs(found[KNOWN_ID].ra - 187.2005) < 1e-3


def test_count_in_cone(verifier):
    assert verifier.count_in_cone(187.2, 2.05, 0.05, limit=10) == 10
