"""Offline tests for the verifier logic using a fake catalog."""

from __future__ import annotations

from harness.geometry import angular_separation_deg
from harness.tap import GaiaSource
from tasks.gaia_cone import verify as V
from tasks.gaia_cone.variants import ConeVariant, generate

VARIANT = ConeVariant(id="t", ra=187.2, dec=2.05, radius=0.2, n_rows=3, seed=0)


class FakeVerifier:
    def __init__(self, catalog: dict[int, tuple[float, float]]):
        self.catalog = catalog
        self.calls = 0

    def lookup(self, ids):
        self.calls += 1
        return {i: GaiaSource(i, *self.catalog[i]) for i in ids if i in self.catalog}


CATALOG = {
    3700386145394808064: (187.20050496833028, 2.039735940891333),
    3700386420272718464: (187.192897633849, 2.0589282613426607),
    3700386454632521216: (187.21060085085534, 2.0567765627316468),
    1234567890123456789: (10.0, -40.0),  # real id, far away
}


def csv_for(ids, coords=None):
    lines = ["source_id,ra,dec"]
    for i in ids:
        ra, dec = (coords or CATALOG)[i]
        lines.append(f"{i},{ra},{dec}")
    return "\n".join(lines) + "\n"


def test_pass():
    ids = list(CATALOG)[:3]
    v = V.verify_csv_text(csv_for(ids), VARIANT, FakeVerifier(CATALOG))
    assert v.passed and v.n_in_cone == 3


def test_extra_columns_and_odd_header_case_ok():
    text = "Source_ID , RA ,dec,phot_g_mean_mag\n" + "\n".join(
        f"{i},{CATALOG[i][0]},{CATALOG[i][1]},15.1" for i in list(CATALOG)[:3]
    )
    assert V.verify_csv_text(text, VARIANT, FakeVerifier(CATALOG)).passed


def test_missing_columns():
    v = V.verify_csv_text("id,x,y\n1,2,3\n", VARIANT, FakeVerifier(CATALOG))
    assert v.reason == V.MISSING_COLUMNS


def test_too_few_rows():
    v = V.verify_csv_text(csv_for(list(CATALOG)[:2]), VARIANT, FakeVerifier(CATALOG))
    assert v.reason == V.TOO_FEW_ROWS


def test_float_source_id_rejected_before_network():
    fake = FakeVerifier(CATALOG)
    v = V.verify_csv_text("source_id,ra,dec\n3.7e18,187.2,2.05\n", VARIANT, fake)
    assert v.reason == V.SOURCE_ID_NOT_INTEGER and fake.calls == 0


def test_unknown_id_is_fabrication():
    ids = list(CATALOG)[:2] + [4242]
    coords = {**CATALOG, 4242: (187.2, 2.05)}
    v = V.verify_csv_text(csv_for(ids, coords), VARIANT, FakeVerifier(CATALOG))
    assert v.reason == V.UNKNOWN_SOURCE_ID and v.bad_ids == [4242] and v.n_found == 2


def test_outside_cone():
    ids = list(CATALOG)[:2] + [1234567890123456789]
    v = V.verify_csv_text(csv_for(ids), VARIANT, FakeVerifier(CATALOG))
    assert v.reason == V.OUTSIDE_CONE


def test_real_id_fake_coordinates():
    ids = list(CATALOG)[:3]
    coords = dict(CATALOG)
    coords[ids[0]] = (187.2, 2.05)  # cone centre, but not where the star is (>1")
    v = V.verify_csv_text(csv_for(ids, coords), VARIANT, FakeVerifier(CATALOG))
    assert v.reason == V.COORDS_MISMATCH and v.bad_ids == [ids[0]]


def test_duplicates():
    ids = [list(CATALOG)[0]] * 3
    v = V.verify_csv_text(csv_for(ids), VARIANT, FakeVerifier(CATALOG))
    assert v.reason == V.DUPLICATE_SOURCE_ID


def test_separation_sanity():
    assert abs(angular_separation_deg(0, 0, 1, 0) - 1.0) < 1e-9
    assert abs(angular_separation_deg(10, 89, 190, 89) - 2.0) < 1e-9  # across the pole


def test_variants_deterministic_and_in_range():
    a = generate(5, seed=7)
    b = generate(5, seed=7)
    assert a == b
    assert len({v.id for v in a}) == 5
    for v in a:
        assert 0 <= v.ra < 360 and -89 <= v.dec <= 89
