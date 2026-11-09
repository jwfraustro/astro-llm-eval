"""Offline tests for the shared Verdict, CSV reading and Variant plumbing."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import pytest

from harness import csvio
from harness.geometry import random_sky_position, uniform_dec
from harness.variants import Variant, write_variants
from harness.verdict import MAX_LIST_ITEMS, Verdict


@dataclass
class MyVerdict(Verdict):
    bad_ids: list[int] = field(default_factory=list)
    note: str | None = None


def test_verdict_metadata_is_log_safe():
    big_id = 3700386145394808064
    md = MyVerdict(False, "unknown_source_id", "x", bad_ids=[big_id] * (MAX_LIST_ITEMS + 5)).as_metadata()
    assert md["reason"] == "unknown_source_id"
    assert md["bad_ids"] == [str(big_id)] * MAX_LIST_ITEMS  # capped and stringified
    assert md["note"] is None and md["infra_error"] is False
    assert Verdict(True).as_metadata()["reason"] == "pass"


def test_read_result_csv_normalises_headers():
    table = csvio.read_result_csv(" Source_ID , RA,dec\n1,2,3\n")
    assert table.fieldnames == ["source_id", "ra", "dec"] and len(table) == 1
    assert csvio.cell(table.rows[0], "source_id") == "1"
    assert csvio.cell(table.rows[0], None) == ""
    assert csvio.find_column(table.fieldnames, ("t_start", "ra")) == "ra"


def test_read_result_csv_failures():
    with pytest.raises(csvio.CsvError) as exc:
        csvio.read_result_csv("")
    assert exc.value.reason == csvio.UNREADABLE_FILE
    table = csvio.read_result_csv("a,b\n1,2\n3,4\n")
    with pytest.raises(csvio.CsvError) as exc:
        csvio.require_columns(table, ("a", "c"))
    assert exc.value.reason == csvio.MISSING_COLUMNS and exc.value.n_rows == 2 and "['c']" in exc.value.detail


@dataclass(frozen=True)
class MyVariant(Variant):
    x: float
    optional: str | None = None


def test_variant_round_trip_and_tolerant_from_dict(tmp_path):
    vs = [MyVariant(id="t-0", seed=1, x=0.5), MyVariant(id="t-1", seed=1, x=1.5, optional="y")]
    path = tmp_path / "v.jsonl"
    write_variants(vs, path)
    assert MyVariant.load(path) == vs
    # A file written before `optional` existed, with a stray key, still loads.
    assert MyVariant.from_dict({"id": "t", "seed": 2, "x": 1.0, "legacy": True}) == MyVariant(id="t", seed=2, x=1.0)
    with pytest.raises(TypeError):
        MyVariant.from_dict({"id": "t", "seed": 2})  # required field missing
    assert json.loads(path.read_text().splitlines()[0])["id"] == "t-0"


def test_sky_sampling_is_uniform_on_sphere_and_banded():
    rng = random.Random(0)
    decs = [uniform_dec(rng, -30.0, 10.0) for _ in range(2000)]
    assert all(-30.0 <= d <= 10.0 for d in decs)
    # Uniform in sin(dec): the mean of sin(dec) is the band's midpoint in sin space.
    import math

    mean_sin = sum(math.sin(math.radians(d)) for d in decs) / len(decs)
    assert abs(mean_sin - (math.sin(math.radians(-30)) + math.sin(math.radians(10))) / 2) < 0.02
    ra, dec = random_sky_position(random.Random(1), dec_min=-90, dec_max=-60)
    assert 0 <= ra < 360 and dec <= -60
