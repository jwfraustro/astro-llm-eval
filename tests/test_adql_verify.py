"""Offline tests for the ADQL translation verifier using a fake TAP service."""

from __future__ import annotations

from harness.tap import QueryRejected, TAPError
from tasks.adql_translate import verify as V
from tasks.adql_translate.variants import TranslateVariant, generate

VARIANT = TranslateVariant(
    id="t",
    source_sql="SELECT table_name, schema_name FROM TAP_SCHEMA.tables LIMIT 3",
    gold_adql="SELECT TOP 3 table_name, schema_name FROM TAP_SCHEMA.tables",
    table="TAP_SCHEMA.tables",
    columns="table_name, schema_name",
    limit=3,
    seed=0,
)


class FakeTAP:
    """Understands exactly enough ADQL for this task: TOP n and a column list."""

    def __init__(self, n_total: int = 5, *, down: bool = False):
        self.n_total = n_total
        self.down = down
        self.queries: list[str] = []

    def sync_query_full(self, adql: str):
        self.queries.append(adql)
        if self.down:
            raise TAPError("service down")
        up = adql.upper()
        if " LIMIT " in up or "TOP" not in up or "NOPE" in up:
            raise QueryRejected("HTTP 400: Unexpected token")
        toks = adql.split()
        n = int(toks[toks.index("TOP") + 1]) if "TOP" in toks else self.n_total
        cols_text = adql[adql.upper().index("TOP") :].split(None, 2)[2].split(" FROM ")[0]
        cols = ["utype", "description", "table_type", "table_name", "schema_name", "table_index"] if cols_text.strip() == "*" else [c.strip() for c in cols_text.split(",")]
        rows = [[f"{c}{i}" for c in cols] for i in range(min(n, self.n_total))]
        return cols, rows


def test_generate_is_deterministic_and_gold_uses_top():
    a, b = generate(5, seed=1), generate(5, seed=1)
    assert a == b
    assert len({v.id for v in a}) == 5
    for v in a:
        assert " LIMIT " in v.source_sql and "TOP" not in v.source_sql
        assert v.gold_adql.startswith(f"SELECT TOP {v.limit} ")


def test_correct_translation_passes():
    v = V.verify_query_text("SELECT TOP 3 table_name, schema_name FROM TAP_SCHEMA.tables\n", VARIANT, FakeTAP())
    assert v.passed and v.n_rows == 3 and v.n_expected == 3


def test_column_order_fences_and_semicolon_are_tolerated():
    text = "```sql\nSELECT TOP 3 schema_name, table_name\nFROM TAP_SCHEMA.tables;\n```\n"
    assert V.verify_query_text(text, VARIANT, FakeTAP()).passed


def test_unchanged_query_is_rejected_by_service():
    v = V.verify_query_text(VARIANT.source_sql, VARIANT, FakeTAP())
    assert not v.passed and v.reason == V.QUERY_REJECTED
    assert v.expected_columns == ["table_name", "schema_name"]


def test_wrong_columns_and_wrong_count():
    bad_cols = V.verify_query_text("SELECT TOP 3 table_name FROM TAP_SCHEMA.tables", VARIANT, FakeTAP())
    assert bad_cols.reason == V.COLUMNS_MISMATCH
    bad_n = V.verify_query_text("SELECT TOP 4 table_name, schema_name FROM TAP_SCHEMA.tables", VARIANT, FakeTAP())
    assert bad_n.reason == V.ROW_COUNT_MISMATCH and bad_n.n_rows == 4


def test_top_larger_than_table_still_matches_gold():
    # TAP_SCHEMA.schemas has 2 rows at MAST; TOP 5 on both sides returns 2.
    assert V.verify_query_text("SELECT TOP 3 table_name, schema_name FROM TAP_SCHEMA.tables", VARIANT, FakeTAP(n_total=2)).passed


def test_empty_file():
    assert V.verify_query_text("\n```\n```\n", VARIANT, FakeTAP()).reason == V.EMPTY_QUERY


def test_service_down_propagates_as_infra_error():
    import pytest

    with pytest.raises(TAPError):
        V.verify_query_text("SELECT TOP 3 table_name, schema_name FROM TAP_SCHEMA.tables", VARIANT, FakeTAP(down=True))
