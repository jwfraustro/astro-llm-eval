"""Seeded variants for the ADQL translation task.

Each variant is a short query written in a SQL dialect that a TAP service will
reject (``LIMIT n`` instead of ``TOP n``), over one of the ``TAP_SCHEMA``
tables every TAP service exposes. The gold ADQL is produced alongside it, so
the verifier can run both against the service and compare results instead of
query text.

This is deliberately the simplest member of the planned ADQL task set (see
``docs/task-ideas.md``, section 1); it exists to give the task registry a
third task and will be replaced by templated questions with real difficulty.

Run as a script to (re)generate ``variants.jsonl``::

    python -m tasks.adql_translate.variants --n 10 --seed 2026 --check
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from harness.variants import Variant, variant_cli

HERE = Path(__file__).resolve().parent
DEFAULT_VARIANTS = HERE / "variants.jsonl"

# (table, column choices). "*" is always an option; the others are columns
# every TAP_SCHEMA implementation has, so the gold query runs anywhere.
TABLES: dict[str, tuple[str, ...]] = {
    "TAP_SCHEMA.schemas": ("schema_name", "schema_name, description"),
    "TAP_SCHEMA.tables": ("table_name", "schema_name, table_name", "table_name, description"),
    "TAP_SCHEMA.columns": ("column_name", "table_name, column_name", "column_name, datatype"),
}
LIMIT_RANGE = (1, 12)


@dataclass(frozen=True)
class TranslateVariant(Variant):
    source_sql: str  # what the model is given
    gold_adql: str  # what the verifier runs; never shown to the model
    table: str
    columns: str
    limit: int


def generate(n: int, *, seed: int, client=None) -> list[TranslateVariant]:
    """Generate ``n`` variants deterministically from ``seed``.

    If ``client`` (a :class:`harness.tap.GaiaVerifier`) is given, each gold
    query is run once to make sure it is accepted by the service.
    """
    rng = random.Random(seed)
    out: list[TranslateVariant] = []
    for idx in range(n):
        table = rng.choice(sorted(TABLES))
        columns = rng.choice(("*",) + TABLES[table])
        limit = rng.randint(*LIMIT_RANGE)
        gold = f"SELECT TOP {limit} {columns} FROM {table}"
        if client is not None:
            client.sync_query_full(gold)  # raises TAPError / QueryRejected if not
        out.append(
            TranslateVariant(
                id=f"adql_translate-{seed}-{idx:03d}",
                seed=seed,
                source_sql=f"SELECT {columns} FROM {table} LIMIT {limit}",
                gold_adql=gold,
                table=table,
                columns=columns,
                limit=limit,
            )
        )
    return out


def load_variants(path: Path = DEFAULT_VARIANTS) -> list[TranslateVariant]:
    return TranslateVariant.load(path)


def _main() -> None:
    from harness.tap import GaiaVerifier

    variant_cli(
        __doc__,
        default_out=DEFAULT_VARIANTS,
        check_client=GaiaVerifier,
        check_help="run each gold query against MAST TAP",
        make=lambda args, client: generate(args.n, seed=args.seed, client=client),
    )


if __name__ == "__main__":
    _main()
