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

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

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
class TranslateVariant:
    id: str
    source_sql: str  # what the model is given
    gold_adql: str  # what the verifier runs; never shown to the model
    table: str
    columns: str
    limit: int
    seed: int

    @classmethod
    def from_dict(cls, d: dict) -> "TranslateVariant":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


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
                source_sql=f"SELECT {columns} FROM {table} LIMIT {limit}",
                gold_adql=gold,
                table=table,
                columns=columns,
                limit=limit,
                seed=seed,
            )
        )
    return out


def write_variants(variants: list[TranslateVariant], path: Path = DEFAULT_VARIANTS) -> None:
    with path.open("w") as fh:
        for v in variants:
            fh.write(json.dumps(asdict(v)) + "\n")


def iter_variants(path: Path = DEFAULT_VARIANTS) -> Iterator[TranslateVariant]:
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                yield TranslateVariant.from_dict(json.loads(line))


def load_variants(path: Path = DEFAULT_VARIANTS) -> list[TranslateVariant]:
    return list(iter_variants(path))


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=DEFAULT_VARIANTS)
    ap.add_argument("--check", action="store_true", help="run each gold query against MAST TAP")
    args = ap.parse_args()

    client = None
    if args.check:
        from harness.tap import GaiaVerifier

        client = GaiaVerifier()
        if not client.healthcheck():
            raise SystemExit("MAST TAP healthcheck failed; refusing to generate unchecked variants with --check")

    variants = generate(args.n, seed=args.seed, client=client)
    write_variants(variants, args.out)
    print(f"wrote {len(variants)} variants to {args.out}")


if __name__ == "__main__":
    _main()
