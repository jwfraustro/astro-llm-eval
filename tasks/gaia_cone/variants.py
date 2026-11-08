"""Seeded cone-search variants for the Gaia fetch task.

A bare "fetch 10 rows of Gaia DR3" is memorisable in principle (real source_ids
appear in papers). Pinning the request to a random cone on the sky removes that
escape hatch: the verifier checks both that every id exists *and* that it sits
inside the cone. One template therefore yields an unbounded supply of trials.

Run as a script to (re)generate ``variants.jsonl``::

    python -m tasks.gaia_cone.variants --n 20 --seed 2026 --check
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
DEFAULT_VARIANTS = HERE / "variants.jsonl"


@dataclass(frozen=True)
class ConeVariant:
    id: str
    ra: float  # ICRS, degrees
    dec: float  # ICRS, degrees
    radius: float  # degrees
    n_rows: int
    seed: int

    @classmethod
    def from_dict(cls, d: dict) -> "ConeVariant":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def random_cone_center(rng: random.Random) -> tuple[float, float]:
    """Uniform point on the sphere, rounded so prompts stay tidy.

    We stay 1 degree clear of the poles: not because anything breaks there, but
    because a cone at |dec| ~ 90 is an unrepresentative edge case for a first
    task. Later tasks can target it deliberately.
    """
    ra = rng.uniform(0.0, 360.0)
    dec = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))
    dec = max(-89.0, min(89.0, dec))
    return round(ra, 4), round(dec, 4)


def generate(
    n: int,
    *,
    seed: int,
    radius: float = 0.2,
    n_rows: int = 10,
    verifier=None,
) -> list[ConeVariant]:
    """Generate ``n`` variants deterministically from ``seed``.

    If ``verifier`` (a :class:`harness.tap.GaiaVerifier`) is given, each cone is
    checked to contain at least ``n_rows`` sources. At 0.2 deg this is
    essentially always true, but a cheap check beats a confusing eval failure.
    """
    rng = random.Random(seed)
    out: list[ConeVariant] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 20 * n:
            raise RuntimeError("could not find enough satisfiable cones")
        ra, dec = random_cone_center(rng)
        if verifier is not None and verifier.count_in_cone(ra, dec, radius, limit=n_rows) < n_rows:
            continue
        idx = len(out)
        out.append(
            ConeVariant(
                id=f"gaia_cone-{seed}-{idx:03d}",
                ra=ra,
                dec=dec,
                radius=radius,
                n_rows=n_rows,
                seed=seed,
            )
        )
    return out


def write_variants(variants: list[ConeVariant], path: Path = DEFAULT_VARIANTS) -> None:
    with path.open("w") as fh:
        for v in variants:
            fh.write(json.dumps(asdict(v)) + "\n")


def load_variants(path: Path = DEFAULT_VARIANTS) -> list[ConeVariant]:
    return list(iter_variants(path))


def iter_variants(path: Path = DEFAULT_VARIANTS) -> Iterator[ConeVariant]:
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ConeVariant.from_dict(json.loads(line))


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--out", type=Path, default=DEFAULT_VARIANTS)
    ap.add_argument("--check", action="store_true", help="verify each cone is satisfiable via MAST TAP")
    args = ap.parse_args()

    verifier = None
    if args.check:
        from harness.tap import GaiaVerifier

        verifier = GaiaVerifier()
        if not verifier.healthcheck():
            raise SystemExit("MAST TAP healthcheck failed; refusing to generate unchecked variants with --check")

    variants = generate(args.n, seed=args.seed, radius=args.radius, n_rows=args.rows, verifier=verifier)
    write_variants(variants, args.out)
    print(f"wrote {len(variants)} variants to {args.out}")


if __name__ == "__main__":
    _main()
