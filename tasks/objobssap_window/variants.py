"""Seeded variants for the ObjObsSAP observability task.

Each variant is a random sky position and a random few-day MJD window. Every
fifth variant is a *negative control*: a far-southern declination that never
rises from Baltimore, so the correct answer is an empty table. Those are where
"guessed a bad FACILITY, got nothing, called it done" becomes distinguishable
from a correct empty answer.

With ``--check`` the generator also records the service's answer in
``expected`` so a later change to the service's observability model shows up
as ``truth_drift`` in the logs instead of silently rewriting history::

    python -m tasks.objobssap_window.variants --n 20 --seed 2026 --check
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from harness.geometry import random_sky_position
from harness.variants import Variant, variant_cli

HERE = Path(__file__).resolve().parent
DEFAULT_VARIANTS = HERE / "variants.jsonl"

# Baltimore is at ~39.3 N; nothing with dec < -50.7 ever rises. Stay well clear.
NEVER_RISES_DEC_MAX = -60.0
# Positive variants avoid the marginal band near the southern horizon limit so
# "a few seconds above the horizon" rounding never decides a pass.
POSITIVE_DEC_MIN = -40.0
# Circumpolar targets (dec > 90 - 39.3 = 50.7 from Baltimore) currently make the
# reference service return HTTP 500 (observed 2026-11-08). Stay below that until
# the service handles never-setting targets; then raise this to 90.
POSITIVE_DEC_MAX = 48.0

MJD_MIN, MJD_MAX = 61000.0, 61700.0  # roughly late 2025 through 2027
DURATION_DAYS = (3, 7)


@dataclass(frozen=True)
class WindowVariant(Variant):
    ra: float
    dec: float
    t0: float  # MJD
    t1: float  # MJD
    min_obs: int  # seconds
    negative_control: bool
    expected: list[list[float]] | None = field(default=None)  # [[t_start, t_stop], ...] from --check


def generate(n: int, *, seed: int, min_obs: int = 30, client=None, negative_every: int = 5) -> list[WindowVariant]:
    rng = random.Random(seed)
    out: list[WindowVariant] = []
    for idx in range(n):
        negative = negative_every > 0 and idx % negative_every == negative_every - 1
        band = (-90.0, NEVER_RISES_DEC_MAX) if negative else (POSITIVE_DEC_MIN, POSITIVE_DEC_MAX)
        ra, dec = random_sky_position(rng, dec_min=band[0], dec_max=band[1])
        ra, dec = round(ra, 4), round(dec, 4)
        t0 = round(rng.uniform(MJD_MIN, MJD_MAX), 1)
        t1 = round(t0 + rng.randint(*DURATION_DAYS), 1)

        expected = None
        if client is not None:
            intervals = client.query(ra, dec, t0, t1, min_obs=min_obs)
            expected = [[iv.t_start, iv.t_stop] for iv in intervals]
            if negative and expected:
                raise RuntimeError(f"negative control {idx} unexpectedly observable: {expected}")
            if not negative and not expected:
                raise RuntimeError(f"positive variant {idx} has no observability; adjust dec bounds")

        out.append(
            WindowVariant(
                id=f"objobssap_window-{seed}-{idx:03d}",
                seed=seed,
                ra=ra,
                dec=dec,
                t0=t0,
                t1=t1,
                min_obs=min_obs,
                negative_control=negative,
                expected=expected,
            )
        )
    return out


def load_variants(path: Path = DEFAULT_VARIANTS) -> list[WindowVariant]:
    return WindowVariant.load(path)


def _main() -> None:
    from harness.objobssap import ObjObsSAPClient

    def add_arguments(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--min-obs", type=int, default=30)

    variant_cli(
        __doc__,
        default_out=DEFAULT_VARIANTS,
        add_arguments=add_arguments,
        check_client=ObjObsSAPClient,
        check_help="record the service's answer for each variant",
        make=lambda args, client: generate(args.n, seed=args.seed, min_obs=args.min_obs, client=client),
        summary=lambda vs: f"{sum(v.negative_control for v in vs)} negative controls",
    )


if __name__ == "__main__":
    _main()
