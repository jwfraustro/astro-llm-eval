"""Seeded-variant plumbing shared by every task.

A task's ``variants.py`` declares a frozen dataclass deriving from
:class:`Variant` and a ``generate(n, *, seed, ...)`` function. The jsonl
round trip, the tolerant ``from_dict`` (a jsonl written before an optional
field was added must still load), and the ``python -m tasks.<task>.variants``
command line live here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar

V = TypeVar("V", bound="Variant")


@dataclass(frozen=True)
class Variant:
    id: str
    seed: int

    @classmethod
    def from_dict(cls: type[V], d: dict) -> V:
        """Build from a dict with possibly extra or (for defaulted fields) missing keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def iter_file(cls: type[V], path: Path) -> Iterator[V]:
        with Path(path).open() as fh:
            for line in fh:
                if line.strip():
                    yield cls.from_dict(json.loads(line))

    @classmethod
    def load(cls: type[V], path: Path) -> list[V]:
        return list(cls.iter_file(path))


def write_variants(variants: Sequence[Variant], path: Path) -> None:
    with Path(path).open("w") as fh:
        for v in variants:
            fh.write(json.dumps(asdict(v)) + "\n")


def variant_cli(
    doc: str | None,
    *,
    default_out: Path,
    make: Callable[[argparse.Namespace, Any], Sequence[Variant]],
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    check_client: Callable[[], Any] | None = None,
    check_help: str = "validate each variant against the live service",
    summary: Callable[[Sequence[Variant]], str] | None = None,
) -> None:
    """The ``python -m tasks.<task>.variants`` entry point.

    Always offers ``--n``, ``--seed``, ``--out`` and, when ``check_client`` is
    given, ``--check``: the client is built, health-checked, and passed to
    ``make(args, client)``; without ``--check`` it receives ``None``.
    """
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=default_out)
    if check_client is not None:
        ap.add_argument("--check", action="store_true", help=check_help)
    if add_arguments is not None:
        add_arguments(ap)
    args = ap.parse_args()

    client = None
    if check_client is not None and args.check:
        client = check_client()
        if not client.healthcheck():
            raise SystemExit("service healthcheck failed; refusing to generate unchecked variants with --check")

    variants = make(args, client)
    write_variants(variants, args.out)
    extra = f" ({summary(variants)})" if summary else ""
    print(f"wrote {len(variants)} variants{extra} to {args.out}")
