"""One place that knows what every task is made of.

The ``@task`` functions are discovered by Inspect from each ``task.py``; this
registry is for the trusted-side scripts that need to know, per task, which
scorer holds the verdict, how to rebuild its variant, which client its
verifier needs and which exception that client raises when the *service*
rather than the model failed. Scripts should never grow a per-task
``if task == ...`` table again; add an entry here instead.

Command line::

    python -m tasks.registry list
    python -m tasks.registry healthcheck gaia_cone
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from harness.objobssap import ObjObsSAPClient, ObjObsSAPError
from harness.tap import GaiaVerifier, TAPError
from tasks.adql_translate import verify as adql_verify_mod
from tasks.adql_translate.variants import TranslateVariant
from tasks.gaia_cone import verify as gaia_verify_mod
from tasks.gaia_cone.variants import ConeVariant
from tasks.objobssap_window import verify as objobssap_verify_mod
from tasks.objobssap_window.variants import WindowVariant


@dataclass(frozen=True)
class TaskInfo:
    name: str  # also the @task function name and the directory under tasks/
    outcome_scorer: str  # scorer name holding CORRECT/INCORRECT/NOANSWER
    variant_cls: type  # has from_dict(dict)
    verify: Callable[[str, Any, Any], Any]  # (result_text, variant, client) -> Verdict
    client_factory: Callable[[], Any]  # the verifier's service client; has healthcheck()
    service_error: type[Exception]  # raised by verify() when the service, not the model, failed
    service_label: str  # for humans, in preflight output
    process_scorer: str | None = None  # optional process-level scorer name

    @property
    def directory(self) -> Path:
        return Path(__file__).resolve().parent / self.name

    @property
    def variants_file(self) -> Path:
        return self.directory / "variants.jsonl"


TASKS: dict[str, TaskInfo] = {
    t.name: t
    for t in (
        TaskInfo(
            name="gaia_cone",
            outcome_scorer="tap_verify",
            variant_cls=ConeVariant,
            verify=gaia_verify_mod.verify_csv_text,
            client_factory=GaiaVerifier,
            service_error=TAPError,
            service_label="MAST Gaia DR3 TAP",
        ),
        TaskInfo(
            name="objobssap_window",
            outcome_scorer="interval_verify",
            process_scorer="protocol_use",
            variant_cls=WindowVariant,
            verify=objobssap_verify_mod.verify_csv_text,
            client_factory=ObjObsSAPClient,
            service_error=ObjObsSAPError,
            service_label="vo-labs.dev ObjObsSAP",
        ),
        TaskInfo(
            name="adql_translate",
            outcome_scorer="adql_verify",
            variant_cls=TranslateVariant,
            verify=adql_verify_mod.verify_query_text,
            client_factory=GaiaVerifier,
            service_error=TAPError,
            service_label="MAST Gaia DR3 TAP",
        ),
    )
}

OUTCOME_SCORERS: tuple[str, ...] = tuple(t.outcome_scorer for t in TASKS.values())
PROCESS_SCORERS: tuple[str, ...] = tuple(t.process_scorer for t in TASKS.values() if t.process_scorer)


def get(name: str) -> TaskInfo:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}") from None


def for_scorers(scorer_names: set[str] | frozenset[str]) -> TaskInfo | None:
    """The task whose outcome scorer appears among ``scorer_names`` (from a log)."""
    return next((t for t in TASKS.values() if t.outcome_scorer in scorer_names), None)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the task names, one per line")
    hc = sub.add_parser("healthcheck", help="exit 0 if the task's verifier service answers")
    hc.add_argument("task")
    args = ap.parse_args()

    if args.cmd == "list":
        for name in TASKS:
            print(name)
        return
    info = get(args.task)
    ok = info.client_factory().healthcheck()
    print(f"{info.service_label} healthcheck:", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()
