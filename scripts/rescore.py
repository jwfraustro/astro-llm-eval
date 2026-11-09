"""Re-verify samples that hit an infra error, using the result text stashed in
their score metadata. Rewrites the log in place (or to --out).

    python scripts/rescore.py logs/2026-11-08T12-00-00_gaia-cone_abc.eval

The task is inferred from the scorers present in the log (see
``tasks.registry``). Only samples whose outcome score has ``infra_error: true``
are touched; model outputs are never re-generated. Run ``summarize.py``
afterwards; it recomputes accuracy from sample scores and ignores the stale
log header.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from inspect_ai.scorer import CORRECT, INCORRECT  # noqa: E402

from harness.selfreport import self_report_agreement  # noqa: E402
from tasks import registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", help="write here instead of overwriting")
    ap.add_argument("--task", help="override the task inferred from the log's scorers")
    args = ap.parse_args()

    log = read_eval_log(args.log)
    samples = log.samples or []
    if args.task:
        info = registry.get(args.task)
    else:
        present = set().union(*((s.scores or {}).keys() for s in samples)) if samples else set()
        info = registry.for_scorers(present)
        if info is None:
            raise SystemExit(f"no known outcome scorer among {sorted(present)}; pass --task")

    client = info.client_factory()
    if not client.healthcheck():
        raise SystemExit(f"{info.service_label} still unreachable; nothing to do")

    fixed = still_broken = 0
    for sample in samples:
        score = (sample.scores or {}).get(info.outcome_scorer)
        if not score or not (score.metadata or {}).get("infra_error"):
            continue
        # "csv_text" is the key older logs used.
        text = score.metadata.get("result_text", score.metadata.get("csv_text"))
        if text is None:
            continue
        variant = info.variant_cls.from_dict(sample.metadata)
        try:
            verdict = info.verify(text, variant, client)
        except info.service_error as exc:
            still_broken += 1
            score.explanation = f"infra_error (rescore): {exc}"
            continue
        score.value = CORRECT if verdict.passed else INCORRECT
        score.explanation = f"{verdict.reason or 'pass'} (rescored): {verdict.detail}"
        score.metadata.update(verdict.as_metadata())
        score.metadata["reason"] = verdict.reason or "pass"
        score.metadata["self_report_agreement"] = self_report_agreement(
            score.metadata.get("self_report", "unlabeled"), verdict
        )
        score.metadata["rescored"] = True
        fixed += 1

    write_eval_log(log, args.out or args.log)
    print(f"[{info.name}] rescored {fixed} sample(s); {still_broken} still failing on infra")


if __name__ == "__main__":
    main()
