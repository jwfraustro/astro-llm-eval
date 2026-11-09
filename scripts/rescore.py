"""Re-verify samples that hit an infra error, using the CSV text stashed in
their score metadata. Rewrites the log in place (or to --out).

    python scripts/rescore.py logs/2026-11-08T12-00-00_gaia-cone_abc.eval

Only samples whose ``tap_verify`` score has ``infra_error: true`` are touched;
model outputs are never re-generated. Run ``summarize.py`` afterwards; it
recomputes accuracy from sample scores and ignores the stale log header.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER  # noqa: E402

from harness.tap import GaiaVerifier, TAPError  # noqa: E402
from tasks.gaia_cone.variants import ConeVariant  # noqa: E402
from harness.selfreport import self_report_agreement  # noqa: E402
from tasks.gaia_cone.verify import verify_csv_text  # noqa: E402

SCORER = "tap_verify"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", help="write here instead of overwriting")
    args = ap.parse_args()

    log = read_eval_log(args.log)
    verifier = GaiaVerifier()
    if not verifier.healthcheck():
        raise SystemExit("MAST TAP still unreachable; nothing to do")

    fixed = still_broken = 0
    for sample in log.samples or []:
        score = (sample.scores or {}).get(SCORER)
        if not score or not (score.metadata or {}).get("infra_error"):
            continue
        csv_text = score.metadata.get("csv_text")
        if csv_text is None:
            continue
        variant = ConeVariant.from_dict(sample.metadata)
        try:
            verdict = verify_csv_text(csv_text, variant, verifier)
        except TAPError as exc:
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
    print(f"rescored {fixed} sample(s); {still_broken} still failing on infra; NOANSWER={NOANSWER!r} entries resolved")


if __name__ == "__main__":
    main()
