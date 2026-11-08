"""Summarise Inspect logs into a markdown table.

Accuracy is recomputed from per-sample scores rather than read from the log
header, so logs that were patched by ``rescore.py`` are reported correctly.

    python scripts/summarize.py logs/ > results/summary.md
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Outcome scorers, one per task; the first one present in a log is used.
PRIMARY_SCORERS = ("tap_verify", "interval_verify")
PROCESS_SCORER = "protocol_use"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarise(log: EvalLog) -> dict | None:
    if not log.samples:
        return None
    present = set().union(*((s.scores or {}).keys() for s in log.samples))
    primary = next((p for p in PRIMARY_SCORERS if p in present), None)
    if primary is None:
        return None

    n = passed = infra = used_protocol = local_compute = 0
    has_process = PROCESS_SCORER in present
    reasons: Counter[str] = Counter()
    calibration: Counter[str] = Counter()
    msgs: list[int] = []
    tokens = 0
    for s in log.samples:
        sc = (s.scores or {}).get(primary)
        if sc is None:
            continue
        n += 1
        proc = (s.scores or {}).get(PROCESS_SCORER)
        if proc is not None:
            used_protocol += proc.value == "C"
            local_compute += bool((proc.metadata or {}).get("local_computation"))
        meta = sc.metadata or {}
        if sc.value == "C":
            passed += 1
        elif meta.get("infra_error"):
            infra += 1
            reasons["infra_error"] += 1
        else:
            reasons[meta.get("reason") or "unknown"] += 1
        calibration[meta.get("self_report_agreement", "unlabeled")] += 1
        msgs.append(meta.get("n_messages") or len(s.messages))
        if s.model_usage:
            tokens += sum(u.total_tokens for u in s.model_usage.values())
    if n == 0:
        return None
    lo, hi = wilson(passed, n)
    return {
        "model": log.eval.model,
        "task": log.eval.task,
        "tier": (log.eval.metadata or {}).get("tool_tier", "?"),
        "n": n,
        "pass": passed,
        "acc": passed / n,
        "ci": f"{lo:.2f}-{hi:.2f}",
        "infra": infra,
        "mean_msgs": sum(msgs) / len(msgs) if msgs else 0.0,
        "tokens": tokens,
        "reasons": ", ".join(f"{k}:{v}" for k, v in reasons.most_common()) or "-",
        # The cell that matters: claimed SUCCESS, verifiably wrong.
        "overconfident": calibration.get("overconfident", 0),
        "honest_fail": calibration.get("honest_failure", 0),
        "unlabeled": calibration.get("unlabeled", 0),
        # Process scorer (ObjObsSAP task only): '-' when the task has none.
        "protocol": f"{used_protocol}/{n}" if has_process else "-",
        "local": str(local_compute) if has_process else "-",
        "started": (log.eval.created or "")[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", nargs="?", default="logs")
    args = ap.parse_args()

    rows = []
    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info.name)
        row = summarise(log)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r["task"], r["tier"], -r["acc"], r["model"]))

    print(
        "| task | model | tier | n | pass | acc | 95% CI | infra | used VO | local calc | "
        "overconf. | honest fail | unlabeled | msgs | tokens | failure reasons | date |"
    )
    print("|---|" * 17)
    for r in rows:
        print(
            f"| {r['task']} | {r['model']} | {r['tier']} | {r['n']} | {r['pass']} | {r['acc']:.2f} | {r['ci']} | "
            f"{r['infra']} | {r['protocol']} | {r['local']} | "
            f"{r['overconfident']} | {r['honest_fail']} | {r['unlabeled']} | "
            f"{r['mean_msgs']:.1f} | {r['tokens']:,} | {r['reasons']} | {r['started']} |"
        )


if __name__ == "__main__":
    main()
