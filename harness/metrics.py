"""Metadata-driven Inspect metrics shared by task scorers.

Caveat that applies to all of these: Inspect reduces epochs *before* metrics
run, and the reducer keeps only the first epoch's metadata. With ``--epochs 1``
they are exact; with more epochs they describe epoch 0. ``scripts/summarize.py``
reads every epoch and is the source of truth.
"""

from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, metric


def histogram(scores: list[SampleScore], key: str, default: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for s in scores:
        label = str((s.score.metadata or {}).get(key) or default)
        counts[label] = counts.get(label, 0) + 1
    total = max(len(scores), 1)
    return {k: v / total for k, v in sorted(counts.items())}


def flag_rate(scores: list[SampleScore], key: str) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if (s.score.metadata or {}).get(key)) / len(scores)


@metric
def infra_error_rate() -> Metric:
    """Fraction of samples where the verifier, not the model, failed."""

    def compute(scores: list[SampleScore]) -> float:
        return flag_rate(scores, "infra_error")

    return compute


@metric
def failure_reasons() -> Metric:
    """Histogram of mechanical failure reasons ('pass' for verified samples)."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return histogram(scores, "reason", default="pass")

    return compute


@metric
def self_report_calibration() -> Metric:
    """Histogram of self-report vs verifier agreement (see harness.selfreport)."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return histogram(scores, "self_report_agreement", default="unlabeled")

    return compute
