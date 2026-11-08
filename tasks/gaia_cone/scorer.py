"""Inspect scorer wrapping the pure verifier.

Scoring policy:

* ``CORRECT``   - all rows exist in Gaia DR3 and fall inside the cone.
* ``INCORRECT`` - any verifiable failure (see ``verify.py`` reasons).
* ``NOANSWER``  - the verifier itself could not reach MAST. ``accuracy()``
  counts this as 0, which is the conservative choice; ``infra_error_rate``
  reports how many samples that affected, and ``scripts/rescore.py`` can
  re-verify them later from the CSV text stashed in the score metadata.

A caveat on the metadata-based metrics (``infra_error_rate``,
``failure_reasons``, ``self_report_calibration``): Inspect reduces epochs
*before* metrics run, and the reducer keeps only the first epoch's metadata.
With ``--epochs 1`` they are exact; with more epochs they describe epoch 0.
``scripts/summarize.py`` reads every epoch and is the source of truth.
"""

from __future__ import annotations

import asyncio

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from harness.tap import GaiaVerifier, TAPError
from tasks.gaia_cone.prompts import RESULT_PATH
from tasks.gaia_cone.variants import ConeVariant
from tasks.gaia_cone.verify import (
    INFRA_ERROR,
    NO_OUTPUT_FILE,
    UNREADABLE_FILE,
    Verdict,
    parse_self_report,
    self_report_agreement,
    verify_csv_text,
)

# Don't stash arbitrarily large files into the log.
_MAX_STASH_CHARS = 200_000


def _histogram(scores: list[SampleScore], key: str, default: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for s in scores:
        label = str((s.score.metadata or {}).get(key) or default)
        counts[label] = counts.get(label, 0) + 1
    total = max(len(scores), 1)
    return {k: v / total for k, v in sorted(counts.items())}


@metric
def infra_error_rate() -> Metric:
    """Fraction of samples where the verifier, not the model, failed."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        return sum(1 for s in scores if (s.score.metadata or {}).get("infra_error")) / len(scores)

    return compute


@metric
def failure_reasons() -> Metric:
    """Histogram of mechanical failure reasons ('pass' for verified samples)."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return _histogram(scores, "reason", default="pass")

    return compute


@metric
def self_report_calibration() -> Metric:
    """Histogram of self-report vs verifier agreement (see verify.py)."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return _histogram(scores, "self_report_agreement", default="unlabeled")

    return compute


@scorer(metrics=[accuracy(), stderr(), infra_error_rate(), failure_reasons(), self_report_calibration()])
def tap_verify(result_path: str = RESULT_PATH, verifier: GaiaVerifier | None = None):
    """Verify the agent's CSV against MAST's Gaia DR3 TAP service."""
    v = verifier or GaiaVerifier()

    async def score(state: TaskState, target: Target) -> Score:
        variant = ConeVariant.from_dict(state.metadata)

        # 1. Pull the file out of the sandbox.
        try:
            text = await sandbox().read_file(result_path)
        except FileNotFoundError:
            return _score(Verdict(False, NO_OUTPUT_FILE, f"{result_path} does not exist"), state, None)
        except Exception as exc:  # IsADirectoryError, UnicodeDecodeError, PermissionError, ...
            return _score(Verdict(False, UNREADABLE_FILE, f"{type(exc).__name__}: {exc}"), state, None)

        # 2. Verify. The TAP client is synchronous; keep the event loop free.
        try:
            verdict = await asyncio.to_thread(verify_csv_text, text, variant, v)
        except TAPError as exc:
            verdict = Verdict(False, INFRA_ERROR, str(exc), infra_error=True)
        return _score(verdict, state, text)

    return score


def _score(verdict: Verdict, state: TaskState, csv_text: str | None) -> Score:
    if verdict.infra_error:
        value = NOANSWER
    else:
        value = CORRECT if verdict.passed else INCORRECT

    submit_message = state.output.completion
    self_report = parse_self_report(submit_message)

    metadata = verdict.as_metadata()
    metadata["reason"] = verdict.reason or "pass"
    metadata["submit_message"] = submit_message
    metadata["self_report"] = self_report
    metadata["self_report_agreement"] = self_report_agreement(self_report, verdict)
    metadata["n_messages"] = len(state.messages)
    if csv_text is not None:
        metadata["csv_text"] = csv_text[:_MAX_STASH_CHARS]
    return Score(
        value=value,
        answer=submit_message,
        explanation=f"{metadata['reason']}: {verdict.detail} [self-report: {self_report}]",
        metadata=metadata,
    )
