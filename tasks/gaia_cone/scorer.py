"""Inspect scorer wrapping the pure verifier.

Scoring policy:

* ``CORRECT``   - all rows exist in Gaia DR3 and fall inside the cone.
* ``INCORRECT`` - any verifiable failure (see ``verify.py`` reasons).
* ``NOANSWER``  - the verifier itself could not reach MAST. ``accuracy()``
  counts this as 0, which is the conservative choice; ``infra_error_rate``
  reports how many samples that affected, and ``scripts/rescore.py`` can
  re-verify them later from the CSV text stashed in the score metadata.

See ``harness.metrics`` for the caveat on metadata-based metrics with epochs > 1.
"""

from __future__ import annotations

import asyncio

from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from harness.metrics import failure_reasons, infra_error_rate, self_report_calibration
from harness.selfreport import parse_self_report, self_report_agreement
from harness.tap import GaiaVerifier, TAPError
from tasks.gaia_cone.prompts import RESULT_PATH
from tasks.gaia_cone.variants import ConeVariant
from tasks.gaia_cone.verify import INFRA_ERROR, NO_OUTPUT_FILE, UNREADABLE_FILE, Verdict, verify_csv_text

# Don't stash arbitrarily large files into the log.
_MAX_STASH_CHARS = 200_000


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
