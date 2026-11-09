"""Inspect scorer wrapping the pure verifier.

The read-file / verify / build-Score skeleton and the scoring policy
(CORRECT / INCORRECT / NOANSWER) live in ``harness.scoring``; this module only
binds them to the Gaia verifier and declares the metrics.

See ``harness.metrics`` for the caveat on metadata-based metrics with epochs > 1.
"""

from __future__ import annotations

from inspect_ai.scorer import accuracy, scorer, stderr

from harness.metrics import failure_reasons, infra_error_rate, self_report_calibration
from harness.prompts import RESULT_PATH
from harness.scoring import file_outcome_scorer, never_submitted_rate
from harness.tap import GaiaVerifier, TAPError
from tasks.gaia_cone.variants import ConeVariant
from tasks.gaia_cone.verify import Verdict, verify_csv_text


@scorer(metrics=[accuracy(), stderr(), infra_error_rate(), failure_reasons(), self_report_calibration(), never_submitted_rate()])
def tap_verify(result_path: str = RESULT_PATH, verifier: GaiaVerifier | None = None):
    """Verify the agent's CSV against MAST's Gaia DR3 TAP service."""
    return file_outcome_scorer(
        verify_csv_text,
        ConeVariant,
        verifier or GaiaVerifier(),
        TAPError,
        verdict_cls=Verdict,
        result_path=result_path,
    )
