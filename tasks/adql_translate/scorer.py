"""Inspect scorer for the ADQL translation task.

The read-file / verify / build-Score skeleton lives in ``harness.scoring``;
this module binds it to the MAST TAP client and declares the metrics.
"""

from __future__ import annotations

from inspect_ai.scorer import accuracy, scorer, stderr

from harness.metrics import failure_reasons, infra_error_rate, self_report_calibration
from harness.scoring import file_outcome_scorer, never_submitted_rate
from harness.tap import GaiaVerifier, TAPError
from tasks.adql_translate.prompts import RESULT_PATH
from tasks.adql_translate.variants import TranslateVariant
from tasks.adql_translate.verify import Verdict, verify_query_text


@scorer(metrics=[accuracy(), stderr(), infra_error_rate(), failure_reasons(), self_report_calibration(), never_submitted_rate()])
def adql_verify(result_path: str = RESULT_PATH, client: GaiaVerifier | None = None):
    """Run the agent's ADQL and the gold ADQL against MAST TAP; compare results."""
    return file_outcome_scorer(
        verify_query_text,
        TranslateVariant,
        client or GaiaVerifier(),
        TAPError,
        verdict_cls=Verdict,
        result_path=result_path,
    )
