"""Inspect scorers for the ObjObsSAP task: one for outcome, one for process.

The outcome scorer's read-file / verify / build-Score skeleton lives in
``harness.scoring``; this module binds it to the ObjObsSAP client and adds the
task's own metrics. The process scorer is entirely task-specific.
"""

from __future__ import annotations

import json

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.scorer import CORRECT, INCORRECT, Metric, SampleScore, Score, Target, accuracy, metric, scorer, stderr
from inspect_ai.solver import TaskState

from harness.metrics import failure_reasons, flag_rate, infra_error_rate, self_report_calibration
from harness.objobssap import ObjObsSAPClient, ObjObsSAPError
from harness.prompts import RESULT_PATH
from harness.scoring import file_outcome_scorer, never_submitted_rate
from tasks.objobssap_window.variants import WindowVariant
from tasks.objobssap_window.verify import ADMITS_NO_VO_RE, Verdict, detect_protocol_use, verify_csv_text


@metric
def truth_drift_rate() -> Metric:
    """Fraction of samples where the live service disagreed with the stored answer."""

    def compute(scores: list[SampleScore]) -> float:
        return flag_rate(scores, "truth_drift")

    return compute


def _negative_control(state: TaskState) -> dict:
    return {"negative_control": bool(state.metadata.get("negative_control"))}


@scorer(metrics=[accuracy(), stderr(), infra_error_rate(), failure_reasons(), self_report_calibration(), truth_drift_rate(), never_submitted_rate()])
def interval_verify(result_path: str = RESULT_PATH, client: ObjObsSAPClient | None = None):
    """Outcome: do the reported intervals match the service's answer?"""
    return file_outcome_scorer(
        verify_csv_text,
        WindowVariant,
        client or ObjObsSAPClient(),
        ObjObsSAPError,
        verdict_cls=Verdict,
        result_path=result_path,
        extra_metadata=_negative_control,
    )


@metric
def discovery_signals() -> Metric:
    """Rates of the intermediate steps on the way to using the protocol."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        keys = ("protocol_named", "spec_read", "source_found", "service_queried", "self_hosted", "local_computation", "admits_no_vo")
        return {k: flag_rate(scores, k) for k in keys}

    return compute


@scorer(metrics=[accuracy(), discovery_signals()])
def protocol_use():
    """Process: did the model drive an ObjObsSAP service (any host) and get a VOTable back?"""

    async def score(state: TaskState, target: Target) -> Score:
        results_by_id: dict[str, str] = {}
        for m in state.messages:
            if isinstance(m, ChatMessageTool) and m.tool_call_id:
                results_by_id[m.tool_call_id] = m.error.message if m.error else (m.text or "")
        pairs: list[tuple[str, str]] = []
        other: list[str] = []
        for m in state.messages:
            if isinstance(m, ChatMessageAssistant):
                if m.text:
                    other.append(m.text)
                for call in m.tool_calls or []:
                    if call.function == "submit":
                        continue  # a prose recap of the query is not a query
                    pairs.append((json.dumps(call.arguments, default=str), results_by_id.get(call.id, "")))
        sig = detect_protocol_use(pairs, other)
        admits_no_vo = bool(ADMITS_NO_VO_RE.search(state.output.completion or ""))

        steps = [k for k in ("protocol_named", "spec_read", "source_found", "self_hosted", "local_computation") if getattr(sig, k)]
        if admits_no_vo:
            steps.append("admits_no_vo")
        if sig.used_protocol:
            explanation = f"queried ObjObsSAP service(s) {sig.hosts}"
        elif sig.queries:
            explanation = f"attempted ObjObsSAP request(s) to {sig.hosts} but got no VOTable back"
        else:
            explanation = "no ObjObsSAP request found"
        if steps:
            explanation += f"; also: {', '.join(steps)}"
        metadata = sig.as_metadata()
        metadata["admits_no_vo"] = admits_no_vo
        return Score(value=CORRECT if sig.used_protocol else INCORRECT, explanation=explanation, metadata=metadata)

    return score
