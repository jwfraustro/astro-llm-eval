"""Inspect scorers for the ObjObsSAP task: one for outcome, one for process."""

from __future__ import annotations

import asyncio
import json

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Metric, SampleScore, Score, Target, accuracy, metric, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from harness.metrics import failure_reasons, flag_rate, infra_error_rate, self_report_calibration
from harness.objobssap import ObjObsSAPClient, ObjObsSAPError
from harness.selfreport import parse_self_report, self_report_agreement
from tasks.objobssap_window.prompts import RESULT_PATH
from tasks.objobssap_window.variants import WindowVariant
from tasks.objobssap_window.verify import INFRA_ERROR, NO_OUTPUT_FILE, UNREADABLE_FILE, Verdict, detect_protocol_use, verify_csv_text

_MAX_STASH_CHARS = 200_000


def _submitted(state: TaskState) -> bool:
    return any(
        call.function == "submit"
        for m in state.messages
        if isinstance(m, ChatMessageAssistant)
        for call in (m.tool_calls or [])
    )


@metric
def truth_drift_rate() -> Metric:
    """Fraction of samples where the live service disagreed with the stored answer."""

    def compute(scores: list[SampleScore]) -> float:
        return flag_rate(scores, "truth_drift")

    return compute


@metric
def never_submitted_rate() -> Metric:
    """Fraction of samples that hit a limit before calling submit."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        return sum(1 for s in scores if not (s.score.metadata or {}).get("submitted", True)) / len(scores)

    return compute


@scorer(metrics=[accuracy(), stderr(), infra_error_rate(), failure_reasons(), self_report_calibration(), truth_drift_rate(), never_submitted_rate()])
def interval_verify(result_path: str = RESULT_PATH, client: ObjObsSAPClient | None = None):
    """Outcome: do the reported intervals match the service's answer?"""
    c = client or ObjObsSAPClient()

    async def score(state: TaskState, target: Target) -> Score:
        variant = WindowVariant.from_dict(state.metadata)
        try:
            text = await sandbox().read_file(result_path)
        except FileNotFoundError:
            return _score(Verdict(False, NO_OUTPUT_FILE, f"{result_path} does not exist"), state, None)
        except Exception as exc:
            return _score(Verdict(False, UNREADABLE_FILE, f"{type(exc).__name__}: {exc}"), state, None)

        try:
            verdict = await asyncio.to_thread(verify_csv_text, text, variant, c)
        except ObjObsSAPError as exc:
            verdict = Verdict(False, INFRA_ERROR, str(exc), infra_error=True)
        return _score(verdict, state, text)

    return score


def _score(verdict: Verdict, state: TaskState, csv_text: str | None) -> Score:
    value = NOANSWER if verdict.infra_error else (CORRECT if verdict.passed else INCORRECT)
    submit_message = state.output.completion
    self_report = parse_self_report(submit_message)
    submitted = _submitted(state)
    metadata = verdict.as_metadata()
    metadata.update(
        submit_message=submit_message,
        submitted=submitted,
        self_report=self_report,
        self_report_agreement=self_report_agreement(self_report, verdict),
        n_messages=len(state.messages),
        negative_control=bool(state.metadata.get("negative_control")),
    )
    if csv_text is not None:
        metadata["csv_text"] = csv_text[:_MAX_STASH_CHARS]
    suffix = f" [self-report: {self_report}]" if submitted else " [never submitted: hit a limit]"
    return Score(value=value, answer=submit_message, explanation=f"{metadata['reason']}: {verdict.detail}{suffix}", metadata=metadata)


@metric
def discovery_signals() -> Metric:
    """Rates of the intermediate steps on the way to using the protocol."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        keys = ("protocol_named", "spec_read", "source_found", "service_queried", "self_hosted", "local_computation")
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
                    pairs.append((json.dumps(call.arguments, default=str), results_by_id.get(call.id, "")))
        sig = detect_protocol_use(pairs, other)

        steps = [k for k in ("protocol_named", "spec_read", "source_found", "self_hosted", "local_computation") if getattr(sig, k)]
        if sig.used_protocol:
            explanation = f"queried ObjObsSAP service(s) {sig.hosts}"
        elif sig.queries:
            explanation = f"attempted ObjObsSAP request(s) to {sig.hosts} but got no VOTable back"
        else:
            explanation = "no ObjObsSAP request found"
        if steps:
            explanation += f"; also: {', '.join(steps)}"
        return Score(value=CORRECT if sig.used_protocol else INCORRECT, explanation=explanation, metadata=sig.as_metadata())

    return score
