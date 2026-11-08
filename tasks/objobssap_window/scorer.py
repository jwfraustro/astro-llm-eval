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


@metric
def truth_drift_rate() -> Metric:
    """Fraction of samples where the live service disagreed with the stored answer."""

    def compute(scores: list[SampleScore]) -> float:
        return flag_rate(scores, "truth_drift")

    return compute


@scorer(metrics=[accuracy(), stderr(), infra_error_rate(), failure_reasons(), self_report_calibration(), truth_drift_rate()])
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
    metadata = verdict.as_metadata()
    metadata.update(
        submit_message=submit_message,
        self_report=self_report,
        self_report_agreement=self_report_agreement(self_report, verdict),
        n_messages=len(state.messages),
        negative_control=bool(state.metadata.get("negative_control")),
    )
    if csv_text is not None:
        metadata["csv_text"] = csv_text[:_MAX_STASH_CHARS]
    return Score(
        value=value,
        answer=submit_message,
        explanation=f"{metadata['reason']}: {verdict.detail} [self-report: {self_report}]",
        metadata=metadata,
    )


@metric
def local_computation_rate() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return flag_rate(scores, "local_computation")

    return compute


@scorer(metrics=[accuracy(), local_computation_rate()])
def protocol_use():
    """Process: is there evidence in the transcript that an ObjObsSAP service was driven?"""

    async def score(state: TaskState, target: Target) -> Score:
        calls: list[str] = []
        results: list[str] = []
        other: list[str] = []
        for m in state.messages:
            if isinstance(m, ChatMessageAssistant):
                if m.text:
                    other.append(m.text)
                for call in m.tool_calls or []:
                    calls.append(json.dumps(call.arguments, default=str))
            elif isinstance(m, ChatMessageTool):
                results.append(m.error.message if m.error else (m.text or ""))
        sig = detect_protocol_use(calls, results, other)
        return Score(
            value=CORRECT if sig.used_protocol else INCORRECT,
            explanation=("used ObjObsSAP" if sig.used_protocol else "no ObjObsSAP service call found")
            + (" (also computed locally)" if sig.local_computation else ""),
            metadata=sig.as_metadata(),
        )

    return score
