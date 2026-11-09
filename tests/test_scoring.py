"""Offline tests for the shared outcome-scorer skeleton in harness.scoring."""

from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from harness import scoring as S
from harness.selfreport import HONEST_FAILURE, INDETERMINATE, OVERCONFIDENT, TRUE_POSITIVE


@dataclass
class Verdict:
    passed: bool
    reason: str | None = None
    detail: str = ""
    infra_error: bool = False

    def as_metadata(self) -> dict:
        return {"passed": self.passed, "reason": self.reason, "detail": self.detail, "infra_error": self.infra_error}


def _state(completion: str, *, submitted: bool = True) -> TaskState:
    messages = [ChatMessageUser(content="do the thing")]
    if submitted:
        messages.append(
            ChatMessageAssistant(
                content="",
                tool_calls=[ToolCall(id="c1", function="submit", arguments={"answer": completion})],
            )
        )
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s",
        epoch=0,
        input="do the thing",
        messages=messages,
        output=ModelOutput.from_content("mockllm/model", completion),
        metadata={},
    )


def test_build_score_values_and_calibration():
    ok = S.build_score(Verdict(True, None, "fine"), _state("SUCCESS: done"), "a,b\n1,2\n")
    assert ok.value == CORRECT
    assert ok.metadata["reason"] == "pass"
    assert ok.metadata["self_report_agreement"] == TRUE_POSITIVE
    assert ok.metadata["csv_text"] == "a,b\n1,2\n"
    assert ok.metadata["submitted"] is True
    assert ok.explanation.startswith("pass: fine") and "[self-report: success]" in ok.explanation

    bad = S.build_score(Verdict(False, "too_few_rows", "0 rows"), _state("SUCCESS: done"), "a\n")
    assert bad.value == INCORRECT
    assert bad.metadata["reason"] == "too_few_rows"
    assert bad.metadata["self_report_agreement"] == OVERCONFIDENT

    honest = S.build_score(Verdict(False, S.NO_OUTPUT_FILE, "missing"), _state("FAIL: nope"), None)
    assert honest.metadata["self_report_agreement"] == HONEST_FAILURE
    assert "csv_text" not in honest.metadata

    infra = S.build_score(Verdict(False, S.INFRA_ERROR, "down", infra_error=True), _state("SUCCESS: done"), "a\n")
    assert infra.value == NOANSWER
    assert infra.metadata["self_report_agreement"] == INDETERMINATE


def test_build_score_extra_metadata_and_never_submitted():
    sc = S.build_score(Verdict(False, S.NO_OUTPUT_FILE, "missing"), _state("", submitted=False), None, negative_control=True)
    assert sc.metadata["submitted"] is False
    assert sc.metadata["negative_control"] is True
    assert "[never submitted: hit a limit]" in sc.explanation

    rate = S.never_submitted_rate()
    scores = [SampleScore(score=sc), SampleScore(score=Score(value=CORRECT, metadata={"submitted": True}))]
    assert rate(scores) == 0.5
    assert rate([]) == 0.0


def test_stash_is_truncated():
    big = "x" * (S.MAX_STASH_CHARS + 10)
    sc = S.build_score(Verdict(True), _state("SUCCESS: ok"), big)
    assert len(sc.metadata["csv_text"]) == S.MAX_STASH_CHARS
