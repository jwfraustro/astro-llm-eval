"""Shared skeleton for outcome scorers that verify a CSV the agent wrote.

Every outcome scorer does the same four things: pull ``/workspace/result.csv``
out of the sandbox, run the task's synchronous verifier off the event loop,
turn a service outage into an infra error (``NOANSWER``, never blamed on the
model), and record the self-report calibration alongside the verdict. The
task supplies the verifier, its variant class, and the exception its service
client raises; it keeps its own ``@scorer`` decorator so metrics stay per task.

Scoring policy:

* ``CORRECT``   - the verifier passed the file.
* ``INCORRECT`` - any verifiable failure (see each task's ``verify.py``).
* ``NOANSWER``  - the verifier itself could not reach its service.
  ``accuracy()`` counts this as 0, the conservative choice;
  ``infra_error_rate`` reports how many samples that affected, and
  ``scripts/rescore.py`` can re-verify them later from the CSV text stashed in
  the score metadata.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Protocol

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Metric, SampleScore, Score, Target, metric
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from harness.prompts import RESULT_PATH
from harness.selfreport import parse_self_report, self_report_agreement

# Mechanical failure reasons every CSV task can produce before its own
# verifier gets a look at the contents.
NO_OUTPUT_FILE = "no_output_file"
UNREADABLE_FILE = "unreadable_file"
INFRA_ERROR = "infra_error"

# Don't stash arbitrarily large files into the log.
MAX_STASH_CHARS = 200_000


class VerdictLike(Protocol):
    passed: bool
    reason: str | None
    detail: str
    infra_error: bool

    def as_metadata(self) -> dict: ...


def submitted(state: TaskState) -> bool:
    """Did the agent ever call ``submit``, or did it hit a limit first?"""
    return any(
        call.function == "submit"
        for m in state.messages
        if isinstance(m, ChatMessageAssistant)
        for call in (m.tool_calls or [])
    )


@metric
def never_submitted_rate() -> Metric:
    """Fraction of samples that hit a limit before calling submit."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        return sum(1 for s in scores if not (s.score.metadata or {}).get("submitted", True)) / len(scores)

    return compute


def build_score(verdict: VerdictLike, state: TaskState, csv_text: str | None, **extra_metadata: Any) -> Score:
    """Turn a verdict into an Inspect ``Score`` with the common metadata."""
    value = NOANSWER if verdict.infra_error else (CORRECT if verdict.passed else INCORRECT)
    submit_message = state.output.completion
    self_report = parse_self_report(submit_message)
    was_submitted = submitted(state)

    metadata = verdict.as_metadata()
    metadata["reason"] = verdict.reason or "pass"
    metadata.update(
        submit_message=submit_message,
        submitted=was_submitted,
        self_report=self_report,
        self_report_agreement=self_report_agreement(self_report, verdict),
        n_messages=len(state.messages),
    )
    metadata.update(extra_metadata)
    if csv_text is not None:
        metadata["csv_text"] = csv_text[:MAX_STASH_CHARS]

    suffix = f" [self-report: {self_report}]" if was_submitted else " [never submitted: hit a limit]"
    return Score(
        value=value,
        answer=submit_message,
        explanation=f"{metadata['reason']}: {verdict.detail}{suffix}",
        metadata=metadata,
    )


def csv_outcome_scorer(
    verify_csv_text: Callable[[str, Any, Any], VerdictLike],
    variant_cls: type,
    client: Any,
    service_error: type[Exception] | tuple[type[Exception], ...],
    *,
    verdict_cls: Callable[..., VerdictLike],
    result_path: str = RESULT_PATH,
    extra_metadata: Callable[[TaskState], dict[str, Any]] | None = None,
) -> Callable[[TaskState, Target], Any]:
    """Build the async ``score`` function for a CSV-verifying outcome scorer.

    ``verify_csv_text(text, variant, client)`` is the task's pure verifier; it
    may raise ``service_error`` when its service is unreachable. ``verdict_cls``
    is the task's ``Verdict`` dataclass, constructed positionally as
    ``(passed, reason, detail)`` for the pre-verification failures here.
    ``extra_metadata(state)`` adds task-specific keys to the score metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        variant = variant_cls.from_dict(state.metadata)
        extra = extra_metadata(state) if extra_metadata else {}

        # 1. Pull the file out of the sandbox.
        try:
            text = await sandbox().read_file(result_path)
        except FileNotFoundError:
            return build_score(verdict_cls(False, NO_OUTPUT_FILE, f"{result_path} does not exist"), state, None, **extra)
        except Exception as exc:  # IsADirectoryError, UnicodeDecodeError, PermissionError, ...
            return build_score(verdict_cls(False, UNREADABLE_FILE, f"{type(exc).__name__}: {exc}"), state, None, **extra)

        # 2. Verify. Service clients are synchronous; keep the event loop free.
        try:
            verdict = await asyncio.to_thread(verify_csv_text, text, variant, client)
        except service_error as exc:
            verdict = verdict_cls(False, INFRA_ERROR, str(exc), infra_error=True)
        return build_score(verdict, state, text, **extra)

    return score
