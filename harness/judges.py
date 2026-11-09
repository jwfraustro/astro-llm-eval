"""LLM-judged failure taxonomy over the agent transcript.

The verifier tells you *that* a trial failed; this tells you *how*. It is a
secondary scorer: cheap, model-graded, and meant to be spot-checked by a human
against the transcript in ``inspect view``.

Pick the judge with a model role so it is independent of the model under test::

    inspect eval ... --model-role judge=anthropic/claude-haiku-5-5
"""

from __future__ import annotations

import json
import re

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import TaskState

FAILURE_LABELS: dict[str, str] = {
    "success": "The agent produced the requested output through a real query to a real service.",
    "hallucinated_endpoint": "Invented or malformed service URL (wrong host, wrong path, nonexistent API).",
    "wrong_table_or_column": "Right service, but a nonexistent table, schema prefix, or column name.",
    "adql_error": "Query language mistakes, e.g. SQL LIMIT instead of ADQL TOP, bad geometry functions.",
    "library_misuse": "Wrong or outdated astroquery/pyvo/astropy API usage that was never corrected.",
    "environment_error": "Stuck on pip, network, filesystem, or shell problems unrelated to astronomy.",
    "output_contract": "Got the data but wrote the wrong file, columns, format, or row count.",
    "gave_up": "Stopped or submitted without a result after hitting errors.",
    "fabricated_output": "Wrote a result file without any successful data retrieval (made-up rows).",
    "ran_out_of_budget": "Was making progress but hit the message/time limit.",
    "other": "Does not fit any label above; explain in evidence.",
}

DEFAULT_TASK_DESCRIPTION = "complete an astronomy data-access task"

_JUDGE_PROMPT = """You are auditing a transcript in which an AI agent was asked to {task_description}
using tools (bash, python, possibly web search). Classify the outcome with
exactly one label from this list:

{labels}

Respond with JSON only, on one line:
{{"label": "<label>", "evidence": "<one or two sentences quoting the decisive moment>"}}

Transcript:
<transcript>
{transcript}
</transcript>
"""


def render_transcript(messages: list[ChatMessage], *, max_chars: int = 60_000, tool_output_chars: int = 1500) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, ChatMessageUser):
            parts.append(f"[user]\n{m.text}")
        elif isinstance(m, ChatMessageAssistant):
            if m.text:
                parts.append(f"[assistant]\n{m.text}")
            for call in m.tool_calls or []:
                args = json.dumps(call.arguments, default=str)
                parts.append(f"[tool_call {call.function}]\n{args[:tool_output_chars]}")
        elif isinstance(m, ChatMessageTool):
            body = m.error.message if m.error else m.text
            parts.append(f"[tool_result {m.function or ''}]\n{(body or '')[:tool_output_chars]}")
        # System messages are omitted; the judge does not need the agent's instructions.
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        head, tail = text[: max_chars // 2], text[-max_chars // 2 :]
        text = f"{head}\n\n[... {len(text) - max_chars} chars elided ...]\n\n{tail}"
    return text


@metric
def label_distribution() -> Metric:
    def compute(scores: list[SampleScore]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for s in scores:
            counts[str(s.score.value)] = counts.get(str(s.score.value), 0) + 1
        total = max(len(scores), 1)
        return {k: v / total for k, v in sorted(counts.items())}

    return compute


@scorer(metrics=[label_distribution()])
def failure_taxonomy(
    judge_model: str | None = None,
    extra_labels: dict[str, str] | None = None,
    task_description: str = DEFAULT_TASK_DESCRIPTION,
):
    """Classify the transcript.

    ``extra_labels`` adds task-specific labels (e.g.
    ``tasks.objobssap_window.task.JUDGE_LABELS``) to the generic set.
    ``task_description`` completes "the agent was asked to ..." in the judge
    prompt, e.g. ``"fetch Gaia DR3 catalog rows"``."""
    all_labels = {**FAILURE_LABELS, **(extra_labels or {})}

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(judge_model) if judge_model else get_model(role="judge")
        labels = "\n".join(f"- {k}: {v}" for k, v in all_labels.items())
        prompt = _JUDGE_PROMPT.format(task_description=task_description, labels=labels, transcript=render_transcript(state.messages))
        result = await judge.generate(prompt)
        label, evidence = _parse_judgement(result.completion, all_labels)
        return Score(value=label, explanation=evidence, metadata={"judge": str(judge.name), "raw": result.completion[:2000]})

    return score


def _parse_judgement(text: str, labels: dict[str, str] | None = None) -> tuple[str, str]:
    labels = labels or FAILURE_LABELS
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = str(obj.get("label", "other"))
            if label not in labels:
                label = "other"
            return label, str(obj.get("evidence", ""))
        except json.JSONDecodeError:
            pass
    return "other", f"unparseable judge output: {text[:200]}"
