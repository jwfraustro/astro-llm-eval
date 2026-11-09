"""Base class for the verifiers' verdicts.

Each task's ``verify.py`` subclasses :class:`Verdict` with its own extra
fields (ids that failed, boundary errors, columns seen, ...). The common part
is what ``harness.scoring`` and ``scripts/rescore.py`` rely on: pass/fail, a
mechanical failure reason, a one-line detail, and the infra-error flag that
turns a verdict into ``NOANSWER`` instead of ``INCORRECT``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# Mechanical failure reasons every file-output task can produce before its own
# verifier gets a look at the contents. Task modules import and re-use these.
NO_OUTPUT_FILE = "no_output_file"
UNREADABLE_FILE = "unreadable_file"
INFRA_ERROR = "infra_error"

# Lists in a verdict are capped in the log metadata; the full list is only
# needed interactively, and a 10k-row failure would otherwise be stored whole.
MAX_LIST_ITEMS = 20


@dataclass
class Verdict:
    passed: bool
    reason: str | None = None  # a failure-reason constant when not passed
    detail: str = ""
    n_rows: int = 0
    infra_error: bool = False

    def as_metadata(self) -> dict:
        """Every field, log-safe: ``reason`` is ``"pass"`` when passed, and list
        items are stringified (64-bit Gaia ids do not survive JSON as numbers)
        and capped at :data:`MAX_LIST_ITEMS`."""
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, list):
                value = [str(x) for x in value[:MAX_LIST_ITEMS]]
            out[f.name] = value
        out["reason"] = self.reason or "pass"
        return out
