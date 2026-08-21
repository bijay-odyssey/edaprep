"""The journal: an append-only record of everything that actually happened.

The plan says what was *decided*; the journal says what was *done*, with measured
counts.  Keeping them separate matters because they can disagree -- a plan may decide
to clip ``income`` and then find nothing above the fence in a particular transform.

Notebook practice persisted ``processed_train.csv`` with no record of how it was
produced.  This module is the fix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..types import Severity, Stage

__all__ = ["JournalEntry", "Journal", "Warning_"]


@dataclass(frozen=True)
class JournalEntry:
    """One recorded action."""

    stage: Stage
    transformer: str
    action: str
    phase: str  # "fit" | "transform"
    columns: Sequence[str] = ()
    params: Dict[str, Any] = field(default_factory=dict)
    #: Measured outcome: rows affected, values imputed, categories grouped, ...
    effect: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": str(self.stage),
            "transformer": self.transformer,
            "action": self.action,
            "phase": self.phase,
            "columns": list(self.columns),
            "params": _plain(self.params),
            "effect": _plain(self.effect),
            "duration_s": round(self.duration_s, 6),
        }

    def __str__(self) -> str:
        where = ", ".join(self.columns[:4])
        if len(self.columns) > 4:
            where += f" (+{len(self.columns) - 4} more)"
        detail = ", ".join(f"{k}={v}" for k, v in self.effect.items()) if self.effect else ""
        tail = f" [{detail}]" if detail else ""
        return f"{self.stage}: {self.action} on {where}{tail}"


@dataclass(frozen=True)
class Warning_:
    """A structured advisory.

    Named with a trailing underscore to avoid shadowing the builtin.  Advisories are
    collected here rather than emitted through the ``warnings`` module alone, so that
    a user who has muted warnings still gets them in ``report.summary()``.
    """

    code: str
    severity: Severity
    message: str
    columns: Sequence[str] = ()
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": str(self.severity),
            "message": self.message,
            "columns": list(self.columns),
            "details": _plain(self.details),
        }

    def __str__(self) -> str:
        marker = {"info": "i", "warning": "!", "error": "x"}[str(self.severity)]
        return f"[{marker}] {self.message}"


def _plain(obj: Any) -> Any:
    """Convert numpy scalars and containers to JSON-serialisable equivalents."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


class Journal:
    """Append-only log, shared by every transformer in one pipeline run."""

    def __init__(self) -> None:
        self._entries: List[JournalEntry] = []
        self._warnings: List[Warning_] = []

    # -- writing -----------------------------------------------------------------

    def record(
        self,
        stage: Stage,
        transformer: str,
        action: str,
        phase: str,
        columns: Sequence[str] = (),
        params: Optional[Dict[str, Any]] = None,
        effect: Optional[Dict[str, Any]] = None,
        duration_s: float = 0.0,
    ) -> JournalEntry:
        entry = JournalEntry(
            stage=stage,
            transformer=transformer,
            action=action,
            phase=phase,
            columns=tuple(columns),
            params=dict(params or {}),
            effect=dict(effect or {}),
            duration_s=duration_s,
        )
        self._entries.append(entry)
        return entry

    def warn(
        self,
        code: str,
        message: str,
        severity: Severity = Severity.WARNING,
        columns: Sequence[str] = (),
        details: Optional[Dict[str, Any]] = None,
    ) -> Warning_:
        warning = Warning_(
            code=code,
            severity=Severity.coerce(severity),
            message=message,
            columns=tuple(columns),
            details=dict(details or {}),
        )
        self._warnings.append(warning)
        return warning

    def timer(self, stage: Stage, transformer: str, action: str, phase: str) -> "_Timer":
        """Context manager that records an entry with its wall-clock duration.

        ``with journal.timer(...) as t: t.effect["n_imputed"] = 12``
        """
        return _Timer(self, stage, transformer, action, phase)

    # -- reading ------------------------------------------------------------------

    @property
    def entries(self) -> List[JournalEntry]:
        return list(self._entries)

    @property
    def warnings(self) -> List[Warning_]:
        return list(self._warnings)

    def for_stage(self, stage: Stage) -> List[JournalEntry]:
        wanted = Stage.coerce(stage)
        return [e for e in self._entries if e.stage is wanted]

    def for_column(self, column: str) -> List[JournalEntry]:
        return [e for e in self._entries if column in e.columns]

    def fit_entries(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.phase == "fit"]

    def transform_entries(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.phase == "transform"]

    def clear_transform_entries(self) -> None:
        """Drop transform-phase entries.

        ``transform`` may be called many times (train, valid, test).  Without this the
        journal would grow without bound and the report would describe an arbitrary
        concatenation of runs rather than the most recent one.
        """
        self._entries = [e for e in self._entries if e.phase != "transform"]

    def total_duration(self, phase: Optional[str] = None) -> float:
        return sum(e.duration_s for e in self._entries if phase is None or e.phase == phase)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[JournalEntry]:
        return iter(self._entries)

    def __repr__(self) -> str:
        return f"Journal({len(self._entries)} entries, {len(self._warnings)} warnings)"


class _Timer:
    def __init__(self, journal: Journal, stage: Stage, transformer: str, action: str, phase: str):
        self._journal = journal
        self._stage = stage
        self._transformer = transformer
        self._action = action
        self._phase = phase
        self.columns: List[str] = []
        self.params: Dict[str, Any] = {}
        self.effect: Dict[str, Any] = {}
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._journal.record(
                self._stage,
                self._transformer,
                self._action,
                self._phase,
                columns=self.columns,
                params=self.params,
                effect=self.effect,
                duration_s=time.perf_counter() - self._start,
            )
        return False
