"""Plan value types: ``Decision``, ``PlannedStep``, ``Plan``.

These are inert data.  A :class:`Plan` holds no references to transformers, no frames,
and no closures; it round-trips through ``to_dict``/``from_dict`` and can be printed,
diffed, stored beside a model artefact, edited, and re-executed.

That single property delivers most of the design goal's transparency requirements at once:
``explain()`` is a renderer over a Plan, "override the automatic decisions" means
editing a Plan, "no magic" means the Plan is printable, and reproducibility means the
Plan is serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .._version import __version__
from ..types import Stage

__all__ = ["Decision", "PlannedStep", "Plan", "DecisionSource"]

#: Where a decision came from.  Distinguishing these is what makes overrides honest:
#: ``explain()`` marks a user override rather than presenting it as the rule's choice.
DecisionSource = str
RULE: DecisionSource = "rule"
USER_OVERRIDE: DecisionSource = "user_override"
DEFAULT: DecisionSource = "default"


@dataclass(frozen=True)
class Decision:
    """One decision about one column."""

    column: str
    stage: Stage
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    rule: str = ""
    confidence: float = 1.0
    source: DecisionSource = RULE
    #: Anything the user should know but that does not change the action.
    notes: Tuple[str, ...] = ()

    @property
    def is_override(self) -> bool:
        return self.source == USER_OVERRIDE

    @property
    def is_noop(self) -> bool:
        return self.action in ("none", "ignore", "keep")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "stage": str(self.stage),
            "action": self.action,
            "params": dict(self.params),
            "rationale": self.rationale,
            "rule": self.rule,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(
            column=data["column"],
            stage=Stage.coerce(data["stage"]),
            action=data["action"],
            params=dict(data.get("params", {})),
            rationale=data.get("rationale", ""),
            rule=data.get("rule", ""),
            confidence=float(data.get("confidence", 1.0)),
            source=data.get("source", RULE),
            notes=tuple(data.get("notes", ())),
        )

    def __str__(self) -> str:
        marker = "*" if self.is_override else ("-" if self.is_noop else "+")
        detail = f" ({self.rationale})" if self.rationale else ""
        return f"{marker} {self.action}{detail}"


@dataclass(frozen=True)
class PlannedStep:
    """One transformer's worth of work: a stage, an action, and the columns it covers.

    Decisions are grouped into steps so that one ``OneHotEncoder`` covers every column
    routed to one-hot encoding, rather than one transformer per column.  The per-column
    reasoning survives in :attr:`decisions`.
    """

    stage: Stage
    transformer: str
    columns: Tuple[str, ...]
    params: Dict[str, Any] = field(default_factory=dict)
    decisions: Tuple[Decision, ...] = ()

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": str(self.stage),
            "transformer": self.transformer,
            "columns": list(self.columns),
            "params": dict(self.params),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedStep":
        return cls(
            stage=Stage.coerce(data["stage"]),
            transformer=data["transformer"],
            columns=tuple(data.get("columns", ())),
            params=dict(data.get("params", {})),
            decisions=tuple(Decision.from_dict(d) for d in data.get("decisions", ())),
        )

    def __str__(self) -> str:
        shown = ", ".join(self.columns[:5])
        if len(self.columns) > 5:
            shown += f", +{len(self.columns) - 5} more"
        return f"{self.transformer}({shown})"


@dataclass(frozen=True)
class Plan:
    """An ordered, serialisable description of what will be done.

    Produced by :class:`~edaprep.planning.planner.Planner` from a profile and a config.
    Consumed by :class:`~edaprep.core.pipeline.Pipeline`, which materialises the
    transformers.  Nothing in between touches data.
    """

    steps: Tuple[PlannedStep, ...] = ()
    target: Optional[str] = None
    model_family: Optional[str] = None
    #: Columns removed before any transformation, with the reason.
    dropped_columns: Dict[str, str] = field(default_factory=dict)
    #: Advisories generated while planning: uncertain types, refused operations.
    notes: Tuple[str, ...] = ()
    edaprep_version: str = __version__

    # -- views ---------------------------------------------------------------------

    def __iter__(self) -> Iterator[PlannedStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def decisions(self) -> List[Decision]:
        return [d for step in self.steps for d in step.decisions]

    def for_column(self, column: str) -> List[Decision]:
        """Every decision affecting ``column``, in stage order."""
        found = [d for d in self.decisions if d.column == column]
        return sorted(found, key=lambda d: d.stage.order)

    def for_stage(self, stage: Stage) -> List[PlannedStep]:
        wanted = Stage.coerce(stage)
        return [s for s in self.steps if s.stage is wanted]

    @property
    def columns(self) -> List[str]:
        """Every column mentioned anywhere in the plan, in first-appearance order."""
        seen: Dict[str, None] = {}
        for step in self.steps:
            for column in step.columns:
                seen.setdefault(column, None)
        for column in self.dropped_columns:
            seen.setdefault(column, None)
        return list(seen)

    @property
    def overrides(self) -> List[Decision]:
        return [d for d in self.decisions if d.is_override]

    @property
    def uses_target(self) -> bool:
        return any(
            d.action in ("encode_target",) for d in self.decisions
        )

    # -- editing --------------------------------------------------------------------

    def without_stage(self, stage: Stage) -> "Plan":
        """A copy with every step of ``stage`` removed.

        Plans are frozen, so editing means deriving a new one.  That keeps a plan that
        has already been executed from changing underneath its report.
        """
        wanted = Stage.coerce(stage)
        return replace(self, steps=tuple(s for s in self.steps if s.stage is not wanted))

    def without_columns(self, columns: Sequence[str]) -> "Plan":
        """A copy with ``columns`` removed from every step."""
        drop = set(columns)
        steps = []
        for step in self.steps:
            kept = tuple(c for c in step.columns if c not in drop)
            if not kept:
                continue
            steps.append(
                replace(
                    step,
                    columns=kept,
                    decisions=tuple(d for d in step.decisions if d.column not in drop),
                )
            )
        return replace(self, steps=tuple(steps))

    # -- serialisation ---------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edaprep_version": self.edaprep_version,
            "target": self.target,
            "model_family": self.model_family,
            "dropped_columns": dict(self.dropped_columns),
            "notes": list(self.notes),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            steps=tuple(PlannedStep.from_dict(s) for s in data.get("steps", ())),
            target=data.get("target"),
            model_family=data.get("model_family"),
            dropped_columns=dict(data.get("dropped_columns", {})),
            notes=tuple(data.get("notes", ())),
            edaprep_version=data.get("edaprep_version", __version__),
        )

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    # -- rendering ---------------------------------------------------------------------

    def summary(self) -> str:
        """Stage-by-stage overview: what will run, on how many columns."""
        lines = ["Preprocessing plan"]
        if self.target:
            lines.append(f"  target: {self.target}")
        if self.model_family:
            lines.append(f"  model family: {self.model_family}")
        if self.dropped_columns:
            lines.append(f"  dropped up front: {len(self.dropped_columns)} column(s)")
            for column, reason in list(self.dropped_columns.items())[:10]:
                lines.append(f"    {column:<28.28} {reason}")
            if len(self.dropped_columns) > 10:
                lines.append(f"    ... {len(self.dropped_columns) - 10} more")
        lines.append("")
        if not self.steps:
            lines.append("  (no steps: the dataset needs no preprocessing)")
        for i, step in enumerate(self.steps, 1):
            shown = ", ".join(step.columns[:6])
            if len(step.columns) > 6:
                shown += f", +{len(step.columns) - 6} more"
            lines.append(f"  {i:>2}. [{str(step.stage):<13}] {step.transformer}")
            if step.columns:
                lines.append(f"      on: {shown}")
            interesting = {k: v for k, v in step.params.items() if v not in (None, {}, [])}
            if interesting:
                rendered = ", ".join(f"{k}={v}" for k, v in sorted(interesting.items()))
                lines.append(f"      params: {rendered}")
        if self.notes:
            lines.append("")
            lines.append("Planning notes")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def explain(self, column: Optional[str] = None, max_columns: int = 40) -> str:
        """Per-column reasoning.

        This is the answer to "why did my data change?".  Every line names the action,
        the rule that chose it, and the measurement behind it.
        """
        lines: List[str] = []
        if column is not None:
            return "\n".join(self._explain_column(column))

        columns = self.columns[:max_columns]
        for name in columns:
            lines.extend(self._explain_column(name))
            lines.append("")
        if len(self.columns) > max_columns:
            lines.append(f"... {len(self.columns) - max_columns} more columns")
        return "\n".join(lines).rstrip()

    def _explain_column(self, column: str) -> List[str]:
        lines = [f"{column}:"]
        if column in self.dropped_columns:
            lines.append(f"  x dropped - {self.dropped_columns[column]}")
            return lines
        decisions = self.for_column(column)
        if not decisions:
            lines.append("  - passed through unchanged")
            return lines
        for decision in decisions:
            if decision.is_override:
                marker = "*"
                suffix = " [user override]"
            elif decision.is_noop:
                marker = "-"
                suffix = ""
            else:
                marker = "+"
                suffix = ""
            text = f"  {marker} {decision.action}"
            if decision.rationale:
                text += f" - {decision.rationale}"
            lines.append(text + suffix)
            for note in decision.notes:
                lines.append(f"      ! {note}")
        return lines

    def __repr__(self) -> str:
        return (
            f"Plan({len(self.steps)} steps, {len(self.decisions)} decisions, "
            f"{len(self.dropped_columns)} dropped, target={self.target!r})"
        )

    def __str__(self) -> str:
        return self.summary()
