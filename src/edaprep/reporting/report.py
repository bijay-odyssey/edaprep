"""The report: what was decided, what happened, and with what settings.

Three things are kept distinct because they answer different questions and can
legitimately disagree:

* the **profile** -- what the data looked like;
* the **plan** -- what was decided, and why (available before any data is transformed);
* the **journal** -- what actually happened, with measured counts.

A plan may decide to clip ``income`` and then find nothing above the fence in a
particular frame.  Collapsing the two would hide that.

Reproducibility metadata (library version, config, random seed, learned parameters) is
recorded so that a persisted output can be traced back to the process that made it --
the thing ``processed_train.csv`` in notebook practice cannot do.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .._version import __version__
from ..config import Config
from ..core.journal import Journal, JournalEntry, Warning_
from ..planning.decisions import Plan
from ..profiling.profiler import DatasetProfile
from ..types import Severity, Stage

__all__ = ["Report"]


@dataclass(frozen=True)
class Report:
    """Machine-readable and human-readable record of a pipeline run."""

    entries: Sequence[JournalEntry] = ()
    warnings: Sequence[Warning_] = ()
    profile: Optional[DatasetProfile] = None
    plan: Optional[Plan] = None
    config: Optional[Config] = None
    feature_names_in: Sequence[str] = ()
    feature_names_out: Sequence[str] = ()
    environment: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @classmethod
    def from_run(
        cls,
        journal: Journal,
        profile: Optional[DatasetProfile] = None,
        plan: Optional[Plan] = None,
        config: Optional[Config] = None,
        feature_names_in: Sequence[str] = (),
        feature_names_out: Sequence[str] = (),
    ) -> "Report":
        return cls(
            entries=tuple(journal.entries),
            warnings=tuple(journal.warnings),
            profile=profile,
            plan=plan,
            config=config,
            feature_names_in=tuple(feature_names_in),
            feature_names_out=tuple(feature_names_out),
            environment={
                "edaprep": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # -- views -----------------------------------------------------------------------

    @property
    def dropped_columns(self) -> Dict[str, str]:
        return dict(self.plan.dropped_columns) if self.plan else {}

    @property
    def n_columns_added(self) -> int:
        return max(0, len(self.feature_names_out) - len(self.feature_names_in))

    def for_stage(self, stage: Stage) -> List[JournalEntry]:
        wanted = Stage.coerce(stage)
        return [e for e in self.entries if e.stage is wanted]

    def for_column(self, column: str) -> List[JournalEntry]:
        return [e for e in self.entries if column in e.columns]

    def warnings_of(self, severity: Severity) -> List[Warning_]:
        target = Severity.coerce(severity)
        return [w for w in self.warnings if w.severity is target]

    @property
    def leakage(self) -> Dict[str, Any]:
        """What was learned from the target, and how it was protected.

        The audit notebook practice needed and did not have.
        """
        target_users = [
            e for e in self.entries if "target_encode" in e.action
        ]
        suspicious: List[str] = []
        if self.profile is not None:
            for issue in self.profile.issues:
                if issue.code == "possible_target_leakage":
                    suspicious.extend(issue.columns)
        return {
            "target": self.plan.target if self.plan else None,
            "transformers_using_target": sorted(
                {e.transformer for e in target_users}
            ),
            "cross_fitted": any(e.action == "target_encode_oof" for e in target_users),
            "columns_suspected_of_leakage": suspicious,
            "statistics_learned_at_fit_only": True,
        }

    # -- serialisation -------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "created_at": self.created_at,
            "environment": dict(self.environment),
            "feature_names_in": list(self.feature_names_in),
            "feature_names_out": list(self.feature_names_out),
            "n_features_in": len(self.feature_names_in),
            "n_features_out": len(self.feature_names_out),
            "leakage": self.leakage,
            "journal": [e.to_dict() for e in self.entries],
            "warnings": [w.to_dict() for w in self.warnings],
        }
        if self.config is not None:
            out["config"] = self.config.to_dict()
        if self.plan is not None:
            out["plan"] = self.plan.to_dict()
        if self.profile is not None:
            out["profile"] = self.profile.to_dict()
        return out

    def to_json(self, indent: int = 2, include_profile: bool = True) -> str:
        import json

        data = self.to_dict()
        if not include_profile:
            data.pop("profile", None)
        return json.dumps(data, indent=indent, default=str)

    def to_html(self, path: Optional[str] = None) -> str:
        from .html import render_html

        html = render_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    # -- rendering ------------------------------------------------------------------------

    def summary(self, max_rows: int = 40) -> str:
        """The human-readable report."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("edaprep report")
        lines.append("=" * 72)

        if self.profile is not None:
            p = self.profile
            lines.append("")
            lines.append("Dataset")
            lines.append(f"  {p.n_rows:,} rows x {p.n_columns:,} columns")
            lines.append(
                f"  {p.total_missing_cells:,} missing cells ({p.missing_fraction:.2%})"
            )
            if p.n_duplicate_rows:
                lines.append(
                    f"  {p.n_duplicate_rows:,} duplicate rows "
                    f"({p.duplicate_row_fraction:.2%})"
                )
            if p.target:
                extra = f", {p.target_classes} classes" if p.target_classes else ""
                if p.target_imbalance_ratio is not None:
                    extra += f", imbalance ratio {p.target_imbalance_ratio:.3f}"
                lines.append(f"  target: {p.target} ({p.target_kind}{extra})")

        lines.append("")
        lines.append(
            f"Features: {len(self.feature_names_in)} in -> "
            f"{len(self.feature_names_out)} out"
        )

        dropped = self.dropped_columns
        if dropped:
            lines.append("")
            lines.append(f"Removed ({len(dropped)})")
            for column, reason in list(dropped.items())[:max_rows]:
                lines.append(f"  {column:<28.28} {reason}")
            if len(dropped) > max_rows:
                lines.append(f"  ... {len(dropped) - max_rows} more")

        lines.extend(self._stage_sections(max_rows))

        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)})")
            for warning in sorted(self.warnings, key=lambda w: -w.severity.rank):
                lines.append(f"  {warning}")

        leakage = self.leakage
        lines.append("")
        lines.append("Leakage audit")
        lines.append("  All statistics were learned during fit, on the training frame only.")
        if leakage["transformers_using_target"]:
            lines.append(
                f"  Transformers that read the target: "
                f"{', '.join(leakage['transformers_using_target'])}"
            )
            lines.append(
                f"  Cross-fitted (no row encoded with its own target): "
                f"{leakage['cross_fitted']}"
            )
        else:
            lines.append("  No transformer read the target.")
        if leakage["columns_suspected_of_leakage"]:
            lines.append(
                f"  ! Columns almost perfectly associated with the target: "
                f"{', '.join(leakage['columns_suspected_of_leakage'])}"
            )

        if self.config is not None:
            lines.append("")
            lines.append("Reproducibility")
            lines.append(f"  edaprep {self.environment.get('edaprep', __version__)}")
            lines.append(f"  random_state: {self.config.random_state}")
            lines.append(f"  model_family: {self.config.model_family}")
            if self.profile is not None and self.profile.sampling.get("used"):
                s = self.profile.sampling
                lines.append(
                    f"  profiling sampled {s['n']:,} of {s['of']:,} rows "
                    f"(random_state={s.get('random_state')})"
                )
        return "\n".join(lines)

    def _stage_sections(self, max_rows: int) -> List[str]:
        lines: List[str] = []
        titles = {
            Stage.CAST: "Type corrections",
            Stage.DEDUPLICATE: "Duplicate rows",
            Stage.DATETIME: "Datetime expansion",
            Stage.MISSING_FLAG: "Missing indicators",
            Stage.OUTLIERS: "Outliers",
            Stage.MISSING: "Missing values",
            Stage.TRANSFORM: "Distribution transforms",
            Stage.RARE_CATEGORY: "Rare categories",
            Stage.ENCODE: "Categorical encoding",
            Stage.SCALE: "Scaling",
            Stage.SELECT: "Feature selection",
        }
        for stage, title in titles.items():
            rendered = self._render_stage(stage, max_rows)
            if rendered:
                lines.append("")
                lines.append(title)
                lines.extend(rendered)
        return lines

    def _render_stage(self, stage: Stage, max_rows: int) -> List[str]:
        """Per-column lines combining the decision with its measured effect."""
        decisions = (
            [d for d in self.plan.decisions if d.stage is stage] if self.plan else []
        )
        entries = [e for e in self.for_stage(stage) if e.phase == "transform"]

        effects: Dict[str, str] = {}
        for entry in entries:
            for key in ("n_values_imputed", "n_outliers", "n_values_grouped"):
                per_column = entry.effect.get(
                    "per_column" if key == "n_values_imputed" else key
                )
                if isinstance(per_column, dict):
                    for column, count in per_column.items():
                        if count:
                            effects[column] = f"{count:,} value(s) affected"

        lines: List[str] = []
        if decisions:
            for decision in decisions[:max_rows]:
                mark = "*" if decision.is_override else " "
                effect = effects.get(decision.column, "")
                suffix = f"  [{effect}]" if effect else ""
                lines.append(
                    f" {mark} {decision.column:<26.26} -> {decision.action}{suffix}"
                )
            if len(decisions) > max_rows:
                lines.append(f"   ... {len(decisions) - max_rows} more")
            return lines

        # No plan (explicit Pipeline): fall back to the journal alone.
        for entry in entries:
            if entry.effect:
                lines.append(f"   {entry}")
        return lines

    def __repr__(self) -> str:
        return (
            f"Report({len(self.entries)} journal entries, {len(self.warnings)} warnings, "
            f"{len(self.feature_names_in)} -> {len(self.feature_names_out)} features)"
        )

    def __str__(self) -> str:
        return self.summary()
