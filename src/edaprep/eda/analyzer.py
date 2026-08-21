"""The EDA engine.

Replaces steps 2-15 of the workflow reconstructed in ``docs/design-rationale.md``:
``head``, ``shape``, ``info``, ``describe``, ``isnull().sum()``, ``duplicated()``, the
per-numeric skew/kurtosis loop, the per-categorical ``value_counts`` loop, the outlier
scan, the correlation heatmap, VIF, and the t-test/ANOVA-against-target block.

Analysis levels
---------------
``quick``     dataset shape, dtypes, missingness, duplicates.  No moments, no
              correlation, no per-column association.  Suitable for a 500-column frame
              you have just loaded and want to look at.
``standard``  the above plus moments, outlier counts, category frequencies,
              correlation (sampled on large frames) and target relationships.
``deep``      the above plus VIF, statistical tests against the target, and the full
              correlation matrix regardless of width.

The distinction is real: ``quick`` skips every O(n log n) and O(p^2) computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..config import Config
from ..exceptions import EmptyDataError
from ..profiling.profiler import DatasetProfile
from ..profiling.profiler import profile as profile_dataset
from ..profiling.quality import QualityIssue
from ..types import AnalysisLevel, SemanticType, Severity
from .categorical import categorical_summary
from .correlation import correlation_matrix, top_correlated_pairs, variance_inflation
from .numerical import numerical_summary
from .outliers import outlier_summary
from .target import target_relationships, target_summary

__all__ = ["EDA", "EDAReport"]


@dataclass
class EDAReport:
    """The result of an analysis.

    Every section is a plain ``DataFrame`` or ``dict``, so it can be inspected, joined,
    exported, or dropped into a notebook without going through the library.
    """

    level: AnalysisLevel
    profile: DatasetProfile
    dataset: Dict[str, Any] = field(default_factory=dict)
    columns: Optional[pd.DataFrame] = None
    missing: Optional[pd.DataFrame] = None
    duplicates: Dict[str, Any] = field(default_factory=dict)
    numerical: Optional[pd.DataFrame] = None
    categorical: Optional[pd.DataFrame] = None
    cardinality: Optional[pd.DataFrame] = None
    outliers: Optional[pd.DataFrame] = None
    correlation: Optional[pd.DataFrame] = None
    correlated_pairs: Optional[pd.DataFrame] = None
    vif: Optional[pd.DataFrame] = None
    target: Dict[str, Any] = field(default_factory=dict)
    target_relationships: Optional[pd.DataFrame] = None
    issues: Sequence[QualityIssue] = ()

    # -- views -------------------------------------------------------------------

    @property
    def warnings(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity is not Severity.INFO]

    def issues_of(self, severity: Severity) -> List[QualityIssue]:
        target = Severity.coerce(severity)
        return [i for i in self.issues if i.severity is target]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    # -- serialisation -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        def frame(value: Optional[pd.DataFrame]) -> Optional[list]:
            if value is None:
                return None
            return value.replace({np.nan: None}).to_dict(orient="records")

        return {
            "level": str(self.level),
            "dataset": self.dataset,
            "duplicates": self.duplicates,
            "target": self.target,
            "columns": frame(self.columns),
            "missing": frame(self.missing),
            "numerical": frame(self.numerical),
            "categorical": frame(self.categorical),
            "cardinality": frame(self.cardinality),
            "outliers": frame(self.outliers),
            "correlated_pairs": frame(self.correlated_pairs),
            "vif": frame(self.vif),
            "target_relationships": frame(self.target_relationships),
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, path: Optional[str] = None) -> str:
        from ..reporting.html import _CSS, _e

        parts = [f"<title>EDA report</title><style>{_CSS}</style><main>"]
        parts.append("<h1>EDA report</h1>")
        parts.append(
            f"<div class='sub'>{self.profile.n_rows:,} rows &times; "
            f"{self.profile.n_columns:,} columns &middot; level {_e(self.level)}</div>"
        )
        sections = [
            ("Columns", self.columns),
            ("Missing values", self.missing),
            ("Numerical", self.numerical),
            ("Categorical", self.categorical),
            ("Outliers", self.outliers),
            ("Correlated pairs", self.correlated_pairs),
            ("Multicollinearity (VIF)", self.vif),
            ("Target relationships", self.target_relationships),
        ]
        for title, frame in sections:
            if frame is None or frame.empty:
                continue
            parts.append(f"<h2>{_e(title)}</h2><div class='scroll'>")
            parts.append(frame.to_html(index=False, border=0, na_rep=""))
            parts.append("</div>")
        if self.issues:
            parts.append("<h2>Findings</h2>")
            for issue in sorted(self.issues, key=lambda i: -i.severity.rank):
                parts.append(
                    f"<div class='w {_e(issue.severity)}'>{_e(issue.message)}</div>"
                )
        parts.append("</main>")
        html = "\n".join(parts)
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    # -- rendering ------------------------------------------------------------------

    def summary(self, max_rows: int = 25) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append(f"EDA report ({self.level})")
        lines.append("=" * 72)
        lines.append("")
        for key, value in self.dataset.items():
            lines.append(f"  {key.replace('_', ' '):<24} {value}")

        if self.duplicates.get("n_duplicate_rows"):
            lines.append("")
            lines.append(
                f"  duplicate rows           {self.duplicates['n_duplicate_rows']:,} "
                f"({self.duplicates['fraction']:.2%})"
            )

        if self.target:
            lines.append("")
            lines.append("Target")
            for key, value in self.target.items():
                if isinstance(value, dict):
                    continue
                lines.append(f"  {key.replace('_', ' '):<24} {value}")

        for title, frame in (
            ("Missing values", self.missing),
            ("Numerical summary", self.numerical),
            ("Categorical summary", self.categorical),
            ("Outliers", self.outliers),
            ("Most correlated pairs", self.correlated_pairs),
            ("Multicollinearity (VIF)", self.vif),
            ("Feature/target relationships", self.target_relationships),
        ):
            if frame is None or frame.empty:
                continue
            lines.append("")
            lines.append(title)
            with pd.option_context(
                "display.max_columns", 20, "display.width", 200, "display.max_rows", max_rows
            ):
                lines.append(_indent(frame.head(max_rows).to_string(index=False)))
            if len(frame) > max_rows:
                lines.append(f"  ... {len(frame) - max_rows} more rows")

        if self.issues:
            lines.append("")
            lines.append("Findings")
            for issue in sorted(self.issues, key=lambda i: -i.severity.rank):
                lines.append(f"  {issue}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EDAReport(level={self.level}, {self.profile.n_rows:,} rows, "
            f"{self.profile.n_columns} columns, {len(self.issues)} findings)"
        )

    def __str__(self) -> str:
        return self.summary()


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class EDA:
    """Exploratory analysis of a dataset.

    ::

        report = edaprep.EDA(df, target="churn").analyze(level="standard")
        print(report.summary())
        report.numerical          # a DataFrame, ready to use
        report.to_html("eda.html")

    Parameters
    ----------
    data :
        The frame.  Never mutated.
    target :
        Optional target column, which unlocks the relationship and imbalance sections.
    config :
        Thresholds and sampling settings.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"EDA expects a pandas DataFrame, got {type(data).__name__}. If you "
                f"have a Series, call .to_frame() first."
            )
        if data.shape[1] == 0:
            raise EmptyDataError.no_columns("analyse a dataset")
        if target is not None and target not in data.columns:
            available = ", ".join(repr(c) for c in list(data.columns)[:10])
            raise KeyError(
                f"target={target!r} is not a column of the dataset. Available columns "
                f"include: {available}."
            )
        self.data = data
        self.target = target
        self.config = config or Config()
        self._profile: Optional[DatasetProfile] = None

    @property
    def profile_(self) -> DatasetProfile:
        """The profile, computed once and reused."""
        if self._profile is None:
            self._profile = profile_dataset(
                self.data, target=self.target, config=self.config
            )
        return self._profile

    def analyze(
        self,
        level: Union[AnalysisLevel, str] = AnalysisLevel.STANDARD,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> EDAReport:
        """Run the analysis.

        Parameters
        ----------
        level :
            ``"quick"``, ``"standard"`` or ``"deep"``.  See the module docstring.
        include, exclude :
            Section names, for ``level="custom"``-style control without a fourth
            level.  ``include`` restricts to exactly those sections; ``exclude``
            removes them from whatever the level would have produced.
        """
        level = AnalysisLevel.coerce(level)
        data = self.data
        wanted = _sections_for(level, include, exclude)

        profile = profile_dataset(
            data,
            target=self.target,
            config=self.config,
            compute_moments=level.rank >= AnalysisLevel.STANDARD.rank,
            compute_target_association=(
                self.target is not None and level.rank >= AnalysisLevel.STANDARD.rank
            ),
            check_quality=level.rank >= AnalysisLevel.STANDARD.rank,
        )
        self._profile = profile

        report = EDAReport(level=level, profile=profile, issues=profile.issues)

        report.dataset = {
            "n_rows": profile.n_rows,
            "n_columns": profile.n_columns,
            "memory": _human_bytes(profile.memory_bytes),
            "missing_cells": profile.total_missing_cells,
            "missing_fraction": f"{profile.missing_fraction:.2%}",
            "numeric_columns": len(profile.numeric_columns),
            "categorical_columns": len(profile.categorical_columns),
            "datetime_columns": len(profile.datetime_columns),
            "text_columns": len(profile.text_columns),
            "identifier_columns": len(profile.identifier_columns),
            "constant_columns": len(profile.constant_columns),
        }
        report.duplicates = {
            "n_duplicate_rows": profile.n_duplicate_rows,
            "fraction": profile.duplicate_row_fraction,
        }

        if "columns" in wanted:
            report.columns = _column_frame(profile)
        if "missing" in wanted:
            report.missing = _missing_frame(profile)
        if "cardinality" in wanted:
            report.cardinality = _cardinality_frame(profile)
        if "numerical" in wanted:
            report.numerical = numerical_summary(profile)
        if "categorical" in wanted:
            report.categorical = categorical_summary(data, profile)
        if "outliers" in wanted:
            report.outliers = outlier_summary(data, profile, self.config)
        if "correlation" in wanted:
            report.correlation = correlation_matrix(
                data,
                profile,
                config=self.config,
                force=(level is AnalysisLevel.DEEP),
            )
            if report.correlation is not None:
                report.correlated_pairs = top_correlated_pairs(
                    report.correlation,
                    threshold=self.config.thresholds.correlation_threshold * 0.8,
                )
        if "vif" in wanted:
            report.vif = variance_inflation(data, profile, config=self.config)
        if self.target is not None:
            if "target" in wanted:
                report.target = target_summary(data, profile, self.target)
            if "target_relationships" in wanted:
                report.target_relationships = target_relationships(
                    data, profile, self.target, deep=(level is AnalysisLevel.DEEP)
                )
        return report

    # -- convenience ---------------------------------------------------------------

    def quick(self) -> EDAReport:
        return self.analyze(AnalysisLevel.QUICK)

    def deep(self) -> EDAReport:
        return self.analyze(AnalysisLevel.DEEP)

    def __repr__(self) -> str:
        return (
            f"EDA({self.data.shape[0]:,} rows x {self.data.shape[1]} columns, "
            f"target={self.target!r})"
        )


_QUICK = {"columns", "missing", "cardinality"}
_STANDARD = _QUICK | {
    "numerical",
    "categorical",
    "outliers",
    "correlation",
    "target",
    "target_relationships",
}
_DEEP = _STANDARD | {"vif"}


def _sections_for(
    level: AnalysisLevel,
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
) -> set:
    base = {
        AnalysisLevel.QUICK: _QUICK,
        AnalysisLevel.STANDARD: _STANDARD,
        AnalysisLevel.DEEP: _DEEP,
    }[level]
    wanted = set(include) if include is not None else set(base)
    if exclude:
        wanted -= set(exclude)
    return wanted


def _column_frame(profile: DatasetProfile) -> pd.DataFrame:
    rows = []
    for name in profile.column_order:
        cp = profile.columns[name]
        rows.append(
            {
                "column": name,
                "dtype": cp.dtype,
                "semantic": str(cp.semantic),
                "confidence": round(cp.semantic_confidence, 2),
                "missing": cp.n_missing,
                "missing_%": round(cp.missing_fraction * 100, 2),
                "unique": cp.n_unique,
                "unique_%": round(cp.unique_ratio * 100, 2),
                "memory_kb": round(cp.memory_bytes / 1024, 1),
            }
        )
    return pd.DataFrame(rows)


def _missing_frame(profile: DatasetProfile) -> pd.DataFrame:
    rows = []
    for name in profile.column_order:
        cp = profile.columns[name]
        if cp.n_missing == 0:
            continue
        rows.append(
            {
                "column": name,
                "semantic": str(cp.semantic),
                "n_missing": cp.n_missing,
                "missing_%": round(cp.missing_fraction * 100, 2),
                "co_missing_with": ", ".join(
                    sorted(
                        {
                            other
                            for a, b, _ in profile.comissing_pairs
                            for other in ((b,) if a == name else (a,) if b == name else ())
                        }
                    )[:3]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("missing_%", ascending=False, ignore_index=True)
    return frame


def _cardinality_frame(profile: DatasetProfile) -> pd.DataFrame:
    rows = []
    for name in profile.column_order:
        cp = profile.columns[name]
        rows.append(
            {
                "column": name,
                "semantic": str(cp.semantic),
                "n_unique": cp.n_unique,
                "unique_%": round(cp.unique_ratio * 100, 2),
                "modal_value": cp.modal_value,
                "modal_%": round(cp.modal_frequency * 100, 2),
                "flag": _cardinality_flag(cp),
            }
        )
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False, ignore_index=True)


def _cardinality_flag(cp) -> str:
    if cp.is_constant:
        return "constant"
    if cp.is_near_constant:
        return "near-constant"
    if cp.is_possible_id:
        return "identifier"
    if cp.semantic is SemanticType.CATEGORICAL and cp.n_unique > 50:
        return "high-cardinality"
    return ""


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover
