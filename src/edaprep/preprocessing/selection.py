"""Feature selection: removing columns that cannot help.

Kept separate from mandatory cleaning, per the design goal.  Nothing here is required for a
dataset to be model-ready; each filter trades information for simplicity, so each is
opt-in and each reports exactly what it dropped and why.

The correlation filter is a redesign, not a port.  The usual version::

    to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]

is order-dependent: for the chain ``a~b``, ``b~c`` with ``a`` and ``c`` uncorrelated it
drops both ``b`` and ``c``, losing ``c``'s independent information.  Here the
above-threshold edges form a graph, connected components are extracted, and one
representative per component is kept by an explicit, reported criterion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.base import Transformer
from ..core.context import FitContext
from ..types import SemanticType, Severity, Stage

__all__ = [
    "ConstantFilter",
    "MissingnessFilter",
    "DuplicateColumnFilter",
    "CorrelationFilter",
    "VarianceFilter",
    "ColumnDropper",
]


class _Dropper(Transformer):
    """Base for filters that remove whole columns."""

    stage = Stage.SELECT

    def _drop_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.to_drop_:
            return X
        keep = [c for c in X.columns if str(c) not in self.to_drop_]
        return X[keep]

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        present = [c for c in self.to_drop_ if c in map(str, X.columns)]
        if present:
            context.journal.record(
                self.stage,
                type(self).__name__,
                "drop_columns",
                "transform",
                columns=present,
                effect={"n_dropped": len(present)},
            )
        return self._drop_columns(X)

    def _compute_feature_names_out(self) -> List[str]:
        return [c for c in self.feature_names_in_ if c not in self.to_drop_]

    @property
    def dropped_(self) -> List[str]:
        return list(self.to_drop_)


class ColumnDropper(_Dropper):
    """Drop an explicit list of columns.

    Used by the planner for identifiers, text columns it will not process, and
    anything the user marked ``config.column(name).drop = True``.
    """

    stage = Stage.DROP_COLUMNS

    def __init__(
        self, columns: Optional[Sequence[str]] = None, reason: str = "explicitly dropped"
    ) -> None:
        super().__init__(columns)
        self.reason = reason

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        return []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        requested = list(self.columns or [])
        present = set(map(str, X.columns))
        self.to_drop_ = [c for c in requested if c in present]
        missing = [c for c in requested if c not in present]
        if missing:
            context.journal.warn(
                "drop_column_not_found",
                f"{len(missing)} column(s) marked for dropping are not in the data and "
                f"were skipped: {', '.join(repr(c) for c in missing[:5])}.",
                Severity.INFO,
                tuple(missing),
            )
        context.journal.record(
            self.stage,
            type(self).__name__,
            "drop_columns",
            "fit",
            columns=self.to_drop_,
            params={"reason": self.reason},
            effect={"n_dropped": len(self.to_drop_)},
        )

    def _resolve_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        # Unlike other transformers, a missing column here is not an error: dropping
        # something that is already absent is a no-op, not a schema violation.
        return [c for c in (self.columns or []) if c in set(map(str, X.columns))]


class ConstantFilter(_Dropper):
    """Drop columns with no variation.

    Parameters
    ----------
    near_constant_ratio :
        Also drop columns where a single value covers at least this fraction of rows.
        ``1.0`` (default) means exactly-constant only.
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        near_constant_ratio: float = 1.0,
        dropna: bool = True,
    ) -> None:
        super().__init__(columns)
        self.near_constant_ratio = near_constant_ratio
        self.dropna = dropna

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.to_drop_: List[str] = []
        self.reasons_: Dict[str, str] = {}
        n_rows = len(X)

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                if column == context.target:
                    continue
                series = X[column]
                try:
                    n_unique = int(series.nunique(dropna=self.dropna))
                except TypeError:
                    n_unique = int(series.astype(str).nunique(dropna=self.dropna))
                if n_unique <= 1:
                    self.to_drop_.append(column)
                    self.reasons_[column] = f"constant ({n_unique} distinct value)"
                    continue
                if self.near_constant_ratio < 1.0 and n_rows:
                    try:
                        top = series.value_counts(dropna=True)
                    except TypeError:
                        continue
                    if len(top) and (top.iloc[0] / n_rows) >= self.near_constant_ratio:
                        self.to_drop_.append(column)
                        self.reasons_[column] = (
                            f"near-constant ({top.iloc[0] / n_rows:.1%} of rows share "
                            f"the value {top.index[0]!r})"
                        )

            timer.columns = list(self.to_drop_)
            timer.effect = {"n_dropped": len(self.to_drop_), "reasons": dict(self.reasons_)}


class MissingnessFilter(_Dropper):
    """Drop columns whose missing fraction exceeds ``threshold``.

    Above the threshold, imputation invents most of the column: the "median" of a
    column that is 90% missing is a statement about 10% of the rows, applied to all of
    them.
    """

    def __init__(self, columns: Optional[Sequence[str]] = None, threshold: float = 0.6) -> None:
        super().__init__(columns)
        self.threshold = threshold

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.to_drop_: List[str] = []
        self.missing_fractions_: Dict[str, float] = {}
        n_rows = len(X)

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                if column == context.target:
                    continue
                fraction = float(X[column].isna().sum()) / n_rows if n_rows else 0.0
                self.missing_fractions_[column] = fraction
                if fraction >= self.threshold:
                    self.to_drop_.append(column)

            if self.to_drop_:
                context.journal.warn(
                    "dropped_high_missingness",
                    f"{len(self.to_drop_)} column(s) were dropped for exceeding "
                    f"{self.threshold:.0%} missing values: "
                    f"{', '.join(repr(c) for c in self.to_drop_[:6])}. Missingness can "
                    f"itself be informative; a MissingIndicator step earlier in the "
                    f"plan preserves that signal.",
                    Severity.WARNING,
                    tuple(self.to_drop_),
                    {c: round(self.missing_fractions_[c], 4) for c in self.to_drop_},
                )
            timer.columns = list(self.to_drop_)
            timer.effect = {"n_dropped": len(self.to_drop_)}


class DuplicateColumnFilter(_Dropper):
    """Drop columns that duplicate another column exactly.

    Hashes each column once and compares only within hash buckets, so the cost is
    O(n*p) rather than the O(n*p^2) of pairwise comparison.  The first column of each
    group, in input order, is kept.
    """

    def __init__(self, columns: Optional[Sequence[str]] = None) -> None:
        super().__init__(columns)

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        from ..profiling.quality import detect_duplicate_columns

        self.to_drop_: List[str] = []
        self.groups_: List[List[str]] = []

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            candidates = [c for c in self.columns_ if c != context.target]
            for group in detect_duplicate_columns(X, candidates):
                ordered = [c for c in self.columns_ if c in set(group)]
                self.groups_.append(ordered)
                self.to_drop_.extend(ordered[1:])

            if self.groups_:
                context.journal.warn(
                    "dropped_duplicate_columns",
                    f"{len(self.to_drop_)} duplicate column(s) removed, keeping the "
                    f"first of each group: "
                    + "; ".join("=".join(g) for g in self.groups_[:5]),
                    Severity.INFO,
                    tuple(self.to_drop_),
                    {"groups": [list(g) for g in self.groups_]},
                )
            timer.columns = list(self.to_drop_)
            timer.effect = {"n_dropped": len(self.to_drop_), "groups": self.groups_}


class VarianceFilter(_Dropper):
    """Drop numeric columns whose variance is at or below ``threshold``.

    Note that variance is scale-dependent: a column measured in kilometres has a
    variance a million times smaller than the same column in metres.  ``threshold=0.0``
    (drop only genuinely constant columns) is therefore the only scale-free default,
    and anything larger should be set with the units in mind.
    """

    def __init__(self, columns: Optional[Sequence[str]] = None, threshold: float = 0.0) -> None:
        super().__init__(columns)
        self.threshold = threshold

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        return [
            str(c)
            for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c].dtype) and str(c) != context.target
        ]

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.to_drop_: List[str] = []
        self.variances_: Dict[str, float] = {}

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                variance = float(finite.var(ddof=0)) if finite.size > 1 else 0.0
                self.variances_[column] = variance
                if variance <= self.threshold:
                    self.to_drop_.append(column)

            if self.threshold > 0 and self.to_drop_:
                context.journal.warn(
                    "variance_filter_is_scale_dependent",
                    f"{len(self.to_drop_)} column(s) were dropped for variance <= "
                    f"{self.threshold}. Variance depends on units, so this threshold "
                    f"treats a column in kilometres very differently from the same "
                    f"column in metres. Consider scaling before filtering, or filter "
                    f"on correlation instead.",
                    Severity.WARNING,
                    tuple(self.to_drop_),
                )
            timer.columns = list(self.to_drop_)
            timer.effect = {"n_dropped": len(self.to_drop_)}


class CorrelationFilter(_Dropper):
    """Drop redundant columns from groups of mutually correlated features.

    Above-threshold pairs form a graph; each connected component keeps one
    representative.  The representative is chosen by an explicit, reported criterion:

    1. highest absolute correlation with the target, when a target is available;
    2. else lowest missing fraction;
    3. else first in input order.

    Deterministic and order-independent, unlike the greedy version usually written.

    Parameters
    ----------
    method :
        ``"pearson"`` or ``"spearman"``.  Spearman is the safer default on skewed data
        -- the usual choice for wide, skewed data -- but costs a rank
        transform.
    sample_size :
        Compute the correlation matrix on at most this many rows; sampling 10,000 is
        conventional for exactly this reason.
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        threshold: float = 0.95,
        method: str = "spearman",
        sample_size: Optional[int] = 10_000,
    ) -> None:
        super().__init__(columns)
        self.threshold = threshold
        self.method = method
        self.sample_size = sample_size

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out = []
        for c in X.columns:
            name = str(c)
            if name == context.target or not pd.api.types.is_numeric_dtype(X[c].dtype):
                continue
            cp = context.column_profile(name)
            if cp is not None and cp.semantic is SemanticType.IDENTIFIER:
                continue
            out.append(name)
        return out

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.to_drop_: List[str] = []
        self.groups_: List[List[str]] = []
        self.kept_: Dict[str, str] = {}

        if len(self.columns_) < 2 or len(X) < 3:
            context.journal.record(
                self.stage, type(self).__name__, "correlation_filter", "fit",
                effect={"n_dropped": 0, "reason": "fewer than 2 numeric columns"},
            )
            return

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            frame = X[self.columns_]
            sampled = False
            if self.sample_size and len(frame) > self.sample_size:
                seed = context.random_state if context.random_state is not None else 0
                frame = frame.sample(n=self.sample_size, random_state=seed)
                sampled = True

            corr = frame.corr(method=self.method, numeric_only=True).abs()
            corr = corr.fillna(0.0).to_numpy()
            names = list(self.columns_)
            np.fill_diagonal(corr, 0.0)

            adjacency = corr >= self.threshold
            components = _connected_components(adjacency)

            scores = self._representative_scores(X, y, names, context)
            for component in components:
                if len(component) < 2:
                    continue
                members = [names[i] for i in sorted(component)]
                keeper = max(members, key=lambda c: (scores[c], -members.index(c)))
                self.groups_.append(members)
                for member in members:
                    if member != keeper:
                        self.to_drop_.append(member)
                        self.kept_[member] = keeper

            if self.to_drop_:
                context.journal.warn(
                    "dropped_correlated_columns",
                    f"{len(self.to_drop_)} column(s) removed from "
                    f"{len(self.groups_)} group(s) of features correlated at or above "
                    f"{self.threshold} ({self.method}). One representative per group "
                    f"was kept: "
                    + "; ".join(
                        f"{k} (dropped, kept {v})" for k, v in list(self.kept_.items())[:5]
                    ),
                    Severity.INFO,
                    tuple(self.to_drop_),
                    {"groups": self.groups_, "kept": self.kept_},
                )

            timer.columns = list(self.to_drop_)
            timer.params = {
                "threshold": self.threshold,
                "method": self.method,
                "sampled": sampled,
            }
            timer.effect = {
                "n_dropped": len(self.to_drop_),
                "n_groups": len(self.groups_),
                "kept": dict(self.kept_),
            }

    def _representative_scores(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        names: Sequence[str],
        context: FitContext,
    ) -> Dict[str, float]:
        """Higher is better.  Target association if available, else completeness."""
        scores: Dict[str, float] = {}
        profile = context.profile
        for name in names:
            cp = profile.columns.get(name) if profile is not None else None
            if cp is not None and cp.target_association is not None:
                scores[name] = float(cp.target_association)
            elif y is not None:
                scores[name] = _abs_corr(X[name], y)
            else:
                scores[name] = 1.0 - float(X[name].isna().mean())
        return scores


def _abs_corr(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if mask.sum() < 3:
        return 0.0
    a, b = xv[mask].to_numpy(), yv[mask].to_numpy()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _connected_components(adjacency: np.ndarray) -> List[List[int]]:
    """Connected components of a boolean adjacency matrix, by iterative DFS."""
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    components: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = np.flatnonzero(adjacency[node] & ~seen)
            seen[neighbours] = True
            stack.extend(neighbours.tolist())
        components.append(component)
    return components
