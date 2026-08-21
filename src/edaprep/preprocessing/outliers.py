"""Outlier detection and handling.

The IQR fence and the z-score fence get rewritten in project after project, and two
bugs recur often enough to be worth restating (docs/design-rationale.md section 5.2):

* ``zscore(col.dropna())`` produces a positional array over the *non-null* subset which
  is then used as a boolean mask against the *full* frame, flagging the wrong rows
  whenever the column has any NaN;
* the variant without ``dropna()`` is worse: ``zscore`` returns all-NaN for a column
  with a single missing value, so the mask is all-False and no outlier is ever found --
  silently.

Every detector here returns a boolean ``Series`` aligned to the input index, with
missing values never flagged.  That single choice removes the whole class of bug.

Detection and removal are also kept apart, because they are not the same decision.  A
value being statistically unusual is not evidence that it is wrong, so the default
strategy is ``"report"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import AUTO, Thresholds
from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError
from ..profiling.statistics import median_abs_deviation
from ..types import SemanticType, Severity, Stage

__all__ = [
    "Bounds",
    "OutlierDetector",
    "IQRDetector",
    "ZScoreDetector",
    "ModifiedZScoreDetector",
    "PercentileDetector",
    "OutlierHandler",
    "detect_outliers",
]

#: Consistency constant making the MAD an unbiased estimator of sigma for normal data.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Bounds:
    """A learned acceptance interval for one column."""

    lower: float
    upper: float
    method: str
    params: Dict[str, float]

    def mask(self, values: np.ndarray) -> np.ndarray:
        """Boolean array: True where the value is outside the fence.

        NaN compares False in both directions, so missing values are never flagged.
        That is deliberate: a missing value is a missing-value problem, not an outlier.
        """
        return (values < self.lower) | (values > self.upper)

    def to_dict(self) -> Dict[str, object]:
        return {
            "lower": _finite(self.lower),
            "upper": _finite(self.upper),
            "method": self.method,
            "params": {k: _finite(v) for k, v in self.params.items()},
        }


def _finite(value: float) -> Optional[float]:
    f = float(value)
    return None if (np.isnan(f) or np.isinf(f)) else f


class OutlierDetector(ABC):
    """Learns a :class:`Bounds` from training values."""

    name: str = "base"

    @abstractmethod
    def fit_bounds(self, values: np.ndarray) -> Bounds:
        """Compute the fence from finite training values."""

    def __call__(self, values: np.ndarray) -> Bounds:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return Bounds(-np.inf, np.inf, self.name, {"n": 0})
        return self.fit_bounds(finite)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class IQRDetector(OutlierDetector):
    """Tukey's fence: ``[Q1 - k*IQR, Q3 + k*IQR]``.

    ``k=1.5`` is Tukey's original and the usual choice; practice widens to ``k=3.0``
    for skewed columns, which the planner reproduces as a named decision rather than an
    unexplained literal.
    """

    name = "iqr"

    def __init__(self, k: float = 1.5) -> None:
        if k <= 0:
            raise ConfigurationError(
                f"IQRDetector(k={k}) must be positive; a non-positive multiplier "
                f"would place the fence inside the interquartile range and flag most "
                f"of the column."
            )
        self.k = float(k)

    def fit_bounds(self, values: np.ndarray) -> Bounds:
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        return Bounds(
            lower=float(q1 - self.k * iqr),
            upper=float(q3 + self.k * iqr),
            method=self.name,
            params={"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "k": self.k},
        )

    def __repr__(self) -> str:
        return f"IQRDetector(k={self.k})"


class ZScoreDetector(OutlierDetector):
    """``[mean - t*sd, mean + t*sd]``.

    ``ddof`` defaults to 0 (population), matching ``scipy.stats.zscore``.  Notebook code
    mixes ``scipy.stats.zscore`` (ddof=0) with ``Series.std()`` (ddof=1) and treats the
    results as the same statistic; naming the parameter makes the difference visible.

    Note that the mean and standard deviation are themselves distorted by the outliers
    being looked for, so this detector is only appropriate for roughly symmetric
    columns.  :class:`ModifiedZScoreDetector` is the robust alternative.
    """

    name = "zscore"

    def __init__(self, threshold: float = 3.0, ddof: int = 0) -> None:
        if threshold <= 0:
            raise ConfigurationError(
                f"ZScoreDetector(threshold={threshold}) must be positive."
            )
        self.threshold = float(threshold)
        self.ddof = int(ddof)

    def fit_bounds(self, values: np.ndarray) -> Bounds:
        mean = float(values.mean())
        sd = float(values.std(ddof=self.ddof)) if values.size > self.ddof else 0.0
        if sd == 0.0:
            return Bounds(-np.inf, np.inf, self.name, {"mean": mean, "std": 0.0})
        return Bounds(
            lower=mean - self.threshold * sd,
            upper=mean + self.threshold * sd,
            method=self.name,
            params={"mean": mean, "std": sd, "threshold": self.threshold},
        )

    def __repr__(self) -> str:
        return f"ZScoreDetector(threshold={self.threshold}, ddof={self.ddof})"


class ModifiedZScoreDetector(OutlierDetector):
    """MAD-based z-score: ``0.6745 * (x - median) / MAD``.

    Robust: the median and MAD are not dragged by the extreme values being detected,
    unlike the mean and standard deviation.  ``threshold=3.5`` is Iglewicz and
    Hoaglin's recommendation.  This is what skewed columns need, and what the plain
    z-score fence is usually reached for instead.

    When the MAD is zero -- more than half the column shares one value, common in
    zero-heavy columns -- the scale is undefined and the detector falls back to the
    IQR fence rather than flagging every non-modal value.
    """

    name = "modified_zscore"

    def __init__(self, threshold: float = 3.5) -> None:
        if threshold <= 0:
            raise ConfigurationError(
                f"ModifiedZScoreDetector(threshold={threshold}) must be positive."
            )
        self.threshold = float(threshold)

    def fit_bounds(self, values: np.ndarray) -> Bounds:
        median = float(np.median(values))
        mad = median_abs_deviation(values)
        if mad == 0.0 or not np.isfinite(mad):
            fallback = IQRDetector(k=3.0).fit_bounds(values)
            return Bounds(
                fallback.lower,
                fallback.upper,
                self.name,
                {
                    "median": median,
                    "mad": 0.0,
                    "threshold": self.threshold,
                    "fallback_to_iqr": 1.0,
                },
            )
        scale = mad * MAD_TO_SIGMA
        return Bounds(
            lower=median - self.threshold * scale,
            upper=median + self.threshold * scale,
            method=self.name,
            params={"median": median, "mad": mad, "threshold": self.threshold},
        )

    def __repr__(self) -> str:
        return f"ModifiedZScoreDetector(threshold={self.threshold})"


class PercentileDetector(OutlierDetector):
    """Fixed quantile fence, e.g. ``[P1, P99]``.

    Distribution-free and the natural partner of winsorising.  It always flags a fixed
    fraction of the training data by construction, which is a property to be aware of:
    it answers "which values are extreme *in this sample*", not "which values are
    implausible".
    """

    name = "percentile"

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        if not 0.0 <= lower < upper <= 1.0:
            raise ConfigurationError(
                f"PercentileDetector(lower={lower}, upper={upper}) must satisfy "
                f"0 <= lower < upper <= 1."
            )
        self.lower = float(lower)
        self.upper = float(upper)

    def fit_bounds(self, values: np.ndarray) -> Bounds:
        lo, hi = np.quantile(values, [self.lower, self.upper])
        return Bounds(
            lower=float(lo),
            upper=float(hi),
            method=self.name,
            params={"lower_q": self.lower, "upper_q": self.upper},
        )

    def __repr__(self) -> str:
        return f"PercentileDetector(lower={self.lower}, upper={self.upper})"


_DETECTORS = {
    "iqr": IQRDetector,
    "zscore": ZScoreDetector,
    "modified_zscore": ModifiedZScoreDetector,
    "percentile": PercentileDetector,
}


def make_detector(method: str, thresholds: Thresholds, skewed: bool = False) -> OutlierDetector:
    """Build a detector from a method name and the configured thresholds."""
    if method == "iqr":
        return IQRDetector(k=thresholds.iqr_k_skewed if skewed else thresholds.iqr_k)
    if method == "zscore":
        return ZScoreDetector(threshold=thresholds.zscore_threshold)
    if method == "modified_zscore":
        return ModifiedZScoreDetector(threshold=thresholds.modified_zscore_threshold)
    if method == "percentile":
        lo, hi = thresholds.percentile_bounds
        return PercentileDetector(lower=lo, upper=hi)
    raise ConfigurationError.unknown_option("outlier_method", method, sorted(_DETECTORS))


def detect_outliers(
    series: pd.Series, method: str = "iqr", **kwargs
) -> pd.Series:
    """Convenience: a boolean Series flagging outliers in ``series``.

    Aligned to the input index; missing values are never flagged.  This is the
    one-liner that gets rewritten in every project.

    >>> detect_outliers(pd.Series([1, 2, 3, 100]), method="iqr").sum()
    1
    """
    if method not in _DETECTORS:
        raise ConfigurationError.unknown_option("method", method, sorted(_DETECTORS))
    detector = _DETECTORS[method](**kwargs)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    bounds = detector(values)
    return pd.Series(bounds.mask(values), index=series.index, name=series.name)


class OutlierHandler(Transformer, ColumnTransformerMixin):
    """Detect outliers on train, then apply the same fence at every transform.

    Parameters
    ----------
    method :
        ``"auto"`` follows the usual own rule (docs/design-rationale.md, axis 3): the IQR
        fence with a widened ``k`` for skewed columns, the z-score fence for
        symmetric ones -- except that the skewed branch uses the *modified* z-score
        where the column is heavily skewed, which is the robust statistic the IQR fence
        reached for the IQR to approximate.
    strategy :
        What to do with a detected outlier.

        ``"report"``   record and count them, change nothing (the default).
        ``"clip"``     clamp to the fence.
        ``"winsorize"`` clamp to the percentile fence regardless of ``method``.
        ``"impute"``   set to NaN, for a later imputation step to fill.  This is the
                       behaviour that usually emerges by accident
                       (docs/design-rationale.md section 5.6), named.
        ``"remove"``   drop the rows.  Fit-time only; see below.
        ``"ignore"``   do nothing and do not even count.
    max_action_fraction :
        If more than this fraction of the training column is flagged, the fence is
        describing the distribution rather than errors in it, so the strategy is
        downgraded to ``"report"`` and a warning is recorded.

    Row removal
    -----------
    ``strategy="remove"`` drops rows at fit time only.  Dropping rows during
    ``transform`` would silently change the number of predictions returned for a test
    set, which no caller expects; the fence is applied as a clip instead and the
    difference is reported.
    """

    stage = Stage.OUTLIERS

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        method: str = AUTO,
        strategy: str = "report",
        per_column: Optional[Dict[str, str]] = None,
        max_action_fraction: Optional[float] = None,
    ) -> None:
        super().__init__(columns)
        self.method = method
        self.strategy = strategy
        self.per_column = dict(per_column) if per_column else None
        self.max_action_fraction = max_action_fraction

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out: List[str] = []
        for name in map(str, X.columns):
            if name == context.target:
                continue
            cp = context.column_profile(name)
            if cp is not None:
                if cp.semantic is not SemanticType.NUMERIC:
                    continue
            elif not pd.api.types.is_numeric_dtype(
                X[name].dtype
            ) or pd.api.types.is_bool_dtype(X[name].dtype):
                continue
            # Columns created mid-pipeline (missing indicators, one-hot columns,
            # calendar flags) have no profile entry, so the semantic check above cannot
            # protect them.  A 0/1 column with under 25% ones has Q1 = Q3 = 0, giving
            # the fence [0, 0]; clipping to it sets every 1 to 0 and destroys the
            # column outright.  "Outlier" is not a meaningful notion for two-valued
            # data, so exclude it here rather than relying on a downstream guard.
            if int(X[name].nunique(dropna=True)) <= 2:
                continue
            out.append(name)
        return out

    def _resolve(self, column: str, context: FitContext) -> Tuple[str, str]:
        """(method, strategy) for one column, honouring overrides."""
        override = context.config.get_column(column)
        method = self.method
        strategy = self.strategy
        if self.per_column and column in self.per_column:
            strategy = self.per_column[column]
        if override is not None:
            if override.outlier_method is not None:
                method = override.outlier_method
            if override.outlier_strategy is not None:
                strategy = override.outlier_strategy
        return method, strategy

    def _choose_method(self, column: str, skew: float, context: FitContext) -> Tuple[str, bool]:
        """Auto method selection.  Returns (method, treat_as_skewed)."""
        thresholds = context.config.thresholds
        if not np.isfinite(skew):
            return "iqr", False
        magnitude = abs(skew)
        if magnitude >= thresholds.skew_heavy:
            return "modified_zscore", True
        if magnitude >= thresholds.skew_moderate:
            return "iqr", True
        return "zscore", False

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.bounds_: Dict[str, Bounds] = {}
        self.strategies_: Dict[str, str] = {}
        self.n_detected_: Dict[str, int] = {}
        self.fraction_detected_: Dict[str, float] = {}

        thresholds = context.config.thresholds
        cap = (
            self.max_action_fraction
            if self.max_action_fraction is not None
            else thresholds.outlier_max_action_fraction
        )

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                values = self._numeric_values(X[column])
                cp = context.column_profile(column)
                skew = cp.skew if cp is not None and cp.numeric is not None else float("nan")

                method, strategy = self._resolve(column, context)
                if strategy == "ignore":
                    self.strategies_[column] = "ignore"
                    continue

                skewed = False
                if method == AUTO:
                    method, skewed = self._choose_method(column, skew, context)
                elif method == "none":
                    self.strategies_[column] = "ignore"
                    continue
                else:
                    skewed = np.isfinite(skew) and abs(skew) >= thresholds.skew_moderate

                if strategy == "winsorize":
                    detector: OutlierDetector = make_detector("percentile", thresholds)
                    method = "percentile"
                else:
                    detector = make_detector(method, thresholds, skewed=skewed)

                bounds = detector(values)
                flagged = bounds.mask(values)
                n_flagged = int(np.count_nonzero(flagged))
                n_finite = int(np.count_nonzero(np.isfinite(values)))
                fraction = (n_flagged / n_finite) if n_finite else 0.0

                # A collapsed fence means the column's central 50% is a single value,
                # so every other value looks "extreme".  Acting on that would replace
                # the whole distribution with its mode.  Defence in depth: binary
                # columns are already excluded above, but zero-heavy and heavily tied
                # columns reach here too.
                if (
                    strategy not in ("report", "ignore")
                    and np.isfinite(bounds.lower)
                    and np.isfinite(bounds.upper)
                    and bounds.upper <= bounds.lower
                ):
                    context.journal.warn(
                        "degenerate_outlier_fence",
                        f"The {method} fence for column {column!r} collapsed to a "
                        f"single point ([{bounds.lower:g}, {bounds.upper:g}]), because "
                        f"most of the column shares one value. Acting on it would "
                        f"replace the distribution with that value, so the strategy "
                        f"was downgraded from {strategy!r} to 'report'.",
                        Severity.WARNING,
                        (column,),
                        {"lower": float(bounds.lower), "upper": float(bounds.upper)},
                    )
                    strategy = "report"

                if strategy not in ("report", "ignore") and fraction > cap:
                    context.journal.warn(
                        "outlier_action_downgraded",
                        f"{fraction:.1%} of column {column!r} falls outside the "
                        f"{method} fence, which is more than the {cap:.0%} ceiling. At "
                        f"that share the fence is describing the distribution rather "
                        f"than errors in it, so the strategy was downgraded from "
                        f"{strategy!r} to 'report'. Raise max_action_fraction, or set "
                        f"config.column({column!r}).outlier_strategy explicitly, to "
                        f"override.",
                        Severity.WARNING,
                        (column,),
                        {"fraction": round(fraction, 4), "method": method, "cap": cap},
                    )
                    strategy = "report"

                self.bounds_[column] = bounds
                self.strategies_[column] = strategy
                self.n_detected_[column] = n_flagged
                self.fraction_detected_[column] = fraction

            timer.columns = list(self.columns_)
            timer.params = {"method": self.method, "strategy": self.strategy}
            timer.effect = {
                "n_columns": len(self.bounds_),
                "n_detected": dict(self.n_detected_),
                "fraction_detected": {
                    k: round(v, 5) for k, v in self.fraction_detected_.items()
                },
                "bounds": {k: v.to_dict() for k, v in self.bounds_.items()},
            }

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        affected: Dict[str, int] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "handle_outliers", "transform"
        ) as timer:
            for column, bounds in self.bounds_.items():
                if column not in X.columns:
                    continue
                strategy = self.strategies_.get(column, "report")
                if strategy in ("report", "ignore"):
                    values = self._numeric_values(X[column])
                    affected[column] = int(np.count_nonzero(bounds.mask(values)))
                    continue

                series = X[column]
                values = self._numeric_values(series)
                mask = bounds.mask(values)
                n = int(np.count_nonzero(mask))
                affected[column] = n
                if n == 0:
                    continue

                if strategy in ("clip", "winsorize", "remove"):
                    # "remove" reaches transform as a clip on purpose: dropping rows here
                    # would change how many predictions the caller gets back.
                    replacements[column] = series.clip(lower=bounds.lower, upper=bounds.upper)
                elif strategy == "impute":
                    replacements[column] = series.mask(pd.Series(mask, index=series.index))

            timer.columns = sorted(affected)
            timer.effect = {
                "strategies": dict(self.strategies_),
                "n_outliers": affected,
                "n_modified": int(sum(affected[c] for c in replacements)),
            }
        return self._rebuild(X, replacements)

    # -- fit-time row removal -------------------------------------------------------

    def rows_to_remove(self, X: pd.DataFrame) -> pd.Series:
        """Boolean mask of training rows flagged for removal.

        Exposed separately from :meth:`transform` because removing rows is a fit-time
        operation: the pipeline applies it to the training frame only.
        """
        mask = pd.Series(False, index=X.index)
        for column, bounds in self.bounds_.items():
            if self.strategies_.get(column) != "remove" or column not in X.columns:
                continue
            values = self._numeric_values(X[column])
            mask |= pd.Series(bounds.mask(values), index=X.index)
        return mask

    def summary(self) -> pd.DataFrame:
        """One row per column: method, fence, count and fraction detected on train."""
        rows = []
        for column, bounds in self.bounds_.items():
            rows.append(
                {
                    "column": column,
                    "method": bounds.method,
                    "strategy": self.strategies_.get(column),
                    "lower": bounds.lower,
                    "upper": bounds.upper,
                    "n_outliers": self.n_detected_.get(column, 0),
                    "fraction": self.fraction_detected_.get(column, 0.0),
                }
            )
        return pd.DataFrame(rows)
