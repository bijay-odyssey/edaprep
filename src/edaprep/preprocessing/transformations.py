"""Distribution transformations for skewed numeric columns.

``np.log1p`` is usually applied by eye, and the *target* often transformed on the full
frame before splitting.  Here the transform is chosen from measured skewness,
validated against the column's actual support, and refused with an explanation rather
than producing silent NaN or -inf.

Validity is the point of this module::

    log      requires x > 0
    log1p    requires x > -1
    sqrt     requires x >= 0
    boxcox   requires x > 0
    yeojohnson  works on the whole real line

A transform is never applied blindly: ``LogTransformer`` on a column with negatives
raises :class:`TransformationError` naming Yeo-Johnson as the alternative.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AUTO
from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError, TransformationError
from ..types import SemanticType, Severity, Stage

__all__ = ["DistributionTransformer", "choose_transform"]

_TRANSFORMS = ("log", "log1p", "sqrt", "boxcox", "yeojohnson", "quantile", "none")


def choose_transform(
    skew: float,
    minimum: float,
    skew_moderate: float,
    skew_heavy: float,
) -> str:
    """Pick a transform from measured skewness and support.

    Reproduces the usual skew tiering (docs/design-rationale.md, axis 1) with the validity
    checks it lacked:

    * ``|skew| < moderate``  -> no transform
    * ``moderate <= |skew| < heavy`` -> ``log1p`` when the column is non-negative,
      otherwise ``yeojohnson``
    * ``|skew| >= heavy`` -> ``yeojohnson``

    Yeo-Johnson rather than Box-Cox as the general answer, because Box-Cox requires
    strictly positive data and a single zero makes it fail -- and zero-heavy columns
    are common (``balance``, ``amount``, any count).
    """
    if not np.isfinite(skew):
        return "none"
    magnitude = abs(skew)
    if magnitude < skew_moderate:
        return "none"
    if magnitude >= skew_heavy:
        return "yeojohnson"
    if minimum >= 0:
        return "log1p"
    return "yeojohnson"


class DistributionTransformer(Transformer, ColumnTransformerMixin):
    """Apply a distribution-correcting transform to skewed numeric columns.

    Parameters
    ----------
    method :
        One of ``log``, ``log1p``, ``sqrt``, ``boxcox``, ``yeojohnson``, ``quantile``,
        ``none``, or ``"auto"`` to choose per column via :func:`choose_transform`.
    standardize :
        For ``boxcox``/``yeojohnson``, whether to zero-mean/unit-variance the output,
        matching ``sklearn.preprocessing.PowerTransformer(standardize=...)``.
    output_distribution :
        For ``quantile``: ``"uniform"`` or ``"normal"``.
    n_quantiles :
        Number of knots for the quantile transform.  Capped at the training row count.
    """

    stage = Stage.TRANSFORM

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        method: str = AUTO,
        standardize: bool = False,
        output_distribution: str = "uniform",
        n_quantiles: int = 1000,
        per_column: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(columns)
        self.method = method
        self.standardize = standardize
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.per_column = dict(per_column) if per_column else None

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out: List[str] = []
        for name in map(str, X.columns):
            if name == context.target:
                continue
            if not pd.api.types.is_numeric_dtype(X[name].dtype):
                continue
            if pd.api.types.is_bool_dtype(X[name].dtype):
                continue
            cp = context.column_profile(name)
            if cp is not None and cp.semantic is not SemanticType.NUMERIC:
                continue
            out.append(name)
        return out

    def _resolve(self, column: str, context: FitContext) -> str:
        override = context.config.get_column(column)
        if override is not None and override.transform is not None:
            return override.transform
        if self.per_column and column in self.per_column:
            return self.per_column[column]
        if self.method != AUTO:
            return self.method
        if context.config.transform_strategy != AUTO:
            return context.config.transform_strategy
        return AUTO

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.methods_: Dict[str, str] = {}
        self.params_: Dict[str, Dict[str, float]] = {}
        self.quantiles_: Dict[str, np.ndarray] = {}
        self.references_: Dict[str, np.ndarray] = {}

        thresholds = context.config.thresholds
        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                values = self._numeric_values(X[column])
                finite = values[np.isfinite(values)]
                method = self._resolve(column, context)

                if method == AUTO:
                    cp = context.column_profile(column)
                    skew = (
                        cp.skew
                        if cp is not None and cp.numeric is not None
                        else _sample_skew(finite)
                    )
                    minimum = float(finite.min()) if finite.size else 0.0
                    method = choose_transform(
                        skew, minimum, thresholds.skew_moderate, thresholds.skew_heavy
                    )

                if method == "none":
                    self.methods_[column] = "none"
                    continue
                if method not in _TRANSFORMS:
                    raise ConfigurationError.unknown_option("transform", method, _TRANSFORMS)

                self._validate(column, method, finite, context)
                self.methods_[column] = method
                self._learn(column, method, finite, context)

            timer.columns = list(self.columns_)
            timer.params = {"method": self.method}
            timer.effect = {
                "methods": dict(self.methods_),
                "n_transformed": sum(1 for m in self.methods_.values() if m != "none"),
            }

    def _validate(
        self, column: str, method: str, finite: np.ndarray, context: FitContext
    ) -> None:
        """Refuse a transform the column's support cannot support."""
        if finite.size == 0:
            raise TransformationError.degenerate(
                column, method, "the column has no finite training values"
            )
        minimum = float(finite.min())
        if method in ("log", "boxcox") and minimum <= 0:
            n_bad = int(np.count_nonzero(finite <= 0))
            raise TransformationError.non_positive(
                column,
                method,
                n_bad,
                "Use 'yeojohnson', which is defined on the whole real line, or "
                "'log1p' if the column is non-negative.",
            )
        if method == "log1p" and minimum <= -1:
            n_bad = int(np.count_nonzero(finite <= -1))
            raise TransformationError.non_positive(
                column,
                method,
                n_bad,
                "log1p requires x > -1. Use 'yeojohnson' instead.",
            )
        if method == "sqrt" and minimum < 0:
            n_bad = int(np.count_nonzero(finite < 0))
            raise TransformationError.non_positive(
                column, method, n_bad, "Use 'yeojohnson' instead."
            )
        if np.ptp(finite) == 0:
            raise TransformationError.degenerate(
                column, method, "the column is constant in the training data"
            )

    def _learn(
        self, column: str, method: str, finite: np.ndarray, context: FitContext
    ) -> None:
        if method in ("log", "log1p", "sqrt"):
            self.params_[column] = {}
            return

        if method in ("boxcox", "yeojohnson"):
            from scipy import stats as scipy_stats

            if method == "boxcox":
                _, lam = scipy_stats.boxcox(finite)
            else:
                _, lam = scipy_stats.yeojohnson(finite)
            params = {"lambda": float(lam)}
            if self.standardize:
                transformed = _apply_power(finite, method, float(lam))
                params["mean"] = float(transformed.mean())
                sd = float(transformed.std(ddof=0))
                params["std"] = sd if sd > 0 else 1.0
            self.params_[column] = params
            return

        if method == "quantile":
            n_knots = int(min(self.n_quantiles, max(2, finite.size)))
            levels = np.linspace(0.0, 1.0, n_knots)
            self.quantiles_[column] = np.quantile(finite, levels)
            self.references_[column] = levels
            self.params_[column] = {"n_quantiles": float(n_knots)}
            if finite.size < 50:
                context.journal.warn(
                    "quantile_transform_small_sample",
                    f"The quantile transform for column {column!r} was fitted on only "
                    f"{finite.size} values. The learned mapping will not generalise; "
                    f"prefer 'yeojohnson' on small samples.",
                    Severity.WARNING,
                    (column,),
                    {"n": int(finite.size)},
                )
            return

        raise ConfigurationError.unknown_option("transform", method, _TRANSFORMS)

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "transform_distribution", "transform"
        ) as timer:
            for column, method in self.methods_.items():
                if method == "none" or column not in X.columns:
                    continue
                values = self._numeric_values(X[column])
                out = self._apply(column, method, values, context)
                replacements[column] = pd.Series(out, index=X.index, name=column)

            timer.columns = sorted(replacements)
            timer.effect = {"methods": {c: self.methods_[c] for c in replacements}}
        return self._rebuild(X, replacements)

    def _apply(
        self, column: str, method: str, values: np.ndarray, context: FitContext
    ) -> np.ndarray:
        finite = np.isfinite(values)
        out = np.full_like(values, np.nan, dtype=np.float64)

        if method in ("log", "log1p", "sqrt", "boxcox"):
            # Values outside the transform's domain can appear at transform time even
            # though the training column was clean.  Producing NaN silently is what the
            # usually happens; here the count is recorded so it is visible in the report.
            domain = _domain_mask(method, values) & finite
            n_bad = int(np.count_nonzero(finite & ~domain))
            if n_bad:
                context.journal.warn(
                    "transform_domain_violation",
                    f"{n_bad} value(s) in column {column!r} fall outside the domain of "
                    f"the {method} transform learned at fit time and became missing. "
                    f"The training data contained no such values. Consider "
                    f"'yeojohnson' for this column, which has no domain restriction.",
                    Severity.WARNING,
                    (column,),
                    {"n_out_of_domain": n_bad, "method": method},
                )
            valid = domain
        else:
            valid = finite

        source = values[valid]
        if source.size == 0:
            return out

        if method == "log":
            out[valid] = np.log(source)
        elif method == "log1p":
            out[valid] = np.log1p(source)
        elif method == "sqrt":
            out[valid] = np.sqrt(source)
        elif method in ("boxcox", "yeojohnson"):
            params = self.params_[column]
            transformed = _apply_power(source, method, params["lambda"])
            if self.standardize:
                transformed = (transformed - params["mean"]) / params["std"]
            out[valid] = transformed
        elif method == "quantile":
            knots = self.quantiles_[column]
            levels = self.references_[column]
            # np.interp clamps outside the training range, so unseen extremes map to
            # 0 or 1 rather than extrapolating into nonsense.
            ranks = np.interp(source, knots, levels)
            if self.output_distribution == "normal":
                from scipy import stats as scipy_stats

                eps = 1e-7
                out[valid] = scipy_stats.norm.ppf(np.clip(ranks, eps, 1 - eps))
            else:
                out[valid] = ranks
        return out

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"column": c, "method": m, **self.params_.get(c, {})}
                for c, m in self.methods_.items()
                if m != "none"
            ]
        )


def _domain_mask(method: str, values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if method in ("log", "boxcox"):
            return values > 0
        if method == "log1p":
            return values > -1
        if method == "sqrt":
            return values >= 0
    return np.ones_like(values, dtype=bool)


def _apply_power(values: np.ndarray, method: str, lam: float) -> np.ndarray:
    """Box-Cox / Yeo-Johnson with the learned lambda.

    Reimplemented rather than calling ``scipy.stats.boxcox(x, lmbda=...)`` so that the
    fitted lambda is applied element-wise to arbitrary new data, including the negative
    branch of Yeo-Johnson, without refitting anything at transform time.
    """
    x = np.asarray(values, dtype=np.float64)
    if method == "boxcox":
        if abs(lam) < 1e-12:
            return np.log(x)
        return (np.power(x, lam) - 1.0) / lam

    out = np.empty_like(x)
    positive = x >= 0
    neg = ~positive
    if abs(lam) < 1e-12:
        out[positive] = np.log1p(x[positive])
    else:
        out[positive] = (np.power(x[positive] + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-12:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _sample_skew(finite: np.ndarray) -> float:
    from ..profiling.statistics import skewness

    return skewness(finite)
