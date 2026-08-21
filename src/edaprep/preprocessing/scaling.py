"""Numeric scaling.

Deliberately conservative.  Not every model needs scaling,
and notebook practice scales unconditionally -- including for tree ensembles, where it is pure
cost.  ``strategy="auto"`` therefore consults ``model_family``: tree families get no
scaling at all, and with no family declared the default is standard scaling only
because that is the least surprising choice, recorded as such in the plan.

Verified against ``sklearn.preprocessing`` in the tests, with one intentional
difference: a zero-variance column is left unscaled here rather than divided by 1.0
silently, and the fact is reported.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AUTO
from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError
from ..types import ModelFamily, SemanticType, Severity, Stage

__all__ = ["Scaler", "resolve_scaling"]

_STRATEGIES = ("standard", "minmax", "robust", "maxabs", "none")


def resolve_scaling(model_family: Optional[ModelFamily]) -> str:
    """The default scaling for a model family.

    Mined from notebook practice, where two notebooks maintain parallel tree and linear
    branches and only the linear branch scales.
    """
    if model_family is None:
        return "standard"
    if model_family is ModelFamily.TREE:
        return "none"
    if model_family is ModelFamily.NEURAL:
        # Bounded inputs keep early-layer activations in a sane range; unbounded
        # standard scores do not.
        return "minmax"
    return "standard"


class Scaler(Transformer, ColumnTransformerMixin):
    """Scale numeric columns with statistics learned on the training data.

    Parameters
    ----------
    strategy :
        ``"standard"`` (zero mean, unit variance), ``"minmax"`` (to
        ``feature_range``), ``"robust"`` (median and IQR), ``"maxabs"`` (divide by the
        largest absolute value, preserving zeros and sparsity), ``"none"``, or
        ``"auto"`` to follow ``model_family``.
    feature_range :
        Target range for ``"minmax"``.
    with_mean, with_std :
        Standard-scaling components, as in scikit-learn.
    """

    stage = Stage.SCALE

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        strategy: str = AUTO,
        feature_range: tuple = (0.0, 1.0),
        with_mean: bool = True,
        with_std: bool = True,
        per_column: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(columns)
        self.strategy = strategy
        self.feature_range = tuple(feature_range)
        self.with_mean = with_mean
        self.with_std = with_std
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
            if cp is not None and cp.semantic in (
                SemanticType.IDENTIFIER,
                SemanticType.CONSTANT,
            ):
                continue
            # Indicator columns -- one-hot dummies, missing flags, weekend flags -- are
            # already on a common 0/1 scale.  Standard-scaling them is harmless
            # arithmetic but destroys the readability of coefficients and of the frame
            # itself, and it is not what a hand-written ColumnTransformer does either.
            # An explicit per_column entry still wins.
            named = self.per_column is not None and name in self.per_column
            if not named and int(X[name].nunique(dropna=True)) <= 2:
                continue
            out.append(name)
        return out

    def _resolve(self, column: str, context: FitContext) -> str:
        override = context.config.get_column(column)
        if override is not None and override.scaling is not None:
            strategy = override.scaling
        elif self.per_column and column in self.per_column:
            strategy = self.per_column[column]
        else:
            strategy = self.strategy
        if strategy == AUTO:
            if context.config.scaling != AUTO:
                strategy = context.config.scaling
            else:
                strategy = resolve_scaling(context.model_family)
        if strategy not in _STRATEGIES:
            raise ConfigurationError.unknown_option("scaling", strategy, _STRATEGIES)
        return strategy

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.strategies_: Dict[str, str] = {}
        self.centers_: Dict[str, float] = {}
        self.scales_: Dict[str, float] = {}

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                strategy = self._resolve(column, context)
                self.strategies_[column] = strategy
                if strategy == "none":
                    continue

                values = self._numeric_values(X[column])
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    self.centers_[column] = 0.0
                    self.scales_[column] = 1.0
                    continue

                center, scale = self._learn(strategy, finite)

                if scale == 0.0 or not np.isfinite(scale):
                    # A zero-variance column has nothing to scale.  sklearn substitutes
                    # 1.0 silently; here the column is left alone and the reason is
                    # recorded, because a constant feature surviving to this stage is
                    # itself worth knowing about.
                    context.journal.warn(
                        "zero_variance_not_scaled",
                        f"Column {column!r} has no variation in the training data "
                        f"(scale = 0), so {strategy} scaling would divide by zero. The "
                        f"column is passed through unscaled. It carries no information "
                        f"and is a candidate for removal.",
                        Severity.INFO,
                        (column,),
                    )
                    self.centers_[column] = center
                    self.scales_[column] = 1.0
                else:
                    self.centers_[column] = center
                    self.scales_[column] = scale

            timer.columns = list(self.columns_)
            timer.params = {"strategy": self.strategy}
            timer.effect = {
                "strategies": dict(self.strategies_),
                "n_scaled": sum(1 for s in self.strategies_.values() if s != "none"),
            }

    def _learn(self, strategy: str, finite: np.ndarray) -> tuple:
        """Return ``(center, scale)`` so that output = (x - center) / scale."""
        if strategy == "standard":
            center = float(finite.mean()) if self.with_mean else 0.0
            # ddof=0, matching sklearn's StandardScaler.
            scale = float(finite.std(ddof=0)) if self.with_std else 1.0
            return center, scale
        if strategy == "minmax":
            lo, hi = float(finite.min()), float(finite.max())
            span = hi - lo
            # Encoded as (x - lo) / span, then mapped to feature_range at transform.
            return lo, span
        if strategy == "robust":
            median = float(np.median(finite))
            q1, q3 = np.quantile(finite, [0.25, 0.75])
            return median, float(q3 - q1)
        if strategy == "maxabs":
            return 0.0, float(np.max(np.abs(finite)))
        raise ConfigurationError.unknown_option("scaling", strategy, _STRATEGIES)

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        lo_target, hi_target = self.feature_range
        span_target = hi_target - lo_target

        with context.journal.timer(
            self.stage, type(self).__name__, "scale", "transform"
        ) as timer:
            for column, strategy in self.strategies_.items():
                if strategy == "none" or column not in X.columns:
                    continue
                values = self._numeric_values(X[column])
                center = self.centers_[column]
                scale = self.scales_[column]
                scaled = (values - center) / scale
                if strategy == "minmax":
                    scaled = scaled * span_target + lo_target
                replacements[column] = pd.Series(scaled, index=X.index, name=column)

            timer.columns = sorted(replacements)
            timer.effect = {"n_columns_scaled": len(replacements)}
        return self._rebuild(X, replacements)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column": column,
                    "strategy": strategy,
                    "center": self.centers_.get(column),
                    "scale": self.scales_.get(column),
                }
                for column, strategy in self.strategies_.items()
            ]
        )
