"""Missing-value handling.

Replaces the usual three-lines-per-column boilerplate::

    age_median = train_df['Age'].median()
    train_df['Age'] = train_df['Age'].fillna(age_median)
    test_df ['Age'] = test_df ['Age'].fillna(age_median)

The semantics there are right -- learn on train, apply to test -- and are exactly what
``fit``/``transform`` gives for free.  What is added is that the fill values are
recorded, the number of values actually filled is measured on every transform, and a
column that is mostly missing is reported rather than quietly invented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type

import numpy as np
import pandas as pd

from ..config import AUTO
from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError
from ..types import SemanticType, Severity, Stage

__all__ = ["MissingValueHandler", "MissingIndicator"]

#: Strategies that need a learned statistic.
_LEARNED = frozenset({"mean", "median", "mode"})
#: Strategies applied row-wise at transform time with no learned state.
_ROWWISE = frozenset({"ffill", "bfill"})
#: Multivariate imputers (scikit-learn, optional dependency).
_SKLEARN_IMPUTE = frozenset({"knn", "iterative"})

_ADVANCED_INSTALL_MSG = (
    "Install the optional dependency with: pip install 'edaprep[advanced]' "
    "(requires scikit-learn>=1.1)."
)


def _valid_imputation_strategies() -> List[str]:
    return sorted(
        _LEARNED
        | _ROWWISE
        | _SKLEARN_IMPUTE
        | {"constant", "missing_category", "none", "drop_rows"}
    )


def _is_numeric_imputation_column(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    )


def _load_sklearn_imputers() -> Tuple[Type[Any], Type[Any]]:
    try:
        from sklearn.impute import KNNImputer
    except ImportError as exc:  # pragma: no cover - exercised via mock in tests
        raise ConfigurationError(
            f"strategy='knn' or 'iterative' requires scikit-learn. {_ADVANCED_INSTALL_MSG}"
        ) from exc
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            f"strategy='iterative' requires scikit-learn. {_ADVANCED_INSTALL_MSG}"
        ) from exc
    return KNNImputer, IterativeImputer


class MissingValueHandler(Transformer, ColumnTransformerMixin):
    """Impute missing values with statistics learned on the training data.

    Parameters
    ----------
    strategy :
        ``"auto"`` picks per column from the semantic type: median for numeric
        (robust to the skew real tabular data is full of), mode for categorical
        and binary, and an explicit ``"missing"`` category for high-cardinality
        categoricals where the mode is not representative.  Any other value pins
        every column to that strategy.  ``"knn"`` and ``"iterative"`` use
        scikit-learn multivariate imputers (``edaprep[advanced]``).
    fill_value :
        Used by ``strategy="constant"``.
    per_column :
        ``{column: strategy}``, overriding ``strategy`` for those columns.
    add_indicator :
        Add a ``{column}__was_missing`` flag.  See :class:`MissingIndicator`; the
        planner normally emits that as a separate step so the flags are created
        *before* imputation destroys the signal.
    missing_category :
        The label used by the ``"missing_category"`` strategy.
    """

    stage = Stage.MISSING

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        strategy: str = AUTO,
        fill_value: Any = None,
        per_column: Optional[Dict[str, str]] = None,
        add_indicator: bool = False,
        missing_category: str = "__missing__",
    ) -> None:
        super().__init__(columns)
        self.strategy = strategy
        self.fill_value = fill_value
        self.per_column = dict(per_column) if per_column else None
        self.add_indicator = add_indicator
        self.missing_category = missing_category

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        # Only columns that actually have missing values at fit time.  A column with no
        # NaN in training may still have them at transform time, so the strategy is
        # resolved for every column and the *selection* only decides what to report.
        return [str(c) for c in X.columns]

    def _resolve_strategy(self, column: str, X: pd.DataFrame, context: FitContext) -> str:
        override = context.config.get_column(column)
        if override is not None and override.imputation is not None:
            return override.imputation
        if self.per_column and column in self.per_column:
            return self.per_column[column]
        if self.strategy != AUTO:
            return self.strategy

        cp = context.column_profile(column)
        series = X[column]
        if cp is not None:
            semantic = cp.semantic
            cardinality = cp.n_unique
        else:
            semantic = (
                SemanticType.NUMERIC
                if pd.api.types.is_numeric_dtype(series.dtype)
                else SemanticType.CATEGORICAL
            )
            cardinality = int(series.nunique(dropna=True))

        if semantic in (SemanticType.NUMERIC, SemanticType.ORDINAL):
            # Median, not mean: the usual own datasets are mostly right-skewed, and
            # the mean of a lognormal column is not a typical value.
            return "median"
        if semantic is SemanticType.DATETIME:
            return "median"
        if semantic in (SemanticType.CATEGORICAL, SemanticType.BINARY):
            if cardinality > context.config.effective_high_cardinality:
                # With hundreds of levels the mode covers a tiny share of the column,
                # so filling with it invents a spurious concentration.  An explicit
                # category is honest and keeps missingness learnable.
                return "missing_category"
            return "mode"
        return "missing_category"

    def _eligible_numeric_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        """All numeric columns in ``X`` usable as multivariate predictors (ordered)."""
        target = context.target
        eligible: List[str] = []
        for column in map(str, X.columns):
            if column == target:
                continue
            if not _is_numeric_imputation_column(X[column]):
                continue
            eligible.append(column)
        return eligible

    def _require_numeric_for_sklearn(self, column: str, series: pd.Series) -> None:
        if _is_numeric_imputation_column(series):
            return
        raise ConfigurationError(
            f"strategy='knn' and 'iterative' are only valid for numeric columns; "
            f"column {column!r} has dtype {series.dtype}. Use 'mode', "
            f"'missing_category', or 'median' as appropriate."
        )

    def _block_matrix(self, X: pd.DataFrame, block: Sequence[str]) -> np.ndarray:
        """Build a float matrix aligned with ``block`` for sklearn imputers."""
        n_rows = len(X)
        matrix = np.empty((n_rows, len(block)), dtype=np.float64)
        for j, column in enumerate(block):
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=np.float64)
            matrix[:, j] = values
        return matrix

    def _fit_sklearn_imputers(self, X: pd.DataFrame, context: FitContext) -> None:
        needs_knn = any(s == "knn" for s in self.strategies_.values())
        needs_iterative = any(s == "iterative" for s in self.strategies_.values())
        self.knn_impute_columns_: List[str] = [
            c for c in self.columns_ if self.strategies_.get(c) == "knn"
        ]
        self.iterative_impute_columns_: List[str] = [
            c for c in self.columns_ if self.strategies_.get(c) == "iterative"
        ]

        if not needs_knn and not needs_iterative:
            self.imputer_block_columns_: List[str] = []
            self.imputer_block_all_missing_: Set[str] = set()
            self.knn_imputer_ = None
            self.iterative_imputer_ = None
            return

        KNNImputer, IterativeImputer = _load_sklearn_imputers()
        eligible = self._eligible_numeric_columns(X, context)
        self.imputer_block_all_missing_ = {
            column
            for column in eligible
            if len(X) and int(X[column].isna().sum()) == len(X)
        }
        block = [column for column in eligible if column not in self.imputer_block_all_missing_]
        self.imputer_block_columns_ = block

        if not block:
            self.knn_imputer_ = None
            self.iterative_imputer_ = None
            return

        fit_matrix = self._block_matrix(X, block)

        if needs_knn:
            self.knn_imputer_ = KNNImputer()
            self.knn_imputer_.fit(fit_matrix)
        else:
            self.knn_imputer_ = None

        if needs_iterative:
            self.iterative_imputer_ = IterativeImputer(
                random_state=context.config.random_state,
                sample_posterior=False,
            )
            self.iterative_imputer_.fit(fit_matrix)
        else:
            self.iterative_imputer_ = None

    def _apply_sklearn_imputations(
        self,
        X: pd.DataFrame,
        replacements: Dict[str, pd.Series],
        filled_counts: Dict[str, int],
    ) -> None:
        block = self.imputer_block_columns_
        if not block:
            return

        transform_matrix = self._block_matrix(X, block)
        block_index = {name: idx for idx, name in enumerate(block)}

        for imputer, output_columns in (
            (self.knn_imputer_, self.knn_impute_columns_),
            (self.iterative_imputer_, self.iterative_impute_columns_),
        ):
            if imputer is None:
                continue
            imputed = imputer.transform(transform_matrix)
            for column in output_columns:
                if column not in X.columns or column not in block_index:
                    continue
                series = X[column]
                mask = series.isna()
                n_missing = int(mask.sum())
                if n_missing == 0:
                    continue
                col_idx = block_index[column]
                filled = series.copy()
                filled.loc[mask] = imputed[mask.to_numpy(), col_idx]
                replacements[column] = filled
                after = int(filled.isna().sum())
                filled_counts[column] = n_missing - after

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.strategies_: Dict[str, str] = {}
        self.fill_values_: Dict[str, Any] = {}
        self.dropped_columns_: List[str] = []

        thresholds = context.config.thresholds
        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                series = X[column]
                n_missing = int(series.isna().sum())
                missing_fraction = n_missing / len(X) if len(X) else 0.0

                strategy = self._resolve_strategy(column, X, context)
                self.strategies_[column] = strategy

                if strategy in ("none", "drop_rows"):
                    continue
                if strategy in _ROWWISE:
                    continue

                if strategy in _SKLEARN_IMPUTE:
                    self._require_numeric_for_sklearn(column, series)
                    if n_missing == len(X):
                        context.journal.warn(
                            "no_data_to_learn_fill",
                            f"Column {column!r} is entirely missing in the training data, "
                            f"so multivariate imputation cannot learn a fill for it. "
                            f"Missing values in this column will be left as-is.",
                            Severity.WARNING,
                            (column,),
                        )
                    continue

                if strategy == "constant":
                    if self.fill_value is None:
                        override = context.config.get_column(column)
                        value = override.imputation_fill_value if override is not None else None
                        if value is None:
                            raise ConfigurationError(
                                f"strategy='constant' for column {column!r} needs a fill "
                                f"value. Pass fill_value=... to MissingValueHandler, or "
                                f"set config.column({column!r}).imputation_fill_value."
                            )
                        self.fill_values_[column] = value
                    else:
                        self.fill_values_[column] = self.fill_value
                    continue

                if strategy == "missing_category":
                    self.fill_values_[column] = self.missing_category
                    continue

                self.fill_values_[column] = self._learn(series, strategy, column, context)

                if missing_fraction >= thresholds.missing_drop_threshold:
                    context.journal.warn(
                        "imputed_mostly_missing_column",
                        f"Column {column!r} is {missing_fraction:.1%} missing and is "
                        f"still being imputed with the {strategy}. Most of the column "
                        f"is now an invented constant. Consider dropping it, or set "
                        f"config.column({column!r}).drop = True.",
                        Severity.WARNING,
                        (column,),
                        {"missing_fraction": round(missing_fraction, 4)},
                    )

            self._fit_sklearn_imputers(X, context)

            timer.columns = list(self.columns_)
            timer.params = {"strategy": self.strategy}
            timer.effect = {
                "n_columns_with_fill": len(self.fill_values_),
                "strategies": dict(self.strategies_),
            }

    def _learn(self, series: pd.Series, strategy: str, column: str, context: FitContext) -> Any:
        clean = series.dropna()
        if clean.empty:
            context.journal.warn(
                "no_data_to_learn_fill",
                f"Column {column!r} is entirely missing in the training data, so no "
                f"{strategy} could be computed. Missing values in this column will be "
                f"left as-is; the column carries no information and is best dropped.",
                Severity.WARNING,
                (column,),
            )
            return None
        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series.dtype):
                raise ConfigurationError(
                    f"strategy='mean' is not valid for column {column!r} of dtype "
                    f"{series.dtype}. Use 'mode' or 'missing_category' for "
                    f"non-numeric columns."
                )
            return float(clean.mean())
        if strategy == "median":
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                return clean.median()
            if not pd.api.types.is_numeric_dtype(series.dtype):
                raise ConfigurationError(
                    f"strategy='median' is not valid for column {column!r} of dtype "
                    f"{series.dtype}. Use 'mode' or 'missing_category' instead."
                )
            return float(clean.median())
        if strategy == "mode":
            modes = clean.mode(dropna=True)
            # Ties are broken by taking the first, which pandas orders deterministically
            # by value, so refitting on identical data gives an identical answer.
            return modes.iloc[0] if len(modes) else None
        raise ConfigurationError.unknown_option(
            "imputation strategy",
            strategy,
            _valid_imputation_strategies(),
        )

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        added: Dict[str, pd.Series] = {}
        filled_counts: Dict[str, int] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "impute", "transform"
        ) as timer:
            self._apply_sklearn_imputations(X, replacements, filled_counts)

            for column in self.columns_:
                if column not in X.columns:
                    continue
                series = X[column]
                mask = series.isna()
                n_missing = int(mask.sum())

                strategy = self.strategies_.get(column, "none")
                if strategy in ("none", "drop_rows"):
                    continue

                if self.add_indicator and n_missing:
                    added[f"{column}__was_missing"] = mask.astype(np.int8)

                if strategy in _SKLEARN_IMPUTE:
                    continue

                if n_missing == 0:
                    continue

                if strategy == "ffill":
                    replacements[column] = series.ffill()
                elif strategy == "bfill":
                    replacements[column] = series.bfill()
                else:
                    value = self.fill_values_.get(column)
                    if value is None:
                        continue
                    replacements[column] = _fill(series, value)

                if column not in filled_counts:
                    after = (
                        int(replacements[column].isna().sum())
                        if column in replacements
                        else n_missing
                    )
                    filled_counts[column] = n_missing - after

            timer.columns = sorted(filled_counts)
            timer.effect = {
                "n_values_imputed": int(sum(filled_counts.values())),
                "per_column": filled_counts,
            }
        return self._rebuild(X, replacements, added)

    def _compute_feature_names_out(self) -> List[str]:
        names = list(self.feature_names_in_)
        if self.add_indicator:
            names += [f"{c}__was_missing" for c in self.columns_]
        return names


def _fill(series: pd.Series, value: Any) -> pd.Series:
    """Fill NaN with ``value``, extending a categorical dtype when necessary.

    ``fillna`` on a categorical raises if the value is not already a category, which is
    precisely the case for the explicit ``__missing__`` label.
    """
    if isinstance(series.dtype, pd.CategoricalDtype) and value not in series.cat.categories:
        return series.cat.add_categories([value]).fillna(value)
    return series.fillna(value)


class MissingIndicator(Transformer, ColumnTransformerMixin):
    """Add ``{column}__was_missing`` flags.

    Emitted by the planner *before* imputation, because missingness is frequently
    informative and imputation destroys it.  ``a census-income notebook`` established that
    ``workclass`` and ``occupation`` go missing together, then imputed both and lost
    the pattern.

    Parameters
    ----------
    threshold :
        Only columns whose training missing fraction is at least this get a flag.  A
        column with three missing values out of a million does not need one.
    """

    stage = Stage.MISSING_FLAG

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        threshold: float = 0.0,
        suffix: str = "__was_missing",
    ) -> None:
        super().__init__(columns)
        self.threshold = threshold
        self.suffix = suffix

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out = []
        for column in map(str, X.columns):
            if column == context.target:
                continue
            fraction = float(X[column].isna().mean()) if len(X) else 0.0
            if fraction > 0 and fraction >= self.threshold:
                out.append(column)
        return out

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.indicator_names_ = {c: f"{c}{self.suffix}" for c in self.columns_}
        context.journal.record(
            self.stage,
            type(self).__name__,
            "add_missing_indicators",
            "fit",
            columns=self.columns_,
            params={"threshold": self.threshold},
            effect={"n_indicators": len(self.columns_)},
        )

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        added = {
            self.indicator_names_[c]: X[c].isna().astype(np.int8)
            for c in self.columns_
            if c in X.columns
        }
        if added:
            context.journal.record(
                self.stage,
                type(self).__name__,
                "add_missing_indicators",
                "transform",
                columns=list(self.columns_),
                effect={
                    "n_indicators": len(added),
                    "n_flagged": {k: int(v.sum()) for k, v in added.items()},
                },
            )
        return self._rebuild(X, {}, added)

    def _compute_feature_names_out(self) -> List[str]:
        return list(self.feature_names_in_) + [self.indicator_names_[c] for c in self.columns_]
