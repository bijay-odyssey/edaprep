"""Categorical encoding.

Two corrections to notebook practice (docs/design-rationale.md 6.3 and section 13 of the design goal):

* **Target encoding is cross-fitted.**  ``category_encoders.TargetEncoder`` inside a
  ``ColumnTransformer`` is train/test-safe but not *row*-safe: a training row's encoded
  value includes that row's own target.  With near-singleton categories the encoding
  approaches the target itself and the model memorises it.  :class:`TargetEncoder` here
  computes the training output from out-of-fold statistics while storing the full-train
  mapping for ``transform``.
* **One-hot encoding is refused, not attempted, on high-cardinality columns.**  The
  notebook practice one-hot encodes whatever arrives; a column with 300 levels silently becomes
  300 columns, and one with 300,000 exhausts memory.

Unseen categories at transform time are handled explicitly rather than crashing or
producing NaN by accident, since a category present only in the test set is the normal
case, not an exception.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AUTO
from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError, LeakageError
from ..types import CATEGORICAL_LIKE, ModelFamily, SemanticType, Severity, Stage

__all__ = [
    "RareCategoryGrouper",
    "OneHotEncoder",
    "OrdinalEncoder",
    "FrequencyEncoder",
    "TargetEncoder",
    "CategoricalEncoder",
    "resolve_encoding",
]

#: Value written for a category never seen during fit, in ordinal encoding.
UNKNOWN_CODE = -1


def resolve_encoding(
    cardinality: int,
    model_family: Optional[ModelFamily],
    high_cardinality: int,
    extreme_cardinality: int,
    has_target: bool,
) -> str:
    """Choose an encoding from cardinality and the consuming model family.

    Mined from notebook practice (docs/design-rationale.md, axis 2), extended with the cardinality
    ceilings it lacked.
    """
    if cardinality <= 2:
        return "ordinal"  # binary: one column either way, no false ordering possible
    if model_family is ModelFamily.TREE:
        # Trees split on thresholds, so an arbitrary integer ordering costs nothing:
        # the tree can isolate any subset of codes with enough splits.
        return "ordinal"
    if cardinality > extreme_cardinality:
        return "frequency"
    if cardinality > high_cardinality:
        return "target" if has_target else "frequency"
    return "onehot"


class _CategoricalBase(Transformer, ColumnTransformerMixin):
    """Shared column selection for categorical transformers."""

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out: List[str] = []
        for name in map(str, X.columns):
            if name == context.target:
                continue
            cp = context.column_profile(name)
            if cp is not None:
                if cp.semantic in CATEGORICAL_LIKE:
                    out.append(name)
                continue
            dtype = X[name].dtype
            if (
                pd.api.types.is_object_dtype(dtype)
                or isinstance(dtype, (pd.CategoricalDtype, pd.StringDtype))
                or pd.api.types.is_bool_dtype(dtype)
            ):
                out.append(name)
        return out

    @staticmethod
    def _as_object(series: pd.Series) -> pd.Series:
        """Normalise to a comparable object Series with NaN preserved.

        Categorical dtypes, string dtypes and booleans all need to hash the same way as
        the values stored in the fitted mapping, or a category learned at fit time will
        not match itself at transform time.
        """
        if isinstance(series.dtype, pd.CategoricalDtype):
            return series.astype(object)
        if pd.api.types.is_bool_dtype(series.dtype):
            return series.astype(object)
        if isinstance(series.dtype, pd.StringDtype):
            return series.astype(object)
        return series


class RareCategoryGrouper(_CategoricalBase):
    """Collapse infrequent categories into a single bucket.

    Rare levels are noise: a category seen 3 times in 100,000 rows cannot support a
    reliable estimate, and one-hot encoding it adds a column that is almost always
    zero.  Grouping them also makes the encoder robust to unseen categories, since an
    unseen level is naturally routed to the same bucket.
    """

    stage = Stage.RARE_CATEGORY

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        min_frequency: Optional[int] = None,
        other_label: str = "__rare__",
        max_categories: Optional[int] = None,
    ) -> None:
        super().__init__(columns)
        self.threshold = threshold
        self.min_frequency = min_frequency
        self.other_label = other_label
        self.max_categories = max_categories

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.kept_categories_: Dict[str, set] = {}
        self.n_grouped_: Dict[str, int] = {}
        n_rows = len(X)

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                override = context.config.get_column(column)
                threshold = self.threshold
                if override is not None and override.rare_category_threshold is not None:
                    threshold = override.rare_category_threshold
                if threshold is None:
                    threshold = context.config.effective_rare_threshold

                counts = self._as_object(X[column]).value_counts(dropna=True)
                floor = (
                    self.min_frequency
                    if self.min_frequency is not None
                    else max(1, int(np.ceil(threshold * n_rows)))
                )
                kept = counts[counts >= floor]
                if self.max_categories is not None and len(kept) > self.max_categories:
                    kept = kept.head(self.max_categories)

                self.kept_categories_[column] = set(kept.index)
                self.n_grouped_[column] = int(len(counts) - len(kept))

            timer.columns = list(self.columns_)
            timer.params = {"threshold": self.threshold, "other_label": self.other_label}
            timer.effect = {
                "n_categories_grouped": dict(self.n_grouped_),
                "n_categories_kept": {
                    c: len(v) for c, v in self.kept_categories_.items()
                },
            }

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        affected: Dict[str, int] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "group_rare", "transform"
        ) as timer:
            for column, kept in self.kept_categories_.items():
                if column not in X.columns:
                    continue
                series = self._as_object(X[column])
                mask = series.notna() & ~series.isin(kept)
                n = int(mask.sum())
                if n == 0:
                    continue
                affected[column] = n
                replacements[column] = series.mask(mask, self.other_label)

            timer.columns = sorted(affected)
            timer.effect = {"n_values_grouped": affected}
        return self._rebuild(X, replacements)


class OneHotEncoder(_CategoricalBase):
    """Expand each category into an indicator column.

    Parameters
    ----------
    drop_first :
        Drop one level per column to avoid the dummy-variable trap.  Matters for
        linear models with an intercept; harmless but wasteful otherwise.
    max_columns :
        Refuse to run if the expansion would add more than this many columns.  The
        error names the offending columns and suggests frequency or target encoding.
    handle_unknown :
        ``"ignore"`` gives an unseen category all-zero indicators, which is the only
        sensible answer; ``"error"`` raises.
    """

    stage = Stage.ENCODE

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        drop_first: bool = False,
        max_columns: Optional[int] = None,
        handle_unknown: str = "ignore",
        dtype: str = "int8",
    ) -> None:
        super().__init__(columns)
        self.drop_first = drop_first
        self.max_columns = max_columns
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.categories_: Dict[str, List[Any]] = {}
        self.output_names_: Dict[str, List[str]] = {}

        cap = (
            self.max_columns
            if self.max_columns is not None
            else context.config.thresholds.max_onehot_columns
        )

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            total = 0
            offenders: List[str] = []
            for column in self.columns_:
                series = self._as_object(X[column])
                # Sorted for determinism: refitting on the same data must produce the
                # same column order, or downstream feature indices shift silently.
                categories = sorted(series.dropna().unique(), key=_sort_key)
                if self.drop_first and len(categories) > 1:
                    categories = categories[1:]
                self.categories_[column] = categories
                self.output_names_[column] = [f"{column}_{_label(c)}" for c in categories]
                total += len(categories)
                if len(categories) > context.config.effective_high_cardinality:
                    offenders.append(f"{column} ({len(categories)} levels)")

            if total > cap:
                raise ConfigurationError(
                    f"One-hot encoding these columns would add {total} columns, above "
                    f"the {cap}-column ceiling. The worst offenders are: "
                    f"{', '.join(offenders[:5]) or 'none individually'}. Use frequency "
                    f"or target encoding for high-cardinality columns, group rare "
                    f"levels first with RareCategoryGrouper, or raise max_columns "
                    f"(Thresholds.max_onehot_columns) if the expansion is intended."
                )

            timer.columns = list(self.columns_)
            timer.params = {"drop_first": self.drop_first}
            timer.effect = {"n_output_columns": total}

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        added: Dict[str, pd.Series] = {}
        unknown_counts: Dict[str, int] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "onehot", "transform"
        ) as timer:
            for column, categories in self.categories_.items():
                if column not in X.columns:
                    continue
                series = self._as_object(X[column])

                # One factorize, then integer comparisons.  Comparing an object array
                # against each category directly (`values == category`) makes NumPy
                # fall back to elementwise Python comparison, which is slow and raises
                # a FutureWarning whenever the array mixes types -- exactly the case for
                # an integer-coded column held as object.
                codes, uniques = pd.factorize(series, use_na_sentinel=True)
                position = {value: i for i, value in enumerate(uniques)}
                # -2 is unreachable as a factorize code, so a category absent from this
                # frame yields an all-zero indicator rather than matching anything.
                wanted_codes = [position.get(c, -2) for c in categories]

                known_codes = {c for c in wanted_codes if c >= 0}
                n_unknown = int(
                    np.count_nonzero(
                        (codes >= 0) & ~np.isin(codes, list(known_codes) or [-2])
                    )
                )
                if n_unknown:
                    unknown_counts[column] = n_unknown
                    if self.handle_unknown == "error":
                        unseen = sorted(
                            {u for i, u in enumerate(uniques) if i not in known_codes},
                            key=_sort_key,
                        )[:5]
                        raise ValueError(
                            f"Column {column!r} contains {n_unknown} value(s) not seen "
                            f"during fit, for example {unseen}. Set "
                            f"handle_unknown='ignore' to encode them as all-zero "
                            f"indicators, or group rare levels before encoding."
                        )

                for code, name in zip(wanted_codes, self.output_names_[column]):
                    added[name] = pd.Series(
                        (codes == code).astype(self.dtype), index=X.index, name=name
                    )

            if unknown_counts:
                context.journal.warn(
                    "unseen_categories",
                    f"{sum(unknown_counts.values())} value(s) across "
                    f"{len(unknown_counts)} column(s) were not present at fit time and "
                    f"were encoded as all-zero indicators: "
                    f"{', '.join(sorted(unknown_counts))}.",
                    Severity.INFO,
                    tuple(unknown_counts),
                    unknown_counts,
                )

            timer.columns = list(self.categories_)
            timer.effect = {
                "n_output_columns": len(added),
                "n_unknown_values": unknown_counts,
            }

        remaining = {
            str(c): X[c] for c in X.columns if str(c) not in self.categories_
        }
        return pd.DataFrame({**remaining, **added}, index=X.index, copy=False)

    def _compute_feature_names_out(self) -> List[str]:
        out = [c for c in self.feature_names_in_ if c not in self.categories_]
        for column in self.categories_:
            out.extend(self.output_names_[column])
        return out


class OrdinalEncoder(_CategoricalBase):
    """Map categories to integer codes.

    Appropriate for tree ensembles, which split on thresholds and can isolate any
    subset of codes, and for genuinely ordered columns.  It imposes a false ordering on
    a nominal column fed to a linear or distance-based model, which is why the planner
    only selects it for ``model_family="tree"`` or ``SemanticType.ORDINAL``.
    """

    stage = Stage.ENCODE

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        unknown_value: int = UNKNOWN_CODE,
        categories: Optional[Dict[str, Sequence[Any]]] = None,
        dtype: str = "int32",
    ) -> None:
        super().__init__(columns)
        self.unknown_value = unknown_value
        self.categories = categories
        self.dtype = dtype

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.mappings_: Dict[str, Dict[Any, int]] = {}
        self.categories_: Dict[str, List[Any]] = {}

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                if self.categories and column in self.categories:
                    ordered = list(self.categories[column])  # user-supplied ordering
                else:
                    series = self._as_object(X[column])
                    if isinstance(X[column].dtype, pd.CategoricalDtype) and X[
                        column
                    ].dtype.ordered:
                        # An ordered categorical already states the ordering; honouring
                        # it is the whole point of the dtype.
                        ordered = list(X[column].cat.categories)
                    else:
                        ordered = sorted(series.dropna().unique(), key=_sort_key)
                self.categories_[column] = ordered
                self.mappings_[column] = {c: i for i, c in enumerate(ordered)}

            timer.columns = list(self.columns_)
            timer.effect = {
                "n_categories": {c: len(v) for c, v in self.categories_.items()}
            }

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        unknown_counts: Dict[str, int] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "ordinal", "transform"
        ) as timer:
            for column, mapping in self.mappings_.items():
                if column not in X.columns:
                    continue
                series = self._as_object(X[column])
                codes = series.map(mapping)
                unknown = series.notna() & codes.isna()
                n_unknown = int(unknown.sum())
                if n_unknown:
                    unknown_counts[column] = n_unknown
                # NaN in the input stays NaN so a later imputer can see it; only
                # genuinely unseen categories get the sentinel code.
                codes = codes.mask(unknown, self.unknown_value)
                replacements[column] = codes.astype("float64")

            if unknown_counts:
                context.journal.warn(
                    "unseen_categories",
                    f"{sum(unknown_counts.values())} value(s) were not seen at fit time "
                    f"and were coded as {self.unknown_value}: "
                    f"{', '.join(sorted(unknown_counts))}.",
                    Severity.INFO,
                    tuple(unknown_counts),
                    unknown_counts,
                )

            timer.columns = sorted(replacements)
            timer.effect = {"n_unknown_values": unknown_counts}
        return self._rebuild(X, replacements)


class FrequencyEncoder(_CategoricalBase):
    """Replace each category by how often it occurs in the training data.

    The right default for high-cardinality columns: one output column regardless of
    cardinality, no target involved (so no leakage risk at all), and unseen categories
    map naturally to zero -- which is the truth, since they were never seen.
    """

    stage = Stage.ENCODE

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        normalize: bool = True,
        unseen_value: float = 0.0,
    ) -> None:
        super().__init__(columns)
        self.normalize = normalize
        self.unseen_value = unseen_value

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.frequencies_: Dict[str, Dict[Any, float]] = {}

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                counts = self._as_object(X[column]).value_counts(
                    dropna=True, normalize=self.normalize
                )
                self.frequencies_[column] = counts.to_dict()
            timer.columns = list(self.columns_)
            timer.params = {"normalize": self.normalize}
            timer.effect = {
                "n_categories": {c: len(v) for c, v in self.frequencies_.items()}
            }

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        with context.journal.timer(
            self.stage, type(self).__name__, "frequency", "transform"
        ) as timer:
            for column, mapping in self.frequencies_.items():
                if column not in X.columns:
                    continue
                series = self._as_object(X[column])
                encoded = series.map(mapping)
                encoded = encoded.mask(series.notna() & encoded.isna(), self.unseen_value)
                replacements[column] = encoded.astype("float64")
            timer.columns = sorted(replacements)
            timer.effect = {"n_columns": len(replacements)}
        return self._rebuild(X, replacements)


class TargetEncoder(_CategoricalBase):
    """Replace each category by a smoothed estimate of the target mean, cross-fitted.

    The usual version leaks within the fold: a training row's encoded value includes
    that row's own target, so a near-singleton category encodes the answer.  Here:

    * ``fit`` learns the full-train mapping, used by every later ``transform``;
    * ``fit_transform`` returns *out-of-fold* values -- each row is encoded from the
      folds that do not contain it -- so a model trained on the output cannot see its
      own label through the encoding.

    That difference is why :attr:`cross_fitted` is True, and why
    ``fit_transform(X, y)`` legitimately differs from ``fit(X, y).transform(X)``.

    Smoothing is the m-estimate::

        encoding(c) = (sum_c + prior * m) / (n_c + m)

    so a category with few observations is pulled towards the global mean.  ``m`` is
    ``Config.target_encoding_smoothing``.
    """

    stage = Stage.ENCODE
    uses_target = True
    cross_fitted = True

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_folds: Optional[int] = None,
        smoothing: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(columns)
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state

    def _numeric_target(self, y: pd.Series, context: FitContext) -> np.ndarray:
        """Targets must be numeric to average.

        Binary/categorical targets are mapped to 0/1 against the *rarer* class, which
        is the convention that makes the encoding read as "probability of the positive
        class" for imbalanced problems.
        """
        if pd.api.types.is_numeric_dtype(y.dtype) and not pd.api.types.is_bool_dtype(
            y.dtype
        ):
            return y.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        counts = y.value_counts(dropna=True)
        if len(counts) > 2:
            raise ConfigurationError(
                f"Target encoding needs a numeric or binary target, but the target has "
                f"{len(counts)} distinct non-numeric values. For multiclass problems, "
                f"use one-hot or frequency encoding instead."
            )
        positive = counts.index[-1]  # rarest class
        return (y == positive).to_numpy(dtype=np.float64)

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        if y is None:
            raise LeakageError.target_required(type(self).__name__)
        target = self._numeric_target(y, context)
        smoothing = (
            self.smoothing
            if self.smoothing is not None
            else context.config.target_encoding_smoothing
        )
        self.prior_ = float(np.nanmean(target)) if np.isfinite(target).any() else 0.0
        self.smoothing_ = float(smoothing)
        self.mappings_: Dict[str, Dict[Any, float]] = {}

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                self.mappings_[column] = _smoothed_means(
                    self._as_object(X[column]), target, self.prior_, self.smoothing_
                )
            timer.columns = list(self.columns_)
            timer.params = {"smoothing": self.smoothing_, "prior": self.prior_}
            timer.effect = {"n_categories": {c: len(m) for c, m in self.mappings_.items()}}

    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext
    ) -> pd.DataFrame:
        self._fit(X, y, context)
        assert y is not None  # _fit raises otherwise
        target = self._numeric_target(y, context)
        n_folds = self.n_folds or context.config.target_encoding_folds
        n_folds = int(min(n_folds, max(2, len(X))))

        seed = self.random_state if self.random_state is not None else context.random_state
        rng = np.random.default_rng(seed)
        folds = rng.permutation(len(X)) % n_folds

        replacements: Dict[str, pd.Series] = {}
        with context.journal.timer(
            self.stage, type(self).__name__, "target_encode_oof", "fit"
        ) as timer:
            for column in self.columns_:
                series = self._as_object(X[column])
                out = np.full(len(X), self.prior_, dtype=np.float64)
                for fold in range(n_folds):
                    holdout = folds == fold
                    rest = ~holdout
                    if not rest.any():
                        continue
                    mapping = _smoothed_means(
                        series[rest], target[rest], self.prior_, self.smoothing_
                    )
                    encoded = series[holdout].map(mapping)
                    out[holdout] = encoded.fillna(self.prior_).to_numpy(dtype=np.float64)
                replacements[column] = pd.Series(out, index=X.index, name=column)
            timer.columns = list(self.columns_)
            timer.params = {"n_folds": n_folds, "random_state": seed}
            timer.effect = {
                "note": "training rows encoded out-of-fold to prevent within-fold leakage"
            }
        return self._rebuild(X, replacements)

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        with context.journal.timer(
            self.stage, type(self).__name__, "target_encode", "transform"
        ) as timer:
            for column, mapping in self.mappings_.items():
                if column not in X.columns:
                    continue
                series = self._as_object(X[column])
                encoded = series.map(mapping).astype("float64")
                # Unseen categories get the prior, which is the best available estimate
                # in the absence of any category-specific evidence.
                replacements[column] = encoded.fillna(self.prior_)
            timer.columns = sorted(replacements)
            timer.effect = {"prior": self.prior_}
        return self._rebuild(X, replacements)


def _smoothed_means(
    series: pd.Series, target: np.ndarray, prior: float, smoothing: float
) -> Dict[Any, float]:
    """m-estimate smoothed target means per category."""
    frame = pd.DataFrame({"c": series.to_numpy(), "y": target})
    grouped = frame.groupby("c", observed=True, dropna=True)["y"].agg(["sum", "count"])
    smoothed = (grouped["sum"] + prior * smoothing) / (grouped["count"] + smoothing)
    return smoothed.to_dict()


def _sort_key(value: Any) -> tuple:
    """Total order across mixed types, so category ordering is always deterministic."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (2, float(value))
    return (3, str(value))


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class CategoricalEncoder(_CategoricalBase):
    """Route each categorical column to the encoder its cardinality warrants.

    This is the user-facing entry point.  It holds one sub-encoder per strategy and
    dispatches columns to them, so the planner can express "one-hot these three,
    frequency-encode that one" as a single step.
    """

    stage = Stage.ENCODE

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        strategy: str = AUTO,
        per_column: Optional[Dict[str, str]] = None,
        drop_first: bool = False,
        handle_unknown: str = "ignore",
    ) -> None:
        super().__init__(columns)
        self.strategy = strategy
        self.per_column = dict(per_column) if per_column else None
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown

    @property
    def uses_target(self) -> bool:  # type: ignore[override]
        """True only when at least one column is actually target-encoded."""
        assignments = getattr(self, "assignments_", None)
        if assignments is None:
            return self.strategy == "target" or bool(
                self.per_column and "target" in self.per_column.values()
            )
        return "target" in assignments.values()

    def _resolve(self, column: str, cardinality: int, context: FitContext) -> str:
        override = context.config.get_column(column)
        if override is not None and override.encoding is not None:
            return override.encoding
        if self.per_column and column in self.per_column:
            return self.per_column[column]
        if self.strategy != AUTO:
            return self.strategy
        if context.config.categorical_encoding != AUTO:
            return context.config.categorical_encoding
        return resolve_encoding(
            cardinality,
            context.model_family,
            context.config.effective_high_cardinality,
            context.config.thresholds.extreme_cardinality_threshold,
            has_target=context.target is not None,
        )

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.assignments_: Dict[str, str] = {}
        for column in self.columns_:
            cp = context.column_profile(column)
            cardinality = (
                cp.n_unique if cp is not None else int(X[column].nunique(dropna=True))
            )
            if cp is not None and cp.semantic is SemanticType.ORDINAL:
                self.assignments_[column] = "ordinal"
                continue
            self.assignments_[column] = self._resolve(column, cardinality, context)

        if "target" in self.assignments_.values() and y is None:
            raise LeakageError.target_required(f"{type(self).__name__} (target encoding)")

        self.encoders_: Dict[str, Transformer] = {}
        by_strategy: Dict[str, List[str]] = {}
        for column, strategy in self.assignments_.items():
            if strategy in ("none", "drop"):
                continue
            by_strategy.setdefault(strategy, []).append(column)

        for strategy, cols in by_strategy.items():
            encoder = self._make(strategy, cols)
            encoder.fit(X, y, context)
            self.encoders_[strategy] = encoder

        self.dropped_ = [c for c, s in self.assignments_.items() if s == "drop"]
        context.journal.record(
            self.stage,
            type(self).__name__,
            "assign_encoders",
            "fit",
            columns=list(self.columns_),
            effect={"assignments": dict(self.assignments_)},
        )

    def _make(self, strategy: str, cols: List[str]) -> Transformer:
        if strategy == "onehot":
            return OneHotEncoder(
                cols, drop_first=self.drop_first, handle_unknown=self.handle_unknown
            )
        if strategy == "ordinal":
            return OrdinalEncoder(cols)
        if strategy in ("frequency", "count"):
            return FrequencyEncoder(cols, normalize=(strategy == "frequency"))
        if strategy == "target":
            return TargetEncoder(cols)
        if strategy == "binary":
            raise ConfigurationError(
                "encoding='binary' is not implemented in this version. Use 'frequency' "
                "or 'target' for high-cardinality columns; both produce a single "
                "column and are better understood."
            )
        raise ConfigurationError.unknown_option(
            "encoding", strategy, ["onehot", "ordinal", "frequency", "count", "target"]
        )

    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext
    ) -> pd.DataFrame:
        # Target encoding must go through its own out-of-fold path; every other encoder
        # is order-independent.  Children are driven through the private hooks: they
        # are chained, so each one sees columns the previous one added, and the public
        # wrapper's schema check would reject those as "unseen at fit time".  The
        # composite has already validated the caller's frame.
        self._fit(X, y, context)
        out = X
        for strategy, encoder in self.encoders_.items():
            if strategy == "target":
                out = encoder._fit_transform(out, y, context)
                encoder._is_fitted = True
            else:
                out = encoder._transform(out, context)
        return self._drop(out)

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        out = X
        for encoder in self.encoders_.values():
            out = encoder._transform(out, context)
        return self._drop(out)

    def _drop(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.dropped_:
            return X
        keep = [c for c in X.columns if str(c) not in self.dropped_]
        return X[keep]

    def _compute_feature_names_out(self) -> List[str]:
        # Must match _transform exactly.  One-hot appends its indicator columns at the
        # end rather than expanding in place (expanding in place would mean rebuilding
        # the frame around each encoded column), so the names have to be appended too.
        # Ordinal, frequency and target encoding all replace their column in position.
        onehot = self.encoders_.get("onehot")
        encoded_away = set(getattr(onehot, "categories_", {})) if onehot else set()
        names = [
            c
            for c in self.feature_names_in_
            if c not in self.dropped_ and c not in encoded_away
        ]
        if onehot is not None:
            for column in onehot.categories_:  # type: ignore[attr-defined]
                names.extend(onehot.output_names_[column])  # type: ignore[attr-defined]
        return names
