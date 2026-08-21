"""Calendar feature expansion.

Notebook practice writes this block verbatim in three notebooks::

    df['dayofweek']  = df[date_col].dt.dayofweek
    df['month']      = df[date_col].dt.month
    df['quarter']    = df[date_col].dt.quarter
    df['year']       = df[date_col].dt.year
    df['dayofmonth'] = df[date_col].dt.day
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

Two changes.  First, only features that *vary* are emitted: expanding ``hour`` from a
column of pure dates produces a constant, and the design goal is explicit about not
generating hundreds of meaningless features.  Second, which features vary is decided at
fit time and frozen, so train and test get identical columns even if the test set
happens to span a single month.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError
from ..types import SemanticType, Stage

__all__ = ["DateTimeExpander", "AVAILABLE_FEATURES"]

#: name -> extractor.  Cyclical encodings are offered but not default: they only help
#: models that cannot represent wrap-around themselves, and they double the columns.
_EXTRACTORS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "year": lambda s: s.dt.year,
    "quarter": lambda s: s.dt.quarter,
    "month": lambda s: s.dt.month,
    "week": lambda s: s.dt.isocalendar().week.astype("float64"),
    "day": lambda s: s.dt.day,
    "dayofweek": lambda s: s.dt.dayofweek,
    "dayofyear": lambda s: s.dt.dayofyear,
    "hour": lambda s: s.dt.hour,
    "minute": lambda s: s.dt.minute,
    "second": lambda s: s.dt.second,
    "is_weekend": lambda s: (s.dt.dayofweek >= 5).astype("float64"),
    "is_month_start": lambda s: s.dt.is_month_start.astype("float64"),
    "is_month_end": lambda s: s.dt.is_month_end.astype("float64"),
    "is_quarter_start": lambda s: s.dt.is_quarter_start.astype("float64"),
    "is_quarter_end": lambda s: s.dt.is_quarter_end.astype("float64"),
    "is_year_start": lambda s: s.dt.is_year_start.astype("float64"),
    "is_year_end": lambda s: s.dt.is_year_end.astype("float64"),
    "month_sin": lambda s: np.sin(2 * np.pi * s.dt.month / 12.0),
    "month_cos": lambda s: np.cos(2 * np.pi * s.dt.month / 12.0),
    "dayofweek_sin": lambda s: np.sin(2 * np.pi * s.dt.dayofweek / 7.0),
    "dayofweek_cos": lambda s: np.cos(2 * np.pi * s.dt.dayofweek / 7.0),
    "hour_sin": lambda s: np.sin(2 * np.pi * s.dt.hour / 24.0),
    "hour_cos": lambda s: np.cos(2 * np.pi * s.dt.hour / 24.0),
}

AVAILABLE_FEATURES = tuple(_EXTRACTORS)

#: The usual own set, which is a sensible default.
DEFAULT_FEATURES = (
    "year",
    "quarter",
    "month",
    "day",
    "dayofweek",
    "hour",
    "is_weekend",
)


class DateTimeExpander(Transformer, ColumnTransformerMixin):
    """Expand datetime columns into calendar features.

    Parameters
    ----------
    features :
        Which features to attempt.  Defaults to the usual set.  Any name in
        :data:`AVAILABLE_FEATURES` is accepted, including cyclical encodings.
    drop_original :
        Drop the source datetime column afterwards.  True by default: a raw
        ``datetime64`` cannot be consumed by a model, and keeping it produces a column
        that every downstream step has to special-case.
    drop_constant :
        Skip features that take a single value across the training data.
    reference :
        If given, also emit ``{column}__days_since_reference``.  Useful for turning an
        absolute date into a duration, which usually generalises better.
    """

    stage = Stage.DATETIME

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        features: Sequence[str] = DEFAULT_FEATURES,
        drop_original: bool = True,
        drop_constant: bool = True,
        reference: Optional[str] = None,
    ) -> None:
        super().__init__(columns)
        self.features = tuple(features)
        self.drop_original = drop_original
        self.drop_constant = drop_constant
        self.reference = reference

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out: List[str] = []
        for name in map(str, X.columns):
            if name == context.target:
                continue
            cp = context.column_profile(name)
            if cp is not None:
                if cp.semantic is SemanticType.DATETIME:
                    out.append(name)
                continue
            if pd.api.types.is_datetime64_any_dtype(X[name].dtype):
                out.append(name)
        return out

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        unknown = [f for f in self.features if f not in _EXTRACTORS]
        if unknown:
            raise ConfigurationError.unknown_option(
                "datetime feature", unknown[0], AVAILABLE_FEATURES
            )

        self.emitted_: Dict[str, List[str]] = {}
        self.reference_ts_: Optional[pd.Timestamp] = (
            pd.Timestamp(self.reference) if self.reference else None
        )

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                override = context.config.get_column(column)
                wanted = (
                    tuple(override.datetime_features)
                    if override is not None and override.datetime_features
                    else self.features
                )
                series = _as_datetime(X[column])
                kept: List[str] = []
                for feature in wanted:
                    values = _EXTRACTORS[feature](series)
                    # Which features vary is decided here and frozen.  Deciding it per
                    # frame would give the test set different columns from the train
                    # set whenever it happens to span less time.
                    if self.drop_constant and int(values.nunique(dropna=True)) <= 1:
                        continue
                    kept.append(feature)
                self.emitted_[column] = kept

            timer.columns = list(self.columns_)
            timer.params = {"features": list(self.features)}
            timer.effect = {
                "emitted": {c: list(v) for c, v in self.emitted_.items()},
                "n_new_columns": sum(len(v) for v in self.emitted_.values())
                + (len(self.emitted_) if self.reference_ts_ else 0),
            }

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        added: Dict[str, pd.Series] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "expand_datetime", "transform"
        ) as timer:
            for column, features in self.emitted_.items():
                if column not in X.columns:
                    continue
                series = _as_datetime(X[column])
                for feature in features:
                    added[f"{column}__{feature}"] = _EXTRACTORS[feature](series).astype(
                        "float64"
                    )
                if self.reference_ts_ is not None:
                    delta = (series - self.reference_ts_).dt.total_seconds() / 86400.0
                    added[f"{column}__days_since_reference"] = delta

            timer.columns = list(self.emitted_)
            timer.effect = {"n_new_columns": len(added)}

        keep = {
            str(c): X[c]
            for c in X.columns
            if not (self.drop_original and str(c) in self.emitted_)
        }
        return pd.DataFrame({**keep, **added}, index=X.index, copy=False)

    def _compute_feature_names_out(self) -> List[str]:
        names = [
            c
            for c in self.feature_names_in_
            if not (self.drop_original and c in self.emitted_)
        ]
        for column, features in self.emitted_.items():
            names.extend(f"{column}__{f}" for f in features)
            if self.reference_ts_ is not None:
                names.append(f"{column}__days_since_reference")
        return names


def _as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")
