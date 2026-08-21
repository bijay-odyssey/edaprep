"""Dtype correction and memory downcasting.

Two jobs:

* **Correction.** Replace sentinel strings with NaN, parse numeric-looking and
  date-looking text columns, and strip stray whitespace.  This is the ``(df == '?')``
  scan from ``a census-income notebook``, generalised and made part of the pipeline instead of
  a one-off cell.
* **Downcasting.** Optionally narrow ``int64`` to the smallest safe integer width and
  ``float64`` to ``float32``.  Off by default: ``float32`` halves memory but also
  halves precision, and silently changing the numerical properties of a user's data is
  exactly the kind of magic the design goal forbids.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..types import SemanticType, Severity, Stage

__all__ = ["DataTypeInference"]


class DataTypeInference(Transformer, ColumnTransformerMixin):
    """Correct dtypes and, optionally, shrink them.

    Parameters
    ----------
    replace_sentinels :
        Replace placeholder strings (``"?"``, ``"N/A"``, ...) with NaN.  Only string
        columns are touched; numeric sentinels such as ``-999`` are reported by the
        profiler and never replaced, because the value may be legitimate.
    parse_numeric, parse_datetime :
        Cast text columns the profiler identified as numeric or datetime.
    strip_whitespace :
        Strip leading/trailing whitespace from string columns.  ``"Yes"`` and
        ``"Yes "`` are otherwise two categories.
    downcast_integers, downcast_floats :
        Narrow numeric dtypes.  ``downcast_floats`` is lossy and off by default.
    """

    stage = Stage.CAST

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        replace_sentinels: bool = True,
        parse_numeric: bool = True,
        parse_datetime: bool = True,
        strip_whitespace: bool = True,
        downcast_integers: bool = False,
        downcast_floats: bool = False,
        sentinels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(columns)
        self.replace_sentinels = replace_sentinels
        self.parse_numeric = parse_numeric
        self.parse_datetime = parse_datetime
        self.strip_whitespace = strip_whitespace
        self.downcast_integers = downcast_integers
        self.downcast_floats = downcast_floats
        self.sentinels = list(sentinels) if sentinels is not None else None

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self.actions_: Dict[str, List[str]] = {}
        self.target_dtypes_: Dict[str, str] = {}
        self.sentinel_set_: set = set()

        config = context.config
        if self.replace_sentinels and config.replace_sentinels:
            source = self.sentinels if self.sentinels is not None else config.sentinels
            self.sentinel_set_ = {str(s).strip().lower() for s in source}

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            for column in self.columns_:
                actions: List[str] = []
                series = X[column]
                cp = context.column_profile(column)
                is_stringy = series.dtype == object or isinstance(
                    series.dtype, pd.StringDtype
                )

                if is_stringy and self.strip_whitespace:
                    actions.append("strip")
                if is_stringy and self.sentinel_set_:
                    actions.append("sentinels_to_nan")

                if cp is not None and cp.suggested_dtype and is_stringy:
                    if cp.suggested_dtype == "float64" and self.parse_numeric:
                        actions.append("to_numeric")
                        self.target_dtypes_[column] = "float64"
                    elif cp.suggested_dtype.startswith("datetime") and self.parse_datetime:
                        actions.append("to_datetime")
                        self.target_dtypes_[column] = "datetime64[ns]"
                    elif cp.suggested_dtype == "boolean":
                        actions.append("to_boolean")
                        self.target_dtypes_[column] = "boolean"

                if self.downcast_integers and pd.api.types.is_integer_dtype(series.dtype):
                    narrowed = _narrow_integer(series)
                    if narrowed != str(series.dtype):
                        actions.append("downcast_int")
                        self.target_dtypes_[column] = narrowed
                if (
                    self.downcast_floats
                    and pd.api.types.is_float_dtype(series.dtype)
                    and str(series.dtype) == "float64"
                    and _float32_safe(series)
                ):
                    actions.append("downcast_float")
                    self.target_dtypes_[column] = "float32"

                if actions:
                    self.actions_[column] = actions

            if self.downcast_floats:
                context.journal.warn(
                    "float_downcast_is_lossy",
                    "downcast_floats=True converts float64 columns to float32, halving "
                    "memory and precision. Values beyond float32's ~7 significant "
                    "digits are rounded. Columns whose range does not fit float32 are "
                    "left alone.",
                    Severity.INFO,
                )

            timer.columns = list(self.actions_)
            timer.effect = {"actions": dict(self.actions_)}

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        replacements: Dict[str, pd.Series] = {}
        effects: Dict[str, Dict[str, int]] = {}

        with context.journal.timer(
            self.stage, type(self).__name__, "cast", "transform"
        ) as timer:
            for column, actions in self.actions_.items():
                if column not in X.columns:
                    continue
                series = X[column]
                counts: Dict[str, int] = {}

                if "strip" in actions:
                    stripped = series.astype(object).map(
                        lambda v: v.strip() if isinstance(v, str) else v
                    )
                    counts["stripped"] = int((stripped != series).sum())
                    series = stripped

                if "sentinels_to_nan" in actions:
                    as_str = series.astype(object)
                    mask = as_str.map(
                        lambda v: isinstance(v, str) and v.strip().lower() in self.sentinel_set_
                    )
                    n = int(mask.sum())
                    if n:
                        counts["sentinels_to_nan"] = n
                        series = series.mask(mask.astype(bool))

                target = self.target_dtypes_.get(column)
                if "to_numeric" in actions:
                    parsed = pd.to_numeric(series, errors="coerce")
                    counts["failed_to_parse"] = int(
                        (series.notna() & parsed.isna()).sum()
                    )
                    series = parsed
                elif "to_datetime" in actions:
                    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                    counts["failed_to_parse"] = int(
                        (series.notna() & parsed.isna()).sum()
                    )
                    series = parsed
                elif "to_boolean" in actions:
                    series = _to_boolean(series)
                elif target is not None:
                    series = series.astype(target)

                replacements[column] = series
                if counts:
                    effects[column] = counts

            for column, counts in effects.items():
                failed = counts.get("failed_to_parse", 0)
                if failed:
                    context.journal.warn(
                        "parse_failures",
                        f"{failed} value(s) in column {column!r} could not be parsed "
                        f"into {self.target_dtypes_.get(column)} and became missing. At "
                        f"fit time every value parsed, so these are new. Inspect them "
                        f"before trusting the imputation that follows.",
                        Severity.WARNING,
                        (column,),
                        {"n_failed": failed},
                    )

            timer.columns = sorted(replacements)
            timer.effect = {"per_column": effects}
        return self._rebuild(X, replacements)


def _narrow_integer(series: pd.Series) -> str:
    """Smallest integer dtype that holds this column's range without loss."""
    clean = series.dropna()
    if clean.empty:
        return str(series.dtype)
    lo, hi = int(clean.min()), int(clean.max())
    candidates = (
        [("int8", -128, 127), ("int16", -32768, 32767), ("int32", -2147483648, 2147483647)]
        if lo < 0
        else [("uint8", 0, 255), ("uint16", 0, 65535), ("uint32", 0, 4294967295)]
    )
    for name, low, high in candidates:
        if low <= lo and hi <= high:
            return name
    return str(series.dtype)


def _float32_safe(series: pd.Series) -> bool:
    """True when narrowing this column to float32 loses nothing that matters.

    Two things are checked, and a third is deliberately not.

    Checked: the range fits float32 (otherwise values become inf), and distinct values
    stay distinct after the round trip (otherwise rows that differ in the data become
    identical to the model, which is real information loss).

    Not checked: whether digits beyond float32's ~7 significant figures survive.  They
    do not, and cannot; that is what ``downcast_floats`` opts into, and rejecting every
    column with more than 7 digits would make the option useless.
    """
    values = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    if np.abs(finite).max() > np.finfo(np.float32).max:
        return False
    round_tripped = finite.astype(np.float32).astype(np.float64)
    return len(np.unique(round_tripped)) == len(np.unique(finite))


_TRUE = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE = frozenset({"false", "f", "no", "n", "0", "0.0"})


def _to_boolean(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")
