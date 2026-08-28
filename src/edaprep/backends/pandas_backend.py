"""The pandas backend: the only implementation, and the reference one.

Every method here is a thin adapter over a pandas call. That is the point: the protocol
exists so a *different* implementation can be written, not to add a layer over the one
that already works.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import Backend

__all__ = ["PandasBackend"]


class PandasBackend(Backend):
    """Backend over ``pandas.DataFrame``."""

    name = "pandas"

    # -- introspection ---------------------------------------------------------------

    def is_frame(self, obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

    def shape(self, frame: pd.DataFrame) -> Tuple[int, int]:
        return frame.shape

    def column_names(self, frame: pd.DataFrame) -> List[str]:
        return [str(c) for c in frame.columns]

    def dtype_of(self, frame: pd.DataFrame, column: str) -> str:
        return str(frame[column].dtype)

    def memory_usage(self, frame: pd.DataFrame, deep: bool = True) -> Dict[str, int]:
        usage = frame.memory_usage(index=True, deep=deep)
        return {"total": int(usage.sum()), **{str(k): int(v) for k, v in usage.items()}}

    # -- column access ----------------------------------------------------------------

    def get_column(self, frame: pd.DataFrame, column: str) -> pd.Series:
        return frame[column]

    def to_float_array(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("float64")
        try:
            return series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        except (TypeError, ValueError):
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)

    def select(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        return frame[list(columns)]

    def assign(self, frame: pd.DataFrame, columns: Mapping[str, Any]) -> pd.DataFrame:
        if not columns:
            return frame
        # Rebuild from the existing column blocks with only the named columns replaced,
        # rather than `frame.copy()` then assigning: a step that touches 5 of 400
        # columns then allocates 5 columns, not 400.
        data: Dict[str, Any] = {}
        for name in frame.columns:
            key = str(name)
            data[key] = columns.get(key, frame[name])
        for key, value in columns.items():
            if key not in data:
                data[key] = value
        return pd.DataFrame(data, index=frame.index, copy=False)

    # -- aggregation -------------------------------------------------------------------

    def null_mask(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        return frame[column].isna().to_numpy()

    def n_unique(self, frame: pd.DataFrame, column: str, dropna: bool = True) -> int:
        try:
            return int(frame[column].nunique(dropna=dropna))
        except TypeError:
            # Unhashable cell values (lists, dicts); the string view is slower but is
            # the only thing that can be counted at all.
            return int(frame[column].astype(str).nunique(dropna=dropna))

    def value_counts(
        self, frame: pd.DataFrame, column: str, dropna: bool = True
    ) -> List[Tuple[Any, int]]:
        counts = frame[column].value_counts(dropna=dropna)
        return [(index, int(value)) for index, value in counts.items()]

    def quantiles(
        self, frame: pd.DataFrame, columns: Sequence[str], levels: Sequence[float]
    ) -> np.ndarray:
        return frame[list(columns)].quantile(list(levels)).to_numpy()

    def group_mean(
        self, frame: pd.DataFrame, value_column: str, group_column: str
    ) -> Dict[Any, float]:
        grouped = frame.groupby(group_column, observed=True)[value_column].mean()
        return {index: float(value) for index, value in grouped.items()}

    def duplicated_rows(
        self, frame: pd.DataFrame, subset: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        try:
            return frame.duplicated(subset=list(subset) if subset else None).to_numpy()
        except TypeError:
            return (
                frame.astype(str).duplicated(subset=list(subset) if subset else None).to_numpy()
            )

    # -- construction --------------------------------------------------------------------

    def concat_columns(self, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
        parts = list(frames)
        if not parts:
            return pd.DataFrame()
        if len(parts) == 1:
            return parts[0]
        return pd.concat(parts, axis=1, copy=False)

    def take_rows(self, frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        return frame.loc[np.asarray(mask, dtype=bool)]
