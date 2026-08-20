"""Vectorised statistic kernels.

Why this module exists
----------------------
Profiling a wide frame the naive way costs one pandas dispatch per statistic per
column::

    for col in df.columns:            # 434 columns in the usual largest frame
        df[col].mean(); df[col].std(); df[col].skew(); df[col].kurt(); ...

That is ~10 dispatches x 434 columns, each with its own NaN mask allocation.  The
kernels here compute the whole family of moment-based statistics for a *block* of
numeric columns in a fixed number of passes over one 2-D array, sharing the NaN mask.

Numerical policy
----------------
Moments are computed by the two-pass centred method (mean first, then centred powers),
not by accumulating raw power sums.  Raw power sums are one pass faster and lose
catastrophic amounts of precision when the mean is large relative to the spread; for a
column like ``price`` with mean 250,000 and sd 50, the naive variance can come out
negative.  Correctness before speed, per the design goal.

Skewness and kurtosis use the bias-corrected sample estimators (the same
Fisher-Pearson adjusted forms pandas uses), so results agree with ``Series.skew()`` and
``Series.kurt()`` to floating-point tolerance.  This is asserted in the tests.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "NumericStats",
    "numeric_block_stats",
    "series_numeric_stats",
    "skewness",
    "kurtosis",
    "median_abs_deviation",
    "quantiles",
    "top_categories",
    "estimate_memory",
]

#: Quantiles computed for every numeric column.  P1/P99 feed the winsorising fence,
#: the quartiles feed the IQR fence, and P5/P95 are reported for context.
DEFAULT_QUANTILES: Tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class NumericStats:
    """Moment and order statistics for a single numeric column."""

    count: int
    n_missing: int
    mean: float
    std: float
    variance: float
    minimum: float
    maximum: float
    skew: float
    kurtosis: float
    quantiles: Dict[float, float]
    n_zeros: int
    n_negative: int
    n_infinite: int
    mad: float

    @property
    def median(self) -> float:
        return self.quantiles.get(0.50, float("nan"))

    @property
    def q1(self) -> float:
        return self.quantiles.get(0.25, float("nan"))

    @property
    def q3(self) -> float:
        return self.quantiles.get(0.75, float("nan"))

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def range(self) -> float:
        return self.maximum - self.minimum

    def to_dict(self) -> Dict[str, object]:
        out = {
            "count": self.count,
            "n_missing": self.n_missing,
            "mean": _clean(self.mean),
            "std": _clean(self.std),
            "variance": _clean(self.variance),
            "min": _clean(self.minimum),
            "max": _clean(self.maximum),
            "skew": _clean(self.skew),
            "kurtosis": _clean(self.kurtosis),
            "median": _clean(self.median),
            "iqr": _clean(self.iqr),
            "mad": _clean(self.mad),
            "n_zeros": self.n_zeros,
            "n_negative": self.n_negative,
            "n_infinite": self.n_infinite,
            "quantiles": {str(q): _clean(v) for q, v in self.quantiles.items()},
        }
        return out


@contextlib.contextmanager
def _quiet_nan_reductions():
    """Silence the two NumPy warnings that all-NaN reductions legitimately raise.

    Scoped to the single expression that can raise them.  ``edaprep`` never installs a
    global warning filter.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        with np.errstate(invalid="ignore", divide="ignore"):
            yield


def _denoise_m2(m2: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Zero out a second moment that is indistinguishable from rounding noise.

    Centring a column of 25 copies of ``38479.9277006`` leaves values of about one ULP
    (~7e-12) rather than exactly zero, because the mean is not exactly representable.
    ``m2`` is then ~5e-23: non-zero, so the skewness formula divides noise by noise and
    returns an arbitrary number near 1.  The column is constant; the answer must be 0.

    pandas guards this with a fixed absolute cut-off (``m2 < 1e-14``), which is
    scale-dependent: it misfires on genuinely tiny data and misses noise on very large
    data.  The guard here is relative -- ``m2`` must exceed the square of one ULP of the
    mean, with a small factor for accumulated error -- so it behaves the same at every
    magnitude.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        noise_floor = (np.finfo(np.float64).eps * np.abs(mean)) ** 2 * 16.0
    degenerate = np.isfinite(noise_floor) & (m2 <= noise_floor)
    if np.any(degenerate):
        m2 = m2.copy()
        m2[degenerate] = 0.0
    return m2


def _clean(value: float) -> Optional[float]:
    """JSON-safe float: NaN and +/-inf become ``None``."""
    if value is None:
        return None
    f = float(value)
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _as_float_block(frame: pd.DataFrame) -> np.ndarray:
    """Materialise a numeric frame as a 2-D float64 array with NaN for missing.

    Nullable extension dtypes (``Int64``, ``Float64``, ``boolean``) do not convert via
    ``.to_numpy()`` without an explicit ``na_value``; handling them here is what keeps
    the rest of the library dtype-agnostic.
    """
    if frame.shape[1] == 0:
        return np.empty((len(frame), 0), dtype=np.float64)
    columns = []
    for name in frame.columns:
        series = frame[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("float64")
        try:
            arr = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        except (TypeError, ValueError):
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        columns.append(arr)
    return np.column_stack(columns) if columns else np.empty((len(frame), 0))


def numeric_block_stats(
    frame: pd.DataFrame,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    compute_moments: bool = True,
    compute_mad: bool = True,
) -> Dict[str, NumericStats]:
    """Compute :class:`NumericStats` for every column of a numeric frame.

    All columns share one NaN mask and one pass per statistic family, rather than one
    dispatch per (column, statistic) pair.

    Parameters
    ----------
    frame :
        A frame whose columns are all numeric (or numeric-coercible).
    quantile_levels :
        Quantiles to compute, in [0, 1].
    compute_moments :
        Set ``False`` to skip skewness and kurtosis, the most expensive part, for
        "quick" analysis levels.
    compute_mad :
        Set ``False`` to skip the median absolute deviation.
    """
    names: List[str] = [str(c) for c in frame.columns]
    n_rows = len(frame)
    if not names:
        return {}

    block = _as_float_block(frame)
    finite_mask = np.isfinite(block)
    valid = ~np.isnan(block)  # NaN is missing; +/-inf is present but not finite
    n_valid = valid.sum(axis=0)
    n_infinite = (valid & ~finite_mask).sum(axis=0)

    # Statistics are computed over *finite* values only.  An infinity in a column would
    # otherwise poison the mean and every downstream moment, silently.
    work = np.where(finite_mask, block, np.nan)
    n_finite = finite_mask.sum(axis=0)

    # An all-missing column makes NumPy emit "Mean of empty slice" / "All-NaN slice
    # encountered".  Both are expected here and the NaN result is the intended answer,
    # so they are silenced locally rather than globally -- the usual
    # ``filterwarnings("ignore")`` at module scope is exactly what hid a real bug.
    with _quiet_nan_reductions():
        mean = np.nanmean(work, axis=0) if n_rows else np.full(len(names), np.nan)
        centred = work - mean
        m2 = _denoise_m2(np.nanmean(centred**2, axis=0), mean)
        variance = _sample_variance(m2, n_finite)
        std = np.sqrt(variance)
        minimum = np.nanmin(work, axis=0) if n_rows else np.full(len(names), np.nan)
        maximum = np.nanmax(work, axis=0) if n_rows else np.full(len(names), np.nan)

        if compute_moments:
            m3 = np.nanmean(centred**3, axis=0)
            m4 = np.nanmean(centred**4, axis=0)
            skew_vals = _skew_from_moments(m2, m3, n_finite)
            kurt_vals = _kurtosis_from_moments(m2, m4, n_finite)
        else:
            skew_vals = np.full(len(names), np.nan)
            kurt_vals = np.full(len(names), np.nan)

        q_levels = np.asarray(quantile_levels, dtype=np.float64)
        if n_rows and np.any(n_finite > 0):
            q_matrix = np.nanquantile(work, q_levels, axis=0)
        else:
            q_matrix = np.full((len(q_levels), len(names)), np.nan)

        n_zeros = np.nansum(work == 0.0, axis=0).astype(np.int64)
        n_negative = np.nansum(work < 0.0, axis=0).astype(np.int64)

        if compute_mad:
            median_row = q_matrix[list(q_levels).index(0.50)] if 0.50 in q_levels else (
                np.nanmedian(work, axis=0)
            )
            mad_vals = np.nanmedian(np.abs(work - median_row), axis=0)
        else:
            mad_vals = np.full(len(names), np.nan)

    out: Dict[str, NumericStats] = {}
    for i, name in enumerate(names):
        out[name] = NumericStats(
            count=int(n_valid[i]),
            n_missing=int(n_rows - n_valid[i]),
            mean=float(mean[i]),
            std=float(std[i]),
            variance=float(variance[i]),
            minimum=float(minimum[i]),
            maximum=float(maximum[i]),
            skew=float(skew_vals[i]),
            kurtosis=float(kurt_vals[i]),
            quantiles={float(q): float(q_matrix[j, i]) for j, q in enumerate(q_levels)},
            n_zeros=int(n_zeros[i]),
            n_negative=int(n_negative[i]),
            n_infinite=int(n_infinite[i]),
            mad=float(mad_vals[i]),
        )
    return out


def series_numeric_stats(
    series: pd.Series, quantile_levels: Sequence[float] = DEFAULT_QUANTILES
) -> NumericStats:
    """Single-column convenience wrapper around :func:`numeric_block_stats`."""
    name = str(series.name) if series.name is not None else "_"
    frame = series.to_frame(name=name)
    return numeric_block_stats(frame, quantile_levels)[name]


def _sample_variance(m2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Convert the population second moment to the unbiased sample variance (ddof=1)."""
    n = n.astype(np.float64)
    out = np.full_like(m2, np.nan, dtype=np.float64)
    ok = n > 1
    out[ok] = m2[ok] * n[ok] / (n[ok] - 1.0)
    return out


def _skew_from_moments(m2: np.ndarray, m3: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Adjusted Fisher-Pearson standardised moment coefficient G1.

    ``G1 = sqrt(n(n-1))/(n-2) * m3 / m2**1.5``, undefined for n < 3.

    Zero-variance columns return ``0.0``, matching ``pandas.Series.skew()``.  SciPy
    returns NaN there instead.  pandas' convention is the more useful one for a
    planner: a constant column has no skew to correct, and ``0.0`` routes it to
    "no transform" without every downstream rule needing a NaN special case.  The
    quantity is mathematically undefined either way (0/0), so this is a convention,
    not a correctness claim, and the tests pin it to pandas.
    """
    n = n.astype(np.float64)
    out = np.full_like(m2, np.nan, dtype=np.float64)
    defined = n > 2
    out[defined & (m2 <= 0)] = 0.0
    ok = defined & (m2 > 0)
    if not np.any(ok):
        return out
    g1 = m3[ok] / np.power(m2[ok], 1.5)
    out[ok] = np.sqrt(n[ok] * (n[ok] - 1.0)) / (n[ok] - 2.0) * g1
    return out


def _kurtosis_from_moments(m2: np.ndarray, m4: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Bias-corrected excess kurtosis G2, matching ``pandas.Series.kurt()``.

    ``G2 = (n-1)/((n-2)(n-3)) * ((n+1) * g2 + 6)`` with ``g2 = m4/m2**2 - 3``.
    Undefined for n < 4; zero-variance columns return ``0.0`` as pandas does.
    """
    n = n.astype(np.float64)
    out = np.full_like(m2, np.nan, dtype=np.float64)
    defined = n > 3
    out[defined & (m2 <= 0)] = 0.0
    ok = defined & (m2 > 0)
    if not np.any(ok):
        return out
    g2 = m4[ok] / (m2[ok] ** 2) - 3.0
    out[ok] = (n[ok] - 1.0) / ((n[ok] - 2.0) * (n[ok] - 3.0)) * ((n[ok] + 1.0) * g2 + 6.0)
    return out


def skewness(values: np.ndarray) -> float:
    """Bias-corrected sample skewness of a 1-D array, ignoring NaN."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return float("nan")
    mean = arr.mean()
    centred = arr - mean
    m2 = _denoise_m2(np.array([np.mean(centred**2)]), np.array([mean]))
    m3 = np.array([np.mean(centred**3)])
    return float(_skew_from_moments(m2, m3, np.array([n]))[0])


def kurtosis(values: np.ndarray) -> float:
    """Bias-corrected sample excess kurtosis of a 1-D array, ignoring NaN."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 4:
        return float("nan")
    mean = arr.mean()
    centred = arr - mean
    m2 = _denoise_m2(np.array([np.mean(centred**2)]), np.array([mean]))
    m4 = np.array([np.mean(centred**4)])
    return float(_kurtosis_from_moments(m2, m4, np.array([n]))[0])


def median_abs_deviation(values: np.ndarray, scale: float = 1.0) -> float:
    """MAD = median(|x - median(x)|), ignoring NaN.

    ``scale=1.4826`` makes it a consistent estimator of the standard deviation for
    normally distributed data; that constant is applied by the modified-z-score
    detector rather than baked in here.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)) * scale)


def quantiles(values: np.ndarray, levels: Sequence[float]) -> Dict[float, float]:
    """Quantiles of a 1-D array, ignoring NaN.  Returns NaN when no finite values."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {float(q): float("nan") for q in levels}
    computed = np.quantile(arr, np.asarray(levels, dtype=np.float64))
    return {float(q): float(v) for q, v in zip(levels, np.atleast_1d(computed))}


def top_categories(series: pd.Series, k: int = 10, dropna: bool = True) -> List[Tuple[object, int]]:
    """The ``k`` most frequent values and their counts, in descending order."""
    counts = series.value_counts(dropna=dropna)
    return [(idx, int(val)) for idx, val in counts.head(k).items()]


def estimate_memory(frame: pd.DataFrame, deep: bool = True) -> Dict[str, int]:
    """Per-column memory usage in bytes, plus the total.

    ``deep=True`` follows object pointers, which is the only way to see that an object
    column of strings costs far more than its 8-bytes-per-pointer shallow estimate.  It
    is O(n) in the number of Python objects, so callers profiling very wide object
    frames may pass ``deep=False``.
    """
    usage = frame.memory_usage(index=True, deep=deep)
    per_column = {str(k): int(v) for k, v in usage.items()}
    total = int(usage.sum())
    return {"total": total, **per_column}
