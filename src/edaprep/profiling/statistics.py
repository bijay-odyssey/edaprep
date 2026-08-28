"""Statistic kernels for profiling.

What the benchmark decided
--------------------------
This module originally hand-rolled the moment computations in NumPy, on the premise
that one fused pass over a 2-D block would beat pandas' per-column dispatch.  The
benchmark said otherwise, at every shape tested:

===================  ==========  ==================  ===================
shape                NumPy block  pandas per-column   pandas frame-level
===================  ==========  ==================  ===================
20,000 x 300            799 ms          557 ms              806 ms
100,000 x 50            731 ms          381 ms              600 ms
500,000 x 12            969 ms          542 ms              810 ms
5,000 x 1000            838 ms         1210 ms              802 ms
===================  ==========  ==================  ===================

and used 298 MiB against pandas' 2 MiB.  Cache-sized chunking recovered some of the
gap but never closed it: pandas computes m2, m3 and m4 in a *single fused Cython pass*
per column, while any NumPy formulation needs a separate traversal for each power and
is therefore memory-bandwidth bound.

So the hand-rolled kernel was deleted.  What remains delegates to pandas and adds only
the things pandas does not provide:

* **infinity handling** -- pandas' mean of a column containing ``inf`` is ``inf``,
  which poisons every downstream moment; infinities are counted and masked first;
* **a scale-relative degeneracy guard** -- pandas zeroes moments below a fixed
  ``1e-14``, which misfires on genuinely small data and misses rounding noise on large
  data (see :func:`_denoise`);
* zero/negative counts, the MAD, and the :class:`NumericStats` value type.

This is section 28 of the design goal applied literally: do not reimplement what pandas
already does well, and let measurement rather than intuition decide which case that is.
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
        return {
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


@contextlib.contextmanager
def _quiet_reductions():
    """Silence the NumPy/pandas warnings that degenerate reductions legitimately raise.

    Scoped to the expressions that can raise them.  ``edaprep`` never installs a global
    warning filter: a module-level ``filterwarnings("ignore")`` is exactly the kind of
    thing that hides a real bug.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="All-NaN axis encountered")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        warnings.filterwarnings("ignore", message="overflow encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
        # Overflow is possible and meaningful: values around 1e300 square to infinity,
        # so the variance genuinely is infinite.  Reporting inf (rendered as null in
        # JSON) is truthful; raising is not.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            yield


def _clean(value: float) -> Optional[float]:
    """JSON-safe float: NaN and +/-inf become ``None``."""
    if value is None:
        return None
    f = float(value)
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _denoise(
    skew: np.ndarray, kurtosis: np.ndarray, std: np.ndarray, mean: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero moments whose spread is indistinguishable from rounding noise.

    Centring a column of 25 copies of ``38479.9277006`` leaves values of about one ULP
    (~7e-12) rather than exactly zero, because the mean is not exactly representable.
    Skewness then divides noise by noise and returns an arbitrary number near 1, when
    the column is in fact constant and the answer must be 0.

    pandas guards this with a fixed absolute cut-off on the moments (``|m| < 1e-14``).
    That is scale-dependent in both directions: it reports 0 for genuinely small data
    where the moments are well determined -- ``[0, a, a, a]`` has skewness exactly -2
    for every ``a``, yet pandas returns 0 once ``a`` drops below about 1e-9 -- and it
    misses rounding noise on data large enough that the noise exceeds 1e-14.

    The guard here compares the standard deviation with one ULP of the mean, so it
    fires exactly when the spread *is* rounding noise, at any magnitude.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        noise_floor = np.finfo(np.float64).eps * np.abs(mean) * 4.0
    degenerate = np.isfinite(noise_floor) & np.isfinite(std) & (std <= noise_floor)
    if not np.any(degenerate):
        return skew, kurtosis
    skew = np.where(degenerate & np.isfinite(skew), 0.0, skew)
    kurtosis = np.where(degenerate & np.isfinite(kurtosis), 0.0, kurtosis)
    return skew, kurtosis


def _to_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A float64 view of ``frame`` with missing values as NaN.

    Nullable extension dtypes (``Int64``, ``Float64``, ``boolean``) and categoricals do
    not participate in arithmetic reductions the same way as plain floats; normalising
    once here is what keeps the rest of the library dtype-agnostic.  Columns that are
    already float64 are passed through without a copy.
    """
    needs_cast = [name for name in frame.columns if frame[name].dtype != np.float64]
    if not needs_cast:
        return frame
    converted = {}
    for name in frame.columns:
        series = frame[name]
        if series.dtype == np.float64:
            converted[str(name)] = series
        elif isinstance(series.dtype, pd.CategoricalDtype):
            converted[str(name)] = series.astype("float64")
        else:
            try:
                converted[str(name)] = pd.Series(
                    series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False),
                    index=frame.index,
                    name=str(name),
                )
            except (TypeError, ValueError):
                converted[str(name)] = pd.to_numeric(series, errors="coerce").astype("float64")
    return pd.DataFrame(converted, index=frame.index, copy=False)


def numeric_block_stats(
    frame: pd.DataFrame,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    compute_moments: bool = True,
    compute_mad: bool = True,
) -> Dict[str, NumericStats]:
    """Compute :class:`NumericStats` for every column of a numeric frame.

    Parameters
    ----------
    frame :
        A frame whose columns are all numeric (or numeric-coercible).
    quantile_levels :
        Quantiles to compute, in [0, 1].
    compute_moments :
        Set ``False`` to skip skewness and kurtosis -- the most expensive statistics --
        for the "quick" analysis level.
    compute_mad :
        Set ``False`` to skip the median absolute deviation, which costs a second
        median pass over a full-size intermediate.
    """
    names: List[str] = [str(c) for c in frame.columns]
    if not names:
        return {}

    n_rows = len(frame)
    width = len(names)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    data = _to_numeric_frame(frame)

    with _quiet_reductions():
        # One pass over the columns for every statistic that would otherwise
        # materialise a full-frame intermediate.  `(data == 0).sum()`,
        # `(data < 0).sum()` and the MAD's `(data - median).abs()` each allocate a
        # whole extra frame: on a 20,000 x 300 input that is three 48 MiB temporaries
        # for statistics that need only one column in memory at a time.
        n_missing = np.zeros(width, dtype=np.int64)
        n_infinite = np.zeros(width, dtype=np.int64)
        n_zeros = np.zeros(width, dtype=np.int64)
        n_negative = np.zeros(width, dtype=np.int64)
        infinite_columns: Dict[str, np.ndarray] = {}

        for i, name in enumerate(data.columns):
            values = data[name].to_numpy(dtype=np.float64, copy=False)
            missing = np.isnan(values)
            n_missing[i] = int(missing.sum())
            # Infinities are present but not finite.  Left in place they turn the mean
            # -- and therefore every moment -- into inf or nan, so they are counted and
            # then masked to NaN, and the remaining statistics describe finite values.
            infinite = np.isinf(values)
            count = int(infinite.sum())
            if count:
                n_infinite[i] = count
                infinite_columns[str(name)] = infinite
            n_zeros[i] = int(np.count_nonzero(values == 0.0))
            n_negative[i] = int(np.count_nonzero(values < 0.0))

        if infinite_columns:
            data = data.copy()
            for name, mask in infinite_columns.items():
                data.loc[mask, name] = np.nan
        del infinite_columns

        n_valid = n_rows - n_missing  # NaN is missing; inf is present but not finite

        if n_rows == 0:
            empty = np.full(width, np.nan)
            mean = std = minimum = maximum = skew = kurt = mad = empty
            q_matrix = np.full((len(levels), width), np.nan)
        else:
            mean, std, minimum, maximum, skew, kurt = _moments(data, compute_moments)
            if compute_moments:
                skew, kurt = _repair_small_magnitude(data, mean, std, skew, kurt)
            skew, kurt = _denoise(skew, kurt, std, mean)

            q_frame = _chunked_quantile(data, levels)
            q_matrix = q_frame.to_numpy()

            if compute_mad:
                medians = (
                    q_frame.loc[0.50].to_numpy()
                    if 0.50 in q_frame.index
                    else data.median().to_numpy()
                )
                mad = np.empty(width, dtype=np.float64)
                for i, name in enumerate(data.columns):
                    values = data[name].to_numpy(dtype=np.float64, copy=False)
                    mad[i] = np.nanmedian(np.abs(values - medians[i]))
            else:
                mad = np.full(width, np.nan)

        variance = std**2

    return {
        name: NumericStats(
            count=int(n_valid[i]),
            n_missing=int(n_missing[i]),
            mean=float(mean[i]),
            std=float(std[i]),
            variance=float(variance[i]),
            minimum=float(minimum[i]),
            maximum=float(maximum[i]),
            skew=float(skew[i]),
            kurtosis=float(kurt[i]),
            quantiles={float(q): float(q_matrix[j, i]) for j, q in enumerate(levels)},
            n_zeros=int(n_zeros[i]),
            n_negative=int(n_negative[i]),
            n_infinite=int(n_infinite[i]),
            mad=float(mad[i]),
        )
        for i, name in enumerate(names)
    }


#: Below this second moment, pandas' absolute ``_zero_out_fperr`` cut-off (1e-14) can
#: fire on data whose moments are in fact well determined.  1e-12 leaves a safe margin.
_SMALL_MOMENT = 1e-13


def _repair_small_magnitude(
    data: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    skew: np.ndarray,
    kurtosis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute moments for columns small enough to trip pandas' absolute cut-off.

    ``pandas.core.nanops`` zeroes any moment whose magnitude is below 1e-14 before
    computing skewness.  For a column of values around 1e-9 the second moment is
    ~1e-18, so ``Series.skew()`` returns 0.0 even though the shape is perfectly well
    determined: ``[0, a, a, a]`` has skewness exactly -2 for every ``a``.

    Skewness and kurtosis are invariant under affine rescaling, so the fix is to
    recompute those columns from ``(x - mean) / std``, which puts them at unit scale
    where the cut-off cannot reach.  Only columns that are actually at risk are
    touched, so the cost is negligible on ordinary data.

    Columns whose spread really *is* rounding noise are handled separately by
    :func:`_denoise`, which runs afterwards and wins.
    """
    at_risk = np.flatnonzero(np.isfinite(std) & (std > 0.0) & (std**2 < _SMALL_MOMENT))
    if at_risk.size == 0:
        return skew, kurtosis
    skew = skew.copy()
    kurtosis = kurtosis.copy()
    columns = list(data.columns)
    for i in at_risk:
        rescaled = (data[columns[i]] - mean[i]) / std[i]
        skew[i] = rescaled.skew()
        kurtosis[i] = rescaled.kurt()
    return skew, kurtosis


#: Target size of one intermediate, in bytes.  ``DataFrame.quantile`` sorts a copy of
#: everything it is given, so calling it on a whole frame allocates the whole frame
#: again; chunking bounds that without giving up the vectorisation, which per-column
#: quantile calls do (measured 1.8x slower on a 20,000 x 300 frame).
_QUANTILE_CHUNK_BYTES = 16 * 1024 * 1024


def _chunked_quantile(data: pd.DataFrame, levels: np.ndarray) -> pd.DataFrame:
    """``data.quantile(levels)`` with a bounded working set."""
    n_rows = len(data)
    per_column = max(n_rows * 8, 1)
    chunk = max(1, min(data.shape[1], _QUANTILE_CHUNK_BYTES // per_column))
    if chunk >= data.shape[1]:
        return data.quantile(list(levels))
    parts = [
        data.iloc[:, start : start + chunk].quantile(list(levels))
        for start in range(0, data.shape[1], chunk)
    ]
    return pd.concat(parts, axis=1)


def _moments(data: pd.DataFrame, compute_moments: bool) -> Tuple[np.ndarray, ...]:
    """Mean, std, min, max and (optionally) skew and kurtosis, as aligned arrays.

    Per-column rather than frame-level, which is the opposite of the obvious choice and
    was settled by measurement.  ``DataFrame.mean()`` and friends operate on pandas'
    internal blocks and allocate intermediates proportional to the *whole frame*:

    ===============  ==========================  =========================
    shape            per-column                  frame-level
    ===============  ==========================  =========================
    20,000 x 300         543 ms /   3.1 MiB          1684 ms / 377.8 MiB
    5,000 x 1000         850 ms /   4.4 MiB           915 ms / 157.6 MiB
    2,000 x 3000        1680 ms /  10.0 MiB          1375 ms / 189.3 MiB
    ===============  ==========================  =========================

    Frame-level only wins on time past about 3000 columns, and then by 1.2x for 19x
    the memory.  Both call the same pandas kernels, so they agree exactly; there is no
    accuracy trade-off, only a resource one, and it points one way.
    """
    width = data.shape[1]
    means, stds, mins, maxs, skews, kurts = [], [], [], [], [], []
    for name in data.columns:
        column = data[name]
        means.append(column.mean())
        stds.append(column.std(ddof=1))
        mins.append(column.min())
        maxs.append(column.max())
        if compute_moments:
            skews.append(column.skew())
            kurts.append(column.kurt())
    empty = np.full(width, np.nan)
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(stds, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
        np.asarray(skews, dtype=np.float64) if compute_moments else empty,
        np.asarray(kurts, dtype=np.float64) if compute_moments else empty,
    )


def series_numeric_stats(
    series: pd.Series, quantile_levels: Sequence[float] = DEFAULT_QUANTILES
) -> NumericStats:
    """Single-column convenience wrapper around :func:`numeric_block_stats`."""
    name = str(series.name) if series.name is not None else "_"
    return numeric_block_stats(series.to_frame(name=name), quantile_levels)[name]


def skewness(values: np.ndarray) -> float:
    """Bias-corrected sample skewness of a 1-D array, ignoring NaN and infinities.

    Zero-variance input returns ``0.0``, matching ``pandas.Series.skew()``; SciPy
    returns NaN there instead.  The quantity is mathematically undefined either way
    (0/0), so this is a convention, and the tests pin it to pandas.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    with _quiet_reductions():
        frame = pd.DataFrame({"_": arr})
        return float(numeric_block_stats(frame, quantile_levels=(0.5,))["_"].skew)


def kurtosis(values: np.ndarray) -> float:
    """Bias-corrected sample excess kurtosis of a 1-D array, ignoring NaN."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return float("nan")
    with _quiet_reductions():
        frame = pd.DataFrame({"_": arr})
        return float(numeric_block_stats(frame, quantile_levels=(0.5,))["_"].kurtosis)


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


def top_categories(
    series: pd.Series, k: int = 10, dropna: bool = True
) -> List[Tuple[object, int]]:
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
    return {"total": int(usage.sum()), **per_column}
