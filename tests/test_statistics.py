"""Correctness of the statistic kernels against trusted implementations.

Section 33 of the design goal: numerical operations are verified against NumPy, pandas and
SciPy rather than against themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from edaprep.profiling.statistics import (
    estimate_memory,
    kurtosis,
    median_abs_deviation,
    numeric_block_stats,
    series_numeric_stats,
    skewness,
)

TOL = dict(rel=1e-9, abs=1e-9)


@pytest.fixture
def block() -> pd.DataFrame:
    gen = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "normal": gen.normal(100, 15, 500),
            "lognormal": gen.lognormal(2, 1, 500),
            "uniform": gen.uniform(-5, 5, 500),
            "integers": gen.integers(0, 50, 500).astype(float),
            "with_nan": np.where(gen.random(500) < 0.2, np.nan, gen.normal(0, 1, 500)),
        }
    )


def test_moments_match_pandas(block: pd.DataFrame) -> None:
    stats = numeric_block_stats(block)
    for name in block.columns:
        series = block[name]
        got = stats[name]
        assert got.mean == pytest.approx(series.mean(), **TOL)
        assert got.std == pytest.approx(series.std(ddof=1), **TOL)
        assert got.variance == pytest.approx(series.var(ddof=1), **TOL)
        assert got.minimum == pytest.approx(series.min(), **TOL)
        assert got.maximum == pytest.approx(series.max(), **TOL)
        # skew/kurt are the interesting ones: bias-corrected forms are easy to get wrong
        assert got.skew == pytest.approx(series.skew(), rel=1e-8, abs=1e-8)
        assert got.kurtosis == pytest.approx(series.kurt(), rel=1e-8, abs=1e-8)


def test_quantiles_match_numpy(block: pd.DataFrame) -> None:
    stats = numeric_block_stats(block)
    for name in block.columns:
        values = block[name].to_numpy()
        finite = values[np.isfinite(values)]
        for level in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert stats[name].quantiles[level] == pytest.approx(
                np.quantile(finite, level), **TOL
            )


def test_scalar_helpers_match_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    gen = np.random.default_rng(1)
    values = gen.lognormal(0, 1, 400)
    assert skewness(values) == pytest.approx(scipy_stats.skew(values, bias=False), rel=1e-9)
    assert kurtosis(values) == pytest.approx(
        scipy_stats.kurtosis(values, bias=False), rel=1e-9
    )
    assert median_abs_deviation(values) == pytest.approx(
        scipy_stats.median_abs_deviation(values, scale=1.0), rel=1e-12
    )


def test_infinities_do_not_poison_moments() -> None:
    """An inf in a column must not turn every statistic into nan.

    ``pandas.Series.mean()`` returns inf here; edaprep computes over finite values and
    reports the infinity count separately, so the summary stays usable.
    """
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.inf, -np.inf, np.nan]})
    stats = numeric_block_stats(frame)["x"]
    assert stats.n_infinite == 2
    assert stats.n_missing == 1
    assert stats.count == 5  # inf is present, just not finite
    assert stats.mean == pytest.approx(2.0)
    assert stats.maximum == pytest.approx(3.0)


def test_constant_column_has_nan_skew_like_pandas() -> None:
    frame = pd.DataFrame({"c": [5.0] * 20})
    stats = numeric_block_stats(frame)["c"]
    assert stats.skew == 0.0
    assert stats.kurtosis == 0.0
    assert stats.variance == pytest.approx(0.0)
    assert pd.Series([5.0] * 20).skew() == 0.0  # pinned to the pandas convention


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_small_samples_do_not_raise(n: int) -> None:
    """Skew needs n>=3 and kurtosis n>=4; below that the answer is nan, not an error."""
    frame = pd.DataFrame({"x": np.arange(n, dtype=float)})
    stats = numeric_block_stats(frame)["x"]
    assert stats.count == n
    if n < 3:
        assert np.isnan(stats.skew)
    if n < 4:
        assert np.isnan(stats.kurtosis)


def test_all_missing_column() -> None:
    frame = pd.DataFrame({"x": [np.nan] * 10})
    stats = numeric_block_stats(frame)["x"]
    assert stats.count == 0
    assert stats.n_missing == 10
    assert np.isnan(stats.mean)
    assert stats.to_dict()["mean"] is None  # JSON-safe


def test_nullable_extension_dtypes() -> None:
    """Int64/Float64 must profile identically to their numpy counterparts."""
    frame = pd.DataFrame(
        {
            "nullable": pd.array([1, 2, 3, None, 5], dtype="Int64"),
            "plain": pd.Series([1.0, 2.0, 3.0, np.nan, 5.0]),
        }
    )
    stats = numeric_block_stats(frame)
    assert stats["nullable"].mean == pytest.approx(stats["plain"].mean)
    assert stats["nullable"].n_missing == 1


def test_large_mean_small_spread_precision() -> None:
    """Two-pass centred moments, not raw power sums.

    With raw power sums the variance of this column comes out negative in float64.
    """
    values = 1e9 + np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    frame = pd.DataFrame({"x": values})
    stats = numeric_block_stats(frame)["x"]
    assert stats.variance == pytest.approx(2.5, rel=1e-9)
    assert stats.std == pytest.approx(np.sqrt(2.5), rel=1e-9)


def test_zero_and_negative_counts() -> None:
    frame = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 0.0, 1.0, np.nan]})
    stats = numeric_block_stats(frame)["x"]
    assert stats.n_zeros == 2
    assert stats.n_negative == 2


def test_series_wrapper_matches_block() -> None:
    series = pd.Series([1.0, 5.0, 2.0, 8.0, 3.0], name="v")
    assert series_numeric_stats(series).mean == pytest.approx(series.mean())


def test_empty_frame_returns_empty_mapping() -> None:
    assert numeric_block_stats(pd.DataFrame(index=range(5))) == {}


def test_estimate_memory_total_matches_pandas() -> None:
    frame = pd.DataFrame({"a": np.arange(100), "b": ["x"] * 100})
    got = estimate_memory(frame, deep=True)
    assert got["total"] == int(frame.memory_usage(index=True, deep=True).sum())


@pytest.mark.parametrize("a", [1.17549435e-38, 3.75689048e-26, 1e-9])
def test_small_magnitude_skew_is_more_accurate_than_pandas(a: float) -> None:
    """Below ~1e-14 pandas zeroes the moments; the scale-relative guard does not.

    ``pandas.core.nanops`` applies ``_zero_out_fperr``, an *absolute* ``|m| < 1e-14``
    cut-off, before computing skewness.  For ``[0, a, a, a]`` the exact skewness is -2
    for every ``a``, but once ``a`` is small enough that ``m2`` falls under that
    cut-off, ``Series.skew()`` returns 0.0.

    edaprep's guard is relative to the column's own magnitude, so it fires only when
    the spread really is rounding noise.  These cases are therefore a deliberate
    *disagreement* with pandas in which edaprep is correct, and the property test
    below excludes the regime rather than pinning us to pandas' behaviour.
    """
    frame = pd.DataFrame({"x": [0.0, a, a, a]})
    assert numeric_block_stats(frame)["x"].skew == pytest.approx(-2.0, rel=1e-9)

    scaled = pd.DataFrame({"x": [0.0, 1.0, 1.0, 1.0]})  # same shape, ordinary scale
    assert numeric_block_stats(scaled)["x"].skew == pytest.approx(
        pd.Series([0.0, 1.0, 1.0, 1.0]).skew(), rel=1e-12
    )


def test_near_constant_at_large_magnitude_reads_as_constant() -> None:
    """The other direction: rounding noise must not be reported as real skew.

    Centring 25 copies of a value that the mean cannot represent exactly leaves ~1 ULP
    of noise.  Dividing that noise by itself yields an arbitrary number near 1, which
    is what an unguarded implementation reports.
    """
    frame = pd.DataFrame({"x": [38479.9277006] * 25})
    stats = numeric_block_stats(frame)["x"]
    assert stats.skew == 0.0
    assert stats.variance == 0.0


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=1, max_dims=1, min_side=4, max_side=200),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=64,
        ),
    )
)
def test_property_moments_agree_with_pandas(values: np.ndarray) -> None:
    """For any finite float column at ordinary scale, moments match pandas."""
    centred = values - values.mean()
    # Exclude the regime documented above, where pandas' absolute 1e-14 cut-off zeroes
    # moments that are in fact well determined.  1e-13 keeps a safe margin above it.
    assume(np.all(np.isfinite(centred**4)) and float(np.mean(centred**2)) > 1e-13)

    series = pd.Series(values, name="x")
    stats = numeric_block_stats(series.to_frame())["x"]
    assert stats.mean == pytest.approx(series.mean(), rel=1e-7, abs=1e-7)
    assert stats.std == pytest.approx(series.std(ddof=1), rel=1e-6, abs=1e-9)
    expected_skew, expected_kurt = series.skew(), series.kurt()
    if pd.notna(expected_skew):
        assert stats.skew == pytest.approx(expected_skew, rel=1e-5, abs=1e-5)
    if pd.notna(expected_kurt):
        assert stats.kurtosis == pytest.approx(expected_kurt, rel=1e-5, abs=1e-5)
