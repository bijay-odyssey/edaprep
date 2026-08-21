"""Leakage prevention.

The reason the library exists.  Three of the common notebook workflows leak; a
convention will not fix that, so these tests assert the structural property directly:
**a fitted transformer's output for a row must not depend on any other data the frame
it is transforming happens to contain.**

The strongest test here is :func:`test_transform_output_is_independent_of_batching`:
it transforms a frame whole, then row by row, and requires identical output.  Any
statistic recomputed at transform time breaks it immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import edaprep
from edaprep.config import Config
from edaprep.core.context import FitContext
from edaprep.exceptions import LeakageError
from edaprep.preprocessing import (
    CategoricalEncoder,
    DistributionTransformer,
    MissingValueHandler,
    OutlierHandler,
    Scaler,
    TargetEncoder,
)
from edaprep.profiling import profile


@pytest.fixture
def split_frame():
    """Train and test frames drawn from *different* distributions.

    The shift is the point: if any statistic is recomputed on the test frame, the
    scaled test output will centre on zero, which is exactly what these tests catch.
    """
    gen = np.random.default_rng(11)
    train = pd.DataFrame(
        {
            "num": gen.normal(0, 1, 400),
            "skewed": gen.lognormal(0, 1, 400),
            "cat": gen.choice(["a", "b", "c"], 400, p=[0.6, 0.3, 0.1]),
            "y": gen.integers(0, 2, 400),
        }
    )
    test = pd.DataFrame(
        {
            "num": gen.normal(50, 10, 120),  # very different location and spread
            "skewed": gen.lognormal(3, 1, 120),
            "cat": gen.choice(["a", "b", "c"], 120),
            "y": gen.integers(0, 2, 120),
        }
    )
    return train, test


def _context(frame: pd.DataFrame, target: str = "y") -> FitContext:
    config = Config(random_state=0)
    return FitContext(
        config=config, profile=profile(frame, target=target, config=config), target=target
    )


# --- the structural property ---------------------------------------------------------


def test_transform_output_is_independent_of_batching(split_frame) -> None:
    """Transforming row by row must equal transforming the whole frame.

    This is the general statement of "no statistic is computed at transform time".  A
    scaler that recentres on the incoming batch, an encoder that recounts frequencies,
    or an imputer that takes the median of the frame it is given all fail here.
    """
    train, test = split_frame
    pipe = edaprep.AutoPipeline(target="y", model_family="linear", random_state=42)
    pipe.fit(train)

    whole = pipe.transform(test)
    row_by_row = pd.concat(
        [pipe.transform(test.iloc[[i]]) for i in range(len(test))], axis=0
    )
    pd.testing.assert_frame_equal(whole, row_by_row, check_exact=False, rtol=1e-12)


def test_transform_is_idempotent_across_calls(split_frame) -> None:
    """Calling transform twice must give the same answer both times."""
    train, test = split_frame
    pipe = edaprep.AutoPipeline(target="y", random_state=42).fit(train)
    pd.testing.assert_frame_equal(pipe.transform(test), pipe.transform(test))


def test_transforming_test_first_does_not_change_train_output(split_frame) -> None:
    """No transformer may accumulate state during transform."""
    train, test = split_frame
    pipe = edaprep.AutoPipeline(target="y", random_state=42).fit(train)
    train_first = pipe.transform(train)
    pipe.transform(test)
    pipe.transform(test)
    pd.testing.assert_frame_equal(train_first, pipe.transform(train))


def test_fit_is_unaffected_by_data_it_never_saw(split_frame) -> None:
    """Fitting on train alone must equal fitting on train from a larger frame's rows."""
    train, test = split_frame
    combined = pd.concat([train, test], ignore_index=True)

    a = edaprep.AutoPipeline(target="y", model_family="linear", random_state=42).fit(train)
    b = edaprep.AutoPipeline(target="y", model_family="linear", random_state=42).fit(
        combined.iloc[: len(train)]
    )
    assert a["scaler"].centers_ == pytest.approx(b["scaler"].centers_)


# --- per-transformer statements ----------------------------------------------------------


def test_scaler_uses_train_statistics_only(split_frame) -> None:
    train, test = split_frame
    context = _context(train)
    scaler = Scaler(["num"], strategy="standard").fit(train, train["y"], context)

    assert scaler.centers_["num"] == pytest.approx(train["num"].mean())
    out = scaler.transform(test, context)
    # Test data centred on 50 must NOT come out centred on 0.
    assert out["num"].mean() > 10
    expected = (test["num"] - train["num"].mean()) / train["num"].std(ddof=0)
    np.testing.assert_allclose(out["num"].to_numpy(), expected.to_numpy())


def test_imputer_uses_train_median_only(split_frame) -> None:
    train, test = split_frame
    train = train.copy()
    train.loc[train.index[:50], "num"] = np.nan
    test = test.copy()
    test.loc[test.index[:20], "num"] = np.nan

    context = _context(train)
    handler = MissingValueHandler(["num"], strategy="median").fit(
        train, train["y"], context
    )
    train_median = train["num"].median()
    assert handler.fill_values_["num"] == pytest.approx(train_median)

    out = handler.transform(test, context)
    filled = out.loc[test.index[:20], "num"]
    assert np.allclose(filled.to_numpy(), train_median)
    assert filled.iloc[0] != pytest.approx(test["num"].median())


def test_outlier_bounds_are_fixed_at_fit(split_frame) -> None:
    """Notebook practice fitted bounds on train and never applied them to test at all."""
    train, test = split_frame
    context = _context(train)
    handler = OutlierHandler(["num"], method="iqr", strategy="clip").fit(
        train, train["y"], context
    )
    upper = handler.bounds_["num"].upper

    out = handler.transform(test, context)
    assert out["num"].max() == pytest.approx(upper)
    # Every test value is above the train fence, so all of them get clipped.
    assert np.allclose(out["num"].to_numpy(), upper)


def test_encoder_categories_are_fixed_at_fit() -> None:
    gen = np.random.default_rng(3)
    train = pd.DataFrame({"c": gen.choice(["a", "b"], 200), "y": gen.integers(0, 2, 200)})
    test = pd.DataFrame({"c": ["a", "b", "zzz_new"], "y": [0, 1, 0]})

    context = _context(train)
    encoder = CategoricalEncoder(["c"], strategy="onehot").fit(train, train["y"], context)
    out = encoder.transform(test, context)

    assert "c_zzz_new" not in out.columns  # an unseen category cannot invent a column
    assert list(out.columns) == list(encoder.get_feature_names_out())
    assert out.iloc[2][["c_a", "c_b"]].tolist() == [0, 0]  # all-zero for the unseen level


def test_power_transform_lambda_is_fixed_at_fit(split_frame) -> None:
    train, test = split_frame
    context = _context(train)
    transformer = DistributionTransformer(["skewed"], method="yeojohnson").fit(
        train, train["y"], context
    )
    learned = transformer.params_["skewed"]["lambda"]

    transformer.transform(test, context)
    assert transformer.params_["skewed"]["lambda"] == learned


# --- target encoding: the usual within-fold leak -----------------------------------------


def test_target_encoding_is_cross_fitted() -> None:
    """A training row must not be encoded using its own target.

    With near-singleton categories, a non-cross-fitted encoder returns (almost) the
    row's own label, which a model memorises.  Here the training output must be
    decorrelated from the target, while the stored mapping still reflects it.
    """
    gen = np.random.default_rng(5)
    n = 600
    # One category per 2 rows: the pathological case for target encoding.
    frame = pd.DataFrame(
        {"c": [f"lvl_{i // 2}" for i in range(n)], "y": gen.integers(0, 2, n)}
    )
    context = _context(frame)

    encoder = TargetEncoder(["c"])
    oof = encoder.fit_transform(frame, frame["y"], context)["c"]
    naive = frame["c"].map(encoder.mappings_["c"])

    corr_oof = abs(np.corrcoef(oof, frame["y"])[0, 1])
    corr_naive = abs(np.corrcoef(naive, frame["y"])[0, 1])

    assert encoder.cross_fitted is True
    assert corr_naive > 0.4, "the naive mapping should leak, or the test proves nothing"
    assert corr_oof < 0.2, f"out-of-fold encoding still leaks (corr={corr_oof:.3f})"
    assert corr_oof < corr_naive / 2


def test_target_encoder_fit_transform_differs_from_fit_then_transform() -> None:
    """The one place the two legitimately differ, and it must be the declared one."""
    gen = np.random.default_rng(6)
    frame = pd.DataFrame(
        {"c": gen.choice(list("abcdefgh"), 300), "y": gen.integers(0, 2, 300)}
    )
    context = _context(frame)

    a = TargetEncoder(["c"]).fit_transform(frame, frame["y"], context)["c"]
    b = TargetEncoder(["c"]).fit(frame, frame["y"], context).transform(frame, context)["c"]
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_target_encoder_transform_is_not_cross_fitted() -> None:
    """At transform time the full-train mapping is correct and must be used."""
    gen = np.random.default_rng(7)
    frame = pd.DataFrame(
        {"c": gen.choice(list("abc"), 300), "y": gen.integers(0, 2, 300)}
    )
    context = _context(frame)
    encoder = TargetEncoder(["c"]).fit(frame, frame["y"], context)
    out = encoder.transform(pd.DataFrame({"c": ["a", "b", "c"]}), context)
    for i, level in enumerate("abc"):
        assert out["c"].iloc[i] == pytest.approx(encoder.mappings_["c"][level])


def test_target_encoder_without_target_raises() -> None:
    frame = pd.DataFrame({"c": ["a", "b"] * 50})
    with pytest.raises(LeakageError, match="requires the target"):
        TargetEncoder(["c"]).fit(frame, None, _context(frame.assign(y=0)))


def test_uses_target_is_declared_honestly() -> None:
    """Only transformers that actually read y may claim to."""
    assert TargetEncoder().uses_target is True
    assert Scaler().uses_target is False
    assert MissingValueHandler().uses_target is False
    assert OutlierHandler().uses_target is False
    assert CategoricalEncoder(strategy="onehot").uses_target is False
    assert CategoricalEncoder(strategy="target").uses_target is True


# --- target containment ---------------------------------------------------------------------


def test_target_never_appears_in_transform_output(split_frame) -> None:
    """The simplest possible leak: the model reads the answer off an input column."""
    train, test = split_frame
    pipe = edaprep.AutoPipeline(target="y", random_state=42).fit(train)
    assert "y" not in pipe.transform(train).columns
    assert "y" not in pipe.transform(test).columns
    assert "y" not in pipe.fit_transform(train).columns


def test_target_is_not_transformed(split_frame) -> None:
    """edaprep never transforms a target column, even a skewed one.

    Notebook practice does ``df['price'] = np.log1p(df['price'])`` in place, which is
    easy to forget to invert at prediction time.
    """
    train, _ = split_frame
    train = train.rename(columns={"skewed": "y2"}).drop(columns=["y"])
    pipe = edaprep.AutoPipeline(target="y2", model_family="linear", random_state=1)
    pipe.fit(train)
    assert not any(d.column == "y2" for d in pipe.plan_.decisions)


def test_report_leakage_audit(split_frame) -> None:
    train, _ = split_frame
    pipe = edaprep.AutoPipeline(target="y", random_state=42).fit(train)
    audit = pipe.report_.leakage
    assert audit["statistics_learned_at_fit_only"] is True
    assert audit["target"] == "y"


def test_leaky_column_is_reported_not_silently_dropped() -> None:
    """A column that encodes the answer is flagged, not removed.

    Removing it automatically would be wrong: a legitimately strong feature is
    indistinguishable from a leak without domain knowledge.
    """
    gen = np.random.default_rng(9)
    y = gen.integers(0, 2, 400)
    frame = pd.DataFrame(
        {"x": gen.normal(size=400), "leak": y + gen.normal(0, 1e-6, 400), "y": y}
    )
    pipe = edaprep.AutoPipeline(target="y", random_state=1).fit(frame)
    assert "leak" in pipe.transform(frame).columns
    assert any("LEAKAGE SUSPECTED" in note for note in pipe.plan_.notes)
    assert "leak" in pipe.report_.leakage["columns_suspected_of_leakage"]
