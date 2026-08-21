"""Individual transformers: correctness, edge cases, and agreement with scikit-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edaprep.config import Config
from edaprep.core.context import FitContext
from edaprep.exceptions import (
    ConfigurationError,
    NotFittedError,
    SchemaError,
    TransformationError,
)
from edaprep.preprocessing import (
    CategoricalEncoder,
    ColumnDropper,
    ConstantFilter,
    CorrelationFilter,
    DataTypeInference,
    DateTimeExpander,
    DistributionTransformer,
    DuplicateColumnFilter,
    DuplicateRowHandler,
    FrequencyEncoder,
    MissingIndicator,
    MissingnessFilter,
    MissingValueHandler,
    OneHotEncoder,
    OrdinalEncoder,
    OutlierHandler,
    RareCategoryGrouper,
    Scaler,
    TextColumnHandler,
    VarianceFilter,
    detect_outliers,
)
from edaprep.preprocessing.outliers import (
    IQRDetector,
    ModifiedZScoreDetector,
    PercentileDetector,
    ZScoreDetector,
)
from edaprep.profiling import profile

sklearn = pytest.importorskip("sklearn")


def ctx(frame: pd.DataFrame = None, target=None) -> FitContext:
    config = Config(random_state=0)
    prof = profile(frame, target=target, config=config) if frame is not None else None
    return FitContext(config=config, profile=prof, target=target)


# ============================== scaling ==============================================


@pytest.fixture
def numeric_frame() -> pd.DataFrame:
    gen = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": gen.normal(10, 3, 200),
            "b": gen.uniform(-5, 5, 200),
            "c": gen.lognormal(0, 1, 200),
        }
    )


@pytest.mark.parametrize(
    "strategy,sk_name",
    [
        ("standard", "StandardScaler"),
        ("minmax", "MinMaxScaler"),
        ("robust", "RobustScaler"),
        ("maxabs", "MaxAbsScaler"),
    ],
)
def test_scaler_matches_sklearn(numeric_frame, strategy, sk_name) -> None:
    from sklearn import preprocessing as skp

    ours = Scaler(strategy=strategy).fit_transform(numeric_frame, None, ctx(numeric_frame))
    theirs = getattr(skp, sk_name)().fit_transform(numeric_frame)
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-10, atol=1e-12)


def test_scaler_minmax_respects_feature_range(numeric_frame) -> None:
    out = Scaler(strategy="minmax", feature_range=(-1, 1)).fit_transform(
        numeric_frame, None, ctx(numeric_frame)
    )
    assert out["a"].min() == pytest.approx(-1.0)
    assert out["a"].max() == pytest.approx(1.0)


def test_scaler_leaves_zero_variance_alone_and_reports() -> None:
    """sklearn substitutes scale=1.0 silently; we say so."""
    frame = pd.DataFrame({"const": [4.0] * 50, "v": np.arange(50.0)})
    context = ctx(frame)
    scaler = Scaler(["const", "v"], strategy="standard")
    out = scaler.fit_transform(frame, None, context)
    assert scaler.scales_["const"] == 1.0
    assert (out["const"] == 0.0).all()  # centred, not divided by zero
    assert any(w.code == "zero_variance_not_scaled" for w in context.journal.warnings)


def test_scaler_skips_tree_family(numeric_frame) -> None:
    context = ctx(numeric_frame)
    context.config.model_family = "tree"
    out = Scaler().fit_transform(numeric_frame, None, context)
    pd.testing.assert_frame_equal(out, numeric_frame)


def test_scaler_preserves_nan(numeric_frame) -> None:
    frame = numeric_frame.copy()
    frame.loc[frame.index[:10], "a"] = np.nan
    out = Scaler(["a"], strategy="standard").fit_transform(frame, None, ctx(frame))
    assert out["a"].isna().sum() == 10


# ============================== outlier detection =====================================


def test_iqr_detector_matches_the_textbook_fence() -> None:
    values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    q1, q3 = np.quantile(values, [0.25, 0.75])
    bounds = IQRDetector(k=1.5)(values)
    assert bounds.lower == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert bounds.upper == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert bounds.mask(values).sum() == 1


def test_zscore_detector_matches_scipy() -> None:
    from scipy import stats as scipy_stats

    gen = np.random.default_rng(1)
    values = np.concatenate([gen.normal(0, 1, 200), [12.0]])
    ours = ZScoreDetector(threshold=3.0)(values).mask(values)
    theirs = np.abs(scipy_stats.zscore(values)) > 3.0
    np.testing.assert_array_equal(ours, theirs)


def test_modified_zscore_is_robust_where_zscore_is_not() -> None:
    """Masking: enough outliers inflate the sd that the fence no longer reaches them.

    The mean and standard deviation are computed from the very values being detected,
    so a cluster of extremes hides itself.  The median and MAD are not, which is why
    the usual skewed columns needed this detector and got the z-score instead.
    """
    # The largest z attainable with a share p of identical extremes is
    # sqrt((1-p)/p), which drops below 3 once p exceeds ~10%.
    values = np.concatenate([np.arange(100.0), np.full(18, 100_000.0)])
    assert ZScoreDetector(threshold=3.0)(values).mask(values).sum() == 0
    assert ModifiedZScoreDetector(threshold=3.5)(values).mask(values).sum() == 18


def test_modified_zscore_falls_back_when_mad_is_zero() -> None:
    """More than half the column shares one value, so the MAD scale is undefined."""
    values = np.concatenate([np.zeros(80), np.arange(1.0, 21.0)])
    bounds = ModifiedZScoreDetector()(values)
    assert bounds.params["fallback_to_iqr"] == 1.0
    assert np.isfinite(bounds.upper)


def test_detector_never_flags_missing_values() -> None:
    """The usual index-misalignment bug lived exactly here."""
    series = pd.Series([1.0, 2.0, np.nan, 3.0, 1000.0])
    mask = detect_outliers(series, method="iqr")
    assert mask.index.equals(series.index)
    assert not mask.iloc[2]  # the NaN
    assert mask.iloc[4]  # the genuine outlier


def test_detector_with_nan_still_finds_the_outlier() -> None:
    """The sibling notebook's bug: zscore(col) with any NaN returned all-NaN."""
    gen = np.random.default_rng(2)
    values = gen.normal(0, 1, 200)
    values[::10] = np.nan
    values[5] = 50.0
    series = pd.Series(values)
    assert detect_outliers(series, method="zscore").sum() >= 1


def test_percentile_detector() -> None:
    values = np.arange(1000.0)
    bounds = PercentileDetector(0.01, 0.99)(values)
    assert bounds.lower == pytest.approx(np.quantile(values, 0.01))
    assert bounds.upper == pytest.approx(np.quantile(values, 0.99))


@pytest.mark.parametrize("bad", [0, -1])
def test_detector_rejects_non_positive_thresholds(bad) -> None:
    with pytest.raises(ConfigurationError):
        IQRDetector(k=bad)
    with pytest.raises(ConfigurationError):
        ZScoreDetector(threshold=bad)


# ============================== outlier handling ======================================


def test_outlier_report_changes_nothing() -> None:
    frame = pd.DataFrame({"x": np.concatenate([np.arange(100.0), [10_000.0]])})
    context = ctx(frame)
    out = OutlierHandler(["x"], method="iqr", strategy="report").fit_transform(
        frame, None, context
    )
    pd.testing.assert_frame_equal(out, frame)


def test_outlier_clip_bounds_the_column() -> None:
    frame = pd.DataFrame({"x": np.concatenate([np.arange(100.0), [10_000.0]])})
    context = ctx(frame)
    handler = OutlierHandler(["x"], method="iqr", strategy="clip")
    out = handler.fit_transform(frame, None, context)
    assert out["x"].max() == pytest.approx(handler.bounds_["x"].upper)


def test_outlier_impute_produces_nan_for_a_later_imputer() -> None:
    frame = pd.DataFrame({"x": np.concatenate([np.arange(100.0), [10_000.0]])})
    out = OutlierHandler(["x"], method="iqr", strategy="impute").fit_transform(
        frame, None, ctx(frame)
    )
    assert out["x"].isna().sum() == 1


def test_outlier_action_downgraded_when_too_many_flagged() -> None:
    """At 30% flagged the fence describes the distribution, not errors in it."""
    gen = np.random.default_rng(3)
    frame = pd.DataFrame({"x": gen.standard_cauchy(500)})  # very heavy tails
    context = ctx(frame)
    handler = OutlierHandler(["x"], method="percentile", strategy="clip")
    handler.max_action_fraction = 0.01
    handler.fit(frame, None, context)
    assert handler.strategies_["x"] == "report"
    assert any(w.code == "outlier_action_downgraded" for w in context.journal.warnings)


def test_binary_columns_are_excluded_from_outlier_handling() -> None:
    """The IQR fence on a 0/1 column is [0,0]; clipping to it zeroes the column."""
    frame = pd.DataFrame({"flag": ([0] * 90 + [1] * 10), "x": np.arange(100.0)})
    handler = OutlierHandler(strategy="clip")
    out = handler.fit_transform(frame, None, ctx(frame))
    assert "flag" not in handler.columns_
    assert out["flag"].sum() == 10


def test_degenerate_fence_downgrades_to_report() -> None:
    """A zero-heavy column: Q1 == Q3 == 0, so the fence collapses to a point."""
    frame = pd.DataFrame({"z": np.concatenate([np.zeros(400), np.arange(1.0, 101.0)])})
    context = ctx(frame)
    handler = OutlierHandler(["z"], method="iqr", strategy="clip")
    out = handler.fit_transform(frame, None, context)
    assert handler.strategies_["z"] == "report"
    assert out["z"].max() == 100.0  # untouched
    assert any(w.code == "degenerate_outlier_fence" for w in context.journal.warnings)


def test_outlier_remove_does_not_drop_rows_during_transform() -> None:
    frame = pd.DataFrame({"x": np.concatenate([np.arange(100.0), [10_000.0]])})
    context = ctx(frame)
    handler = OutlierHandler(["x"], method="iqr", strategy="remove").fit(
        frame, None, context
    )
    assert len(handler.transform(frame, context)) == len(frame)
    assert handler.rows_to_remove(frame).sum() == 1


# ============================== missing values ========================================


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_imputer_matches_sklearn(strategy) -> None:
    from sklearn.impute import SimpleImputer

    gen = np.random.default_rng(4)
    values = gen.normal(size=200)
    values[::7] = np.nan
    frame = pd.DataFrame({"x": values})
    ours = MissingValueHandler(["x"], strategy=strategy).fit_transform(
        frame, None, ctx(frame)
    )
    theirs = SimpleImputer(strategy=strategy).fit_transform(frame)
    np.testing.assert_allclose(ours.to_numpy(), theirs)


def test_mode_imputation_matches_sklearn() -> None:
    from sklearn.impute import SimpleImputer

    # np.nan rather than None: sklearn's SimpleImputer raises TypeError on a None in an
    # object column (it sorts the values). edaprep handles both, which the next test
    # pins down; here the point is only that the chosen mode agrees.
    frame = pd.DataFrame({"c": ["a", "a", "b", np.nan, np.nan, "c"]})
    ours = MissingValueHandler(["c"], strategy="mode").fit_transform(
        frame, None, ctx(frame)
    )
    theirs = SimpleImputer(strategy="most_frequent").fit_transform(frame)
    assert ours["c"].tolist() == [v[0] for v in theirs]


def test_mode_imputation_handles_none_which_sklearn_cannot() -> None:
    """``None`` and ``np.nan`` both mean missing in an object column."""
    frame = pd.DataFrame({"c": ["a", "a", "b", None, np.nan]})
    out = MissingValueHandler(["c"], strategy="mode").fit_transform(frame, None, ctx(frame))
    assert out["c"].tolist() == ["a", "a", "b", "a", "a"]


def test_auto_uses_median_for_numeric_and_mode_for_categorical() -> None:
    frame = pd.DataFrame(
        {"n": [1.0, 2.0, 100.0, np.nan], "c": ["a", "a", "b", None], "y": [0, 1, 0, 1]}
    )
    handler = MissingValueHandler().fit(frame, frame["y"], ctx(frame, target="y"))
    assert handler.strategies_["n"] == "median"
    assert handler.strategies_["c"] == "mode"


def test_high_cardinality_gets_an_explicit_missing_category() -> None:
    gen = np.random.default_rng(5)
    values = gen.choice([f"c{i}" for i in range(300)], 1000).astype(object)
    values[:50] = None
    frame = pd.DataFrame({"c": values})
    context = ctx(frame)
    handler = MissingValueHandler(["c"]).fit(frame, None, context)
    assert handler.strategies_["c"] == "missing_category"
    out = handler.transform(frame, context)
    assert (out["c"].iloc[:50] == "__missing__").all()


def test_missing_category_works_on_categorical_dtype() -> None:
    """fillna on a categorical raises unless the value is already a category."""
    frame = pd.DataFrame({"c": pd.Categorical(["a", "b", None, "a"])})
    out = MissingValueHandler(["c"], strategy="missing_category").fit_transform(
        frame, None, ctx(frame)
    )
    assert out["c"].isna().sum() == 0


def test_mostly_missing_column_is_imputed_but_reported() -> None:
    frame = pd.DataFrame({"x": [1.0] + [np.nan] * 99})
    context = ctx(frame)
    MissingValueHandler(["x"], strategy="median").fit(frame, None, context)
    assert any(
        w.code == "imputed_mostly_missing_column" for w in context.journal.warnings
    )


def test_constant_imputation_without_a_value_is_an_error() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ConfigurationError, match="fill_value"):
        MissingValueHandler(["x"], strategy="constant").fit(frame, None, ctx(frame))


def test_mean_on_a_string_column_is_an_error_with_a_suggestion() -> None:
    frame = pd.DataFrame({"c": ["a", None, "b"]})
    with pytest.raises(ConfigurationError, match="Use 'mode'"):
        MissingValueHandler(["c"], strategy="mean").fit(frame, None, ctx(frame))


def test_missing_indicator_flags_before_imputation() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})
    out = MissingIndicator().fit_transform(frame, None, ctx(frame))
    assert out["x__was_missing"].tolist() == [0, 1, 0, 1]


def test_missing_indicator_respects_threshold() -> None:
    frame = pd.DataFrame({"x": [np.nan] + [1.0] * 999})  # 0.1% missing
    indicator = MissingIndicator(threshold=0.05).fit(frame, None, ctx(frame))
    assert indicator.columns_ == []


# ============================== encoding ==============================================


def test_onehot_matches_sklearn() -> None:
    from sklearn.preprocessing import OneHotEncoder as SkOneHot

    gen = np.random.default_rng(6)
    frame = pd.DataFrame({"c": gen.choice(["x", "y", "z"], 100)})
    ours = OneHotEncoder(["c"]).fit_transform(frame, None, ctx(frame))
    theirs = SkOneHot(sparse_output=False).fit_transform(frame)
    np.testing.assert_array_equal(ours.to_numpy().astype(float), theirs)


def test_onehot_refuses_extreme_expansion() -> None:
    gen = np.random.default_rng(7)
    frame = pd.DataFrame({"c": gen.choice([f"v{i}" for i in range(500)], 2000)})
    with pytest.raises(ConfigurationError, match="ceiling"):
        OneHotEncoder(["c"], max_columns=100).fit(frame, None, ctx(frame))


def test_onehot_column_order_is_deterministic() -> None:
    gen = np.random.default_rng(8)
    frame = pd.DataFrame({"c": gen.choice(list("dcba"), 100)})
    a = OneHotEncoder(["c"]).fit_transform(frame, None, ctx(frame))
    b = OneHotEncoder(["c"]).fit_transform(frame.iloc[::-1], None, ctx(frame))
    assert list(a.columns) == list(b.columns)
    assert list(a.columns) == ["c_a", "c_b", "c_c", "c_d"]


def test_onehot_drop_first() -> None:
    frame = pd.DataFrame({"c": ["a", "b", "c"] * 20})
    out = OneHotEncoder(["c"], drop_first=True).fit_transform(frame, None, ctx(frame))
    assert list(out.columns) == ["c_b", "c_c"]


def test_ordinal_encoder_honours_an_ordered_categorical() -> None:
    dtype = pd.CategoricalDtype(["low", "medium", "high"], ordered=True)
    frame = pd.DataFrame({"c": pd.Series(["high", "low", "medium"], dtype=dtype)})
    out = OrdinalEncoder(["c"]).fit_transform(frame, None, ctx(frame))
    assert out["c"].tolist() == [2.0, 0.0, 1.0]


def test_ordinal_encoder_marks_unseen_but_keeps_nan_as_nan() -> None:
    train = pd.DataFrame({"c": ["a", "b"] * 20})
    context = ctx(train)
    encoder = OrdinalEncoder(["c"]).fit(train, None, context)
    out = encoder.transform(pd.DataFrame({"c": ["a", "zzz", None]}), context)
    assert out["c"].tolist()[0] == 0.0
    assert out["c"].tolist()[1] == -1.0  # unseen
    assert pd.isna(out["c"].tolist()[2])  # missing stays missing


def test_frequency_encoder_and_unseen_categories() -> None:
    train = pd.DataFrame({"c": ["a"] * 70 + ["b"] * 30})
    context = ctx(train)
    encoder = FrequencyEncoder(["c"]).fit(train, None, context)
    out = encoder.transform(pd.DataFrame({"c": ["a", "b", "new"]}), context)
    assert out["c"].tolist() == [0.7, 0.3, 0.0]


def test_rare_category_grouping() -> None:
    frame = pd.DataFrame({"c": ["common"] * 990 + [f"rare{i}" for i in range(10)]})
    out = RareCategoryGrouper(["c"], threshold=0.01).fit_transform(
        frame, None, ctx(frame)
    )
    assert set(out["c"].unique()) == {"common", "__rare__"}
    assert (out["c"] == "__rare__").sum() == 10


def test_categorical_encoder_routes_by_cardinality() -> None:
    gen = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "low": gen.choice(list("abc"), 500),
            "high": gen.choice([f"v{i}" for i in range(200)], 500),
            "y": gen.integers(0, 2, 500),
        }
    )
    context = ctx(frame, target="y")
    encoder = CategoricalEncoder().fit(frame, frame["y"], context)
    assert encoder.assignments_["low"] == "onehot"
    assert encoder.assignments_["high"] == "target"


def test_categorical_encoder_routes_by_model_family() -> None:
    gen = np.random.default_rng(10)
    frame = pd.DataFrame({"c": gen.choice(list("abc"), 200), "y": gen.integers(0, 2, 200)})
    context = ctx(frame, target="y")
    context.config.model_family = "tree"
    encoder = CategoricalEncoder().fit(frame, frame["y"], context)
    assert encoder.assignments_["c"] == "ordinal"


# ============================== transformations =======================================


def test_yeojohnson_matches_sklearn() -> None:
    from sklearn.preprocessing import PowerTransformer

    gen = np.random.default_rng(11)
    frame = pd.DataFrame({"x": gen.lognormal(0, 1, 300)})
    ours = DistributionTransformer(
        ["x"], method="yeojohnson", standardize=True
    ).fit_transform(frame, None, ctx(frame))
    theirs = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(frame)
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-6, atol=1e-8)


def test_boxcox_matches_scipy() -> None:
    from scipy import stats as scipy_stats

    gen = np.random.default_rng(12)
    values = gen.lognormal(0, 1, 300)
    frame = pd.DataFrame({"x": values})
    ours = DistributionTransformer(["x"], method="boxcox").fit_transform(
        frame, None, ctx(frame)
    )
    theirs, _ = scipy_stats.boxcox(values)
    np.testing.assert_allclose(ours["x"].to_numpy(), theirs, rtol=1e-8)


def test_log_on_negative_values_raises_with_an_alternative() -> None:
    frame = pd.DataFrame({"x": [-1.0, 2.0, 3.0]})
    with pytest.raises(TransformationError, match="yeojohnson"):
        DistributionTransformer(["x"], method="log").fit(frame, None, ctx(frame))


def test_sqrt_on_negative_values_raises() -> None:
    frame = pd.DataFrame({"x": [-1.0, 2.0, 3.0]})
    with pytest.raises(TransformationError):
        DistributionTransformer(["x"], method="sqrt").fit(frame, None, ctx(frame))


def test_constant_column_cannot_be_transformed() -> None:
    frame = pd.DataFrame({"x": [5.0] * 20})
    with pytest.raises(TransformationError, match="constant"):
        DistributionTransformer(["x"], method="log1p").fit(frame, None, ctx(frame))


def test_auto_transform_picks_log1p_for_non_negative_moderate_skew() -> None:
    gen = np.random.default_rng(13)
    frame = pd.DataFrame({"x": gen.lognormal(0, 0.5, 2000)})
    transformer = DistributionTransformer(["x"]).fit(frame, None, ctx(frame))
    assert 1.0 <= abs(frame["x"].skew()) < 5.0, "fixture must sit in the moderate tier"
    assert transformer.methods_["x"] == "log1p"


def test_auto_transform_picks_yeojohnson_for_negative_values() -> None:
    """log is undefined below zero, so the moderate tier must fall back."""
    gen = np.random.default_rng(23)
    values = gen.lognormal(0, 0.5, 2000) - 5.0
    frame = pd.DataFrame({"x": values})
    transformer = DistributionTransformer(["x"]).fit(frame, None, ctx(frame))
    assert transformer.methods_["x"] == "yeojohnson"


def test_auto_transform_leaves_symmetric_columns_alone() -> None:
    gen = np.random.default_rng(14)
    frame = pd.DataFrame({"x": gen.normal(0, 1, 500)})
    transformer = DistributionTransformer(["x"]).fit(frame, None, ctx(frame))
    assert transformer.methods_["x"] == "none"


def test_domain_violation_at_transform_time_is_reported() -> None:
    """Values outside the learned transform's domain must not become NaN silently."""
    train = pd.DataFrame({"x": np.arange(1.0, 101.0)})
    context = ctx(train)
    transformer = DistributionTransformer(["x"], method="log").fit(train, None, context)
    out = transformer.transform(pd.DataFrame({"x": [-5.0, 10.0]}), context)
    assert pd.isna(out["x"].iloc[0])
    assert any(
        w.code == "transform_domain_violation" for w in context.journal.warnings
    )


def test_quantile_transform_clamps_unseen_extremes() -> None:
    train = pd.DataFrame({"x": np.arange(100.0)})
    context = ctx(train)
    transformer = DistributionTransformer(["x"], method="quantile").fit(
        train, None, context
    )
    out = transformer.transform(pd.DataFrame({"x": [-1000.0, 1000.0]}), context)
    assert out["x"].tolist() == [0.0, 1.0]


# ============================== datetime ==============================================


def test_datetime_expansion_keeps_only_varying_features() -> None:
    frame = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=400, freq="D")})
    context = ctx(frame)
    expander = DateTimeExpander(["d"]).fit(frame, None, context)
    out = expander.transform(frame, context)
    assert "d__hour" not in out.columns  # pure dates: hour is always 0
    assert "d__month" in out.columns
    assert "d" not in out.columns  # original dropped


def test_datetime_feature_set_is_frozen_at_fit() -> None:
    """Test data spanning one month must still get the year and month columns."""
    train = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=400, freq="D")})
    context = ctx(train)
    expander = DateTimeExpander(["d"]).fit(train, None, context)
    narrow = pd.DataFrame({"d": pd.date_range("2021-03-01", periods=5, freq="D")})
    out = expander.transform(narrow, context)
    assert list(out.columns) == list(expander.get_feature_names_out())


def test_datetime_rejects_unknown_features() -> None:
    frame = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=10)})
    with pytest.raises(ConfigurationError):
        DateTimeExpander(["d"], features=["nonsense"]).fit(frame, None, ctx(frame))


# ============================== casting ===============================================


def test_sentinels_become_nan() -> None:
    frame = pd.DataFrame({"c": ["a", "?", "b", "N/A", " NA "]})
    context = ctx(frame)
    out = DataTypeInference(["c"]).fit_transform(frame, None, context)
    assert out["c"].isna().sum() == 3


def test_whitespace_is_stripped() -> None:
    frame = pd.DataFrame({"c": ["USA", " usa ", "USA"]})
    out = DataTypeInference(["c"]).fit_transform(frame, None, ctx(frame))
    assert out["c"].tolist() == ["USA", "usa", "USA"]


def test_numeric_strings_are_parsed() -> None:
    gen = np.random.default_rng(15)
    frame = pd.DataFrame({"n": [f"{v:.2f}" for v in gen.normal(50, 10, 200)]})
    out = DataTypeInference().fit_transform(frame, None, ctx(frame))
    assert pd.api.types.is_numeric_dtype(out["n"].dtype)


def test_integer_downcast_preserves_values() -> None:
    frame = pd.DataFrame({"i": np.arange(200, dtype="int64")})
    out = DataTypeInference(["i"], downcast_integers=True).fit_transform(
        frame, None, ctx(frame)
    )
    assert out["i"].dtype == np.uint8
    np.testing.assert_array_equal(out["i"].to_numpy(), frame["i"].to_numpy())


def test_float_downcast_is_refused_when_values_would_collide() -> None:
    """Distinct rows must not become identical to the model."""
    frame = pd.DataFrame({"f": [1.000000001, 1.000000002, 2.0]})
    out = DataTypeInference(["f"], downcast_floats=True).fit_transform(
        frame, None, ctx(frame)
    )
    assert out["f"].dtype == np.float64
    assert out["f"].nunique() == 3


def test_float_downcast_is_refused_when_out_of_range() -> None:
    frame = pd.DataFrame({"f": [1e39, 2e39, 3.0]})
    out = DataTypeInference(["f"], downcast_floats=True).fit_transform(
        frame, None, ctx(frame)
    )
    assert out["f"].dtype == np.float64  # float32 would overflow to inf


def test_float_downcast_applies_when_safe() -> None:
    frame = pd.DataFrame({"f": np.arange(100.0)})
    context = ctx(frame)
    out = DataTypeInference(["f"], downcast_floats=True).fit_transform(
        frame, None, context
    )
    assert out["f"].dtype == np.float32
    assert any(w.code == "float_downcast_is_lossy" for w in context.journal.warnings)


# ============================== selection =============================================


def test_constant_filter() -> None:
    frame = pd.DataFrame({"c": [1] * 50, "v": np.arange(50)})
    out = ConstantFilter().fit_transform(frame, None, ctx(frame))
    assert list(out.columns) == ["v"]


def test_near_constant_filter() -> None:
    frame = pd.DataFrame({"nc": ["a"] * 999 + ["b"], "v": np.arange(1000)})
    out = ConstantFilter(near_constant_ratio=0.99).fit_transform(frame, None, ctx(frame))
    assert list(out.columns) == ["v"]


def test_missingness_filter() -> None:
    frame = pd.DataFrame({"m": [1.0] + [np.nan] * 99, "v": np.arange(100.0)})
    out = MissingnessFilter(threshold=0.6).fit_transform(frame, None, ctx(frame))
    assert list(out.columns) == ["v"]


def test_duplicate_column_filter_keeps_the_first() -> None:
    frame = pd.DataFrame({"a": [1.0, 2, 3], "b": [1.0, 2, 3], "c": [4.0, 5, 6]})
    out = DuplicateColumnFilter().fit_transform(frame, None, ctx(frame))
    assert list(out.columns) == ["a", "c"]


def test_variance_filter_warns_about_scale_dependence() -> None:
    frame = pd.DataFrame({"small": np.arange(100) * 1e-6, "big": np.arange(100.0)})
    context = ctx(frame)
    VarianceFilter(threshold=1e-6).fit(frame, None, context)
    assert any(
        w.code == "variance_filter_is_scale_dependent" for w in context.journal.warnings
    )


def test_correlation_filter_is_order_independent() -> None:
    """The usual greedy version gives different answers for different column orders."""
    gen = np.random.default_rng(16)
    base = gen.normal(size=500)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base + gen.normal(0, 0.001, 500),
            "c": base + gen.normal(0, 0.001, 500),
            "d": gen.normal(size=500),
        }
    )
    forward = CorrelationFilter(threshold=0.95).fit(frame, None, ctx(frame))
    reversed_frame = frame[["d", "c", "b", "a"]]
    backward = CorrelationFilter(threshold=0.95).fit(
        reversed_frame, None, ctx(reversed_frame)
    )
    # The same group is identified either way; only the representative may differ.
    assert {frozenset(g) for g in forward.groups_} == {
        frozenset(g) for g in backward.groups_
    }
    assert len(forward.to_drop_) == len(backward.to_drop_) == 2


def test_correlation_filter_keeps_chain_information() -> None:
    """a~b and b~c with a,c independent: notebook practice drops both b and c."""
    gen = np.random.default_rng(17)
    a = gen.normal(size=800)
    c = gen.normal(size=800)
    b = (a + c) / np.sqrt(2)
    frame = pd.DataFrame({"a": a, "b": b, "c": c})
    filt = CorrelationFilter(threshold=0.6).fit(frame, None, ctx(frame))
    kept = [col for col in frame.columns if col not in filt.to_drop_]
    assert len(kept) >= 1
    assert len(filt.groups_) == 1  # one connected component, not two independent drops


def test_column_dropper_tolerates_absent_columns() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3]})
    context = ctx(frame)
    out = ColumnDropper(["a", "nonexistent"]).fit_transform(frame, None, context)
    assert list(out.columns) == []
    assert any(w.code == "drop_column_not_found" for w in context.journal.warnings)


# ============================== duplicates / text ======================================


def test_duplicate_rows_reported_by_default() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    context = ctx(frame)
    handler = DuplicateRowHandler(strategy="report")
    out = handler.fit_transform(frame, None, context)
    assert len(out) == 3
    assert handler.stats_["n_duplicate_rows"] == 1


def test_duplicate_rows_removed_at_fit_only() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2]})
    context = ctx(frame)
    handler = DuplicateRowHandler(strategy="remove")
    assert len(handler.fit_transform(frame, None, context)) == 2
    assert len(handler.transform(frame, context)) == 3  # transform never drops rows


def test_text_columns_dropped_with_a_note() -> None:
    frame = pd.DataFrame(
        {"t": [f"A long free text remark number {i} about the service." for i in range(100)]}
    )
    context = ctx(frame)
    out = TextColumnHandler().fit_transform(frame, None, context)
    assert list(out.columns) == []
    assert any(w.code == "text_columns_dropped" for w in context.journal.warnings)


def test_text_length_features() -> None:
    frame = pd.DataFrame(
        {"t": [f"A long free text remark number {i} about the service." for i in range(100)]}
    )
    out = TextColumnHandler(strategy="length_features").fit_transform(
        frame, None, ctx(frame)
    )
    assert "t__length" in out.columns and "t__n_words" in out.columns


# ============================== the contract ===========================================


ALL_TRANSFORMERS = [
    Scaler,
    MissingValueHandler,
    MissingIndicator,
    OutlierHandler,
    CategoricalEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    FrequencyEncoder,
    RareCategoryGrouper,
    DistributionTransformer,
    DateTimeExpander,
    DataTypeInference,
    ConstantFilter,
    MissingnessFilter,
    DuplicateColumnFilter,
    VarianceFilter,
    CorrelationFilter,
    TextColumnHandler,
]


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS, ids=lambda c: c.__name__)
def test_transform_before_fit_raises(cls) -> None:
    with pytest.raises(NotFittedError, match="not fitted"):
        cls().transform(pd.DataFrame({"a": [1.0, 2.0]}))


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS, ids=lambda c: c.__name__)
def test_get_set_params_round_trip(cls) -> None:
    transformer = cls()
    params = transformer.get_params(deep=False)
    assert "columns" in params
    transformer.set_params(**params)
    assert transformer.get_params(deep=False) == params


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS, ids=lambda c: c.__name__)
def test_repr_does_not_raise_before_or_after_fit(cls) -> None:
    transformer = cls()
    assert repr(transformer)
    gen = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "n": gen.normal(size=60),
            "c": gen.choice(list("abc"), 60),
            "d": pd.date_range("2020-01-01", periods=60),
        }
    )
    transformer.fit(frame, None, ctx(frame))
    assert repr(transformer)


def test_set_params_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Invalid parameter"):
        Scaler().set_params(nonsense=1)


def test_missing_column_at_transform_raises_schema_error() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    context = ctx(frame)
    scaler = Scaler(["a", "b"]).fit(frame, None, context)
    with pytest.raises(SchemaError, match="missing from the input"):
        scaler.transform(pd.DataFrame({"a": [1.0]}), context)


def test_unexpected_column_at_transform_raises_by_default() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    context = ctx(frame)
    scaler = Scaler(["a"]).fit(frame, None, context)
    with pytest.raises(SchemaError, match="not seen at 'fit' time"):
        scaler.transform(pd.DataFrame({"a": [1.0], "surprise": [2.0]}), context)


def test_unexpected_column_can_be_tolerated() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    context = ctx(frame)
    context.config.on_unknown_columns = "ignore"
    scaler = Scaler(["a"]).fit(frame, None, context)
    out = scaler.transform(pd.DataFrame({"a": [1.0], "surprise": [2.0]}), context)
    assert "surprise" in out.columns


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS, ids=lambda c: c.__name__)
def test_input_frame_is_never_mutated(cls) -> None:
    gen = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "n": np.where(gen.random(60) < 0.1, np.nan, gen.normal(size=60)),
            "c": gen.choice(list("abc"), 60),
            "d": pd.date_range("2020-01-01", periods=60),
        }
    )
    before = frame.copy(deep=True)
    cls().fit_transform(frame, None, ctx(frame))
    pd.testing.assert_frame_equal(frame, before)
