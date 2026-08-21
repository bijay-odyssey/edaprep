"""The EDA engine and the config system."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import edaprep
from edaprep import EDA, Config
from edaprep.config import Thresholds
from edaprep.eda.categorical import category_frequencies
from edaprep.eda.correlation import (
    correlation_matrix,
    top_correlated_pairs,
    variance_inflation,
)
from edaprep.eda.target import benjamini_hochberg
from edaprep.exceptions import ConfigurationError, EmptyDataError
from edaprep.types import AnalysisLevel, ModelFamily, Severity


@pytest.fixture
def analysis_frame() -> pd.DataFrame:
    gen = np.random.default_rng(31)
    n = 500
    signal = gen.normal(size=n)
    y = (signal + gen.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame(
        {
            "informative": signal,
            "noise": gen.normal(size=n),
            "collinear_a": signal * 2 + gen.normal(0, 1e-3, n),
            "skewed": gen.lognormal(0, 1.5, n),
            "with_missing": np.where(gen.random(n) < 0.2, np.nan, gen.normal(size=n)),
            "cat": gen.choice(list("abcd"), n, p=[0.5, 0.3, 0.15, 0.05]),
            "const": 1.0,
            "y": y,
        }
    )


# ============================== analysis levels =======================================


def test_quick_skips_expensive_sections(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("quick")
    assert report.columns is not None
    assert report.missing is not None
    assert report.numerical is None
    assert report.correlation is None
    assert report.vif is None


def test_standard_includes_distributions_and_correlation(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    assert report.numerical is not None and not report.numerical.empty
    assert report.categorical is not None
    assert report.outliers is not None
    assert report.correlation is not None
    assert report.target_relationships is not None
    assert report.vif is None  # deep only


def test_deep_adds_vif_and_significance_tests(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("deep")
    assert report.vif is not None
    assert "p_value" in report.target_relationships.columns
    assert "q_value" in report.target_relationships.columns


def test_include_and_exclude(analysis_frame) -> None:
    eda = EDA(analysis_frame, target="y")
    only = eda.analyze("standard", include=["numerical"])
    assert only.numerical is not None
    assert only.categorical is None

    without = eda.analyze("standard", exclude=["correlation"])
    assert without.correlation is None
    assert without.numerical is not None


def test_quick_is_measurably_cheaper(analysis_frame) -> None:
    """The levels must differ in work done, not just in what is displayed."""
    import time

    eda = EDA(analysis_frame, target="y")
    start = time.perf_counter()
    eda.analyze("quick")
    quick = time.perf_counter() - start
    start = time.perf_counter()
    eda.analyze("deep")
    deep = time.perf_counter() - start
    assert quick < deep


# ============================== content ================================================


def test_numerical_summary_is_sorted_by_skew(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    frame = report.numerical
    assert frame.iloc[0]["column"] == "skewed"
    assert "heavily right-skewed" in frame.iloc[0]["distribution"]


def test_categorical_summary_reports_cardinality(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    row = report.categorical.set_index("column").loc["cat"]
    assert row["n_unique"] == 4
    assert row["modal_value"] == "a"


def test_missing_report_lists_only_columns_with_gaps(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("quick")
    assert set(report.missing["column"]) == {"with_missing"}


def test_outlier_summary_shows_methods_disagreeing(analysis_frame) -> None:
    """Different fences give different counts; showing that is the point."""
    report = EDA(analysis_frame, target="y").analyze("standard")
    row = report.outliers.set_index("column").loc["skewed"]
    assert row["n_iqr"] != row["n_modified_z"]
    assert row["recommended"] in ("iqr", "iqr (k=3)", "zscore", "modified_zscore")


def test_correlated_pairs_finds_the_collinear_pair(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    pairs = {frozenset((a, b)) for a, b in report.correlated_pairs[["column_a", "column_b"]].to_numpy()}
    assert frozenset(("informative", "collinear_a")) in pairs


def test_target_summary_reports_imbalance() -> None:
    gen = np.random.default_rng(32)
    frame = pd.DataFrame(
        {"x": gen.normal(size=1000), "y": (gen.random(1000) < 0.03).astype(int)}
    )
    report = EDA(frame, target="y").analyze("standard")
    assert report.target["kind"] == "classification"
    assert report.target["imbalance_ratio"] < 0.1
    assert "resampling" in report.target["note"]


def test_regression_target_summary_warns_about_transforming_in_place() -> None:
    gen = np.random.default_rng(33)
    frame = pd.DataFrame({"x": gen.normal(size=500), "y": gen.lognormal(0, 1.5, 500)})
    report = EDA(frame, target="y").analyze("standard")
    assert report.target["kind"] == "regression"
    assert "log1p" in report.target["note"]


def test_target_relationships_rank_the_informative_feature_first(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    assert report.target_relationships.iloc[0]["column"] in (
        "informative",
        "collinear_a",
    )


def test_significance_tests_are_named(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("deep")
    tests = set(report.target_relationships["test"].dropna())
    assert "welch t-test" in tests  # numeric feature, binary target
    assert "chi-squared" in tests  # categorical feature, binary target


# ============================== VIF ====================================================


def test_vif_matches_the_regression_definition() -> None:
    """VIF_i = 1/(1-R2_i) from regressing feature i on the others."""
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    gen = np.random.default_rng(34)
    a = gen.normal(size=400)
    b = gen.normal(size=400)
    frame = pd.DataFrame({"a": a, "b": b, "c": a + b + gen.normal(0, 0.3, 400)})
    result = variance_inflation(frame, edaprep.profile(frame)).set_index("column")

    others = frame[["b", "c"]].to_numpy()
    model = sklearn_lm.LinearRegression().fit(others, frame["a"].to_numpy())
    r2 = model.score(others, frame["a"].to_numpy())
    assert result.loc["a", "vif"] == pytest.approx(1.0 / (1.0 - r2), rel=1e-6)


def test_vif_reports_perfect_collinearity_as_infinite() -> None:
    gen = np.random.default_rng(35)
    a = gen.normal(size=300)
    b = gen.normal(size=300)
    frame = pd.DataFrame({"a": a, "b": b, "c": a + b})  # exact linear combination
    result = variance_inflation(frame, edaprep.profile(frame)).set_index("column")
    assert not np.isfinite(result.loc["c", "vif"])
    assert "collinear" in result.loc["c", "note"]


def test_vif_flags_severity() -> None:
    gen = np.random.default_rng(36)
    a = gen.normal(size=500)
    frame = pd.DataFrame({"a": a, "b": a + gen.normal(0, 0.1, 500), "c": gen.normal(size=500)})
    result = variance_inflation(frame, edaprep.profile(frame)).set_index("column")
    assert "multicollinearity" in result.loc["a", "note"]
    assert result.loc["c", "note"] == ""


# ============================== multiple testing ========================================


def test_benjamini_hochberg_controls_the_false_discovery_rate() -> None:
    """400 null features at alpha=0.05 give ~20 spurious 'significant' hits."""
    gen = np.random.default_rng(37)
    p_values = gen.uniform(size=400)  # all null
    q_values = benjamini_hochberg(p_values)
    assert (p_values < 0.05).sum() > 10  # the problem
    assert (q_values < 0.05).sum() == 0  # the fix


def test_benjamini_hochberg_is_monotone_and_bounded() -> None:
    gen = np.random.default_rng(38)
    p_values = np.sort(gen.uniform(size=100))
    q_values = benjamini_hochberg(p_values)
    assert np.all(np.diff(q_values) >= -1e-12)
    assert np.all(q_values >= p_values - 1e-12)
    assert np.all(q_values <= 1.0)


def test_benjamini_hochberg_keeps_a_real_signal() -> None:
    p_values = np.concatenate([[1e-10, 1e-9], np.random.default_rng(39).uniform(size=98)])
    assert (benjamini_hochberg(p_values)[:2] < 0.05).all()


def test_benjamini_hochberg_handles_nan() -> None:
    q = benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(q[1])
    assert np.isfinite(q[0])


# ============================== correlation helpers =====================================


def test_correlation_matrix_skipped_on_wide_frames_unless_forced() -> None:
    gen = np.random.default_rng(40)
    frame = pd.DataFrame({f"c{i}": gen.normal(size=50) for i in range(300)})
    prof = edaprep.profile(frame)
    config = Config()
    config.thresholds = Thresholds(correlation_max_columns=100)
    assert correlation_matrix(frame, prof, config=config) is None
    assert correlation_matrix(frame, prof, config=config, force=True) is not None


def test_top_correlated_pairs_returns_upper_triangle_only() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
        columns=list("abc"),
        index=list("abc"),
    )
    pairs = top_correlated_pairs(corr, threshold=0.5)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["column_a", "column_b"]]) == {"a", "b"}


def test_category_frequencies_includes_missing_by_default() -> None:
    frame = pd.DataFrame({"c": ["a", "a", "b", None]})
    table = category_frequencies(frame, "c")
    assert table["count"].sum() == 4
    assert table["cumulative_%"].iloc[-1] == pytest.approx(100.0)


# ============================== rendering / serialisation ================================


def test_eda_report_serialises(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    data = json.loads(report.to_json())
    assert data["level"] == "standard"
    assert isinstance(data["numerical"], list)
    assert data["dataset"]["n_rows"] == len(analysis_frame)


def test_eda_report_summary_and_html(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    text = report.summary()
    assert "EDA report" in text and "Numerical summary" in text
    html = report.to_html()
    assert "<title>EDA report</title>" in html
    assert "http://" not in html and "https://" not in html


def test_eda_report_exposes_findings(analysis_frame) -> None:
    report = EDA(analysis_frame, target="y").analyze("standard")
    codes = {i.code for i in report.issues}
    assert "constant_columns" in codes
    assert all(i.severity is not Severity.INFO for i in report.warnings)


def test_eda_rejects_bad_input() -> None:
    with pytest.raises(TypeError, match="to_frame"):
        EDA(pd.Series([1, 2, 3]))  # type: ignore[arg-type]
    with pytest.raises(EmptyDataError):
        EDA(pd.DataFrame())
    with pytest.raises(KeyError, match="not a column"):
        EDA(pd.DataFrame({"a": [1]}), target="nope")


def test_eda_never_mutates_input(analysis_frame) -> None:
    before = analysis_frame.copy(deep=True)
    EDA(analysis_frame, target="y").analyze("deep")
    pd.testing.assert_frame_equal(analysis_frame, before)


def test_eda_on_zero_row_frame() -> None:
    frame = pd.DataFrame({"a": pd.Series(dtype="float64")})
    report = EDA(frame).analyze("quick")
    assert report.dataset["n_rows"] == 0


# ============================== configuration ===========================================


def test_config_round_trips_through_dict() -> None:
    config = Config(random_state=7, model_family="tree", outlier_strategy="clip")
    config.column("a").imputation = "mean"
    config.thresholds.skew_heavy = 4.0

    restored = Config.from_dict(config.to_dict())
    assert restored.random_state == 7
    assert restored.model_family is ModelFamily.TREE
    assert restored.thresholds.skew_heavy == 4.0
    assert restored.column("a").imputation == "mean"


def test_config_copy_is_deep() -> None:
    config = Config()
    config.column("a").imputation = "mean"
    clone = config.copy()
    clone.column("a").imputation = "median"
    assert config.column("a").imputation == "mean"


def test_config_rejects_unknown_strategies() -> None:
    with pytest.raises(ConfigurationError, match="not a valid value"):
        Config(missing_strategy="telepathy")
    with pytest.raises(ConfigurationError, match="not a valid value"):
        Config(scaling="vibes")


def test_config_rejects_contradictory_thresholds() -> None:
    with pytest.raises(ConfigurationError, match="skew_heavy"):
        Config(thresholds=Thresholds(skew_moderate=5.0, skew_heavy=1.0))
    with pytest.raises(ConfigurationError, match="high_cardinality_threshold"):
        Config(
            thresholds=Thresholds(
                high_cardinality_threshold=2000, extreme_cardinality_threshold=100
            )
        )


def test_config_rejects_out_of_range_values() -> None:
    with pytest.raises(ConfigurationError, match="out of range"):
        Config(thresholds=Thresholds(id_unique_ratio=1.5))
    with pytest.raises(ConfigurationError, match="out of range"):
        Config(rare_category_threshold=2.0)


def test_target_encoding_folds_must_allow_holdout() -> None:
    with pytest.raises(ConfigurationError, match="at least 2"):
        Config(target_encoding_folds=1)


def test_constant_imputation_needs_a_value() -> None:
    config = Config()
    config.column("a").imputation = "constant"
    with pytest.raises(ConfigurationError, match="imputation_fill_value"):
        config.validate()


def test_column_accessor_creates_on_demand_but_get_does_not() -> None:
    config = Config()
    assert config.get_column("a") is None
    config.column("a")
    assert config.get_column("a") is not None


def test_effective_accessors_prefer_explicit_values() -> None:
    config = Config(rare_category_threshold=0.05, high_cardinality_threshold=10)
    assert config.effective_rare_threshold == 0.05
    assert config.effective_high_cardinality == 10
    assert Config().effective_rare_threshold == Thresholds().rare_category_threshold


def test_thresholds_actually_change_behaviour() -> None:
    """A threshold nobody reads is a magic number with extra steps."""
    gen = np.random.default_rng(41)
    frame = pd.DataFrame(
        {"c": gen.choice([f"v{i}" for i in range(30)], 600), "y": gen.integers(0, 2, 600)}
    )
    strict = Config(random_state=0, high_cardinality_threshold=10, model_family="linear")
    loose = Config(random_state=0, high_cardinality_threshold=100, model_family="linear")

    a = edaprep.AutoPipeline(target="y", config=strict).fit(frame)
    b = edaprep.AutoPipeline(target="y", config=loose).fit(frame)
    assert a["categorical_encoder"].assignments_["c"] == "target"
    assert b["categorical_encoder"].assignments_["c"] == "onehot"


def test_analysis_level_and_severity_coercion() -> None:
    assert AnalysisLevel.coerce("QUICK") is AnalysisLevel.QUICK
    assert Severity.coerce("warning") is Severity.WARNING
    assert ModelFamily.coerce("Tree") is ModelFamily.TREE
    with pytest.raises(ValueError, match="not a valid"):
        AnalysisLevel.coerce("exhaustive")


def test_config_repr_is_readable() -> None:
    config = Config(random_state=1, model_family="tree")
    config.column("a").imputation = "mean"
    text = repr(config)
    assert "random_state=1" in text
    assert "column override" in text
