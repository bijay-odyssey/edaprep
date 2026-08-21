"""Dataset profiling: measurements, quality detection, and edge cases."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from edaprep.config import Config, Thresholds
from edaprep.exceptions import EmptyDataError
from edaprep.profiling import profile
from edaprep.profiling.quality import (
    detect_case_variants,
    detect_duplicate_columns,
    detect_numeric_sentinels,
    detect_sentinels,
    missingness_correlation,
)
from edaprep.types import SemanticType, Severity


def test_shape_and_memory(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert p.n_rows == len(messy_frame)
    assert p.n_columns == messy_frame.shape[1]
    assert p.memory_bytes > 0
    assert p.total_missing_cells == int(messy_frame.isna().sum().sum())


def test_input_frame_is_never_mutated(messy_frame: pd.DataFrame) -> None:
    before = messy_frame.copy(deep=True)
    profile(messy_frame, target="target")
    pd.testing.assert_frame_equal(messy_frame, before)


def test_semantic_assignment(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert p["customer_id"].semantic is SemanticType.IDENTIFIER
    assert p["age"].semantic is SemanticType.NUMERIC
    assert p["income"].semantic is SemanticType.NUMERIC
    assert p["city"].semantic is SemanticType.CATEGORICAL
    assert p["signup_date"].semantic is SemanticType.DATETIME
    assert p["notes"].semantic is SemanticType.TEXT
    assert p["constant_col"].semantic is SemanticType.CONSTANT
    assert p["is_active"].semantic is SemanticType.BINARY


def test_column_views(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert "age" in p.numeric_columns
    assert "city" in p.categorical_columns
    assert "signup_date" in p.datetime_columns
    assert "customer_id" in p.identifier_columns
    assert "constant_col" in p.constant_columns
    assert "target" not in p.feature_columns


def test_skew_tiers_are_measured(messy_frame: pd.DataFrame) -> None:
    """The usual skew tiering depends on these being right."""
    p = profile(messy_frame, target="target")
    assert abs(p["age"].skew) < 1.0  # symmetric
    assert p["income"].skew > 1.0  # lognormal, right-skewed


def test_zero_and_negative_counts(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert p["balance"].has_zero
    assert p["balance"].has_negative
    assert not p["income"].has_negative


def test_duplicate_column_detection(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    groups = [set(g) for g in p.duplicate_columns]
    assert {"income", "income_copy"} in groups


def test_sentinel_detection(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert "workclass" in p.sentinels
    assert "?" in p.sentinels["workclass"]


def test_comissingness(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    pairs = {frozenset((a, b)) for a, b, _ in p.comissing_pairs}
    assert frozenset(("income", "income_copy")) in pairs


def test_leakage_flagged_as_error(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    errors = p.issues_of(Severity.ERROR)
    codes = {i.code for i in errors}
    assert "possible_target_leakage" in codes
    leak = next(i for i in errors if i.code == "possible_target_leakage")
    assert "leaky" in leak.columns


def test_target_classification_and_imbalance(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert p.target_kind == "classification"
    assert p.target_classes == 2
    assert 0.0 < p.target_imbalance_ratio < 1.0


def test_regression_target() -> None:
    gen = np.random.default_rng(0)
    frame = pd.DataFrame({"x": gen.normal(size=300), "y": gen.normal(size=300)})
    p = profile(frame, target="y")
    assert p.target_kind == "regression"
    assert p.target_classes is None


def test_target_association_is_computed(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert p["leaky"].target_association > 0.9
    assert p["leaky"].target_association_kind == "eta"
    assert p["target"].target_association is None  # the target itself is skipped


def test_whitespace_and_case_variants(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    assert "country" in p.whitespace_columns
    assert "country" in p.case_variant_columns


def test_unknown_target_raises_with_a_helpful_message(simple_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not a column"):
        profile(simple_frame, target="nope")


def test_non_dataframe_input_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="to_frame"):
        profile(pd.Series([1, 2, 3]))  # type: ignore[arg-type]


def test_no_columns_raises(empty_frame: pd.DataFrame) -> None:
    with pytest.raises(EmptyDataError):
        profile(pd.DataFrame())


def test_zero_row_frame_profiles(empty_frame: pd.DataFrame) -> None:
    """0 rows is a legitimate, if degenerate, input; it must not crash."""
    p = profile(empty_frame)
    assert p.n_rows == 0
    assert p.n_columns == 2
    assert "empty_dataset" in {i.code for i in p.issues}


def test_single_row_frame() -> None:
    p = profile(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert p.n_rows == 1
    assert p["a"].is_constant


def test_single_column_frame() -> None:
    p = profile(pd.DataFrame({"a": np.arange(100.0)}))
    assert p.n_columns == 1


def test_all_missing_column_is_constant() -> None:
    p = profile(pd.DataFrame({"a": [np.nan] * 50, "b": np.arange(50.0)}))
    assert p["a"].semantic is SemanticType.CONSTANT
    assert p["a"].missing_fraction == 1.0


def test_serialisation_round_trips_through_json(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target")
    text = json.dumps(p.to_dict())

    def reject(constant: str) -> None:  # pragma: no cover - only runs on failure
        raise AssertionError(f"non-standard JSON constant {constant!r} in the report")

    # NaN/Infinity must have become null.  json.dumps emits them happily and
    # json.loads accepts them, but they are not valid JSON and every strict parser
    # downstream rejects them, so parse_constant is what actually pins this down.
    restored = json.loads(text, parse_constant=reject)
    assert restored["n_rows"] == p.n_rows
    assert set(restored["columns"]) == set(p.column_order)


def test_summary_renders(messy_frame: pd.DataFrame) -> None:
    text = profile(messy_frame, target="target").summary()
    assert "Dataset" in text
    assert "Semantic types" in text
    assert "target" in text


def test_sampling_is_recorded_and_deterministic() -> None:
    gen = np.random.default_rng(0)
    frame = pd.DataFrame({"x": gen.normal(size=5000), "y": gen.integers(0, 2, 5000)})
    config = Config(random_state=42, sample_size=500)
    config.thresholds = Thresholds(sampling_row_threshold=1000)
    first = profile(frame, target="y", config=config)
    second = profile(frame, target="y", config=config)
    assert first.sampling["used"] is True
    assert first.sampling["n"] == 500
    assert first.sampling["of"] == 5000
    assert first["x"].numeric.mean == second["x"].numeric.mean


def test_missing_counts_stay_exact_under_sampling() -> None:
    """Missing fractions drive drop/impute decisions, so they must not be estimated."""
    gen = np.random.default_rng(1)
    values = np.where(gen.random(5000) < 0.3, np.nan, gen.normal(size=5000))
    frame = pd.DataFrame({"x": values})
    config = Config(random_state=0, sample_size=200)
    config.thresholds = Thresholds(sampling_row_threshold=500)
    p = profile(frame, config=config)
    assert p["x"].n_missing == int(np.isnan(values).sum())
    assert p["x"].numeric.n_missing == int(np.isnan(values).sum())


def test_quick_mode_skips_moments(messy_frame: pd.DataFrame) -> None:
    p = profile(messy_frame, target="target", compute_moments=False)
    assert np.isnan(p["income"].skew)


def test_config_semantic_override_is_honoured(messy_frame: pd.DataFrame) -> None:
    config = Config()
    config.column("grade").semantic_type = "numeric"
    p = profile(messy_frame, target="target", config=config)
    assert p["grade"].semantic is SemanticType.NUMERIC
    assert p["grade"].semantic_confidence == 1.0


# --- quality helpers in isolation -------------------------------------------------


def test_detect_sentinels_is_case_and_whitespace_insensitive() -> None:
    frame = pd.DataFrame({"a": ["x", "N/A", " na ", "NA", "y"]})
    found = detect_sentinels(frame, ["n/a", "na"])
    assert found["a"]["n/a"] == 1
    assert found["a"]["na"] == 2


def test_detect_sentinels_skips_numeric_columns() -> None:
    frame = pd.DataFrame({"n": [1, 2, 3]})
    assert detect_sentinels(frame, ["1"]) == {}


def test_numeric_sentinels_only_when_outside_the_distribution() -> None:
    gen = np.random.default_rng(2)
    outside = pd.DataFrame({"x": np.concatenate([gen.normal(50, 5, 990), [-999.0] * 10])})
    assert "x" in detect_numeric_sentinels(outside, [-999.0])

    inside = pd.DataFrame({"x": np.concatenate([gen.uniform(-2000, 2000, 990), [-999.0] * 10])})
    assert detect_numeric_sentinels(inside, [-999.0]) == {}


def test_missingness_correlation_finds_shared_causes() -> None:
    gen = np.random.default_rng(3)
    mask = gen.random(500) < 0.2
    frame = pd.DataFrame(
        {
            "a": np.where(mask, np.nan, 1.0),
            "b": np.where(mask, np.nan, 2.0),
            "c": np.where(gen.random(500) < 0.2, np.nan, 3.0),
        }
    )
    pairs = missingness_correlation(frame, threshold=0.5)
    assert ("a", "b") in {(x, y) for x, y, _ in pairs}


def test_duplicate_columns_requires_identical_nan_positions() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan],
            "b": [1.0, 2.0, np.nan],
            "c": [1.0, 2.0, 3.0],
        }
    )
    groups = [set(g) for g in detect_duplicate_columns(frame)]
    assert {"a", "b"} in groups
    assert not any("c" in g for g in groups)


def test_case_variants() -> None:
    frame = pd.DataFrame({"c": ["USA", "usa", "Usa", "UK"]})
    assert detect_case_variants(frame)["c"] == [["USA", "Usa", "usa"]]


def test_unhashable_values_do_not_crash_profiling() -> None:
    frame = pd.DataFrame({"a": [[1, 2], [3], [4, 5]], "b": [1.0, 2.0, 3.0]})
    p = profile(frame)
    assert p.n_columns == 2
