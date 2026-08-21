"""Semantic type inference.

The cases here are the ones notebook code gets wrong: integer-coded categories,
numeric IDs, dates stored as strings, and boolean-ish text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edaprep.config import Thresholds
from edaprep.profiling.column_types import (
    infer_semantic_type,
    is_integral,
    name_suggests,
)
from edaprep.types import SemanticType


def infer(values, name="col", **kwargs) -> SemanticType:
    series = pd.Series(values, name=name)
    return infer_semantic_type(series, **kwargs).semantic


N = 200


def test_continuous_float_is_numeric() -> None:
    gen = np.random.default_rng(0)
    assert infer(gen.normal(0, 1, N), "temperature") is SemanticType.NUMERIC


def test_integer_coded_category_is_not_numeric() -> None:
    """The headline correction: the dtype split sends this straight to StandardScaler."""
    gen = np.random.default_rng(1)
    assert infer(gen.integers(0, 5, N), "payment_type") is SemanticType.CATEGORICAL


def test_float_holding_whole_numbers_is_still_category_like() -> None:
    gen = np.random.default_rng(2)
    values = gen.integers(0, 4, N).astype(float)
    assert infer(values, "cluster") is SemanticType.CATEGORICAL


def test_consecutive_integer_levels_with_ordinal_name_are_ordinal() -> None:
    gen = np.random.default_rng(3)
    assert infer(gen.integers(1, 6, N), "satisfaction_level") is SemanticType.ORDINAL


def test_sequential_integer_id_is_identifier() -> None:
    assert infer(np.arange(N), "customer_id") is SemanticType.IDENTIFIER


def test_high_cardinality_string_id_is_identifier() -> None:
    values = [f"a3f{i:07d}" for i in range(N)]
    assert infer(values, "transaction_ref") is SemanticType.IDENTIFIER


def test_unique_float_column_is_not_an_identifier() -> None:
    """Distinct floats are the normal case, not a sign of a row key."""
    gen = np.random.default_rng(4)
    inference = infer_semantic_type(pd.Series(gen.normal(0, 1, N), name="price_id"))
    assert inference.semantic is SemanticType.NUMERIC


def test_target_is_never_an_identifier() -> None:
    assert infer(np.arange(N), "y", is_target=True) is not SemanticType.IDENTIFIER


def test_two_valued_columns_are_binary() -> None:
    assert infer([0, 1] * 100, "flag") is SemanticType.BINARY
    assert infer(["yes", "no"] * 100, "consent") is SemanticType.BINARY
    assert infer([True, False] * 100, "active") is SemanticType.BINARY


def test_boolean_like_strings_suggest_a_boolean_dtype() -> None:
    inference = infer_semantic_type(pd.Series(["True", "False"] * 100, name="flag"))
    assert inference.semantic is SemanticType.BINARY
    assert inference.suggested_dtype == "boolean"


def test_datetime_dtype_and_date_strings() -> None:
    dates = pd.date_range("2020-01-01", periods=N)
    assert infer(dates, "signup") is SemanticType.DATETIME
    assert infer(dates.strftime("%Y-%m-%d").tolist(), "signup") is SemanticType.DATETIME


def test_bare_integers_are_not_parsed_as_years() -> None:
    """``pd.to_datetime("2020")`` succeeds; that must not make an ID column a date."""
    values = [str(2000 + i) for i in range(N)]
    assert infer(values, "code") is not SemanticType.DATETIME


def test_free_text_is_text_not_categorical() -> None:
    values = [
        f"The customer number {i} reported a problem with delivery and asked for help."
        for i in range(N)
    ]
    assert infer(values, "comment") is SemanticType.TEXT


def test_short_strings_are_categorical() -> None:
    assert infer(["red", "green", "blue"] * 70, "colour") is SemanticType.CATEGORICAL


def test_numeric_strings_are_recognised_and_a_cast_is_suggested() -> None:
    gen = np.random.default_rng(5)
    values = [f"{v:.3f}" for v in gen.normal(50, 10, N)]
    inference = infer_semantic_type(pd.Series(values, name="measure"))
    assert inference.semantic is SemanticType.NUMERIC
    assert inference.suggested_dtype == "float64"


def test_constant_and_all_missing() -> None:
    assert infer([1.0] * N, "c") is SemanticType.CONSTANT
    assert infer([np.nan] * N, "c") is SemanticType.CONSTANT


def test_user_hint_wins_with_full_confidence() -> None:
    inference = infer_semantic_type(
        pd.Series(np.arange(N), name="zip"), hint=SemanticType.CATEGORICAL
    )
    assert inference.semantic is SemanticType.CATEGORICAL
    assert inference.confidence == 1.0


def test_thresholds_scale_with_row_count() -> None:
    """12 distinct integers is categorical in a big frame, continuous in a tiny one."""
    gen = np.random.default_rng(6)
    big = pd.Series(gen.integers(0, 12, 5000), name="v")
    small = pd.Series(np.arange(12), name="v")
    assert infer_semantic_type(big).semantic is SemanticType.CATEGORICAL
    assert infer_semantic_type(small).semantic is SemanticType.NUMERIC


def test_name_alone_never_decides_identifier() -> None:
    """A column called ``user_id`` with 3 values in 200 rows is not an identifier."""
    gen = np.random.default_rng(7)
    assert infer(gen.integers(0, 3, N), "user_id") is not SemanticType.IDENTIFIER


def test_uncertainty_is_exposed() -> None:
    gen = np.random.default_rng(8)
    inference = infer_semantic_type(pd.Series(gen.integers(0, 8, N), name="v"))
    assert 0.0 < inference.confidence < 1.0
    assert inference.alternatives
    assert inference.reasons


def test_categorical_dtype_is_honoured() -> None:
    series = pd.Series(pd.Categorical(["a", "b", "c"] * 70), name="v")
    assert infer_semantic_type(series).semantic is SemanticType.CATEGORICAL


def test_extension_dtypes() -> None:
    gen = np.random.default_rng(9)
    nullable = pd.Series(pd.array(gen.normal(0, 1, N), dtype="Float64"), name="v")
    assert infer_semantic_type(nullable).semantic is SemanticType.NUMERIC
    string_series = pd.Series(["a", "b"] * 100, dtype="string", name="v")
    assert infer_semantic_type(string_series).semantic is SemanticType.BINARY


def test_timedelta_is_numeric() -> None:
    series = pd.Series(pd.to_timedelta(np.arange(N), unit="D"), name="elapsed")
    assert infer_semantic_type(series).semantic is SemanticType.NUMERIC


@pytest.mark.parametrize(
    "name,key",
    [
        ("customer_id", "identifier"),
        ("order_no", "identifier"),
        ("created_at", "datetime"),
        ("satisfaction_rating", "ordinal"),
        ("free_text_comment", "text"),
        ("country", "categorical"),
    ],
)
def test_name_heuristics(name: str, key: str) -> None:
    assert name_suggests(name)[key]


def test_residual_does_not_match_the_id_pattern() -> None:
    """Word-boundary anchoring: 'residual' contains 'id' but is not an identifier."""
    assert not name_suggests("residual_sugar")["identifier"]


def test_is_integral() -> None:
    assert is_integral(pd.Series([1, 2, 3]))
    assert is_integral(pd.Series([1.0, 2.0, np.nan]))
    assert not is_integral(pd.Series([1.5, 2.0]))
    assert not is_integral(pd.Series(["a", "b"]))


def test_custom_thresholds_change_the_outcome() -> None:
    gen = np.random.default_rng(10)
    values = pd.Series(gen.integers(0, 30, 5000), name="v")
    assert infer_semantic_type(values).semantic is SemanticType.NUMERIC
    loose = Thresholds(numeric_as_categorical_max=40)
    assert infer_semantic_type(values, thresholds=loose).semantic is SemanticType.CATEGORICAL


def test_inference_is_deterministic() -> None:
    gen = np.random.default_rng(11)
    series = pd.Series(gen.integers(0, 100, 3000).astype(str), name="v")
    first = infer_semantic_type(series, random_state=42)
    second = infer_semantic_type(series, random_state=42)
    assert first.semantic is second.semantic
    assert first.confidence == second.confidence
