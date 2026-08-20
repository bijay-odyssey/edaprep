"""Shared fixtures.

``messy_frame`` deliberately reproduces every pathology found in notebook practice:
sentinel strings, an ID column, a constant, a near-constant, a duplicated column,
co-missing pairs, a heavily skewed column, a zero-heavy column, a high-cardinality
category, an integer-coded category, whitespace variants, and a leaky column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20240821)


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    n = 600
    gen = np.random.default_rng(7)

    age = gen.normal(40, 12, n).round().clip(18, 90)
    age[gen.choice(n, 30, replace=False)] = np.nan

    income = gen.lognormal(10.5, 1.1, n)  # heavily right-skewed
    income[gen.choice(n, 12, replace=False)] = np.nan

    balance = gen.normal(0, 500, n)
    balance[gen.choice(n, 200, replace=False)] = 0.0  # zero-heavy

    city = gen.choice(
        [f"city_{i:03d}" for i in range(180)], n
    )  # high cardinality categorical

    workclass = gen.choice(["Private", "Gov", "SelfEmp", "?"], n, p=[0.6, 0.2, 0.15, 0.05])
    occupation = np.where(workclass == "?", "?", gen.choice(["Tech", "Sales", "Admin"], n))

    grade = gen.integers(1, 6, n)  # integer-coded ordinal

    target = (gen.random(n) < 0.18).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": np.arange(100000, 100000 + n),
            "age": age,
            "income": income,
            "balance": balance,
            "city": city,
            "workclass": workclass,
            "occupation": occupation,
            "grade": grade,
            "country": np.where(gen.random(n) < 0.5, "USA", " usa "),
            "signup_date": pd.to_datetime("2020-01-01")
            + pd.to_timedelta(gen.integers(0, 1500, n), unit="D"),
            "notes": [
                f"Customer {i} left a free text remark about the service they received."
                for i in range(n)
            ],
            "constant_col": 1.0,
            "near_constant": np.where(gen.random(n) < 0.995, "A", "B"),
            "mostly_missing": np.where(gen.random(n) < 0.05, 1.0, np.nan),
            "is_active": gen.choice([True, False], n),
            "target": target,
            "leaky": target * 1.0 + gen.normal(0, 0.001, n),
        }
    )
    frame["income_copy"] = frame["income"]  # exact duplicate column
    return frame


@pytest.fixture
def simple_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "cat": ["a", "b", "a", "c", "b", None],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
