"""End-to-end worked example: raw dataset to ML-ready, with nothing hidden.

    raw dataset -> EDA -> profiling -> automatic plan -> cleaning -> preprocessing
    -> final ML-ready dataset

Run it::

    python examples/end_to_end.py
    python examples/end_to_end.py --html        # also write eda.html and report.html

The synthetic dataset below is built to contain every pathology found in the reference
repositories (see ``docs/design-rationale.md``), so the output shows the library
reacting to real problems rather than to a clean toy frame.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import edaprep


def make_messy_dataset(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """A customer-churn frame containing every defect this library exists to catch."""
    gen = np.random.default_rng(seed)

    tenure = gen.integers(0, 96, n)
    support_calls = gen.poisson(1.4, n)
    monthly_spend = gen.lognormal(3.6, 0.55, n)

    # A real signal, so the target is learnable rather than noise.
    risk = (
        0.9 * (support_calls > 3)
        + 0.7 * (tenure < 12)
        - 0.5 * (monthly_spend > 60)
        + gen.normal(0, 0.8, n)
    )
    churn = (risk > 1.0).astype(int)

    frame = pd.DataFrame(
        {
            # an identifier: unique, uninformative, and dropped in every notebook by hand
            "customer_id": [f"CUST-{i:07d}" for i in range(n)],
            "tenure_months": tenure.astype(float),
            "support_calls": support_calls.astype(float),
            # heavily right-skewed, the usual most common numeric shape
            "monthly_spend": monthly_spend,
            # zero-heavy: 40% of customers carry no balance
            "account_balance": np.where(gen.random(n) < 0.40, 0.0, gen.normal(0, 300, n)),
            # a 5-level ordered scale stored as integers
            "satisfaction_rating": gen.integers(1, 6, n).astype(float),
            # high-cardinality categorical: one-hot would add 400 columns
            "city": gen.choice([f"city_{i:03d}" for i in range(400)], n),
            # low-cardinality categorical, with '?' standing in for missing
            "contract_type": gen.choice(
                ["monthly", "annual", "two_year", "?"], n, p=[0.5, 0.3, 0.15, 0.05]
            ),
            # inconsistent whitespace and casing: three spellings of one country
            "country": gen.choice(["USA", " usa ", "Usa"], n, p=[0.7, 0.2, 0.1]),
            # numbers stored as text
            "credit_score": [f"{v:.0f}" for v in gen.normal(680, 90, n)],
            "signup_date": pd.to_datetime("2019-01-01")
            + pd.to_timedelta(gen.integers(0, 1800, n), unit="D"),
            # free text: detected, reported, and not silently one-hot encoded
            "last_comment": [
                f"Customer {i} contacted support about their most recent invoice."
                for i in range(n)
            ],
            "is_paperless": gen.choice([True, False], n),
            "legacy_flag": 1.0,  # constant
            # 92% missing: imputing would invent the column.  Built via pd.Series
            # rather than np.where, which coerces np.nan to the *string* "nan" when the
            # other branch is text -- a real export artefact, but not what is wanted
            # here (contract_type already demonstrates sentinel detection).
            "referral_code": pd.Series(
                np.where(gen.random(n) < 0.08, "REF", None), dtype="object"
            ),
            "churn": churn,
        }
    )

    # Missing values, some of them correlated: two columns that go missing together,
    # the kind of pattern usually noticed by hand, one dataset at a time.
    shared_gap = gen.random(n) < 0.12
    frame.loc[shared_gap, "monthly_spend"] = np.nan
    frame.loc[shared_gap, "account_balance"] = np.nan
    frame.loc[gen.random(n) < 0.05, "tenure_months"] = np.nan

    # An exactly duplicated column, and some duplicated rows.
    frame["spend_copy"] = frame["monthly_spend"]
    return pd.concat([frame, frame.iloc[:60]], ignore_index=True)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", action="store_true", help="also write HTML reports")
    parser.add_argument("--rows", type=int, default=6000)
    args = parser.parse_args()

    df = make_messy_dataset(args.rows)

    # ------------------------------------------------------------------ 1. raw data
    rule("1. THE RAW DATASET")
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head(3).to_string())
    print("\ndtypes say almost nothing about what these columns mean:")
    print(df.dtypes.to_string())

    # ------------------------------------------------------------------ 2. profiling
    rule("2. PROFILING  -- measurement only, no decisions")
    profile = edaprep.profile(df, target="churn", config=edaprep.Config(random_state=42))
    print(profile.summary(max_columns=20))

    rule("2a. WHAT THE DTYPE-BASED SPLIT WOULD HAVE MISSED")
    naive_numeric = set(df.select_dtypes(include=["int64", "float64"]).columns)
    semantic_numeric = set(profile.numeric_columns)
    print("  select_dtypes(['int64','float64']) says numeric:")
    print(f"    {sorted(naive_numeric)}")
    print("  edaprep says numeric:")
    print(f"    {sorted(semantic_numeric)}")
    print("\n  the disagreements, and why they matter:")
    for name in sorted(naive_numeric ^ semantic_numeric):
        column = profile[name]
        side = "dtype-only" if name in naive_numeric else "edaprep-only"
        print(
            f"    {name:<22} {side:<13} -> {column.semantic} "
            f"({column.semantic_reasons[0] if column.semantic_reasons else ''})"
        )
    missed = [c for c in ("credit_score",) if c not in naive_numeric]
    if missed:
        print(
            f"\n    {missed[0]!r} is numeric data stored as text; the dtype split drops "
            f"it from the numeric branch entirely."
        )

    # ------------------------------------------------------------------ 3. EDA
    rule("3. EDA  -- the notebook loops, as tables")
    eda = edaprep.EDA(df, target="churn", config=edaprep.Config(random_state=42))
    report = eda.analyze("standard")

    print("Numerical summary (most-skewed first):")
    print(report.numerical[
        ["column", "missing_%", "mean", "median", "skew", "n_zeros", "distribution"]
    ].to_string(index=False))

    print("\nCategorical summary:")
    print(report.categorical[
        ["column", "n_unique", "modal_value", "modal_%", "n_rare_levels", "note"]
    ].to_string(index=False))

    print("\nOutliers -- the three fences disagree, which is the point:")
    print(report.outliers[
        ["column", "skew", "n_iqr", "n_zscore", "n_modified_z", "recommended"]
    ].to_string(index=False))

    print("\nFeature/target association:")
    print(report.target_relationships.head(8).to_string(index=False))

    # ------------------------------------------------------------------ 4. split
    rule("4. SPLIT FIRST  -- everything after this is safe by construction")
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(len(shuffled) * 0.8)
    train_df, test_df = shuffled.iloc[:cut].copy(), shuffled.iloc[cut:].copy()
    print(f"train: {train_df.shape[0]:,} rows    test: {test_df.shape[0]:,} rows")
    print("\nScaling the full frame and splitting afterwards is the classic mistake.")
    print("first is the whole discipline; edaprep makes the rest automatic.")

    # ------------------------------------------------------------------ 5. plan
    rule("5. THE PLAN  -- decided before a single row is touched")
    config = edaprep.Config(random_state=42)
    config.column("satisfaction_rating").semantic_type = "ordinal"
    config.column("account_balance").outlier_strategy = "clip"   # a user override

    pipe = edaprep.AutoPipeline(
        target="churn", model_family="linear", config=config, random_state=42
    )
    plan = pipe.plan(train_df)
    print(plan.summary())

    rule("5a. EXPLAIN  -- every decision, with the measurement behind it")
    for column in ("monthly_spend", "city", "account_balance", "customer_id", "referral_code"):
        print(plan.explain(column))
        print()

    overrides = plan.overrides
    if overrides:
        print(f"user overrides in this plan ({len(overrides)}):")
        for decision in overrides:
            print(f"  {decision.column:<22} {decision.action}  [{decision.rationale}]")

    # ------------------------------------------------------------------ 6. fit
    rule("6. FIT ON TRAIN, TRANSFORM BOTH")
    pipe.fit(train_df)
    X_train = pipe.transform(train_df)
    X_test = pipe.transform(test_df)
    y_train, y_test = train_df["churn"], test_df["churn"]

    print(f"train: {train_df.shape} -> {X_train.shape}")
    print(f"test:  {test_df.shape} -> {X_test.shape}")
    print(f"identical columns:      {list(X_train.columns) == list(X_test.columns)}")
    all_numeric = all(pd.api.types.is_numeric_dtype(d) for d in X_train.dtypes)
    print(f"all numeric:            {all_numeric}")
    print(f"missing values left:    {int(X_train.isna().sum().sum())} train, "
          f"{int(X_test.isna().sum().sum())} test")
    print(f"target absent from X:   {'churn' not in X_train.columns}")

    print("\nfinal columns:")
    for i in range(0, len(X_train.columns), 4):
        print("  " + "  ".join(f"{c:<26.26}" for c in X_train.columns[i : i + 4]))

    # ------------------------------------------------------------------ 7. no leakage
    rule("7. THE LEAKAGE CHECK")
    row_by_row = pd.concat(
        [pipe.transform(test_df.iloc[[i]]) for i in range(min(40, len(test_df)))]
    )
    matches = np.allclose(
        X_test.iloc[: len(row_by_row)].to_numpy(dtype=float),
        row_by_row.to_numpy(dtype=float),
        equal_nan=True,
    )
    print("Transforming the test frame whole, then one row at a time, and comparing:")
    print(f"  identical: {matches}")
    print("  If any step recomputed a statistic on the frame it was given, these would")
    print("  differ. This is asserted for the whole library in tests/test_leakage.py.")

    print("\nLeakage audit from the report:")
    for key, value in pipe.report_.leakage.items():
        print(f"  {key:<34} {value}")

    # ------------------------------------------------------------------ 8. report
    rule("8. THE REPORT  -- what actually happened, with counts")
    print(pipe.report_.summary())

    # ------------------------------------------------ 8a. acting on a finding
    rule("8a. ACTING ON A FINDING  -- report, then decide")
    print("The profile flagged 'country' for case variants: USA / Usa / usa are one")
    print("country spelled three ways. edaprep strips whitespace (unambiguous) but does")
    print("NOT fold case, because that is a judgement about the data, not a defect with")
    print("one right answer -- 'IT' and 'it' may be a country and a department.")
    print()
    print(f"  as planned: {[c for c in X_train.columns if c.startswith('country_')]}")

    fixed_train = train_df.assign(country=train_df["country"].str.strip().str.lower())
    fixed_pipe = edaprep.AutoPipeline(
        target="churn", model_family="linear", config=config, random_state=42
    ).fit(fixed_train)
    fixed_X = fixed_pipe.transform(fixed_train)
    remaining = [c for c in fixed_X.columns if c.startswith("country")]
    print(f"  after folding case: {remaining or 'none -- the column became constant'}")
    print(f"  columns: {X_train.shape[1]} -> {fixed_X.shape[1]}")
    print()
    print("Folding case revealed that every customer is in one country, so the column")
    print("carries no information at all and the planner dropped it. Three spurious")
    print("one-hot columns became zero. The library found the problem and named it;")
    print("you decided what it meant.")

    # ------------------------------------------------------------------ 9. model
    rule("9. THE POINT OF ALL THIS: a model-ready frame")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"LogisticRegression on the prepared data -- test AUC: {auc:.4f}")
        print("\nNo preprocessing code was written for this model. The imbalance was")
        print("reported by edaprep and handled here, at fit time, where it belongs.")
    except ImportError:
        print("scikit-learn is not installed; skipping the modelling step.")
        print("Install it with: pip install 'edaprep[advanced]'")

    # ------------------------------------------------------------------ 10. artefacts
    if args.html:
        rule("10. ARTEFACTS")
        report.to_html("eda.html")
        pipe.report_.to_html("report.html")
        with open("plan.json", "w", encoding="utf-8") as handle:
            handle.write(pipe.plan_.to_json())
        print("wrote eda.html, report.html, plan.json")
        print("plan.json records exactly how these features were built, so the frame")
        print("can be traced back to the process that produced it.")


if __name__ == "__main__":
    main()
