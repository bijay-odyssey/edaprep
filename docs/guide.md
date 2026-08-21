# User guide

From installation to production. Every example runs as written.

---

## 1. Installation

```bash
pip install edaprep                    # core: numpy, pandas, scipy
pip install "edaprep[visualization]"   # + matplotlib, for the plotting helpers
pip install "edaprep[advanced]"        # + scikit-learn, for interoperability
pip install "edaprep[all]"
```

Python 3.9 or newer. The core has three dependencies and no build step.

---

## 2. Understanding a dataset

### 2.1 Profiling

`profile()` measures. It makes no decisions and changes nothing.

```python
import edaprep

profile = edaprep.profile(df, target="churn")
print(profile.summary())
```

The result is a frozen, serialisable value:

```python
profile.n_rows, profile.n_columns
profile.numeric_columns          # by *semantic* type, not dtype
profile.categorical_columns
profile.identifier_columns
profile.uncertain_columns        # where the type inference was not confident

col = profile["income"]
col.semantic                     # SemanticType.NUMERIC
col.semantic_confidence          # 0.9
col.semantic_reasons             # ('continuous numeric values',)
col.missing_fraction             # 0.02
col.skew                         # 3.22
col.numeric.q1, col.numeric.q3, col.numeric.mad
col.target_association           # 0.14

profile.to_dict()                # JSON-safe: NaN and inf become null
```

### 2.2 Why semantic types matter

A column's dtype says how it is stored; its semantic type says what it means. Confusing
the two is the single most common defect in notebook preprocessing.

```python
df = pd.DataFrame({
    "zip_code":    [90210, 10001, 60601] * 100,   # int64, but categorical
    "temperature": [21.4, 19.8, 23.1]   * 100,    # int64-adjacent, genuinely numeric
    "rating":      [1, 2, 3, 4, 5]      * 60,     # int64, ordered scale
})
p = edaprep.profile(df)
p["zip_code"].semantic      # CATEGORICAL  (integer-coded, few distinct values)
p["temperature"].semantic   # NUMERIC
p["rating"].semantic        # ORDINAL      (name suggests an ordered scale)
```

When the inference is unsure it says so, rather than guessing quietly:

```python
for name in profile.uncertain_columns:
    col = profile[name]
    print(name, col.semantic, col.semantic_confidence, col.semantic_alternatives)
```

Correct it with a hint, which is honoured with full confidence:

```python
config = edaprep.Config()
config.column("zip_code").semantic_type = "categorical"
```

### 2.3 Data-quality findings

```python
for issue in profile.issues:
    print(issue)
```

Detected without being asked: constant and near-constant columns, identifiers, high
missingness, duplicate rows, duplicate columns, sentinel strings (`"?"`, `"N/A"`, …),
suspicious numeric placeholders (`-999`), correlated missingness, mixed Python types in
one column, stray whitespace, categories differing only by case, missing targets, class
imbalance, and columns almost perfectly associated with the target.

Each is a structured record, not just a message:

```python
issue.code       # "possible_target_leakage"
issue.severity   # Severity.ERROR
issue.columns    # ("leaky",)
issue.details    # {"leaky": 0.9998}
```

---

## 3. Exploratory analysis

```python
report = edaprep.EDA(df, target="churn").analyze("standard")
print(report.summary())
```

Every section is a plain `DataFrame` or `dict`:

```python
report.columns               # dtype, semantic, missing, cardinality per column
report.missing               # only columns with gaps, plus what they co-miss with
report.numerical             # moments and quantiles, most-skewed first
report.categorical           # cardinality, concentration, rare levels, encoding cost
report.outliers              # counts under each fence, side by side
report.correlation           # matrix (sampled on large frames)
report.correlated_pairs      # the pairs worth looking at
report.vif                   # deep level only
report.target                # distribution and imbalance
report.target_relationships  # association and significance per feature
```

### 3.1 Analysis levels

| level | includes | on 100k × 14 |
|---|---|---|
| `quick` | shape, dtypes, missingness, cardinality, duplicates | 521 ms |
| `standard` | + moments, categories, outliers, correlation, target relationships | 1205 ms |
| `deep` | + VIF, significance tests, full correlation regardless of width | 1411 ms |

The levels are different amounts of *work*, not different amounts of display.

```python
edaprep.EDA(df, target="y").analyze("standard", exclude=["correlation"])
edaprep.EDA(df, target="y").analyze("standard", include=["numerical", "outliers"])
```

### 3.2 Reading the outlier table

Three fences are shown side by side deliberately — they disagree, often substantially,
and seeing the disagreement stops "outlier" being read as a property of a value rather
than an artefact of the fence chosen.

```
column    skew   n_iqr  n_zscore  n_modified_z  recommended
income    3.22     412        88          1104  iqr (k=3)
age       0.14      23        21            25  zscore
```

`recommended` names the method the planner would pick for that column, so the table
doubles as a preview of the plan.

### 3.3 Visualisation

Optional; requires the `visualization` extra.

```python
viz = edaprep.visualization
ax = viz.histogram(df, "income")
ax = viz.missing_bar(df)
ax = viz.correlation_heatmap(report.correlation)
fig = viz.plot_profile(df, target="churn")     # one overview figure
```

Every helper takes an `ax` and returns it, so plots compose into your figure. Nothing
calls `plt.show()`.

---

## 4. Preparing data

### 4.1 Automatic

```python
pipe = edaprep.AutoPipeline(
    target="churn",
    model_family="tree",     # "linear" | "tree" | "distance" | "neural" | None
    random_state=42,
)
pipe.fit(train_df)
X_train = pipe.transform(train_df)
X_test  = pipe.transform(test_df)
```

Inspect the plan before it runs a single row:

```python
plan = edaprep.AutoPipeline(target="churn").plan(train_df)
print(plan.summary())
```

### 4.2 Why `model_family` matters

Sophisticated hand-written pipelines end up maintaining parallel preprocessing
branches for tree and linear models. That is now one argument.

| | `"tree"` | `"linear"` / `"distance"` | `"neural"` | `None` |
|---|---|---|---|---|
| scaling | none | standard (robust if skewed) | min-max | standard |
| encoding | ordinal | one-hot, target above 50 levels | one-hot / target | one-hot |
| skew transform | none | log1p / Yeo-Johnson | yes | heavy skew only |
| outliers | report | clip | clip | report |

Trees are invariant to monotone transforms and to rescaling, so both are skipped —
and the plan *says* they were skipped and why, rather than silently omitting them:

```
age:
  - no_scaling - tree models are invariant to monotone rescaling, so scaling would
                 cost a pass over the data and change no split
```

`None` is the default and selects a conservative branch that makes no modelling
assumptions.

### 4.3 Explicit

```python
pipe = (
    edaprep.Pipeline(target="churn")
    .infer_types()
    .drop_columns(["customer_id"])
    .flag_missing()
    .expand_datetime()
    .handle_outliers(method="iqr", strategy="clip")
    .handle_missing(strategy="median")
    .transform_distributions(method="yeojohnson")
    .group_rare_categories(threshold=0.01)
    .encode_categorical(strategy="onehot")
    .scale_numeric(strategy="standard")
)
X = pipe.fit_transform(train_df)
```

Or as a list, which is the same thing:

```python
from edaprep import Pipeline, MissingValueHandler, Scaler

pipe = Pipeline([
    ("impute", MissingValueHandler(strategy="median")),
    ("scale",  Scaler(strategy="standard")),
], target="churn")
```

Every transformer also works standalone:

```python
from edaprep import OutlierHandler
handler = OutlierHandler(["income"], method="modified_zscore", strategy="clip")
handler.fit(train_df)
handler.summary()        # method, fence, count and fraction per column
```

---

## 5. The train/test workflow

This is the part the library exists for.

### 5.1 The rule

**Fit on training data only. Transform everything with the fitted statistics.**

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["churn"]
)

pipe = edaprep.AutoPipeline(target="churn", model_family="tree", random_state=42)
pipe.fit(train_df)                       # <- sees train only

X_train = pipe.transform(train_df)
X_test  = pipe.transform(test_df)        # <- train's medians, fences, categories
y_train, y_test = train_df["churn"], test_df["churn"]
```

Split **first**. Everything after that is safe by construction.

### 5.2 What "safe by construction" means

Every learned statistic — medians, category mappings, outlier fences, power-transform
lambdas, quantile knots, scaler centres — is stored on the transformer during `fit`.
`transform` reads them and computes nothing new. The library asserts this directly:

```python
whole      = pipe.transform(test_df)
row_by_row = pd.concat([pipe.transform(test_df.iloc[[i]]) for i in range(len(test_df))])
assert whole.equals(row_by_row)          # a test in the suite
```

If any step recomputed a statistic on its input, those two would differ.

### 5.3 Target encoding and cross-fitting

Target encoding is the one place `fit_transform` legitimately differs from
`fit().transform()`, and the difference is the point.

```python
X_train = pipe.fit_transform(train_df)   # training rows encoded OUT OF FOLD
X_test  = pipe.transform(test_df)        # full-train mapping
```

A training row's encoded value is computed from the folds that do **not** contain it.
Without that, a category seen twice encodes almost exactly that row's own label, the
model memorises it, and validation looks excellent right up until production.

```python
pipe.report_.leakage
# {'target': 'churn',
#  'transformers_using_target': ['TargetEncoder'],
#  'cross_fitted': True,
#  'columns_suspected_of_leakage': [],
#  'statistics_learned_at_fit_only': True}
```

### 5.4 Validation sets and cross-validation

A validation set is just another frame to `transform`:

```python
X_valid = pipe.transform(valid_df)
```

For k-fold cross-validation, refit inside each fold — reusing one fitted pipeline
across folds leaks the whole training set into every validation fold:

```python
from sklearn.model_selection import StratifiedKFold

for train_idx, valid_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    fold_pipe = edaprep.AutoPipeline(target="churn", random_state=42)
    fold_pipe.fit(df.iloc[train_idx])
    X_tr = fold_pipe.transform(df.iloc[train_idx])
    X_va = fold_pipe.transform(df.iloc[valid_idx])
```

Or put the transformers in an sklearn pipeline and let `cross_val_score` handle it
(section 8).

### 5.5 What `edaprep` will not do for you

- **It does not split.** Splitting is a modelling decision — stratified, grouped,
  time-ordered — and getting it wrong is not something a preprocessing library should
  guess at.
- **It does not resample.** Class imbalance is measured and reported. Resampling must
  happen after the split, on the training fold only, and belongs with the model.
- **It does not transform the target.** A skewed target is reported with a
  recommendation to model `log1p(y)` and invert the prediction. Transforming a target
  column in place is easy to forget to undo.
- **It does not drop suspected leaks.** A column almost perfectly associated with the
  target is flagged as an error, not removed: a legitimately strong feature looks
  identical from here, and only you know which it is.

---

## 6. Configuration

### 6.1 Global settings

```python
config = edaprep.Config(
    missing_strategy="auto",
    outlier_strategy="report",
    categorical_encoding="auto",
    scaling="auto",
    duplicate_strategy="report",
    model_family="tree",
    random_state=42,
    rare_category_threshold=0.01,
    high_cardinality_threshold=50,
    correlation_filter=False,
)
```

`"auto"` hands the decision to the planner. Anything else pins every applicable column.

### 6.2 Per-column overrides

```python
config.column("age").imputation = "mean"
config.column("income").outlier_strategy = "clip"
config.column("income").transform = "log1p"
config.column("city").encoding = "frequency"
config.column("zip").semantic_type = "categorical"
config.column("internal_note").drop = True
config.column("signup").datetime_features = ["year", "month", "dayofweek"]
```

Or in bulk:

```python
config.set_columns({
    "age":    {"imputation": "median"},
    "income": {"transform": "yeojohnson", "scaling": "robust"},
})
```

An override is tagged `source="user_override"` in the plan, so `explain()` marks it as
yours rather than presenting it as the planner's reasoning.

### 6.3 Thresholds

Every number that influences a decision is named, in one place, with its provenance.

```python
config.thresholds.skew_moderate = 1.0            # |skew| below this: no transform
config.thresholds.skew_heavy = 5.0               # at or above: Yeo-Johnson
config.thresholds.iqr_k = 1.5                    # Tukey's fence
config.thresholds.iqr_k_skewed = 3.0             # widened for skewed columns
config.thresholds.missing_drop_threshold = 0.60  # above this: drop, do not impute
config.thresholds.high_cardinality_threshold = 50
config.thresholds.outlier_max_action_fraction = 0.10
config.thresholds.sampling_row_threshold = 200_000
```

Contradictions are rejected at construction, not discovered later:

```python
edaprep.Config(thresholds=edaprep.Thresholds(skew_moderate=5.0, skew_heavy=1.0))
# ConfigurationError: skew_heavy (1.0) must be greater than skew_moderate (5.0);
# they define adjacent tiers of a single scale.
```

### 6.4 Saving and restoring

```python
import json
json.dump(config.to_dict(), open("config.json", "w"))
config = edaprep.Config.from_dict(json.load(open("config.json")))
```

---

## 7. Reports

```python
print(pipe.report_.summary())          # human-readable
pipe.report_.to_json("report.json")    # machine-readable
pipe.report_.to_html("report.html")    # self-contained page, no external assets
```

Three things are kept distinct because they answer different questions:

| | question | available |
|---|---|---|
| `report.profile` | what did the data look like? | after `fit` |
| `report.plan` | what was decided, and why? | before any transform |
| `report.entries` | what actually happened, with counts? | after `transform` |

They can legitimately disagree — a plan may decide to clip `income` and then find
nothing above the fence in a particular batch. Collapsing them would hide that.

```python
pipe.report_.for_column("income")       # every journal entry touching one column
pipe.report_.for_stage("encode")
pipe.report_.warnings                   # structured advisories
pipe.report_.leakage                    # the audit
```

The report also records the library version, the configuration, the random seed and
whether profiling sampled — so a persisted output can be traced back to the process that
made it.

---

## 8. scikit-learn interoperability

`edaprep` transformers implement the estimator protocol, so they drop straight into an
sklearn pipeline:

```python
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = SkPipeline([
    ("impute", edaprep.MissingValueHandler(strategy="median")),
    ("encode", edaprep.CategoricalEncoder(strategy="onehot")),
    ("scale",  edaprep.Scaler(strategy="standard")),
    ("clf",    LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(model, X, y, cv=5)
model.set_params(scale__strategy="robust")
```

`edaprep` does not import scikit-learn to do this — it reimplements the ~40-line
`get_params`/`set_params` protocol, so scikit-learn stays an optional dependency.

---

## 9. Large datasets

Profiling switches to a deterministic sample once a frame crosses
`Thresholds.sampling_row_threshold` (200,000 rows by default). Cheap statistics — null
counts, dtypes, min/max — always run on the full frame, because they are single
vectorised passes and because missing fractions drive drop/impute decisions and must be
exact.

Whether sampling happened is recorded, so no statistic is ever silently approximate:

```python
profile.sampling
# {'used': True, 'n': 50000, 'of': 590540, 'random_state': 42,
#  'statistics': ['moments', 'quantiles', 'distinct counts', ...]}
```

Tune it:

```python
config = edaprep.Config(random_state=42, sample_size=100_000)
config.thresholds.sampling_row_threshold = 1_000_000
```

Other cost controls: `EDA.analyze("quick")`, `profile(..., compute_moments=False)`,
`profile(..., check_quality=False)`, `profile(..., deep_memory=False)`, and
`Thresholds.correlation_max_columns`, which skips the correlation matrix on very wide
frames unless `deep` asks for it.

---

## 10. Errors

`edaprep` raises typed exceptions with actionable messages, and never a bare
`ValueError`.

```
TransformationError: Column 'balance' contains 200 non-positive value(s), so the log
transformation cannot be applied. Use 'yeojohnson', which is defined on the whole real
line, or 'log1p' if the column is non-negative.
```

```
SchemaError: 2 column(s) required by Scaler are missing from the input: 'age',
'income'. The frame passed to 'transform' must contain every column present at 'fit'
time.
```

| exception | means |
|---|---|
| `ConfigurationError` | unknown, contradictory or out-of-range configuration |
| `NotFittedError` | `transform` before `fit` |
| `SchemaError` | transform-time columns disagree with fit-time columns |
| `EmptyDataError` | no rows or no columns where data is required |
| `TransformationError` | the data cannot support the requested transformation |
| `LeakageError` | a fit-time-only operation was reached at transform time |

Extra columns at transform time are an error by default, because tolerating them
silently is how train/serve skew becomes invisible. Downgrade deliberately:

```python
config = edaprep.Config(on_unknown_columns="ignore")
```

---

## 11. Where to go next

- [`examples/end_to_end.py`](../examples/end_to_end.py) — a complete worked run
- [`docs/architecture.md`](architecture.md) — how the planner works
- [`docs/extending.md`](extending.md) — custom transformers, rules and backends
- [`docs/design-rationale.md`](design-rationale.md) — the analysis this was built from
