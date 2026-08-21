# edaprep

Transparent, leakage-safe EDA and ML preprocessing, with an explainable planner.

`edaprep` looks at a dataset, works out which preprocessing operations actually apply
to it, tells you what it intends to do and why, and then does it — fitting every
statistic on the training data alone.

```python
import edaprep

pipe = edaprep.AutoPipeline(target="churn", model_family="tree", random_state=42)
pipe.fit(train_df)
pipe.explain()

X_train = pipe.transform(train_df)
X_test  = pipe.transform(test_df)
```

```
income:
  + outliers_report - skew 3.22 is moderate (>= 1.0); IQR fence widened to k=3.0 for the asymmetry
  + impute_median - 2.0% missing; median rather than mean because it is unaffected by
                    the skew (3.22) and by outliers
  + transform_log1p - skew 3.22 is moderate and the column is non-negative (min 2576.27),
                      so log1p applies and is invertible
  + scale_robust - skew 3.22; robust scaling (median and IQR) rather than standard,
                   whose standard deviation is dominated by the tail

city:
  + group_rare_categories - 163 levels; those appearing in fewer than 5 rows (1.0%) are
                            grouped, since they cannot support a reliable estimate
  + encode_target - 163 levels exceeds the 50-level one-hot ceiling; target encoding is
                    used with 5-fold cross-fitting so no row is encoded using its own target

customer_id:
  x dropped - identifier: 100.0% of values are distinct, so it cannot generalise beyond
              the rows it was fitted on
```

---

## Why it exists

It was built by mining common notebook workflows for the EDA and
preprocessing workflow they have in common — a broad survey of notebook workflows.
The findings are written up in [`docs/design-rationale.md`](docs/design-rationale.md), and
they shaped every design decision:

- The dtype-based column split (`select_dtypes(include=['int64','float64'])`) appears
  **39 times** and is the largest single source of error: it sends a zip code and a
  temperature down the same path. `edaprep` infers a *semantic* type and reports its
  confidence.
- The IQR outlier fence is rewritten **12 times**, the z-score fence 6 times, with the
  multiplier drifting between 1.5 and 3.0 for no recorded reason. Both are now single
  parameterised, named, reported operations.
- **Leakage is easy to introduce.** One fits a `StandardScaler` on the full
  frame, writes the result to CSV, and splits afterwards. `edaprep` makes that
  structurally impossible rather than merely discouraged.
- Two notebooks independently maintain *parallel preprocessing branches* for tree and
  linear models. That insight became `model_family`, a first-class planning input.

## Installation

```bash
pip install edaprep                    # core: numpy, pandas, scipy
pip install "edaprep[visualization]"   # + matplotlib
pip install "edaprep[advanced]"        # + scikit-learn
pip install "edaprep[all]"
```

Python 3.9+.

---

## What it does

### Understand a dataset

```python
profile = edaprep.profile(df, target="churn")
print(profile.summary())
```

```
Dataset
  600 rows x 18 columns
  300.3 KB in memory
  628 missing cells (5.81%)
  target: churn (classification, 2 classes, minority/majority ratio 0.232)

Semantic types
  numeric           5
  binary            4
  categorical       3
  ...

Data-quality findings
  [x] 1 column(s) are almost perfectly associated with the target (>= 0.98). This
      usually means the column encodes the answer: 'leaky'.
  [!] 2 column(s) contain placeholder strings that most likely mean 'missing' but are
      not recognised as NaN: 'workclass', 'occupation'.
  [!] 1 group(s) of identical columns: income=income_copy
  [i] 1 column pair(s) go missing together, which usually means a shared cause:
      income~income_copy (1.00)
```

### Explore it

```python
report = edaprep.EDA(df, target="churn").analyze("standard")
print(report.summary())
report.numerical        # a DataFrame
report.to_html("eda.html")
```

Three levels that differ in work done, not just in what is shown: `quick` skips every
O(n log n) and O(p²) computation, `standard` adds moments, outliers, correlation and
target relationships, `deep` adds VIF and significance tests with a
Benjamini-Hochberg adjustment.

### Prepare it

```python
pipe = edaprep.AutoPipeline(target="churn", model_family="linear", random_state=42)
X_train = pipe.fit_transform(train_df)
X_test  = pipe.transform(test_df)

pipe.plan_             # the decisions, serialisable and editable
pipe.report_           # what actually happened, with counts
pipe.transformations_  # one row per decision, as a DataFrame
pipe.statistics_       # every learned parameter
```

Or say exactly what should happen:

```python
pipe = (
    edaprep.Pipeline(target="churn")
    .flag_missing()
    .handle_outliers(strategy="clip")
    .handle_missing()
    .encode_categorical()
    .scale_numeric()
)
```

### Override anything

```python
config = edaprep.Config(random_state=42)
config.column("age").imputation = "mean"
config.column("income").outlier_strategy = "clip"
config.column("city").encoding = "frequency"
config.column("zip").semantic_type = "categorical"
config.thresholds.skew_heavy = 4.0

pipe = edaprep.AutoPipeline(target="churn", config=config)
```

Overrides are tagged in the plan, so `explain()` marks them as yours rather than
presenting them as the planner's reasoning.

---

## Design guarantees

**No leakage, structurally.** Learned state lives only in attributes written inside
`fit`; `transform` is a pure function of that state. The property is asserted directly:
a test transforms a frame whole and then row by row and requires identical output, which
fails immediately if anything recomputes a statistic at transform time.

**Nothing silent.** Dropped columns, imputed values, grouped categories, clipped rows
and unseen categories are all counted and reported. `edaprep` never calls
`warnings.filterwarnings`.

**Everything explainable.** Every automatic decision carries an English rationale naming
the measurement behind it. The plan is inert, serialisable data — printable, diffable,
storable next to a model artefact, and re-executable.

**Reproducible.** `random_state` seeds every stochastic step. The report records the
library version, the configuration, the seed, whether profiling sampled, and every
learned parameter.

**Conservative.** Outliers are reported, not deleted, by default. Duplicate rows are
reported, not removed — repeated observations are legitimate in transactional data.
Class imbalance is measured and reported; resampling is a modelling decision that
belongs after the split, so `edaprep` does not do it.

---

## Performance

Measured, not asserted. See [`docs/performance.md`](docs/performance.md).

| operation | edaprep | baseline |
|---|---|---|
| `Scaler` (standard) | 6.9 ms | sklearn `StandardScaler` 26.7 ms |
| `MissingValueHandler` (median) | 5.2 ms | sklearn `SimpleImputer` 17.0 ms |
| `OutlierHandler` (IQR clip) | 25.2 ms | the usual IQR block 41.4 ms |
| `numeric_block_stats` (20k × 300) | 578 ms | equivalent pandas loop 1056 ms |
| `AutoPipeline.transform` | 240 ms / 35.6 MiB | `ColumnTransformer` 104 ms / 67.2 MiB |

100,000 rows unless stated. The most instructive result is one that went the other way:
a hand-written NumPy kernel in this library turned out to be **2.1× slower** than the
pandas code it replaced, so it was deleted. That story is in `docs/performance.md` §1.

No native code. Nothing here is un-vectorisable, and the one place a hand-written kernel
looked promising was slower than pandas.

---

## Documentation

| | |
|---|---|
| [Workflow mining](docs/design-rationale.md) | what 13 repositories revealed, and the 9 defects found |
| [Architecture](docs/architecture.md) | package design, the planner, execution model |
| [User guide](docs/guide.md) | installation to production, with the train/test workflow |
| [Performance](docs/performance.md) | benchmarks, method, and what optimisation actually changed |
| [Extending](docs/extending.md) | custom transformers, rules and backends |
| [Example](examples/end_to_end.py) | raw dataset to ML-ready, end to end |

---

## Scope

**In:** dataset inspection, EDA, data quality, cleaning, missing values, duplicates,
outliers, dtype inference, categorical encoding, numeric transformation, scaling,
feature selection, datetime expansion, leakage-safe train/test preparation, pipelines,
reporting.

**Out, deliberately:** model training, resampling, hyperparameter search, NLP,
forecasting, deep learning, distributed execution. Extension points exist for each
(`docs/extending.md`), and none is implemented in v1.

## Development

```bash
pip install -e ".[dev]"
pytest                       # 353 tests
python benchmarks/bench.py
```

## Licence

MIT.
