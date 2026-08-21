# Architecture

**Phase 2 deliverable.** Design decisions for `edaprep`, and the reasoning behind them.
Read `design-rationale.md` first: every decision here traces back to something measured
there.

---

## 1. The one idea

Most EDA/preprocessing libraries are *function libraries*: you call `impute()`, then
`encode()`, then `scale()`. But those functions are the easy part — most practitioners
have already written each of them several times. What is actually hard is the layer
above:

> Given this dataset, and given what I intend to do with it, **which** operations apply,
> **in what order**, **with what parameters**, and **why**?

So `edaprep` separates *deciding* from *doing*:

```
        DataFrame
            |
            v
  +---------------------+
  |   DatasetProfile    |   pure measurement, no decisions
  +---------------------+
            |
            v
  +---------------------+
  |      Planner        |   pure decision, no data touched
  |   (rules + config)  |
  +---------------------+
            |
            v
  +---------------------+
  |       Plan          |   an inert, serialisable value
  +---------------------+
            |
            v
  +---------------------+
  |      Pipeline       |   materialises transformers, fit/transform
  +---------------------+
            |
       +----+----+
       v         v
   DataFrame   Report
```

The three interfaces between these boxes are **plain data**, not objects with behaviour:

- `DatasetProfile` is a frozen dataclass of measurements.
- `Plan` is a frozen dataclass holding a list of `PlannedStep`s.
- `Report` is a frozen dataclass of records.

All three round-trip through `to_dict()`/`from_dict()`. That single property buys most
of the design goal's requirements at once: `explain()` is a renderer over `Plan`;
"users can override decisions" means editing a `Plan`; "no magic" means the `Plan` is
printable; reproducibility means the `Plan` is serialisable and re-executable.

### 1.1 Why the planner does not touch data

`Planner.plan(profile, config)` takes a `DatasetProfile` and a `Config` and returns a
`Plan`. It never sees a DataFrame. Consequences:

- The planner is trivially unit-testable: construct a synthetic profile, assert on the
  plan. No fixtures, no I/O, microsecond tests.
- The planner cannot leak, because it has nothing to leak *from*. Since the profile
  passed to it is computed on the training frame only, no test-set quantity can reach a
  decision.
- Planning a 50 GB dataset costs the same as planning a 50-row one.

---

## 2. Package layout

```
src/edaprep/
├── __init__.py             public API surface, lazy submodule import
├── _version.py
├── types.py                SemanticType / ColumnRole / enums, no dependencies
├── exceptions.py           typed, message-rich exception hierarchy
├── config.py               Config, ColumnConfig, threshold constants
├── core/
│   ├── base.py             Transformer ABC: fit/transform/get_params/set_params
│   ├── context.py          FitContext: target, profile, RNG, journal
│   ├── journal.py          append-only record of everything that happened
│   └── pipeline.py         Pipeline (explicit) and AutoPipeline (automatic)
├── profiling/
│   ├── statistics.py       statistic kernels (delegating; see performance.md)
│   ├── column_types.py     semantic type inference
│   ├── quality.py          sentinels, co-missingness, quality issues
│   └── profiler.py         DatasetProfile, ColumnProfile, profile()
├── planning/
│   ├── decisions.py        Decision / PlannedStep / Plan value types
│   ├── rules.py            the rule set (see design-rationale.md)
│   └── planner.py          Planner: profile + config -> Plan
├── preprocessing/
│   ├── missing.py          MissingValueHandler, MissingIndicator
│   ├── duplicates.py       DuplicateRowHandler
│   ├── outliers.py         detectors + OutlierHandler
│   ├── encoding.py         OneHot / Ordinal / Frequency / Target / RareGrouper
│   ├── scaling.py          Standard / MinMax / Robust / MaxAbs
│   ├── transformations.py  Log1p / Sqrt / BoxCox / YeoJohnson / Quantile
│   ├── datetime_features.py
│   ├── text.py             lightweight text detection + extension point
│   ├── selection.py        constant / duplicate-col / missingness / correlation
│   └── casting.py          DataTypeInference (dtype correction + downcasting)
├── eda/
│   ├── analyzer.py         EDA facade, analysis levels
│   ├── numerical.py
│   ├── categorical.py
│   ├── correlation.py      pearson/spearman + VIF
│   ├── outliers.py         EDA-side outlier summary
│   └── target.py           target relationship, imbalance, leakage suspicion
├── reporting/
│   ├── report.py           Report value type, summary + JSON renderers
│   └── html.py             self-contained HTML renderer (no external assets)
├── visualization/
│   └── plots.py            optional matplotlib renderers
└── backends/
    ├── base.py             Backend protocol
    └── pandas_backend.py   the only implementation today
```

### 2.1 Why `src/` layout

Tests import the installed package, not the source tree. This catches missing
`__init__.py`, missing package data, and accidental reliance on the CWD. It is the
standard for serious packages.

### 2.2 Why `types.py` has no dependencies

`SemanticType` and friends are imported by everything. Keeping that module
dependency-free (stdlib `enum` only) prevents import cycles without needing
`TYPE_CHECKING` guards throughout.

---

## 3. The transformer contract

```python
class Transformer(ABC):
    stage: Stage                   # where it belongs in the ordering
    uses_target: bool = False      # declares y-consumption; audited
    cross_fitted: bool = False     # declares fit_transform != fit().transform()

    def fit(self, X, y=None, context=None) -> Self
    def transform(self, X) -> DataFrame
    def fit_transform(self, X, y=None, context=None) -> DataFrame
    def get_params(deep=True) -> dict
    def set_params(**params) -> Self
    def get_feature_names_out(input_features=None) -> ndarray
```

Four rules make leakage structurally impossible rather than merely discouraged:

1. **All learned state lives in attributes with a trailing underscore**, set only inside
   `_fit`. `transform` reads them and nothing else.
2. **`transform` never computes a statistic over its input.** Enforced two ways. A
   structural test (`test_leakage.py::test_transform_methods_compute_no_statistics`)
   parses every `_transform` body with `ast` and fails on a call to `mean`, `median`,
   `quantile`, `std`, `var`, `mode`, `value_counts`, `nunique`, `corr` or `factorize`
   outside a recorded allow-list. A behavioural test transforms a frame whole and then
   row by row and requires identical output, which fails on *any* transform-time
   statistic, including ones the parser cannot name.
3. **`fit_transform` is not free to differ from `fit().transform()`** except where
   cross-fitting demands it (target encoding). Those transformers set
   `cross_fitted = True` and are tested for the intended difference.
4. **`uses_target=True` is a declaration**, and `Pipeline.fit` raises if such a
   transformer is reached with `y=None`.

`get_params`/`set_params` are implemented once on the base class by introspecting
`__init__`, exactly like scikit-learn, which is what makes `edaprep` transformers
droppable into an `sklearn.pipeline.Pipeline`. `edaprep` does **not** subclass
`BaseEstimator`, because scikit-learn is an optional dependency; it reimplements the
~40-line protocol so interoperability costs nothing at import time.

### 3.1 Column scoping

Every transformer takes `columns: Sequence[str] | None`. `None` means "all columns this
transformer considers applicable", resolved at `fit` time and frozen into
`self.columns_`. `transform` operates only on `self.columns_` and passes everything else
through untouched. This is what lets the planner emit one transformer per *decision*
rather than one per column, while keeping per-column behaviour explicit in the plan.

---

## 4. Semantic type inference

docs/design-rationale.md identifies this as the highest-value correction. The inference is a
**cascade of falsifiable checks with recorded confidence**, not a heuristic soup.

```
dtype is datetime64 / has parseable date strings   -> DATETIME
dtype is bool, or 2 unique values                  -> BINARY
n_unique == 1 (or <= 1 ignoring NaN)               -> CONSTANT
unique_ratio > id_unique_ratio and (int or string) -> IDENTIFIER
   and name matches id-like pattern                    (confidence boost)
dtype is object/string:
    mean length > text_min_length or
       mean token count > text_min_tokens          -> TEXT
    else                                           -> CATEGORICAL
dtype is numeric:
    n_unique <= numeric_as_categorical_max and
       values are integral and                     -> CATEGORICAL (low conf)
       name matches code-like pattern                 or ORDINAL if name suggests
    else                                           -> NUMERIC
```

Three deliberate design points:

- **Confidence is returned, not discarded.** `ColumnProfile.semantic_confidence` is a
  float and `semantic_alternatives` lists runner-ups. Section 6 of the design goal
  ("when uncertain, expose the uncertainty") is met by making low confidence a
  *reported quality issue*, so `report.summary()` prints
  `city_code: inferred CATEGORICAL (confidence 0.55) - alternatives: NUMERIC`.
- **Name heuristics are advisory and never decisive alone.** A column called `user_id`
  with 3 unique values in 100k rows is not an identifier. The name only moves
  confidence; cardinality decides.
- **Thresholds scale with `n_rows`.** A 12-unique-value column is categorical in a
  100k-row frame and is probably continuous in a 15-row frame. The identifier and
  numeric-as-categorical thresholds are functions of `n_rows`, not constants.

---

## 5. The planner

### 5.1 Rules are objects, not `if` statements

```python
@dataclass(frozen=True)
class Rule:
    name: str
    applies_to: Callable[[ColumnProfile, Config], bool]
    decide: Callable[[ColumnProfile, Config], Decision | None]
    stage: Stage
    priority: int
```

Rules are registered per `Stage`. Within a stage, rules are evaluated in priority order
and the first rule that produces a `Decision` for a column wins (a documented, testable
conflict-resolution policy). Every `Decision` carries:

```python
@dataclass(frozen=True)
class Decision:
    column: str
    stage: Stage
    action: str            # "impute_median", "encode_onehot", ...
    params: dict
    rationale: str         # human sentence, shown by explain()
    rule: str              # which rule fired
    confidence: float
    source: Literal["rule", "user_override", "default"]
```

`source` is what makes overrides honest: when a user sets
`config.column("age").imputation = "mean"`, the produced decision is tagged
`user_override` and `explain()` marks it, rather than silently pretending the rule chose
it.

### 5.2 Stage order

The stage order is fixed and is itself a design claim, derived from common practice
plus the correctness fixes:

```
 1. CAST            dtype correction, sentinel -> NaN, downcast
 2. DROP_COLUMNS    constants, IDs, duplicate columns, over-missing columns
 3. DEDUPLICATE     exact duplicate rows   (rows, so before per-column fitting)
 4. MISSING_FLAG    add indicator columns while the original columns still exist
 5. DATETIME        expand datetime -> calendar features; then treat as numeric
 6. OUTLIERS        detect and act; bounds fitted here, applied in transform
 7. MISSING         impute (after outliers, so "outlier -> NaN -> impute" is coherent)
 8. TRANSFORM       skew correction (log1p / Yeo-Johnson / quantile)
 9. RARE_CATEGORY   group rare levels before encoding
10. ENCODE          categorical -> numeric
11. SCALE           numeric scaling (after encoding, so encoded cols can be scaled)
12. SELECT          near-constant, correlation, variance filters
```

Two orderings differ from the usual notebook order, deliberately:

- **MISSING_FLAG before DATETIME, OUTLIERS and MISSING.** Missingness is frequently
  informative (`a census-income notebook` established that `workclass` and `occupation`
  co-miss). If imputation runs first the signal is gone, and notebook code rarely
  captures it. It must also precede datetime expansion, which consumes the original column: a
  `NaT` has to be flagged while the column it belongs to still exists.
- **OUTLIERS before MISSING.** In notebook code this ordering usually happens by
  accident (`design-rationale.md` §5.6). Here it is chosen: the `"impute"` outlier
  strategy sets outliers to NaN and the imputation stage then fills them with a
  statistic computed *excluding* those outliers. That is the statistically defensible
  version of what usually happens by chance.

`SELECT` runs last because correlation between *encoded* features is what matters to a
model, not correlation between raw categorical labels.

### 5.3 Model family

`model_family` is a planning input with four values plus `None`:

| family | scaling | categorical encoding | skew transform | outlier default |
|---|---|---|---|---|
| `"linear"` | required | one-hot (low card) / target (high card) | yes | clip |
| `"tree"` | skipped | ordinal | no | report only |
| `"distance"` | required | one-hot | yes | clip |
| `"neural"` | required (MinMax) | one-hot / target | yes | clip |
| `None` | conservative default: standard-scale, one-hot, transform on heavy skew only | | | report |

This is mined directly from section 5 of docs/design-rationale.md. The default when the user
says nothing is `None`, which is the conservative branch, because by design:
"do not assume every ML model requires scaling".

---

## 6. Execution and copy discipline

`Pipeline.transform` does **not** do `X.copy()`. It maintains a dict of
`{name: array-or-Series}` seeded lazily from the input frame, and each transformer
writes only the columns it owns. Untouched columns are passed through by reference and
the final frame is constructed once, at the end, with `copy=False`.

Practical consequences:

- A pipeline that touches 5 of 400 columns allocates 5 columns, not 400.
- The input frame is never mutated (verified by a test that hashes the input frame
  before and after every public call).
- No intermediate whole-frame copies between stages.

Where a transformer genuinely needs a materialised frame (correlation filtering), it
takes one for the subset of columns it needs, not the whole frame.

### 6.1 Sampling

`profile(df, sample_size=N)` draws a deterministic sample (seeded `random_state`) for the
*expensive* statistics only: skewness, kurtosis, quantiles, correlation, cardinality on
wide frames. Cheap statistics (null counts, dtype, min/max) always run on the full frame
because they are single vectorised passes. The report records
`profile.sampling = {"used": True, "n": 50_000, "of": 590_540, "random_state": 42}` so
that no statistic is ever silently approximate.

### 6.2 Backends

`backends/base.py` defines a narrow `Backend` protocol (about 15 operations: column
access, null mask, unique count, quantiles, groupby-mean, concat). `pandas_backend.py`
is the only implementation. This is not speculative abstraction: it exists because the
docs/design-rationale.md shows the frames of interest reach 590k x 434, and an Arrow-backed
implementation is a foreseeable need. The protocol is deliberately tiny, and the rest of
the library is written against it only where the operation is hot. Cold paths use pandas
directly, because pretending otherwise would be abstraction for its own sake.

---

## 7. Performance strategy

Correct first, measure, then optimise. The plan:

1. **Reference implementation in pandas/NumPy.** Done first, with correctness tests
   against NumPy and scikit-learn where a trusted reference exists.
2. **Benchmark suite** (`benchmarks/`) measuring wall time and peak memory
   (`tracemalloc`) for profiling, each transformer, and end-to-end pipelines, against
   hand-written pandas and scikit-learn baselines.
3. **Optimise only what the benchmark shows.** The predicted hot spots, from the shape of
   the work rather than from guesswork, are:
   - per-column `.skew()`/`.kurt()` on wide frames (pandas dispatches per column; a
     single fused NumPy moment pass over a 2-D block is much cheaper);
   - `nunique()` on wide frames;
   - the correlation matrix on wide frames (already addressed by sampling);
   - one-hot encoding materialising dense float64.
4. **No native code in v1.** There is nothing in this workload that NumPy does not
   already vectorise. Introducing Rust or Cython before a benchmark justifies it would
   contradict the design goal and cost every user a build toolchain. The `backends/` seam is
   where a native implementation would go if measurements later demand it.

Measured results and the decisions they drove are recorded in `docs/performance.md`.

---

## 8. Reporting

`Report` is assembled from the `Journal`, an append-only list of records written by
transformers during `fit` and `transform`. A record is
`(stage, transformer, column, action, params, effect)` where `effect` holds measured
outcomes (rows affected, values imputed, categories grouped).

Two distinct things are reported and must not be confused:

- **the plan** — what was decided, and why (available before any data is transformed);
- **the journal** — what actually happened, with counts (available after).

`report.summary()` renders both. `report.to_dict()` / `to_json()` give the
machine-readable form; `to_html()` is a self-contained page with no external assets.

---

## 9. Error handling

`edaprep.exceptions` defines a small hierarchy rooted at `EdaPrepError`. Every raise site
supplies: what was attempted, on which column, why it failed, and what to do instead.

```
EdaPrepError
├── ConfigurationError      contradictory or unknown configuration
├── NotFittedError          transform before fit
├── SchemaError             transform-time columns disagree with fit-time
├── DataError               data cannot support the requested operation
│   ├── EmptyDataError
│   └── TransformationError   e.g. log on negative values
└── LeakageError            a fit-time-only operation was reached at transform time
```

`SchemaError` deserves emphasis: silently tolerating a missing or extra column at
transform time is how train/serve skew becomes invisible. `edaprep` raises by default
and offers `Config(on_unknown_columns="ignore"|"error")` for the cases where tolerance is
genuinely wanted.

---

## 10. Extension points (designed, not implemented)

There are future capabilities. Each has a named seam:

| future capability | seam |
|---|---|
| feature engineering | a `Stage.FEATURE_ENGINEERING` slot exists in the stage enum, unused |
| new transformers | subclass `Transformer`, register a `Rule` via `rules.register()` |
| model selection / HPO | consumes `Plan` + `Report`; outside the package |
| NLP | `preprocessing/text.py` detects and reports; a `TextVectorizer` slots into `Stage.ENCODE` |
| time series | `Stage.DATETIME` already produces calendar features; lag/rolling features are a further transformer |
| GPU / distributed | `backends/` protocol |
| data validation | `profiling/quality.py` issue records are already schema-shaped |

Nothing above is implemented in v1. The seams cost nothing (an unused enum member, a
registration function that already exists for internal use) and their absence would
force a redesign later.
