# Design rationale

Why `edaprep` is shaped the way it is.

Every EDA and preprocessing library is a bet about which parts of the workflow are
worth automating. This document states the bet: it catalogues the workflow that tabular
ML notebooks converge on, names the places that workflow reliably goes wrong, and
derives each of the library's design decisions from one of them.

Read this before `architecture.md`. That document says *what* the library does; this one
says *why* any of it is necessary.

---

## 1. The workflow everyone writes

Tabular ML notebooks converge on nearly the same sequence, regardless of domain:

```
 1. load csv
 2. .head()
 3. .shape
 4. .info()
 5. .describe()  /  .describe(include='all').T
 6. .isnull().sum()
 7. missing-value matrix / heatmap
 8. .duplicated().sum()
 9. split columns:
      num_cols = df.select_dtypes(include=['int64','float64'])
      cat_cols = df.select_dtypes(include=['object'])
10. per-numeric loop:     skew + kurtosis + histplot + boxplot
11. per-categorical loop: value_counts + countplot
12. outlier detection:    IQR fence and/or |z| > 3
13. correlation heatmap
14. VIF
15. statistical tests against the target (t-test / ANOVA)
16. train_test_split(..., stratify=y)
17. impute -> encode -> scale
18. class imbalance: SMOTE or class_weight
19. model loop over a dict of estimators
```

Steps 1–15 are **identical in intent across projects** and differ only in column names.
Steps 16–18 are where practice varies most, and where the correctness problems
concentrate.

That split is the whole thesis. Steps 1–15 are mechanical enough to automate. Steps
16–18 are where a library earns its place by being *correct* rather than merely
convenient.

---

## 2. The dependency stack this implies

Weighted by how often each appears in real notebook preprocessing:

```
matplotlib / seaborn   visual only
pandas, numpy          the hard core
scipy                  zscore, boxcox, yeojohnson
scikit-learn           SimpleImputer, ColumnTransformer, and the modelling half
statsmodels            VIF, and little else
category_encoders      TargetEncoder
imbalanced-learn       SMOTE
```

So: `numpy + pandas + scipy` are required. `scikit-learn`, `matplotlib` and `pyarrow`
are extras. `statsmodels` is not a dependency at all — VIF is 20 lines of NumPy
(section 5.3), and taking a whole package for one function is not a trade worth making.

---

## 3. The code that gets rewritten every time

### 3.1 The `select_dtypes` split

```python
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
```

This is the most-typed line in tabular data science, and the most consequential.

**Problems**

- Hard-coding `int64`/`float64` silently drops `float32`, `int32`, `Int64` (nullable),
  `uint8`, and Arrow-backed columns. On a memory-optimised frame it excludes most of the
  data.
- `include=['object']` misses `category` and `string[python]` dtypes.
- It is a **dtype** split, not a **semantic** split. A `zip_code` stored as `int64` lands
  in `num_cols` and gets standard-scaled; a 5-level Likert column encoded `1..5` gets
  treated as continuous; a numeric column stored as text is dropped from the numeric
  branch entirely.

→ `edaprep.profiling.column_types` replaces it with semantic inference that returns a
confidence and a list of runner-up types.

### 3.2 The IQR fence

```python
Q1 = df[col].quantile(0.25); Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
```

Written once per project, per column, with the multiplier drifting between `1.5` and
`3.0` and no record of why.

→ `preprocessing.outliers.IQRDetector(k=...)`, with the multiplier a named, reported
parameter.

### 3.3 The z-score fence

```python
z_scores = zscore(df[col]);  outliers = df[np.abs(z_scores) > 3]
```

→ `ZScoreDetector`, plus the `ModifiedZScoreDetector` (MAD-based) that skewed columns
actually need — see section 5.2.

### 3.4 The fit-on-train / apply-to-test triplet

Careful notebooks do this correctly, and pay for the correctness with pure boilerplate —
three lines per column, per statistic:

```python
age_median = train_df['Age'].median()
train_df['Age'] = train_df['Age'].fillna(age_median)
test_df ['Age'] = test_df ['Age'].fillna(age_median)

embarked_mode = train_df['Embarked'].mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
test_df ['Embarked'] = test_df ['Embarked'].fillna(embarked_mode)

fare_cap = train_df['Fare'].quantile(0.95)
train_df['Fare'] = train_df['Fare'].clip(upper=fare_cap)
test_df ['Fare'] = test_df ['Fare'].clip(upper=fare_cap)
```

This is the clearest statement of the library's purpose: **the semantics here are right
and the ergonomics are terrible.** `edaprep` makes this the default behaviour of
`fit`/`transform` and reduces it to zero lines of user code.

### 3.5 The sentinel scan

```python
question_mark_counts = (df == '?').sum()
cols_with_question_marks = question_mark_counts[question_mark_counts > 0]
...
df[col] = df[col].replace('?', np.nan)
```

Usually followed, in the better notebooks, by a genuinely good piece of analysis — a
**co-missingness correlation**, which reveals that two columns go missing together and
therefore share a cause:

```python
corr = df[cols_with_question_mark].isnull().astype(int).corr()
```

→ `profiling.quality.detect_sentinels()` (a configurable vocabulary: `?`, `-`, `NA`,
`N/A`, `null`, `none`, `missing`, `unknown`, …) and
`profiling.quality.missingness_correlation()`.

### 3.6 The greedy high-correlation drop

```python
corr_matrix = df[num_cols].sample(10000, random_state=42).corr('spearman').abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
df_reduced = df.drop(columns=to_drop)
```

→ Extracted, but redesigned. See section 5.4.

---

## 4. The idea worth formalising: routing, not sequencing

The most sophisticated hand-written pipelines stop being a *sequence* and become a
*decision procedure*. They look like this:

```python
skews = X_train[num_cols].apply(lambda x: x.skew(skipna=True)).fillna(0)
normal_features   = skews[abs(skews) < 1].index.tolist()
moderate_features = skews[(abs(skews) >= 1) & (abs(skews) < 5)].index.tolist()
heavy_features    = skews[abs(skews) >= 5].index.tolist()

for col in moderate_features:                       # clip the middle tier
    lower, upper = X_train[col].quantile([0.01, 0.99])
    X_train[col] = X_train[col].clip(lower, upper)

def make_preprocessor(model_name):
    standard_pipe = [SimpleImputer('median'), StandardScaler()]
    robust_pipe   = [SimpleImputer('median'), RobustScaler()]
    power_pipe    = [SimpleImputer('median'), PowerTransformer('yeo-johnson'), RobustScaler()]
    cat_pipe_tree   = [SimpleImputer('most_frequent'), OrdinalEncoder(unknown=-1)]
    cat_pipe_target = [SimpleImputer('most_frequent'), TargetEncoder()]
    cat_pipe = cat_pipe_target if model_name in [
        'LogisticRegression', 'SVM(RBF)', 'KNN', 'NaiveBayes'] else cat_pipe_tree
    return ColumnTransformer([...])
```

That is a planner, written by hand, buried in a closure. Three independent routing axes
are encoded in it, and formalising them is the highest-value thing this library does.

**Axis 1 — route numeric columns by distribution shape.**

| abs(skew) | treatment |
|---|---|
| `< 1` | median impute, then StandardScaler |
| `1 – 5` | median impute, clip to [P1, P99], then RobustScaler |
| `>= 5` | median impute, Yeo-Johnson, then RobustScaler |

**Axis 2 — route categorical columns by the consuming model family.**

| model family | encoder | scaling |
|---|---|---|
| linear / distance-based / naive Bayes | target (or one-hot) | required |
| tree / gradient boosting | ordinal | unnecessary |

The same idea shows up structurally elsewhere: pipelines that maintain **two parallel
frames**, one for trees and one for linear models, dropping different columns from each
(`education` vs `education_num` — the nominal and ordinal encodings of one variable).

**Axis 3 — choose the outlier method by skewness.**

```python
if abs(X_train[col].skew()) > 1:      # skewed -> IQR with a wide k=3 fence
    Q1, Q3 = X_train[col].quantile([0.25, 0.75]); IQR = Q3 - Q1
    lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
else:                                 # symmetric -> z-score
    z_scores = zscore(X_train[col]);  mask = np.abs(z_scores) <= 3
```

**These three axes are the seed rule set of `edaprep.planning.rules`.** The library's
contribution is not inventing them — it is giving them names, making the thresholds
configurable, and making the reasoning *printable*, so that the reason a column got
`RobustScaler` is stated rather than buried in a closure.

---

## 5. Where the workflow reliably goes wrong

Each of these is a defect that recurs across projects, and each maps to a specific
design decision.

### 5.1 Fitting before the split

```python
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])   # cell 9
df.to_csv('processed_train.csv')                              # cell 10
...
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)      # cell 11
```

The mean and standard deviation used to scale the training rows were computed with the
test rows included. The leaked quantity is small for `StandardScaler` on a large sample,
but when the processed frame is **persisted and reused**, the leak propagates to every
downstream experiment.

The same shape appears with `if df['price'].skew() > 1: df['price'] = np.log1p(...)`
applied to the whole frame before splitting, and with VIF or correlation computed on the
full frame (diagnostic only, lower severity, still wrong).

**Design response.** No transformer can observe data it was not fitted on. Every learned
statistic lives in fitted state populated only inside `fit()`, and `transform()` is a
pure function of that state and `X`. A leakage audit records which statistics were
learned and whether they consumed the target; `report.leakage` lists them. Transformers
that legitimately consume `y` declare `uses_target = True` and are forced through inner
cross-fitting (5.3).

### 5.2 Index misalignment in the outlier handler

```python
z_scores = zscore(X_train[col].dropna())        # length = n_non_null
outliers_b = X_train.loc[np.abs(z_scores) > 3]  # boolean mask of length n_non_null
X_train[col] = X_train[col].where(np.abs(z_scores) <= 3)   # applied to length-n column
```

`z_scores` is a positional array over the **non-null subset**, then used as a boolean
mask against the **full** frame. When the column contains any NaN the mask is shorter
than the frame and the rows flagged are the wrong rows.

The variant without `dropna()` is worse: `scipy.stats.zscore` returns all-NaN whenever
the column has a single missing value, so the mask is all-`False` and **no outlier is
ever detected** — silently.

There is also a units problem hiding here: `scipy.stats.zscore` uses `ddof=0` while
`pandas.Series.std()` uses `ddof=1`. Code that mixes them is not computing one statistic.

**Design response.** Outlier masks are aligned boolean `Series` over the original index.
NaN handling is explicit — missing values are never flagged as outliers. `ddof` is a
named parameter defaulting to `0`. All three are asserted in tests.

### 5.3 Target encoding without cross-fitting

`category_encoders.TargetEncoder` inside a `ColumnTransformer` is train/test-safe, but it
is **not row-safe**: for a training row in category *c*, the encoded value includes that
row's own target. For a high-cardinality column with near-singleton categories the
encoding approaches the target itself, the model memorises it, and validation scores
computed on the same fold are optimistic.

**Design response.** `edaprep`'s `TargetEncoder` performs **inner K-fold cross-fitting**
on `fit`: the value written into the training output for a row is computed from the
*other* folds only, while the mapping stored for `transform` is the full-train mapping.
Smoothing is m-estimate style with a configurable prior weight. This matches the design
of `sklearn.preprocessing.TargetEncoder` (1.3+) and is verified against it in the tests.

### 5.4 Order-dependent correlation pruning

```python
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
```

For a group of three mutually correlated columns `{a,b,c}` this drops `b` and `c`
(keeping `a`), which is defensible. But for the chain `a~b`, `b~c` with `a` and `c`
uncorrelated it drops **both** `b` and `c`, losing `c`'s independent information. The
outcome depends entirely on column order. Computed on the full frame before the split,
it also lets the test set decide which features exist.

**Design response.** `CorrelationFilter` builds a graph of edges above the threshold,
extracts connected components, and keeps one representative per component — chosen by an
explicit, reported criterion (highest absolute target correlation, else lowest
missingness, else first in column order). Deterministic, order-independent, fitted on
train only.

### 5.5 Bounds fitted on train and never applied to test

Outlier handlers that mutate `X_train` in place and never touch `X_test` are common. This
is not leakage; it is worse in a subtle way. The model is trained on a clipped
distribution and served an unclipped one, so the train/serve distributions differ **by
construction**.

**Design response.** Clipping bounds are fitted state, applied identically in every
`transform()` call. The report states the bounds and the fraction of rows affected.

### 5.6 A strategy emerging by accident

A common accidental sequence: the outlier step replaces extreme values with NaN, then a
`SimpleImputer` later in the `ColumnTransformer` fills them with the median. The net
effect — "replace extreme values with the median" — is a real strategy, and a reasonable
one. But it was never chosen, and never reported.

**Design response.** `edaprep` names it (`outlier_strategy="impute"`) and the plan prints
it. The order between missing-value handling and outlier handling is an explicit property
of the plan, not an accident of cell order.

### 5.7 Double-correcting class imbalance

Applying `SMOTE` after the split, on the training set only, is correct. Also passing
`class_weight='balanced'` to the same model double-corrects it.

**Scope decision.** Resampling is a *modelling* concern. `edaprep` **detects and reports**
class imbalance (with the ratio and a recommendation), exposes
`report.target.imbalance_ratio`, and stops there. It does not resample.

### 5.8 `inplace=True` and unmanaged copies

`inplace=True` and `.copy()` used without a policy. On a 590k × 434 frame,
`df1 = df.copy(); df2 = df.copy()` triples peak memory.

**Design response.** `edaprep` never mutates the caller's frame. `transform()` returns a
new frame assembled column-by-column, reusing untouched column blocks rather than doing a
whole-frame `.copy()` up front. See `architecture.md` §6.

### 5.9 `warnings.filterwarnings("ignore")`

Near-universal at the top of a notebook. It suppresses, among other things, the pandas
`SettingWithCopyWarning` raised by mutating a slice — the very warning that would have
surfaced 5.2.

**Design response.** `edaprep` never calls `filterwarnings` at module scope. It raises
typed exceptions (`edaprep.exceptions`) with actionable messages, and routes advisories
through structured warning records in the report rather than the `warnings` module alone.

---

## 6. What belongs in a preprocessing library, and what does not

| capability | in `edaprep`? |
|---|---|
| dtype-based column splitting | **yes** — replaced by semantic inference |
| IQR / z-score / modified-z fences | **yes** — `preprocessing.outliers` |
| skew tiering and transform choice | **yes** — `planning.rules` |
| median/mode imputation fitted on train | **yes** — `preprocessing.missing` |
| one-hot / ordinal / target / frequency encoding | **yes** — `preprocessing.encoding` |
| Standard/Robust/MinMax/MaxAbs scaling | **yes** — `preprocessing.scaling` |
| constant and duplicate-column removal | **yes** — `preprocessing.selection` |
| correlation pruning | **yes** — redesigned (5.4) |
| sentinel detection | **yes** — `profiling.quality` |
| missingness co-occurrence | **yes** — `profiling.quality` |
| calendar feature expansion | **yes** — `preprocessing.datetime_features` |
| VIF | **yes** — `eda.correlation`, pure-NumPy reimplementation |
| t-test / ANOVA vs target | **yes** — `eda.target` |
| missing-value matrix / heatmap | **yes** — `visualization`, optional dep |
| domain feature engineering (`Title` regex, `FamilySize`) | no — project-specific |
| SMOTE / `class_weight` | no — modelling. **Report only** |
| model dicts, learning curves, SHAP | no — out of scope |
| Prophet / `seasonal_decompose` | no — extension point reserved |
| KMeans/DBSCAN/LOF anomaly *models* | no — `IsolationForest` kept as an optional outlier detector only |

---

## 7. What this analysis demands of the design

1. **Semantic typing is mandatory, not a nicety.** The most-typed line in the workflow
   (3.1) is also the most consequential bug source.
2. **The planner is the product.** A hand-written planner already exists in practice
   (section 4). Formalising it is higher-value than any individual transformer.
3. **Model family is a first-class planning input.** Real pipelines route on it, so
   `AutoPipeline(model_family=...)` must exist.
4. **Leakage prevention must be structural, not documentary.** A convention will not
   prevent 5.1; an architecture where `transform` cannot see un-fitted state will.
5. **Every automatic action must be printed.** 5.6 shows an entire strategy emerging by
   accident. The plan must state it.
6. **Sampling must be built in.** Wide frames already force `.sample()` before `.corr()`
   in practice, because the full computation is intractable. The profiler needs a
   principled `sample_size` with the sampling recorded.
7. **Reports must be persistable.** A processed CSV with no record of how it was produced
   is the norm. `report.to_json()` is the fix.

---

## 8. Target state

The generic, repeated EDA and preprocessing logic in a typical project runs to well over
a thousand lines. The goal is that it becomes:

```python
import edaprep

profile = edaprep.profile(train_df, target="is_fraud")
report  = edaprep.EDA(train_df, target="is_fraud").analyze(level="standard")

pipe = edaprep.AutoPipeline(target="is_fraud", model_family="tree", random_state=42)
pipe.fit(train_df)
pipe.explain()

X_train = pipe.transform(train_df)
X_test  = pipe.transform(test_df)
```

with `pipe.plan_`, `pipe.report_` and `pipe.explain()` making every decision above
visible, overridable and reproducible.
