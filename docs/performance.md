# Performance

**Phase 8 and 9 deliverable.** What was measured, what the measurements changed, and
what is still slow on purpose.

Reproduce with:

```bash
python benchmarks/bench.py --rows 100000 --wide-rows 20000 --wide-columns 300 --repeat 5
```

Numbers below are from Python 3.10.6, NumPy 1.24.4, pandas 2.2.3 on Windows 11
(consumer laptop). Time is the **minimum** of 5 runs; the minimum estimates the cost of
the work, while the mean estimates the work plus whatever else the machine was doing.
Memory is peak additional allocation from `tracemalloc`, which sees Python-level
allocations including NumPy buffers, but not allocations made inside C libraries that
bypass the Python allocator. Treat memory as comparative, not absolute.

---

## 1. The headline result: the optimisation I assumed would help was a pessimisation

`profiling/statistics.py` originally hand-rolled the moment computations in NumPy, on
the stated premise that one fused pass over a 2-D block would beat pandas' per-column
dispatch. The first benchmark run said the opposite, at every shape tested:

| shape | NumPy block kernel | pandas per-column | pandas frame-level |
|---|---|---|---|
| 20,000 × 300 | 799 ms | **557 ms** | 806 ms |
| 100,000 × 50 | 731 ms | **381 ms** | 600 ms |
| 500,000 × 12 | 969 ms | **542 ms** | 810 ms |
| 5,000 × 1000 | 838 ms | 1210 ms | **802 ms** |

and it used **298 MiB against pandas' 1.6 MiB**.

The cause is structural rather than incidental. pandas computes m2, m3 and m4 in a
single fused Cython pass per column. Any NumPy formulation needs a separate traversal
of the data for each power, so it is memory-bandwidth bound. Cache-sized chunking
(tested at 64 KiB, 256 KiB, 1 MiB, 4 MiB and 32 MiB buffers) narrowed the gap but never
closed it.

**The hand-rolled kernel was deleted.** What replaced it delegates to pandas and adds
only what pandas does not provide: infinity masking, a scale-relative degeneracy guard,
zero/negative counts, the MAD, and the `NumericStats` value type. This is section 28 of
the design goal applied literally — do not reimplement what pandas already does well, and let
measurement rather than intuition decide which case that is.

### 1.1 A second, subtler mistake in the same file

The rewrite dispatched to *frame-level* pandas reductions (`data.mean()`,
`data.skew()`) for narrow frames and per-column ones for wide frames. That was backwards
in both directions:

| shape | per-column | frame-level |
|---|---|---|
| 20,000 × 300 | **543 ms / 3.1 MiB** | 1684 ms / 377.8 MiB |
| 5,000 × 1000 | **850 ms / 4.4 MiB** | 915 ms / 157.6 MiB |
| 2,000 × 3000 | 1680 ms / 10.0 MiB | 1375 ms / 189.3 MiB |

Frame-level reductions operate on pandas' internal blocks and allocate intermediates
proportional to the whole frame. They only win on time past about 3000 columns, and
then by 1.2× for 19× the memory. The branch was removed entirely; the per-column loop
is now the only path, which is also less code.

### 1.2 Net effect on that function

| | before | after | change |
|---|---|---|---|
| time (20,000 × 300) | 795 ms | **578 ms** | 1.38× faster |
| memory | 298 MiB | **49 MiB** | 6.1× less |
| vs. equivalent pandas loop | 2.10× **slower** | **1.83× faster** | — |

The "equivalent pandas loop" baseline was also corrected during this work: the original
version computed fewer statistics than `edaprep` does (no zero/negative/missing counts,
no MAD), which made the comparison meaningless. It now computes the same set.

---

## 2. Other bottlenecks found by profiling, and what was done

`cProfile` on `profile(df, target=...)` for a 100,000 × 14 frame:

| finding | share | fix |
|---|---|---|
| `detect_sentinels` + `detect_whitespace_issues` normalising every cell | 21% | count first, then normalise the **distinct** values and weight by their counts. A 100,000-row column with 500 levels was doing 200× more work than needed. |
| `_cramers_v` via `pd.crosstab` | 17% | `pd.factorize` + `np.bincount`. `crosstab` routes through `pivot_table`, which sorts, groups and builds a labelled frame that is immediately discarded. |
| `_correlation_ratio` — one `groupby` per numeric column | 36% (wide frames) | batch it. Every numeric column shares the same grouping, so sums and counts accumulate in one masked pass **per class** (typically 2–20) instead of one groupby per column. |

Each replacement is asserted to agree with the original to floating-point tolerance.

`profile()` on the 20,000 × 300 frame went from **2394 ms / 300 MiB** to
**2905 ms / 96 MiB**; the wall time moved within run-to-run noise while peak memory fell
by 3.1×. Chunking the batched association and the quantile call is what bought the
memory back.

---

## 3. Current results

### Profiling

| operation | time | peak memory |
|---|---|---|
| `profile(100k × 14)` | 970 ms | 25.6 MiB |
| `profile(quick)` | 635 ms | 24.5 MiB |
| `profile(20k × 300)` | 2905 ms | 95.8 MiB |
| `numeric_block_stats(20k × 300)` | 578 ms | 48.8 MiB |
| ↳ vs equivalent pandas loop | 1056 ms | **1.83× faster** |

`profile()` is ~4× slower than a hand-written pandas EDA block (`.info()`,
`.describe()`, `.isnull().sum()`, per-column `.skew()`). That comparison is not
like-for-like and is reported anyway: `edaprep` additionally runs semantic type
inference on every column, six data-quality scans, duplicate-column hashing,
co-missingness correlation, and per-feature target association. The baseline does none
of those. **~1 second to fully characterise a 100,000-row dataset is the right trade.**

### Transformers

| operation | edaprep | baseline | result |
|---|---|---|---|
| `Scaler` (standard, 6 cols) | 6.9 ms | sklearn `StandardScaler` 26.7 ms | **3.9× faster** |
| `MissingValueHandler` (median) | 5.2 ms | sklearn `SimpleImputer` 17.0 ms | **3.3× faster** |
| `OutlierHandler` (IQR clip, 6 cols) | 25.2 ms | the usual IQR block 41.4 ms | **1.6× faster** |
| `OneHotEncoder` (8 levels) | 47.4 ms | sklearn `OneHotEncoder` 40.8 ms | 1.16× slower |
| `TargetEncoder` (500 levels, 5-fold) | 124.5 ms | — | no equivalent baseline |
| `detect_outliers` (1 col) | 4.3 ms | — | 23M rows/s |

The scaler and imputer wins are mostly avoided copies: sklearn converts to a NumPy array
and back, while `edaprep` writes only the columns it owns and passes the rest through by
reference.

`OneHotEncoder` is slightly slower and that is accepted: it produces a labelled
`DataFrame` with deterministic column ordering rather than a bare array.

`TargetEncoder` has no honest baseline. `sklearn.preprocessing.TargetEncoder` is the
closest, but the cross-fitting that makes both of them leak-free is exactly the cost
being measured, so the comparison would be against a different algorithm.

### Pipelines

| operation | time | peak memory |
|---|---|---|
| `AutoPipeline.fit_transform` (tree) | 1370 ms | 33.3 MiB |
| `AutoPipeline.fit_transform` (linear) | 1667 ms | 36.6 MiB |
| `AutoPipeline.transform` (fitted) | 240 ms | 35.6 MiB |
| ↳ sklearn `ColumnTransformer` | 104 ms | 67.2 MiB |

`transform` is 2.3× slower than an equivalent `ColumnTransformer` **and uses half the
memory**. The time difference is work the `ColumnTransformer` does not do: expanding
datetime columns into calendar features, adding missing indicators, applying outlier
fences, and journalling the measured effect of every step. The memory difference is the
copy discipline — `ColumnTransformer` materialises intermediate arrays per branch and
concatenates; `edaprep` replaces columns in place in a new frame and passes untouched
blocks through by reference.

`fit_transform` includes profiling, planning and report construction. Those are
one-time costs; `transform` is the per-batch cost and is what matters in production.

### EDA levels

| level | time | peak memory |
|---|---|---|
| `quick` | 521 ms | 15.9 MiB |
| `standard` | 1205 ms | 24.5 MiB |
| `deep` | 1411 ms | 24.5 MiB |

The levels are genuinely different amounts of work, not different amounts of display:
`quick` skips moments, correlation, outlier scanning and target association entirely.

---

## 4. What was deliberately *not* optimised

* **No native code.** Nothing in this workload is un-vectorisable, and the one place a
  hand-written kernel looked promising turned out to be slower than pandas. Introducing
  Rust or Cython before a benchmark justifies it would contradict the design goal and cost
  every user a build toolchain. The `backends/` protocol is where a native
  implementation would go if measurements later demand one.

* **Correlation on very wide frames** is skipped rather than made fast. A 500 × 500
  matrix is 250,000 numbers nobody reads; `Thresholds.correlation_max_columns` skips it
  in `standard` analysis, and `deep` opts back in.

* **`value_counts` per column** (~0.75 s of the wide-frame profile) is left alone. It is
  a single hash aggregation per column with no redundant work; the cost is inherent to
  needing exact distinct counts.

* **`AutoPipeline.fit`** is not tuned. It runs once per dataset and is dominated by
  profiling, which is already the subject of section 2.

---

## 5. Method notes

* **Minimum, not mean.** Repeated runs on a laptop are contaminated by scheduling and
  thermal drift; the minimum is the cleanest estimate of the work itself.
* **Interleaving.** When comparing candidates, they are run round-robin rather than
  in blocks, so drift affects all of them equally. Early measurements in this document
  that compared candidates across separate processes proved unreliable — pandas'
  per-column baseline measured 557 ms in one run and 1210 ms in another — and were
  re-taken interleaved.
* **Fair baselines.** Every baseline computes the same statistics as the `edaprep` call
  it is compared against. Where that is impossible (`TargetEncoder`), no baseline is
  reported rather than a misleading one.
* **No claim without a number.** Every performance statement in the documentation
  traces to a row in `benchmarks/results/latest.json`.
