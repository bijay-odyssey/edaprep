# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is `0.x`, the public API may change between minor versions; anything that does
will be listed under **Changed** with a migration note.

## [0.2.0] — 2026-08-26

### Removed

- **`Config.n_jobs`.** It was accepted and never read by anything — a false promise to
  anyone setting `n_jobs=-1` and expecting work to be parallelised. Removed by
  [@qiaobochi040726-source](https://github.com/qiaobochi040726-source) in
  [#11](https://github.com/bijay-odyssey/edaprep/pull/11), after benchmarking by
  [@zbs-ops](https://github.com/zbs-ops) and a second independent run established that
  threading the per-column loop in `profiling/statistics.py` helps at one frame shape
  (20,000 × 300, 1.27×) and hurts at three others (1.15–1.38× slower). Exploiting that
  would need a shape-dependent branch, which is the same thing §1 of
  `docs/performance.md` records deleting once already.

  *Migration:* delete the argument. It never did anything, so nothing else changes.
  A `Config` saved by an earlier version still loads — see below.

### Changed

- **`Config.from_dict` no longer raises on settings it does not recognise.** It drops
  them and warns instead. `Report.to_dict()` embeds the configuration, so a report
  written before a setting was retired has to keep loading afterwards; `n_jobs` is the
  first case. Dropping is warned about rather than silent, because an unrecognised key
  is equally likely to be a misspelling.

### Fixed

- **Placeholder strings converted at the cast step are now imputed.** A column stored
  as text purely because a handful of values are blank (or `?`, `N/A`, …) is measured
  by the profiler as 0% missing, because at that point those values are still strings.
  The cast then parses the column to a real dtype and turns them into `NaN`, after
  which the imputation rule declined to act — it was keyed on the profile's
  `n_missing`, which was zero. The result was `NaN` in output the library described as
  ML-ready, which then raises in any estimator that does not accept them.

  The rule now also consults the placeholder counts the profiler already records, and
  the rationale names them rather than reporting "0.0% missing" while imputing anyway:

  ```
  + impute_median - 11 placeholder value(s) become NaN when the column is cast, so it
                    needs imputation despite reporting 0.0% missing; median is robust
                    to outliers
  ```

  Found while preparing a worked example on the Telco Customer Churn dataset, whose
  `TotalCharges` column is exactly this shape. Columns that cast cleanly are
  unaffected and still get no imputation step.

## [0.1.0] — 2026-08-22

First public release. Available on PyPI: `pip install edaprep`.

### Added

**Profiling**
- `profile()` returns a frozen, JSON-serialisable `DatasetProfile`.
- Semantic column typing that returns a *confidence* and runner-up types rather than a
  bare label, replacing the dtype-based `select_dtypes` split.
- Data-quality detection: sentinel strings, numeric placeholders, constant and
  near-constant columns, identifiers, duplicate columns, correlated missingness, mixed
  Python types, stray whitespace, case-variant categories, class imbalance, and columns
  suspiciously associated with the target.

**Planning**
- `Planner` maps `(DatasetProfile, Config) → Plan` without ever seeing a DataFrame.
- `Plan` is inert, serialisable data: printable, diffable, editable, re-executable.
- Every decision carries an English rationale naming the measurement behind it, and is
  tagged with its source (`rule`, `user_override`, `default`).
- Rules are registrable objects, so the decision logic can be extended or pre-empted
  without subclassing anything.

**Preprocessing**
- Missing values, duplicates, outliers, categorical encoding, numeric scaling,
  distribution transforms, datetime expansion, text handling, dtype correction and
  feature selection.
- `TargetEncoder` cross-fits on inner K folds, so no training row is ever encoded using
  its own target.
- Outlier detectors return index-aligned boolean masks and never flag missing values.
- Transform validity is checked against each column's actual support and refused with a
  named alternative rather than emitting silent `NaN`.

**Pipelines**
- `AutoPipeline` — profile, plan, explain, execute.
- `Pipeline` — explicit, with a chainable builder.
- scikit-learn estimator protocol (`get_params`/`set_params`/`__sklearn_tags__`)
  implemented without importing scikit-learn, which stays an optional dependency.

**EDA**
- Three analysis levels that differ in work done, not just in what is displayed.
- VIF reimplemented in NumPy as a single matrix decomposition, dropping the
  `statsmodels` dependency and correctly reporting perfectly collinear columns as
  infinite rather than 1.0.
- Benjamini-Hochberg adjustment on feature/target tests, so ranking hundreds of columns
  by p-value does not manufacture significance.

**Reporting**
- Text, JSON and self-contained HTML output, with no external assets.
- A leakage audit recording which transformers read the target and whether they
  cross-fitted.
- Reproducibility metadata: library version, configuration, random seed, and whether
  profiling sampled.

### Guarantees

- Every learned statistic is fitted on the training frame only; `transform` is a pure
  function of that fitted state. Enforced by two tests: an AST check that no
  `_transform` body calls an aggregation, and a behavioural check that transforming a
  frame whole equals transforming it row by row.
- The input frame is never mutated.
- Nothing is discarded silently — dropped columns, imputed values, grouped categories,
  clipped rows and unseen categories are all counted and reported.

### Known limitations

- Text columns are detected and reported, not vectorised.
- Only a pandas backend; the `backends/` protocol exists but has one implementation.
- No resampling: class imbalance is measured and reported, because resampling belongs
  after the train/test split and with the model.

[0.2.0]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.2.0
[0.1.0]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.1.0
