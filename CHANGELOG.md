# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is `0.x`, the public API may change between minor versions; anything that does
will be listed under **Changed** with a migration note.

## [0.3.0] — 2026-09-16

A codebase audit ([#36](https://github.com/bijay-odyssey/edaprep/issues/36)–[#47](https://github.com/bijay-odyssey/edaprep/issues/47))
found twelve genuine bugs in one pass. Seven are fixed here, plus one new feature;
the remaining five ([#36](https://github.com/bijay-odyssey/edaprep/issues/36),
[#37](https://github.com/bijay-odyssey/edaprep/issues/37),
[#38](https://github.com/bijay-odyssey/edaprep/issues/38),
[#44](https://github.com/bijay-odyssey/edaprep/issues/44),
[#46](https://github.com/bijay-odyssey/edaprep/issues/46)) are still open — one of
them (#36) deliberately held back for careful review; see below. Everything shipped
here landed in less than a day, most of it from new contributors picking issues up
within hours of them being filed.

### Added

- **`knn` and `iterative` imputation strategies** for `MissingValueHandler`
  ([#48](https://github.com/bijay-odyssey/edaprep/pull/48), closes
  [#3](https://github.com/bijay-odyssey/edaprep/issues/3), by
  [@feyzasagman](https://github.com/feyzasagman)), wrapping scikit-learn's
  `KNNImputer`/`IterativeImputer` behind the `[advanced]` extra. Both are fitted once
  in `_fit` and never refit at transform time — asserted directly by a test that
  monkeypatches `.fit` to raise if called during `transform`. All-missing predictor
  columns are excluded from the block so they can't distort neighbouring columns; an
  all-missing *target* column is left as `NaN` with a warning rather than silently
  producing nothing. Opt-in per column or globally; `"auto"` does not select them yet.

### Fixed

All seven below were found in the same audit and are independent of each other.
`TargetEncoder`'s cross-fit leak ([#36](https://github.com/bijay-odyssey/edaprep/issues/36))
is the most severe finding of the audit and is deliberately *not* in this release —
it touches the library's central leakage guarantee and is being reviewed carefully
rather than shipped quickly. `Plan.without_columns` silently dropping steps (#37),
sample-scoped quality detection (#38), timedelta profiling (#44) and duplicate
column names crashing with opaque errors (#46) are real but still open too.

- **`Config.from_dict` now tolerates retired `Thresholds` and per-column settings**
  ([#53](https://github.com/bijay-odyssey/edaprep/pull/53), closes
  [#39](https://github.com/bijay-odyssey/edaprep/issues/39), by
  [Erol Tasci](https://github.com/Voyagerroc-Lab)), the same warn-and-drop tolerance
  the top-level fields already had — `n_jobs` was the first case, in 0.2.0. Direct
  `Config.set_columns()` still raises on an unrecognised key; only `from_dict`
  (deserialising a report saved by an older version) warns and drops.
- **`OrdinalEncoder`'s `dtype` parameter is now actually applied**
  ([#55](https://github.com/bijay-odyssey/edaprep/pull/55), closes
  [#40](https://github.com/bijay-odyssey/edaprep/issues/40), by
  [Erol Tasci](https://github.com/Voyagerroc-Lab)), previously silently ignored in
  favour of a hardcoded `float64`. Falls back to `float64` when a non-nullable
  integer dtype can't represent a missing value; nullable dtypes (`"Int32"`, ...) are
  preserved with their missingness intact.
- **`DateTimeExpander`'s boolean calendar features** (`is_weekend`, `is_month_start`,
  `is_month_end`, `is_quarter_start`, `is_quarter_end`, `is_year_start`,
  `is_year_end`) **now propagate `NaN` for a missing date**
  ([#52](https://github.com/bijay-odyssey/edaprep/pull/52), closes
  [#41](https://github.com/bijay-odyssey/edaprep/issues/41), by
  [Erol Tasci](https://github.com/Voyagerroc-Lab)) instead of fabricating `False` —
  the numeric calendar features already did this correctly; the boolean ones didn't.
- **`DataTypeInference`'s `"stripped"` journal count no longer counts missing values**
  ([#56](https://github.com/bijay-odyssey/edaprep/pull/56), closes
  [#42](https://github.com/bijay-odyssey/edaprep/issues/42), by
  [Alaa Bakr](https://github.com/alaa-bakr-analyst)). `NaN != NaN` was inflating the
  count with rows nothing happened to.
- **`categorical_summary`'s rare-level floor now matches the planner's**
  ([#54](https://github.com/bijay-odyssey/edaprep/pull/54), closes
  [#43](https://github.com/bijay-odyssey/edaprep/issues/43), by
  [Erol Tasci](https://github.com/Voyagerroc-Lab)) — both now round up
  (`ceil(threshold * n_rows)`); the EDA table previously truncated, so it could
  report "nothing rare here" for a level the planner would in fact group.
- **`TextColumnHandler`'s `length_features` no longer measures the string `"nan"`
  for missing text** ([#50](https://github.com/bijay-odyssey/edaprep/pull/50), closes
  [#45](https://github.com/bijay-odyssey/edaprep/issues/45), by
  [Erol Tasci](https://github.com/Voyagerroc-Lab)) — `astype(str)` on the raw column
  turned a missing value into the literal 3-character string it now excludes before
  measuring.
- **`_rule_impute`'s rationale now names `outlier_strategy='impute'`** as the trigger
  when imputation is planned for a column with genuinely zero missing values
  ([#49](https://github.com/bijay-odyssey/edaprep/pull/49), closes
  [#47](https://github.com/bijay-odyssey/edaprep/issues/47), by
  [Erol Tasci](https://github.com/Voyagerroc-Lab)), rather than reporting a bare,
  uninformative `0.0% missing`.

## [0.2.2] — 2026-09-15

### Fixed

- **`missing_fraction` in a decision's `params` now means the same thing across
  `drop_high_missing`, `missing_indicator` and `impute_by_type`: the fraction the
  decision was actually taken on, including placeholders `Stage.CAST` turns into
  `NaN`.** Previously all three stored the pre-cast `ColumnProfile.missing_fraction`
  while their rationale text quoted the post-cast figure — a decision reading "above
  the 5.0% flag threshold" in its rationale would report `missing_fraction=0.0` in
  `params`, the machine-readable half a plan diff actually compares. `cast_missing` is
  unchanged and still present alongside it; `n_rows` is now recorded too, so both the
  raw and effective fractions are reconstructible from a serialised plan without the
  profile alongside it.

  Centralised in a new `_post_cast_missing()` helper in `planning/rules.py`, matching
  the design [@luziyi123448-gif](https://github.com/luziyi123448-gif) put in
  [#17](https://github.com/bijay-odyssey/edaprep/pull/17), applied to all three rules
  and extended with `n_rows` as [#18](https://github.com/bijay-odyssey/edaprep/issues/18)
  asked. The rationale text is unchanged — this only corrects what `params` records.

## [0.2.1] — 2026-09-08

### Fixed

- **The missing-indicator and drop-high-missing rules now see placeholders that
  become `NaN` at the cast step.** [0.2.0] fixed this for the imputation rule
  (`_rule_impute`, [#12](https://github.com/bijay-odyssey/edaprep/pull/12)); the other
  two rules keyed on `missing_fraction` were still reading the profiler's figure, which
  is measured on the raw frame where a placeholder is still the string `''` (or `?`,
  `N/A`, …) rather than `NaN`. `Stage.CAST` converts those first, so by the time these
  rules run their input is stale.

  - `_rule_missing_indicator` — a column that is 8% blank strings now earns a
    missing-indicator column, where before it earned none.
  - `_rule_drop_high_missing` — a column that is 70% blank strings is now dropped,
    rather than imputed from the 30% that actually parsed.

  Placeholder counts come from the profiling sample, so when sampling is on the
  effective fraction is a lower bound: the rule can miss a threshold crossing but
  cannot manufacture one, which is the safe direction for a rule that deletes a
  column. Fixed by [@Jeferson681](https://github.com/Jeferson681) in
  [#16](https://github.com/bijay-odyssey/edaprep/pull/16), closing
  [#14](https://github.com/bijay-odyssey/edaprep/issues/14).

  `missing_fraction` in a decision's `params` still means the raw figure in these
  rules while the rationale quotes the effective one; making that consistent across
  all three rules is tracked in
  [#18](https://github.com/bijay-odyssey/edaprep/issues/18).

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

[0.3.0]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.3.0
[0.2.2]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.2.2
[0.2.1]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.2.1
[0.2.0]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.2.0
[0.1.0]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.1.0
