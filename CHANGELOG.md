# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is `0.x`, the public API may change between minor versions; anything that does
will be listed under **Changed** with a migration note.

## [0.1.0] — 2026-08-22

First public release.

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

[0.1.0]: https://github.com/bijay-odyssey/edaprep/releases/tag/v0.1.0
