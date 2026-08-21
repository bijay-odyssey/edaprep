"""Benchmark suite.

Measures wall time and peak memory for profiling, each transformer, and end-to-end
pipelines, against hand-written pandas and scikit-learn baselines.

Method
------
Time is the *minimum* of ``repeat`` runs, not the mean.  The minimum estimates the
cost of the work; the mean estimates the cost of the work plus whatever else the
machine was doing, and on a laptop that is mostly noise.

Memory is peak *additional* allocation measured with ``tracemalloc``, which counts
Python-level allocations including NumPy buffers created through the Python API.  It
does not see allocations made inside C libraries that bypass the Python allocator, so
treat the numbers as comparative rather than absolute.

Run::

    python benchmarks/bench.py                    # default sizes
    python benchmarks/bench.py --rows 200000      # bigger
    python benchmarks/bench.py --json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import edaprep
from edaprep.config import Config
from edaprep.core.context import FitContext
from edaprep.preprocessing import (
    CategoricalEncoder,
    MissingValueHandler,
    OneHotEncoder,
    OutlierHandler,
    Scaler,
    detect_outliers,
)
from edaprep.profiling import profile


@dataclass
class Result:
    name: str
    group: str
    seconds: float
    peak_mib: float
    rows: int
    columns: int
    baseline_seconds: Optional[float] = None
    baseline_peak_mib: Optional[float] = None
    baseline_name: Optional[str] = None
    notes: str = ""

    @property
    def speedup(self) -> Optional[float]:
        if not self.baseline_seconds or not self.seconds:
            return None
        return self.baseline_seconds / self.seconds

    def render(self) -> str:
        rate = f"{self.rows / self.seconds / 1e6:.2f}M rows/s" if self.seconds else "-"
        line = (
            f"  {self.name:<38.38} {self.seconds * 1000:>9.1f} ms "
            f"{self.peak_mib:>8.1f} MiB  {rate:>14}"
        )
        if self.baseline_seconds:
            ratio = self.speedup or 0.0
            verdict = "faster" if ratio >= 1 else "slower"
            line += (
                f"\n  {'  vs ' + (self.baseline_name or 'baseline'):<38.38} "
                f"{self.baseline_seconds * 1000:>9.1f} ms "
                f"{(self.baseline_peak_mib or 0):>8.1f} MiB  "
                f"{max(ratio, 1 / ratio if ratio else 0):>8.2f}x {verdict}"
            )
        return line


def measure(fn: Callable[[], object], repeat: int = 3) -> tuple:
    """Return ``(min_seconds, peak_mib)`` for ``fn``."""
    gc.collect()
    timings: List[float] = []
    peak_bytes = 0
    for index in range(repeat):
        gc.collect()
        if index == 0:
            tracemalloc.start()
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if index == 0:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        del result
        timings.append(elapsed)
    return min(timings), peak_bytes / (1024 * 1024)


def make_frame(rows: int, seed: int = 0) -> pd.DataFrame:
    """A frame with the shape of notebook practice's datasets."""
    gen = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "age": np.where(gen.random(rows) < 0.08, np.nan, gen.normal(40, 12, rows)),
            "income": gen.lognormal(10, 1.1, rows),
            "balance": np.where(gen.random(rows) < 0.3, 0.0, gen.normal(0, 500, rows)),
            "score": gen.normal(600, 80, rows),
            "ratio": gen.beta(2, 5, rows),
            "tenure": gen.integers(0, 120, rows).astype(float),
            "city": gen.choice([f"city_{i:04d}" for i in range(500)], rows),
            "segment": gen.choice(
                ["a", "b", "c", "d", "e"], rows, p=[0.4, 0.3, 0.15, 0.1, 0.05]
            ),
            "channel": gen.choice(["web", "app", "branch"], rows),
            "country": gen.choice([f"c{i}" for i in range(30)], rows),
            "active": gen.choice([True, False], rows),
            "signup": pd.to_datetime("2019-01-01")
            + pd.to_timedelta(gen.integers(0, 2000, rows), unit="D"),
            "target": (gen.random(rows) < 0.25).astype(int),
        }
    )


def make_wide_frame(rows: int, columns: int, seed: int = 1) -> pd.DataFrame:
    """A wide numeric frame: the profiler's worst case, and the IEEE frame's shape."""
    gen = np.random.default_rng(seed)
    data = {f"v{i}": gen.normal(size=rows) for i in range(columns)}
    for i in range(0, columns, 10):
        column = data[f"v{i}"]
        column[gen.random(rows) < 0.1] = np.nan
    data["target"] = (gen.random(rows) < 0.3).astype(int)
    return pd.DataFrame(data)


def _context(frame: pd.DataFrame, target: str = "target") -> FitContext:
    config = Config(random_state=0)
    return FitContext(
        config=config, profile=profile(frame, target=target, config=config), target=target
    )


# ============================== benchmark cases ========================================


def bench_profiling(rows: int, repeat: int) -> List[Result]:
    results: List[Result] = []
    frame = make_frame(rows)

    seconds, peak = measure(lambda: profile(frame, target="target"), repeat)
    # The nearest hand-written equivalent from notebook practice: the statistics its EDA
    # notebooks actually compute, column by column.
    def baseline() -> Dict[str, object]:
        out: Dict[str, object] = {
            "shape": frame.shape,
            "dtypes": frame.dtypes.to_dict(),
            "missing": frame.isnull().sum().to_dict(),
            "duplicates": int(frame.duplicated().sum()),
            "describe": frame.describe(include="all"),
        }
        numeric = frame.select_dtypes(include=["int64", "float64"]).columns
        out["skew"] = {c: frame[c].skew() for c in numeric}
        out["kurt"] = {c: frame[c].kurt() for c in numeric}
        out["nunique"] = {c: frame[c].nunique() for c in frame.columns}
        return out

    base_seconds, base_peak = measure(baseline, repeat)
    results.append(
        Result(
            "profile(df, target=...)",
            "profiling",
            seconds,
            peak,
            rows,
            frame.shape[1],
            base_seconds,
            base_peak,
            "hand-written pandas EDA block",
            notes="edaprep additionally infers semantic types and runs quality checks",
        )
    )

    seconds, peak = measure(
        lambda: profile(frame, target="target", compute_moments=False, check_quality=False),
        repeat,
    )
    results.append(
        Result("profile(quick)", "profiling", seconds, peak, rows, frame.shape[1])
    )
    return results


def bench_wide_profiling(rows: int, columns: int, repeat: int) -> List[Result]:
    frame = make_wide_frame(rows, columns)
    results: List[Result] = []

    seconds, peak = measure(lambda: profile(frame, target="target"), repeat)
    results.append(
        Result(f"profile ({columns} columns)", "profiling", seconds, peak, rows, columns)
    )

    numeric = [c for c in frame.columns if c != "target"]
    from edaprep.profiling.statistics import numeric_block_stats

    seconds, peak = measure(lambda: numeric_block_stats(frame[numeric]), repeat)

    def per_column() -> Dict[str, tuple]:
        """The same statistic set edaprep produces, written the obvious way.

        Including the zero/negative/missing counts and the MAD matters: without them
        the baseline is doing strictly less work and the comparison is not a
        comparison.
        """
        out = {}
        for c in numeric:
            column = frame[c]
            median = column.median()
            out[c] = (
                column.mean(),
                column.std(),
                column.min(),
                column.max(),
                column.skew(),
                column.kurt(),
                column.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
                int(column.isna().sum()),
                int((column == 0).sum()),
                int((column < 0).sum()),
                (column - median).abs().median(),
            )
        return out

    base_seconds, base_peak = measure(per_column, repeat)
    results.append(
        Result(
            f"numeric_block_stats ({columns} cols)",
            "profiling",
            seconds,
            peak,
            rows,
            columns,
            base_seconds,
            base_peak,
            "per-column pandas dispatch",
        )
    )
    return results


def bench_transformers(rows: int, repeat: int) -> List[Result]:
    frame = make_frame(rows)
    context = _context(frame)
    y = frame["target"]
    results: List[Result] = []

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder as SkOneHot
    from sklearn.preprocessing import StandardScaler

    numeric = ["age", "income", "balance", "score", "ratio", "tenure"]

    seconds, peak = measure(
        lambda: Scaler(numeric, strategy="standard").fit_transform(frame, y, context), repeat
    )
    base_seconds, base_peak = measure(
        lambda: StandardScaler().fit_transform(frame[numeric]), repeat
    )
    results.append(
        Result(
            "Scaler (standard, 6 cols)",
            "transformers",
            seconds,
            peak,
            rows,
            6,
            base_seconds,
            base_peak,
            "sklearn StandardScaler",
            notes="edaprep returns a labelled DataFrame; sklearn returns a bare array",
        )
    )

    seconds, peak = measure(
        lambda: MissingValueHandler(["age"], strategy="median").fit_transform(
            frame, y, context
        ),
        repeat,
    )
    base_seconds, base_peak = measure(
        lambda: SimpleImputer(strategy="median").fit_transform(frame[["age"]]), repeat
    )
    results.append(
        Result(
            "MissingValueHandler (median)",
            "transformers",
            seconds,
            peak,
            rows,
            1,
            base_seconds,
            base_peak,
            "sklearn SimpleImputer",
        )
    )

    seconds, peak = measure(
        lambda: OneHotEncoder(["segment", "channel"]).fit_transform(frame, y, context),
        repeat,
    )
    base_seconds, base_peak = measure(
        lambda: SkOneHot(sparse_output=False).fit_transform(frame[["segment", "channel"]]),
        repeat,
    )
    results.append(
        Result(
            "OneHotEncoder (8 levels)",
            "transformers",
            seconds,
            peak,
            rows,
            2,
            base_seconds,
            base_peak,
            "sklearn OneHotEncoder (dense)",
        )
    )

    seconds, peak = measure(
        lambda: CategoricalEncoder(["city"], strategy="target").fit_transform(
            frame, y, context
        ),
        repeat,
    )
    results.append(
        Result(
            "TargetEncoder (500 levels, 5-fold)",
            "transformers",
            seconds,
            peak,
            rows,
            1,
            notes="cross-fitted; no sklearn baseline is equivalent",
        )
    )

    seconds, peak = measure(
        lambda: OutlierHandler(numeric, method="iqr", strategy="clip").fit_transform(
            frame, y, context
        ),
        repeat,
    )

    def iqr_baseline() -> pd.DataFrame:
        out = frame.copy()
        for column in numeric:
            q1, q3 = out[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            out[column] = out[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return out

    base_seconds, base_peak = measure(iqr_baseline, repeat)
    results.append(
        Result(
            "OutlierHandler (IQR clip, 6 cols)",
            "transformers",
            seconds,
            peak,
            rows,
            6,
            base_seconds,
            base_peak,
            "the usual IQR block",
        )
    )

    seconds, peak = measure(lambda: detect_outliers(frame["income"], "iqr"), repeat)
    results.append(Result("detect_outliers (1 col)", "transformers", seconds, peak, rows, 1))
    return results


def bench_pipelines(rows: int, repeat: int) -> List[Result]:
    frame = make_frame(rows)
    results: List[Result] = []

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import OneHotEncoder as SkOneHot
    from sklearn.preprocessing import StandardScaler

    for family in ("tree", "linear"):
        seconds, peak = measure(
            lambda f=family: edaprep.AutoPipeline(
                target="target", model_family=f, random_state=0
            ).fit_transform(frame),
            repeat,
        )
        results.append(
            Result(
                f"AutoPipeline.fit_transform ({family})",
                "pipelines",
                seconds,
                peak,
                rows,
                frame.shape[1],
                notes="includes profiling, planning and reporting",
            )
        )

    pipe = edaprep.AutoPipeline(target="target", model_family="linear", random_state=0)
    pipe.fit(frame)
    seconds, peak = measure(lambda: pipe.transform(frame), repeat)

    numeric = ["age", "income", "balance", "score", "ratio", "tenure"]
    categorical = ["segment", "channel", "country"]
    sk = SkPipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            SkPipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                        (
                            "cat",
                            SkPipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "encode",
                                        SkOneHot(
                                            handle_unknown="ignore", sparse_output=False
                                        ),
                                    ),
                                ]
                            ),
                            categorical,
                        ),
                    ]
                ),
            )
        ]
    )
    sk.fit(frame, frame["target"])
    base_seconds, base_peak = measure(lambda: sk.transform(frame), repeat)
    results.append(
        Result(
            "AutoPipeline.transform (fitted)",
            "pipelines",
            seconds,
            peak,
            rows,
            frame.shape[1],
            base_seconds,
            base_peak,
            "sklearn ColumnTransformer",
            notes="edaprep also expands datetimes, flags missingness and journals effects",
        )
    )
    return results


def bench_eda(rows: int, repeat: int) -> List[Result]:
    frame = make_frame(rows)
    results: List[Result] = []
    for level in ("quick", "standard", "deep"):
        seconds, peak = measure(
            lambda lv=level: edaprep.EDA(frame, target="target").analyze(lv), repeat
        )
        results.append(
            Result(f"EDA.analyze({level!r})", "eda", seconds, peak, rows, frame.shape[1])
        )
    return results


# ============================== runner ==================================================


@dataclass
class Suite:
    rows: int
    wide_rows: int
    wide_columns: int
    repeat: int
    results: List[Result] = field(default_factory=list)

    def run(self) -> "Suite":
        self.results.extend(bench_profiling(self.rows, self.repeat))
        self.results.extend(
            bench_wide_profiling(self.wide_rows, self.wide_columns, self.repeat)
        )
        self.results.extend(bench_transformers(self.rows, self.repeat))
        self.results.extend(bench_pipelines(self.rows, self.repeat))
        self.results.extend(bench_eda(self.rows, self.repeat))
        return self

    def render(self) -> str:
        lines = [
            "=" * 84,
            "edaprep benchmarks",
            "=" * 84,
            f"  python {sys.version.split()[0]} | numpy {np.__version__} | "
            f"pandas {pd.__version__}",
            f"  {platform.platform()}",
            f"  rows={self.rows:,}  wide={self.wide_rows:,}x{self.wide_columns}  "
            f"repeat={self.repeat} (minimum reported)",
        ]
        for group in ("profiling", "transformers", "pipelines", "eda"):
            rows = [r for r in self.results if r.group == group]
            if not rows:
                continue
            lines.append("")
            lines.append(group.upper())
            for result in rows:
                lines.append(result.render())
                if result.notes:
                    lines.append(f"      note: {result.notes}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
                "edaprep": edaprep.__version__,
            },
            "settings": {
                "rows": self.rows,
                "wide_rows": self.wide_rows,
                "wide_columns": self.wide_columns,
                "repeat": self.repeat,
            },
            "results": [asdict(r) for r in self.results],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="edaprep benchmarks")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--wide-rows", type=int, default=20_000)
    parser.add_argument("--wide-columns", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    suite = Suite(args.rows, args.wide_rows, args.wide_columns, args.repeat).run()
    print(suite.render())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(suite.to_dict(), handle, indent=2)
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
