"""Self-contained HTML rendering for a :class:`~edaprep.reporting.report.Report`.

No external assets: no CDN scripts, no web fonts, no images.  A report is often read
from a machine with no network access, and a report that renders differently depending
on connectivity is not a record.
"""

from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from .report import Report

__all__ = ["render_html"]

_CSS = """
:root { --fg:#1a1a1a; --muted:#666; --bg:#fff; --line:#e3e3e3;
        --warn:#8a5a00; --err:#a11; --accent:#2b5797; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e6e6; --muted:#9a9a9a; --bg:#161616; --line:#333;
          --warn:#e0a63a; --err:#e57373; --accent:#7aa7e8; }
}
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:1000px; margin:0 auto; }
h1 { font-size:1.6rem; margin:0 0 .25rem; }
h2 { font-size:1.1rem; margin:2rem 0 .6rem; padding-bottom:.3rem;
     border-bottom:1px solid var(--line); }
.sub { color:var(--muted); font-size:.85rem; margin-bottom:1.5rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; }
.card { border:1px solid var(--line); border-radius:6px; padding:.75rem; }
.card .n { font-size:1.4rem; font-weight:600; }
.card .l { color:var(--muted); font-size:.78rem; text-transform:uppercase;
           letter-spacing:.04em; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.87rem; }
th,td { text-align:left; padding:.35rem .6rem; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:.78rem; text-transform:uppercase;
     letter-spacing:.03em; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em; }
.w { padding:.5rem .7rem; border-left:3px solid var(--line); margin:.4rem 0; }
.w.warning { border-color:var(--warn); }
.w.error { border-color:var(--err); }
.w.info { border-color:var(--accent); }
.tag { display:inline-block; padding:.05rem .4rem; border-radius:3px; font-size:.75rem;
       border:1px solid var(--line); color:var(--muted); }
.override { color:var(--accent); font-weight:600; }
.muted { color:var(--muted); }
"""


def _e(value: object) -> str:
    return _html.escape(str(value))


def _card(number: str, label: str) -> str:
    return (
        f"<div class='card'><div class='n'>{_e(number)}</div>"
        f"<div class='l'>{_e(label)}</div></div>"
    )


def render_html(report: "Report") -> str:
    """Render a full report as one self-contained HTML page."""
    parts: List[str] = []
    add = parts.append

    add("<title>edaprep report</title>")
    add(f"<style>{_CSS}</style>")
    add("<main>")
    add("<h1>edaprep report</h1>")
    add(
        f"<div class='sub'>generated {_e(report.created_at)} &middot; "
        f"edaprep {_e(report.environment.get('edaprep', ''))} &middot; "
        f"python {_e(report.environment.get('python', ''))}</div>"
    )

    profile = report.profile
    add("<div class='grid'>")
    if profile is not None:
        add(_card(f"{profile.n_rows:,}", "rows"))
        add(_card(f"{profile.n_columns:,}", "columns"))
        add(_card(f"{profile.missing_fraction:.1%}", "missing cells"))
        add(_card(f"{profile.n_duplicate_rows:,}", "duplicate rows"))
    add(_card(f"{len(report.feature_names_in)}", "features in"))
    add(_card(f"{len(report.feature_names_out)}", "features out"))
    add("</div>")

    if report.dropped_columns:
        add("<h2>Removed columns</h2><div class='scroll'><table>")
        add("<tr><th>column</th><th>reason</th></tr>")
        for column, reason in report.dropped_columns.items():
            add(f"<tr><td><code>{_e(column)}</code></td><td>{_e(reason)}</td></tr>")
        add("</table></div>")

    if report.plan is not None and report.plan.decisions:
        add("<h2>Decisions</h2><div class='scroll'><table>")
        add(
            "<tr><th>column</th><th>stage</th><th>action</th>"
            "<th>why</th><th>source</th></tr>"
        )
        for d in sorted(report.plan.decisions, key=lambda d: (d.column, d.stage.order)):
            source = (
                "<span class='override'>user override</span>"
                if d.is_override
                else f"<span class='muted'>{_e(d.rule)}</span>"
            )
            add(
                f"<tr><td><code>{_e(d.column)}</code></td>"
                f"<td><span class='tag'>{_e(d.stage)}</span></td>"
                f"<td><code>{_e(d.action)}</code></td>"
                f"<td>{_e(d.rationale)}</td><td>{source}</td></tr>"
            )
        add("</table></div>")

    leakage = report.leakage
    add("<h2>Leakage audit</h2>")
    add(
        "<div class='w info'>All statistics were learned during <code>fit</code>, on "
        "the training frame only. <code>transform</code> is a pure function of that "
        "fitted state.</div>"
    )
    if leakage["transformers_using_target"]:
        add(
            f"<div class='w info'>Transformers that read the target: "
            f"<code>{_e(', '.join(leakage['transformers_using_target']))}</code>. "
            f"Cross-fitted: <strong>{_e(leakage['cross_fitted'])}</strong>.</div>"
        )
    for column in leakage["columns_suspected_of_leakage"]:
        add(
            f"<div class='w error'>Column <code>{_e(column)}</code> is almost "
            f"perfectly associated with the target. Investigate before modelling.</div>"
        )

    if report.warnings:
        add("<h2>Warnings</h2>")
        for warning in sorted(report.warnings, key=lambda w: -w.severity.rank):
            add(f"<div class='w {_e(warning.severity)}'>{_e(warning.message)}</div>")

    transform_entries = [e for e in report.entries if e.phase == "transform"]
    if transform_entries:
        add("<h2>What happened</h2><div class='scroll'><table>")
        add("<tr><th>stage</th><th>transformer</th><th>action</th><th>effect</th></tr>")
        for entry in transform_entries:
            effect = ", ".join(
                f"{k}={v}" for k, v in entry.effect.items() if not isinstance(v, dict)
            )
            add(
                f"<tr><td><span class='tag'>{_e(entry.stage)}</span></td>"
                f"<td>{_e(entry.transformer)}</td>"
                f"<td><code>{_e(entry.action)}</code></td>"
                f"<td class='muted'>{_e(effect)}</td></tr>"
            )
        add("</table></div>")

    if report.config is not None:
        add("<h2>Reproducibility</h2><div class='scroll'><table>")
        add(f"<tr><th>random_state</th><td>{_e(report.config.random_state)}</td></tr>")
        add(f"<tr><th>model_family</th><td>{_e(report.config.model_family)}</td></tr>")
        if profile is not None and profile.sampling.get("used"):
            s = profile.sampling
            add(
                f"<tr><th>profiling sample</th><td>{s['n']:,} of {s['of']:,} rows "
                f"(seed {_e(s.get('random_state'))})</td></tr>"
            )
        add("</table></div>")

    add("</main>")
    return "\n".join(parts)
