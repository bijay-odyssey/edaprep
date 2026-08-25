# Security policy

## Supported versions

`edaprep` is pre-1.0. Only the latest release receives fixes.

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please report privately rather than in a public issue:

- GitHub [private vulnerability reporting](https://github.com/bijay-odyssey/edaprep/security/advisories/new) (preferred), or
- email **bijaybeezoe@gmail.com**

You can expect an acknowledgement within a few days. As a single-maintainer project
there is no formal SLA, but reports are taken seriously and you will be credited unless
you prefer otherwise.

## Scope

`edaprep` is a data-processing library. It does not open sockets, execute user-supplied
code, or deserialise untrusted formats. Realistic issues are therefore things like:

- a crafted input frame causing unbounded memory growth or a hang;
- a path traversal or overwrite via a `to_html(path=...)` / `to_json(path=...)` argument;
- sensitive values leaking into a report that a user then shares.

That last one is worth stating plainly: **reports embed data**. `report.to_dict()`,
`to_json()` and `to_html()` include column names, modal values, quantiles and example
categories. Treat a generated report with the same care as the dataset it describes.

## Not vulnerabilities

- Statistical or numerical disagreements — please open a normal issue; several are
  deliberate and documented in `docs/performance.md`.
- Dependency advisories with no reachable path from this library's API. Report those to
  the dependency.
