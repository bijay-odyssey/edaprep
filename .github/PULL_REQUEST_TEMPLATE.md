## What this changes

<!-- One or two sentences. The "why" matters more than the "what". -->

Closes #

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check src/ tests/ benchmarks/ examples/` passes
- [ ] Added a test for the behaviour that changed
- [ ] If this fixes a bug: I confirmed the new test **fails without the fix**

## If you touched a transformer

- [ ] `_transform` computes no statistic over its input (the AST test enforces this)
- [ ] The input frame is not mutated
- [ ] Effects are recorded via `context.journal`

## If you added a decision threshold

- [ ] It lives in `Thresholds`, named, with a comment saying where the number came from

## If you are claiming a performance improvement

- [ ] Benchmark numbers included, using the method in `docs/performance.md` §5
      (minimum of N runs, interleaved candidates, like-for-like baseline)
