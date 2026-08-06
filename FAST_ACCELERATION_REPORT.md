# v6 Fast acceleration report

This release accelerates the strict `unit_local` implementation without changing
its primary kNN MI, granular-ball, bootstrap, LCB or budget-selection formulas.

## Exact-score optimizations

1. **Band-batched MI**
   - v5 called the kNN estimator once per frequency band inside every ball.
   - v6 flattens `[unit, band]` into independent feature columns and evaluates
     them in one call. The estimator is column-wise, so the result is unchanged.

2. **Nested-ball memoization**
   - Coarse, medium and fine granularities are nested and often contain the same
     balls. v5 recomputed MI for the same ball at every granularity.
   - v6 computes every unique ball once and reuses it in all snapshots.

3. **Vectorized per-unit standardization**
   - The same per-unit z-score transformation is now performed for all units in
     one NumPy operation.

4. **Compiled small-problem kNN MI**
   - Ball-level MI contains only a few features but is called millions of times.
   - When Numba is installed, v6 uses a compiled implementation with the same
     jitter, k-th same-class radius, global neighbor count and digamma formula.
   - It automatically falls back to the v5 NumPy implementation if Numba is not
     available. Disable with `--no-paper_fast_small_mi` for diagnostics.

5. **Chunked unit workers**
   - Each thread handles a block of units instead of scheduling 11,008 tiny
     futures. Results are written back by original unit index.

6. **Parallel LCB repeats with deterministic sampling**
   - All bootstrap indices are generated serially from the original random seed
     before parallel execution. Therefore the same repeated samples are used.
   - Total nested concurrency remains bounded by `paper_gb_workers`.

7. **No diagnostic hierarchy inside LCB repeats**
   - Diagnostic balls are only needed for figures and CSV summaries. v6 still
     rebuilds every scoring ball in every repeat, but does not build an extra
     unused layer-level diagnostic hierarchy.

8. **Task-band relevance cache**
   - MI for a previously evaluated frequency interval is reused during adjacent
     band merging.

## Auxiliary KDE acceleration

`--paper_kde_scope probe` evaluates granular KDE on the representative probe
units used by the task-driven frequency merge. KDE is an auxiliary consistency
check and is never used by MI, GB, LCB, the pruning mask or PPL. Therefore this
option reduces diagnostic cost without changing pruning performance.

Use `--paper_kde_scope all` when a full per-unit KDE table is explicitly needed.

## Validation

- Project tests: 12 passed.
- v5/v6 synthetic score comparison:
  - MI maximum absolute difference: 0
  - GB maximum absolute difference: approximately 1e-15
  - LCB maximum absolute difference: approximately 1e-15
- A 96-observation, 512-unit benchmark that did not finish in six minutes under
  v5 completed in about 19 seconds under v6 on the build machine. Actual server
  speed depends on CPU count, memory bandwidth, ball count and event balance.

The tiny floating differences come from equivalent batched summation order. They
should not change masks unless two units are tied at machine precision.
