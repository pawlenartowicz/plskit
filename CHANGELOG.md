# Changelog

All notable changes to this project will be documented here.

## [0.4.0] - 2026-08-03

- Changed: `split_perm` and `split_perm_nr` are merged into one method,
  `split_exact` — the permutation-calibrated split-half test. The engine
  picks the no-refit route at K = 1 (dense input) or the refit route
  otherwise; there is no route knob, and `split_perm` / `split_perm_nr`
  are no longer valid `method` values. This changes the refit route's
  numbers versus 0.3.0's `split_perm`: splits are now drawn once and held
  fixed across permutation replicates instead of redrawn per replicate —
  redrawing folded split-to-split scatter into the null and miscalibrated
  `split_perm`'s p-values — and the reported statistic moved from mean-r
  to `tanh(z̄)` to match. Re-running a 0.3.0 `split_perm` analysis under
  0.4.0 will produce different numbers. The no-refit route's numbers are
  unchanged from 0.3.0's `split_perm_nr` (bit-identical), but it now also
  accepts weighted input at K = 1, which `split_perm_nr` used to reject.
- Added: `split_nb` auto-gate. `split_nb`'s Fisher-z correction drifts
  off level when `n_eff < 25` or the stable rank of the standardized `X`
  is `< 3`. Stable rank can never exceed the column count, so `X` with 4
  columns or fewer is rerouted outright without consulting the computed
  rank. A request that trips any of the three clauses now reroutes to
  `split_exact` (at `n_perm=1000`) and `result.method` reports the
  method actually run. Pass `args={'force': True}` to run `split_nb`
  anyway. Python raises a `UserWarning` when a reroute happens.
- Added: `stable_rank` on `ConfirmatoryTestResult`, `FindKOptimalResult`
  and `FindKSequenceResult` — the stable rank the `split_nb` auto-gate
  saw, populated whenever `split_nb` was requested (fired or not,
  including under `force`); `None` for every other method. On the two
  `find_k` results it is the sequence-level gate's value, read off the
  undeflated `X`, so a rerouted run can say which clause fired.
- Added: `split_nb_gate(X, weights=None)` — ask whether the auto-gate
  flags a design without running a test. Returns `fires`, `stable_rank`
  and `n_eff`. It evaluates the same rule the test functions apply
  internally, so it cannot drift from them.
- Changed: the recommended default for `pls1_confirmatory_test` is now
  K = 1 with `method="split_exact"` (previously `split_nb`).

## [0.3.0] - 2026-07-31

- Added: `split_perm_nr` confirmatory test method — the same statistic as
  `split_nb` (mean Fisher-z of held-out correlations, reported as
  `tanh(z̄)`), compared against a permutation reference instead of the t
  approximation. K = 1 and unweighted input only; raises rather than
  degrading on ineligible input.
- Added: `rho_hat` on `ConfirmatoryTestResult` — reported for the `split_nb`
  arm (`None` for every other method, and `None` for `split_nb` itself when
  the input is weighted or the test half is too small).
- Changed: `pls1_find_k_optimal` and `pls1_find_k_sequence` name the offending
  method when it has no sequential variant, instead of always reporting
  `score`. `split_perm_nr` is the second such method.
- Fixed: pinned `time` to 0.3.41 and `deflate64` to 0.1.9 in `Cargo.lock`.
  Dependency bumps had pulled in `time-core` 0.1.8 (requires rustc 1.88) and
  `deflate64` 0.1.12 (uses `unbounded_shifts`, stable in 1.87), both past the
  declared MSRV of 1.85. Dev-dependency-only, reached via `ndarray-npy` →
  `zip`; nothing in the published crate or wheel is affected.

## [0.2.0] - 2026-06-12

- Added: `spls1_*` sparse PLS1 family (`spls1_fit`, `spls1_find_keep_optimal`, `spls1_find_k_optimal`, `spls1_find_k_sequence`) — hard keep-count NIPALS selection with dense bit-parity at `keep = n_features`.
- Changed (numerical — outputs shift from 0.1.0; pin the version for
  reproducibility):
  - `pls1_find_k_sequence` now standardizes and deflates with the supplied
    observation weights (0.1.0 ran the incremental steps unweighted and only
    weighted the final per-step test).
  - universal-inference e-value (`method="e"`) fixes σ²_alt on the training
    half instead of the test half.
  - χ²(df) survival function computed via the upper incomplete gamma directly
    (no 1 − lower complement round-trip).
  - subsample subspace leverage computed without Procrustes alignment (the
    hat matrix is rotation-invariant).
  - varimax final `w_rot` matmul via faer (`Par::Seq`) instead of a scalar loop.
  - rotation-stability paired-bootstrap seed derived from the parent RNG
    stream instead of a fixed `0xB007` offset.
- R and Julia wrappers remain at 0.0.1 (no `spls1` surface yet).

## [0.1.0] - 2026-05-09

Initial release of plskit. PLS1 with modern inference (canonical
percentile CIs, `split_nb`, `split_perm`). Ships the Rust engine
(`plskit` on crates.io) and the Python wrapper (`plskit` on PyPI);
R and Julia wrappers are not yet at feature parity and ship at a
lower version.
