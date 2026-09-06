# Changelog

All notable changes to this project will be documented here.

## [0.5.0] - 2026-09-06

- Added: `pls3_fit` (alias `plssvd_fit`) — SVD-PLS / PLSC. One SVD of the
  standardized cross-covariance `X'Y`, no deflation, so all `k ≤ min(p, q)`
  components come out of a single decomposition and are orthogonal by
  construction. The SVD acts on a `p × q` matrix and never on a `p × p`
  one, so `p ≫ n` is the ordinary case for this family. Salience signs are
  pinned — the largest-magnitude entry of each `U` column is positive and
  the matching `V` column flips with it — so repeated and cross-platform
  fits agree.
- Added: `pls3_transform` (alias `plssvd_transform`) — project new X and/or
  Y onto a fitted PLS3's latent-variable scores, selected by
  `which="x_scores" | "y_scores" | "both"`. There is no `pls3_predict`:
  PLS3 is symmetric, so neither block is the outcome.
- Added: `pls3_confirmatory_test` — omnibus test at LV1 on the held-out
  latent-variable correlation `r = cor(X_te @ u1, Y_te @ v1)`, Fisher-z
  averaged across splits and reported as `tanh(z_bar)`. Two methods over
  that one statistic: `method="split_exact"` calibrates it by permuting the
  rows of Y against X with the splits held fixed, and `method="split_nb"`
  compares it against a t reference instead, costing `n_splits` fits in
  total rather than `n_perm * n_splits`. `split_exact` is the
  recommendation — it holds its level on any design. Both sides of the
  correlation are estimated on the training half, but that costs the t
  reference nothing: conditional on the training half the two held-out
  score vectors are fixed linear combinations of independent test-half
  rows, so under the null `r` follows the ordinary null correlation law,
  and on Gaussian, heavy-tailed, low-stable-rank and real two-block designs
  `split_nb` measured conservative rather than anti-conservative.
  `raw_perm` needs a CV statistic a method without `predict` does not have,
  and `score` and `e` have no symmetric formulation. `k=1` only: above LV1
  the training-half component ordering need not survive to the test half.
- Added: the `split_nb` auto-gate applies to `pls3_confirmatory_test` on
  the same terms as `pls1_confirmatory_test`, on X only. A flagged design
  runs `split_exact` instead, `result.method` says so, Python warns, and
  `args={"force": True}` overrides. Y never enters the gate: `q` is small
  by construction in PLSC, so a stable-rank floor on Y would flag almost
  every design. The thresholds are the PLS1 ones and have not been
  re-derived for a two-block design.
- Added: `PLS3Result` and `PLS3Scores` result objects. `pls3_confirmatory_test`
  reuses `ConfirmatoryTestResult`, with `ci` always `None` and `n_eff` equal
  to `n`. `rho_hat` is populated for `split_nb` only, `stable_rank` whenever
  `split_nb` was requested, and `n_perm` is `None` for `split_nb`.
- Added: six `pls3_*` reference-corpus cases (`testdata/`), regenerated
  from the Rust core.
- Note: observation weights are not implemented anywhere in the PLS3
  family. `pls3_fit(weights=...)` raises rather than silently ignoring the
  argument.
- Changed: `pls1_confirmatory_test(method="raw_perm")` at `k=1` and
  `pls3_confirmatory_test` now pick between two execution routes internally.
  The new Gram route builds the training Gram `X_tr X_tr'` and the
  train-to-test map `X_te X_tr'` once per fold or split, so each
  permutation replicate afterwards costs nothing in the feature count — the
  win is concentrated where `p` is large and `n` is small (voxel- or
  embedding-width data at small sample size). The route is chosen from
  `(n_tr, p, n_perm, q)` alone and is taken only when it is estimated
  faster; there is no knob and nothing is reported on the result, because it
  is an execution strategy and not a statistic. Both routes see the same
  splits, folds and permutations at the same seed, and every replicate is
  checked against an honest refit, agreeing within the project's numerical
  tolerance. Weighted input, `k >= 2` and sparse `keep` always take the
  original route.

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

## [0.2.1] - 2026-06-21

- Fixed: the `coverage_mc` test's oracle for `leverage_ci_*` coverage now
  estimates the population value of the same finite-n estimand each
  per-dataset CI targets — Monte Carlo over `N_ORACLE = 200` freshly drawn
  size-`n` datasets run through the identical engine path — instead of a
  single asymptotic 50,000-row fit. Per-coordinate leverage coverage is no
  longer asserted above `k = 1` (the synthetic DGP's true signal rank): the
  centered-scaled leverage CI is anti-conservative outside the low-`d`/
  large-`n` regime (measured between-dataset SD / reported SE ≈ 1.2, rising
  to ≈ 2.1 at `d=20, n=100`), so those numbers are now printed for
  diagnostic monitoring only. `holdout_corr` remains the sole asserted
  calibration guarantee. Test-only; no change to library behavior.

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
