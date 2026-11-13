# Changelog

All notable changes to this project will be documented here.

## [0.9.0] - 2026-11-11

- Added: `covariates` on `pls1_fit`, `spls1_fit`, `pls1_predict`, `pls1_confirmatory_test`, `pls1_perm_null`, the PLS1 and sPLS1 k-selection functions, `factors` and `rotate`. X is standardized, then X and y are replaced by their (weighted) least-squares residuals on `[1, Z]`; the permutation tests of `split_exact` and `raw_perm` use Freedman–Lane null columns (approximately exact, not exact), and `"auto"` runs `split_exact`. `split_nb`, `score` and `e` refuse covariates; `profiles` and `profiles_search` refuse a model fitted with them. Aliased columns of Z are dropped, as `lm` does. Results gain `n_covariates`, `PLS1Result` gains `covariate_coef` and `FactorsResult` gains `covariate_adjusted`; a prediction is `intercept + X_new·beta + Z_new·covariate_coef`. The new error code `insufficient_residual_df` covers too few rows after the covariates are removed.
- Added: `null_y` on `pls1_confirmatory_test`: your own null outcomes (`n × B`) for `split_exact`, which replace its permutations. It refuses `covariates` and an explicit `n_perm`.
- Added: `gram_matrix(X, pre_standardized=False)` returns the all-row Gram matrix of `X` the held-out tests compute (`GramMatrixResult`: `gram`, `n_features`, `pre_standardized`), so it can be computed once and reused across calls. `result.subset(rows)` gives the Gram matrix of a row subsample on the same centring and scale.
- Added: `gram=` on `pls1_confirmatory_test` and `pls3_confirmatory_test`, in place of `X`: a `GramMatrixResult` from `gram_matrix`, or an `(n, n)` array with `n_features` (and `pre_standardized` / `pre_standardized_X` for how it was scaled). Every method runs from the Gram matrix, which the call centres over its rows (so `result.subset(rows)` is accepted) and checks on its entries; on `gram_matrix(X, ...)` the result equals the call on `X` at the same seed to rounding, `score` on X centred over all rows. `weights`, `ci` and `k` above 10 are refused with `gram=` (in Rust also `keep`, and `keep_x` below the column count). `X` and `y` / `Y` of both tests now default to `None`; `y` / `Y` stays required and exactly one of `X` and `gram` is passed. In Rust: `ConfirmatoryTestInput::Gram` and `pls3_confirmatory_test_gram`. In Julia, `pls1_confirmatory_test(X, y; ...)` and `pls3_confirmatory_test(X, Y; ...)` keep `X` and `y` / `Y` positional; a `gram=` call passes `y` / `Y` by keyword: `pls1_confirmatory_test(; gram=g, y=y)`.
- Added: `pls1_perm_alpha` takes a `GramMatrixResult` as `gram` (Rust: `pls1_perm_alpha_gram`), which it centres before checking its entries, so a subset and a matrix built under `pre_standardized` are accepted; an array still gets every check, and `n_features` is required only with an array.
- Added: `pls1_perm_alpha` serves `k` from 1 to 10 (was 1 only).
- Added: `split_nb_gate` takes `pre_standardized` (default `False`). With it, X is read as given and only centred, as the tests read it under the same flag, so the query answers for flagged calls too.
- Added: `pls1_confirmatory_test` takes a set of `k` (Python a list, a `range` or an integer array; R an integer vector; Julia an `AbstractVector{Int}`; Rust `k: Option<&[usize]>` on `ConfirmatoryTestInput::Raw` and `::Gram`) and asks whether X predicts y at some `k` in the set. `split_exact` and `raw_perm` draw their splits or folds once for the largest `k`, compute every `k`'s statistic from one fit per split or fold and column, and combine the per-`k` p-values by min-p (Westfall and Young), which keeps the permutation p exact. A one-element set gives the same p and statistic as a call at that `k`. A set is sorted; an empty set, a duplicate, a `k` of 0, or a set past a limit of the method is an error. `split_nb`, `score`, `e`, `ci=True` and (in Rust) `keep` refuse a set of more than one `k`. The result names no `k` to fit at.
- Added: `ConfirmatoryTestResult.pvalues_by_k` and `statistics_by_k` (Rust `ConfirmatoryTestOutput`, `BTreeMap<usize, f64>`): each tested `k`'s own p, not adjusted for the set, and its statistic. `pls3_confirmatory_test` fills them for its one `k`.
- Changed: with `weights`, `split_exact` and `split_nb` (in `pls1_confirmatory_test`, the steps of `pls1_find_k_sequence` / `spls1_find_k_sequence` and the `diagnostic` that runs them) subtract y's weighted mean over all rows before the test, and each `null_y` column its own weighted mean (the weighted statistic reads a column's location, so a null column must be centred as y is). They correlated the held-out predictions with the uncentred `√w·y`, so the farther y's mean was from 0, the less power they had. Results of weighted calls change. Without weights (and with `gram=`) they subtract the plain mean, which moves results in their last bits, except at k = 1 on a y whose mean is far from 0 next to its spread, where the test gets back precision it lost (about 3e-6 in the statistic at a mean of 1e11 spreads).
- Changed: with `weights`, the `split_nb` gate (in `split_nb`, `auto`, the steps that run them and `split_nb_gate`) reads X with the √w row scaling the tests apply, so a row of zero weight no longer counts toward its stable rank. A zero-weight row far from the rest made it fire (a stable rank of about 1 for one +20 row on 200 × 2000). `stable_rank` of weighted calls changes, and so can the gate's decision.
- Changed: the held-out tests (`split_exact`, `split_nb`, `e` and `raw_perm` in `pls1_confirmatory_test`, `pls3_confirmatory_test`, the steps of `pls1_find_k_sequence` / `spls1_find_k_sequence`, and the `diagnostic` of `pls1_find_k_optimal` / `spls1_find_k_optimal`, which runs that sequence) scale X (and Y in `pls3_confirmatory_test`) once over all rows, with weighted moments when weights are given, and only centre: each split or fold centres its training part with its own means and its held-out part with the training means; nothing is scaled per split or fold. The `split_nb` gate reads X under the same rule. Results change at the same seed.
- Changed: for these tests `pre_standardized`, `pre_standardized_X` and `pre_standardized_Y` mean "use as given": the one scaling over all rows is skipped and each part is still centred as above. `score`, and the full-data reference fit and the β and leverage resamples of the CI of `pls1_confirmatory_test`, still use X and y exactly as given under the flag. In `pls3_confirmatory_test` the two flags now take effect.
- Changed: the cross-validation folds of `pls1_find_k_optimal`, `spls1_find_k_optimal` and `spls1_find_keep_optimal` read X under the same rule (scaled once over all rows, or used as given under the flag) and only centre. In `pls1_find_k_sequence` / `spls1_find_k_sequence`, steps from 2 on test the deflated residual on step 1's scale. `split_exact` at k = 1 draws its null columns as the refit route does. Results change at the same seed.
- Changed: `split_exact`, `split_nb`, `pls3_confirmatory_test` and the sequence steps that run them draw their J splits from one balanced design over random cells of rows. When n ≥ H, the number of cells (n ≥ 64 at the default J = 32), every pair of splits overlaps in exact quarters of the cell rows; below that there are no cells and the splits are independent random halves. Each split is still a random half. Both routes of `split_exact` now test the same splits at a seed. `e` keeps its single random half. Results change at the same seed, and so does the CI of `pls1_confirmatory_test`, which draws after the test.
- Changed: the default `n_splits` is 32 (was 50); results change at the same seed.
- Changed: `test_method="auto"` keeps `split_exact` up to Kish `n_eff` 1000 at `k = 1` without `keep` (every step of `pls1_find_k_sequence`, and of `spls1_find_k_sequence` at `keep = n_features`); the floor stays 250 at `k ≥ 2` or with a sparse `keep`. A request with `n_eff` from 250 to 1000 at `k = 1` without `keep` ran `split_nb` and now runs `split_exact`; `n_perm` is set and `stable_rank` is `None` on it.
- Changed: `pls3_confirmatory_test` with `test_method="auto"` (the default) runs `split_exact` on every design. It ran `split_nb` when n ≥ 250, X had more than 4 columns and the `split_nb` gate did not fire; on those designs the result changes, `n_perm` is set and `stable_rank` is `None`. An explicit `split_nb` request is unchanged.
- Changed: the CI of `pls1_confirmatory_test` (`ci=True`) is beta, not to be trusted yet. It reads X and y as the held-out tests read X: scaled once over all rows (with weighted moments when weights are given; used as given under `pre_standardized`, divided by one power of two outside `2^-256 .. 2^256`), and each half split centred with its own means, the held-out rows with the training means; each subsample and bootstrap resample is centred with its own means too, except under `pre_standardized`, where it is used as given, as the reference fit is. `holdout_corr` reads the held-out y on the all-row scale too, so a y in small units (a standard deviation below about 1e-16) no longer makes every half split fail. It scaled each of them with its own moments, so a column that barely varied on a half's rows blew up its held-out rows (a column of 1e-44 noise with one row at 0.5 moved `holdout_corr.point` from 0.338 to 0.195 on 200 × 400). `beta` and its CI stay on the raw X scale. Results of `ci=True` change at the same seed.
- Changed: on wide X the held-out tests of `pls1_confirmatory_test` (`split_exact`, `split_nb`, `raw_perm`) and `pls1_perm_null` decide every column, truncation included, from the Gram matrix of X, with no refit on X. A column whose outcome is orthogonal to X up to rounding, or whose fit asks for components past the rank of X, can get a different value than an X refit gives, since the Gram matrix resolves `‖X̃'z‖` only to its own rounding. The Gram matrix's precision follows the largest column of X: with `pre_standardized` and columns of very different scale, or under any scaling with a row far from the rest (about 1e9 of the spread or more), other columns can lose precision too, by up to O(1); elsewhere other columns are unchanged. The Gram route now serves up to k = 10 components (was 2); past that, wide fits refit on X.
- Changed: X with entries near `±f64::MAX` of both signs is centred without overflow wherever it is standardized: in the fits, in `score`, `e`, every route of `split_exact`, `split_nb` and `raw_perm` in `pls1_confirmatory_test`, in `pls3_confirmatory_test` and in the cross-validation folds. Results are those of the same X divided by 8. 0.8.0 raised `non_finite_input` in the fits, `score`, `e` and `raw_perm` and scored an overflowing split as r = 0. The `split_nb` gate reads a finite stable rank there, so a NaN no longer makes it fire.
- Changed: under `pre_standardized` (`pre_standardized_X` / `pre_standardized_Y` in PLS3), the held-out tests, the cross-validation folds of the k-selection functions, `split_nb_gate` and `gram_matrix` divide an X whose largest entry is outside `2^-256 .. 2^256` (about 1e-77 to 1e77) by one power of two near that entry. Their results do not change with a common factor on X, so they are those of X near unit scale. 0.8.0 returned a statistic of 0 or NaN there from about 1e153 up and below about 1e-151. `gram_matrix` returns the Gram matrix of the divided X and no longer refuses X of about 1e154 or more. The fits and the other paths that read X as given under the flag divide it the same way and map every result back to X's scale: `pls1_fit` and `spls1_fit` (`coef`, `beta`, `q`, the scores), `pls3_fit`, `spls3_fit` and `plssvd_fit` (Y too, under `pre_standardized_Y`; singular values and scores), `score` (its statistic) and the reference fit of `pls1_confirmatory_test`, `pls1_perm_null`, `factors`, `pca_basis`, the parallel analysis of `profiles_find_n`, and `pls1_find_k_sequence` / `spls1_find_k_sequence`. 0.8.0 gave NaN or wrong results or an error there, from about 1e153 up and below about 1e-151 (`pls1_fit`: NaN `coef` above, a truncation error below).
- Changed: the cross-validation folds of `profiles_find_n` read X as the held-out tests do: divided once by its column standard deviations over all rows (used as given under `pre_standardized`, which before decided only the parallel analysis) and centred by each fold's training means, and the fold fit runs on those rows; no raw copy of a training fold is written. A column that barely varied on a fold's training rows blew up its held-out rows (held-out R² of -2.5e80 on a column of 1e-44 noise with one row at 0.5). Results change at the same seed.
- Changed: `pls3_confirmatory_test` decides every column without a refit on X: `split_exact` on its Gram route works from the Gram matrix of X, and dense `split_nb` from that Gram matrix on wide X where its cost rule takes that route, and from its blocks of X otherwise. A column whose outcome block is orthogonal to X up to rounding can get a different value than an X refit gives, and so can, with `keep_y`, a column near a tie in which Y columns are kept or in which sweep the sparse fit stops; neither value is more correct there. As on the PLS1 Gram routes, precision follows the largest column of X: with `pre_standardized_X` and columns of very different scale, or under any scaling with a row far from the rest (about 1e9 of the spread or more), other columns can lose precision too. Elsewhere other columns are unchanged. A split whose Gram blocks are not finite gives r = 0.
- Changed: on tall X the held-out tests of `pls1_confirmatory_test` (`split_exact`, `split_nb`, `raw_perm`) and `pls1_perm_null` decide every column, truncation included, with no refit on X: fits on the cross-product route come from the cross-product matrix of each training part, and `split_exact` and `split_nb` at k = 1 apply the truncation rule to their own products. A column whose outcome is orthogonal to X up to rounding, or whose fit asks for components past the rank of X, can get a different value than an X refit gives. The cross-product matrix squares the condition number of X, so a fit whose components reach a direction in which columns nearly coincide loses accuracy against an X refit: with one such pair at k = p, for any p, and earlier when several pairs nearly coincide (two pairs 1e-4 apart at p = 8 move r by about 1e-9 from k = 7). A row far from the rest can bury the other rows. The cross-product route now serves up to k = 8 components (was 3, and k = 3 only up to n/p of 3000).
- Changed: `pls1_perm_alpha` fits every column from `gram`: `alpha` is never NaN; `k_used` is below `k` where a fit stops at a truncation exit.
- Changed (performance): on tall X, unweighted `split_exact` in `pls1_confirmatory_test`, and `split_exact` (so also `auto`) in `pls3_confirmatory_test`, sum every split's training rows from the cells of its balanced design when n ≥ H: one pass over X per batch of null columns and one Walsh–Hadamard transform over the H cells give the sums of all J splits, instead of one pass over the training rows per split. `pls3_confirmatory_test` on this route also builds each null column once per call instead of once per split. Results agree with the per-split sums to rounding; in `pls1_confirmatory_test` a column whose fit sits at the truncation floor in a split can decide differently there. These arms, and the one that scores the held-out halves from moments, run only when their buffers fit `PLSKIT_MEMORY_BUDGET`, so results across budget values agree to rounding (about 1e-15 relative), not bit for bit.
- Changed: the default `k` of `pls1_confirmatory_test` is the set `1..8` (was 1), cut at `min(8, p, ⌊n/2⌋ − 2)` and below the weights' Kish `n_eff` (`n_eff ≥ k + 1`) and the covariates' residual df (`n_pos − rank([1, Z]) ≥ k + 1`), never below 1. It stays `{1}` for `split_nb`, `score`, `e` and `ci=True`. Results of a call without `k` change.
- Changed: `test_method="auto"` with a set of more than one `k`, the default set included, runs `split_exact` at any `n_eff`, and `stable_rank` is then `None`. On data where `"auto"` picked `split_nb`, the default call is now a refit permutation test up to k = 8 and costs much more; `k=1, test_method="split_nb"` is the fast choice.
- Changed: `ConfirmatoryTestResult.k` (Rust `ConfirmatoryTestOutput.k`) is the tested set, a list (Python `list[int]`, R integer vector, Julia `Vector{Int}`, Rust `Vec<usize>`); one `k` gives a one-element list, in `pls3_confirmatory_test` too. `statistic` is null for a set of more than one `k` (Python `None`, R `NULL`, Julia `nothing`, Rust `Option<f64>`).
- Changed: the default `k` of `pls1_fit` is `"optimal"` (was 1): it chooses `k` with `pls1_find_k_optimal` (selector `r2_se`) and fits there. Under `"optimal"` or `"sequence"`, an unset `k_max` is `min(9, p)`, lowered below the weights' Kish `n_eff` and the covariates' residual df, never below 1; an explicit `k_max` behaves as before. Without a seed the CV folds are random, so the chosen `k` can differ between runs. A `y` with no first component now raises `optimal_no_component` from the default call, where `k = 1` returned the `k_used = 0` zero model; an explicit integer `k` still returns it. Rust `pls1_fit` keeps its explicit `KSpec`.
- Changed (performance): `pls3_confirmatory_test` counts the per-Y-column cost of its Gram route (`Ỹ'GỸ` and the q × q eigendecomposition) and the primal route's p × q SVD when it picks a route, for `split_exact` and dense `split_nb`. With many Y columns it no longer takes the Gram route where that costs more. Results agree across routes to rounding.
- Fixed: Python and Julia refuse a 1-D bool array for an integer argument (for example `k=np.array([True])`, Julia `k=[true]`), as R refuses a logical vector; it ran as `k=[1]`. Bool data arrays are still read as 1/0.
- Removed: `PermAlphaResult.resolved` (Rust: `PermAlphaOutput.resolved`), which was always true.
- The testdata corpus has 107 cases: the 83 of 0.8.0, two new `gram_matrix` cases, six new `gram=` cases (`pls1_confirmatory_test_gram_*`: `split_exact` at k = 1 and 2, `split_nb`, `raw_perm`, `score`; `pls3_confirmatory_test_gram_split_exact`), nine new `covariates` cases (`pls1_fit_covariates` and `_weighted`, `pls1_predict_covariates`, `pls1_confirmatory_split_exact_covariates` and `_weighted`, `pls1_confirmatory_raw_perm_covariates`, `pls1_confirmatory_test_gram_split_exact_covariates`, `pls1_find_k_sequence_covariates`, `pls1_perm_null_covariates`), one `null_y` case (`pls1_confirmatory_split_exact_null_y`) and six set-of-`k` cases (`pls1_confirmatory_kset_split_exact`, `_raw_perm`, `_default_auto`, `_weighted_split_exact`, `_wide_split_exact`, and `pls1_confirmatory_test_gram_kset_split_exact`). Every `pls1_confirmatory_test` and `pls3_confirmatory_test` output stores `k` as a list and gains `pvalues_by_k` and `statistics_by_k`. 46 outputs of the 83 change: 21 of `pls1_confirmatory_test` (the `split_exact`, `split_nb`, `e` and `raw_perm` cases; in the four `*split_nb_ci*` cases the CI changes too, as it now reads X on the all-row scale), 4 of `pls3_confirmatory_test`, 5 of `pls1_find_k_sequence`, 7 of `spls1_find_k_sequence`, 4 of `pls1_find_k_optimal`, 2 of `spls1_find_k_optimal`, `spls1_find_keep_optimal_k1`, `profiles_find_n_n100_d12_k2` (its folds now read X on the all-row scale), and `pls1_perm_alpha_n40_p60_m4_k1`, which loses its `resolved` array (its other arrays are unchanged). The no-refit routes (wide, tall and PLS3) and the `pls1_perm_alpha` fits move no output beyond the last bits of three p-values (about 1e-14 relative) that `split_nb` at k = 1 computes on tall X; the weighted `split_exact` and `split_nb` outputs move through the centring of y above.

## [0.8.0] - 2026-11-05

- Added: `factors(coef, x, n, pre_standardized)` splits a coefficient vector into `n` orthogonal factors that sum to it. It builds the basis from the principal components of `X`, so the basis is the same whatever fit the coefficient came from. A part of the coefficient outside the row space of the standardized kept columns of `X` (for example from a sparse fit with more kept columns than `N − 1`) is appended, normalized, as one extra last basis direction, so it lands in the last factor.
- Added: `FactorsOutput::rotated(r)` rotates the factors by an orthogonal matrix, sorts them by size and pins their signs. In the wrappers this is an overload of `rotate`, and `rotate(f, X)` rotates with varimax on the row scores `Xs W`.
- Added: `Pls3Model::rotated(r)` and, in the wrappers, `rotate(PLS3Result)`, which rotates `U`, `V` and both score matrices, recomputes `singular_values`, sorts and pins signs. It warns with kind `rotated_sparse` on a sparse fit, whose rotated columns are not unit-norm. `rotate(PLS3Result, X)` uses the fit's stored `x_scores` as the varimax target, and `pre_standardized` is refused for a `PLS3Result`.
- Added: `pca_basis(x, pre_standardized)` returns the right singular vectors and singular values of standardized `X`.
- Added: `profiles_find_n(x, y, k, n_max, opts)` chooses the number of factors by parallel analysis and cross-validated R².
- Added: `profiles(model_or_coef, X, n)` in `plskit-bind` and the Python, R and Julia wrappers, which splits and rotates the coefficient vector of a fit in one call.
- Added: `profiles_search(model, X, y)` in `plskit-bind` and the wrappers, which runs `profiles` at the `n` that `profiles_find_n` picks and returns a `ProfilesSearchResult` with that search on it.
- Changed: `holdout_corr` in the confirmatory CI (`pls1_confirmatory_test` with `ci`) comes from half splits: each replicate fits on `m_h = ⌊n/2⌋` rows whatever `m_rate` is, and `point` is the mean over the half splits. A CI request with `⌊n/2⌋ < k + 2` is refused with `invalid_argument`. A replicate whose half-split fit fails has a NaN `holdout_corr` and is kept for the other readouts; this includes a half split with too few effective observations, which now counts toward `max_failure_rate`, not `max_skip_rate`.
- Changed: `pls1_perm_null` at `k = 1` holds `c` at `c_ref` instead of refitting it on each permuted `y`, so its null is that of `X_j'y`. At `k ≥ 2` it is unchanged.
- Removed: `rotate(PLS1Result)` and the `PLS1Result.rotation_spec` field. Use `profiles`, or `rotate(m.W, ...)` for the weights alone.
- Removed: `pls1_rotation_stability`, with `RotationStabilityOutput` and the `procrustes` dependency.
- Removed (Rust): `PlsKitError::AlreadyRotated`. The wrappers still raise the `already_rotated` code.
- The testdata corpus gains `factors_pls1_n50_d10_k2_n3` and `profiles_find_n_n100_d12_k2` and loses the three `pls1_rotation_stability_*` cases, so it goes from 84 to 83 cases. Five outputs change: the four `pls1_confirmatory_*split_nb_ci*` files and `pls1_perm_null_wide_n60_d3000_k1`.

## [0.7.2] - 2026-10-19

No engine code changes: results are unchanged; the testdata corpus is
unchanged.

- Added (docs): the user-facing docs are published at
  https://pawlenartowicz.pl/plskit/. The `_docs/` directory is renamed to
  `docs/`, and every path that pointed at it (docstrings, comments, this
  changelog, scripts) is updated. The new `docs.yml` workflow publishes
  `docs/` untagged on every push to `main` that touches it (the site's
  "latest (live)" version) and tagged `vX.Y.Z` on every engine tag. It is
  separate from `release.yml`, so a docs failure does not block a release.
- Added (tests): the code examples in the READMEs run as tests, in Python
  (`plskit-py/tests/test_readme.py`), R (`tests/testthat/test-readme.R`)
  and Julia (`test/readme.jl`). The READMEs are rewritten to run as written.
- Changed (tests): the Python corpus test reads the corpus from
  `$PLSKIT_TESTDATA` when set, and fails instead of skipping when the
  corpus is missing and `PLSKIT_REQUIRE_CORPUS` is set, which the Linux CI
  job does. R and Julia already used the same variables.
- Changed (tests): the `ParChoice::Seq` vs `Auto` check for `spls3_fit`
  moves from `byte_parity.rs` into `thread_count_parity.rs`, which runs it
  in every thread pool. `memory_budget_parity.rs` drops its ignored bits
  probe.
- Added (CI): a `julia-condapkg` job in `ci.yml` runs the Julia
  `Pkg.test()` on every push with CondaPkg left on, so CondaPkg resolves
  Python from conda-forge as it does for a user. Until the pinned
  `plskit` is on PyPI, the job pins a wheel built from the checkout instead.
- Removed (CI): the non-blocking R bit canary (R against the corpus at
  `atol` 0), with its `PLSKIT_CORPUS_EXACT` switch in `test-corpus.R`.
- Removed (R): the registry test of result field order, and the per-type
  field lists it read from the internal registry.
- Changed (R): `Suggests` needs `testthat (>= 3.1.5)`.

## [0.7.1] - 2026-10-16

- Added (Python): a wheel for Pyodide (`wasm32-unknown-emscripten`, built
  and tested against Pyodide 314.0.7, platform tag `pyemscripten_2026_0`),
  attached to each `vX.Y.Z-py` GitHub release. It is not uploaded to PyPI.
  Results are unchanged; the testdata corpus is unchanged.
- Added (CI): `scripts/julia-default-install-smoke.jl` installs PLSKit.jl
  with no pre-installed Python (CondaPkg installs it) and fits once. The
  `publish-jl` job runs it on Julia 1 and lts, and the new weekly
  `julia-default-install.yml` workflow runs it on the latest `vX.Y.Z-jl`
  tag.
- Changed: the engine no longer hits a compile-time overflow on 32-bit
  targets (checked on `wasm32-unknown-emscripten`); building the crates.io
  crate for Emscripten still needs a patched `atomic-wait`. The Python
  wheels and sdist build `atomic-wait` 1.1.0 from a copy in `vendor/` that
  adds an Emscripten module; other targets compile the crates.io code
  unchanged. On 32-bit targets `pls1_perm_alpha` accepts `n_features` up
  to `usize::MAX / 8` (536,870,911), a whole float count too large for a
  32-bit `usize` is rejected instead of saturating, and its error for too
  large an `n_features` gives the bound as a number
  (`9007199254740992` on 64-bit targets) instead of `2^53`. Results are
  unchanged; the testdata corpus is unchanged.
- Changed (memory, numerics): `selector="bic"` in `pls1_find_k_optimal`,
  `spls1_find_k_optimal` and `pls1_fit(k="optimal")` fits `X` in place
  instead of on a standardized copy, as does the check of the first
  component that opens every selection by cross-validation
  (`selector="r2_se"` or `"r2_max"`, and `spls1_find_keep_optimal`). BIC
  selection holds a copy of `X` only where its fit falls back to one (a
  column mean far from zero relative to its scale, or a component near the
  truncation floor). `bic_scores` differ from 0.7.0, and between C and F
  order of the same `X`, at rounding level, not to the bit. A near tie of
  two BIC scores or a fit at its truncation floor can select another `k`,
  and where the first component sits at its truncation floor a selection
  by cross-validation can return `k_star` (or `keep_star`) 0 in one
  release or layout and not in the other. The testdata corpus is
  unchanged.
- Changed (memory, numerics): the folds of the selections by
  cross-validation (`selector="r2_se"` or `"r2_max"` in
  `pls1_find_k_optimal`, `spls1_find_k_optimal` and
  `pls1_fit(k="optimal")`, and `spls1_find_keep_optimal`) fit `X` in place
  instead of on standardized copies of their training and validation rows.
  A fold holds a copy only where its fit falls back to one (a column mean
  far from zero relative to its scale, a component near the truncation
  floor, or a non-finite result), so the folds of a wide `X` can run in
  parallel at the default memory budget. `cv_scores` and `cv_scores_se`
  differ from 0.7.0, and between C and F order of the same `X`, at
  rounding level, not to the bit. A near tie (for `r2_se`, a score at the
  one-SE threshold) or a fold fit at its truncation floor can select
  another `k_star` or `keep_star`. The testdata corpus is unchanged.
- Changed (performance): the fits on raw `X` keep at most one strip's
  standardization moments per thread alive, which removes a 0.7.0 slowdown
  of about 8% on one thread at 200 × 60,000 (heap memory returned and
  page-faulted back on each fit). Results are unchanged to the bit.
- Changed (R): the repository no longer carries
  `plskit-r/src/rust/Cargo.lock`. The `vX.Y.Z-r` release job writes it in
  the staged package from the workspace `Cargo.lock` against the published
  `plskit` and `plskit-bind`, so `vX.Y.Z-r` can be tagged on the same
  commit as `vX.Y.Z`. Install a release from r-universe
  (`install.packages("plskit", repos = "https://pawlenartowicz.r-universe.dev")`)
  or from the `r-universe` branch
  (`devtools::install_github("pawlenartowicz/plskit", ref = "r-universe")`).
  A checkout of `main` builds only in dev mode (`PLSKIT_CARGO_CONFIG`).
- Fixed: a finite `X` with entries near `±f64::MAX`, whose
  standardization overflows, no longer gives a NaN statistic or a panic
  in `split_nb`, `split_exact` and `auto`; nor does such a `Y` in
  `pls3_confirmatory_test`. A split whose held-out half of `X` (or, in
  `pls3_confirmatory_test`, of `Y`) overflows contributes `r = 0` to
  `split_nb` and `split_exact` (`pls1_confirmatory_test` and
  `pls3_confirmatory_test`, every route), as a split whose training half
  overflows already did. The `split_nb` gate
  (`split_nb_gate`, and the gate inside `split_nb`, `auto` and the
  `find_k` functions) reports `stable_rank = NaN` and fires when the Gram
  matrix of the standardized `X` is not finite, where it could panic in
  its eigendecomposition or report `fires = false`.
  `linalg::stable_rank` returns NaN when the Gram matrix of its input is
  not finite. `pls1_confirmatory_test` with `test_method="raw_perm"`
  returns `non_finite_input` on its K = 1 Gram route when a training fold
  of `X` overflows, as its other routes do. Two results change on input
  whose standardization does not overflow: a weighted gate whose Gram
  matrix overflows (a zero-weight row holding `1e200`) now fires, and a
  PLS3 `split_exact` split on the Gram route whose held-out product
  overflows gets the primal route's value. Results on other input are
  unchanged to the bit. The testdata corpus is unchanged.
- Fixed: the memory estimate behind `PLSKIT_MEMORY_BUDGET` counts what a
  unit allocates: each copy of `X`, of a half of it or of a Gram matrix
  with its rows padded to a multiple of 8, as the matrix library stores
  it, and the two halves of a split or fold as two matrices; the blocks in
  which `X` is read at their size; and the buffers of the per-split fits
  of `pls3_confirmatory_test` (`split_nb` and `split_exact`), and, where a
  split's or fold's permutations are fitted in parallel
  (`pls1_confirmatory_test` with `raw_perm` or `split_exact`, and
  `pls3_confirmatory_test` with `split_exact`), one fit per thread. Batch
  sizes change with the estimate, mostly smaller, so the peak memory stays
  nearer the budget. Results are unchanged to the bit.
- Fixed (R): under webR's build R, `cargo-build.R` builds the Rust library
  for `wasm32-unknown-emscripten` with panics set to abort, so a panic ends
  the R session instead of raising `plskit_error` (untested on
  r-universe); it finds `cargo` without `Sys.which()`, and the source
  tarball carries an `atomic-wait` with Emscripten code.
- Fixed (R): on Linux and macOS the build deletes the Rust static library
  and the unpacked vendored crates once `plskit.so` is linked (Windows
  removes only the unpacked crates), so `R CMD check` reports no `abort` or
  `_exit` in the compiled code (checked on Linux; the warning appeared on
  macOS).

## [0.7.0] - 2026-10-11

- Changed (performance): `pls1_fit`, `spls1_fit`, `pls3_fit`, `spls3_fit`
  and `plssvd_fit` on raw `X` read `X` in strips of columns for its
  standardization moments (2 MiB strips, so that each strip's sweeps run
  from cache when `X` has at most 32,768 rows; a C-ordered `X` with more
  than 1,024 rows in wider strips, about one per thread, so that each
  strip's rows stay long), on large inputs over several threads. Results
  are unchanged to the bit. The testdata corpus
  is unchanged.
- Changed (numerics): every PLS1 fit on standardized `X` (`pls1_fit` and
  `spls1_fit` with `pre_standardized`, and the fits on a standardized copy
  of `X` inside the selection and sequence functions,
  `pls1_confirmatory_test` with `test_method="e"`, the confirmatory CI,
  `pls1_rotation_stability` and the reference fit of `pls1_perm_null`)
  forms the sum of squares behind its truncation floor in a fixed number
  of pieces, over several threads on large inputs. A fit whose component sits within rounding of its floor
  can stop at a different `k_used`, and the results built on it change
  with it; other results are unchanged. The testdata corpus is unchanged.
- Changed (performance, numerics): `pls1_fit` and `spls1_fit` with
  `pre_standardized` and no weights, on a C-ordered `X` with at least
  2,048 columns, form `X'·t` block by block over the rows of `X`, as they
  do without `pre_standardized`. So do the fits on raw C-ordered `X` of
  that width that fall back to a standardized copy (column means far from
  zero relative to their scale, or a component near the truncation
  floor). Results differ from the previous release at rounding level; so
  do `pls1_rotation_stability` and the confirmatory CI on such input. The
  testdata corpus is unchanged.
- Changed (performance, numerics): `selector="bic"` in
  `pls1_find_k_optimal`, `spls1_find_k_optimal` and
  `pls1_fit(k="optimal")` takes each `k`'s residual sum of squares from the
  fit's scores, with no pass over `X` per `k`. `bic_scores` differ from the
  previous release at rounding level, and a near tie can select the other
  `k`. The testdata corpus is unchanged.
- Changed (performance): `pls1_find_k_sequence`, `spls1_find_k_sequence`,
  `pls1_fit(k="sequence")` and the `diagnostic` of `pls1_find_k_optimal`
  and `spls1_find_k_optimal` standardize `X` once per call instead of once
  per step. Results are unchanged to the bit. The testdata corpus is
  unchanged.
- Added: `pls1_perm_alpha(gram, Y, k=1, *, n_features)` in every wrapper.
  For each column of `Y` it returns the PLS1 dual coefficients `alpha`
  (`coef = X̃'·alpha`) from the Gram matrix `X̃X̃'` of a standardized
  block the caller holds; plskit never sees the block. `k = 1` only. A
  column the fit cannot certify from the Gram matrix alone is reported
  (`resolved = False`, `alpha` NaN), not refitted. `gram` is checked for
  a non-negative diagonal, symmetry, off-diagonal entries within the
  Cauchy-Schwarz bound of the diagonal, centring and its trace. One new
  testdata fixture.
- Added: `PLSKIT_MEMORY_BUDGET`, an environment variable set and checked
  like `PLSKIT_NUM_THREADS` in Rust, Python, R and Julia: the bytes of working
  memory, beyond the inputs, for the buffers that grow with the data
  (copies of `X` or of its halves, Gram matrices, the blocks in which `X`
  is read). It takes a whole number of bytes with an optional `K`, `M`,
  `G` or `T` (powers of 1024) and an optional `B`, or `iB` after the
  letter, in any case;
  the default is `2G`, and `0` runs one unit at a time. An invalid value
  is `InvalidArgument` from every function that reads
  `PLSKIT_NUM_THREADS`, before any work. The splits, folds and resamples
  of `pls1_confirmatory_test` (`split_nb`, `split_exact`, `raw_perm` and
  the `ci` pass), `pls3_confirmatory_test` (`split_nb`, `split_exact`),
  the cross-validated selection of `pls1_find_k_optimal`,
  `spls1_find_k_optimal` and `spls1_find_keep_optimal`, and
  `pls1_rotation_stability` run in parallel batches sized by the budget,
  at least one unit per batch, so their peak memory is bounded by it
  whatever the thread count: where each unit copies `X`, an `X` too large
  for two copies in the budget runs one unit at a time, with a peak of
  about `X` plus one copy.
  The budget is an estimate of the large buffers, not an exact ceiling.
  Results do not depend on it. See
  [Memory budget](docs/concepts/threads.md#memory-budget).
- Changed (numerics): for shapes with `n·d·k` at or above 1,000,000 (`n`
  training rows, `d` columns, `k` components), the PLS1 fits inside
  `pls1_confirmatory_test`'s `split_nb` and `split_exact` (the per-split
  fits of their refit routes, and the standardized-copy fallback of the
  `k = 1` routes), the cross-validated selection, the `ci` pass of
  `pls1_confirmatory_test` and `pls1_rotation_stability` use the fixed
  8-piece parallel arithmetic that `pls1_fit` uses at that size, not one
  thread, so their outputs differ from the previous release in the last
  bits. The steps of `pls1_find_k_sequence` and
  `spls1_find_k_sequence`, the `diagnostic` of `pls1_find_k_optimal` and
  `spls1_find_k_optimal`, and `pls1_fit` with `k="optimal"` or
  `"sequence"` (Python, R and Julia) run these and move with them.
  Smaller shapes are unchanged. Results do not depend on the thread count
  or the budget. The testdata corpus is unchanged and passes within its
  tolerance.
- Changed (performance): `pls1_confirmatory_test` at `k = 1` on dense
  input (`split_exact`, `split_nb`, and `raw_perm` on its Gram route) and
  `pls3_confirmatory_test` on dense `X` (`split_exact` on its Gram route;
  `split_nb` with dense `Y`) no longer write a standardized copy of `X`
  per split or fold: `X` is read one block of columns at a time and only
  the small products are kept. A split, fold or column the block-by-block
  route cannot decide safely is computed on a standardized copy as
  before. The `split_nb` auto-gate does the same on `X` with at least as
  many columns as rows.
- Changed (numerics): on those requests results differ from the previous
  release at rounding level. Products are summed block by block and the
  truncation floors take `‖X‖_F` from the standardization moments. Dense
  `split_nb` also changes formula: `pls1_confirmatory_test` at `k = 1`
  takes `split_exact`'s no-refit formula, and `pls3_confirmatory_test`
  scores the held-out half through `X'Y`; each is tested against the
  refit to `1e-10`. The steps of `pls1_find_k_sequence` and of the
  `diagnostic` of `pls1_find_k_optimal`, and of their `spls1_` forms at
  `keep = n_features`, run the `k = 1` test and move with it.
  `pls3_confirmatory_test`'s dense `split_nb` statistic agrees with
  `split_exact`'s at the same seed to rounding (tested to `1e-10`); where
  `split_exact` takes its primal route the two were bit-identical in the
  previous release. The testdata corpus is unchanged.
- Changed (performance): the `split_nb` auto-gate takes the stable rank
  from the Gram matrix on the shorter side of the standardized `X`, not
  from an SVD of it, so on `X` with more rows than columns the gate holds
  one copy of `X` where it held two. `stable_rank` differs from the
  previous release in the last bits. The testdata corpus is unchanged.
- Changed (performance): `pls3_fit`, `plssvd_fit` and `spls3_fit` no longer
  write a standardized copy of `X`: `X'Y` and the X scores are formed from
  the raw `X`, read in its own memory layout (a C-ordered NumPy array in
  place), with the centering and scaling applied inline, as `pls1_fit` /
  `spls1_fit` do. When a column's |mean| exceeds 1000 times its scale, `X`
  is standardized into a copy as before. A fit in which one of the `k`
  requested components has its `σ` below `2·(1 + max |mean| / scale)` times
  the truncation floor is refitted on a standardized copy, which costs
  about two fits and the copy's memory. The truncation floor takes `‖X‖_F`
  from the standardization moments. `Y` is standardized into a copy as
  before, and a `pre_standardized_X=True` fit is unchanged to the bit.
- Changed (numerics): `pls3_fit`, `plssvd_fit` and `spls3_fit` without
  `pre_standardized_X` are no longer bit-identical across memory layouts of
  `X`; row- and column-major inputs agree to rounding (the corpus
  tolerance). Their results also differ from the previous release in the
  last bits. The testdata corpus is unchanged.
- Added: `test_method="auto"`, now the default of `pls1_confirmatory_test`,
  `pls3_confirmatory_test`, `pls1_find_k_sequence` and
  `spls1_find_k_sequence`, in Rust, Python, R and Julia. It runs
  `split_exact` or `split_nb`, chosen once per call from `X` and the weights,
  and `result.test_method` reports the method that ran. The two confirmatory
  tests no longer require `test_method`.
- Changed (breaking, Rust): the `Default` of `ConfirmatoryTestOpts` and
  `Pls3ConfirmatoryTestOpts` changes from `SplitExact` to `Auto`, so large
  designs now run `split_nb` by default.
- Changed (breaking): the default `test_method` of `pls1_find_k_sequence` and
  `spls1_find_k_sequence` changes from `split_nb` to `"auto"`, which can
  change the method that runs by default. `pls1_fit(k="sequence")` inherits
  it.
- Changed: `spls1_find_k_sequence` runs `split_exact` by default on designs
  where `"auto"` picks it (`n_eff < 250`, or the `split_nb` auto-gate
  fires). That is a refit route, about `(n_perm + 1) * n_splits` fits per step
  instead of `n_splits`.
- Changed (breaking): `disable_parallelism` is removed from every function,
  in Rust, Python, R and Julia. Set `PLSKIT_NUM_THREADS` to cap the thread
  count; `PLSKIT_NUM_THREADS=1` runs on one core. Results do not depend on
  the setting. The testdata manifest drops the `disable_parallelism` key;
  numerical outputs are unchanged.
- Changed (breaking): `pls1_confirmatory_test` and `pls3_confirmatory_test`
  take `test_method=` instead of `method=`, and their result reports
  `result.test_method` instead of `result.method`, in Python, R and Julia.
  The Rust `ConfirmatoryTestOutput.method` field is now `test_method`. This
  matches the `test_method` argument of `pls1_find_k_sequence` and
  `spls1_find_k_sequence`. There is no alias for the old name. Error
  messages that name the argument say `test_method=` too. The testdata
  corpus renames the `method` output entry and manifest kwargs key of the
  confirmatory cases to `test_method`; numerical outputs are unchanged.
- Changed (breaking, Rust): `linalg::standardize_weighted` and
  `linalg::standardize1_weighted` panic when `weights` is `Some` and its
  length differs from the row count. They used to read the first `n`
  entries of a longer `weights`.
- Added: the `plskit-bind` workspace crate, a language-neutral binding
  layer over the plskit engine. It exposes the whole Python surface
  through one `call(fn_name, Record)` entry point. The R wrapper calls the
  engine through it; the Julia wrapper uses only its registry dump, to
  render its stubs.
- Changed (Python, stricter input): `pls1_fit`, `spls1_fit`,
  `spls1_find_keep_optimal`, `spls1_find_k_optimal` and
  `spls1_find_k_sequence` reject a `k` or `keep` that is not a non-negative
  whole number with `PlsKitError(code="invalid_argument")` instead of
  coercing it: a fractional float (`k=2.7` used to fit 2 components) or a
  `bool` now raises where it used to coerce; a negative value already
  raised, but as a PyO3 `OverflowError` rather than a `PlsKitError`, and now
  raises `PlsKitError(code="invalid_argument")` instead. A whole float such
  as `2.0` is still accepted. `pls1_fit` with an integer `k` rejects a
  `k_max` or `find_k_args` argument instead of ignoring it; for
  `spls1_find_k_optimal` and `spls1_find_k_sequence`, only `keep` is
  checked this way, not `k_max`. All of this matches the rules the R
  wrapper applies through `plskit-bind`; the Julia wrapper inherits them
  from the Python package.
- Changed (Python): the extension calls the engine through `plskit-bind`.
  Newly accepted: a plain `dict` holding a model's fields as `model` (a
  dict missing a field raises `invalid_argument`); a decimal
  string as `seed` (`seed="7"` is seed 7); a 0-D array where a scalar is
  expected (`pre_standardized=np.array(False)`,
  `test_method=np.array("score")`), also inside `args` / `rotation_args`.
  Newly rejected, with
  `PlsKitError(code="invalid_argument")`: a number type with no plain
  int / float form (`Decimal`, `Fraction`) inside `args` / `rotation_args`;
  `rotate(model)` with a `P` whose row count differs from `W`'s, or with a
  `None` or mistyped field the rotation does not use (`coef=None`);
  `pls1_predict` and `pls3_transform` on a model with a mistyped field
  (`pre_standardized=1`, `intercept="3.5"`) or an unconvertible value in a
  field they do not use (`selection_result=object()`); a subclass of a
  result class as `model`; a model field that is a nested list
  (`T=m.T.tolist()`) or a flat list holding a float (`Q=m.Q.tolist()`) in
  `rotate`, `pls1_predict` and `pls3_transform`; an int of `2**64` or
  more, or below `-2**63`, as a float argument (`tol=2**64`,
  `tol=-2**63 - 1`). Error codes: a wrong-typed value inside
  `find_k_args` is `invalid_args` (was `invalid_argument`); non-string keys
  in `args` / `rotation_args` / `find_k_args`, and any other value in
  `args` / `rotation_args` that cannot be converted (`object()`,
  `{'n_perm': 2**64}`), are `invalid_argument` (was `invalid_args`);
  `rotate(model)` with `T`, `P` or `Q` not matching `W`'s column count
  raises `PlsKitError` with `invalid_argument` (was numpy's `ValueError`);
  a Rust panic raises `PlsKitError` with `internal` (was `PanicException`).
  Several error messages are now worded as `plskit-bind` words them.
  `rotate(model)` forms `T·R`, `P·R` and `Rᵀ·Q` in faer instead of numpy,
  so results agree to rounding, not bit for bit. `rotate(model)` returns
  new, equal objects for the fields it does not rotate
  (`selection_result`, `coef`, `beta`) instead of the caller's objects.
  The reroute warning from `pls1_find_k_optimal`, `pls1_find_k_sequence`
  and `pls1_fit(k="optimal" | "sequence")` now points at the caller's
  line.
- Added (R): the R package `plskit` (`plskit-r/`) with the whole Python
  surface: the same 20 functions, argument names and result fields,
  through the `plskit-bind` layer. Results are classed named lists
  (`c("pls1_result", "plskit_result")`, ...), seeds are decimal strings,
  and errors are `plskit_error` conditions carrying the Python error
  codes. It passes the shared `testdata/` corpus. Until `plskit-bind` is
  on crates.io it builds only from a checkout, with `PLSKIT_CARGO_CONFIG`
  naming a cargo config that patches in the in-repo crates
  (`docs/r/installation.md`). The package version moves from 0.0.1 to
  0.7.0, the engine version. The placeholder `version()` export, which
  masked `base::version`, is removed.
- Added (Julia): the Julia package `PLSKit.jl` (`plskit-jl/`) with the
  whole Python surface: it runs the Python `plskit` package through
  PythonCall.jl, so every function, argument name and result field is the
  Python one. Results are `PlsKitResult{T}` values with one alias per
  result type (`PLS1Result`, ...), errors are a single `PlsKitError`
  carrying Python's `code` as a `Symbol`, and Python warnings are
  re-emitted with `@warn`. Its tests check one `testdata/` fixture per
  function at the corpus tolerances; the numbers come from the Python
  wheel, which passes the whole corpus. The
  package and module are renamed from `plskit` to `PLSKit` (UUID kept),
  version 0.7.0, pinning PyPI `plskit==0.7.0` through `CondaPkg.toml`.
  The placeholder `version()` export is removed.
  - The Julia package version, its `CondaPkg.toml` pin and
    `PLSKIT_PY_VERSION` are always equal and name the `plskit-py` release
    the package runs on; Julia may lag Python, never lead it. CI builds
    `plskit-py` from source when the checkout matches the pin and installs
    the pinned wheel from PyPI when the checkout is newer
    (`scripts/julia-python-pin.py`); the `-jl` release job tests against
    PyPI only.
- Added (CI): `scripts/check-profile-sync.py` keeps the R crate's
  `[profile.release]`, `[workspace]` table and dependency versions in
  step with the workspace; `scripts/render-r-stubs.py --check` and
  `scripts/render-jl-stubs.py --check` fail when the committed R stubs or
  Julia stubs and aliases drift from the `plskit-bind` registry;
  `scripts/check-docs-drift.py` also checks the R `NAMESPACE` exports and
  the Julia `export`s against the Python surface. The R package is checked
  with `R CMD check` on Linux, macOS and Windows, and the Julia tests run
  on the same three OSes (Julia LTS and current).
- Fixed: `pls1_confirmatory_test` confidence intervals (normal-theory
  leverage CI and NB-Wald `holdout_corr` CI) used a wrong inverse normal.
  Below `level` 0.85 the critical value was off by up to 20%
  (Φ⁻¹(0.9) gave 1.0253, not 1.2816): CIs were too narrow near level 0.8
  and too wide near level 0.5. At the default level 0.95 the bounds move by
  about 2e-10 relative. `standard_normal_inv` now matches Wichura's AS241
  in every branch, including a far-tail coefficient typo that the public
  API cannot reach.
- Fixed: a column of `X` (or `y`, or a column of `Y`) that is constant to
  rounding now gets scale `max(1, |mean|)` instead of `1`, so it
  standardizes to zeros up to rounding at any magnitude. Before, a constant
  column near `1e300` truncated `pls1_fit` to `k_used = 0` or gave NaN
  coefficients and made `pls3_fit` fail with "SVD of X'Y failed to
  converge", and one near `1e20` silently shifted the other `pls1_fit`
  coefficients. `X_scale` / `Y_scale` (from `preprocess` and PLS3 fits) now
  report `|mean|` for such a column when `|mean| > 1`; constants of
  magnitude at most 1 keep scale 1. Such columns no longer force
  `pls1_fit` onto its materialized-copy route.
- Fixed: a column whose standard deviation underflows (zero or subnormal,
  e.g. a single `5e-324` among zeros) is treated as constant (only
  centered) instead of getting scale `0` or a subnormal scale and a silent
  all-NaN `pls1_fit` or a failed `pls3_fit`.
- Fixed: all-equal observation weights are now identical to absent weights
  at every entry: results are bit-identical to `weights=None` (including
  `rho_hat` on `split_nb`, the `split_nb` auto-gate, the replicate route,
  BIC, and `n_eff = n` exactly), and an `n_eff < k + 1` failure raises
  `invalid_argument` rather than `invalid_weights` /
  `insufficient_effective_n`. Before, equal non-unit weights could round
  `n_eff` just below `n`, rejecting a feasible `n = k + 1` call or firing
  the `split_nb` gate at `n = 25`. The same holds inside
  `pls1_confirmatory_test`: a `raw_perm` fold or a `split_exact` /
  `split_nb` half whose own weights are all equal (or all zero) trains
  unweighted, as `pls1_fit` does on those rows.
- Fixed: `pls1_confirmatory_test` rejects `k = 0` with `invalid_argument`
  ("k must be >= 1") for every method; `split_exact`, `split_nb` and
  `score` used to return a p-value.
- Fixed (Python): `preprocess` with a 2-D `Y` containing NaN or infinity
  raises `non_finite_input`, like a 1-D `y`. Validation and
  standardization of a multi-column `Y` moved into the core
  (`plskit::preprocess::preprocess_block`); outputs for finite input are
  bit-identical.
- Fixed (Python): an unknown `method`, `selector`, `diagnostic` or
  `test_method` raises `PlsKitError(code="invalid_args")` (the `code` used
  to be empty). Model-dict fields missing on `pls1_predict` /
  `pls3_transform` raise `invalid_argument` instead of an uncoded error or
  a panic.
- Fixed (Python): an unusable argument value no longer escapes as a raw
  `TypeError` / `OverflowError`: a negative, fractional or non-numeric
  count, a seed outside `[0, 2^64)`, a non-number float option, a non-bool
  flag, a non-string method name or a non-dict `args`. Inside an `args` /
  `rotation_args` dict the error is `invalid_args`, for a top-level
  argument `invalid_argument`; messages match `plskit-bind`.
- Changed (Python): argument rules now follow `plskit-bind`. `None` for an
  optional argument or an args-dict key means its default, a whole float
  such as `100.0` is accepted as a count, and a `bool` is rejected where a
  number is expected. `pls1_rotation_stability` parses `rotation_args` with
  `rotate`'s parser, and `pls1_fit` validates `seed` when `k` is an int.
- Changed (Python): `RotationSpec.args` records the varimax arguments the
  engine resolved (defaults filled, counts as `int`) instead of echoing the
  caller's dict. The private `_plskit.VARIMAX_DEFAULTS` is removed.
- Changed (Python): `ConfirmatoryTestResult` fields are ordered `n_eff,
  rho_hat, stable_rank`, matching `results.md` and the R/Julia wrappers;
  `scripts/check-docs-drift.py` now fails when a Python result dataclass
  orders its fields differently from `results.md`.
- Changed (testing): corpus comparisons (Rust, Python, R, Julia, bind and
  the generator's settle step) add a relative term, `|a - e| <= atol +
  1e-14 * |e|` for finite `e` (an infinity equals only itself), so
  cross-host drift of an ulp or two in large values (the ~6126 score
  statistic) no longer sits at the edge of `atol = 1e-12`.
- Testdata: regenerated the three `pls1_confirmatory_test` `split_nb` CI
  fixtures (AS241 fix) and added `pls1_confirmatory_split_nb_ci_level80`,
  the first CI fixture off the default level, which pins the level-0.8
  critical value and fails a wrapper that drops `level`.
  `producing_version` is 0.7.0. The manifest drops the per-case `tolerance`
  field, which no corpus reader used; the tolerances are in
  `testdata/README.md` ("Tolerance").
- Docs: `length_mismatch` is listed as an `invalid_weights` reason; the
  `split_nb` docstrings state the actual auto-gate rule (at most 4 columns,
  `n_eff < 25`, or stable rank < 3);
  `RotationStabilityResult.degenerate_baseline` no longer claims
  `variance_ratio.point` is always NaN when set (only the `V_unrot = 0`
  trigger gives NaN).
- Fixed: `k = 0` and `k_max = 0` raise `invalid_argument` ("k must be >= 1"
  / "k_max must be >= 1") at every entry, in every wrapper.
  `pls1_rotation_stability`, the `find_k` family, `spls1_find_keep_optimal`
  and `pls1_fit(k="optimal"|"sequence")` raised `k_exceeds_max`,
  `pls1_perm_null` used a different message, and `pls3_confirmatory_test`
  reported its `k = 1` restriction. `k_exceeds_max` now means only a count
  above its maximum.
- Fixed: `preprocess` reports `n_eff` exactly `n` for all-equal weights, as
  every fit does (Kish's ratio could round an ulp below `n`).
  `weights_normalized` is still returned.
- Fixed: `rotate` on a one-column `W` (K = 1) reports `V_converged` for the
  target the K >= 2 path starts from: `L` if given, else `W`, row-normalized
  under `kaiser_normalize`. It used to skip the row normalization, so with
  the default `kaiser_normalize=True` it reported the criterion of the raw
  column. `W_rot`, `R` and `sweeps` are unchanged.
- Fixed (Python): an array argument numpy cannot read as real numbers
  (strings, including numeric strings, ragged nested lists, non-numeric
  objects, complex, datetime or structured dtypes) raises
  `PlsKitError(code="invalid_argument")` naming the argument, as
  `plskit-bind` does, instead of numpy's `ValueError` / `TypeError`; a
  complex array used to be cast silently to its real part. `None` and
  `pd.NA` in an object array read as NaN (`non_finite_input`). A 0-d
  `weights` or `Y` is rejected as not 1-D / 2-D.
- Changed (Python, performance and numerics): a C- or F-ordered float64 `X`
  reaches the engine without a copy at every entry that takes one. This
  reverses the 0.6.1 rule that the Python API passes every `X` on in C
  order: an F-ordered `X` no longer pays a full copy (about 25 to 40 ms per
  fit at 1e7 entries), and results for different memory layouts of the same
  values agree to rounding (corpus tolerance) rather than bit for bit.
  `pls1_fit`, `spls1_fit`, `pls1_predict`, `pls1_rotation_stability`,
  `pls1_confirmatory_test` with `ci` and, without `pre_standardized_X`, `pls3_fit`, `plssvd_fit` and `spls3_fit` can
  differ in the last bits between C and F order; the same array in the same
  layout still gives byte-identical results across runs and thread counts.
  Julia arrays are column-major and now read in place, so Julia
  `pls1_fit` / `spls1_fit` results can differ from 0.6.1 in the last bits
  (they now match the Rust core's column-major bits).
- Changed (performance): the permutation and split loops take `‖X̃‖_F` once
  per block instead of once per replicate on their primal route
  (`pls1_perm_null`, the `raw_perm` CV folds, the `split_exact` refit
  splits). On one core at 1000 × 1000 with 1000 permutations, a k = 1
  `pls1_perm_null` drops from 1.41 s to 0.66 s and a k = 1 `raw_perm` from
  5.86 s to 2.88 s, so k = 1 is again faster than k = 2 at that shape.
  Results are bit-identical.
- Changed (internal): the p-space Gram backend's per-replicate diagnostics
  are compiled only into tests; production replicates no longer fill or
  allocate them. Results are unchanged.
- Fixed: `pls3_fit`, `plssvd_fit`, `spls3_fit`, `preprocess` and
  `preprocess_block` raise `invalid_argument` ("insufficient n: need
  n >= 1") on zero rows, as `pls1_fit` does. They used to return NaN
  standardization moments, the PLS3 fits with `k_used = 0`. One row is
  still not an error for the PLS3 family: its `k` is not bounded by `n`, so
  the fit truncates.
- Fixed (numerics): `pls3_fit`, `plssvd_fit` and `spls3_fit` give
  bit-identical results for every memory layout of `Y`, and of an `X`
  passed with `pre_standardized_X`, also with `pre_standardized_Y`. A
  pre-standardized row-major or negative-stride block used to move the
  scores in the last bits (up to about 4e-14 for a reversed row order) and
  could move `k_used` on the truncation floor; such a block is now copied
  column-major. Column-major pre-standardized
  results are unchanged to the bit. In Python, a default C-ordered `X` with
  `pre_standardized_X=True` now gives the same bits as its F-ordered copy.
- Changed (R, plskit-bind): bool data is read as 1/0 in every numeric
  array argument, as numpy reads a bool array in Python and Julia a `Bool`
  array, with the same bits as the equal 0/1 doubles. In R a logical matrix
  keeps its shape, and logical vectors and logical data.frame columns are
  accepted; `NA` in logical data raises `non_finite_input`, as in numeric
  data. Flags and integer arguments stay strict.
- Fixed (Python): `pls1_predict`, `pls3_transform`, `plssvd_transform` and
  `rotate` given a model of the wrong type (a `PLS3Result` where a
  `PLS1Result` is expected, `None`, ...) raise
  `PlsKitError(code="invalid_argument")` in `plskit-bind`'s words
  ("model must be a PLS1Result, got a PLS3Result"), as R does, instead of
  `AttributeError` or `TypeError`. `rotate` reads a nested-list `W` like any
  other array argument.
- Changed (Python, performance): an aligned F-ordered float64 `Y`, `Y_new`,
  `W` or `L` reaches the engine without a copy. C-ordered and strided ones
  are still copied column-major, so results are bit-identical to before for
  every layout.
- Changed (performance): `pls1_fit` / `spls1_fit` on a row-major
  (C-ordered, the Python default) `X` at least 2048 columns wide read the
  `X'·t` products in blocks of 8 rows (the standardization moments are
  read in strips of columns, see the entry on strips of columns above). On
  one M4 core a 1000 × 10000 C-ordered k = 1 fit drops from about 24 ms to
  about 13 ms (it was about 1.4x slower than its column-major copy, and is
  now no slower). Moments are
  bit-identical. Fits of such inputs that are not pre-standardized move in
  the last bits, agreeing with 0.6.1 to corpus tolerance but not bit for
  bit, and stay byte-identical across thread counts; for pre-standardized
  ones, see the entry on `pre_standardized` and C-ordered `X` above.
  Column-major `X` and narrower `X` are unchanged bit for bit by this
  change. The reference fit inside `pls1_rotation_stability`, `pls1_confirmatory_test`
  with `ci`, and the final fit of `pls1_fit(k="optimal"|"sequence")` take
  the same path.
- Changed (performance): the rule that sends a `perm_null`, `raw_perm` or
  `split_exact` resampling loop to the p-space Gram route (build
  `C = X'X` once, then one `C·r` per replicate) now weighs the build and
  the per-replicate product at their measured costs:
  `p·(0.33·n_tr + 1.3·B·k) < 2·B·k·n_tr`, was `p·(n_tr + B·k) <
  2·B·k·n_tr`. Shapes near the boundary that stayed primal although the
  Gram route was faster now take it, and break-even shapes that took it
  stay primal (over a 164-shape sweep: 91.3 s to 85.5 s in total;
  `split_exact` at 1600 × 640, k = 2, 199 permutations: about 1.2x faster
  on one M4 core). The
  coefficients were fitted on an Apple M4. No corpus fixture changes
  route. On shapes that do, results agree with 0.6.1 to corpus tolerance
  (observed at most 1.6e-14), not bit for bit.
- Changed (performance): `split_exact` at k = 2 on the n-space Gram route
  (`p` well above the training size) runs its kernel over runs of 16
  outcome columns as matrix products instead of one matrix-vector product
  per column: 1000 × 1000, 1000 permutations, k = 2 drops from about 12.2 s
  to about 6.0 s on one M4 core. k = 1 is unchanged. The statistic and
  p-value were unchanged in every case tested; the per-split statistics can
  move by about 1e-16 when the last run of columns is partial
  (`n_perm + 1` not a multiple of 16) at small training sizes, within
  corpus tolerance. Results stay byte-identical across thread counts, and
  splits that fall back to the primal refit are unchanged bit for bit.
- Changed (release): a `vX.Y.Z` tag now publishes `plskit-bind` to
  crates.io after `plskit`, for the R package to build against. The
  workspace pins its `plskit` dependency exactly (`=X.Y.Z`), so a version
  bump that misses it fails to build instead of drifting. The `-r` release
  job installs the R tarball against crates.io with the committed lock file
  before uploading it.
- Changed (dependencies): `chacha20` (through `rand`) moves from the yanked
  0.10.0 to 0.10.2, so a freshly resolved R lock file gets the same version
  as the workspace. No numeric change: corpus, byte-parity and
  thread-count tests pass unchanged.

## [0.6.1] - 2026-09-29

- Changed (performance): `pls1_fit` / `spls1_fit` no longer write a
  standardized copy of `X`: the kernel forms its products from the raw `X`,
  read in its own memory layout (a C-ordered NumPy array in place), with the
  centering and scaling applied inline. When a column's |mean| exceeds 1000
  times its scale, `X` is standardized into a copy as before, since the
  inline form would lose digits to cancellation. A fit that stops before `k`
  components on its truncation floor, or keeps one within
  `2·(1 + max |mean| / scale)` times that floor, is refitted on a
  standardized copy, which costs about two fits and the copy's memory.
  `pre_standardized=True` no longer copies `X` into column-major order (a
  weighted fit still forms its √w-scaled `X`). The truncation floor takes `‖X‖_F` from the standardization
  moments, and the finiteness check from the same pass.
- Changed (numerics): `pls1_fit` / `spls1_fit` results from Rust are no
  longer bit-identical across memory layouts of `X`; row- and column-major
  inputs agree to rounding, except that a fit sitting on its truncation
  floor (typically a `pre_standardized` `X` outside the scale contract) can
  stop at a different `k_used`. The Python API passes every `X` on in C
  order, so its results do not depend on the input's layout; a
  `pre_standardized=True` fit of a C-ordered array is now read row-major
  rather than copied column-major, so its last bits (and, on the floor, its
  `k_used`) can differ from 0.6.0. A standardizing fit forms its products from
  the raw `X`, so its coefficients differ from 0.6.0 by about
  `max |mean| / scale · 1e-16` relative; a stop on the truncation floor
  (`k_used`) is decided on the standardized copy whenever it is close enough
  that the raw-`X` products could decide it differently.

## [0.6.0] - 2026-09-27

- Added: `spls3_fit`, sparse PLS3 / PLSSVD with a keep-count per side
  (`keep_X` non-zeros per column of `U`, `keep_Y` per column of `V`, ties
  broken toward the lower index). `keep_Y < n_targets` forces each latent
  dimension onto a few outcomes. Full keep-counts delegate to `pls3_fit`
  (bit-identical); otherwise saliences are unit-norm but not orthogonal.
  `Pls3Model` / `PLS3Result` gain `keep_X`, `keep_Y` (`None` on a dense
  fit), `converged` and `n_iter`; reaching `max_iter` is not an error.
- Added (Rust only): `keep_x` / `keep_y` on `Pls3ConfirmatoryTestOpts`,
  applied inside each training half (the wrappers do not expose them);
  `PermNullOpts: Default`; public `SPLIT_NB_REROUTE_N_PERM`; error variants
  `PlsKitError::OptimalNoComponent` and `SequenceNoRejection { alpha }`
  (an exhaustive `match` needs two new arms); `k_to_fit()` on the find-K
  outputs.
- Changed (behaviour): these inputs now raise `invalid_argument` instead of
  returning a degenerate result. Check for them before upgrading.
  - `pls1_confirmatory_test(method="raw_perm")` with `n_folds >= n` (was
    statistic `0`, `p = 1`).
  - `raw_perm` as `test_method` / `diagnostic` at `n <= 5`, through the
    `*_find_k_*` functions and `pls1_fit(k="sequence")`.
  - The CV selectors (`r2_se`, `r2_max`) at `n <= 2`, through
    `*_find_k_optimal`, `spls1_find_keep_optimal` and
    `pls1_fit(k="optimal")`. `bic` is unaffected.
  - `preprocess` with a 2-D `Y` whose row count does not match `X` or
    `weights` now raises `dimension_mismatch` / `invalid_weights`.
- Changed (behaviour): truncation floors are relative to the data. Every
  PLS1 component must clear `max(n, p)·ε·‖X‖_F·‖y‖`, every PLS3 component
  `max(n, p, q)·ε·‖X̃‖_F·‖Ỹ‖_F`, instead of an absolute `1e-14`. Fits no
  longer keep noise components once `y` is exhausted or on a
  rank-deficient `X'Y`, so `k_used` can be smaller (`n = 2000`, `p = 40` of
  rank 39, `k = 40`: 40 → 13). A `y` or `Y` orthogonal to `X` up to
  rounding gives `k_used = 0`, and then `*_find_k_optimal` returns
  `k_star = 0` and `spls1_find_keep_optimal` returns `keep_star = 0`. No
  corpus fixture changes.
- Changed: PLS1 fits use Improved Kernel PLS (Dayal and MacGregor 1997)
  instead of explicit deflation. Same model; one-component fits are
  bit-identical, multi-component fits may differ in the last bits.
- Changed: `pls1_fit(k="optimal" | "sequence")` raises
  `optimal_no_component` / `sequence_no_rejection` from the engine, not
  the Python wrapper. Codes unchanged, wording slightly different.
- Changed (Python): resampling defaults (`n_boot`, `m_rate`, `level`,
  `max_failure_rate`, `max_skip_rate`, `n_perm`, `alpha`, and `spls3_fit`'s
  `max_iter` / `tol`) are now `None`, resolving to the engine default.
  Values and results are unchanged.
- Changed (Rust only): `ConfirmatoryTestOpts::default()` selects
  `split_exact` (`n_perm = 1000`, `n_splits = 50`) instead of `split_nb`.
- Fixed: results for a fixed seed no longer depend on the Rayon pool size.
  faer split parallel products by pool size, so large fits (`n·d·k >= 1e6`
  for PLS1, `n·p·q >= 1e6` for PLS3) and everything built on them moved in
  the last bits with `RAYON_NUM_THREADS`. Every faer call now uses a fixed
  8-way split or `Par::Seq`; `tests/thread_count_parity.rs` requires
  bit-identical output at 1, 3 and 5 threads.
- Fixed: `ConfirmatoryCI` (the subsampling CI of `pls1_confirmatory_test`).
  - `beta_sign_z` is `|β_ref[j]| / beta_se[j]`, using the finite-population
    rate `√(m/(n − m))` and a correction for subsample shrinkage `κ̂`
    (`beta_sign_z_signed` keeps the sign). The old z flagged 63–92% of
    noise variables.
  - `beta_ci_*` / `beta_se` carry the same corrections: coverage at nominal
    95% rose from 77–89% to 91–96% at `K = 1`. β remains uncalibrated at
    `K ≥ 2`.
  - `leverage_*` come from an n-out-of-n bootstrap, with a normal interval
    clamped to `[0, 1]`: signal-variable coverage 0.92–0.97 at `K = 1`,
    0.92–0.98 at `K = 2, 3` (was 40–65% at `D = 20`).
  - The `pls1_confirmatory_*_ci` fixtures are regenerated.
- Fixed: scale invariance. Standardization's constant-column test and the
  split-half correlation (`split_exact`, `split_nb`,
  `pls3_confirmatory_test`) were absolute (`sd <= 1e-12`, `ss < 1e-15`)
  and overflowed or underflowed past about `1e±154`, so rescaling the data
  could zero a column or return statistic `0`. Both are now relative and
  exactly rescaled by a power of two; wherever the old sums stayed in
  range, outputs are bit-identical.
- Fixed: weighted `pls1_rotation_stability` standardized each resample with
  unweighted moments. Weighted results change; unweighted are identical.
- Fixed: the `split_nb` auto-gate uses one `n_eff` everywhere, so
  `*_find_k_sequence` and the confirmatory test agree at the floor.
- Fixed: the `split_exact` no-refit route at `k = 1`, the `raw_perm` Gram
  route at `k = 1`, and the sparse-Y Gram route of `pls3_confirmatory_test`
  now recompute on the primal route any case that sits within rounding of
  a decision threshold, so the route choice is again invisible.
- Fixed (Python): misaligned `float64` arrays were read in place
  (undefined behaviour in Rust); they are now copied.
- Performance: Gram routes for resampling loops, chosen from shape alone,
  with a primal refit for any replicate they cannot certify. A p-space
  `X'X` route serves `pls1_perm_null`, `raw_perm` and `split_exact` refits
  on tall data (dense, weighted, sparse); an n-space `X X'` route serves
  the same loops at `k <= 2` when `p ≫ n`. At a fixed seed, p-values are
  unchanged except where a null statistic lies within `1e-10` of the
  observed one.
- Performance: IKPLS reads `X` twice per component and never writes it;
  resampling loops no longer copy `X` per replicate; standardization is
  faster and bit-identical; Python no longer copies a C-ordered `X` before
  fitting. `pls1_fit` from Python now beats scikit-learn on all four rows
  of `scripts/bench_pls1_fit.py`'s default grid.
- Tests and docs: `tests/corpus.rs` checks every fixture family from Rust;
  three `spls3_fit` fixtures added. README examples use `k=1` with
  `split_exact`.

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
