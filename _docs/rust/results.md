# Result objects (Rust)

User-facing field shapes for each public Rust result struct. Field names
are snake_case and match across language wrappers; types follow Rust
convention. Containers are owned (`Mat<f64>` / `Col<f64>`) — the wrapper
copies into language-native arrays at the FFI boundary.

The matrix and column types are [faer](https://crates.io/crates/faer)
types, re-exported as `plskit::Mat`, `plskit::Col`, `plskit::MatRef`,
`plskit::ColRef`. Downstream Rust callers do not need a direct faer
dependency — `use plskit::{Mat, Col};` is sufficient. The choice to
expose faer types directly avoids round-trip allocations through
intermediate `Vec<Vec<f64>>` representations and keeps the result
objects usable with any faer-aware linear-algebra code.

The Rust core uses long-form snake_case identifiers. Wrappers may rename
short-form matrix names (`T`, `P`, `W`, `Q`) at the user-facing boundary
to match the standard PLS notation; the underlying field name is the
long form below.

## `PreprocessResult` — what `preprocess` returns

| Field | Rust type | Notes |
|---|---|---|
| `x_std` | `Option<(Mat<f64>, Col<f64>, Col<f64>)>` | `Some((X_std, X_mean, X_scale))` when `X` was passed |
| `y_std` | `Option<(Col<f64>, f64, f64)>` | `Some((y_std, y_mean, y_scale))` when `y` was passed |
| `weights_normalized` | `Option<Col<f64>>` | `Some(w')` (normalized to mean 1, Σ = n) when weights were passed |
| `n_eff` | `Option<f64>` | populated when weights were passed; `None` otherwise |

## `Pls1Model` — what `pls1_fit` returns

| Field | Rust type | Shape |
|---|---|---|
| `t_scores` | `Mat<f64>` | `(n, K)` |
| `p_loadings` | `Mat<f64>` | `(D, K)` |
| `w_star` | `Mat<f64>` | `(D, K)` |
| `q_loadings` | `Col<f64>` | `(K,)` |
| `coef` | `Col<f64>` | `(D,)` |
| `beta` | `Col<f64>` | `(D,)` |
| `intercept` | `f64` | scalar |
| `k_used` | `usize` | scalar |
| `pre_standardized` | `bool` | scalar |
| `weights` | `Option<Col<f64>>` | `(n,)` when present; `None` for uniform/absent weights |
| `n_eff` | `f64` | scalar; Kish's effective sample size, equals `n` for uniform/absent weights |
| `keep` | `Option<usize>` | `Some(keep)` for `spls1_fit`; `None` for `pls1_fit` |

There is no `k_was_auto` flag and no `find_k_certificate` field. The
2026-04 confirmatory-vs-exploratory overhaul moved K-selection
diagnostics onto the K-selection result structs themselves, where
they originate. Rotation is stamped wrapper-side: the Python
`PLS1Result` carries a `rotation_spec` field that `rotate(model, ...)`
sets on a copy of the model, but there is no Rust `RotationSpec` type
and no `rotation_spec` field on the Rust `Pls1Model` — it never
crosses the FFI seam.

## `Pls3Model` — what `pls3_fit` / `plssvd_fit` returns

| Field | Rust type | Shape |
|---|---|---|
| `u_saliences` | `Mat<f64>` | `(p, k_used)`, orthonormal columns |
| `v_saliences` | `Mat<f64>` | `(q, k_used)`, orthonormal columns |
| `singular_values` | `Col<f64>` | `(k_used,)`, descending |
| `x_scores` | `Mat<f64>` | `(n, k_used)` |
| `y_scores` | `Mat<f64>` | `(n, k_used)` |
| `x_mean` | `Col<f64>` | `(p,)`; zeros when `pre_standardized_x` |
| `x_scale` | `Col<f64>` | `(p,)`; ones when `pre_standardized_x` |
| `y_mean` | `Col<f64>` | `(q,)`; zeros when `pre_standardized_y` |
| `y_scale` | `Col<f64>` | `(q,)`; ones when `pre_standardized_y` |
| `k_used` | `usize` | scalar; `< k` when a singular value fell below `1e-14` |
| `pre_standardized_x` | `bool` | scalar |
| `pre_standardized_y` | `bool` | scalar |

There is no `beta`, no `coef` and no `intercept`: PLS3 is symmetric, so
there is nothing to regress. There is no `weights` / `n_eff` pair either —
observation weights are not implemented for this family.

## `Pls3Scores` — what `pls3_transform` / `plssvd_transform` returns

| Field | Rust type | When populated |
|---|---|---|
| `x_scores` | `Option<Mat<f64>>` | `which ∈ {XScores, Both}`; shape `(n_new, k_used)` |
| `y_scores` | `Option<Mat<f64>>` | `which ∈ {YScores, Both}`; shape `(n_new, k_used)` |

`pls3_confirmatory_test` returns the same `ConfirmatoryTestOutput` struct
`pls1_confirmatory_test` does. For PLS3 `ci` is always `None` and `n_eff`
equals `n`. `rho_hat` is populated for `split_nb` only, and only when the
test half has at least 4 rows; `stable_rank` is populated whenever
`split_nb` was requested.

## `FindKOptimalOutput` — what `pls1_find_k_optimal` returns

| Field | Rust type | When populated |
|---|---|---|
| `k_star` | `usize` | always |
| `selector` | `String` | always |
| `cv_scores` | `Option<BTreeMap<usize, f64>>` | `selector ∈ {r2_se, r2_max}` |
| `cv_scores_se` | `Option<BTreeMap<usize, f64>>` | `selector="r2_se"` only |
| `bic_scores` | `Option<BTreeMap<usize, f64>>` | `selector="bic"` |
| `pvalues` | `Option<Col<f64>>` | `diagnostic.is_some()` |
| `diagnostic` | `Option<String>` | `diagnostic.is_some()` |
| `seed` | `u64` | always |
| `n_eff` | `f64` | always; equals `n_samples` for uniform/absent weights |
| `stable_rank` | `Option<f64>` | `Some` when `diagnostic == Some(SplitNb)` — fired or not, including under `force` |

When `diagnostic` is set, `pvalues` carries the per-component p-values
of a same-sample sequential test up to `k_star`, and `diagnostic`
echoes the method name. Selection and the diagnostic share the same
data, so the pvalues are a robustness check, not honest inference — a
fresh sample is required for a confirmatory claim.

## `FindKSequenceOutput` — what `pls1_find_k_sequence` returns

| Field | Rust type | When populated |
|---|---|---|
| `k_star` | `usize` | always (0 if no component rejects at α) |
| `pvalues` | `Col<f64>` | always; trailing entries are `f64::NAN` if stop-early kicked in |
| `test_method` | `String` | always |
| `alpha` | `f64` | always |
| `seed` | `u64` | always |
| `n_eff` | `f64` | always; equals `n_samples` for uniform/absent weights |
| `stable_rank` | `Option<f64>` | `Some` when `test_method == SplitNb` — fired or not, including under `force` |

Closed testing on nested H is exact, so `pvalues[..k_star]` is an
honest FWER-controlled sequence. The path-max p-value is
`pvalues[..k_star].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))`.

On both result types `stable_rank` is what the auto-gate saw on the
undeflated `X`, so when the reported method reads `"split_exact"`
after a `split_nb` request, it and `n_eff` are the two numbers that
explain why. `split_nb_gate` returns them without running a test.

## `FindKeepOptimalOutput` — what `spls1_find_keep_optimal` returns

| Field | Rust type | Notes |
|---|---|---|
| `keep_star` | `usize` | sparsest `keep` within 1 SE of the best mean CV R² |
| `k` | `usize` | the fixed component count the sweep ran at |
| `cv_scores` | `BTreeMap<usize, f64>` | keep → mean CV R² across folds |
| `cv_scores_se` | `BTreeMap<usize, f64>` | keep → SE of the CV R² |
| `keep_grid` | `Vec<usize>` | the logged geometric grid swept (powers of two; endpoints 1 and n_features always included) |
| `seed` | `u64` | always |
| `n_eff` | `f64` | effective sample size; `f64::NAN` if unavailable |

Selection criterion: the 1-SE rule on mean CV R² — `keep_star` is the
sparsest keep whose mean CV R² is within 1 SE of the maximum. The full
field semantics are identical to the Python counterpart; see
[Python results](../python/results.md).

## `ConfirmatoryTestOutput` — what `pls1_confirmatory_test` returns

| Field | Rust type | Notes |
|---|---|---|
| `pvalue` | `f64` | always |
| `statistic` | `f64` | always |
| `method` | `String` | one of `"raw_perm"` / `"split_nb"` / `"split_exact"` / `"score"` / `"e"`; `"split_exact"` when a `"split_nb"` request was rerouted by the auto-gate |
| `k` | `usize` | the K tested (echoed from the input) |
| `n_perm` | `Option<usize>` | `Some` for resampling-family methods, `None` for `score` / `e` |
| `n_splits` | `Option<usize>` | `Some` for `split_*` methods, `None` for `raw_perm` / `score` / `e` |
| `seed` | `u64` | always |
| `n_eff` | `f64` | always; Kish effective sample size, equals `n_samples` for uniform/absent weights |
| `rho_hat` | `Option<f64>` | `Some` for `split_nb` only, and only when unweighted with a test half of at least 4 rows; `None` for every other method, including `split_exact` |
| `stable_rank` | `Option<f64>` | stable rank of the standardized `X`, as seen by the `split_nb` auto-gate; `Some` whenever `split_nb` was requested (fired or not, including under `force`), `None` for every other requested method |
| `ci` | `Option<ConfirmatoryCI>` | `Some` when called with `ci=true`; carries the rotation-invariant subsample CIs |

## `SplitNbGateOutput` — what `split_nb_gate` returns

| Field | Rust type | Notes |
|---|---|---|
| `fires` | `bool` | `true` → a `split_nb` request on this `X` reroutes to `split_exact` |
| `stable_rank` | `f64` | stable rank of the standardized `X` |
| `n_eff` | `f64` | Kish effective sample size; equals `n_samples` for uniform/absent weights |

Every field is always populated. This is the same rule the test
functions apply internally, on the same standardized `X` — querying it
costs one SVD and no resampling.

## `PermNullOutput` — what `pls1_perm_null` returns

| Field | Rust type | Notes |
|---|---|---|
| `n_perm` | `usize` | number of permutations actually run |
| `k` | `usize` | K used for fitting |
| `seed` | `u64` | RNG seed actually used |
| `n_eff` | `f64` | effective sample size (`sum(w)² / sum(w²)`); equals `n` when weights are uniform |
| `beta_ref` | `Vec<f64>` | full-data β reference, length D |
| `beta_perm_mean` | `Vec<f64>` | mean of β under permuted y, length D; ≈ 0 under H0 (calibration diagnostic) |
| `beta_perm_sd` | `Vec<f64>` | SD of β under permuted y, length D |
| `beta_perm_z` | `Vec<f64>` | signed per-voxel z = β_ref / β_perm_sd; NaN where SD ≈ 0, length D |
| `beta_perm_matrix` | `Option<Vec<f64>>` | optional `(n_perm, D)` β matrix, row-major; `Some` when `opts.return_perm_matrix == true` |

## `CIScalar` — scalar subsample CI

Centered-scaled subsampling CI for a scalar functional, plus its SD.

| Field | Rust type |
|---|---|
| `point` | `f64` |
| `lower` | `f64` |
| `upper` | `f64` |
| `sd` | `f64` |

## `ConfirmatoryCI` — what `pls1_confirmatory_test(ci=true)` adds

Rotation-invariant readouts only.

| Field | Rust type | Shape |
|---|---|---|
| `n_boot` | `usize` | scalar |
| `m` | `usize` | scalar (resolved subsample size, `m = ceil(n^m_rate)`) |
| `m_rate` | `f64` | scalar |
| `level` | `f64` | scalar |
| `beta_sign_z` | `Vec<f64>` | `(D,)` per-variable folded sign-stability z |
| `beta_sign_z_signed` | `Vec<f64>` | `(D,)` per-variable signed = `sign(β_ref[j]) · |beta_sign_z[j]|` |
| `leverage_ci_lower` | `Vec<f64>` | `(D,)` per-variable |
| `leverage_ci_upper` | `Vec<f64>` | `(D,)` per-variable |
| `leverage_se` | `Vec<f64>` | `(D,)` per-variable |
| `beta_ci_lower` | `Vec<f64>` | `(D,)` per-coordinate centered-scaled CI on β; PLS1-only diagnostic |
| `beta_ci_upper` | `Vec<f64>` | `(D,)` per-coordinate |
| `beta_se` | `Vec<f64>` | `(D,)` `= √(m/n) · sd(β_b[j])` |
| `holdout_corr` | `CIScalar` | scalar (Fisher z-transformed NB-Wald CI on out-of-sample predictive correlation) |
| `n_boot_finite` | `usize` | scalar; resamples whose worker fit succeeded (≤ `n_boot`) |
| `n_boot_finite_holdout_corr` | `usize` | scalar; subset whose holdout_corr is finite |

The full caveats on `beta_ci_*` (PLS shrinkage bias on small m, no
multiple-comparison correction, K=1-only theoretical asymptotic
normality, standardization-mode interaction, sign-z vs leverage-ci vs
beta-ci three-way distinction) are stated alongside the Python type
table — see [Python results](../python/results.md). The semantics are
identical.

## `RotateOutput` — what `rotate` returns

| Field | Rust type | Notes |
|---|---|---|
| `w_rot` | `Mat<f64>` | rotated weights `W @ R`, shape `(D, K)` |
| `r` | `Mat<f64>` | orthogonal rotation, shape `(K, K)`; `w_rot = w @ r` |
| `sweeps` | `usize` | number of Kaiser sweeps actually run (≤ `args.max_iter`) |
| `v_converged` | `f64` | final value of the varimax criterion `V = Σ_j Var(target[:, j]²)` |

## `RotationStabilityOutput` — what `pls1_rotation_stability` returns

| Field | Rust type | Notes |
|---|---|---|
| `method` | `String` | the rotation method used |
| `n_boot` | `usize` | resolved subsample iterations |
| `m` | `usize` | resolved subsample size |
| `m_rate` | `f64` | echoed from the input |
| `level` | `f64` | echoed from the input |
| `seed` | `u64` | always |
| `variance_ratio` | `CIScalar` | headline aggregate variance ratio `ρ = V_rot / V_unrot` with paired-bootstrap percentile CI |
| `variance_ratio_per_axis` | `Vec<CIScalar>` | per-axis ratio `ρ_k = V_rot,k / V_unrot,k`, length K, reference-axis order |
| `variance_unrot` | `f64` | aggregate `V_unrot = (1/B) Σ_b Σ_k α²_unrot,b,k` |
| `variance_rot` | `f64` | aggregate `V_rot = (1/B) Σ_b Σ_k α²_rot,b,k` |
| `variance_unrot_per_axis` | `Vec<f64>` | per-axis `V_unrot,k`, length K, reference-axis order |
| `variance_rot_per_axis` | `Vec<f64>` | per-axis `V_rot,k`, length K, reference-axis order |
| `degenerate_baseline` | `bool` | `true` iff `V_unrot = 0` on the engine pass or more than 5% of bootstrap iterations had `V_unrot* = 0`; when set, `variance_ratio.point` is `NaN` |
| `n_boot_finite` | `usize` | number of resamples that produced finite per-axis squared residuals (≤ `n_boot`) |
| `n_eff` | `f64` | effective sample size `(Σ wᵢ)² / Σ wᵢ²` from the full normalized weight vector; equals `n` for uniform weights |

`RotationSpec` (the record of a `rotate(model, ...)` call — method,
args, `R`, sweeps, convergence value) exists only on the Python
`PLS1Result` dataclass. There is no Rust `RotationSpec` type and no
Rust counterpart on `Pls1Model` or `RotationStabilityOutput`; the Rust
`rotate` function returns `RotateOutput` (`w_rot, r, sweeps,
v_converged`), and the Python wrapper assembles `RotationSpec` from
that plus the caller's method/args.
