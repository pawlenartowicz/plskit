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
| `rotation_spec` | `Option<RotationSpec>` | — |
| `keep` | `Option<usize>` | `Some(keep)` for `spls1_fit`; `None` for `pls1_fit` |

There is no `k_was_auto` flag and no `find_k_certificate` field. The
2026-04 confirmatory-vs-exploratory overhaul moved K-selection
diagnostics onto the K-selection result structs themselves, where
they originate. `rotation_spec` is `None` until `rotate(model, ...)`
stamps it onto a copy of the model.

## `FindKOptimalResult` — what `pls1_find_k_optimal` returns

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

When `diagnostic` is set, `pvalues` carries the per-component p-values
of a same-sample sequential test up to `k_star`, and `diagnostic`
echoes the method name. Selection and the diagnostic share the same
data, so the pvalues are a robustness check, not honest inference — a
fresh sample is required for a confirmatory claim.

## `FindKSequenceResult` — what `pls1_find_k_sequence` returns

| Field | Rust type | When populated |
|---|---|---|
| `k_star` | `usize` | always (0 if no component rejects at α) |
| `pvalues` | `Col<f64>` | always; trailing entries are `f64::NAN` if stop-early kicked in |
| `test_method` | `String` | always |
| `alpha` | `f64` | always |
| `seed` | `u64` | always |

Closed testing on nested H is exact, so `pvalues[..k_star]` is an
honest FWER-controlled sequence. The path-max p-value is
`pvalues[..k_star].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))`.

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

## `ConfirmatoryTestResult` — what `pls1_confirmatory_test` returns

| Field | Rust type | Notes |
|---|---|---|
| `pvalue` | `f64` | always |
| `statistic` | `f64` | always |
| `method` | `String` | one of `"raw_perm"` / `"split_nb"` / `"split_perm"` / `"score"` / `"e"` |
| `k` | `usize` | the K tested (echoed from the input) |
| `n_perm` | `Option<usize>` | `Some` for resampling-family methods, `None` for `score` / `e` |
| `n_splits` | `Option<usize>` | `Some` for `split_*` methods, `None` for `raw_perm` / `score` / `e` |
| `seed` | `u64` | always |
| `ci` | `Option<ConfirmatoryCI>` | `Some` when called with `ci=true`; carries the rotation-invariant subsample CIs |

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

## `RotationStabilityResult` — what `pls1_rotation_stability` returns

| Field | Rust type | Notes |
|---|---|---|
| `method` | `String` | the rotation method used |
| `n_boot` | `usize` | resolved subsample iterations |
| `m` | `usize` | resolved subsample size |
| `m_rate` | `f64` | echoed from the input |
| `level` | `f64` | echoed from the input |
| `seed` | `u64` | always |
| `agreement` | `CIScalar` | post-procrustes Frobenius CI; `agreement.point` is `0.0` by construction (full-data fit aligns to itself), so the CI width is the diagnostic |

## `RotationSpec` — stamped by `rotate(model, ...)`

| Field | Rust type | Notes |
|---|---|---|
| `method` | `String` | `"varimax"` today; future `"promax"` / `"oblimin"` / `"geomin"` |
| `args` | `RotationArgs` (enum) | method-specific kwargs used at rotate-time |
| `R` | `Mat<f64>` | `(K, K)` rotation matrix; `W_rot = W * R` |
| `sweeps` | `usize` | varimax iterations to convergence |
| `V_converged` | `f64` | final varimax criterion value |
| `L_was_provided` | `bool` | whether caller passed a loading basis |

`rotation_spec` is `None` until `rotate(model, ...)` stamps it on a
copy of the model.
