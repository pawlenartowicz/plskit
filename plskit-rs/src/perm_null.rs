//! Permutation-null engine for signed per-voxel z statistics on PLS1 β.
//!
//! Produces a signed per-feature test statistic suitable for downstream
//! TFCE / cluster-mass / max-stat FWER pipelines (PALM, FSL randomise,
//! nltools). Independent from the subsampling engine in `subsample.rs` —
//! different question (null distribution under permuted y, not sampling
//! distribution under the true DGP) and different per-resample workload
//! (full-size fit on permuted y, no procrustes alignment).

use faer::{ColRef, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::signal_test::ReplicateRoute;

/// Tuning knobs for `pls1_perm_null`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct PermNullOpts {
    /// Number of permutations. Must be ≥ 100 (per-voxel-z noise floor).
    pub n_perm: usize,
    /// Whether to return the full `(n_perm, D)` β-matrix on the output.
    /// The b×d buffer is always materialized — the two-pass reduction over it
    /// is the byte-exact reduction path (pinned by `streaming_matches_retained_byte_exact`),
    /// so no true single-pass Welford accumulator exists; this flag only
    /// controls whether that buffer is handed back to the caller.
    pub return_perm_matrix: bool,
    /// Caller asserts X is already column-standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
}

impl Default for PermNullOpts {
    fn default() -> Self {
        Self {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
        }
    }
}

impl PermNullOpts {
    /// Validate args (`n_perm` ≥ 100, k ≥ 1).
    ///
    /// # Errors
    ///
    /// Returns `PlsKitError::InvalidArgument` if `n_perm < 100` or `k < 1`.
    pub fn validate(&self, k: usize) -> PlsKitResult<()> {
        if self.n_perm < 100 {
            return Err(PlsKitError::InvalidArgument(format!(
                "n_perm must be ≥ 100, got {}",
                self.n_perm
            )));
        }
        if k < 1 {
            return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
        }
        Ok(())
    }
}

/// Output of `pls1_perm_null`. All per-voxel arrays length `D`.
#[derive(Debug, Clone)]
#[allow(clippy::doc_markdown)]
pub struct PermNullOutput {
    /// Number of permutations actually run.
    pub n_perm: usize,
    /// K used for fitting.
    pub k: usize,
    /// RNG seed actually used.
    pub seed: u64,
    /// Effective sample size (sum(w)² / sum(w²)); equals n when weights are uniform.
    pub n_eff: f64,
    /// Full-data β reference. Length D.
    pub beta_ref: Vec<f64>,
    /// Mean of β under permuted y. Length D. ≈ 0 under H0 (calibration diagnostic).
    pub beta_perm_mean: Vec<f64>,
    /// SD of β under permuted y. Length D.
    pub beta_perm_sd: Vec<f64>,
    /// Signed per-voxel z = β_ref / β_perm_sd. NaN where SD ≈ 0. Length D.
    pub beta_perm_z: Vec<f64>,
    /// Optional `(n_perm, D)` β matrix in row-major layout (length n_perm·D).
    /// Some when `opts.return_perm_matrix == true`.
    pub beta_perm_matrix: Option<Vec<f64>>,
}

/// Permutation-null engine for PLS1 β. See module docs.
///
/// # Shapes
/// - `x`: `(n, d)`
/// - `y`: `(n,)`
/// - `k`: components retained per fit; `1 ≤ k ≤ d`
///
/// # Errors
/// - `PlsKitError::InvalidArgument` when `n_perm < 100` or `k < 1`
/// - `PlsKitError::DimensionMismatch` when `y.len() != x.nrows()`
/// - `PlsKitError::KExceedsMax` when `k > d`
/// - `PlsKitError::NonFiniteInput` when X or y contains NaN/inf
/// - `PlsKitError::InvalidWeights` when weights are invalid
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::needless_pass_by_value, clippy::many_single_char_names)]
pub fn pls1_perm_null(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: PermNullOpts,
    seed: Option<u64>,
) -> PlsKitResult<PermNullOutput> {
    crate::fit::with_thread_limit(|| perm_null_impl(x, y, k, weights, opts, seed))
}

/// Body of [`pls1_perm_null`], on the caller's pool.
#[allow(clippy::needless_pass_by_value, clippy::many_single_char_names)]
pub(crate) fn perm_null_impl(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: PermNullOpts,
    seed: Option<u64>,
) -> PlsKitResult<PermNullOutput> {
    use crate::fit::{
        fit_row_scale, pls1_fit_impl, scale_rows, validate_weights_for_k, FitOpts, KSpec,
    };
    use crate::linalg::{standardize, standardize1, standardize1_weighted, standardize_weighted};
    use faer::{Col, Mat};

    opts.validate(k)?;

    let n = x.nrows();
    let d = x.ncols();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, d),
            y: y.nrows(),
        });
    }
    if k > d {
        return Err(PlsKitError::KExceedsMax { k, k_max: d });
    }
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_col(y)?;

    let (w_norm, n_eff_val) = validate_weights_for_k(weights, n, k)?;
    let wref = w_norm.as_ref().map(Col::as_ref);

    // Standardize once. Subsequent permutations operate on standardized arrays
    // — permuting y after standardization is equivalent to permuting raw y and
    // re-standardizing because mean/scale are permutation-invariant.
    //
    // Convention: weighted moments when weights are present, matching pls1_fit's
    // own path (standardize_weighted, fit.rs). beta_ref is taken from a reference
    // fit run on these pre-standardized arrays, so it equals the standardized-scale
    // coefficient (Pls1Model.coef) of a direct weighted pls1_fit on the same input
    // — verified to f64 epsilon. (Not the raw-scale Pls1Model.beta, which fit
    // back-projects by x_scale/y_scale; beta_ref stays on the standardized scale,
    // which is the scale the per-voxel z statistic is defined on.) The
    // y-permutation argument still holds: the weighted moments depend on w, and w
    // is NOT permuted (it stays tied to row i), so the transform applied to
    // observed and permuted y is identical and the null is internally consistent.
    //
    // Under `pre_standardized` the caller's X is used as given: a
    // column-major view is borrowed and any other layout is copied into
    // column-major storage (`linalg::col_major_or_copy` says why the layout
    // matters).
    let (xs_owned, ys_owned): (Option<Mat<f64>>, Col<f64>) = if opts.pre_standardized {
        (
            crate::linalg::col_major_or_copy(x),
            Col::<f64>::from_fn(n, |i| y[i]),
        )
    } else if wref.is_some() {
        // wref are the same mean-1 normalized weights pls1_fit standardizes with.
        let (xs, _, _) = standardize_weighted(x, wref);
        let (ys, _, _) = standardize1_weighted(y, wref);
        (Some(xs), ys)
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (Some(xs), ys)
    };
    let xs: MatRef<'_, f64> = xs_owned.as_ref().map_or(x, Mat::as_ref);
    let ys = ys_owned.as_ref();

    // Reference fit on full standardized data.
    let fit_ref = pls1_fit_impl(
        xs,
        ys,
        KSpec::Fixed(k),
        wref,
        FitOpts {
            pre_standardized: true,
            // check_n_eff: false — n_eff was already validated at the top-level
            // entry; truncation here just yields β at k_used (fit at what exists).
            check_n_eff: false,
            ..FitOpts::default()
        },
    )?;
    let beta_ref: Vec<f64> = (0..d).map(|j| fit_ref.beta[j]).collect();

    // Weights stay tied to rows under every permutation, so the √w row
    // scaling `pls1_fit` applies to X̃ is the same matrix in every replicate:
    // build it once. The factor is `pls1_fit`'s own (it renormalizes the
    // weights it is handed), so it is √(normalize_weights(w')), not √w'.
    // Once the scaled copy exists nothing reads the standardized one, so it
    // is freed here: a single n×d buffer lives through the permutation loop.
    let sqw: Option<Col<f64>> = wref.map(fit_row_scale);
    let xs_fit_owned: Option<Mat<f64>> = match sqw.as_ref() {
        Some(s) => {
            let scaled = scale_rows(xs, s.as_ref());
            drop(xs_owned);
            Some(scaled)
        }
        None => xs_owned,
    };
    let xs_fit: MatRef<'_, f64> = xs_fit_owned.as_ref().map_or(x, Mat::as_ref);

    let (seed_used, mut rng) = crate::rng::resolve_seed(seed)?;

    // Route choice: `perm_null_route` owns the rule. Decided here, on the
    // calling thread, before any parallel work.
    let route = perm_null_route(n, d, opts.n_perm, k, wref.is_some());

    run_engine(
        route,
        xs_fit,
        ys,
        k,
        sqw.as_ref().map(Col::as_ref),
        beta_ref,
        n_eff_val,
        opts,
        seed_used,
        &mut rng,
    )
}

// The row driver of the permutation null: the route's block is built once,
// then one row per permutation is written in place into the single
// row-major (B, D) buffer the two-pass reduction reads (no single-pass
// Welford path exists; `streaming_matches_retained_byte_exact` pins the two
// outputs equal). `opts.return_perm_matrix` only decides whether that buffer
// is returned. Returns Result to leave room for per-permutation hard
// failures in future revisions.
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    clippy::many_single_char_names
)]
fn run_engine(
    route: ReplicateRoute,
    xs_fit: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    sqw: Option<faer::ColRef<'_, f64>>,
    beta_ref: Vec<f64>,
    n_eff_val: f64,
    opts: PermNullOpts,
    seed_used: u64,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<PermNullOutput> {
    debug_assert!(k <= xs_fit.ncols(), "k <= p is the caller's precondition");
    let n = xs_fit.nrows();
    let d = xs_fit.ncols();
    let b = opts.n_perm;
    let block = perm_block(route, xs_fit, crate::fit::par_fixed());

    // One row per permutation, written in place into the single row-major
    // (B, D) buffer; a failed permutation leaves a NaN row (fail-soft), the
    // whole row overwritten whatever the unit wrote before failing. This
    // buffer is the dominant allocation (~0.8–8 GB at D = 1e5–1e6, B ≥ 1e3,
    // the fMRI target scale), held once: a deliberate
    // determinism-over-memory trade, since no streaming Welford accumulator
    // exists.
    let flat = crate::resample::parallel_fill_rows_seeded(rng, b, d, |_, child, row| {
        // The permutation is the first draw off the child stream, as in
        // the replicate-by-replicate worker.
        let perm = crate::resample::permute_indices(n, child);
        if perm_row(&block, xs_fit, sqw, ys, &perm, k, row).is_err() {
            row.fill(f64::NAN);
        }
    });

    // Two-pass per-column reduction.
    let (beta_perm_mean, beta_perm_sd) = reduce_two_pass(&flat, b, d);
    let beta_perm_z = signed_z(&beta_ref, &beta_perm_sd);

    Ok(PermNullOutput {
        n_perm: b,
        k,
        seed: seed_used,
        n_eff: n_eff_val,
        beta_ref,
        beta_perm_mean,
        beta_perm_sd,
        beta_perm_z,
        beta_perm_matrix: if opts.return_perm_matrix {
            Some(flat)
        } else {
            None
        },
    })
}

/// Two-pass mean / SD over a flat row-major (B, D) buffer.
/// Skips NaN rows in both numerator and denominator (failed permutations).
fn reduce_two_pass(flat: &[f64], b: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0_f64; d];
    let mut count = vec![0_usize; d];
    for bi in 0..b {
        let off = bi * d;
        for j in 0..d {
            let v = flat[off + j];
            if v.is_finite() {
                mean[j] += v;
                count[j] += 1;
            }
        }
    }
    for j in 0..d {
        if count[j] > 0 {
            mean[j] /= count[j] as f64;
        }
    }
    let mut m2 = vec![0.0_f64; d];
    for bi in 0..b {
        let off = bi * d;
        for j in 0..d {
            let v = flat[off + j];
            if v.is_finite() {
                let dv = v - mean[j];
                m2[j] += dv * dv;
            }
        }
    }
    let sd: Vec<f64> = (0..d)
        .map(|j| {
            if count[j] > 1 {
                (m2[j] / (count[j] - 1) as f64).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    (mean, sd)
}

/// β_ref / β_perm_sd, NaN-guarded with ε = √f64::EPSILON.
#[allow(clippy::doc_markdown)]
fn signed_z(beta_ref: &[f64], sd: &[f64]) -> Vec<f64> {
    let eps = f64::EPSILON.sqrt();
    beta_ref
        .iter()
        .zip(sd.iter())
        .map(|(b, s)| if *s > eps { b / s } else { f64::NAN })
        .collect()
}

/// `pls1_perm_null`'s selector, called once per call on the calling thread
/// before any parallel work. `Nspace` for dense, unweighted input the
/// n-space rule admits (`dual_route::nspace_eligible_perm_null`), unless
/// `gram_routes_disabled()`. `GramP` when `gram_p::gram_p_eligible` admits
/// the block (tried after `Nspace`: in the overlap band
/// `n_tr < p < 1.54·n_tr` the n-space route is cheaper), unless
/// `gram_routes_disabled()`; its replicate count is `n_perm`, since the
/// observed fit stays on the Primal route. `Primal` otherwise.
pub(crate) fn perm_null_route(
    n: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    weighted: bool,
) -> ReplicateRoute {
    if crate::signal_test::gram_routes_disabled() {
        return ReplicateRoute::Primal;
    }
    if crate::dual_route::nspace_eligible_perm_null(n, p, n_perm, k, !weighted) {
        ReplicateRoute::Nspace
    } else if crate::gram_p::gram_p_eligible(n, p, n_perm, k, None) {
        ReplicateRoute::GramP
    } else {
        ReplicateRoute::Primal
    }
}

/// Per-call state of a route, built once by `run_engine` and shared
/// read-only by every permutation row. The primal route needs only
/// `‖xs_fit‖_F`.
enum PermBlock<'x> {
    /// `x_fro = ‖xs_fit‖_F` (`norm_l2` of the view), the norm the kernel's
    /// truncation floor reads. Taken once per call instead of once per row:
    /// the same call on the same view, so the same bits.
    Primal {
        /// `xs_fit.norm_l2()`.
        x_fro: f64,
    },
    /// n-space Gram route: `G = X̃X̃'` with its norms, built once per call on
    /// the calling thread.
    Nspace(crate::dual_route::NspaceGram),
    /// The p-space Gram block of `xs_fit`, built once per call.
    GramP(crate::gram_p::GramPBlock<'x>),
}

/// The block of `route` for the prepared `xs_fit`, its precompute run under
/// `par` (`fit::par_fixed()`). `Special` is not a `perm_null` route;
/// handed it, the builder returns the primal block. `Nspace` builds `G` and
/// its norms once; `GramP` builds `C = X̃'X̃` and its norm estimate once;
/// `Primal` takes `‖xs_fit‖_F` once.
fn perm_block(route: ReplicateRoute, xs_fit: MatRef<'_, f64>, par: faer::Par) -> PermBlock<'_> {
    match route {
        ReplicateRoute::Nspace => {
            PermBlock::Nspace(crate::dual_route::NspaceGram::new(xs_fit, par))
        }
        ReplicateRoute::GramP => PermBlock::GramP(crate::gram_p::GramPBlock::new(xs_fit, par)),
        ReplicateRoute::Special | ReplicateRoute::Primal => PermBlock::Primal {
            x_fro: xs_fit.norm_l2(),
        },
    }
}

/// One permutation row: β of the refit on the permuted standardized outcome
/// `ys_std[perm[i]]` (scaled by `sqw` when weighted), written into `out`
/// (length d). `xs_fit` is standardized and, when weighted, √w-scaled.
/// Weights are NOT permuted: `w[i]` stays tied to row `i` whichever `y`
/// value lands there (key invariant), which is why `xs_fit` can be shared.
/// Every input check `pls1_fit` would make already passed on the reference
/// fit (same X, a permutation of the same y, same `k` and weights), so the
/// primal arm runs the kernel tail directly, on the block's `‖xs_fit‖_F`;
/// β is `coef` because the arrays are pre-standardized.
///
/// # Errors
/// What the kernel returns; `run_engine` then writes a NaN row.
fn perm_row(
    block: &PermBlock<'_>,
    xs_fit: MatRef<'_, f64>,
    sqw: Option<ColRef<'_, f64>>,
    ys_std: ColRef<'_, f64>,
    perm: &[usize],
    k: usize,
    out: &mut [f64],
) -> PlsKitResult<()> {
    use crate::fit::{pls1_fit_prepared_fro, scale_col, ParChoice};
    match block {
        &PermBlock::Primal { x_fro } => {
            let n = xs_fit.nrows();
            let y_perm = faer::Col::<f64>::from_fn(n, |i| ys_std[perm[i]]);
            // Seq inside the per-permutation worker: outer Rayon owns the threadpool.
            let fit = match sqw {
                Some(s) => {
                    let ys_fit = scale_col(y_perm.as_ref(), s);
                    pls1_fit_prepared_fro(xs_fit, ys_fit.as_ref(), k, None, ParChoice::Seq, x_fro)?
                }
                None => {
                    pls1_fit_prepared_fro(xs_fit, y_perm.as_ref(), k, None, ParChoice::Seq, x_fro)?
                }
            };
            for (j, v) in out.iter_mut().enumerate() {
                *v = fit.coef[j];
            }
            Ok(())
        }
        PermBlock::Nspace(gram) => {
            debug_assert!(k >= 1 && sqw.is_none());
            if crate::dual_route::nspace_perm_row(gram, xs_fit, ys_std, perm, k, out) {
                Ok(())
            } else {
                // Unresolved: the Primal arm's body, call for call.
                perm_row(
                    &PermBlock::Primal {
                        x_fro: gram.x_fro(),
                    },
                    xs_fit,
                    sqw,
                    ys_std,
                    perm,
                    k,
                    out,
                )
            }
        }
        PermBlock::GramP(gram) => {
            debug_assert!(k >= 1);
            if let Some(coef) = perm_row_coef(gram, sqw, ys_std, perm, k) {
                for (j, v) in out.iter_mut().enumerate() {
                    *v = coef[j];
                }
                return Ok(());
            }
            perm_row(
                &PermBlock::Primal {
                    x_fro: gram.x_fro(),
                },
                xs_fit,
                sqw,
                ys_std,
                perm,
                k,
                out,
            )
        }
    }
}

/// The Gram-p arm's fit of one permutation row: the fit target
/// the Primal arm builds (`ys_std` permuted by `perm`, then `scale_col`-ed by
/// `sqw` when weighted, weights tied to rows), fitted on the block's `C`.
/// `None` when a gate refuses the replicate; the arm then runs the Primal
/// arm's body, so the row is the Primal route's to the bit. A non-finite
/// target fails the `tt` gate (every gate is an `a >= b` conjunction) and
/// reaches that body too. `pre_standardized = true` inside the loop, so the
/// row is `coef`.
fn perm_row_coef(
    gram: &crate::gram_p::GramPBlock<'_>,
    sqw: Option<ColRef<'_, f64>>,
    ys_std: ColRef<'_, f64>,
    perm: &[usize],
    k: usize,
) -> Option<faer::Col<f64>> {
    let n = ys_std.nrows();
    let y_perm = faer::Col::<f64>::from_fn(n, |i| ys_std[perm[i]]);
    let ys_fit = match sqw {
        Some(s) => crate::fit::scale_col(y_perm.as_ref(), s),
        None => y_perm,
    };
    gram.fit_replicate(ys_fit.as_ref(), k, None)
        .map(|(coef, _)| coef)
}

#[cfg(test)]
mod layout_invariance {
    use super::*;
    use crate::test_support::{assert_bits_eq, copy_free_families, for_each_layout};
    use faer::Col;

    fn assert_out_bits(a: &PermNullOutput, b: &PermNullOutput, what: &str) {
        assert_eq!(
            (a.n_perm, a.k, a.seed),
            (b.n_perm, b.k, b.seed),
            "{what}: counts"
        );
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "{what}.n_eff");
        assert_bits_eq(&a.beta_ref, &b.beta_ref, &format!("{what}.beta_ref"));
        assert_bits_eq(
            &a.beta_perm_mean,
            &b.beta_perm_mean,
            &format!("{what}.beta_perm_mean"),
        );
        assert_bits_eq(
            &a.beta_perm_sd,
            &b.beta_perm_sd,
            &format!("{what}.beta_perm_sd"),
        );
        assert_bits_eq(
            &a.beta_perm_z,
            &b.beta_perm_z,
            &format!("{what}.beta_perm_z"),
        );
        match (&a.beta_perm_matrix, &b.beta_perm_matrix) {
            (Some(x), Some(y)) => assert_bits_eq(x, y, &format!("{what}.beta_perm_matrix")),
            (None, None) => {}
            _ => panic!("{what}: beta_perm_matrix presence differs"),
        }
    }

    /// `pls1_perm_null` copies X column-major before the loop (standardizing,
    /// √w-scaling, or `col_major_or_copy` under `pre_standardized`), so every
    /// memory layout of X gives the owned column-major matrix's bits, on the
    /// Primal route and (the unweighted `wide` family) the n-space route.
    #[test]
    fn pls1_perm_null_is_bit_identical_across_layouts() {
        for f in copy_free_families() {
            let wr = f.w.as_ref().map(Col::as_ref);
            for pre in [false, true] {
                let (x0, y0) = f.inputs(pre);
                let opts = PermNullOpts {
                    n_perm: 100,
                    return_perm_matrix: true,
                    pre_standardized: pre,
                    verbose: false,
                };
                for_each_layout(
                    &x0,
                    |_, xv| pls1_perm_null(xv, y0.as_ref(), 2, wr, opts, Some(31)).unwrap(),
                    |view, got, want| {
                        assert_out_bits(got, want, &format!("{} {view} pre={pre}", f.name));
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests_engine_streaming {
    use super::*;
    use crate::test_support::{bits, synth};

    #[test]
    fn streaming_matches_retained_byte_exact() {
        // Both paths fill the same flat buffer (parallel_fill_rows_seeded) and
        // run the two-pass reduce over it; only retained returns the matrix.
        let (x, y) = synth(100, 5, 2, 4.0, 42);
        let opts_retained = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: false,
            verbose: false,
        };
        let opts_streaming = PermNullOpts {
            return_perm_matrix: false,
            ..opts_retained
        };
        let r1 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_retained, Some(99)).unwrap();
        let r2 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_streaming, Some(99)).unwrap();
        assert_eq!((r1.n_perm, r1.k), (200, 2));
        assert_eq!(r1.beta_perm_matrix.as_ref().map(Vec::len), Some(200 * 5));
        assert!(r2.beta_perm_matrix.is_none());
        assert_eq!(
            bits(&r1.beta_perm_mean),
            bits(&r2.beta_perm_mean),
            "beta_perm_mean must be byte-exact between retained and streaming"
        );
        assert_eq!(
            bits(&r1.beta_perm_sd),
            bits(&r2.beta_perm_sd),
            "beta_perm_sd must be byte-exact between retained and streaming"
        );
        assert_eq!(
            bits(&r1.beta_perm_z),
            bits(&r2.beta_perm_z),
            "beta_perm_z must be byte-exact between retained and streaming"
        );
    }
}

#[cfg(test)]
mod tests_calibration {
    use super::*;
    use crate::test_support::synth;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    /// Planted sparse signal in feature 0 with positive sign.
    fn synth_h1_signed(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| 2.0 * x[(i, 0)] + 0.3 * rng.random_range(-1.0..1.0));
        (x, y)
    }

    #[test]
    fn h0_mean_perm_close_to_zero() {
        // Under H0, β under permuted y has population mean 0; sampling SD scales like 1/√B.
        let (x, y) = synth(80, 5, 0, 0.0, 1);
        let opts = PermNullOpts {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, opts, Some(7)).unwrap();
        // Tolerance: 5σ band around 0 with σ ≈ sd/√B. `sd[j]/√1000 * 5` ~ generous.
        for j in 0..5 {
            let band = 5.0 * out.beta_perm_sd[j] / (1000.0_f64).sqrt();
            assert!(
                out.beta_perm_mean[j].abs() < band,
                "mean_perm[{j}] = {} exceeds band {}",
                out.beta_perm_mean[j],
                band,
            );
        }
    }

    #[test]
    fn h0_uncorrected_fpr_close_to_alpha() {
        // Average across many features and a few seeds: fraction of |z| > 1.96
        // should land near 0.05. Use a loose Monte-Carlo band (D × n_seeds is small,
        // so the test is loose but should catch order-of-magnitude regressions).
        let mut total = 0_usize;
        let mut rejects = 0_usize;
        for seed in 0..3_u64 {
            let (x, y) = synth(80, 30, 0, 0.0, seed * 17 + 3);
            let opts = PermNullOpts {
                n_perm: 500,
                return_perm_matrix: false,
                pre_standardized: false,
                verbose: false,
            };
            let out =
                pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, opts, Some(seed * 11 + 7)).unwrap();
            for &z in &out.beta_perm_z {
                if z.is_finite() {
                    total += 1;
                    if z.abs() > 1.96 {
                        rejects += 1;
                    }
                }
            }
        }
        let fpr = rejects as f64 / total as f64;
        // σ ≈ √(0.05·0.95/total) ≈ 0.023 at total = 90, so [0.01, 0.15] spans
        // about −1.7σ to +4.4σ around 0.05. Loose but informative.
        assert!(
            (0.01..=0.15).contains(&fpr),
            "FPR={fpr} (rejects={rejects} / total={total}) outside [0.01, 0.15]",
        );
    }

    #[test]
    fn h1_signed_signal_recovered() {
        let (x, y) = synth_h1_signed(150, 8, 23);
        let opts = PermNullOpts {
            n_perm: 500,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, opts, Some(31)).unwrap();
        // Feature 0 carries the planted signal with positive sign.
        assert!(
            out.beta_perm_z[0] > 0.0,
            "z[0] = {} not positive",
            out.beta_perm_z[0]
        );
        let abs_z: Vec<f64> = out.beta_perm_z.iter().map(|z| z.abs()).collect();
        let max_other = abs_z[1..].iter().copied().fold(0.0_f64, f64::max);
        assert!(
            abs_z[0] > max_other,
            "|z[0]|={} not larger than max |z[1..]|={}",
            abs_z[0],
            max_other,
        );
    }
}

#[cfg(test)]
mod tests_validation {
    use super::*;
    use faer::{Col, Mat};

    #[test]
    fn rejects_dim_mismatch() {
        let x = Mat::<f64>::zeros(10, 5);
        let y = Col::<f64>::zeros(9);
        let opts = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "dimension_mismatch");
    }

    #[test]
    fn rejects_k_exceeds_max() {
        let x = Mat::<f64>::zeros(20, 4);
        let y = Col::<f64>::zeros(20);
        let opts = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 5, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "k_exceeds_max");
    }

    #[test]
    fn rejects_low_n_perm() {
        // `opts.validate(k)` runs first, so the all-zero X is never reached.
        let x = Mat::<f64>::zeros(20, 4);
        let y = Col::<f64>::zeros(20);
        let opts = PermNullOpts {
            n_perm: 50,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(format!("{err}").contains("n_perm must be"), "{err}");
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests_nspace {
    use super::*;
    use crate::dual_route::{nspace_blocks_built, K_DUAL_MAX};
    use crate::linalg::{standardize, standardize1};
    use crate::signal_test::with_gram_routes_disabled;
    use crate::test_support::{assert_rows_close, assert_summaries_close, bits, synth};
    use faer::{Col, Mat};

    fn opts() -> PermNullOpts {
        PermNullOpts {
            n_perm: 100,
            return_perm_matrix: true,
            pre_standardized: false,
            verbose: false,
        }
    }

    fn run(x: &Mat<f64>, y: &Col<f64>, k: usize, o: PermNullOpts) -> PermNullOutput {
        pls1_perm_null(x.as_ref(), y.as_ref(), k, None, o, Some(17)).expect("perm_null")
    }

    /// The whole B×D matrix bit for bit: the rows a fallback computes are
    /// the primal route's exactly.
    fn assert_matrix_bits_eq(a: &PermNullOutput, b: &PermNullOutput, what: &str) {
        assert_eq!(
            bits(a.beta_perm_matrix.as_deref().expect("matrix")),
            bits(b.beta_perm_matrix.as_deref().expect("matrix")),
            "{what}: every row falls back, so every bit is the primal's"
        );
    }

    #[test]
    fn the_public_call_builds_one_nspace_block() {
        let (x, y) = synth(40, 2000, 1, 1.0, 1);
        let before = nspace_blocks_built();
        let _ = run(&x, &y, 2, opts());
        assert_eq!(
            nspace_blocks_built(),
            before + 1,
            "the n-space route must run"
        );
        let before = nspace_blocks_built();
        let _ = with_gram_routes_disabled(|| run(&x, &y, 2, opts()));
        assert_eq!(nspace_blocks_built(), before, "override: primal only");
        let w = Col::<f64>::from_fn(40, |i| 0.5 + (i % 3) as f64 * 0.5);
        let before = nspace_blocks_built();
        let _ = pls1_perm_null(
            x.as_ref(),
            y.as_ref(),
            2,
            Some(w.as_ref()),
            opts(),
            Some(17),
        )
        .expect("weighted perm_null");
        assert_eq!(nspace_blocks_built(), before, "weighted input stays primal");
    }

    #[test]
    fn every_replicate_matches_the_primal_route() {
        // Closeness alone would pass if the n-space route fell back on every
        // row and returned the primal answer, so the bits must also differ
        // from the primal route's somewhere: the n-space kernel must run.
        let mut differing = 0usize;
        for k in 1..=K_DUAL_MAX {
            let (x, y) = synth(40, 2000, 1, 0.5, 20 + k as u64);
            let gram = run(&x, &y, k, opts());
            let primal = with_gram_routes_disabled(|| run(&x, &y, k, opts()));
            assert_rows_close(
                gram.beta_perm_matrix.as_deref().expect("matrix"),
                primal.beta_perm_matrix.as_deref().expect("matrix"),
                gram.beta_ref.len(),
                &format!("k={k}"),
            );
            assert_summaries_close(&gram, &primal, &format!("k={k}"));
            assert_eq!(
                bits(&gram.beta_ref),
                bits(&primal.beta_ref),
                "the observed fit stays primal"
            );
            if bits(gram.beta_perm_matrix.as_deref().expect("matrix"))
                != bits(primal.beta_perm_matrix.as_deref().expect("matrix"))
            {
                differing += 1;
            }
        }
        assert!(
            differing > 0,
            "no replicate's bits differed from the primal route: the n-space \
             route may be falling back on every row"
        );
    }

    #[test]
    fn tiny_scale_pre_standardized_x_matches_the_primal_route() {
        // The one input that reaches the absolute-floor gates.
        let (x, y) = synth(40, 2000, 1, 1.0, 3);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let o = PermNullOpts {
            pre_standardized: true,
            ..opts()
        };
        for s in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
            let xt = Mat::<f64>::from_fn(40, 2000, |i, j| xs[(i, j)] * s);
            let gram = run(&xt, &ys, 2, o);
            let primal = with_gram_routes_disabled(|| run(&xt, &ys, 2, o));
            assert_rows_close(
                gram.beta_perm_matrix.as_deref().expect("matrix"),
                primal.beta_perm_matrix.as_deref().expect("matrix"),
                gram.beta_ref.len(),
                &format!("scale {s:e}"),
            );
        }
        // At these two scales no replicate clears the gates, so the whole
        // matrix is the primal route's to the bit.
        for s in [1e-8, 1e-12] {
            let xt = Mat::<f64>::from_fn(40, 2000, |i, j| xs[(i, j)] * s);
            let gram = run(&xt, &ys, 2, o);
            let primal = with_gram_routes_disabled(|| run(&xt, &ys, 2, o));
            assert_matrix_bits_eq(&gram, &primal, &format!("scale {s:e}"));
        }
    }

    #[test]
    fn pre_standardized_outcomes_of_extreme_magnitude_match_the_primal_route() {
        // Squares of z underflow (1e-160) or overflow (1e160); the gates
        // must fail closed and hand those replicates to the primal.
        let (x, y) = synth(40, 2000, 1, 1.0, 4);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let o = PermNullOpts {
            pre_standardized: true,
            ..opts()
        };
        for m in [1e-160, 1e-120, 1e120, 1e160] {
            let yt = Col::<f64>::from_fn(40, |i| ys[i] * m);
            let gram = run(&xs, &yt, 2, o);
            let primal = with_gram_routes_disabled(|| run(&xs, &yt, 2, o));
            assert_rows_close(
                gram.beta_perm_matrix.as_deref().expect("matrix"),
                primal.beta_perm_matrix.as_deref().expect("matrix"),
                gram.beta_ref.len(),
                &format!("|y| ~ {m:e}"),
            );
        }
        // At these two magnitudes `‖z‖²` underflows or overflows for every
        // replicate, so the whole matrix is the primal route's to the bit.
        for m in [1e-160, 1e160] {
            let yt = Col::<f64>::from_fn(40, |i| ys[i] * m);
            let gram = run(&xs, &yt, 2, o);
            let primal = with_gram_routes_disabled(|| run(&xs, &yt, 2, o));
            assert_matrix_bits_eq(&gram, &primal, &format!("|y| ~ {m:e}"));
        }
    }
}

#[cfg(test)]
mod tests_gram_p {
    use super::*;
    use crate::signal_test::{with_gram_routes_disabled, ReplicateRoute};
    use crate::test_support::{
        assert_rows_close, assert_summaries_close, bits, conditioned, gram_p_weights, linear_y,
        unif,
    };
    use faer::{Col, Mat, Par};

    #[test]
    fn gram_p_arm_falls_back_to_the_primal_arm_to_the_bit() {
        // κ = 1e9 at k = p = 20: every permutation fails a gate (Tt by
        // component 4 on this design), so every GramP row is the
        // Primal arm's row, bit for bit. The route rule never sends k = 20
        // here; the arm itself accepts any k.
        let xs = conditioned(2000, 20, 1e9, 5);
        let (ys, _, _) = crate::linalg::standardize1(linear_y(&xs, 0.1, 6).as_ref());
        let g_block = perm_block(ReplicateRoute::GramP, xs.as_ref(), Par::Seq);
        let p_block = perm_block(ReplicateRoute::Primal, xs.as_ref(), Par::Seq);
        let PermBlock::GramP(gram) = &g_block else {
            panic!("perm_block(GramP) built another block")
        };
        let (mut row_g, mut row_p) = (vec![0.0_f64; 20], vec![0.0_f64; 20]);
        for b in 0..20u64 {
            let perm = crate::resample::permutation_from_seed(2000, b);
            assert!(
                perm_row_coef(gram, None, ys.as_ref(), &perm, 20).is_none(),
                "premise: permutation {b} falls back"
            );
            let rg = perm_row(
                &g_block,
                xs.as_ref(),
                None,
                ys.as_ref(),
                &perm,
                20,
                &mut row_g,
            );
            let rp = perm_row(
                &p_block,
                xs.as_ref(),
                None,
                ys.as_ref(),
                &perm,
                20,
                &mut row_p,
            );
            assert_eq!(rg.is_ok(), rp.is_ok(), "permutation {b}: same outcome");
            assert!(
                row_g
                    .iter()
                    .zip(&row_p)
                    .all(|(a, c)| a.to_bits() == c.to_bits()),
                "permutation {b}: the fallback row is the Primal row"
            );
        }
    }

    #[test]
    fn perm_null_gram_route_is_invisible() {
        let x = unif(2000, 50, 75);
        let y = linear_y(&x, 2.0, 76);
        let w = gram_p_weights(2000);
        for (label, wopt, k) in [("dense", None, 3usize), ("weighted", Some(&w), 2usize)] {
            assert_eq!(
                perm_null_route(2000, 50, 300, k, wopt.is_some()),
                ReplicateRoute::GramP,
                "{label}: premise"
            );
            let opts = PermNullOpts {
                n_perm: 300,
                return_perm_matrix: true,
                pre_standardized: false,
                verbose: false,
            };
            let call = || {
                pls1_perm_null(
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    wopt.map(Col::as_ref),
                    opts,
                    Some(77),
                )
                .expect("perm_null")
            };
            let g = call();
            let xr = with_gram_routes_disabled(call);
            // The observed fit stays on the Primal route on both runs.
            assert!(g
                .beta_ref
                .iter()
                .zip(&xr.beta_ref)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_summaries_close(&g, &xr, label);
            let (gm, xm) = (
                g.beta_perm_matrix.as_deref().expect("matrix"),
                xr.beta_perm_matrix.as_deref().expect("matrix"),
            );
            assert_rows_close(gm, xm, 50, "beta_perm_matrix");
            // The public call must actually run the Gram route: some row's
            // bits differ from the Primal route's.
            assert_ne!(
                bits(gm),
                bits(xm),
                "{label}: the public call ran no Gram row"
            );
        }
    }

    #[test]
    fn tiny_scale_pre_standardized_x_matches_the_primal_route() {
        // At 1e-6 the Gram route resolves; at 1e-8, tt approaches the
        // absolute band and it must fall back. Either way: the Primal
        // route's rows. beta scales as 1/scale, so the rows are compared
        // with the relative row rule.
        let x = unif(2000, 50, 78);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize1(linear_y(&x, 1.0, 79).as_ref());
        for scale in [1e-6, 1e-8] {
            let xt = Mat::<f64>::from_fn(2000, 50, |i, j| xs[(i, j)] * scale);
            // Premise: the GramP block resolves some permutations at 1e-6
            // and none at 1e-8.
            let g_block = perm_block(ReplicateRoute::GramP, xt.as_ref(), Par::Seq);
            let PermBlock::GramP(gram) = &g_block else {
                panic!("perm_block(GramP) built another block")
            };
            let resolved = (0..50u64)
                .filter(|&b| {
                    let perm = crate::resample::permutation_from_seed(2000, b);
                    perm_row_coef(gram, None, ys.as_ref(), &perm, 2).is_some()
                })
                .count();
            if scale > 1e-7 {
                assert!(resolved > 0, "premise: scale {scale:e} resolves some rows");
            } else {
                assert_eq!(resolved, 0, "premise: scale {scale:e} resolves no row");
            }
            let opts = PermNullOpts {
                n_perm: 300,
                return_perm_matrix: true,
                pre_standardized: true,
                verbose: false,
            };
            let call = || {
                pls1_perm_null(xt.as_ref(), ys.as_ref(), 2, None, opts, Some(80))
                    .expect("perm_null")
            };
            let g = call();
            let xr = with_gram_routes_disabled(call);
            assert_rows_close(
                &g.beta_perm_matrix.expect("matrix"),
                &xr.beta_perm_matrix.expect("matrix"),
                50,
                &format!("scale {scale:e}"),
            );
        }
    }
}
