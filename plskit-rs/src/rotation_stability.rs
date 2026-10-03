//! Rotation-stability diagnostic for PLS1. Standalone subsampling pass that
//! asks "does varimax rotation make axes more replicable than they would have
//! been on the unrotated NIPALS basis?" Output is a paired
//! rotated-vs-unrotated variance ratio with paired-bootstrap CIs.

#![allow(clippy::doc_markdown)]

use faer::{Col, ColRef, Mat, MatRef};
use rand::{RngExt, SeedableRng};

use crate::error::{PlsKitError, PlsKitResult};
use crate::rotate::{RotationMethod, VarimaxArgs};
use crate::subsample::CIScalar;

/// Hardcoded number of paired-bootstrap iterations.
/// Not a user knob: percentile resolution at the `n_boot = 100` floor is
/// already bounded by `1/B`, not `1/B'`.
const N_BOOT_PAIRED: usize = 1000;

/// Threshold on the bootstrap-iteration skip rate that flips
/// `degenerate_baseline` to `true` (secondary trigger).
const DEGENERATE_BOOTSTRAP_SKIP_THRESHOLD: f64 = 0.05;

/// Tuning knobs for `pls1_rotation_stability`.
#[derive(Debug, Clone, Copy)]
pub struct RotationStabilityOpts {
    /// Number of subsampling resamples. Must be ≥ 100.
    pub n_boot: usize,
    /// Subsample-size exponent: `m = ceil(n^m_rate)`. Must be in `(0.5, 0.95)`.
    pub m_rate: f64,
    /// Nominal CI level (e.g. 0.95). Must satisfy `0.5 ≤ level ≤ 0.99`.
    pub level: f64,
    /// Set when the caller has already column-standardized `X`.
    pub pre_standardized: bool,
    /// Optional fixed RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Reserved for future progress reporting.
    pub verbose: bool,
    /// Maximum fraction of subsamples that may be skipped (due to weight
    /// degeneracy) before returning `ResamplingDegenerate`. Default `0.01`.
    pub max_skip_rate: f64,
}

impl Default for RotationStabilityOpts {
    fn default() -> Self {
        Self {
            n_boot: 1000,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            seed: None,
            verbose: false,
            max_skip_rate: 0.01,
        }
    }
}

/// Method-axis dispatch payload. v0.x ships varimax only; the enum is
/// parameterized so promax/oblimin/geomin can land later without changing
/// the surface (mirrors `RotationMethod` in `rotate.rs`).
#[derive(Debug, Clone)]
pub enum RotationStabilityMethod {
    /// Varimax via pairwise Kaiser sweeps.
    Varimax(VarimaxArgs),
}

/// Engine output. Marshalled into `RotationStabilityResult` at the wrapper layer.
#[derive(Debug, Clone)]
pub struct RotationStabilityOutput {
    /// Method label echoed back to the caller (e.g. `"varimax"`).
    pub method: String,
    /// Number of subsampling resamples requested by the caller.
    pub n_boot: usize,
    /// Resolved subsample size `m`.
    pub m: usize,
    /// Subsample-size exponent used to derive `m`.
    pub m_rate: f64,
    /// Nominal CI level used for the percentile bootstrap.
    pub level: f64,
    /// Concrete RNG seed used (resolved from `opts.seed`).
    pub seed: u64,

    /// Headline aggregate variance ratio `ρ = V_rot / V_unrot` with
    /// paired-bootstrap percentile CI.
    pub variance_ratio: CIScalar,
    /// Per-axis ratio `ρ_k = V_rot,k / V_unrot,k`. Length K, indexed in
    /// reference-axis order.
    pub variance_ratio_per_axis: Vec<CIScalar>,

    /// Aggregate `V_unrot = (1/B) Σ_b Σ_k α²_unrot,b,k`.
    pub variance_unrot: f64,
    /// Aggregate `V_rot = (1/B) Σ_b Σ_k α²_rot,b,k`.
    pub variance_rot: f64,
    /// Per-axis `V_unrot,k`. Length K, reference-axis order.
    pub variance_unrot_per_axis: Vec<f64>,
    /// Per-axis `V_rot,k`. Length K, reference-axis order.
    pub variance_rot_per_axis: Vec<f64>,

    /// Diagnostic flag: `true` iff `V_unrot = 0` on the engine
    /// pass or > 5 % of bootstrap iterations had `V_unrot* = 0`. Only the
    /// first trigger makes `variance_ratio.point` `NaN` (with every CI
    /// bound); under the second alone the point is the finite
    /// `V_rot / V_unrot` and the CI is built from the draws that kept
    /// `V_unrot* > 0`.
    pub degenerate_baseline: bool,
    /// Number of resamples that produced finite per-axis squared residuals
    /// (`≤ n_boot`). Falls below `n_boot` only when the per-resample fit
    /// fails; `n_boot - n_boot_finite` resamples were skipped.
    pub n_boot_finite: usize,
    /// Effective sample size `n_eff = (Σ wᵢ)² / Σ wᵢ²` from the full
    /// (normalized) weight vector. Equals `n` when weights are uniform.
    pub n_eff: f64,
}

/// PLS1 rotation-stability diagnostic.
///
/// # Errors
/// - `KExceedsMax` when `k > n_features`
/// - `InvalidArgument` when `k = 0`, `k = 1` (rotation indeterminacy diagnostic
///   is meaningless on a 1-D subspace), `k > 7` (brute-force enumeration
///   tractability cap), `m_rate`/`level`/`n_boot` out of range, or `L`
///   shape-incompatible with the loadings.
/// - `DimensionMismatch` when `y.nrows() != x.nrows()`.
/// - `NonFiniteInput` for NaN/inf in inputs.
/// - `InvalidWeights` for invalid weight vectors.
/// - `ResamplingDegenerate` when more than `opts.max_skip_rate` fraction
///   of resamples fail to fit.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_pass_by_value)]
pub fn pls1_rotation_stability(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    method: RotationStabilityMethod,
    l: Option<MatRef<'_, f64>>,
    weights: Option<ColRef<'_, f64>>,
    opts: RotationStabilityOpts,
) -> PlsKitResult<RotationStabilityOutput> {
    crate::fit::with_thread_limit(|| rotation_stability_impl(x, y, k, method, l, weights, opts))
}

/// Body of [`pls1_rotation_stability`], on the caller's pool.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn rotation_stability_impl(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    method: RotationStabilityMethod,
    l: Option<MatRef<'_, f64>>,
    weights: Option<ColRef<'_, f64>>,
    opts: RotationStabilityOpts,
) -> PlsKitResult<RotationStabilityOutput> {
    // ── Validation ──
    let n = x.nrows();
    let d = x.ncols();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, d),
            y: y.nrows(),
        });
    }
    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k > d {
        return Err(PlsKitError::KExceedsMax { k, k_max: d });
    }
    if k == 1 {
        return Err(PlsKitError::InvalidArgument(
            "k=1: rotation indeterminacy diagnostic is meaningless on a 1-D subspace".into(),
        ));
    }
    if k > 7 {
        return Err(PlsKitError::InvalidArgument(format!(
            "k>7 ({k}): signed-permutation enumeration is not tractable for k > 7 \
             (2^k * k! candidates per replicate). Use k <= 7, or reduce the \
             subspace dimension before calling this function."
        )));
    }
    if let Some(l_ref) = l {
        if l_ref.ncols() != k {
            return Err(PlsKitError::ShapeMismatch(format!(
                "L.ncols={} but k={}",
                l_ref.ncols(),
                k
            )));
        }
    }
    // Reuse SubsampleOpts validation for shared knobs.
    let sub_opts = crate::subsample::SubsampleOpts {
        n_boot: opts.n_boot,
        m_rate: opts.m_rate,
        level: opts.level,
        pre_standardized: opts.pre_standardized,
        max_failure_rate: 1.0,
        // rotation_stability has its own skip-rate check; this value is only
        // used for shared-knob validation via validate(), not the CI loop.
        max_skip_rate: 1.0,
    };
    sub_opts.validate()?;
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_col(y)?;

    // ── Validate + normalize weights ──
    let (w_norm, n_eff_val) = crate::fit::validate_weights_for_k(weights, n, k)?;

    // ── Resolve seed + reference fit + reference rotation ──
    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    let fit_ref = {
        use crate::fit::{pls1_fit_impl, FitOpts, KSpec};
        pls1_fit_impl(
            x,
            y,
            KSpec::Fixed(k),
            w_norm.as_ref().map(faer::Col::as_ref),
            FitOpts {
                pre_standardized: opts.pre_standardized,
                ..FitOpts::default()
            },
        )?
    };

    let RotationStabilityMethod::Varimax(varimax_args) = &method;
    let varimax_args = *varimax_args;
    let rot_ref = crate::rotate::rotate(
        fit_ref.w_star.as_ref(),
        RotationMethod::Varimax(varimax_args),
        l,
    )?;
    let w_rot_ref = rot_ref.w_rot;

    // ── Resolve m and parallel-loop ──
    let m = crate::subsample::resolve_m(n, opts.m_rate);
    // m == n collapses every subsample to the full sample → zero between-resample
    // variance, degenerate baseline flag, and no stability signal. Reject early.
    if m >= n {
        return Err(PlsKitError::InvalidArgument(format!(
            "resolved m = {m} (from n={n}, m_rate={}) equals n; need m < n",
            opts.m_rate
        )));
    }
    if m < k + 2 {
        return Err(PlsKitError::InvalidArgument(format!(
            "resolved m = {m} (from n={n}, m_rate={}) is too small for k={k}; need m ≥ k+2",
            opts.m_rate
        )));
    }

    let pre_std = opts.pre_standardized;
    let rows: Vec<RotationStabilityWorkerRow> =
        crate::resample::parallel_for_each_seeded(&mut rng, opts.n_boot, |_, child| {
            run_one_rotation_stability(
                x,
                y,
                k,
                m,
                pre_std,
                fit_ref.w_star.as_ref(),
                w_rot_ref.as_ref(),
                varimax_args,
                l,
                w_norm.as_ref().map(faer::Col::as_ref),
                child,
            )
            // All worker errors (numerical failure and weight-validation skip
            // alike) collapse to one NaN row counted by a single skip counter.
            // Deliberate: this diagnostic only gates on the aggregate skip rate.
            // The confirmatory path (subsample.rs WorkerOutcome) is the granular
            // one that splits Skipped vs Failed; replicating it here would add
            // surface this diagnostic does not need.
            .unwrap_or_else(|_| RotationStabilityWorkerRow::nan(k))
        });

    // ── Derive a second child seed for the paired bootstrap; nested from the
    // post-subsample parent state to avoid any overlap with subsample-draw seeds.
    let boot_seed = {
        use rand::Rng;
        rng.next_u64()
    };

    // ── Reduce ──
    reduce_variance_ratio(
        &rows,
        k,
        opts.n_boot,
        m,
        opts.m_rate,
        opts.level,
        seed_used,
        boot_seed,
        n_eff_val,
        opts.max_skip_rate,
    )
}

/// Per-resample outputs: paired length-K vectors of squared post-alignment
/// Frobenius residuals against the unrotated and rotated references. NaN
/// entries flag a failed worker fit and are filtered before reduction.
#[derive(Debug, Clone)]
struct RotationStabilityWorkerRow {
    /// `α²_unrot,k` for `k = 0..K`. Length K.
    sq_unrot_per_axis: Vec<f64>,
    /// `α²_rot,k` for `k = 0..K`. Length K.
    sq_rot_per_axis: Vec<f64>,
}

impl RotationStabilityWorkerRow {
    fn nan(k: usize) -> Self {
        Self {
            sq_unrot_per_axis: vec![f64::NAN; k],
            sq_rot_per_axis: vec![f64::NAN; k],
        }
    }

    fn is_finite(&self) -> bool {
        self.sq_unrot_per_axis.iter().all(|v| v.is_finite())
            && self.sq_rot_per_axis.iter().all(|v| v.is_finite())
    }
}

/// Fit `k` components on the rows `sample_idx` of `(x, y)` and return the
/// weight matrix `W` of that fit: the per-replicate fit of
/// `run_one_rotation_stability`.
///
/// Weights are sliced to the subsample and re-normalized, and the rows are
/// standardized with the subsample's own weighted moments (plain moments
/// when unweighted), as `pls1_fit` standardizes the full data for the
/// reference fit. Under `pre_standardized_x` the rows are taken as they are.
#[allow(clippy::many_single_char_names)]
fn fit_subsample(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    sample_idx: &[usize],
    k: usize,
    pre_standardized_x: bool,
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<Mat<f64>> {
    use crate::fit::{pls1_fit_impl, validate_and_normalize_weights, FitOpts, KSpec};
    use crate::linalg::{col_row_subset, row_subset, standardize1_weighted};

    let m = sample_idx.len();
    let y_sub = col_row_subset(y, sample_idx);

    // Slice + re-normalize weights for this subsample. A too-small n_eff_sub
    // surfaces from the pls1_fit call below, not here; the caller maps either
    // error to a NaN row.
    let w_sub_norm: Option<Col<f64>> = match weights {
        Some(w_full) => {
            let w_sub = col_row_subset(w_full, sample_idx);
            let (w_norm_sub, _) = validate_and_normalize_weights(Some(w_sub.as_ref()), m, k)?;
            w_norm_sub
        }
        None => None,
    };

    // Pre-standardized: the subsample rows are the block. Otherwise they are
    // gathered and standardized in one pass.
    let wref = w_sub_norm.as_ref().map(Col::as_ref);
    let (xs, ys) = if pre_standardized_x {
        (row_subset(x, sample_idx), y_sub)
    } else {
        let (xs, _, _) = crate::linalg::standardize_rows(x, sample_idx, wref, None);
        let (ys, _, _) = standardize1_weighted(y_sub.as_ref(), wref);
        (xs, ys)
    };

    let fit_b = pls1_fit_impl(
        xs.as_ref(),
        ys.as_ref(),
        KSpec::Fixed(k),
        w_sub_norm.as_ref().map(Col::as_ref),
        FitOpts {
            pre_standardized: true,
            // Seq inside the per-resample worker — outer Rayon owns the threadpool.
            par: crate::fit::ParChoice::Seq,
            ..FitOpts::default()
        },
    )?;
    Ok(fit_b.w_star)
}

/// One subsample worker pass. Draws `m` indices, fits PLS1, and computes
/// signed-permutation-aligned squared per-axis Frobenius residuals against
/// both the unrotated and rotated references.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
fn run_one_rotation_stability(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    m: usize,
    pre_standardized_x: bool,
    w_unrot_ref: MatRef<'_, f64>,
    w_rot_ref: MatRef<'_, f64>,
    varimax_args: VarimaxArgs,
    l: Option<MatRef<'_, f64>>,
    weights: Option<ColRef<'_, f64>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RotationStabilityWorkerRow> {
    let n = x.nrows();
    let d = x.ncols();

    let (sample_idx, _holdout_idx) = crate::subsample::subsample_indices(n, m, rng);
    let w_b = fit_subsample(x, y, &sample_idx, k, pre_standardized_x, weights)?;

    // Unrotated alignment — signed-permutation against the unrotated ref.
    // Per-axis squared residual is read directly from the alignment payload
    // (no need to materialize an aligned matrix); see the cost
    // identity on `SignedPermutationAlignment.residual_frobenius`.
    // `?` so that a truncated `w_b` (NIPALS short-circuit, k_used < k)
    // propagates as Err to the outer worker, which maps it to a NaN row.
    let aln_unrot = procrustes::signed_permutation(w_b.as_ref(), w_unrot_ref, false)?;
    let sq_unrot_per_axis: Vec<f64> = (0..k)
        .map(|kk| {
            let src = aln_unrot.assigned[kk];
            let s = aln_unrot.signs[kk];
            let mut acc = 0.0_f64;
            for j in 0..d {
                let diff = s * w_b[(j, src)] - w_unrot_ref[(j, kk)];
                acc += diff * diff;
            }
            acc
        })
        .collect();

    // Continuous-orthogonal alignment is scaffolding — puts W_b into the
    // same orthogonal frame as the reference so varimax converges to a
    // comparable simple-structure target. Residual is discarded.
    // `procrustes::orthogonal`'s rotation, on a sequential SVD (that crate
    // reads faer's global parallelism). The shapes were validated by the
    // `signed_permutation` call above, which fails on the same inputs.
    let r_orth = crate::linalg::orthogonal_rotation(w_b.as_ref(), w_unrot_ref);
    let mut w_b_rot_input = Mat::<f64>::zeros(d, k);
    faer::linalg::matmul::matmul(
        w_b_rot_input.as_mut(),
        faer::Accum::Replace,
        w_b.as_ref(),
        r_orth.as_ref(),
        1.0,
        faer::Par::Seq,
    );

    let rot_b = crate::rotate::rotate(
        w_b_rot_input.as_ref(),
        RotationMethod::Varimax(varimax_args),
        l,
    )?;
    let w_b_rot = rot_b.w_rot;

    // Rotated alignment — signed-permutation against the rotated ref.
    let aln_rot = procrustes::signed_permutation(w_b_rot.as_ref(), w_rot_ref, false)?;
    let sq_rot_per_axis: Vec<f64> = (0..k)
        .map(|kk| {
            let src = aln_rot.assigned[kk];
            let s = aln_rot.signs[kk];
            let mut acc = 0.0_f64;
            for j in 0..d {
                let diff = s * w_b_rot[(j, src)] - w_rot_ref[(j, kk)];
                acc += diff * diff;
            }
            acc
        })
        .collect();

    Ok(RotationStabilityWorkerRow {
        sq_unrot_per_axis,
        sq_rot_per_axis,
    })
}

/// Paired percentile-bootstrap reduction.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
fn reduce_variance_ratio(
    rows: &[RotationStabilityWorkerRow],
    k: usize,
    n_boot: usize,
    m: usize,
    m_rate: f64,
    level: f64,
    seed: u64,
    boot_seed: u64,
    n_eff: f64,
    max_skip_rate: f64,
) -> PlsKitResult<RotationStabilityOutput> {
    // Filter out NaN rows (failed fits). Error out if skip rate exceeds threshold.
    let finite: Vec<&RotationStabilityWorkerRow> = rows.iter().filter(|r| r.is_finite()).collect();
    let n_boot_finite = finite.len();
    let total = rows.len();
    let skipped = total - n_boot_finite;
    #[allow(clippy::cast_precision_loss)]
    {
        let skip_rate = skipped as f64 / total.max(1) as f64;
        if skip_rate > max_skip_rate {
            return Err(PlsKitError::ResamplingDegenerate {
                skipped,
                total,
                skip_rate,
                threshold: max_skip_rate,
            });
        }
    }

    let b = n_boot_finite;
    if b == 0 {
        // Every resample failed but the skip-rate gate let it through (e.g.
        // max_skip_rate=1.0). With no finite rows the V's are all NaN and the
        // degenerate_baseline flag (NaN == 0.0 → false) would mislabel an
        // all-NaN result as healthy. Fail loudly instead.
        #[allow(clippy::cast_precision_loss)]
        return Err(PlsKitError::ResamplingDegenerate {
            skipped,
            total,
            skip_rate: skipped as f64 / total.max(1) as f64,
            threshold: max_skip_rate,
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let b_f = b as f64;

    // Aggregate point-estimate variance components.
    let mut v_unrot_per_axis = vec![0.0_f64; k];
    let mut v_rot_per_axis = vec![0.0_f64; k];
    for row in &finite {
        for kk in 0..k {
            v_unrot_per_axis[kk] += row.sq_unrot_per_axis[kk];
            v_rot_per_axis[kk] += row.sq_rot_per_axis[kk];
        }
    }
    for kk in 0..k {
        v_unrot_per_axis[kk] /= b_f;
        v_rot_per_axis[kk] /= b_f;
    }
    let v_unrot: f64 = v_unrot_per_axis.iter().sum();
    let v_rot: f64 = v_rot_per_axis.iter().sum();

    // Primary degeneracy trigger: V_unrot = 0.
    let primary_degenerate = v_unrot == 0.0;

    // Headline ratios.
    let rho_point = if primary_degenerate {
        f64::NAN
    } else {
        v_rot / v_unrot
    };
    let rho_per_axis_point: Vec<f64> = (0..k)
        .map(|kk| {
            if v_unrot_per_axis[kk] == 0.0 {
                f64::NAN
            } else {
                v_rot_per_axis[kk] / v_unrot_per_axis[kk]
            }
        })
        .collect();

    // Paired percentile bootstrap on the per-resample paired vector.
    // Skipped iterations (V_unrot* = 0) propagate to the
    // bootstrap_skip counter and degenerate-baseline flag.
    let mut boot_rng = rand_chacha::ChaCha8Rng::seed_from_u64(boot_seed);
    let mut rho_star: Vec<f64> = Vec::with_capacity(N_BOOT_PAIRED);
    let mut rho_per_axis_star: Vec<Vec<f64>> = vec![Vec::with_capacity(N_BOOT_PAIRED); k];
    let mut bootstrap_skipped = 0usize;

    if b > 0 && !primary_degenerate {
        let mut idx_buf: Vec<usize> = vec![0; b];
        for _ in 0..N_BOOT_PAIRED {
            for slot in &mut idx_buf {
                *slot = boot_rng.random_range(0..b);
            }
            let mut v_unrot_b_per_axis = vec![0.0_f64; k];
            let mut v_rot_b_per_axis = vec![0.0_f64; k];
            for &i in &idx_buf {
                let row = finite[i];
                for kk in 0..k {
                    v_unrot_b_per_axis[kk] += row.sq_unrot_per_axis[kk];
                    v_rot_b_per_axis[kk] += row.sq_rot_per_axis[kk];
                }
            }
            for kk in 0..k {
                v_unrot_b_per_axis[kk] /= b_f;
                v_rot_b_per_axis[kk] /= b_f;
            }
            let v_unrot_b: f64 = v_unrot_b_per_axis.iter().sum();
            let v_rot_b: f64 = v_rot_b_per_axis.iter().sum();
            if v_unrot_b > 0.0 {
                rho_star.push(v_rot_b / v_unrot_b);
            } else {
                bootstrap_skipped += 1;
            }
            for kk in 0..k {
                if v_unrot_b_per_axis[kk] > 0.0 {
                    rho_per_axis_star[kk].push(v_rot_b_per_axis[kk] / v_unrot_b_per_axis[kk]);
                }
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let bootstrap_skip_rate = bootstrap_skipped as f64 / N_BOOT_PAIRED as f64;
    let degenerate_baseline =
        primary_degenerate || bootstrap_skip_rate > DEGENERATE_BOOTSTRAP_SKIP_THRESHOLD;

    let alpha = 1.0 - level;
    let variance_ratio = build_ciscalar_from_bootstrap(rho_point, &mut rho_star[..], alpha);
    let variance_ratio_per_axis: Vec<CIScalar> = (0..k)
        .map(|kk| {
            build_ciscalar_from_bootstrap(
                rho_per_axis_point[kk],
                &mut rho_per_axis_star[kk][..],
                alpha,
            )
        })
        .collect();

    Ok(RotationStabilityOutput {
        method: "varimax".to_owned(),
        n_boot,
        m,
        m_rate,
        level,
        seed,
        variance_ratio,
        variance_ratio_per_axis,
        variance_unrot: v_unrot,
        variance_rot: v_rot,
        variance_unrot_per_axis: v_unrot_per_axis,
        variance_rot_per_axis: v_rot_per_axis,
        degenerate_baseline,
        n_boot_finite,
        n_eff,
    })
}

/// Build a `CIScalar` from a point estimate and a bootstrap-sample slice.
/// Sorts in place; if the sample slice is empty (degenerate axis) returns
/// a NaN-filled CI carrying the (possibly NaN) point estimate.
fn build_ciscalar_from_bootstrap(point: f64, samples: &mut [f64], alpha: f64) -> CIScalar {
    if samples.is_empty() {
        return CIScalar {
            point,
            lower: f64::NAN,
            upper: f64::NAN,
            sd: f64::NAN,
        };
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
    let lower = crate::linalg::empirical_quantile(samples, alpha / 2.0);
    let upper = crate::linalg::empirical_quantile(samples, 1.0 - alpha / 2.0);
    #[allow(clippy::cast_precision_loss)]
    let n = samples.len() as f64;
    let mean: f64 = samples.iter().sum::<f64>() / n;
    let sd: f64 = if samples.len() > 1 {
        let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    } else {
        0.0
    };
    CIScalar {
        point,
        lower,
        upper,
        sd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotate::VarimaxArgs;
    use crate::test_support::synth;
    use faer::Mat;
    use rand::RngExt;
    use rand::SeedableRng;

    /// A weighted replicate is standardized the way the weighted reference
    /// fit standardizes the full data: with weighted moments. Its `W` is
    /// then the `W` of `pls1_fit` on the same rows and weights (which
    /// standardizes internally), up to the last-bit effect of normalizing
    /// the weights once more on the reference path.
    #[test]
    fn weighted_subsample_uses_weighted_moments() {
        let (x, y) = synth(60, 5, 2, 2.0, 7);
        let w = faer::Col::<f64>::from_fn(60, |i| 0.2 + (i % 5) as f64);
        let idx: Vec<usize> = (0..60).filter(|i| i % 3 != 0).collect();
        let k = 2;
        let got = fit_subsample(x.as_ref(), y.as_ref(), &idx, k, false, Some(w.as_ref())).unwrap();

        let x_sub = crate::linalg::row_subset(x.as_ref(), &idx);
        let y_sub = crate::linalg::col_row_subset(y.as_ref(), &idx);
        let w_sub = crate::linalg::col_row_subset(w.as_ref(), &idx);
        let want = crate::fit::pls1_fit(
            x_sub.as_ref(),
            y_sub.as_ref(),
            crate::fit::KSpec::Fixed(k),
            Some(w_sub.as_ref()),
            crate::fit::FitOpts {
                check_n_eff: false,
                ..crate::fit::FitOpts::default()
            },
        )
        .unwrap()
        .w_star;
        for a in 0..k {
            for j in 0..5 {
                assert!(
                    (got[(j, a)] - want[(j, a)]).abs() <= 1e-12,
                    "W[({j}, {a})]: {} vs {}",
                    got[(j, a)],
                    want[(j, a)]
                );
            }
        }
    }

    /// Synthesize a 2-factor model where simple-structure axes are NOT
    /// aligned with PLS components — the regime where rotation provably
    /// reduces per-axis variance.
    ///
    /// Construction: two correlated latent factors (`corr(f1, f2) ≈ 0.05`)
    /// with block-disjoint simple-structure loadings. The small positive
    /// cross-correlation tilts the principal axes ~45° toward (avg,
    /// contrast) directions, leaving the eigenvalue gap small enough that
    /// finite-sample NIPALS axes drift within the close-σ block. Varimax
    /// rotates back to the block-disjoint simple structure, which is
    /// pinned by the loading sparsity criterion.
    #[allow(clippy::many_single_char_names)]
    fn synth_factor_model(n: usize, seed: u64) -> (Mat<f64>, faer::Col<f64>) {
        let d = 8usize;
        let rho = 0.05_f64;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let f1 = faer::Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let z = faer::Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let f2 = faer::Col::<f64>::from_fn(n, |i| rho * f1[i] + (1.0 - rho * rho).sqrt() * z[i]);
        let mut x = Mat::<f64>::zeros(n, d);
        for i in 0..n {
            for j in 0..d {
                let base = if j < 4 { f1[i] } else { f2[i] };
                let noise = rng.random_range(-0.05..0.05);
                x[(i, j)] = base + noise;
            }
        }
        // y = f1 + f2 puts PLS's first component on the (avg) direction
        // — orthogonal to simple structure, matching the regime where
        // varimax rotation is supposed to help.
        let y = faer::Col::<f64>::from_fn(n, |i| f1[i] + f2[i] + 0.1 * rng.random_range(-1.0..1.0));
        (x, y)
    }

    fn run_one(
        x: &Mat<f64>,
        y: &faer::Col<f64>,
        k: usize,
        n_boot: usize,
        seed: u64,
    ) -> RotationStabilityOutput {
        let opts = RotationStabilityOpts {
            n_boot,
            m_rate: 0.7,
            level: 0.95,
            seed: Some(seed),
            ..Default::default()
        };
        pls1_rotation_stability(
            x.as_ref(),
            y.as_ref(),
            k,
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            None,
            None,
            opts,
        )
        .unwrap()
    }

    /// Structural sanity — on a factor-model design with simple-structure
    /// loadings, the diagnostic produces a finite ratio in a reasonable
    /// range without flagging `degenerate_baseline`.
    ///
    /// The bound (`upper < 1.5`) only rules out a blow-up: PLS1, unlike
    /// PCA, pins both components through y-driven deflation, so the
    /// unrotated axes do not drift on this design and the ratio stays near
    /// 1 rather than below it.
    #[test]
    fn variance_ratio_is_bounded_on_factor_model() {
        let (x, y) = synth_factor_model(300, 17);
        let r = run_one(&x, &y, 2, 500, 23);
        assert!(
            !r.degenerate_baseline,
            "factor-model design should not flag degenerate baseline"
        );
        assert!(
            r.variance_ratio.point.is_finite(),
            "ratio must be finite on factor-model design (point={})",
            r.variance_ratio.point
        );
        assert!(
            r.variance_ratio.upper < 1.5,
            "rotation should not blow up the variance ratio on a factor-model \
             design, got upper={} (point={})",
            r.variance_ratio.upper,
            r.variance_ratio.point,
        );
    }

    /// Random y: rotation should neither help nor strongly hurt.
    /// Tolerance reflects bootstrap variance at small B and is empirical.
    #[test]
    fn variance_ratio_one_under_pure_noise() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let x = Mat::<f64>::from_fn(200, 6, |_, _| rng.random_range(-1.0..1.0));
        let y = faer::Col::<f64>::from_fn(200, |_| rng.random_range(-1.0..1.0));
        let r = run_one(&x, &y, 2, 200, 31);
        assert!(
            r.variance_ratio.point.is_finite(),
            "ratio must be finite on noise design"
        );
        assert!(
            (0.5..=1.6).contains(&r.variance_ratio.point),
            "expected ratio ≈ 1 under pure noise, got {}",
            r.variance_ratio.point
        );
    }

    /// Aggregate ratio is consistent with per-axis decomposition:
    /// sum of per-axis V's must reproduce aggregate V.
    #[test]
    fn per_axis_decomposition_sums_to_aggregate() {
        let (x, y) = synth(150, 6, 2, 4.0, 11);
        let r = run_one(&x, &y, 2, 200, 5);
        assert_eq!(r.method, "varimax");
        let sum_unrot: f64 = r.variance_unrot_per_axis.iter().sum();
        let sum_rot: f64 = r.variance_rot_per_axis.iter().sum();
        assert!(
            (sum_unrot - r.variance_unrot).abs() < 1e-10,
            "sum_unrot={} aggregate={}",
            sum_unrot,
            r.variance_unrot,
        );
        assert!(
            (sum_rot - r.variance_rot).abs() < 1e-10,
            "sum_rot={} aggregate={}",
            sum_rot,
            r.variance_rot,
        );
        if r.variance_unrot > 0.0 {
            let expected = r.variance_rot / r.variance_unrot;
            assert!(
                (r.variance_ratio.point - expected).abs() < 1e-10,
                "ratio={} V_rot/V_unrot={}",
                r.variance_ratio.point,
                expected,
            );
        }
    }

    /// Bootstrap CI must bracket the point estimate.
    #[test]
    fn paired_bootstrap_ci_contains_point_estimate() {
        let (x, y) = synth(120, 6, 2, 3.0, 9);
        let r = run_one(&x, &y, 2, 200, 14);
        assert!(
            r.variance_ratio.lower <= r.variance_ratio.point + 1e-10,
            "lower={} > point={}",
            r.variance_ratio.lower,
            r.variance_ratio.point,
        );
        assert!(
            r.variance_ratio.point <= r.variance_ratio.upper + 1e-10,
            "point={} > upper={}",
            r.variance_ratio.point,
            r.variance_ratio.upper,
        );
        for (kk, ci) in r.variance_ratio_per_axis.iter().enumerate() {
            if ci.point.is_finite() {
                assert!(
                    ci.lower <= ci.point + 1e-10 && ci.point <= ci.upper + 1e-10,
                    "axis {kk}: lower={} point={} upper={}",
                    ci.lower,
                    ci.point,
                    ci.upper,
                );
            }
        }
    }

    /// `degenerate_baseline` on both triggers, at the reducer: V_unrot = 0
    /// on the engine pass (the point and every CI bound are NaN), and more
    /// than 5% of the paired bootstrap draws with V_unrot* = 0. With every
    /// replicate failed the reducer errors even when the skip gate lets the
    /// rows through.
    #[test]
    fn reduce_variance_ratio_flags_degenerate_baseline() {
        let row = |u: [f64; 2], r: [f64; 2]| RotationStabilityWorkerRow {
            sq_unrot_per_axis: u.to_vec(),
            sq_rot_per_axis: r.to_vec(),
        };
        let reduce = |rows: &[RotationStabilityWorkerRow], max_skip: f64| {
            reduce_variance_ratio(rows, 2, rows.len(), 10, 0.7, 0.95, 1, 2, 20.0, max_skip)
        };
        // Primary trigger: V_unrot = 0 on every replicate.
        let r = reduce(&vec![row([0.0, 0.0], [0.5, 0.5]); 20], 0.01).unwrap();
        assert!(r.degenerate_baseline);
        let v = &r.variance_ratio;
        assert!(v.point.is_nan() && v.lower.is_nan() && v.upper.is_nan() && v.sd.is_nan());
        assert!(r.variance_ratio_per_axis.iter().all(|c| c.point.is_nan()));
        // Secondary trigger: one non-zero replicate of 20, so ~36% of the
        // 1000 paired bootstrap draws have V_unrot* = 0 (> 5%). The flag is
        // set but the point stays the finite V_rot / V_unrot = 1 / 0.1.
        let mut rows = vec![row([0.0, 0.0], [0.5, 0.5]); 20];
        rows[0] = row([1.0, 1.0], [0.5, 0.5]);
        let r = reduce(&rows, 0.01).unwrap();
        assert!(r.degenerate_baseline);
        let point = r.variance_ratio.point;
        assert!(point.is_finite());
        assert_eq!(
            point.to_bits(),
            (r.variance_rot / r.variance_unrot).to_bits()
        );
        // Every replicate failed: an error even when the skip gate allows it.
        let nan = vec![RotationStabilityWorkerRow::nan(2); 20];
        assert!(matches!(
            reduce(&nan, 1.0),
            Err(PlsKitError::ResamplingDegenerate {
                skipped: 20,
                total: 20,
                ..
            })
        ));
    }

    /// The worker reads X only through row gathers of owned column-major
    /// copies, so every layout of X gives the owned matrix's per-axis
    /// residuals to the bit once the references are fixed. The references
    /// are fitted on the owned matrix: `pls1_fit` itself is equal
    /// across layouts only to rounding.
    #[test]
    fn rotation_stability_worker_is_layout_invariant() {
        crate::test_support::assert_families_layout_invariant(
            "rotation_stability",
            |c| {
                let k = 2;
                let n = c.x.nrows();
                let (w_norm, _) = crate::fit::validate_and_normalize_weights(c.w, n, k)?;
                let wr = w_norm.as_ref().map(faer::Col::as_ref);
                let w_unrot = crate::fit::pls1_fit(
                    c.x_owned.as_ref(),
                    c.y,
                    crate::fit::KSpec::Fixed(k),
                    wr,
                    crate::fit::FitOpts {
                        pre_standardized: c.pre,
                        ..Default::default()
                    },
                )?
                .w_star;
                let w_rot = crate::rotate::rotate(
                    w_unrot.as_ref(),
                    RotationMethod::Varimax(VarimaxArgs::default()),
                    None,
                )?
                .w_rot;
                let m = crate::subsample::resolve_m(n, 0.7);
                let mut out = Vec::new();
                for seed in [1_u64, 2, 3] {
                    let (_, mut rng) = crate::rng::resolve_seed(Some(seed))?;
                    let row = run_one_rotation_stability(
                        c.x,
                        c.y,
                        k,
                        m,
                        c.pre,
                        w_unrot.as_ref(),
                        w_rot.as_ref(),
                        VarimaxArgs::default(),
                        None,
                        wr,
                        &mut rng,
                    )?;
                    out.extend(row.sq_unrot_per_axis);
                    out.extend(row.sq_rot_per_axis);
                }
                Ok(out)
            },
            Vec::clone,
        );
    }

    #[test]
    fn rotation_stability_rejects_k_eq_1() {
        let (x, y) = synth(80, 5, 2, 3.0, 1);
        let err = pls1_rotation_stability(
            x.as_ref(),
            y.as_ref(),
            1,
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            None,
            None,
            RotationStabilityOpts::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(format!("{err}").contains("k=1"));
    }

    #[test]
    fn rotation_stability_rejects_k_gt_7() {
        let (x, y) = synth(80, 10, 2, 3.0, 1);
        let err = pls1_rotation_stability(
            x.as_ref(),
            y.as_ref(),
            8,
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            None,
            None,
            RotationStabilityOpts::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(format!("{err}").contains("k>7") || format!("{err}").contains("k > 7"));
    }

    #[test]
    fn rotation_stability_rejects_l_shape_mismatch() {
        let (x, y) = synth(80, 6, 2, 3.0, 1);
        let l_bad = Mat::<f64>::zeros(4, 3);
        let err = pls1_rotation_stability(
            x.as_ref(),
            y.as_ref(),
            2,
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            Some(l_bad.as_ref()),
            None,
            RotationStabilityOpts::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), "shape_mismatch");
    }
}
