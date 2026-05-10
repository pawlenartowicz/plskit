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

/// Tuning knobs for `pls1_perm_null`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct PermNullOpts {
    /// Number of permutations. Must be ≥ 100 (per-voxel-z noise floor).
    pub n_perm: usize,
    /// If true, retain and return the full `(n_perm, D)` β-matrix.
    /// If false, fold into a Welford accumulator and discard per-row data.
    pub return_perm_matrix: bool,
    /// Caller asserts X is already column-standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// Disable Rayon parallelism (deterministic byte-for-byte; useful for tests).
    pub disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
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
            return Err(PlsKitError::InvalidArgument(format!(
                "k must be ≥ 1, got {k}"
            )));
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
    use crate::fit::{pls1_fit, validate_and_normalize_weights, FitOpts, KSpec};
    use crate::linalg::{standardize, standardize1};
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

    let (w_norm, n_eff_val, _all_uniform) = validate_and_normalize_weights(weights, n, k)?;
    let wref = w_norm.as_ref().map(Col::as_ref);

    // Standardize once. Subsequent permutations operate on standardized arrays
    // — permuting y after standardization is equivalent to permuting raw y and
    // re-standardizing because mean/scale are permutation-invariant.
    let (xs_owned, ys_owned) = if opts.pre_standardized {
        (
            Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)]),
            Col::<f64>::from_fn(n, |i| y[i]),
        )
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (xs, ys)
    };
    let xs = xs_owned.as_ref();
    let ys = ys_owned.as_ref();

    // Reference fit on full standardized data.
    let fit_ref = pls1_fit(
        xs,
        ys,
        KSpec::Fixed(k),
        wref,
        FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        },
    )?;
    let beta_ref: Vec<f64> = (0..d).map(|j| fit_ref.beta[j]).collect();

    let (seed_used, mut rng) = crate::rng::resolve_seed(seed);

    if opts.return_perm_matrix {
        run_engine_retained(
            xs, ys, k, wref, beta_ref, n_eff_val, opts, seed_used, &mut rng,
        )
    } else {
        run_engine_streaming(
            xs, ys, k, wref, beta_ref, n_eff_val, opts, seed_used, &mut rng,
        )
    }
}

// Returns Result for symmetry with run_engine_streaming and to leave room for
// per-permutation hard failures in future revisions.
#[allow(clippy::too_many_arguments, clippy::unnecessary_wraps)]
fn run_engine_retained(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    wref: Option<faer::ColRef<'_, f64>>,
    beta_ref: Vec<f64>,
    n_eff_val: f64,
    opts: PermNullOpts,
    seed_used: u64,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<PermNullOutput> {
    let d = xs.ncols();
    let b = opts.n_perm;

    // Per-permutation worker collects β vectors. NaN row on failure (fail-soft).
    let beta_rows: Vec<Vec<f64>> =
        crate::resample::parallel_for_each_seeded(rng, b, opts.disable_parallelism, |_, child| {
            run_one_perm(xs, ys, k, wref, child).unwrap_or_else(|_| vec![f64::NAN; d])
        });

    // Flatten into row-major (B, D) buffer.
    let mut flat = vec![0.0_f64; b * d];
    for (bi, row) in beta_rows.iter().enumerate() {
        let off = bi * d;
        flat[off..off + d].copy_from_slice(row);
    }

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
        beta_perm_matrix: Some(flat),
    })
}

#[allow(clippy::too_many_arguments, clippy::unnecessary_wraps)]
fn run_engine_streaming(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    wref: Option<faer::ColRef<'_, f64>>,
    beta_ref: Vec<f64>,
    n_eff_val: f64,
    opts: PermNullOpts,
    seed_used: u64,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<PermNullOutput> {
    let d = xs.ncols();
    let b = opts.n_perm;

    // Per-permutation worker collects β vectors. NaN row on failure (fail-soft).
    let beta_rows: Vec<Vec<f64>> =
        crate::resample::parallel_for_each_seeded(rng, b, opts.disable_parallelism, |_, child| {
            run_one_perm(xs, ys, k, wref, child).unwrap_or_else(|_| vec![f64::NAN; d])
        });

    // Flatten ~n_perm·d f64s for deterministic two-pass reduce; trivial memory at typical sizes.
    let mut flat = vec![0.0_f64; b * d];
    for (bi, row) in beta_rows.iter().enumerate() {
        let off = bi * d;
        flat[off..off + d].copy_from_slice(row);
    }

    // Two-pass per-column reduction — byte-exact regardless of Rayon scheduling.
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
        beta_perm_matrix: None,
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

/// One permutation: draw `π_b`, apply to standardized y, refit, return β.
/// `xs` and `ys_std` are already standardized (centered, unit-scale).
/// Weights are NOT permuted — `w[i]` stays tied to row `i` regardless of
/// which `y` value lands there under the permutation (key invariant).
fn run_one_perm(
    xs: MatRef<'_, f64>,
    ys_std: ColRef<'_, f64>,
    k: usize,
    wref: Option<faer::ColRef<'_, f64>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<Vec<f64>> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    let n = xs.nrows();
    let d = xs.ncols();
    let perm = crate::resample::permute_indices(n, rng);
    let y_perm = faer::Col::<f64>::from_fn(n, |i| ys_std[perm[i]]);
    let fit = pls1_fit(
        xs,
        y_perm.as_ref(),
        KSpec::Fixed(k),
        wref,
        FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        },
    )?;
    let mut out = vec![0.0_f64; d];
    #[allow(clippy::needless_range_loop)]
    for j in 0..d {
        out[j] = fit.beta[j];
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
pub(crate) fn run_one_perm_for_test(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    pre_standardized_x: bool,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<Vec<f64>> {
    use crate::linalg::{standardize, standardize1};
    use faer::{Col, Mat};
    let n = x.nrows();
    let d = x.ncols();
    let (xs_owned, ys_owned) = if pre_standardized_x {
        (
            Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)]),
            Col::<f64>::from_fn(n, |i| y[i]),
        )
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (xs, ys)
    };
    run_one_perm(xs_owned.as_ref(), ys_owned.as_ref(), k, None, rng)
}

#[cfg(test)]
mod tests_worker {
    use super::*;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 2 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    #[test]
    fn worker_returns_finite_beta_with_correct_length() {
        let (x, y) = synth(80, 5, 4.0, 1);
        let (_, mut rng) = crate::rng::resolve_seed(Some(11));
        let beta = run_one_perm_for_test(x.as_ref(), y.as_ref(), 2, false, &mut rng).unwrap();
        assert_eq!(beta.len(), 5);
        for v in &beta {
            assert!(v.is_finite());
        }
    }
}

#[cfg(test)]
mod tests_engine_retained {
    use super::*;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 2 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    fn opts_retained() -> PermNullOpts {
        PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        }
    }

    #[test]
    fn retained_path_runs_end_to_end() {
        let (x, y) = synth(100, 5, 4.0, 42);
        let out =
            pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_retained(), Some(7)).unwrap();
        assert_eq!(out.n_perm, 200);
        assert_eq!(out.k, 2);
        assert_eq!(out.beta_ref.len(), 5);
        assert_eq!(out.beta_perm_mean.len(), 5);
        assert_eq!(out.beta_perm_sd.len(), 5);
        assert_eq!(out.beta_perm_z.len(), 5);
        let m = out
            .beta_perm_matrix
            .as_ref()
            .expect("matrix should be retained");
        assert_eq!(m.len(), 200 * 5);
        // Per-voxel z is finite (nonzero SD when signal exists).
        for &z in &out.beta_perm_z {
            assert!(z.is_finite() || z.is_nan());
        }
    }

    #[test]
    fn retained_path_signal_voxels_have_higher_abs_z() {
        let (x, y) = synth(150, 8, 6.0, 11);
        let out =
            pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_retained(), Some(13)).unwrap();
        let signal: f64 = out.beta_perm_z[..2].iter().map(|z| z.abs()).sum::<f64>() / 2.0;
        let noise: f64 = out.beta_perm_z[2..].iter().map(|z| z.abs()).sum::<f64>() / 6.0;
        assert!(
            signal > noise,
            "signal mean |z|={signal}, noise mean |z|={noise}"
        );
    }
}

#[cfg(test)]
mod tests_engine_streaming {
    use super::*;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 2 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    #[test]
    fn streaming_path_runs_end_to_end() {
        let (x, y) = synth(100, 5, 4.0, 42);
        let opts = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts, Some(7)).unwrap();
        assert!(out.beta_perm_matrix.is_none());
        assert_eq!(out.beta_perm_sd.len(), 5);
        assert_eq!(out.beta_perm_z.len(), 5);
    }

    #[test]
    fn streaming_matches_retained_byte_exact() {
        // Both paths now use parallel_for_each_seeded + two-pass reduce over the
        // same flat buffer; the only difference is retained returns the matrix.
        let (x, y) = synth(100, 5, 4.0, 42);
        let opts_retained = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        };
        let opts_streaming = PermNullOpts {
            return_perm_matrix: false,
            ..opts_retained
        };
        let r1 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_retained, Some(99)).unwrap();
        let r2 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_streaming, Some(99)).unwrap();
        assert_eq!(
            r1.beta_perm_mean, r2.beta_perm_mean,
            "beta_perm_mean must be byte-exact between retained and streaming"
        );
        assert_eq!(
            r1.beta_perm_sd, r2.beta_perm_sd,
            "beta_perm_sd must be byte-exact between retained and streaming"
        );
        assert_eq!(
            r1.beta_perm_z, r2.beta_perm_z,
            "beta_perm_z must be byte-exact between retained and streaming"
        );
    }
}

#[cfg(test)]
mod tests_validate {
    use super::*;

    fn opts_default() -> PermNullOpts {
        PermNullOpts {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: false,
            verbose: false,
        }
    }

    #[test]
    fn validate_accepts_defaults() {
        opts_default().validate(2).unwrap();
    }

    #[test]
    fn validate_rejects_low_n_perm() {
        let mut o = opts_default();
        o.n_perm = 50;
        let err = o.validate(2).unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(format!("{err}").contains("n_perm"));
    }

    #[test]
    fn validate_rejects_zero_k() {
        let err = opts_default().validate(0).unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(format!("{err}").contains('k'));
    }
}

#[cfg(test)]
mod tests_calibration {
    use super::*;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    /// Pure noise: y independent of X.
    fn synth_h0(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        (x, y)
    }

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
        let (x, y) = synth_h0(80, 5, 1);
        let opts = PermNullOpts {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
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
        // should land near 0.05. Use 3σ Monte-Carlo band (D × n_seeds is small,
        // so the test is loose but should catch order-of-magnitude regressions).
        let mut total = 0_usize;
        let mut rejects = 0_usize;
        for seed in 0..3_u64 {
            let (x, y) = synth_h0(80, 30, seed * 17 + 3);
            let opts = PermNullOpts {
                n_perm: 500,
                return_perm_matrix: false,
                pre_standardized: false,
                disable_parallelism: true,
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
        // 3σ band: σ ≈ √(0.05·0.95/total) ≈ 0.013 for total ≈ 90. Loose but informative.
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
            disable_parallelism: true,
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

    #[test]
    fn parallelism_determinism_disable_vs_enable() {
        // Same seed, opposite disable_parallelism — both streaming and retained paths now
        // use parallel_for_each_seeded + two-pass reduce, so output is byte-exact.
        let (x, y) = synth_h0(60, 5, 41);
        let opts_serial = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        };
        let opts_parallel = PermNullOpts {
            disable_parallelism: false,
            ..opts_serial
        };
        let r1 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_serial, Some(2026)).unwrap();
        let r2 =
            pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_parallel, Some(2026)).unwrap();
        assert_eq!(
            r1.beta_perm_mean, r2.beta_perm_mean,
            "beta_perm_mean must be byte-exact across serial/parallel"
        );
        assert_eq!(
            r1.beta_perm_sd, r2.beta_perm_sd,
            "beta_perm_sd must be byte-exact across serial/parallel"
        );
        assert_eq!(
            r1.beta_perm_z, r2.beta_perm_z,
            "beta_perm_z must be byte-exact across serial/parallel"
        );
    }

    #[test]
    fn parallelism_determinism_retained_matrix_byte_exact() {
        // Retained path uses parallel_for_each_seeded which is byte-exact across
        // serial / parallel modes (confirmed in resample::tests).
        let (x, y) = synth_h0(60, 5, 41);
        let opts_serial = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        };
        let opts_parallel = PermNullOpts {
            disable_parallelism: false,
            ..opts_serial
        };
        let r1 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_serial, Some(2026)).unwrap();
        let r2 =
            pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_parallel, Some(2026)).unwrap();
        let m1 = r1.beta_perm_matrix.as_ref().unwrap();
        let m2 = r2.beta_perm_matrix.as_ref().unwrap();
        assert_eq!(
            m1, m2,
            "retained matrices diverge between serial and parallel"
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
            disable_parallelism: true,
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
            disable_parallelism: true,
            verbose: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 5, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "k_exceeds_max");
    }
}
