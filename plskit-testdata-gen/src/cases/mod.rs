//! Case definitions: each `pub fn` materializes one fixture
//! (input npz + output npz) and returns its `Case` manifest entry.

use crate::manifest::Case;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub mod pls1_confirmatory_test;
pub mod pls1_find_k_optimal;
pub mod pls1_find_k_sequence;
pub mod pls1_fit;
pub mod pls1_perm_null;
pub mod pls1_predict;
pub mod pls1_rotation_stability;
pub mod pls3;
pub mod preprocess;
pub mod rotate;
pub mod spls1_find_k_optimal;
pub mod spls1_find_k_sequence;
pub mod spls1_find_keep_optimal;
pub mod spls1_fit;

/// Resolved relative and absolute paths for a fixture case's input/output files.
///
/// Use [`CasePaths::build`] to construct; it also creates the parent directories.
pub(crate) struct CasePaths {
    /// Relative path to the inputs `.npz` (e.g. `"inputs/foo.npz"`).
    pub rel_inputs: String,
    /// Absolute path to the inputs `.npz`.
    pub abs_inputs: PathBuf,
    /// Relative path to the outputs `.npz` (e.g. `"outputs/pls1_fit/foo.npz"`).
    pub rel_outputs: String,
    /// Absolute path to the outputs `.npz`.
    pub abs_outputs: PathBuf,
}

impl CasePaths {
    /// Build relative + absolute paths for a case under `root` and ensure
    /// parent directories exist.
    ///
    /// # Errors
    /// Returns an error if the parent directories cannot be created.
    pub(crate) fn build(root: &Path, function: &str, name: &str) -> std::io::Result<Self> {
        let rel_inputs = format!("inputs/{name}.npz");
        let rel_outputs = format!("outputs/{function}/{name}.npz");
        let abs_inputs = root.join(&rel_inputs);
        let abs_outputs = root.join(&rel_outputs);
        if let Some(p) = abs_inputs.parent() {
            std::fs::create_dir_all(p)?;
        }
        if let Some(p) = abs_outputs.parent() {
            std::fs::create_dir_all(p)?;
        }
        Ok(Self {
            rel_inputs,
            abs_inputs,
            rel_outputs,
            abs_outputs,
        })
    }
}

pub(crate) use synth_helpers::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64,
    synth_data, synth_xy,
};

mod synth_helpers {
    use faer::{Col, Mat};
    use ndarray::{Array1, Array2};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StandardNormal;

    /// Generate `(X, y)` with `k_signal` active features at signal-to-noise ratio `snr`.
    ///
    /// `X` is an `(n, d)` matrix of standard-normal values. `y` is constructed as
    /// `X[:, :k_signal].sum(axis=1) * snr + noise` where noise is also standard-normal.
    /// The RNG is a deterministic `ChaCha8` seeded with `seed`.
    #[must_use]
    pub fn synth_data(
        n: usize,
        d: usize,
        k_signal: usize,
        snr: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut x = Array2::<f64>::zeros((n, d));
        for v in &mut x {
            *v = rng.sample::<f64, _>(StandardNormal);
        }
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..k_signal {
                acc += x[(i, j)];
            }
            y[i] = acc * snr + rng.sample::<f64, _>(StandardNormal);
        }
        (x, y)
    }

    /// Generate a two-block `(X, Y)` sharing `k_signal` latent factors at
    /// signal-to-noise ratio `snr` — the shape PLS3 / PLSSVD analyses.
    ///
    /// Factor `a` drives `X[:, a]` and `Y[:, a]` for every `a < k_signal`;
    /// every other column of X and Y is pure noise. This keeps the leading
    /// singular triplet of `X'Y` well separated so the fixtures are not
    /// recording noise. There is no wraparound: a `k_signal` greater than
    /// `q` silently leaves the surplus factors out of Y rather than
    /// wrapping them, so callers must keep `k_signal <= q` (and, for the
    /// same reason, `<= p`) for every factor to land in both blocks. The
    /// RNG is a deterministic `ChaCha8` seeded with `seed`; factors are
    /// drawn first, then X, then Y, so the byte stream is fixed by
    /// `(n, p, q, k_signal, seed)` alone.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn synth_xy(
        n: usize,
        p: usize,
        q: usize,
        k_signal: usize,
        snr: f64,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut factors = Array2::<f64>::zeros((n, k_signal));
        for v in &mut factors {
            *v = rng.sample::<f64, _>(StandardNormal);
        }
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                let noise: f64 = rng.sample(StandardNormal);
                x[(i, j)] = if j < k_signal {
                    snr * factors[(i, j)] + noise
                } else {
                    noise
                };
            }
        }
        let mut y = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for j in 0..q {
                let noise: f64 = rng.sample(StandardNormal);
                y[(i, j)] = if j < k_signal {
                    snr * factors[(i, j)] + noise
                } else {
                    noise
                };
            }
        }
        (x, y)
    }

    /// Convert an `ndarray::Array2<f64>` to a `faer::Mat<f64>` (column-major).
    pub fn ndarray_to_faer_mat(a: &Array2<f64>) -> Mat<f64> {
        Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
    }

    /// Convert an `ndarray::Array1<f64>` to a `faer::Col<f64>`.
    pub fn ndarray_to_faer_col(a: &Array1<f64>) -> Col<f64> {
        Col::from_fn(a.len(), |i| a[i])
    }

    /// Convert a `faer::Col<f64>` to a 1-D `ndarray::ArrayD<f64>`.
    pub fn faer_col_to_array(col: &Col<f64>) -> ndarray::ArrayD<f64> {
        let n = col.nrows();
        let v: Vec<f64> = (0..n).map(|i| col[i]).collect();
        ndarray::Array1::from_vec(v).into_dyn()
    }

    /// Wrap a scalar `f64` as a 0-D `ndarray::ArrayD<f64>`.
    pub fn scalar_f64(v: f64) -> ndarray::ArrayD<f64> {
        ndarray::arr0(v).into_dyn()
    }

    /// Wrap a scalar `i64` as a 0-D `ndarray::ArrayD<i64>`.
    pub fn scalar_i64(v: i64) -> ndarray::ArrayD<i64> {
        ndarray::arr0(v).into_dyn()
    }
}

/// Materialize every fixture under `root` and return the manifest entries.
///
/// All cases are required; any failure is propagated as an error.
///
/// # Errors
/// Returns an error if any case fails to write its fixture files.
// `?` operators inside individual push calls prevent collapsing to `vec![]`.
#[allow(clippy::vec_init_then_push)]
pub fn all_cases(root: &Path) -> Result<Vec<Case>> {
    let mut cases = Vec::new();

    cases.push(pls1_fit::small_n50_d10_k1(root)?);
    cases.push(pls1_fit::small_n50_d10_k3(root)?);
    cases.push(pls1_fit::small_n50_d10_sequence(root)?);
    cases.push(pls1_fit::wide_n30_d100_k1(root)?);
    cases.push(pls1_fit::wide_n30_d100_k3(root)?);
    cases.push(pls1_fit::skinny_n200_d5_k1(root)?);
    cases.push(pls1_fit::weighted_n50_d10_k2(root)?);

    cases.push(pls1_find_k_optimal::r2_se(root)?);
    cases.push(pls1_find_k_optimal::r2_max(root)?);
    cases.push(pls1_find_k_optimal::bic(root)?);
    cases.push(pls1_find_k_optimal::r2_se_diagnostic(root)?);

    cases.push(pls1_find_k_sequence::raw_perm(root)?);
    cases.push(pls1_find_k_sequence::split_nb(root)?);
    cases.push(pls1_find_k_sequence::split_exact(root)?);
    cases.push(pls1_find_k_sequence::e(root)?);

    cases.push(pls1_confirmatory_test::raw_perm(root)?);
    cases.push(pls1_confirmatory_test::split_nb(root)?);
    cases.push(pls1_confirmatory_test::split_exact(root)?);
    cases.push(pls1_confirmatory_test::split_exact_k1(root)?);
    cases.push(pls1_confirmatory_test::score(root)?);
    cases.push(pls1_confirmatory_test::e(root)?);
    cases.push(pls1_confirmatory_test::split_nb_ci(root)?);
    cases.push(pls1_confirmatory_test::weighted_raw_perm(root)?);
    cases.push(pls1_confirmatory_test::weighted_split_nb(root)?);
    cases.push(pls1_confirmatory_test::weighted_split_exact(root)?);
    cases.push(pls1_confirmatory_test::weighted_score(root)?);
    cases.push(pls1_confirmatory_test::weighted_e(root)?);
    cases.push(pls1_confirmatory_test::raw_perm_wide(root)?);

    cases.push(pls1_predict::basic_n80_d6_k2(root)?);
    cases.push(rotate::varimax_d6_k2(root)?);
    cases.push(preprocess::n50_d10_with_weights(root)?);
    cases.push(pls1_perm_null::basic_n80_d6_k2(root)?);
    cases.push(pls1_rotation_stability::n80_d6_k2(root)?);

    cases.push(spls1_fit::wide_n30_d100_k2_keep8(root)?);
    cases.push(spls1_fit::small_n50_d10_k2_keep3(root)?);
    cases.push(spls1_find_keep_optimal::k1(root)?);
    cases.push(spls1_find_k_optimal::r2_se_keep3(root)?);
    cases.push(spls1_find_k_sequence::split_nb_keep3(root)?);
    cases.push(spls1_find_k_sequence::split_nb_keep3_gated(root)?);

    cases.push(pls3::fit_small_n50_p10_q4_k1(root)?);
    cases.push(pls3::fit_small_n50_p10_q4_k3(root)?);
    cases.push(pls3::fit_wide_n30_p100_q3_k2(root)?);
    cases.push(pls3::transform_basic_n80_p6_q3_k2(root)?);
    cases.push(pls3::confirmatory_split_exact(root)?);
    cases.push(pls3::confirmatory_split_exact_wide(root)?);
    cases.push(pls3::confirmatory_split_nb(root)?);

    Ok(cases)
}
