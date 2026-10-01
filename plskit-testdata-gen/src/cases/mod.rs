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
pub mod spls3_fit;

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
    default_tolerance, faer_col_to_array, faer_mat_to_array, i64_vec, ndarray_to_faer_col,
    ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data, synth_xy,
};

/// A case's input arrays: `X`, `y` and, for weighted cases, `weights`.
///
/// [`Xyw::write`] stores them in that order, the order every single-response
/// case writes, so two cases that share an inputs file write the same bytes.
pub(crate) struct Xyw {
    pub x: ndarray::Array2<f64>,
    pub y: ndarray::Array1<f64>,
    pub w: Option<ndarray::Array1<f64>>,
}

impl Xyw {
    /// Write `X`, `y` (and `weights` when present) to `path`.
    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        let mut w = crate::npz::NpzWriter::create(path)?;
        w.add_f64("X", &self.x.clone().into_dyn())?;
        w.add_f64("y", &self.y.clone().into_dyn())?;
        if let Some(wt) = &self.w {
            w.add_f64("weights", &wt.clone().into_dyn())?;
        }
        w.finish()
    }

    pub(crate) fn x_faer(&self) -> faer::Mat<f64> {
        ndarray_to_faer_mat(&self.x)
    }

    pub(crate) fn y_faer(&self) -> faer::Col<f64> {
        ndarray_to_faer_col(&self.y)
    }

    pub(crate) fn w_faer(&self) -> Option<faer::Col<f64>> {
        self.w.as_ref().map(ndarray_to_faer_col)
    }
}

/// Weights of the weighted n = 80, d = 6 fixtures: uneven, mean not one, and
/// zero on every tenth row. The entry points renormalize them, so the
/// fixtures pin the √w row scaling, zero-weight rows included.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn perm_null_weights(n: usize) -> ndarray::Array1<f64> {
    ndarray::Array1::from_shape_fn(n, |i| {
        if i % 10 == 9 {
            0.0
        } else {
            1.0 + (i % 4) as f64 * 0.5
        }
    })
}

/// Inputs stem of [`weighted_n80_d6`]: the file
/// `pls1_perm_null::weighted_n80_d6_k2` writes, shared by every case on it.
pub(crate) const WEIGHTED_N80_D6_INPUTS: &str = "pls1_perm_null_weighted_n80_d6_k2";

/// `synth_data(80, 6, 2, 4.0, 42)` with [`perm_null_weights`]`(80)`.
pub(crate) fn weighted_n80_d6() -> Xyw {
    let (x, y) = synth_data(80, 6, 2, 4.0, 42);
    Xyw {
        x,
        y,
        w: Some(perm_null_weights(80)),
    }
}

/// Inputs stem of [`weighted_prestd_n80_d6`].
pub(crate) const WEIGHTED_PRESTD_N80_D6_INPUTS: &str = "pls1_weighted_prestd_n80_d6_inputs";

/// [`weighted_n80_d6`] with X and y standardized by their unweighted moments,
/// same weights. Run weighted with `pre_standardized = true`, the call keeps
/// these unweighted-standardized columns; with `pre_standardized = false` it
/// restandardizes by the weighted moments. So a fixture on these inputs moves
/// when the `pre_standardized` flag is dropped.
pub(crate) fn weighted_prestd_n80_d6() -> Xyw {
    let raw = weighted_n80_d6();
    let (xs, _, _) = plskit::linalg::standardize(raw.x_faer().as_ref());
    let (ys, _, _) = plskit::linalg::standardize1(raw.y_faer().as_ref());
    Xyw {
        x: ndarray::Array2::from_shape_fn((xs.nrows(), xs.ncols()), |(i, j)| xs[(i, j)]),
        y: ndarray::Array1::from_shape_fn(ys.nrows(), |i| ys[i]),
        w: raw.w,
    }
}

/// Create the parent directories of `root/rel_inputs` and `root/rel_outputs`
/// and return both absolute paths.
pub(crate) fn case_files(
    root: &Path,
    rel_inputs: &str,
    rel_outputs: &str,
) -> std::io::Result<(PathBuf, PathBuf)> {
    let (abs_inputs, abs_outputs) = (root.join(rel_inputs), root.join(rel_outputs));
    for dir in [abs_inputs.parent(), abs_outputs.parent()]
        .into_iter()
        .flatten()
    {
        std::fs::create_dir_all(dir)?;
    }
    Ok((abs_inputs, abs_outputs))
}

/// Generator guard for a fixture meant to discriminate an option: fail unless
/// `with` and `without` (the same call with and without it) differ somewhere
/// by more than `1e-6`, far outside the corpus tolerances, so a consumer that
/// drops the option fails the fixture.
///
/// # Errors
/// Returns an error when the two outputs agree to `1e-6`.
pub(crate) fn ensure_moved(case: &str, field: &str, with: &[f64], without: &[f64]) -> Result<()> {
    let moved = with.len() != without.len()
        || with
            .iter()
            .zip(without)
            .any(|(a, b)| a.is_nan() != b.is_nan() || (a - b).abs() > 1e-6);
    anyhow::ensure!(
        moved,
        "{case}: {field} does not move with the option ({with:?} vs {without:?}); the fixture would not discriminate it"
    );
    Ok(())
}

/// The manifest entry of a case whose files are already written under `root`,
/// at the default tolerance.
///
/// # Errors
/// Returns an error if either file cannot be hashed.
pub(crate) fn manifest_case(
    root: &Path,
    name: &str,
    function: &str,
    rel_inputs: String,
    rel_outputs: String,
    kwargs: serde_json::Value,
) -> Result<Case> {
    let hashes = crate::manifest::Hashes {
        inputs_sha256: crate::npz::sha256_of_file(&root.join(&rel_inputs))?,
        outputs_sha256: crate::npz::sha256_of_file(&root.join(&rel_outputs))?,
    };
    Ok(Case {
        name: name.to_string(),
        function: function.to_string(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs,
        hashes,
        tolerance: Some(default_tolerance()),
    })
}

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

    /// Convert a 2-D `faer::Mat` to a dynamic `ndarray` for the npz writer.
    ///
    /// The single shared version: every case module imports this one.
    pub fn faer_mat_to_array(m: &Mat<f64>) -> ndarray::ArrayD<f64> {
        ndarray::Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_dyn()
    }

    /// Wrap a scalar `f64` as a 0-D `ndarray::ArrayD<f64>`.
    pub fn scalar_f64(v: f64) -> ndarray::ArrayD<f64> {
        ndarray::arr0(v).into_dyn()
    }

    /// Convert a `Vec<i64>` to a 1-D `ndarray::ArrayD<i64>`.
    pub fn i64_vec(v: Vec<i64>) -> ndarray::ArrayD<i64> {
        ndarray::Array1::from_vec(v).into_dyn()
    }

    /// Wrap a scalar `i64` as a 0-D `ndarray::ArrayD<i64>`.
    pub fn scalar_i64(v: i64) -> ndarray::ArrayD<i64> {
        ndarray::arr0(v).into_dyn()
    }

    /// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10,
    /// rtol=1e-14 (`|a − e| ≤ atol + rtol·|e|`; `testdata/README.md`
    /// "Tolerance").
    ///
    /// The single shared version, recorded as each `Case`'s `tolerance`
    /// field in `manifest.json` and read by the settle step. The wrappers'
    /// corpus tests hard-code the same numbers. It never touches fixture bytes.
    pub fn default_tolerance() -> serde_json::Value {
        serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10, "rtol": 1e-14})
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
    cases.push(pls1_fit::pre_standardized_weighted_n50_d10_k3(root)?);
    cases.push(pls1_fit::offset_n50_d10_k3(root)?);
    cases.push(pls1_fit::mean_heavy_weighted_n50_d10_k3(root)?);

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
    cases.push(pls1_confirmatory_test::split_nb_ci_level80(root)?);
    cases.push(pls1_confirmatory_test::weighted_raw_perm(root)?);
    cases.push(pls1_confirmatory_test::weighted_split_nb(root)?);
    cases.push(pls1_confirmatory_test::weighted_split_exact(root)?);
    cases.push(pls1_confirmatory_test::weighted_score(root)?);
    cases.push(pls1_confirmatory_test::weighted_e(root)?);
    cases.push(pls1_confirmatory_test::raw_perm_wide(root)?);
    cases.push(pls1_confirmatory_test::raw_perm_wide_k2(root)?);
    cases.push(pls1_confirmatory_test::split_exact_wide_k2(root)?);
    cases.push(pls1_confirmatory_test::raw_perm_tall_k2(root)?);
    cases.push(pls1_confirmatory_test::score_wide_pre_standardized(root)?);
    cases.push(pls1_confirmatory_test::weighted_split_exact_k2(root)?);
    cases.push(pls1_confirmatory_test::split_exact_wide_k1(root)?);
    cases.push(pls1_confirmatory_test::weighted_split_nb_ci(root)?);
    cases.push(pls1_confirmatory_test::weighted_prestd_split_nb_ci(root)?);
    cases.push(pls1_confirmatory_test::weighted_score_wide_pre_standardized(root)?);

    cases.push(pls1_predict::basic_n80_d6_k2(root)?);
    cases.push(rotate::varimax_d6_k2(root)?);
    cases.push(preprocess::n50_d10_with_weights(root)?);
    cases.push(pls1_perm_null::basic_n80_d6_k2(root)?);
    cases.push(pls1_perm_null::weighted_n80_d6_k2(root)?);
    cases.push(pls1_perm_null::wide_n60_d3000_k1(root)?);
    cases.push(pls1_perm_null::wide_n60_d3000_k2(root)?);
    cases.push(pls1_perm_null::tall_n2000_d50_k3(root)?);
    cases.push(pls1_perm_null::tall_weighted_n2000_d50_k2(root)?);
    cases.push(pls1_perm_null::pre_standardized_n80_d6_k2(root)?);
    cases.push(pls1_perm_null::weighted_n40_d1_k1(root)?);
    cases.push(pls1_perm_null::weighted_prestd_n80_d6_k2(root)?);
    cases.push(pls1_rotation_stability::n80_d6_k2(root)?);
    cases.push(pls1_rotation_stability::weighted_n80_d6_k2(root)?);
    cases.push(pls1_rotation_stability::weighted_prestd_n80_d6_k2(root)?);

    cases.push(spls1_fit::wide_n30_d100_k2_keep8(root)?);
    cases.push(spls1_fit::small_n50_d10_k2_keep3(root)?);
    cases.push(spls1_find_keep_optimal::k1(root)?);
    cases.push(spls1_find_k_optimal::r2_se_keep3(root)?);
    cases.push(spls1_find_k_optimal::r2_se_keep3_weighted(root)?);
    cases.push(spls1_find_k_sequence::split_nb_keep3(root)?);
    cases.push(spls1_find_k_sequence::split_nb_keep3_gated(root)?);
    cases.push(spls1_find_k_sequence::split_exact_keep3(root)?);
    cases.push(spls1_find_k_sequence::split_exact_tall_keep10(root)?);
    cases.push(spls1_find_k_sequence::raw_perm_keep3(root)?);
    cases.push(spls1_find_k_sequence::e_keep3(root)?);
    cases.push(spls1_find_k_sequence::split_exact_keep3_weighted_prestd(
        root,
    )?);

    cases.push(pls3::fit_small_n50_p10_q4_k1(root)?);
    cases.push(pls3::fit_small_n50_p10_q4_k3(root)?);
    cases.push(pls3::fit_wide_n30_p100_q3_k2(root)?);
    cases.push(pls3::transform_basic_n80_p6_q3_k2(root)?);
    cases.push(pls3::confirmatory_split_exact(root)?);
    cases.push(pls3::confirmatory_split_exact_wide(root)?);
    cases.push(pls3::confirmatory_split_nb(root)?);

    cases.push(spls3_fit::small_n50_p10_q4_keep3_2_k2(root)?);
    cases.push(spls3_fit::dense_endpoint_n50_p10_q4_k2(root)?);
    cases.push(spls3_fit::wide_n30_p100_q3_keep10_2_k2(root)?);

    Ok(cases)
}
