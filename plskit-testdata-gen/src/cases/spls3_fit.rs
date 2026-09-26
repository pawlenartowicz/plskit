//! `spls3_fit` fixture cases (sparse PLS3 / sparse PLSSVD).

use std::path::Path;

use anyhow::Result;
use plskit::{spls3_fit, Pls3FitOpts};

use crate::cases::{
    default_tolerance, faer_col_to_array, faer_mat_to_array, i64_vec, ndarray_to_faer_mat,
    scalar_i64, synth_xy, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};

/// Shared signal strength and factor count for every `spls3_fit` fixture,
/// matching `cases/pls3.rs`'s constants so the dense-endpoint case can
/// reuse the same `(n, p, q)` shape and land on the identical byte stream
/// as `pls3_fit_small_n50_p10_q4_k3`.
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const CASE_SEED: u64 = 42;

/// Synth shape and sparse settings for one `spls3_fit` case.
///
/// Every field here is the single source of truth for both the fit and the
/// manifest `kwargs` block: `fit_case` passes these values to `spls3_fit`
/// and then derives the JSON from the same values, so a producer/consumer
/// divergence is not expressible. (`plskit-rs/tests/corpus.rs` and
/// `plskit-py/tests/test_corpus.py` both re-fit from those `kwargs`.)
struct Spls3FitCase<'a> {
    name: &'a str,
    n: usize,
    p: usize,
    q: usize,
    k: usize,
    keep_x: usize,
    keep_y: usize,
    /// Fit knobs. Only `max_iter` and `tol` are recorded in `kwargs`:
    /// `par` is a parallelism choice with no wrapper equivalent, and the
    /// `pre_standardized_*` flags are spelled with a capital `X`/`Y` in the
    /// wrappers, so serializing the whole struct would break the `**kwargs`
    /// splat in `plskit-py/tests/test_corpus.py`.
    opts: Pls3FitOpts,
}

/// Generic `spls3_fit` case writer, shared by all three cases.
#[allow(clippy::many_single_char_names)]
fn fit_case(root: &Path, c: &Spls3FitCase<'_>) -> Result<Case> {
    let function = "spls3_fit";
    let paths = CasePaths::build(root, function, c.name)?;
    let (x, y) = synth_xy(c.n, c.p, c.q, SYNTH_K_SIGNAL, SYNTH_SNR, CASE_SEED);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("Y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let xf = ndarray_to_faer_mat(&x);
    let yf = ndarray_to_faer_mat(&y);
    let model = spls3_fit(
        xf.as_ref(),
        yf.as_ref(),
        c.k,
        c.keep_x,
        c.keep_y,
        None,
        c.opts,
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("U", &faer_mat_to_array(&model.u_saliences))?;
        w.add_f64("V", &faer_mat_to_array(&model.v_saliences))?;
        w.add_f64(
            "singular_values",
            &faer_col_to_array(&model.singular_values),
        )?;
        w.add_f64("x_scores", &faer_mat_to_array(&model.x_scores))?;
        w.add_f64("y_scores", &faer_mat_to_array(&model.y_scores))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        w.add_i64(
            "keep_X",
            &scalar_i64(i64::try_from(
                model.keep_x.expect("spls3_fit always sets keep_x"),
            )?),
        )?;
        w.add_i64(
            "keep_Y",
            &scalar_i64(i64::try_from(
                model.keep_y.expect("spls3_fit always sets keep_y"),
            )?),
        )?;
        // bool has no npz scalar type in this writer; 0/1 keeps the
        // fixture readable from numpy without a dtype special case.
        w.add_i64(
            "converged",
            &i64_vec(
                model
                    .converged
                    .expect("spls3_fit always sets converged")
                    .into_iter()
                    .map(i64::from)
                    .collect(),
            ),
        )?;
        w.add_i64(
            "n_iter",
            &i64_vec(
                model
                    .n_iter
                    .expect("spls3_fit always sets n_iter")
                    .into_iter()
                    .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
                    .collect(),
            ),
        )?;
        w.finish()?;
    }

    Ok(Case {
        name: c.name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "k": c.k,
            "keep_X": c.keep_x,
            "keep_Y": c.keep_y,
            "max_iter": c.opts.max_iter,
            "tol": c.opts.tol,
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: small (n=50, p=10, q=4), k=2, `keep_X=3`, `keep_Y=2`: the ordinary
/// sparse regime with both blocks thresholded.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls3_fit` fails.
pub fn small_n50_p10_q4_keep3_2_k2(root: &Path) -> Result<Case> {
    fit_case(
        root,
        &Spls3FitCase {
            name: "spls3_fit_small_n50_p10_q4_keep3_2_k2",
            n: 50,
            p: 10,
            q: 4,
            k: 2,
            keep_x: 3,
            keep_y: 2,
            opts: Pls3FitOpts::default(),
        },
    )
}

/// Case: `keep_X = n_features` and `keep_Y = n_targets` on the same
/// `(n=50, p=10, q=4)` shape as `pls3_fit_small_n50_p10_q4_k3`. This is the
/// dense endpoint: `spls3_fit` delegates to `pls3_fit` internally and the
/// shared fields (`U`, `V`, `singular_values`, `x_scores`, `y_scores`,
/// `k_used`) must be byte-identical to that fixture's first two components,
/// since PLS3 components are nested. The case is registered in `mod.rs`, so
/// every wrapper's corpus test re-proves the delegation with this fixture
/// rather than leaving it to the Rust unit test
/// `spls3_dense_endpoint_is_bit_identical_to_pls3` in `plskit-rs/src/pls3.rs`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls3_fit` fails.
pub fn dense_endpoint_n50_p10_q4_k2(root: &Path) -> Result<Case> {
    fit_case(
        root,
        &Spls3FitCase {
            name: "spls3_fit_dense_endpoint_n50_p10_q4_k2",
            n: 50,
            p: 10,
            q: 4,
            k: 2,
            keep_x: 10,
            keep_y: 4,
            opts: Pls3FitOpts::default(),
        },
    )
}

/// Case: wide (n=30, p=100, q=3), k=2, `keep_X=10`, `keep_Y=2`: the `p ≫ n`
/// regime, sparse on both sides.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls3_fit` fails.
pub fn wide_n30_p100_q3_keep10_2_k2(root: &Path) -> Result<Case> {
    fit_case(
        root,
        &Spls3FitCase {
            name: "spls3_fit_wide_n30_p100_q3_keep10_2_k2",
            n: 30,
            p: 100,
            q: 3,
            k: 2,
            keep_x: 10,
            keep_y: 2,
            opts: Pls3FitOpts::default(),
        },
    )
}
