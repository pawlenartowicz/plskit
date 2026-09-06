//! PLS3 / PLSSVD fixture cases: `pls3_fit`, `pls3_transform`,
//! `pls3_confirmatory_test`.

use std::path::Path;

use anyhow::Result;
use plskit::{
    pls3_confirmatory_test, pls3_fit, pls3_transform, ConfirmatoryArgs, Pls3ConfirmatoryTestOpts,
    Pls3FitOpts, TransformWhich,
};

use crate::cases::{
    faer_col_to_array, ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_xy, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Shared signal strength and factor count for every PLS3 fixture.
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const CASE_SEED: u64 = 42;

/// Convert a 2-D `faer::Mat` to a dynamic `ndarray` for the npz writer.
///
/// Module-private on purpose: there is no shared version, and
/// `cases/rotate.rs` carries its own copy for the same reason. The `Col`
/// equivalent *is* shared — this module imports
/// `crate::cases::faer_col_to_array` rather than repeating it.
fn faer_mat_to_array(m: &plskit::Mat<f64>) -> ndarray::ArrayD<f64> {
    ndarray::Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_dyn()
}

/// Synth parameters for a fixed-k `pls3_fit` case.
struct Pls3FitCase<'a> {
    name: &'a str,
    n: usize,
    p: usize,
    q: usize,
    k: usize,
}

/// Generic `pls3_fit` case writer, shared by all three fixed-k cases.
#[allow(clippy::many_single_char_names)]
fn fit_case(root: &Path, c: &Pls3FitCase<'_>) -> Result<Case> {
    let function = "pls3_fit";
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
    let m = pls3_fit(xf.as_ref(), yf.as_ref(), c.k, None, Pls3FitOpts::default())?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("U", &faer_mat_to_array(&m.u_saliences))?;
        w.add_f64("V", &faer_mat_to_array(&m.v_saliences))?;
        w.add_f64("singular_values", &faer_col_to_array(&m.singular_values))?;
        w.add_f64("x_scores", &faer_mat_to_array(&m.x_scores))?;
        w.add_f64("y_scores", &faer_mat_to_array(&m.y_scores))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(m.k_used)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: c.name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": c.k, "seed": CASE_SEED}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: small (n=50, p=10, q=4), k=1. Covers the LV1-only fit the
/// confirmatory test uses internally.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls3_fit` fails.
pub fn fit_small_n50_p10_q4_k1(root: &Path) -> Result<Case> {
    fit_case(
        root,
        &Pls3FitCase {
            name: "pls3_fit_small_n50_p10_q4_k1",
            n: 50,
            p: 10,
            q: 4,
            k: 1,
        },
    )
}

/// Case: small (n=50, p=10, q=4), k=3 — multiple components out of one SVD.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls3_fit` fails.
pub fn fit_small_n50_p10_q4_k3(root: &Path) -> Result<Case> {
    fit_case(
        root,
        &Pls3FitCase {
            name: "pls3_fit_small_n50_p10_q4_k3",
            n: 50,
            p: 10,
            q: 4,
            k: 3,
        },
    )
}

/// Case: wide (n=30, p=100, q=3), k=2 — the `p ≫ n` regime PLS3 targets.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls3_fit` fails.
pub fn fit_wide_n30_p100_q3_k2(root: &Path) -> Result<Case> {
    fit_case(
        root,
        &Pls3FitCase {
            name: "pls3_fit_wide_n30_p100_q3_k2",
            n: 30,
            p: 100,
            q: 3,
            k: 2,
        },
    )
}

/// Case: `pls3_transform` at k=2, `which="both"`, applied to a *second*
/// dataset the fit never saw.
///
/// Transforming the training block would pin nothing — the scores would be
/// byte-identical to the fit fixture's in-sample `x_scores` / `y_scores`,
/// so the case would only re-check `pls3_fit`. The held-out block exercises
/// what `pls3_transform` actually does that the fit does not: apply the
/// stored standardization moments to new rows. Same two-seed shape as
/// `pls1_predict` (`cases/pls1_predict.rs`).
///
/// # Errors
/// Returns an error if fixture files cannot be written or either call fails.
///
/// # Panics
/// Panics if `pls3_transform` returns `None` for a block requested via
/// `which=Both` — unreachable given the `which` value passed here.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub fn transform_basic_n80_p6_q3_k2(root: &Path) -> Result<Case> {
    let name = "pls3_transform_basic_n80_p6_q3_k2";
    let function = "pls3_transform";
    let paths = CasePaths::build(root, function, name)?;
    let (x, y) = synth_xy(80, 6, 3, SYNTH_K_SIGNAL, SYNTH_SNR, CASE_SEED);
    let (x_new, y_new) = synth_xy(40, 6, 3, SYNTH_K_SIGNAL, SYNTH_SNR, CASE_SEED + 1);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("Y", &y.clone().into_dyn())?;
        w.add_f64("X_new", &x_new.clone().into_dyn())?;
        w.add_f64("Y_new", &y_new.clone().into_dyn())?;
        w.finish()?;
    }

    let xf = ndarray_to_faer_mat(&x);
    let yf = ndarray_to_faer_mat(&y);
    let xf_new = ndarray_to_faer_mat(&x_new);
    let yf_new = ndarray_to_faer_mat(&y_new);
    let m = pls3_fit(xf.as_ref(), yf.as_ref(), 2, None, Pls3FitOpts::default())?;
    let s = pls3_transform(
        &m,
        Some(xf_new.as_ref()),
        Some(yf_new.as_ref()),
        TransformWhich::Both,
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64(
            "x_scores",
            &faer_mat_to_array(s.x_scores.as_ref().expect("which=both yields x_scores")),
        )?;
        w.add_f64(
            "y_scores",
            &faer_mat_to_array(s.y_scores.as_ref().expect("which=both yields y_scores")),
        )?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "k": 2, "which": "both",
            "seed_train": CASE_SEED, "seed_new": CASE_SEED + 1
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Synth shape and permutation counts for one `pls3_confirmatory_test`
/// `split_exact` case.
struct Pls3SplitExactCase<'a> {
    name: &'a str,
    n: usize,
    p: usize,
    q: usize,
    n_perm: usize,
    n_splits: usize,
}

/// Generic `pls3_confirmatory_test` / `split_exact` case writer.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
///
/// # Panics
/// Panics if the result omits `n_perm` / `n_splits` — unreachable for
/// `method=split_exact`, which always reports both.
#[allow(clippy::many_single_char_names)]
fn split_exact_case(root: &Path, c: &Pls3SplitExactCase<'_>) -> Result<Case> {
    let function = "pls3_confirmatory_test";
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
    let r = pls3_confirmatory_test(
        xf.as_ref(),
        yf.as_ref(),
        1,
        Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: c.n_perm,
                n_splits: c.n_splits,
            },
            seed: Some(CASE_SEED),
            ..Pls3ConfirmatoryTestOpts::default()
        },
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        w.add_f64("statistic", &scalar_f64(r.statistic))?;
        w.add_string("method", &r.method)?;
        w.add_i64("k", &scalar_i64(i64::try_from(r.k)?))?;
        w.add_i64(
            "n_perm",
            &scalar_i64(i64::try_from(
                r.n_perm.expect("split_exact reports n_perm"),
            )?),
        )?;
        w.add_i64(
            "n_splits",
            &scalar_i64(i64::try_from(
                r.n_splits.expect("split_exact reports n_splits"),
            )?),
        )?;
        w.add_f64("n_eff", &scalar_f64(r.n_eff))?;
        // The resolved seed is part of the result and gets pinned, matching
        // `cases/pls1_confirmatory_test.rs`.
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: c.name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "k": 1,
            "method": "split_exact",
            "args": {"n_perm": c.n_perm, "n_splits": c.n_splits},
            "seed": CASE_SEED
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: `pls3_confirmatory_test` with `method=split_exact`, `n_perm=200`,
/// `n_splits=30`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_split_exact(root: &Path) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name: "pls3_confirmatory_split_exact",
            n: 80,
            p: 6,
            q: 3,
            n_perm: 200,
            n_splits: 30,
        },
    )
}

/// Case: `pls3_confirmatory_test` with `method=split_exact` on a wide design
/// (n=30, p=100, q=3), `n_perm=100`, `n_splits=30`, `seed=42`.
///
/// The corpus's only `pls3_confirmatory_test` fixture that reaches the Gram
/// route in `plskit-rs/src/dual_route.rs`. The permutation branch takes that
/// route when `n_tr·(B·q + p) < p·B·q` holds, with `n_tr = 15` (half of `n`,
/// via `resample::split_sizes`), `q = 3` and `B = n_perm + 1`. At this shape
/// the inequality reduces to `n_perm ≥ 5`, so `n_perm = 100` clears it by
/// 20×. Narrowing X or shrinking `n_perm` past that bound silently sends the
/// fixture back onto the primal route and the Gram route loses its only
/// cross-language coverage on this family.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_split_exact_wide(root: &Path) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name: "pls3_confirmatory_split_exact_wide",
            n: 30,
            p: 100,
            q: 3,
            n_perm: 100,
            n_splits: 30,
        },
    )
}

/// Case: `pls3_confirmatory_test` with `method=split_nb`, `n_splits=30`,
/// `seed=42`. Same design as `confirmatory_split_exact`, which clears the
/// auto-gate on X, so the run is not rerouted and the pinned `method` is
/// `split_nb`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
///
/// # Panics
/// Panics if the result omits `n_splits` — unreachable for a split method.
pub fn confirmatory_split_nb(root: &Path) -> Result<Case> {
    let name = "pls3_confirmatory_split_nb";
    let function = "pls3_confirmatory_test";
    let paths = CasePaths::build(root, function, name)?;
    let (x, y) = synth_xy(80, 6, 3, SYNTH_K_SIGNAL, SYNTH_SNR, CASE_SEED);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("Y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let xf = ndarray_to_faer_mat(&x);
    let yf = ndarray_to_faer_mat(&y);
    let r = pls3_confirmatory_test(
        xf.as_ref(),
        yf.as_ref(),
        1,
        Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 30,
                force: false,
            },
            seed: Some(CASE_SEED),
            ..Pls3ConfirmatoryTestOpts::default()
        },
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        w.add_f64("statistic", &scalar_f64(r.statistic))?;
        w.add_string("method", &r.method)?;
        w.add_i64("k", &scalar_i64(i64::try_from(r.k)?))?;
        // No `n_perm` key: split_nb runs no permutations. `stable_rank` is
        // what the gate saw, mirroring `cases/pls1_confirmatory_test.rs`.
        w.add_i64(
            "n_splits",
            &scalar_i64(i64::try_from(
                r.n_splits.expect("split_nb reports n_splits"),
            )?),
        )?;
        if let Some(sr) = r.stable_rank {
            w.add_f64("stable_rank", &scalar_f64(sr))?;
        }
        w.add_f64("n_eff", &scalar_f64(r.n_eff))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "k": 1,
            "method": "split_nb",
            "args": {"n_splits": 30},
            "seed": CASE_SEED
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
