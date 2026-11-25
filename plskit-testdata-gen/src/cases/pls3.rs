//! PLS3 / PLSSVD fixture cases: `pls3_fit`, `pls3_transform`,
//! `pls3_confirmatory_test`.

use std::path::Path;

use anyhow::Result;
use plskit::{
    ConfirmatoryArgs, Pls3ConfirmatoryTestOpts, Pls3FitOpts, TransformWhich, gram_matrix,
    pls3_confirmatory_test, pls3_confirmatory_test_gram, pls3_fit, pls3_transform,
};

use crate::cases::pls1_confirmatory_test::write_k_set;
use crate::cases::{
    CasePaths, add_count, case_files, faer_col_to_array, faer_mat_to_array, ndarray_to_faer_mat,
    scalar_f64, scalar_i64, synth_xy,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{NpzWriter, sha256_of_file};

/// Shared signal strength and factor count of the PLS3 fixtures; the
/// weak-signal cases set their own `snr`.
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const CASE_SEED: u64 = 42;

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
    })
}

/// Synth shape and permutation counts for one `pls3_confirmatory_test`
/// `split_exact` case. `auto` requests `test_method=auto` instead, on a
/// design that resolves to `split_exact`. `gram` passes `gram_matrix(X)`
/// in place of X (kwarg `"gram": true`).
struct Pls3SplitExactCase<'a> {
    name: &'a str,
    /// Stem of the inputs file, `inputs/{inputs_name}.npz`. Cases that share
    /// a file must share its design, seed and `snr`.
    inputs_name: &'a str,
    auto: bool,
    gram: bool,
    /// `Some(rotate)` runs `sv_perm` with that flag instead; `n_splits` is then unused.
    sv_rotate: Option<bool>,
    snr: f64,
    n: usize,
    p: usize,
    q: usize,
    n_perm: usize,
    n_splits: usize,
}

/// Generic `pls3_confirmatory_test` / `split_exact` case writer.
///
/// # Errors
/// Returns an error if fixture files cannot be written, the test fails, or
/// [`crate::cases::ensure_off_floor`] rejects its p-value.
///
/// # Panics
/// Panics if the result omits `n_perm` / `n_splits` — unreachable for
/// `test_method=split_exact`, which always reports both.
#[allow(clippy::many_single_char_names)]
fn split_exact_case(root: &Path, c: &Pls3SplitExactCase<'_>) -> Result<Case> {
    let function = "pls3_confirmatory_test";
    let rel_inputs = format!("inputs/{}.npz", c.inputs_name);
    let rel_outputs = format!("outputs/{function}/{}.npz", c.name);
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    let (x, y) = synth_xy(c.n, c.p, c.q, SYNTH_K_SIGNAL, c.snr, CASE_SEED);

    {
        let mut w = NpzWriter::create(&abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("Y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let xf = ndarray_to_faer_mat(&x);
    let yf = ndarray_to_faer_mat(&y);
    let opts = Pls3ConfirmatoryTestOpts {
        args: if let Some(rotate) = c.sv_rotate {
            ConfirmatoryArgs::SvPerm {
                n_perm: c.n_perm,
                rotate,
            }
        } else if c.auto {
            ConfirmatoryArgs::Auto {
                n_perm: c.n_perm,
                n_splits: c.n_splits,
            }
        } else {
            ConfirmatoryArgs::SplitExact {
                n_perm: c.n_perm,
                n_splits: c.n_splits,
            }
        },
        seed: Some(CASE_SEED),
        ..Pls3ConfirmatoryTestOpts::default()
    };
    let r = if c.gram {
        pls3_confirmatory_test_gram(&gram_matrix(xf.as_ref(), false)?, yf.as_ref(), 1, opts)?
    } else {
        pls3_confirmatory_test(xf.as_ref(), yf.as_ref(), 1, opts)?
    };
    crate::cases::ensure_off_floor(c.name, r.pvalue, r.n_perm)?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        w.add_f64(
            "statistic",
            &scalar_f64(r.statistic.expect("PLS3 tests one k")),
        )?;
        w.add_string("test_method", &r.test_method)?;
        write_k_set(&mut w, &r)?;
        w.add_i64(
            "n_perm",
            &scalar_i64(i64::try_from(
                r.n_perm.expect("split_exact reports n_perm"),
            )?),
        )?;
        if let Some(ns) = r.n_splits {
            w.add_i64("n_splits", &scalar_i64(i64::try_from(ns)?))?;
        }
        add_count(&mut w, "n_failed_splits", r.n_failed_splits)?;
        add_count(&mut w, "n_failed_perms", r.n_failed_perms)?;
        add_count(&mut w, "n_failed_folds", r.n_failed_folds)?;
        if let Some(sr) = r.stable_rank {
            w.add_f64("stable_rank", &scalar_f64(sr))?;
        }
        w.add_f64("n_eff", &scalar_f64(r.n_eff))?;
        // The resolved seed is part of the result and gets pinned, matching
        // `cases/pls1_confirmatory_test.rs`.
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: c.name.to_string(),
        function: function.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs: {
            let mut kw = serde_json::json!({
                "k": 1,
                "test_method": if c.sv_rotate.is_some() { "sv_perm" } else if c.auto { "auto" } else { "split_exact" },
                "args": {"n_perm": c.n_perm, "n_splits": c.n_splits},
                "seed": CASE_SEED
            });
            if let Some(rotate) = c.sv_rotate {
                kw["args"] = serde_json::json!({"n_perm": c.n_perm, "rotate": rotate});
            }
            if c.gram {
                kw["gram"] = serde_json::Value::Bool(true);
            }
            kw
        },
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
    })
}

/// Signal of `inputs/pls3_confirmatory_split_exact_weak.npz` (n=80, p=6,
/// q=3): puts p mid-range so the fixtures pin the null columns' ranks.
const SPLIT_EXACT_WEAK_SNR: f64 = 0.6;

/// Case: `pls3_confirmatory_test` with `test_method=split_exact`, `n_perm=200`,
/// `n_splits=30`, `seed=42`, on the weak-signal inputs
/// `inputs/pls3_confirmatory_split_exact_weak.npz`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_split_exact(root: &Path) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name: "pls3_confirmatory_split_exact",
            inputs_name: "pls3_confirmatory_split_exact_weak",
            auto: false,
            gram: false,
            sv_rotate: None,
            snr: SPLIT_EXACT_WEAK_SNR,
            n: 80,
            p: 6,
            q: 3,
            n_perm: 200,
            n_splits: 30,
        },
    )
}

/// Case: `pls3_confirmatory_test` with `test_method=auto`, `n_perm=200`,
/// `n_splits=30`, `seed=42` on n=80, p=6, q=3. `auto` runs `split_exact` on
/// every PLS3 design. It reads the inputs of [`confirmatory_split_exact`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_auto_split_exact(root: &Path) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name: "pls3_confirmatory_auto_split_exact",
            inputs_name: "pls3_confirmatory_split_exact_weak",
            auto: true,
            gram: false,
            sv_rotate: None,
            snr: SPLIT_EXACT_WEAK_SNR,
            n: 80,
            p: 6,
            q: 3,
            n_perm: 200,
            n_splits: 30,
        },
    )
}

/// Case: `pls3_confirmatory_test` with `test_method=split_exact` on a wide design
/// (n=30, p=100, q=3), `n_perm=100`, `n_splits=30`, `seed=42`.
///
/// Its route is pinned in `plskit-rs/src/fixture_route_pins.rs`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_split_exact_wide(root: &Path) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name: "pls3_confirmatory_split_exact_wide",
            inputs_name: "pls3_confirmatory_split_exact_wide",
            auto: false,
            gram: false,
            sv_rotate: None,
            snr: SYNTH_SNR,
            n: 30,
            p: 100,
            q: 3,
            n_perm: 100,
            n_splits: 30,
        },
    )
}

/// Case: `pls3_confirmatory_test` on `gram_matrix(X)` in place of X, with
/// the design and counts of [`confirmatory_split_exact_wide`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_gram_split_exact(root: &Path) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name: "pls3_confirmatory_test_gram_split_exact",
            inputs_name: "pls3_confirmatory_test_gram_split_exact",
            auto: false,
            gram: true,
            sv_rotate: None,
            snr: SYNTH_SNR,
            n: 30,
            p: 100,
            q: 3,
            n_perm: 100,
            n_splits: 30,
        },
    )
}

/// Weak signal for the raw `sv_perm` pair and the Gram `sv_perm` case with
/// `rotate`: at the shared `SYNTH_SNR` the observed singular value beats
/// every permutation and p sits at the floor.
const SV_PERM_RAW_SNR: f64 = 0.1;

/// A `pls3_confirmatory_test` `sv_perm` case (`n_perm = 99`).
fn sv_perm_case(
    root: &Path,
    (name, inputs_name): (&str, &str),
    (gram, rotate): (bool, bool),
    (n, p): (usize, usize),
    snr: f64,
) -> Result<Case> {
    split_exact_case(
        root,
        &Pls3SplitExactCase {
            name,
            inputs_name,
            auto: false,
            gram,
            sv_rotate: Some(rotate),
            snr,
            n,
            p,
            q: 3,
            n_perm: 99,
            n_splits: 0,
        },
    )
}

/// Case: `pls3_confirmatory_test` with `test_method=sv_perm`, `n_perm=99`,
/// `rotate=false` on n=80, p=6, q=3.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_sv_perm(root: &Path) -> Result<Case> {
    sv_perm_case(
        root,
        ("pls3_confirmatory_sv_perm", "pls3_confirmatory_sv_perm"),
        (false, false),
        (80, 6),
        SV_PERM_RAW_SNR,
    )
}

/// Case: [`confirmatory_sv_perm`] with `rotate=true`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_sv_perm_rotate(root: &Path) -> Result<Case> {
    sv_perm_case(
        root,
        (
            "pls3_confirmatory_sv_perm_rotate",
            "pls3_confirmatory_sv_perm_rotate",
        ),
        (false, true),
        (80, 6),
        SV_PERM_RAW_SNR,
    )
}

/// Case: `sv_perm` with `rotate=false` on `gram_matrix(X)` of the inputs of
/// [`confirmatory_sv_perm_rotate_gram`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_sv_perm_gram(root: &Path) -> Result<Case> {
    sv_perm_case(
        root,
        (
            "pls3_confirmatory_test_gram_sv_perm",
            "pls3_confirmatory_test_gram_sv_perm",
        ),
        (true, false),
        (30, 100),
        SYNTH_SNR,
    )
}

/// Case: `sv_perm` with `rotate=true` on `gram_matrix(X)` of the wide
/// design (n=30, p=100, q=3), on weak-signal inputs of its own.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the test fails.
pub fn confirmatory_sv_perm_rotate_gram(root: &Path) -> Result<Case> {
    sv_perm_case(
        root,
        (
            "pls3_confirmatory_test_gram_sv_perm_rotate",
            "pls3_confirmatory_test_gram_sv_perm_rotate_weak",
        ),
        (true, true),
        (30, 100),
        SV_PERM_RAW_SNR,
    )
}

/// Case: `pls3_confirmatory_test` with `test_method=split_nb`, `n_splits=30`,
/// `seed=42`, on n = 80, p = 6, q = 3 at `SYNTH_SNR`, whose X clears the
/// auto-gate, so the run is not rerouted and the pinned `test_method` is
/// `split_nb`.
///
/// # Errors
/// Returns an error if fixture files cannot be written, the test fails, or
/// [`crate::cases::ensure_off_floor`] rejects its p-value (only possible
/// after a reroute to a permutation method).
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
    crate::cases::ensure_off_floor(name, r.pvalue, r.n_perm)?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        w.add_f64(
            "statistic",
            &scalar_f64(r.statistic.expect("PLS3 tests one k")),
        )?;
        w.add_string("test_method", &r.test_method)?;
        write_k_set(&mut w, &r)?;
        // No `n_perm` key: split_nb runs no permutations. `stable_rank` is
        // what the gate saw, mirroring `cases/pls1_confirmatory_test.rs`.
        w.add_i64(
            "n_splits",
            &scalar_i64(i64::try_from(
                r.n_splits.expect("split_nb reports n_splits"),
            )?),
        )?;
        add_count(&mut w, "n_failed_splits", r.n_failed_splits)?;
        add_count(&mut w, "n_failed_perms", r.n_failed_perms)?;
        add_count(&mut w, "n_failed_folds", r.n_failed_folds)?;
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
            "test_method": "split_nb",
            "args": {"n_splits": 30},
            "seed": CASE_SEED
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
    })
}
