//! `pls1_confirmatory_test` fixture cases.
//!
//! Five unweighted cases share one inputs file (`inputs/pls1_confirmatory_inputs.npz`):
//! `split_nb`, `score`, `e` and two `split_nb` CI-bundle variants (`level`
//! 0.95 and 0.8). The `split_nb_gate` case `split_nb_gate_n80_d6` reads it
//! too.
//! Four weighted cases (`split_nb`, `score`, `e`, `split_nb_ci`) share a
//! separate inputs file (`inputs/pls1_confirmatory_weighted_inputs.npz`) that
//! also carries `weights`.
//! The wide (n=30, p=100) inputs file `inputs/pls1_confirmatory_wide_inputs.npz`
//! is read by two Gram cases (`split_nb` and `score` on `gram_matrix(X)`).
//! Three K=2 cases on a wider design (n=60, p=3000), `raw_perm_wide_k2`,
//! `split_exact_wide_k2` and `split_nb_wide_k2`, share
//! `inputs/pls1_confirmatory_wide_n60_d3000_inputs.npz`. The first two run
//! the primal engine at p ≫ n; `split_nb_wide_k2` runs the n-space route,
//! not the primal engine.
//! The generator refuses any case whose p sits at the permutation floor
//! `1/(n_perm + 1)`. The 19 permutation cases listed below read weak-signal
//! inputs, so their p sits mid-range. `synth_data` draws X before y, so the
//! signal strength changes only y: each of these files has the X that
//! `synth_data` draws at the same shape and seed:
//! - `inputs/pls1_confirmatory_sv_perm_inputs.npz` (n=80, p=6): `raw_perm`,
//!   `split_exact` (refit route, K = 2), `split_exact_k1` (no-refit route),
//!   `kset_split_exact`, `kset_raw_perm` and `kset_default_auto`, beside the
//!   `sv_perm` and `score` with `n_perm` cases.
//! - `inputs/pls1_confirmatory_weighted_weak_inputs.npz`: weighted `raw_perm`,
//!   `split_exact`, `split_exact_k2` and `kset_split_exact`.
//! - `inputs/pls1_confirmatory_wide_weak_inputs.npz` (n=30, p=100): the Gram
//!   `split_exact` cases at K = 1, K = 2 and over a set of k.
//! - `inputs/pls1_confirmatory_wide_raw_perm_weak_inputs.npz` (n=30, p=100):
//!   `raw_perm_wide`, the only shape that routes `raw_perm` through the
//!   n-space route at k = 1, and `test_gram_raw_perm`.
//! - `inputs/pls1_confirmatory_auto_split_exact_weak_inputs.npz` (n=60, p=20):
//!   `auto_split_exact`.
//! - `inputs/pls1_confirmatory_covariates_weak_inputs.npz`:
//!   `split_exact_covariates` and `raw_perm_covariates`.
//! - `inputs/pls1_confirmatory_covariates_weighted_weak_inputs.npz`:
//!   `split_exact_covariates_weighted`.
//!
//! The other permutation cases at `SYNTH_SNR` are off the floor on their
//! strong-signal inputs: `raw_perm_wide_k2`, `split_exact_wide_k1`,
//! `split_exact_wide_k2`, `kset_wide_split_exact` and
//! `test_gram_split_exact_covariates`.
//!
//! Six `kset_*` cases test a set of k: `split_exact`, `raw_perm`, weighted
//! `split_exact`, `split_exact` on `gram_matrix(X)`, `auto` with no `k` (the
//! default set), and `split_exact` on the wide n=60 inputs, whose combined p
//! is off the permutation floor.
//! Five `sv_perm` / `score` `n_perm` cases (`n_perm = 99`) read weak-signal
//! inputs: `sv_perm` unweighted, weighted, with covariates and on
//! `gram_matrix(X)`, and `score` with `n_perm`.

use std::path::Path;

use anyhow::{Context, Result};

use plskit::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryCI, ConfirmatoryMethod, ConfirmatoryTestInput,
    ConfirmatoryTestOpts, ConfirmatoryTestOutput, gram_matrix, pls1_confirmatory_test,
    split_nb_gate,
};

use crate::cases::pls1_find_k_optimal::write_btreemap;
use crate::cases::{
    WEIGHTED_PRESTD_N80_D6_INPUTS, Xyw, add_count, array_descriptors, case_files, ensure_moved,
    ensure_off_floor, manifest_case, perm_null_weights, scalar_f64, scalar_i64, synth_covariates,
    synth_data, weighted_prestd_n80_d6,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{NpzWriter, sha256_of_file};

/// Default synth design shared by every case except `raw_perm_wide`, which
/// needs `n < p` and sets its own `n` / `d` on the descriptor.
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
/// Signal of the weighted (n = 80, p = 6) and `auto_split_exact` (n = 60,
/// p = 20) permutation cases: puts p mid-range so the fixtures pin the null
/// columns' ranks.
const WEIGHTED_AND_AUTO_WEAK_SNR: f64 = 0.2;
/// Signal of the wide (n = 30, p = 100) Gram `split_exact` cases: puts p
/// mid-range so the fixtures pin the null columns' ranks.
const WIDE_WEAK_SNR: f64 = 0.5;
/// Signal of the wide (n = 30, p = 100) `raw_perm` cases, X form and Gram
/// form: puts p mid-range so the fixtures pin the null columns' ranks.
const WIDE_RAW_PERM_WEAK_SNR: f64 = 0.3;
/// Signal of the unweighted covariate permutation cases: puts p mid-range so
/// the fixtures pin the null columns' ranks.
const COVARIATES_WEAK_SNR: f64 = 0.4;
/// Signal of the weighted covariate permutation case: puts p mid-range so the
/// fixtures pin the null columns' ranks.
const COVARIATES_WEIGHTED_WEAK_SNR: f64 = 0.3;
/// Shared RNG seed for all cases.
const CASE_SEED: u64 = 42;
/// Function name for the manifest.
const FUNCTION: &str = "pls1_confirmatory_test";

/// Descriptor for one `pls1_confirmatory_test` case.
struct ConfirmatoryCase {
    name: &'static str,
    args: ConfirmatoryArgs,
    ci: Option<CIOpts>,
    kwargs: serde_json::Value,
    /// Inputs file stem and synth seed; set to the weighted variants for weighted cases.
    inputs_name: &'static str,
    /// Synth design shape. Every case sharing an `inputs_name` must agree on
    /// these — the inputs file is written once per name and the last writer wins.
    n: usize,
    d: usize,
    synth_seed: u64,
    /// The tested set; `None` takes the default set.
    k: Option<Vec<usize>>,
    /// Non-uniform weights array (first 40 obs get 2.0, rest 1.0); `None` for unweighted cases.
    weights: Option<ndarray::Array1<f64>>,
    /// Signal-to-noise ratio passed to `synth_data`.
    snr: f64,
}

/// Write the `ConfirmatoryCI` bundle fields into `w`.
///
/// Each `Vec<f64>` field is encoded as a 1-D `ArrayD<f64>`.
/// Each `CIScalar` is split into four separate 0-D float fields.
/// Integral counts use `i64::try_from` to avoid silent truncation.
fn write_ci_fields(w: &mut NpzWriter, ci: &ConfirmatoryCI) -> Result<()> {
    let to_arr = |v: &Vec<f64>| ndarray::Array1::from_vec(v.clone()).into_dyn();

    w.add_i64("n_boot", &scalar_i64(i64::try_from(ci.n_boot)?))?;
    w.add_i64("m", &scalar_i64(i64::try_from(ci.m)?))?;
    w.add_f64("m_rate", &scalar_f64(ci.m_rate))?;
    w.add_f64("level", &scalar_f64(ci.level))?;
    w.add_f64("beta_sign_z", &to_arr(&ci.beta_sign_z))?;
    w.add_f64("beta_sign_z_signed", &to_arr(&ci.beta_sign_z_signed))?;
    w.add_f64("leverage_ci_lower", &to_arr(&ci.leverage_ci_lower))?;
    w.add_f64("leverage_ci_upper", &to_arr(&ci.leverage_ci_upper))?;
    w.add_f64("leverage_se", &to_arr(&ci.leverage_se))?;
    w.add_f64("beta_ci_lower", &to_arr(&ci.beta_ci_lower))?;
    w.add_f64("beta_ci_upper", &to_arr(&ci.beta_ci_upper))?;
    w.add_f64("beta_se", &to_arr(&ci.beta_se))?;
    w.add_f64("holdout_corr_point", &scalar_f64(ci.holdout_corr.point))?;
    w.add_f64("holdout_corr_lower", &scalar_f64(ci.holdout_corr.lower))?;
    w.add_f64("holdout_corr_upper", &scalar_f64(ci.holdout_corr.upper))?;
    w.add_f64("holdout_corr_sd", &scalar_f64(ci.holdout_corr.sd))?;
    w.add_i64(
        "n_boot_finite",
        &scalar_i64(i64::try_from(ci.n_boot_finite)?),
    )?;
    w.add_i64(
        "n_boot_finite_holdout_corr",
        &scalar_i64(i64::try_from(ci.n_boot_finite_holdout_corr)?),
    )?;
    Ok(())
}

/// Write the tested set `k` (1-D) and the per-k maps `pvalues_by_k` /
/// `statistics_by_k` (flattened as `pls1_find_k_optimal`'s `cv_scores`).
///
/// # Errors
/// Returns an error if a field cannot be written.
pub(crate) fn write_k_set(w: &mut NpzWriter, r: &ConfirmatoryTestOutput) -> Result<()> {
    let ks =
        r.k.iter()
            .map(|&k| i64::try_from(k))
            .collect::<std::result::Result<Vec<_>, _>>()?;
    w.add_i64("k", &ndarray::Array1::from_vec(ks).into_dyn())?;
    write_btreemap(
        w,
        "pvalues_by_k__keys",
        "pvalues_by_k__values",
        &r.pvalues_by_k,
    )?;
    write_btreemap(
        w,
        "statistics_by_k__keys",
        "statistics_by_k__values",
        &r.statistics_by_k,
    )
}

/// Generic runner shared by every synth-input `pls1_confirmatory_test` case
/// (unweighted + weighted): `synth_data(c.n, c.d, 2, c.snr, c.synth_seed)`
/// with `c.weights`, through [`run_confirmatory_on`] with no extra check.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
fn run_confirmatory_case(root: &Path, c: &ConfirmatoryCase) -> Result<Case> {
    let (x, y) = synth_data(c.n, c.d, SYNTH_K_SIGNAL, c.snr, c.synth_seed);
    let data = Xyw {
        x,
        y,
        w: c.weights.clone(),
        ..Default::default()
    };
    run_confirmatory_on(root, c, &data, |_| Ok(()))
}

/// The call a case describes, on `data`. `pre_standardized` is read from the
/// case's `kwargs`, so the manifest records exactly the flag the call used.
/// With `"gram": true` in the `kwargs`, the call takes `gram_matrix(X,
/// pre_standardized)` in place of X.
fn confirmatory_call(c: &ConfirmatoryCase, data: &Xyw) -> Result<ConfirmatoryTestOutput> {
    let flag = |key: &str| {
        c.kwargs
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    let pre_standardized = flag("pre_standardized");
    let x_faer = data.x_faer();
    let y_faer = data.y_faer();
    let weights_faer = data.w_faer();
    let z_faer = data.z_faer();
    let null_y_faer = data.null_y_faer();
    let opts = ConfirmatoryTestOpts {
        args: c.args,
        pre_standardized,
        seed: Some(CASE_SEED),
        ci: c.ci,
        max_skip_rate: 0.01,
        keep: None,
    };
    if flag("gram") {
        let g = gram_matrix(x_faer.as_ref(), pre_standardized)?;
        return Ok(pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: y_faer.as_ref(),
                k: c.k.as_deref(),
                covariates: z_faer.as_ref().map(faer::Mat::as_ref),
                null_y: null_y_faer.as_ref().map(faer::Mat::as_ref),
            },
            ConfirmatoryTestOpts {
                pre_standardized: false,
                ..opts
            },
        )?);
    }
    Ok(pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x_faer.as_ref(),
            y: y_faer.as_ref(),
            k: c.k.as_deref(),
            weights: weights_faer.as_ref().map(faer::Col::as_ref),
            covariates: z_faer.as_ref().map(faer::Mat::as_ref),
            null_y: null_y_faer.as_ref().map(faer::Mat::as_ref),
        },
        opts,
    )?)
}

/// Write `data` to the case's inputs file (idempotent: same bytes every
/// call for a shared file), run the call, apply the generator guard `check`
/// and [`ensure_off_floor`] to its output, and write the case's outputs file.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or `check` or [`ensure_off_floor`] rejects the output.
fn run_confirmatory_on(
    root: &Path,
    c: &ConfirmatoryCase,
    data: &Xyw,
    check: impl FnOnce(&ConfirmatoryTestOutput) -> Result<()>,
) -> Result<Case> {
    let rel_inputs = format!("inputs/{}.npz", c.inputs_name);
    let rel_outputs = format!("outputs/{FUNCTION}/{}.npz", c.name);
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    data.write(&abs_inputs)?;

    let r = confirmatory_call(c, data)?;
    check(&r)?;
    ensure_off_floor(c.name, r.pvalue, r.n_perm)?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        if let Some(s) = r.statistic {
            w.add_f64("statistic", &scalar_f64(s))?;
        }
        w.add_string("test_method", &r.test_method)?;
        write_k_set(&mut w, &r)?;
        if let Some(np) = r.n_perm {
            w.add_i64("n_perm", &scalar_i64(i64::try_from(np)?))?;
        }
        if let Some(ns) = r.n_splits {
            w.add_i64("n_splits", &scalar_i64(i64::try_from(ns)?))?;
        }
        add_count(&mut w, "n_failed_splits", r.n_failed_splits)?;
        add_count(&mut w, "n_failed_perms", r.n_failed_perms)?;
        add_count(&mut w, "n_failed_folds", r.n_failed_folds)?;
        if let Some(sr) = r.stable_rank {
            w.add_f64("stable_rank", &scalar_f64(sr))?;
        }
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        if let Some(nc) = r.n_covariates {
            w.add_i64("n_covariates", &scalar_i64(i64::try_from(nc)?))?;
        }
        if let Some(ci) = &r.ci {
            write_ci_fields(&mut w, ci)?;
        }
        w.finish()?;
    }

    let mut kwargs = c.kwargs.clone();
    array_descriptors(&mut kwargs, data);
    Ok(Case {
        name: c.name.to_string(),
        function: FUNCTION.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs,
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
    })
}

/// Case: `pls1_confirmatory_test` with `test_method=raw_perm`, `n_perm=200`, `n_folds=5`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn raw_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_raw_perm",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_sv_perm_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SV_PERM_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=split_nb`, `n_splits=30`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_nb(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_nb",
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 30,
                force: false,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "split_nb",
                "args": {"n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=split_exact`, `n_perm=200`, `n_splits=30`,
/// `seed=42`. `k=2` sends this through `split_exact`'s honest-refit route.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_exact",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_sv_perm_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SV_PERM_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=split_exact`, `n_perm=200`, `n_splits=30`,
/// `seed=42`, `k=1`. Unweighted dense K = 1: exercises `split_exact`'s no-refit route
/// (the [`split_exact`] case above covers the refit route via K = 2;
/// [`weighted_split_exact`] covers the no-refit route under weights).
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact_k1(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_exact_k1",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_sv_perm_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: Some(vec![1]),
            weights: None,
            snr: SV_PERM_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=score`, `seed=42`.
///
/// Closed-form score test — no permutation or split count.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn score(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_score",
            args: ConfirmatoryArgs::Score { n_perm: None },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "score",
                "args": {},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=e`, `seed=42`.
///
/// Universal-inference split-LR e-value — no permutation or split count.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn e(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_e",
            args: ConfirmatoryArgs::E,
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "e",
                "args": {},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=split_nb` + CI bundle (`n_boot=300`), `seed=42`.
///
/// Exercises the `ci = Some(CIOpts { ... })` path.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_nb_ci(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &split_nb_ci_case("pls1_confirmatory_split_nb_ci", 0.95),
    )
}

/// Case: [`split_nb_ci`]'s call at `level = 0.8`. The normal quantile of the
/// leverage and `holdout_corr` CIs is then Φ⁻¹(0.9), from AS241's central
/// branch (`|p − 0.5| ≤ 0.425`); every other CI fixture runs at the default
/// 0.95 (Φ⁻¹(0.975), the tail branch). It is also the only fixture that
/// fails a consumer dropping `level`: the generator refuses it unless
/// `holdout_corr_lower` moves against the 0.95 call.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or the output does not depend on `level`.
pub fn split_nb_ci_level80(root: &Path) -> Result<Case> {
    let c = split_nb_ci_case("pls1_confirmatory_split_nb_ci_level80", 0.8);
    let (x, y) = synth_data(c.n, c.d, SYNTH_K_SIGNAL, c.snr, c.synth_seed);
    let data = Xyw {
        x,
        y,
        w: None,
        ..Default::default()
    };
    let default_level = confirmatory_call(&split_nb_ci_case(c.name, 0.95), &data)?;
    run_confirmatory_on(root, &c, &data, |r| {
        let lower = |o: &ConfirmatoryTestOutput| {
            o.ci.as_ref()
                .map(|ci| vec![ci.holdout_corr.lower])
                .unwrap_or_default()
        };
        crate::cases::ensure_moved(
            c.name,
            "holdout_corr_lower vs level 0.95",
            &lower(r),
            &lower(&default_level),
        )
    })
}

/// The call of [`split_nb_ci`] (`k=2`, `n_splits=30`, CI with `n_boot=300`,
/// serial, unweighted inputs) at `level`.
fn split_nb_ci_case(name: &'static str, level: f64) -> ConfirmatoryCase {
    ConfirmatoryCase {
        name,
        args: ConfirmatoryArgs::SplitNb {
            n_splits: 30,
            force: false,
        },
        ci: Some(CIOpts {
            n_boot: 300,
            m_rate: 0.7,
            level,
            max_failure_rate: 0.0,
        }),
        kwargs: serde_json::json!({
            "k": 2,
            "test_method": "split_nb",
            "args": {"n_splits": 30},
            "ci": true,
            "n_boot": 300,
            "m_rate": 0.7,
            "level": level,
            "seed": 42,
            "max_failure_rate": 0.0
        }),
        inputs_name: "pls1_confirmatory_inputs",
        n: SYNTH_N,
        d: SYNTH_D,
        synth_seed: 42,
        k: Some(vec![2]),
        weights: None,
        snr: SYNTH_SNR,
    }
}

/// Case: `pls1_confirmatory_test` with `test_method=auto` on n=60, p=20, `k=2`,
/// `n_perm=200`, `n_splits=30`, `seed=42`. n is below the refit floor of 250 (k = 2), so the
/// rule resolves to `split_exact`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn auto_split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_auto_split_exact",
            args: ConfirmatoryArgs::Auto {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "auto",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_auto_split_exact_weak_inputs",
            n: 60,
            d: 20,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: WEIGHTED_AND_AUTO_WEAK_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=auto` on n=300, p=10, `k=2`,
/// `n_perm=200`, `n_splits=30`, `seed=42`. n clears the refit floor of 250 (k = 2), p is in
/// range and the stable rank is at least 3, so the rule resolves to `split_nb`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn auto_split_nb(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_auto_split_nb",
            args: ConfirmatoryArgs::Auto {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "auto",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_auto_split_nb_inputs",
            n: 300,
            d: 10,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=raw_perm` on a wide design
/// (n=30, p=100), `k=1`, `n_perm=100`, `n_folds=5`.
///
/// The n-space route's `k = 1` reference. `run_raw_perm` admits that route
/// when `keep` is unset, the design is unweighted and the largest training
/// fold is under `0.75·p` (`4·n_tr < 3·p`, here 24 against 75), and takes
/// it when the selector's cost rule does (pinned in
/// `plskit-rs/src/fixture_route_pins.rs`). Lowering `n_perm`, raising
/// `n_folds` or narrowing X can send the fixture back onto the primal
/// route, which would leave the X form's n-space route at `k = 1` without a
/// corpus case.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn raw_perm_wide(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_raw_perm_wide",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 100,
                n_folds: 5,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "raw_perm",
                "args": {"n_perm": 100, "n_folds": 5},
                "seed": 42
            }),
            inputs_name: WIDE_RAW_PERM_WEAK_INPUTS.0,
            n: 30,
            d: 100,
            synth_seed: 42,
            k: Some(vec![1]),
            weights: None,
            snr: WIDE_RAW_PERM_WEAK_INPUTS.1,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=raw_perm` on a wide design
/// (n=60, p=3000), `k=2`, `n_perm=200`, `n_folds=5`.
///
/// `raw_perm` at p ≫ n on the n-space route (pinned in
/// `plskit-rs/src/fixture_route_pins.rs`): the reference of that route at
/// `k ≥ 2`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn raw_perm_wide_k2(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_raw_perm_wide_k2",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_wide_n60_d3000_inputs",
            n: 60,
            d: 3000,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=split_exact` on the wide
/// design of [`raw_perm_wide_k2`], `k=2`, `n_perm=200`, `n_splits=20`.
///
/// `k = 2` takes the refit route (the no-refit route is K = 1 only), dense
/// and at p ≫ n: the primal reference of the refit route's split loop.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact_wide_k2(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_exact_wide_k2",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 20,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 20},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_wide_n60_d3000_inputs",
            n: 60,
            d: 3000,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=split_nb` on the wide
/// design of [`raw_perm_wide_k2`], `k=2`, `n_splits=30`, without `force`.
///
/// At k = 2 and p ≫ n the split loop takes the n-space route, and the gate
/// does not fire, so the call runs `split_nb`. The generator refuses the
/// case unless the stored `test_method` is `split_nb`, so a gate change
/// cannot turn the fixture into a test of another method.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or the call does not run `split_nb`.
pub fn split_nb_wide_k2(root: &Path) -> Result<Case> {
    let c = ConfirmatoryCase {
        name: "pls1_confirmatory_split_nb_wide_k2",
        args: ConfirmatoryArgs::SplitNb {
            n_splits: 30,
            force: false,
        },
        ci: None,
        kwargs: serde_json::json!({
            "k": 2,
            "test_method": "split_nb",
            "args": {"n_splits": 30},
            "seed": 42
        }),
        inputs_name: "pls1_confirmatory_wide_n60_d3000_inputs",
        n: 60,
        d: 3000,
        synth_seed: 42,
        k: Some(vec![2]),
        weights: None,
        snr: SYNTH_SNR,
    };
    let (x, y) = synth_data(c.n, c.d, SYNTH_K_SIGNAL, c.snr, c.synth_seed);
    let data = Xyw {
        x,
        y,
        ..Default::default()
    };
    run_confirmatory_on(root, &c, &data, |r| {
        anyhow::ensure!(
            r.test_method == "split_nb",
            "{}: test_method = {}, want split_nb",
            c.name,
            r.test_method
        );
        Ok(())
    })
}

/// Non-uniform weights for weighted confirmatory cases: first 40 of 80 obs get 2.0, rest 1.0.
fn weighted_confirmatory_weights() -> ndarray::Array1<f64> {
    ndarray::Array1::from_shape_fn(SYNTH_N, |i| if i < 40 { 2.0_f64 } else { 1.0_f64 })
}

/// Case: weighted `pls1_confirmatory_test` with `test_method=raw_perm`, `n_perm=200`, `n_folds=5`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_raw_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_raw_perm",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_weak_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1]),
            weights: Some(weighted_confirmatory_weights()),
            snr: WEIGHTED_AND_AUTO_WEAK_SNR,
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `test_method=split_nb`, `n_splits=50`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_split_nb(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_split_nb",
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 50,
                force: false,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "split_nb",
                "args": {"n_splits": 50},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1]),
            weights: Some(weighted_confirmatory_weights()),
            snr: SYNTH_SNR,
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `test_method=split_exact`, `n_perm=200`,
/// `n_splits=50`. `k=1` dense under weights: exercises `split_exact`'s weighted
/// no-refit route.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_split_exact",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 50,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 50},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_weak_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1]),
            weights: Some(weighted_confirmatory_weights()),
            snr: WEIGHTED_AND_AUTO_WEAK_SNR,
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `test_method=score`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_score(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_score",
            args: ConfirmatoryArgs::Score { n_perm: None },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "score",
                "args": {},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1]),
            weights: Some(weighted_confirmatory_weights()),
            snr: SYNTH_SNR,
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `test_method=e`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_e(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_e",
            args: ConfirmatoryArgs::E,
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "e",
                "args": {},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1]),
            weights: Some(weighted_confirmatory_weights()),
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=raw_perm` on a tall design
/// (n=2000, p=50), `k=2`, `n_perm=1000`, `n_folds=5`, `seed=42`.
///
/// Generated by the explicit-deflation (primal) kernel, so it is an
/// independent reference for the p-space Gram route. The largest training fold has `n_tr = n − n/n_folds = 1600` rows
/// and `B = n_perm + 1 = 1001` columns, so the X backend's work per block is
/// `B·(2k + 1)·n_tr·p = 4.004e8`, 4× the Gram backend's starting work floor,
/// and the n-space Gram route cannot claim it (`n_tr > p`).
/// `plskit-rs/src/fixture_route_pins.rs` pins its route.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn raw_perm_tall_k2(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_raw_perm_tall_k2",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 1000,
                n_folds: 5,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "raw_perm",
                "args": {"n_perm": 1000, "n_folds": 5},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_tall_inputs",
            n: 2000,
            d: 50,
            synth_seed: 42,
            k: Some(vec![2]),
            weights: None,
            snr: RAW_PERM_TALL_SNR,
        },
    )
}

/// Weak signal for `raw_perm_tall_k2`: the observed CV R² must sit inside
/// the null distribution, so that the p-value depends on how every one of
/// the 1000 null replicates ranks against it. At the shared `SYNTH_SNR`
/// the observed value beats every null and the p-value is `1/1001`
/// whatever the nulls are, which pins nothing about them.
const RAW_PERM_TALL_SNR: f64 = 0.05;

/// Case: `test_method=score`, `k=1`, `pre_standardized=true` on the wide
/// (n=30, p=100) design of [`raw_perm_wide`], stored in its own inputs file
/// with every column of X centered and scaled to unit Euclidean norm and y
/// centered and scaled to unit (population) variance.
///
/// Pins score's `d > n` Gram branch (`X X'`, through the p-value) and its
/// skip-standardize branch in one fixture. The statistic `‖X'y‖²` is then `n`
/// times the sum of the squared column correlations (about 140), where
/// unit-variance columns would give `n` times that and the raw inputs about
/// `1.3e5`, whose last bit alone exceeds the `1e-12` scalar tolerance. The
/// p-value depends on the scale of y only, so unit-variance y keeps it
/// interior. A consumer that drops `pre_standardized` restandardizes X and
/// moves the statistic by a factor of `n = 30`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn score_wide_pre_standardized(root: &Path) -> Result<Case> {
    let c = ConfirmatoryCase {
        name: "pls1_confirmatory_score_wide_pre_standardized",
        args: ConfirmatoryArgs::Score { n_perm: None },
        ci: None,
        kwargs: serde_json::json!({
            "k": 1,
            "test_method": "score",
            "args": {},
            "pre_standardized": true,
            "seed": 42
        }),
        inputs_name: "pls1_confirmatory_score_wide_pre_standardized",
        n: 30,
        d: 100,
        synth_seed: CASE_SEED,
        k: Some(vec![1]),
        weights: None,
        snr: SYNTH_SNR,
    };
    run_confirmatory_on(root, &c, &score_wide_unit_norm(None), |_| Ok(()))
}

/// The inputs of [`score_wide_pre_standardized`]: `synth_data(30, 100, 2,
/// 4.0, 42)` with every column of X centered and scaled to unit Euclidean
/// norm and y centered and scaled to unit (population) variance, plus
/// `weights` when given.
fn score_wide_unit_norm(weights: Option<ndarray::Array1<f64>>) -> Xyw {
    let (mut x, mut y) = synth_data(30, 100, SYNTH_K_SIGNAL, SYNTH_SNR, CASE_SEED);
    // Center `c` and scale it to Euclidean norm `norm`.
    let rescale = |mut c: ndarray::ArrayViewMut1<'_, f64>, norm: f64| {
        let mean = c.mean().unwrap_or(0.0);
        c -= mean;
        let current = c.dot(&c).sqrt();
        c *= norm / current;
    };
    x.columns_mut().into_iter().for_each(|c| rescale(c, 1.0));
    let unit_variance_norm = 30.0_f64.sqrt();
    rescale(y.view_mut(), unit_variance_norm);
    Xyw {
        x,
        y,
        w: weights,
        ..Default::default()
    }
}

/// Case: [`score_wide_pre_standardized`]'s call under weights
/// [`perm_null_weights`]`(30)`, on its inputs plus `weights` (own inputs
/// file). Under `pre_standardized = true` the weights still row-scale X and
/// y by √w, and at `d > n` the score reads the trace and squared Frobenius
/// norm of the √w-scaled `X X'`; the only fixture of either. The generator
/// refuses it unless the statistic stays below `1e3` (the `1e-12` scalar
/// tolerance must stay above its last bit), the p-value is interior (so it
/// pins those two sums), and the output moves both without the weights and
/// without the flag.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or a guard fails.
pub fn weighted_score_wide_pre_standardized(root: &Path) -> Result<Case> {
    let case = |name: &'static str, pre_standardized: bool, weighted: bool| {
        let mut kwargs = serde_json::json!({
            "k": 1,
            "test_method": "score",
            "args": {},
            "pre_standardized": pre_standardized,
            "seed": 42
        });
        if weighted {
            kwargs["weights"] = serde_json::json!("nonuniform");
        }
        ConfirmatoryCase {
            name,
            args: ConfirmatoryArgs::Score { n_perm: None },
            ci: None,
            kwargs,
            inputs_name: "pls1_confirmatory_weighted_score_wide_pre_standardized",
            n: 30,
            d: 100,
            synth_seed: CASE_SEED,
            k: Some(vec![1]),
            weights: weighted.then(|| perm_null_weights(30)),
            snr: SYNTH_SNR,
        }
    };
    let name = "pls1_confirmatory_weighted_score_wide_pre_standardized";
    let c = case(name, true, true);
    let data = score_wide_unit_norm(c.weights.clone());
    let unweighted = confirmatory_call(&case(name, true, false), &score_wide_unit_norm(None))?;
    let without_flag = confirmatory_call(&case(name, false, true), &data)?;
    run_confirmatory_on(root, &c, &data, |r| {
        let statistic = r.statistic.context("one k")?;
        anyhow::ensure!(
            statistic < 1e3 && r.pvalue > 1e-3 && r.pvalue < 0.999,
            "{name}: statistic {} (want < 1e3), pvalue {} (want interior)",
            statistic,
            r.pvalue
        );
        for (what, other) in [
            ("unweighted", &unweighted),
            ("pre_standardized = false", &without_flag),
        ] {
            crate::cases::ensure_moved(
                name,
                &format!("[statistic, pvalue] vs {what}"),
                &[statistic, r.pvalue],
                &[other.statistic.context("one k")?, other.pvalue],
            )?;
        }
        Ok(())
    })
}

/// Case: weighted `test_method=split_exact`, `k=2`, `n_perm=200`, `n_splits=50`.
/// `k=2` takes the honest-refit route: the only fixture of that route under
/// weights.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_split_exact_k2(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_split_exact_k2",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 50,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 2,
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 50},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_weak_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![2]),
            weights: Some(weighted_confirmatory_weights()),
            snr: WEIGHTED_AND_AUTO_WEAK_SNR,
        },
    )
}

/// Case: `test_method=split_exact`, `k=1`, `n_perm=200`, `n_splits=20` on the wide
/// (n=60, p=3000) inputs of [`raw_perm_wide_k2`]. Dense `K = 1` takes the
/// no-refit route, and at this shape its association product runs in the
/// order `30·30·(3000 + 201) < 60·3000·201` selects; [`split_exact_k1`] pins
/// the other order.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact_wide_k1(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_exact_wide_k1",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 20,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": 1,
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 20},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_wide_n60_d3000_inputs",
            n: 60,
            d: 3000,
            synth_seed: 42,
            k: Some(vec![1]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// The call of [`split_nb_ci`] (`k=2`, `n_splits=30`, CI with `n_boot=300`,
/// serial) under weights, on inputs `inputs_name`, with `pre_standardized`
/// recorded in the kwargs when set.
fn weighted_split_nb_ci_case(
    name: &'static str,
    inputs_name: &'static str,
    synth_seed: u64,
    weights: ndarray::Array1<f64>,
    pre_standardized: bool,
) -> ConfirmatoryCase {
    let mut kwargs = serde_json::json!({
        "k": 2,
        "test_method": "split_nb",
        "args": {"n_splits": 30},
        "ci": true,
        "n_boot": 300,
        "m_rate": 0.7,
        "level": 0.95,
        "seed": 42,
        "max_failure_rate": 0.0,
        "weights": "nonuniform"
    });
    if pre_standardized {
        kwargs["pre_standardized"] = serde_json::json!(true);
    }
    ConfirmatoryCase {
        name,
        args: ConfirmatoryArgs::SplitNb {
            n_splits: 30,
            force: false,
        },
        ci: Some(CIOpts {
            n_boot: 300,
            m_rate: 0.7,
            level: 0.95,
            max_failure_rate: 0.0,
        }),
        kwargs,
        inputs_name,
        n: SYNTH_N,
        d: SYNTH_D,
        synth_seed,
        k: Some(vec![2]),
        weights: Some(weights),
        snr: SYNTH_SNR,
    }
}

/// Case: [`split_nb_ci`]'s call under weights, on the weighted inputs
/// `inputs/pls1_confirmatory_weighted_inputs.npz`. The only fixture of the weighted subsampling CI
/// bundle. The generator refuses it unless all 300 replicates are finite, so
/// the bundle is not a fail-soft remnant.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or a CI replicate is non-finite.
pub fn weighted_split_nb_ci(root: &Path) -> Result<Case> {
    let c = weighted_split_nb_ci_case(
        "pls1_confirmatory_weighted_split_nb_ci",
        "pls1_confirmatory_weighted_inputs",
        77,
        weighted_confirmatory_weights(),
        false,
    );
    let (x, y) = synth_data(c.n, c.d, SYNTH_K_SIGNAL, c.snr, c.synth_seed);
    let data = Xyw {
        x,
        y,
        w: c.weights.clone(),
        ..Default::default()
    };
    run_confirmatory_on(root, &c, &data, |r| {
        let finite = r.ci.as_ref().map(|ci| ci.n_boot_finite);
        anyhow::ensure!(
            finite == Some(300),
            "{}: n_boot_finite = {finite:?}, want Some(300)",
            c.name
        );
        Ok(())
    })
}

/// Case: [`weighted_split_nb_ci`]'s call with `pre_standardized = true` on
/// X and y standardized by their unweighted moments
/// (`cases::weighted_prestd_n80_d6`). The generator refuses it unless the CI
/// bundle moves when the flag is dropped, so the fixture fails a consumer
/// that ignores `pre_standardized`.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or the output does not depend on `pre_standardized`.
pub fn weighted_prestd_split_nb_ci(root: &Path) -> Result<Case> {
    let data = weighted_prestd_n80_d6();
    let c = weighted_split_nb_ci_case(
        "pls1_confirmatory_weighted_prestd_split_nb_ci",
        WEIGHTED_PRESTD_N80_D6_INPUTS,
        42,
        perm_null_weights(SYNTH_N),
        true,
    );
    let without_flag =
        weighted_split_nb_ci_case(c.name, c.inputs_name, 42, perm_null_weights(SYNTH_N), false);
    let without = confirmatory_call(&without_flag, &data)?;
    run_confirmatory_on(root, &c, &data, |r| {
        let lower = |o: &ConfirmatoryTestOutput| {
            o.ci.as_ref()
                .map(|ci| ci.beta_ci_lower.clone())
                .unwrap_or_default()
        };
        crate::cases::ensure_moved(c.name, "beta_ci_lower", &lower(r), &lower(&without))
    })
}

/// One `pls1_confirmatory_test` case on `gram_matrix(X)` in place of X
/// (kwarg `"gram": true`), on wide inputs (n = 30, p = 100): `(inputs_name, snr)` is
/// [`WIDE_INPUTS`] or [`WIDE_WEAK_INPUTS`].
fn gram_case(
    root: &Path,
    (inputs_name, snr): (&'static str, f64),
    name: &'static str,
    args: ConfirmatoryArgs,
    k: &[usize],
    kwargs: serde_json::Value,
) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name,
            args,
            ci: None,
            kwargs,
            inputs_name,
            n: 30,
            d: 100,
            synth_seed: 42,
            k: Some(k.to_vec()),
            weights: None,
            snr,
        },
    )
}

/// The wide inputs of the Gram `split_nb` and `score` cases.
const WIDE_INPUTS: (&str, f64) = ("pls1_confirmatory_wide_inputs", SYNTH_SNR);

/// Wide inputs with the X of [`WIDE_INPUTS`] and a weaker y, for the
/// `raw_perm` cases.
const WIDE_RAW_PERM_WEAK_INPUTS: (&str, f64) = (
    "pls1_confirmatory_wide_raw_perm_weak_inputs",
    WIDE_RAW_PERM_WEAK_SNR,
);

/// Wide inputs with the X of [`WIDE_INPUTS`] and a weaker y, for the Gram
/// `split_exact` cases.
const WIDE_WEAK_INPUTS: (&str, f64) = ("pls1_confirmatory_wide_weak_inputs", WIDE_WEAK_SNR);

/// Case: `split_exact` at `k = 1` (`n_perm = 100`, `n_splits = 20`) on
/// `gram_matrix(X)`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_split_exact_k1(root: &Path) -> Result<Case> {
    gram_case(
        root,
        WIDE_WEAK_INPUTS,
        "pls1_confirmatory_test_gram_split_exact_k1",
        ConfirmatoryArgs::SplitExact {
            n_perm: 100,
            n_splits: 20,
        },
        &[1],
        serde_json::json!({
            "k": 1, "test_method": "split_exact", "args": {"n_perm": 100, "n_splits": 20},
            "seed": 42, "gram": true
        }),
    )
}

/// Case: `split_exact` at `k = 2` (`n_perm = 100`, `n_splits = 20`) on
/// `gram_matrix(X)`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_split_exact_k2(root: &Path) -> Result<Case> {
    gram_case(
        root,
        WIDE_WEAK_INPUTS,
        "pls1_confirmatory_test_gram_split_exact_k2",
        ConfirmatoryArgs::SplitExact {
            n_perm: 100,
            n_splits: 20,
        },
        &[2],
        serde_json::json!({
            "k": 2, "test_method": "split_exact", "args": {"n_perm": 100, "n_splits": 20},
            "seed": 42, "gram": true
        }),
    )
}

/// Case: `split_nb` (`n_splits = 20`) on `gram_matrix(X)`; the gate reads
/// the stable rank of G.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_split_nb(root: &Path) -> Result<Case> {
    gram_case(
        root,
        WIDE_INPUTS,
        "pls1_confirmatory_test_gram_split_nb",
        ConfirmatoryArgs::SplitNb {
            n_splits: 20,
            force: false,
        },
        &[1],
        serde_json::json!({
            "k": 1, "test_method": "split_nb", "args": {"n_splits": 20},
            "seed": 42, "gram": true
        }),
    )
}

/// Case: `raw_perm` (`n_perm = 100`, `n_folds = 5`) on `gram_matrix(X)`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_raw_perm(root: &Path) -> Result<Case> {
    gram_case(
        root,
        WIDE_RAW_PERM_WEAK_INPUTS,
        "pls1_confirmatory_test_gram_raw_perm",
        ConfirmatoryArgs::RawPerm {
            n_perm: 100,
            n_folds: 5,
        },
        &[1],
        serde_json::json!({
            "k": 1, "test_method": "raw_perm", "args": {"n_perm": 100, "n_folds": 5},
            "seed": 42, "gram": true
        }),
    )
}

/// Case: `score` on `gram_matrix(X)`: `y'Gy` and the trace and squared
/// Frobenius norm of G.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_score(root: &Path) -> Result<Case> {
    gram_case(
        root,
        WIDE_INPUTS,
        "pls1_confirmatory_test_gram_score",
        ConfirmatoryArgs::Score { n_perm: None },
        &[1],
        serde_json::json!({"k": 1, "test_method": "score", "seed": 42, "gram": true}),
    )
}

/// Run `c` on `data`, which carries covariates or `null_y`, and refuse the
/// fixture unless its statistic moves against the same call without them.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or the statistic does not move.
fn covariate_case(root: &Path, c: &ConfirmatoryCase, data: &Xyw) -> Result<Case> {
    let plain_data = Xyw {
        x: data.x.clone(),
        y: data.y.clone(),
        w: data.w.clone(),
        ..Default::default()
    };
    let plain = confirmatory_call(c, &plain_data)?;
    run_confirmatory_on(root, c, data, |r| {
        ensure_moved(
            c.name,
            "statistic",
            &[r.statistic.context("one k")?],
            &[plain.statistic.context("one k")?],
        )
    })
}

/// `synth_data(n, d, 2, snr, seed)` with [`synth_covariates`]`(x, 7)`.
#[allow(clippy::many_single_char_names)]
fn covariate_data(
    (n, d): (usize, usize),
    seed: u64,
    weights: Option<ndarray::Array1<f64>>,
    snr: f64,
) -> Xyw {
    let (x, y) = synth_data(n, d, SYNTH_K_SIGNAL, snr, seed);
    let z = Some(synth_covariates(&x, 7));
    Xyw {
        x,
        y,
        w: weights,
        z,
        ..Default::default()
    }
}

/// A `ConfirmatoryCase` on the covariate inputs `inputs_name`, whose `n`,
/// `d`, `synth_seed`, `weights` and `snr` only label the data
/// ([`covariate_case`] takes the data itself). `snr` is the strength the
/// caller builds its data at.
fn covariate_descriptor(
    name: &'static str,
    inputs_name: &'static str,
    args: ConfirmatoryArgs,
    k: usize,
    kwargs: serde_json::Value,
    snr: f64,
) -> ConfirmatoryCase {
    ConfirmatoryCase {
        name,
        args,
        ci: None,
        kwargs,
        inputs_name,
        n: SYNTH_N,
        d: SYNTH_D,
        synth_seed: CASE_SEED,
        k: Some(vec![k]),
        weights: None,
        snr,
    }
}

/// Case: `split_exact` (`n_perm = 200`, `n_splits = 30`) at `k = 2` with three
/// covariates, the third an alias of the first (Freedman-Lane null columns).
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact_covariates(root: &Path) -> Result<Case> {
    covariate_case(
        root,
        &covariate_descriptor(
            "pls1_confirmatory_split_exact_covariates",
            "pls1_confirmatory_covariates_weak_inputs",
            ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            2,
            serde_json::json!({
                "k": 2, "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30}, "seed": 42
            }),
            COVARIATES_WEAK_SNR,
        ),
        &covariate_data((SYNTH_N, SYNTH_D), CASE_SEED, None, COVARIATES_WEAK_SNR),
    )
}

/// Case: `raw_perm` (`n_perm = 200`, `n_folds = 5`) at `k = 1` with the
/// covariates of [`split_exact_covariates`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn raw_perm_covariates(root: &Path) -> Result<Case> {
    covariate_case(
        root,
        &covariate_descriptor(
            "pls1_confirmatory_raw_perm_covariates",
            "pls1_confirmatory_covariates_weak_inputs",
            ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            1,
            serde_json::json!({
                "k": 1, "test_method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5}, "seed": 42
            }),
            COVARIATES_WEAK_SNR,
        ),
        &covariate_data((SYNTH_N, SYNTH_D), CASE_SEED, None, COVARIATES_WEAK_SNR),
    )
}

/// Case: weighted `split_exact` (`n_perm = 200`, `n_splits = 50`) at `k = 1`
/// with covariates, on the X and weights of [`weighted_split_exact`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact_covariates_weighted(root: &Path) -> Result<Case> {
    covariate_case(
        root,
        &covariate_descriptor(
            "pls1_confirmatory_split_exact_covariates_weighted",
            "pls1_confirmatory_covariates_weighted_weak_inputs",
            ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 50,
            },
            1,
            serde_json::json!({
                "k": 1, "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 50}, "seed": 42,
                "weights": "nonuniform"
            }),
            COVARIATES_WEIGHTED_WEAK_SNR,
        ),
        &covariate_data(
            (SYNTH_N, SYNTH_D),
            77,
            Some(weighted_confirmatory_weights()),
            COVARIATES_WEIGHTED_WEAK_SNR,
        ),
    )
}

/// Case: `split_exact` (`n_perm = 100`, `n_splits = 20`) at `k = 1` on
/// `gram_matrix(X)` of the wide inputs (n = 30, p = 100) with three
/// covariates.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_split_exact_covariates(root: &Path) -> Result<Case> {
    covariate_case(
        root,
        &covariate_descriptor(
            "pls1_confirmatory_test_gram_split_exact_covariates",
            "pls1_confirmatory_wide_covariates_inputs",
            ConfirmatoryArgs::SplitExact {
                n_perm: 100,
                n_splits: 20,
            },
            1,
            serde_json::json!({
                "k": 1, "test_method": "split_exact",
                "args": {"n_perm": 100, "n_splits": 20}, "seed": 42, "gram": true
            }),
            SYNTH_SNR,
        ),
        &covariate_data((30, 100), CASE_SEED, None, SYNTH_SNR),
    )
}

/// Signal-to-noise ratio of [`split_exact_null_y`]: weak, so the p-value sits
/// mid-range and pins the rank count against the null columns.
const NULL_Y_SNR: f64 = 0.1;

/// Case: `split_exact` (`n_splits = 30`) at `k = 1` against caller-built null
/// outcomes: B = 100 columns of standard-normal noise scaled to the sd of y.
/// The manifest names no `n_perm`; the output's `n_perm` is B.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
#[allow(clippy::cast_precision_loss)]
pub fn split_exact_null_y(root: &Path) -> Result<Case> {
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;
    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, NULL_Y_SNR, CASE_SEED);
    let mean = y.sum() / y.len() as f64;
    let sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() - 1) as f64).sqrt();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let null_y = ndarray::Array2::from_shape_fn((SYNTH_N, 100), |_| {
        sd * rng.sample::<f64, _>(StandardNormal)
    });
    let data = Xyw {
        x,
        y,
        null_y: Some(null_y),
        ..Default::default()
    };
    // The core ignores `n_perm` under `null_y`; the call takes the default.
    let args = match ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitExact) {
        ConfirmatoryArgs::SplitExact { n_perm, .. } => ConfirmatoryArgs::SplitExact {
            n_perm,
            n_splits: 30,
        },
        other => other,
    };
    let c = covariate_descriptor(
        "pls1_confirmatory_split_exact_null_y",
        "pls1_confirmatory_null_y_inputs",
        args,
        1,
        serde_json::json!({
            "k": 1, "test_method": "split_exact", "args": {"n_splits": 30}, "seed": 42
        }),
        NULL_Y_SNR,
    );
    run_confirmatory_on(root, &c, &data, |r| {
        anyhow::ensure!(
            r.n_perm == Some(100),
            "null_y: n_perm is {:?}, not B",
            r.n_perm
        );
        Ok(())
    })
}

/// A `split_exact` / `raw_perm` case over the set `k` on the weak-signal
/// inputs of [`split_exact`] (n = 80, p = 6).
fn kset_case(
    name: &'static str,
    args: ConfirmatoryArgs,
    k: Option<Vec<usize>>,
    kwargs: serde_json::Value,
) -> ConfirmatoryCase {
    ConfirmatoryCase {
        name,
        args,
        ci: None,
        kwargs,
        inputs_name: "pls1_confirmatory_sv_perm_inputs",
        n: SYNTH_N,
        d: SYNTH_D,
        synth_seed: 42,
        k,
        weights: None,
        snr: SV_PERM_SNR,
    }
}

/// Case: `split_exact` (`n_perm = 200`, `n_splits = 30`) over `k = [1, 2, 3]`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn kset_split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &kset_case(
            "pls1_confirmatory_kset_split_exact",
            ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            Some(vec![1, 2, 3]),
            serde_json::json!({
                "k": [1, 2, 3], "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30}, "seed": 42
            }),
        ),
    )
}

/// Case: `raw_perm` (`n_perm = 200`, `n_folds = 5`) over `k = [1, 2, 3]`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn kset_raw_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &kset_case(
            "pls1_confirmatory_kset_raw_perm",
            ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            Some(vec![1, 2, 3]),
            serde_json::json!({
                "k": [1, 2, 3], "test_method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5}, "seed": 42
            }),
        ),
    )
}

/// Case: `auto` (`n_perm = 200`, `n_splits = 30`) with no `k`: the default
/// set, `1..=6` at n = 80, p = 6. The generator refuses it unless the set has
/// more than one k and `auto` resolved to `split_exact`.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or the guard fails.
pub fn kset_default_auto(root: &Path) -> Result<Case> {
    let c = kset_case(
        "pls1_confirmatory_kset_default_auto",
        ConfirmatoryArgs::Auto {
            n_perm: 200,
            n_splits: 30,
        },
        None,
        serde_json::json!({
            "test_method": "auto", "args": {"n_perm": 200, "n_splits": 30}, "seed": 42
        }),
    );
    let (x, y) = synth_data(c.n, c.d, SYNTH_K_SIGNAL, c.snr, c.synth_seed);
    let data = Xyw {
        x,
        y,
        ..Default::default()
    };
    run_confirmatory_on(root, &c, &data, |r| {
        anyhow::ensure!(
            r.k.len() > 1 && r.test_method == "split_exact",
            "{}: k = {:?}, test_method = {} (want a set, split_exact)",
            c.name,
            r.k,
            r.test_method
        );
        Ok(())
    })
}

/// Case: weighted `split_exact` (`n_perm = 200`, `n_splits = 50`) over
/// `k = [1, 2]` on the inputs of [`weighted_split_exact`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn kset_weighted_split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_kset_weighted_split_exact",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 50,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": [1, 2],
                "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 50},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_weak_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1, 2]),
            weights: Some(weighted_confirmatory_weights()),
            snr: WEIGHTED_AND_AUTO_WEAK_SNR,
        },
    )
}

/// Case: `split_exact` (`n_perm = 100`, `n_splits = 20`) over `k = [1, 2, 3]`
/// on `gram_matrix(X)`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn gram_kset_split_exact(root: &Path) -> Result<Case> {
    gram_case(
        root,
        WIDE_WEAK_INPUTS,
        "pls1_confirmatory_test_gram_kset_split_exact",
        ConfirmatoryArgs::SplitExact {
            n_perm: 100,
            n_splits: 20,
        },
        &[1, 2, 3],
        serde_json::json!({
            "k": [1, 2, 3], "test_method": "split_exact", "args": {"n_perm": 100, "n_splits": 20},
            "seed": 42, "gram": true
        }),
    )
}

/// Case: `split_exact` (`n_perm = 200`, `n_splits = 30`) over `k = [1, 2]` on
/// the wide inputs of [`split_exact_wide_k1`] (n = 60, p = 3000). Each k's p
/// is off the floor, so the combined p pins the min-p step itself.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn kset_wide_split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_kset_wide_split_exact",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            kwargs: serde_json::json!({
                "k": [1, 2], "test_method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30}, "seed": 42
            }),
            inputs_name: "pls1_confirmatory_wide_n60_d3000_inputs",
            n: 60,
            d: 3000,
            synth_seed: 42,
            k: Some(vec![1, 2]),
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Weak signal of three inputs files at n = 80, p = 6:
/// `inputs/pls1_confirmatory_sv_perm_inputs.npz` (the `sv_perm` and `score`
/// with `n_perm` cases, and the unweighted `raw_perm` / `split_exact`
/// permutation cases), `inputs/pls1_confirmatory_sv_perm_weighted_inputs.npz`
/// and `inputs/pls1_confirmatory_sv_perm_covariates_inputs.npz`. It puts p
/// mid-range so the fixtures pin the null columns' ranks. At `SYNTH_SNR` the
/// observed statistic beats every permutation and p sits at the floor
/// `1/(n_perm + 1)`, which pins nothing about the null.
const SV_PERM_SNR: f64 = 0.1;

/// A permutation-quadratic case (`sv_perm`, or `score` with `n_perm`) at
/// `k = 1` on the weak-signal inputs (n = 80, p = 6), raw or (kwarg
/// `"gram": true`) on `gram_matrix(X)` of the same X.
fn quad_perm_case(
    name: &'static str,
    args: ConfirmatoryArgs,
    kwargs: serde_json::Value,
) -> ConfirmatoryCase {
    ConfirmatoryCase {
        name,
        args,
        ci: None,
        kwargs,
        inputs_name: "pls1_confirmatory_sv_perm_inputs",
        n: SYNTH_N,
        d: SYNTH_D,
        synth_seed: CASE_SEED,
        k: Some(vec![1]),
        weights: None,
        snr: SV_PERM_SNR,
    }
}

const SV_PERM_99: ConfirmatoryArgs = ConfirmatoryArgs::SvPerm {
    n_perm: 99,
    rotate: false,
};

/// Case: `sv_perm` (`n_perm = 99`) at `k = 1` on the weak-signal inputs.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn sv_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &quad_perm_case(
            "pls1_confirmatory_sv_perm",
            SV_PERM_99,
            serde_json::json!({
                "k": 1, "test_method": "sv_perm", "args": {"n_perm": 99}, "seed": 42
            }),
        ),
    )
}

/// Case: `sv_perm` (`n_perm = 99`) on weak-signal inputs with the weights of
/// [`weighted_split_exact`].
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn sv_perm_weighted(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_sv_perm_weighted",
            args: SV_PERM_99,
            ci: None,
            kwargs: serde_json::json!({
                "k": 1, "test_method": "sv_perm", "args": {"n_perm": 99}, "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_sv_perm_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: Some(vec![1]),
            weights: Some(weighted_confirmatory_weights()),
            snr: SV_PERM_SNR,
        },
    )
}

/// Case: `sv_perm` (`n_perm = 99`) on weak-signal inputs with the covariates
/// of [`split_exact_covariates`]; `gram` runs it on `gram_matrix(X)`.
fn sv_perm_covariates_case(root: &Path, name: &'static str, gram: bool) -> Result<Case> {
    let mut kwargs = serde_json::json!({
        "k": 1, "test_method": "sv_perm", "args": {"n_perm": 99}, "seed": 42
    });
    if gram {
        kwargs["gram"] = serde_json::Value::Bool(true);
    }
    covariate_case(
        root,
        &covariate_descriptor(
            name,
            "pls1_confirmatory_sv_perm_covariates_inputs",
            SV_PERM_99,
            1,
            kwargs,
            SV_PERM_SNR,
        ),
        &covariate_data((SYNTH_N, SYNTH_D), CASE_SEED, None, SV_PERM_SNR),
    )
}

/// Case: `sv_perm` with covariates (Freedman-Lane null columns).
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn sv_perm_covariates(root: &Path) -> Result<Case> {
    sv_perm_covariates_case(root, "pls1_confirmatory_sv_perm_covariates", false)
}

/// Case: [`sv_perm_covariates`] on `gram_matrix(X)` of the same X.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn sv_perm_covariates_gram(root: &Path) -> Result<Case> {
    sv_perm_covariates_case(root, "pls1_confirmatory_test_gram_sv_perm_covariates", true)
}

/// Case: `score` with `n_perm = 99` (permutation p in place of the closed
/// form) on the inputs of [`sv_perm`]; at one seed it gives the same p.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn score_n_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &quad_perm_case(
            "pls1_confirmatory_score_n_perm",
            ConfirmatoryArgs::Score { n_perm: Some(99) },
            serde_json::json!({
                "k": 1, "test_method": "score", "args": {"n_perm": 99}, "seed": 42
            }),
        ),
    )
}

/// Case: [`score_n_perm`] on `gram_matrix(X)` of the same X.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn score_n_perm_gram(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &quad_perm_case(
            "pls1_confirmatory_test_gram_score_n_perm",
            ConfirmatoryArgs::Score { n_perm: Some(99) },
            serde_json::json!({
                "k": 1, "test_method": "score", "args": {"n_perm": 99}, "seed": 42,
                "gram": true
            }),
        ),
    )
}

/// Case: [`sv_perm`] on `gram_matrix(X)` of the same X.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn sv_perm_gram(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &quad_perm_case(
            "pls1_confirmatory_test_gram_sv_perm",
            SV_PERM_99,
            serde_json::json!({
                "k": 1, "test_method": "sv_perm", "args": {"n_perm": 99}, "seed": 42,
                "gram": true
            }),
        ),
    )
}

/// Case: `split_nb_gate` on `inputs/pls1_confirmatory_inputs.npz` (the
/// default n = 80, p = 6 design, no weights) with `pre_standardized =
/// false`.
///
/// Inputs: `X`. Outputs: `fires` (0/1), `stable_rank`, `n_eff`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `split_nb_gate` fails.
pub fn split_nb_gate_n80_d6(root: &Path) -> Result<Case> {
    let name = "split_nb_gate_n80_d6";
    let function = "split_nb_gate";
    let rel_inputs = "inputs/pls1_confirmatory_inputs.npz".to_string();
    let rel_outputs = format!("outputs/{function}/{name}.npz");
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, CASE_SEED);
    let data = Xyw {
        x,
        y,
        ..Default::default()
    };
    data.write(&abs_inputs)?;

    let r = split_nb_gate(data.x_faer().as_ref(), None, false)?;
    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        // bool has no npz scalar type in this writer; stored as 0/1.
        w.add_i64("fires", &scalar_i64(i64::from(r.fires)))?;
        w.add_f64("stable_rank", &scalar_f64(r.stable_rank))?;
        w.add_f64("n_eff", &scalar_f64(r.n_eff))?;
        w.finish()?;
    }

    manifest_case(
        root,
        name,
        function,
        rel_inputs,
        rel_outputs,
        serde_json::json!({"pre_standardized": false}),
    )
}
