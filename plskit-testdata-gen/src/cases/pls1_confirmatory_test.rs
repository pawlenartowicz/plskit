//! `pls1_confirmatory_test` fixture cases.
//!
//! Eight unweighted cases share one inputs file (`inputs/pls1_confirmatory_inputs.npz`):
//! six base-method cases (no CI; two of them, `split_exact` and `split_exact_k1`, cover
//! `split_exact`'s refit and no-refit routes respectively) and two CI-bundle variants
//! (`level` 0.95 and 0.8).
//! Five weighted cases share a separate inputs file
//! (`inputs/pls1_confirmatory_weighted_inputs.npz`) that also carries `weights`.
//! One further case, `raw_perm_wide`, carries its own wide (n=30, p=100) inputs
//! file because it is the only shape that routes through `dual_route`. Two
//! K=2 cases on a wider design (n=60, p=3000), `raw_perm_wide_k2` and
//! `split_exact_wide_k2`, share `inputs/pls1_confirmatory_wide_n60_d3000_inputs.npz`:
//! both run the primal engine at p ≫ n.

use std::path::Path;

use anyhow::Result;

use plskit::{
    pls1_confirmatory_test, CIOpts, ConfirmatoryArgs, ConfirmatoryCI, ConfirmatoryTestInput,
    ConfirmatoryTestOpts, ConfirmatoryTestOutput,
};

use crate::cases::{
    case_files, default_tolerance, perm_null_weights, scalar_f64, scalar_i64, synth_data,
    weighted_prestd_n80_d6, Xyw, WEIGHTED_PRESTD_N80_D6_INPUTS,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};

/// Default synth design shared by every case except `raw_perm_wide`, which
/// needs `n < p` and sets its own `n` / `d` on the descriptor.
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
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
    k: usize,
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
    };
    run_confirmatory_on(root, c, &data, |_| Ok(()))
}

/// The call a case describes, on `data`. `pre_standardized` is read from the
/// case's `kwargs`, so the manifest records exactly the flag the call used.
fn confirmatory_call(c: &ConfirmatoryCase, data: &Xyw) -> Result<ConfirmatoryTestOutput> {
    let pre_standardized = c
        .kwargs
        .get("pre_standardized")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let x_faer = data.x_faer();
    let y_faer = data.y_faer();
    let weights_faer = data.w_faer();
    Ok(pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x_faer.as_ref(),
            y: y_faer.as_ref(),
            k: c.k,
            weights: weights_faer.as_ref().map(faer::Col::as_ref),
        },
        ConfirmatoryTestOpts {
            args: c.args,
            pre_standardized,
            seed: Some(CASE_SEED),
            verbose: false,
            ci: c.ci,
            max_skip_rate: 0.01,
            keep: None,
        },
    )?)
}

/// Write `data` to the case's inputs file (idempotent: same bytes every
/// call for a shared file), run the call, apply the generator guard `check`
/// to its output, and write the case's outputs file.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_confirmatory_test`
/// fails, or `check` rejects the output.
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

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        w.add_f64("statistic", &scalar_f64(r.statistic))?;
        w.add_string("test_method", &r.test_method)?;
        w.add_i64("k", &scalar_i64(i64::try_from(r.k)?))?;
        if let Some(np) = r.n_perm {
            w.add_i64("n_perm", &scalar_i64(i64::try_from(np)?))?;
        }
        if let Some(ns) = r.n_splits {
            w.add_i64("n_splits", &scalar_i64(i64::try_from(ns)?))?;
        }
        if let Some(sr) = r.stable_rank {
            w.add_f64("stable_rank", &scalar_f64(sr))?;
        }
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        if let Some(ci) = &r.ci {
            write_ci_fields(&mut w, ci)?;
        }
        w.finish()?;
    }

    Ok(Case {
        name: c.name.to_string(),
        function: FUNCTION.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs: c.kwargs.clone(),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
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
            inputs_name: "pls1_confirmatory_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: 2,
            weights: None,
            snr: SYNTH_SNR,
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
            k: 2,
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
            inputs_name: "pls1_confirmatory_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: 2,
            weights: None,
            snr: SYNTH_SNR,
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
            inputs_name: "pls1_confirmatory_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 42,
            k: 1,
            weights: None,
            snr: SYNTH_SNR,
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
            args: ConfirmatoryArgs::Score,
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
            k: 2,
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
            k: 2,
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
    let data = Xyw { x, y, w: None };
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
        k: 2,
        weights: None,
        snr: SYNTH_SNR,
    }
}

/// Case: `pls1_confirmatory_test` with `test_method=auto` on n=60, p=20, `k=2`,
/// `n_perm=200`, `n_splits=30`, `seed=42`. n is below the cutoff of 250, so the
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
            inputs_name: "pls1_confirmatory_auto_split_exact_inputs",
            n: 60,
            d: 20,
            synth_seed: 42,
            k: 2,
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=auto` on n=300, p=10, `k=2`,
/// `n_perm=200`, `n_splits=30`, `seed=42`. n clears the cutoff of 250, p is in
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
            k: 2,
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=raw_perm` on a wide design
/// (n=30, p=100), `k=1`, `n_perm=100`, `n_folds=5`.
///
/// The corpus's only fixture that reaches the Gram route in
/// `plskit-rs/src/dual_route.rs`. `run_raw_perm` takes that route when
/// `k == 1`, `keep` is unset, the design is unweighted, and
/// `n_tr·(B·q + p) < p·B·q` holds for `n_tr = n − n/n_folds = 24`, `q = 1`
/// and `B = n_perm + 1`. At this shape the inequality reduces to
/// `n_perm ≥ 31`, so `n_perm = 100` clears it by more than 3×. Lowering
/// `n_perm`, raising `n_folds` or narrowing X silently sends the fixture
/// back onto the primal route, which would leave the Gram route with no
/// cross-language coverage at all.
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
            inputs_name: "pls1_confirmatory_wide_inputs",
            n: 30,
            d: 100,
            synth_seed: 42,
            k: 1,
            weights: None,
            snr: SYNTH_SNR,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `test_method=raw_perm` on a wide design
/// (n=60, p=3000), `k=2`, `n_perm=200`, `n_folds=5`.
///
/// The primal `raw_perm` route at p ≫ n: `k = 2` keeps it off the K = 1 Gram
/// closed form, and the shape is eligible for an n-space Gram route at
/// K ≥ 2 (both pinned in `plskit-rs/src/fixture_route_pins.rs`), so this
/// fixture is the primal reference such a route must reproduce.
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
            k: 2,
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
            k: 2,
            weights: None,
            snr: SYNTH_SNR,
        },
    )
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
            inputs_name: "pls1_confirmatory_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
            snr: SYNTH_SNR,
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
            k: 1,
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
            inputs_name: "pls1_confirmatory_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
            snr: SYNTH_SNR,
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
            args: ConfirmatoryArgs::Score,
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
            k: 1,
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
            k: 1,
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
            k: 2,
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
/// Pins score's `d > n` eigen branch (`X X'`, through the p-value) and its
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
        args: ConfirmatoryArgs::Score,
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
        k: 1,
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
    Xyw { x, y, w: weights }
}

/// Case: [`score_wide_pre_standardized`]'s call under weights
/// [`perm_null_weights`]`(30)`, on its inputs plus `weights` (own inputs
/// file). Under `pre_standardized = true` the weights still row-scale X and
/// y by √w, and at `d > n` the score's eigenvalues come from the √w-scaled
/// `X X'`; the only fixture of either. The generator refuses it unless the
/// statistic stays below `1e3` (the `1e-12` scalar tolerance must stay above
/// its last bit), the p-value is interior (so it pins the eigenvalues), and
/// the output moves both without the weights and without the flag.
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
            args: ConfirmatoryArgs::Score,
            ci: None,
            kwargs,
            inputs_name: "pls1_confirmatory_weighted_score_wide_pre_standardized",
            n: 30,
            d: 100,
            synth_seed: CASE_SEED,
            k: 1,
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
        anyhow::ensure!(
            r.statistic < 1e3 && r.pvalue > 1e-3 && r.pvalue < 0.999,
            "{name}: statistic {} (want < 1e3), pvalue {} (want interior)",
            r.statistic,
            r.pvalue
        );
        for (what, other) in [
            ("unweighted", &unweighted),
            ("pre_standardized = false", &without_flag),
        ] {
            crate::cases::ensure_moved(
                name,
                &format!("[statistic, pvalue] vs {what}"),
                &[r.statistic, r.pvalue],
                &[other.statistic, other.pvalue],
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
            inputs_name: "pls1_confirmatory_weighted_inputs",
            n: SYNTH_N,
            d: SYNTH_D,
            synth_seed: 77,
            k: 2,
            weights: Some(weighted_confirmatory_weights()),
            snr: SYNTH_SNR,
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
            k: 1,
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
        k: 2,
        weights: Some(weights),
        snr: SYNTH_SNR,
    }
}

/// Case: [`split_nb_ci`]'s call under weights, on the weighted inputs of
/// [`weighted_raw_perm`]. The only fixture of the weighted subsampling CI
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
