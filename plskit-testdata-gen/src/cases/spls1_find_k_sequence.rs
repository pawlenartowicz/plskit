//! `spls1_find_k_sequence` fixture cases (sparse PLS1: sequential test at fixed keep).

use std::path::Path;

use anyhow::Result;

use crate::cases::{
    case_files, ensure_moved, faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat,
    scalar_f64, scalar_i64, synth_data, weighted_prestd_n80_d6, Xyw, WEIGHTED_PRESTD_N80_D6_INPUTS,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use plskit::{spls1_find_k_sequence, ConfirmatoryMethod, FindKSequenceOpts, FindKSequenceOutput};

/// Shared synth parameters. `n` varies between cases, and `e_keep3` lowers
/// the SNR to `E_SNR`.
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
const K_MAX: usize = 4;
const KEEP: usize = 3;
const FUNCTION: &str = "spls1_find_k_sequence";

/// Case: `spls1_find_k_sequence` with `test_method=split_nb`, `keep=3`, `n_splits=20`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_sequence` fails.
pub fn split_nb_keep3(root: &Path) -> Result<Case> {
    split_nb_case(
        root,
        "spls1_find_k_sequence_split_nb_keep3",
        "spls1_find_k_sequence_inputs",
        80,
    )
}

/// Case: same call as `split_nb_keep3` but at `n = 20`, where the design's
/// `n_eff` (20 < `SPLIT_NB_GATE_MIN_N_EFF` = 25, see `signal_test.rs`) trips
/// the hoisted `split_nb` auto-gate. This is the corpus's one case documenting
/// the `split_nb` -> `split_exact` reroute: `test_method` on this fixture reads
/// `"split_exact"` even though the requested method is `split_nb` (every other
/// `split_nb` fixture clears the gate and reports genuine NB output).
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_sequence` fails.
pub fn split_nb_keep3_gated(root: &Path) -> Result<Case> {
    split_nb_case(
        root,
        "spls1_find_k_sequence_split_nb_keep3_gated",
        "spls1_find_k_sequence_gated_inputs",
        20,
    )
}

fn split_nb_case(root: &Path, name: &str, inputs_name: &str, synth_n: usize) -> Result<Case> {
    sequence_case(
        root,
        name,
        inputs_name,
        &synth(synth_n),
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitNb,
            n_splits: 20,
            seed: Some(42),
            ..FindKSequenceOpts::default()
        },
        serde_json::json!({
            "k_max": K_MAX,
            "keep": KEEP,
            "test_method": "split_nb",
            "args": {"n_splits": 20},
            "seed": 42
        }),
        |_| Ok(()),
    )
}

/// Case: `spls1_find_k_sequence` with `test_method=split_exact`, `keep=3`,
/// `n_perm=200`, `n_splits=20`, `seed=42`, on the `split_nb_keep3` inputs.
///
/// The only public path to `split_exact`'s sparse refit route: each step
/// tests at `k = 1` on the deflated residual with `keep` set, and `keep`
/// rules out the K = 1 no-refit route. On these inputs the third step's
/// p-value is interior, so the fixture pins the refit statistic, not only
/// its `1/(n_perm + 1)` floor.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_sequence` fails.
pub fn split_exact_keep3(root: &Path) -> Result<Case> {
    sequence_case(
        root,
        "spls1_find_k_sequence_split_exact_keep3",
        "spls1_find_k_sequence_inputs",
        &synth(80),
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitExact,
            n_perm: 200,
            n_splits: 20,
            seed: Some(42),
            ..FindKSequenceOpts::default()
        },
        serde_json::json!({
            "k_max": K_MAX,
            "keep": KEEP,
            "test_method": "split_exact",
            "args": {"n_perm": 200, "n_splits": 20},
            "seed": 42
        }),
        |_| Ok(()),
    )
}

/// `synth_data(synth_n, 6, 2, 4.0, 42)`, unweighted: the inputs of every
/// synth case in this module except the tall one and `e_keep3`.
fn synth(synth_n: usize) -> Xyw {
    let (x, y) = synth_data(synth_n, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, SYNTH_SEED);
    Xyw { x, y, w: None }
}

/// Shared body of every case in this module: write the inputs (idempotent,
/// same bytes every call), run `spls1_find_k_sequence` at `k_max = 4`,
/// `keep = 3` with `opts` and `data`'s weights, apply the generator guard
/// `check`, write the outputs, and return the manifest entry with `kwargs`.
fn sequence_case(
    root: &Path,
    name: &str,
    inputs_name: &str,
    data: &Xyw,
    opts: FindKSequenceOpts,
    kwargs: serde_json::Value,
    check: impl FnOnce(&FindKSequenceOutput) -> Result<()>,
) -> Result<Case> {
    let rel_inputs = format!("inputs/{inputs_name}.npz");
    let rel_outputs = format!("outputs/{FUNCTION}/{name}.npz");
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    data.write(&abs_inputs)?;

    let r = sequence_call(data, opts)?;
    check(&r)?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(r.k_star)?))?;
        w.add_f64("pvalues", &faer_col_to_array(&r.pvalues))?;
        w.add_string("test_method", &r.test_method)?;
        w.add_f64("alpha", &scalar_f64(r.alpha))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
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

/// `spls1_find_k_sequence` at `k_max = 4`, `keep = 3` on `data`.
fn sequence_call(data: &Xyw, opts: FindKSequenceOpts) -> Result<FindKSequenceOutput> {
    let w = data.w_faer();
    Ok(spls1_find_k_sequence(
        data.x_faer().as_ref(),
        data.y_faer().as_ref(),
        K_MAX,
        KEEP,
        w.as_ref().map(faer::Col::as_ref),
        opts,
    )?)
}

/// Case: `spls1_find_k_sequence` with `test_method=raw_perm`, `keep=3`,
/// `n_perm=100`, `seed=42`, on the `split_nb_keep3` inputs. The only fixture
/// of `raw_perm` with a sparse inner fit; the steps after the signal give
/// interior p-values, so the null columns are pinned.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_sequence` fails.
pub fn raw_perm_keep3(root: &Path) -> Result<Case> {
    sequence_case(
        root,
        "spls1_find_k_sequence_raw_perm_keep3",
        "spls1_find_k_sequence_inputs",
        &synth(80),
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::RawPerm,
            n_perm: 100,
            seed: Some(42),
            ..FindKSequenceOpts::default()
        },
        serde_json::json!({
            "k_max": K_MAX,
            "keep": KEEP,
            "test_method": "raw_perm",
            "args": {"n_perm": 100},
            "seed": 42
        }),
        |_| Ok(()),
    )
}

/// Signal-to-noise ratio of `e_keep3`'s inputs: weak enough that the
/// e-value of the first step is finite and interior (at the other cases'
/// `SYNTH_SNR` it is about `1e-22`, below the corpus tolerance).
const E_SNR: f64 = 0.3;

/// Case: `spls1_find_k_sequence` with `test_method=e`, `keep=3`, `seed=42`,
/// on `synth_data(80, 6, 2, E_SNR, 42)`. The only fixture of the e-value
/// test with a sparse inner fit. The generator refuses it unless the first
/// step's p-value is interior (`1e-6 < p < 0.9`): the e-value p is
/// `min(1, 1/E)`, so a strong signal records a p-value under the corpus
/// tolerance and a null step records the clamp, and neither pins the
/// statistic.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `spls1_find_k_sequence`
/// fails, or the guard fails.
pub fn e_keep3(root: &Path) -> Result<Case> {
    let name = "spls1_find_k_sequence_e_keep3";
    let (x, y) = synth_data(80, SYNTH_D, SYNTH_K_SIGNAL, E_SNR, SYNTH_SEED);
    sequence_case(
        root,
        name,
        "spls1_find_k_sequence_weak_inputs",
        &Xyw { x, y, w: None },
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::E,
            seed: Some(42),
            ..FindKSequenceOpts::default()
        },
        serde_json::json!({
            "k_max": K_MAX,
            "keep": KEEP,
            "test_method": "e",
            "args": {},
            "seed": 42
        }),
        |r| {
            let p = r.pvalues[0];
            anyhow::ensure!(
                p > 1e-6 && p < 0.9,
                "{name}: first-step p-value {p} is not interior; retune E_SNR"
            );
            Ok(())
        },
    )
}

/// Case: [`split_exact_keep3`]'s call under weights with
/// `pre_standardized = true`, on X and y standardized by their unweighted
/// moments (`cases::weighted_prestd_n80_d6`). Pins the weighted, keep-3
/// deflation between steps. The generator refuses it unless at least two
/// steps reject, the first non-rejecting step's p-value is interior (a
/// floor p-value pins nothing), and the p-values move when the flag is dropped.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `spls1_find_k_sequence`
/// fails, or the guard fails.
pub fn split_exact_keep3_weighted_prestd(root: &Path) -> Result<Case> {
    let name = "spls1_find_k_sequence_split_exact_keep3_weighted_prestd";
    let n_perm = 200;
    let opts = |pre_standardized| FindKSequenceOpts {
        test_method: ConfirmatoryMethod::SplitExact,
        n_perm,
        n_splits: 20,
        pre_standardized,
        seed: Some(42),
        ..FindKSequenceOpts::default()
    };
    let data = weighted_prestd_n80_d6();
    let without = sequence_call(&data, opts(false))?;
    sequence_case(
        root,
        name,
        WEIGHTED_PRESTD_N80_D6_INPUTS,
        &data,
        opts(true),
        serde_json::json!({
            "k_max": K_MAX,
            "keep": KEEP,
            "test_method": "split_exact",
            "args": {"n_perm": n_perm, "n_splits": 20},
            "seed": 42,
            "weights": "nonuniform",
            "pre_standardized": true
        }),
        |r| {
            let p_min = 1.0 / f64::from(u32::try_from(n_perm + 1)?);
            anyhow::ensure!(
                r.k_star >= 2 && r.k_star < r.pvalues.nrows(),
                "{name}: k_star = {} is not in [2, k_max)",
                r.k_star
            );
            let p = r.pvalues[r.k_star];
            anyhow::ensure!(
                p > 1.5 * p_min && p < 1.0,
                "{name}: p-value {p} of the first non-rejecting step is not interior"
            );
            let col = |c: &faer::Col<f64>| (0..c.nrows()).map(|i| c[i]).collect::<Vec<_>>();
            ensure_moved(name, "pvalues", &col(&r.pvalues), &col(&without.pvalues))
        },
    )
}

/// Tall weak-signal design of `split_exact_tall_keep10`: one active column at
/// a signal-to-noise ratio low enough that the first step's p-value is
/// interior (the generator refuses to write the case otherwise).
const TALL_N: usize = 2000;
const TALL_D: usize = 200;
const TALL_K_SIGNAL: usize = 1;
const TALL_SNR: f64 = 0.05;
const TALL_SYNTH_SEED: u64 = 7;
const TALL_KEEP: usize = 10;
const TALL_K_MAX: usize = 3;
const TALL_N_PERM: usize = 1000;
const TALL_N_SPLITS: usize = 20;

/// Case: `spls1_find_k_sequence` with `test_method=split_exact`, `keep=10`,
/// `n_perm=1000`, `n_splits=20`, `seed=42`, on a tall weak-signal design
/// (n=2000, d=200).
///
/// The sparse `split_exact` refit route is reachable from the public surface
/// only through this function, whose steps test at k = 1 on the deflated
/// residual. Each half has `n_tr = 1000` rows and `B = n_perm + 1 = 1001`
/// columns, so the X backend's work per block is `B·3·n_tr·d = 6.006e8`,
/// about 6× the Gram backend's starting work floor (d = 50 would clear it by
/// only 1.5×). The corpus record holds only `k_star`, `pvalues`,
/// `test_method`, `alpha` and `seed`, and a step rejected at `p = 1/1001`
/// pins nothing, hence the weak signal and the interior-p guard below.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `spls1_find_k_sequence`
/// fails, or the first step's p-value is not interior.
pub fn split_exact_tall_keep10(root: &Path) -> Result<Case> {
    let name = "spls1_find_k_sequence_split_exact_tall_keep10";
    let inputs_name = "spls1_find_k_sequence_tall_inputs";
    let rel_inputs = format!("inputs/{inputs_name}.npz");
    let rel_outputs = format!("outputs/{FUNCTION}/{name}.npz");
    let abs_inputs = root.join(&rel_inputs);
    let abs_outputs = root.join(&rel_outputs);
    if let Some(p) = abs_inputs.parent() {
        std::fs::create_dir_all(p)?;
    }
    if let Some(p) = abs_outputs.parent() {
        std::fs::create_dir_all(p)?;
    }

    let (x, y) = synth_data(TALL_N, TALL_D, TALL_K_SIGNAL, TALL_SNR, TALL_SYNTH_SEED);
    {
        let mut w = NpzWriter::create(&abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let r = spls1_find_k_sequence(
        x_faer.as_ref(),
        y_faer.as_ref(),
        TALL_K_MAX,
        TALL_KEEP,
        None,
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitExact,
            n_perm: TALL_N_PERM,
            n_splits: TALL_N_SPLITS,
            seed: Some(42),
            ..FindKSequenceOpts::default()
        },
    )?;

    let p_min = 1.0 / f64::from(u32::try_from(TALL_N_PERM + 1)?);
    let first = r.pvalues[0];
    anyhow::ensure!(
        first > 1.5 * p_min && first < 1.0,
        "{name}: first-step p-value {first} is not interior; lower TALL_SNR"
    );

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(r.k_star)?))?;
        w.add_f64("pvalues", &faer_col_to_array(&r.pvalues))?;
        w.add_string("test_method", &r.test_method)?;
        w.add_f64("alpha", &scalar_f64(r.alpha))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: FUNCTION.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs: serde_json::json!({
            "k_max": TALL_K_MAX,
            "keep": TALL_KEEP,
            "test_method": "split_exact",
            "args": {"n_perm": TALL_N_PERM, "n_splits": TALL_N_SPLITS},
            "seed": 42
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
    })
}
