//! `spls1_find_k_optimal` fixture cases (sparse PLS1, mode 3 — k sweep at fixed keep).

use std::path::Path;

use anyhow::Result;

use crate::cases::pls1_find_k_optimal::write_btreemap;
use crate::cases::{
    case_files, default_tolerance, ensure_moved, faer_col_to_array, scalar_i64, synth_data,
    weighted_n80_d6, Xyw, WEIGHTED_N80_D6_INPUTS,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use plskit::{spls1_find_k_optimal, FindKOptimalOpts, FindKOptimalOutput, Selector};

/// Shared synth parameters (mirrors dense `pls1_find_k_optimal` cases).
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
const K_MAX: usize = 4;
const KEEP: usize = 3;
/// Shared inputs filename stem.
const INPUTS_NAME: &str = "spls1_find_k_optimal_inputs";
const FUNCTION: &str = "spls1_find_k_optimal";

/// Case: `spls1_find_k_optimal` with `selector=r2_se`, `keep=3`, `n_folds=5`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_optimal` fails.
pub fn r2_se_keep3(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, SYNTH_SEED);
    optimal_on(
        root,
        "spls1_find_k_optimal_r2_se_keep3",
        INPUTS_NAME,
        &Xyw { x, y, w: None },
        |_| Ok(()),
    )
}

/// Case: [`r2_se_keep3`]'s call under weights (`cases::weighted_n80_d6`, the
/// inputs file of `pls1_perm_null_weighted_n80_d6_k2`). Pins the weighted
/// fold slicing and renormalization of the CV path every optimal entry point
/// shares. The generator refuses it unless the CV scores move when the
/// weights are dropped.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `spls1_find_k_optimal`
/// fails, or the weights do not move the CV scores.
pub fn r2_se_keep3_weighted(root: &Path) -> Result<Case> {
    let name = "spls1_find_k_optimal_r2_se_keep3_weighted";
    let data = weighted_n80_d6();
    let unweighted = r2_se_keep3_call(&Xyw {
        x: data.x.clone(),
        y: data.y.clone(),
        w: None,
    })?;
    optimal_on(root, name, WEIGHTED_N80_D6_INPUTS, &data, |r| {
        let scores = |o: &FindKOptimalOutput| -> Vec<f64> {
            o.cv_scores
                .as_ref()
                .map(|m| m.values().copied().collect())
                .unwrap_or_default()
        };
        ensure_moved(name, "cv_scores", &scores(r), &scores(&unweighted))
    })
}

/// `spls1_find_k_optimal` at `k_max=4`, `keep=3`, `selector=r2_se`,
/// `n_folds=5`, `seed=42`, with `data`'s weights.
fn r2_se_keep3_call(data: &Xyw) -> Result<FindKOptimalOutput> {
    let w = data.w_faer();
    Ok(spls1_find_k_optimal(
        data.x_faer().as_ref(),
        data.y_faer().as_ref(),
        K_MAX,
        KEEP,
        w.as_ref().map(faer::Col::as_ref),
        FindKOptimalOpts {
            selector: Selector::R2Se,
            n_folds: 5,
            seed: Some(42),
            ..FindKOptimalOpts::default()
        },
    )?)
}

/// Write `data` to `inputs/{inputs_name}.npz` (idempotent: same bytes every
/// call for a shared file), run [`r2_se_keep3_call`], apply the generator
/// guard `check`, and write the outputs.
fn optimal_on(
    root: &Path,
    name: &str,
    inputs_name: &str,
    data: &Xyw,
    check: impl FnOnce(&FindKOptimalOutput) -> Result<()>,
) -> Result<Case> {
    let rel_inputs = format!("inputs/{inputs_name}.npz");
    let rel_outputs = format!("outputs/{FUNCTION}/{name}.npz");
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    data.write(&abs_inputs)?;

    let r = r2_se_keep3_call(data)?;
    check(&r)?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(r.k_star)?))?;
        w.add_string("selector", &r.selector)?;
        if let Some(m) = &r.cv_scores {
            write_btreemap(&mut w, "cv_scores__keys", "cv_scores__values", m)?;
        }
        if let Some(m) = &r.cv_scores_se {
            write_btreemap(&mut w, "cv_scores_se__keys", "cv_scores_se__values", m)?;
        }
        if let Some(m) = &r.bic_scores {
            write_btreemap(&mut w, "bic_scores__keys", "bic_scores__values", m)?;
        }
        if let Some(ref col) = r.pvalues {
            w.add_f64("pvalues", &faer_col_to_array(col))?;
        }
        if let Some(ref s) = r.diagnostic {
            w.add_string("diagnostic", s)?;
        }
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    let mut kwargs = serde_json::json!({
        "k_max": K_MAX,
        "keep": KEEP,
        "selector": "r2_se",
        "args": {"n_folds": 5},
        "seed": 42
    });
    if data.w.is_some() {
        kwargs["weights"] = serde_json::json!("nonuniform");
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
        tolerance: Some(default_tolerance()),
    })
}
