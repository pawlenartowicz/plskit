//! Site-level tests for the n-space route at `raw_perm`, `split_exact` and
//! `split_nb`: the units and the drivers on both routes (called directly
//! with `ReplicateRoute::Nspace` and `ReplicateRoute::Primal`), the blocks
//! taken from the all-row Gram matrix against the products of the halves,
//! route invisibility on the public surface, and degenerate folds and
//! halves.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]

use super::*;
use crate::held_out_gram::{HeldOutGram, grams_built};
use crate::linalg::fold_split;
use crate::resample::Columns;
use crate::rng::{child_seeds, resolve_seed};
use crate::signal_test::{
    ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput,
    ReplicateRoute, SplitIdx, draw_splits, first_row, fold_block, fold_columns_nspace, fold_unit,
    pls1_confirmatory_test, pooled_cv_r2_columns, prepare_cv_fold, prepare_split, raw_perm_route,
    split_block, split_columns_nspace, split_exact_refit_route, split_unit, split_zbars_columns,
    with_gram_routes_disabled,
};
use crate::test_support::{orthonormal_basis, project_off};
use faer::{Col, Mat, MatRef, Par};
use rand::{RngExt, SeedableRng};

fn wide(n: usize, p: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
    let e = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| snr * x[(i, 0)] + e[i]);
    (x, y)
}

fn seeds_for(n_perm: usize, seed: u64) -> Vec<u64> {
    let (_, mut rng) = resolve_seed(Some(seed)).expect("seed");
    child_seeds(&mut rng, n_perm)
}

fn folds_of(n: usize, n_folds: usize, seed: u64) -> Vec<Vec<usize>> {
    use rand::seq::SliceRandom;
    let (_, mut rng) = resolve_seed(Some(seed)).expect("seed");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    fold_split(&idx, n_folds)
}

/// One fold's `(ss_res, ss_tot)` on both routes: `ss_tot` bit for bit (it
/// depends only on `y`), `|Δss_res| ≤ 1e-10·max(1, ss_tot)`.
fn assert_fold_contribution_close(a: (f64, f64), b: (f64, f64), what: &str) {
    assert_eq!(a.1.to_bits(), b.1.to_bits(), "{what}: ss_tot");
    assert!(
        (a.0 - b.0).abs() <= 1e-10 * b.1.max(1.0),
        "{what}: ss_res {:e} vs {:e}",
        a.0,
        b.0
    );
}

fn raw_perm(
    x: &Mat<f64>,
    y: &Col<f64>,
    k: usize,
    n_folds: usize,
    seed: u64,
) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: Some(&[k]),
            weights: None,
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 50,
                n_folds,
            },
            seed: Some(seed),
            ..Default::default()
        },
    )
    .expect("raw_perm")
}

/// The public call on both routes: the selector picks `Nspace`, the call
/// builds one block per fold, the statistic agrees within `1e-10` and the
/// p-value to the bit. (A null within `1e-10` of the observed statistic
/// would be the one legitimate p-value difference; the driver tests check
/// counts with that exception, and these seeds have no such tie.)
fn assert_public_raw_perm_invisible(
    x: &Mat<f64>,
    y: &Col<f64>,
    k: usize,
    n_folds: usize,
    seed: u64,
) {
    let (n, p) = (x.nrows(), x.ncols());
    assert_eq!(
        raw_perm_route(n, n_folds, p, 50, k, None, false),
        ReplicateRoute::Nspace,
        "k={k}"
    );
    // Budget 0 runs the folds one at a time on this thread, where the block
    // counter lives; the bits do not depend on the budget.
    let before = nspace_blocks_built();
    let gram = crate::fit::with_budget(0, || raw_perm(x, y, k, n_folds, seed));
    assert_eq!(
        nspace_blocks_built(),
        before + n_folds,
        "k={k}: the n-space route must run"
    );
    let primal = with_gram_routes_disabled(|| raw_perm(x, y, k, n_folds, seed));
    assert!(
        (gram.statistic.unwrap() - primal.statistic.unwrap()).abs() <= 1e-10,
        "k={k}: {:e} vs {:e}",
        gram.statistic.unwrap(),
        primal.statistic.unwrap()
    );
    assert_eq!(
        gram.pvalue.to_bits(),
        primal.pvalue.to_bits(),
        "k={k}: p-value"
    );
}

/// Fold `fi`'s `(ss_res, ss_tot)` for the outcome `y` on the n-space route,
/// from the all-row Gram matrix of `x`.
fn nspace_fold(
    x: &Mat<f64>,
    folds: &[Vec<usize>],
    fi: usize,
    y: &Col<f64>,
    k: usize,
) -> (f64, f64) {
    let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
    let g = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
    let r = fold_columns_nspace(folds, fi, &g, &[k], 1, &|_| y.clone());
    r[0].as_ref().expect("nspace unit")[0]
}

#[test]
fn raw_perm_fold_units_match_the_primal_arm() {
    let (x, y) = wide(40, 2000, 0.5, 1);
    let folds = folds_of(40, 5, 2);
    let seeds = seeds_for(20, 3);
    let cols = Columns {
        y: y.as_ref().as_mat(),
        seeds: &seeds,
        null_map: None,
    };
    // Closeness alone would pass if the n-space route returned the primal
    // arm's answer, so this also counts units whose ss_res differs in bits
    // from the primal arm's and requires at least one such unit.
    let mut differing = 0usize;
    let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
    let g = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
    for fi in 0..folds.len() {
        let fold = prepare_cv_fold(x.as_ref(), &folds, fi, None, scale.as_ref()).expect("fold");
        let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
        for k in 2..=K_DUAL_MAX {
            let nspace = fold_columns_nspace(&folds, fi, &g, &[k], cols.len(), &|c| cols.column(c));
            for (c, unit) in nspace.iter().enumerate() {
                let yc = cols.column(c);
                let y_of = |i: usize| yc[i];
                let a = unit.as_ref().expect("nspace unit")[0];
                let b = fold_unit(&primal, &fold, &y_of, &[k], None).expect("primal unit")[0];
                assert_fold_contribution_close(a, b, &format!("fold {fi}, k={k}, column {c}"));
                if a.0.to_bits() != b.0.to_bits() {
                    differing += 1;
                }
            }
        }
    }
    assert!(
        differing > 0,
        "no unit's ss_res differed from the primal arm in bits: the n-space \
         kernel did not run"
    );
}

#[test]
fn raw_perm_is_route_invisible_on_the_public_surface() {
    for k in 2..=K_DUAL_MAX {
        let seed = 20 + k as u64;
        let (x, y) = wide(40, 2000, 0.3, seed);
        assert_public_raw_perm_invisible(&x, &y, k, 5, seed);
    }
}

#[test]
fn raw_perm_keeps_its_other_routes() {
    // k = 1: the closed form; past K_DUAL_MAX, sparse or weighted: primal.
    assert_eq!(
        raw_perm_route(40, 5, 2000, 50, 1, None, false),
        ReplicateRoute::Special
    );
    assert_eq!(
        raw_perm_route(40, 5, 2000, 50, K_DUAL_MAX + 1, None, false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        raw_perm_route(40, 5, 2000, 50, 2, Some(3), false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        raw_perm_route(40, 5, 2000, 50, 2, None, true),
        ReplicateRoute::Primal
    );
    assert_eq!(
        with_gram_routes_disabled(|| raw_perm_route(40, 5, 2000, 50, 2, None, false)),
        ReplicateRoute::Primal
    );
    let (x, y) = wide(40, 2000, 0.3, 24);
    let (grams, before) = (grams_built(), nspace_blocks_built());
    let _ = raw_perm(&x, &y, 1, 5, 24);
    assert_eq!(nspace_blocks_built(), before, "k = 1 keeps the closed form");
    assert_eq!(grams_built(), grams + 1, "k = 1: one G per call");
}

#[test]
fn k_at_the_smallest_fold_ceiling_matches_the_primal() {
    // k = K_DUAL_MAX with the smallest training fold of 5 at k + 2 rows
    // (n = 15), one above rank(X̃_tr) = k + 1, on the public call.
    let k = K_DUAL_MAX;
    let n: usize = 15;
    assert_eq!(
        n - n.div_ceil(5),
        k + 2,
        "n = 15 must sit at the k + 2 fold-size boundary"
    );
    let (x, y) = wide(n, 1500, 0.5, 45);
    assert_public_raw_perm_invisible(&x, &y, k, 5, 44);
}

#[test]
fn a_constant_training_outcome_is_the_zero_model_to_the_bit() {
    // y is zero off fold 0, so fold 0's training outcome is constant,
    // z is exactly zero, and both routes predict zero.
    let (x, _) = wide(40, 2000, 0.0, 51);
    let folds = fold_split(&(0..40).collect::<Vec<_>>(), 5);
    let y = Col::<f64>::from_fn(40, |i| if i < 8 { i as f64 - 3.5 } else { 0.0 });
    let fold = prepare_cv_fold(
        x.as_ref(),
        &folds,
        0,
        None,
        crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
    )
    .expect("fold");
    let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
    let y_of = |i: usize| y[i];
    let (res_g, tot_g) = nspace_fold(&x, &folds, 0, &y, K_DUAL_MAX);
    let (res_p, tot_p) =
        fold_unit(&primal, &fold, &y_of, &[K_DUAL_MAX], None).expect("primal unit")[0];
    assert_eq!(
        (res_g.to_bits(), tot_g.to_bits()),
        (res_p.to_bits(), tot_p.to_bits())
    );
}

#[test]
fn a_training_outcome_orthogonal_to_x_is_decided_from_g() {
    // Rank-5 X, so a training outcome orthogonal to every column of X̃_tr
    // exists; ‖X̃_tr'z‖ is then rounding. The n-space unit decides it from
    // G alone: a finite contribution with the primal's ss_tot.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(61);
    let a = Mat::<f64>::from_fn(40, 5, |_, _| rng.random_range(-1.0..1.0));
    let b = Mat::<f64>::from_fn(5, 800, |_, _| rng.random_range(-1.0..1.0));
    let x = a.as_ref() * b.as_ref();
    let folds = fold_split(&(0..40).collect::<Vec<_>>(), 5);
    let fold = prepare_cv_fold(
        x.as_ref(),
        &folds,
        0,
        None,
        crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
    )
    .expect("fold");
    let n_tr = fold.train_idx.len();
    let ones = Col::<f64>::from_fn(n_tr, |_| 1.0);
    let basis = orthonormal_basis(ones.as_ref(), fold.xs_tr.as_ref(), 1e-8);
    let mut z = Col::<f64>::from_fn(n_tr, |_| rng.random_range(-1.0..1.0));
    project_off(&basis, &mut z);
    let mut y = Col::<f64>::zeros(40);
    for (i, &row) in fold.train_idx.iter().enumerate() {
        y[row] = z[i];
    }
    for &row in &fold.val_idx {
        y[row] = rng.random_range(-1.0..1.0);
    }
    let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
    let y_of = |i: usize| y[i];
    let (res_g, tot_g) = nspace_fold(&x, &folds, 0, &y, 2);
    let (res_p, tot_p) = fold_unit(&primal, &fold, &y_of, &[2], None).expect("primal unit")[0];
    assert!(
        res_g.is_finite() && res_p.is_finite(),
        "{res_g:e}, {res_p:e}"
    );
    assert_eq!(tot_g.to_bits(), tot_p.to_bits());
}

fn split_exact(x: &Mat<f64>, y: &Col<f64>, k: usize, seed: u64) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: Some(&[k]),
            weights: None,
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 50,
                n_splits: 10,
            },
            seed: Some(seed),
            ..Default::default()
        },
    )
    .expect("split_exact")
}

fn halves() -> SplitIdx {
    SplitIdx {
        tr: (0..20).collect(),
        te: (20..40).collect(),
    }
}

/// One outcome's `r` on the n-space route (from the all-row Gram matrix)
/// and on the primal one (`split_unit`).
fn both_r(x: &Mat<f64>, sp: &SplitIdx, y_of: &dyn Fn(usize) -> f64, k: usize) -> (f64, f64) {
    let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
    let prep = prepare_split(x.as_ref(), sp, None, scale.as_ref());
    let primal = split_block(ReplicateRoute::Primal, &prep, Par::Seq);
    let g = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
    let y = Col::<f64>::from_fn(x.nrows(), y_of);
    (
        split_columns_nspace(&g, sp, 1, &[k], &|_| y.clone())[0][0],
        split_unit(&primal, &prep, sp, y_of, &[k], None)[0],
    )
}

#[test]
fn split_exact_units_match_the_primal_arm() {
    let (x, y) = wide(40, 2000, 0.5, 71);
    let (_, mut rng) = resolve_seed(Some(72)).expect("seed");
    let splits = draw_splits(40, 2, 5, &mut rng).expect("splits");
    let seeds = seeds_for(20, 73);
    let cols = Columns {
        y: y.as_ref().as_mat(),
        seeds: &seeds,
        null_map: None,
    };
    // Closeness alone would pass if the n-space route returned the primal
    // arm's answer, so this also counts units whose r differs in bits from
    // the primal arm's and requires at least one such unit.
    let mut differing = 0usize;
    for (si, sp) in splits.iter().enumerate() {
        for k in 2..=K_DUAL_MAX {
            for c in 0..cols.len() {
                let yc = cols.column(c);
                let (r_g, r_p) = both_r(&x, sp, &|i: usize| yc[i], k);
                assert!(
                    (r_g - r_p).abs() <= 1e-10,
                    "split {si}, k={k}, column {c}: {r_g:e} vs {r_p:e}"
                );
                if r_g.to_bits() != r_p.to_bits() {
                    differing += 1;
                }
            }
        }
    }
    assert!(
        differing > 0,
        "no unit's r differed from the primal arm in bits: the n-space \
         kernel did not run"
    );
}

#[test]
fn split_exact_is_route_invisible_on_the_public_surface() {
    for k in 2..=K_DUAL_MAX {
        let seed = 90 + k as u64;
        let (x, y) = wide(40, 2000, 0.3, seed);
        assert_eq!(
            split_exact_refit_route(40, 2000, 50, 10, k, None, false, false),
            ReplicateRoute::Nspace
        );
        // Budget 0 runs the splits one at a time on this thread, where the
        // block counter lives; the bits do not depend on the budget.
        let before = nspace_blocks_built();
        let gram = crate::fit::with_budget(0, || split_exact(&x, &y, k, seed));
        assert_eq!(
            nspace_blocks_built(),
            before + 10,
            "k={k}: one block per split"
        );
        let primal = with_gram_routes_disabled(|| split_exact(&x, &y, k, seed));
        assert!(
            (gram.statistic.unwrap() - primal.statistic.unwrap()).abs() <= 1e-10,
            "k={k}: {:e} vs {:e}",
            gram.statistic.unwrap(),
            primal.statistic.unwrap()
        );
        assert_eq!(
            gram.pvalue.to_bits(),
            primal.pvalue.to_bits(),
            "k={k}: p-value"
        );
    }
}

/// `|a − b| ≤ 1e-12·max(|a|, |b|)`.
fn assert_rel_close_1e12(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
        "{what}: {a:e} vs {b:e}"
    );
}

/// Shapes the per-unit rule kept primal and the one-G rules send to the
/// n-space route: the public statistic is within a relative 1e-12 of the
/// call with the Gram routes taken out, the p-value the same bits.
#[test]
fn nspace_shapes_the_one_gram_rules_add_match_the_primal() {
    // raw_perm 130 x 2000, 5 folds (n_tr 104), n_perm 50, k = 2.
    assert!(!use_dual_route(104, 2000, 51, 2), "premise: per-fold rule");
    assert_eq!(
        raw_perm_route(130, 5, 2000, 50, 2, None, false),
        ReplicateRoute::Nspace
    );
    let (x, y) = wide(130, 2000, 0.3, 151);
    let gram = raw_perm(&x, &y, 2, 5, 152);
    let primal = with_gram_routes_disabled(|| raw_perm(&x, &y, 2, 5, 152));
    assert_rel_close_1e12(
        gram.statistic.unwrap(),
        primal.statistic.unwrap(),
        "raw_perm",
    );
    assert_eq!(gram.pvalue.to_bits(), primal.pvalue.to_bits(), "raw_perm p");
    // split_exact 240 x 2000 (halves of 120), n_perm 50, J = 10, k = 2.
    let (n_train, _) = crate::resample::split_sizes(240, 2);
    assert!(
        !use_dual_route(n_train, 2000, 51, 2),
        "premise: per-split rule"
    );
    assert_eq!(
        split_exact_refit_route(240, 2000, 50, 10, 2, None, false, false),
        ReplicateRoute::Nspace
    );
    let (x, y) = wide(240, 2000, 0.3, 153);
    let gram = split_exact(&x, &y, 2, 154);
    let primal = with_gram_routes_disabled(|| split_exact(&x, &y, 2, 154));
    assert_rel_close_1e12(
        gram.statistic.unwrap(),
        primal.statistic.unwrap(),
        "split_exact",
    );
    assert_eq!(
        gram.pvalue.to_bits(),
        primal.pvalue.to_bits(),
        "split_exact p"
    );
}

#[test]
fn split_exact_keeps_its_other_routes() {
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 10, 1, None, false, false),
        ReplicateRoute::Special
    );
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 10, K_DUAL_MAX + 1, None, false, false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 10, 2, Some(3), false, false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 10, 2, None, true, false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        with_gram_routes_disabled(|| split_exact_refit_route(
            40, 2000, 50, 10, 2, None, false, false
        )),
        ReplicateRoute::Primal
    );
}

#[test]
fn identical_test_rows_give_r_zero_on_both_routes() {
    // Every test-half row of X is the same row, so both routes' scores are
    // constant up to rounding, and both guards return r = 0.
    let (x0, y) = wide(40, 2000, 0.5, 111);
    let x = Mat::<f64>::from_fn(
        40,
        2000,
        |i, j| if i < 20 { x0[(i, j)] } else { x0[(20, j)] },
    );
    let (r_g, r_p) = both_r(&x, &halves(), &|i: usize| y[i], 2);
    assert_eq!(r_g.to_bits(), r_p.to_bits());
}

#[test]
fn a_run_of_columns_answers_each_column_as_its_own_unit() {
    // One batched run mixing every way a column can leave the kernel: an
    // ordinary column, a constant training half (the zero model), a
    // constant test half (the guard's degenerate y) and a training outcome
    // orthogonal to the rank-5 X̃_tr. The ordinary columns get their own
    // unit's r within the route tolerance and the degenerate ones the
    // primal's r to the bit, whatever their run-mates are; the orthogonal
    // one, decided at the rounding of `G`, a finite r.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(121);
    let a = Mat::<f64>::from_fn(40, 5, |_, _| rng.random_range(-1.0..1.0));
    let b = Mat::<f64>::from_fn(5, 800, |_, _| rng.random_range(-1.0..1.0));
    let x = a.as_ref() * b.as_ref();
    let sp = halves();
    let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
    let prep = prepare_split(x.as_ref(), &sp, None, scale.as_ref());
    let noise = Col::<f64>::from_fn(40, |_| rng.random_range(-1.0..1.0));
    let ordinary = Col::<f64>::from_fn(40, |i| x[(i, 0)] + 0.5 * noise[i]);
    let constant_train = Col::<f64>::from_fn(40, |i| if i < 20 { 2.5 } else { noise[i] });
    let constant_test = Col::<f64>::from_fn(40, |i| if i < 20 { noise[i] } else { -1.0 });
    let ones = Col::<f64>::from_fn(20, |_| 1.0);
    let basis = orthonormal_basis(ones.as_ref(), prep.xs_tr.as_ref(), 1e-8);
    let mut z = Col::<f64>::from_fn(20, |_| rng.random_range(-1.0..1.0));
    project_off(&basis, &mut z);
    let orthogonal = Col::<f64>::from_fn(40, |i| if i < 20 { z[i] } else { noise[i] });
    let cols = [
        &ordinary,
        &constant_train,
        &ordinary,
        &constant_test,
        &orthogonal,
    ];
    let g = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
    let ns = NspaceSplit::from_gram(&g, &sp);
    let column_y = |c: usize| cols[c].clone();
    let run: Vec<f64> = split_columns_nspace(&g, &sp, cols.len(), &[2], &column_y)
        .into_iter()
        .map(|r| r[0])
        .collect();
    for (j, y) in cols.iter().enumerate() {
        let (r_g, r_p) = both_r(&x, &sp, &|i: usize| y[i], 2);
        if j == 0 || j == 2 {
            assert!(
                (run[j] - r_g).abs() <= 1e-10,
                "column {j}: {:e} vs {r_g:e}",
                run[j]
            );
            assert!(
                (run[j] - r_p).abs() <= 1e-10,
                "column {j}: {:e} vs {r_p:e}",
                run[j]
            );
        } else if j == 4 {
            assert!(
                run[j].is_finite() && run[j].abs() <= 1.0,
                "column {j}: {:e}",
                run[j]
            );
        } else {
            assert_eq!(run[j].to_bits(), r_p.to_bits(), "column {j}");
            assert_eq!(r_g.to_bits(), r_p.to_bits(), "column {j}: single unit");
        }
    }
    assert_eq!(run[0].to_bits(), run[2].to_bits(), "same column, same run");
    // Premise: the ordinary column keeps both components.
    let (z, _, _) = standardize1(Col::<f64>::from_fn(20, |i| ordinary[i]).as_ref());
    assert!(matches!(
        pls1_nspace_kernel(&ns.gram.block(), z.as_ref(), 2),
        NspaceOutcome::Resolved { k_used: 2, .. }
    ));
}

/// The two n-space drivers, called directly on the inputs of
/// `raw_perm_fold_units_match_the_primal_arm` (five folds) and
/// `split_exact_units_match_the_primal_arm` (five splits), with their units
/// one at a time (budget 0) and all in one batch (no limit): the same bits.
/// At budget 0 every block is built on this thread, which shows the n-space
/// route ran.
#[test]
fn the_drivers_give_the_same_bits_at_every_budget() {
    let bits = |v: &[f64]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
    let (x, y) = wide(40, 2000, 0.5, 1);
    let folds = folds_of(40, 5, 2);
    let seeds = seeds_for(20, 3);
    let cols = Columns {
        y: y.as_ref().as_mat(),
        seeds: &seeds,
        null_map: None,
    };
    for k in 2..=K_DUAL_MAX {
        let run = |budget| {
            crate::fit::with_budget(budget, || {
                pooled_cv_r2_columns(
                    ReplicateRoute::Nspace,
                    x.as_ref(),
                    &folds,
                    None,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                    &[k],
                    None,
                    cols.len(),
                    &|c| cols.column(c),
                )
                .expect("columns")
            })
        };
        let row_bits = |m: Mat<f64>| bits(&first_row(&m));
        let before = nspace_blocks_built();
        let one = run(0);
        assert_eq!(
            nspace_blocks_built(),
            before + folds.len(),
            "k={k}: one block per fold"
        );
        assert_eq!(row_bits(one), row_bits(run(u64::MAX)), "raw_perm k={k}");
    }

    let (x, y) = wide(40, 2000, 0.5, 71);
    let (_, mut rng) = resolve_seed(Some(72)).expect("seed");
    let splits = draw_splits(40, 2, 5, &mut rng).expect("splits");
    let seeds = seeds_for(20, 73);
    let cols = Columns {
        y: y.as_ref().as_mat(),
        seeds: &seeds,
        null_map: None,
    };
    for k in 2..=K_DUAL_MAX {
        let run = |budget| {
            crate::fit::with_budget(budget, || {
                let z = split_zbars_columns(
                    ReplicateRoute::Nspace,
                    x.as_ref(),
                    &splits,
                    None,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                    &[k],
                    None,
                    cols.len(),
                    &|c| cols.column(c),
                    None,
                    None,
                );
                first_row(&z)
            })
        };
        let before = nspace_blocks_built();
        let one = run(0);
        assert_eq!(
            nspace_blocks_built(),
            before + splits.len(),
            "k={k}: one block per split"
        );
        assert_eq!(bits(&one), bits(&run(u64::MAX)), "split_exact k={k}");
    }
}

/// `|a − b| ≤ tol·max|b|` entrywise, with the largest difference in the
/// message.
fn assert_entries_close(a: MatRef<'_, f64>, b: MatRef<'_, f64>, tol: f64, what: &str) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{what}: shape"
    );
    let mut scale = 0.0_f64;
    let mut diff = 0.0_f64;
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            scale = scale.max(b[(i, j)].abs());
            diff = diff.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    assert!(
        diff <= tol * scale,
        "{what}: max |Δ| = {diff:e} against max |entry| = {scale:e}"
    );
}

fn rel_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * b.abs(), "{what}: {a:e} vs {b:e}");
}

/// `X` of `wide` with a different offset on every column, so `G` must
/// centre, and the first 8 rows shifted by 0.5 in every column, so a part
/// holding some of them has means away from the all-row means and the two
/// centrings differ.
fn uncentred(x: &Mat<f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| {
        x[(i, j)] + (j % 7) as f64 - 2.0 + if i < 8 { 0.5 } else { 0.0 }
    })
}

#[test]
fn nspace_blocks_from_gram_match_blocks_from_halves() {
    let (x0, _) = wide(40, 2000, 0.5, 131);
    let x = uncentred(&x0);
    let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
    let g = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
    let (_, mut rng) = resolve_seed(Some(132)).expect("seed");
    for (si, sp) in draw_splits(40, 2, 5, &mut rng)
        .expect("splits")
        .iter()
        .enumerate()
    {
        let halves = NspaceSplit::new(
            &prepare_split(x.as_ref(), sp, None, scale.as_ref()),
            Par::Seq,
        );
        let from_g = NspaceSplit::from_gram(&g, sp);
        let what = format!("split {si}");
        assert_entries_close(
            from_g.gram.g.as_ref(),
            halves.gram.g.as_ref(),
            1e-12,
            &format!("{what}: K_A"),
        );
        assert_entries_close(
            from_g.m.as_ref(),
            halves.m.as_ref(),
            1e-12,
            &format!("{what}: M"),
        );
        rel_close(
            from_g.gram.x_fro,
            halves.gram.x_fro,
            1e-12,
            &format!("{what}: x_fro"),
        );
        assert_eq!(from_g.gram.p, halves.gram.p, "{what}: p");
    }
    let folds = folds_of(40, 5, 133);
    for fi in 0..folds.len() {
        let fold = prepare_cv_fold(x.as_ref(), &folds, fi, None, scale.as_ref()).expect("fold");
        let halves = NspaceFold::new(&fold, Par::Seq);
        let from_g = NspaceFold::from_gram(&g, &fold.train_idx, &fold.val_idx);
        let what = format!("fold {fi}");
        assert_entries_close(
            from_g.gram.g.as_ref(),
            halves.gram.g.as_ref(),
            1e-12,
            &format!("{what}: K_A"),
        );
        assert_entries_close(
            from_g.m.as_ref(),
            halves.m.as_ref(),
            1e-12,
            &format!("{what}: M"),
        );
        rel_close(
            from_g.gram.x_fro,
            halves.gram.x_fro,
            1e-12,
            &format!("{what}: x_fro"),
        );
    }
}

#[test]
fn split_exact_k2_builds_one_gram_per_call() {
    let (x, y) = wide(40, 2000, 0.3, 141);
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 10, 2, None, false, false),
        ReplicateRoute::Nspace
    );
    // Budget 0 runs the splits one at a time on this thread, where both
    // counters live.
    let (grams, blocks) = (grams_built(), nspace_blocks_built());
    let _ = crate::fit::with_budget(0, || split_exact(&x, &y, 2, 141));
    assert_eq!(grams_built(), grams + 1, "one G per call");
    assert_eq!(nspace_blocks_built(), blocks + 10, "one block per split");
}

/// `split_nb` at `k` on the public surface, the gate on.
fn split_nb(x: &Mat<f64>, y: &Col<f64>, k: usize, seed: u64) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: Some(&[k]),
            weights: None,
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 10,
                force: false,
            },
            seed: Some(seed),
            ..Default::default()
        },
    )
    .expect("split_nb")
}

#[test]
fn raw_perm_k2_and_split_nb_k2_from_gram_match_primal() {
    for (seed, snr) in [(151_u64, 2.0), (152, 0.3)] {
        let (x0, y) = wide(40, 2000, snr, seed);
        let x = uncentred(&x0);
        // `raw_perm` at k = 2: one G, one block per fold.
        let (grams, blocks) = (grams_built(), nspace_blocks_built());
        let gram = crate::fit::with_budget(0, || raw_perm(&x, &y, 2, 5, seed));
        assert_eq!(grams_built(), grams + 1, "raw_perm: one G");
        assert_eq!(
            nspace_blocks_built(),
            blocks + 5,
            "raw_perm: one block per fold"
        );
        let primal = with_gram_routes_disabled(|| raw_perm(&x, &y, 2, 5, seed));
        rel_close(
            gram.statistic.unwrap(),
            primal.statistic.unwrap(),
            1e-12,
            "raw_perm statistic",
        );
        rel_close(gram.pvalue, primal.pvalue, 1e-12, "raw_perm p");
        // `split_nb` at k = 2: the gate's G, one block per split.
        let (grams, blocks) = (grams_built(), nspace_blocks_built());
        let gram = crate::fit::with_budget(0, || split_nb(&x, &y, 2, seed));
        assert_eq!(gram.test_method, "split_nb", "the gate must pass");
        assert_eq!(grams_built(), grams + 1, "split_nb: the gate's G only");
        assert_eq!(
            nspace_blocks_built(),
            blocks + 10,
            "split_nb: one block per split"
        );
        let primal = with_gram_routes_disabled(|| split_nb(&x, &y, 2, seed));
        assert_eq!(primal.test_method, "split_nb");
        rel_close(
            gram.statistic.unwrap(),
            primal.statistic.unwrap(),
            1e-12,
            "split_nb statistic",
        );
        rel_close(gram.pvalue, primal.pvalue, 1e-12, "split_nb p");
    }
}
