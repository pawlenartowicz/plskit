//! Site-level tests for the n-space route at `raw_perm` and `split_exact`:
//! the units and the drivers on both routes (called directly with
//! `ReplicateRoute::Nspace` and `ReplicateRoute::Primal`), route
//! invisibility on the public surface, and degenerate folds and halves.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]

use super::*;
use crate::linalg::fold_split;
use crate::resample::Columns;
use crate::rng::{child_seeds, resolve_seed};
use crate::signal_test::{
    draw_splits, fold_block, fold_unit, pls1_confirmatory_test, prepare_cv_fold, prepare_split,
    raw_perm_route, split_block, split_exact_refit_route, split_unit, with_new_routes_disabled,
    ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput,
    ReplicateRoute, SplitIdx,
};
use crate::test_support::{orthonormal_basis, project_off};
use faer::{Col, Mat, Par};
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
            k,
            weights: None,
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
    let before = nspace_blocks_built();
    let gram = raw_perm(x, y, k, n_folds, seed);
    assert_eq!(
        nspace_blocks_built(),
        before + n_folds,
        "k={k}: the n-space route must run"
    );
    let primal = with_new_routes_disabled(|| raw_perm(x, y, k, n_folds, seed));
    assert!(
        (gram.statistic - primal.statistic).abs() <= 1e-10,
        "k={k}: {:e} vs {:e}",
        gram.statistic,
        primal.statistic
    );
    assert_eq!(
        gram.pvalue.to_bits(),
        primal.pvalue.to_bits(),
        "k={k}: p-value"
    );
}

#[test]
fn raw_perm_fold_units_match_the_primal_arm() {
    let (x, y) = wide(40, 2000, 0.5, 1);
    let folds = folds_of(40, 5, 2);
    let seeds = seeds_for(20, 3);
    let cols = Columns {
        y: y.as_ref(),
        seeds: &seeds,
    };
    // Closeness alone would pass even if the n-space route always fell back
    // to the primal arm and simply returned the primal answer, so this also
    // counts units whose ss_res differs in bits from the primal arm's and
    // requires at least one such unit: the n-space kernel must actually run.
    let mut differing = 0usize;
    for fi in 0..folds.len() {
        let fold = prepare_cv_fold(x.as_ref(), &folds, fi, None).expect("fold");
        let nspace = fold_block(ReplicateRoute::Nspace, &fold, Par::Seq);
        let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
        for k in 2..=K_DUAL_MAX {
            for c in 0..cols.len() {
                let yc = cols.column(c);
                let y_of = |i: usize| yc[i];
                let a = fold_unit(&nspace, &fold, &y_of, k, None).expect("nspace unit");
                let b = fold_unit(&primal, &fold, &y_of, k, None).expect("primal unit");
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
         route may be falling back on every unit"
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
        with_new_routes_disabled(|| raw_perm_route(40, 5, 2000, 50, 2, None, false)),
        ReplicateRoute::Primal
    );
    let (x, y) = wide(40, 2000, 0.3, 24);
    let before = nspace_blocks_built();
    let _ = raw_perm(&x, &y, 1, 5, 24);
    assert_eq!(nspace_blocks_built(), before, "k = 1 keeps the closed form");
}

#[test]
fn k_at_the_smallest_fold_ceiling_matches_the_primal() {
    // k = K_DUAL_MAX with the smallest training fold of 5 at k + 2 rows
    // (n = 6), one above rank(X̃_tr) = k + 1, on the public call.
    let k = K_DUAL_MAX;
    let n: usize = 6;
    assert_eq!(
        n - n.div_ceil(5),
        k + 2,
        "n = 6 must sit at the k + 2 fold-size boundary"
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
    let fold = prepare_cv_fold(x.as_ref(), &folds, 0, None).expect("fold");
    let nspace = fold_block(ReplicateRoute::Nspace, &fold, Par::Seq);
    let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
    let y_of = |i: usize| y[i];
    let (res_g, tot_g) = fold_unit(&nspace, &fold, &y_of, K_DUAL_MAX, None).expect("gram unit");
    let (res_p, tot_p) = fold_unit(&primal, &fold, &y_of, K_DUAL_MAX, None).expect("primal unit");
    assert_eq!(
        (res_g.to_bits(), tot_g.to_bits()),
        (res_p.to_bits(), tot_p.to_bits())
    );
}

#[test]
fn a_training_outcome_orthogonal_to_x_goes_to_the_primal_body() {
    // Rank-5 X, so a training outcome orthogonal to every column of X̃_tr
    // exists; ‖X̃_tr'z‖ is then rounding and the kernel must hand the unit
    // to the primal body, whose answer is returned to the bit.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(61);
    let a = Mat::<f64>::from_fn(40, 5, |_, _| rng.random_range(-1.0..1.0));
    let b = Mat::<f64>::from_fn(5, 800, |_, _| rng.random_range(-1.0..1.0));
    let x = a.as_ref() * b.as_ref();
    let folds = fold_split(&(0..40).collect::<Vec<_>>(), 5);
    let fold = prepare_cv_fold(x.as_ref(), &folds, 0, None).expect("fold");
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
    let nspace = fold_block(ReplicateRoute::Nspace, &fold, Par::Seq);
    let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
    let y_of = |i: usize| y[i];
    let (res_g, tot_g) = fold_unit(&nspace, &fold, &y_of, 2, None).expect("gram unit");
    let (res_p, tot_p) = fold_unit(&primal, &fold, &y_of, 2, None).expect("primal unit");
    assert_eq!(
        (res_g.to_bits(), tot_g.to_bits()),
        (res_p.to_bits(), tot_p.to_bits())
    );
}

fn split_exact(x: &Mat<f64>, y: &Col<f64>, k: usize, seed: u64) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k,
            weights: None,
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

/// `split_unit` on both routes for one outcome.
fn both_r(x: &Mat<f64>, sp: &SplitIdx, y_of: &dyn Fn(usize) -> f64, k: usize) -> (f64, f64) {
    let prep = prepare_split(x.as_ref(), sp, None);
    let nspace = split_block(ReplicateRoute::Nspace, &prep, Par::Seq);
    let primal = split_block(ReplicateRoute::Primal, &prep, Par::Seq);
    (
        split_unit(&nspace, &prep, sp, y_of, k, None),
        split_unit(&primal, &prep, sp, y_of, k, None),
    )
}

#[test]
fn split_exact_units_match_the_primal_arm() {
    let (x, y) = wide(40, 2000, 0.5, 71);
    let (_, mut rng) = resolve_seed(Some(72)).expect("seed");
    let splits = draw_splits(40, 2, 5, true, &mut rng).expect("splits");
    let seeds = seeds_for(20, 73);
    let cols = Columns {
        y: y.as_ref(),
        seeds: &seeds,
    };
    // Closeness alone would pass even if the n-space route always fell back
    // to the primal arm and simply returned the primal answer, so this also
    // counts units whose r differs in bits from the primal arm's and
    // requires at least one such unit: the n-space kernel must actually run.
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
         route may be falling back on every unit"
    );
}

#[test]
fn split_exact_is_route_invisible_on_the_public_surface() {
    for k in 2..=K_DUAL_MAX {
        let seed = 90 + k as u64;
        let (x, y) = wide(40, 2000, 0.3, seed);
        assert_eq!(
            split_exact_refit_route(40, 2000, 50, k, None, false),
            ReplicateRoute::Nspace
        );
        let before = nspace_blocks_built();
        let gram = split_exact(&x, &y, k, seed);
        assert_eq!(
            nspace_blocks_built(),
            before + 10,
            "k={k}: one block per split"
        );
        let primal = with_new_routes_disabled(|| split_exact(&x, &y, k, seed));
        assert!(
            (gram.statistic - primal.statistic).abs() <= 1e-10,
            "k={k}: {:e} vs {:e}",
            gram.statistic,
            primal.statistic
        );
        assert_eq!(
            gram.pvalue.to_bits(),
            primal.pvalue.to_bits(),
            "k={k}: p-value"
        );
    }
}

#[test]
fn split_exact_keeps_its_other_routes() {
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 1, None, false),
        ReplicateRoute::Special
    );
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, K_DUAL_MAX + 1, None, false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 2, Some(3), false),
        ReplicateRoute::Primal
    );
    assert_eq!(
        split_exact_refit_route(40, 2000, 50, 2, None, true),
        ReplicateRoute::Primal
    );
    assert_eq!(
        with_new_routes_disabled(|| split_exact_refit_route(40, 2000, 50, 2, None, false)),
        ReplicateRoute::Primal
    );
}

#[test]
fn constant_halves_give_the_primal_r_to_the_bit() {
    // A constant training half (z exactly zero, the zero model) and a
    // constant test half (the guard's degenerate y).
    let (x, noise) = wide(40, 2000, 0.0, 101);
    let sp = halves();
    let constant_train = Col::<f64>::from_fn(40, |i| if i < 20 { 2.5 } else { noise[i] });
    let constant_test = Col::<f64>::from_fn(40, |i| if i < 20 { noise[i] } else { -1.0 });
    for (what, y) in [("training", &constant_train), ("test", &constant_test)] {
        let (r_g, r_p) = both_r(&x, &sp, &|i: usize| y[i], K_DUAL_MAX);
        assert_eq!(r_g.to_bits(), r_p.to_bits(), "constant {what} half");
    }
}

#[test]
fn identical_test_rows_fall_back_through_the_score_gate() {
    // Every test-half row of X is the same row, so both routes' scores are
    // constant up to rounding: the score gate must hand the column to the
    // primal refit, whose guard then returns r = 0.
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
    // One batched run mixing every way a column can leave the kernel: a
    // resolved column, a constant training half (the zero model), a
    // constant test half (the guard's degenerate y) and a training outcome
    // orthogonal to the rank-5 X̃_tr (Unresolved, the primal body). Each
    // column must get what its own unit gives it: the decided and
    // fallback columns the primal's r to the bit, the resolved ones within
    // the route tolerance, whatever its run-mates are.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(121);
    let a = Mat::<f64>::from_fn(40, 5, |_, _| rng.random_range(-1.0..1.0));
    let b = Mat::<f64>::from_fn(5, 800, |_, _| rng.random_range(-1.0..1.0));
    let x = a.as_ref() * b.as_ref();
    let sp = halves();
    let prep = prepare_split(x.as_ref(), &sp, None);
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
    let ns = crate::dual_route::NspaceSplit::new(&prep, Par::Seq);
    let column_y = |c: usize| cols[c].clone();
    let run =
        crate::signal_test::split_columns_nspace(&prep, &sp, &ns, cols.len(), 2, true, &column_y);
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
        } else {
            assert_eq!(run[j].to_bits(), r_p.to_bits(), "column {j}");
        }
    }
    assert_eq!(run[0].to_bits(), run[2].to_bits(), "same column, same run");
    // Premises: the ordinary column resolves, the orthogonal one does not.
    let kernel = |y: &Col<f64>| {
        let (z, _, _) = standardize1(Col::<f64>::from_fn(20, |i| y[i]).as_ref());
        pls1_nspace_kernel(&ns.gram.block(), z.as_ref(), 2)
    };
    assert!(matches!(kernel(&ordinary), NspaceOutcome::Resolved { .. }));
    assert!(matches!(kernel(&orthogonal), NspaceOutcome::Unresolved));
}
