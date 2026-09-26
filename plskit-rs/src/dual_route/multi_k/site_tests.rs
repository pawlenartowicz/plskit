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
use crate::error::PlsKitError;
use crate::linalg::fold_split;
use crate::resample::Columns;
use crate::rng::{child_seeds, resolve_seed};
use crate::signal_test::{
    draw_splits, fold_block, fold_unit, pls1_confirmatory_test, pooled_cv_r2_columns,
    prepare_cv_fold, prepare_split, raw_perm_route, split_block, split_exact_refit_route,
    split_unit, split_zbars_columns, with_new_routes_disabled, ConfirmatoryArgs,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput, ReplicateRoute, SplitIdx,
};
use crate::test_support::{orthonormal_basis, project_off};
use faer::{Col, ColRef, Mat, MatRef, Par};
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

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().copied().map(f64::to_bits).collect()
}

/// The `raw_perm` driver on `route`, for the observed `y` and one null
/// column per seed.
fn raw_columns(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    folds: &[Vec<usize>],
    seeds: &[u64],
    k: usize,
    disable_parallelism: bool,
) -> Vec<f64> {
    let cols = Columns { y, seeds };
    pooled_cv_r2_columns(
        route,
        x,
        folds,
        None,
        k,
        None,
        cols.len(),
        disable_parallelism,
        &|c: usize| cols.column(c),
    )
    .expect("pooled columns")
}

/// Exceedance counts as the runners form them (a non-finite null counts),
/// compared unless some primal null lies within `tol` of the observed
/// value, where `>=` may resolve either way.
fn assert_same_exceedances(gram: &[f64], primal: &[f64], tol: f64, what: &str) {
    let near_tie = primal[1..].iter().any(|v| (v - primal[0]).abs() <= tol);
    if near_tie {
        return;
    }
    let count = |c: &[f64]| {
        c[1..]
            .iter()
            .filter(|v| !v.is_finite() || **v >= c[0])
            .count()
    };
    assert_eq!(count(gram), count(primal), "{what}: exceedances");
}

/// Route-invisible statistics: `1e-10` absolute per column.
fn assert_columns_close(gram: &[f64], primal: &[f64], what: &str) {
    for (c, (a, b)) in gram.iter().zip(primal).enumerate() {
        assert!(
            a.is_nan() == b.is_nan() && (a.is_nan() || (a - b).abs() <= 1e-10),
            "{what}, column {c}: {a:e} vs {b:e}"
        );
    }
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

/// Like [`raw_perm`], but returns the `Result` instead of unwrapping: use
/// this to assert on an input the public call is expected to reject.
fn raw_perm_checked(
    x: &Mat<f64>,
    y: &Col<f64>,
    k: usize,
    n_folds: usize,
    seed: u64,
) -> Result<ConfirmatoryTestOutput, PlsKitError> {
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
fn raw_perm_columns_match_the_primal_and_count_the_same_exceedances() {
    for k in 2..=K_DUAL_MAX {
        let seed = 10 + k as u64;
        let (x, y) = wide(40, 2000, 0.3, seed);
        let folds = folds_of(40, 5, seed + 1);
        let seeds = seeds_for(50, seed + 2);
        let run = |route, dp| raw_columns(route, x.as_ref(), y.as_ref(), &folds, &seeds, k, dp);
        let gram = run(ReplicateRoute::Nspace, false);
        let primal = run(ReplicateRoute::Primal, false);
        assert_columns_close(&gram, &primal, &format!("k={k}"));
        assert_same_exceedances(&gram, &primal, 1e-10, &format!("k={k}"));
        assert_eq!(
            bits(&run(ReplicateRoute::Nspace, true)),
            bits(&gram),
            "k={k}: serial vs parallel"
        );
    }
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
fn leave_one_out_raw_perm_matches_the_primal() {
    // One validation row per fold, M is 1 × 19 over p = 4000, so
    // n_tr·p = 76 000 ≥ 65 536.
    //
    // The pooled statistic is degenerate here: with a one-row validation
    // fold, ss_tot is always 0, so every pooled column is 0 and p = 1 on
    // both routes regardless of whether the n-space kernel actually ran.
    // So this test does not compare the pooled columns; instead, as
    // `raw_perm_fold_units_match_the_primal_arm` does, it compares the
    // per-(fold, column) `fold_unit` `ss_res` between the Nspace and
    // Primal routes directly and requires at least one unit whose ss_res
    // differs in bits, so a route that silently fell back to the primal
    // on every unit would fail this test.
    let (x, y) = wide(20, 4000, 0.5, 31);
    let folds = folds_of(20, 20, 32);
    let seeds = seeds_for(50, 33);
    let cols = Columns {
        y: y.as_ref(),
        seeds: &seeds,
    };
    let mut differing = 0usize;
    for fi in 0..folds.len() {
        let fold = prepare_cv_fold(x.as_ref(), &folds, fi, None).expect("fold");
        let nspace = fold_block(ReplicateRoute::Nspace, &fold, Par::Seq);
        let primal = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
        for c in 0..cols.len() {
            let yc = cols.column(c);
            let y_of = |i: usize| yc[i];
            let a = fold_unit(&nspace, &fold, &y_of, 2, None).expect("nspace unit");
            let b = fold_unit(&primal, &fold, &y_of, 2, None).expect("primal unit");
            assert_fold_contribution_close(a, b, &format!("fold {fi}, column {c}"));
            if a.0.to_bits() != b.0.to_bits() {
                differing += 1;
            }
        }
    }
    assert!(
        differing > 0,
        "no unit's ss_res differed from the primal arm in bits: the n-space \
         route may be falling back on every unit"
    );
    // n_folds == n is leave-one-out: `pls1_confirmatory_test` rejects it at
    // the public boundary (raw_perm's pooled CV R² is undefined with
    // one-row validation folds), so there is no public-dispatch route to
    // compare here the way `assert_public_raw_perm_invisible` does for the
    // other site tests. Assert the rejection instead.
    let r = raw_perm_checked(&x, &y, 2, 20, 34);
    assert!(
        matches!(r, Err(PlsKitError::InvalidArgument(_))),
        "expected InvalidArgument for n_folds == n, got {r:?}"
    );
}

#[test]
fn k_at_the_smallest_fold_ceiling_matches_the_primal() {
    // k = K_DUAL_MAX, and n chosen so the smallest training fold of 5
    // holds k + 2 rows, one above rank(X̃_tr) = k + 1: near exhaustion
    // on every replicate.
    let k = K_DUAL_MAX;
    // `5..=4*(k+2)` suffices: at n = 5, n - ceil(n/5) = 4, and each unit
    // increase in n raises n - ceil(n/5) by at most 1, so by n = 4*(k+2) it
    // has reached at least k + 2. This is the smallest such n, so every one
    // of the 5 training folds sits at exactly k + 2 rows (the tightest
    // ceiling this test can exercise); it is also n_folds == n, leave-
    // one-out, which the public boundary rejects but the internal
    // `raw_columns` Nspace-vs-Primal comparison below does not go through
    // (it calls `prepare_cv_fold`/`fold_unit` directly), so it stays at
    // n = 5 here.
    let n = (5_usize..=4 * (k + 2))
        .find(|&n| n - n.div_ceil(5) == k + 2)
        .expect("some n gives the smallest fold k + 2 rows");
    let (x, y) = wide(n, 1500, 0.5, 41);
    let folds = folds_of(n, 5, 42);
    let seeds = seeds_for(50, 43);
    let gram = raw_columns(
        ReplicateRoute::Nspace,
        x.as_ref(),
        y.as_ref(),
        &folds,
        &seeds,
        k,
        false,
    );
    let primal = raw_columns(
        ReplicateRoute::Primal,
        x.as_ref(),
        y.as_ref(),
        &folds,
        &seeds,
        k,
        false,
    );
    assert_columns_close(&gram, &primal, "k at the ceiling");
    assert_same_exceedances(&gram, &primal, 1e-10, "k at the ceiling");

    // The trailing public-dispatch check needs `n_folds < n`, which n = 5
    // (n_folds = 5) cannot give: that combination is leave-one-out, which
    // `pls1_confirmatory_test` rejects. n = 6 still lands one row past the
    // same fold-size boundary (6 - ceil(6/5) = 4 = k + 2) without being
    // leave-one-out, so use a fresh dataset at n = 6 for this check only;
    // the Nspace-vs-Primal comparison above, which is what actually
    // exercises the tightest ceiling, is unaffected.
    let n_pub: usize = 6;
    assert_eq!(
        n_pub - n_pub.div_ceil(5),
        k + 2,
        "n = 6 must still sit at the k + 2 fold-size boundary"
    );
    let (x_pub, y_pub) = wide(n_pub, 1500, 0.5, 45);
    assert_public_raw_perm_invisible(&x_pub, &y_pub, k, 5, 44);
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

/// The `split_exact` driver on `route`.
fn split_columns(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    splits: &[SplitIdx],
    seeds: &[u64],
    k: usize,
    disable_parallelism: bool,
) -> Vec<f64> {
    let cols = Columns { y, seeds };
    split_zbars_columns(
        route,
        x,
        splits,
        None,
        k,
        None,
        cols.len(),
        disable_parallelism,
        &|c: usize| cols.column(c),
    )
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
fn split_exact_columns_match_the_primal_and_count_the_same_exceedances() {
    for k in 2..=K_DUAL_MAX {
        let seed = 80 + k as u64;
        let (x, y) = wide(40, 2000, 0.3, seed);
        let (_, mut rng) = resolve_seed(Some(seed + 1)).expect("seed");
        let splits = draw_splits(40, k, 10, true, &mut rng).expect("splits");
        let seeds = seeds_for(50, seed + 2);
        let run = |route, dp| split_columns(route, x.as_ref(), y.as_ref(), &splits, &seeds, k, dp);
        let gram = run(ReplicateRoute::Nspace, false);
        let primal = run(ReplicateRoute::Primal, false);
        assert_columns_close(&gram, &primal, &format!("k={k}"));
        assert_same_exceedances(&gram, &primal, 1e-10, &format!("k={k}"));
        assert_eq!(
            bits(&run(ReplicateRoute::Nspace, true)),
            bits(&gram),
            "k={k}: serial vs parallel"
        );
    }
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
