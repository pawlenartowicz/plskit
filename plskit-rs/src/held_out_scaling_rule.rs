//! The PLS1 held-out tests scale X once over all rows and centre both parts
//! with the training part's means. With `pre_standardized` X is used as
//! given and each part is only centred. Checked on wide and tall X, with and without
//! weights, for `split_exact` (both routes), `split_nb`, `e` and
//! `raw_perm` (at k = 1 and 2).
//!
//! The PLS3 held-out tests do the same for X and for Y, with
//! `pre_standardized_x` and `pre_standardized_y` per block. Checked on wide
//! and tall X for `split_exact` (dense, and with `keep_y = 2`) and
//! `split_nb`.
//!
//! The cross-validation folds of `pls1_find_k_optimal` (`r2_se`, `r2_max`)
//! and `spls1_find_keep_optimal` follow the same rule, on the same cases.
//!
//! On wide and on tall X the PLS1 held-out tests give a finite statistic on
//! degenerate columns (a constant outcome, an outcome orthogonal to X, `k`
//! above the rank of X, a column constant over some rows), `r = 0` for a
//! constant outcome, and `pls1_perm_alpha` the zero model on an outcome
//! exactly orthogonal to X̃. That these come from the Gram matrices alone,
//! with no refit on X, is shown by the crate's unit tests, which reach the
//! routes.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]

use crate::linalg::{standardize, standardize_weighted};
use crate::{
    ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, FindKOptimalOpts,
    FindKeepOptimalOpts, Pls3ConfirmatoryTestOpts, Selector, pls1_confirmatory_test,
    pls1_find_k_optimal, pls1_perm_alpha, pls3_confirmatory_test, spls1_find_keep_optimal,
};
use faer::{Col, ColRef, Mat, MatRef};

/// `SplitMix64` mapped to `[-1, 1)`: deterministic data without a dev-dependency.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }

    /// Box-Muller; `1 - u` keeps the logarithm's argument in `(0, 1]`.
    fn normal(&mut self) -> f64 {
        let u1 = f64::midpoint(self.next(), 1.0);
        let u2 = f64::midpoint(self.next(), 1.0);
        (-2.0 * (1.0 - u1).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `n × p` X with column `j` scaled by `exp(N(0, 1))` and shifted, and a
/// `y` carrying a signal on the first three columns.
fn data(n: usize, p: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
    let mut s = Stream(seed);
    let z = Mat::<f64>::from_fn(n, p, |_, _| s.normal());
    let c: Vec<f64> = (0..p).map(|_| s.normal().exp()).collect();
    let x = Mat::<f64>::from_fn(n, p, |i, j| z[(i, j)] * c[j] + 0.5 * j as f64);
    let y = Col::<f64>::from_fn(n, |i| {
        0.6 * (z[(i, 0)] + z[(i, 1)] - z[(i, 2)]) + s.normal()
    });
    (x, y)
}

fn weights(n: usize, seed: u64) -> Col<f64> {
    let mut s = Stream(seed);
    Col::<f64>::from_fn(n, |_| 1.25 + 0.75 * s.next())
}

#[derive(Clone, Copy, Debug)]
enum Method {
    /// `split_exact` at k = 1, dense: the no-refit route.
    ExactNoRefit,
    /// `split_exact` at k = 1 with `keep = p / 2`: the refit route.
    ExactKeep,
    /// `split_exact` at k = 2: the refit route.
    ExactK2,
    /// `split_nb`, forced past the gate.
    NbForce,
    E,
    /// `raw_perm` at k = 1.
    RawPermK1,
    /// `raw_perm` at k = 2.
    RawPermK2,
}

const METHODS: [Method; 7] = [
    Method::ExactNoRefit,
    Method::ExactKeep,
    Method::ExactK2,
    Method::NbForce,
    Method::E,
    Method::RawPermK1,
    Method::RawPermK2,
];

/// `(statistic, p-value)` of one call. `p` sets `keep = p / 2` for
/// [`Method::ExactKeep`], so a call on X with an added column keeps the same
/// count.
fn run(
    m: Method,
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    p: usize,
    w: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> (f64, f64) {
    let exact = ConfirmatoryArgs::SplitExact {
        n_perm: 39,
        n_splits: 8,
    };
    let (args, k, keep) = match m {
        Method::ExactNoRefit => (exact, 1, None),
        Method::ExactKeep => (exact, 1, Some(p / 2)),
        Method::ExactK2 => (exact, 2, None),
        Method::NbForce => (
            ConfirmatoryArgs::SplitNb {
                n_splits: 8,
                force: true,
            },
            1,
            None,
        ),
        Method::E => (ConfirmatoryArgs::E, 1, None),
        Method::RawPermK1 | Method::RawPermK2 => (
            ConfirmatoryArgs::RawPerm {
                n_perm: 39,
                n_folds: 5,
            },
            if matches!(m, Method::RawPermK1) { 1 } else { 2 },
            None,
        ),
    };
    let r = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x,
            y,
            k: Some(&[k]),
            weights: w,
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args,
            pre_standardized,
            seed: Some(17),
            keep,
            ..ConfirmatoryTestOpts::default()
        },
    )
    .unwrap();
    (r.statistic.unwrap(), r.pvalue)
}

/// A case name, X, y and the weights.
type Case = (String, Mat<f64>, Col<f64>, Option<Col<f64>>);

/// Wide and tall X, each without and with weights.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (shape, n, p, seed) in [("wide", 40, 120, 3), ("tall", 200, 6, 5)] {
        let (x, y) = data(n, p, seed);
        out.push((format!("{shape} unweighted"), x.clone(), y.clone(), None));
        out.push((
            format!("{shape} weighted"),
            x,
            y,
            Some(weights(n, seed + 100)),
        ));
    }
    out
}

fn assert_near(a: (f64, f64), b: (f64, f64), what: &str) {
    for (name, u, v) in [("statistic", a.0, b.0), ("p-value", a.1, b.1)] {
        assert!(
            (u - v).abs() <= 1e-12 * u.abs().max(v.abs()),
            "{what}: {name} {u:e} vs {v:e}"
        );
    }
}

fn hand_standardized(x: MatRef<'_, f64>, w: Option<ColRef<'_, f64>>) -> Mat<f64> {
    standardize_weighted(x, w).0
}

fn with_column(x: MatRef<'_, f64>, c: impl Fn(usize) -> f64) -> Mat<f64> {
    let p = x.ncols();
    Mat::<f64>::from_fn(
        x.nrows(),
        p + 1,
        |i, j| if j < p { x[(i, j)] } else { c(i) },
    )
}

#[test]
fn flag_on_hand_standardized_x_equals_call_on_raw_x() {
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xs = hand_standardized(x.as_ref(), w);
        for m in METHODS {
            let raw = run(m, x.as_ref(), y.as_ref(), x.ncols(), w, false);
            let flagged = run(m, xs.as_ref(), y.as_ref(), x.ncols(), w, true);
            assert_near(raw, flagged, &format!("{name}, {m:?}"));
        }
    }
}

#[test]
fn flag_uses_column_scale_as_given() {
    let d = [1.0, 10.0, 0.1];
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xd = Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * d[j % 3]);
        for m in [
            Method::ExactNoRefit,
            Method::E,
            Method::RawPermK1,
            Method::RawPermK2,
        ] {
            let plain = run(m, x.as_ref(), y.as_ref(), x.ncols(), w, true);
            let scaled = run(m, xd.as_ref(), y.as_ref(), x.ncols(), w, true);
            assert!(
                (plain.0 - scaled.0).abs() > 1e-6,
                "{name}, {m:?}: statistic {:e} did not move with the column scale",
                plain.0
            );
        }
    }
}

#[test]
fn flag_centres_each_part() {
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xs = hand_standardized(x.as_ref(), w);
        let shifted = Mat::<f64>::from_fn(xs.nrows(), xs.ncols(), |i, j| xs[(i, j)] + 5.0);
        for m in METHODS {
            let a = run(m, xs.as_ref(), y.as_ref(), x.ncols(), w, true);
            let b = run(m, shifted.as_ref(), y.as_ref(), x.ncols(), w, true);
            assert_near(a, b, &format!("{name}, {m:?}"));
        }
    }
}

#[test]
fn constant_column_adds_nothing() {
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xc = with_column(x.as_ref(), |_| 7.0);
        for m in METHODS {
            let a = run(m, x.as_ref(), y.as_ref(), x.ncols(), w, false);
            let b = run(m, xc.as_ref(), y.as_ref(), x.ncols(), w, false);
            assert_near(a, b, &format!("{name}, {m:?}"));
        }
    }
}

#[test]
fn column_constant_on_some_rows_stays_finite() {
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xz = with_column(x.as_ref(), |i| if i % 10 == 0 { 1.0 } else { 0.0 });
        let xs = hand_standardized(xz.as_ref(), w);
        for m in METHODS {
            let raw = run(m, xz.as_ref(), y.as_ref(), x.ncols(), w, false);
            assert!(raw.0.is_finite(), "{name}, {m:?}: statistic {}", raw.0);
            let flagged = run(m, xs.as_ref(), y.as_ref(), x.ncols(), w, true);
            assert_near(raw, flagged, &format!("{name}, {m:?}"));
        }
    }
}

/// `xs·xs'` by hand.
fn gram_of(xs: MatRef<'_, f64>) -> Mat<f64> {
    let n = xs.nrows();
    Mat::<f64>::from_fn(n, n, |i, j| {
        (0..xs.ncols()).map(|l| xs[(i, l)] * xs[(j, l)]).sum()
    })
}

/// Wide X (40 × 120) of 20 rows each taken twice (rows `2m` and `2m + 1`),
/// and an outcome antisymmetric across each pair, so orthogonal to X̃ over
/// any set of whole pairs (on every row exactly: X̃'y sums pairs `x_m·a_m`
/// and `x_m·(−a_m)`).
fn duplicated_pairs() -> (Mat<f64>, Col<f64>) {
    let (base, _) = data(20, 120, 31);
    let x = Mat::<f64>::from_fn(40, 120, |i, j| base[(i / 2, j)]);
    let mut s = Stream(32);
    let a: Vec<f64> = (0..20).map(|_| 1.0 + s.next().abs()).collect();
    let y = Col::<f64>::from_fn(40, |i| if i % 2 == 0 { a[i / 2] } else { -a[i / 2] });
    (x, y)
}

/// Wide X, every held-out method and `pls1_perm_alpha`: a constant outcome
/// gives `r = 0` (statistic 0); an outcome orthogonal to X over whole
/// pairs of duplicated rows, `k` above the rank of a rank-one X, and a
/// column constant over some rows each give a finite statistic and
/// p-value. `pls1_perm_alpha` on the duplicated rows gives the zero model
/// for the orthogonal outcome and for a constant one, and a finite fit of
/// at most one component for an outcome orthogonal to X up to rounding.
#[test]
#[allow(clippy::float_cmp)] // the zero model's statistic is exactly 0
fn degenerate_columns_give_finite_results() {
    let finite = |r: (f64, f64), what: &str| {
        assert!(
            r.0.is_finite() && r.1.is_finite() && r.1 > 0.0 && r.1 <= 1.0,
            "{what}: statistic {:e}, p-value {:e}",
            r.0,
            r.1
        );
    };
    let (x, _) = data(40, 120, 3);
    let constant = Col::<f64>::from_fn(40, |_| 2.5);
    let (xd, yd) = duplicated_pairs();
    let (rank1, y1) = {
        let (b, y) = data(40, 120, 33);
        (
            Mat::<f64>::from_fn(40, 120, |i, j| b[(i, 0)] * (1.0 + j as f64)),
            y,
        )
    };
    let (xz, yz) = {
        let (b, y) = data(40, 120, 34);
        (
            with_column(b.as_ref(), |i| if i < 20 { 1.0 } else { i as f64 }),
            y,
        )
    };
    for m in METHODS {
        let c = run(m, x.as_ref(), constant.as_ref(), 120, None, false);
        finite(c, &format!("{m:?}, constant y"));
        if !matches!(m, Method::E) {
            assert_eq!(c.0, 0.0, "{m:?}, constant y: statistic");
        }
        finite(
            run(m, xd.as_ref(), yd.as_ref(), 120, None, false),
            &format!("{m:?}, orthogonal y"),
        );
        finite(
            run(m, rank1.as_ref(), y1.as_ref(), 120, None, false),
            &format!("{m:?}, rank one"),
        );
        finite(
            run(m, xz.as_ref(), yz.as_ref(), 120, None, false),
            &format!("{m:?}, column constant on half the rows"),
        );
    }

    let alpha = |x: &Mat<f64>, y: &Col<f64>| {
        let xs = standardize(x.as_ref()).0;
        let ym = Mat::<f64>::from_fn(y.nrows(), 1, |i, _| y[i]);
        pls1_perm_alpha(gram_of(xs.as_ref()).as_ref(), ym.as_ref(), 1, x.ncols())
            .expect("perm_alpha")
    };
    for (what, y) in [("orthogonal y", &yd), ("constant y", &constant)] {
        let out = alpha(&xd, y);
        assert_eq!(out.k_used[0], 0, "perm_alpha, {what}: the zero model");
        assert!(
            (0..40).all(|i| out.alpha[(i, 0)] == 0.0),
            "perm_alpha, {what}: alpha"
        );
    }
    // Rank one: an outcome orthogonal to the one direction of X̃ up to
    // rounding.
    let xs1 = standardize(rank1.as_ref()).0;
    let u = Col::<f64>::from_fn(40, |i| xs1[(i, 0)]);
    let uu: f64 = (0..40).map(|i| u[i] * u[i]).sum();
    let mut v = Col::<f64>::from_fn(40, |i| ((i * 7) % 11) as f64 - 5.0);
    for _ in 0..2 {
        let mean = (0..40).map(|i| v[i]).sum::<f64>() / 40.0;
        let c: f64 = (0..40).map(|i| u[i] * v[i]).sum::<f64>() / uu;
        for i in 0..40 {
            v[i] -= mean + c * u[i];
        }
    }
    let out = alpha(&rank1, &v);
    assert!(out.k_used[0] <= 1, "perm_alpha, rank one");
    assert!(
        (0..40).all(|i| out.alpha[(i, 0)].is_finite()),
        "perm_alpha, rank one: alpha"
    );
}

/// Tall X (10 000 × 10), the held-out methods at shapes where the tall Gram
/// routes run (`n_perm = 399`: the k = 2 refits take the p-space Gram route,
/// `split_exact` at k = 1 scores from moments): a constant outcome gives
/// `r = 0` (statistic 0); an outcome orthogonal to X over whole pairs of
/// duplicated rows, `k = 2` on a rank-one X, and a column constant over half
/// the rows each give a finite statistic and p-value.
#[test]
#[allow(clippy::float_cmp)] // the zero model's statistic is exactly 0
fn degenerate_columns_give_finite_results_on_tall_data() {
    let (n, p) = (10_000_usize, 10_usize);
    let tall = |m: Method, x: &Mat<f64>, y: &Col<f64>| {
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 399,
            n_splits: 8,
        };
        let (args, k) = match m {
            Method::ExactNoRefit => (exact, 1),
            Method::ExactK2 => (exact, 2),
            Method::NbForce => (
                ConfirmatoryArgs::SplitNb {
                    n_splits: 8,
                    force: true,
                },
                1,
            ),
            Method::E => (ConfirmatoryArgs::E, 1),
            Method::RawPermK1 | Method::RawPermK2 => (
                ConfirmatoryArgs::RawPerm {
                    n_perm: 399,
                    n_folds: 5,
                },
                if matches!(m, Method::RawPermK1) { 1 } else { 2 },
            ),
            Method::ExactKeep => unreachable!("not run here"),
        };
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[k]),
                weights: None,
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args,
                seed: Some(19),
                ..ConfirmatoryTestOpts::default()
            },
        )
        .unwrap();
        (r.statistic.unwrap(), r.pvalue)
    };
    let finite = |r: (f64, f64), what: &str| {
        assert!(
            // `e`'s p-value underflows to 0 on this strong a signal.
            r.0.is_finite() && (0.0..=1.0).contains(&r.1),
            "{what}: statistic {:e}, p-value {:e}",
            r.0,
            r.1
        );
    };
    let (x, _) = data(n, p, 41);
    let constant = Col::<f64>::from_fn(n, |_| 2.5);
    let (xd, yd) = {
        let (base, _) = data(n / 2, p, 42);
        let mut s = Stream(43);
        let a: Vec<f64> = (0..n / 2).map(|_| 1.0 + s.next().abs()).collect();
        (
            Mat::<f64>::from_fn(n, p, |i, j| base[(i / 2, j)]),
            Col::<f64>::from_fn(n, |i| if i % 2 == 0 { a[i / 2] } else { -a[i / 2] }),
        )
    };
    let (rank1, y1) = {
        let (b, y) = data(n, p, 44);
        (
            Mat::<f64>::from_fn(n, p, |i, j| b[(i, 0)] * (1.0 + j as f64)),
            y,
        )
    };
    let (xz, yz) = {
        let (b, y) = data(n, p, 45);
        (
            with_column(b.as_ref(), |i| if i < n / 2 { 1.0 } else { i as f64 }),
            y,
        )
    };
    for m in [
        Method::ExactNoRefit,
        Method::ExactK2,
        Method::NbForce,
        Method::E,
        Method::RawPermK1,
        Method::RawPermK2,
    ] {
        let c = tall(m, &x, &constant);
        finite(c, &format!("{m:?}, constant y"));
        if !matches!(m, Method::E) {
            assert_eq!(c.0, 0.0, "{m:?}, constant y: statistic");
        }
        finite(tall(m, &xd, &yd), &format!("{m:?}, orthogonal y"));
        finite(tall(m, &rank1, &y1), &format!("{m:?}, rank one"));
        finite(
            tall(m, &xz, &yz),
            &format!("{m:?}, column constant on half the rows"),
        );
    }
}

/// `n × p` X and `n × 4` Y, every column scaled by `exp(N(0, 1))` and
/// shifted, sharing a factor through the first three columns of each.
fn pls3_data(n: usize, p: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    let q = 4;
    let mut s = Stream(seed);
    let f: Vec<f64> = (0..n).map(|_| s.normal()).collect();
    let mut block = |d: usize| {
        let c: Vec<f64> = (0..d).map(|_| s.normal().exp()).collect();
        let z = Mat::<f64>::from_fn(n, d, |i, j| {
            s.normal() + if j < 3 { 0.6 * f[i] } else { 0.0 }
        });
        Mat::<f64>::from_fn(n, d, |i, j| z[(i, j)] * c[j] - 0.5 * j as f64)
    };
    let x = block(p);
    let y = block(q);
    (x, y)
}

#[derive(Clone, Copy, Debug)]
enum Pls3Method {
    /// `split_exact`, dense: the dual route on wide X, the primal on tall.
    Exact,
    /// `split_exact` with `keep_y = 2`.
    ExactKeepY,
    /// `split_nb`, forced past the gate.
    NbForce,
}

const PLS3_METHODS: [Pls3Method; 3] = [
    Pls3Method::Exact,
    Pls3Method::ExactKeepY,
    Pls3Method::NbForce,
];

/// `(statistic, p-value)` of one call.
fn run3(
    m: Pls3Method,
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    pre_x: bool,
    pre_y: bool,
) -> (f64, f64) {
    let exact = ConfirmatoryArgs::SplitExact {
        n_perm: 39,
        n_splits: 8,
    };
    let (args, keep_y) = match m {
        Pls3Method::Exact => (exact, None),
        Pls3Method::ExactKeepY => (exact, Some(2)),
        Pls3Method::NbForce => (
            ConfirmatoryArgs::SplitNb {
                n_splits: 8,
                force: true,
            },
            None,
        ),
    };
    let r = pls3_confirmatory_test(
        x,
        y,
        1,
        Pls3ConfirmatoryTestOpts {
            args,
            pre_standardized_x: pre_x,
            pre_standardized_y: pre_y,
            keep_y,
            seed: Some(17),
            ..Pls3ConfirmatoryTestOpts::default()
        },
    )
    .unwrap();
    (r.statistic.unwrap(), r.pvalue)
}

/// Wide X (the dual route for `split_exact`) and tall X (the primal).
fn pls3_cases() -> Vec<(&'static str, Mat<f64>, Mat<f64>)> {
    [("wide", 40, 120, 7), ("tall", 200, 6, 9)]
        .into_iter()
        .map(|(name, n, p, seed)| {
            let (x, y) = pls3_data(n, p, seed);
            (name, x, y)
        })
        .collect()
}

fn rescaled(a: MatRef<'_, f64>) -> Mat<f64> {
    let d = [1.0, 10.0, 0.1];
    Mat::<f64>::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] * d[j % 3])
}

#[test]
fn pls3_flags_on_hand_standardized_blocks_equal_call_on_raw_blocks() {
    for (name, x, y) in pls3_cases() {
        let xs = hand_standardized(x.as_ref(), None);
        let ys = hand_standardized(y.as_ref(), None);
        for m in PLS3_METHODS {
            let raw = run3(m, x.as_ref(), y.as_ref(), false, false);
            let flagged = run3(m, xs.as_ref(), ys.as_ref(), true, true);
            assert_near(raw, flagged, &format!("{name}, {m:?}"));
        }
    }
}

#[test]
fn pls3_flag_x_uses_column_scale_as_given() {
    for (name, x, y) in pls3_cases() {
        let xd = rescaled(x.as_ref());
        for m in PLS3_METHODS {
            let plain = run3(m, x.as_ref(), y.as_ref(), true, false);
            let scaled = run3(m, xd.as_ref(), y.as_ref(), true, false);
            assert!(
                (plain.0 - scaled.0).abs() > 1e-6,
                "{name}, {m:?}: statistic {:e} did not move with X's column scale",
                plain.0
            );
        }
    }
}

#[test]
fn pls3_flag_y_uses_column_scale_as_given() {
    for (name, x, y) in pls3_cases() {
        let yd = rescaled(y.as_ref());
        for m in PLS3_METHODS {
            let plain = run3(m, x.as_ref(), y.as_ref(), false, true);
            let scaled = run3(m, x.as_ref(), yd.as_ref(), false, true);
            assert!(
                (plain.0 - scaled.0).abs() > 1e-6,
                "{name}, {m:?}: statistic {:e} did not move with Y's column scale",
                plain.0
            );
        }
    }
}

#[test]
fn pls3_flags_centre_each_part() {
    for (name, x, y) in pls3_cases() {
        let xs = hand_standardized(x.as_ref(), None);
        let ys = hand_standardized(y.as_ref(), None);
        let xs5 = Mat::<f64>::from_fn(xs.nrows(), xs.ncols(), |i, j| xs[(i, j)] + 5.0);
        let ys3 = Mat::<f64>::from_fn(ys.nrows(), ys.ncols(), |i, j| ys[(i, j)] - 3.0);
        for m in PLS3_METHODS {
            let a = run3(m, xs.as_ref(), ys.as_ref(), true, true);
            let b = run3(m, xs5.as_ref(), ys3.as_ref(), true, true);
            assert_near(a, b, &format!("{name}, {m:?}"));
        }
    }
}

/// `(k_star, cv_scores in key order)` of `pls1_find_k_optimal` to
/// `k_max = 3`.
fn find_k(
    selector: Selector,
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> (usize, Vec<f64>) {
    let r = pls1_find_k_optimal(
        x,
        y,
        3,
        w,
        None,
        FindKOptimalOpts {
            selector,
            pre_standardized,
            seed: Some(17),
            ..FindKOptimalOpts::default()
        },
    )
    .unwrap();
    (r.k_star, r.cv_scores.unwrap().into_values().collect())
}

/// `(position of keep_star in the keep grid, cv_scores in grid order)` of
/// `spls1_find_keep_optimal` at `k = 2`: positions, so that X with an added
/// column (whose grid ends at `p + 1`) compares.
fn find_keep(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
) -> (usize, Vec<f64>) {
    let r = spls1_find_keep_optimal(
        x,
        y,
        2,
        w,
        None,
        FindKeepOptimalOpts {
            seed: Some(17),
            ..FindKeepOptimalOpts::default()
        },
    )
    .unwrap();
    let pos = r.keep_grid.iter().position(|&v| v == r.keep_star).unwrap();
    (pos, r.cv_scores.into_values().collect())
}

const SELECTORS: [Selector; 2] = [Selector::R2Se, Selector::R2Max];

fn assert_scores_near(a: &(usize, Vec<f64>), b: &(usize, Vec<f64>), what: &str) {
    assert_eq!(a.0, b.0, "{what}: pick");
    assert_eq!(a.1.len(), b.1.len(), "{what}: score count");
    for (i, (&u, &v)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(
            (u - v).abs() <= 1e-12 * u.abs().max(v.abs()),
            "{what}: cv_score {i} {u:e} vs {v:e}"
        );
    }
}

#[test]
fn find_k_flag_on_hand_standardized_x_equals_call_on_raw_x() {
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xs = hand_standardized(x.as_ref(), w);
        for sel in SELECTORS {
            let raw = find_k(sel, x.as_ref(), y.as_ref(), w, false);
            let flagged = find_k(sel, xs.as_ref(), y.as_ref(), w, true);
            assert_scores_near(&raw, &flagged, &format!("{name}, {sel:?}"));
        }
    }
}

#[test]
fn find_k_flag_uses_column_scale_as_given() {
    let d = [1.0, 10.0, 0.1];
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xd = Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * d[j % 3]);
        for sel in SELECTORS {
            let plain = find_k(sel, x.as_ref(), y.as_ref(), w, true);
            let scaled = find_k(sel, xd.as_ref(), y.as_ref(), w, true);
            assert!(
                plain
                    .1
                    .iter()
                    .zip(&scaled.1)
                    .any(|(u, v)| (u - v).abs() > 1e-6),
                "{name}, {sel:?}: cv_scores {:?} did not move with the column scale",
                plain.1
            );
        }
    }
}

#[test]
fn find_k_constant_column_adds_nothing() {
    for (name, x, y, w) in cases() {
        let w = w.as_ref().map(Col::as_ref);
        let xc = with_column(x.as_ref(), |_| 7.0);
        for sel in SELECTORS {
            let a = find_k(sel, x.as_ref(), y.as_ref(), w, false);
            let b = find_k(sel, xc.as_ref(), y.as_ref(), w, false);
            assert_scores_near(&a, &b, &format!("{name}, {sel:?}"));
        }
        let a = find_keep(x.as_ref(), y.as_ref(), w);
        let b = find_keep(xc.as_ref(), y.as_ref(), w);
        assert_scores_near(&a, &b, &format!("{name}, keep"));
    }
}
