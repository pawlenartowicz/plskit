//! Rescaling a column of `X` or `y` / `Y` by a positive constant changes no
//! scale-free output beyond rounding, however small or large the constant.
//!
//! Standardization used to classify a column as constant whenever its
//! standard deviation was at most an absolute `1e-12`, so a column rescaled
//! by `1e-13` was centered but not rescaled: `pls1_fit` then truncated to
//! `k_used = 0` and the split-half statistics collapsed to `0` with `p = 1`.
//! The classification is now relative to the column's own magnitude
//! (`linalg::standardize_weighted`), so these outputs do not depend on the
//! units the data are recorded in.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]

use faer::{Col, Mat};
use plskit::fit::{pls1_fit, FitOpts, KSpec};
use plskit::{
    pls1_confirmatory_test, pls3_confirmatory_test, pls3_fit, ConfirmatoryArgs,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, Pls3ConfirmatoryTestOpts, Pls3FitOpts,
};

/// Factors spanning both sides of the old absolute `1e-12` floor. Every
/// column below has standard deviation of order one, so `1e-13` puts it
/// under the old floor and `1e-8` keeps it above.
const FACTORS: [f64; 4] = [1e-13, 1e-8, 1e8, 1e13];

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
}

/// `(X, Y)` sharing one latent factor through the first two columns of `X`
/// and every column of `Y`, plus independent noise.
fn linked(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    let mut s = Stream(seed);
    let f: Vec<f64> = (0..n).map(|_| s.next()).collect();
    let x = Mat::<f64>::from_fn(n, p, |i, j| {
        let e = s.next();
        if j < 2 {
            2.0 * f[i] + e
        } else {
            e
        }
    });
    let y = Mat::<f64>::from_fn(n, q, |i, _| 1.5 * f[i] + s.next());
    (x, y)
}

fn scaled(m: &Mat<f64>, factor: f64) -> Mat<f64> {
    Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * factor)
}

fn scaled_col(c: &Col<f64>, factor: f64) -> Col<f64> {
    Col::<f64>::from_fn(c.nrows(), |i| c[i] * factor)
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a:e} vs {b:e}");
}

#[test]
fn pls1_fit_is_invariant_to_the_scale_of_x_and_y() {
    let (x, ymat) = linked(60, 6, 1, 11);
    let y = ymat.col(0).to_owned();
    let w = Col::<f64>::from_fn(60, |i| 0.5 + (i % 5) as f64 * 0.25);
    for weighted in [false, true] {
        let wref = weighted.then_some(w.as_ref());
        let fit = |x: &Mat<f64>, y: &Col<f64>| {
            pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(3),
                wref,
                FitOpts::default(),
            )
            .unwrap()
        };
        let base = fit(&x, &y);
        assert_eq!(base.k_used, 3);
        for factor in FACTORS {
            for (side, m) in [
                ("X", fit(&scaled(&x, factor), &y)),
                ("y", fit(&x, &scaled_col(&y, factor))),
            ] {
                let what = format!("weighted={weighted} {side}×{factor:e}");
                assert_eq!(m.k_used, base.k_used, "{what}: k_used");
                for j in 0..x.ncols() {
                    assert_close(
                        m.coef[j],
                        base.coef[j],
                        1e-10,
                        &format!("{what}: coef[{j}]"),
                    );
                }
                for i in 0..x.nrows() {
                    for a in 0..base.k_used {
                        assert_close(
                            m.t_scores[(i, a)],
                            base.t_scores[(i, a)],
                            1e-10,
                            &format!("{what}: t_scores[({i}, {a})]"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pls1_confirmatory_test_is_invariant_to_the_scale_of_x_and_y() {
    let (x, ymat) = linked(60, 6, 1, 13);
    let y = ymat.col(0).to_owned();
    // k = 1 takes split_exact's no-refit route, k = 2 the refit route.
    for k in [1_usize, 2] {
        let run = |x: &Mat<f64>, y: &Col<f64>| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 49,
                        n_splits: 6,
                    },
                    seed: Some(7),
                    ..ConfirmatoryTestOpts::default()
                },
            )
            .unwrap()
        };
        let base = run(&x, &y);
        assert!(
            base.statistic > 0.3,
            "k={k}: statistic = {}",
            base.statistic
        );
        for factor in FACTORS {
            for (side, r) in [
                ("X", run(&scaled(&x, factor), &y)),
                ("y", run(&x, &scaled_col(&y, factor))),
            ] {
                let what = format!("k={k} {side}×{factor:e}");
                assert_close(
                    r.statistic,
                    base.statistic,
                    1e-10,
                    &format!("{what}: statistic"),
                );
                assert_eq!(
                    r.pvalue.to_bits(),
                    base.pvalue.to_bits(),
                    "{what}: pvalue {} vs {}",
                    r.pvalue,
                    base.pvalue
                );
            }
        }
    }
}

#[test]
fn pls3_confirmatory_test_is_invariant_to_the_scale_of_x_and_y() {
    let (x, y) = linked(60, 6, 3, 17);
    let run = |x: &Mat<f64>, y: &Mat<f64>| {
        pls3_confirmatory_test(
            x.as_ref(),
            y.as_ref(),
            1,
            Pls3ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 49,
                    n_splits: 6,
                },
                seed: Some(7),
                ..Pls3ConfirmatoryTestOpts::default()
            },
        )
        .unwrap()
    };
    let base = run(&x, &y);
    assert!(base.statistic > 0.3, "statistic = {}", base.statistic);
    for factor in FACTORS {
        for (side, r) in [
            ("X", run(&scaled(&x, factor), &y)),
            ("Y", run(&x, &scaled(&y, factor))),
        ] {
            let what = format!("{side}×{factor:e}");
            assert_close(
                r.statistic,
                base.statistic,
                1e-10,
                &format!("{what}: statistic"),
            );
            assert_eq!(
                r.pvalue.to_bits(),
                base.pvalue.to_bits(),
                "{what}: pvalue {} vs {}",
                r.pvalue,
                base.pvalue
            );
        }
    }
}

#[test]
fn pls3_fit_is_invariant_to_the_scale_of_x_and_y() {
    let (x, y) = linked(50, 6, 3, 19);
    let fit = |x: &Mat<f64>, y: &Mat<f64>| {
        pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap()
    };
    let base = fit(&x, &y);
    assert_eq!(base.k_used, 2);
    for factor in FACTORS {
        for (side, m) in [
            ("X", fit(&scaled(&x, factor), &y)),
            ("Y", fit(&x, &scaled(&y, factor))),
        ] {
            let what = format!("{side}×{factor:e}");
            assert_eq!(m.k_used, base.k_used, "{what}: k_used");
            for a in 0..base.k_used {
                let sv = base.singular_values[a];
                assert_close(
                    m.singular_values[a] / sv,
                    1.0,
                    1e-10,
                    &format!("{what}: singular_values[{a}]"),
                );
                for j in 0..x.ncols() {
                    assert_close(
                        m.u_saliences[(j, a)],
                        base.u_saliences[(j, a)],
                        1e-10,
                        &format!("{what}: u_saliences[({j}, {a})]"),
                    );
                }
            }
        }
    }
}
