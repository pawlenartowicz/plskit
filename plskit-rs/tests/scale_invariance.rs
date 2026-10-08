//! Rescaling a column of `X` or `y` / `Y` by a positive constant changes no
//! scale-free output beyond rounding, however small or large the constant.
//!
//! Standardization classifies a column as constant relative to the column's
//! own magnitude (`linalg::standardize_weighted`), not by an absolute
//! threshold. An absolute `1e-12` cutoff would leave a column rescaled by
//! `1e-13` centered but not rescaled, so `pls1_fit` would truncate to
//! `k_used = 0` and the split-half statistics would collapse to `0` with
//! `p = 1`. These outputs therefore do not depend on the units the data are
//! recorded in.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]

use faer::{Col, Mat};
use plskit::fit::{pls1_fit, FitOpts, KSpec};
use plskit::{
    pls3_confirmatory_test, pls3_fit, ConfirmatoryArgs, Pls3ConfirmatoryTestOpts, Pls3FitOpts,
};

/// Factors spanning both sides of an absolute `1e-12` threshold (a naive
/// constant-column test). Every column below has standard deviation of order
/// one, so `1e-13` puts it under that threshold and `1e-8` keeps it above.
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
    // 3e307 puts the entries of X near `f64::MAX`, where `X'Ỹ` formed from
    // the raw X overflows although every standardized quantity is of order
    // one.
    for factor in FACTORS.into_iter().chain([3e307]) {
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

/// Columns that carry no information, next to `n` rows of real data: constant
/// to rounding at several magnitudes (exact, and with relative jitter
/// `1e-15`), or with a spread below the normal range. `1e6 + 0.1` and `0.1`
/// are controls: their centering noise is small at scale 1.
fn columns_without_information(n: usize) -> Vec<(&'static str, Vec<f64>)> {
    let mut s = Stream(99);
    let jitter: Vec<f64> = (0..n).map(|_| s.next()).collect();
    vec![
        ("1e300", vec![1e300; n]),
        ("-1.7e300", vec![-1.7e300; n]),
        ("1.234567e20", vec![1.234_567e20; n]),
        ("1e6 + 0.1 (control)", vec![1e6 + 0.1; n]),
        (
            "1e300 with relative jitter 1e-15",
            jitter.iter().map(|e| 1e300 * (1.0 + 1e-15 * e)).collect(),
        ),
        ("0.1 (control)", vec![0.1; n]),
        (
            "one 5e-324",
            (0..n).map(|i| if i == 0 { 5e-324 } else { 0.0 }).collect(),
        ),
        (
            "1e-310 on half",
            (0..n)
                .map(|i| if i % 2 == 0 { 1e-310 } else { 0.0 })
                .collect(),
        ),
    ]
}

/// `x` with `c` appended as its last column.
fn with_column(x: &Mat<f64>, c: &[f64]) -> Mat<f64> {
    let p = x.ncols();
    Mat::<f64>::from_fn(
        x.nrows(),
        p + 1,
        |i, j| if j < p { x[(i, j)] } else { c[i] },
    )
}

/// A column that carries no information ([`columns_without_information`])
/// leaves `pls1_fit` as it is without it, whatever its magnitude: the same
/// `k_used`, the same coefficients on the other columns, and a zero
/// coefficient of its own. With scale 1 at every magnitude, the centering
/// noise of a constant near `1e300` would truncate the fit to `k_used = 0` or
/// overflow it to NaN coefficients, and a constant near `1e20` would move the
/// other coefficients by about 10%; a spread that underflowed would give
/// scale `0` or a subnormal scale and NaN coefficients.
#[test]
fn pls1_fit_ignores_a_column_without_information_at_any_magnitude() {
    let n = 60;
    let (x0, ymat) = linked(n, 3, 1, 11);
    let y = ymat.col(0).to_owned();
    let w = Col::<f64>::from_fn(n, |i| 0.5 + (i % 5) as f64 * 0.25);
    for weighted in [false, true] {
        let wref = weighted.then_some(w.as_ref());
        let fit = |x: &Mat<f64>| {
            pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(2),
                wref,
                FitOpts::default(),
            )
            .unwrap()
        };
        let base = fit(&x0);
        assert_eq!(base.k_used, 2);
        for (name, c) in &columns_without_information(n) {
            let m = fit(&with_column(&x0, c));
            let what = format!("weighted={weighted} {name}");
            assert_eq!(m.k_used, base.k_used, "{what}: k_used");
            for j in 0..3 {
                assert_close(
                    m.coef[j],
                    base.coef[j],
                    1e-10,
                    &format!("{what}: coef[{j}]"),
                );
            }
            assert_close(m.coef[3], 0.0, 1e-10, &format!("{what}: coef[3]"));
            assert!(m.beta[3].is_finite(), "{what}: beta[3] = {}", m.beta[3]);
            assert_close(
                m.intercept,
                base.intercept,
                1e-9,
                &format!("{what}: intercept"),
            );
        }
    }
}

/// The same columns leave `pls3_fit` as it is without them: the same
/// `k_used` and singular values, the same X saliences on the other
/// columns and a zero salience on the new one, the same Y saliences.
/// A constant near `1e300`, its jittered twin and the single
/// `5e-324` can make a naive SVD of X'Y fail to converge.
#[test]
fn pls3_fit_ignores_a_column_without_information_at_any_magnitude() {
    // `pls3_fit` takes no weights yet.
    let n = 60;
    let (x0, y) = linked(n, 3, 3, 23);
    let fit = |x: &Mat<f64>| pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default());
    let base = fit(&x0).unwrap();
    assert_eq!(base.k_used, 2);
    for (name, c) in &columns_without_information(n) {
        let m = fit(&with_column(&x0, c)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(m.k_used, base.k_used, "{name}: k_used");
        for a in 0..base.k_used {
            assert_close(
                m.singular_values[a] / base.singular_values[a],
                1.0,
                1e-10,
                &format!("{name}: singular_values[{a}]"),
            );
            for j in 0..3 {
                assert_close(
                    m.u_saliences[(j, a)],
                    base.u_saliences[(j, a)],
                    1e-10,
                    &format!("{name}: u_saliences[({j}, {a})]"),
                );
            }
            assert_close(
                m.u_saliences[(3, a)],
                0.0,
                1e-10,
                &format!("{name}: u_saliences[(3, {a})]"),
            );
            for j in 0..y.ncols() {
                assert_close(
                    m.v_saliences[(j, a)],
                    base.v_saliences[(j, a)],
                    1e-10,
                    &format!("{name}: v_saliences[({j}, {a})]"),
                );
            }
        }
    }
}
