//! Kish effective sample size tests.

#![allow(clippy::cast_precision_loss)]

use faer::Col;
use plskit::linalg::compute_n_eff;

#[test]
fn n_eff_uniform_equals_n() {
    for n in [5, 50, 500] {
        let w = Col::<f64>::from_fn(n, |_| 1.0);
        let n_eff = compute_n_eff(w.as_ref());
        assert!((n_eff - n as f64).abs() < 1e-12);
    }
}

#[test]
fn n_eff_single_row_concentration_equals_one() {
    let w = Col::<f64>::from_fn(100, |i| if i == 0 { 1.0 } else { 0.0 });
    assert!((compute_n_eff(w.as_ref()) - 1.0).abs() < 1e-12);
}

#[test]
fn n_eff_one_two_three_kish() {
    let w = Col::<f64>::from_fn(3, |i| (i + 1) as f64);
    assert!((compute_n_eff(w.as_ref()) - 36.0 / 14.0).abs() < 1e-12);
}
