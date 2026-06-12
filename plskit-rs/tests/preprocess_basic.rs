//! Basic preprocess entry point tests.

#![allow(clippy::cast_precision_loss)]

use approx::assert_relative_eq;
use faer::{Col, Mat};
use plskit::linalg::standardize_weighted;
use plskit::preprocess::{preprocess, PreprocessInput};

fn small() -> (Mat<f64>, Col<f64>, Col<f64>) {
    let x = Mat::from_fn(5, 3, |i, j| (i + j) as f64);
    let y = Col::<f64>::from_fn(5, |i| i as f64);
    let w = Col::<f64>::from_fn(5, |i| (i + 1) as f64);
    (x, y, w)
}

#[test]
fn empty_input_returns_all_none() {
    let r = preprocess(PreprocessInput {
        x: None,
        y: None,
        weights: None,
    })
    .unwrap();
    assert!(r.x_std.is_none());
    assert!(r.y_std.is_none());
    assert!(r.weights_normalized.is_none());
}

#[test]
fn x_only_populates_x_fields() {
    let (x, _, _) = small();
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: None,
        weights: None,
    })
    .unwrap();
    let (xs, m, s) = r.x_std.unwrap();
    assert_eq!(xs.nrows(), 5);
    assert_eq!(m.nrows(), 3);
    assert_eq!(s.nrows(), 3);
    assert!(r.y_std.is_none());
    assert!(r.weights_normalized.is_none());
}

#[test]
fn x_y_weights_all_populated() {
    let (x, y, w) = small();
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: Some(y.as_ref()),
        weights: Some(w.as_ref()),
    })
    .unwrap();
    assert!(r.x_std.is_some());
    assert!(r.y_std.is_some());
    let wn = r.weights_normalized.unwrap();
    let sum: f64 = (0..wn.nrows()).map(|i| wn[i]).sum();
    assert!((sum - 5.0).abs() < 1e-12);
}

#[test]
fn weights_only_skips_shape_check() {
    let (_, _, w) = small();
    let r = preprocess(PreprocessInput {
        x: None,
        y: None,
        weights: Some(w.as_ref()),
    })
    .unwrap();
    assert!(r.weights_normalized.is_some());
}

#[test]
fn x_y_shape_mismatch_errors() {
    let (x, _, _) = small();
    let y_bad = Col::<f64>::from_fn(3, |_| 0.0); // length 3, X has 5 rows
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: Some(y_bad.as_ref()),
        weights: None,
    });
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// Numeric value assertions — fixture: n=4, d=2 cols, w_raw=[2,1,1,1]
//
// Convention (mirrors standardize_weighted source):
//   w'_i = w_i · n / Σw  (normalized so Σw' = n)
//   μ_j  = Σ(w'_i · x_{ij}) / n
//   var_j = Σ(w'_i · (x_{ij} − μ_j)²) / n   (population, ddof=0)
//   scale_j = √var_j  (or 1 if √var ≤ 1e-12)
// ---------------------------------------------------------------------------

/// Small fixture with non-uniform weights so weighted stats differ from unweighted.
/// X columns are [1,2,3,4] and [10,20,30,40]; y = [1,2,3,4]; `w_raw` = [2,1,1,1].
fn numeric_fixture() -> (Mat<f64>, Col<f64>, Col<f64>) {
    let x = Mat::from_fn(4, 2, |i, j| {
        (i + 1) as f64 * if j == 0 { 1.0 } else { 10.0 }
    });
    let y = Col::<f64>::from_fn(4, |i| (i + 1) as f64);
    let w = Col::<f64>::from_fn(4, |i| if i == 0 { 2.0 } else { 1.0 });
    (x, y, w)
}

/// Hand-compute expected weighted mean and scale for col 0 of the fixture.
/// `w_raw`=[2,1,1,1], Σw=5, n=4 → w'=[1.6, 0.8, 0.8, 0.8]
/// μ = (1.6·1 + 0.8·2 + 0.8·3 + 0.8·4) / 4 = 8.8/4 = 2.2
/// var = (1.6·(1-2.2)² + 0.8·(2-2.2)² + 0.8·(3-2.2)² + 0.8·(4-2.2)²) / 4 = 5.44/4 = 1.36
/// → scale = √1.36 ≈ 1.16619
fn expected_col0() -> (f64, f64) {
    let n = 4.0_f64;
    let sum_w = 5.0_f64;
    let w_prime: [f64; 4] = [
        2.0 * n / sum_w,
        1.0 * n / sum_w,
        1.0 * n / sum_w,
        1.0 * n / sum_w,
    ];
    let x: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let mean: f64 = w_prime
        .iter()
        .zip(x.iter())
        .map(|(w, v)| w * v)
        .sum::<f64>()
        / n;
    let var: f64 = w_prime
        .iter()
        .zip(x.iter())
        .map(|(w, v)| w * (v - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

#[test]
fn x_weighted_col_means_are_zero() {
    // After weighted standardization, Σ(w'_i · xs_{ij}) / n = 0 for each column j.
    let (x, _, w) = numeric_fixture();
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: None,
        weights: Some(w.as_ref()),
    })
    .unwrap();
    let (xs, _, _) = r.x_std.unwrap();
    let wn = r.weights_normalized.unwrap();
    let n = xs.nrows() as f64;
    for j in 0..xs.ncols() {
        let wmean: f64 = (0..xs.nrows()).map(|i| wn[i] * xs[(i, j)]).sum::<f64>() / n;
        assert_relative_eq!(wmean, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn x_scale_matches_hand_computed_weighted_population_std() {
    let (x, _, w) = numeric_fixture();
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: None,
        weights: Some(w.as_ref()),
    })
    .unwrap();
    let (_, _, scale) = r.x_std.unwrap();
    let (_, expected_scale0) = expected_col0();
    // Col 1 is 10 × col 0, so its std is 10 × col-0 std.
    assert_relative_eq!(scale[0], expected_scale0, epsilon = 1e-12);
    assert_relative_eq!(scale[1], 10.0 * expected_scale0, epsilon = 1e-12);
}

#[test]
fn y_std_matches_hand_computed_weighted_population_std() {
    // y = [1,2,3,4] is identical to col 0 of X, so expected (mean, scale) are the same.
    let (_, y, w) = numeric_fixture();
    let r = preprocess(PreprocessInput {
        x: None,
        y: Some(y.as_ref()),
        weights: Some(w.as_ref()),
    })
    .unwrap();
    let (_, mean_y, scale_y) = r.y_std.unwrap();
    let (expected_mean, expected_scale) = expected_col0();
    assert_relative_eq!(mean_y, expected_mean, epsilon = 1e-12);
    assert_relative_eq!(scale_y, expected_scale, epsilon = 1e-12);
}

#[test]
fn x_std_agrees_with_linalg_standardize_weighted() {
    // preprocess delegates to standardize_weighted; cross-check at 1e-15.
    let (x, _, w) = numeric_fixture();
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: None,
        weights: Some(w.as_ref()),
    })
    .unwrap();
    let (xs_pre, mean_pre, scale_pre) = r.x_std.unwrap();
    // Reproduce what preprocess does: normalize weights, then call standardize_weighted.
    let wn = r.weights_normalized.unwrap();
    let (xs_la, mean_la, scale_la) = standardize_weighted(x.as_ref(), Some(wn.as_ref()));
    for j in 0..x.ncols() {
        assert_relative_eq!(mean_pre[j], mean_la[j], epsilon = 1e-15);
        assert_relative_eq!(scale_pre[j], scale_la[j], epsilon = 1e-15);
        for i in 0..x.nrows() {
            assert_relative_eq!(xs_pre[(i, j)], xs_la[(i, j)], epsilon = 1e-15);
        }
    }
}
