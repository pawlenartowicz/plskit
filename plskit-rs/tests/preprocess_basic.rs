//! Basic preprocess entry point tests.

#![allow(clippy::cast_precision_loss)]

use faer::{Col, Mat};
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
