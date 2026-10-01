//! Bit-near parity test against reference values stored in
//! `varimax_reference_seed42.json`: the rotation matrix R and the sweep values
//! from a Kaiser-normalized varimax (Kaiser 1958, Psychometrika 23(3):187-200)
//! on seed 42. Locks in that `plskit::rotate` reproduces them.

use faer::{Mat, MatRef};
use plskit::{rotate, RotationMethod, VarimaxArgs};
use serde_json::Value;
use std::fs;

fn load_2d(v: &Value) -> Mat<f64> {
    let outer = v.as_array().expect("outer array");
    let n = outer.len();
    let k = outer[0].as_array().expect("inner array").len();
    Mat::<f64>::from_fn(n, k, |i, j| outer[i][j].as_f64().expect("f64"))
}

fn approx_eq(a: MatRef<'_, f64>, b: MatRef<'_, f64>, tol: f64) -> bool {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return false;
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if (a[(i, j)] - b[(i, j)]).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[test]
fn rotate_matches_varimax_reference() {
    let raw = fs::read_to_string("tests/varimax_reference_seed42.json")
        .expect("tests/varimax_reference_seed42.json is missing");
    let v: Value = serde_json::from_str(&raw).unwrap();

    let w = load_2d(&v["W"]);
    let r_expected = load_2d(&v["R"]);
    #[allow(clippy::cast_possible_truncation)]
    let sweeps_expected = v["sweeps"].as_u64().unwrap() as usize;

    let out = rotate(
        w.as_ref(),
        RotationMethod::Varimax(VarimaxArgs::default()),
        None,
    )
    .unwrap();

    // Bit-near tolerance (array envelope: 1e-10).
    assert!(
        approx_eq(out.r.as_ref(), r_expected.as_ref(), 1e-10),
        "R diverges from the reference"
    );
    assert_eq!(out.sweeps, sweeps_expected, "sweep count differs");
    assert!(
        (out.v_converged - v["V_converged"].as_f64().unwrap()).abs() < 1e-12,
        "V_converged differs from the reference"
    );
}
