//! Basic preprocess entry point tests.

#![allow(clippy::cast_precision_loss)]

use approx::assert_relative_eq;
use faer::{Col, Mat};
use plskit::error::PlsKitError;
use plskit::preprocess::{preprocess, preprocess_block, PreprocessBlockInput, PreprocessInput};

fn small() -> (Mat<f64>, Col<f64>, Col<f64>) {
    let x = Mat::from_fn(5, 3, |i, j| (i + j) as f64);
    let y = Col::<f64>::from_fn(5, |i| i as f64);
    let w = Col::<f64>::from_fn(5, |i| (i + 1) as f64);
    (x, y, w)
}

/// Each output is populated exactly when its input was given: `x_std` for
/// X, `y_std` for y, `weights_normalized` and `n_eff` for weights. With
/// weights alone there is no shape to check against.
#[test]
fn preprocess_populates_exactly_the_inputs_given() {
    let (x, y, w) = small();
    for (has_x, has_y, has_w) in [
        (false, false, false),
        (true, false, false),
        (true, true, true),
        (false, false, true),
    ] {
        let r = preprocess(PreprocessInput {
            x: has_x.then(|| x.as_ref()),
            y: has_y.then(|| y.as_ref()),
            weights: has_w.then(|| w.as_ref()),
        })
        .unwrap();
        let what = format!("x={has_x} y={has_y} w={has_w}");
        assert_eq!(r.x_std.is_some(), has_x, "{what}: x_std");
        assert_eq!(r.y_std.is_some(), has_y, "{what}: y_std");
        assert_eq!(r.weights_normalized.is_some(), has_w, "{what}: weights");
        assert_eq!(r.n_eff.is_some(), has_w, "{what}: n_eff");
        if let Some((xs, mean, scale)) = &r.x_std {
            assert_eq!(
                (xs.nrows(), xs.ncols(), mean.nrows(), scale.nrows()),
                (5, 3, 3, 3),
                "{what}"
            );
        }
    }
}

/// An X with no columns is accepted and comes back with no columns.
#[test]
fn zero_column_x_keeps_its_shape() {
    let x = Mat::<f64>::zeros(3, 0);
    let r = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: None,
        weights: None,
    })
    .unwrap();
    let (xs, mean, scale) = r.x_std.unwrap();
    assert_eq!(
        (xs.nrows(), xs.ncols(), mean.nrows(), scale.nrows()),
        (3, 0, 0, 0)
    );
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
    assert!(
        matches!(r, Err(plskit::error::PlsKitError::DimensionMismatch { .. })),
        "{r:?}"
    );
}

/// The 1-D form rejects a bad weights vector: by reason for a negative, an
/// all-zero or a wrong-length one, and as non-finite input for NaN or
/// infinity.
#[test]
fn preprocess_rejects_invalid_weights() {
    let (x, y, _) = small();
    let cases: [(&str, Vec<f64>, &str, Option<&str>); 5] = [
        (
            "negative",
            vec![1.0, -0.5, 1.0, 1.0, 1.0],
            "invalid_weights",
            Some("negative"),
        ),
        (
            "all zero",
            vec![0.0; 5],
            "invalid_weights",
            Some("all_zero"),
        ),
        (
            "nan",
            vec![1.0, f64::NAN, 1.0, 1.0, 1.0],
            "non_finite_input",
            None,
        ),
        (
            "inf",
            vec![1.0, f64::INFINITY, 1.0, 1.0, 1.0],
            "non_finite_input",
            None,
        ),
        (
            "length",
            vec![1.0; 4],
            "invalid_weights",
            Some("length_mismatch"),
        ),
    ];
    for (what, w, code, reason) in cases {
        let w = Col::<f64>::from_fn(w.len(), |i| w[i]);
        let err = preprocess(PreprocessInput {
            x: Some(x.as_ref()),
            y: Some(y.as_ref()),
            weights: Some(w.as_ref()),
        })
        .unwrap_err();
        assert_eq!(err.code(), code, "{what}: {err:?}");
        if let Some(reason) = reason {
            assert!(
                matches!(err, PlsKitError::InvalidWeights { reason: r } if r == reason),
                "{what}: {err:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric value assertions — fixture: n=4, d=2 cols, w_raw=[2,1,1,1]
//
// Convention (mirrors standardize_weighted source):
//   w'_i = w_i · n / Σw  (normalized so Σw' = n)
//   μ_j  = Σ(w'_i · x_{ij}) / n
//   var_j = Σ(w'_i · (x_{ij} − μ_j)²) / n   (population, ddof=0)
//   scale_j = √var_j  (or max(1, |mean|) for a column constant to rounding, or whose std underflows)
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

// ---------------------------------------------------------------------------
// preprocess_block: the multi-column-Y form
// ---------------------------------------------------------------------------

/// All-equal weights: `weights_normalized` is still echoed, and `n_eff` is
/// exactly `n`, the value `pls1_fit` reports for the same weights (Kish's
/// ratio of 0.3s rounds an ulp below 3). Unequal weights keep Kish's value.
/// The fitted model echoes all-equal weights as `None` and unequal ones
/// normalized to mean one (`w·n/Σw`).
#[test]
#[allow(clippy::many_single_char_names)]
fn all_equal_weights_report_n_eff_n_like_the_fits() {
    let x = Mat::from_fn(3, 2, |i, j| ((i + 1) * (j + 2)) as f64 + (i * i) as f64);
    let y = Col::<f64>::from_fn(3, |i| [1.0, 3.0, 2.0][i]);
    let yb = Mat::from_fn(3, 1, |i, _| y[i]);
    for (w, want, echo) in [
        ([0.3; 3], 3.0_f64, None),
        ([1e6; 3], 3.0, None),
        ([1.0, 1.0, 2.0], 16.0 / 6.0, Some([0.75, 0.75, 1.5])),
    ] {
        let w = Col::<f64>::from_fn(3, |i| w[i]);
        let r = preprocess(PreprocessInput {
            x: Some(x.as_ref()),
            y: None,
            weights: Some(w.as_ref()),
        })
        .unwrap();
        let b = preprocess_block(PreprocessBlockInput {
            x: None,
            y: Some(yb.as_ref()),
            weights: Some(w.as_ref()),
        })
        .unwrap();
        let fit = plskit::pls1_fit(
            x.as_ref(),
            y.as_ref(),
            plskit::KSpec::Fixed(1),
            Some(w.as_ref()),
            plskit::FitOpts::default(),
        )
        .unwrap();
        let what = format!("w = {:?}", w[0]);
        assert!(r.weights_normalized.is_some(), "{what}");
        assert_eq!(r.n_eff.map(f64::to_bits), Some(want.to_bits()), "{what}");
        assert_eq!(b.n_eff.map(f64::to_bits), Some(want.to_bits()), "{what}");
        assert_eq!(fit.n_eff.to_bits(), want.to_bits(), "{what}: pls1_fit");
        match (&fit.weights, echo) {
            (None, None) => {}
            (Some(got), Some(want)) => {
                assert_eq!(got.nrows(), 3, "{what}: echoed weights");
                for i in 0..3 {
                    assert_relative_eq!(got[i], want[i], epsilon = 1e-15);
                }
            }
            (got, want) => panic!("{what}: echoed weights {got:?}, expected {want:?}"),
        }
    }
}

/// Y block for the block tests: col 0 = y of `small`, col 1 its reverse,
/// col 2 constant (so the constant rule applies).
fn small_block() -> Mat<f64> {
    Mat::from_fn(5, 3, |i, j| match j {
        0 => i as f64,
        1 => (4 - i) as f64 * 3.5,
        _ => 2.0,
    })
}

/// Column `j` of the block result is `preprocess` on `y = Y[:, j]`, bit
/// for bit, with and without weights (the constant column included); X and
/// the weights come out as `preprocess` returns them.
#[test]
fn block_columns_are_preprocess_on_each_column() {
    let (x, _, w) = small();
    let yb = small_block();
    for weights in [None, Some(w.as_ref())] {
        let b = preprocess_block(PreprocessBlockInput {
            x: Some(x.as_ref()),
            y: Some(yb.as_ref()),
            weights,
        })
        .unwrap();
        let (ys, mean, scale) = b.y_std.as_ref().unwrap();
        assert_eq!(
            (ys.nrows(), ys.ncols(), mean.nrows(), scale.nrows()),
            (5, 3, 3, 3)
        );
        for j in 0..yb.ncols() {
            let col = yb.col(j).to_owned();
            let one = preprocess(PreprocessInput {
                x: Some(x.as_ref()),
                y: Some(col.as_ref()),
                weights,
            })
            .unwrap();
            let (ys1, mean1, scale1) = one.y_std.unwrap();
            let what = format!("col {j}, weighted={}", weights.is_some());
            assert_eq!(mean[j].to_bits(), mean1.to_bits(), "{what}: mean");
            assert_eq!(scale[j].to_bits(), scale1.to_bits(), "{what}: scale");
            for i in 0..ys.nrows() {
                assert_eq!(ys[(i, j)].to_bits(), ys1[i].to_bits(), "{what}: row {i}");
            }
            assert_eq!(
                b.x_std.as_ref().map(|t| &t.0),
                one.x_std.as_ref().map(|t| &t.0)
            );
            assert_eq!(b.weights_normalized, one.weights_normalized, "{what}");
            assert_eq!(
                b.n_eff.map(f64::to_bits),
                one.n_eff.map(f64::to_bits),
                "{what}"
            );
        }
    }
}

/// A `PreprocessBlockInput` over owned test matrices.
fn input<'a>(
    x: Option<&'a Mat<f64>>,
    y: &'a Mat<f64>,
    w: Option<&'a Col<f64>>,
) -> PreprocessBlockInput<'a> {
    PreprocessBlockInput {
        x: x.map(Mat::as_ref),
        y: Some(y.as_ref()),
        weights: w.map(Col::as_ref),
    }
}

/// The block form rejects what `preprocess` rejects for a 1-D `y`: a
/// non-finite Y entry, a row count differing from X, and weights of the
/// wrong length.
#[test]
fn block_rejects_what_preprocess_rejects() {
    type Expect = fn(&PlsKitError) -> bool;
    let non_finite: Expect = |e| matches!(e, PlsKitError::NonFiniteInput);
    let rows: Expect = |e| matches!(e, PlsKitError::DimensionMismatch { .. });
    let length: Expect = |e| {
        matches!(
            e,
            PlsKitError::InvalidWeights {
                reason: "length_mismatch"
            }
        )
    };
    let (x, _, w) = small();
    let mut nan = small_block();
    nan[(3, 1)] = f64::NAN;
    let mut inf = small_block();
    inf[(0, 2)] = f64::INFINITY;
    let short = Mat::from_fn(4, 2, |i, j| (i * j) as f64);
    let good = small_block();
    let w_short = Col::<f64>::from_fn(4, |_| 1.0);
    let cases = [
        ("nan", input(Some(&x), &nan, None), non_finite),
        ("inf, no X", input(None, &inf, Some(&w)), non_finite),
        ("rows", input(Some(&x), &short, None), rows),
        ("weights vs Y", input(None, &good, Some(&w_short)), length),
        (
            "weights vs X",
            input(Some(&x), &good, Some(&w_short)),
            length,
        ),
    ];
    for (what, inp, expect) in cases {
        match preprocess_block(inp) {
            Err(e) => assert!(expect(&e), "{what}: wrong error {e:?}"),
            Ok(_) => panic!("{what}: accepted"),
        }
    }
}

/// Zero rows: `preprocess` and `preprocess_block` raise `invalid_argument`
/// (the PLS3 fits' "insufficient n" error) instead of returning the NaN
/// moments of an empty column, whether the empty block is X, y / Y or
/// both. One row is fine (each column is constant: mean the value, scale
/// `max(1, |mean|)`).
#[test]
fn zero_rows_error_one_row_is_finite() {
    let check = |what: &str, r: Result<(), PlsKitError>| match r {
        Err(PlsKitError::InvalidArgument(m)) => assert!(m.contains("need n >= 1"), "{what}: {m}"),
        r => panic!("{what}: {r:?}"),
    };
    for n in [0usize, 1] {
        let x = Mat::<f64>::from_fn(n, 2, |i, j| (i + j) as f64 + 0.5);
        let y = Col::<f64>::from_fn(n, |i| i as f64 - 3.0);
        let yy = Mat::<f64>::from_fn(n, 3, |i, j| (i * j) as f64 + 2.0);
        let w = Col::<f64>::from_fn(n, |_| 1.0);
        for (xi, yi) in [(true, false), (false, true), (true, true)] {
            for wi in [false, true] {
                let what = format!("n={n} x={xi} y={yi} w={wi}");
                let one = preprocess(PreprocessInput {
                    x: xi.then(|| x.as_ref()),
                    y: yi.then(|| y.as_ref()),
                    weights: wi.then(|| w.as_ref()),
                });
                let block = preprocess_block(PreprocessBlockInput {
                    x: xi.then(|| x.as_ref()),
                    y: yi.then(|| yy.as_ref()),
                    weights: wi.then(|| w.as_ref()),
                });
                if n == 0 {
                    check(&format!("preprocess {what}"), one.map(|_| ()));
                    check(&format!("preprocess_block {what}"), block.map(|_| ()));
                } else {
                    let (one, block) = (one.unwrap(), block.unwrap());
                    let mut vals: Vec<f64> = Vec::new();
                    if let Some((_, m, s)) = &one.x_std {
                        vals.extend(m.iter().chain(s.iter()));
                    }
                    if let Some((_, m, s)) = one.y_std {
                        vals.extend([m, s]);
                    }
                    if let Some((_, m, s)) = &block.y_std {
                        vals.extend(m.iter().chain(s.iter()));
                    }
                    assert!(vals.iter().all(|v| v.is_finite()), "{what}: {vals:?}");
                }
            }
        }
    }
}
