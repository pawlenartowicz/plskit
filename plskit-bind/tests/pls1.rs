//! Integration tests for `preprocess`, `pls1_fit`, `pls1_predict`,
//! `spls1_fit` and `split_nb_gate` through `call`.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{call, check_record, MatF64, Record, Value, VecF64};

fn fit(k: Value<'static>) -> plskit_bind::Outcome {
    let (x, y) = data(60, 6, 1);
    ok("pls1_fit", vec![("X", x), ("y", y), ("k", k)])
}

#[test]
fn unknown_function_is_invalid_argument() {
    let e = call("pls2_fit", Record::new()).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
}

#[test]
fn unexpected_argument_is_invalid_argument() {
    let (x, y) = data(60, 6, 1);
    let e = err(
        "pls1_fit",
        vec![("X", x), ("y", y), ("ncomp", Value::I64(2))],
    );
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'ncomp'"), "{}", e.message);
}

#[test]
fn missing_or_null_required_argument() {
    let (x, _) = data(60, 6, 1);
    let e = err("pls1_fit", vec![("X", x.clone())]);
    assert!(
        e.message.contains("missing required argument 'y'"),
        "{}",
        e.message
    );
    let e = err("pls1_fit", vec![("X", x), ("y", Value::Null)]);
    assert_eq!(e.code, "invalid_argument");
}

#[test]
fn pls1_fit_returns_a_conforming_pls1_result() {
    let o = fit(Value::I64(2));
    let r = record(&o);
    assert_eq!(r.type_name(), Some("PLS1Result"));
    check_record(r).unwrap();
    assert_eq!(i(r, "k_used"), 2);
    assert!(field(r, "rotation_spec").is_null());
    assert!(field(r, "selection_result").is_null());
    assert!(field(r, "keep").is_null());
    assert!(o.warnings.is_empty());
}

#[test]
fn whole_double_k_is_accepted_and_fractional_k_is_not() {
    let a = fit(Value::F64(2.0));
    let b = fit(Value::I64(2));
    assert!(same(&a.result, &b.result));
    let (x, y) = data(60, 6, 1);
    let e = err("pls1_fit", vec![("X", x), ("y", y), ("k", Value::F64(2.5))]);
    assert_eq!(e.code, "invalid_argument");
    assert_eq!(e.message, "k must be a non-negative whole number, got 2.5");
}

#[test]
fn k_max_or_find_k_args_with_an_integer_k_is_rejected() {
    let (x, y) = data(60, 6, 1);
    let e = err(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(2)),
            ("k_max", Value::I64(4)),
        ],
    );
    assert_eq!(e.code, "invalid_argument");
    let e = err(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::I64(2)),
            ("find_k_args", Value::Record(Record::new())),
        ],
    );
    assert_eq!(e.code, "invalid_argument");
    assert_eq!(
        e.message,
        "k_max and find_k_args apply only when k is 'optimal' or 'sequence'"
    );
}

#[test]
fn seed_with_an_integer_k_is_accepted_and_ignored() {
    let (x, y) = data(60, 6, 1);
    let a = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(2)),
            ("seed", Value::text("7")),
        ],
    );
    let b = ok("pls1_fit", vec![("X", x), ("y", y), ("k", Value::I64(2))]);
    assert!(same(&a.result, &b.result));
}

// A null explicit optional argument means "absent": the default applies,
// same as omitting it, for both a bool default (pre_standardized) and an
// int default (k).
#[test]
fn null_optional_argument_takes_its_default() {
    let (x, y) = data(60, 6, 1);
    let a = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(2)),
            ("pre_standardized", Value::Null),
        ],
    );
    let b = ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y.clone()), ("k", Value::I64(2))],
    );
    assert!(same(&a.result, &b.result));

    let a = ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y.clone()), ("k", Value::Null)],
    );
    let b = ok("pls1_fit", vec![("X", x), ("y", y)]);
    assert!(same(&a.result, &b.result));
}

#[test]
fn predict_round_trips_the_record() {
    let (x, y) = data(60, 6, 1);
    let model = fit(Value::I64(2)).result;
    let p = ok(
        "pls1_predict",
        vec![("model", model.clone()), ("X_new", x.clone())],
    );
    let Value::Vec(yhat) = &p.result else {
        panic!("expected a vector")
    };
    assert_eq!(yhat.len(), 60);

    // Same numbers as the engine called directly.
    let Value::Mat(xm) = &x else { panic!() };
    let Value::Vec(yv) = &y else { panic!() };
    let m = plskit::pls1_fit(
        xm.as_mat(),
        yv.as_col(),
        plskit::KSpec::Fixed(2),
        None,
        plskit::FitOpts::default(),
    )
    .unwrap();
    let direct = plskit::pls1_predict(&m, xm.as_mat()).unwrap();
    for (row, &got) in yhat.as_slice().iter().enumerate() {
        assert!((got - direct[row]).abs() <= 1e-12);
    }
}

// Review Focus 1: R hands back a k = 1 model's Q as a length-1 vector,
// which the seam turns into a scalar.
#[test]
fn predict_widens_a_scalar_q() {
    let (x, _) = data(60, 6, 1);
    let Value::Record(mut model) = fit(Value::I64(1)).result else {
        panic!()
    };
    let q = floats(field(&model, "Q"))[0];
    let before = ok(
        "pls1_predict",
        vec![
            ("model", Value::Record(model.clone())),
            ("X_new", x.clone()),
        ],
    );
    model.set("Q", Value::F64(q));
    let after = ok(
        "pls1_predict",
        vec![("model", Value::Record(model)), ("X_new", x)],
    );
    assert!(same(&before.result, &after.result));
}

// Review Focus 4: a y with no first component gives the zero model.
#[test]
fn a_zero_component_model_round_trips() {
    let (x, _) = data(40, 5, 3);
    let y = Value::Vec(VecF64::Owned(vec![2.0; 40]));
    let o = ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y), ("k", Value::I64(2))],
    );
    let r = record(&o);
    check_record(r).unwrap();
    assert_eq!(i(r, "k_used"), 0);
    let Value::Mat(t) = field(r, "T") else {
        panic!()
    };
    assert_eq!(t.ncols(), 0);
    let p = ok(
        "pls1_predict",
        vec![("model", o.result.clone()), ("X_new", x)],
    );
    assert!(floats(&p.result).iter().all(|&v| (v - 2.0).abs() < 1e-12));
}

#[test]
fn predict_rejects_a_broken_model() {
    let (x, _) = data(60, 6, 1);
    let Value::Record(mut model) = fit(Value::I64(2)).result else {
        panic!()
    };
    let mut no_w = Record::typed("PLS1Result");
    for (k, v) in model.iter() {
        if k != "W" {
            no_w.push(k, v.clone()).unwrap();
        }
    }
    let e = err(
        "pls1_predict",
        vec![("model", Value::Record(no_w)), ("X_new", x.clone())],
    );
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'W'"), "{}", e.message);

    model.set_type_name("PLS3Result");
    let e = err(
        "pls1_predict",
        vec![("model", Value::Record(model)), ("X_new", x)],
    );
    assert_eq!(e.code, "invalid_argument");
}

#[test]
fn preprocess_1d_and_2d_y() {
    let (x, y) = data(50, 4, 2);
    let o = ok("preprocess", vec![("X", x.clone()), ("Y", y)]);
    let r = record(&o);
    check_record(r).unwrap();
    assert!(matches!(field(r, "Y_mean"), Value::F64(_)));
    assert!(field(r, "n_eff").is_null());

    let y2 = block_y(50, 3, &x, 5);
    let o = ok("preprocess", vec![("X", x.clone()), ("Y", y2)]);
    let r = record(&o);
    check_record(r).unwrap();
    assert!(matches!(field(r, "Y_mean"), Value::Vec(v) if v.len() == 3));

    let Value::Mat(MatF64::Owned {
        mut data,
        nrows,
        ncols,
    }) = block_y(50, 3, &x, 5)
    else {
        panic!()
    };
    data[nrows + 7] = f64::NAN;
    let nan_y = Value::Mat(MatF64::Owned { data, nrows, ncols });
    let e = err("preprocess", vec![("X", x.clone()), ("Y", nan_y)]);
    assert_eq!(e.code, "non_finite_input");

    let short = Value::Mat(MatF64::Owned {
        data: vec![0.0; 30],
        nrows: 10,
        ncols: 3,
    });
    let e = err("preprocess", vec![("X", x), ("Y", short)]);
    assert_eq!(e.code, "dimension_mismatch");
}

#[test]
fn spls1_fit_records_keep() {
    let (x, y) = data(60, 6, 1);
    let o = ok(
        "spls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::I64(2)),
            ("keep", Value::I64(3)),
        ],
    );
    let r = record(&o);
    check_record(r).unwrap();
    assert_eq!(i(r, "keep"), 3);
}

#[test]
fn split_nb_gate_record() {
    let (x, _) = data(60, 6, 1);
    let o = ok("split_nb_gate", vec![("X", x)]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("SplitNbGateResult"));
    check_record(r).unwrap();
}

#[test]
fn engine_errors_keep_their_code_and_details() {
    let (x, y) = data(60, 6, 1);
    let Value::Mat(MatF64::Owned {
        mut data,
        nrows,
        ncols,
    }) = x.clone()
    else {
        panic!()
    };
    data[0] = f64::NAN;
    let nan_x = Value::Mat(MatF64::Owned { data, nrows, ncols });
    assert_eq!(
        err("pls1_fit", vec![("X", nan_x), ("y", y.clone())]).code,
        "non_finite_input"
    );

    let mut w = vec![1.0; 60];
    w[3] = -1.0;
    let e = err(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("weights", Value::Vec(VecF64::Owned(w))),
        ],
    );
    assert_eq!(e.code, "invalid_weights");
    assert!(matches!(e.details.get("reason"), Some(Value::Str(s)) if s == "negative"));
}

// Review Minor 6: the R and Julia seams pass Borrowed views
// (MatF64::from_col_major / VecF64::Borrowed), not Owned; both must take
// the same path through call() to a bit-identical result.
#[test]
fn borrowed_and_owned_inputs_give_bit_identical_results() {
    let n = 40;
    let p = 5;
    let mut rng = Rng::new(7);
    let x_data: Vec<f64> = (0..n * p).map(|_| rng.next()).collect();
    let y_data: Vec<f64> = (0..n)
        .map(|i| 4.0 * (x_data[i] + x_data[i + n]) + 0.5 * rng.next())
        .collect();

    let owned = ok(
        "pls1_fit",
        vec![
            (
                "X",
                Value::Mat(MatF64::Owned {
                    data: x_data.clone(),
                    nrows: n,
                    ncols: p,
                }),
            ),
            ("y", Value::Vec(VecF64::Owned(y_data.clone()))),
            ("k", Value::I64(2)),
        ],
    );

    let mut br = Record::new();
    br.push(
        "X",
        Value::Mat(MatF64::from_col_major(&x_data, n, p).unwrap()),
    )
    .unwrap();
    br.push("y", Value::Vec(VecF64::Borrowed(&y_data))).unwrap();
    br.push("k", Value::I64(2)).unwrap();
    let borrowed = call("pls1_fit", br).unwrap_or_else(|e| panic!("{}: {}", e.code, e.message));

    assert!(same(&owned.result, &borrowed.result));
}

// Bool data is accepted as 0/1, the way numpy casts a bool array: a
// logical `y` or `weights` fits exactly like its 0/1 double twin. A bool
// flag stays a flag (only `to_bool` reads it).
#[test]
fn bool_vectors_are_data_and_bool_flags_stay_flags() {
    let (x, y) = data(60, 6, 1);
    let yv = floats(&y);
    let y_bool: Vec<bool> = yv.iter().map(|&v| v > 0.0).collect();
    let y_01: Vec<f64> = y_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let a = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", Value::BoolVec(y_bool)),
            ("weights", Value::BoolVec(vec![true; 60])),
        ],
    );
    let b = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", Value::Vec(VecF64::Owned(y_01))),
            ("weights", Value::Vec(VecF64::Owned(vec![1.0; 60]))),
        ],
    );
    assert!(same(&a.result, &b.result));

    let e = err(
        "pls1_fit",
        vec![("X", Value::BoolVec(vec![true; 60])), ("y", y.clone())],
    );
    assert_eq!(e.code, "invalid_argument");
    assert_eq!(e.message, "X must be 2-D, got 1-D");

    let e = err(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("pre_standardized", Value::BoolVec(vec![true, false])),
        ],
    );
    assert_eq!(e.code, "invalid_argument");
    assert!(
        e.message.starts_with("pre_standardized must be a bool"),
        "{}",
        e.message
    );
}
