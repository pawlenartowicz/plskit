//! Integration tests for `pls3_fit`, `plssvd_fit`, `pls3_transform`,
//! `plssvd_transform` and `spls3_fit` through `call`.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{check_record, Record, Value, VecF64};

fn blocks() -> (Value<'static>, Value<'static>) {
    let (x, _) = data(60, 6, 1);
    let y = block_y(60, 3, &x, 4);
    (x, y)
}

#[test]
fn pls3_and_plssvd_fit_are_the_same_function() {
    let (x, y) = blocks();
    let a = ok(
        "pls3_fit",
        vec![("X", x.clone()), ("Y", y.clone()), ("k", Value::I64(2))],
    );
    let b = ok("plssvd_fit", vec![("X", x), ("Y", y), ("k", Value::I64(2))]);
    check_record(record(&a)).unwrap();
    assert!(same(&a.result, &b.result));
    assert!(field(record(&a), "converged").is_null());
}

#[test]
fn a_one_column_y_matrix_is_2d_and_a_vector_is_not() {
    let (x, _) = data(60, 6, 1);
    let y1 = block_y(60, 1, &x, 4);
    ok(
        "pls3_fit",
        vec![("X", x.clone()), ("Y", y1), ("k", Value::I64(1))],
    );
    let e = err(
        "pls3_fit",
        vec![("X", x), ("Y", Value::Vec(VecF64::Owned(vec![0.0; 60])))],
    );
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("use pls1_fit"), "{}", e.message);
}

#[test]
fn spls3_at_full_keep_reproduces_the_dense_saliences() {
    let (x, y) = blocks();
    let dense = ok(
        "pls3_fit",
        vec![("X", x.clone()), ("Y", y.clone()), ("k", Value::I64(2))],
    );
    let sparse = ok(
        "spls3_fit",
        vec![
            ("X", x),
            ("Y", y),
            ("k", Value::I64(2)),
            ("keep_X", Value::I64(6)),
            ("keep_Y", Value::I64(3)),
        ],
    );
    let (d, s) = (record(&dense), record(&sparse));
    check_record(s).unwrap();
    assert!(same(field(d, "U"), field(s, "U")));
    assert!(same(field(d, "V"), field(s, "V")));
    assert!(matches!(field(s, "converged"), Value::BoolVec(v) if v.len() == 2));
    assert!(matches!(field(s, "n_iter"), Value::IntVec(v) if v.len() == 2));
    assert_eq!(i(s, "keep_X"), 6);
}

#[test]
fn transform_round_trips_and_validates() {
    let (x, y) = blocks();
    let model = ok(
        "pls3_fit",
        vec![("X", x.clone()), ("Y", y.clone()), ("k", Value::I64(2))],
    )
    .result;
    let both = ok(
        "pls3_transform",
        vec![
            ("model", model.clone()),
            ("X_new", x.clone()),
            ("Y_new", y.clone()),
        ],
    );
    let r = record(&both);
    check_record(r).unwrap();
    // In-sample transform reproduces the fit's own scores.
    let fit_scores = floats(field(
        match &model {
            Value::Record(m) => m,
            _ => panic!(),
        },
        "x_scores",
    ));
    let got = floats(field(r, "x_scores"));
    assert!(fit_scores
        .iter()
        .zip(&got)
        .all(|(a, b)| (a - b).abs() < 1e-10));

    let alias = ok(
        "plssvd_transform",
        vec![
            ("model", model.clone()),
            ("X_new", x.clone()),
            ("Y_new", y.clone()),
        ],
    );
    assert!(same(&both.result, &alias.result));

    let xo = ok(
        "pls3_transform",
        vec![
            ("model", model.clone()),
            ("X_new", x.clone()),
            ("which", Value::text("x_scores")),
        ],
    );
    assert!(field(record(&xo), "y_scores").is_null());

    let e = err(
        "pls3_transform",
        vec![
            ("model", model.clone()),
            ("X_new", x.clone()),
            ("which", Value::text("z")),
        ],
    );
    assert_eq!(e.code, "invalid_args");

    let Value::Record(m) = model else { panic!() };
    let mut no_v = Record::typed("PLS3Result");
    for (k, v) in m.iter() {
        if k != "V" {
            no_v.push(k, v.clone()).unwrap();
        }
    }
    let e = err(
        "pls3_transform",
        vec![("model", Value::Record(no_v)), ("X_new", x)],
    );
    assert_eq!(e.code, "invalid_argument");
}

// Review Focus 1 again: R hands back a q = 1 model's Y moments as scalars.
#[test]
fn transform_widens_scalar_moments() {
    let (x, _) = data(60, 6, 1);
    let y1 = block_y(60, 1, &x, 4);
    let Value::Record(mut m) = ok(
        "pls3_fit",
        vec![("X", x.clone()), ("Y", y1.clone()), ("k", Value::I64(1))],
    )
    .result
    else {
        panic!()
    };
    let before = ok(
        "pls3_transform",
        vec![
            ("model", Value::Record(m.clone())),
            ("X_new", x.clone()),
            ("Y_new", y1.clone()),
        ],
    );
    let mean = floats(field(&m, "Y_mean"))[0];
    let scale = floats(field(&m, "Y_scale"))[0];
    m.set("Y_mean", Value::F64(mean));
    m.set("Y_scale", Value::F64(scale));
    let after = ok(
        "pls3_transform",
        vec![("model", Value::Record(m)), ("X_new", x), ("Y_new", y1)],
    );
    assert!(same(&before.result, &after.result));
}
