//! Integration tests for `rotate`'s array and model overloads.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{check_record, MatF64, Record, Value, VecF64};

fn fit(k: i64) -> Value<'static> {
    let (x, y) = data(60, 6, 1);
    ok("pls1_fit", vec![("X", x), ("y", y), ("k", Value::I64(k))]).result
}

fn rec_of<'r>(v: &'r Value<'static>) -> &'r Record<'static> {
    match v {
        Value::Record(r) => r,
        other => panic!("expected a record, got {other:?}"),
    }
}

#[test]
fn array_overload_returns_a_rotate_result_with_resolved_args() {
    let model = fit(3);
    let w = field(rec_of(&model), "W").clone();
    let o = ok("rotate", vec![("model_or_W", w)]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("RotateResult"));
    check_record(r).unwrap();
    let Value::Record(spec) = field(r, "spec") else {
        panic!()
    };
    let Value::Record(args) = field(spec, "args") else {
        panic!()
    };
    assert_eq!(
        args.keys().collect::<Vec<_>>(),
        ["max_iter", "tol", "kaiser_normalize"]
    );
    assert_eq!(i(args, "max_iter"), 50);
    assert!(matches!(field(spec, "L_was_provided"), Value::Bool(false)));
}

#[test]
fn model_overload_composes_scores_loadings_and_q() {
    let model = fit(3);
    let m = rec_of(&model);
    let arr = ok("rotate", vec![("model_or_W", field(m, "W").clone())]);
    let o = ok("rotate", vec![("model_or_W", model.clone())]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("PLS1Result"));
    check_record(r).unwrap();
    assert!(same(field(r, "W"), field(record(&arr), "W_rot")));

    let Value::Record(spec) = field(r, "rotation_spec") else {
        panic!("rotation_spec missing")
    };
    let Value::Mat(rm) = field(spec, "R") else {
        panic!()
    };
    let rm = rm.as_mat();
    let (Value::Mat(t0), Value::Mat(t1)) = (field(m, "T"), field(r, "T")) else {
        panic!()
    };
    let (t0, t1) = (t0.as_mat(), t1.as_mat());
    for row in 0..t0.nrows() {
        for c in 0..3 {
            let want: f64 = (0..3).map(|j| t0[(row, j)] * rm[(j, c)]).sum();
            assert!((t1[(row, c)] - want).abs() < 1e-12);
        }
    }
    let (q0, q1) = (floats(field(m, "Q")), floats(field(r, "Q")));
    for c in 0..3 {
        let want: f64 = (0..3).map(|j| rm[(j, c)] * q0[j]).sum();
        assert!((q1[c] - want).abs() < 1e-12);
    }
    // Rotation leaves the regression untouched.
    assert!(same(field(m, "coef"), field(r, "coef")));
}

#[test]
fn a_rotated_model_cannot_be_rotated_again() {
    let once = ok("rotate", vec![("model_or_W", fit(2))]).result;
    let e = err("rotate", vec![("model_or_W", once)]);
    assert_eq!(e.code, "already_rotated");
}

// Review Focus 5: every field the rotation does not touch survives it.
#[test]
fn rotation_carries_selection_result_and_keep_through() {
    let (x, y) = data(80, 6, 1);
    let model = ok(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::text("optimal")),
            ("k_max", Value::I64(3)),
            ("seed", Value::U64(11)),
        ],
    )
    .result;
    let m = rec_of(&model);
    // Whatever K the selection picks (the engine rotates K = 1 too), the
    // fields rotation does not touch must come back bit-identical.
    let o = ok("rotate", vec![("model_or_W", model.clone())]);
    let r = record(&o);
    assert!(same(
        field(m, "selection_result"),
        field(r, "selection_result")
    ));
    assert!(same(field(m, "keep"), field(r, "keep")));
    assert!(same(field(m, "weights"), field(r, "weights")));
}

#[test]
fn rotate_errors() {
    let model = fit(2);
    let w = field(rec_of(&model), "W").clone();
    let e = err(
        "rotate",
        vec![("model_or_W", w.clone()), ("method", Value::text("promax"))],
    );
    assert_eq!(e.code, "rotation_method_not_implemented");
    let bad = Value::Record(rec(vec![("bogus", Value::I64(1))]));
    let e = err("rotate", vec![("model_or_W", w.clone()), ("args", bad)]);
    assert_eq!(e.code, "invalid_args");
    let (x, _) = data(60, 6, 1);
    let y = block_y(60, 3, &x, 4);
    let pls3 = ok("pls3_fit", vec![("X", x), ("Y", y), ("k", Value::I64(2))]).result;
    let e = err("rotate", vec![("model_or_W", pls3)]);
    assert_eq!(e.code, "invalid_argument");
    let e = err("rotate", vec![("model_or_W", Value::F64(1.0))]);
    assert_eq!(e.code, "invalid_argument");
}

// Review Important #1: a hand-edited model whose T/P/Q no longer agree
// with W's shape must raise invalid_argument, not panic inside faer.
#[test]
fn rotate_rejects_a_model_with_mismatched_shapes() {
    let model = fit(3);
    let m = rec_of(&model).clone();

    let mut bad_t = m.clone();
    let Value::Mat(t) = field(&bad_t, "T") else {
        panic!()
    };
    let nrows = t.nrows();
    bad_t.set(
        "T",
        Value::Mat(MatF64::Owned {
            data: vec![0.0; nrows],
            nrows,
            ncols: 1,
        }),
    );
    let e = err("rotate", vec![("model_or_W", Value::Record(bad_t))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'T'"));

    let mut bad_p_cols = m.clone();
    let Value::Mat(p) = field(&bad_p_cols, "P") else {
        panic!()
    };
    let nrows = p.nrows();
    bad_p_cols.set(
        "P",
        Value::Mat(MatF64::Owned {
            data: vec![0.0; nrows],
            nrows,
            ncols: 1,
        }),
    );
    let e = err("rotate", vec![("model_or_W", Value::Record(bad_p_cols))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'P'"));

    let mut bad_p_rows = m.clone();
    let Value::Mat(p) = field(&bad_p_rows, "P") else {
        panic!()
    };
    let ncols = p.ncols();
    bad_p_rows.set(
        "P",
        Value::Mat(MatF64::Owned {
            data: vec![0.0; ncols],
            nrows: 1,
            ncols,
        }),
    );
    let e = err("rotate", vec![("model_or_W", Value::Record(bad_p_rows))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'P'"));

    let mut bad_q = m.clone();
    bad_q.set("Q", Value::Vec(VecF64::Owned(vec![0.0; 5])));
    let e = err("rotate", vec![("model_or_W", Value::Record(bad_q))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'Q'"));

    // W/T mismatch: shrink W's columns below T's.
    let mut bad_w = m.clone();
    let Value::Mat(w) = field(&bad_w, "W") else {
        panic!()
    };
    let nrows = w.nrows();
    bad_w.set(
        "W",
        Value::Mat(MatF64::Owned {
            data: vec![0.0; nrows * 2],
            nrows,
            ncols: 2,
        }),
    );
    let e = err("rotate", vec![("model_or_W", Value::Record(bad_w))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("'T'"));
}

// Review Minor 1: an untagged model record missing a nullable field and
// carrying an extra field must still yield output that matches the
// PLS1Result shape exactly (Python's dataclasses.replace semantics).
#[test]
fn rotate_output_matches_pls1result_shape_for_an_untagged_partial_model() {
    let model = fit(2);
    let m = rec_of(&model);
    let mut untagged = Record::new();
    for (k, v) in m.iter() {
        // Drop the nullable "weights" field and skip re-tagging.
        if k == "weights" {
            continue;
        }
        untagged.push(k, v.clone()).unwrap();
    }
    untagged.push("extra_junk", Value::I64(7)).unwrap();

    let o = ok("rotate", vec![("model_or_W", Value::Record(untagged))]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("PLS1Result"));
    check_record(r).unwrap();
    assert!(matches!(field(r, "weights"), Value::Null));
    assert!(r.get("extra_junk").is_none());
}

#[test]
fn a_loading_basis_is_recorded() {
    let model = fit(2);
    let w = field(rec_of(&model), "W").clone();
    // L needs ncols == k; the weights themselves are a valid basis.
    let o = ok("rotate", vec![("model_or_W", w.clone()), ("L", w)]);
    let Value::Record(spec) = field(record(&o), "spec") else {
        panic!()
    };
    assert!(matches!(field(spec, "L_was_provided"), Value::Bool(true)));
}
