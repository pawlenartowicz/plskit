//! Integration tests for `rotate`: the matrix, `FactorsResult` and `PLS3Result` forms and their refusals.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{MatF64, Record, Value, VecF64, check_record};

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
fn rotate_refuses_a_pls1_result() {
    let e = err("rotate", vec![("model_or_W", fit(2))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("profiles"), "{}", e.message);
    assert!(e.message.contains("rotate(m.W"), "{}", e.message);
}

#[test]
fn rotate_refuses_an_untagged_record() {
    let untagged = Value::Record(rec(vec![("W", Value::I64(1))]));
    let e = err("rotate", vec![("model_or_W", untagged)]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("FactorsResult"), "{}", e.message);
    assert!(e.message.contains("PLS3Result"), "{}", e.message);
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
    let e = err("rotate", vec![("model_or_W", Value::F64(1.0))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("PLS3Result"), "{}", e.message);
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

fn mat_of(data: Vec<f64>, nrows: usize, ncols: usize) -> Value<'static> {
    Value::Mat(MatF64::Owned { data, nrows, ncols })
}

/// `rotate(model_or_W, ...)` with the extra fields.
fn rotate(first: &Value<'static>, extra: Vec<(&str, Value<'static>)>) -> plskit_bind::Outcome {
    let mut f = vec![("model_or_W", first.clone())];
    f.extend(extra);
    ok("rotate", f)
}

fn rotate_err(
    first: &Value<'static>,
    extra: Vec<(&str, Value<'static>)>,
) -> plskit_bind::BindError {
    let mut f = vec![("model_or_W", first.clone())];
    f.extend(extra);
    err("rotate", f)
}

/// `factors(m, x, n)` for a fitted model `m`.
fn factors_of(m: &Value<'static>, x: &Value<'static>, n: i64) -> Value<'static> {
    ok(
        "factors",
        vec![
            ("model_or_coef", m.clone()),
            ("X", x.clone()),
            ("n", Value::I64(n)),
        ],
    )
    .result
}

/// `x` with column `j` multiplied by `j + 1`.
#[allow(clippy::cast_precision_loss)]
fn uneven_scales(x: &Value<'static>) -> Value<'static> {
    let Value::Mat(m) = x else { panic!() };
    let n = m.nrows();
    let data = floats(x)
        .iter()
        .enumerate()
        .map(|(idx, v)| v * (idx / n + 1) as f64)
        .collect();
    mat_of(data, n, m.ncols())
}

fn pls3_fit(k: i64) -> (Value<'static>, Value<'static>) {
    let (x, _) = data(60, 6, 1);
    let y = block_y(60, 3, &x, 4);
    let m = ok(
        "pls3_fit",
        vec![("X", x.clone()), ("Y", y), ("k", Value::I64(k))],
    )
    .result;
    (x, m)
}

fn with_field(v: &Value<'static>, key: &str, new: Value<'static>) -> Value<'static> {
    let mut r = rec_of(v).clone();
    r.set(key, new);
    Value::Record(r)
}

#[test]
fn x_gives_the_row_scores_target() {
    let (x, y) = data(60, 6, 1);
    let m = ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y), ("k", Value::I64(3))],
    )
    .result;
    let f = factors_of(&m, &x, 3);
    let w = field(rec_of(&f), "W").clone();
    let l = mat_of(matmul(&xstd(&x), &w), 60, 3);
    let a = rotate(&f, vec![("X", x)]);
    let b = rotate(&f, vec![("L", l)]);
    assert!(same(&a.result, &b.result));
    assert!(matches!(
        field(rec_of(field(record(&a), "rotation_spec")), "L_was_provided"),
        Value::Bool(true)
    ));
}

#[test]
fn profiles_is_factors_then_rotate() {
    let (x, y) = data(60, 6, 1);
    let m = ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y), ("k", Value::I64(3))],
    )
    .result;
    let p = ok(
        "profiles",
        vec![
            ("model_or_coef", m.clone()),
            ("X", x.clone()),
            ("n", Value::I64(3)),
        ],
    );
    let r = rotate(&factors_of(&m, &x, 3), vec![("X", x)]);
    assert!(same(field(record(&p), "factors"), &r.result));
}

#[test]
fn rotate_with_x_follows_the_result() {
    let (x, y) = data(60, 6, 1);
    // A result with pre_standardized = true: X is used as given.
    let xs = xstd(&x);
    let m = ok(
        "pls1_fit",
        vec![
            ("X", xs.clone()),
            ("y", y.clone()),
            ("k", Value::I64(3)),
            ("pre_standardized", Value::Bool(true)),
        ],
    )
    .result;
    let f = factors_of(&m, &xs, 3);
    assert!(matches!(
        field(rec_of(&f), "pre_standardized"),
        Value::Bool(true)
    ));
    let l = mat_of(matmul(&xs, field(rec_of(&f), "W")), 60, 3);
    let a = rotate(&f, vec![("X", xs)]);
    let b = rotate(&f, vec![("L", l)]);
    assert!(same(&a.result, &b.result));

    // A result with pre_standardized = false, and X of unequal scales: the
    // caller says X is already standardized, so it is used as given.
    let x = uneven_scales(&x);
    let m = ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y), ("k", Value::I64(3))],
    )
    .result;
    let f = factors_of(&m, &x, 3);
    assert!(matches!(
        field(rec_of(&f), "pre_standardized"),
        Value::Bool(false)
    ));
    let l = mat_of(matmul(&x, field(rec_of(&f), "W")), 60, 3);
    let a = rotate(&f, vec![("X", x), ("pre_standardized", Value::Bool(true))]);
    let b = rotate(&f, vec![("L", l)]);
    assert!(same(&a.result, &b.result));
}

#[test]
fn a_factors_result_keeps_its_pre_standardized_through_rotate() {
    let (x, y) = data(60, 6, 1);
    let xs = xstd(&x);
    let m = ok(
        "pls1_fit",
        vec![
            ("X", xs.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("pre_standardized", Value::Bool(true)),
        ],
    )
    .result;
    let f = factors_of(&m, &xs, 2);
    let r = rotate(&f, vec![]);
    assert!(matches!(
        field(record(&r), "pre_standardized"),
        Value::Bool(true)
    ));

    let mut stripped = Record::typed("FactorsResult");
    for (k, v) in rec_of(&f).iter().filter(|(k, _)| *k != "pre_standardized") {
        stripped.push(k, v.clone()).unwrap();
    }
    let e = rotate_err(&Value::Record(stripped), vec![]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("pre_standardized"), "{}", e.message);
}

#[test]
fn the_matrix_form_takes_x() {
    let (x, _) = data(60, 6, 1);
    let m = fit(3);
    let w = field(rec_of(&m), "W").clone();
    let l = mat_of(matmul(&xstd(&x), &w), 60, 3);
    let a = rotate(&w, vec![("X", x)]);
    let b = rotate(&w, vec![("L", l)]);
    assert!(same(&a.result, &b.result));
}

/// A sparse fit keeps 3 of 8 columns, so 5 rows of `W` are zero.
#[test]
fn sparse_factors_keep_their_zeros() {
    let (x, y) = data(60, 8, 1);
    let m = ok(
        "spls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("keep", Value::I64(3)),
        ],
    )
    .result;
    let f = factors_of(&m, &x, 2);
    let zero_rows = |v: &Value<'static>| -> Vec<usize> {
        let Value::Mat(m) = v else { panic!() };
        let d = floats(v);
        (0..m.nrows())
            .filter(|&r| (0..m.ncols()).all(|c| d[r + c * m.nrows()].to_bits() == 0))
            .collect()
    };
    let before = zero_rows(field(rec_of(&f), "W"));
    assert!(before.len() >= 4, "the fit left out no variable");
    for extra in [vec![], vec![("X", x)]] {
        let o = rotate(&f, extra);
        assert!(o.warnings.is_empty());
        let r = record(&o);
        assert_eq!(zero_rows(field(r, "W")), before);
        assert_eq!(zero_rows(field(r, "F")), before);
    }
}

#[test]
fn pls3_result_is_rotated() {
    let (x, m3) = pls3_fit(3);
    let o = rotate(&m3, vec![("X", x)]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("PLS3Result"));
    check_record(r).unwrap();
    assert!(o.warnings.is_empty());

    // L = V has another row count than U.
    let v = field(rec_of(&m3), "V").clone();
    let o = rotate(&m3, vec![("L", v)]);
    assert_eq!(record(&o).type_name(), Some("PLS3Result"));
    assert!(matches!(
        field(rec_of(field(record(&o), "rotation_spec")), "L_was_provided"),
        Value::Bool(true)
    ));
}

#[test]
fn a_rotated_pls3_cannot_be_rotated_again() {
    let (_, m3) = pls3_fit(2);
    let once = rotate(&m3, vec![]).result;
    let e = rotate_err(&once, vec![]);
    assert_eq!(e.code, "already_rotated");
}

#[test]
fn sparse_pls3_warns() {
    let (x, _) = data(60, 8, 1);
    let y = block_y(60, 4, &x, 4);
    let m = ok(
        "spls3_fit",
        vec![
            ("X", x),
            ("Y", y),
            ("k", Value::I64(2)),
            ("keep_X", Value::I64(3)),
            ("keep_Y", Value::I64(2)),
        ],
    )
    .result;
    let o = rotate(&m, vec![]);
    assert_eq!(o.warnings.len(), 1);
    assert_eq!(s(&o.warnings[0], "kind"), "rotated_sparse");
    let (before, after) = (rec_of(&m), record(&o));
    assert!(!field(before, "converged").is_null());
    assert!(same(field(before, "converged"), field(after, "converged")));
    assert!(same(field(before, "n_iter"), field(after, "n_iter")));

    let (_, dense) = pls3_fit(2);
    assert!(rotate(&dense, vec![]).warnings.is_empty());
}

#[test]
fn a_zero_component_pls3_is_invalid_input() {
    let (_, m3) = pls3_fit(2);
    let r = rec_of(&m3);
    let cols0 = |key: &str| {
        let nrows = match field(r, key) {
            Value::Mat(m) => m.nrows(),
            other => panic!("{other:?}"),
        };
        mat_of(Vec::new(), nrows, 0)
    };
    let mut z = r.clone();
    for key in ["U", "V", "x_scores", "y_scores"] {
        z.set(key, cols0(key));
    }
    z.set("singular_values", Value::Vec(VecF64::Owned(Vec::new())));
    z.set("k_used", Value::I64(0));
    let e = rotate_err(&Value::Record(z), vec![]);
    assert_eq!(e.code, "invalid_input");
}

#[test]
fn a_pls3_result_with_non_finite_values_is_invalid_input() {
    let (_, m3) = pls3_fit(2);
    let bad = with_field(
        &m3,
        "singular_values",
        Value::Vec(VecF64::Owned(vec![f64::NAN, 1.0])),
    );
    let e = rotate_err(&bad, vec![]);
    assert_eq!(e.code, "invalid_input");
    for key in ["V", "x_scores"] {
        let Value::Mat(m) = field(rec_of(&m3), key) else {
            panic!()
        };
        let mut d = floats(field(rec_of(&m3), key));
        d[0] = f64::NAN;
        let bad = with_field(&m3, key, mat_of(d, m.nrows(), m.ncols()));
        assert_eq!(rotate_err(&bad, vec![]).code, "invalid_input", "{key}");
    }
}

#[test]
fn rotate_refuses() {
    let (x, m3) = pls3_fit(3);
    let w = field(rec_of(&fit(3)), "W").clone();
    let l = mat_of(vec![1.0; 60 * 3], 60, 3);

    let e = rotate_err(&w, vec![("X", x.clone()), ("L", l)]);
    assert_eq!(e.code, "invalid_argument");

    let e = rotate_err(&w, vec![("pre_standardized", Value::Bool(true))]);
    assert_eq!(e.code, "invalid_argument");

    // The PLS3 target is its stored x_scores, so there is nothing to standardize.
    let e = rotate_err(
        &m3,
        vec![("X", x.clone()), ("pre_standardized", Value::Bool(false))],
    );
    assert_eq!(e.code, "invalid_argument");

    // X with other rows than the PLS3Result was fit on.
    let (x50, _) = data(50, 6, 1);
    let e = rotate_err(&m3, vec![("X", x50)]);
    assert_eq!(e.code, "invalid_input");

    // X with other columns than W has rows (W is 6 x 3).
    let (x5, _) = data(60, 5, 1);
    let e = rotate_err(&w, vec![("X", x5.clone())]);
    assert_eq!(e.code, "shape_mismatch");
    let e = rotate_err(&m3, vec![("X", x5)]);
    assert_eq!(e.code, "shape_mismatch");

    // A hand-edited V with two columns while k_used is 3.
    let bad_v = mat_of(vec![0.5; 3 * 2], 3, 2);
    let e = rotate_err(&with_field(&m3, "V", bad_v), vec![("X", x)]);
    assert_eq!(e.code, "shape_mismatch");
}

fn get_mat(r: &Record<'static>, key: &str) -> Value<'static> {
    field(r, key).clone()
}

fn u_of(o: &plskit_bind::Outcome) -> Value<'static> {
    get_mat(record(o), "U")
}

#[test]
fn the_pls3_target_follows_x_and_l() {
    let (x, m3) = pls3_fit(3);
    let u = get_mat(rec_of(&m3), "U");
    let v = get_mat(rec_of(&m3), "V");

    // X gives L = the stored x_scores.
    let a = rotate(&m3, vec![("X", x)]);
    let b = rotate(&m3, vec![("L", get_mat(rec_of(&m3), "x_scores"))]);
    assert!(same(&a.result, &b.result));

    // No X and no L: the target is U itself, and it is not V.
    let none = rotate(&m3, vec![]);
    assert!(same(
        &u_of(&none),
        &u_of(&rotate(&m3, vec![("L", u.clone())]))
    ));
    assert!(!same(&u_of(&none), &u_of(&rotate(&m3, vec![("L", v)]))));
}

#[test]
fn the_matrix_form_takes_pre_standardized() {
    let (x, _) = data(60, 6, 1);
    let w = field(rec_of(&fit(3)), "W").clone();
    let a = rotate(
        &w,
        vec![("X", x.clone()), ("pre_standardized", Value::Bool(true))],
    );
    let b = rotate(&w, vec![("L", mat_of(matmul(&x, &w), 60, 3))]);
    assert!(same(&a.result, &b.result));
}

#[test]
fn non_finite_x_is_invalid_input_with_either_setting() {
    let w = field(rec_of(&fit(3)), "W").clone();
    let (x, _) = data(60, 6, 1);
    let mut d = floats(&x);
    d[0] = f64::NAN;
    let bad = mat_of(d, 60, 6);
    for extra in [vec![], vec![("pre_standardized", Value::Bool(true))]] {
        let mut f = vec![("X", bad.clone())];
        f.extend(extra);
        assert_eq!(rotate_err(&w, f).code, "invalid_input");
    }
}
