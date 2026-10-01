//! Integration tests for the confirmatory-test family, permutation null
//! and rotation stability through `call`.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{check_record, Value, VecF64};

fn args(fields: Vec<(&str, Value<'static>)>) -> Value<'static> {
    Value::Record(rec(fields))
}

fn exact_args() -> Value<'static> {
    args(vec![
        ("n_perm", Value::I64(100)),
        ("n_splits", Value::I64(10)),
    ])
}

fn confirm(extra: Vec<(&'static str, Value<'static>)>) -> Vec<(&'static str, Value<'static>)> {
    let (x, y) = data(60, 6, 1);
    let mut v = vec![("X", x), ("y", y)];
    v.extend(extra);
    v
}

#[test]
fn split_exact_records_its_seed_and_reproduces_from_it() {
    let a = ok(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_exact")),
            ("args", exact_args()),
        ]),
    );
    let ra = record(&a);
    check_record(ra).unwrap();
    let Value::U64(seed) = field(ra, "seed") else {
        panic!("seed must be U64")
    };
    let seed = *seed;
    let b = ok(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_exact")),
            ("args", exact_args()),
            ("seed", Value::U64(seed)),
        ]),
    );
    assert!(same(&a.result, &b.result));
    let c = ok(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_exact")),
            ("args", exact_args()),
            ("seed", Value::text(&seed.to_string())),
        ]),
    );
    assert!(same(&a.result, &c.result));
}

// Seeds above 2^63 travel as decimal strings (R's form).
#[test]
fn a_seed_above_2_63_is_recorded_exactly() {
    let o = ok(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_exact")),
            ("args", exact_args()),
            ("seed", Value::text("18446744073709551615")),
        ]),
    );
    assert!(matches!(field(record(&o), "seed"), Value::U64(u64::MAX)));
}

#[test]
fn bad_methods_and_args_are_invalid_args() {
    let e = err(
        "pls1_confirmatory_test",
        confirm(vec![("test_method", Value::text("split_perm"))]),
    );
    assert_eq!(
        (e.code, e.message.as_str()),
        ("invalid_args", "unknown method: split_perm")
    );
    let e = err(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("raw_perm")),
            ("args", args(vec![("n_splits", Value::I64(3))])),
        ]),
    );
    assert_eq!(e.code, "invalid_args");
    let e = err(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_exact")),
            ("args", args(vec![("n_perm", Value::F64(2.5))])),
        ]),
    );
    assert_eq!(e.code, "invalid_args");
    let e = err(
        "pls1_confirmatory_test",
        confirm(vec![("test_method", Value::I64(1))]),
    );
    assert_eq!(e.code, "invalid_argument");
}

#[test]
fn ci_knobs_are_inert_without_ci_and_fill_the_bundle_with_it() {
    let off = ok(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_nb")),
            ("n_boot", Value::I64(50)),
            ("seed", Value::U64(1)),
        ]),
    );
    assert!(field(record(&off), "ci").is_null());
    let on = ok(
        "pls1_confirmatory_test",
        confirm(vec![
            ("test_method", Value::text("split_nb")),
            ("ci", Value::Bool(true)),
            ("n_boot", Value::I64(100)),
            ("seed", Value::U64(1)),
        ]),
    );
    let r = record(&on);
    check_record(r).unwrap();
    let Value::Record(ci) = field(r, "ci") else {
        panic!("ci missing")
    };
    assert_eq!(i(ci, "n_boot"), 100);
}

#[test]
fn a_rerouted_split_nb_warns_with_the_python_sentence() {
    let (x, y) = data(60, 3, 2); // 3 columns: the gate fires on ncols <= 4
    let o = ok(
        "pls1_confirmatory_test",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("test_method", Value::text("split_nb")),
            ("seed", Value::U64(3)),
        ],
    );
    let r = record(&o);
    assert_eq!(s(r, "test_method"), "split_exact");
    assert_eq!(o.warnings.len(), 1);
    let w = &o.warnings[0];
    assert_eq!(s(w, "kind"), "rerouted");
    assert_eq!(s(w, "requested"), "split_nb");
    assert_eq!(s(w, "actual"), "split_exact");
    let msg = s(w, "message");
    assert!(
        msg.starts_with(
            "'split_nb' was rerouted to 'split_exact': the split_nb auto-gate flagged this design \
             (stable rank of the standardized X = "
        ),
        "{msg}"
    );
    assert!(
        msg.contains("; n_eff = 60). The fallback runs n_perm="),
        "{msg}"
    );
    assert!(
        msg.ends_with("Pass args={'force': True} to run split_nb anyway."),
        "{msg}"
    );

    let forced = ok(
        "pls1_confirmatory_test",
        vec![
            ("X", x),
            ("y", y),
            ("test_method", Value::text("split_nb")),
            ("args", args(vec![("force", Value::Bool(true))])),
            ("seed", Value::U64(3)),
        ],
    );
    assert_eq!(s(record(&forced), "test_method"), "split_nb");
    assert!(forced.warnings.is_empty());
}

#[test]
fn pls3_confirmatory_test_runs_two_methods_and_never_a_ci() {
    let (x, _) = data(60, 6, 1);
    let y = block_y(60, 3, &x, 4);
    let o = ok(
        "pls3_confirmatory_test",
        vec![
            ("X", x.clone()),
            ("Y", y.clone()),
            ("test_method", Value::text("split_exact")),
            ("args", exact_args()),
        ],
    );
    let r = record(&o);
    check_record(r).unwrap();
    assert!(field(r, "ci").is_null());
    let e = err(
        "pls3_confirmatory_test",
        vec![
            ("X", x.clone()),
            ("Y", y),
            ("test_method", Value::text("raw_perm")),
        ],
    );
    assert_eq!(e.code, "invalid_args");
    let y1 = Value::Vec(VecF64::Owned(vec![0.0; 60]));
    let e = err(
        "pls3_confirmatory_test",
        vec![
            ("X", x),
            ("Y", y1),
            ("test_method", Value::text("split_exact")),
        ],
    );
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("PLS1 problem"), "{}", e.message);
}

#[test]
fn perm_null_matrix_matches_the_engine_row_for_row() {
    let (x, y) = data(60, 6, 1);
    let o = ok(
        "pls1_perm_null",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(2)),
            ("n_perm", Value::I64(100)),
            ("return_perm_matrix", Value::Bool(true)),
            ("seed", Value::U64(5)),
        ],
    );
    let r = record(&o);
    check_record(r).unwrap();
    let Value::Mat(m) = field(r, "beta_perm_matrix") else {
        panic!()
    };
    assert_eq!((m.nrows(), m.ncols()), (100, 6));
    let (Value::Mat(xm), Value::Vec(yv)) = (&x, &y) else {
        panic!()
    };
    let out = plskit::pls1_perm_null(
        xm.as_mat(),
        yv.as_col(),
        2,
        None,
        plskit::PermNullOpts {
            n_perm: 100,
            return_perm_matrix: true,
            ..plskit::PermNullOpts::default()
        },
        Some(5),
    )
    .unwrap();
    let flat = out.beta_perm_matrix.unwrap();
    for row in 0..2 {
        for j in 0..6 {
            assert_eq!(m.as_mat()[(row, j)].to_bits(), flat[row * 6 + j].to_bits());
        }
    }
}

#[test]
fn rotation_stability_record_and_errors() {
    let (x, y) = data(80, 6, 1);
    let base = || {
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(2)),
            ("n_boot", Value::I64(100)),
            ("seed", Value::U64(3)),
        ]
    };
    let o = ok("pls1_rotation_stability", base());
    let r = record(&o);
    check_record(r).unwrap();
    assert!(matches!(field(r, "variance_ratio_per_axis"), Value::List(l) if l.len() == 2));

    let mut v = base();
    v.push(("rotation_method", Value::text("promax")));
    assert_eq!(
        err("pls1_rotation_stability", v).code,
        "rotation_method_not_implemented"
    );

    let mut v = base();
    v.push(("rotation_args", args(vec![("bogus", Value::I64(1))])));
    assert_eq!(err("pls1_rotation_stability", v).code, "invalid_args");
}
