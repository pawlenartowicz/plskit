//! Integration tests for `factors`, `pca_basis` and `rotate`'s
//! `FactorsResult` overload.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{MatF64, Record, Value, VecF64, check_record};

fn vecv(v: Vec<f64>) -> Value<'static> {
    Value::Vec(VecF64::Owned(v))
}

/// Orthonormal columns by Gram-Schmidt on pseudo-random data, `d x k`.
fn basis(d: usize, k: usize, seed: u64) -> Value<'static> {
    let mut rng = Rng::new(seed);
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for _ in 0..k {
        let mut c: Vec<f64> = (0..d).map(|_| rng.next()).collect();
        for p in &cols {
            let dot: f64 = c.iter().zip(p).map(|(a, b)| a * b).sum();
            for (a, b) in c.iter_mut().zip(p) {
                *a -= dot * b;
            }
        }
        let nrm = c.iter().map(|a| a * a).sum::<f64>().sqrt();
        for a in &mut c {
            *a /= nrm;
        }
        cols.push(c);
    }
    Value::Mat(MatF64::Owned {
        data: cols.concat(),
        nrows: d,
        ncols: k,
    })
}

fn coef(d: usize, k: usize, seed: u64) -> Value<'static> {
    // In the span of `basis(d, k, seed)`.
    let b = floats(&basis(d, k, seed));
    let mut rng = Rng::new(seed + 100);
    let a: Vec<f64> = (0..k).map(|_| rng.next() + 1.5).collect();
    vecv(
        (0..d)
            .map(|i| (0..k).map(|j| b[i + j * d] * a[j]).sum())
            .collect(),
    )
}

fn make_factors(n: i64) -> Value<'static> {
    ok(
        "factors",
        vec![
            ("model_or_coef", coef(12, 5, 3)),
            ("X", data(30, 12, 3).0),
            ("n", Value::I64(n)),
        ],
    )
    .result
}

fn rec_of<'r>(v: &'r Value<'static>) -> &'r Record<'static> {
    match v {
        Value::Record(r) => r,
        other => panic!("expected a record, got {other:?}"),
    }
}

#[test]
fn factors_returns_a_checked_record() {
    let v = make_factors(3);
    let r = rec_of(&v);
    assert_eq!(r.type_name(), Some("FactorsResult"));
    check_record(r).unwrap();
    assert_eq!(
        r.keys().collect::<Vec<_>>(),
        [
            "F",
            "W",
            "size",
            "coef",
            "pre_standardized",
            "rotation_spec"
        ]
    );
    assert!(field(r, "rotation_spec").is_null());
    let (f, c) = (floats(field(r, "F")), floats(field(r, "coef")));
    for row in 0..12 {
        let sum: f64 = (0..3).map(|j| f[row + 12 * j]).sum();
        assert!((sum - c[row]).abs() < 1e-12);
    }
}

#[test]
fn pca_basis_returns_a_checked_record() {
    let (x, _) = data(40, 6, 2);
    let o = ok("pca_basis", vec![("X", x)]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("PCABasisResult"));
    check_record(r).unwrap();
    let Value::Mat(v) = field(r, "V") else {
        panic!()
    };
    assert_eq!((v.nrows(), v.ncols()), (6, 6));
    assert_eq!(floats(field(r, "s")).len(), 6);
}

#[test]
fn rotate_overload_returns_factors_with_a_spec() {
    let before = make_factors(3);
    let l = basis(3, 3, 9);
    let o = ok("rotate", vec![("model_or_W", before.clone()), ("L", l)]);
    let r = record(&o);
    assert_eq!(r.type_name(), Some("FactorsResult"));
    check_record(r).unwrap();
    let Value::Record(spec) = field(r, "rotation_spec") else {
        panic!("rotation_spec missing")
    };
    assert_eq!(spec.type_name(), Some("RotationSpec"));
    assert!(matches!(field(spec, "L_was_provided"), Value::Bool(true)));
    let Value::Record(args) = field(spec, "args") else {
        panic!()
    };
    assert!(matches!(field(args, "kaiser_normalize"), Value::Bool(_)));

    let w0 = floats(field(rec_of(&before), "W"));
    let w1 = floats(field(r, "W"));
    let rm = floats(field(spec, "R"));
    for row in 0..12 {
        for c in 0..3 {
            let want: f64 = (0..3).map(|j| w0[row + 12 * j] * rm[j + 3 * c]).sum();
            assert!((w1[row + 12 * c] - want).abs() < 1e-12);
        }
    }
    let size = floats(field(r, "size"));
    assert!(size.windows(2).all(|p| p[0] >= p[1]));
    // The factors still add up to coef.
    let (f, c) = (floats(field(r, "F")), floats(field(r, "coef")));
    for row in 0..12 {
        let sum: f64 = (0..3).map(|j| f[row + 12 * j]).sum();
        assert!((sum - c[row]).abs() < 1e-12);
    }
}

#[test]
fn rotate_on_rotated_factors_is_already_rotated() {
    let once = ok("rotate", vec![("model_or_W", make_factors(3))]).result;
    let e = err("rotate", vec![("model_or_W", once)]);
    assert_eq!(e.code, "already_rotated");
}

#[test]
fn rotate_with_mismatched_factors_is_shape_mismatch() {
    let v = make_factors(3);
    let mut r = rec_of(&v).clone();
    r.set("coef", vecv(vec![1.0; 5]));
    let e = err("rotate", vec![("model_or_W", Value::Record(r))]);
    assert_eq!(e.code, "shape_mismatch");
}

#[test]
fn rotate_refuses_an_untagged_factors_record() {
    let v = make_factors(3);
    let src = rec_of(&v);
    let mut r = Record::new();
    for k in src.keys() {
        r.push(k, field(src, k).clone()).unwrap();
    }
    let e = err("rotate", vec![("model_or_W", Value::Record(r))]);
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("FactorsResult"), "{}", e.message);
}

#[test]
fn rotate_factors_unknown_method() {
    let e = err(
        "rotate",
        vec![
            ("model_or_W", make_factors(3)),
            ("method", Value::text("oblimin")),
        ],
    );
    assert_eq!(e.code, "rotation_method_not_implemented");
}

// ---- profiles, profiles_search and profiles_find_n ----

fn text(s: &str) -> Value<'static> {
    Value::text(s)
}

fn noise_y(n: usize, seed: u64) -> Value<'static> {
    let mut rng = Rng::new(seed);
    vecv((0..n).map(|_| rng.next()).collect())
}

fn find_k_args() -> Value<'static> {
    Value::Record(rec(vec![
        ("selector", text("r2_se")),
        ("diagnostic", text("split_nb")),
        (
            "args",
            Value::Record(rec(vec![
                ("n_folds", Value::I64(5)),
                ("n_splits", Value::I64(50)),
            ])),
        ),
    ]))
}

fn rotation_r(r: &Record<'static>) -> Value<'static> {
    let Value::Record(spec) = field(r, "rotation_spec") else {
        panic!("rotation_spec missing")
    };
    field(spec, "R").clone()
}

fn fit(x: &Value<'static>, y: &Value<'static>, k: i64) -> plskit_bind::Outcome {
    ok(
        "pls1_fit",
        vec![("X", x.clone()), ("y", y.clone()), ("k", Value::I64(k))],
    )
}

fn record_of<'r>(v: &'r Value<'static>) -> &'r Record<'static> {
    match v {
        Value::Record(r) => r,
        other => panic!("expected a record, got {other:?}"),
    }
}

#[test]
fn profiles_equals_the_separate_calls() {
    let (x, y) = data(60, 8, 1);
    let m = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y),
            ("k", text("optimal")),
            ("k_max", Value::I64(5)),
            ("find_k_args", find_k_args()),
            ("seed", Value::U64(7)),
        ],
    );
    let coef_v = field(record(&m), "coef").clone();
    let xs = xstd(&x);
    let f = ok(
        "factors",
        vec![
            ("model_or_coef", coef_v),
            ("X", x.clone()),
            ("n", Value::I64(3)),
        ],
    );
    let w = field(record(&f), "W").clone();
    let l = Value::Mat(MatF64::Owned {
        data: matmul(&xs, &w),
        nrows: 60,
        ncols: 3,
    });
    let want = ok("rotate", vec![("model_or_W", f.result.clone()), ("L", l)]);

    let got = ok(
        "profiles",
        vec![
            ("model_or_coef", m.result.clone()),
            ("X", x),
            ("n", Value::I64(3)),
        ],
    );
    let (g, w) = (record(&got), record(&want));
    check_record(g).unwrap();
    assert_eq!(g.type_name(), Some("ProfilesResult"));
    assert!(same(field(g, "model"), &m.result));
    let gf = record_of(field(g, "factors"));
    for key in ["F", "W", "size", "coef"] {
        assert!(same(field(gf, key), field(w, key)), "{key}");
    }
    assert!(same(&rotation_r(gf), &rotation_r(w)));
}

#[test]
fn scores_sum_to_the_fit_score() {
    let (x, y) = data(60, 8, 2);
    let o = ok(
        "profiles",
        vec![
            ("model_or_coef", fit(&x, &y, 2).result),
            ("X", x.clone()),
            ("n", Value::I64(3)),
        ],
    );
    let r = record(&o);
    let coef_v = floats(field(record_of(field(r, "model")), "coef"));
    let xs = xstd(&x);
    let xd = floats(&xs);
    let scores = floats(field(r, "scores"));
    for row in 0..60 {
        let want: f64 = (0..8).map(|j| xd[row + 60 * j] * coef_v[j]).sum();
        let got: f64 = (0..3).map(|i| scores[row + 60 * i]).sum();
        assert!(
            (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "row {row}"
        );
    }
}

#[test]
fn profiles_follows_the_models_pre_standardized() {
    let (x, y) = data(40, 6, 7);
    // Scale 2, so standardizing it again would change the result.
    let x2 = Value::Mat(MatF64::Owned {
        data: floats(&xstd(&x)).iter().map(|v| 2.0 * v).collect(),
        nrows: 40,
        ncols: 6,
    });
    let m = ok(
        "pls1_fit",
        vec![
            ("X", x2.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("pre_standardized", Value::Bool(true)),
        ],
    );
    let o = ok(
        "profiles",
        vec![
            ("model_or_coef", m.result),
            ("X", x2.clone()),
            ("n", Value::I64(2)),
        ],
    );
    let r = record(&o);
    let want = matmul(&x2, field(record_of(field(r, "factors")), "F"));
    let got = floats(field(r, "scores"));
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()));
    }
}

/// Three blocks of correlated columns, with `y` built from two of the three
/// block traits, so the search has more than one axis to find.
fn trait_data(n: usize, seed: u64) -> (Value<'static>, Value<'static>) {
    let mut rng = Rng::new(seed);
    let z: Vec<f64> = (0..3 * n).map(|_| rng.next()).collect();
    let d = 9;
    let x: Vec<f64> = (0..n * d)
        .map(|c| z[c % n + n * (c / n % 3)] + 0.3 * rng.next())
        .collect();
    let y = (0..n)
        .map(|i| 2.0 * z[i] - z[i + n] + 0.3 * rng.next())
        .collect();
    (
        Value::Mat(MatF64::Owned {
            data: x,
            nrows: n,
            ncols: d,
        }),
        vecv(y),
    )
}

#[test]
fn search_is_profiles_at_n_star_plus_the_search() {
    let (x, y) = trait_data(120, 6);
    let m = fit(&x, &y, 2);
    let args = || {
        Value::Record(rec(vec![
            ("n_folds", Value::I64(3)),
            ("n_repeats", Value::I64(1)),
            ("n_perm", Value::I64(5)),
        ]))
    };
    let s = ok(
        "profiles_search",
        vec![
            ("model", m.result.clone()),
            ("X", x.clone()),
            ("y", y.clone()),
            ("n_max", Value::I64(3)),
            ("find_n_args", args()),
            ("seed", Value::U64(9)),
        ],
    );
    let want_search = ok(
        "profiles_find_n",
        vec![
            ("X", x.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("n_max", Value::I64(3)),
            ("args", args()),
            ("seed", Value::U64(9)),
        ],
    );
    let r = record(&s);
    check_record(r).unwrap();
    assert_eq!(r.type_name(), Some("ProfilesSearchResult"));
    assert!(same(field(r, "n_search"), &want_search.result));
    let n_star = i(record(&want_search), "n_star");
    // Above 1, so a search result that `profiles_search` ignored would show.
    assert!(n_star >= 2, "n_star {n_star}");
    let p = ok(
        "profiles",
        vec![
            ("model_or_coef", m.result),
            ("X", x),
            ("n", Value::I64(n_star)),
        ],
    );
    for key in ["model", "factors", "scores"] {
        assert!(same(field(r, key), field(record(&p), key)), "{key}");
    }
}

#[test]
fn search_with_no_signal_returns_coef() {
    let (x, _) = data(60, 8, 5);
    let y = noise_y(60, 11);
    let o = ok(
        "profiles_search",
        vec![
            ("model", fit(&x, &y, 1).result),
            ("X", x),
            ("y", y),
            ("seed", Value::U64(3)),
        ],
    );
    let r = record(&o);
    let ns = record_of(field(r, "n_search"));
    assert_eq!(i(ns, "n_star"), 0);
    let f = record_of(field(r, "factors"));
    let Value::Mat(fm) = field(f, "F") else {
        panic!()
    };
    assert_eq!(fm.ncols(), 1);
    let coef_v = floats(field(record_of(field(r, "model")), "coef"));
    // F = W * size, so it equals coef up to rounding.
    let fv = floats(field(f, "F"));
    let nrm = coef_v.iter().map(|c| c * c).sum::<f64>().sqrt();
    for (a, b) in fv.iter().zip(&coef_v) {
        assert!((a - b).abs() <= 1e-12 * nrm);
    }
}

#[test]
fn search_seed_is_recorded_and_reproduces() {
    let (x, y) = data(60, 8, 4);
    let m = fit(&x, &y, 2);
    let call = |seed: Option<u64>| {
        let mut f = vec![
            ("model", m.result.clone()),
            ("X", x.clone()),
            ("y", y.clone()),
        ];
        if let Some(s) = seed {
            f.push(("seed", Value::U64(s)));
        }
        ok("profiles_search", f)
    };
    let first = call(None);
    let Value::U64(s) = field(record(&first), "seed") else {
        panic!("seed not recorded")
    };
    let again = call(Some(*s));
    let mean = |o: &plskit_bind::Outcome| {
        floats(field(record_of(field(record(o), "n_search")), "r2_mean"))
    };
    assert_eq!(mean(&first), mean(&again));
}

#[test]
fn profiles_takes_no_search_mode() {
    let (x, y) = data(40, 6, 3);
    let e = err(
        "profiles",
        vec![
            ("model_or_coef", fit(&x, &y, 2).result),
            ("X", x),
            ("n", text("search")),
        ],
    );
    assert_eq!(e.code, "invalid_argument");
}

#[test]
fn rotation_other_than_varimax_raises_before_any_work() {
    let (x, y) = data(20, 4, 3);
    let m = fit(&x, &y, 1);
    let e = err(
        "profiles",
        vec![
            ("model_or_coef", m.result.clone()),
            ("X", x.clone()),
            ("n", Value::I64(2)),
            ("rotation", text("oblimin")),
        ],
    );
    assert_eq!(e.code, "rotation_method_not_implemented");
    let e = err(
        "profiles_search",
        vec![
            ("model", m.result),
            ("X", x),
            // Too short, so the search would raise something else first.
            ("y", vecv(vec![1.0; 3])),
            ("rotation", text("oblimin")),
        ],
    );
    assert_eq!(e.code, "rotation_method_not_implemented");
}

#[test]
fn unknown_find_n_args_key_is_invalid_args() {
    let (x, y) = data(60, 8, 3);
    let e = err(
        "profiles_search",
        vec![
            ("model", fit(&x, &y, 1).result),
            ("X", x),
            ("y", y),
            (
                "find_n_args",
                Value::Record(rec(vec![("bogus", Value::I64(1))])),
            ),
        ],
    );
    assert_eq!(e.code, "invalid_args");
    assert!(
        e.message
            .contains("find_n_args does not accept arg 'bogus'"),
        "{}",
        e.message
    );
}

#[test]
fn profiles_find_n_returns_a_checked_record() {
    let (x, y) = data(60, 8, 8);
    let o = ok(
        "profiles_find_n",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::I64(2)),
            ("seed", Value::U64(5)),
        ],
    );
    let r = record(&o);
    assert_eq!(r.type_name(), Some("ProfilesFindNResult"));
    check_record(r).unwrap();
    assert!(matches!(field(r, "seed"), Value::U64(5)));
}

#[test]
fn negative_counts_are_invalid_input() {
    let (x, y) = data(40, 6, 3);
    let m = fit(&x, &y, 1);
    for neg in [Value::I64(-1), Value::F64(-1.0)] {
        let e = err(
            "factors",
            vec![
                ("model_or_coef", coef(12, 5, 3)),
                ("X", data(30, 12, 3).0),
                ("n", neg.clone()),
            ],
        );
        assert_eq!(e.code, "invalid_input", "factors n={neg:?}");
        let e = err(
            "profiles",
            vec![
                ("model_or_coef", m.result.clone()),
                ("X", x.clone()),
                ("n", neg.clone()),
            ],
        );
        assert_eq!(e.code, "invalid_input", "profiles n={neg:?}");
        let e = err(
            "profiles_search",
            vec![
                ("model", m.result.clone()),
                ("X", x.clone()),
                ("y", y.clone()),
                ("n_max", neg.clone()),
            ],
        );
        assert_eq!(e.code, "invalid_input", "profiles_search n_max={neg:?}");
        let e = err(
            "profiles_find_n",
            vec![
                ("X", x.clone()),
                ("y", y.clone()),
                ("k", Value::I64(1)),
                ("n_max", neg.clone()),
            ],
        );
        assert_eq!(e.code, "invalid_input", "profiles_find_n n_max={neg:?}");
    }
    let e = err(
        "factors",
        vec![
            ("model_or_coef", coef(12, 5, 3)),
            ("X", data(30, 12, 3).0),
            ("n", Value::F64(1.5)),
        ],
    );
    assert_eq!(e.code, "invalid_argument");
}

#[test]
fn profiles_with_a_zero_coef_says_so() {
    let (x, _) = data(40, 6, 3);
    let y = vecv(vec![1.0; 40]);
    let m = fit(&x, &y, 1);
    let e = err(
        "profiles_search",
        vec![("model", m.result), ("X", x), ("y", y)],
    );
    assert!(e.message.contains("coef is zero"), "{}", e.message);
}

fn vals(v: &[f64]) -> Value<'static> {
    vecv(v.to_vec())
}

fn matv(data: Vec<f64>, nrows: usize) -> Value<'static> {
    let ncols = data.len() / nrows;
    Value::Mat(MatF64::Owned { data, nrows, ncols })
}

fn pre_standardized_of(o: &plskit_bind::Outcome) -> bool {
    matches!(field(record(o), "pre_standardized"), Value::Bool(true))
}

#[test]
fn the_two_forms_agree() {
    let (x, y) = data(60, 8, 1);
    let m = fit(&x, &y, 2);
    let coef_v = field(record(&m), "coef").clone();
    let a = ok(
        "profiles",
        vec![
            ("model_or_coef", m.result.clone()),
            ("X", x.clone()),
            ("n", Value::I64(3)),
        ],
    );
    let b = ok(
        "profiles",
        vec![("model_or_coef", coef_v), ("X", x), ("n", Value::I64(3))],
    );
    for key in ["factors", "scores"] {
        assert!(
            same(field(record(&a), key), field(record(&b), key)),
            "{key}"
        );
    }
    assert!(field(record(&b), "model").is_null());
    check_record(record(&b)).unwrap();
}

#[test]
fn pre_standardized_overrides_the_result() {
    // Unequal column scales, so standardizing or not changes the scores.
    let (x, y) = data(40, 6, 5);
    let scaled = matv(
        floats(&x)
            .iter()
            .enumerate()
            .map(|(i, v)| v * f64::from(1 + u8::try_from(i / 40).unwrap()))
            .collect(),
        40,
    );
    let call = |m: &Value<'static>, x: &Value<'static>, ps: Option<bool>| {
        let mut f = vec![
            ("model_or_coef", m.clone()),
            ("X", x.clone()),
            ("n", Value::I64(2)),
        ];
        if let Some(b) = ps {
            f.push(("pre_standardized", Value::Bool(b)));
        }
        ok("profiles", f)
    };
    let m = fit(&scaled, &y, 2);
    let dflt = call(&m.result, &scaled, None);
    let forced = call(&m.result, &scaled, Some(true));
    let f = record_of(field(record(&forced), "factors"));
    assert!(matches!(field(f, "pre_standardized"), Value::Bool(true)));
    let want = matmul(&scaled, field(f, "F"));
    let got = floats(field(record(&forced), "scores"));
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()));
    }
    let base = floats(field(record(&dflt), "scores"));
    let gap = got
        .iter()
        .zip(&base)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(gap > 1e-6, "gap {gap}");

    // A fit made on standardized data, told to standardize.
    let xs = xstd(&scaled);
    let m2 = ok(
        "pls1_fit",
        vec![
            ("X", xs.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("pre_standardized", Value::Bool(true)),
        ],
    );
    let off = call(&m2.result, &scaled, Some(false));
    let f = record_of(field(record(&off), "factors"));
    assert!(matches!(field(f, "pre_standardized"), Value::Bool(false)));
    let want = matmul(&xs, field(f, "F"));
    for (a, b) in floats(field(record(&off), "scores")).iter().zip(&want) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()));
    }
}

#[test]
fn null_follows_the_result_and_false_for_an_array() {
    let (x, y) = data(40, 6, 4);
    let xs = xstd(&x);
    let m = ok(
        "pls1_fit",
        vec![
            ("X", xs.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("pre_standardized", Value::Bool(true)),
        ],
    );
    let from_model = ok(
        "factors",
        vec![
            ("model_or_coef", m.result.clone()),
            ("X", xs),
            ("n", Value::I64(3)),
        ],
    );
    assert!(pre_standardized_of(&from_model));
    let from_vec = ok(
        "factors",
        vec![
            ("model_or_coef", field(record(&m), "coef").clone()),
            ("X", x),
            ("n", Value::I64(3)),
        ],
    );
    assert!(!pre_standardized_of(&from_vec));
}

#[test]
fn a_sparse_fit_is_split() {
    let (x, y) = data(60, 10, 2);
    let m = ok(
        "spls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y),
            ("k", Value::I64(1)),
            ("keep", Value::I64(4)),
        ],
    );
    let c = floats(field(record(&m), "coef"));
    assert!(c.contains(&0.0));
    let o = ok(
        "factors",
        vec![
            ("model_or_coef", m.result.clone()),
            ("X", x.clone()),
            ("n", Value::I64(2)),
        ],
    );
    let f = floats(field(record(&o), "F"));
    // A zero row stays zero under rotation, so `profiles` keeps it too.
    let p = ok(
        "profiles",
        vec![("model_or_coef", m.result), ("X", x), ("n", Value::I64(2))],
    );
    let Value::Record(pf) = field(record(&p), "factors") else {
        panic!()
    };
    let g = floats(field(pf, "F"));
    for (j, cj) in c.iter().enumerate() {
        if *cj == 0.0 {
            assert!(f[j] == 0.0 && f[j + 10] == 0.0, "factors row {j}");
            assert!(g[j] == 0.0 && g[j + 10] == 0.0, "profiles row {j}");
        }
    }
}

#[test]
fn a_wide_sparse_fit_is_split_and_rotated() {
    // 20 rows, more than 19 variables kept: part of coef is outside the row
    // space of X. At n = 20 that part is the last factor alone, so the
    // rotation target `Xs W` has a zero column.
    let (x, y) = data(20, 200, 5);
    let m = ok(
        "spls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y),
            ("k", Value::I64(2)),
            ("keep", Value::I64(40)),
        ],
    )
    .result;
    let f = ok(
        "factors",
        vec![
            ("model_or_coef", m.clone()),
            ("X", x.clone()),
            ("n", Value::I64(20)),
        ],
    )
    .result;
    let (fd, c) = (
        floats(field(rec_of(&f), "F")),
        floats(field(rec_of(&f), "coef")),
    );
    let nrm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    for (row, cr) in c.iter().enumerate() {
        let sum: f64 = (0..20).map(|j| fd[row + 200 * j]).sum();
        assert!((sum - cr).abs() <= 1e-12 * nrm, "row {row}");
    }
    let r = ok("rotate", vec![("model_or_W", f), ("X", x.clone())]);
    for key in ["F", "W", "size"] {
        assert!(
            floats(field(record(&r), key)).iter().all(|v| v.is_finite()),
            "{key}"
        );
    }
    let p = ok(
        "profiles",
        vec![("model_or_coef", m), ("X", x), ("n", Value::I64(20))],
    );
    check_record(record(&p)).unwrap();
}

#[test]
#[allow(clippy::float_cmp)] // exact zero vs nonzero coef
fn a_constant_column_is_harmless() {
    let (x, y) = data(40, 6, 9);
    let with_col = |v: f64| {
        let mut d = floats(&x);
        for r in 0..40 {
            d[r + 40 * 2] = v;
        }
        matv(d, 40)
    };
    let factors_of = |xc: &Value<'static>| {
        let m = fit(xc, &y, 2);
        let c = floats(field(record(&m), "coef"));
        let o = ok(
            "profiles",
            vec![
                ("model_or_coef", m.result),
                ("X", xc.clone()),
                ("n", Value::I64(2)),
            ],
        );
        (
            c,
            floats(field(record_of(field(record(&o), "factors")), "F")),
        )
    };
    // Nonzero constant: coef at rounding level, the column stays in.
    let (c, f) = factors_of(&with_col(3.0));
    assert_ne!(c[2], 0.0);
    let nrm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(f[2].abs().max(f[2 + 6].abs()) < 1e-12 * nrm);
    for r in 0..6 {
        assert!((f[r] + f[r + 6] - c[r]).abs() <= 1e-12 * nrm, "row {r}");
    }
    // All-zero column: an exact zero in coef and in F.
    let (c, f) = factors_of(&with_col(0.0));
    assert_eq!(c[2], 0.0);
    assert!(f[2] == 0.0 && f[2 + 6] == 0.0);
}

#[test]
fn a_one_variable_coef_is_read_as_a_vector() {
    let x = matv((0..8).map(|i| f64::from(i) - 3.5).collect(), 8);
    let o = ok(
        "factors",
        vec![
            ("model_or_coef", Value::F64(2.0)),
            ("X", x),
            ("n", Value::I64(1)),
        ],
    );
    assert_eq!(floats(field(record(&o), "F")), [2.0]);
}

#[test]
fn factors_refuses() {
    let (x, y) = data(40, 6, 3);
    let call = |name: &'static str, first: Value<'static>, x: &Value<'static>, n: i64| {
        err(
            name,
            vec![
                ("model_or_coef", first),
                ("X", x.clone()),
                ("n", Value::I64(n)),
            ],
        )
    };
    let dense = fit(&x, &y, 2);
    let c = field(record(&dense), "coef").clone();
    let weights = vecv((0..40).map(|i| 1.0 + f64::from(i % 2)).collect());
    let weighted = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(1)),
            ("weights", weights.clone()),
        ],
    );
    let weighted_sparse = ok(
        "spls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(1)),
            ("keep", Value::I64(3)),
            ("weights", weights),
        ],
    );
    let flat = fit(&x, &vecv(vec![1.0; 40]), 1);
    let p3 = ok(
        "pls3_fit",
        vec![
            ("X", x.clone()),
            ("Y", block_y(40, 2, &x, 4)),
            ("k", Value::I64(1)),
        ],
    );
    let (other, _) = data(30, 6, 3);
    // Two equal columns and an orthogonal one: coef has a 1e-14 share on the
    // second component.
    let ab = matv(
        vec![
            1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        ],
        4,
    );
    for name in ["factors", "profiles"] {
        for m in [&weighted, &weighted_sparse] {
            let e = call(name, m.result.clone(), &x, 1);
            assert_eq!(e.code, "invalid_input", "{name} weighted");
            assert!(e.message.contains("weights"), "{}", e.message);
        }
        let e = call(name, flat.result.clone(), &x, 1);
        assert_eq!(e.code, "invalid_input", "{name} k_used 0");
        assert!(e.message.contains("kept no component"), "{}", e.message);
        let e = call(name, vals(&[0.0; 6]), &x, 1);
        assert_eq!(e.code, "invalid_input", "{name} zero vector");
        let e = call(name, dense.result.clone(), &other, 1);
        assert_eq!(e.code, "invalid_input", "{name} rows differ");
        assert!(e.message.contains("X has 30 rows"), "{}", e.message);
        let mut bad = floats(&c);
        bad[1] = f64::NAN;
        let e = call(name, vecv(bad), &x, 1);
        assert_eq!(e.code, "invalid_input", "{name} coef not finite");
        assert!(e.message.contains("non-finite"), "{}", e.message);
        let mut xb = floats(&x);
        xb[5] = f64::INFINITY;
        let e = call(name, c.clone(), &matv(xb, 40), 1);
        assert_eq!(e.code, "invalid_input", "{name} X not finite");
        assert!(e.message.contains("non-finite"), "{}", e.message);
        let e = call(name, vals(&[1.0, 2.0]), &x, 1);
        assert_eq!(e.code, "shape_mismatch", "{name} coef length");
        let e = call(name, p3.result.clone(), &x, 1);
        assert_eq!(e.code, "invalid_argument", "{name} PLS3Result");
        for n in [0, 7] {
            let e = call(name, c.clone(), &x, n);
            assert_eq!(e.code, "invalid_input", "{name} n={n}");
        }
        let e = call(name, vals(&[1.0, 1.0, 1e-14]), &ab, 2);
        assert_eq!(e.code, "invalid_input", "{name} empty block");
        assert!(e.message.contains("no signal"), "{}", e.message);
    }
}

#[test]
fn profiles_search_still_refuses_sparse() {
    let (x, y) = data(40, 6, 3);
    let sparse = ok(
        "spls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(1)),
            ("keep", Value::I64(3)),
        ],
    );
    let e = err(
        "profiles_search",
        vec![("model", sparse.result), ("X", x.clone()), ("y", y.clone())],
    );
    assert_eq!(e.code, "invalid_input");
    assert!(e.message.contains("keep"), "{}", e.message);

    let weighted = ok(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(1)),
            (
                "weights",
                vecv((0..40).map(|i| 1.0 + f64::from(i % 2)).collect()),
            ),
        ],
    );
    let e = err(
        "profiles_search",
        vec![("model", weighted.result), ("X", x), ("y", y)],
    );
    assert_eq!(e.code, "invalid_input");
    assert!(e.message.contains("weights"), "{}", e.message);
}
