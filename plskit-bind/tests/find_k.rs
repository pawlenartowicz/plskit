//! K selection (`*_find_k_optimal`, `pls1_find_k_sequence`,
//! `spls1_find_keep_optimal`) and `pls1_fit`'s string `k`.
#![allow(clippy::many_single_char_names)]

mod common;

use common::*;
use plskit_bind::{check_record, Value, VecF64};

fn args(fields: Vec<(&str, Value<'static>)>) -> Value<'static> {
    Value::Record(rec(fields))
}

fn xy() -> (Value<'static>, Value<'static>) {
    data(80, 6, 1)
}

fn optimal(
    extra: Vec<(&'static str, Value<'static>)>,
) -> Result<plskit_bind::Outcome, plskit_bind::BindError> {
    let (x, y) = xy();
    let mut v = vec![
        ("X", x),
        ("y", y),
        ("k_max", Value::I64(4)),
        ("seed", Value::U64(1)),
    ];
    v.extend(extra);
    plskit_bind::call("pls1_find_k_optimal", rec(v))
}

#[test]
fn r2_se_scores_cover_one_to_k_max() {
    let o = optimal(vec![]).unwrap();
    let r = record(&o);
    check_record(r).unwrap();
    let Value::IntMap(cv) = field(r, "cv_scores") else {
        panic!()
    };
    assert_eq!(cv.iter().map(|p| p.0).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert!(field(r, "bic_scores").is_null());
}

#[test]
fn optimal_args_rules_match_plskit_py() {
    let cases: Vec<(Vec<(&'static str, Value<'static>)>, &str)> = vec![
        (
            vec![
                ("selector", Value::text("bic")),
                ("args", args(vec![("n_folds", Value::I64(5))])),
            ],
            "selector='bic' does not accept arg 'n_folds'",
        ),
        (
            vec![("args", args(vec![("n_perm", Value::I64(100))]))],
            "args['n_perm'] requires diagnostic to be set",
        ),
        (
            vec![
                ("diagnostic", Value::text("raw_perm")),
                ("args", args(vec![("n_splits", Value::I64(10))])),
            ],
            "args['n_splits'] only valid for diagnostic in {split_nb, split_exact}",
        ),
        (
            vec![
                ("diagnostic", Value::text("split_exact")),
                ("args", args(vec![("force", Value::Bool(true))])),
            ],
            "args['force'] only valid for diagnostic='split_nb'",
        ),
        (
            vec![("diagnostic", Value::text("split_perm"))],
            "unknown method: split_perm",
        ),
        (
            vec![("selector", Value::text("cv"))],
            "unknown selector: cv",
        ),
    ];
    for (extra, want) in cases {
        let e = optimal(extra).unwrap_err();
        assert_eq!((e.code, e.message.as_str()), ("invalid_args", want));
    }
}

#[test]
fn sequence_rejects_score_and_runs_split_exact() {
    let (x, y) = xy();
    let base = || {
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k_max", Value::I64(3)),
            ("seed", Value::U64(1)),
        ]
    };
    let mut v = base();
    v.push(("test_method", Value::text("score")));
    let e = err("pls1_find_k_sequence", v);
    assert_eq!(
        (e.code, e.message.as_str()),
        (
            "invalid_args",
            "test_method='score' has no sequential variant"
        )
    );

    let mut v = base();
    v.push(("test_method", Value::text("split_exact")));
    v.push((
        "args",
        args(vec![
            ("n_perm", Value::I64(100)),
            ("n_splits", Value::I64(10)),
        ]),
    ));
    let o = ok("pls1_find_k_sequence", v);
    let r = record(&o);
    check_record(r).unwrap();
    assert_eq!(floats(field(r, "pvalues")).len(), 3);
}

#[test]
fn a_rerouted_diagnostic_warns() {
    let (x, y) = data(60, 3, 2);
    let o = ok(
        "pls1_find_k_optimal",
        vec![
            ("X", x),
            ("y", y),
            ("k_max", Value::I64(2)),
            ("diagnostic", Value::text("split_nb")),
            ("seed", Value::U64(1)),
        ],
    );
    assert_eq!(s(record(&o), "diagnostic"), "split_exact");
    assert_eq!(o.warnings.len(), 1);
    assert_eq!(i(&o.warnings[0], "n_perm"), 1000);
}

#[test]
fn pls1_fit_optimal_attaches_the_selection() {
    let (x, y) = xy();
    let o = ok(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::text("optimal")),
            ("k_max", Value::I64(4)),
            ("seed", Value::U64(1)),
        ],
    );
    let r = record(&o);
    check_record(r).unwrap();
    let Value::Record(sel) = field(r, "selection_result") else {
        panic!()
    };
    assert_eq!(sel.type_name(), Some("FindKOptimalResult"));
    assert_eq!(i(r, "k_used"), i(sel, "k_star"));
}

#[test]
fn pls1_fit_mode_errors() {
    let (x, y) = xy();
    let e = err(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::text("optimal")),
        ],
    );
    assert_eq!(
        (e.code, e.message.as_str()),
        ("invalid_argument", "k='optimal' requires k_max")
    );
    let e = err(
        "pls1_fit",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::text("bogus")),
            ("k_max", Value::I64(3)),
        ],
    );
    assert_eq!(
        e.message,
        "unknown k mode 'bogus'; use int, 'optimal', or 'sequence'"
    );
    let e = err(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::text("optimal")),
            ("k_max", Value::I64(3)),
            ("find_k_args", args(vec![("bogus", Value::I64(1))])),
        ],
    );
    assert_eq!(e.code, "invalid_args");
    assert_eq!(
        e.message,
        "find_k_args does not accept arg 'bogus'; allowed: ['selector', 'diagnostic', 'args']"
    );
}

#[test]
fn pls1_fit_sequence_with_no_rejection_raises_the_engine_code() {
    let (x, _) = xy();
    let mut rng = Rng::new(99);
    let noise = Value::Vec(VecF64::Owned((0..80).map(|_| rng.next()).collect()));
    // split_exact with 100 permutations cannot reach p < 1/101, so alpha = 1e-12 rejects nothing.
    let e = err(
        "pls1_fit",
        vec![
            ("X", x),
            ("y", noise),
            ("k", Value::text("sequence")),
            ("k_max", Value::I64(3)),
            ("seed", Value::U64(1)),
            (
                "find_k_args",
                args(vec![
                    ("test_method", Value::text("split_exact")),
                    ("alpha", Value::F64(1e-12)),
                    (
                        "args",
                        args(vec![
                            ("n_perm", Value::I64(100)),
                            ("n_splits", Value::I64(10)),
                        ]),
                    ),
                ]),
            ),
        ],
    );
    assert_eq!(e.code, "sequence_no_rejection");
}

#[test]
fn sparse_selection_at_keep_equal_p_is_the_dense_one() {
    let (x, y) = xy();
    let dense = ok(
        "pls1_find_k_optimal",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k_max", Value::I64(3)),
            ("seed", Value::U64(4)),
        ],
    );
    let sparse = ok(
        "spls1_find_k_optimal",
        vec![
            ("X", x),
            ("y", y),
            ("k_max", Value::I64(3)),
            ("keep", Value::I64(6)),
            ("seed", Value::U64(4)),
        ],
    );
    assert!(same(&dense.result, &sparse.result));
}

#[test]
fn keep_optimal_reports_its_grid() {
    let (x, y) = xy();
    let o = ok(
        "spls1_find_keep_optimal",
        vec![
            ("X", x.clone()),
            ("y", y.clone()),
            ("k", Value::I64(1)),
            ("seed", Value::U64(1)),
        ],
    );
    let r = record(&o);
    check_record(r).unwrap();
    assert!(matches!(field(r, "keep_grid"), Value::IntVec(g) if g.first() == Some(&1)));
    let e = err(
        "spls1_find_keep_optimal",
        vec![
            ("X", x),
            ("y", y),
            ("k", Value::I64(1)),
            ("args", args(vec![("n_perm", Value::I64(1))])),
        ],
    );
    assert_eq!(e.code, "invalid_args");
}
