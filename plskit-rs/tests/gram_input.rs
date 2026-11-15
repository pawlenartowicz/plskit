//! The held-out tests and `pls1_perm_alpha` on a caller's Gram matrix
//! (`gram_matrix`'s output): every method of `pls1_confirmatory_test` and
//! `pls3_confirmatory_test` from G equals the call on X at the same seed,
//! a row subsample of G equals the call on those rows scaled over all rows,
//! and the refusals of the `gram` form.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
// Exact comparisons are intended where the two calls share the value's bits.
#![allow(clippy::float_cmp)]

use faer::{Col, ColRef, Mat, MatRef};
use plskit::linalg::{standardize, standardize1};
use plskit::{
    ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput, FitOpts,
    GramMatrixOutput, KSpec, Pls3ConfirmatoryTestOpts, PlsKitError, gram_matrix,
    pls1_confirmatory_test, pls1_fit, pls1_perm_alpha, pls1_perm_alpha_gram,
    pls3_confirmatory_test, pls3_confirmatory_test_gram,
};

/// `SplitMix64` mapped to `[-1, 1)`.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }
}

/// `n × p` X, column `j` scaled by `1 + j/10` and shifted by `1 + j/20`,
/// and a `y` carrying a signal on the first two columns.
fn data(n: usize, p: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
    let mut s = Stream(seed);
    let z = Mat::<f64>::from_fn(n, p, |_, _| s.next());
    let x = Mat::<f64>::from_fn(n, p, |i, j| {
        z[(i, j)] * (1.0 + j as f64 / 10.0) + 1.0 + j as f64 / 20.0
    });
    let y = Col::<f64>::from_fn(n, |i| 2.0 * (z[(i, 0)] + 0.5 * z[(i, 1)]) + s.next());
    (x, y)
}

/// A three-column Y sharing the signal of `x`'s first two columns.
fn outcomes(x: MatRef<'_, f64>, seed: u64) -> Mat<f64> {
    let mut s = Stream(seed);
    Mat::<f64>::from_fn(x.nrows(), 3, |i, j| {
        (j as f64 + 1.0) * x[(i, j % 2)] + 2.0 * s.next()
    })
}

/// `x` with each column centred by its own mean.
fn centred(x: MatRef<'_, f64>) -> Mat<f64> {
    let n = x.nrows() as f64;
    let mean: Vec<f64> = (0..x.ncols())
        .map(|j| (0..x.nrows()).map(|i| x[(i, j)]).sum::<f64>() / n)
        .collect();
    Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] - mean[j])
}

/// The PLS1 methods, each with its `k`.
fn pls1_methods(k: usize) -> Vec<ConfirmatoryArgs> {
    vec![
        ConfirmatoryArgs::RawPerm {
            n_perm: 19,
            n_folds: 5,
        },
        ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: false,
        },
        ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: true,
        },
        ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: 8,
        },
        ConfirmatoryArgs::Auto {
            n_perm: 19,
            n_splits: 8,
        },
        ConfirmatoryArgs::E,
    ]
    .into_iter()
    .chain((k == 1).then_some(ConfirmatoryArgs::Score { n_perm: None }))
    .collect()
}

fn pls3_methods() -> [ConfirmatoryArgs; 4] {
    [
        ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: 8,
        },
        ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: false,
        },
        ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: true,
        },
        ConfirmatoryArgs::Auto {
            n_perm: 19,
            n_splits: 8,
        },
    ]
}

fn opts(args: ConfirmatoryArgs, pre_standardized: bool) -> ConfirmatoryTestOpts {
    ConfirmatoryTestOpts {
        args,
        pre_standardized,
        seed: Some(7),
        ..ConfirmatoryTestOpts::default()
    }
}

fn pls3_opts(args: ConfirmatoryArgs, pre_standardized_x: bool) -> Pls3ConfirmatoryTestOpts {
    Pls3ConfirmatoryTestOpts {
        args,
        pre_standardized_x,
        seed: Some(7),
        ..Pls3ConfirmatoryTestOpts::default()
    }
}

fn on_x(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    o: ConfirmatoryTestOpts,
) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x,
            y,
            k: Some(&[k]),
            weights: None,
            covariates: None,
            null_y: None,
        },
        o,
    )
    .expect("X call")
}

fn on_gram(
    g: &GramMatrixOutput,
    y: ColRef<'_, f64>,
    k: usize,
    o: ConfirmatoryTestOpts,
) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Gram {
            gram: g,
            y,
            k: Some(&[k]),
            covariates: None,
            null_y: None,
        },
        o,
    )
    .expect("gram call")
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

/// The fields the two calls must share: statistic and p-value within
/// `1e-12` relative, the stable rank too, the rest exactly. `None` when
/// they agree, else what differs.
fn differs(a: &ConfirmatoryTestOutput, b: &ConfirmatoryTestOutput) -> Option<String> {
    let rank = match (a.stable_rank, b.stable_rank) {
        (Some(s), Some(t)) => close(s, t),
        (s, t) => s.is_none() && t.is_none(),
    };
    let same = close(a.statistic.unwrap(), b.statistic.unwrap())
        && close(a.pvalue, b.pvalue)
        && rank
        && a.test_method == b.test_method
        && (&a.k, a.n_perm, a.n_splits, a.seed) == (&b.k, b.n_perm, b.n_splits, b.seed)
        && a.n_eff == b.n_eff
        && a.rho_hat.is_some() == b.rho_hat.is_some()
        && a.rho_hat.zip(b.rho_hat).is_none_or(|(s, t)| close(s, t));
    (!same).then(|| format!("X {a:?}\n    gram {b:?}"))
}

/// On wide and tall X, both flags, `k` = 1 and 2, every PLS1 method from G
/// equals the call on X, and so does every dense PLS3 method. The gram call
/// reads the flag from `g`. Under the flag X is centred first: `score` reads
/// X as given there, and G is always centred. A PLS3 call near a tie of the
/// sparse Y side may differ between routes; none of these calls is sparse.
/// The 1100 x 300 shape takes `auto` and the `split_nb` gate past their
/// `n_eff` rules, to the stable rank they read from G.
#[test]
fn gram_call_equals_x_call() {
    for (n, p) in [(30, 80), (60, 8)] {
        let (x0, y) = data(n, p, 11);
        let yy = outcomes(x0.as_ref(), 12);
        for flag in [false, true] {
            let x = if flag {
                centred(x0.as_ref())
            } else {
                x0.clone()
            };
            let g = gram_matrix(x.as_ref(), flag).expect("gram_matrix");
            for k in [1, 2] {
                for args in pls1_methods(k) {
                    let what = format!("{n}x{p} flag={flag} k={k} {args:?}");
                    let a = on_x(x.as_ref(), y.as_ref(), k, opts(args, flag));
                    let b = on_gram(&g, y.as_ref(), k, opts(args, false));
                    if let Some(d) = differs(&a, &b) {
                        panic!("{what}:\n    {d}");
                    }
                }
            }
            for args in pls3_methods() {
                let a = pls3_confirmatory_test(x.as_ref(), yy.as_ref(), 1, pls3_opts(args, flag))
                    .expect("PLS3 X call");
                let b = pls3_confirmatory_test_gram(&g, yy.as_ref(), 1, pls3_opts(args, false))
                    .expect("PLS3 gram call");
                if let Some(d) = differs(&a, &b) {
                    panic!("PLS3 {n}x{p} flag={flag} {args:?}:\n    {d}");
                }
            }
        }
    }
    let (x, y) = data(1100, 300, 13);
    let g = gram_matrix(x.as_ref(), false).expect("gram_matrix");
    let gated = [
        ConfirmatoryArgs::Auto {
            n_perm: 19,
            n_splits: 4,
        },
        ConfirmatoryArgs::SplitNb {
            n_splits: 4,
            force: false,
        },
    ];
    for k in [1, 2] {
        for args in gated {
            let a = on_x(x.as_ref(), y.as_ref(), k, opts(args, false));
            let b = on_gram(&g, y.as_ref(), k, opts(args, false));
            assert!(
                a.stable_rank.is_some(),
                "k={k} {args:?}: the rule read no stable rank"
            );
            if let Some(d) = differs(&a, &b) {
                panic!("1100x300 k={k} {args:?}:\n    {d}");
            }
        }
    }
}

/// `subset(S)` of G is the Gram matrix of the rows `S` on the all-row
/// centring and scale. Every PLS1 method on it equals the call on `X[S]`
/// divided by the all-row scale (ones under the flag), centred by its own
/// means and read under `pre_standardized`; `score` there reads `y` as
/// given, so without the flag it gets `y` standardized. A G built under
/// `pre_standardized` on Pareto-scaled X is accepted too.
const AUTO: ConfirmatoryArgs = ConfirmatoryArgs::Auto {
    n_perm: 19,
    n_splits: 8,
};

/// `auto` at n = 600 is on the no-refit side of its floor at k = 1
/// (`split_exact` up to 1000) and on the refit side at k = 2 and with
/// `keep` (`split_nb` from 250), on X and on G alike.
#[test]
fn auto_floor_follows_the_route_on_x_and_gram() {
    let (x, y) = data(600, 10, 81);
    let g = gram_matrix(x.as_ref(), false).unwrap();
    for (k, want) in [(1, "split_exact"), (2, "split_nb")] {
        let a = on_x(x.as_ref(), y.as_ref(), k, opts(AUTO, false));
        let b = on_gram(&g, y.as_ref(), k, opts(AUTO, false));
        assert_eq!(
            (a.test_method.as_str(), b.test_method.as_str()),
            (want, want),
            "k={k}"
        );
    }
    let o = ConfirmatoryTestOpts {
        keep: Some(3),
        ..opts(AUTO, false)
    };
    assert_eq!(
        on_x(x.as_ref(), y.as_ref(), 1, o).test_method,
        "split_nb",
        "keep"
    );
}

#[test]
fn gram_subset_equals_x_call_scaled_over_all_rows() {
    for (n, p) in [(40, 100), (60, 8)] {
        let (x, y) = data(n, p, 21);
        let rows: Vec<usize> = (0..n).filter(|i| i % 5 != 2).rev().collect();
        let ys = Col::<f64>::from_fn(rows.len(), |i| y[rows[i]]);
        for flag in [false, true] {
            let (_, _, sd) = standardize(x.as_ref());
            let scale = |j: usize| if flag { 1.0 } else { sd[j] };
            let xs_raw = Mat::<f64>::from_fn(rows.len(), p, |i, j| x[(rows[i], j)] / scale(j));
            let xs = centred(xs_raw.as_ref());
            let g = gram_matrix(x.as_ref(), flag)
                .expect("gram_matrix")
                .subset(&rows)
                .expect("subset");
            for k in [1, 2] {
                for args in pls1_methods(k) {
                    let what = format!("{n}x{p} flag={flag} k={k} {args:?}");
                    let y_x = if matches!(args, ConfirmatoryArgs::Score { .. }) && !flag {
                        standardize1(ys.as_ref()).0
                    } else {
                        ys.clone()
                    };
                    let a = on_x(xs.as_ref(), y_x.as_ref(), k, opts(args, true));
                    let b = on_gram(&g, ys.as_ref(), k, opts(args, false));
                    if let Some(d) = differs(&a, &b) {
                        panic!("{what}:\n    {d}");
                    }
                }
            }
        }
    }

    let (x, y) = data(40, 100, 22);
    let (_, _, sd) = standardize(x.as_ref());
    // Centred, so `score`, which reads X as given under the flag, sees G's X̃.
    let pareto = centred(Mat::<f64>::from_fn(40, 100, |i, j| x[(i, j)] / sd[j].sqrt()).as_ref());
    let g = gram_matrix(pareto.as_ref(), true).expect("gram_matrix");
    for args in pls1_methods(1) {
        let a = on_x(pareto.as_ref(), y.as_ref(), 1, opts(args, true));
        let b = on_gram(&g, y.as_ref(), 1, opts(args, false));
        assert!(differs(&a, &b).is_none(), "Pareto {args:?}");
    }
    let yy = Mat::<f64>::from_fn(40, 1, |i, _| y[i]);
    pls1_perm_alpha_gram(&g, yy.as_ref(), 2).expect("perm_alpha on a flagged G");
}

/// The raw form of `pls1_perm_alpha` keeps its row-sum check: a subset of
/// G, which is not centred, is refused there and accepted as the object.
#[test]
fn perm_alpha_raw_form_keeps_its_checks() {
    let (x, y) = data(40, 100, 31);
    let rows: Vec<usize> = (0..30).collect();
    let g = gram_matrix(x.as_ref(), false)
        .unwrap()
        .subset(&rows)
        .unwrap();
    let yy = Mat::<f64>::from_fn(30, 1, |i, _| y[i]);
    let e = pls1_perm_alpha(g.gram.as_ref(), yy.as_ref(), 1, 100).unwrap_err();
    assert!(
        matches!(&e, PlsKitError::InvalidInput(m) if m.contains("row")),
        "{e:?}"
    );
    pls1_perm_alpha_gram(&g, yy.as_ref(), 1).expect("the object is centred first");
}

/// `r` is `InvalidArgument` with a message holding `want`.
fn invalid_argument<T: std::fmt::Debug>(r: Result<T, PlsKitError>, want: &str) {
    match r {
        Err(PlsKitError::InvalidArgument(m)) if m.contains(want) => {}
        other => panic!("expected InvalidArgument with {want:?}, got {other:?}"),
    }
}

fn gram_fixture() -> (GramMatrixOutput, Col<f64>, Mat<f64>) {
    let (x, y) = data(30, 80, 41);
    let yy = outcomes(x.as_ref(), 42);
    (gram_matrix(x.as_ref(), false).unwrap(), y, yy)
}

#[test]
fn gram_refuses_keep() {
    let (g, y, _) = gram_fixture();
    for args in pls1_methods(1) {
        let o = ConfirmatoryTestOpts {
            keep: Some(5),
            ..opts(args, false)
        };
        invalid_argument(
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Gram {
                    gram: &g,
                    y: y.as_ref(),
                    k: Some(&[1]),
                    covariates: None,
                    null_y: None,
                },
                o,
            ),
            "keep does not combine",
        );
    }
}

#[test]
fn gram_refuses_keep_x_below_p() {
    let (g, _, yy) = gram_fixture();
    let o = Pls3ConfirmatoryTestOpts {
        keep_x: Some(79),
        ..pls3_opts(pls3_methods()[0], false)
    };
    invalid_argument(
        pls3_confirmatory_test_gram(&g, yy.as_ref(), 1, o),
        "keep_X below n_features",
    );
    // The dense endpoint and the Y side are not refused.
    let o = Pls3ConfirmatoryTestOpts {
        keep_x: Some(80),
        keep_y: Some(2),
        ..pls3_opts(pls3_methods()[0], false)
    };
    pls3_confirmatory_test_gram(&g, yy.as_ref(), 1, o).expect("keep_x = p, keep_y = 2");
}

#[test]
fn gram_refuses_ci() {
    let (g, y, _) = gram_fixture();
    let o = ConfirmatoryTestOpts {
        ci: Some(plskit::CIOpts::default()),
        ..opts(pls1_methods(1)[3], false)
    };
    invalid_argument(
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: y.as_ref(),
                k: Some(&[1]),
                covariates: None,
                null_y: None,
            },
            o,
        ),
        "ci does not combine",
    );
}

/// `k` above 10, the largest the n-space fit serves, is refused by every
/// `gram` form; `k` = 10 runs.
#[test]
fn gram_refuses_k_above_k_dual_max() {
    let (g, y, _) = gram_fixture();
    let call = |k| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: y.as_ref(),
                k: Some(&[k]),
                covariates: None,
                null_y: None,
            },
            opts(pls1_methods(1)[3], false),
        )
    };
    invalid_argument(call(11), "gram serves k <= 10");
    call(10).expect("k = 10");
    let yy = Mat::<f64>::from_fn(30, 1, |i, _| y[i]);
    invalid_argument(pls1_perm_alpha_gram(&g, yy.as_ref(), 11), "serves k <= 10");
    invalid_argument(
        pls1_perm_alpha(g.gram.as_ref(), yy.as_ref(), 11, 80),
        "serves k <= 10",
    );
}

#[test]
fn gram_refuses_a_size_mismatch() {
    let (g, y, yy) = gram_fixture();
    let short = Col::<f64>::from_fn(29, |i| y[i]);
    invalid_argument(
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: short.as_ref(),
                k: Some(&[1]),
                covariates: None,
                null_y: None,
            },
            opts(pls1_methods(1)[3], false),
        ),
        "gram needs one row and column",
    );
    let short = Mat::<f64>::from_fn(29, 3, |i, j| yy[(i, j)]);
    invalid_argument(
        pls3_confirmatory_test_gram(&g, short.as_ref(), 1, pls3_opts(pls3_methods()[0], false)),
        "gram needs one row and column",
    );
    // The object form of `pls1_perm_alpha` too; its raw form keeps
    // `ShapeMismatch`.
    invalid_argument(
        pls1_perm_alpha_gram(&g, short.as_ref(), 1),
        "gram needs one row and column",
    );
    assert!(matches!(
        pls1_perm_alpha(g.gram.as_ref(), short.as_ref(), 1, 80),
        Err(PlsKitError::ShapeMismatch(_))
    ));
}

/// The X call's `n >= k + 1` check holds on the gram form: `score`, which
/// draws no split, on 3 rows at `k = 3`.
#[test]
fn gram_refuses_k_above_n_minus_1() {
    let (x, y) = data(3, 80, 61);
    let g = gram_matrix(x.as_ref(), false).unwrap();
    for r in [
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[3]),
                weights: None,
                covariates: None,
                null_y: None,
            },
            opts(ConfirmatoryArgs::Score { n_perm: None }, false),
        ),
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: y.as_ref(),
                k: Some(&[3]),
                covariates: None,
                null_y: None,
            },
            opts(ConfirmatoryArgs::Score { n_perm: None }, false),
        ),
    ] {
        invalid_argument(r, "insufficient n for k=3");
    }
}

/// Both forms of `pls1_perm_alpha` run at every `k` from 1 to 10 and agree,
/// and `X̃'·alpha` is `pls1_fit`'s coefficient wherever the fit keeps `k`
/// components.
#[test]
fn perm_alpha_runs_at_k_up_to_k_dual_max() {
    let (n, p) = (40, 200);
    let (x, y) = data(n, p, 51);
    let g = gram_matrix(x.as_ref(), false).unwrap();
    let (xs, _, _) = standardize(x.as_ref());
    let yy = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
    for k in 1..=10 {
        let raw = pls1_perm_alpha(g.gram.as_ref(), yy.as_ref(), k, p).expect("raw form");
        let obj = pls1_perm_alpha_gram(&g, yy.as_ref(), k).expect("object form");
        assert_eq!(raw.k_used, obj.k_used, "k = {k}");
        assert!(raw.k_used[0] <= k);
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(k),
            None,
            None,
            FitOpts::default(),
        )
        .expect("pls1_fit");
        if raw.k_used[0] != k || fit.k_used != k {
            continue;
        }
        let big = (0..p).fold(1.0_f64, |m, j| m.max(fit.coef[j].abs()));
        for out in [&raw, &obj] {
            for j in 0..p {
                let c: f64 = (0..n).map(|i| xs[(i, j)] * out.alpha[(i, 0)]).sum();
                assert!(
                    (c - fit.coef[j]).abs() <= 1e-10 * big,
                    "k = {k}, coef[{j}]: {c:e} vs {:e}",
                    fit.coef[j]
                );
            }
        }
    }
}

/// `n × 2` covariates: `x`'s first two columns plus noise.
fn covariates(x: MatRef<'_, f64>, seed: u64) -> Mat<f64> {
    let mut s = Stream(seed);
    Mat::<f64>::from_fn(x.nrows(), 2, |i, j| x[(i, j)] + s.next())
}

/// The methods that take covariates: `split_exact` at `k` 1 and 2 and
/// `raw_perm` at `k` 1.
fn covariate_cases() -> [(ConfirmatoryArgs, usize); 3] {
    let se = ConfirmatoryArgs::SplitExact {
        n_perm: 19,
        n_splits: 8,
    };
    let rp = ConfirmatoryArgs::RawPerm {
        n_perm: 19,
        n_folds: 5,
    };
    [(se, 1), (se, 2), (rp, 1)]
}

fn raw_with_z(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    z: MatRef<'_, f64>,
    k: usize,
    o: ConfirmatoryTestOpts,
) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x,
            y,
            k: Some(&[k]),
            weights: None,
            covariates: Some(z),
            null_y: None,
        },
        o,
    )
    .expect("X call")
}

fn gram_with_z(
    g: &GramMatrixOutput,
    y: ColRef<'_, f64>,
    z: MatRef<'_, f64>,
    k: usize,
    o: ConfirmatoryTestOpts,
) -> Result<ConfirmatoryTestOutput, PlsKitError> {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Gram {
            gram: g,
            y,
            k: Some(&[k]),
            covariates: Some(z),
            null_y: None,
        },
        o,
    )
}

#[test]
fn gram_with_covariates_equals_x_call() {
    for (n, p) in [(40, 90), (60, 8)] {
        let (x, y) = data(n, p, 61);
        let z = covariates(x.as_ref(), 62);
        let g = gram_matrix(x.as_ref(), false).unwrap();
        for (args, k) in covariate_cases() {
            let a = raw_with_z(x.as_ref(), y.as_ref(), z.as_ref(), k, opts(args, false));
            let b = gram_with_z(&g, y.as_ref(), z.as_ref(), k, opts(args, false)).unwrap();
            assert_eq!(b.n_covariates, Some(2));
            assert_eq!(a.n_covariates, b.n_covariates);
            if let Some(d) = differs(&a, &b) {
                panic!("{n}x{p} k={k} {args:?}:\n    {d}");
            }
        }
    }
}

/// `auto` on G with covariates runs `split_exact` on a design that runs
/// `split_nb` without them.
#[test]
fn gram_auto_with_covariates_runs_split_exact() {
    let (x, y) = data(1100, 10, 83);
    let z = covariates(x.as_ref(), 84);
    let g = gram_matrix(x.as_ref(), false).unwrap();
    let plain = on_gram(&g, y.as_ref(), 1, opts(AUTO, false));
    assert_eq!(plain.test_method, "split_nb", "premise");
    let got = gram_with_z(&g, y.as_ref(), z.as_ref(), 1, opts(AUTO, false)).unwrap();
    assert_eq!(got.test_method, "split_exact");
}

#[test]
fn gram_subset_with_covariates() {
    let (n, p) = (60, 40);
    let (x, y) = data(n, p, 63);
    let z = covariates(x.as_ref(), 64);
    let rows: Vec<usize> = (0..n).filter(|i| i % 5 != 2).rev().collect();
    let (_, _, sd) = standardize(x.as_ref());
    let xs = Mat::<f64>::from_fn(rows.len(), p, |i, j| x[(rows[i], j)] / sd[j]);
    let ys = Col::<f64>::from_fn(rows.len(), |i| y[rows[i]]);
    let zs = Mat::<f64>::from_fn(rows.len(), 2, |i, j| z[(rows[i], j)]);
    let g = gram_matrix(x.as_ref(), false)
        .unwrap()
        .subset(&rows)
        .unwrap();
    for (args, k) in covariate_cases() {
        let a = raw_with_z(xs.as_ref(), ys.as_ref(), zs.as_ref(), k, opts(args, true));
        let b = gram_with_z(&g, ys.as_ref(), zs.as_ref(), k, opts(args, false)).unwrap();
        if let Some(d) = differs(&a, &b) {
            panic!("k={k} {args:?}:\n    {d}");
        }
    }
}

#[test]
fn gram_subset_with_full_length_z() {
    let (n, p) = (40, 30);
    let (x, y) = data(n, p, 65);
    let z = covariates(x.as_ref(), 66);
    let rows: Vec<usize> = (0..30).collect();
    let ys = Col::<f64>::from_fn(rows.len(), |i| y[rows[i]]);
    let g = gram_matrix(x.as_ref(), false)
        .unwrap()
        .subset(&rows)
        .unwrap();
    let (args, k) = covariate_cases()[0];
    match gram_with_z(&g, ys.as_ref(), z.as_ref(), k, opts(args, false)) {
        Err(PlsKitError::ShapeMismatch(_)) => {}
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn gram_with_covariates_refuses_split_nb_score_e() {
    let (x, y) = data(40, 30, 67);
    let z = covariates(x.as_ref(), 68);
    let g = gram_matrix(x.as_ref(), false).unwrap();
    for args in [
        ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: false,
        },
        ConfirmatoryArgs::Score { n_perm: None },
        ConfirmatoryArgs::E,
    ] {
        let r = gram_with_z(&g, y.as_ref(), z.as_ref(), 1, opts(args, false));
        match r {
            Err(PlsKitError::InvalidArgument(m)) => {
                assert!(
                    m.contains("split_exact") && m.contains(args.method().as_str()),
                    "{m}"
                );
            }
            other => panic!("{args:?}: {other:?}"),
        }
    }
}

/// With a y that holds no signal the p-value is interior, so it shows the
/// gram path applies Freedman-Lane: it equals the X call and differs from
/// the gram call without covariates.
#[test]
fn gram_with_covariates_p_is_interior_and_uses_the_null_map() {
    let (n, p) = (40, 60);
    let (x, _) = data(n, p, 71);
    let mut s = Stream(72);
    let y = Col::<f64>::from_fn(n, |_| s.next());
    let z = covariates(x.as_ref(), 73);
    let g = gram_matrix(x.as_ref(), false).unwrap();
    for (args, k) in covariate_cases().map(|(a, k)| (more_perms(a), k)) {
        let a = raw_with_z(x.as_ref(), y.as_ref(), z.as_ref(), k, opts(args, false));
        let b = gram_with_z(&g, y.as_ref(), z.as_ref(), k, opts(args, false)).unwrap();
        let plain = on_gram(&g, y.as_ref(), k, opts(args, false));
        eprintln!(
            "k={k} {args:?}: X {} gram {} no-z {}",
            a.pvalue, b.pvalue, plain.pvalue
        );
        assert!(differs(&a, &b).is_none(), "k={k} {args:?}");
        assert!(b.pvalue > 1.5 / 200.0, "k={k} {args:?}: p at the floor");
        assert!(
            b.pvalue != plain.pvalue,
            "k={k} {args:?}: same p without covariates"
        );
    }
}

/// `args` with 199 permutations, so the p-value grid is fine.
fn more_perms(args: ConfirmatoryArgs) -> ConfirmatoryArgs {
    match args {
        ConfirmatoryArgs::SplitExact { n_splits, .. } => ConfirmatoryArgs::SplitExact {
            n_perm: 199,
            n_splits,
        },
        ConfirmatoryArgs::RawPerm { n_folds, .. } => ConfirmatoryArgs::RawPerm {
            n_perm: 199,
            n_folds,
        },
        other => other,
    }
}

/// `sv_perm` and `score` with `n_perm` on G equal the X call at the same
/// seed, plain and with covariates: the same permutations, so the same
/// p-value, and the statistic to rounding.
#[test]
fn sv_perm_gram_equals_raw() {
    let methods = [
        ConfirmatoryArgs::SvPerm {
            n_perm: 199,
            rotate: false,
        },
        ConfirmatoryArgs::Score { n_perm: Some(199) },
    ];
    for (n, p) in [(40, 90), (60, 8)] {
        let (x, y) = data(n, p, 71);
        let z = covariates(x.as_ref(), 72);
        let g = gram_matrix(x.as_ref(), false).unwrap();
        for args in methods {
            let plain = (
                on_x(x.as_ref(), y.as_ref(), 1, opts(args, false)),
                on_gram(&g, y.as_ref(), 1, opts(args, false)),
            );
            let with_z = (
                raw_with_z(x.as_ref(), y.as_ref(), z.as_ref(), 1, opts(args, false)),
                gram_with_z(&g, y.as_ref(), z.as_ref(), 1, opts(args, false)).unwrap(),
            );
            for (case, (raw, gram)) in [("plain", plain), ("covariates", with_z)] {
                assert_eq!(raw.pvalue, gram.pvalue, "{n}x{p} {args:?} {case}");
                assert_eq!(raw.test_method, gram.test_method);
                assert_eq!(raw.n_perm, gram.n_perm);
                let rel = (raw.statistic.unwrap() / gram.statistic.unwrap() - 1.0).abs();
                assert!(rel < 1e-10, "{n}x{p} {args:?} {case}: rel {rel}");
            }
        }
    }
}

/// PLS3 `sv_perm` on G equals the X call at the same seed, both `rotate`
/// settings: the same p-value and the statistic within 1e-10 relative. On
/// the wide shapes (40 x 90, 20 x 200) the X call takes the Gram route too,
/// so the X call's `HeldOutGram::from_x` is checked against `gram_matrix`;
/// on 60 x 8 the X call forms X̃ itself (the primal route). At q = 4 ≤ p,
/// rotation takes the `r = q` closed form.
#[test]
fn pls3_sv_perm_gram_equals_raw() {
    for (n, p, q) in [(40, 90, 3), (60, 8, 3), (20, 200, 4)] {
        let (x, _) = data(n, p, 81);
        let mut s = Stream(82);
        let yy = Mat::<f64>::from_fn(n, q, |i, j| x[(i, j % 2)] + 2.0 * s.next());
        let g = gram_matrix(x.as_ref(), false).unwrap();
        for rotate in [false, true] {
            let args = ConfirmatoryArgs::SvPerm {
                n_perm: 199,
                rotate,
            };
            let raw = pls3_confirmatory_test(x.as_ref(), yy.as_ref(), 1, pls3_opts(args, false))
                .expect("PLS3 X call");
            let gram = pls3_confirmatory_test_gram(&g, yy.as_ref(), 1, pls3_opts(args, false))
                .expect("PLS3 gram call");
            let what = format!("{n}x{p} q={q} rotate={rotate}");
            assert_eq!(raw.pvalue, gram.pvalue, "{what}");
            assert!(raw.pvalue.is_finite(), "{what}");
            assert_eq!(raw.test_method, gram.test_method);
            assert_eq!(raw.n_perm, gram.n_perm);
            let rel = (raw.statistic.unwrap() / gram.statistic.unwrap() - 1.0).abs();
            assert!(rel < 1e-10, "{what}: rel {rel}");
        }
    }
}
