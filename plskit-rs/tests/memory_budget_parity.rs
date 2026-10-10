//! `PLSKIT_MEMORY_BUDGET` end to end, and the bits of the resampling routes
//! that fit PLS1 inside each unit.
//!
//! `memory_budget_does_not_change_results` is the binary's only test that
//! runs by default, so setting the process-wide variable races nothing (the
//! tests of one binary run on parallel threads). Every resampling loop the
//! budget batches runs through its public entry point at the budgets `0`
//! (one unit at a time), a value that gives two or three units at once, and
//! unset (the default, every unit of these small shapes at once), each
//! inside Rayon pools of 1 and 4 threads (`ThreadPoolBuilder::install`); the
//! `Debug` renderings of the six results must be identical (see
//! `thread_count_parity.rs` for why that compares bits).
//!
//! `memory_budget_bits_probe` (ignored; run it with `--ignored --nocapture`)
//! prints the outputs of those routes on shapes past the `ParChoice::Auto`
//! threshold of the fit inside one unit (`n·d·k >= 1e6`, `fit::resolve_par`)
//! and on shapes below it. Every `f64` is printed with `{:?}`, the shortest
//! string that round-trips, so two equal strings are equal bits. Each line
//! starts with `probe`, then the shape, the case, the route the case is meant
//! to take and the `n·d·k` of the fit inside one unit (`n` its training rows,
//! `d` the feature count, `k` its component count).

mod common;

use common::synth;
use faer::{Col, Mat};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_fit, pls1_rotation_stability,
    pls3_confirmatory_test, spls1_find_keep_optimal, CIOpts, ConfirmatoryArgs,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput, FindKOptimalOpts,
    FindKeepOptimalOpts, FitOpts, KSpec, Pls3ConfirmatoryTestOpts, PlsKitError,
    RotationStabilityMethod, RotationStabilityOpts, Selector, VarimaxArgs,
};

/// Component count of every case.
const K: usize = 2;
/// Split-half repetitions of the split tests.
const N_SPLITS: usize = 4;
/// CV folds of the selectors.
const N_FOLDS: usize = 5;
/// Resamples of the CI and of rotation stability (the floor of both).
const N_BOOT: usize = 100;
/// Subsample rate of the CI and of rotation stability: `m = ceil(n^0.9)`,
/// large enough for the subsample fits on the wide shape to cross the
/// threshold.
const M_RATE: f64 = 0.9;

fn row_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| 0.5 + f64::from(u32::try_from(i % 4).unwrap()) * 0.25)
}

struct Shape {
    x: Mat<f64>,
    y: Col<f64>,
    tag: String,
}

impl Shape {
    fn new(n: usize, p: usize, seed: u64) -> Self {
        let (x, y) = synth(n, p, 1.0, seed);
        Self {
            x,
            y,
            tag: format!("{n}x{p}"),
        }
    }

    fn n(&self) -> usize {
        self.x.nrows()
    }

    fn p(&self) -> usize {
        self.x.ncols()
    }

    /// Training rows of one split half at `K` (`n / 2` at these shapes).
    fn n_split_tr(&self) -> usize {
        self.n() / 2
    }

    /// Training rows of one CV fold (`n` is a multiple of `N_FOLDS` here).
    fn n_fold_tr(&self) -> usize {
        self.n() - self.n() / N_FOLDS
    }

    fn confirm(
        &self,
        args: ConfirmatoryArgs,
        weights: Option<&Col<f64>>,
        keep: Option<usize>,
        ci: Option<CIOpts>,
        seed: u64,
    ) -> ConfirmatoryTestOutput {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: self.x.as_ref(),
                y: self.y.as_ref(),
                k: K,
                weights: weights.map(Col::as_ref),
            },
            ConfirmatoryTestOpts {
                args,
                seed: Some(seed),
                keep,
                ci,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn print_test(&self, case: &str, route: &str, ndk: usize, out: &ConfirmatoryTestOutput) {
        println!(
            "probe {} {case} route={route} ndk={ndk} {:?} {:?} test_method={}",
            self.tag, out.statistic, out.pvalue, out.test_method
        );
    }
}

/// The cases both the above-threshold and the below-threshold shape run.
#[allow(clippy::too_many_lines)]
fn run_shape_cases(s: &Shape) {
    let (n, p) = (s.n(), s.p());
    let split_ndk = s.n_split_tr() * p * K;

    let out = s.confirm(
        ConfirmatoryArgs::SplitNb {
            n_splits: N_SPLITS,
            force: true,
        },
        None,
        None,
        None,
        101,
    );
    s.print_test("split_nb dense", "refit-primal", split_ndk, &out);

    // n_perm = 9 keeps the dense unweighted block off the n-space route
    // (`use_dual_route` needs more replicates at these shapes).
    let out = s.confirm(
        ConfirmatoryArgs::SplitExact {
            n_perm: 9,
            n_splits: N_SPLITS,
        },
        None,
        None,
        None,
        102,
    );
    s.print_test("split_exact dense n_perm=9", "Primal", split_ndk, &out);

    let w = row_weights(n);
    let out = s.confirm(
        ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: N_SPLITS,
        },
        Some(&w),
        None,
        None,
        103,
    );
    s.print_test("split_exact weighted n_perm=19", "Primal", split_ndk, &out);

    let keep = p / 10;
    let out = s.confirm(
        ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: N_SPLITS,
        },
        None,
        Some(keep),
        None,
        104,
    );
    s.print_test(
        &format!("split_exact keep={keep} n_perm=19"),
        "Primal",
        split_ndk,
        &out,
    );

    // Control: n_perm = 59 puts the dense unweighted block on the n-space
    // route, whose per-unit fit is the n-space kernel (the primal arm only
    // on a column the kernel leaves undecided).
    let out = s.confirm(
        ConfirmatoryArgs::SplitExact {
            n_perm: 59,
            n_splits: N_SPLITS,
        },
        None,
        None,
        None,
        105,
    );
    s.print_test(
        "split_exact dense n_perm=59",
        "Nspace(control)",
        split_ndk,
        &out,
    );

    let fold_ndk = s.n_fold_tr() * p * K;
    let out = pls1_find_k_optimal(
        s.x.as_ref(),
        s.y.as_ref(),
        K,
        None,
        FindKOptimalOpts {
            selector: Selector::R2Se,
            n_folds: N_FOLDS,
            seed: Some(106),
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "probe {} find_k_optimal r2_se k_max={K} route=cv-fold-fit ndk={fold_ndk} k_star={} \
         cv_scores={:?} cv_scores_se={:?}",
        s.tag, out.k_star, out.cv_scores, out.cv_scores_se
    );

    let out = spls1_find_keep_optimal(
        s.x.as_ref(),
        s.y.as_ref(),
        K,
        None,
        FindKeepOptimalOpts {
            n_folds: N_FOLDS,
            seed: Some(107),
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "probe {} spls1_find_keep_optimal k={K} route=cv-fold-fit ndk={fold_ndk} keep_star={} \
         cv_scores={:?} cv_scores_se={:?}",
        s.tag, out.keep_star, out.cv_scores, out.cv_scores_se
    );

    // The CI runs on `score`, whose test performs no fit, so the line
    // isolates the subsample (`m` rows) and bootstrap (`n` rows) fits.
    let out = s.confirm(
        ConfirmatoryArgs::Score,
        None,
        None,
        Some(CIOpts {
            n_boot: N_BOOT,
            m_rate: M_RATE,
            ..CIOpts::default()
        }),
        108,
    );
    let ci = out.ci.as_ref().unwrap();
    println!(
        "probe {} ci(score) m={} route=subsample-fit ndk={} boot_ndk={} \
         beta_ci_lower[0]={:?} beta_ci_upper[0]={:?} holdout_corr={:?}",
        s.tag,
        ci.m,
        ci.m * p * K,
        n * p * K,
        ci.beta_ci_lower[0],
        ci.beta_ci_upper[0],
        ci.holdout_corr
    );
    println!("probe {} ci(score) full {ci:?}", s.tag);

    let out = pls1_rotation_stability(
        s.x.as_ref(),
        s.y.as_ref(),
        K,
        RotationStabilityMethod::Varimax(VarimaxArgs::default()),
        None,
        None,
        RotationStabilityOpts {
            n_boot: N_BOOT,
            m_rate: M_RATE,
            seed: Some(109),
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "probe {} rotation_stability m={} route=subsample-fit ndk={} variance_ratio={:?} \
         full={out:?}",
        s.tag,
        out.m,
        out.m * p * K,
        out.variance_ratio
    );
}

/// `split_exact` on the p-space Gram route (`GramP`): dense, unweighted,
/// `n_tr ≥ p` (off the n-space route), the Gram cost rule and the
/// `B·(2k + 1)·n_tr·p ≥ 1e8` work floor met with `B = n_perm + 1`.
fn run_gram_p_case(n: usize, p: usize, n_perm: usize, seed: u64) {
    let s = Shape::new(n, p, seed);
    let out = s.confirm(
        ConfirmatoryArgs::SplitExact {
            n_perm,
            n_splits: N_SPLITS,
        },
        None,
        None,
        None,
        seed + 1,
    );
    s.print_test(
        &format!("split_exact dense n_perm={n_perm}"),
        "GramP",
        s.n_split_tr() * p * K,
        &out,
    );
}

#[test]
#[ignore = "bit probe: prints outputs, asserts nothing"]
fn memory_budget_bits_probe() {
    // Past the threshold: split halves of 100 rows, folds of 160, m = 118.
    run_shape_cases(&Shape::new(200, 6000, 1));
    // Below it: split halves of 20 rows, folds of 32, m = 28.
    run_shape_cases(&Shape::new(40, 200, 2));
    // GramP past the threshold (n_tr = 2500, 2500·250·2 = 1.25e6) and below
    // it (2500·50·2 = 2.5e5).
    run_gram_p_case(5000, 250, 49, 3);
    run_gram_p_case(5000, 50, 199, 5);
}

const BUDGET_VAR: &str = "PLSKIT_MEMORY_BUDGET";

/// Rows of every parity case. Splits take `split_sizes(40, k) = (20, 20)`
/// at `k <= 2`; folds of 8 rows leave 32 training rows.
const PN: usize = 40;
/// Columns of X in the parity cases: `p >> n`, off every p-space Gram route.
const PP: usize = 400;
/// Columns of X in the constant-X cases on a Gram route: a copy of X
/// (`8·40·13_120` bytes) is within 0.2% of the Gram unit, so one budget
/// gives both passes two units at once.
const PP_WIDE: usize = 13_120;
/// Columns of Y in the PLS3 cases.
const PQ: usize = 3;
/// Training rows of a split half and of a fold.
const P_SPLIT_TR: usize = 20;
const P_FOLD_TR: usize = 32;
/// Splits and folds per case: five units, so a two-unit batch ends on a
/// partial one.
const P_UNITS: usize = 5;
/// Resamples of the CI and of rotation stability (the floor of both).
const P_BOOT: usize = 100;
/// Rows and columns of the case whose per-split fit is past the
/// `ParChoice::Auto` threshold: halves of 100 rows, `100·6000·2 = 1.2e6`.
const AUTO_N: usize = 200;
const AUTO_P: usize = 6000;
/// Rows and columns of the `raw_perm` case on the p-space Gram route
/// (`GramP`), the shape of `thread_count_parity.rs`'s gram-p case.
const GRAM_P_N: usize = 2000;
const GRAM_P_P: usize = 50;
const P_M_RATE: f64 = 0.7;
/// Rotation stability's subsample rows, `ceil(40^0.7)`.
const P_ROT_M: usize = 14;

/// Bytes of an `f64` copy of `rows × cols` values, as the loops count it.
const fn copy(rows: usize, cols: usize) -> usize {
    8 * rows * cols
}

/// The two 2 MiB stream blocks every Gram-route and streamed-pass unit counts.
const STREAM_BLOCKS: usize = 2 * (2 << 20);

/// One public call and its middle budget: `units` and a half times the
/// `unit_bytes` of the loop the case exists for, so the budget gives
/// `units` at once even if the loop counts its unit slightly larger.
struct Case<'a> {
    name: &'static str,
    mid: usize,
    run: Box<dyn Fn() -> String + Sync + 'a>,
}

fn case<'a>(
    name: &'static str,
    unit_bytes: usize,
    units: usize,
    run: impl Fn() -> String + Sync + 'a,
) -> Case<'a> {
    Case {
        name,
        mid: unit_bytes * units + unit_bytes / 2,
        run: Box::new(run),
    }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
}

#[allow(clippy::many_single_char_names)]
fn two_block(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
    let y = Mat::<f64>::from_fn(n, q, |i, j| x[(i, j)] + 0.5 * rng.random_range(-1.0..1.0));
    (x, y)
}

/// Every column constant (column `j` is `j + 1`): standardized, X is the
/// zero matrix, which no first pass decides.
fn constant_x(n: usize, p: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(n, p, |_, j| f64::from(u32::try_from(j + 1).unwrap()))
}

fn pls1_test(
    x: &Mat<f64>,
    y: &Col<f64>,
    k: usize,
    weights: Option<&Col<f64>>,
    args: ConfirmatoryArgs,
    ci: Option<CIOpts>,
    seed: u64,
) -> ConfirmatoryTestOutput {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k,
            weights: weights.map(Col::as_ref),
        },
        ConfirmatoryTestOpts {
            args,
            seed: Some(seed),
            ci,
            ..Default::default()
        },
    )
    .unwrap()
}

fn pls3_test(
    x: &Mat<f64>,
    y: &Mat<f64>,
    args: ConfirmatoryArgs,
    keep_x: Option<usize>,
    keep_y: Option<usize>,
    seed: u64,
) -> ConfirmatoryTestOutput {
    pls3_confirmatory_test(
        x.as_ref(),
        y.as_ref(),
        1,
        Pls3ConfirmatoryTestOpts {
            args,
            keep_x,
            keep_y,
            seed: Some(seed),
            ..Default::default()
        },
    )
    .unwrap()
}

/// The `Debug` rendering of a `split_nb` result, after checking that
/// `split_nb` ran (`force: true` skips the auto-gate).
fn split_nb_out(out: &ConfirmatoryTestOutput) -> String {
    assert_eq!(out.test_method, "split_nb");
    format!("{out:?}")
}

const SPLIT_NB: ConfirmatoryArgs = ConfirmatoryArgs::SplitNb {
    n_splits: P_UNITS,
    force: true,
};
const SPLIT_EXACT: ConfirmatoryArgs = ConfirmatoryArgs::SplitExact {
    n_perm: 19,
    n_splits: P_UNITS,
};

/// The cases, one or more per batched loop, each with the loop it reaches
/// and the bytes of one of its units.
#[allow(clippy::too_many_lines)]
fn parity_cases(d: &ParityData) -> Vec<Case<'_>> {
    let (x, y, w) = (&d.x, &d.y, &d.w);
    let one_copy = copy(PN, PP);
    let fold_fit_copy = copy(P_FOLD_TR, PP);
    // A Gram-route unit: `G` and `M` (`n_tr · n` values) and the stream
    // blocks.
    let split_gram = copy(P_SPLIT_TR, PN) + STREAM_BLOCKS;
    let fold_gram = copy(P_FOLD_TR, PN) + STREAM_BLOCKS;
    let wide_copy = copy(PN, PP_WIDE);
    let raw_perm = |n_perm, n_folds| ConfirmatoryArgs::RawPerm { n_perm, n_folds };
    vec![
        // `split_half_correlations` (K >= 2 refits on prepared halves; the
        // √w scaling is written into the halves, so weights add no copy).
        case("pls1 split_nb k=2", one_copy, 2, move || {
            split_nb_out(&pls1_test(x, y, 2, None, SPLIT_NB, None, 11))
        }),
        case("pls1 split_nb k=2 weighted", one_copy, 2, move || {
            split_nb_out(&pls1_test(x, y, 2, Some(w), SPLIT_NB, None, 12))
        }),
        // The same loop with each split's fit past the `Auto` threshold, so
        // a batch runs fits that use the pool themselves.
        case(
            "pls1 split_nb k=2 200x6000",
            copy(AUTO_N, AUTO_P),
            2,
            move || {
                split_nb_out(&pls1_test(
                    &d.x_auto, &d.y_auto, 2, None, SPLIT_NB, None, 32,
                ))
            },
        ),
        // `split_nb_correlations_nr`'s second pass: a constant X leaves
        // every split undecided. Its first pass runs one split at a time at
        // this budget.
        case("pls1 split_nb k=1 constant X", one_copy, 2, move || {
            split_nb_out(&pls1_test(&d.xc, y, 1, None, SPLIT_NB, None, 13))
        }),
        // `split_zbars_columns` on the n-space route (dense, unweighted,
        // k = 2, 20·(40 + 400) < 400·40): a copy of X plus `G` and `M`.
        case(
            "split_exact k=2",
            one_copy + copy(P_SPLIT_TR, PN),
            2,
            move || format!("{:?}", pls1_test(x, y, 2, None, SPLIT_EXACT, None, 14)),
        ),
        // The no-refit route's first pass on route B (20·20·(400 + 20) <
        // 40·400·20): the stream blocks and `M`. A random X leaves nothing
        // for `fill_pending`.
        case(
            "split_exact k=1",
            STREAM_BLOCKS + copy(PN - P_SPLIT_TR, P_SPLIT_TR),
            2,
            move || format!("{:?}", pls1_test(x, y, 1, None, SPLIT_EXACT, None, 15)),
        ),
        // `split_perm_nr_zbars`' second pass: every split undecided. Its
        // first pass runs one split at a time at this budget.
        case("split_exact k=1 constant X", one_copy, 2, move || {
            format!("{:?}", pls1_test(&d.xc, y, 1, None, SPLIT_EXACT, None, 16))
        }),
        // `pooled_cv_r2_columns` on the primal route (weights take it off
        // the n-space route; p >> n_tr keeps it off the p-space one): a copy
        // of X plus the √w-scaled training copy.
        case(
            "raw_perm k=2 weighted",
            one_copy + fold_fit_copy,
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(x, y, 2, Some(w), raw_perm(19, P_UNITS), None, 17)
                )
            },
        ),
        // `pooled_cv_r2_columns` on the n-space route (32·(40 + 400) <
        // 400·40): a copy of X plus `G` and `M`.
        case(
            "raw_perm k=2",
            one_copy + copy(P_FOLD_TR, PN),
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(x, y, 2, None, raw_perm(19, P_UNITS), None, 18)
                )
            },
        ),
        // `pls1_cv_r2_columns`' first pass (the K = 1 Gram route, 32·(40 +
        // 400) < 400·40); a random X leaves no column pending.
        case("raw_perm k=1", fold_gram, 2, move || {
            format!(
                "{:?}",
                pls1_test(x, y, 1, None, raw_perm(39, P_UNITS), None, 19)
            )
        }),
        // Both passes of `pls1_cv_r2_columns`: a constant X leaves every
        // column of every fold pending. The budget gives each pass two
        // units.
        case(
            "raw_perm k=1 constant X wide",
            fold_gram.max(wide_copy),
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(&d.xc_wide, y, 1, None, raw_perm(39, P_UNITS), None, 20)
                )
            },
        ),
        // `pooled_cv_r2_columns` on the p-space Gram route: a copy of X
        // plus `C` (`p × p`).
        case(
            "raw_perm gram-p 2000x50 k=2",
            copy(GRAM_P_N, GRAM_P_P) + copy(GRAM_P_P, GRAM_P_P),
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(&d.xg, &d.yg, 2, None, raw_perm(300, P_UNITS), None, 33)
                )
            },
        ),
        // `pls3_split_lv_correlations` (a sparse X side).
        case("pls3 split_nb keep_x=40", one_copy, 2, move || {
            split_nb_out(&pls3_test(&d.x3, &d.y3, SPLIT_NB, Some(40), None, 21))
        }),
        // The streamed route's first pass: the stream blocks, `A` and its
        // SVD's `U`. A random X leaves nothing for `fill_pending`.
        case(
            "pls3 split_nb dense",
            STREAM_BLOCKS + 2 * copy(PP, PQ),
            2,
            move || split_nb_out(&pls3_test(&d.x3, &d.y3, SPLIT_NB, None, None, 22)),
        ),
        // `pls3_split_lv_correlations_streamed`' second pass: every split
        // undecided. Its first pass runs one split at a time at this budget.
        case("pls3 split_nb dense constant X", one_copy, 2, move || {
            split_nb_out(&pls3_test(&d.xc, &d.y3, SPLIT_NB, None, None, 23))
        }),
        // `pls3_split_zbars_columns`' first pass (the Gram route, 20·(60 +
        // 400) < 400·60).
        case("pls3 split_exact dense", split_gram, 2, move || {
            format!("{:?}", pls3_test(&d.x3, &d.y3, SPLIT_EXACT, None, None, 24))
        }),
        // `pls3_split_zbars_columns_primal` (a sparse X side).
        case("pls3 split_exact keep_x=40", one_copy, 2, move || {
            format!(
                "{:?}",
                pls3_test(&d.x3, &d.y3, SPLIT_EXACT, Some(40), None, 25)
            )
        }),
        // Both passes of `pls3_split_zbars_columns`. On a constant X the
        // dense arm decides every column (exactly 0); a sparse Y side falls
        // back at its selection band (every `|v|` is 0), so every column
        // of every split is pending. The budget gives each pass two units.
        case(
            "pls3 split_exact constant X wide keep_y=1",
            split_gram.max(wide_copy),
            2,
            move || {
                format!(
                    "{:?}",
                    pls3_test(&d.xc_wide, &d.y3, SPLIT_EXACT, None, Some(1), 26)
                )
            },
        ),
        // `cv_select`: the fold's standardized slices of X, plus the
        // √w-scaled training copy when weighted.
        case(
            "pls1_find_k_optimal r2_se k_max=2",
            one_copy,
            2,
            move || format!("{:?}", find_k(x, y, None, 27)),
        ),
        case(
            "pls1_find_k_optimal r2_se k_max=2 weighted",
            one_copy + fold_fit_copy,
            2,
            move || format!("{:?}", find_k(x, y, Some(w), 28)),
        ),
        case("spls1_find_keep_optimal k=2", one_copy, 2, move || {
            let out = spls1_find_keep_optimal(
                x.as_ref(),
                y.as_ref(),
                2,
                None,
                FindKeepOptimalOpts {
                    n_folds: P_UNITS,
                    seed: Some(29),
                    ..Default::default()
                },
            )
            .unwrap();
            format!("{out:?}")
        }),
        // `pls1_subsample_inference_confirmatory`: two n-row copies of X
        // per replicate. `score` performs no fit, so the CI is the only
        // loop.
        case("ci (score) k=2", 2 * one_copy, 3, move || {
            let ci = CIOpts {
                n_boot: P_BOOT,
                m_rate: P_M_RATE,
                ..CIOpts::default()
            };
            format!(
                "{:?}",
                pls1_test(x, y, 2, None, ConfirmatoryArgs::Score, Some(ci), 30)
            )
        }),
        // `rotation_stability_impl`: the m subsample rows of X.
        case(
            "pls1_rotation_stability k=2",
            copy(P_ROT_M, PP),
            3,
            move || {
                let out = pls1_rotation_stability(
                    x.as_ref(),
                    y.as_ref(),
                    2,
                    RotationStabilityMethod::Varimax(VarimaxArgs::default()),
                    None,
                    None,
                    RotationStabilityOpts {
                        n_boot: P_BOOT,
                        m_rate: P_M_RATE,
                        seed: Some(31),
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(out.m, P_ROT_M);
                format!("{out:?}")
            },
        ),
    ]
}

fn find_k(
    x: &Mat<f64>,
    y: &Col<f64>,
    w: Option<&Col<f64>>,
    seed: u64,
) -> plskit::FindKOptimalOutput {
    pls1_find_k_optimal(
        x.as_ref(),
        y.as_ref(),
        2,
        w.map(Col::as_ref),
        FindKOptimalOpts {
            selector: Selector::R2Se,
            n_folds: P_UNITS,
            seed: Some(seed),
            ..Default::default()
        },
    )
    .unwrap()
}

struct ParityData {
    x: Mat<f64>,
    y: Col<f64>,
    w: Col<f64>,
    xc: Mat<f64>,
    xc_wide: Mat<f64>,
    x3: Mat<f64>,
    y3: Mat<f64>,
    x_auto: Mat<f64>,
    y_auto: Col<f64>,
    xg: Mat<f64>,
    yg: Col<f64>,
}

/// Removes `PLSKIT_MEMORY_BUDGET` when dropped, also when the test panics.
struct UnsetBudget;

impl Drop for UnsetBudget {
    fn drop(&mut self) {
        std::env::remove_var(BUDGET_VAR);
    }
}

/// The message of the `InvalidArgument` that `pls1_fit` returns while the
/// variable holds a bad value.
fn bad_budget_message(x: &Mat<f64>, y: &Col<f64>) -> String {
    match pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    ) {
        Err(PlsKitError::InvalidArgument(msg)) => msg,
        other => panic!("{BUDGET_VAR} bad: {:?}", other.map(|_| ())),
    }
}

#[test]
fn memory_budget_does_not_change_results() {
    let _unset = UnsetBudget;
    // A `PLSKIT_NUM_THREADS` cap would run the calls on plskit's own pool,
    // not on the pools installed below.
    std::env::remove_var("PLSKIT_NUM_THREADS");
    let (x, y) = synth(PN, PP, 1.0, 1);
    let (x3, y3) = two_block(PN, PP, PQ, 2);
    let (x_auto, y_auto) = synth(AUTO_N, AUTO_P, 1.0, 3);
    let (xg, yg) = synth(GRAM_P_N, GRAM_P_P, 0.3, 62);
    let data = ParityData {
        x,
        y,
        w: row_weights(PN),
        xc: constant_x(PN, PP),
        xc_wide: constant_x(PN, PP_WIDE),
        x3,
        y3,
        x_auto,
        y_auto,
        xg,
        yg,
    };
    let pools = [(1, pool(1)), (4, pool(4))];
    let mut failures = Vec::new();
    for c in parity_cases(&data) {
        let mid = c.mid.to_string();
        let mut runs: Vec<(String, String)> = Vec::new();
        for budget in [Some("0"), Some(mid.as_str()), None] {
            match budget {
                Some(b) => std::env::set_var(BUDGET_VAR, b),
                None => std::env::remove_var(BUDGET_VAR),
            }
            for (t, pl) in &pools {
                let label = format!("budget {} on {t} thread(s)", budget.unwrap_or("unset"));
                runs.push((label, pl.install(|| (c.run)())));
            }
        }
        let (label0, out0) = &runs[0];
        if let Some((label, out)) = runs[1..].iter().find(|(_, o)| o != out0) {
            let at = out0
                .bytes()
                .zip(out.bytes())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| out0.len().min(out.len()));
            let lo = at.saturating_sub(100);
            let ctx = |s: &str| s.get(lo..(at + 40).min(s.len())).unwrap_or("").to_owned();
            failures.push(format!(
                "{}: {label0} and {label} differ at byte {at}\n  ...{}\n  ...{}",
                c.name,
                ctx(out0),
                ctx(out)
            ));
        }
    }
    std::env::remove_var(BUDGET_VAR);
    assert!(
        failures.is_empty(),
        "{} case(s) depend on the budget or the pool:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // A bad value is rejected before any work, quoted as given.
    std::env::set_var(BUDGET_VAR, "2X");
    let msg = bad_budget_message(&data.x, &data.y);
    assert!(msg.contains(BUDGET_VAR) && msg.contains("\"2X\""), "{msg}");
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        std::env::set_var(BUDGET_VAR, std::ffi::OsStr::from_bytes(b"2\xffG"));
        let msg = bad_budget_message(&data.x, &data.y);
        assert!(
            msg.contains(BUDGET_VAR) && msg.contains(r#""2\xFFG""#),
            "{msg}"
        );
    }
    std::env::remove_var(BUDGET_VAR);
}
