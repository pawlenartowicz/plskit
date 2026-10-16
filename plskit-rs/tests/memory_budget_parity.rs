//! `PLSKIT_MEMORY_BUDGET` end to end, and the bits of the resampling routes
//! that fit PLS1 inside each unit.
//!
//! `memory_budget_does_not_change_results` re-runs itself in child processes
//! (`common::run_child`), one per budget, because the library reads the
//! variable on each call and the test thread cannot change the environment
//! safely. Every resampling loop the budget batches runs through its public
//! entry point at the budgets `0` (one unit at a time), a value that gives two
//! or three units at once (`Case::mid`, which differs per case, so that
//! budget gets one child per case), and unset (the default, every unit of
//! these small shapes at once), each inside Rayon pools of 1 and 4 threads
//! (`ThreadPoolBuilder::install`); a child prints one result per case and
//! pool. The `Debug` renderings of the six results of a case must be identical
//! (see `thread_count_parity.rs` for why that compares bits).
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

use std::ffi::OsStr;

use common::{child_arg, emit_result, run_child, synth};
use faer::linalg::svd::{ComputeSvdVectors, svd_scratch};
use faer::{Col, Mat};
use plskit::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput,
    FindKOptimalOpts, FindKeepOptimalOpts, FitOpts, KSpec, Pls3ConfirmatoryTestOpts, PlsKitError,
    RotationStabilityMethod, RotationStabilityOpts, Selector, VarimaxArgs, pls1_confirmatory_test,
    pls1_find_k_optimal, pls1_fit, pls1_rotation_stability, pls3_confirmatory_test,
    spls1_find_keep_optimal,
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
/// A `PLSKIT_NUM_THREADS` cap would run the calls on plskit's own pool, not on
/// the pools installed by the test.
const THREADS_VAR: &str = "PLSKIT_NUM_THREADS";
/// Must equal the name of the test function.
const TEST: &str = "memory_budget_does_not_change_results";

/// Rows of every parity case. Splits take `split_sizes(40, k) = (20, 20)`
/// at `k <= 2`; folds of 8 rows leave 32 training rows.
const PN: usize = 40;
/// Columns of X in the parity cases: `p >> n`, off every p-space Gram route.
const PP: usize = 400;
/// Columns of X in the constant-X cases on a Gram route. At each case's
/// middle budget, the cases `raw_perm k=1 constant X wide` and
/// `pls3 split_exact constant X wide keep_y=1` run in batches of units; their
/// comments give the batch sizes.
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

/// Bytes of an owned `f64` matrix of `rows × cols`, as the loops count it:
/// `resample::copy_bytes` (change together), the rows padded to a multiple
/// of 8.
const fn copy(rows: usize, cols: usize) -> usize {
    8 * rows.next_multiple_of(8) * cols
}

/// The two stream blocks a Gram-route or streamed-pass unit counts for
/// halves of `n_a` and `n_b` rows: `linalg::stream_blocks_bytes` (change
/// together), at `linalg::stream_block_width(max(n_a, n_b))` columns, 2 MiB
/// of the larger half rounded down to a multiple of 8 columns, at least 8.
const fn stream_blocks(n_a: usize, n_b: usize) -> usize {
    let rows = if n_a > n_b { n_a } else { n_b };
    let w = (2 << 20) / (8 * rows) / 8 * 8;
    let w = if w < 8 { 8 } else { w };
    copy(n_a, w) + copy(n_b, w)
}

/// One PLS3 LV1 fit on a `p × q` cross-product: `pls3::lv1_bytes` (change
/// together), `A`, the thin SVD's `U` and `V`, and faer's SVD scratch.
fn lv1(p: usize, q: usize) -> usize {
    let r = p.min(q);
    let scratch = svd_scratch::<f64>(
        p,
        q,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        faer::Par::Seq,
        faer::Spec::default(),
    )
    .size_bytes();
    copy(p, q) + copy(p, r) + copy(q, r) + scratch
}

/// One PLS1 fit on `p` columns to `k` components: `fit::pls1_fit_bytes`
/// (change together).
const fn pls1_fit_bytes(p: usize, k: usize) -> usize {
    5 * copy(p, k) + 4 * copy(p, 1)
}

/// One fold of `find_k::cv_select`: `find_k::cv_fold_bytes` (change
/// together), one PLS1 fit to `k` components, a `p × (k + 2)` matrix and a
/// block of validation rows of X: 2 MiB of `p`-wide rows rounded down to a
/// multiple of 8, at least 8 (`linalg::stream_block_width(p)`), and at most
/// `n_val`.
const fn cv_fold(p: usize, k: usize, n_val: usize) -> usize {
    let rows = (2 << 20) / (8 * p) / 8 * 8;
    let rows = if rows < 8 { 8 } else { rows };
    let rows = if rows < n_val { rows } else { n_val };
    pls1_fit_bytes(p, k) + copy(p, k + 2) + copy(rows, p)
}

/// The larger of the two pools each case runs on. A unit whose columns run
/// in parallel counts one fit per thread, so the middle budgets of those
/// cases are sized on this pool, and the pool of 1 can run more units at once.
const MAX_THREADS: usize = 4;

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
    // Two blocks of `cols` columns on the training rows and on the other
    // rows of a split or a fold: two matrices, padded apart
    // (`signal_test::split_blocks_bytes`, `resample::fold_blocks_bytes`; change
    // together).
    let split_blocks = |cols| 2 * copy(P_SPLIT_TR, cols);
    let fold_blocks = |cols| copy(P_FOLD_TR, cols) + copy(PN - P_FOLD_TR, cols);
    let one_copy = copy(PN, PP);
    let split_copy = split_blocks(PP);
    let fold_copy = fold_blocks(PP);
    let fold_fit_copy = copy(P_FOLD_TR, PP);
    // A Gram-route unit: `G` (`n_tr × n_tr`), `M` (`n_te × n_tr`) and the
    // stream blocks.
    let split_gram = split_blocks(P_SPLIT_TR) + stream_blocks(P_SPLIT_TR, PN - P_SPLIT_TR);
    let fold_gram = fold_blocks(P_FOLD_TR) + stream_blocks(P_FOLD_TR, PN - P_FOLD_TR);
    let wide_fold_copy = fold_blocks(PP_WIDE);
    let lv1_pls3 = lv1(PP, PQ);
    let raw_perm = |n_perm, n_folds| ConfirmatoryArgs::RawPerm { n_perm, n_folds };
    vec![
        // `split_half_correlations` (K >= 2 refits on prepared halves; the
        // √w scaling is written into the halves, so weights add no copy).
        case("pls1 split_nb k=2", split_copy, 2, move || {
            split_nb_out(&pls1_test(x, y, 2, None, SPLIT_NB, None, 11))
        }),
        case("pls1 split_nb k=2 weighted", split_copy, 2, move || {
            split_nb_out(&pls1_test(x, y, 2, Some(w), SPLIT_NB, None, 12))
        }),
        // The same loop with each split's fit past the `Auto` threshold, so
        // a batch runs fits that use the pool themselves.
        case(
            "pls1 split_nb k=2 200x6000",
            2 * copy(AUTO_N / 2, AUTO_P),
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
        case("pls1 split_nb k=1 constant X", split_copy, 2, move || {
            split_nb_out(&pls1_test(&d.xc, y, 1, None, SPLIT_NB, None, 13))
        }),
        // `split_zbars_columns` on the n-space route (dense, unweighted,
        // k = 2, 20·(40 + 400) < 400·40): the halves of X plus `G` and `M`.
        case(
            "split_exact k=2",
            split_copy + split_blocks(P_SPLIT_TR),
            2,
            move || format!("{:?}", pls1_test(x, y, 2, None, SPLIT_EXACT, None, 14)),
        ),
        // The no-refit route's first pass on route B (20·20·(400 + 20) <
        // 40·400·20): the stream blocks and `M`. A random X leaves nothing
        // for `fill_pending`.
        case(
            "split_exact k=1",
            stream_blocks(P_SPLIT_TR, PN - P_SPLIT_TR) + copy(PN - P_SPLIT_TR, P_SPLIT_TR),
            2,
            move || format!("{:?}", pls1_test(x, y, 1, None, SPLIT_EXACT, None, 15)),
        ),
        // `split_perm_nr_zbars`' second pass: every split undecided. The
        // halves of X and one PLS1 fit per thread (19 + 1 columns); sized on
        // the pool of 4, so the pool of 1 runs three units. Its first pass
        // runs one split at a time at this budget.
        case(
            "split_exact k=1 constant X",
            split_copy + MAX_THREADS * pls1_fit_bytes(PP, 1),
            2,
            move || format!("{:?}", pls1_test(&d.xc, y, 1, None, SPLIT_EXACT, None, 16)),
        ),
        // `pooled_cv_r2_columns` on the primal route (weights take it off
        // the n-space route; p >> n_tr keeps it off the p-space one): the
        // fold's two blocks of X, the √w-scaled training copy and one PLS1
        // fit per thread (19 + 1 columns); sized on the pool of 4, so the
        // pool of 1 runs three units.
        case(
            "raw_perm k=2 weighted",
            fold_copy + fold_fit_copy + MAX_THREADS * pls1_fit_bytes(PP, 2),
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(x, y, 2, Some(w), raw_perm(19, P_UNITS), None, 17)
                )
            },
        ),
        // `pooled_cv_r2_columns` on the n-space route (32·(40 + 400) <
        // 400·40): the fold's two blocks of X plus `G` and `M`.
        case(
            "raw_perm k=2",
            fold_copy + fold_blocks(P_FOLD_TR),
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
        // column of every fold pending. The second pass counts the fold's
        // two blocks of X and one PLS1 fit per thread (39 + 1 columns);
        // sized on the pool of 4, so the pool of 1 runs three units there.
        // The first pass runs all five folds in one batch at this budget
        // (`raw_perm k=1` covers it at two units).
        case(
            "raw_perm k=1 constant X wide",
            fold_gram.max(wide_fold_copy + MAX_THREADS * pls1_fit_bytes(PP_WIDE, 1)),
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(&d.xc_wide, y, 1, None, raw_perm(39, P_UNITS), None, 20)
                )
            },
        ),
        // `pooled_cv_r2_columns` on the p-space Gram route: the fold's two
        // blocks of X (5 folds of 400 rows), `C` (`p × p`) and one PLS1 fit
        // per thread (300 + 1 columns). The fit term is small, so both pools
        // run two units.
        case(
            "raw_perm gram-p 2000x50 k=2",
            copy(GRAM_P_N * 4 / 5, GRAM_P_P)
                + copy(GRAM_P_N / 5, GRAM_P_P)
                + copy(GRAM_P_P, GRAM_P_P)
                + MAX_THREADS * pls1_fit_bytes(GRAM_P_P, 2),
            2,
            move || {
                format!(
                    "{:?}",
                    pls1_test(&d.xg, &d.yg, 2, None, raw_perm(300, P_UNITS), None, 33)
                )
            },
        ),
        // `pls3_split_lv_correlations` (a sparse X side).
        case(
            "pls3 split_nb keep_x=40",
            split_copy + lv1_pls3,
            2,
            move || split_nb_out(&pls3_test(&d.x3, &d.y3, SPLIT_NB, Some(40), None, 21)),
        ),
        // The streamed route's first pass: the stream blocks and one LV1
        // fit. A random X leaves nothing for `fill_pending`.
        case(
            "pls3 split_nb dense",
            stream_blocks(P_SPLIT_TR, PN - P_SPLIT_TR) + lv1_pls3,
            2,
            move || split_nb_out(&pls3_test(&d.x3, &d.y3, SPLIT_NB, None, None, 22)),
        ),
        // `pls3_split_lv_correlations_streamed`' second pass: every split
        // undecided. Its first pass runs one split at a time at this budget.
        case(
            "pls3 split_nb dense constant X",
            split_copy + lv1_pls3,
            2,
            move || split_nb_out(&pls3_test(&d.xc, &d.y3, SPLIT_NB, None, None, 23)),
        ),
        // `pls3_split_zbars_columns`' first pass (the Gram route, 20·(60 +
        // 400) < 400·60).
        case("pls3 split_exact dense", split_gram, 2, move || {
            format!("{:?}", pls3_test(&d.x3, &d.y3, SPLIT_EXACT, None, None, 24))
        }),
        // `pls3_split_zbars_columns_primal` (a sparse X side): the halves of
        // X and one LV1 fit per thread (19 + 1 columns); sized on the pool of
        // 4, so the pool of 1 runs three units.
        case(
            "pls3 split_exact keep_x=40",
            split_copy + MAX_THREADS * lv1_pls3,
            2,
            move || {
                format!(
                    "{:?}",
                    pls3_test(&d.x3, &d.y3, SPLIT_EXACT, Some(40), None, 25)
                )
            },
        ),
        // Both passes of `pls3_split_zbars_columns`. On a constant X the
        // dense arm decides every column (exactly 0); a sparse Y side falls
        // back at its selection band (every `|v|` is 0), so every column
        // of every split is pending. Pass 1 runs four units at this budget,
        // pass 2 two (three on the pool of 1).
        case(
            "pls3 split_exact constant X wide keep_y=1",
            split_gram.max(split_blocks(PP_WIDE) + MAX_THREADS * lv1(PP_WIDE, PQ)),
            2,
            move || {
                format!(
                    "{:?}",
                    pls3_test(&d.xc_wide, &d.y3, SPLIT_EXACT, None, Some(1), 26)
                )
            },
        ),
        // `cv_select`: one fold fit on X in place, weighted or not
        // (`k_max = 2` gives `max_comp = 2`; the keep grid fits at k = 2).
        case(
            "pls1_find_k_optimal r2_se k_max=2",
            cv_fold(PP, 2, PN - P_FOLD_TR),
            2,
            move || format!("{:?}", find_k(x, y, None, 27)),
        ),
        case(
            "pls1_find_k_optimal r2_se k_max=2 weighted",
            cv_fold(PP, 2, PN - P_FOLD_TR),
            2,
            move || format!("{:?}", find_k(x, y, Some(w), 28)),
        ),
        case(
            "spls1_find_keep_optimal k=2",
            cv_fold(PP, 2, PN - P_FOLD_TR),
            2,
            move || {
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
            },
        ),
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

/// Runs a child of the test with the budget `budget` (`None`: unset) and no
/// thread cap.
fn run_budget_child(arg: &str, budget: Option<&OsStr>) -> Vec<String> {
    run_child(TEST, arg, &[(THREADS_VAR, None), (BUDGET_VAR, budget)])
}

#[test]
fn memory_budget_does_not_change_results() {
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
    let cases = parity_cases(&data);
    // Child argument: `bad` checks the bad-value message, a case index runs
    // that case only, anything else runs every case.
    if let Some(arg) = child_arg() {
        if arg == "bad" {
            emit_result(&bad_budget_message(&data.x, &data.y));
            return;
        }
        let only = arg.parse::<usize>().ok();
        let pools = [pool(1), pool(MAX_THREADS)];
        for (i, c) in cases.iter().enumerate() {
            if only.is_none_or(|o| o == i) {
                for pl in &pools {
                    emit_result(&pl.install(|| (c.run)()));
                }
            }
        }
        return;
    }

    let zero = run_budget_child("all", Some(OsStr::new("0")));
    let unset = run_budget_child("all", None);
    assert_eq!(zero.len(), 2 * cases.len(), "budget 0 results");
    assert_eq!(unset.len(), 2 * cases.len(), "unset budget results");
    let mut failures = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let mid = c.mid.to_string();
        let mid_runs = run_budget_child(&i.to_string(), Some(OsStr::new(&mid)));
        assert_eq!(mid_runs.len(), 2, "{}: mid budget results", c.name);
        let mut runs: Vec<(String, &String)> = Vec::new();
        for (budget, results) in [
            ("0", &zero[2 * i..2 * i + 2]),
            (mid.as_str(), &mid_runs[..]),
            ("unset", &unset[2 * i..2 * i + 2]),
        ] {
            for (t, out) in [1, MAX_THREADS].into_iter().zip(results) {
                runs.push((format!("budget {budget} on {t} thread(s)"), out));
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
    assert!(
        failures.is_empty(),
        "{} case(s) depend on the budget or the pool:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // A bad value is rejected before any work, quoted as given.
    let msg = run_budget_child("bad", Some(OsStr::new("2X"))).swap_remove(0);
    assert!(msg.contains(BUDGET_VAR) && msg.contains("\"2X\""), "{msg}");
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let msg = run_budget_child("bad", Some(OsStr::from_bytes(b"2\xffG"))).swap_remove(0);
        assert!(
            msg.contains(BUDGET_VAR) && msg.contains(r#""2\xFFG""#),
            "{msg}"
        );
    }
}
