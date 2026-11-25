// The parity cases of `PLSKIT_MEMORY_BUDGET`, written once and included by
// `tests/memory_budget_parity.rs` (budgets through the environment, results
// compared bit for bit) and by `src/memory_budget_cases.rs` (budgets through
// `fit::with_budget`, routes and batch sizes read from the test logs). The
// includer brings the names into scope: the `faer`, `plskit` and `rayon`
// items used below, and `synth` (`tests/common/synth.rs`).

/// Rows of every parity case. Splits take `split_sizes(40, k) = (20, 20)`
/// at `k <= 2`; folds of 8 rows leave 32 training rows.
const PN: usize = 40;
/// Columns of X in the parity cases: `p >> n`, off every p-space Gram route.
const PP: usize = 400;
/// Columns of X in the wide constant-X cases.
const PP_WIDE: usize = 13_120;
/// Columns of Y in the PLS3 cases.
const PQ: usize = 3;
/// Columns of X in the PLS3 case on the streamed body: `P_SPLIT_TR ≥ PS_P`
/// leaves no Gram route.
const PS_P: usize = 15;
/// Training rows of a split half and of a fold.
const P_SPLIT_TR: usize = 20;
const P_FOLD_TR: usize = 32;
/// Splits and folds per case: five units, so a two-unit batch ends on a
/// partial one.
const P_UNITS: usize = 5;
/// Resamples of the CI (the floor).
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
/// The larger of the two pools each case runs on.
const MAX_THREADS: usize = 4;

fn row_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| 0.5 + f64::from(u32::try_from(i % 4).unwrap()) * 0.25)
}

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

/// One fold of `profiles_find_n`'s first pass on `n_tr ≤ p` training rows
/// of `p` columns: `factors::fold_svd_bytes` (change together) on its
/// `n_tr ≤ p` arm, the fold's copy of X, `V` and faer's SVD scratch.
fn fold_svd(n_tr: usize, p: usize) -> usize {
    let scratch = svd_scratch::<f64>(
        n_tr,
        p,
        ComputeSvdVectors::No,
        ComputeSvdVectors::Thin,
        faer::Par::Seq,
        faer::Spec::default(),
    )
    .size_bytes();
    copy(n_tr, p) + copy(p, n_tr) + scratch
}

/// One replicate column's copies of Y on a PLS3 split of `n_tr` and `n_te`
/// rows: `pls3_signal_test::y_column_bytes` (change together), the permuted
/// block and its two centred halves.
#[allow(clippy::similar_names)] // n_tr, n_te
const fn y_column(n_tr: usize, n_te: usize, q: usize) -> usize {
    copy(n_tr + n_te, q) + copy(n_tr, q) + copy(n_te, q)
}

/// One n-space kernel call on `m` outcome columns of length `n`, on a
/// training block of `n_tr` rows at `k` components:
/// `dual_route::nspace_run_bytes` (change together).
const fn nspace_run(n: usize, n_tr: usize, k: usize, m: usize) -> usize {
    (copy(n, 1) + copy(n_tr, k)) * m + 3 * copy(n_tr, m) + 8 * n_tr * m * (3 * k + 2)
}

/// One PLS1 fit on `p` columns to `k` components: `fit::pls1_fit_bytes`
/// (change together).
const fn pls1_fit_bytes(p: usize, k: usize) -> usize {
    5 * copy(p, k) + 4 * copy(p, 1)
}

/// One batch of `b` lanes of the primal route on `p` columns to `k`
/// components, scored at one k on `n_te` held-out rows:
/// `fit::lanes::lane_batch_bytes` (change together), the batch's outcome
/// columns, its training targets and `T`, `b` PLS1 fits, `R` and `P`, the
/// coefficients and the held-out scores.
#[allow(clippy::similar_names)] // n_tr, n_te
const fn lane_batch(n_tr: usize, n_te: usize, p: usize, k: usize, b: usize) -> usize {
    copy(n_tr + n_te, b)
        + 2 * copy(n_tr, b)
        + b * pls1_fit_bytes(p, k)
        + 2 * copy(p, b)
        + copy(p, b)
        + copy(n_te, b)
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

/// One public call of a batched loop and `unit`, a mirror of the bytes of
/// one unit of the route it takes, at one thread. Its low budget is the
/// call's final `needed` (`final_needed`), found by running it; its middle
/// budget is [`Case::mid`] over that.
struct Case<'a> {
    name: &'static str,
    unit: usize,
    run: Box<dyn Fn() -> PlsKitResult<String> + Sync + 'a>,
}

impl Case<'_> {
    /// The middle budget over a final `needed` of `low` on a pool, which is
    /// the route's `call + u` there: `low + 2.5·unit`. That runs three units
    /// at once when `unit = u`, and two to four while the mirror is within
    /// 0.4 and 1.6 times `u` (a per-thread term on the pool of 4), so always
    /// fewer than the `P_UNITS` the unset run takes at once.
    fn mid(&self, low: u64) -> u64 {
        low + self.unit as u64 * 5 / 2
    }
}

fn case<'a>(
    name: &'static str,
    unit: usize,
    run: impl Fn() -> PlsKitResult<String> + Sync + 'a,
) -> Case<'a> {
    Case {
        name,
        unit,
        run: Box::new(run),
    }
}

/// The budget at which `at` first runs, its result there and the number of
/// `MemoryBudget` rounds before it: `at(1)` must fail with `MemoryBudget`,
/// then `at` runs at each `needed` it returns (a later loop of the call can
/// name more) until it returns `Ok`, at most eight times after the first.
/// Panics on any other error, on a `needed` that does not exceed the budget
/// it failed at, and when the rounds run out.
fn final_needed<T>(name: &str, mut at: impl FnMut(u64) -> PlsKitResult<T>) -> (u64, T, usize) {
    let mut budget = 1;
    for round in 0..=8 {
        match at(budget) {
            Ok(out) => {
                assert!(round > 0, "{name}: ran at budget 1");
                return (budget, out, round);
            }
            Err(PlsKitError::MemoryBudget { needed, .. }) if needed > budget => budget = needed,
            Err(e) => panic!("{name} at budget {budget}: {e:?}"),
        }
    }
    panic!("{name}: MemoryBudget after 8 rounds, the last at {budget}")
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

/// The `Debug` rendering of a call's output.
fn render<T: std::fmt::Debug>(out: PlsKitResult<T>) -> PlsKitResult<String> {
    out.map(|o| format!("{o:?}"))
}

fn pls1_test(
    x: &Mat<f64>,
    y: &Col<f64>,
    k: usize,
    weights: Option<&Col<f64>>,
    args: ConfirmatoryArgs,
    ci: Option<CIOpts>,
    seed: u64,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: Some(&[k]),
            weights: weights.map(Col::as_ref),
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args,
            seed: Some(seed),
            ci,
            ..Default::default()
        },
    )
}

fn pls3_test(
    x: &Mat<f64>,
    y: &Mat<f64>,
    args: ConfirmatoryArgs,
    keep_x: Option<usize>,
    keep_y: Option<usize>,
    seed: u64,
) -> PlsKitResult<ConfirmatoryTestOutput> {
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
}

/// The `Debug` rendering of a `split_nb` result, after checking that
/// `split_nb` ran (`force: true` skips the auto-gate).
fn split_nb_out(out: PlsKitResult<ConfirmatoryTestOutput>) -> PlsKitResult<String> {
    out.map(|o| {
        assert_eq!(o.test_method, "split_nb");
        format!("{o:?}")
    })
}

const SPLIT_NB: ConfirmatoryArgs = ConfirmatoryArgs::SplitNb {
    n_splits: P_UNITS,
    force: true,
};
const SPLIT_EXACT: ConfirmatoryArgs = ConfirmatoryArgs::SplitExact {
    n_perm: 19,
    n_splits: P_UNITS,
};

fn find_k(
    x: &Mat<f64>,
    y: &Col<f64>,
    w: Option<&Col<f64>>,
    seed: u64,
) -> PlsKitResult<FindKOptimalOutput> {
    pls1_find_k_optimal(
        x.as_ref(),
        y.as_ref(),
        2,
        w.map(Col::as_ref),
        None,
        FindKOptimalOpts {
            selector: Selector::R2Se,
            n_folds: P_UNITS,
            seed: Some(seed),
            ..Default::default()
        },
    )
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
    w_auto: Col<f64>,
    x3s: Mat<f64>,
    y3s: Mat<f64>,
    xg: Mat<f64>,
    yg: Col<f64>,
}

fn parity_data() -> ParityData {
    let (x, y) = synth(PN, PP, 1.0, 1);
    let (x3, y3) = two_block(PN, PP, PQ, 2);
    let (x_auto, y_auto) = synth(AUTO_N, AUTO_P, 1.0, 3);
    let (xg, yg) = synth(GRAM_P_N, GRAM_P_P, 0.3, 62);
    let (x3s, y3s) = two_block(PN, PS_P, PQ, 4);
    ParityData {
        x,
        y,
        w: row_weights(PN),
        xc: constant_x(PN, PP),
        xc_wide: constant_x(PN, PP_WIDE),
        x3,
        y3,
        x_auto,
        y_auto,
        w_auto: row_weights(AUTO_N),
        x3s,
        y3s,
        xg,
        yg,
    }
}

/// The cases, one or more per batched loop, each with the route it takes
/// (the route log of `src/memory_budget_cases.rs` checks it) and the bytes
/// of one unit of that route.
#[allow(clippy::too_many_lines)]
fn parity_cases(d: &ParityData) -> Vec<Case<'_>> {
    let (x, y, w) = (&d.x, &d.y, &d.w);
    // Two blocks of `cols` columns on the training rows and on the other
    // rows of a split or a fold: two matrices, padded apart
    // (`signal_test::split_blocks_bytes`, `resample::fold_blocks_bytes`; change
    // together).
    let split_blocks = |cols| 2 * copy(P_SPLIT_TR, cols);
    let fold_blocks = |cols| copy(P_FOLD_TR, cols) + copy(PN - P_FOLD_TR, cols);
    // The n-space and no-refit route G units: `G` and `M` of a split
    // (`signal_test::split_unit_bytes`' `Nspace` arm, `nr_split_unit_bytes`)
    // or of a fold.
    let split_gram = split_blocks(P_SPLIT_TR);
    // A no-refit split on route G (`signal_test::nr_split_unit_bytes`):
    // `K_A` and `M`, and the split's rows of `cols` outcome columns, `y_tr`,
    // `y_te` and the scores `t_te`.
    let nr_gram = |cols| split_gram + copy(P_SPLIT_TR, cols) + 2 * copy(PN - P_SPLIT_TR, cols);
    let fold_gram = fold_blocks(P_FOLD_TR);
    // A PLS3 n-space unit at one thread (`pls3_signal_test::gram_unit_bytes`):
    // `G` and `M`, and one column's copies of Y and `G·Ỹ_tr`.
    let pls3_gram = split_gram + y_column(P_SPLIT_TR, PN - P_SPLIT_TR, PQ) + copy(P_SPLIT_TR, PQ);
    // A PLS3 primal unit at one thread (`primal_split_unit_bytes`): the
    // halves of X and one column's LV1 fit and copies of Y.
    let pls3_primal = split_blocks(PP) + lv1(PP, PQ) + y_column(P_SPLIT_TR, PN - P_SPLIT_TR, PQ);
    let raw_perm = |n_perm, n_folds| ConfirmatoryArgs::RawPerm { n_perm, n_folds };
    vec![
        // `split_nb` on the n-space route (dense, unweighted, the gate's
        // Gram matrix held).
        case(
            "pls1 split_nb k=2",
            split_gram + nspace_run(PN, P_SPLIT_TR, 2, 1),
            move || split_nb_out(pls1_test(x, y, 2, None, SPLIT_NB, None, 11)),
        ),
        // Weights take `split_nb` off the n-space route onto the primal
        // refit (`split_half_correlations`; `split_half_unit_bytes`): the
        // prepared halves of X and one PLS1 fit.
        case(
            "pls1 split_nb k=2 weighted",
            split_blocks(PP) + pls1_fit_bytes(PP, 2),
            move || split_nb_out(pls1_test(x, y, 2, Some(w), SPLIT_NB, None, 12)),
        ),
        // The same primal refit with each split's fit past the `Auto`
        // threshold, so a batch runs fits that use the pool themselves.
        case(
            "pls1 split_nb k=2 200x6000 weighted",
            2 * copy(AUTO_N / 2, AUTO_P) + pls1_fit_bytes(AUTO_P, 2),
            move || {
                split_nb_out(pls1_test(
                    &d.x_auto,
                    &d.y_auto,
                    2,
                    Some(&d.w_auto),
                    SPLIT_NB,
                    None,
                    32,
                ))
            },
        ),
        // `split_nb_correlations_nr` on K = 1 route G, which decides a
        // constant X from the blocks of the gate's Gram matrix.
        case("pls1 split_nb k=1 constant X", nr_gram(1), move || {
            split_nb_out(pls1_test(&d.xc, y, 1, None, SPLIT_NB, None, 13))
        }),
        // `split_zbars_columns` on the n-space route (dense, unweighted,
        // k = 2).
        case(
            "split_exact k=2",
            split_gram + nspace_run(PN, P_SPLIT_TR, 2, 16),
            move || render(pls1_test(x, y, 2, None, SPLIT_EXACT, None, 14)),
        ),
        // The no-refit route on K = 1 route G.
        case("split_exact k=1", nr_gram(20), move || {
            render(pls1_test(x, y, 1, None, SPLIT_EXACT, None, 15))
        }),
        // The same route on a constant X, which route G decides as the zero
        // model on every split.
        case("split_exact k=1 constant X", nr_gram(20), move || {
            render(pls1_test(&d.xc, y, 1, None, SPLIT_EXACT, None, 16))
        }),
        // `pooled_cv_r2_columns` on the primal route (weights take it off
        // the n-space route; p >> n_tr keeps it off the p-space one). At
        // `low` it takes the gather source, which holds no copy of the fold;
        // from `mid` up the copy source: the fold's two blocks of X and one
        // batch of lanes for all 19 + 1 columns.
        case(
            "raw_perm k=2 weighted",
            fold_blocks(PP) + lane_batch(P_FOLD_TR, PN - P_FOLD_TR, PP, 2, 20),
            move || render(pls1_test(x, y, 2, Some(w), raw_perm(19, P_UNITS), None, 17)),
        ),
        // `pooled_cv_r2_columns` on the n-space route: the fold's `G` and
        // `M`, and one column's kernel call.
        case(
            "raw_perm k=2",
            fold_gram + nspace_run(PN, P_FOLD_TR, 2, 1),
            move || render(pls1_test(x, y, 2, None, raw_perm(19, P_UNITS), None, 18)),
        ),
        // `pooled_cv_r2_columns` on the n-space route at k = 1: the fold's
        // `G` and `M`, and one column's kernel call.
        case(
            "raw_perm k=1",
            fold_gram + nspace_run(PN, P_FOLD_TR, 1, 1),
            move || render(pls1_test(x, y, 1, None, raw_perm(39, P_UNITS), None, 19)),
        ),
        // The same route on a constant X, which leaves every column zero.
        case(
            "raw_perm k=1 constant X wide",
            fold_gram + nspace_run(PN, P_FOLD_TR, 1, 1),
            move || {
            render(pls1_test(
                &d.xc_wide,
                y,
                1,
                None,
                raw_perm(39, P_UNITS),
                None,
                20,
            ))
        }),
        // `pooled_cv_r2_columns` on the p-space Gram route: the fold's two
        // blocks of X (5 folds of 400 rows), `C` (`p × p`) and one PLS1 fit
        // per column that runs at once, at one thread.
        case(
            "raw_perm gram-p 2000x50 k=2",
            copy(GRAM_P_N * 4 / 5, GRAM_P_P)
                + copy(GRAM_P_N / 5, GRAM_P_P)
                + copy(GRAM_P_P, GRAM_P_P)
                + pls1_fit_bytes(GRAM_P_P, 2),
            move || {
                render(pls1_test(
                    &d.xg,
                    &d.yg,
                    2,
                    None,
                    raw_perm(300, P_UNITS),
                    None,
                    33,
                ))
            },
        ),
        // `pls3_split_lv_correlations` on the primal route (a sparse X
        // side).
        case("pls3 split_nb keep_x=40", pls3_primal, move || {
            split_nb_out(pls3_test(&d.x3, &d.y3, SPLIT_NB, Some(40), None, 21))
        }),
        // The n-space route (dense, `n_tr < p`).
        case("pls3 split_nb dense", pls3_gram, move || {
            split_nb_out(pls3_test(&d.x3, &d.y3, SPLIT_NB, None, None, 22))
        }),
        // The n-space route on a constant X: every split is the zero model.
        case("pls3 split_nb dense constant X", pls3_gram, move || {
            split_nb_out(pls3_test(&d.xc, &d.y3, SPLIT_NB, None, None, 23))
        }),
        // The streamed body (`Primal(Gather)`, `pls3_split_lv_correlations_streamed`):
        // `n_tr ≥ p` leaves no Gram route (`streamed_unit_bytes`): one LV1
        // fit, the stream blocks, the halves of Y and the summed held-out
        // scores.
        case(
            "pls3 split_nb dense 40x15",
            lv1(PS_P, PQ)
                + stream_blocks(P_SPLIT_TR, PN - P_SPLIT_TR)
                + copy(P_SPLIT_TR, PQ)
                + 2 * copy(PN - P_SPLIT_TR, PQ),
            move || split_nb_out(pls3_test(&d.x3s, &d.y3s, SPLIT_NB, None, None, 34)),
        ),
        // `pls3_split_zbars_columns` on the n-space route; the copies of Y
        // count once per thread.
        case("pls3 split_exact dense", pls3_gram, move || {
            render(pls3_test(&d.x3, &d.y3, SPLIT_EXACT, None, None, 24))
        }),
        // `pls3_split_zbars_columns_primal` (a sparse X side); the LV1 fit
        // and copies of Y count once per thread.
        case("pls3 split_exact keep_x=40", pls3_primal, move || {
            render(pls3_test(&d.x3, &d.y3, SPLIT_EXACT, Some(40), None, 25))
        }),
        // `pls3_split_zbars_columns` on a constant X with a sparse Y side,
        // on the n-space route: every column is the zero model.
        case(
            "pls3 split_exact constant X wide keep_y=1",
            pls3_gram,
            move || render(pls3_test(&d.xc_wide, &d.y3, SPLIT_EXACT, None, Some(1), 26)),
        ),
        // `cv_select`: one fold fit on X in place, weighted or not
        // (`k_max = 2` gives `max_comp = 2`; the keep grid fits at k = 2).
        case(
            "pls1_find_k_optimal r2_se k_max=2",
            cv_fold(PP, 2, PN - P_FOLD_TR),
            move || render(find_k(x, y, None, 27)),
        ),
        case(
            "pls1_find_k_optimal r2_se k_max=2 weighted",
            cv_fold(PP, 2, PN - P_FOLD_TR),
            move || render(find_k(x, y, Some(w), 28)),
        ),
        case(
            "spls1_find_keep_optimal k=2",
            cv_fold(PP, 2, PN - P_FOLD_TR),
            move || {
                render(spls1_find_keep_optimal(
                    x.as_ref(),
                    y.as_ref(),
                    2,
                    None,
                    None,
                    FindKeepOptimalOpts {
                        n_folds: P_UNITS,
                        seed: Some(29),
                    },
                ))
            },
        ),
        // `profiles_find_n`: the shuffles of its parallel analysis and its
        // two fold passes. A fold of the first pass (`P_FOLD_TR` training
        // rows) is the largest unit, so it sets `low`; at `mid` each loop
        // runs a few units at once.
        case("profiles_find_n k=2", fold_svd(P_FOLD_TR, PP), move || {
            let opts = ProfilesFindNOpts {
                seed: Some(31),
                ..Default::default()
            };
            render(profiles_find_n(x.as_ref(), y.as_ref(), 2, 9, opts))
        }),
        // `pls1_subsample_inference_confirmatory`: two n-row copies of X
        // and one PLS1 fit per replicate. `score` performs no fit, so the
        // CI is the only loop.
        case(
            "ci (score) k=2",
            2 * copy(PN, PP) + pls1_fit_bytes(PP, 2),
            move || {
                let ci = CIOpts {
                    n_boot: P_BOOT,
                    m_rate: P_M_RATE,
                    ..CIOpts::default()
                };
                render(pls1_test(
                    x,
                    y,
                    2,
                    None,
                    ConfirmatoryArgs::Score { n_perm: None },
                    Some(ci),
                    30,
                ))
            },
        ),
    ]
}
