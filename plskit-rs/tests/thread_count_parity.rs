//! Byte-parity across Rayon pool sizes.
//!
//! One process has one global pool size, so a result whose bits depend on
//! *how many* threads the pool has cannot show up in a single run. This file
//! runs the same public call inside Rayon pools of 1, 3 and 5 threads
//! (`ThreadPoolBuilder::install`) in one process and requires the outputs
//! to be bitwise identical.
//!
//! Outputs are compared through their `Debug` rendering. Rust prints an
//! `f64` with `{:?}` as the shortest string that round-trips to the same
//! value, so two finite values render identically exactly when their bits
//! are equal (and `-0.0` renders apart from `0.0`); faer's `Mat` / `Col`
//! `Debug` prints every entry that way. That covers every field of every
//! output struct without a per-type comparison, which is the point: a new
//! field is covered the day it is added. NaN payloads are not
//! distinguished, which no result contract depends on.
//!
//! The pool comparison only has teeth where some product is past a
//! pool-sensitive split: `ParChoice::Auto`'s `n·d·k >= 1e6` (`n·p·q` for
//! PLS3), faer's `256²`-entry column-major GEMV split, or the parallel
//! threshold of its EVD / SVD. Every case below marked "pool" differs
//! across pool sizes when a pool-sized `Par` is used. Cases
//! marked "guard only" sit below those thresholds, so the pool comparison
//! cannot fail there today; they are kept for the global-parallelism guard
//! below, which fails on them the day a path they reach reads faer's
//! global. The guard does not see a pool-sized `Par` passed explicitly
//! (`Par::rayon(0)`), so only the "pool" cases pin that.
//!
//! Every call also runs with faer's global parallelism disabled
//! (`faer::disable_global_parallelism`, process-wide, which is why this is
//! its own test binary). faer's operator `*` and its high-level
//! decompositions read that global, whose default `Par::rayon(0)` has the
//! pool's size as its degree; with it disabled, any such read on a path
//! these calls reach panics, naming the site, instead of passing by luck
//! at a shape too small to split.

mod common;

use std::fmt::Debug;

use common::{perm_opts, synth};
use faer::{Col, Mat, MatRef};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, pls1_fit, pls1_perm_null,
    pls1_rotation_stability, pls3_confirmatory_test, pls3_fit, split_nb_gate,
    spls1_find_k_sequence, spls1_fit, spls3_fit, CIOpts, ConfirmatoryArgs, ConfirmatoryMethod,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, FindKOptimalOpts, FindKSequenceOpts, FitOpts,
    KSpec, PermNullOpts, Pls3ConfirmatoryTestOpts, Pls3FitOpts, RotationStabilityMethod,
    RotationStabilityOpts, Selector, VarimaxArgs,
};

const POOL_SIZES: [usize; 3] = [1, 3, 5];

#[allow(clippy::many_single_char_names)]
fn two_block(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
    let y = Mat::<f64>::from_fn(n, q, |i, j| x[(i, j)] + 0.5 * rng.random_range(-1.0..1.0));
    (x, y)
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
}

/// Collects every case whose output depends on the pool size, so a failing
/// test names all of them rather than the first.
#[derive(Default)]
struct Checker {
    failures: Vec<String>,
}

impl Checker {
    /// Run `f` inside a pool of each size in `POOL_SIZES` and record a
    /// failure unless the `Debug` renderings of the results are identical.
    fn check<T: Debug>(&mut self, name: &str, f: impl Fn() -> T + Sync) {
        faer::disable_global_parallelism();
        let outs: Vec<String> = POOL_SIZES
            .iter()
            .map(|&t| pool(t).install(|| format!("{:?}", f())))
            .collect();
        for (t, out) in POOL_SIZES.iter().zip(&outs).skip(1) {
            if *out != outs[0] {
                let at = outs[0]
                    .bytes()
                    .zip(out.bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| outs[0].len().min(out.len()));
                let lo = at.saturating_sub(100);
                let ctx = |s: &str| s.get(lo..(at + 40).min(s.len())).unwrap_or("").to_owned();
                self.failures.push(format!(
                    "{name}: 1-thread and {t}-thread pools differ at byte {at}\n  \
                     1 thread : ...{}\n  {t} threads: ...{}",
                    ctx(&outs[0]),
                    ctx(out)
                ));
                return;
            }
        }
    }

    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} case(s) depend on the pool size:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

#[test]
fn faer_rayon_degree_follows_the_installed_pool() {
    // The mechanism behind every case below: `Par::rayon(0)`, the default
    // of faer's global parallelism, resolves to
    // `rayon::current_num_threads()` at the call, i.e. the size of the pool
    // the call runs in, and faer's split (and so the rounding) follows that
    // degree. `Par::rayon(n)` with `n > 0` does not.
    for t in POOL_SIZES {
        let degree = pool(t).install(|| faer::Par::rayon(0).degree());
        assert_eq!(degree, t, "Par::rayon(0) inside a {t}-thread pool");
        let degree = pool(t).install(|| faer::Par::rayon(8).degree());
        assert_eq!(degree, 8, "Par::rayon(8) inside a {t}-thread pool");
    }
}

#[test]
fn pls1_fit_auto_is_pool_size_invariant() {
    let mut c = Checker::default();
    // Pool: n·d·k >= 1e6 in every case, so `ParChoice::Auto` takes the
    // Rayon arm.
    for (n, d, k) in [
        (2000usize, 600usize, 1usize),
        (2000, 600, 5),
        (20_000, 60, 5),
    ] {
        let (x, y) = synth(n, d, 1.0, 3);
        c.check(&format!("pls1_fit {n}x{d} k={k}"), || {
            pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(k),
                None,
                FitOpts::default(),
            )
            .unwrap()
        });
    }
    // The Python seam hands the kernel a row-major *view* of `X`
    // (`np_mat_view`, `MatRef::from_row_major_slice`) instead of a
    // column-major copy. The standardized X keeps a row-major input's
    // layout, so that view must be pool-invariant on its own.
    let (x_col_major, y5) = synth(2000, 600, 1.0, 3);
    let mut x_row_major: Vec<f64> = Vec::with_capacity(2000 * 600);
    for i in 0..2000 {
        for j in 0..600 {
            x_row_major.push(x_col_major[(i, j)]);
        }
    }
    let x_row_view = MatRef::from_row_major_slice(&x_row_major, 2000, 600);
    c.check("pls1_fit row-major view 2000x600 k=5", || {
        pls1_fit(
            x_row_view,
            y5.as_ref(),
            KSpec::Fixed(5),
            None,
            FitOpts::default(),
        )
        .unwrap()
    });
    // Pool: a row-major view at least 2048 columns wide forms `Xs'·t` with
    // plskit's own row split (a fixed number of row pieces, summed in piece
    // order), not faer's GEMV, so it is pinned here on its own. At k = 3,
    // n·d·k = 3.0e6 clears the 1e6 Rayon threshold threefold, so a retuned
    // threshold keeps this case on the row split. The transpose of a stored
    // transpose is a row-major view.
    let (x_wide, y_wide) = synth(203, 5000, 1.0, 5);
    let x_wide_t = x_wide.transpose().to_owned();
    c.check("pls1_fit row-major view 203x5000 k=3", || {
        pls1_fit(
            x_wide_t.transpose(),
            y_wide.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap()
    });

    let (x, y) = synth(2000, 600, 1.0, 4);
    let w = row_weights(2000);
    c.check("pls1_fit weighted 2000x600 k=5", || {
        pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(5),
            Some(w.as_ref()),
            FitOpts::default(),
        )
        .unwrap()
    });
    c.check("spls1_fit 2000x600 k=5 keep=100", || {
        spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(5),
            100,
            None,
            FitOpts::default(),
        )
        .unwrap()
    });
    c.finish();
}

#[test]
fn pls3_fits_auto_are_pool_size_invariant() {
    let mut c = Checker::default();
    // Pool: n·p·q = 200·2000·6 >= 1e6, the Rayon arm of `resolve_par`.
    let (x, y) = two_block(200, 2000, 6, 31);
    c.check("pls3_fit 200x2000x6 k=3", || {
        pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap()
    });
    // Pool: sparse PLS3 needs a parallel split inside the alternation's
    // `A v` products, i.e. `p·q >= 256²` for the p x q cross-covariance
    // (20000·4 here); at 200x2000x6 `spls3_fit` stays below that
    // split.
    let (xs, ys) = two_block(100, 20_000, 4, 32);
    c.check("spls3_fit 100x20000x4 k=2 keep=500,2", || {
        spls3_fit(
            xs.as_ref(),
            ys.as_ref(),
            2,
            500,
            2,
            None,
            Pls3FitOpts::default(),
        )
        .unwrap()
    });
    // The Python seam hands both fits a C-ordered `X` as a row-major view,
    // standardized or `pre_standardized_x` (copied column-major by the
    // fit): each must be pool-invariant in that layout too.
    let row_major = |x: &Mat<f64>| -> Vec<f64> {
        (0..x.nrows())
            .flat_map(|i| (0..x.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| x[(i, j)])
            .collect()
    };
    let dense_buf = row_major(&x);
    let dense_view = MatRef::from_row_major_slice(&dense_buf, x.nrows(), x.ncols());
    let sparse_buf = row_major(&xs);
    let sparse_view = MatRef::from_row_major_slice(&sparse_buf, xs.nrows(), xs.ncols());
    for pre in [false, true] {
        let opts = Pls3FitOpts {
            pre_standardized_x: pre,
            ..Pls3FitOpts::default()
        };
        c.check(
            &format!("pls3_fit row-major 200x2000x6 k=3 pre={pre}"),
            || pls3_fit(dense_view, y.as_ref(), 3, None, opts).unwrap(),
        );
        c.check(
            &format!("spls3_fit row-major 100x20000x4 k=2 keep=500,2 pre={pre}"),
            || spls3_fit(sparse_view, ys.as_ref(), 2, 500, 2, None, opts).unwrap(),
        );
    }
    c.finish();
}

fn confirmatory(
    x: &Mat<f64>,
    y: &Col<f64>,
    k: usize,
    weights: Option<&Col<f64>>,
    args: ConfirmatoryArgs,
    ci: Option<CIOpts>,
    seed: u64,
) -> plskit::ConfirmatoryTestOutput {
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

fn row_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| 0.5 + f64::from(u32::try_from(i % 4).unwrap()) * 0.25)
}

const CI: CIOpts = CIOpts {
    n_boot: 100,
    m_rate: 0.7,
    level: 0.9,
    max_failure_rate: 0.5,
};

#[test]
fn split_exact_primal_is_pool_size_invariant() {
    let mut c = Checker::default();
    // Pool: 40x4000 dense, k = 3 (past the n-space route's
    // cap, so the primal refit and its held-out scoring GEMV).
    let (x, y) = synth(40, 4000, 1.0, 5);
    let split_exact = ConfirmatoryArgs::SplitExact {
        n_perm: 50,
        n_splits: 20,
    };
    c.check("split_exact 40x4000 k=3", || {
        confirmatory(&x, &y, 3, None, split_exact, None, 23)
    });
    // Pool: the weighted primal arm, same shape.
    let w = row_weights(40);
    c.check("split_exact weighted 40x4000 k=3", || {
        confirmatory(&x, &y, 3, Some(&w), split_exact, None, 24)
    });
    // Pool: tall (Gram-p route) with the CI bundle.
    let (xt, yt) = synth(5000, 50, 1.0, 25);
    c.check("split_exact 5000x50 k=2 + ci", || {
        confirmatory(
            &xt,
            &yt,
            2,
            None,
            ConfirmatoryArgs::SplitExact {
                n_perm: 50,
                n_splits: 4,
            },
            Some(CI),
            26,
        )
    });
    // Guard only (not shown to be pool-sensitive): the Gram-p refit route
    // with `keep` (shape pinned in
    // `fixture_route_pins::thread_count_parity_shapes_take_their_routes`).
    let (xk, yk) = synth(2000, 200, 0.3, 63);
    c.check("split_exact gram-p 2000x200 k=1 keep=10", || {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: xk.as_ref(),
                y: yk.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 400,
                    n_splits: 5,
                },
                seed: Some(3033),
                keep: Some(10),
                ..Default::default()
            },
        )
        .unwrap()
    });
    c.finish();
}

#[test]
fn split_exact_nspace_is_pool_size_invariant() {
    let mut c = Checker::default();
    // Guard only: the n-space refit route (dense k = 2, 40x2000; the shape
    // and `n_perm` are pinned in
    // `fixture_route_pins::thread_count_parity_shapes_take_their_routes`). Its 58
    // outcome columns run as three full runs of `dual_route::NSPACE_BATCH`
    // (16) and a partial one of 10.
    let (x, y) = synth(40, 2000, 1.0, 31);
    let split_exact = ConfirmatoryArgs::SplitExact {
        n_perm: 57,
        n_splits: 10,
    };
    c.check("split_exact n-space 40x2000 k=2", || {
        confirmatory(&x, &y, 2, None, split_exact, None, 32)
    });
    // Same route on a rank-1 X: k = 2 exceeds the rank, so every column
    // leaves the kernel unresolved and runs the primal fallback map.
    let (left, right) = (synth(40, 2, 0.0, 33).0, synth(2000, 2, 0.0, 34).0);
    let x1 = Mat::<f64>::from_fn(40, 2000, |i, j| left[(i, 0)] * right[(j, 0)]);
    c.check("split_exact n-space rank-1 40x2000 k=2", || {
        confirmatory(&x1, &y, 2, None, split_exact, None, 35)
    });
    c.finish();
}

#[test]
fn other_pls1_confirmatory_methods_are_pool_size_invariant() {
    let mut c = Checker::default();
    let (x, y) = synth(40, 4000, 1.0, 6);
    // Pool: `split_nb`'s statistic and `stable_rank`.
    c.check("split_nb 40x4000 k=3", || {
        confirmatory(
            &x,
            &y,
            3,
            None,
            ConfirmatoryArgs::SplitNb {
                n_splits: 20,
                force: true,
            },
            None,
            12,
        )
    });
    // Pool: score and e read the full design once (`X'y`, the Gram matrix
    // and its eigenvalues; e scores a 1000x600 half), past the GEMV split
    // and the EVD's parallel threshold.
    let (xt, yt) = synth(2000, 600, 1.0, 7);
    c.check("score 2000x600 k=1", || {
        confirmatory(&xt, &yt, 1, None, ConfirmatoryArgs::Score, None, 13)
    });
    c.check("e 2000x600 k=2", || {
        confirmatory(&xt, &yt, 2, None, ConfirmatoryArgs::E, None, 13)
    });
    // Pool: the `split_nb` auto-gate's `stable_rank`.
    c.check("split_nb_gate 300x400", || {
        split_nb_gate(xt.as_ref().subrows(0, 300).subcols(0, 400), None).unwrap()
    });
    // Guard only: a 10x4000 validation fold is under the GEMV split.
    c.check("raw_perm 40x4000 k=3", || {
        confirmatory(
            &x,
            &y,
            3,
            None,
            ConfirmatoryArgs::RawPerm {
                n_perm: 20,
                n_folds: 4,
            },
            None,
            11,
        )
    });
    // Guard only (not shown to be pool-sensitive): `raw_perm` on the n-space
    // and the p-space Gram routes (shapes pinned in
    // `fixture_route_pins::thread_count_parity_shapes_take_their_routes`).
    // n_folds = 20 puts n_tr·d = 38·2000 past faer's GEMV split, but no pool
    // size has been seen to change its output.
    let (xw, yw) = synth(40, 2000, 1.0, 3);
    for n_folds in [5, 20] {
        c.check(
            &format!("raw_perm n-space 40x2000 k=2 n_folds={n_folds}"),
            || {
                confirmatory(
                    &xw,
                    &yw,
                    2,
                    None,
                    ConfirmatoryArgs::RawPerm {
                        n_perm: 50,
                        n_folds,
                    },
                    None,
                    21,
                )
            },
        );
    }
    let (xg, yg) = synth(2000, 50, 0.3, 62);
    c.check("raw_perm gram-p 2000x50 k=2", || {
        confirmatory(
            &xg,
            &yg,
            2,
            None,
            ConfirmatoryArgs::RawPerm {
                n_perm: 300,
                n_folds: 5,
            },
            None,
            3032,
        )
    });
    c.finish();
}

#[test]
fn subsample_ci_holdout_is_pool_size_invariant() {
    let mut c = Checker::default();
    // Pool: the CI bundle's holdout correlation scores the held-out rows
    // with the subsample fit's β; 30 holdout rows x 4000 features clears
    // faer's GEMV split.
    let (x, y) = synth(100, 4000, 1.0, 8);
    c.check("score + ci 100x4000 k=2", || {
        confirmatory(&x, &y, 2, None, ConfirmatoryArgs::Score, Some(CI), 14)
    });
    c.finish();
}

#[test]
fn perm_null_is_pool_size_invariant() {
    let mut c = Checker::default();
    // Pool: the reference fit runs under `ParChoice::Auto` (n·d·k >= 1e6).
    let (x, y) = synth(2000, 600, 1.0, 9);
    c.check("perm_null 2000x600 k=2", || {
        pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, perm_opts(), Some(15)).unwrap()
    });
    // Guard only: wide at k = 3, the primal permutation refits (Seq).
    let (x, y) = synth(40, 4000, 1.0, 10);
    c.check("perm_null 40x4000 k=3", || {
        pls1_perm_null(x.as_ref(), y.as_ref(), 3, None, perm_opts(), Some(16)).unwrap()
    });
    // Guard only (not shown to be pool-sensitive): the public
    // calls on the n-space and the p-space Gram routes (shapes pinned in
    // `fixture_route_pins::thread_count_parity_shapes_take_their_routes`). Their
    // block builds are pinned kernel-level by
    // `dual_route::multi_k::tests::gram_products_are_thread_count_invariant_at_the_route_shapes`
    // and `gram_p::tests_block::c_build_is_thread_count_invariant_at_the_route_shapes`.
    let (x, y) = synth(40, 2000, 1.0, 12);
    c.check("perm_null n-space 40x2000 k=2", || {
        pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, perm_opts(), Some(17)).unwrap()
    });
    // `n_perm = 300`: at `perm_opts()`'s 100 this shape is under the p-space
    // work floor and runs the primal route.
    let (x, y) = synth(2000, 50, 0.3, 13);
    c.check("perm_null gram-p 2000x50 k=2", || {
        let opts = PermNullOpts {
            n_perm: 300,
            ..perm_opts()
        };
        pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts, Some(18)).unwrap()
    });
    c.finish();
}

#[test]
fn find_k_and_rotation_stability_are_pool_size_invariant() {
    let mut c = Checker::default();
    // Pool: the BIC sweep's full fit and its scoring run on the fixed split.
    let (xt, yt) = synth(2000, 600, 1.0, 21);
    c.check("find_k_optimal bic 2000x600", || {
        pls1_find_k_optimal(
            xt.as_ref(),
            yt.as_ref(),
            3,
            None,
            FindKOptimalOpts {
                selector: Selector::Bic,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap()
    });
    // Pool: the sequence's deflation fits and products.
    c.check("find_k_sequence split_nb 2000x600", || {
        pls1_find_k_sequence(
            xt.as_ref(),
            yt.as_ref(),
            2,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 6,
                force: true,
                alpha: 0.5,
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap()
    });
    // Pool: the reference fit runs under `ParChoice::Auto`.
    c.check("rotation_stability 2000x600 k=2", || {
        pls1_rotation_stability(
            xt.as_ref(),
            yt.as_ref(),
            2,
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            None,
            None,
            RotationStabilityOpts {
                n_boot: 100,
                seed: Some(4),
                ..Default::default()
            },
        )
        .unwrap()
    });
    // Guard only: 60x3000 folds, halves and subsamples are under every
    // split.
    let (x, y) = synth(60, 3000, 1.0, 11);
    c.check("find_k_optimal 60x3000", || {
        pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            FindKOptimalOpts {
                selector: Selector::R2Se,
                n_folds: 5,
                seed: Some(17),
                ..Default::default()
            },
        )
        .unwrap()
    });
    c.check("find_k_sequence 60x3000", || {
        pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitExact,
                n_perm: 20,
                n_splits: 4,
                alpha: 0.5,
                seed: Some(18),
                ..Default::default()
            },
        )
        .unwrap()
    });
    c.check("rotation_stability 60x3000 k=2", || {
        pls1_rotation_stability(
            x.as_ref(),
            y.as_ref(),
            2,
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            None,
            None,
            RotationStabilityOpts {
                n_boot: 100,
                seed: Some(19),
                ..Default::default()
            },
        )
        .unwrap()
    });
    c.finish();
}

#[test]
fn sparse_weighted_sequence_is_pool_size_invariant() {
    let mut c = Checker::default();
    // Guard only: the sparse weighted sequence (deflation, then one
    // confirmatory test per k) under weights, keep and `pre_standardized`,
    // on the two refitting engines. A weak signal and alpha = 0.95 keep
    // every step running, with p-values off the 1 / (n_perm + 1) floor.
    let (xs, ys) = {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
        let x = Mat::<f64>::from_fn(60, 8, |_, _| rng.random_range(-1.0..1.0));
        let noise = Col::<f64>::from_fn(60, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(60, |i| x[(i, 0)] * 0.5 + noise[i]);
        (x, y)
    };
    let ws = Col::<f64>::from_fn(60, |i| {
        if i % 9 == 8 {
            0.0
        } else {
            0.5 + f64::from(u32::try_from(i % 5).unwrap()) * 0.25
        }
    });
    for test_method in [ConfirmatoryMethod::RawPerm, ConfirmatoryMethod::SplitExact] {
        c.check(
            &format!("spls1_find_k_sequence 60x8 {test_method:?}"),
            || {
                spls1_find_k_sequence(
                    xs.as_ref(),
                    ys.as_ref(),
                    3,
                    3,
                    Some(ws.as_ref()),
                    FindKSequenceOpts {
                        test_method,
                        n_perm: 50,
                        n_splits: 4,
                        alpha: 0.95,
                        pre_standardized: true,
                        seed: Some(31),
                        ..Default::default()
                    },
                )
                .unwrap()
            },
        );
    }
    c.finish();
}

#[test]
fn pls3_confirmatory_is_pool_size_invariant() {
    let mut c = Checker::default();
    let pls3_test = |x: &Mat<f64>, y: &Mat<f64>, args: ConfirmatoryArgs, seed: u64| {
        pls3_confirmatory_test(
            x.as_ref(),
            y.as_ref(),
            1,
            Pls3ConfirmatoryTestOpts {
                args,
                seed: Some(seed),
                ..Default::default()
            },
        )
        .unwrap()
    };
    // Pool: dense `split_nb` at q = 4.
    let (x4, y4) = two_block(60, 3000, 4, 11);
    c.check("pls3 split_nb 60x3000x4", || {
        pls3_test(
            &x4,
            &y4,
            ConfirmatoryArgs::SplitNb {
                n_splits: 6,
                force: true,
            },
            5,
        )
    });
    // Guard only: the Gram route's blocks are under every split here.
    let (x, y) = two_block(60, 3000, 3, 37);
    c.check("pls3 split_exact 60x3000x3", || {
        pls3_test(
            &x,
            &y,
            ConfirmatoryArgs::SplitExact {
                n_perm: 20,
                n_splits: 4,
            },
            20,
        )
    });
    c.finish();
}
