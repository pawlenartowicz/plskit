//! Byte-parity across Rayon pool sizes.
//!
//! `byte_parity.rs` compares serial (`disable_parallelism`) with parallel
//! execution at whatever pool size the process has, so it cannot see a
//! result whose bits depend on *how many* threads the pool has: one process
//! has one `RAYON_NUM_THREADS`. This file runs the same public call inside
//! Rayon pools of 1, 3 and 5 threads (`ThreadPoolBuilder::install`) in one
//! process and requires the outputs to be bitwise identical.
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
//! threshold of its EVD / SVD. Every case below marked "pool" was checked
//! to differ across pool sizes before the fix (it fails without it). Cases
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

use std::fmt::Debug;

use faer::{Col, Mat, MatRef};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, pls1_fit, pls1_perm_null,
    pls1_rotation_stability, pls3_confirmatory_test, pls3_fit, split_nb_gate, spls1_fit, spls3_fit,
    CIOpts, ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts,
    FindKOptimalOpts, FindKSequenceOpts, FitOpts, KSpec, PermNullOpts, Pls1Model,
    Pls3ConfirmatoryTestOpts, Pls3FitOpts, RotationStabilityMethod, RotationStabilityOpts,
    Selector, VarimaxArgs,
};

const POOL_SIZES: [usize; 3] = [1, 3, 5];

/// `a` and `b` agree entry by entry to `tol` relative to `1 + |b|`, with
/// the same `k_used` and flags: the check for two fits whose only
/// difference is the memory layout of `X`.
fn assert_fit_close(a: &Pls1Model, b: &Pls1Model, tol: f64, what: &str) {
    fn mat(m: &Mat<f64>) -> Vec<f64> {
        (0..m.ncols())
            .flat_map(|j| (0..m.nrows()).map(move |i| m[(i, j)]))
            .collect()
    }
    fn col(c: &Col<f64>) -> Vec<f64> {
        (0..c.nrows()).map(|i| c[i]).collect()
    }
    assert_eq!(a.k_used, b.k_used, "{what}: k_used");
    assert_eq!(
        (a.pre_standardized, a.keep),
        (b.pre_standardized, b.keep),
        "{what}: flags"
    );
    let pairs = [
        ("t_scores", mat(&a.t_scores), mat(&b.t_scores)),
        ("p_loadings", mat(&a.p_loadings), mat(&b.p_loadings)),
        ("w_star", mat(&a.w_star), mat(&b.w_star)),
        ("q", col(&a.q_loadings), col(&b.q_loadings)),
        ("coef", col(&a.coef), col(&b.coef)),
        ("beta", col(&a.beta), col(&b.beta)),
        (
            "scalars",
            vec![a.intercept, a.n_eff],
            vec![b.intercept, b.n_eff],
        ),
    ];
    for (name, x, y) in pairs {
        assert_eq!(x.len(), y.len(), "{what}: {name} length");
        for (i, (u, v)) in x.iter().zip(&y).enumerate() {
            assert!(
                (u - v).abs() <= tol * (1.0 + v.abs()),
                "{what}: {name}[{i}] {u} vs {v}"
            );
        }
    }
}

fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| (x[(i, 0)] + 0.5 * x[(i, 1)]) * snr + noise[i]);
    (x, y)
}

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
    // M5: the Python seam hands the kernel a row-major *view* of `X`
    // (`np_mat_view`, `MatRef::from_row_major_slice`) instead of a
    // column-major copy. The standardized X keeps a row-major input's
    // layout, so that view agrees with the column-major case above to
    // 1e-10 (not bit for bit), and must be pool-invariant on its own: a
    // future change could make the row-major path pool-sensitive while
    // each half-test still passes in isolation.
    let (x_col_major, y5) = synth(2000, 600, 1.0, 3);
    let mut x_row_major: Vec<f64> = Vec::with_capacity(2000 * 600);
    for i in 0..2000 {
        for j in 0..600 {
            x_row_major.push(x_col_major[(i, j)]);
        }
    }
    let x_row_view = MatRef::from_row_major_slice(&x_row_major, 2000, 600);
    let col_major_out = pool(1).install(|| {
        pls1_fit(
            x_col_major.as_ref(),
            y5.as_ref(),
            KSpec::Fixed(5),
            None,
            FitOpts::default(),
        )
        .unwrap()
    });
    let row_major_out = pool(1).install(|| {
        pls1_fit(
            x_row_view,
            y5.as_ref(),
            KSpec::Fixed(5),
            None,
            FitOpts::default(),
        )
        .unwrap()
    });
    assert_fit_close(
        &row_major_out,
        &col_major_out,
        1e-10,
        "pls1_fit 2000x600 k=5: row-major view vs column-major copy",
    );
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
    // (20000·4 here); at 200x2000x6 `spls3_fit` is pool-invariant even
    // before the fix.
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
    // Pool. The reproducer: 40x4000 dense, k = 3 (past the n-space route's
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

fn perm_opts() -> PermNullOpts {
    PermNullOpts {
        n_perm: 100,
        return_perm_matrix: true,
        pre_standardized: false,
        disable_parallelism: false,
        verbose: false,
    }
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
