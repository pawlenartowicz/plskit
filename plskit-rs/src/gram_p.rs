//! p-space Gram backend for the fixed-X PLS1 resampling loops at `n ≫ p`
//! (Improved Kernel PLS, Algorithm 2 of Dayal and MacGregor 1997).
//!
//! `pls1_perm_null`, the `raw_perm` folds and the `split_exact` refit route
//! refit on one fixed block `Xs` (`n × p`: the standardized training rows,
//! `√w`-scaled when weighted) while only the outcome changes. This module
//! builds `C = Xs'Xs` (`p × p`) once per block and runs the shared PLS1
//! component loop (`fit::pls1_component_loop`) on it, so a replicate costs
//! one GEMV `s₁ = Xs'ys` (formed by the caller, [`first_product`]) plus
//! `O(k·p²)`, against the X backend's `(2k + 1)·n·p`.
//!
//! The route selectors weigh this route's work ([`gram_p_work`]) against
//! the X backend's ([`gram_p_primal_work`]) only on the shapes
//! [`gram_p_allowed`] admits, where the loop does enough work to matter;
//! everything else never builds `C`. The route is one arm (`GramP`) of the replicate drivers in
//! `signal_test.rs` and `perm_null.rs`; this module holds no loop.
//!
//! **Every replicate is decided from `C`.** The loop truncates as the X
//! backend does: it stops at the first component whose `‖s_a‖` is below
//! `max(NIPALS_ABS_FLOOR, w_rel_floor(n, p, ‖Xs‖_F, ‖ys‖))` or whose
//! `t't = r'Cr` is below `NIPALS_ABS_FLOOR`, and keeps the components
//! before it (none is the zero model). No replicate is recomputed on X.
//! Near those floors (an outcome orthogonal to `Xs`, the components past
//! its rank) `C`'s rounding and X's can decide differently, so the two
//! backends can keep different numbers of components there; [`K_GRAM_MAX`]
//! records the agreement the calibration sweep (`tests_sweep`) measures
//! elsewhere.
//!
//! **Precision.** `Ĉᵢⱼ` carries rounding of about `n·u·‖xᵢ‖·‖xⱼ‖`
//! (`u = 2⁻⁵³`), relative to the column norms, and `C` squares the
//! condition number of `Xs`. A fit whose components reach a direction in
//! which columns nearly coincide loses accuracy against the X backend:
//! with one such pair at `k = p`, for any `p`, and earlier when several
//! pairs nearly coincide (two pairs `1e-4` apart at `p = 8`: `|Δr|` about
//! `1e-9` from `k = 7`). [`K_GRAM_MAX`] does not bound that loss. A row far
//! from the rest dominates every column norm and buries the other rows'
//! part of `C`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::fit::{
    ComponentBackend, Gate, LoopOutcome, ParChoice, Scored, pls1_coef_at_k, pls1_component_loop,
    resolve_par, w_rel_floor,
};

/// Largest component count the route accepts: the largest `k ≤ 10` up to
/// which the calibration sweep
/// (`gram_p::tests_sweep::gram_p_matches_x_backend_up_to_k_gram_max`, 35
/// designs from `200 × 10` to `100 000 × 8`, dense, sparse and weighted)
/// meets `|Δcoef| ≤ 1e-10·max(1, ‖coef‖_∞)` on all rows and `|Δr| ≤ 1e-12`
/// on a split against the X backend, on every fit where both keep the same
/// number of components. Worst errors per `k`:
///
/// | k | worst `|Δcoef|/max(1, ‖coef‖_∞)` | worst `|Δr|` |
/// |---|---|---|
/// | 1 | 1.2e-14 (`n_much_larger_than_p`) | 8.9e-16 (`n_much_larger_than_p`) |
/// | 2 | 1.2e-14 (`n_much_larger_than_p`) | 1.0e-15 (`n_much_larger_than_p`) |
/// | 3 | 1.2e-14 (`n_much_larger_than_p`) | 6.7e-16 (`n_much_larger_than_p`) |
/// | 4 | 1.2e-14 (`n_much_larger_than_p`) | 5.6e-16 (`n_much_larger_than_p`) |
/// | 5 | 1.2e-14 (`n_much_larger_than_p`) | 4.4e-16 (`ordinary`) |
/// | 6 | 1.5e-14 (`collinear_1e6`) | 5.6e-16 (`collinear_1e6`) |
/// | 7 | 5.4e-13 (`collinear_pair_1e-2`) | 6.7e-16 (`collinear_pair_1e-2`) |
/// | 8 | 2.7e-12 (`collinear_pair_1e-3`) | 1.6e-15 (`collinear_pair_1e-2`) |
/// | 9 | 9.3e-10 (`collinear_pair_1e-4`) | 1.4e-13 (`collinear_pair_1e-3`) |
/// | 10 | 7.7e-8 (`collinear_pair_1e-4`) | 6.6e-12 (`collinear_pair_1e-4`) |
///
/// The two backends kept the same component count on every fit of the
/// sweep. Past `k = 8` the `collinear_pair` designs (two of 10 columns
/// `1e-4` apart) reach the direction of their difference. The cap is
/// therefore the sweep's, set by that family's `p = 10`, and is no accuracy
/// bound for X with nearly coinciding columns: such a fit loses accuracy at
/// `k = p` for any `p`, and earlier with several such pairs (see the module
/// doc).
pub(crate) const K_GRAM_MAX: usize = 8;

/// Least X-backend work per block, `B·(2k + 1)·n_tr·p` multiply-adds, for
/// which the route is taken. Below it the whole loop costs milliseconds on
/// the X backend, and without it this route would become the default for
/// ordinary small data (every existing corpus fixture clears the cost rule
/// alone). Set from `gram_p_work_floor_benchmark`.
pub(crate) const GRAM_P_MIN_WORK: f64 = 1e8;

/// Cost of building `C = Xs'Xs` per multiply-add, in units of one
/// multiply-add of an X-backend pass (`Xs'v` or `Xs·v`): the build is one
/// BLAS-3 product, which runs at about three times a matrix-vector
/// product's rate. See [`gram_p_work`] for how it enters the rule and
/// for how both coefficients were fitted.
const GRAM_P_BUILD_COST: f64 = 0.33;

/// Cost of one Gram-backend component's `C·r` product per multiply-add, in
/// the same units as [`GRAM_P_BUILD_COST`]: slower per multiply-add than an
/// X pass, since a `p × p` product has less work per element loaded than
/// the X pass's `n_tr × p` one at the shapes the rule decides.
const GRAM_P_PRODUCT_COST: f64 = 1.3;

/// The shapes at which a fixed-X loop may take the p-space Gram route (the
/// route selectors weigh its work, [`gram_p_work`]): `1 ≤ k ≤ min(p,
/// K_GRAM_MAX)` and the X backend's work `B·(2k + 1)·n_tr·p` at least
/// `GRAM_P_MIN_WORK`, with `n_tr` the block's row count (the largest fold
/// or half), `p` the feature count, `n_replicates` the number of outcomes
/// one `C` serves (`n_perm + 1` at `raw_perm` and `split_exact`, `n_perm`
/// at `perm_null`, whose observed fit stays on the Primal route) and `k`
/// the component count. A selector with `keep` also requires
/// `1 ≤ keep ≤ p`: anything else stays on the Primal route, whose per-unit
/// input checks report it, so no argument error ever depends on the route.
pub(crate) fn gram_p_allowed(n_tr: usize, p: usize, n_replicates: usize, k: usize) -> bool {
    let (n_f, p_f, b_f, k_f) = (n_tr as f64, p as f64, n_replicates as f64, k as f64);
    (1..=p.min(K_GRAM_MAX)).contains(&k) && b_f * (2.0 * k_f + 1.0) * n_f * p_f >= GRAM_P_MIN_WORK
}

/// The Gram backend's work for one block, in X-pass multiply-adds:
/// `p²·(GRAM_P_BUILD_COST·n_tr + GRAM_P_PRODUCT_COST·B·k)`.
///
/// # The rule
/// Counting multiply-adds per block with `B = n_replicates`:
///
/// ```text
/// X backend:     B·(n·p + 2·k·n·p)
/// Gram backend:  n·p² + B·(n·p + k·p²)
/// ```
///
/// Weighting the Gram backend's two `p²` terms by their measured cost per
/// multiply-add relative to an X pass, `a = GRAM_P_BUILD_COST` for the
/// `C` build and `b = GRAM_P_PRODUCT_COST` for the per-component `C·r`
/// products, the Gram backend pays when
/// `a·n·p² + b·B·k·p² < 2·B·k·n·p`, that is
/// `p·(a·n_tr + b·B·k) < 2·B·k·n_tr` (this function against
/// [`gram_p_primal_work`]). That implies `p < (2/b)·n_tr` (about
/// `1.54·n_tr`).
///
/// # Fitting `a` and `b`
/// Fitted on an Apple M4 (single thread) from
/// `bench_work_floor::gram_p_route_boundary_benchmark` over 164 shapes
/// around the boundary (`n_tr` from 300 to 4000, `k` 1 to 3, `B` 100 to
/// 1000, `p` from 0.75 to 1.3 times the boundary value, passed through
/// `GRAM_P_BENCH_SHAPES`), taking the faster of two runs per shape: with
/// `c = x_ms / (B·(2k + 1)·n_tr·p)` the X backend's time per multiply-add
/// (median over shapes), a least-squares fit of
/// `gram_ms ≈ c·(a·n_tr·p² + b·B·k·p² + s·B·n_tr·p)` gave `a = 0.33`,
/// `b = 1.30`, `s = 1.05` (the `s = Xs'ys` pass costs one X pass, as the
/// count assumes). Over those shapes the fitted rule costs 85.5 s against
/// 91.3 s for the unweighted count (`a = b = 1`, which kept the X backend on
/// 61 shapes where the Gram backend was faster by more than 2%, up to 2x)
/// and 84.9 s for a per-shape oracle. To re-fit on other hardware, run the
/// benchmark over such a sweep and repeat the fit; a stale fit costs time
/// only.
///
/// Evaluated in f64: at every shape a caller can afford to run, the
/// products stay inside f64's range with room to spare, and a rounding
/// could only pick the slower of two routes.
pub(crate) fn gram_p_work(n_tr: f64, p: f64, n_replicates: f64, k: f64) -> f64 {
    let bk = n_replicates * k;
    p * (p * (GRAM_P_BUILD_COST * n_tr + GRAM_P_PRODUCT_COST * bk))
}

/// The `C` build's part of [`gram_p_work`] over `rows` rows:
/// `p²·GRAM_P_BUILD_COST·rows`.
pub(crate) fn gram_p_build_work(rows: f64, p: f64) -> f64 {
    p * (p * (GRAM_P_BUILD_COST * rows))
}

/// The X backend's work for one block against [`gram_p_work`]:
/// `2·B·k·n_tr·p`, the passes the Gram backend saves (the first product
/// `Xs'ys` and the test-row scoring are on both sides and not counted).
pub(crate) fn gram_p_primal_work(n_tr: f64, p: f64, n_replicates: f64, k: f64) -> f64 {
    let bk = n_replicates * k;
    p * (2.0 * bk * n_tr)
}

/// `ŝ₁ = Xs'ys`, the first product of the X backend's fit of `ys` on the
/// same view (`fit::pls1_kernel`'s `s₁` on a column-major `Xs`), under the
/// `Par` that fit resolves from `primal_par` and `k`. The callers pass it to
/// [`GramPBlock::fit`], so the first weight vector is the X backend's bits.
pub(crate) fn first_product(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    primal_par: ParChoice,
    k: usize,
) -> Col<f64> {
    let par = resolve_par(primal_par, xs.nrows(), xs.ncols(), k);
    let mut s = Col::<f64>::zeros(xs.ncols());
    matmul(
        s.as_mut().as_mat_mut(),
        Accum::Replace,
        xs.transpose(),
        ys.as_mat(),
        1.0,
        par,
    );
    s
}

/// One fixed block of a replicate loop, prepared for the p-space Gram
/// backend: `C = Xs'Xs` and the scalars the truncation floor reads. Built
/// once per block by the `GramP` arm of the site's block constructor and
/// shared read-only by every worker; per replicate only `O(k·p)` vectors
/// are allocated.
pub(crate) struct GramPBlock {
    /// `Ĉ = Xs'Xs` as computed.
    c: Mat<f64>,
    /// The row count of `Xs`, the `n` of the relative floor.
    n: usize,
    /// `‖Xs‖_F`: the `norm_l2` call the X backend makes on the same view,
    /// so `w_floor` is bit-identical across backends.
    x_fro: f64,
}

impl GramPBlock {
    /// Build `C` for the block `xs` with `par` (the driver passes
    /// `fit::par_fixed()`).
    /// `tests_block::c_build_is_thread_count_invariant_at_the_route_shapes`
    /// pins that every `par` gives the same bits.
    pub(crate) fn new(xs: MatRef<'_, f64>, par: Par) -> Self {
        let p = xs.ncols();
        let mut c = Mat::<f64>::zeros(p, p);
        matmul(c.as_mut(), Accum::Replace, xs.transpose(), xs, 1.0, par);
        Self {
            c,
            n: xs.nrows(),
            x_fro: xs.norm_l2(),
        }
    }

    /// The block of a training half whose `C = Xs'Xs` was formed elsewhere
    /// (from the cells of the split design, `cell_sums::cell_grams`), with
    /// `n` rows and `x_fro = ‖Xs‖_F` as the caller formed it.
    pub(crate) fn from_gram(c: Mat<f64>, n: usize, x_fro: f64) -> Self {
        Self { c, n, x_fro }
    }

    /// The PLS1 fit of one replicate from `C`: `s1` is `Xs'ys`
    /// ([`first_product`] on the block's `Xs`, or any other evaluation of
    /// the same product) and `ys_norm` is `‖ys‖`, `ys` the replicate's
    /// outcome as the X backend would receive it (standardized,
    /// `√w`-scaled exactly as the block). The component loop and its
    /// truncation rule are the X backend's (see the module doc); only
    /// `t't = r'Cr`, `p = Cr/t't` and `q = s'w/t't` are formed from `C`.
    /// The caller decides non-finite outcomes before calling: a NaN in
    /// `s1` passes every floor test, as in the X backend, and gives a NaN
    /// `coef`.
    pub(crate) fn fit(
        &self,
        s1: ColRef<'_, f64>,
        ys_norm: f64,
        k: usize,
        keep: Option<usize>,
    ) -> GramFit {
        let p = self.c.nrows();
        let w_floor = w_rel_floor(self.n, p, self.x_fro, ys_norm);
        let mut backend = GramBackend {
            c: self.c.as_ref(),
            cr: Col::<f64>::zeros(p),
        };
        match pls1_component_loop(&mut backend, s1.to_owned(), w_floor, k, keep) {
            LoopOutcome::Done {
                p: p_mat,
                w: w_mat,
                q,
                ..
            } => {
                let k_used = w_mat.ncols();
                GramFit {
                    coef: pls1_coef_at_k(&w_mat, &p_mat, &q, k_used, Par::Seq),
                    k_used,
                    w: w_mat,
                    p: p_mat,
                    q,
                }
            }
            LoopOutcome::Unresolved => {
                unreachable!("the Gram backend never returns Gate::Unresolved")
            }
        }
    }
}

/// One replicate's fit on the Gram backend.
pub(crate) struct GramFit {
    /// `W (P'W)⁻¹ Q` through `fit::pls1_coef_at_k`, as on the X backend; zero
    /// for the zero model.
    pub(crate) coef: Col<f64>,
    /// Components kept, `0..=k`.
    pub(crate) k_used: usize,
    /// Weight vectors `W`, `(p, k_used)`; the first column is the X
    /// backend's bits.
    pub(crate) w: Mat<f64>,
    /// X-loadings `P`, `(p, k_used)`.
    pub(crate) p: Mat<f64>,
    /// y-loadings `Q`, `(k_used,)`.
    pub(crate) q: Col<f64>,
}

/// The Gram backend of [`pls1_component_loop`]: `C` and the current
/// component's `Ĉr`, which `loading` reuses.
struct GramBackend<'b> {
    c: MatRef<'b, f64>,
    cr: Col<f64>,
}

impl ComponentBackend for GramBackend<'_> {
    fn gate_s(&mut self, _a: usize, _s: &Col<f64>, _keep: Option<usize>) -> Gate {
        Gate::Continue
    }

    fn score(&mut self, r: &Col<f64>) -> Scored {
        matmul(
            self.cr.as_mut().as_mat_mut(),
            Accum::Replace,
            self.c,
            r.as_ref().as_mat(),
            1.0,
            Par::Seq,
        );
        let tt: f64 = (0..self.cr.nrows()).map(|j| r[j] * self.cr[j]).sum();
        Scored { tt, t: None }
    }

    fn gate_tt(&mut self, _a: usize, _r: &Col<f64>, _tt: f64) -> Gate {
        Gate::Continue
    }

    fn loading(&mut self, _r: &Col<f64>, _t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        Col::<f64>::from_fn(self.cr.nrows(), |j| self.cr[j] * inv_tt)
    }

    fn q(&mut self, s: &Col<f64>, w: &Col<f64>, _t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        let sw: f64 = (0..s.nrows()).map(|j| s[j] * w[j]).sum();
        sw * inv_tt
    }
}

#[cfg(test)]
mod tests_route_rule {
    use super::*;

    #[test]
    fn never_when_p_is_at_least_twice_n_tr() {
        for n_tr in [2usize, 10, 100, 1000, 1999] {
            for p in [2 * n_tr, 2 * n_tr + 1, 3 * n_tr] {
                for b in [1usize, 100, 10_000, 1_000_000] {
                    for k in [1usize, 2, K_GRAM_MAX] {
                        let (n, p, b, k) = (n_tr as f64, p as f64, b as f64, k as f64);
                        assert!(
                            gram_p_work(n, p, b, k) >= gram_p_primal_work(n, p, b, k),
                            "n_tr={n} p={p} B={b} k={k}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_component_cap_binds() {
        assert!(gram_p_allowed(10_000, 50, 1000, K_GRAM_MAX));
        assert!(!gram_p_allowed(10_000, 50, 1000, K_GRAM_MAX + 1));
        assert!(
            gram_p_allowed(10_000, 1000, 1001, K_GRAM_MAX),
            "wide-ish 10000x1000 at the component cap"
        );
    }

    #[test]
    fn work_floor_refuses_a_shape_the_cost_rule_admits() {
        // perm_null n = 80, d = 6, n_perm = 200, k = 2: the cost rule alone
        // admits it, the work floor does not.
        let (n, p, b, k) = (80usize, 6usize, 200usize, 2usize);
        let (nf, pf, bf, kf) = (n as f64, p as f64, b as f64, k as f64);
        assert!(
            gram_p_work(nf, pf, bf, kf) < gram_p_primal_work(nf, pf, bf, kf),
            "premise: the cost rule admits the corpus shape"
        );
        assert!(!gram_p_allowed(n, p, b, k));
    }

    #[test]
    fn degenerate_inputs_stay_on_the_x_backend() {
        // The shape the rows below put a zero into is allowed.
        assert!(gram_p_allowed(10_000, 10, 1000, 2));
        assert!(!gram_p_allowed(0, 0, 0, 0));
        assert!(!gram_p_allowed(0, 10, 1000, 2));
        assert!(!gram_p_allowed(10_000, 0, 1000, 2));
        assert!(!gram_p_allowed(10_000, 10, 0, 2));
        assert!(!gram_p_allowed(10_000, 10, 1000, 0));
        // More components than features.
        assert!(!gram_p_allowed(10_000, 1, 1000, 2));
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests_block {
    use super::*;
    use crate::test_support::{linear_y, prepared, unif};

    #[test]
    fn c_build_is_thread_count_invariant_at_the_route_shapes() {
        // The block built with `Par::Seq` is compared with the block built
        // with `fit::par_fixed()`, the fixed-degree Rayon split, so the block
        // must come out with the same bits serial and in pools of two and
        // seven threads. Shapes: the fixture blocks (perm_null, raw_perm fold,
        // split_exact half), the memprobe shape, a block large enough for
        // faer's parallel GEMM, and d = 1, once short and once long enough
        // (n ≥ 256² entries) for faer to split the reduction dimension.
        for (n, p) in [
            (2000usize, 50usize),
            (1600, 50),
            (1000, 200),
            (10_000, 500),
            (4000, 256),
            (5000, 1),
            (70_000, 1),
        ] {
            let x = unif(n, p, 31);
            let (xs, _) = prepared(&x, &linear_y(&x, 1.0, 32), None);
            let seq = GramPBlock::new(xs.as_ref(), Par::Seq);
            for threads in [2usize, 7] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool");
                let par = pool.install(|| GramPBlock::new(xs.as_ref(), crate::fit::par_fixed()));
                for j in 0..p {
                    for i in 0..p {
                        assert_eq!(
                            par.c[(i, j)].to_bits(),
                            seq.c[(i, j)].to_bits(),
                            "{n}×{p}, {threads} threads: C[{i}, {j}]"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_kernel {
    use super::*;
    use crate::fit::{ParChoice, pls1_fit_prepared_fro};
    use crate::test_support::{linear_y, prepared, unif};
    use faer::Col;

    /// The Gram fit of `ys` on the block of `xs`, from the X backend's `s₁`.
    pub(super) fn gram_fit(
        block: &GramPBlock,
        xs: MatRef<'_, f64>,
        ys: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
    ) -> GramFit {
        let s1 = first_product(xs, ys, ParChoice::Seq, k);
        block.fit(s1.as_ref(), ys.norm_l2(), k, keep)
    }

    /// `max_j |a_j − b_j| ≤ 1e-10·max(1, ‖b‖∞)`, `b` the X backend's.
    fn assert_coef_close(a: &Col<f64>, b: &Col<f64>, what: &str) {
        let scale = (0..b.nrows()).map(|j| b[j].abs()).fold(1.0_f64, f64::max);
        let err = (0..a.nrows())
            .map(|j| (a[j] - b[j]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err <= 1e-10 * scale,
            "{what}: max |Δcoef| {err:e} above 1e-10·{scale:e}"
        );
    }

    #[test]
    fn first_component_is_the_x_backends_bits() {
        // s₁, w_floor and the selection at a = 1 are the X backend's own
        // bits, so the first weight vector is too (dense, sparse, weighted).
        let x = unif(400, 25, 3);
        let y = linear_y(&x, 1.0, 4);
        let w = Col::<f64>::from_fn(400, |i| 0.5 + (i % 3) as f64 * 0.5);
        for (label, wopt, keep) in [
            ("dense", None, None),
            ("sparse", None, Some(6)),
            ("weighted", Some(&w), None),
        ] {
            let (xs, ys) = prepared(&x, &y, wopt);
            let block = GramPBlock::new(xs.as_ref(), Par::Seq);
            let g = gram_fit(&block, xs.as_ref(), ys.as_ref(), 2, keep);
            let xf = pls1_fit_prepared_fro(
                xs.as_ref(),
                ys.as_ref(),
                2,
                keep,
                ParChoice::Seq,
                xs.norm_l2(),
            );
            for j in 0..25 {
                assert_eq!(
                    g.w[(j, 0)].to_bits(),
                    xf.w_star[(j, 0)].to_bits(),
                    "{label}: w[{j}, 0]"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact zeros of a constant column, by construction
    fn single_feature_and_constant_column_match_the_x_backend() {
        // p = 1.
        let x = unif(3000, 1, 9);
        let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 10), None);
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        let xf = pls1_fit_prepared_fro(
            xs.as_ref(),
            ys.as_ref(),
            1,
            None,
            ParChoice::Seq,
            xs.norm_l2(),
        );
        let g = gram_fit(&block, xs.as_ref(), ys.as_ref(), 1, None);
        assert_eq!(g.k_used, xf.k_used);
        assert_coef_close(&g.coef, &xf.coef, "p = 1");
        // A constant column standardizes to an exact zero column, so C has
        // a zero row and column.
        let mut x = unif(400, 10, 11);
        for i in 0..400 {
            x[(i, 3)] = 2.5;
        }
        let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 12), None);
        assert!((0..400).all(|i| xs[(i, 3)] == 0.0), "premise");
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        for k in 1..=K_GRAM_MAX.min(9) {
            let xf = pls1_fit_prepared_fro(
                xs.as_ref(),
                ys.as_ref(),
                k,
                None,
                ParChoice::Seq,
                xs.norm_l2(),
            );
            let g = gram_fit(&block, xs.as_ref(), ys.as_ref(), k, None);
            assert_eq!(g.k_used, xf.k_used, "k={k}");
            assert_coef_close(&g.coef, &xf.coef, &format!("k={k}"));
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // the zero model's coefficients are exact zeros
    fn a_zero_outcome_and_an_orthogonal_outcome_give_the_zero_model() {
        // `ys = 0` (a constant outcome after standardization) and an outcome
        // orthogonal to every column of `Xs` up to rounding: `s₁` is the X
        // backend's, so both backends stop at a = 1.
        let x = unif(500, 12, 13);
        let (xs, _) = prepared(&x, &linear_y(&x, 1.0, 14), None);
        let orth = crate::test_support::orthogonal_y(&xs, 15);
        let (ys_orth, _, _) = crate::linalg::standardize1(orth.as_ref());
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        for (label, ys) in [("zero", Col::<f64>::zeros(500)), ("orthogonal", ys_orth)] {
            for k in [1usize, 2, 5] {
                let g = gram_fit(&block, xs.as_ref(), ys.as_ref(), k, None);
                let xf = pls1_fit_prepared_fro(
                    xs.as_ref(),
                    ys.as_ref(),
                    k,
                    None,
                    ParChoice::Seq,
                    xs.norm_l2(),
                );
                assert_eq!((g.k_used, xf.k_used), (0, 0), "{label}, k = {k}");
                assert!((0..12).all(|j| g.coef[j] == 0.0), "{label}, k = {k}");
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]
mod tests_sweep {
    use super::*;
    use crate::fit::{ParChoice, pls1_fit_prepared_fro};
    use crate::signal_test::{
        SplitIdx, prepare_split, split_column_r, split_column_r_gram_p, split_train_target,
    };
    use crate::test_support::{conditioned, linear_y, orthogonal_y, prepared, unif, unif_col};
    use std::collections::BTreeMap;

    /// The component counts the sweep measures.
    const K_SWEEP_MAX: usize = 10;

    /// The agreement tolerances: `|Δcoef| ≤ COEF_TOL·max(1, ‖coef‖_∞)` and
    /// `|Δr| ≤ R_TOL`.
    const COEF_TOL: f64 = 1e-10;
    const R_TOL: f64 = 1e-12;

    /// One design: raw `x` and `y` (the coefficients are compared on
    /// `prepared(x, y, w)`, the split on `prepare_split` of `x` with the
    /// all-row scale), weights and `keep`.
    struct Design {
        family: &'static str,
        x: Mat<f64>,
        y: Col<f64>,
        w: Option<Col<f64>>,
        keep: Option<usize>,
    }

    fn design(family: &'static str, x: Mat<f64>, y: Col<f64>) -> Design {
        Design {
            family,
            x,
            y,
            w: None,
            keep: None,
        }
    }

    /// `y = Σ_{l<m} (l + 1)·u_l` over the leading left singular vectors of
    /// `xs`: exactly `m` components are real.
    #[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
    fn span_y(xs: &Mat<f64>, m: usize) -> Col<f64> {
        let svd = xs.thin_svd().expect("svd");
        let u = svd.U();
        Col::<f64>::from_fn(xs.nrows(), |i| {
            (0..m).map(|l| (1.0 + l as f64) * u[(i, l)]).sum::<f64>()
        })
    }

    /// Rank-`r` `n × p` matrix `A·B` with uniform factors.
    fn low_rank(n: usize, p: usize, r: usize, seed: u64) -> Mat<f64> {
        let a = unif(n, r, seed);
        let b = unif(r, p, seed + 1);
        crate::linalg::mat_mul(a.as_ref(), b.as_ref(), Par::Seq)
    }

    #[allow(clippy::too_many_lines)]
    fn designs() -> Vec<Design> {
        let mut out = Vec::new();
        // Ordinary: signal and null outcomes.
        for seed in 0..3u64 {
            let x = unif(400, 30, 100 + seed);
            out.push(design("ordinary", x.clone(), linear_y(&x, 1.0, 200 + seed)));
            out.push(design("ordinary", x, unif_col(400, 300 + seed)));
        }
        // n ≫ p, up to the release-gate shape.
        for seed in 0..2u64 {
            let x = unif(5000, 8, 110 + seed);
            let y = linear_y(&x, 1.0, 210 + seed);
            out.push(design("n_much_larger_than_p", x, y));
        }
        {
            let x = unif(100_000, 8, 112);
            let y = linear_y(&x, 1.0, 212);
            out.push(design("n_much_larger_than_p", x, y));
        }
        // Rank-deficient: columns 20..30 are sums of earlier ones (rank 20).
        {
            let mut x = unif(300, 30, 120);
            for j in 20..30 {
                for i in 0..300 {
                    x[(i, j)] = x[(i, j - 20)] + x[(i, j - 19)];
                }
            }
            let y = linear_y(&x, 1.0, 220);
            out.push(design("rank_deficient", x, y));
        }
        // y exhausted: X of rank 4, y independent of X, so the components
        // past the fourth sit at the floor.
        {
            let x = low_rank(400, 20, 4, 121);
            out.push(design("y_exhausted", x, unif_col(400, 221)));
        }
        // Weak signal: 1e-3 of the signal on top of noise.
        for seed in 0..2u64 {
            let x = unif(400, 30, 130 + seed);
            let e = unif_col(400, 230 + seed);
            let s = linear_y(&x, 0.0, 0);
            let y = Col::<f64>::from_fn(400, |i| e[i] + 1e-3 * s[i]);
            out.push(design("weak_signal", x, y));
        }
        // Collinear columns, κ(X) = 1e2, 1e4, 1e6.
        for (tag, kappa) in [
            ("collinear_1e2", 1e2),
            ("collinear_1e4", 1e4),
            ("collinear_1e6", 1e6),
        ] {
            let x = conditioned(400, 20, kappa, 140);
            let y = linear_y(&x, 0.1, 240);
            out.push(design(tag, x, y));
        }
        // A nearly collinear pair inside an ordinary block, 1e-2 to 1e-4
        // apart.
        for (tag, eps) in [
            ("collinear_pair_1e-2", 1e-2),
            ("collinear_pair_1e-3", 1e-3),
            ("collinear_pair_1e-4", 1e-4),
        ] {
            let mut x = unif(1000, 10, 145);
            let e = unif_col(1000, 146);
            for i in 0..1000 {
                x[(i, 1)] = x[(i, 0)] + eps * e[i];
            }
            let y = linear_y(&x, 1.0, 245);
            out.push(design(tag, x, y));
        }
        // Duplicated rows.
        {
            let x0 = unif(150, 20, 150);
            let x = Mat::<f64>::from_fn(300, 20, |i, j| x0[(i % 150, j)]);
            let y0 = linear_y(&x0, 1.0, 250);
            let y = Col::<f64>::from_fn(300, |i| y0[i % 150]);
            out.push(design("duplicated_rows", x, y));
        }
        // y orthogonal to every column of X (and to 1) up to rounding.
        for seed in [41u64, 43] {
            let x = unif(200, 10, seed);
            let (xs0, _, _) = crate::linalg::standardize(x.as_ref());
            let y = orthogonal_y(&xs0, seed + 1);
            out.push(design("y_orthogonal", x, y));
        }
        // y in the span of the first m = 2 directions of the standardized X:
        // the components past the second sit at the floor.
        {
            let x = unif(500, 30, 21);
            let (xs, _, _) = crate::linalg::standardize(x.as_ref());
            let y = span_y(&xs, 2);
            out.push(design("y_in_span_m2", x, y));
        }
        // Sparse keep, signal and null.
        for keep in [1usize, 5, 15, 29] {
            let x = unif(400, 30, 160);
            for y in [linear_y(&x, 1.0, 260), unif_col(400, 261)] {
                out.push(Design {
                    keep: Some(keep),
                    ..design("sparse_keep", x.clone(), y)
                });
            }
        }
        // Duplicated columns under keep: exact ties at the boundary.
        {
            let x0 = unif(400, 15, 170);
            let x = Mat::<f64>::from_fn(400, 30, |i, j| x0[(i, j % 15)]);
            let y = linear_y(&x0, 1.0, 270);
            for keep in [3usize, 10] {
                out.push(Design {
                    keep: Some(keep),
                    ..design("duplicated_columns_keep", x.clone(), y.clone())
                });
            }
        }
        // Weighted, every 7th weight zero; dense and sparse.
        {
            let x = unif(400, 30, 180);
            let y = linear_y(&x, 1.0, 280);
            let w = Col::<f64>::from_fn(400, |i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    0.5 + (i % 5) as f64 * 0.3
                }
            });
            for keep in [None, Some(10)] {
                out.push(Design {
                    w: Some(w.clone()),
                    keep,
                    ..design("weighted_zero_weights", x.clone(), y.clone())
                });
            }
        }
        out
    }

    /// Worst errors of one (k, family) cell on the fits where both backends
    /// keep the same component count.
    #[derive(Default)]
    struct Cell {
        coef: f64,
        r: f64,
    }

    /// One fit where the two backends keep different component counts.
    struct Flip {
        family: &'static str,
        design: usize,
        k: usize,
        /// `"coef"` (all rows) or `"r"` (the split's training half).
        on: &'static str,
        gram: usize,
        x: usize,
        /// `|Δcoef| / max(1, ‖coef‖_∞)` or `|Δr|`.
        err: f64,
    }

    /// The fits where `C`'s rounding and X's may decide a component
    /// differently: the components past the rank of X (rank 4 in
    /// `y_exhausted`) or past the span of `y` (2 in `y_in_span_m2`), where
    /// `‖s_a‖` is rounding, and an outcome orthogonal to X.
    fn flip_expected(family: &str, k: usize) -> bool {
        family == "y_orthogonal"
            || (family == "y_exhausted" && k > 4)
            || (family == "y_in_span_m2" && k > 2)
    }

    /// `|Δcoef| / max(1, ‖b‖_∞)`.
    fn coef_err(a: &Col<f64>, b: &Col<f64>) -> f64 {
        let scale = (0..b.nrows()).map(|j| b[j].abs()).fold(1.0_f64, f64::max);
        (0..a.nrows())
            .map(|j| (a[j] - b[j]).abs())
            .fold(0.0_f64, f64::max)
            / scale
    }

    /// Every design, `k = 1..=min(10, p)`: the Gram backend against the X
    /// backend on the same block. Coefficients on all rows (`prepared`);
    /// `r` on the split of the first half against the second,
    /// `split_column_r_gram_p` against `split_column_r` on `prepare_split`'s
    /// halves. On the fits where both keep the same component count, every
    /// family meets `|Δcoef| ≤ 1e-10·max(1, ‖coef‖_∞)` and `|Δr| ≤ 1e-12` at
    /// every `k ≤ K_GRAM_MAX`; a fit where the counts differ is listed, and
    /// is allowed only where [`flip_expected`] says. `cargo test -p plskit
    /// --release gram_p_matches_x_backend -- --nocapture` prints the table.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn gram_p_matches_x_backend_up_to_k_gram_max() {
        let mut cells: BTreeMap<(usize, &'static str), Cell> = BTreeMap::new();
        let mut flips: Vec<Flip> = Vec::new();
        let mut seen: BTreeMap<&'static str, usize> = BTreeMap::new();
        let primal = |xs: MatRef<'_, f64>, ys: ColRef<'_, f64>, k: usize, keep| {
            pls1_fit_prepared_fro(xs, ys, k, keep, ParChoice::Seq, xs.norm_l2())
        };
        let gram = |block: &GramPBlock, xs: MatRef<'_, f64>, ys: ColRef<'_, f64>, k, keep| {
            let s1 = first_product(xs, ys, ParChoice::Seq, k);
            block.fit(s1.as_ref(), ys.norm_l2(), k, keep)
        };
        for d in designs() {
            let design = {
                let c = seen.entry(d.family).or_default();
                *c += 1;
                *c - 1
            };
            let (n, p) = (d.x.nrows(), d.x.ncols());
            let (xs, ys) = prepared(&d.x, &d.y, d.w.as_ref());
            let block = GramPBlock::new(xs.as_ref(), Par::Seq);
            let wn =
                d.w.as_ref()
                    .map(|w| crate::linalg::normalize_weights(w.as_ref()).expect("weights"));
            let scale =
                crate::linalg::held_out_scale(d.x.as_ref(), wn.as_ref().map(Col::as_ref), false);
            let n_tr = n - n / 2;
            let half = SplitIdx {
                tr: (0..n_tr).collect(),
                te: (n_tr..n).collect(),
            };
            let prep = prepare_split(
                d.x.as_ref(),
                &half,
                wn.as_ref().map(Col::as_ref),
                scale.as_ref(),
            );
            let block_h = GramPBlock::new(prep.xs_tr.as_ref(), Par::Seq);
            let y_of = |i: usize| d.y[i];
            let ys_h = split_train_target(prep.w_tr.as_ref(), &half, &y_of);
            for k in 1..=K_SWEEP_MAX.min(p) {
                let cell = cells.entry((k, d.family)).or_default();
                let xf = primal(xs.as_ref(), ys.as_ref(), k, d.keep);
                let gf = gram(&block, xs.as_ref(), ys.as_ref(), k, d.keep);
                let err = coef_err(&gf.coef, &xf.coef);
                if gf.k_used == xf.k_used {
                    cell.coef = cell.coef.max(err);
                } else {
                    flips.push(Flip {
                        family: d.family,
                        design,
                        k,
                        on: "coef",
                        gram: gf.k_used,
                        x: xf.k_used,
                        err,
                    });
                }
                let r_x = split_column_r(&prep, &half, &y_of, &[k], d.keep)[0];
                let r_g = split_column_r_gram_p(&block_h, &prep, &half, &y_of, &[k], d.keep)[0];
                let ku_x = primal(prep.xs_tr.as_ref(), ys_h.as_ref(), k, d.keep).k_used;
                let ku_g = gram(&block_h, prep.xs_tr.as_ref(), ys_h.as_ref(), k, d.keep).k_used;
                if ku_g == ku_x {
                    cell.r = cell.r.max((r_g - r_x).abs());
                } else {
                    flips.push(Flip {
                        family: d.family,
                        design,
                        k,
                        on: "r",
                        gram: ku_g,
                        x: ku_x,
                        err: (r_g - r_x).abs(),
                    });
                }
            }
        }
        let meets = |k: usize| {
            cells
                .range((k, "")..(k + 1, ""))
                .all(|(_, c)| c.coef <= COEF_TOL && c.r <= R_TOL)
        };
        let k_max = (1..=K_SWEEP_MAX)
            .take_while(|&k| meets(k))
            .last()
            .unwrap_or(0);
        eprintln!("k  family                    |dcoef|/max(1,|coef|)  |dr|   (equal counts)");
        for ((k, family), c) in &cells {
            eprintln!("{k:>2} {family:<25} {:>21.1e}  {:.1e}", c.coef, c.r);
        }
        eprintln!("/// | k | worst |Δcoef|/max(1, ‖coef‖_∞) | worst |Δr| |");
        for k in 1..=K_SWEEP_MAX {
            let row = cells.range((k, "")..(k + 1, ""));
            let worst = |f: &dyn Fn(&Cell) -> f64| {
                row.clone().fold(
                    ("", 0.0_f64),
                    |a, ((_, fam), c)| {
                        if f(c) > a.1 { (*fam, f(c)) } else { a }
                    },
                )
            };
            let (fc, wc) = worst(&|c| c.coef);
            let (fr, wr) = worst(&|c| c.r);
            eprintln!("/// | {k} | {wc:.1e} (`{fc}`) | {wr:.1e} (`{fr}`) |");
        }
        eprintln!("/// Flips (family, design, k, on, kept Gram / X, error):");
        for f in &flips {
            eprintln!(
                "/// - {} {}, k = {}, {}: {} / {}, {:.1e}",
                f.family, f.design, f.k, f.on, f.gram, f.x, f.err
            );
        }
        eprintln!("largest k meeting the tolerances at every k up to it: {k_max}");
        let unexpected: Vec<String> = flips
            .iter()
            .filter(|f| !flip_expected(f.family, f.k))
            .map(|f| format!("{} {} k={} {}", f.family, f.design, f.k, f.on))
            .collect();
        assert!(
            unexpected.is_empty(),
            "component counts differ outside the expected families: {unexpected:?}"
        );
        assert_eq!(
            k_max, K_GRAM_MAX,
            "K_GRAM_MAX must be the largest k up to which the sweep meets the tolerances"
        );
    }
}

#[cfg(test)]
mod tests_site_route {
    use super::*;
    use crate::dual_route::{
        nspace_allowed_perm_null, nspace_allowed_raw_perm, nspace_allowed_split,
    };
    use crate::perm_null::perm_null_route;
    use crate::signal_test::{
        PrimalSource, ReplicateRoute, raw_perm_route, split_exact_refit_route,
        with_gram_routes_disabled,
    };

    #[test]
    fn k3_at_high_tall_ratio_takes_gram_p() {
        // 100 000 × 8 at k = 3, n_tr/p from 6250 to 12 500: every site's
        // selector takes the Gram-p route.
        let (n, p, n_perm, k) = (100_000usize, 8usize, 300usize, 3usize);
        assert!(gram_p_allowed(n, p, n_perm, k));
        assert_eq!(
            perm_null_route(n, p, n_perm, k, false, false)
                .unwrap()
                .route,
            ReplicateRoute::GramP
        );
        assert_eq!(
            raw_perm_route(n, 5, p, n_perm, &[k], None, false)
                .unwrap()
                .route,
            ReplicateRoute::GramP
        );
        assert_eq!(
            split_exact_refit_route(n, p, n_perm, 32, &[k], None, false, false)
                .unwrap()
                .route
                .route,
            ReplicateRoute::GramP
        );
    }

    #[test]
    fn weighted_replicates_prefer_gram_p_in_the_overlap_band() {
        // perm_null: n = 1000 < p = 1200 < 1.5n, k = 2, n_perm = 5000. Both
        // routes are eligible (premises asserted: the n-space edge is
        // p ~ 1111, the Gram-p edge p ~ 1500); the n-space route is cheaper
        // there.
        assert!(nspace_allowed_perm_null(1000, 2, true), "premise");
        assert!(gram_p_allowed(1000, 1200, 5000, 2), "premise");
        assert_eq!(
            perm_null_route(1000, 1200, 5000, 2, false, false)
                .unwrap()
                .route,
            ReplicateRoute::Nspace
        );
        // Weighted: the n-space route refuses, Gram-p takes it.
        assert_eq!(
            perm_null_route(1000, 1200, 5000, 2, true, false)
                .unwrap()
                .route,
            ReplicateRoute::GramP
        );
        // raw_perm: n = 1250, n_folds = 5, so n_tr = 1000; p = 1400 clears
        // the n-space rule's `n_tr < 0.75·p`.
        assert!(nspace_allowed_raw_perm(1250, 5, 1400, 2, true), "premise");
        assert!(gram_p_allowed(1000, 1400, 5001, 2), "premise");
        assert_eq!(
            raw_perm_route(1250, 5, 1400, 5000, &[2], None, false)
                .unwrap()
                .route,
            ReplicateRoute::Nspace
        );
        // keep: the n-space route refuses, Gram-p takes it.
        assert_eq!(
            raw_perm_route(1250, 5, 1400, 5000, &[2], Some(100), false)
                .unwrap()
                .route,
            ReplicateRoute::GramP
        );
        // split_exact: n = 2000, so n_train = 1000.
        assert!(nspace_allowed_split(1000, 2, true), "premise");
        assert!(gram_p_allowed(1000, 1200, 5001, 2), "premise");
        assert_eq!(
            split_exact_refit_route(2000, 1200, 5000, 32, &[2], None, false, false)
                .unwrap()
                .route
                .route,
            ReplicateRoute::Nspace
        );
    }

    #[test]
    fn special_routes_keep_their_inputs() {
        // split_exact k = 1 without keep: the no-refit route, even on a
        // Gram-eligible tall shape.
        assert!(gram_p_allowed(1000, 50, 1001, 1), "premise");
        assert_eq!(
            split_exact_refit_route(2000, 50, 1000, 32, &[1], None, false, false)
                .unwrap()
                .route
                .route,
            ReplicateRoute::Special
        );
    }

    #[test]
    fn k1_on_tall_data_takes_gram_p_where_no_special_route_claims_it() {
        assert_eq!(
            raw_perm_route(2000, 5, 50, 1000, &[1], None, false)
                .unwrap()
                .route,
            ReplicateRoute::GramP
        );
    }

    #[test]
    fn invalid_arguments_never_take_the_gram_p_route() {
        // k = 0, k > p, keep = 0 and keep > p reach the Primal units, whose
        // input checks report them, independent of the route.
        assert_eq!(
            raw_perm_route(2000, 5, 50, 1000, &[0], None, false)
                .unwrap()
                .route,
            ReplicateRoute::Primal(PrimalSource::Copy)
        );
        assert_eq!(
            perm_null_route(2000, 3, 1000, 4, false, false)
                .unwrap()
                .route,
            ReplicateRoute::Primal(PrimalSource::Copy)
        );
        for keep in [Some(0usize), Some(51)] {
            assert_eq!(
                raw_perm_route(2000, 5, 50, 1000, &[2], keep, false)
                    .unwrap()
                    .route,
                ReplicateRoute::Primal(PrimalSource::Copy),
                "keep {keep:?}"
            );
            assert_eq!(
                split_exact_refit_route(2000, 50, 1000, 32, &[2], keep, false, false)
                    .unwrap()
                    .route
                    .route,
                ReplicateRoute::Primal(PrimalSource::Copy),
                "keep {keep:?}"
            );
        }
        assert!(!gram_p_allowed(2000, 50, 1000, 51));
        assert!(
            raw_perm_route(2000, 5, 50, 1000, &[2], Some(50), false)
                .unwrap()
                .route
                == ReplicateRoute::GramP,
            "keep = p is valid"
        );
    }

    #[test]
    fn gram_routes_disabled_restricts_to_the_special_and_primal_routes() {
        let tall = || {
            perm_null_route(2000, 50, 1000, 3, false, false)
                .unwrap()
                .route
        };
        assert_eq!(tall(), ReplicateRoute::GramP);
        assert_eq!(
            with_gram_routes_disabled(tall),
            ReplicateRoute::Primal(PrimalSource::Copy)
        );
        assert_eq!(
            with_gram_routes_disabled(|| raw_perm_route(2000, 5, 50, 1000, &[2], Some(10), false)
                .unwrap()
                .route),
            ReplicateRoute::Primal(PrimalSource::Copy)
        );
        // k = 1 leaves the Gram routes like every other k.
        assert_eq!(
            with_gram_routes_disabled(|| raw_perm_route(60, 5, 3000, 200, &[1], None, false)
                .unwrap()
                .route),
            ReplicateRoute::Primal(PrimalSource::Copy)
        );
        // The guard restores the previous state on exit.
        assert_eq!(tall(), ReplicateRoute::GramP);
    }
}

#[cfg(test)]
mod bench_work_floor {
    use super::*;
    use crate::fit::{ParChoice, pls1_fit_prepared_fro};
    use crate::test_support::{linear_y, prepared, unif};
    use std::hint::black_box;
    use std::time::Instant;

    // One fixed block per shape, single thread: B X-backend fits (with
    // `‖Xs‖_F` taken once for the block) against one C build plus B Gram fits. Run with
    // `cargo test -p plskit --release gram_p_work_floor_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: sets GRAM_P_MIN_WORK"]
    fn gram_p_work_floor_benchmark() {
        // (n_tr, p, k, B); every k is at most K_GRAM_MAX.
        let shapes = [
            (500usize, 20usize, 2usize, 200usize),
            (1000, 20, 2, 300),
            (1000, 20, 2, 1000),
            (2000, 30, 2, 1000),
            (1600, 50, 2, 1001),
            (1000, 200, 1, 1001),
            (2000, 50, 3, 1000),
            (5000, 100, 3, 1000),
            (10_000, 500, 3, 100),
        ];
        eprintln!("n_tr\tp\tk\tB\twork\tcost_rule\tx_ms\tgram_ms\tspeedup\tsaved_ms");
        for (n, p, k, b) in shapes {
            let x = unif(n, p, 91);
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 92), None);
            let perms: Vec<Col<f64>> = (0..b)
                .map(|i| {
                    let perm = crate::rng::null_permutation(n, i + 1, 0);
                    Col::<f64>::from_fn(n, |r| ys[perm[r] as usize])
                })
                .collect();
            // `‖Xs‖_F` taken once, as `perm_null`'s Primal arm takes it.
            let x_fro = xs.norm_l2();
            let t0 = Instant::now();
            for yb in &perms {
                black_box(pls1_fit_prepared_fro(
                    xs.as_ref(),
                    yb.as_ref(),
                    k,
                    None,
                    ParChoice::Seq,
                    x_fro,
                ));
            }
            let x_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let block = GramPBlock::new(xs.as_ref(), Par::Seq);
            for yb in &perms {
                let s1 = first_product(xs.as_ref(), yb.as_ref(), ParChoice::Seq, k);
                black_box(block.fit(s1.as_ref(), yb.norm_l2(), k, None));
            }
            let g_ms = t1.elapsed().as_secs_f64() * 1e3;
            let (nf, pf, bf, kf) = (n as f64, p as f64, b as f64, k as f64);
            let work = bf * (2.0 * kf + 1.0) * nf * pf;
            let rule = gram_p_work(nf, pf, bf, kf) < gram_p_primal_work(nf, pf, bf, kf);
            eprintln!(
                "{n}\t{p}\t{k}\t{b}\t{work:.1e}\t{rule}\t{x_ms:.1}\t{g_ms:.1}\t{:.2}\t{:.1}",
                x_ms / g_ms,
                x_ms - g_ms
            );
        }
    }

    /// Median of `reps` timings of `f`, in ms.
    fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
        let mut v: Vec<f64> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v[reps / 2]
    }

    // Shapes around the unweighted flop count's boundary
    // `p·(n_tr + B·k) = 2·B·k·n_tr` (`p` at 0.8x to 1.5x of that value),
    // single thread: B X-backend fits with `‖Xs‖_F` taken once, as
    // `perm_null`'s Primal arm runs them, against one C build plus B Gram
    // fits, medians of 3.
    // `GRAM_P_BENCH_SHAPES="n,p,k,B;..."` replaces the default shapes; the
    // sweep that fitted `gram_p_work`'s cost coefficients passes its
    // own (see "Fitting `a` and `b`" there). Run
    // with `cargo test -p plskit --release gram_p_route_boundary_benchmark --
    // --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: checks gram_p_work near its boundary"]
    fn gram_p_route_boundary_benchmark() {
        let default_shapes = || {
            vec![
                (300, 369, 1, 1000),
                (300, 531, 1, 1000),
                (1000, 800, 1, 1000),
                (1000, 1000, 1, 1000),
                (1000, 1300, 1, 1000),
                (1000, 333, 1, 200),
                (1000, 500, 1, 200),
                (600, 738, 2, 1000),
                (600, 1062, 2, 1000),
                (1600, 640, 2, 200),
                (300, 545, 3, 1000),
                (1000, 862, 3, 200),
            ]
        };
        let parse = |s: String| -> Vec<(usize, usize, usize, usize)> {
            s.split(';')
                .map(|c| {
                    let v: Vec<usize> = c
                        .split(',')
                        .map(|t| t.trim().parse().expect("usize"))
                        .collect();
                    (v[0], v[1], v[2], v[3])
                })
                .collect()
        };
        let shapes = std::env::var("GRAM_P_BENCH_SHAPES").map_or_else(|_| default_shapes(), parse);
        eprintln!("n_tr\tp\tk\tB\trule\tx_ms\tgram_ms\tspeedup");
        for (n, p, k, b) in shapes {
            let x = unif(n, p, 91);
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 92), None);
            let perms: Vec<Col<f64>> = (0..b)
                .map(|i| {
                    let perm = crate::rng::null_permutation(n, i + 1, 0);
                    Col::<f64>::from_fn(n, |r| ys[perm[r] as usize])
                })
                .collect();
            // The X backend as `perm_null` runs it: `‖Xs‖_F` taken once.
            let x_fro = xs.norm_l2();
            let x_ms = median_ms(3, || {
                for yb in &perms {
                    black_box(crate::fit::pls1_fit_prepared_fro(
                        xs.as_ref(),
                        yb.as_ref(),
                        k,
                        None,
                        ParChoice::Seq,
                        x_fro,
                    ));
                }
            });
            let g_ms = median_ms(3, || {
                let block = GramPBlock::new(xs.as_ref(), Par::Seq);
                for yb in &perms {
                    let s1 = first_product(xs.as_ref(), yb.as_ref(), ParChoice::Seq, k);
                    black_box(block.fit(s1.as_ref(), yb.norm_l2(), k, None));
                }
            });
            let (nf, pf, bf, kf) = (n as f64, p as f64, b as f64, k as f64);
            let rule = gram_p_allowed(n, p, b, k)
                && gram_p_work(nf, pf, bf, kf) < gram_p_primal_work(nf, pf, bf, kf);
            eprintln!(
                "{n}\t{p}\t{k}\t{b}\t{rule}\t{x_ms:.1}\t{g_ms:.1}\t{:.2}",
                x_ms / g_ms
            );
        }
    }
}
