//! p-space Gram backend for the fixed-X PLS1 resampling loops at `n ≫ p`
//! (Improved Kernel PLS, Algorithm 2 of Dayal and MacGregor 1997).
//!
//! `pls1_perm_null`, the `raw_perm` folds and the `split_exact` refit route
//! refit on one fixed block `Xs` (`n × p`: the standardized training rows,
//! `√w`-scaled when weighted) while only the outcome changes. This module
//! builds `C = Xs'Xs` (`p × p`) once per block and runs the shared PLS1
//! component loop (`fit::pls1_component_loop`) on it, so a replicate costs
//! one sequential GEMV `s = Xs'ys` plus `O(k·p²)`, against the X backend's
//! `(2k + 1)·n·p`.
//!
//! **The X backend stays primary.**
//! [`use_gram_p_route`](crate::gram_p::use_gram_p_route) admits an input
//! only when its cost rule favours this route and the loop does enough
//! work to matter; everything else never builds `C`. The route is one arm
//! (`GramP`) of the replicate drivers in `signal_test.rs` and
//! `perm_null.rs`; this module holds no loop.
//!
//! **The route changes no decision.** The gates rest on first-order running
//! bounds and a conditioning ratio gate. A replicate whose truncation
//! decision or conditioning this backend cannot certify is recomputed by
//! the Primal arm of its site, so its output is the Primal route's to the
//! bit. The accuracy of resolved replicates is validated by the calibration
//! sweep (`tests_sweep`), not gated per replicate.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::dual_route::{ABS_BAND, RESOLVE_BAND};
use crate::fit::{
    pls1_coef_at_k, pls1_component_loop, w_rel_floor, ComponentBackend, Gate, LoopOutcome, Scored,
    NIPALS_ABS_FLOOR,
};

/// Unit roundoff of f64, `2⁻⁵³`: the `u` of the rounding bounds in this
/// module.
const U: f64 = f64::EPSILON * 0.5;

/// Largest feature count the route accepts. `C` is `p²` f64s, 128 MB at
/// the cap, and one block is alive at a time.
pub(crate) const GRAM_P_MAX: usize = 4000;

/// Largest component count the route accepts. At `k = 4` ordinary families
/// fall back above 1% (`n_much_larger_than_p`, `5000 × 8`: both replicates
/// fall back at component 4), so the cap is 3.
/// `python3 scripts/gate_feasibility.py gram_p` covers that `n ≫ p` shape
/// alongside the tall corpus fixture blocks, the memprobe shape
/// `10 000 × 500` and an ordinary `500 × 20` block, and prints the same
/// value: the largest `k ≤ 20` whose fallback rate on ordinary data is at
/// most 1% at every one of those shapes, for every `k` up to it. The
/// calibration sweep
/// (`gram_p::tests_sweep::gram_backend_matches_x_backend_across_design_families`)
/// may only lower it further.
/// Calibration range: at `k = 3` the decision gates fall back on most
/// replicates once `n/p` is above about 2000 (measured with the recursion
/// in `scripts/gate_feasibility.py`: `20 000 × 8` at 7%, `100 000 × 8` at 100%), which costs a
/// wasted attempt before the Primal arm but never a wrong answer; `k = 2`
/// never fell back on any shape tried.
pub(crate) const K_GRAM_MAX: usize = 3;

/// Least X-backend work per block, `B·(2k + 1)·n_tr·p` multiply-adds, for
/// which the route is taken. Below it the whole loop costs milliseconds on
/// the X backend, and without it the gated path would become the default
/// for ordinary small data (every existing corpus fixture clears the cost
/// rule alone). Set from `gram_p_work_floor_benchmark`.
pub(crate) const GRAM_P_MIN_WORK: f64 = 1e8;

/// Largest `n_tr / p` a `k = 3` attempt is offered at. Below `K_GRAM_MAX`
/// the loop's decision gates (`gate_s`'s `SNorm` and `gate_tt`'s `Tt`, see
/// [`GramPBlock::fit_replicate_diag`]) fall back on a rounding-driven share
/// of replicates that grows with `n_tr / p`: the per-component bounds carry
/// a `c2_norm·δr` term where `c2_norm` tracks `‖Xs‖_F² ≈ n_tr·p`, so a
/// larger training block leaves less of `RESOLVE_BAND`'s margin for the
/// third component to clear. The wasted attempt (the `s1 = Xs'ys` GEMV,
/// about `1/(2k + 1)` of a primal replicate) is cheap next to a primal
/// replicate, so the route still wins up to a high fallback rate: with a
/// fallback share `f` of `B` replicates, the Gram route's total work is
/// about `B·f·(1 + 1/(2k + 1))` primal-replicate units against the
/// primal-only `B·1`, so it pays while `f < (2k + 1)/(2k + 2)` (`7/8` at
/// `k = 3`).
///
/// Measured with `tests_sweep::tall_shapes_n_over_p_sweep` (300 permuted
/// replicates per cell, `p ∈ {4, 8, 16}`, `n_tr/p` from 1000 to 20000): the
/// fallback rate crosses `7/8` at `n_tr/p` of about 7300 (`p = 4`), 5300-6800
/// (`p = 8`) and 3900 (`p = 16`), so the break-even ratio falls as `p`
/// grows and no single-`p` measurement covers the family. `3000` sits under
/// every measured break-even with a comfortable margin (fallback at most
/// 20% at `n_tr/p = 3000` across the three `p`, an over 4x net win even in
/// the worst cell) while still admitting `n_tr/p = 2500`
/// (`20000×8` at `k = 3`: 7.9x faster with a 1.5% fallback rate); the calibration sweep may only lower it further.
pub(crate) const GRAM_P_TALL_RATIO_MAX_K3: f64 = 3000.0;

/// Cost of building `C = Xs'Xs` per multiply-add, in units of one
/// multiply-add of an X-backend pass (`Xs'v` or `Xs·v`): the build is one
/// BLAS-3 product, which runs at about three times a matrix-vector
/// product's rate. See [`use_gram_p_route`] for how it enters the rule and
/// for how both coefficients were fitted.
const GRAM_P_BUILD_COST: f64 = 0.33;

/// Cost of one Gram-backend component's `C·r` product per multiply-add, in
/// the same units as [`GRAM_P_BUILD_COST`]: slower per multiply-add than an
/// X pass, since a `p × p` product has less work per element loaded than
/// the X pass's `n_tr × p` one at the shapes the rule decides.
const GRAM_P_PRODUCT_COST: f64 = 1.3;

/// Should this fixed-X loop take the p-space Gram route?
///
/// `n_tr` is the block's row count (the largest fold or half), `p` the
/// feature count, `n_replicates` the number of outcomes one `C` serves
/// (`n_perm + 1` at `raw_perm` and `split_exact`, `n_perm` at `perm_null`,
/// whose observed fit stays on the Primal route), and `k` the component
/// count.
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
/// `p·(a·n_tr + b·B·k) < 2·B·k·n_tr`. That implies `p < (2/b)·n_tr`
/// (about `1.54·n_tr`), while `dual_route::use_dual_route` implies
/// `n_tr < p`; in the band between, the n-space route is cheaper on both
/// counts and the route selectors try it first. On top of the cost rule:
/// `1 ≤ k ≤ min(p, K_GRAM_MAX)`, `p ≤ GRAM_P_MAX`, and the X backend's work
/// `B·(2k + 1)·n_tr·p` at least `GRAM_P_MIN_WORK`.
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
/// benchmark over such a sweep and repeat the fit; the rule only picks the
/// faster of two routes that agree, so a stale fit costs time, never
/// accuracy.
///
/// At `k ≥ 3` the cost rule alone is not enough: on a very tall design
/// (`n_tr / p` past [`GRAM_P_TALL_RATIO_MAX_K3`]) the third component's
/// decision gates fall back on so many replicates that the wasted attempts
/// outweigh the rule's projected saving (see
/// `GRAM_P_TALL_RATIO_MAX_K3`'s doc comment for the calibration). That check
/// costs nothing at `k ≤ 2`, where the gates never fell back at any
/// measured shape.
///
/// Evaluated in f64 like `use_dual_route`: at every shape a caller can
/// afford to run, the products stay inside f64's range with room to spare,
/// and a rounding could only pick the slower of two routes that agree.
#[allow(clippy::many_single_char_names)]
pub(crate) fn use_gram_p_route(n_tr: usize, p: usize, n_replicates: usize, k: usize) -> bool {
    if n_tr == 0 || p == 0 || n_replicates == 0 || k == 0 {
        return false;
    }
    if p > GRAM_P_MAX || k > K_GRAM_MAX || k > p {
        return false;
    }
    let (n_f, p_f, b_f, k_f) = (n_tr as f64, p as f64, n_replicates as f64, k as f64);
    if k >= 3 && n_f > GRAM_P_TALL_RATIO_MAX_K3 * p_f {
        return false;
    }
    let pays = gram_p_cost_rule(n_f, p_f, b_f, k_f);
    let work = b_f * (2.0 * k_f + 1.0) * n_f * p_f;
    pays && work >= GRAM_P_MIN_WORK
}

/// The cost rule of [`use_gram_p_route`] alone:
/// `p·(GRAM_P_BUILD_COST·n_tr + GRAM_P_PRODUCT_COST·B·k) < 2·B·k·n_tr`.
pub(crate) fn gram_p_cost_rule(n_tr: f64, p: f64, n_replicates: f64, k: f64) -> bool {
    let bk = n_replicates * k;
    p * (GRAM_P_BUILD_COST * n_tr + GRAM_P_PRODUCT_COST * bk) < 2.0 * bk * n_tr
}

/// The selectors' `GramP` test: [`use_gram_p_route`] on the block shape,
/// and arguments the Gram-p arms accept (`1 ≤ k ≤ p`; with `keep`,
/// `1 ≤ keep ≤ p`). Anything else stays on the Primal route, whose per-unit
/// input checks report it, so no argument error ever depends on
/// the route.
pub(crate) fn gram_p_eligible(
    n_tr: usize,
    p: usize,
    n_replicates: usize,
    k: usize,
    keep: Option<usize>,
) -> bool {
    (1..=p).contains(&k)
        && keep.is_none_or(|kp| (1..=p).contains(&kp))
        && use_gram_p_route(n_tr, p, n_replicates, k)
}

/// Product cap of the power iteration in [`c2_norm_estimate`].
const POWER_MAX_ITERS: usize = 50;

/// Stop rule of the power iteration in [`c2_norm_estimate`]: `λ̂` moved by
/// at most this fraction of itself.
const POWER_REL_TOL: f64 = 1e-3;

/// The `‖C‖₂` the first-order bounds use: `min(2·λ̂, ‖Xs‖_F²)`.
///
/// `λ̂` is the Rayleigh quotient `v'Cv` of a power iteration on `Ĉ`
/// (`Par::Seq`, start `1/√p` in every entry, at most `POWER_MAX_ITERS`
/// products, stopping once `λ̂` moves by at most `POWER_REL_TOL` of itself).
/// Power iteration estimates `λ_max` from below, hence the factor 2;
/// `‖Xs‖_F² = tr C ≥ ‖C‖₂` caps it, and is the value whenever `λ̂` is not
/// positive (a zero `C`, or a start with no component along any nonzero
/// direction of `C`). A first-order estimate, not a guaranteed bound.
/// Transcribed in `scripts/gate_feasibility.py` (`c2_norm_estimate`):
/// change both together.
fn c2_norm_estimate(c: MatRef<'_, f64>, x_fro: f64) -> f64 {
    let p = c.nrows();
    let fro2 = x_fro * x_fro;
    let start = 1.0 / (p as f64).sqrt();
    let mut v = Col::<f64>::from_fn(p, |_| start);
    let mut w = Col::<f64>::zeros(p);
    let mut lambda = 0.0_f64;
    for _ in 0..POWER_MAX_ITERS {
        matmul(
            w.as_mut().as_mat_mut(),
            Accum::Replace,
            c,
            v.as_ref().as_mat(),
            1.0,
            Par::Seq,
        );
        let lambda_new: f64 = (0..p).map(|j| v[j] * w[j]).sum();
        let w_norm = w.norm_l2();
        if w_norm.is_nan() || w_norm <= 0.0 {
            lambda = lambda_new;
            break;
        }
        let inv = 1.0 / w_norm;
        for j in 0..p {
            v[j] = w[j] * inv;
        }
        let done = (lambda_new - lambda).abs() <= POWER_REL_TOL * lambda_new;
        lambda = lambda_new;
        if done {
            break;
        }
    }
    if lambda.is_nan() || lambda <= 0.0 {
        return fro2;
    }
    (2.0 * lambda).min(fro2)
}

/// One fixed block of a replicate loop, prepared for the p-space Gram
/// backend: `C = Xs'Xs` and the scalars the first-order bounds need. Built
/// once per block by the `GramP` arm of the site's block constructor and
/// shared read-only by every worker; per replicate only `O(k·p)` vectors
/// are allocated.
pub(crate) struct GramPBlock<'a> {
    /// The block the X backend runs on (`n × p`); each replicate reads it
    /// once for `s = Xs'ys`.
    xs: MatRef<'a, f64>,
    /// `Ĉ = Xs'Xs` as computed.
    c: Mat<f64>,
    /// `‖Xs‖_F`: the `norm_l2` call the X backend makes on the same view,
    /// so `w_floor` is bit-identical across backends.
    x_fro: f64,
    /// [`c2_norm_estimate`] of `Ĉ`: the `‖C‖₂` of the first-order bounds.
    c2_norm: f64,
}

impl<'a> GramPBlock<'a> {
    /// Build `C` for the block `xs` with `par` (the driver passes
    /// `resample::block_par(disable_parallelism)`), and its `‖C‖₂`
    /// estimate (sequential). `tests_block::c_build_is_thread_count_invariant_at_the_route_shapes`
    /// pins that every `par` gives the same bits.
    pub(crate) fn new(xs: MatRef<'a, f64>, par: Par) -> Self {
        let p = xs.ncols();
        let mut c = Mat::<f64>::zeros(p, p);
        matmul(c.as_mut(), Accum::Replace, xs.transpose(), xs, 1.0, par);
        let x_fro = xs.norm_l2();
        let c2_norm = c2_norm_estimate(c.as_ref(), x_fro);
        Self {
            xs,
            c,
            x_fro,
            c2_norm,
        }
    }

    /// `‖Xs‖_F` (`xs.norm_l2()`), for a site's Primal fallback, which would
    /// otherwise take the same norm of the same view again.
    pub(crate) fn x_fro(&self) -> f64 {
        self.x_fro
    }
}

/// Conditioning floor of the ratio gate: component `a` stays on this
/// backend only when `tt_a ≥ RATIO_MIN·‖Xs‖_F²·‖r_a‖²`, that is, when the
/// Rayleigh quotient of `C` along `r_a` is at least `RATIO_MIN` of
/// `tr C = ‖Xs‖_F²`.
///
/// `C` squares the condition number of `Xs`, and the first-order bounds do
/// not model the accuracy lost along a nearly singular direction of `C`:
/// the relative error of `tt_a` and `p_a` grows like `u·tr C` over that
/// Rayleigh quotient. The gate sends such replicates to the Primal arm
/// (never to a truncation), which also covers rank-deficient `C` without an
/// eigenvalue test. It is a guard, not a bound: the calibration sweep
/// (`tests_sweep`) calibrates it, raising it only, so that every design
/// family whose replicates resolve meets the tolerance with a 10× margin.
/// Shared with `scripts/gate_feasibility.py`
/// (`GRAM_P_RATIO_MIN`): change both together.
pub(crate) const RATIO_MIN: f64 = 1e-6;

/// First-order bound on the difference between the Gram route's and the
/// Primal route's centered test-half scores `X̃_te·coef` at `split_exact`:
///
/// ```text
/// D = ‖X_te‖_F·(2·e_coef + 2·(p + 2)·u·‖coef‖) + 2·(n_te + 2)·u·‖s_te‖
/// ```
///
/// `e_coef` is the first-order estimate of each route's `coef` error
/// against the exact reference (`GramFit::coef_err`), so their `coef`s
/// differ by about twice it; each route's GEMV `X_te·coef` adds at most
/// `p·u·‖X_te‖_F·‖coef‖`, and each centering `(n_te + 2)·u·‖s_te‖`.
/// `t_norm` is `‖s_te‖`, the norm of the uncentered test-half scores. The
/// score gate uses it to protect the score-degeneracy decision only.
pub(crate) fn score_discrepancy_bound(
    xte_fro: f64,
    coef_err: f64,
    coef_norm: f64,
    p: usize,
    n_te: usize,
    t_norm: f64,
) -> f64 {
    xte_fro * (2.0 * coef_err + 2.0 * (p as f64 + 2.0) * U * coef_norm)
        + 2.0 * (n_te as f64 + 2.0) * U * t_norm
}

/// Why a replicate was handed to the Primal arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateFail {
    /// `‖select(s_a)‖` (`a ≥ 2`) within `RESOLVE_BAND` of twice its bound,
    /// or of the floors.
    SNorm,
    /// The keep-boundary gap (`a ≥ 2`) within `RESOLVE_BAND` of twice the
    /// per-entry bound of `s_a`.
    KeepGap,
    /// `tt` within `RESOLVE_BAND` of twice its bound, or of the absolute
    /// floor.
    Tt,
    /// `tt_a` below `RATIO_MIN·‖Xs‖_F²·‖r_a‖²`: `r_a` points into a nearly
    /// singular direction of `C`.
    Ratio,
    /// The loop stopped at some `a ≥ 2` with no gate refusing first. The
    /// gates make it unreachable; kept so such a stop is never `Resolved`.
    LateStop,
}

/// A replicate the Gram backend resolved.
pub(crate) struct GramFit {
    /// `W (P'W)⁻¹ Q` through `fit::pls1_coef_at_k`, as on the X backend.
    pub(crate) coef: Col<f64>,
    /// Components kept; equal to the X backend's.
    pub(crate) k_used: usize,
    /// Weight vectors `W`; the tests read the first column, which is the X
    /// backend's bits. Test-only, like the other diagnostics below.
    #[cfg(test)]
    pub(crate) w: Mat<f64>,
    /// First-order estimate of `‖coef − coef*‖₂`, either backend against
    /// the exact reference (see `fit_replicate_diag`); read by the
    /// `split_exact` score gate through [`score_discrepancy_bound`].
    pub(crate) coef_err: f64,
    /// Largest per-component `max(δtt/tt, δp/‖p‖)`: a diagnostic for the
    /// sweep report, never a gate. Test-only.
    #[cfg(test)]
    pub(crate) max_rel: f64,
    /// `(tt_a, δtt_a, δp_a)` per component: a diagnostic, reproduced by
    /// `scripts/gate_feasibility.py` (`tests_kernel::bounds_match_gate_feasibility_script`).
    /// Test-only: production replicates neither fill nor allocate it.
    #[cfg(test)]
    pub(crate) bounds: Vec<(f64, f64, f64)>,
}

impl GramPBlock<'_> {
    /// One replicate on the Gram backend, with the reason when it is not
    /// resolved. `ys` is the replicate's outcome as the X backend would
    /// receive it (standardized, `√w`-scaled exactly as the block).
    ///
    /// # Reference and notation
    /// `u = 2⁻⁵³`, `F = ‖Xs‖_F`, `Xs` is `n × p`. Both backends start from
    /// the same bits: `ŝ₁ = Xs'ys` is the same sequential GEMV on the same
    /// view, `w_floor` the same function of the same `‖Xs‖_F` and `‖ys‖`,
    /// and `ŵ₁ = r̂₁` the same `select_and_normalize` of `ŝ₁`. The reference
    /// is exact arithmetic continued from those shared bits. Every bound
    /// below is a first-order estimate of one backend's distance from that
    /// reference for either backend (where their local rounding differs,
    /// the sum is taken), so the two backends differ by about twice it. It
    /// is not a worst-case bound: second-order terms are dropped and
    /// `‖C‖₂` is estimated. The reference obeys these identities:
    /// `s_{a+1} = s_a − (s_a'w_a)·p_a`, `r_a = w_a − R_{<a}(P_{<a}'w_a)`,
    /// `tt_a = r_a'C r_a = ‖Xs r_a‖²`, `p_a = C r_a / tt_a`, and
    /// `q_a·tt_a = s_a'w_a = ŝ₁'r_a = ys'Xs r_a` up to `ŝ₁ ≠ Xs'ys`.
    ///
    /// # Forming `C`
    /// `Ĉᵢⱼ` is a length-`n` dot product in some order, so
    /// `|ΔCᵢⱼ| ≤ n·u·‖xᵢ‖·‖xⱼ‖`; for any `r`, `|r'ΔC r| ≤ n·u·F²·‖r‖²` and
    /// `‖ΔC‖_F ≤ n·u·F²`. `c₂` is the block's `c2_norm`,
    /// `min(2·λ̂, F²)` from a power iteration (`c2_norm_estimate`).
    ///
    /// # Local rounding at component `a`
    /// Gram: `c = Ĉr` (`p·u·F²·‖r‖` plus `ΔC·r`), `tt = r'c`, `p = c·(1/tt)`.
    /// X: `t = Xs r` (`‖Δt‖ ≤ p·u·F·‖r‖`), `tt = t't`, `p = Xs't·(1/tt)`.
    /// With `‖t‖ ≤ F·‖r‖` and `‖c‖ ≤ F²·‖r‖`, both obey
    ///
    /// ```text
    /// L_tt = (n + 2p + 4)·u·F²·‖r‖²
    /// L_p  = (n + p + 2)·u·F²·‖r‖/tt + 2·u·‖p‖
    /// L_qt = (2n + p + 3)·u·‖ys‖·F·‖r‖ + (p + 3)·u·‖ŝ‖
    /// ```
    ///
    /// `L_qt` covers the factor `q·tt` of the `s` update: the Gram backend's
    /// `ŝ'ŵ` (`p·u·‖ŝ‖`) plus the X backend's `ys't` (the dot, the
    /// rounding of `t`, and `Xs'ys` versus `ŝ₁`).
    ///
    /// # Propagation (first order), `ρ = ‖R_{<a}‖_F·‖P_{<a}‖_F`
    ///
    /// ```text
    /// δs₁ = δw₁ = δr₁ = 0
    /// δw_a  = 2·δs_a/σ_a + (p + 2)·u                σ_a = ‖select(ŝ_a)‖
    /// δr_a  = δw_a·(1 + ρ) + Σ_{j<a} (δr_j·‖p_j‖ + ‖r_j‖·δp_j) + (p + a + 2)·u·(1 + ρ)
    /// δtt_a = L_tt + c₂·δr_a·(2‖r_a‖ + δr_a)
    /// δp_a  = L_p + c₂·δr_a/tt_a + ‖p_a‖·δtt_a/tt_a
    /// δqt_a = δs_a + ‖ŝ_a‖·δw_a + ‖ŝ₁‖·δr_a + L_qt
    /// δq_a  = δqt_a/tt_a + |q_a|·δtt_a/tt_a
    /// δs_{a+1} = δs_a + |q_a·tt_a|·δp_a + ‖p_a‖·δqt_a + 2·u·(‖ŝ_a‖ + |q_a·tt_a|·‖p_a‖)
    /// ```
    ///
    /// Each line differences one reference identity, keeps first-order
    /// terms, and adds the local rounding of the operation that produced
    /// the quantity. The running sums carry every earlier component's error
    /// into later ones (the history), so no separate history term is added.
    /// `δs_a` bounds every entry of `ŝ_a` as well (`‖·‖∞ ≤ ‖·‖₂`): it is the
    /// per-entry bound the keep gap uses. With a shared support (the gap
    /// gate), selection is a coordinate projection and does not enlarge
    /// `δs_a`. `scripts/gate_feasibility.py` (`gram_p_gates`) transcribes
    /// this recursion; `tests_kernel::bounds_match_gate_feasibility_script`
    /// holds the two together.
    ///
    /// # Gates (decisions only)
    /// Each is an `a >= b` conjunction, so a NaN fails it.
    /// - `a ≥ 2`, before selection: `σ_a ≥ RESOLVE_BAND·2δs_a`,
    ///   `σ_a ≥ RESOLVE_BAND·w_floor`, `σ_a ≥ ABS_BAND·NIPALS_ABS_FLOOR`,
    ///   and with `keep`, the gap between the `keep`-th and `(keep+1)`-th
    ///   largest `|ŝ_a|` at least `RESOLVE_BAND·2δs_a`. Past them the X
    ///   backend selects the same support and its selected norm is at least
    ///   `(1 − 1/RESOLVE_BAND)·σ_a ≥ 3·w_floor`, so it keeps the component.
    ///   The `w_floor` term also covers the X backend's error as
    ///   `w_rel_floor` measures it (its `first_noise_max` figure), in
    ///   case the first-order model misses a term.
    /// - Every `a`, after `tt`: `tt ≥ RESOLVE_BAND·2δtt` and
    ///   `tt ≥ ABS_BAND·NIPALS_ABS_FLOOR` (the `tt` floor decision), then
    ///   the ratio gate `tt ≥ RATIO_MIN·F²·‖r‖²` (conditioning).
    /// - `a = 1`: the `‖ŝ₁‖` floor test is the X backend's on the same bits,
    ///   so no band; a stop there is `Resolved` with `k_used = 0`.
    ///
    /// This backend never decides a stop at `a ≥ 2`: at a noise component
    /// its `‖ŝ_a‖` lies within its own bound, so any component inside or
    /// below a band sends the replicate to the Primal arm. There is no
    /// per-replicate accuracy gate: `δtt/tt` and `δp/‖p‖` compound
    /// geometrically across components and would refuse ordinary data at
    /// `k ≥ 2`. Accuracy is validated empirically by `tests_sweep` (families
    /// including collinear `κ(Xs)` up to 1e6, rank-deficient, duplicated
    /// rows and columns), as `dual_route.rs` does for its routes.
    ///
    /// # `coef`
    /// In exact arithmetic `W (P'W)⁻¹ = R`, so `coef = R·q` and
    ///
    /// ```text
    /// e_coef = Σ_a (δr_a·|q_a| + ‖r_a‖·δq_a) + (3k + p + 4)·u·√k·‖P‖_F·‖R‖_F·Σ_a ‖r_a‖·|q_a|
    /// ```
    ///
    /// the last term the forward error of `pls1_coef_at_k`'s `k × k` solve,
    /// with `cond(P'W)` bounded by `√k·‖P‖_F·‖R‖_F`. A first-order
    /// diagnostic: only the `split_exact` score gate reads it, through
    /// [`score_discrepancy_bound`].
    ///
    /// # Conditioning
    /// `C` squares the condition number of `Xs`. `L_tt/tt` is about
    /// `(n + 2p)·u·F²·‖r‖²/(r'C r)`, large when `r` points into a nearly
    /// singular direction; the `tt` band refuses a replicate only past `1/8`
    /// relative error, and [`RATIO_MIN`] refuses the directions whose
    /// Rayleigh quotient is a tiny fraction of `tr C`.
    #[allow(clippy::similar_names)]
    pub(crate) fn fit_replicate_diag(
        &self,
        ys: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
    ) -> Result<GramFit, GateFail> {
        let n = self.xs.nrows();
        let p = self.xs.ncols();
        // s₁ = Xs'ys, the call `fit::pls1_kernel` makes for its first `s`
        // under `ParChoice::Seq`, on the same view: the same bits.
        let mut s0 = Col::<f64>::zeros(p);
        matmul(
            s0.as_mut().as_mat_mut(),
            Accum::Replace,
            self.xs.transpose(),
            ys.as_mat(),
            1.0,
            Par::Seq,
        );
        let ys_norm = ys.norm_l2();
        let w_floor = w_rel_floor(n, p, self.x_fro, ys_norm);
        let mut backend = GramBackend::new(self, ys_norm, w_floor);
        match pls1_component_loop(&mut backend, s0, w_floor, k, keep) {
            LoopOutcome::Unresolved => Err(backend.fail.unwrap_or(GateFail::LateStop)),
            LoopOutcome::Done {
                p: p_mat,
                w: w_mat,
                q,
                ..
            } => {
                let k_used = w_mat.ncols();
                if k_used != 0 && k_used != k {
                    return Err(GateFail::LateStop);
                }
                let coef = pls1_coef_at_k(&w_mat, &p_mat, &q, k_used, Par::Seq);
                let coef_err = backend.coef_err(k_used);
                Ok(GramFit {
                    coef,
                    k_used,
                    #[cfg(test)]
                    w: w_mat,
                    coef_err,
                    #[cfg(test)]
                    max_rel: backend.max_rel,
                    #[cfg(test)]
                    bounds: std::mem::take(&mut backend.bounds),
                })
            }
        }
    }

    /// [`Self::fit_replicate_diag`] without the reason.
    pub(crate) fn fit_replicate_full(
        &self,
        ys: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
    ) -> Option<GramFit> {
        self.fit_replicate_diag(ys, k, keep).ok()
    }

    /// `coef` and `k_used` for one replicate, or `None` when unresolved (the
    /// caller then runs the Primal arm's body).
    pub(crate) fn fit_replicate(
        &self,
        ys: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
    ) -> Option<(Col<f64>, usize)> {
        self.fit_replicate_full(ys, k, keep)
            .map(|f| (f.coef, f.k_used))
    }
}

/// `(‖select(s)‖, gap)`: the norm of the `keep` largest-magnitude entries
/// and the gap between the `keep`-th and `(keep+1)`-th largest `|s|`.
/// Dense (`None`, or `keep ≥ p`): the full norm and an infinite gap. Order
/// statistics are exact values, so a partial selection reads the numbers a
/// full sort would.
pub(crate) fn selected_norm_and_gap(s: &Col<f64>, keep: Option<usize>) -> (f64, f64) {
    let p = s.nrows();
    match keep {
        Some(kp) if kp < p => {
            let mut mags: Vec<f64> = (0..p).map(|j| s[j].abs()).collect();
            mags.select_nth_unstable_by(kp, |a, b| b.total_cmp(a));
            let kth_next = mags[kp];
            let kth = mags[..kp].iter().copied().fold(f64::INFINITY, f64::min);
            let sel2: f64 = mags[..kp].iter().map(|m| m * m).sum();
            (sel2.sqrt(), kth - kth_next)
        }
        _ => (s.norm_l2(), f64::INFINITY),
    }
}

/// Per-replicate state of the Gram backend: the block, the replicate's
/// scalars, and the running first-order bounds of
/// [`GramPBlock::fit_replicate_diag`].
#[allow(clippy::struct_field_names)]
struct GramBackend<'b> {
    c: MatRef<'b, f64>,
    n: usize,
    p: usize,
    x_fro: f64,
    /// The block's `‖C‖₂` estimate, `c₂` of the recursion.
    c2_norm: f64,
    ys_norm: f64,
    w_floor: f64,
    /// `Ĉr` of the current component; `loading` reuses it.
    cr: Col<f64>,
    s1_norm: f64,
    s_norm: f64,
    d_s: f64,
    d_w: f64,
    d_r: f64,
    d_tt: f64,
    d_p: f64,
    r_norm: f64,
    tt: f64,
    p_norm: f64,
    /// `‖R_{<a}‖_F²`, `‖P_{<a}‖_F²`.
    r_fro2: f64,
    p_fro2: f64,
    /// `Σ_{j<a} (δr_j·‖p_j‖ + ‖r_j‖·δp_j)`.
    rot_prop: f64,
    /// `Σ_a (δr_a·|q_a| + ‖r_a‖·δq_a)`.
    coef_prop: f64,
    /// `Σ_a ‖r_a‖·|q_a|`.
    coef_scale: f64,
    /// `GramFit::max_rel` (test-only).
    #[cfg(test)]
    max_rel: f64,
    /// `(tt_a, δtt_a, δp_a)` per component that reached `gate_tt`
    /// (test-only, `GramFit::bounds`).
    #[cfg(test)]
    bounds: Vec<(f64, f64, f64)>,
    fail: Option<GateFail>,
}

impl<'b> GramBackend<'b> {
    fn new(block: &'b GramPBlock<'_>, ys_norm: f64, w_floor: f64) -> Self {
        let p = block.c.nrows();
        Self {
            c: block.c.as_ref(),
            n: block.xs.nrows(),
            p,
            x_fro: block.x_fro,
            c2_norm: block.c2_norm,
            ys_norm,
            w_floor,
            cr: Col::<f64>::zeros(p),
            s1_norm: 0.0,
            s_norm: 0.0,
            d_s: 0.0,
            d_w: 0.0,
            d_r: 0.0,
            d_tt: 0.0,
            d_p: 0.0,
            r_norm: 0.0,
            tt: 0.0,
            p_norm: 0.0,
            r_fro2: 0.0,
            p_fro2: 0.0,
            rot_prop: 0.0,
            coef_prop: 0.0,
            coef_scale: 0.0,
            #[cfg(test)]
            max_rel: 0.0,
            #[cfg(test)]
            bounds: Vec::new(),
            fail: None,
        }
    }

    fn coef_err(&self, k_used: usize) -> f64 {
        let k = k_used as f64;
        self.coef_prop
            + (3.0 * k + self.p as f64 + 4.0)
                * U
                * k.sqrt()
                * self.p_fro2.sqrt()
                * self.r_fro2.sqrt()
                * self.coef_scale
    }
}

// The names follow the doc comment's formulas (s, w, q, n, p; d_tt, d_p).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
impl ComponentBackend for GramBackend<'_> {
    fn gate_s(&mut self, a: usize, s: &Col<f64>, keep: Option<usize>) -> Gate {
        let (sel_norm, gap) = selected_norm_and_gap(s, keep);
        self.s_norm = s.norm_l2();
        if a == 1 {
            // ŝ₁, w_floor and the selection are the X backend's own bits.
            self.s1_norm = self.s_norm;
            self.d_w = 0.0;
            return Gate::Continue;
        }
        self.d_w = 2.0 * self.d_s / sel_norm + (self.p as f64 + 2.0) * U;
        let band = RESOLVE_BAND * 2.0 * self.d_s;
        let s_ok = sel_norm >= band
            && sel_norm >= RESOLVE_BAND * self.w_floor
            && sel_norm >= ABS_BAND * NIPALS_ABS_FLOOR;
        let gap_ok = gap >= band;
        if s_ok && gap_ok {
            Gate::Continue
        } else {
            self.fail = Some(if s_ok {
                GateFail::KeepGap
            } else {
                GateFail::SNorm
            });
            Gate::Unresolved
        }
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
        let tt: f64 = (0..self.p).map(|j| r[j] * self.cr[j]).sum();
        Scored { tt, t: None }
    }

    fn gate_tt(&mut self, a: usize, r: &Col<f64>, tt: f64) -> Gate {
        let (n, p) = (self.n as f64, self.p as f64);
        let fro2 = self.x_fro * self.x_fro;
        let r_norm = r.norm_l2();
        let d_r = if a == 1 {
            0.0
        } else {
            let rho = self.r_fro2.sqrt() * self.p_fro2.sqrt();
            self.d_w * (1.0 + rho) + self.rot_prop + (p + a as f64 + 2.0) * U * (1.0 + rho)
        };
        let l_tt = (n + 2.0 * p + 4.0) * U * fro2 * r_norm * r_norm;
        let d_tt = l_tt + self.c2_norm * d_r * (2.0 * r_norm + d_r);
        let p_norm = self.cr.norm_l2() / tt;
        let l_p = (n + p + 2.0) * U * fro2 * r_norm / tt + 2.0 * U * p_norm;
        let d_p = l_p + self.c2_norm * d_r / tt + p_norm * d_tt / tt;
        self.r_norm = r_norm;
        self.tt = tt;
        self.p_norm = p_norm;
        self.d_r = d_r;
        self.d_tt = d_tt;
        self.d_p = d_p;
        #[cfg(test)]
        {
            self.bounds.push((tt, d_tt, d_p));
            let rel = (d_tt / tt).max(d_p / p_norm);
            if rel > self.max_rel {
                self.max_rel = rel;
            }
        }
        let decided = tt >= RESOLVE_BAND * 2.0 * d_tt && tt >= ABS_BAND * NIPALS_ABS_FLOOR;
        let conditioned = tt >= RATIO_MIN * fro2 * r_norm * r_norm;
        if decided && conditioned {
            Gate::Continue
        } else {
            self.fail = Some(if decided {
                GateFail::Ratio
            } else {
                GateFail::Tt
            });
            Gate::Unresolved
        }
    }

    fn loading(&mut self, _r: &Col<f64>, _t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        Col::<f64>::from_fn(self.p, |j| self.cr[j] * inv_tt)
    }

    fn q(&mut self, s: &Col<f64>, w: &Col<f64>, _t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        let sw: f64 = (0..self.p).map(|j| s[j] * w[j]).sum();
        let q = sw * inv_tt;
        let (n, p) = (self.n as f64, self.p as f64);
        let qt = (q * self.tt).abs();
        let l_qt = (2.0 * n + p + 3.0) * U * self.ys_norm * self.x_fro * self.r_norm
            + (p + 3.0) * U * self.s_norm;
        let d_qt = self.d_s + self.s_norm * self.d_w + self.s1_norm * self.d_r + l_qt;
        let d_q = d_qt / self.tt + q.abs() * self.d_tt / self.tt;
        self.d_s += qt * self.d_p + self.p_norm * d_qt + 2.0 * U * (self.s_norm + qt * self.p_norm);
        self.r_fro2 += self.r_norm * self.r_norm;
        self.p_fro2 += self.p_norm * self.p_norm;
        self.rot_prop += self.d_r * self.p_norm + self.r_norm * self.d_p;
        self.coef_prop += self.d_r * q.abs() + self.r_norm * d_q;
        self.coef_scale += self.r_norm * q.abs();
        q
    }
}

#[cfg(test)]
mod tests_route_rule {
    use super::*;

    #[test]
    fn never_when_p_is_at_least_twice_n_tr() {
        for n_tr in [2usize, 10, 100, 1000, 1999] {
            for p in [2 * n_tr, 2 * n_tr + 1, 3 * n_tr] {
                if p > GRAM_P_MAX {
                    continue;
                }
                for b in [1usize, 100, 10_000, 1_000_000] {
                    for k in [1usize, 2, K_GRAM_MAX] {
                        assert!(
                            !use_gram_p_route(n_tr, p, b, k),
                            "n_tr={n_tr} p={p} B={b} k={k}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn caps_bind() {
        // p: cost rule and work floor both hold at p = GRAM_P_MAX + 1; the
        // cap is what refuses it. The premise is asserted so a future edit
        // cannot make the test vacuous.
        let (n_tr, b, k) = (100_000usize, 10_000usize, 2usize);
        let rule = |p: usize| gram_p_cost_rule(n_tr as f64, p as f64, b as f64, k as f64);
        assert!(
            rule(GRAM_P_MAX + 1),
            "premise: cost rule admits p = cap + 1"
        );
        assert!(use_gram_p_route(n_tr, GRAM_P_MAX, b, k));
        assert!(!use_gram_p_route(n_tr, GRAM_P_MAX + 1, b, k));
        // k: same story at the component cap.
        assert!(use_gram_p_route(10_000, 50, 1000, K_GRAM_MAX));
        assert!(!use_gram_p_route(10_000, 50, 1000, K_GRAM_MAX + 1));
        assert!(
            use_gram_p_route(10_000, 1000, 1001, K_GRAM_MAX),
            "wide-ish 10000x1000 at the component cap"
        );
    }

    #[test]
    fn work_floor_keeps_the_existing_corpus_shapes_on_the_x_backend() {
        // perm_null n = 80, d = 6, n_perm = 200, k = 2: the cost rule alone
        // admits it, the work floor does not.
        let (n, p, b, k) = (80usize, 6usize, 200usize, 2usize);
        assert!(
            gram_p_cost_rule(n as f64, p as f64, b as f64, k as f64),
            "premise: the cost rule admits the corpus shape"
        );
        assert!(!use_gram_p_route(n, p, b, k));
        // raw_perm (n_tr = 64, B = 201) and split_exact (n_train = 40,
        // B = 201) at the same n, d, k.
        assert!(!use_gram_p_route(64, 6, 201, 2));
        assert!(!use_gram_p_route(40, 6, 201, 2));
        // The wide route fixtures (n = 60, p = 3000) have p ≥ 2·n_tr.
        assert!(!use_gram_p_route(60, 3000, 200, 1));
        assert!(!use_gram_p_route(48, 3000, 201, 2));
    }

    #[test]
    fn very_tall_k3_falls_back_past_the_ratio_cap() {
        // p = 8, B = 300, k = 3: the cost rule and work floor hold on both
        // sides (checked as premises), so the ratio cap alone decides.
        let (p, b, k) = (8usize, 300usize, 3usize);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n_at_cap = GRAM_P_TALL_RATIO_MAX_K3 as usize * p;
        let rule = |n: usize| gram_p_cost_rule(n as f64, p as f64, b as f64, k as f64);
        let work = |n: usize| b as f64 * (2.0 * k as f64 + 1.0) * n as f64 * p as f64;
        assert!(rule(n_at_cap) && rule(n_at_cap + p), "premise: cost rule");
        assert!(
            work(n_at_cap) >= GRAM_P_MIN_WORK && work(n_at_cap + p) >= GRAM_P_MIN_WORK,
            "premise: work floor"
        );
        // Just inside the ratio cap (n_tr / p == GRAM_P_TALL_RATIO_MAX_K3):
        // still offered.
        assert!(use_gram_p_route(n_at_cap, p, b, k));
        // One p past it (n_tr / p > GRAM_P_TALL_RATIO_MAX_K3): refused.
        assert!(!use_gram_p_route(n_at_cap + p, p, b, k));
        // k = 2 at the same shapes is untouched by the ratio cap.
        assert!(use_gram_p_route(n_at_cap, p, b, 2));
        assert!(use_gram_p_route(n_at_cap + p, p, b, 2));
        // A measured win this cap must keep: n_tr / p = 2500 at k = 3
        // (20 000 × 8: 7.9x faster, 1.5% fallback).
        assert!(use_gram_p_route(20_000, 8, 1000, 3));
        // A measured loss this cap must refuse: n_tr / p = 12500
        // (100 000 × 8: fallback about 100%).
        assert!(!use_gram_p_route(100_000, 8, 300, 3));
    }

    #[test]
    fn degenerate_inputs_stay_on_the_x_backend() {
        assert!(!use_gram_p_route(0, 10, 1000, 2));
        assert!(!use_gram_p_route(10_000, 0, 1000, 2));
        assert!(!use_gram_p_route(10_000, 10, 0, 2));
        assert!(!use_gram_p_route(10_000, 10, 1000, 0));
        // More components than features.
        assert!(!use_gram_p_route(10_000, 1, 1000, 2));
    }
}

/// Design generators shared by the Gram-p tests here, in `perm_null.rs`
/// and in `signal_test.rs`.
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
pub(crate) mod test_designs {
    use faer::{Col, Mat};
    use rand::{RngExt, SeedableRng};

    /// `n × d` uniform on `[-1, 1)`.
    pub(crate) fn unif(n: usize, d: usize, seed: u64) -> Mat<f64> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0))
    }

    /// Length-`n` uniform on `[-1, 1)`.
    pub(crate) fn unif_col(n: usize, seed: u64) -> Col<f64> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0))
    }

    /// `y = X b + noise·e`, `b_j = 1/(j + 1)`, `e` uniform.
    pub(crate) fn linear_y(x: &Mat<f64>, noise: f64, seed: u64) -> Col<f64> {
        let e = unif_col(x.nrows(), seed);
        Col::<f64>::from_fn(x.nrows(), |i| {
            (0..x.ncols())
                .map(|j| x[(i, j)] / (j as f64 + 1.0))
                .sum::<f64>()
                + noise * e[i]
        })
    }

    /// The arrays a fixed-X loop hands its kernel: X and y standardized
    /// (weighted moments under `normalize_weights(w)` when weighted), both
    /// then scaled by `fit_row_scale` of those weights.
    pub(crate) fn prepared(
        x: &Mat<f64>,
        y: &Col<f64>,
        w: Option<&Col<f64>>,
    ) -> (Mat<f64>, Col<f64>) {
        use crate::linalg::{
            normalize_weights, standardize, standardize1, standardize1_weighted,
            standardize_weighted,
        };
        match w {
            None => {
                let (xs, _, _) = standardize(x.as_ref());
                let (ys, _, _) = standardize1(y.as_ref());
                (xs, ys)
            }
            Some(w) => {
                let wn = normalize_weights(w.as_ref()).expect("some weight is nonzero");
                let (xs, _, _) = standardize_weighted(x.as_ref(), Some(wn.as_ref()));
                let (ys, _, _) = standardize1_weighted(y.as_ref(), Some(wn.as_ref()));
                let sqw = crate::fit::fit_row_scale(wn.as_ref());
                (
                    crate::fit::scale_rows(xs.as_ref(), sqw.as_ref()),
                    crate::fit::scale_col(ys.as_ref(), sqw.as_ref()),
                )
            }
        }
    }

    /// `√n · U diag(σ) V'` with `σ_j = κ^(−j/(p−1))`: condition number
    /// exactly `kappa` up to rounding. Columns are not centered; the kernel
    /// comparison does not need them to be.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn conditioned(n: usize, p: usize, kappa: f64, seed: u64) -> Mat<f64> {
        let a = unif(n, p, seed);
        let b = unif(p, p, seed + 1);
        let ua = a.thin_svd().expect("svd");
        let vb = b.thin_svd().expect("svd");
        let (u, v) = (ua.U(), vb.U());
        let sig = |j: usize| {
            if p == 1 {
                1.0
            } else {
                kappa.powf(-(j as f64) / (p as f64 - 1.0))
            }
        };
        let scale = (n as f64).sqrt();
        Mat::<f64>::from_fn(n, p, |i, j| {
            (0..p).map(|l| u[(i, l)] * sig(l) * v[(j, l)]).sum::<f64>() * scale
        })
    }

    /// A `y` orthogonal to `1` and to every column of `xs` (Gram-Schmidt,
    /// each projection applied twice): `X'y` is rounding noise.
    pub(crate) fn orthogonal_y(xs: &Mat<f64>, seed: u64) -> Col<f64> {
        let n = xs.nrows();
        let basis = crate::test_support::orthonormal_basis(
            Col::<f64>::from_fn(n, |_| 1.0).as_ref(),
            xs.as_ref(),
            0.0,
        );
        let e = unif_col(n, seed);
        let mut y = Col::<f64>::from_fn(n, |i| 5.0 + e[i]);
        crate::test_support::project_off(&basis, &mut y);
        y
    }

    /// `y = Σ_{l<m} (l + 1)·u_l` over the leading left singular vectors of
    /// `xs`: exactly `m` components are real.
    pub(crate) fn span_y(xs: &Mat<f64>, m: usize) -> Col<f64> {
        let svd = xs.thin_svd().expect("svd");
        let u = svd.U();
        Col::<f64>::from_fn(xs.nrows(), |i| {
            (0..m).map(|l| (1.0 + l as f64) * u[(i, l)]).sum::<f64>()
        })
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests_block {
    use super::test_designs::{linear_y, prepared, unif};
    use super::*;
    use crate::resample::block_par;

    fn lambda_max(c: &Mat<f64>) -> f64 {
        let ev = c.self_adjoint_eigenvalues(faer::Side::Lower).expect("evd");
        *ev.last().expect("p ≥ 1")
    }

    #[test]
    fn c_build_is_thread_count_invariant_at_the_route_shapes() {
        // byte_parity compares Par::Seq (disable_parallelism) with
        // `block_par(false)`, the fixed-degree Rayon split, so the block must
        // come out with the same bits serial and in pools of two and seven
        // threads. Shapes: the fixture blocks (perm_null, raw_perm fold,
        // split_exact half), the memprobe shape, a block large enough for
        // faer's parallel GEMM, and d = 1, once short and once long enough
        // (n ≥ 256² entries) for faer to split the reduction dimension.
        assert_eq!(block_par(true), Par::Seq);
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
            let seq = GramPBlock::new(xs.as_ref(), block_par(true));
            for threads in [2usize, 7] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool");
                let par = pool.install(|| GramPBlock::new(xs.as_ref(), block_par(false)));
                for j in 0..p {
                    for i in 0..p {
                        assert_eq!(
                            par.c[(i, j)].to_bits(),
                            seq.c[(i, j)].to_bits(),
                            "{n}×{p}, {threads} threads: C[{i}, {j}]"
                        );
                    }
                }
                assert_eq!(
                    par.c2_norm.to_bits(),
                    seq.c2_norm.to_bits(),
                    "{n}×{p}: c2_norm"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // the cap is `min(2·λ̂, fro2)`, compared exactly
    fn c2_norm_is_bracketed_by_the_top_eigenvalue_and_the_trace() {
        // Uniform, a constant (zero after standardization) column, p = 1,
        // two exactly opposite columns (the all-ones start is orthogonal to
        // C's top direction), and a κ = 1e6 block.
        let mut designs: Vec<(&str, Mat<f64>)> = Vec::new();
        let x = unif(500, 20, 35);
        designs.push(("uniform", prepared(&x, &linear_y(&x, 1.0, 36), None).0));
        let mut xc = unif(400, 10, 37);
        for i in 0..400 {
            xc[(i, 3)] = 2.5;
        }
        designs.push((
            "constant_column",
            prepared(&xc, &linear_y(&xc, 1.0, 38), None).0,
        ));
        let x1 = unif(3000, 1, 39);
        designs.push(("p1", prepared(&x1, &linear_y(&x1, 1.0, 40), None).0));
        let (xs1, _, _) = crate::linalg::standardize(unif(600, 1, 41).as_ref());
        designs.push((
            "opposite_columns",
            Mat::<f64>::from_fn(
                600,
                2,
                |i, j| if j == 0 { xs1[(i, 0)] } else { -xs1[(i, 0)] },
            ),
        ));
        designs.push((
            "kappa_1e6",
            super::test_designs::conditioned(400, 20, 1e6, 42),
        ));
        for (label, xs) in &designs {
            let block = GramPBlock::new(xs.as_ref(), Par::Seq);
            let fro2 = block.x_fro * block.x_fro;
            let lam = lambda_max(&block.c);
            assert!(block.c2_norm <= fro2, "{label}: capped by tr C");
            assert!(
                lam <= block.c2_norm * (1.0 + 1e-12),
                "{label}: λ_max {lam:e} above the estimate {:e}",
                block.c2_norm
            );
            assert!(
                block.c2_norm == fro2 || block.c2_norm <= 2.0 * lam * (1.0 + 1e-12),
                "{label}: estimate {:e} above 2·λ_max {:e} and below tr C",
                block.c2_norm,
                2.0 * lam
            );
        }
        assert_eq!(
            c2_norm_estimate(Mat::<f64>::zeros(3, 3).as_ref(), 2.0),
            4.0,
            "zero C falls back to x_fro²"
        );
    }
}

#[cfg(test)]
mod tests_kernel {
    use super::test_designs::{linear_y, prepared, unif, unif_col};
    use super::*;
    use crate::fit::{pls1_fit_prepared_fro, ParChoice};
    use faer::{Col, Mat};

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
            let g = block
                .fit_replicate_diag(ys.as_ref(), 2, keep)
                .expect("resolves");
            let xf = pls1_fit_prepared_fro(
                xs.as_ref(),
                ys.as_ref(),
                2,
                keep,
                ParChoice::Seq,
                xs.norm_l2(),
            )
            .expect("X backend");
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
    fn a_non_finite_outcome_is_never_resolved() {
        // Fail closed: a NaN or infinite entry in ys must reach the Primal
        // arm (which reports it), never a Gram fit or a zero model.
        let x = unif(300, 12, 23);
        let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 24), None);
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut yb = ys.clone();
            yb[7] = bad;
            for keep in [None, Some(4)] {
                for k in 1..=K_GRAM_MAX {
                    assert!(
                        block.fit_replicate_diag(yb.as_ref(), k, keep).is_err(),
                        "ys[7] = {bad}, k = {k}, keep = {keep:?}: resolved"
                    );
                }
            }
        }
    }

    #[test]
    fn the_ratio_gate_refuses_a_nearly_collinear_direction() {
        // Two columns 1e-3 apart: C's small eigenvalue is about 2.5e-7 of
        // tr C, under RATIO_MIN, while the tt band still decides at a = 2
        // (δtt/tt about 2e-2 against the band's 1/8). The transcription in
        // `scripts/gate_feasibility.py` gives Ratio at a = 2 on every draw of this design.
        let x1 = unif(1000, 1, 13);
        let e = unif_col(1000, 14);
        let x = Mat::<f64>::from_fn(1000, 2, |i, j| {
            if j == 0 {
                x1[(i, 0)]
            } else {
                x1[(i, 0)] + 1e-3 * e[i]
            }
        });
        let y = Col::<f64>::from_fn(1000, |i| x1[(i, 0)] + 10.0 * e[i]);
        let (xs, ys) = prepared(&x, &y, None);
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        assert!(
            block.fit_replicate_diag(ys.as_ref(), 1, None).is_ok(),
            "k = 1 resolves"
        );
        let r = block.fit_replicate_diag(ys.as_ref(), 2, None);
        assert!(matches!(r, Err(GateFail::Ratio)), "{:?}", r.err());
    }

    #[test]
    fn a_keep_boundary_inside_rounding_falls_back() {
        // Columns 15..30 duplicate 0..15, so from a = 2 on every |s| entry
        // has a twin and an odd keep puts the boundary between twins, a tie
        // the two backends need not break alike.
        let x0 = unif(400, 15, 7);
        let x = Mat::<f64>::from_fn(400, 30, |i, j| x0[(i, j % 15)]);
        let (xs, ys) = prepared(&x, &linear_y(&x0, 1.0, 8), None);
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        let r = block.fit_replicate_diag(ys.as_ref(), 3, Some(3));
        assert!(matches!(r, Err(GateFail::KeepGap)), "{:?}", r.err());
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
        )
        .expect("X backend");
        let (coef, k_used) = block
            .fit_replicate(ys.as_ref(), 1, None)
            .expect("p = 1 resolves");
        assert_eq!(k_used, xf.k_used);
        assert_coef_close(&coef, &xf.coef, "p = 1");
        // A constant column standardizes to an exact zero column, so C has
        // a zero row and column.
        let mut x = unif(400, 10, 11);
        for i in 0..400 {
            x[(i, 3)] = 2.5;
        }
        let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 12), None);
        assert!((0..400).all(|i| xs[(i, 3)] == 0.0), "premise");
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        for k in 1..=K_GRAM_MAX {
            let xf = pls1_fit_prepared_fro(
                xs.as_ref(),
                ys.as_ref(),
                k,
                None,
                ParChoice::Seq,
                xs.norm_l2(),
            )
            .expect("X backend");
            match block.fit_replicate_diag(ys.as_ref(), k, None) {
                Ok(g) => {
                    assert_eq!(g.k_used, xf.k_used, "k={k}");
                    assert_coef_close(&g.coef, &xf.coef, &format!("k={k}"));
                }
                Err(f) => assert!(k > 2, "k={k} falls back ({f:?}) on well-posed data"),
            }
        }
    }

    #[test]
    // The reference values are pasted as `scripts/gate_feasibility.py` prints them; the design
    // names follow the formulas (n, p, x, y, i, j).
    #[allow(clippy::unreadable_literal, clippy::many_single_char_names)]
    fn bounds_match_gate_feasibility_script() {
        // Printed by `python3 scripts/gate_feasibility.py gram_p --reference`:
        // `(tt_a, δtt_a, δp_a)` for a = 1..=3 on the
        // closed-form design below, with the ‖C‖₂ estimate. The script
        // transcribes this backend's recursion, so the two agree to
        // rounding; a mismatch means one transcription left the formulas of
        // `fit_replicate_diag`'s doc comment.
        const SCRIPT_C2_NORM: f64 = 1719.5640898689182;
        const SCRIPT_REFERENCE: &[(f64, f64, f64)] = &[
            (
                599.6834622091338,
                2.2808421817899206e-10,
                7.820128791640594e-13,
            ),
            (
                466.82408412069884,
                5.1466229828362756e-08,
                1.6251478230266874e-10,
            ),
            (
                651.4987223020075,
                3.018567086830902e-05,
                6.930737553433065e-08,
            ),
        ];
        let (n, p) = (400usize, 12usize);
        let x = Mat::<f64>::from_fn(n, p, |i, j| {
            let (fi, fj) = (i as f64, j as f64);
            ((fi + 1.0) * (fj + 1.0) * 0.37).sin() + 0.5 * (1.1 * fi - 0.7 * fj).cos()
        });
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let y = Col::<f64>::from_fn(n, |i| (0.9 * i as f64 + 0.3).cos() + 0.5 * xs[(i, 0)]);
        let (ys, _, _) = crate::linalg::standardize1(y.as_ref());
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        assert!(
            rel(block.c2_norm, SCRIPT_C2_NORM) <= 1e-12,
            "c2_norm {:e} vs script {SCRIPT_C2_NORM:e}",
            block.c2_norm
        );
        let g = block
            .fit_replicate_diag(ys.as_ref(), 3, None)
            .expect("the reference design resolves at k = 3");
        assert_eq!(g.bounds.len(), SCRIPT_REFERENCE.len());
        for (a, (&(tt, d_tt, d_p), &(s_tt, s_dtt, s_dp))) in
            g.bounds.iter().zip(SCRIPT_REFERENCE).enumerate()
        {
            for (what, got, want) in [("tt", tt, s_tt), ("d_tt", d_tt, s_dtt), ("d_p", d_p, s_dp)] {
                assert!(
                    rel(got, want) <= 1e-12,
                    "component {}: {what} {got:e} vs script {want:e}",
                    a + 1
                );
            }
        }
    }
}

#[cfg(test)]
mod tests_sweep {
    use super::test_designs::{
        conditioned, linear_y, orthogonal_y, prepared, span_y, unif, unif_col,
    };
    use super::*;
    use crate::fit::{pls1_fit_prepared_fro, ParChoice};
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Tally {
        total: usize,
        resolved: usize,
        fails: BTreeMap<String, usize>,
        fails_by_k: BTreeMap<usize, usize>,
        /// Largest `max_j |coef_gram − coef_x| / max(1, ‖coef_x‖∞)` over
        /// resolved replicates (the tolerance is 1e-10).
        max_err: f64,
        /// Largest per-component `max(δtt/tt, δp/‖p‖)`, a diagnostic.
        max_rel: f64,
    }

    impl Tally {
        fn fallback_rate(&self) -> f64 {
            (self.total - self.resolved) as f64 / self.total.max(1) as f64
        }
    }

    /// One replicate on both backends: gate soundness (a Resolved replicate
    /// has the X backend's `k_used`) is asserted at once; the normalized
    /// `coef` error and the outcome are tallied.
    fn check_one(
        t: &mut Tally,
        label: &str,
        block: &GramPBlock<'_>,
        xs: MatRef<'_, f64>,
        ys: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
    ) {
        let x = pls1_fit_prepared_fro(xs, ys, k, keep, ParChoice::Seq, xs.norm_l2())
            .expect("X backend fit");
        t.total += 1;
        match block.fit_replicate_diag(ys, k, keep) {
            Ok(g) => {
                assert_eq!(
                    g.k_used, x.k_used,
                    "{label}, k={k}, keep={keep:?}: Resolved with k_used {} where the X \
                     backend has {} (an unsound decision gate: fix it, never recalibrate)",
                    g.k_used, x.k_used
                );
                assert!(
                    (0..xs.ncols()).all(|j| g.coef[j].is_finite()),
                    "{label}, k={k}: non-finite coef"
                );
                let scale = (0..xs.ncols())
                    .map(|j| x.coef[j].abs())
                    .fold(1.0_f64, f64::max);
                let err = (0..xs.ncols())
                    .map(|j| (g.coef[j] - x.coef[j]).abs())
                    .fold(0.0_f64, f64::max)
                    / scale;
                t.resolved += 1;
                t.max_err = t.max_err.max(err);
                t.max_rel = t.max_rel.max(g.max_rel);
            }
            Err(f) => {
                *t.fails.entry(format!("{f:?}")).or_default() += 1;
                *t.fails_by_k.entry(k).or_default() += 1;
            }
        }
    }

    /// Every k from 1 to `min(k_max, K_GRAM_MAX, p)` on one design: the
    /// component counts the route can reach.
    fn run_design(
        t: &mut Tally,
        label: &str,
        xs: &Mat<f64>,
        ys: &Col<f64>,
        k_max: usize,
        keep: Option<usize>,
    ) {
        let block = GramPBlock::new(xs.as_ref(), Par::Seq);
        for k in 1..=k_max.min(K_GRAM_MAX).min(xs.ncols()) {
            check_one(t, label, &block, xs.as_ref(), ys.as_ref(), k, keep);
        }
    }

    fn report(title: &str, tallies: &BTreeMap<&'static str, Tally>) {
        eprintln!("== {title} (RATIO_MIN = {RATIO_MIN:e}, K_GRAM_MAX = {K_GRAM_MAX})");
        eprintln!("family\ttotal\tresolved\tfallback\tmax_err\tmax_rel\tby_gate\tby_k");
        for (name, t) in tallies {
            eprintln!(
                "{name}\t{}\t{}\t{:.4}\t{:.3e}\t{:.3e}\t{:?}\t{:?}",
                t.total,
                t.resolved,
                t.fallback_rate(),
                t.max_err,
                t.max_rel,
                t.fails,
                t.fails_by_k
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn gram_backend_matches_x_backend_across_design_families() {
        let mut tallies: BTreeMap<&'static str, Tally> = BTreeMap::new();
        // Ordinary: signal and null outcomes.
        for seed in 0..4u64 {
            let x = unif(400, 30, 100 + seed);
            for y in [linear_y(&x, 1.0, 200 + seed), unif_col(400, 300 + seed)] {
                let (xs, ys) = prepared(&x, &y, None);
                run_design(
                    tallies.entry("ordinary").or_default(),
                    "ordinary",
                    &xs,
                    &ys,
                    20,
                    None,
                );
            }
        }
        // n ≫ p.
        for seed in 0..2u64 {
            let x = unif(5000, 8, 110 + seed);
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 210 + seed), None);
            run_design(
                tallies.entry("n_much_larger_than_p").or_default(),
                "n>>p",
                &xs,
                &ys,
                8,
                None,
            );
        }
        // Rank-deficient: columns 20..30 are sums of earlier ones (rank 20).
        {
            let mut x = unif(300, 30, 120);
            for j in 20..30 {
                for i in 0..300 {
                    x[(i, j)] = x[(i, j - 20)] + x[(i, j - 19)];
                }
            }
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 220), None);
            run_design(
                tallies.entry("rank_deficient").or_default(),
                "rank_deficient",
                &xs,
                &ys,
                20,
                None,
            );
        }
        // y exhausted: rank 39 of 40, y independent of X (fit.rs design).
        {
            let mut x = unif(2000, 40, 11);
            for i in 0..2000 {
                x[(i, 39)] = x[(i, 0)] + x[(i, 1)];
            }
            let (xs, ys) = prepared(&x, &unif_col(2000, 12), None);
            run_design(
                tallies.entry("y_exhausted").or_default(),
                "y_exhausted",
                &xs,
                &ys,
                20,
                None,
            );
        }
        // Weak signal: 1e-3 of the signal on top of noise.
        for seed in 0..2u64 {
            let x = unif(400, 30, 130 + seed);
            let e = unif_col(400, 230 + seed);
            let s = linear_y(&x, 0.0, 0);
            let y = Col::<f64>::from_fn(400, |i| e[i] + 1e-3 * s[i]);
            let (xs, ys) = prepared(&x, &y, None);
            run_design(
                tallies.entry("weak_signal").or_default(),
                "weak_signal",
                &xs,
                &ys,
                20,
                None,
            );
        }
        // Collinear columns, κ(Xs) = 1e2, 1e4, 1e6.
        for (tag, kappa) in [
            ("collinear_1e2", 1e2),
            ("collinear_1e4", 1e4),
            ("collinear_1e6", 1e6),
        ] {
            let xs = conditioned(400, 20, kappa, 140);
            let (ys, _, _) = crate::linalg::standardize1(linear_y(&xs, 0.1, 240).as_ref());
            run_design(tallies.entry(tag).or_default(), tag, &xs, &ys, 20, None);
        }
        // A nearly collinear pair inside an ordinary block (the ratio
        // gate's case), at 1e-2, 1e-3 and 1e-4 apart.
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
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 245), None);
            run_design(tallies.entry(tag).or_default(), tag, &xs, &ys, 10, None);
        }
        // Duplicated rows.
        {
            let x0 = unif(150, 20, 150);
            let x = Mat::<f64>::from_fn(300, 20, |i, j| x0[(i % 150, j)]);
            let y0 = linear_y(&x0, 1.0, 250);
            let y = Col::<f64>::from_fn(300, |i| y0[i % 150]);
            let (xs, ys) = prepared(&x, &y, None);
            run_design(
                tallies.entry("duplicated_rows").or_default(),
                "duplicated_rows",
                &xs,
                &ys,
                20,
                None,
            );
        }
        // y orthogonal to X: the zero model, decided exactly at a = 1.
        for seed in [41u64, 43, 45] {
            let x = unif(200, 10, seed);
            let (xs0, _, _) = crate::linalg::standardize(x.as_ref());
            let (xs, ys) = prepared(&x, &orthogonal_y(&xs0, seed + 1), None);
            run_design(
                tallies.entry("y_orthogonal").or_default(),
                "y_orthogonal",
                &xs,
                &ys,
                10,
                None,
            );
        }
        // y in the span of the first m = 2 directions.
        {
            let x = unif(500, 30, 21);
            let (xs, _, _) = crate::linalg::standardize(x.as_ref());
            let (ys, _, _) = crate::linalg::standardize1(span_y(&xs, 2).as_ref());
            run_design(
                tallies.entry("y_in_span_m2").or_default(),
                "y_in_span_m2",
                &xs,
                &ys,
                10,
                None,
            );
        }
        // Sparse keep, signal and null.
        for keep in [1usize, 5, 15, 29] {
            let x = unif(400, 30, 160);
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 260), None);
            run_design(
                tallies.entry("sparse_keep").or_default(),
                "sparse_keep",
                &xs,
                &ys,
                10,
                Some(keep),
            );
            let (xs, ys) = prepared(&x, &unif_col(400, 261), None);
            run_design(
                tallies.entry("sparse_keep").or_default(),
                "sparse_keep",
                &xs,
                &ys,
                10,
                Some(keep),
            );
        }
        // Duplicated columns under keep: exact ties at the boundary.
        {
            let x0 = unif(400, 15, 170);
            let x = Mat::<f64>::from_fn(400, 30, |i, j| x0[(i, j % 15)]);
            let (xs, ys) = prepared(&x, &linear_y(&x0, 1.0, 270), None);
            for keep in [3usize, 10] {
                run_design(
                    tallies.entry("duplicated_columns_keep").or_default(),
                    "dup_cols_keep",
                    &xs,
                    &ys,
                    6,
                    Some(keep),
                );
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
            let (xs, ys) = prepared(&x, &y, Some(&w));
            run_design(
                tallies.entry("weighted_zero_weights").or_default(),
                "weighted",
                &xs,
                &ys,
                20,
                None,
            );
            run_design(
                tallies.entry("weighted_zero_weights_keep").or_default(),
                "weighted_keep",
                &xs,
                &ys,
                10,
                Some(10),
            );
        }

        report("design family sweep", &tallies);
        for (fam, t) in &tallies {
            assert!(
                t.max_err <= 1e-10,
                "{fam}: a resolved replicate is {:e}·max(1, ‖coef‖∞) from the X backend, \
                 outside the 1e-10 tolerance: raise RATIO_MIN (see \
                 gram_p::tests_sweep::gram_backend_matches_x_backend_across_design_families)",
                t.max_err
            );
        }
        let global_err = tallies.values().map(|t| t.max_err).fold(0.0_f64, f64::max);
        assert!(
            global_err <= 1e-11,
            "max normalized |Δcoef| {global_err:e} over the sweep leaves under a 10x margin to \
             1e-10: raise RATIO_MIN (see \
             gram_p::tests_sweep::gram_backend_matches_x_backend_across_design_families)"
        );
        for fam in [
            "ordinary",
            "n_much_larger_than_p",
            "weak_signal",
            "weighted_zero_weights",
        ] {
            let t = &tallies[fam];
            assert!(
                t.fallback_rate() <= 0.01,
                "{fam}: fallback rate {} on ordinary data ({:?}, by k {:?}): lower \
                 K_GRAM_MAX (see gram_p::tests_sweep::gram_backend_matches_x_backend_across_design_families)",
                t.fallback_rate(),
                t.fails,
                t.fails_by_k
            );
        }
        let orth = &tallies["y_orthogonal"];
        assert_eq!(
            orth.resolved, orth.total,
            "a stop at a = 1 is decided exactly"
        );
    }

    #[test]
    #[ignore = "diagnostic: sweeps n/p at k = 3 (and k = 2) to find the gate that fails on very \
                tall designs. Run with `cargo test -p plskit --release \
                tall_shapes_n_over_p_sweep -- --ignored --nocapture`."]
    fn tall_shapes_n_over_p_sweep() {
        // n/p ratios from 1000 to 20000, p in {4, 8, 16}, k = 3 and k = 2
        // (control). One block, 300 permuted replicates of a real signal
        // outcome per cell, on the perm_null path's own shape.
        let ratios = [
            1000usize, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 12500, 16000, 20000,
        ];
        let mut tallies: BTreeMap<&'static str, Tally> = BTreeMap::new();
        for &p in &[4usize, 8, 16] {
            for &ratio in &ratios {
                let n = ratio * p;
                let x = unif(n, p, 500 + p as u64);
                let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 600 + p as u64), None);
                let block = GramPBlock::new(xs.as_ref(), Par::Seq);
                for &k in &[3usize, 2] {
                    let label = format!("p{p}_ratio{ratio}_k{k}");
                    let t = tallies
                        .entry(Box::leak(label.into_boxed_str()))
                        .or_default();
                    for b in 0..300u64 {
                        let perm = crate::resample::permutation_from_seed(n, 10_000 + b);
                        let yb = Col::<f64>::from_fn(n, |i| ys[perm[i]]);
                        check_one(t, "sweep", &block, xs.as_ref(), yb.as_ref(), k, None);
                    }
                }
            }
        }
        report("n/p sweep at k = 3 and k = 2", &tallies);
    }

    #[test]
    fn fallback_is_rare_on_ordinary_data_at_route_shapes() {
        // The fixture blocks and the memprobe shape at their k, permuted
        // outcomes as the replicate loops feed them.
        let mut tallies: BTreeMap<&'static str, Tally> = BTreeMap::new();
        for (tag, n, p, k, keep, reps) in [
            (
                "perm_null_n2000_p50_k3",
                2000usize,
                50usize,
                3usize,
                None,
                200usize,
            ),
            ("raw_perm_fold_n1600_p50_k2", 1600, 50, 2, None, 200),
            (
                "split_half_n1000_p200_k1_keep10",
                1000,
                200,
                1,
                Some(10usize),
                200,
            ),
            ("memprobe_n10000_p500_k3", 10_000, 500, 3, None, 20),
        ] {
            let x = unif(n, p, 190);
            let (xs, ys) = prepared(&x, &linear_y(&x, 2.0, 290), None);
            let block = GramPBlock::new(xs.as_ref(), Par::Seq);
            let t = tallies.entry(tag).or_default();
            for b in 0..reps {
                let perm = crate::resample::permutation_from_seed(n, 1000 + b as u64);
                let yb = Col::<f64>::from_fn(n, |i| ys[perm[i]]);
                check_one(t, tag, &block, xs.as_ref(), yb.as_ref(), k, keep);
            }
        }
        report("route shapes", &tallies);
        for (tag, t) in &tallies {
            assert!(
                t.max_err <= 1e-10,
                "{tag}: max_err {:e} outside tolerance",
                t.max_err
            );
            assert!(
                t.fallback_rate() <= 0.01,
                "{tag}: fallback rate {} ({:?}): see gram_p::tests_sweep::fallback_is_rare_on_ordinary_data_at_route_shapes",
                t.fallback_rate(),
                t.fails
            );
        }
    }
}

#[cfg(test)]
mod tests_site_route {
    use super::*;
    use crate::dual_route::{
        nspace_eligible_perm_null, nspace_eligible_raw_perm, nspace_eligible_split_exact,
    };
    use crate::perm_null::perm_null_route;
    use crate::signal_test::{
        raw_perm_route, split_exact_refit_route, with_gram_routes_disabled, ReplicateRoute,
    };

    #[test]
    fn weighted_replicates_prefer_gram_p_in_the_overlap_band() {
        // perm_null: n = 1000 < p = 1200 < 1.5n, k = 2, n_perm = 5000. Both
        // routes are eligible (premises asserted: the n-space edge is
        // p ~ 1111, the Gram-p edge p ~ 1500); the n-space route is cheaper
        // there.
        assert!(
            nspace_eligible_perm_null(1000, 1200, 5000, 2, true),
            "premise"
        );
        assert!(gram_p_eligible(1000, 1200, 5000, 2, None), "premise");
        assert_eq!(
            perm_null_route(1000, 1200, 5000, 2, false),
            ReplicateRoute::Nspace
        );
        // Weighted: the n-space route refuses, Gram-p takes it.
        assert_eq!(
            perm_null_route(1000, 1200, 5000, 2, true),
            ReplicateRoute::GramP
        );
        // raw_perm: n = 1250, n_folds = 5, so n_tr = 1000.
        assert!(
            nspace_eligible_raw_perm(1250, 5, 1200, 5000, 2, true),
            "premise"
        );
        assert!(gram_p_eligible(1000, 1200, 5001, 2, None), "premise");
        assert_eq!(
            raw_perm_route(1250, 5, 1200, 5000, 2, None, false),
            ReplicateRoute::Nspace
        );
        // keep: the n-space route refuses, Gram-p takes it.
        assert_eq!(
            raw_perm_route(1250, 5, 1200, 5000, 2, Some(100), false),
            ReplicateRoute::GramP
        );
        // split_exact: n = 2000, so n_train = 1000.
        assert!(
            nspace_eligible_split_exact(1000, 1200, 5000, 2, true),
            "premise"
        );
        assert!(gram_p_eligible(1000, 1200, 5001, 2, None), "premise");
        assert_eq!(
            split_exact_refit_route(2000, 1200, 5000, 2, None, false),
            ReplicateRoute::Nspace
        );
    }

    #[test]
    fn special_routes_keep_their_inputs() {
        // split_exact k = 1 without keep: the no-refit route, even on a
        // Gram-eligible tall shape.
        assert!(gram_p_eligible(1000, 50, 1001, 1, None), "premise");
        assert_eq!(
            split_exact_refit_route(2000, 50, 1000, 1, None, false),
            ReplicateRoute::Special
        );
    }

    #[test]
    fn k1_on_tall_data_takes_gram_p_where_no_special_route_claims_it() {
        assert_eq!(
            raw_perm_route(2000, 5, 50, 1000, 1, None, false),
            ReplicateRoute::GramP
        );
        assert_eq!(
            perm_null_route(2000, 50, 1000, 1, false),
            ReplicateRoute::GramP
        );
    }

    #[test]
    fn invalid_arguments_never_take_the_gram_p_route() {
        // k = 0, k > p, keep = 0 and keep > p reach the Primal units, whose
        // input checks report them, independent of the route.
        assert_eq!(
            raw_perm_route(2000, 5, 50, 1000, 0, None, false),
            ReplicateRoute::Primal
        );
        assert_eq!(
            perm_null_route(2000, 3, 1000, 4, false),
            ReplicateRoute::Primal
        );
        for keep in [Some(0usize), Some(51)] {
            assert_eq!(
                raw_perm_route(2000, 5, 50, 1000, 2, keep, false),
                ReplicateRoute::Primal,
                "keep {keep:?}"
            );
            assert_eq!(
                split_exact_refit_route(2000, 50, 1000, 2, keep, false),
                ReplicateRoute::Primal,
                "keep {keep:?}"
            );
        }
        assert!(!gram_p_eligible(2000, 50, 1000, 51, None));
        assert!(
            gram_p_eligible(2000, 50, 1000, 2, Some(50)),
            "keep = p is valid"
        );
    }

    #[test]
    fn gram_routes_disabled_restricts_to_the_special_and_primal_routes() {
        let tall = || perm_null_route(2000, 50, 1000, 3, false);
        assert_eq!(tall(), ReplicateRoute::GramP);
        assert_eq!(with_gram_routes_disabled(tall), ReplicateRoute::Primal);
        assert_eq!(
            with_gram_routes_disabled(|| raw_perm_route(2000, 5, 50, 1000, 2, Some(10), false)),
            ReplicateRoute::Primal
        );
        // Special is not a new route and survives the guard.
        assert_eq!(
            with_gram_routes_disabled(|| raw_perm_route(60, 5, 3000, 200, 1, None, false)),
            ReplicateRoute::Special
        );
        // The guard restores the previous state on exit.
        assert_eq!(tall(), ReplicateRoute::GramP);
    }
}

#[cfg(test)]
mod bench_work_floor {
    use super::test_designs::{linear_y, prepared, unif};
    use super::*;
    use crate::fit::{pls1_fit_prepared_fro, ParChoice};
    use std::hint::black_box;
    use std::time::Instant;

    // One fixed block per shape, single thread: B X-backend fits (with
    // `‖Xs‖_F` taken once for the block) against one C build plus B Gram fits
    // (X-backend fallbacks included). Run with
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
        eprintln!("n_tr\tp\tk\tB\twork\tcost_rule\tx_ms\tgram_ms\tspeedup\tsaved_ms\tfallbacks");
        for (n, p, k, b) in shapes {
            let x = unif(n, p, 91);
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 92), None);
            let perms: Vec<Col<f64>> = (0..b)
                .map(|i| {
                    let perm = crate::resample::permutation_from_seed(n, i as u64);
                    Col::<f64>::from_fn(n, |r| ys[perm[r]])
                })
                .collect();
            // `‖Xs‖_F` taken once, as `perm_null`'s Primal arm and the CV
            // folds take it.
            let x_fro = xs.norm_l2();
            let t0 = Instant::now();
            for yb in &perms {
                black_box(
                    pls1_fit_prepared_fro(xs.as_ref(), yb.as_ref(), k, None, ParChoice::Seq, x_fro)
                        .expect("X backend"),
                );
            }
            let x_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let block = GramPBlock::new(xs.as_ref(), Par::Seq);
            let mut fallbacks = 0usize;
            for yb in &perms {
                if black_box(block.fit_replicate(yb.as_ref(), k, None)).is_none() {
                    fallbacks += 1;
                    black_box(
                        pls1_fit_prepared_fro(
                            xs.as_ref(),
                            yb.as_ref(),
                            k,
                            None,
                            ParChoice::Seq,
                            block.x_fro(),
                        )
                        .expect("X backend"),
                    );
                }
            }
            let g_ms = t1.elapsed().as_secs_f64() * 1e3;
            let (nf, pf, bf, kf) = (n as f64, p as f64, b as f64, k as f64);
            let work = bf * (2.0 * kf + 1.0) * nf * pf;
            let rule = gram_p_cost_rule(nf, pf, bf, kf);
            eprintln!(
                "{n}\t{p}\t{k}\t{b}\t{work:.1e}\t{rule}\t{x_ms:.1}\t{g_ms:.1}\t{:.2}\t{:.1}\t{fallbacks}",
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
    // fits (fallbacks included), medians of 3.
    // `GRAM_P_BENCH_SHAPES="n,p,k,B;..."` replaces the default shapes; the
    // sweep that fitted `use_gram_p_route`'s cost coefficients passes its
    // own (see "Fitting `a` and `b`" there). Run
    // with `cargo test -p plskit --release gram_p_route_boundary_benchmark --
    // --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: checks use_gram_p_route near its boundary"]
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
        eprintln!("n_tr\tp\tk\tB\trule\tx_ms\tgram_ms\tspeedup\tfallbacks");
        for (n, p, k, b) in shapes {
            let x = unif(n, p, 91);
            let (xs, ys) = prepared(&x, &linear_y(&x, 1.0, 92), None);
            let perms: Vec<Col<f64>> = (0..b)
                .map(|i| {
                    let perm = crate::resample::permutation_from_seed(n, i as u64);
                    Col::<f64>::from_fn(n, |r| ys[perm[r]])
                })
                .collect();
            // The X backend as `perm_null` runs it: `‖Xs‖_F` taken once.
            let x_fro = xs.norm_l2();
            let x_ms = median_ms(3, || {
                for yb in &perms {
                    black_box(
                        crate::fit::pls1_fit_prepared_fro(
                            xs.as_ref(),
                            yb.as_ref(),
                            k,
                            None,
                            ParChoice::Seq,
                            x_fro,
                        )
                        .expect("X backend"),
                    );
                }
            });
            let mut fallbacks = 0usize;
            let g_ms = median_ms(3, || {
                fallbacks = 0;
                let block = GramPBlock::new(xs.as_ref(), Par::Seq);
                for yb in &perms {
                    if black_box(block.fit_replicate(yb.as_ref(), k, None)).is_none() {
                        fallbacks += 1;
                        black_box(
                            crate::fit::pls1_fit_prepared_fro(
                                xs.as_ref(),
                                yb.as_ref(),
                                k,
                                None,
                                ParChoice::Seq,
                                block.x_fro(),
                            )
                            .expect("X backend"),
                        );
                    }
                }
            });
            let rule = use_gram_p_route(n, p, b, k);
            eprintln!(
                "{n}\t{p}\t{k}\t{b}\t{rule}\t{x_ms:.1}\t{g_ms:.1}\t{:.2}\t{fallbacks}",
                x_ms / g_ms
            );
        }
    }
}
