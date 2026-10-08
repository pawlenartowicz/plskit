//! PLS3 / PLSSVD — symmetric X↔Y covariance analysis. Public entry points:
//! `pls3_fit` (alias `plssvd_fit`) and `pls3_transform` (alias
//! `plssvd_transform`).
//!
//! The fit is one thin SVD of the standardized cross-covariance `X̃'Ỹ`
//! (`p × q`), not `k` sequential passes: there is no deflation, so all
//! components come out orthogonal by construction. The SVD never touches a
//! `p × p` matrix, which is why `p ≫ n` is the ordinary case here rather
//! than a problem.
//!
//! `spls3_fit` is the sparse member of the family and the one exception to
//! the paragraph above: a keep-count on either side cannot come out of a
//! single SVD, so it alternates per component with deflation between
//! components. Its saliences are unit-norm but not orthogonal, and its
//! `singular_values` are the per-component `u'Av` on the deflated `A`, not
//! singular values of `X̃'Ỹ`. See `_docs/concepts/sPLS3/sparse-saliences.md`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{
    check_finite_mat, resolve_par, select_and_normalize, ParChoice, IMPLICIT_MAX_MEAN_RATIO,
};
use crate::linalg::owned_col_slice;

/// Knobs for [`pls3_fit`].
#[derive(Debug, Clone, Copy)]
pub struct Pls3FitOpts {
    /// Skip X centering/scaling; caller asserts X is already standardized.
    /// The returned `x_mean` / `x_scale` are then the identity moments
    /// (0 / 1), matching what `pls1_fit` records on the same branch.
    ///
    /// **Scale contract.** Same failure mode as
    /// [`FitOpts::pre_standardized`](crate::fit::FitOpts::pre_standardized)
    /// (see the "Scale contract" paragraph of its doc comment), deliberately
    /// reused here. When `pre_standardized_x=true`, `pls3_fit` compares
    /// every `σ` of `X̃'Ỹ` against the fixed `SIGMA_FLOOR = 1e-14` absolute
    /// threshold (as well as a floor relative to `‖X̃‖_F·‖Ỹ‖_F`, which is
    /// scale-free). If the "pre-standardized" X
    /// passed in is not genuinely close to zero-mean / unit-variance, that
    /// threshold can silently truncate `k_used` below the requested `k`
    /// instead of erroring. Unlike `pls1_fit`, PLS3 has no `check_n_eff`-equivalent
    /// opt-in to turn truncation into a hard error (out of scope for this
    /// family) — callers passing `pre_standardized_x=true` must ensure the
    /// inputs are genuinely standardized.
    pub pre_standardized_x: bool,
    /// Skip Y centering/scaling; same contract as `pre_standardized_x`.
    pub pre_standardized_y: bool,
    /// Parallelism strategy for the score matmuls. See [`ParChoice`].
    /// Resamplers force `Seq` — outer Rayon already owns the cores.
    pub par: ParChoice,
    /// Iteration cap for the sparse alternation. Ignored on the dense path,
    /// which takes one SVD and never iterates.
    pub max_iter: usize,
    /// Convergence tolerance for the sparse alternation, read as
    /// `‖v − v_prev‖_∞`. `u` is a deterministic function of the previous
    /// `v` (`u = normalize(select(Av_prev))`), so a `v`-only test is not
    /// the weaker one, and it is the only test the Gram route in
    /// `dual_route.rs` can evaluate without entering `p`-space.
    /// Ignored on the dense path.
    pub tol: f64,
}

impl Default for Pls3FitOpts {
    fn default() -> Self {
        Self {
            pre_standardized_x: false,
            pre_standardized_y: false,
            par: ParChoice::Auto,
            max_iter: 100,
            tol: 1e-8,
        }
    }
}

/// Owned PLS3 fit. Fields use long `snake_case` names; the wrapper
/// translates to short Python-facing names at the FFI seam.
#[derive(Debug, Clone)]
pub struct Pls3Model {
    /// X-side saliences `U`; shape `(n_features, k_used)`. Orthonormal
    /// columns on a dense fit. A sparse fit (`spls3_fit`) normalizes each
    /// column but does not orthogonalize across columns: hard thresholding
    /// restricts each component to its own support, and two supports may
    /// overlap.
    pub u_saliences: Mat<f64>,
    /// Y-side saliences `V`; shape `(n_targets, k_used)`. Same
    /// dense-only orthonormality caveat as `u_saliences`.
    pub v_saliences: Mat<f64>,
    /// Per-component `σ`; shape `(k_used,)`. On a dense fit these are the
    /// singular values of `X̃'Ỹ`, descending. On a sparse fit they are
    /// `σ_a = u_a' A_a v_a` on the deflated `A_a`, which is neither a
    /// singular value of `X̃'Ỹ` nor guaranteed descending, and
    /// `Σσ² / ‖A‖_F²` is not an explained share.
    pub singular_values: Col<f64>,
    /// In-sample X-side LV scores `X̃·U`; shape `(n_samples, k_used)`.
    pub x_scores: Mat<f64>,
    /// In-sample Y-side LV scores `Ỹ·V`; shape `(n_samples, k_used)`.
    pub y_scores: Mat<f64>,
    /// Column means of X used by the fit; zeros when `pre_standardized_x`.
    pub x_mean: Col<f64>,
    /// Column scales of X used by the fit; ones when `pre_standardized_x`.
    pub x_scale: Col<f64>,
    /// Column means of Y used by the fit; zeros when `pre_standardized_y`.
    pub y_mean: Col<f64>,
    /// Column scales of Y used by the fit; ones when `pre_standardized_y`.
    pub y_scale: Col<f64>,
    /// Number of components actually retained (≤ requested `k`).
    pub k_used: usize,
    /// Echoes the caller's `pre_standardized_x` flag.
    pub pre_standardized_x: bool,
    /// Echoes the caller's `pre_standardized_y` flag.
    pub pre_standardized_y: bool,
    /// Resolved X-side keep-count. `None` for dense fits, mirroring
    /// `Pls1Model::keep`.
    pub keep_x: Option<usize>,
    /// Resolved Y-side keep-count. `None` for dense fits.
    pub keep_y: Option<usize>,
    /// Per-component convergence flag for the sparse alternation; length
    /// `k_used`. `None` for dense fits (one SVD, nothing to converge).
    /// `false` means `max_iter` was reached, which is a reported outcome,
    /// not an error.
    pub converged: Option<Vec<bool>>,
    /// Per-component alternation sweep count; length `k_used`. `None` for
    /// dense fits.
    pub n_iter: Option<Vec<usize>>,
}

/// Absolute floor on a retained singular value. Mirrors the `1e-14` norm
/// guard in `fit::pls1_kernel`: below it the component is numerical dust
/// and is dropped rather than reported, so a zero-variance block truncates
/// instead of emitting NaN saliences. The Gram route in
/// `dual_route::pls3_split_zbars_columns` keeps its own formula only where
/// it can prove `σ₁` clears this floor by a wide margin (see "Truncation"
/// there), and recomputes every other column on the primal route.
///
/// Every component, the first included, must also clear
/// [`sigma_rel_floor`], which is what catches the structurally zero
/// singular values of a rank-deficient `X̃'Ỹ` and the rounding-noise `σ₁`
/// of a Y orthogonal to X: those land near `eps·‖X̃‖_F·‖Ỹ‖_F`, which grows
/// with `n`, so they clear this absolute floor.
///
/// `pub(crate)` so the Gram route can name this constant directly rather
/// than restate the literal.
pub(crate) const SIGMA_FLOOR: f64 = 1e-14;

/// Relative floor on every retained `σ`, the first included:
/// `max(n, p, q)·eps·‖X̃‖_F·‖Ỹ‖_F`, where `X̃` (`n × p`) and `Ỹ` (`n × q`)
/// are the blocks `A = X̃'Ỹ` is formed from (standardized, or the caller's
/// own under `pre_standardized_*`).
///
/// It catches the structurally zero singular values of a rank-deficient
/// `A` (`n − 1 < k`; a Y block holding subscales plus their total with
/// `k = q`), which clear [`SIGMA_FLOOR`] because they scale with the data.
/// What they scale with is the rounding in forming `A`, not the SVD: each
/// entry of `A` is a length-`n` dot product whose rounding error is bounded
/// by `n·eps` times the corresponding entry of `|X̃|'|Ỹ|`, and the
/// standardization that produced `X̃` and `Ỹ` perturbs them relatively as
/// well, so the zero singular values land near a small multiple of
/// `eps·‖X̃‖_F·‖Ỹ‖_F` whatever `σ₁` is. The SVD's own backward error,
/// `max(p, q)·eps·σ₁`, is never larger, since `σ₁ ≤ ‖X̃‖_F·‖Ỹ‖_F`.
///
/// `σ₁` itself is therefore no reference. It is only a lower bound on the
/// rounding scale and sits far below it when Y is nearly orthogonal to X
/// (on standardized blocks `‖X̃‖_F·‖Ỹ‖_F = n·√(pq)`, while `σ₁` can be as
/// small as the signal). A floor of `max(n, p, q)·eps·σ₁` would keep the noise
/// components of such designs: with Y built as `δ·XB` plus noise projected
/// off the span of X, and a total-score column, the structurally zero `σ`
/// sits up to 121× above it at `δ = 1e-4` and 7e5× at `δ = 1e-8`, and when Y
/// is exactly orthogonal to X every requested component would be kept.
///
/// Measured on total-score, duplicated-column and `n − 1 < k` designs,
/// crossed with strong, null, `δ` from 1e-2 to 1e-8 and exactly orthogonal
/// signal, `n` from 10 to 2e5, the structurally zero `σ` sat at most at
/// `1.2·eps·‖X̃‖_F·‖Ỹ‖_F`, i.e. at most `0.03×` this floor, while every
/// component carrying more than `1e-9` of `‖X̃‖_F·‖Ỹ‖_F` sat at least `22×`
/// above it (at `n = 2e5`, where the floor is highest). `σ_a / (‖X̃‖_F·‖Ỹ‖_F)`
/// is the component's covariance in units of the largest the two blocks
/// could carry, and a component below `1e-9` of it is orders of magnitude
/// under the sampling noise of any `n` that fits in memory. `max(n, p, q)`
/// keeps the worst-case `n·eps` of the dot products and the `max(p, q)·eps`
/// of the SVD, and matches the factor `fit::pls1_kernel` uses for its own
/// relative floor.
///
/// Applied from the first component on, as `fit::pls1_kernel` applies its
/// own floor: a Y orthogonal to X up to rounding, whose `σ₁` is itself
/// rounding noise, returns `k_used = 0` (the zero model `pls1_fit` returns
/// on the same input at `q = 1`) rather than a noise LV1 pointing nowhere
/// in particular. The Gram route in `dual_route::pls3_split_zbars_columns`
/// cannot resolve `σ₁` to the accuracy this floor needs, so it keeps its
/// own formula only on columns where it can prove the primal fit keeps LV1,
/// and recomputes every other column on the primal route (see
/// "Truncation" there).
///
/// `spls3_fit` applies the same floor, computed from the same blocks: its
/// `A` is this `A`, and each deflation `A ← A − σ u v'` adds rounding of
/// order `eps·σ ≤ eps·‖X̃‖_F·‖Ỹ‖_F`. Its first sparse `σ` is no better a
/// reference than `σ₁` (it is at most `σ₁`): on planted sparse designs with
/// weak signal the first noise component sat up to 2.7e6× above
/// `max(n, p, q)·eps·σ_first` and at most `0.003×` this floor.
///
/// Computed as `fit::w_rel_floor` with `d = max(p, q)`, the same integer
/// and the same multiplication order, so the two floors cannot drift apart.
pub(crate) fn sigma_rel_floor(n: usize, p: usize, q: usize, x_fro: f64, y_fro: f64) -> f64 {
    crate::fit::w_rel_floor(n, p.max(q), x_fro, y_fro)
}

/// Everything `pls3_fit` and `spls3_fit` both need before their paths
/// diverge (one thin SVD vs. a deflating alternation), as `prepare`
/// produces it: the standardize-or-skip decision with the moments it
/// records and the `par` choice. `prepare` also validates and returns the
/// cross-covariance `A` alongside.
///
/// It exists so the sparse path need not obtain those by running a whole
/// dense fit and discarding its SVD, its sign pinning and both of its
/// score matmuls. Both the dense and the sparse fit go through `prepare`,
/// so they form `A` and the scores with the same expressions and produce
/// the same bits.
///
/// # The implicit route
/// A standardizing fit does not write `X̃ = (X − 1·mean')·diag(1/scale)`.
/// It only needs `X̃'Ỹ` and `X̃U`, and forms both from the caller's `x`
/// with a rank-one correction, the algebra of `fit::ImplicitXBackend`:
///
/// - `X̃'Ỹ = (X'Ỹ − mean·(1'Ỹ)) / scale`, row `j` divided by `scale_j`;
/// - `X̃U  = X·Ū − 1·(mean'Ū)`, with `Ū` = `U`, row `j` divided by `scale_j`.
///
/// The corrections cost column `j` about `log10(1 + |mean_j| / scale_j)`
/// digits, so the route is taken only up to [`IMPLICIT_MAX_MEAN_RATIO`],
/// and the rounding it adds to `A` lifts the structurally zero `σ` that
/// [`sigma_rel_floor`] is sized for on a written `X̃`. So the implicit `A`
/// never decides a truncation: its components are tested against
/// [`Prepared::stop_floor`], a band above the floor, and a fit that would
/// keep fewer than `k` of them is refitted on the written copy
/// ([`Prepared::materialize`]), whose `σ` the floor is sized for.
/// An `A` that overflows (entries of `x` near `f64::MAX`) is formed from
/// the written copy as well.
/// With `n − 1 < k` the copy is written from the start: a dense fit there
/// always truncates. A sparse fit takes the copy too, though it need not
/// truncate.
/// `x`'s layout is read as given, so the last bits of a fit on this route
/// depend on it.
struct Prepared {
    /// The written `X̃`: the standardized copy off the implicit route, or
    /// under `pre_standardized_x` a column-major copy of the caller's X
    /// when `linalg::col_major_or_copy` makes one. `None` on the implicit
    /// route, and under `pre_standardized_x` when the caller's matrix is
    /// used as-is. Resolve with [`Prepared::xs`].
    xs_owned: Option<Mat<f64>>,
    /// Standardized Y; `None` under `pre_standardized_y` when the caller's
    /// matrix is used as-is. Resolve with [`Prepared::ys`].
    ys_owned: Option<Mat<f64>>,
    /// `Some(max_j |mean_j| / scale_j)` on the implicit route.
    x_implicit: Option<f64>,
    /// `‖X̃‖_F` from the standardization moments (`linalg::FitX::fro`);
    /// `None` under `pre_standardized_x`, where it is the block's
    /// `norm_l2`.
    x_fro: Option<f64>,
    /// Resolved parallelism for every matmul downstream of here.
    par: Par,
    x_mean: Col<f64>,
    x_scale: Col<f64>,
    y_mean: Col<f64>,
    y_scale: Col<f64>,
}

impl Prepared {
    /// The written `X̃`: the copy, or under `pre_standardized_x` the
    /// caller's `x` when it is column-major. Not for the implicit route,
    /// where no `X̃` is written.
    fn xs<'a>(&'a self, x: MatRef<'a, f64>) -> MatRef<'a, f64> {
        debug_assert!(self.x_implicit.is_none());
        self.xs_owned.as_ref().map_or(x, faer::Mat::as_ref)
    }

    /// `Ỹ`: the standardized copy, or under `pre_standardized_y` the
    /// caller's `y` (as given when column-major, else its column-major
    /// copy).
    fn ys<'a>(&'a self, y: MatRef<'a, f64>) -> MatRef<'a, f64> {
        self.ys_owned.as_ref().map_or(y, faer::Mat::as_ref)
    }

    /// The cross-covariance `A = X̃'Ỹ`, shape `(n_features, n_targets)`.
    /// Never a `p × p` matrix. On the implicit route, the first identity of
    /// "The implicit route" on [`Prepared`].
    fn cross_cov(&self, x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> Mat<f64> {
        let ys = self.ys(y);
        let mut a = Mat::<f64>::zeros(x.ncols(), ys.ncols());
        if self.x_implicit.is_none() {
            let xs = self.xs(x).transpose();
            matmul(a.as_mut(), Accum::Replace, xs, ys, 1.0, self.par);
            return a;
        }
        matmul(a.as_mut(), Accum::Replace, x.transpose(), ys, 1.0, self.par);
        let (mean, scale) = (
            owned_col_slice(&self.x_mean),
            owned_col_slice(&self.x_scale),
        );
        for c in 0..ys.ncols() {
            let sum_y: f64 = ys.col(c).iter().sum();
            for ((v, &m), &s) in a.col_as_slice_mut(c).iter_mut().zip(mean).zip(scale) {
                *v = (*v - m * sum_y) / s;
            }
        }
        a
    }

    /// The X-side scores `X̃·U`, shape `(n_samples, k_used)`. On the
    /// implicit route, the second identity of "The implicit route" on
    /// [`Prepared`].
    fn x_scores(&self, x: MatRef<'_, f64>, u: MatRef<'_, f64>) -> Mat<f64> {
        let mut t = Mat::<f64>::zeros(x.nrows(), u.ncols());
        if self.x_implicit.is_none() {
            matmul(t.as_mut(), Accum::Replace, self.xs(x), u, 1.0, self.par);
            return t;
        }
        let (mean, scale) = (
            owned_col_slice(&self.x_mean),
            owned_col_slice(&self.x_scale),
        );
        let mut u_bar = u.to_owned();
        let mut offset = vec![0.0_f64; u.ncols()];
        for (c, off) in offset.iter_mut().enumerate() {
            for ((v, &m), &s) in u_bar.col_as_slice_mut(c).iter_mut().zip(mean).zip(scale) {
                *v /= s;
                *off += m * *v;
            }
        }
        matmul(t.as_mut(), Accum::Replace, x, u_bar.as_ref(), 1.0, self.par);
        for (c, &off) in offset.iter().enumerate() {
            for v in t.col_as_slice_mut(c) {
                *v -= off;
            }
        }
        t
    }

    /// [`sigma_rel_floor`] on `X̃` and `Ỹ`. `‖X̃‖_F` is the one the
    /// standardization moments give, or under `pre_standardized_x` the
    /// block's `norm_l2`; it only gates truncation.
    fn rel_floor(&self, x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> f64 {
        let x_fro = self.x_fro.unwrap_or_else(|| self.xs(x).norm_l2());
        sigma_rel_floor(x.nrows(), x.ncols(), y.ncols(), x_fro, self.ys(y).norm_l2())
    }

    /// What the component stage tests every `σ` against: [`Self::rel_floor`],
    /// or on the implicit route the band
    /// `2·(1 + max_j |mean_j| / scale_j)·max(SIGMA_FLOOR, rel_floor)`, the
    /// one `fit::ImplicitXBackend` gates its own stop decisions with.
    fn stop_floor(&self, x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> f64 {
        let floor = self.rel_floor(x, y);
        match self.x_implicit {
            Some(ratio) => 2.0 * (1.0 + ratio) * floor.max(SIGMA_FLOOR),
            None => floor,
        }
    }

    /// Whether a component stage that kept `k_used` of `k` components has
    /// to be rerun on the written copy: on the implicit route any
    /// truncation is undecided (see "The implicit route" on [`Prepared`]).
    fn unresolved(&self, k_used: usize, k: usize) -> bool {
        self.x_implicit.is_some() && k_used < k
    }

    /// Leaves the implicit route: writes the standardized copy a fit off
    /// the route has from the start, and returns `A` formed from it.
    fn materialize(&mut self, x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> Mat<f64> {
        self.xs_owned = Some(crate::linalg::standardize(x).0);
        self.x_implicit = None;
        self.cross_cov(x, y)
    }

    /// The model both fits return: the scores `X̃U` and `ỸV`, the recorded
    /// moments, and `opts`' flags echoed.
    /// `sparse` carries `spls3_fit`'s `(keep_x, keep_y, converged,
    /// n_iter)`; `pls3_fit` passes `None`.
    #[allow(clippy::many_single_char_names)]
    fn into_model(
        self,
        x: MatRef<'_, f64>,
        y: MatRef<'_, f64>,
        opts: Pls3FitOpts,
        (u, v, singular_values): (Mat<f64>, Mat<f64>, Col<f64>),
        sparse: Option<SparseMeta>,
    ) -> Pls3Model {
        let n_samples = x.nrows();
        let k_used = u.ncols();
        let x_scores = self.x_scores(x, u.as_ref());
        let mut y_scores = Mat::<f64>::zeros(n_samples, k_used);
        matmul(
            y_scores.as_mut(),
            Accum::Replace,
            self.ys(y),
            v.as_ref(),
            1.0,
            self.par,
        );
        let (keep_x, keep_y, converged, n_iter) = match sparse {
            Some(m) => (
                Some(m.keep_x),
                Some(m.keep_y),
                Some(m.converged),
                Some(m.n_iter),
            ),
            None => (None, None, None, None),
        };
        Pls3Model {
            u_saliences: u,
            v_saliences: v,
            singular_values,
            x_scores,
            y_scores,
            x_mean: self.x_mean,
            x_scale: self.x_scale,
            y_mean: self.y_mean,
            y_scale: self.y_scale,
            k_used,
            pre_standardized_x: opts.pre_standardized_x,
            pre_standardized_y: opts.pre_standardized_y,
            keep_x,
            keep_y,
            converged,
            n_iter,
        }
    }
}

/// `spls3_fit`'s sparse metadata on [`Pls3Model`].
struct SparseMeta {
    keep_x: usize,
    keep_y: usize,
    converged: Vec<bool>,
    n_iter: Vec<usize>,
}

/// Run the shared PLS3 prologue. See [`Prepared`]. Also returns the
/// cross-covariance `A = X̃'Ỹ`, shape `(n_features, n_targets)`, separately
/// so the sparse path can deflate it in place while `Prepared` stays
/// borrowed for the relative floor.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn prepare(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<(Prepared, Mat<f64>)> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_targets = y.ncols();

    if y.nrows() != n_samples {
        return Err(PlsKitError::DimensionMismatch {
            x: (n_samples, n_features),
            y: y.nrows(),
        });
    }
    if weights.is_some() {
        return Err(PlsKitError::InvalidArgument(
            "observation weights are not implemented for pls3_fit; pass weights=None. \
             The weighted PLS3 path row-scales (X, Y) by √w before forming X'Y and is \
             not implemented yet across the PLS2/PLS3 family."
                .into(),
        ));
    }
    check_finite_mat(x)?;
    check_finite_mat(y)?;

    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    let k_max = n_features.min(n_targets);
    if k > k_max {
        return Err(PlsKitError::KExceedsMax { k, k_max });
    }
    // Last, as `pls1_fit`'s `n >= k + 1` check is. PLS3 bounds `k` without
    // `n` (`n - 1 < k` truncates; see `sigma_rel_floor`), so only an empty
    // block, whose standardization moments would be NaN, is refused.
    crate::preprocess::check_has_rows(n_samples)?;

    // Standardize OR skip, recording the moments either way so
    // `pls3_transform` can apply the same transform to new data. A skipped
    // block that is row-major or has a negative column stride is copied
    // column-major, as the standardized copy is: faer's products below (the
    // scores, `‖·‖_F` for the relative floor) sum in the operand's layout
    // order, so reading it as given would move the last bits of the scores
    // and could move `k_used` on the floor. A column-major block is read as
    // given, so its bits are unchanged. A standardized X is written only
    // off the implicit route (see "The implicit route" on `Prepared`); its
    // moments are `linalg::standardize`'s bit for bit.
    let (xs_owned, x_mean, x_scale, x_implicit, x_fro) = if opts.pre_standardized_x {
        (
            crate::linalg::col_major_or_copy(x),
            Col::<f64>::zeros(n_features),
            Col::<f64>::from_fn(n_features, |_| 1.0),
            None,
            None,
        )
    } else {
        let fx = crate::linalg::fit_x_moments(x, None);
        let ratio = fx.max_mean_ratio();
        // Written so that a NaN ratio takes the copy. So does `n - 1 < k`:
        // the dense fit then always truncates, and would be refitted on
        // the copy.
        let implicit = ratio <= IMPLICIT_MAX_MEAN_RATIO && n_samples > k;
        (
            (!implicit).then(|| crate::linalg::standardize(x).0),
            fx.mean,
            fx.scale,
            implicit.then_some(ratio),
            Some(fx.fro),
        )
    };
    let (ys_owned, y_mean, y_scale) = if opts.pre_standardized_y {
        (
            crate::linalg::col_major_or_copy(y),
            Col::<f64>::zeros(n_targets),
            Col::<f64>::from_fn(n_targets, |_| 1.0),
        )
    } else {
        let (ys, m, s) = crate::linalg::standardize(y);
        (Some(ys), m, s)
    };

    // The SVD cost is min(p,q)·p·q; the matmuls are n·p·q. Size the par
    // decision on the matmul term, which dominates whenever n ≥ min(p,q).
    let par = resolve_par(opts.par, n_samples, n_features, n_targets);

    let mut prepared = Prepared {
        xs_owned,
        ys_owned,
        x_implicit,
        x_fro,
        par,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    };
    let mut a = prepared.cross_cov(x, y);
    // Entries of `x` near `f64::MAX` overflow the implicit route's raw
    // `X'Ỹ`; the written copy's products stay finite.
    if prepared.x_implicit.is_some() && !a.is_all_finite() {
        a = prepared.materialize(x, y);
    }
    Ok((prepared, a))
}

/// The leading `k` components of the dense `a`: one thin SVD, truncated at
/// the first `σ` under the floors, signs pinned. Returns `(U, V, σ)` with
/// `k_used` columns.
///
/// Truncating at the first below-floor singular value mirrors the PLS1 kernel's early
/// break rather than emitting components with unidentified directions.
/// Every component, the leading one included, must clear both
/// `SIGMA_FLOOR` and `rel_floor()`. `rel_floor()` is called once, and only
/// when `σ₁` clears `SIGMA_FLOOR` (a zero block never pays for its two
/// norms).
#[allow(clippy::many_single_char_names)]
fn dense_components(
    a: MatRef<'_, f64>,
    k: usize,
    par: Par,
    rel_floor: impl FnOnce() -> f64,
) -> PlsKitResult<(Mat<f64>, Mat<f64>, Col<f64>)> {
    let svd = crate::linalg::thin_svd(a, par)
        .map_err(|_| PlsKitError::Internal("SVD of X'Y failed to converge in pls3_fit".into()))?;
    let sigma = svd.s.as_ref();
    // `1 <= k <= min(p, q)` (checked in `prepare`, or by
    // `pls3_lv1_prestd`'s caller), so `sigma[0]` exists.
    let mut k_used = 0;
    if sigma[0] >= SIGMA_FLOOR {
        let rel_floor = rel_floor();
        while k_used < k && sigma[k_used] >= SIGMA_FLOOR && sigma[k_used] >= rel_floor {
            k_used += 1;
        }
    }
    let mut u = svd.u.subcols(0, k_used).to_owned();
    let mut v = svd.v.subcols(0, k_used).to_owned();
    pin_component_signs(&mut u, &mut v);
    Ok((u, v, Col::<f64>::from_fn(k_used, |i| sigma[i])))
}

/// LV1 saliences `(u₁, v₁)` of already-standardized blocks, or `None` when
/// no first component clears the floors: the first columns of what
/// `pls3_fit(xs, ys, 1, None, opts)` returns, or `spls3_fit(xs, ys, 1,
/// keep_x, keep_y, None, opts)` when either keep-count selects fewer than
/// all columns, with `pre_standardized_x = pre_standardized_y = true` and
/// `par = ParChoice::Seq`. Bit for bit: the same `A`, the same component
/// stage, the same sign pinning.
///
/// For the resampling loops of `pls3_signal_test`, which run this once per
/// `(split, replicate column)` and read nothing else. It skips what those
/// fits do around the component stage: validation the caller has already
/// done once for the whole test (`k`, the keep-counts, `max_iter`, `tol`),
/// and the two score matmuls, as costly as forming `A` itself at small `q`.
/// The finiteness checks stay: they are what turns a non-finite
/// standardized block into the `Err` the callers degrade to `r = 0`. A
/// block with no columns (which the caller's `k_max` check excludes)
/// returns `None` rather than reaching the SVD.
///
/// `xs_fro` must be `xs.norm_l2()`, the X-side input of the relative floor
/// ([`sigma_rel_floor`]). The loops hold `xs` fixed across every replicate
/// column of a split, so they compute it once per split rather than once
/// per fit; the same call on the same matrix gives the same bits, so the
/// floor is the one `pls3_fit` computes.
///
/// # Errors
/// `NonFiniteInput` when either block holds NaN/inf, and `Internal` when
/// an SVD fails to converge, as in the two fits.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pls3_lv1_prestd(
    xs: MatRef<'_, f64>,
    xs_fro: f64,
    ys: MatRef<'_, f64>,
    keep_x: Option<usize>,
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
) -> PlsKitResult<Option<(Col<f64>, Col<f64>)>> {
    check_finite_mat(xs)?;
    check_finite_mat(ys)?;
    let (p, q) = (xs.ncols(), ys.ncols());
    if p == 0 || q == 0 {
        return Ok(None);
    }
    let mut a = Mat::<f64>::zeros(p, q);
    matmul(
        a.as_mut(),
        Accum::Replace,
        xs.transpose(),
        ys,
        1.0,
        Par::Seq,
    );
    let (kx, ky) = (keep_x.unwrap_or(p), keep_y.unwrap_or(q));
    // `Prepared::rel_floor` on these blocks: under `pre_standardized_*` the
    // blocks the fits form `A` from are the caller's, i.e. `xs` and `ys`.
    let rel_floor = || sigma_rel_floor(xs.nrows(), p, q, xs_fro, ys.norm_l2());
    let (u, v) = if kx == p && ky == q {
        let (u, v, _) = dense_components(a.as_ref(), 1, Par::Seq, rel_floor)?;
        (u, v)
    } else {
        let (u, v, _, _) = sparse_components(a, 1, kx, ky, max_iter, tol, Par::Seq, rel_floor)?;
        (u, v)
    };
    if u.ncols() == 0 {
        return Ok(None);
    }
    Ok(Some((u.col(0).to_owned(), v.col(0).to_owned())))
}

/// Fit PLS3 / PLSSVD: SVD of the standardized cross-covariance `X̃'Ỹ`.
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - `y`: `(n_samples, n_targets)`
/// - `k`: `1..=min(n_features, n_targets)`
/// - returns `Pls3Model` with `u_saliences: (n_features, k_used)`,
///   `v_saliences: (n_targets, k_used)`, `singular_values: (k_used,)`,
///   `x_scores` / `y_scores`: `(n_samples, k_used)`.
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when `y.nrows() != x.nrows()`
/// - `PlsKitError::InvalidArgument` when `k == 0`, or when `weights` is `Some`
///   (observation weights are not implemented for this family; see
///   "Observation weights are refused" in `_docs/concepts/PLS3/fit-and-transform.md`)
/// - `PlsKitError::KExceedsMax` when `k > min(n_features, n_targets)`
/// - `PlsKitError::InvalidArgument` when X and Y have zero rows (checked
///   after `k`; one row is not an error, see "Truncation" in
///   `_docs/concepts/PLS3/fit-and-transform.md`)
/// - `PlsKitError::NonFiniteInput` when X or Y contains NaN/inf
/// - `PlsKitError::Internal` when faer's SVD fails to converge
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub fn pls3_fit(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    crate::fit::with_thread_limit(|| pls3_fit_impl(x, y, k, weights, opts))
}

/// Body of [`pls3_fit`], on the caller's pool.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn pls3_fit_impl(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    let (mut prepared, a) = prepare(x, y, k, weights, opts)?;
    let mut components =
        dense_components(a.as_ref(), k, prepared.par, || prepared.stop_floor(x, y))?;
    if prepared.unresolved(components.0.ncols(), k) {
        let a = prepared.materialize(x, y);
        components = dense_components(a.as_ref(), k, prepared.par, || prepared.stop_floor(x, y))?;
    }
    Ok(prepared.into_model(x, y, opts, components, None))
}

/// One sparse component's alternation outcome.
struct SparseComponent {
    u: Col<f64>,
    v: Col<f64>,
    sigma: f64,
    converged: bool,
    n_iter: usize,
}

/// Hard-threshold alternating power iteration for one component of the
/// current (deflated) `a`.
///
/// Returns `None` when the component is numerical dust: either side can
/// collapse to a below-floor norm after selection, which is the sparse
/// analogue of the `w_norm < 1e-14` break in `fit::pls1_kernel`.
///
/// The initializer is the leading singular pair of `a`. faer's SVD gives
/// `a v_0 = σ_0 u_0` with consistent signs, so the first `u = a v` step
/// reproduces `u_0` up to a positive factor and no sign realignment is
/// needed inside the loop. A joint sign flip of `(u, v)` is in any case
/// exactly equivariant here (negation is exact in IEEE-754 and every step
/// is sign-homogeneous), so `pin_component_signs` can settle the sign
/// afterwards without perturbing a single bit.
///
/// Unlike the dense SVD, this alternation has no monotone-convergence
/// guarantee: hard thresholding is not a projection onto a convex set, and
/// the support can enter a period-2 cycle. That is what `converged = false`
/// reports.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn spls3_component(
    a: MatRef<'_, f64>,
    keep_x: usize,
    keep_y: usize,
    max_iter: usize,
    tol: f64,
    par: Par,
) -> PlsKitResult<Option<SparseComponent>> {
    let p = a.nrows();
    let q = a.ncols();

    let svd = crate::linalg::thin_svd(a, par)
        .map_err(|_| PlsKitError::Internal("SVD of the deflated X'Y failed in spls3_fit".into()))?;
    // Only `v` seeds the alternation. `u` is a pure destination: the first
    // thing every sweep does is overwrite all `p` entries of it with `A v`
    // (`Accum::Replace`), and `max_iter >= 1` is enforced by the caller, so
    // the loop always runs at least once and `svd.u` is never read. The
    // buffer still has to exist at the right length for that matmul.
    let mut u: Col<f64> = Col::<f64>::zeros(p);
    let mut v: Col<f64> = svd.v.col(0).to_owned();
    // Pin the seed's sign (largest-magnitude entry positive, ties to the
    // lowest index): faer returns it with opposite signs under `Par::Seq`
    // and `Par::rayon`. The alternation is sign-equivariant, so the two
    // would end on exact negatives; `pin_component_signs` then flips one
    // back, and negating turns its unselected `+0.0` entries into `-0.0`,
    // breaking serial/parallel byte parity.
    let mut best = 0usize;
    for j in 1..q {
        if v[j].abs() > v[best].abs() {
            best = j;
        }
    }
    if q > 0 && v[best] < 0.0 {
        for j in 0..q {
            v[j] = -v[j];
        }
    }
    // One scratch column reused across sweeps instead of a fresh clone per
    // sweep; it is written before it is read on every pass.
    let mut v_prev: Col<f64> = Col::<f64>::zeros(q);

    let mut converged = false;
    let mut n_iter = 0usize;
    for it in 1..=max_iter {
        n_iter = it;
        v_prev.as_mut().copy_from(v.as_ref());

        // u = A v, then select and normalize.
        matmul(
            u.as_mut().as_mat_mut(),
            Accum::Replace,
            a,
            v.as_ref().as_mat(),
            1.0,
            par,
        );
        // `keep == dim` is a LITERAL skip, not a no-op call: that is what
        // makes the dense endpoint's float sequence provably untouched.
        if select_and_normalize(&mut u, keep_x, SIGMA_FLOOR).is_none() {
            return Ok(None);
        }

        // v = A' u, then select and normalize.
        matmul(
            v.as_mut().as_mat_mut(),
            Accum::Replace,
            a.transpose(),
            u.as_ref().as_mat(),
            1.0,
            par,
        );
        if select_and_normalize(&mut v, keep_y, SIGMA_FLOOR).is_none() {
            return Ok(None);
        }

        // Convergence on `v` alone: `u` is a deterministic function of the
        // previous `v`, so this is not the weaker test, and it is the only
        // one `dual_route::pls3_split_zbars_columns` can evaluate (it never
        // forms `u`, which is the point of the Gram route). The two must
        // use the same criterion or they stop on different sweeps and their
        // held-out `r` separates by O(tol), far above the 1e-10 the
        // equivalence test asserts.
        let dv = (0..q)
            .map(|j| (v[j] - v_prev[j]).abs())
            .fold(0.0_f64, f64::max);
        if dv < tol {
            converged = true;
            break;
        }
    }

    // σ = u'Av on the current (deflated) A. Formed through matmul rather
    // than a hand-rolled double loop so the reduction order matches the
    // rest of the crate.
    // Mathematically this equals `nv`, the norm computed at the end of the
    // last sweep (u'Av = u'(A'u)'... = ‖A'u‖ once v = A'u/‖A'u‖). It is
    // recomputed anyway: the three reductions have different shapes and
    // disagree in the last bits, and σ is both an output and the deflation
    // coefficient below, so the two are not interchangeable here.
    let mut av: Col<f64> = Col::<f64>::zeros(p);
    matmul(
        av.as_mut().as_mat_mut(),
        Accum::Replace,
        a,
        v.as_ref().as_mat(),
        1.0,
        par,
    );
    let sigma: f64 = (0..p).map(|i| u[i] * av[i]).sum();

    Ok(Some(SparseComponent {
        u,
        v,
        sigma,
        converged,
        n_iter,
    }))
}

/// The sparse knobs shared by [`spls3_fit`] and
/// `pls3_confirmatory_test`: each keep-count set (`Some`) must be in
/// `1..=dim`, and when either selects fewer than all columns, `max_iter`
/// must be at least `1` and `tol` finite and non-negative.
///
/// `max_iter` and `tol` are checked only off the dense endpoint. There no
/// alternation runs (the fit takes one SVD and reports `n_iter = 0`), so
/// neither is read and `max_iter = 0` is not a contradiction. Off it, zero
/// sweeps would return the unselected leading singular pair and silently
/// break the support-size contract. `tol` is a bound on `‖v − v_prev‖_∞`,
/// so a negative or NaN one has no meaning (either would silently run
/// every sweep and report `converged = false`), and `tol = inf` would stop
/// after one sweep and report `converged = true`. `tol = 0` is allowed:
/// it asks for every sweep.
///
/// # Errors
/// `InvalidArgument` for any of the above.
pub(crate) fn validate_sparse(
    keep_x: Option<usize>,
    keep_y: Option<usize>,
    n_features: usize,
    n_targets: usize,
    max_iter: usize,
    tol: f64,
) -> PlsKitResult<()> {
    if let Some(kx) = keep_x {
        crate::fit::validate_keep_arg(kx, n_features, "keep_x", "n_features")?;
    }
    if let Some(ky) = keep_y {
        crate::fit::validate_keep_arg(ky, n_targets, "keep_y", "n_targets")?;
    }
    let selects =
        keep_x.is_some_and(|kx| kx < n_features) || keep_y.is_some_and(|ky| ky < n_targets);
    if !selects {
        return Ok(());
    }
    if max_iter == 0 {
        return Err(PlsKitError::InvalidArgument(
            "max_iter must be >= 1 when keep_x or keep_y selects fewer than all \
             columns (max_iter=0 would skip selection entirely)"
                .into(),
        ));
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(PlsKitError::InvalidArgument(format!(
            "tol must be finite and >= 0 when keep_x or keep_y selects fewer than \
             all columns (got tol={tol})"
        )));
    }
    Ok(())
}

/// Sparse PLS3 / sparse PLSSVD: keep-count selection on both salience
/// sides. Head of the `spls3_*` family, the PLS3 counterpart of
/// `spls1_fit`.
///
/// `keep_x` bounds the non-zeros per column of `U`, `keep_y` the non-zeros
/// per column of `V`. Both are scalars broadcast to all `k` components
/// (per-component budgets are not supported). The Y-side count is
/// the reason this exists: `keep_y < n_targets` forces each dimension onto
/// a few outcomes, so the outcomes separate into hard groups instead of
/// every dimension loading a little on everything.
///
/// `keep_x == n_features && keep_y == n_targets` delegates to
/// [`pls3_fit`], so the dense endpoint is bit-identical rather than merely
/// close: the alternation is a different float sequence from the SVD and
/// would not reproduce it.
///
/// # What this does NOT inherit from `pls3_fit`
/// - `U` and `V` have normalized but non-orthogonal columns.
/// - `singular_values` holds `σ_a = u_a'A_a v_a` on the deflated `A_a`,
///   not singular values of `X̃'Ỹ`, and they need not descend.
/// - `converged` can be `false` on a component; that is a report, not an
///   error.
/// - `k_used` is decided independently of the dense fit's and may differ
///   from `pls3_fit`'s `k_used` for the same `(x, y, k)`.
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`, `y`: `(n_samples, n_targets)`
/// - `keep_x`: `1..=n_features`, `keep_y`: `1..=n_targets`
/// - returns a [`Pls3Model`] with `keep_x` / `keep_y` / `converged` /
///   `n_iter` populated.
///
/// # Errors
/// Everything [`pls3_fit`] returns, plus `InvalidArgument` when `keep_x`
/// or `keep_y` is `0` or exceeds its dimension, or away from the dense
/// endpoint when `opts.max_iter` is `0` or `opts.tol` is NaN, infinite or
/// negative (at the dense endpoint no alternation runs, so both are simply
/// ignored).
///
/// # Panics
/// Never (all indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub fn spls3_fit(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    keep_x: usize,
    keep_y: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    crate::fit::with_thread_limit(|| spls3_fit_impl(x, y, k, keep_x, keep_y, weights, opts))
}

/// Body of [`spls3_fit`], on the caller's pool.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn spls3_fit_impl(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    keep_x: usize,
    keep_y: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    let n_features = x.ncols();
    let n_targets = y.ncols();
    validate_sparse(
        Some(keep_x),
        Some(keep_y),
        n_features,
        n_targets,
        opts.max_iter,
        opts.tol,
    )?;
    if keep_x == n_features && keep_y == n_targets {
        // Dense endpoint: one SVD, byte for byte. The metadata still says
        // this was a sparse call, the same way `spls1_fit` reports
        // `keep = Some(n_features)` at its own dense endpoint.
        let mut m = pls3_fit_impl(x, y, k, weights, opts)?;
        m.keep_x = Some(keep_x);
        m.keep_y = Some(keep_y);
        m.converged = Some(vec![true; m.k_used]);
        m.n_iter = Some(vec![0; m.k_used]);
        return Ok(m);
    }

    // The prologue `pls3_fit` runs is shared verbatim (see `prepare`):
    // validation, standardization with the moments recorded for
    // `pls3_transform`, the `par` choice and `A`. Everything a dense fit
    // does past that point (the thin SVD, the sign pinning) would be
    // discarded here, so this path never runs it.
    let (mut prepared, a) = prepare(x, y, k, weights, opts)?;
    let stage = |prepared: &Prepared, a: Mat<f64>| {
        sparse_components(
            a,
            k,
            keep_x,
            keep_y,
            opts.max_iter,
            opts.tol,
            prepared.par,
            || prepared.stop_floor(x, y),
        )
    };
    let mut out = stage(&prepared, a)?;
    if prepared.unresolved(out.0.ncols(), k) {
        let a = prepared.materialize(x, y);
        out = stage(&prepared, a)?;
    }
    let (u, v, sigma, (converged, n_iter)) = out;
    let meta = SparseMeta {
        keep_x,
        keep_y,
        converged,
        n_iter,
    };
    Ok(prepared.into_model(x, y, opts, (u, v, sigma), Some(meta)))
}

/// The component loop of [`sparse_components`]: one [`spls3_component`] per
/// component on `a`, deflating it in place, truncated at the first `σ`
/// under the floors. Returns the retained components with the signs
/// [`spls3_component`] gave them (not pinned), and `a` deflated by all of
/// them.
///
/// `rel_floor()` is called at most once, when the first component that
/// clears `SIGMA_FLOOR` is tested; it applies to every component, the first
/// included (see [`sigma_rel_floor`] for why the first sparse `σ` is no
/// reference).
#[allow(clippy::too_many_arguments)]
fn sparse_component_loop(
    mut a: Mat<f64>,
    k: usize,
    keep_x: usize,
    keep_y: usize,
    max_iter: usize,
    tol: f64,
    par: Par,
    rel_floor: impl Fn() -> f64,
) -> PlsKitResult<(Vec<SparseComponent>, Mat<f64>)> {
    // One Vec, not five in lockstep: `SparseComponent` already bundles the
    // five per-component outputs, and unpacking them here only to re-pair
    // them in `sparse_components` is how the five can drift out of step.
    let mut components: Vec<SparseComponent> = Vec::with_capacity(k);
    let mut floor: Option<f64> = None;

    for _ in 0..k {
        let Some(c) = spls3_component(a.as_ref(), keep_x, keep_y, max_iter, tol, par)? else {
            break;
        };
        // Truncate at the first below-floor σ, as the dense fit does.
        // Note the difference in force: dense σ are descending, so the
        // first sub-floor one bounds every later one. Sparse σ are not
        // ordered, so this is a stopping heuristic matched to the dense
        // path's shape, not a proof that nothing further exists.
        //
        // Defensive, and deliberately untested: σ = u'Av equals the last
        // sweep's `nv` in exact arithmetic, and `nv < SIGMA_FLOOR` has
        // already taken the dust exit in `spls3_component`, so this fires
        // only when σ rounds under the floor while `nv` sits on it.
        if c.sigma < SIGMA_FLOOR {
            break;
        }
        // The dense fit's relative floor, from the same blocks, applied to
        // every component as the dense fit applies it.
        if c.sigma < *floor.get_or_insert_with(&rel_floor) {
            break;
        }
        // A <- A - σ u v'
        matmul(
            a.as_mut(),
            Accum::Add,
            c.u.as_ref().as_mat(),
            c.v.as_ref().as_mat().transpose(),
            -c.sigma,
            par,
        );
        components.push(c);
    }
    Ok((components, a))
}

/// The leading `k` sparse components of `a`: [`sparse_component_loop`]'s
/// components packed into matrices, signs pinned. Returns
/// `(U, V, σ, (converged, n_iter))` with `k_used` columns / entries.
/// `rel_floor` is passed on to the loop, which calls it at most once.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::many_single_char_names)]
fn sparse_components(
    a: Mat<f64>,
    k: usize,
    keep_x: usize,
    keep_y: usize,
    max_iter: usize,
    tol: f64,
    par: Par,
    rel_floor: impl Fn() -> f64,
) -> PlsKitResult<(Mat<f64>, Mat<f64>, Col<f64>, (Vec<bool>, Vec<usize>))> {
    let (p, q) = (a.nrows(), a.ncols());
    let (components, _) =
        sparse_component_loop(a, k, keep_x, keep_y, max_iter, tol, par, rel_floor)?;

    let k_used = components.len();
    let mut u = Mat::<f64>::zeros(p, k_used);
    let mut v = Mat::<f64>::zeros(q, k_used);
    for (a_idx, c) in components.iter().enumerate() {
        u.col_mut(a_idx).copy_from(&c.u);
        v.col_mut(a_idx).copy_from(&c.v);
    }
    pin_component_signs(&mut u, &mut v);
    Ok((
        u,
        v,
        Col::<f64>::from_fn(k_used, |i| components[i].sigma),
        (
            components.iter().map(|c| c.converged).collect(),
            components.iter().map(|c| c.n_iter).collect(),
        ),
    ))
}

/// Pin the SVD sign convention: flip `(u_i, v_i)` together until the
/// largest-magnitude entry of `u_i` is positive, ties broken by lowest row
/// index. faer promises no sign, and the reference corpus stores the
/// saliences verbatim, so without this the fixtures would be
/// platform-dependent. A *joint* flip is what keeps it free: it leaves
/// `u_i σ_i v_i'`, and therefore `X'Y`, exactly where it was, and it negates
/// both held-out score vectors at once, so the held-out LV correlation is
/// unchanged and the confirmatory test needs no alignment step (see
/// `_docs/concepts/PLS3/fit-and-transform.md` and
/// `_docs/concepts/PLS3/inference.md`).
fn pin_component_signs(u: &mut Mat<f64>, v: &mut Mat<f64>) {
    for a in 0..u.ncols() {
        let mut best = 0usize;
        for i in 1..u.nrows() {
            if u[(i, a)].abs() > u[(best, a)].abs() {
                best = i;
            }
        }
        if u.nrows() > 0 && u[(best, a)] < 0.0 {
            for i in 0..u.nrows() {
                u[(i, a)] = -u[(i, a)];
            }
            for i in 0..v.nrows() {
                v[(i, a)] = -v[(i, a)];
            }
        }
    }
}

/// Alias for [`pls3_fit`] under the SVD-PLS name the chemometrics
/// literature uses. Same function, same arguments, same result: documented
/// as an alias in `_docs/python/api.md` §2c.1, mirroring how `spls1_fit` delegates to
/// `pls1_fit` in `fit.rs`.
///
/// # Errors
/// Everything [`pls3_fit`] returns.
pub fn plssvd_fit(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    crate::fit::with_thread_limit(|| pls3_fit_impl(x, y, k, weights, opts))
}

/// Which score block [`pls3_transform`] should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformWhich {
    /// Project new X only.
    XScores,
    /// Project new Y only.
    YScores,
    /// Project both blocks.
    Both,
}

impl TransformWhich {
    /// Public string identifier (`snake_case`) used in wrapper APIs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TransformWhich::XScores => "x_scores",
            TransformWhich::YScores => "y_scores",
            TransformWhich::Both => "both",
        }
    }

    fn needs_x(self) -> bool {
        matches!(self, TransformWhich::XScores | TransformWhich::Both)
    }

    fn needs_y(self) -> bool {
        matches!(self, TransformWhich::YScores | TransformWhich::Both)
    }
}

/// LV scores for new data. A block is `None` exactly when `which` did not
/// ask for it.
#[derive(Debug, Clone)]
pub struct Pls3Scores {
    /// `X_new` projected on the X-side saliences; shape `(n_new, k_used)`.
    pub x_scores: Option<Mat<f64>>,
    /// `Y_new` projected on the Y-side saliences; shape `(n_new, k_used)`.
    pub y_scores: Option<Mat<f64>>,
}

/// Project new data onto a fitted PLS3's LV scores.
///
/// New data is standardized with the *fit's* moments — the same
/// train-moments-on-test-data rule `split_half_correlations` uses in
/// `signal_test.rs`. On a `pre_standardized_*` fit those moments are the
/// identity, so the caller must apply its own transform first, exactly as
/// `pls1_predict` requires.
///
/// PLS3 is symmetric, so there is no regression `predict` here: neither
/// block is the outcome (see "No `predict`, by design" in `_docs/concepts/PLS3/index.md`).
///
/// # Shapes
/// - `x_new`: `(n_new, n_features)` — `n_features` must equal `model.u_saliences.nrows()`
/// - `y_new`: `(n_new, n_targets)` — `n_targets` must equal `model.v_saliences.nrows()`
/// - returns scores of shape `(n_new, model.k_used)` per requested block
///
/// # Errors
/// - `PlsKitError::InvalidArgument` when `which` asks for a block that was not supplied
/// - `PlsKitError::ShapeMismatch` on a column-count mismatch against the model, or
///   on a model whose saliences, `k_used` and moments disagree with each other
/// - `PlsKitError::NonFiniteInput` when the supplied data contains NaN/inf
///
/// # Panics
/// Never (shapes validated at entry).
pub fn pls3_transform(
    model: &Pls3Model,
    x_new: Option<MatRef<'_, f64>>,
    y_new: Option<MatRef<'_, f64>>,
    which: TransformWhich,
) -> PlsKitResult<Pls3Scores> {
    // Par::Seq: transform is reachable from inside resampler Rayon workers,
    // so it must not dispatch to the global pool (mirrors pls1_predict).
    let par = Par::Seq;

    let x_scores = if which.needs_x() {
        let xn = x_new.ok_or_else(|| {
            PlsKitError::InvalidArgument(format!(
                "which='{}' requires x_new, which was not supplied",
                which.as_str()
            ))
        })?;
        let p = model.u_saliences.nrows();
        if xn.ncols() != p {
            // ShapeMismatch, not DimensionMismatch: the latter is documented
            // and formatted as "X has shape {x}, y has length {y}" for a
            // *row*-count disagreement between X and y, which would print
            // nonsense here ("y has length 7" for a 9-column X).
            return Err(PlsKitError::ShapeMismatch(format!(
                "x_new has {} columns, model was fit on {p}",
                xn.ncols()
            )));
        }
        // A hand-built model (the Python surface exposes every field
        // separately) can carry a k_used, a salience width and moments that
        // disagree. matmul and standardize_apply would panic on the
        // mismatch instead of reporting it.
        if model.u_saliences.ncols() != model.k_used {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model u_saliences has {} columns, k_used is {}",
                model.u_saliences.ncols(),
                model.k_used
            )));
        }
        if model.x_mean.nrows() != p || model.x_scale.nrows() != p {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model x_mean/x_scale have lengths {}/{}, expected {p}",
                model.x_mean.nrows(),
                model.x_scale.nrows()
            )));
        }
        check_finite_mat(xn)?;
        let xs =
            crate::linalg::standardize_apply(xn, model.x_mean.as_ref(), model.x_scale.as_ref());
        let mut t = Mat::<f64>::zeros(xn.nrows(), model.k_used);
        matmul(
            t.as_mut(),
            Accum::Replace,
            xs.as_ref(),
            model.u_saliences.as_ref(),
            1.0,
            par,
        );
        Some(t)
    } else {
        None
    };

    let y_scores = if which.needs_y() {
        let yn = y_new.ok_or_else(|| {
            PlsKitError::InvalidArgument(format!(
                "which='{}' requires y_new, which was not supplied",
                which.as_str()
            ))
        })?;
        let q = model.v_saliences.nrows();
        if yn.ncols() != q {
            // ShapeMismatch — see the x_new branch.
            return Err(PlsKitError::ShapeMismatch(format!(
                "y_new has {} columns, model was fit on {q}",
                yn.ncols()
            )));
        }
        // Model self-consistency — see the x_new branch.
        if model.v_saliences.ncols() != model.k_used {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model v_saliences has {} columns, k_used is {}",
                model.v_saliences.ncols(),
                model.k_used
            )));
        }
        if model.y_mean.nrows() != q || model.y_scale.nrows() != q {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model y_mean/y_scale have lengths {}/{}, expected {q}",
                model.y_mean.nrows(),
                model.y_scale.nrows()
            )));
        }
        check_finite_mat(yn)?;
        let ys =
            crate::linalg::standardize_apply(yn, model.y_mean.as_ref(), model.y_scale.as_ref());
        let mut t = Mat::<f64>::zeros(yn.nrows(), model.k_used);
        matmul(
            t.as_mut(),
            Accum::Replace,
            ys.as_ref(),
            model.v_saliences.as_ref(),
            1.0,
            par,
        );
        Some(t)
    } else {
        None
    };

    Ok(Pls3Scores { x_scores, y_scores })
}

/// Alias for [`pls3_transform`] under the SVD-PLS name (`_docs/python/api.md` §2c.2).
///
/// # Errors
/// Everything [`pls3_transform`] returns.
pub fn plssvd_transform(
    model: &Pls3Model,
    x_new: Option<MatRef<'_, f64>>,
    y_new: Option<MatRef<'_, f64>>,
    which: TransformWhich,
) -> PlsKitResult<Pls3Scores> {
    pls3_transform(model, x_new, y_new, which)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests {
    use super::*;
    use crate::test_support::project_off;
    use approx::assert_relative_eq;

    /// (X, Y) sharing one latent factor: X's first two columns and Y's
    /// first column all load on `f`, so LV1 is a strong, well-separated
    /// singular triplet and the tests below are not measuring noise.
    #[allow(clippy::many_single_char_names)]
    fn shared_factor_data(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let f: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x = Mat::<f64>::from_fn(n, p, |i, j| {
            let noise = rng.random_range(-0.3..0.3);
            if j < 2 {
                f[i] + noise
            } else {
                noise
            }
        });
        let y = Mat::<f64>::from_fn(n, q, |i, j| {
            let noise = rng.random_range(-0.3..0.3);
            if j == 0 {
                f[i] + noise
            } else {
                noise
            }
        });
        (x, y)
    }

    #[test]
    fn pls3_fit_opts_default_has_iteration_caps() {
        let o = Pls3FitOpts::default();
        assert_eq!(o.max_iter, 100);
        assert!((o.tol - 1e-8).abs() < f64::EPSILON);
    }

    #[test]
    fn saliences_are_orthonormal_and_sigmas_descending() {
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap();
        for a in 0..m.k_used {
            let norm: f64 = (0..8).map(|j| m.u_saliences[(j, a)].powi(2)).sum::<f64>();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
            let norm_v: f64 = (0..4).map(|j| m.v_saliences[(j, a)].powi(2)).sum::<f64>();
            assert_relative_eq!(norm_v, 1.0, epsilon = 1e-12);
        }
        let cross: f64 = (0..8)
            .map(|j| m.u_saliences[(j, 0)] * m.u_saliences[(j, 1)])
            .sum();
        assert!(cross.abs() < 1e-10, "u1 ⟂ u2 violated: {cross}");
        for a in 1..m.k_used {
            assert!(m.singular_values[a - 1] >= m.singular_values[a]);
        }
    }

    #[test]
    fn singular_triplet_reproduces_cross_covariance_action() {
        // A v₁ = σ₁ u₁ is the identity the dual (Gram) route rests on;
        // assert it here so the fit owns the invariant.
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 1, None, Pls3FitOpts::default()).unwrap();
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize(y.as_ref());
        let a: Mat<f64> = xs.transpose() * ys.as_ref();
        let av: Col<f64> = a.as_ref() * m.v_saliences.col(0);
        for j in 0..8 {
            assert_relative_eq!(
                av[j],
                m.singular_values[0] * m.u_saliences[(j, 0)],
                epsilon = 1e-9
            );
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn signs_are_pinned_to_largest_u_entry_positive() {
        // The convention: the largest-|.| entry of each u column is positive.
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap();
        for a in 0..m.k_used {
            let piv = sign_pivot(m.u_saliences.col(a));
            assert!(
                m.u_saliences[(piv, a)] > 0.0,
                "component {a}: largest-|.| u entry is negative"
            );
        }
        // A sign check alone is vacuous when every raw component already
        // comes out positive, so compare each fit with the raw SVD of the
        // same `A`: the pinned pair must be the raw pair jointly negated
        // exactly when the raw pivot is negative, and the sweep must contain
        // such a component. Negating Y negates A, so each (Y, -Y) pair
        // should see one of the two raw signs.
        let mut n_flipped = 0usize;
        for seed in [2u64, 8, 11, 13, 25] {
            let (x, y) = shared_factor_data(60, 8, 4, seed);
            let y_neg = Mat::<f64>::from_fn(y.nrows(), y.ncols(), |i, j| -y[(i, j)]);
            for (label, yv) in [("Y", &y), ("-Y", &y_neg)] {
                let opts = Pls3FitOpts::default();
                let m = pls3_fit(x.as_ref(), yv.as_ref(), 3, None, opts).unwrap();
                let (Prepared { par, .. }, a) =
                    prepare(x.as_ref(), yv.as_ref(), 3, None, opts).unwrap();
                let raw = crate::linalg::thin_svd(a.as_ref(), par).unwrap();
                for c in 0..m.k_used {
                    let flip = raw.u[(sign_pivot(raw.u.col(c)), c)] < 0.0;
                    n_flipped += usize::from(flip);
                    let s = if flip { -1.0 } else { 1.0 };
                    for i in 0..8 {
                        assert_eq!(
                            m.u_saliences[(i, c)].to_bits(),
                            (s * raw.u[(i, c)]).to_bits(),
                            "seed {seed} {label} component {c}: u[{i}] (flip={flip})"
                        );
                    }
                    for j in 0..4 {
                        assert_eq!(
                            m.v_saliences[(j, c)].to_bits(),
                            (s * raw.v[(j, c)]).to_bits(),
                            "seed {seed} {label} component {c}: v[{j}] not flipped jointly"
                        );
                    }
                }
            }
        }
        assert!(
            n_flipped > 0,
            "no raw component needed a flip: the sign check would be vacuous"
        );
    }

    #[test]
    fn pre_standardized_path_matches_manual_standardization() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize(y.as_ref());
        let a = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let b = pls3_fit(
            xs.as_ref(),
            ys.as_ref(),
            2,
            None,
            Pls3FitOpts {
                pre_standardized_x: true,
                pre_standardized_y: true,
                ..Pls3FitOpts::default()
            },
        )
        .unwrap();
        for j in 0..6 {
            assert_relative_eq!(
                a.u_saliences[(j, 0)],
                b.u_saliences[(j, 0)],
                epsilon = 1e-12
            );
        }
        assert_relative_eq!(a.singular_values[0], b.singular_values[0], epsilon = 1e-10);
        // Pre-standardized fits report identity moments, matching pls1_fit.
        for j in 0..6 {
            assert_relative_eq!(b.x_mean[j], 0.0, epsilon = 1e-15);
            assert_relative_eq!(b.x_scale[j], 1.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn weights_are_rejected() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let w = Col::<f64>::from_fn(40, |_| 1.0);
        let r = pls3_fit(
            x.as_ref(),
            y.as_ref(),
            1,
            Some(w.as_ref()),
            Pls3FitOpts::default(),
        );
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn k_above_min_p_q_errors() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let r = pls3_fit(x.as_ref(), y.as_ref(), 4, None, Pls3FitOpts::default());
        assert!(matches!(
            r,
            Err(PlsKitError::KExceedsMax { k: 4, k_max: 3 })
        ));
    }

    #[test]
    fn row_count_mismatch_errors() {
        let x = Mat::<f64>::zeros(10, 4);
        let y = Mat::<f64>::zeros(9, 2);
        let r = pls3_fit(x.as_ref(), y.as_ref(), 1, None, Pls3FitOpts::default());
        assert!(matches!(r, Err(PlsKitError::DimensionMismatch { .. })));
    }

    #[test]
    fn non_finite_input_errors() {
        let (mut x, y) = shared_factor_data(20, 4, 2, 3);
        x[(0, 0)] = f64::NAN;
        let r = pls3_fit(x.as_ref(), y.as_ref(), 1, None, Pls3FitOpts::default());
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)));
    }

    #[test]
    fn constant_columns_truncate_to_zero_components() {
        // Every X column constant ⇒ standardize's zero-variance branch zeroes
        // X̃ ⇒ X̃'Ỹ is the zero matrix ⇒ σ₁ = 0 ⇒ nothing is retained.
        let x = Mat::<f64>::from_fn(30, 4, |_, _| 2.5);
        let (_, y) = shared_factor_data(30, 4, 3, 9);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 0);
        assert_eq!(m.u_saliences.ncols(), 0);
    }

    #[test]
    fn transform_on_training_data_reproduces_in_sample_scores() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let s =
            pls3_transform(&m, Some(x.as_ref()), Some(y.as_ref()), TransformWhich::Both).unwrap();
        let xs = s.x_scores.unwrap();
        let ys = s.y_scores.unwrap();
        for a in 0..2 {
            for i in 0..50 {
                assert_relative_eq!(xs[(i, a)], m.x_scores[(i, a)], epsilon = 1e-10);
                assert_relative_eq!(ys[(i, a)], m.y_scores[(i, a)], epsilon = 1e-10);
            }
        }
        let s = pls3_transform(&m, Some(x.as_ref()), None, TransformWhich::XScores).unwrap();
        assert!(s.x_scores.is_some());
        assert!(s.y_scores.is_none());
    }

    #[test]
    fn transform_missing_required_block_errors() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let r = pls3_transform(&m, None, Some(y.as_ref()), TransformWhich::Both);
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
        let r = pls3_transform(&m, None, None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn transform_column_count_mismatch_errors() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let bad = Mat::<f64>::zeros(5, 9);
        let r = pls3_transform(&m, Some(bad.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))));
    }

    #[test]
    fn transform_inconsistent_model_errors() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();

        let mut bad = m.clone();
        bad.x_mean = Col::<f64>::zeros(3);
        let r = pls3_transform(&bad, Some(x.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))));

        let mut bad = m.clone();
        bad.k_used = 5;
        let r = pls3_transform(&bad, Some(x.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))));

        // The Y side has its own guard. `y` has 3 columns and the fit 2
        // components.
        let y_err = |bad: &Pls3Model| match pls3_transform(
            bad,
            None,
            Some(y.as_ref()),
            TransformWhich::YScores,
        ) {
            Err(PlsKitError::ShapeMismatch(msg)) => msg,
            other => panic!("{other:?}"),
        };
        let mut bad = m.clone();
        bad.v_saliences = Mat::<f64>::zeros(3, 1);
        assert!(y_err(&bad).contains("v_saliences"), "{}", y_err(&bad));
        let mut bad = m.clone();
        bad.y_mean = Col::<f64>::zeros(2);
        assert!(y_err(&bad).contains("y_mean/y_scale"), "{}", y_err(&bad));
        let mut bad = m;
        bad.y_scale = Col::<f64>::zeros(2);
        assert!(y_err(&bad).contains("y_mean/y_scale"), "{}", y_err(&bad));
    }

    #[test]
    fn transform_rejects_non_finite_new_data() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let mut bad = Mat::<f64>::zeros(5, 7);
        bad[(0, 0)] = f64::INFINITY;
        let r = pls3_transform(&m, Some(bad.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)));
    }

    /// Two latent directions, each driving a disjoint pair of outcomes:
    /// f1 -> (Y0, Y1) and f2 -> (Y2, Y3), with f1 loading X columns 0..3 and
    /// f2 loading X columns 4..7. With `keep_y` = 2 and k = 2, the supports of
    /// V must come out as exactly those two pairs.
    fn planted_two_block(n: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let f1: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let f2: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut noise = |s: f64| s * rng.random_range(-1.0..1.0);
        let x = Mat::<f64>::from_fn(n, 8, |i, j| {
            let base = if j < 4 { f1[i] } else { f2[i] };
            base + noise(0.10)
        });
        let y = Mat::<f64>::from_fn(n, 4, |i, j| {
            let base = if j < 2 { f1[i] } else { f2[i] };
            base + noise(0.10)
        });
        (x, y)
    }

    #[test]
    fn spls3_dense_endpoint_is_bit_identical_to_pls3() {
        let (x, y) = shared_factor_data(50, 10, 4, 11);
        let dense = pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap();
        assert_eq!(dense.k_used, 3);
        assert!(
            dense.keep_x.is_none()
                && dense.keep_y.is_none()
                && dense.converged.is_none()
                && dense.n_iter.is_none(),
            "a dense fit carries no sparse metadata"
        );
        let sparse = spls3_fit(
            x.as_ref(),
            y.as_ref(),
            3,
            10, // keep_x == n_features
            4,  // keep_y == n_targets
            None,
            Pls3FitOpts::default(),
        )
        .unwrap();
        assert_eq!(sparse.k_used, dense.k_used);
        for a in 0..dense.k_used {
            assert_eq!(
                sparse.singular_values[a].to_bits(),
                dense.singular_values[a].to_bits()
            );
            for i in 0..dense.u_saliences.nrows() {
                assert_eq!(
                    sparse.u_saliences[(i, a)].to_bits(),
                    dense.u_saliences[(i, a)].to_bits()
                );
            }
            for i in 0..dense.v_saliences.nrows() {
                assert_eq!(
                    sparse.v_saliences[(i, a)].to_bits(),
                    dense.v_saliences[(i, a)].to_bits()
                );
            }
        }
        // The metadata does differ: this was a sparse call at the dense endpoint.
        assert_eq!(sparse.keep_x, Some(10));
        assert_eq!(sparse.keep_y, Some(4));
        assert_eq!(sparse.n_iter, Some(vec![0; dense.k_used]));
        assert_eq!(sparse.converged, Some(vec![true; dense.k_used]));
    }

    #[test]
    #[allow(clippy::float_cmp)] // asserting exact zeros from hard_select_keep
    fn spls3_support_sizes_are_exact() {
        let (x, y) = shared_factor_data(50, 10, 4, 13);
        let m = spls3_fit(
            x.as_ref(),
            y.as_ref(),
            2,
            3,
            2,
            None,
            Pls3FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.k_used, 2);
        assert_eq!(m.converged, Some(vec![true; m.k_used]));
        // Regression lock on the exact stopping sweep, not a round number:
        // The Gram (dual) route in dual_route.rs must halt on the same sweep as this primal
        // one or the two held-out statistics separate by O(tol). Observed
        // values, both far below the default max_iter = 100.
        assert_eq!(m.n_iter, Some(vec![4, 9]));
        for a in 0..m.k_used {
            let nz_u = (0..m.u_saliences.nrows())
                .filter(|&i| m.u_saliences[(i, a)] != 0.0)
                .count();
            let nz_v = (0..m.v_saliences.nrows())
                .filter(|&i| m.v_saliences[(i, a)] != 0.0)
                .count();
            assert_eq!(nz_u, 3, "component {a} u support");
            assert_eq!(nz_v, 2, "component {a} v support");
        }
    }

    #[test]
    fn spls3_pins_signs_on_raw_negative_components() {
        // The sparse path must pin signs the way the dense one does:
        // `pin_component_signs` flips `(u_a, v_a)` jointly until the
        // largest-magnitude entry of `u_a` is positive, ties broken toward
        // the lowest row index. Nothing else in the sparse suite would
        // notice if that call were dropped from `spls3_fit`: the support
        // tests are sign-blind, the planted-pairs test is sign-blind, the
        // dense endpoint delegates to `pls3_fit` (which pins separately),
        // and byte parity compares one binary against itself. faer promises
        // no sign, so this check is what keeps a sparse fit reproducible
        // across platforms.
        //
        // A sign check on its own goes vacuous whenever every component
        // happens to come out positive unpinned (seed 13 alone does). So
        // each fit is compared against its own unpinned components from
        // `spls3_unpinned`: the fitted pair must be the raw pair jointly
        // negated exactly when the raw pivot is negative, and the sweep
        // must contain at least one such component. Every seed is fitted
        // against both Y and -Y: negating Y negates A, and with faer's
        // current SVD that flips the raw sign of every component, so one
        // of each pair needs a flip; the count below fails loudly if that
        // stops being true rather than letting the check pass on nothing.
        let mut n_flipped = 0usize;
        for seed in [2u64, 8, 13, 25] {
            let (xs, ys) = shared_factor_data(50, 10, 4, seed);
            let ys_neg = Mat::<f64>::from_fn(ys.nrows(), ys.ncols(), |i, j| -ys[(i, j)]);
            for (label, yv) in [("Y", &ys), ("-Y", &ys_neg)] {
                let opts = Pls3FitOpts::default();
                let ms = spls3_fit(xs.as_ref(), yv.as_ref(), 2, 3, 2, None, opts).unwrap();
                let (raw, _, _) = spls3_unpinned(xs.as_ref(), yv.as_ref(), 2, 3, 2, opts);
                assert_eq!(ms.k_used, 2, "seed {seed} {label}");
                assert_eq!(raw.len(), ms.k_used, "seed {seed} {label}");
                for (a, c) in raw.iter().enumerate() {
                    let piv = sign_pivot(c.u.as_ref());
                    let flip = c.u[piv] < 0.0;
                    n_flipped += usize::from(flip);
                    let s = if flip { -1.0 } else { 1.0 };
                    assert!(
                        ms.u_saliences[(piv, a)] > 0.0,
                        "seed {seed} {label} component {a}: largest-|u| entry sits at \
                         row {piv} and is {}, expected positive",
                        ms.u_saliences[(piv, a)]
                    );
                    for i in 0..c.u.nrows() {
                        assert_eq!(
                            ms.u_saliences[(i, a)].to_bits(),
                            (s * c.u[i]).to_bits(),
                            "seed {seed} {label} component {a}: u[{i}] is not the \
                             raw entry {}",
                            if flip { "negated" } else { "kept" }
                        );
                    }
                    for i in 0..c.v.nrows() {
                        assert_eq!(
                            ms.v_saliences[(i, a)].to_bits(),
                            (s * c.v[i]).to_bits(),
                            "seed {seed} {label} component {a}: v[{i}] not flipped \
                             jointly with u"
                        );
                    }
                }
            }
        }
        assert!(
            n_flipped > 0,
            "no swept component had a negative unpinned pivot, so the sign \
             assertions above are vacuous; add an input that needs a flip"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn spls3_recovers_planted_disjoint_outcome_pairs() {
        let (x, y) = planted_two_block(400, 17);
        let m = spls3_fit(
            x.as_ref(),
            y.as_ref(),
            2,
            4,
            2,
            None,
            Pls3FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.k_used, 2);
        let support = |a: usize| -> Vec<usize> {
            (0..m.v_saliences.nrows())
                .filter(|&i| m.v_saliences[(i, a)] != 0.0)
                .collect()
        };
        let mut supports = vec![support(0), support(1)];
        supports.sort();
        assert_eq!(supports, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn spls3_rejects_zero_max_iter() {
        let (x, y) = shared_factor_data(50, 10, 4, 29);
        let e = spls3_fit(
            x.as_ref(),
            y.as_ref(),
            1,
            3,
            2,
            None,
            Pls3FitOpts {
                max_iter: 0,
                ..Pls3FitOpts::default()
            },
        )
        .unwrap_err();
        assert!(matches!(e, PlsKitError::InvalidArgument(_)));

        // ... but the dense endpoint runs no sweeps at all, so `max_iter =
        // 0` is not a contradiction there and must not be refused.
        let m = spls3_fit(
            x.as_ref(),
            y.as_ref(),
            1,
            10, // keep_x == n_features
            4,  // keep_y == n_targets
            None,
            Pls3FitOpts {
                max_iter: 0,
                ..Pls3FitOpts::default()
            },
        )
        .unwrap();
        assert_eq!(m.n_iter, Some(vec![0; m.k_used]));
    }

    #[test]
    fn spls3_rejects_non_finite_or_negative_tol() {
        let (x, y) = shared_factor_data(50, 10, 4, 31);
        let fit = |keep_x: usize, keep_y: usize, tol: f64| {
            spls3_fit(
                x.as_ref(),
                y.as_ref(),
                1,
                keep_x,
                keep_y,
                None,
                Pls3FitOpts {
                    tol,
                    ..Pls3FitOpts::default()
                },
            )
        };
        for tol in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-8] {
            assert!(
                matches!(fit(3, 2, tol), Err(PlsKitError::InvalidArgument(_))),
                "tol={tol} must be refused off the dense endpoint"
            );
            // The dense endpoint never reads `tol`.
            assert!(fit(10, 4, tol).is_ok(), "tol={tol} at the dense endpoint");
        }
        // `tol = 0` is a legitimate "run every sweep" request.
        assert!(fit(3, 2, 0.0).is_ok());
    }

    #[test]
    fn spls3_rejects_out_of_range_keeps() {
        let (x, y) = shared_factor_data(50, 10, 4, 19);
        let o = Pls3FitOpts::default();
        for (kx, ky) in [(11, 4), (10, 5)] {
            let e = spls3_fit(x.as_ref(), y.as_ref(), 2, kx, ky, None, o).unwrap_err();
            assert!(
                matches!(e, PlsKitError::InvalidArgument(_)),
                "keep_x={kx} keep_y={ky} gave {e:?}"
            );
        }
    }

    #[test]
    fn spls3_reports_non_convergence_without_erroring() {
        let (x, y) = shared_factor_data(50, 10, 4, 23);
        let m = spls3_fit(
            x.as_ref(),
            y.as_ref(),
            1,
            3,
            2,
            None,
            Pls3FitOpts {
                max_iter: 1,
                ..Pls3FitOpts::default()
            },
        )
        .unwrap();
        assert_eq!(m.n_iter, Some(vec![1]));
        assert_eq!(m.converged, Some(vec![false]));
    }

    /// `spls3_fit`'s component loop with the final `pin_component_signs`
    /// call left out. Returns the retained components exactly as
    /// `spls3_component` produced them (raw signs) and `A` deflated by all
    /// of them, so a test can (i) read the unpinned sign of each component
    /// and (ii) call `spls3_component` once more on the returned `A` to see
    /// which of the two truncation exits stopped the loop.
    #[allow(clippy::many_single_char_names)]
    fn spls3_unpinned(
        x: MatRef<'_, f64>,
        y: MatRef<'_, f64>,
        k: usize,
        keep_x: usize,
        keep_y: usize,
        opts: Pls3FitOpts,
    ) -> (Vec<SparseComponent>, Mat<f64>, Par) {
        let (mut prepared, a) = prepare(x, y, k, None, opts).unwrap();
        let par = prepared.par;
        let stage = |prepared: &Prepared, a: Mat<f64>| {
            sparse_component_loop(a, k, keep_x, keep_y, opts.max_iter, opts.tol, par, || {
                prepared.stop_floor(x, y)
            })
            .unwrap()
        };
        let mut out = stage(&prepared, a);
        if prepared.unresolved(out.0.len(), k) {
            let a = prepared.materialize(x, y);
            out = stage(&prepared, a);
        }
        (out.0, out.1, par)
    }

    /// `sigma_rel_floor` exactly as `pls3_fit` / `spls3_fit` evaluate it.
    fn prepared_rel_floor(x: MatRef<'_, f64>, y: MatRef<'_, f64>, opts: Pls3FitOpts) -> f64 {
        let (prepared, _) = prepare(x, y, 1, None, opts).unwrap();
        prepared.rel_floor(x, y)
    }

    /// `A = X̃'Ỹ` from the written `X̃`, which is the `A` a truncating fit
    /// decides on (see "The implicit route" on `Prepared`).
    fn written_cross_cov(x: MatRef<'_, f64>, y: MatRef<'_, f64>, opts: Pls3FitOpts) -> Mat<f64> {
        let (mut prepared, a) = prepare(x, y, 1, None, opts).unwrap();
        if prepared.x_implicit.is_some() {
            prepared.materialize(x, y)
        } else {
            a
        }
    }

    /// Row index `pin_component_signs` pivots on: largest `|u_i|`, ties to
    /// the lowest index (strict `>` while scanning upward).
    fn sign_pivot(u: ColRef<'_, f64>) -> usize {
        let mut best = 0usize;
        for i in 1..u.nrows() {
            if u[i].abs() > u[best].abs() {
                best = i;
            }
        }
        best
    }

    /// Shape checks for a model that truncated to `k_used` components.
    #[allow(clippy::many_single_char_names)]
    fn assert_shapes_match_k_used(m: &Pls3Model, n: usize, p: usize, q: usize) {
        let k = m.k_used;
        assert_eq!((m.u_saliences.nrows(), m.u_saliences.ncols()), (p, k));
        assert_eq!((m.v_saliences.nrows(), m.v_saliences.ncols()), (q, k));
        assert_eq!(m.singular_values.nrows(), k);
        assert_eq!((m.x_scores.nrows(), m.x_scores.ncols()), (n, k));
        assert_eq!((m.y_scores.nrows(), m.y_scores.ncols()), (n, k));
        assert_eq!(m.converged.as_ref().map(Vec::len), Some(k));
        assert_eq!(m.n_iter.as_ref().map(Vec::len), Some(k));
    }

    /// `X = I_3` and `Y = a` with both blocks declared pre-standardized, so
    /// the cross-covariance the fit sees is `a` itself, bit for bit (every
    /// product in `I'a` is against an exact 0 or 1).
    fn identity_x_with_cross_cov(a: &Mat<f64>) -> (Mat<f64>, Mat<f64>, Pls3FitOpts) {
        let x = Mat::<f64>::identity(a.nrows(), a.nrows());
        let opts = Pls3FitOpts {
            pre_standardized_x: true,
            pre_standardized_y: true,
            ..Pls3FitOpts::default()
        };
        (x, a.clone(), opts)
    }

    #[allow(clippy::many_single_char_names)]
    #[test]
    fn spls3_dust_component_truncates_k_used() {
        // A = [[1, 0, 0], [0, d, d], [0, 0, 0]] with d = 1e-16. Component 1
        // lands exactly on the (0, 0) entry (keep_x = 1; the row-0 support
        // of A'u is a single entry), so σ₁ = 1 and deflation leaves only
        // the d-block. Component 2 then starts from that block, and any
        // u = select(A₂v) has ‖u‖ ≤ ‖A₂‖₂ = √2·d ≈ 1.4e-16 < SIGMA_FLOOR,
        // so `spls3_component` returns `Ok(None)` on the first sweep,
        // before σ is ever formed. The margin is two orders of magnitude,
        // not a last-bit coincidence.
        let d = 1e-16;
        let a0 = Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1 | 2) => d,
            _ => 0.0,
        });
        let (x, y, opts) = identity_x_with_cross_cov(&a0);
        let (k, keep_x, keep_y) = (2, 1, 2);
        let m = spls3_fit(x.as_ref(), y.as_ref(), k, keep_x, keep_y, None, opts).unwrap();
        assert_eq!(m.k_used, 1, "expected truncation below k = {k}");
        assert_shapes_match_k_used(&m, 3, 3, 3);

        // Pin the exit, not just the count: on the deflated A the next
        // component must be dust (`None`), not a sub-floor σ.
        let (comps, a_rest, par) = spls3_unpinned(x.as_ref(), y.as_ref(), k, keep_x, keep_y, opts);
        assert_eq!(comps.len(), m.k_used);
        let next = spls3_component(
            a_rest.as_ref(),
            keep_x,
            keep_y,
            opts.max_iter,
            opts.tol,
            par,
        )
        .unwrap();
        assert!(
            next.is_none(),
            "component 2 should take the Ok(None) dust exit, got sigma = {:?}",
            next.map(|c| c.sigma)
        );
    }

    /// `A = [[s, 0, 0], [0, d, d], [0, 0, 0]]` with `(keep_x, keep_y) = (1, 2)`:
    /// component 1 is the exact `(0, 0)` entry (`σ₁ = s`, deflation exact),
    /// and component 2 is the `d` block with `σ₂ ≈ √2·d`.
    fn two_block_cross_cov(s: f64, d: f64) -> Mat<f64> {
        Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => s,
            (1, 1 | 2) => d,
            _ => 0.0,
        })
    }

    #[allow(clippy::many_single_char_names)]
    #[test]
    fn spls3_relative_floor_truncates_k_used() {
        // The relative floor, not the absolute one, must stop the loop, with
        // wide margins on both sides so no platform's last bits can move
        // the outcome: ‖X‖_F·‖Y‖_F = √3·‖A‖_F ≈ √3·1e6 puts the floor at
        // 3·eps·√3·1e6 ≈ 1.2e-9, and σ₂ ≈ √2·1e-11 ≈ 1.4e-11 sits ~80× under
        // it and ~1400× over SIGMA_FLOOR (so neither the dust exit nor the
        // absolute break fires).
        //
        // The absolute `c.sigma < SIGMA_FLOOR` break in `spls3_fit` is not
        // exercised by any test on purpose: σ = u'Av equals the last sweep's
        // `nv` in exact arithmetic, and `nv < SIGMA_FLOOR` already takes the
        // dust exit, so that break fires only when σ rounds under the floor
        // while `nv` sits on it. An input that hits it is a bit-searched
        // one-ulp coincidence that faer's per-platform kernels can undo.
        let (k, keep_x, keep_y) = (2, 1, 2);
        let (x, y, opts) = identity_x_with_cross_cov(&two_block_cross_cov(1e6, 1e-11));
        let m = spls3_fit(x.as_ref(), y.as_ref(), k, keep_x, keep_y, None, opts).unwrap();
        assert_eq!(m.k_used, 1, "expected truncation below k = {k}");
        assert_shapes_match_k_used(&m, 3, 3, 3);

        // Pin the exit: the next component is `Some`, clears the absolute
        // floor by far, and sits well under the relative one.
        let (comps, a_rest, par) = spls3_unpinned(x.as_ref(), y.as_ref(), k, keep_x, keep_y, opts);
        assert_eq!(comps.len(), m.k_used);
        let next = spls3_component(
            a_rest.as_ref(),
            keep_x,
            keep_y,
            opts.max_iter,
            opts.tol,
            par,
        )
        .unwrap()
        .expect("component 2 took the dust exit");
        let rel = prepared_rel_floor(x.as_ref(), y.as_ref(), opts);
        assert!(
            next.sigma > 100.0 * SIGMA_FLOOR && next.sigma < rel / 5.0,
            "component 2 sigma = {:e} not in (100·SIGMA_FLOOR, rel_floor/5 = {:e})",
            next.sigma,
            rel / 5.0
        );

        // Control: the same shape with a σ₂ far above the relative floor
        // keeps both components, so the truncation above is the floor's.
        let (x, y, opts) = identity_x_with_cross_cov(&two_block_cross_cov(1e6, 1e-6));
        let m = spls3_fit(x.as_ref(), y.as_ref(), k, keep_x, keep_y, None, opts).unwrap();
        assert_eq!(m.k_used, 2);
    }

    /// Rank-deficient Y (the second case in the "Truncation" section of
    /// `_docs/concepts/PLS3/fit-and-transform.md`): a Y block holding
    /// subscales plus their total has rank `q − 1`, so `k = q` asks for a
    /// component that does not exist. Its σ lands near `eps·σ₁` (above the
    /// absolute 1e-14 once `σ₁` is past ~50; 2.3e-13 here) and must be
    /// dropped by the relative floor.
    #[test]
    fn total_score_column_truncates_at_true_rank() {
        let (x, mut y) = shared_factor_data(200, 1000, 5, 11);
        for i in 0..200 {
            y[(i, 4)] = (0..4).map(|j| y[(i, j)]).sum();
        }
        let m = pls3_fit(x.as_ref(), y.as_ref(), 5, None, Pls3FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 4, "sigma = {:?}", m.singular_values);
        assert!(m.singular_values[3] > 1e-3 * m.singular_values[0]);
    }

    /// `n − 1 < k`: centered rows span at most `n − 1` dimensions. X is
    /// standardized here and then scaled by 1e3 under
    /// `pre_standardized_x`, which scales every σ by 1e3 and leaves their
    /// ratios alone: σ₁₀ then clears the absolute 1e-14 by orders of
    /// magnitude on any platform (unscaled it lands on either side of it,
    /// 4.7e-15 on aarch64-apple-darwin), so only the relative floor can
    /// stop at 9.
    #[test]
    fn wide_short_design_truncates_at_n_minus_1() {
        let (x, y) = shared_factor_data(10, 50, 20, 12);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let xs = Mat::<f64>::from_fn(10, 50, |i, j| 1e3 * xs[(i, j)]);
        let opts = Pls3FitOpts {
            pre_standardized_x: true,
            ..Pls3FitOpts::default()
        };
        let (Prepared { ys_owned, .. }, a) =
            prepare(xs.as_ref(), y.as_ref(), 15, None, opts).unwrap();
        let sigma = a
            .as_ref()
            .thin_svd()
            .unwrap()
            .S()
            .column_vector()
            .to_owned();
        let ys = ys_owned.unwrap();
        let rel = sigma_rel_floor(10, 50, 20, xs.norm_l2(), ys.norm_l2());
        assert!(
            sigma[9] > 100.0 * SIGMA_FLOOR && sigma[9] < rel / 5.0 && sigma[8] > 1e3 * rel,
            "premise: sigma = {sigma:?}, rel_floor = {rel:e}"
        );
        let m = pls3_fit(xs.as_ref(), y.as_ref(), 15, None, opts).unwrap();
        assert_eq!(m.k_used, 9, "sigma = {:?}", m.singular_values);
    }

    /// Standard normal draws (Box-Muller), row-major fill order.
    #[allow(clippy::many_single_char_names)]
    fn gaussian_mat(rng: &mut rand_chacha::ChaCha8Rng, n: usize, m: usize) -> Mat<f64> {
        use rand::RngExt;
        Mat::<f64>::from_fn(n, m, |_, _| {
            let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
    }

    /// Orthonormal basis of `span{1, columns of m}`.
    fn basis_with_ones(m: MatRef<'_, f64>) -> Vec<Col<f64>> {
        crate::test_support::orthonormal_basis(
            Col::<f64>::from_fn(m.nrows(), |_| 1.0).as_ref(),
            m,
            0.0,
        )
    }

    /// Weak-signal Y: `δ·X·B` plus Gaussian noise projected off
    /// `span{1, X}`. In exact arithmetic `X̃'Ỹ` is `δ·X̃'X·B` with its
    /// columns rescaled, so it has the rank of `B` and a `σ₁` of order
    /// `δ`, while `‖X̃‖_F·‖Ỹ‖_F` stays `n·√(pq)`: the regime where a floor
    /// relative to `σ₁` sits far under the rounding in `X̃'Ỹ`. With
    /// `total`, one more column holding the sum of the others is appended.
    #[allow(clippy::many_single_char_names)]
    fn weak_signal_y(x: &Mat<f64>, b: &Mat<f64>, delta: f64, total: bool, seed: u64) -> Mat<f64> {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (n, r) = (x.nrows(), b.ncols());
        let mut z = gaussian_mat(&mut rng, n, r);
        let basis = basis_with_ones(x.as_ref());
        for j in 0..r {
            let mut v = z.col(j).to_owned();
            project_off(&basis, &mut v);
            z.col_mut(j).copy_from(&v);
        }
        let xb = x * b;
        let q = if total { r + 1 } else { r };
        Mat::<f64>::from_fn(n, q, |i, j| {
            if j < r {
                delta * xb[(i, j)] + z[(i, j)]
            } else {
                (0..r).map(|jj| delta * xb[(i, jj)] + z[(i, jj)]).sum()
            }
        })
    }

    /// A floor relative to the first component's `σ`,
    /// `max(n, p, q)·eps·σ_ref` with `σ_ref` that `σ`: the naive alternative,
    /// kept so the tests below show it would keep the noise component they drop.
    #[allow(clippy::cast_precision_loss)]
    fn sigma1_relative_floor(n: usize, p: usize, q: usize, sigma_ref: f64) -> f64 {
        (n.max(p).max(q) as f64) * f64::EPSILON * sigma_ref
    }

    /// Weak signal plus a total-score column: `X̃'Ỹ` has rank 3 and a
    /// `σ₁` of order `δ·n`, while its structurally zero `σ₄` is rounding
    /// noise of order `eps·‖X̃‖_F·‖Ỹ‖_F`, far above `max(n, p, q)·eps·σ₁`.
    /// A floor relative to `σ₁` kept it (`k_used = 8`). Measured on
    /// aarch64-apple-darwin: `σ₈ ≈ 3e-13`, 30× over `SIGMA_FLOOR`, 55× over
    /// the `σ₁`-relative floor and 2.5e6× under this one; `σ₇ ≈ 1.6e-4`
    /// sits 200× above it.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn weak_signal_total_score_column_truncates_at_true_rank() {
        use rand::SeedableRng;
        let (n, p, k) = (20_000, 10, 8);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(21);
        let x = gaussian_mat(&mut rng, n, p);
        let b = gaussian_mat(&mut rng, p, k - 1);
        let y = weak_signal_y(&x, &b, 1e-8, true, 22);
        let opts = Pls3FitOpts::default();
        let a = written_cross_cov(x.as_ref(), y.as_ref(), opts);
        let sigma = a
            .as_ref()
            .thin_svd()
            .unwrap()
            .S()
            .column_vector()
            .to_owned();
        let rel = prepared_rel_floor(x.as_ref(), y.as_ref(), opts);
        let old = sigma1_relative_floor(n, p, k, sigma[0]);
        assert!(
            sigma[k - 1] > 10.0 * old.max(SIGMA_FLOOR)
                && sigma[k - 1] < rel / 10.0
                && sigma[k - 2] > 10.0 * rel,
            "premise: sigma = {sigma:?}, rel_floor = {rel:e}, sigma1 floor = {old:e}"
        );
        let m = pls3_fit(x.as_ref(), y.as_ref(), k, None, opts).unwrap();
        assert_eq!(m.k_used, k - 1, "sigma = {:?}", m.singular_values);
    }

    /// Y exactly orthogonal to X (in exact arithmetic `X̃'Ỹ = 0`): every
    /// `σ` is rounding noise, `σ₁` included, and the relative floor applies
    /// from the first component on, so nothing is kept (`k_used = 0`), the
    /// zero model `pls1_fit` returns on the same input. The absolute floor
    /// alone kept LV1 here (`σ₁` clears it), and a floor relative to `σ₁`
    /// kept every requested component. Measured on aarch64-apple-darwin:
    /// `σ₁ ≈ 1e-12`, `σ₂ ≈ 7e-13`, the relative floor 4e-7.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn orthogonal_blocks_keep_no_component() {
        use rand::SeedableRng;
        let (n, p, q) = (20_000, 6, 4);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(31);
        let x = gaussian_mat(&mut rng, n, p);
        let y = weak_signal_y(&x, &Mat::<f64>::zeros(p, q), 0.0, false, 32);
        let opts = Pls3FitOpts::default();
        let a = written_cross_cov(x.as_ref(), y.as_ref(), opts);
        let sigma = a
            .as_ref()
            .thin_svd()
            .unwrap()
            .S()
            .column_vector()
            .to_owned();
        let rel = prepared_rel_floor(x.as_ref(), y.as_ref(), opts);
        let old = sigma1_relative_floor(n, p, q, sigma[0]);
        assert!(
            sigma[0] > 10.0 * SIGMA_FLOOR
                && sigma[0] < rel / 10.0
                && sigma[1] > 10.0 * old.max(SIGMA_FLOOR),
            "premise: sigma = {sigma:?}, rel_floor = {rel:e}, sigma1 floor = {old:e}"
        );
        for k in [1, q] {
            let m = pls3_fit(x.as_ref(), y.as_ref(), k, None, opts).unwrap();
            assert_eq!(m.k_used, 0, "k = {k}, sigma = {:?}", m.singular_values);
            assert_eq!((m.u_saliences.nrows(), m.u_saliences.ncols()), (p, 0));
            assert_eq!((m.x_scores.nrows(), m.x_scores.ncols()), (n, 0));
            // The zero model still transforms: empty score matrices.
            let t = pls3_transform(&m, Some(x.as_ref()), Some(y.as_ref()), TransformWhich::Both)
                .unwrap();
            assert_eq!(t.x_scores.map(|a| (a.nrows(), a.ncols())), Some((n, 0)));
            assert_eq!(t.y_scores.map(|a| (a.nrows(), a.ncols())), Some((n, 0)));
        }

        // `q = 1` is the PLS1 shape: the two families agree on it.
        let y1 = Mat::<f64>::from_fn(n, 1, |i, _| y[(i, 0)]);
        let m3 = pls3_fit(x.as_ref(), y1.as_ref(), 1, None, opts).unwrap();
        let m1 = crate::fit::pls1_fit(
            x.as_ref(),
            y1.col(0),
            crate::fit::KSpec::Fixed(1),
            None,
            crate::fit::FitOpts::default(),
        )
        .unwrap();
        assert_eq!((m3.k_used, m1.k_used), (0, 0));

        // The sparse fit applies the same floor to its first component.
        let ms = spls3_fit(x.as_ref(), y.as_ref(), 2, 3, 2, None, opts).unwrap();
        assert_eq!(ms.k_used, 0, "sigma = {:?}", ms.singular_values);
        assert_shapes_match_k_used(&ms, n, p, q);
    }

    /// Sparse counterpart: X has orthonormal centered columns, so
    /// `X̃'Ỹ` is exactly `δ·B` up to column scaling, and `B` is rank 1 on
    /// a 3 × 2 support matching `(keep_x, keep_y)`. The first sparse
    /// component recovers it, and the deflated `A` is rounding noise. That
    /// noise sits far above `max(n, p, q)·eps` times the first sparse `σ`
    /// (the `σ₁`-relative alternative) and far under the floor from the blocks.
    /// Measured on aarch64-apple-darwin: the second `σ ≈ 6e-13`, 600× over
    /// the first-`σ` floor and 2e5× under this one.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn spls3_weak_signal_truncates_after_planted_component() {
        use rand::SeedableRng;
        let (n, p, q, k, keep_x, keep_y) = (10_000, 8, 5, 2, 3, 2);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(41);
        let g = gaussian_mat(&mut rng, n, p);
        let basis = basis_with_ones(g.as_ref());
        let x = Mat::<f64>::from_fn(n, p, |i, j| basis[j + 1][i]);
        let b = Mat::<f64>::from_fn(p, q, |i, j| {
            if i < 3 && j < 2 {
                (1.0 + 0.5 * i as f64) * (1.0 + 0.3 * j as f64)
            } else {
                0.0
            }
        });
        let y = weak_signal_y(&x, &b, 1e-6, false, 42);
        let opts = Pls3FitOpts::default();
        let m = spls3_fit(x.as_ref(), y.as_ref(), k, keep_x, keep_y, None, opts).unwrap();
        let (comps, a_rest, par) = spls3_unpinned(x.as_ref(), y.as_ref(), k, keep_x, keep_y, opts);
        let next = spls3_component(
            a_rest.as_ref(),
            keep_x,
            keep_y,
            opts.max_iter,
            opts.tol,
            par,
        )
        .unwrap()
        .expect("component 2 took the dust exit");
        let rel = prepared_rel_floor(x.as_ref(), y.as_ref(), opts);
        let old = sigma1_relative_floor(n, p, q, comps[0].sigma);
        assert!(
            next.sigma > 10.0 * old.max(SIGMA_FLOOR) && next.sigma < rel / 10.0,
            "premise: sigma_1 = {:e}, sigma_2 = {:e}, rel_floor = {rel:e}, first-sigma floor = {old:e}",
            comps[0].sigma,
            next.sigma
        );
        assert_eq!(m.k_used, 1, "sigma = {:?}", m.singular_values);
        assert_eq!(comps.len(), 1);
    }

    /// Saliences, `σ` and both score blocks, flattened: what a fit of
    /// manually standardized blocks shares with the fit that standardizes.
    fn components_flat(m: &Pls3Model) -> Vec<f64> {
        use crate::test_support::{col_vals, mat_vals};
        let mut v = mat_vals(m.u_saliences.as_ref());
        v.extend(mat_vals(m.v_saliences.as_ref()));
        v.extend(col_vals(m.singular_values.as_ref()));
        v.extend(mat_vals(m.x_scores.as_ref()));
        v.extend(mat_vals(m.y_scores.as_ref()));
        v
    }

    fn pre_both() -> Pls3FitOpts {
        Pls3FitOpts {
            pre_standardized_x: true,
            pre_standardized_y: true,
            ..Pls3FitOpts::default()
        }
    }

    /// Offset X on both sides of `IMPLICIT_MAX_MEAN_RATIO`: `+100` stays on
    /// the implicit route, `+1e6` takes the written copy from the start.
    /// On either, the dense and the sparse fit agree with the fit of the
    /// manually standardized X to the corpus tolerance and record
    /// `linalg::standardize`'s moments to the bit. Y is standardized by the
    /// fit, or passed under `pre_standardized_y` with its columns off
    /// center (`+0.5`), where the mean correction of `X̃'Ỹ` does not vanish.
    #[test]
    fn offset_x_agrees_with_the_written_copy_on_both_routes() {
        use crate::test_support::{assert_agree, col_vals, Agree};
        let (x0, y) = shared_factor_data(60, 40, 5, 51);
        let (ys, _, _) = crate::linalg::standardize(y.as_ref());
        let y_off = Mat::<f64>::from_fn(60, 5, |i, j| y[(i, j)] + 0.5);
        for (c, implicit) in [(100.0, true), (1e6, false)] {
            let x = Mat::<f64>::from_fn(60, 40, |i, j| x0[(i, j)] + c);
            let (xs, x_mean, x_scale) = crate::linalg::standardize(x.as_ref());
            for pre_y in [false, true] {
                let opts = Pls3FitOpts {
                    pre_standardized_y: pre_y,
                    ..Pls3FitOpts::default()
                };
                let (y_fit, y_copy) = if pre_y { (&y_off, &y_off) } else { (&y, &ys) };
                let (prepared, _) = prepare(x.as_ref(), y_fit.as_ref(), 3, None, opts).unwrap();
                let what = format!(
                    "offset {c:e} (ratio {:?}) pre_y={pre_y}",
                    prepared.x_implicit
                );
                assert_eq!(
                    prepared.x_implicit.is_some(),
                    implicit,
                    "{what}: left its route"
                );
                assert!(
                    prepared.x_implicit.is_none_or(|r| r > 100.0),
                    "{what}: not mean-heavy"
                );
                for keeps in [None, Some((10, 3))] {
                    let fit = |x: MatRef<'_, f64>, y: MatRef<'_, f64>, opts| match keeps {
                        None => pls3_fit(x, y, 3, None, opts).unwrap(),
                        Some((kx, ky)) => spls3_fit(x, y, 3, kx, ky, None, opts).unwrap(),
                    };
                    let m = fit(x.as_ref(), y_fit.as_ref(), opts);
                    let copy = fit(xs.as_ref(), y_copy.as_ref(), pre_both());
                    let what = format!("{what} keeps={keeps:?}");
                    assert_eq!(m.k_used, 3, "{what}");
                    assert_agree(
                        &components_flat(&m),
                        &components_flat(&copy),
                        Agree::Corpus(X_LAYOUT_TOL),
                        &what,
                    );
                    for (got, want) in [(&m.x_mean, &x_mean), (&m.x_scale, &x_scale)] {
                        assert_agree(
                            &col_vals(got.as_ref()),
                            &col_vals(want.as_ref()),
                            Agree::Bits,
                            &what,
                        );
                    }
                }
            }
        }
    }

    /// A truncating fit is the written copy's to the bit. A Y block with a
    /// total-score column has rank `q − 1`, so `k = q` truncates; on an X
    /// offset by `+100` (the implicit route) the fit is rerun on the copy
    /// and returns the components of the fit of the manually standardized
    /// blocks.
    #[test]
    fn truncating_fit_on_offset_x_is_the_written_copys() {
        use crate::test_support::{assert_agree, Agree};
        let (x0, mut y) = shared_factor_data(200, 30, 5, 52);
        for i in 0..200 {
            y[(i, 4)] = (0..4).map(|j| y[(i, j)]).sum();
        }
        let x = Mat::<f64>::from_fn(200, 30, |i, j| x0[(i, j)] + 100.0);
        let opts = Pls3FitOpts::default();
        let (prepared, _) = prepare(x.as_ref(), y.as_ref(), 5, None, opts).unwrap();
        assert!(prepared.x_implicit.is_some(), "premise: the implicit route");
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize(y.as_ref());

        let m = pls3_fit(x.as_ref(), y.as_ref(), 5, None, opts).unwrap();
        let copy = pls3_fit(xs.as_ref(), ys.as_ref(), 5, None, pre_both()).unwrap();
        assert_eq!(
            (m.k_used, copy.k_used),
            (4, 4),
            "sigma = {:?}",
            m.singular_values
        );
        assert_agree(
            &components_flat(&m),
            &components_flat(&copy),
            Agree::Bits,
            "dense",
        );
    }

    /// Y orthogonal to an offset X, below `IMPLICIT_MAX_MEAN_RATIO`: the
    /// designs of `fit::tests::orthogonal_y_on_offset_x_keeps_no_component`
    /// with a two-column Y. On some of them the implicit `A` puts `σ₁`
    /// above the relative floor, where the written copy puts it below; the
    /// band of `Prepared::stop_floor` sends every one to the copy, and
    /// nothing is kept.
    #[test]
    fn orthogonal_y_on_offset_x_keeps_no_component() {
        use rand::{RngExt, SeedableRng};
        let designs = [[0.37, 99.7, -1.3, 89.73], [0.37, 0.0, 1.0, 999.3]];
        let mut above_floor = 0_usize;
        for (c, n) in designs
            .iter()
            .flat_map(|c| [8_usize, 12, 16].map(|n| (c, n)))
        {
            // Column j is `c[2j]·a + c[2j + 1]`, `a = ±1` alternating.
            let x = Mat::<f64>::from_fn(n, 2, |i, j| {
                let a = if i % 2 == 0 { 1.0 } else { -1.0 };
                c[2 * j] * a + c[2 * j + 1]
            });
            let ones = Col::<f64>::from_fn(n, |_| 1.0);
            let basis = crate::test_support::orthonormal_basis(ones.as_ref(), x.as_ref(), 1e-12);
            for seed in 0..20_u64 {
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                let mut y = Mat::<f64>::zeros(n, 2);
                for j in 0..2 {
                    let mut col = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
                    project_off(&basis, &mut col);
                    y.col_mut(j).copy_from(&col);
                }
                let what = format!("{c:?} n={n} seed={seed}");
                let opts = Pls3FitOpts::default();
                let (prepared, a) = prepare(x.as_ref(), y.as_ref(), 1, None, opts).unwrap();
                assert!(
                    prepared.x_implicit.is_some(),
                    "{what}: premise: the implicit route"
                );
                let sigma = a
                    .as_ref()
                    .thin_svd()
                    .unwrap()
                    .S()
                    .column_vector()
                    .to_owned();
                above_floor += usize::from(sigma[0] >= prepared.rel_floor(x.as_ref(), y.as_ref()));
                let m = pls3_fit(x.as_ref(), y.as_ref(), 1, None, opts).unwrap();
                assert_eq!(m.k_used, 0, "{what}: sigma = {:?}", m.singular_values);
                let ms = spls3_fit(x.as_ref(), y.as_ref(), 1, 1, 1, None, opts).unwrap();
                assert_eq!(ms.k_used, 0, "{what}: sigma = {:?}", ms.singular_values);
            }
        }
        assert!(
            above_floor > 0,
            "premise: no implicit sigma_1 cleared the floor"
        );
    }

    /// `pls3_fit` / `spls3_fit` / `pls3_transform` output flattened for the
    /// layout tests: saliences, `σ`, both score blocks and the four moment
    /// vectors.
    fn layout_flat(m: &Pls3Model) -> Vec<f64> {
        use crate::test_support::{col_vals, mat_vals};
        let mut v = mat_vals(m.u_saliences.as_ref());
        v.extend(mat_vals(m.v_saliences.as_ref()));
        v.extend(col_vals(m.singular_values.as_ref()));
        v.extend(mat_vals(m.x_scores.as_ref()));
        v.extend(mat_vals(m.y_scores.as_ref()));
        for c in [&m.x_mean, &m.x_scale, &m.y_mean, &m.y_scale] {
            v.extend(col_vals(c.as_ref()));
        }
        v
    }

    /// The unweighted `copy_free_families` (PLS3 refuses weights) with a
    /// three-column Y built from each family's `(x, y)`, crossed with both
    /// `pre_standardized_*` flags: `(label, x, y, opts)`, each block
    /// standardized without weights when its flag is set.
    fn layout_cases() -> Vec<(String, Mat<f64>, Mat<f64>, Pls3FitOpts)> {
        let mut out = Vec::new();
        for f in crate::test_support::copy_free_families()
            .into_iter()
            .filter(|f| f.w.is_none())
        {
            let n = f.x.nrows();
            #[allow(clippy::cast_precision_loss)]
            let yy = Mat::<f64>::from_fn(n, 3, |i, j| f.y[i] * (j + 1) as f64 + f.x[(i, j)]);
            for pre_x in [false, true] {
                for pre_y in [false, true] {
                    let x = if pre_x {
                        crate::linalg::standardize(f.x.as_ref()).0
                    } else {
                        f.x.clone()
                    };
                    let y = if pre_y {
                        crate::linalg::standardize(yy.as_ref()).0
                    } else {
                        yy.clone()
                    };
                    let opts = Pls3FitOpts {
                        pre_standardized_x: pre_x,
                        pre_standardized_y: pre_y,
                        ..Pls3FitOpts::default()
                    };
                    out.push((format!("{} pre=({pre_x}, {pre_y})", f.name), x, y, opts));
                }
            }
        }
        out
    }

    /// Layouts of a standardized X agree to rounding, not bit for bit: the
    /// fits form their products from X in X's own layout ("The implicit
    /// route" on `Prepared`). `Agree::Corpus(1e-10)` is the corpus array
    /// tolerance, `1e-10 + 1e-14 · |value|`, which `_docs/python/api.md`
    /// (Conventions) promises across layouts.
    const X_LAYOUT_TOL: f64 = 1e-10;

    /// Layout invariance in both blocks: a padded submatrix, a row-major
    /// view and a negative-column-stride view of X, or of Y, give the
    /// owned column-major block's output, for the dense fit at
    /// `k = 1, 2`, the sparse fit (selecting, and at its dense endpoint)
    /// and `pls3_transform`, with and without `pre_standardized_*` on
    /// either block. The other block stays owned. The fits agree to
    /// `X_LAYOUT_TOL` across layouts of a standardized X; everything else
    /// agrees to the bit.
    #[test]
    fn pls3_family_agrees_across_layouts() {
        use crate::test_support::{assert_layout_agree, mat_vals, Agree};
        /// The fit inputs `(x, y)` with the varied side replaced by `v`.
        fn blocks<'a>(
            side: &str,
            v: MatRef<'a, f64>,
            x: &'a Mat<f64>,
            y: &'a Mat<f64>,
        ) -> (MatRef<'a, f64>, MatRef<'a, f64>) {
            if side == "X" {
                (v, y.as_ref())
            } else {
                (x.as_ref(), v)
            }
        }
        for (label, x, y, opts) in layout_cases() {
            let p = x.ncols();
            let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, opts).unwrap();
            for side in ["X", "Y"] {
                let varied = if side == "X" { &x } else { &y };
                let fit_agree = if side == "X" && !opts.pre_standardized_x {
                    Agree::Corpus(X_LAYOUT_TOL)
                } else {
                    Agree::Bits
                };
                for k in [1, 2] {
                    let what = format!("{side} layout: pls3_fit k={k} {label}");
                    assert_layout_agree(varied, &what, fit_agree, |v| {
                        let (xv, yv) = blocks(side, v, &x, &y);
                        Ok(layout_flat(&pls3_fit(xv, yv, k, None, opts)?))
                    })
                    .unwrap();
                }
                for (keep_x, keep_y) in [(3, 2), (p, 3)] {
                    let what =
                        format!("{side} layout: spls3_fit keep=({keep_x}, {keep_y}) {label}");
                    assert_layout_agree(varied, &what, fit_agree, |v| {
                        let (xv, yv) = blocks(side, v, &x, &y);
                        Ok(layout_flat(&spls3_fit(
                            xv, yv, 2, keep_x, keep_y, None, opts,
                        )?))
                    })
                    .unwrap();
                }
                let what = format!("{side} layout: pls3_transform {label}");
                assert_layout_agree(varied, &what, Agree::Bits, |v| {
                    let s = if side == "X" {
                        pls3_transform(&m, Some(v), None, TransformWhich::XScores)?.x_scores
                    } else {
                        pls3_transform(&m, None, Some(v), TransformWhich::YScores)?.y_scores
                    };
                    Ok(mat_vals(s.unwrap().as_ref()))
                })
                .unwrap();
            }
        }
    }

    /// Zero rows: every PLS3 fit raises `invalid_argument`, as `pls1_fit`
    /// does, instead of returning `k_used = 0` with NaN moments (the mean
    /// of an empty column). The argument checks keep their precedence, as
    /// in `pls1_fit`, whose row-count check also comes last. One row is
    /// not an error: centering leaves `n − 1 = 0` dimensions, so the
    /// standardizing fit truncates to `k_used = 0` with finite moments
    /// ("Truncation" in `_docs/concepts/PLS3/fit-and-transform.md`).
    #[test]
    fn zero_rows_error_and_one_row_truncates() {
        type Fit =
            fn(MatRef<'_, f64>, MatRef<'_, f64>, usize, Pls3FitOpts) -> PlsKitResult<Pls3Model>;
        let fits: [(&str, Fit); 4] = [
            ("pls3_fit", |x, y, k, o| pls3_fit(x, y, k, None, o)),
            ("plssvd_fit", |x, y, k, o| plssvd_fit(x, y, k, None, o)),
            ("spls3_fit selecting", |x, y, k, o| {
                spls3_fit(x, y, k, 2, 2, None, o)
            }),
            ("spls3_fit dense endpoint", |x, y, k, o| {
                spls3_fit(x, y, k, 4, 3, None, o)
            }),
        ];
        let empty_x = Mat::<f64>::zeros(0, 4);
        let empty_y = Mat::<f64>::zeros(0, 3);
        let (x1, y1) = shared_factor_data(1, 4, 3, 2);
        for (name, fit) in fits {
            for (pre_x, pre_y) in [(false, false), (true, false), (false, true), (true, true)] {
                let o = Pls3FitOpts {
                    pre_standardized_x: pre_x,
                    pre_standardized_y: pre_y,
                    ..Pls3FitOpts::default()
                };
                let what = format!("{name} pre=({pre_x}, {pre_y})");
                match fit(empty_x.as_ref(), empty_y.as_ref(), 1, o) {
                    Err(PlsKitError::InvalidArgument(m)) => {
                        assert!(m.contains("need n >= 1"), "{what}: {m}");
                    }
                    r => panic!("{what}: {r:?}"),
                }
                assert!(
                    matches!(
                        fit(empty_x.as_ref(), empty_y.as_ref(), 4, o),
                        Err(PlsKitError::KExceedsMax { k: 4, k_max: 3 })
                    ),
                    "{what}: k > k_max is reported first"
                );
                if !(pre_x || pre_y) {
                    let m = fit(x1.as_ref(), y1.as_ref(), 1, o).unwrap();
                    assert_eq!(m.k_used, 0, "{what}: one row");
                    let moments = [&m.x_mean, &m.x_scale, &m.y_mean, &m.y_scale];
                    assert!(
                        moments.iter().all(|c| c.iter().all(|v| v.is_finite())),
                        "{what}: one row, moments {moments:?}"
                    );
                }
            }
        }
    }
}
