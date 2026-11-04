//! Splitting one coefficient vector into orthogonal factors that sum to it.
//!
//! [`factors`] leaves out the variables whose `coef` is exactly zero and
//! takes the PCA basis `V` of the columns left in, standardized (unless `X`
//! is told to be already). When `coef` has a part outside the span of `V`
//! (a sparse fit on wide `X` keeps more variables than `X` has rows), that
//! part, normalized, is appended to `V` as one last column, which `X` maps
//! to zero. `W` is `V` padded with zero rows to `(D, K)`, so it has
//! orthonormal columns. With `coef = W c` in their span, the betas
//! `c = W' coef` are cut into `n` consecutive blocks of about equal `Σ c²`. Block `j` gives `piece_j = W[:, b_j] c[b_j]`; the pieces are
//! orthogonal, so `coef = Σ_j piece_j` and `Σ_j ‖piece_j‖² = ‖coef‖²`. The
//! variables left out hold exact zeros in every piece.
//! [`FactorsOutput::rotated`] then rotates the unit directions by an
//! orthogonal `R` and re-projects `coef` onto each rotated axis, which keeps
//! both identities for every `R`.
//!
//! [`profiles_find_n`] chooses `n`: it counts the rotated components of `X`
//! that raise held-out R² for `y`.
//!
//! # References
//!
//! - Horn, J. L. (1965). A rationale and test for the number of factors in
//!   factor analysis. Psychometrika 30, 179–185.
//! - Kaiser, H. F. (1958). The varimax criterion for analytic rotation in
//!   factor analysis. Psychometrika 23, 187–200.

use faer::{Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{FitOpts, KSpec, ParChoice, pls1_fit_impl};
use crate::linalg::{
    col_row_subset, constant_to_rounding, empirical_quantile, fold_split, lu_solve_in_place,
    mat_mul, mat_vec, pow2_scale, row_subset, scaled_moments, self_adjoint_eigen, standardize,
    standardize_apply,
};
use crate::pca::pca_basis;
use crate::resample::{permutation_from_seed, permute_indices};
use crate::rng::{child_rng, child_seeds, resolve_seed};
use crate::rotate::{RotationMethod, VarimaxArgs, rotate};

/// Tolerance on `max|W'W − I|`, on `‖coef − W c‖ / ‖coef‖` and on
/// `max|R'R − I|`.
pub(crate) const ORTHO_TOL: f64 = 1e-10;
/// A block whose size is at most this times `‖coef‖` is treated as empty.
const EMPTY_BLOCK_TOL: f64 = 1e-12;

/// Factors of one coefficient vector.
#[derive(Debug, Clone)]
pub struct FactorsOutput {
    /// Factors, shape `(D, n)`. Columns sum to `coef`.
    pub f: Mat<f64>,
    /// Unit directions, shape `(D, n)`, orthonormal. `f = w * size` column-wise.
    pub w: Mat<f64>,
    /// `‖f[:, i]‖`, length `n`, nonnegative.
    pub size: Col<f64>,
    /// The input coefficient vector, length `D`.
    pub coef: Col<f64>,
    /// Whether `X` was taken as already standardized.
    pub pre_standardized: bool,
}

/// Block end positions (exclusive) for cutting `c` into `n` blocks.
///
/// With `S(e) = Σ_{i<e} c_i² / Σ c²`, block `j` (for `j = 1 .. n-1`) ends at
/// the smallest `e` with `S(e) ≥ j / n`, then `e` is raised to at least
/// `start + 1` and lowered to at most `K - (n - j)`, so every block keeps a
/// component. Block `n` ends at `K`. The test `S(e) ≥ j / n` is evaluated as
/// `n · cum ≥ j · total`, which is the same inequality without a division.
/// Requires `1 ≤ n ≤ c.len()`.
#[allow(clippy::many_single_char_names)] // c, k, n, j, e, p are the math names
fn block_ends(c: &[f64], n: usize) -> Vec<usize> {
    let k = c.len();
    // c is multiplied by a power of two near 1/max|c| (as `scaled_moments`
    // does), which is exact. On raw c, c² underflows below about 1e-162 and
    // overflows above about 1e154, and the cuts would depend on magnitude;
    // scaled, the squares match the raw ones bit for bit wherever raw is in
    // range, so exact ties such as S(e) = 1/2 are kept. Requires some c ≠ 0.
    let (_, inv) = pow2_scale(c.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
    let mut cum = Vec::with_capacity(k);
    let mut run = 0.0;
    for v in c {
        run += (v * inv) * (v * inv);
        cum.push(run);
    }
    let total = run;
    let mut ends = Vec::with_capacity(n);
    let mut start = 0;
    for j in 1..n {
        let target = j as f64 * total;
        // The last entry of `cum` is `total`, and `n · total ≥ j · total`
        // for j < n, so some entry always passes.
        let p = cum
            .iter()
            .position(|&s| s * n as f64 >= target)
            .expect("the last cumulative sum equals total");
        let e = (p + 1).max(start + 1).min(k - (n - j));
        ends.push(e);
        start = e;
    }
    ends.push(k);
    ends
}

/// Split `coef` into `n` orthogonal factors, using the PCA basis of `x`.
///
/// Steps: check `coef` and `x`; leave out the variables where `coef` is
/// exactly zero; take the PCA basis of the rest, standardized unless
/// `pre_standardized`; when `coef` is not in its span (`‖r‖ > 1e-10 ‖coef‖`
/// for the residual `r` of `coef` on the basis), append `r / ‖r‖` as a last
/// column, so the part of `coef` that `X` maps to zero lands in the last
/// factor; cut `W' coef` into `n` blocks of about equal `Σ c²`. Identities:
/// `coef = Σ_j F[:, j]`, `W_f' W_f = I` and `Σ_j size_j² = ‖coef‖²`, where
/// `W_f = out.w` and `F = W_f diag(size)`. Rows of `f` and `w` for the
/// variables left out are exactly zero. Factors stay in block order (not
/// sorted by size).
///
/// # Shapes
/// - `coef`: `(D,)`
/// - `x`: `(N, D)`, `N ≥ 1`
/// - `n`: `1 ≤ n ≤ K`, where `K` is the rank of the columns left in, plus
///   one when the residual column is appended
/// - returns `f: (D, n)`, `w: (D, n)`, `size: (n,)`, `coef: (D,)`
///
/// # Errors
///
/// - [`PlsKitError::ShapeMismatch`] — `coef.nrows() != x.ncols()`.
/// - [`PlsKitError::InvalidInput`] — non-finite `coef` or `x`; zero `coef`;
///   `n` outside `[1, K]`; a block with `size ≤ 1e-12 ‖coef‖` (the message
///   names the block's component range); `‖coef‖` or `W' coef` overflowing
///   to infinity; `x` with no rows, with every column left in constant
///   (when standardized), or with all singular values zero.
/// - [`PlsKitError::Internal`] — the SVD failed.
#[allow(clippy::many_single_char_names)] // x, n, d, c, w, j are the math names
pub fn factors(
    coef: ColRef<'_, f64>,
    x: MatRef<'_, f64>,
    n: usize,
    pre_standardized: bool,
) -> PlsKitResult<FactorsOutput> {
    let d = x.ncols();
    if coef.nrows() != d {
        return Err(PlsKitError::ShapeMismatch(format!(
            "coef has length {} but X has {d} columns",
            coef.nrows()
        )));
    }
    if !coef.is_all_finite() {
        return Err(PlsKitError::InvalidInput(
            "coef contains non-finite values".into(),
        ));
    }
    if !x.is_all_finite() {
        return Err(PlsKitError::InvalidInput(
            "X contains non-finite values".into(),
        ));
    }
    let kept: Vec<usize> = (0..d).filter(|&j| coef[j] != 0.0).collect();
    if kept.is_empty() {
        return Err(PlsKitError::InvalidInput("coef is zero".into()));
    }
    let x_kept = Mat::<f64>::from_fn(x.nrows(), kept.len(), |i, j| x[(i, kept[j])]);
    let mut v = pca_basis(x_kept.as_ref(), pre_standardized)?.v;
    let coef_kept = Col::<f64>::from_fn(kept.len(), |i| coef[kept[i]]);
    let resid = |u: ColRef<'_, f64>| {
        let vu = mat_vec(
            v.as_ref(),
            mat_vec(v.transpose(), u, Par::Seq).as_ref(),
            Par::Seq,
        );
        Col::<f64>::from_fn(u.nrows(), |i| u[i] - vu[i])
    };
    let r = resid(coef_kept.as_ref());
    if r.norm_l2() > ORTHO_TOL * coef_kept.norm_l2() {
        // Projected twice: after one pass `V' r` is rounding of `‖coef‖`,
        // which is not small next to a small `r`.
        let r = resid(r.as_ref());
        let r_norm = r.norm_l2();
        let k = v.ncols();
        v = Mat::<f64>::from_fn(kept.len(), k + 1, |i, j| {
            if j < k { v[(i, j)] } else { r[i] / r_norm }
        });
    }
    // W'W = V'V = I and W' coef = V' coef[kept], because coef is zero exactly
    // where W has zero rows, so the split acts on the full coef unrestricted.
    let mut w = Mat::<f64>::zeros(d, v.ncols());
    for (row, &j) in kept.iter().enumerate() {
        for col in 0..v.ncols() {
            w[(j, col)] = v[(row, col)];
        }
    }
    split_on_basis(w.as_ref(), coef, n, pre_standardized)
}

/// Split `coef` into `n` orthogonal factors along the columns of `w`.
///
/// Identities: `coef = Σ_j F[:, j]`, `W_f' W_f = I` and
/// `Σ_j size_j² = ‖coef‖²`, where `W_f = out.w` and `F = W_f diag(size)`.
/// Factors stay in block order (not sorted by size). The caller has checked
/// that `coef` is finite (`w` comes from an SVD of finite input), that `coef` is not zero and that
/// `coef.nrows() == w.nrows()`.
///
/// # Shapes
/// - `w`: `(D, K)`, orthonormal columns
/// - `coef`: `(D,)`, in the span of `w`
/// - `n`: `1 ≤ n ≤ K`
/// - returns `f: (D, n)`, `w: (D, n)`, `size: (n,)`, `coef: (D,)`
///
/// # Errors
///
/// - [`PlsKitError::InvalidInput`] — `n` outside `[1, K]`;
///   `max|W'W − I| > 1e-10`; `‖coef − W c‖ > 1e-10 ‖coef‖`; a block with
///   `size ≤ 1e-12 ‖coef‖` (the message names the block's component range);
///   `‖coef‖` or `W' coef` overflowing to infinity.
#[allow(clippy::many_single_char_names)] // w, c, n, d, k, j are the math names
fn split_on_basis(
    w: MatRef<'_, f64>,
    coef: ColRef<'_, f64>,
    n: usize,
    pre_standardized: bool,
) -> PlsKitResult<FactorsOutput> {
    let (d, k) = w.shape();
    if n < 1 || n > k {
        return Err(PlsKitError::InvalidInput(format!(
            "n={n} is outside [1, K] with K={k}"
        )));
    }
    let (c, coef_norm) = check_basis(w, coef, "the basis of X")?;
    if !coef_norm.is_finite() || !c.is_all_finite() {
        return Err(PlsKitError::InvalidInput(
            "||coef|| or W' coef overflows".into(),
        ));
    }

    let c_slice: Vec<f64> = (0..k).map(|i| c[i]).collect();
    let ends = block_ends(&c_slice, n);
    let mut f = Mat::<f64>::zeros(d, n);
    let mut w_f = Mat::<f64>::zeros(d, n);
    let mut size = Col::<f64>::zeros(n);
    let mut start = 0;
    for (j, &end) in ends.iter().enumerate() {
        let piece = mat_vec(
            w.subcols(start, end - start),
            c.as_ref().subrows(start, end - start),
            Par::Seq,
        );
        let s = piece.norm_l2();
        if s <= EMPTY_BLOCK_TOL * coef_norm {
            return Err(PlsKitError::InvalidInput(format!(
                "factor {j} holds no signal: components {start}..{end} (0-based, end excluded) of the basis have size {s:e}"
            )));
        }
        for i in 0..d {
            f[(i, j)] = piece[i];
            w_f[(i, j)] = piece[i] / s;
        }
        size[j] = s;
        start = end;
    }
    Ok(FactorsOutput {
        f,
        w: w_f,
        size,
        coef: coef.to_owned(),
        pre_standardized,
    })
}

impl FactorsOutput {
    /// Rotate the directions by orthogonal `r` and re-project `coef`.
    ///
    /// With `W_rot = W R` and `a = W_rot' coef`: each column of `W_rot` (and
    /// of `R`) is negated where `a_i < 0`, so `a_i ≥ 0`; columns are then
    /// sorted by `a` descending (stable); `size_i = a_i` and
    /// `F[:, i] = W_rot[:, i] a_i`. `Σ_i F[:, i] = coef` and
    /// `Σ_i size_i² = ‖coef‖²` hold for every orthogonal `r`, because `coef`
    /// lies in the span of `W_rot`.
    ///
    /// Returns the new output and the `R` after sign pinning and sorting, so
    /// that `out.w = self.w · R` exactly as computed.
    ///
    /// # Shapes
    /// - `r`: `(n, n)` with `n = self.w.ncols()`, orthogonal
    ///
    /// # Errors
    ///
    /// - [`PlsKitError::ShapeMismatch`] — `r` is not `n × n`, or
    ///   `self.coef` has a length other than `self.w.nrows()`.
    /// - [`PlsKitError::InvalidInput`] — non-finite `self.w`, `self.coef`
    ///   or `r`, `max|R'R − I| > 1e-10`, `max|W'W − I| > 1e-10`, or
    ///   `‖coef − W W' coef‖ > 1e-10 ‖coef‖`.
    #[allow(clippy::many_single_char_names)] // w, r, a, d, n are the math names
    pub fn rotated(&self, r: MatRef<'_, f64>) -> PlsKitResult<(FactorsOutput, Mat<f64>)> {
        let (d, n) = self.w.shape();
        if r.shape() != (n, n) {
            return Err(PlsKitError::ShapeMismatch(format!(
                "R has shape {:?} but the factors have n={n}",
                r.shape()
            )));
        }
        if self.coef.nrows() != d {
            return Err(PlsKitError::ShapeMismatch(format!(
                "coef has length {} but W has {d} rows",
                self.coef.nrows()
            )));
        }
        if !self.w.is_all_finite() || !self.coef.is_all_finite() || !r.is_all_finite() {
            return Err(PlsKitError::InvalidInput(
                "W, coef or R contains non-finite values".into(),
            ));
        }
        let defect = orthonormality_defect(r);
        if defect > ORTHO_TOL {
            return Err(PlsKitError::InvalidInput(format!(
                "R is not orthogonal: max|R'R - I| = {defect:e}"
            )));
        }
        check_basis(self.w.as_ref(), self.coef.as_ref(), "W")?;

        let mut w_rot = mat_mul(self.w.as_ref(), r, Par::Seq);
        let mut r_pin = r.to_owned();
        let mut a = mat_vec(w_rot.transpose(), self.coef.as_ref(), Par::Seq);
        for i in 0..n {
            if a[i] < 0.0 {
                a[i] = -a[i];
                for row in 0..d {
                    w_rot[(row, i)] = 0.0 - w_rot[(row, i)];
                }
                for row in 0..n {
                    r_pin[(row, i)] = 0.0 - r_pin[(row, i)];
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&x, &y| a[y].total_cmp(&a[x]));

        let w_out = Mat::<f64>::from_fn(d, n, |i, j| w_rot[(i, order[j])]);
        let r_out = Mat::<f64>::from_fn(n, n, |i, j| r_pin[(i, order[j])]);
        let size = Col::<f64>::from_fn(n, |j| a[order[j]]);
        let f = Mat::<f64>::from_fn(d, n, |i, j| w_out[(i, j)] * size[j]);
        Ok((
            FactorsOutput {
                f,
                w: w_out,
                size,
                coef: self.coef.clone(),
                pre_standardized: self.pre_standardized,
            },
            r_out,
        ))
    }
}

/// Checks `max|W'W − I| ≤ 1e-10` and `‖coef − W W' coef‖ ≤ 1e-10 ‖coef‖`;
/// `what` names the basis in the error message. Returns `c = W' coef` and `‖coef‖`.
#[allow(clippy::many_single_char_names)] // w, c, d are the math names
fn check_basis(
    w: MatRef<'_, f64>,
    coef: ColRef<'_, f64>,
    what: &str,
) -> PlsKitResult<(Col<f64>, f64)> {
    let defect = orthonormality_defect(w);
    if defect > ORTHO_TOL {
        return Err(PlsKitError::InvalidInput(format!(
            "W columns are not orthonormal: max|W'W - I| = {defect:e}"
        )));
    }
    let c = mat_vec(w.transpose(), coef, Par::Seq);
    let wc = mat_vec(w, c.as_ref(), Par::Seq);
    let coef_norm = coef.norm_l2();
    let resid = Col::<f64>::from_fn(coef.nrows(), |i| coef[i] - wc[i]).norm_l2();
    if resid > ORTHO_TOL * coef_norm {
        return Err(PlsKitError::InvalidInput(format!(
            "coef is outside the span of {what}: ||coef - W c|| / ||coef|| = {:e}",
            resid / coef_norm
        )));
    }
    Ok((c, coef_norm))
}

/// `max|A'A − I|`, the departure of the columns of `a` from orthonormal.
pub(crate) fn orthonormality_defect(a: MatRef<'_, f64>) -> f64 {
    let g = mat_mul(a.transpose(), a, Par::Seq);
    let mut worst = 0.0_f64;
    for j in 0..g.ncols() {
        for i in 0..g.nrows() {
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((g[(i, j)] - want).abs());
        }
    }
    worst
}

/// Options for [`profiles_find_n`]. Defaults: `n_folds` 5, `n_repeats` 2,
/// `n_perm` 20, `pre_standardized` false, `seed` `None`.
#[derive(Debug, Clone, Copy)]
pub struct ProfilesFindNOpts {
    /// Folds per repeat, `2 ≤ n_folds ≤ N`.
    pub n_folds: usize,
    /// Repeats of the fold split, at least 1.
    pub n_repeats: usize,
    /// Column shuffles in the parallel analysis, at least 1.
    pub n_perm: usize,
    /// `x` is already standardized. Decides only whether the parallel
    /// analysis on the full data standardizes first; each fold's training
    /// rows are always standardized with their own means and scales.
    pub pre_standardized: bool,
    /// Seed. `None` draws one from OS entropy; the value used is returned.
    pub seed: Option<u64>,
}

impl Default for ProfilesFindNOpts {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_repeats: 2,
            n_perm: 20,
            pre_standardized: false,
            seed: None,
        }
    }
}

/// Output of [`profiles_find_n`].
#[derive(Debug, Clone)]
pub struct ProfilesFindNOutput {
    /// Chosen number of factors: 0 when the best mean held-out R² minus two
    /// of its standard errors is at most zero, otherwise the smallest `j`
    /// whose mean is at least that value.
    pub n_star: usize,
    /// The `j` searched: `1 ..= min(k_star, n_max)`.
    pub n_grid: Vec<usize>,
    /// Mean held-out R² over the usable folds at each `j`, length `n_grid.len()`.
    pub r2_mean: Vec<f64>,
    /// `sd / sqrt(folds)` (`ddof = 1`) of the held-out R² at each `j`.
    pub r2_se: Vec<f64>,
    /// Rotated components counted by the parallel analysis (`K*`), capped at
    /// the smallest training-fold rank.
    pub k_star: usize,
    /// Folds not used: a constant training column, or a held-out `SST = 0` to rounding.
    pub n_failed_folds: usize,
    /// The seed used.
    pub seed: u64,
}

/// Choose the number of factors `n` for [`factors`].
///
/// 1. `K*`: parallel analysis (Horn, 1965) on standardized `X`. Each column
///    is shuffled on its own `n_perm` times; the eigenvalues of the
///    correlation matrix `Xs'Xs / N` of each shuffle give a null for every
///    rank. `K*` counts the leading ranks whose observed eigenvalue is
///    strictly above the 95th percentile (R type 7) of the null at the same
///    rank, at least 1, and at most the smallest training-fold rank.
/// 2. `n_repeats` repeats of `n_folds` folds. In each fold the training
///    rows are standardized, the leading `K*` right singular vectors `V`
///    are rotated by varimax (Kaiser, 1958) to `Q`, and the axes are ordered
///    by `|Q' coef|`, largest first (stable), where `coef` is the
///    `pls1_fit` coefficient on the training rows at `k`.
/// 3. For `j = 1 ..= min(K*, n_max)`, `y` is regressed by ordinary least
///    squares on the scores of the top `j` axes of the training rows, and
///    held-out R² is `1 - SSE / SST`, both around the training mean of `y`.
/// 4. Mean over folds at each `j`, SE `sd / sqrt(folds)` with `ddof = 1`.
///    With `best` the `j` of the largest mean: `n_star = 0` when
///    `mean[best] - 2 SE[best] ≤ 0`, otherwise the smallest `j` with
///    `mean[j] ≥ mean[best] - 2 SE[best]`.
///
/// A fold with a constant training column is not used, nor is a fold whose
/// held-out R² is not finite at some `j` (`SST = 0` to rounding); both are counted in
/// `n_failed_folds`.
///
/// # Shapes
/// - `x`: `(N, D)`, `y`: `(N,)`, `k ≥ 1` PLS components for the fold fits
///
/// # Errors
///
/// - [`PlsKitError::DimensionMismatch`] — `y.nrows() != x.nrows()`.
/// - [`PlsKitError::NonFiniteInput`] — non-finite values in `x` or `y`.
/// - [`PlsKitError::InvalidInput`] — `x` has no columns; `n_max < 1`;
///   `n_folds` outside `[2, N]`; `n_repeats < 1`; `n_perm < 1`; fewer than
///   two usable folds.
/// - Errors of the fold fits (for example `k` too large) are returned as
///   they are.
/// - [`PlsKitError::Internal`] — an SVD or eigendecomposition failed.
// x, y, k, n, d, j are the math names; tr/te pairs are train/test.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn profiles_find_n(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_max: usize,
    opts: ProfilesFindNOpts,
) -> PlsKitResult<ProfilesFindNOutput> {
    let (n, d) = x.shape();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, d),
            y: y.nrows(),
        });
    }
    if d == 0 {
        return Err(PlsKitError::InvalidInput("X has no columns".into()));
    }
    if n_max < 1 {
        return Err(PlsKitError::InvalidInput("n_max must be at least 1".into()));
    }
    if opts.n_folds < 2 || opts.n_folds > n {
        return Err(PlsKitError::InvalidInput(format!(
            "n_folds={} is outside [2, N] with N={n}",
            opts.n_folds
        )));
    }
    if opts.n_repeats < 1 || opts.n_perm < 1 {
        return Err(PlsKitError::InvalidInput(
            "n_repeats and n_perm must be at least 1".into(),
        ));
    }
    if !x.is_all_finite() || !y.is_all_finite() {
        return Err(PlsKitError::NonFiniteInput);
    }

    let (seed, mut rng) = resolve_seed(opts.seed)?;
    let perm_seeds = child_seeds(&mut rng, opts.n_perm);
    let rep_seeds = child_seeds(&mut rng, opts.n_repeats);

    let k_raw = if opts.pre_standardized {
        parallel_k(x, &perm_seeds)?
    } else {
        parallel_k(standardize(x).0.as_ref(), &perm_seeds)?
    };

    // Pass 1: the fold's leading right singular vectors. Folds are kept for
    // pass 2 so that K* can be capped by the smallest rank before any
    // fold is scored.
    let mut n_failed_folds = 0;
    let mut kept: Vec<KeptFold> = Vec::new();
    for &rep_seed in &rep_seeds {
        let folds = fold_split(&permutation_from_seed(n, rep_seed), opts.n_folds);
        for (fi, te) in folds.iter().enumerate() {
            let tr: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|&(g, _)| g != fi)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            if constant_column(x, &tr) {
                n_failed_folds += 1;
                continue;
            }
            let (xs_tr, _, _) = standardize(row_subset(x, &tr).as_ref());
            let basis = pca_basis(xs_tr.as_ref(), true)?;
            let rank = basis.v.ncols();
            kept.push(KeptFold {
                tr,
                te: te.clone(),
                v: basis.v.as_ref().subcols(0, k_raw.min(rank)).to_owned(),
            });
        }
    }
    if kept.len() < 2 {
        return Err(PlsKitError::InvalidInput(format!(
            "{} usable fold(s) of {}: a constant training column in the others; at least 2 are needed",
            kept.len(),
            opts.n_folds * opts.n_repeats
        )));
    }
    // Every fold kept min(k_raw, its rank) columns, so the minimum is
    // min(k_raw, smallest rank), and at least 1 because a fold with a
    // nonconstant column has rank at least 1.
    let k_star = kept.iter().fold(k_raw, |m, f| m.min(f.v.ncols()));
    let j_max = k_star.min(n_max);

    // Pass 2: rotate, order, score.
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(kept.len());
    for fold in &kept {
        let x_tr = row_subset(x, &fold.tr);
        let y_tr = col_row_subset(y, &fold.tr);
        let (xs_tr, mean, scale) = standardize(x_tr.as_ref());
        let fit = pls1_fit_impl(
            x_tr.as_ref(),
            y_tr.as_ref(),
            KSpec::Fixed(k),
            None,
            FitOpts {
                check_n_eff: false,
                par: ParChoice::Seq,
                ..Default::default()
            },
        )?;
        let rot = rotate(
            fold.v.as_ref().subcols(0, k_star),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )?;
        let a = mat_vec(rot.w_rot.transpose(), fit.coef.as_ref(), Par::Seq);
        let mut order: Vec<usize> = (0..k_star).collect();
        order.sort_by(|&p, &q| a[q].abs().total_cmp(&a[p].abs()));
        let axes = Mat::<f64>::from_fn(d, j_max, |i, j| rot.w_rot[(i, order[j])]);
        let z_tr = mat_mul(xs_tr.as_ref(), axes.as_ref(), Par::Seq);
        let xs_te = standardize_apply(
            row_subset(x, &fold.te).as_ref(),
            mean.as_ref(),
            scale.as_ref(),
        );
        let z_te = mat_mul(xs_te.as_ref(), axes.as_ref(), Par::Seq);
        let r2 = heldout_r2(
            z_tr.as_ref(),
            y_tr.as_ref(),
            z_te.as_ref(),
            col_row_subset(y, &fold.te).as_ref(),
        );
        if r2.iter().all(|v| v.is_finite()) {
            rows.push(r2);
        } else {
            n_failed_folds += 1;
        }
    }
    if rows.len() < 2 {
        return Err(PlsKitError::InvalidInput(format!(
            "{} fold(s) with a finite held-out R²: at least 2 are needed",
            rows.len()
        )));
    }
    let (n_star, r2_mean, r2_se) = pick_n(&rows);
    Ok(ProfilesFindNOutput {
        n_star,
        n_grid: (1..=j_max).collect(),
        r2_mean,
        r2_se,
        k_star,
        n_failed_folds,
        seed,
    })
}

/// A training fold that passed the constant-column check: its rows and its
/// leading right singular vectors `V`, shape `(D, min(k_raw, rank))`, where
/// `k_raw` is the parallel-analysis count before the rank cap.
struct KeptFold {
    tr: Vec<usize>,
    te: Vec<usize>,
    v: Mat<f64>,
}

/// Eigenvalues of the correlation matrix `Xs'Xs / N` of standardized `xs`
/// (`ddof = 0` scales), largest first. When `N < D` they come from the
/// `N × N` matrix `Xs Xs' / N`, which has the same nonzero eigenvalues.
fn corr_eigenvalues(xs: MatRef<'_, f64>) -> PlsKitResult<Vec<f64>> {
    let (n, d) = xs.shape();
    let g = if d <= n {
        mat_mul(xs.transpose(), xs, Par::Seq)
    } else {
        mat_mul(xs, xs.transpose(), Par::Seq)
    };
    // shortcut: the eigenvectors are computed and dropped; add a
    // values-only variant to `linalg` when D is large enough to notice.
    let (lambda, _) = self_adjoint_eigen(g.as_ref(), Par::Seq)
        .map_err(|e| PlsKitError::Internal(format!("eigendecomposition failed: {e:?}")))?;
    Ok((0..lambda.nrows())
        .rev()
        .map(|i| lambda[i] / n as f64)
        .collect())
}

/// Parallel analysis (Horn, 1965): the number of leading eigenvalues of the
/// correlation matrix of `xs` that are strictly above the 95th percentile of
/// the shuffled-column eigenvalue at the same rank, at least 1. Each seed in
/// `perm_seeds` shuffles every column on its own; a shuffle keeps a column's
/// mean and scale, so `xs` needs no re-standardizing.
fn parallel_k(xs: MatRef<'_, f64>, perm_seeds: &[u64]) -> PlsKitResult<usize> {
    let (n, d) = xs.shape();
    let observed = corr_eigenvalues(xs)?;
    let mut null: Vec<Vec<f64>> = vec![Vec::with_capacity(perm_seeds.len()); observed.len()];
    let mut shuffled = Mat::<f64>::zeros(n, d);
    for &seed in perm_seeds {
        let mut rng = child_rng(seed);
        for j in 0..d {
            let p = permute_indices(n, &mut rng);
            for i in 0..n {
                shuffled[(i, j)] = xs[(p[i], j)];
            }
        }
        for (r, v) in corr_eigenvalues(shuffled.as_ref())?.into_iter().enumerate() {
            null[r].push(v);
        }
    }
    let mut run = 0;
    for (r, &obs) in observed.iter().enumerate() {
        null[r].sort_by(f64::total_cmp);
        if obs > empirical_quantile(&null[r], 0.95) {
            run += 1;
        } else {
            break;
        }
    }
    Ok(run.max(1))
}

/// True when some column of `x`, over the rows `rows`, is constant to
/// rounding: the rule the standardizer uses.
fn constant_column(x: MatRef<'_, f64>, rows: &[usize]) -> bool {
    (0..x.ncols())
        .any(|j| scaled_moments(rows.len(), |i| x[(rows[i], j)], None).is_constant(rows.len()))
}

/// Held-out R² for each `j = 1 ..= J` (`J = z_tr.ncols()`): ordinary least
/// squares of `y_tr` on the first `j` columns of `z_tr`, around the training
/// mean `ȳ`. With `G = Z_tr'Z_tr` and `b = Z_tr'(y_tr − ȳ)`, `β_j` solves
/// `G[..j, ..j] β_j = b[..j]`; then `SSE = Σ (y_te − ȳ − Z_te[:, ..j] β_j)²`,
/// `SST = Σ (y_te − ȳ)²` and R² is `1 − SSE / SST`. The columns of `z_tr`
/// must have mean zero (scores of centered rows), so `ȳ` is the intercept.
/// `SST` counts as 0 when `y_te` equals `ȳ` to rounding (`constant_to_rounding`),
/// and R² is then not finite.
#[allow(clippy::similar_names)] // tr/te are train/test
fn heldout_r2(
    z_tr: MatRef<'_, f64>,
    y_tr: ColRef<'_, f64>,
    z_te: MatRef<'_, f64>,
    y_te: ColRef<'_, f64>,
) -> Vec<f64> {
    let ybar = (0..y_tr.nrows()).map(|i| y_tr[i]).sum::<f64>() / y_tr.nrows() as f64;
    let yc = Col::<f64>::from_fn(y_tr.nrows(), |i| y_tr[i] - ybar);
    let g = mat_mul(z_tr.transpose(), z_tr, Par::Seq);
    let b = mat_vec(z_tr.transpose(), yc.as_ref(), Par::Seq);
    let sst: f64 = (0..y_te.nrows()).map(|i| (y_te[i] - ybar).powi(2)).sum();
    let sq: f64 = (0..y_te.nrows()).map(|i| y_te[i].powi(2)).sum();
    // The rounding here comes from `ȳ`, so the band follows the training rows.
    let sst = if constant_to_rounding(sst, sq, y_tr.nrows()) {
        0.0
    } else {
        sst
    };
    (1..=z_tr.ncols())
        .map(|j| {
            let mut beta = Mat::<f64>::from_fn(j, 1, |i, _| b[i]);
            lu_solve_in_place(g.as_ref().submatrix(0, 0, j, j), beta.as_mut());
            let pred = mat_vec(z_te.subcols(0, j), beta.as_ref().col(0), Par::Seq);
            let sse: f64 = (0..y_te.nrows())
                .map(|i| (y_te[i] - ybar - pred[i]).powi(2))
                .sum();
            1.0 - sse / sst
        })
        .collect()
}

/// Mean and SE over folds at each `j` (`rows[fold][j]`, all finite, at least
/// two folds), and the choice: `best` is the `j` of the largest mean (first
/// on ties); `n_star = 0` when `mean[best] − 2 SE[best] ≤ 0`, otherwise the
/// smallest 1-based `j` with `mean[j] ≥ mean[best] − 2 SE[best]`. SE is
/// `sd / sqrt(folds)` with `ddof = 1`.
fn pick_n(rows: &[Vec<f64>]) -> (usize, Vec<f64>, Vec<f64>) {
    let folds = rows.len() as f64;
    let width = rows[0].len();
    let mean: Vec<f64> = (0..width)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / folds)
        .collect();
    let se: Vec<f64> = (0..width)
        .map(|j| {
            let ss: f64 = rows.iter().map(|r| (r[j] - mean[j]).powi(2)).sum();
            (ss / (folds - 1.0)).sqrt() / folds.sqrt()
        })
        .collect();
    let best = (1..width).fold(0, |b, j| if mean[j] > mean[b] { j } else { b });
    let bar = mean[best] - 2.0 * se[best];
    let n_star = if bar <= 0.0 {
        0
    } else {
        // `best` itself satisfies mean ≥ bar because se ≥ 0.
        1 + mean
            .iter()
            .position(|&m| m >= bar)
            .expect("the best j passes its own bar")
    };
    (n_star, mean, se)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::thin_svd;
    use crate::test_support::{bits, unif, unif_col};

    const D: usize = 20;
    const K: usize = 7;

    /// Orthonormal `W` (D × K) and `coef = W c` with a fixed `c`.
    fn basis_and_coef() -> (Mat<f64>, Col<f64>) {
        let w = thin_svd(unif(D, K, 11).as_ref(), Par::Seq).unwrap().u;
        let c = Col::<f64>::from_fn(K, |i| [3.0, -2.0, 1.5, 1.0, -0.7, 0.5, 0.2][i]);
        let coef = mat_vec(w.as_ref(), c.as_ref(), Par::Seq);
        (w, coef)
    }

    /// Orthogonal `n × n`: Givens rotations on a few index pairs.
    #[allow(clippy::many_single_char_names)] // Givens c, s are the math names
    fn givens(n: usize) -> Mat<f64> {
        let mut r = Mat::<f64>::identity(n, n);
        for (p, q, th) in [(0, n - 1, 0.7), (0, 1, -1.1), (n - 1, n / 2, 0.4)] {
            if p == q || p >= n || q >= n {
                continue;
            }
            let (c, s) = (f64::cos(th), f64::sin(th));
            for row in 0..n {
                let (a, b) = (r[(row, p)], r[(row, q)]);
                r[(row, p)] = c * a - s * b;
                r[(row, q)] = s * a + c * b;
            }
        }
        r
    }

    fn assert_identities(out: &FactorsOutput, coef: ColRef<'_, f64>) {
        let (d, n) = out.f.shape();
        let cn = coef.norm_l2();
        for i in 0..d {
            let sum: f64 = (0..n).map(|j| out.f[(i, j)]).sum();
            assert!((sum - coef[i]).abs() <= 1e-12 * cn, "sum row {i}");
        }
        assert!(orthonormality_defect(out.w.as_ref()) <= 1e-12, "W'W");
        let ss: f64 = (0..n).map(|j| out.size[j] * out.size[j]).sum();
        assert!((ss - cn * cn).abs() <= 1e-12 * cn * cn, "sizes");
        for j in 0..n {
            for i in 0..d {
                assert!(
                    (out.f[(i, j)] - out.w[(i, j)] * out.size[j]).abs() <= 1e-12 * cn,
                    "F = W size"
                );
            }
        }
    }

    #[test]
    fn block_ends_cut_the_worked_examples() {
        assert_eq!(block_ends(&[1., 1., 1., 1., 1., 1., 1.], 3), [3, 5, 7]);
        assert_eq!(block_ends(&[1., 1., 1., 100.], 3), [2, 3, 4]); // upper limit applies
        assert_eq!(block_ends(&[5., 1., 1.], 3), [1, 2, 3]); // lower limit applies
        // S(2) = 1/2 exactly, so the first block ends at 2, not 3.
        assert_eq!(block_ends(&[2., 3., 3., 2.], 2), [2, 4]);
    }

    #[test]
    fn block_ends_do_not_depend_on_magnitude() {
        for c in [
            [1.0; 7],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0],
        ] {
            for n in [2, 3, 5] {
                let want = block_ends(&c, n);
                for scale in [1e-200, 1e200] {
                    let scaled: Vec<f64> = c.iter().map(|v| v * scale).collect();
                    assert_eq!(block_ends(&scaled, n), want, "{c:?} n={n} scale={scale:e}");
                }
            }
        }
    }

    #[test]
    fn sum_orthonormality_and_sizes_hold() {
        let (w, coef) = basis_and_coef();
        for n in [3, 1, K] {
            let out = split_on_basis(w.as_ref(), coef.as_ref(), n, false).unwrap();
            assert_eq!(out.f.shape(), (D, n));
            assert_identities(&out, coef.as_ref());
        }
    }

    #[test]
    fn identities_hold_after_a_hand_built_rotation() {
        let (w, coef) = basis_and_coef();
        for n in [3, 1, K] {
            let out = split_on_basis(w.as_ref(), coef.as_ref(), n, false).unwrap();
            let (rot, _) = out.rotated(givens(n).as_ref()).unwrap();
            assert_identities(&rot, coef.as_ref());
        }
    }

    #[test]
    fn flipping_a_rotation_column_leaves_f_unchanged() {
        let (w, coef) = basis_and_coef();
        for n in [3, K] {
            let out = split_on_basis(w.as_ref(), coef.as_ref(), n, false).unwrap();
            let r = givens(n);
            let mut r_flip = r.clone();
            for row in 0..n {
                r_flip[(row, 0)] = -r_flip[(row, 0)];
            }
            let (a, ra) = out.rotated(r.as_ref()).unwrap();
            let (b, rb) = out.rotated(r_flip.as_ref()).unwrap();
            for j in 0..n {
                assert!((a.size[j] - b.size[j]).abs() <= 1e-12);
                for i in 0..D {
                    assert!((a.f[(i, j)] - b.f[(i, j)]).abs() <= 1e-12, "F[{i},{j}]");
                }
                for i in 0..n {
                    assert!((ra[(i, j)] - rb[(i, j)]).abs() <= 1e-12, "R[{i},{j}]");
                }
            }
        }
    }

    #[test]
    fn rotated_is_sorted_nonnegative_and_r_is_exact() {
        let (w, coef) = basis_and_coef();
        for n in [3, K] {
            let out = split_on_basis(w.as_ref(), coef.as_ref(), n, false).unwrap();
            let (rot, r) = out.rotated(givens(n).as_ref()).unwrap();
            for j in 0..n {
                assert!(rot.size[j] >= 0.0);
                if j > 0 {
                    assert!(rot.size[j - 1] >= rot.size[j], "size order");
                }
            }
            let back = mat_mul(out.w.as_ref(), r.as_ref(), Par::Seq);
            for j in 0..n {
                for i in 0..D {
                    assert!((back[(i, j)] - rot.w[(i, j)]).abs() <= 1e-12, "W R");
                }
            }
        }
    }

    #[test]
    fn one_factor_after_rotation_is_coef() {
        let (w, coef) = basis_and_coef();
        let out = split_on_basis(w.as_ref(), coef.as_ref(), 1, false).unwrap();
        for i in 0..D {
            assert!((out.f[(i, 0)] - coef[i]).abs() <= 1e-12);
        }
        let (rot, _) = out.rotated(Mat::<f64>::identity(1, 1).as_ref()).unwrap();
        for i in 0..D {
            assert!((rot.f[(i, 0)] - coef[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn output_ignores_basis_column_signs() {
        let (w, coef) = basis_and_coef();
        let base = split_on_basis(w.as_ref(), coef.as_ref(), 3, false).unwrap();
        let w_neg = Mat::<f64>::from_fn(D, K, |i, j| {
            if j == 1 || j == 3 {
                -w[(i, j)]
            } else {
                w[(i, j)]
            }
        });
        let flipped = split_on_basis(w_neg.as_ref(), coef.as_ref(), 3, false).unwrap();
        for j in 0..3 {
            for i in 0..D {
                assert!((base.f[(i, j)] - flipped.f[(i, j)]).abs() <= 1e-12);
                assert!((base.w[(i, j)] - flipped.w[(i, j)]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn factors_error_table() {
        let (w, coef) = basis_and_coef();
        let invalid =
            |r: PlsKitResult<FactorsOutput>| matches!(r, Err(PlsKitError::InvalidInput(_)));

        assert!(
            invalid(split_on_basis(w.as_ref(), coef.as_ref(), 0, false)),
            "n = 0"
        );
        assert!(
            invalid(split_on_basis(w.as_ref(), coef.as_ref(), K + 1, false)),
            "n = K + 1"
        );

        let w_scaled = Mat::<f64>::from_fn(D, K, |i, j| w[(i, j)] * 1.001);
        assert!(
            invalid(split_on_basis(w_scaled.as_ref(), coef.as_ref(), 3, false)),
            "non-orthonormal"
        );

        let mut off_span = coef.clone();
        off_span[0] += 1e-3;
        match split_on_basis(w.as_ref(), off_span.as_ref(), 3, false) {
            Err(PlsKitError::InvalidInput(m)) => assert!(m.contains("basis of X"), "{m}"),
            other => panic!("outside span: {other:?}"),
        }

        let e = Mat::<f64>::from_fn(5, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let c = Col::<f64>::from_fn(5, |i| if i == 0 { 1.0 } else { 0.0 });
        match split_on_basis(e.as_ref(), c.as_ref(), 3, false) {
            Err(PlsKitError::InvalidInput(msg)) => {
                assert!(msg.contains("1..2"), "message names the range: {msg}");
                assert!(msg.contains("0-based, end excluded"), "convention: {msg}");
            }
            other => panic!("zero block: {other:?}"),
        }
    }

    /// `coef = Xs' a` for a seeded `x` (50 × 10) and `a`, so `coef` lies in the
    /// row space of `Xs`.
    fn x_and_coef() -> (Mat<f64>, Col<f64>) {
        let x = unif(50, 10, 21);
        let xs = standardize(x.as_ref()).0;
        let coef = mat_vec(xs.transpose(), unif_col(50, 22).as_ref(), Par::Seq);
        (x, coef)
    }

    #[test]
    fn factors_matches_the_basis_split_on_the_same_x() {
        let (x, coef) = x_and_coef();
        let xs = standardize(x.as_ref()).0;
        let got = factors(coef.as_ref(), x.as_ref(), 3, false).unwrap();
        let v = pca_basis(xs.as_ref(), true).unwrap().v;
        let want = split_on_basis(v.as_ref(), coef.as_ref(), 3, false).unwrap();
        for j in 0..3 {
            assert!((got.size[j] - want.size[j]).abs() <= 1e-12, "size {j}");
            for i in 0..10 {
                assert!((got.f[(i, j)] - want.f[(i, j)]).abs() <= 1e-12, "f");
                assert!((got.w[(i, j)] - want.w[(i, j)]).abs() <= 1e-12, "w");
            }
        }
    }

    #[test]
    fn left_out_variables_hold_exact_zeros() {
        let (x, mut coef) = x_and_coef();
        coef[3] = 0.0;
        coef[7] = 0.0;
        let out = factors(coef.as_ref(), x.as_ref(), 3, false).unwrap();
        for j in 0..3 {
            for i in [3, 7] {
                assert_eq!(out.f[(i, j)], 0.0);
                assert_eq!(out.w[(i, j)], 0.0);
            }
        }
        assert_identities(&out, coef.as_ref());
    }

    #[test]
    fn n_is_bounded_by_the_rank_after_leaving_out() {
        let (x, full) = x_and_coef();
        let coef = Col::<f64>::from_fn(10, |i| if i < 4 { full[i] } else { 0.0 });
        assert!(factors(coef.as_ref(), x.as_ref(), 4, false).is_ok());
        match factors(coef.as_ref(), x.as_ref(), 5, false) {
            Err(PlsKitError::InvalidInput(m)) => assert!(m.contains("K=4"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn coef_outside_the_row_space_lands_in_the_last_factor() {
        // Standardized 20 x 200 X has rank 19, so a generic coef is mostly
        // outside its row space: K = 19 + 1 with the residual column.
        let x = unif(20, 200, 31);
        let coef = unif_col(200, 32);
        let xs = standardize(x.as_ref()).0;
        for n in [2, 20] {
            let out = factors(coef.as_ref(), x.as_ref(), n, false).unwrap();
            assert_identities(&out, coef.as_ref());
        }
        // At n = K every block is one column, so the last factor is the residual.
        let out = factors(coef.as_ref(), x.as_ref(), 20, false).unwrap();
        let unseen = mat_vec(xs.as_ref(), out.w.as_ref().col(19), Par::Seq);
        assert!(
            unseen.norm_l2() <= 1e-10 * xs.norm_l2(),
            "{}",
            unseen.norm_l2()
        );
        match factors(coef.as_ref(), x.as_ref(), 21, false) {
            Err(PlsKitError::InvalidInput(m)) => assert!(m.contains("K=20"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pre_standardized_uses_x_as_given() {
        let raw = unif(50, 10, 41);
        let x = Mat::<f64>::from_fn(50, 10, |i, j| {
            raw[(i, j)] * [1.0, 10.0, 100.0, 1000.0][j % 4]
        });
        let coef = mat_vec(x.transpose(), unif_col(50, 42).as_ref(), Par::Seq);
        let v = pca_basis(x.as_ref(), true).unwrap().v;
        let want = split_on_basis(v.as_ref(), coef.as_ref(), 3, true).unwrap();
        let got = factors(coef.as_ref(), x.as_ref(), 3, true).unwrap();
        let other = factors(coef.as_ref(), x.as_ref(), 3, false).unwrap();
        let mut gap = 0.0_f64;
        for j in 0..3 {
            assert!((got.size[j] - want.size[j]).abs() <= 1e-12, "size {j}");
            for i in 0..10 {
                assert!((got.f[(i, j)] - want.f[(i, j)]).abs() <= 1e-12, "f");
                assert!((got.w[(i, j)] - want.w[(i, j)]).abs() <= 1e-12, "w");
                gap = gap.max((got.f[(i, j)] - other.f[(i, j)]).abs());
            }
        }
        assert!(gap > 1e-6, "gap {gap:e}");
        assert!(got.pre_standardized && want.pre_standardized);
        assert!(!other.pre_standardized);
    }

    #[test]
    fn factors_checks_come_first() {
        let message = |r: PlsKitResult<FactorsOutput>| match r {
            Err(PlsKitError::InvalidInput(m)) => m,
            other => panic!("{other:?}"),
        };
        let (mut x, coef) = x_and_coef();
        let mut left_out = coef.clone();
        left_out[2] = 0.0;
        x[(5, 2)] = f64::NAN;
        let m = message(factors(left_out.as_ref(), x.as_ref(), 3, false));
        assert!(m.contains("X contains non-finite"), "NaN in x: {m}");

        let (x, coef) = x_and_coef();
        let mut coef_inf = coef.clone();
        coef_inf[1] = f64::INFINITY;
        let m = message(factors(coef_inf.as_ref(), x.as_ref(), 3, false));
        assert!(m.contains("coef contains non-finite"), "inf coef: {m}");

        let zero = Col::<f64>::zeros(10);
        let m = message(factors(zero.as_ref(), x.as_ref(), 3, false));
        assert!(m.contains("coef is zero"), "zero coef: {m}");

        let short = Col::<f64>::zeros(9);
        assert!(matches!(
            factors(short.as_ref(), x.as_ref(), 3, false),
            Err(PlsKitError::ShapeMismatch(_))
        ));

        let no_rows = Mat::<f64>::zeros(0, 10);
        message(factors(coef.as_ref(), no_rows.as_ref(), 3, false));

        // Every column left in is constant; the one that varies is left out.
        let constant = Mat::<f64>::from_fn(50, 10, |i, j| if j == 3 { x[(i, j)] } else { 0.1 });
        let mut left_out = coef.clone();
        left_out[3] = 0.0;
        let m = message(factors(left_out.as_ref(), constant.as_ref(), 3, false));
        assert!(m.contains("constant"), "constant columns: {m}");
    }

    #[test]
    fn rotated_keeps_left_out_rows_at_positive_zero() {
        let (x, mut coef) = x_and_coef();
        coef[3] = 0.0;
        coef[7] = 0.0;
        let out = factors(coef.as_ref(), x.as_ref(), 3, false).unwrap();
        // A column flip with a negative a_i negates every row, zeros included.
        let mut flip = givens(3);
        for row in 0..3 {
            flip[(row, 0)] = -flip[(row, 0)];
        }
        for r in [givens(3), flip] {
            let (rot, _) = out.rotated(r.as_ref()).unwrap();
            for j in 0..3 {
                for i in [3, 7] {
                    assert_eq!(rot.f[(i, j)].to_bits(), 0.0_f64.to_bits(), "f");
                    assert_eq!(rot.w[(i, j)].to_bits(), 0.0_f64.to_bits(), "w");
                }
            }
        }
    }

    #[test]
    fn rotated_carries_pre_standardized() {
        let (x, coef) = x_and_coef();
        for flag in [false, true] {
            let out = factors(coef.as_ref(), x.as_ref(), 3, flag).unwrap();
            let (rot, _) = out.rotated(givens(3).as_ref()).unwrap();
            assert_eq!(rot.pre_standardized, flag);
        }
    }

    #[test]
    fn rotated_rejects_bad_shapes() {
        let (w, coef) = basis_and_coef();
        let out = split_on_basis(w.as_ref(), coef.as_ref(), 3, false).unwrap();
        let shape = |r: PlsKitResult<(FactorsOutput, Mat<f64>)>| {
            matches!(r, Err(PlsKitError::ShapeMismatch(_)))
        };
        assert!(shape(out.rotated(Mat::<f64>::identity(2, 2).as_ref())));
        assert!(shape(out.rotated(Mat::<f64>::zeros(3, 4).as_ref())));

        let mut bad = out.clone();
        bad.coef = Col::<f64>::zeros(D + 1);
        assert!(shape(bad.rotated(Mat::<f64>::identity(3, 3).as_ref())));

        let skew = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.0 });
        assert!(matches!(
            out.rotated(skew.as_ref()),
            Err(PlsKitError::InvalidInput(_))
        ));
    }

    /// `n` rows, 5 traits of 12 columns each (trait plus noise, equal
    /// variances). `y` is `sqrt(r2/m)` times the sum of the first `m` traits
    /// plus noise, so `m = 0` is pure noise.
    #[allow(clippy::many_single_char_names)] // n, m, t, e, x, y, a are the math names
    fn traits(n: usize, m: usize, r2: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        let sd = 3.0_f64.sqrt(); // unif on [-1, 1) has variance 1/3
        let t = unif(n, 5, seed);
        let e = unif(n, 60, seed + 1);
        let x = Mat::<f64>::from_fn(n, 60, |i, j| sd * (t[(i, j / 12)] + 0.5 * e[(i, j)]));
        let ey = unif_col(n, seed + 2);
        let a = (r2 / m.max(1) as f64).sqrt();
        let y = Col::<f64>::from_fn(n, |i| {
            a * sd * (0..m).map(|g| t[(i, g)]).sum::<f64>() + (1.0 - r2).sqrt() * sd * ey[i]
        });
        (x, y)
    }

    fn opts(seed: u64) -> ProfilesFindNOpts {
        ProfilesFindNOpts {
            seed: Some(seed),
            ..Default::default()
        }
    }

    /// `x` with one more column that is nonzero on row `row` only.
    fn with_one_row_column(x: &Mat<f64>, row: usize) -> Mat<f64> {
        let d = x.ncols();
        Mat::<f64>::from_fn(x.nrows(), d + 1, |i, j| {
            if j < d {
                x[(i, j)]
            } else {
                f64::from(u8::from(i == row))
            }
        })
    }

    #[test]
    fn pick_applies_the_two_se_rule() {
        // Two folds, two j: means .2 and .3, SE .1, threshold .1, so j = 1 passes.
        assert_eq!(pick_n(&[vec![0.1, 0.2], vec![0.3, 0.4]]).0, 1);
        // Best mean minus 2 SE is at or below zero.
        assert_eq!(pick_n(&[vec![-0.1, 0.0], vec![0.1, 0.2]]).0, 0);
        // Three folds: means 1/3 and 2/3; sd of j = 2 is sqrt(1/3), so its SE is 1/3.
        let (_, mean, se) = pick_n(&[vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        assert!((mean[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((se[1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn heldout_r2_matches_hand_values() {
        // Train z = (1, -1, 1, -1), y = (3, 1, 3, 1): ybar 2, beta 1.
        // Test z = (1, -1), y = (3, 2): SSE 1, SST 1, around the training mean.
        let ztr = Mat::<f64>::from_fn(4, 1, |i, _| if i % 2 == 0 { 1.0 } else { -1.0 });
        let ytr = Col::<f64>::from_fn(4, |i| if i % 2 == 0 { 3.0 } else { 1.0 });
        let zte = Mat::<f64>::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { -1.0 });
        let yte = Col::<f64>::from_fn(2, |i| if i == 0 { 3.0 } else { 2.0 });
        let r2 = heldout_r2(ztr.as_ref(), ytr.as_ref(), zte.as_ref(), yte.as_ref());
        assert_eq!(r2.len(), 1);
        assert!(r2[0].abs() < 1e-12, "{}", r2[0]);
    }

    #[test]
    fn heldout_r2_solves_the_leading_block_for_each_j() {
        // Correlated centered scores z1 = (1, 1, -1, -1), z2 = (1, 0, 0, -1),
        // y = z1 + z2 = (2, 1, -1, -2), ybar 0. G = [[4, 2], [2, 2]], b = (6, 4).
        // j = 1: beta = 6/4 = 1.5. j = 2: beta = (1, 1), which differs from the
        // first entry of a truncated j = 2 solution.
        let zs = [[1.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, -1.0]];
        let ztr = Mat::<f64>::from_fn(4, 2, |i, j| zs[i][j]);
        let ytr = Col::<f64>::from_fn(4, |i| [2.0, 1.0, -1.0, -2.0][i]);
        // Test rows (1, 1) and (-1, 0), y = (3, 0).
        let zte = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 1.0], [-1.0, 0.0]][i][j]);
        let yte = Col::<f64>::from_fn(2, |i| [3.0, 0.0][i]);
        let r2 = heldout_r2(ztr.as_ref(), ytr.as_ref(), zte.as_ref(), yte.as_ref());
        // j = 1: predictions (1.5, -1.5), SSE 4.5, SST 9. j = 2: (2, -1), SSE 2.
        assert!((r2[0] - 0.5).abs() < 1e-12, "{r2:?}");
        assert!((r2[1] - 7.0 / 9.0).abs() < 1e-12, "{r2:?}");
    }

    #[test]
    fn heldout_sst_zero_is_not_finite() {
        let ztr = Mat::<f64>::from_fn(4, 1, |i, _| if i % 2 == 0 { 1.0 } else { -1.0 });
        let ytr = Col::<f64>::from_fn(4, |i| if i % 2 == 0 { 3.0 } else { 1.0 });
        let yte = Col::<f64>::from_fn(2, |_| 2.0); // the training mean
        // SSE > 0 gives -inf, SSE = 0 gives NaN.
        let off = Mat::<f64>::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { -1.0 });
        let on = Mat::<f64>::zeros(2, 1);
        for zte in [off, on] {
            let r2 = heldout_r2(ztr.as_ref(), ytr.as_ref(), zte.as_ref(), yte.as_ref());
            assert!(!r2[0].is_finite(), "{}", r2[0]);
        }
    }

    #[test]
    fn constant_y_fails_whatever_its_mean_rounds_to() {
        // The mean of 80 copies of 0.1 rounds away from 0.1, so SST is tiny, not 0.
        let (x, _) = traits(100, 3, 0.5, 8);
        for v in [3.0, 0.1, 1.0 / 3.0] {
            let y = Col::<f64>::from_fn(100, |_| v);
            let r = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(7));
            assert!(matches!(r, Err(PlsKitError::InvalidInput(_))), "{v}: {r:?}");
        }
    }

    #[test]
    fn five_traits_three_predictive_returns_three() {
        let (x, y) = traits(1000, 3, 0.5, 1);
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(7)).unwrap();
        assert_eq!(out.n_star, 3);
        assert_eq!(out.k_star, 5);
        assert_eq!(out.n_grid, [1, 2, 3, 4, 5]);
        assert_eq!(out.n_failed_folds, 0);
        assert_eq!(out.seed, 7);
    }

    #[test]
    fn pure_noise_returns_zero() {
        let (x, y) = traits(1000, 0, 0.0, 2);
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(7)).unwrap();
        assert_eq!(out.n_star, 0);
        assert!(out.k_star >= 1);
    }

    #[test]
    fn n_max_limits_the_search() {
        let (x, y) = traits(1000, 3, 0.5, 1);
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 2, opts(7)).unwrap();
        assert_eq!(out.n_star, 2);
        assert_eq!(out.n_grid, [1, 2]);
        assert_eq!(out.k_star, 5);
    }

    #[test]
    fn constant_training_column_counts_as_failed() {
        let (x, y) = traits(200, 3, 0.5, 3);
        let x = with_one_row_column(&x, 17);
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(7)).unwrap();
        // The fold holding row 17 out has an all-zero training column: one per repeat.
        assert_eq!(out.n_failed_folds, 2);
        assert_eq!(out.n_grid.len(), out.k_star.min(9));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn same_seed_same_result() {
        let (x, y) = traits(300, 2, 0.5, 4);
        let run = |seed| profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(seed)).unwrap();
        let (a, b, c) = (run(5), run(5), run(6));
        assert_eq!(a.n_star, b.n_star);
        assert_eq!(a.n_grid, b.n_grid);
        assert_eq!(bits(&a.r2_mean), bits(&b.r2_mean));
        assert_eq!(bits(&a.r2_se), bits(&b.r2_se));
        assert_eq!(
            (a.k_star, a.n_failed_folds, a.seed),
            (b.k_star, b.n_failed_folds, b.seed)
        );
        assert_ne!(bits(&a.r2_mean), bits(&c.r2_mean));
    }

    #[test]
    fn seeded_stream_is_pinned() {
        // Recorded from this crate's own output, not an outside reference.
        let (x, y) = traits(300, 2, 0.5, 4);
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(5)).unwrap();
        let mean = [
            0.201_421_225_481_178_83,
            0.450_848_690_320_225_84,
            0.452_152_166_016_964_5,
            0.447_034_765_180_479_5,
            0.442_415_635_240_151_7,
        ];
        let se = [
            0.025_719_111_159_345_522,
            0.018_493_690_229_511_556,
            0.018_476_751_957_864_513,
            0.017_742_420_984_952_607,
            0.014_851_552_999_753_449,
        ];
        assert_eq!(out.r2_mean.len(), mean.len());
        for (a, b) in out.r2_mean.iter().zip(mean).chain(out.r2_se.iter().zip(se)) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    /// `N = 30`, `D = 100`, rank 18: `U diag(s) V'` with `s²` the mean
    /// eigenvalue spectrum of shuffled data, kept on the top 18 ranks. After
    /// column standardization every kept rank sits above the shuffled
    /// spectrum, so the parallel analysis passes more than 14 ranks.
    fn wide_low_rank() -> Mat<f64> {
        let (n, d, rank) = (30, 100, 18);
        let mut null = vec![0.0; n];
        for seed in 0..20 {
            let xs = standardize(unif(n, d, 100 + seed).as_ref()).0;
            for (r, v) in corr_eigenvalues(xs.as_ref()).unwrap().iter().enumerate() {
                null[r] += v / 20.0;
            }
        }
        let u = thin_svd(unif(n, rank, 1).as_ref(), Par::Seq).unwrap().u;
        let v = thin_svd(unif(d, rank, 2).as_ref(), Par::Seq).unwrap().u;
        let us = Mat::<f64>::from_fn(n, rank, |i, r| u[(i, r)] * null[r].sqrt());
        mat_mul(us.as_ref(), v.transpose(), Par::Seq)
    }

    #[test]
    fn wide_data_caps_k_star_at_the_training_rank() {
        let x = wide_low_rank();
        let y = unif_col(30, 6);
        // Two folds of 15 rows: a training fold has centered rank at most 14.
        let o = ProfilesFindNOpts {
            n_folds: 2,
            ..opts(7)
        };
        let k_raw = parallel_k(standardize(x.as_ref()).0.as_ref(), &[1, 2, 3, 4, 5]).unwrap();
        assert!(k_raw > 14, "parallel analysis passes {k_raw} ranks");
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 20, o).unwrap();
        assert_eq!(out.k_star, 14);
        assert_eq!(out.n_grid.len(), 14);
    }

    #[test]
    fn wide_data_runs() {
        let x = unif(30, 100, 5);
        let y = unif_col(30, 6);
        let out = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(7)).unwrap();
        assert_eq!(out.n_grid.len(), out.k_star.min(9));
    }

    #[test]
    fn pre_standardized_input_gives_the_same_result() {
        let (x, y) = traits(300, 2, 0.5, 4);
        let xs = standardize(x.as_ref()).0;
        let raw = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(5)).unwrap();
        let pre = ProfilesFindNOpts {
            pre_standardized: true,
            ..opts(5)
        };
        let std = profiles_find_n(xs.as_ref(), y.as_ref(), 1, 9, pre).unwrap();
        assert_eq!((raw.n_star, raw.k_star), (std.n_star, std.k_star));
        // Standardizing twice differs from once only by rounding.
        for (a, b) in raw.r2_mean.iter().zip(&std.r2_mean) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn every_fold_failed_is_invalid_input() {
        let (x, y) = traits(200, 3, 0.5, 3);
        let x = Mat::<f64>::from_fn(200, 61, |i, j| if j < 60 { x[(i, j)] } else { 0.0 });
        let r = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, opts(7));
        assert!(matches!(r, Err(PlsKitError::InvalidInput(_))), "{r:?}");
    }

    #[test]
    fn one_usable_fold_is_invalid_input() {
        let (x, y) = traits(200, 3, 0.5, 3);
        let x = with_one_row_column(&x, 17);
        let o = ProfilesFindNOpts {
            n_folds: 2,
            n_repeats: 1,
            ..opts(7)
        };
        let r = profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, o);
        assert!(matches!(r, Err(PlsKitError::InvalidInput(_))), "{r:?}");
    }

    #[test]
    fn bad_inputs_are_errors() {
        let (x, y) = traits(100, 3, 0.5, 8);
        let run = |o: ProfilesFindNOpts, n_max: usize| {
            profiles_find_n(x.as_ref(), y.as_ref(), 1, n_max, o)
        };
        let invalid =
            |r: PlsKitResult<ProfilesFindNOutput>| matches!(r, Err(PlsKitError::InvalidInput(_)));
        assert!(invalid(run(opts(1), 0)), "n_max = 0");
        for (name, o) in [
            (
                "n_folds = 1",
                ProfilesFindNOpts {
                    n_folds: 1,
                    ..opts(1)
                },
            ),
            (
                "n_folds > N",
                ProfilesFindNOpts {
                    n_folds: 101,
                    ..opts(1)
                },
            ),
            (
                "n_repeats = 0",
                ProfilesFindNOpts {
                    n_repeats: 0,
                    ..opts(1)
                },
            ),
            (
                "n_perm = 0",
                ProfilesFindNOpts {
                    n_perm: 0,
                    ..opts(1)
                },
            ),
        ] {
            assert!(invalid(run(o, 9)), "{name}");
        }

        let short = Col::<f64>::zeros(99);
        assert!(matches!(
            profiles_find_n(x.as_ref(), short.as_ref(), 1, 9, opts(1)),
            Err(PlsKitError::DimensionMismatch { .. })
        ));

        let mut y_nan = y.clone();
        y_nan[5] = f64::NAN;
        assert!(matches!(
            profiles_find_n(x.as_ref(), y_nan.as_ref(), 1, 9, opts(1)),
            Err(PlsKitError::NonFiniteInput)
        ));

        let mut x_nan = x.clone();
        x_nan[(3, 4)] = f64::NAN;
        assert!(matches!(
            profiles_find_n(x_nan.as_ref(), y.as_ref(), 1, 9, opts(1)),
            Err(PlsKitError::NonFiniteInput)
        ));
    }

    #[test]
    fn rotated_checks_its_own_basis_and_span() {
        let (w, coef) = basis_and_coef();
        let out = split_on_basis(w.as_ref(), coef.as_ref(), 3, false).unwrap();
        let r = Mat::<f64>::identity(3, 3);
        let invalid =
            |o: &FactorsOutput| matches!(o.rotated(r.as_ref()), Err(PlsKitError::InvalidInput(_)));

        let mut swapped = out.clone();
        swapped.coef = Col::<f64>::from_fn(D, |i| i as f64 + 1.0);
        assert!(invalid(&swapped), "coef outside the span of W");

        let mut scaled = out.clone();
        scaled.w = Mat::<f64>::from_fn(D, 3, |i, j| out.w[(i, j)] * 1.001);
        assert!(invalid(&scaled), "W not orthonormal");
    }

    #[test]
    fn overflowing_coef_norm_is_invalid_input() {
        // W spans coef, but c = W' coef overflows to infinity.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let w = Mat::<f64>::from_fn(3, 2, |i, j| match (i, j) {
            (0 | 1, 0) => h,
            (2, 1) => 1.0,
            _ => 0.0,
        });
        let coef = Col::<f64>::from_fn(3, |i| if i < 2 { 1.7e308 } else { 1.0 });
        let r = split_on_basis(w.as_ref(), coef.as_ref(), 2, false);
        match r {
            Err(PlsKitError::InvalidInput(m)) => assert!(m.contains("overflows"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)] // t, e, x, y, o are the math names
    fn pre_standardized_flag_changes_k_star_on_raw_data() {
        // Four traits of 12 columns; one column on a 1e6 scale. Treated as
        // already standardized, that column dominates the eigenvalues.
        let sd = 3.0_f64.sqrt();
        let t = unif(300, 4, 21);
        let e = unif(300, 48, 22);
        let x = Mat::<f64>::from_fn(300, 48, |i, j| {
            let v = sd * (t[(i, j / 12)] + 0.5 * e[(i, j)]);
            if j == 0 { v * 1e6 } else { v }
        });
        let y = unif_col(300, 23);
        let run = |pre: bool| {
            let o = ProfilesFindNOpts {
                pre_standardized: pre,
                ..opts(5)
            };
            profiles_find_n(x.as_ref(), y.as_ref(), 1, 9, o)
                .unwrap()
                .k_star
        };
        assert_ne!(run(false), run(true));
    }
}
