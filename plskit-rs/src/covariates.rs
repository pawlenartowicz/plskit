//! Covariate adjustment: removal of `[1, Z]` from columns by weighted least
//! squares.
//!
//! `A = [1, Z_kept]` is the `n × r` design: an intercept column, then the
//! columns of `Z` that are not aliased, in order. With observation weights
//! `w` (normalized to mean one; `w = 1` unweighted), `A_w = diag(√w)·A =
//! Q·R` is the thin QR factorization (`Q` is `n × r` with orthonormal
//! columns, `R` is `r × r` upper triangular). For a column `v`:
//!
//! - WLS coefficients: `b = argmin Σ wᵢ (vᵢ − aᵢ'b)² = (A'WA)⁻¹ A'W v =
//!   R⁻¹ Q' (√w ∘ v)`.
//! - Residual: `v − A·b`. On rows with `wᵢ > 0` this is `M_Z·v` with
//!   `M_Z = W^{-1/2} (I − QQ') W^{1/2}`; a row with `wᵢ = 0` gets the finite
//!   residual `vᵢ − aᵢ'b` from the same coefficients.
//!
//! `Q` and `R` come from modified Gram–Schmidt with each column
//! orthogonalized twice against the kept ones ("twice is enough": Giraud,
//! Langou & Rozložník 2005, *Comput. Math. Appl.* 50, 1069–1075). A column
//! whose √w-weighted norm after projection is at most `ALIAS_TOL` times its
//! norm before is aliased (constant, or a linear combination of the
//! intercept and the earlier kept columns) and is dropped, as R's `lm` does.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatMut, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};

/// Relative norm below which a column of `[1, Z]` is aliased: R's
/// `lm.fit` default `tol = 1e-7`.
const ALIAS_TOL: f64 = 1e-7;

/// `[1, Z]` with aliased columns dropped, and its weighted thin QR
/// factorization (see the module documentation).
#[derive(Debug, Clone)]
pub struct CovariateBasis {
    /// `A = [1, Z_kept]`, unscaled; `(n, rank)`.
    a: Mat<f64>,
    /// Orthonormal `Q` of `diag(√w)·A`, stored as `diag(√w)·Q` so that
    /// `coef` reads `v` without a weighted copy of it; `(n, rank)`.
    q: Mat<f64>,
    /// Upper-triangular `R`, `diag(√w)·A = Q·R`; `(rank, rank)`.
    r: Mat<f64>,
    /// `√w`, or `None` unweighted; `(n,)`.
    sqw: Option<Col<f64>>,
    /// Column of `Z` behind each kept covariate (column `j + 1` of `A`).
    kept: Vec<usize>,
    /// Columns of the input `Z`.
    p_z: usize,
    /// Rows with positive weight (`n` unweighted).
    n_pos: usize,
}

impl CovariateBasis {
    /// Build the basis of `[1, Z]` for `z` of shape `(n, p_z)`. `weights` are
    /// the normalized observation weights (mean one), length `n`, or `None`
    /// unweighted.
    ///
    /// # Errors
    /// - `ShapeMismatch` when `z` has no rows or `weights` does not have
    ///   `n` entries.
    /// - `NonFiniteInput` when `z` holds a NaN or an infinity.
    #[allow(clippy::many_single_char_names)]
    pub fn new(z: MatRef<'_, f64>, weights: Option<ColRef<'_, f64>>) -> PlsKitResult<Self> {
        let (n, p_z) = (z.nrows(), z.ncols());
        if let Some(w) = weights {
            if w.nrows() != n {
                return Err(PlsKitError::ShapeMismatch(format!(
                    "covariates have {n} rows, weights have {}",
                    w.nrows()
                )));
            }
        }
        if n == 0 {
            return Err(PlsKitError::ShapeMismatch("covariates have no rows".into()));
        }
        crate::fit::check_finite_mat(z)?;
        let sqw = weights.map(crate::linalg::sqrt_col);
        let mut q_cols: Vec<Col<f64>> = Vec::new();
        let mut r_cols: Vec<Vec<f64>> = Vec::new();
        let mut kept = Vec::new();
        for c in 0..=p_z {
            let raw = |i: usize| if c == 0 { 1.0 } else { z[(i, c - 1)] };
            let mut v = Col::<f64>::from_fn(n, |i| sqw.as_ref().map_or(1.0, |s| s[i]) * raw(i));
            let norm0 = v.norm_l2();
            let mut rc = vec![0.0; q_cols.len() + 1];
            for _ in 0..2 {
                for (l, ql) in q_cols.iter().enumerate() {
                    let d: f64 = (0..n).map(|i| ql[i] * v[i]).sum();
                    for i in 0..n {
                        v[i] -= d * ql[i];
                    }
                    rc[l] += d;
                }
            }
            let norm = v.norm_l2();
            if norm <= ALIAS_TOL * norm0 {
                continue;
            }
            rc[q_cols.len()] = norm;
            q_cols.push(Col::<f64>::from_fn(n, |i| v[i] / norm));
            r_cols.push(rc);
            if c > 0 {
                kept.push(c - 1);
            }
        }
        let rank = q_cols.len();
        let n_pos = weights.map_or(n, |w| (0..n).filter(|&i| w[i] > 0.0).count());
        let a = Mat::<f64>::from_fn(
            n,
            rank,
            |i, j| {
                if j == 0 { 1.0 } else { z[(i, kept[j - 1])] }
            },
        );
        let q = Mat::<f64>::from_fn(n, rank, |i, j| {
            sqw.as_ref().map_or(1.0, |s| s[i]) * q_cols[j][i]
        });
        let r = Mat::<f64>::from_fn(rank, rank, |i, j| if i <= j { r_cols[j][i] } else { 0.0 });
        Ok(Self {
            a,
            q,
            r,
            sqw,
            kept,
            p_z,
            n_pos,
        })
    }

    /// `rank([1, Z])`: the columns of `A`, the intercept included.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.a.ncols()
    }

    /// Kept covariates, `rank() − 1`.
    #[must_use]
    pub fn n_covariates(&self) -> usize {
        self.rank() - 1
    }

    /// Columns of the input `Z`, dropped ones included.
    #[must_use]
    pub fn p_z(&self) -> usize {
        self.p_z
    }

    /// The residual degrees of freedom `n − rank([1, Z])`, with `n` the rows
    /// with positive weight (0 when the rank exceeds them).
    pub(crate) fn residual_df(&self) -> usize {
        self.n_pos.saturating_sub(self.rank())
    }

    /// The residual degrees of freedom `n − rank([1, Z])`, with `n` the rows
    /// with positive weight, must be at least `k + 1`.
    ///
    /// # Errors
    /// `InsufficientResidualDf` when `n − rank < k + 1`.
    pub fn check_df(&self, k: usize) -> PlsKitResult<()> {
        let (n, rank) = (self.n_pos, self.rank());
        if n < rank + k + 1 {
            return Err(PlsKitError::InsufficientResidualDf { n, rank, k });
        }
        Ok(())
    }

    /// WLS coefficients `b = R⁻¹ Q' (√w ∘ v)` of each column of `v` on `A`.
    ///
    /// # Shapes
    /// - `v`: `(n, q)`
    /// - returns `(rank, q)`
    #[must_use]
    pub fn coef(&self, v: MatRef<'_, f64>) -> Mat<f64> {
        let mut b = crate::linalg::mat_mul(self.q.transpose(), v, Par::Seq);
        // Back substitution on the upper-triangular R.
        let rank = self.rank();
        for c in 0..b.ncols() {
            for i in (0..rank).rev() {
                let s: f64 = (i + 1..rank).map(|l| self.r[(i, l)] * b[(l, c)]).sum();
                b[(i, c)] = (b[(i, c)] - s) / self.r[(i, i)];
            }
        }
        b
    }

    /// `v ← v − A·coef(v)`, in place: the WLS residuals of `v`'s columns on
    /// `[1, Z]`. A column whose √w-weighted residual norm is at most
    /// `ALIAS_TOL` times its norm before is set to 0.
    ///
    /// # Shapes
    /// - `v`: `(n, q)`
    pub fn residualize(&self, mut v: MatMut<'_, f64>) {
        let wnorm = |v: MatRef<'_, f64>, j: usize| match &self.sqw {
            None => v.col(j).norm_l2(),
            Some(s) => Col::<f64>::from_fn(v.nrows(), |i| s[i] * v[(i, j)]).norm_l2(),
        };
        let before: Vec<f64> = (0..v.ncols()).map(|j| wnorm(v.as_ref(), j)).collect();
        let b = self.coef(v.as_ref());
        matmul(
            v.as_mut(),
            Accum::Add,
            self.a.as_ref(),
            b.as_ref(),
            -1.0,
            Par::Seq,
        );
        // A column in the span of `[1, Z]` is aliased, by the rule `new` applies to Z's columns.
        for (j, &norm0) in before.iter().enumerate() {
            if wnorm(v.as_ref(), j) <= ALIAS_TOL * norm0 {
                v.as_mut().col_mut(j).fill(0.0);
            }
        }
    }

    /// `v_new ← v_new − [1, Z_new,kept]·coef`, in place: residuals of new
    /// rows from coefficients fitted elsewhere (by [`Self::coef`] of this
    /// basis). The kept columns of `z_new` are the ones kept in `Z`.
    ///
    /// # Shapes
    /// - `z_new`: `(m, p_z)`
    /// - `v_new`: `(m, q)`
    /// - `coef`: `(rank, q)`
    pub(crate) fn residualize_new(
        &self,
        z_new: MatRef<'_, f64>,
        v_new: MatMut<'_, f64>,
        coef: MatRef<'_, f64>,
    ) {
        // The callers build all three from one basis, so these asserts guard an internal invariant.
        assert_eq!(z_new.ncols(), self.p_z, "residualize_new: columns of z_new");
        assert_eq!(
            z_new.nrows(),
            v_new.nrows(),
            "residualize_new: rows of v_new"
        );
        assert_eq!(coef.nrows(), self.rank(), "residualize_new: rows of coef");
        let a_new = Mat::<f64>::from_fn(z_new.nrows(), self.rank(), |i, j| {
            if j == 0 {
                1.0
            } else {
                z_new[(i, self.kept[j - 1])]
            }
        });
        matmul(v_new, Accum::Add, a_new.as_ref(), coef, -1.0, Par::Seq);
    }

    /// `M_Z·G·M_Z` with `M_Z = I − QQ'`, the Gram matrix of the residualized
    /// `X`: `G − QQ'G − GQQ' + Q(Q'GQ)Q'`, in `O(n²·rank)`. Defined for an
    /// unweighted basis only.
    ///
    /// # Shapes
    /// - `g`: `(n, n)`
    /// - returns `(n, n)`
    ///
    /// # Panics
    /// When the basis is weighted or `g` is not `(n, n)`.
    #[must_use]
    pub(crate) fn project_gram(&self, g: MatRef<'_, f64>) -> Mat<f64> {
        let n = self.a.nrows();
        assert!(self.sqw.is_none(), "project_gram: weighted basis");
        assert!(
            g.nrows() == n && g.ncols() == n,
            "project_gram: g is not (n, n)"
        );
        let q = self.q.as_ref();
        // H = (I − QQ')G, then H(I − QQ').
        let mut h = g.to_owned();
        let qtg = crate::linalg::mat_mul(q.transpose(), g, Par::Seq);
        matmul(h.as_mut(), Accum::Add, q, qtg.as_ref(), -1.0, Par::Seq);
        let hq = crate::linalg::mat_mul(h.as_ref(), q, Par::Seq);
        matmul(
            h.as_mut(),
            Accum::Add,
            hq.as_ref(),
            q.transpose(),
            -1.0,
            Par::Seq,
        );
        h
    }

    /// Coefficients indexed by `[1, Z]` from coefficients indexed by `A`:
    /// entry 0 is the intercept, entry `1 + c` belongs to column `c` of `Z`
    /// and is 0 for a dropped column.
    ///
    /// # Shapes
    /// - `b`: `(rank,)`
    /// - returns `(1 + p_z,)`
    #[must_use]
    pub fn full_coef(&self, b: ColRef<'_, f64>) -> Col<f64> {
        let mut out = Col::<f64>::zeros(1 + self.p_z);
        out[0] = b[0];
        for (j, &c) in self.kept.iter().enumerate() {
            out[1 + c] = b[1 + j];
        }
        out
    }
}

/// X and y standardized, then with `[1, Z]` removed: `X̃ = M_Z·X_std`,
/// `ỹ = M_Z·y_std`, with the standardization moments and the basis.
pub(crate) struct Adjusted {
    /// `X̃`; `(n, p)`.
    pub x: Mat<f64>,
    /// `ỹ`; `(n,)`.
    pub y: Col<f64>,
    /// Column means of X (zeros under `pre_standardized`); `(p,)`.
    pub x_mean: Col<f64>,
    /// Column scales of X (ones under `pre_standardized`); `(p,)`.
    pub x_scale: Col<f64>,
    /// Scale of y (1 under `pre_standardized`).
    pub y_scale: f64,
    /// The basis of `[1, Z]` under the same weights.
    pub basis: CovariateBasis,
}

/// Standardize X and y as `pls1_fit` does (weighted moments under
/// `w_norm`; as given, with means 0 and scales 1, under `pre_standardized`),
/// then replace both by their WLS residuals on `[1, Z]`.
///
/// # Errors
/// [`basis_for`]'s.
pub(crate) fn adjust(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    z: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> PlsKitResult<Adjusted> {
    let basis = basis_for(x, z, w_norm)?;
    Ok(adjust_on(x, y, basis, w_norm, pre_standardized))
}

/// The basis of `[1, Z]` for X, under the normalized weights `w_norm`.
///
/// # Errors
/// - `ShapeMismatch` when `z` does not have one row per row of `x`.
/// - `NonFiniteInput` when `z` holds a NaN or an infinity.
pub(crate) fn basis_for(
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
) -> PlsKitResult<CovariateBasis> {
    if z.nrows() != x.nrows() {
        return Err(PlsKitError::ShapeMismatch(format!(
            "covariates have {} rows, X has {}",
            z.nrows(),
            x.nrows()
        )));
    }
    CovariateBasis::new(z, w_norm)
}

/// [`adjust`] on the `basis` [`basis_for`] built for `x` under the same
/// `w_norm`.
#[allow(clippy::many_single_char_names)]
pub(crate) fn adjust_on(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    basis: CovariateBasis,
    w_norm: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> Adjusted {
    let p = x.ncols();
    // `standardize_weighted` gives the bits of `fit_x_moments` and of the
    // standardized matrix `pls1_fit` writes.
    let (mut xs, x_mean, x_scale, mut ys, y_scale) = if pre_standardized {
        (
            x.to_owned(),
            Col::<f64>::zeros(p),
            Col::<f64>::from_fn(p, |_| 1.0),
            y.to_owned(),
            1.0,
        )
    } else {
        let (xs, xm, xsc) = crate::linalg::standardize_weighted(x, w_norm);
        let (ys, _, ysc) = crate::linalg::standardize1_weighted(y, w_norm);
        (xs, xm, xsc, ys, ysc)
    };
    basis.residualize(xs.as_mut());
    basis.residualize(ys.as_mut().as_mat_mut());
    Adjusted {
        x: xs,
        y: ys,
        x_mean,
        x_scale,
        y_scale,
        basis,
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::linalg::normalize_weights;
    use crate::test_support::{
        assert_close_rel, test_weights, unif, unif_col, with_intercept, wls_normal_equations,
    };

    /// Unweighted, and uneven normalized weights with zeros.
    fn weight_cases(n: usize) -> [Option<Col<f64>>; 2] {
        [None, normalize_weights(test_weights(n).as_ref())]
    }

    #[test]
    fn basis_drops_aliased_columns_in_order() {
        let n = 40;
        let (z1, z2) = (unif_col(n, 1), unif_col(n, 2));
        let z = Mat::<f64>::from_fn(n, 6, |i, j| match j {
            0 | 2 => z1[i],
            1 => 1.0,
            3 => 3.0 * z1[i] + 2.0,
            4 => 5.0,
            _ => z2[i],
        });
        for w in weight_cases(n) {
            let basis = CovariateBasis::new(z.as_ref(), w.as_ref().map(Col::as_ref)).unwrap();
            assert_eq!(basis.rank(), 3);
            assert_eq!(basis.n_covariates(), 2);
            assert_eq!(basis.p_z(), 6);
            let b = Col::<f64>::from_fn(3, |i| i as f64 + 1.0);
            let full = basis.full_coef(b.as_ref());
            let want = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0];
            assert_eq!((0..7).map(|i| full[i]).collect::<Vec<_>>(), want);
        }
    }

    #[test]
    fn residuals_are_orthogonal_to_the_basis() {
        let n = 50;
        let z = unif(n, 2, 3);
        let v = unif(n, 3, 4);
        let a = with_intercept(z.as_ref());
        for w in weight_cases(n) {
            let wr = w.as_ref().map(Col::as_ref);
            let basis = CovariateBasis::new(z.as_ref(), wr).unwrap();
            let mut r = v.clone();
            basis.residualize(r.as_mut());
            let mut worst = 0.0_f64;
            for l in 0..a.ncols() {
                for c in 0..r.ncols() {
                    let s: f64 = (0..n)
                        .map(|i| a[(i, l)] * wr.map_or(1.0, |w| w[i]) * r[(i, c)])
                        .sum();
                    worst = worst.max(s.abs());
                }
            }
            assert!(worst <= 1e-12 * v.norm_l2(), "max |A'W r| = {worst:e}");
        }
    }

    /// `residualize_new` on the training rows with their own coefficients
    /// is `residualize`.
    #[test]
    fn residualize_new_on_training_rows_equals_residualize() {
        let n = 50;
        let z = unif(n, 2, 5);
        let v = unif(n, 2, 6);
        for w in weight_cases(n) {
            let basis = CovariateBasis::new(z.as_ref(), w.as_ref().map(Col::as_ref)).unwrap();
            let mut r = v.clone();
            basis.residualize(r.as_mut());
            let mut r_new = v.clone();
            basis.residualize_new(z.as_ref(), r_new.as_mut(), basis.coef(v.as_ref()).as_ref());
            assert_close_rel(r_new.as_ref(), r.as_ref(), 1e-12, "residualize_new");
        }
    }

    #[test]
    fn coef_matches_the_normal_equations() {
        let n = 50;
        let mut z = unif(n, 3, 7);
        for i in 0..n {
            z[(i, 0)] += 2.0;
        }
        let v = unif(n, 2, 8);
        let a = with_intercept(z.as_ref());
        for w in weight_cases(n) {
            let wr = w.as_ref().map(Col::as_ref);
            let basis = CovariateBasis::new(z.as_ref(), wr).unwrap();
            let want = wls_normal_equations(a.as_ref(), wr, v.as_ref());
            assert_close_rel(
                basis.coef(v.as_ref()).as_ref(),
                want.as_ref(),
                1e-10,
                "coef",
            );
        }
    }

    #[test]
    fn zero_weight_rows_get_finite_residuals() {
        let n = 50;
        let zeros = [3, 17, 40];
        let w = Col::<f64>::from_fn(n, |i| {
            if zeros.contains(&i) {
                0.0
            } else {
                0.5 + (i % 4) as f64 * 0.25
            }
        });
        let w = normalize_weights(w.as_ref()).unwrap();
        let z = unif(n, 2, 9);
        let v = unif(n, 2, 10);
        let a = with_intercept(z.as_ref());
        let basis = CovariateBasis::new(z.as_ref(), Some(w.as_ref())).unwrap();
        let b = basis.coef(v.as_ref());
        let mut r = v.clone();
        basis.residualize(r.as_mut());
        assert!(r.is_all_finite() && b.is_all_finite());
        for &i in &zeros {
            for c in 0..v.ncols() {
                let fitted: f64 = (0..a.ncols()).map(|l| a[(i, l)] * b[(l, c)]).sum();
                let want = v[(i, c)] - fitted;
                assert!(
                    (r[(i, c)] - want).abs() <= 1e-14 * want.abs().max(1.0),
                    "row {i}: {} vs {want}",
                    r[(i, c)]
                );
            }
        }
    }
}
