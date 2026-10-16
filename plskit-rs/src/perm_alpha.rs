//! PLS1 dual coefficients from a Gram matrix: [`pls1_perm_alpha`].
//!
//! For a standardized block `X̃` (`n × p`) that the caller holds, and many
//! outcomes, returns per outcome the n-vector `alpha` with
//! `coef = X̃'·alpha`, where `coef` is the PLS1 coefficient at `k`
//! components on the standardized scale (`Pls1Model.coef`). The input is
//! `G = X̃X̃'` and `p`; nothing of size `p` enters or leaves. Each outcome
//! is one call of the n-space kernel (`dual_route::pls1_nspace_kernel`) on
//! one block built from `G`; a fit the kernel cannot certify is reported,
//! since there is no `X̃` to refit it on.

use faer::{Col, Mat, MatRef};
use rayon::prelude::*;

use crate::dual_route::{NspaceGram, NspaceOutcome, pls1_nspace_kernel};
use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::check_finite_mat;
use crate::linalg::standardize1;

/// Multiple of the worst-case rounding that the checks on `gram` allow.
/// The checks are sanity gates against a wrong caller, not certificates;
/// observed rounding sits orders of magnitude below the worst case.
const GRAM_CHECK_BAND: f64 = 4.0;

/// Largest `|mean| / sd` of a raw column of `X` whose centring rounding
/// the row-sum check on `gram` admits at its worst case. `gram` does not
/// carry it, so it is a fixed policy rather than a derived bound (see
/// [`check_gram`]).
const CENTRING_KAPPA: f64 = 1e6;

/// Largest `n_features` accepted: `2^53`, or `usize::MAX / 8` where `usize`
/// is narrower (32-bit targets such as wasm32). Every count up to `2^53` is
/// exact in f64 (the bindings' `F64_EXACT_MAX`). It keeps the kernel's
/// rounding bounds, computed in `usize` (`2·p + 3·n` in
/// `nspace_base_bounds`), from overflowing. On 32-bit, `2·p` is at most
/// `usize::MAX / 4`, and `n` is bounded by memory (8·n bytes per column),
/// so the sum fits. From about `2^50` the gates already leave every column
/// unresolved, so the cap rejects no usable input.
const N_FEATURES_MAX: usize = if usize::BITS > 53 {
    1 << 53
} else {
    usize::MAX / 8
};

/// Output of [`pls1_perm_alpha`]. `m` is the number of columns of `y`.
#[derive(Debug, Clone)]
pub struct PermAlphaOutput {
    /// Dual coefficients, `(n, m)`: column `j` gives `coef = X̃'·alpha[:, j]`
    /// on the standardized scale. NaN in a column that is not resolved.
    pub alpha: Mat<f64>,
    /// Components per column, length `m`: `k` for a resolved fit, `0` for a
    /// zero model and for an unresolved column.
    pub k_used: Vec<usize>,
    /// Per column, length `m`: `false` where the fit cannot be certified
    /// from `gram` alone (refit that column on `X`).
    pub resolved: Vec<bool>,
    /// `k` as passed.
    pub k: usize,
    /// `n_features` as passed.
    pub n_features: usize,
}

/// PLS1 dual coefficients for every column of `y` from the Gram matrix of
/// a standardized block, without the block.
///
/// `gram` is `X̃X̃'` for an `X̃` (`n × n_features`) that the caller holds
/// and multiplies `alpha` by. The caller guarantees: `gram` is the Gram
/// matrix of that `X̃`; its columns are centred and scaled as
/// [`crate::linalg::standardize`] scales them (`ddof = 0`, sum of squares
/// `n`, a constant column zero to rounding); each entry is computed in f64
/// within the rounding of one length-`n_features` dot product (sums over
/// column blocks and one final scaling stay within it); `n_features`
/// counts every column summed into `gram`, constant and zeroed ones
/// included (a count that leaves out only zero columns passes the trace
/// check). Then
/// `X̃'·alpha[:, j]` is `pls1_fit(X, y[:, j], k).coef` to rounding. For a
/// store `D` of unit-norm columns passed as `n·DD'`, `X̃ = √n·D`, so
/// `coef = √n·D'·alpha[:, j]`.
///
/// Column `j` of `y` is one raw outcome, standardized here by
/// [`crate::linalg::standardize1`]. An outcome that standardizes to exact
/// zeros is the zero model (`alpha` zero, `k_used = 0`), and so is every
/// outcome when every entry of `gram` is exactly zero. An outcome the
/// kernel cannot certify is reported: `resolved[j] = false`,
/// `k_used[j] = 0` and column `j` of `alpha` NaN. Column `j` depends only
/// on `gram`, `y[:, j]`, `k` and `n_features`: its bits do not change with
/// the other columns, the thread count or the memory layout of `gram` and
/// `y`. Columns run in parallel on the pool `PLSKIT_NUM_THREADS` selects.
///
/// # Shapes
/// - `gram`: `(n, n)`
/// - `y`: `(n, m)`, `m ≥ 0`
/// - `k`: `1`
///
/// # Errors
/// - `PlsKitError::InvalidArgument` when `k == 0`, `k ≥ 2`, `n_features == 0`
///   or `n_features > 2^53` (on 64-bit targets)
/// - `PlsKitError::ShapeMismatch` when `gram` is not square or `y` has
///   another row count
/// - `PlsKitError::KExceedsMax` when `k > n − 1`
/// - `PlsKitError::NonFiniteInput` when `gram` or `y` contains NaN or infinity
/// - `PlsKitError::InvalidInput` when `gram` has a negative diagonal entry,
///   is not symmetric, has an entry `|gram[i, j]|` above
///   `√(gram[i, i]·gram[j, j])`, has a trace above `n · n_features`, or has
///   a row sum away from zero, the last four beyond the rounding the
///   contract allows
///
/// # Panics
/// Never.
pub fn pls1_perm_alpha(
    gram: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    n_features: usize,
) -> PlsKitResult<PermAlphaOutput> {
    crate::fit::with_thread_limit(|| perm_alpha_impl(gram, y, k, n_features))
}

/// Body of [`pls1_perm_alpha`], on the caller's pool.
fn perm_alpha_impl(
    gram: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    n_features: usize,
) -> PlsKitResult<PermAlphaOutput> {
    check_args(gram, y, k, n_features)?;
    check_gram(gram, n_features)?;
    let (n, m) = (gram.nrows(), y.ncols());
    // Every entry exactly zero (`+0.0` or `-0.0`): every column of `X̃` is
    // zero, so every outcome is the zero model, as `pls1_fit` gives on
    // `X̃ = 0`. Decided on the entries; after `check_gram` a zero trace
    // gives the same answer, since its Cauchy-Schwarz check admits only
    // exact zeros in a row whose diagonal entry is zero.
    if (0..n).all(|j| (0..n).all(|i| gram[(i, j)] == 0.0)) {
        return Ok(PermAlphaOutput {
            alpha: Mat::<f64>::zeros(n, m),
            k_used: vec![0; m],
            resolved: vec![true; m],
            k,
            n_features,
        });
    }
    // One block, on the calling thread; the columns share it read-only.
    let block = NspaceGram::from_gram(gram, n_features);
    let mut alpha = Mat::<f64>::zeros(n, m);
    let mut k_used = vec![0; m];
    let mut resolved = vec![false; m];
    alpha
        .as_mut()
        .par_col_iter_mut()
        .zip(k_used.par_iter_mut())
        .zip(resolved.par_iter_mut())
        .enumerate()
        .for_each(
            |(j, ((mut a, ku), r))| match alpha_column(&block, y, j, k) {
                Some((aj, kj)) => {
                    a.copy_from(&aj);
                    *ku = kj;
                    *r = true;
                }
                None => a.fill(f64::NAN),
            },
        );
    Ok(PermAlphaOutput {
        alpha,
        k_used,
        resolved,
        k,
        n_features,
    })
}

/// Column `j` of `y`, standardized by `standardize1`, fitted by one kernel
/// call on `block`: `Some((alpha, k_used))` when resolved (the zero model
/// is a zero `alpha` with `k_used = 0`), `None` when unresolved. One call
/// per column, so the column's bits do not depend on the other columns.
#[allow(clippy::many_single_char_names)]
fn alpha_column(
    block: &NspaceGram,
    y: MatRef<'_, f64>,
    j: usize,
    k: usize,
) -> Option<(Col<f64>, usize)> {
    let n = y.nrows();
    let yj = Col::<f64>::from_fn(n, |i| y[(i, j)]);
    let (z, _, _) = standardize1(yj.as_ref());
    match pls1_nspace_kernel(&block.block(), z.as_ref(), k) {
        NspaceOutcome::Resolved { alpha, k_used } => Some((alpha, k_used)),
        NspaceOutcome::ZeroModel => Some((Col::<f64>::zeros(n), 0)),
        NspaceOutcome::Unresolved => None,
    }
}

/// The argument, shape and finiteness checks of [`pls1_perm_alpha`], in
/// the order its `# Errors` lists them.
fn check_args(
    gram: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    n_features: usize,
) -> PlsKitResult<()> {
    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k >= 2 {
        return Err(PlsKitError::InvalidArgument(format!(
            "pls1_perm_alpha supports k = 1 only, got k = {k}"
        )));
    }
    if n_features == 0 {
        return Err(PlsKitError::InvalidArgument(
            "n_features must be >= 1".into(),
        ));
    }
    if n_features > N_FEATURES_MAX {
        return Err(PlsKitError::InvalidArgument(format!(
            "n_features must be <= {N_FEATURES_MAX}, got {n_features}"
        )));
    }
    let n = gram.nrows();
    if gram.ncols() != n {
        return Err(PlsKitError::ShapeMismatch(format!(
            "gram must be square, got {n} x {}",
            gram.ncols()
        )));
    }
    if y.nrows() != n {
        return Err(PlsKitError::ShapeMismatch(format!(
            "Y has {} rows, gram is {n} x {n}",
            y.nrows()
        )));
    }
    // `k ≤ n − 1`, the rank of a centred block.
    if k >= n {
        return Err(PlsKitError::KExceedsMax {
            k,
            k_max: n.saturating_sub(1),
        });
    }
    check_finite_mat(gram)?;
    check_finite_mat(y)
}

/// The structural checks on a finite square `gram` (`d_i = gram[i, i]`,
/// `p = n_features`, `ε = f64::EPSILON`, `B = GRAM_CHECK_BAND`), each the
/// worst-case rounding of the caller's contract times `B`. The contract
/// lets each entry carry `(p + 1)·ε·√(d_i d_j)`: one length-`p` dot
/// product and one final scaling.
///
/// 1. `d_i ≥ 0`: a computed sum of squares is never negative.
/// 2. Symmetry: `|G_ij − G_ji| ≤ B·2(p + 1)·ε·√d_i·√d_j`.
/// 3. Cauchy-Schwarz, for `i > j`: `|G_ij| ≤ (1 + B·2(p + 1)·ε)·√d_i·√d_j`,
///    computed left to right. Exact `G` has `|G_ij| ≤ √(d_i d_j)`
///    (Cauchy-Schwarz on the rows of `X̃`); the computed entry carries
///    `(p + 1)·ε·√(d_i d_j)`, and `√d_i·√d_j` the diagonal's `(p + 1)·ε`
///    relative. A zero `d_i` admits only exact zeros in row `i`, so a `G`
///    that passes with trace 0 is exactly zero. A bound that overflows
///    rejects.
/// 4. Trace: `Σ_i d_i ≤ n·p·(1 + B·(p + 2n + 6)·ε)`. A non-constant column
///    standardized with `ddof = 0` has sum of squares `n` to `(n + 6)·ε`
///    relative, a constant one about zero; each `d_i` adds `(p + 1)·ε`
///    and the sum `(n − 1)·ε`. Equality holds only without constant
///    columns, which `gram` cannot tell, so only the upper bound is
///    checked. It runs before the centring check, whose bound assumes it.
/// 5. Centring: `|Σ_j G_ij| ≤ B·(p + 2n + 2 + n·CENTRING_KAPPA)·ε·n·√p·√d_i`.
///    The row sum carries the entries' rounding (`p + 1`), its own sum
///    (`n − 1`) and the centring rounding of `X̃`, `x̃_i'·s` with `s` the
///    column sums: a column standardized in f64 has
///    `|s_l| ≤ n·ε·(n·(κ_l + 1) + 2)` for `κ_l = |mean_l| / sd_l` of the
///    raw column, and `√n·√trace(G) ≤ n·√p` once check 4 has passed.
///
/// # Errors
/// `InvalidInput` naming the first check that fails.
// Entry `(i, j)` reads `sq[i]` and `sq[j]`, so the loops index rather than iterate.
#[allow(clippy::needless_range_loop)]
fn check_gram(g: MatRef<'_, f64>, n_features: usize) -> PlsKitResult<()> {
    let n = g.nrows();
    let (nf, pf, eps) = (n as f64, n_features as f64, f64::EPSILON);
    let d: Vec<f64> = (0..n).map(|i| g[(i, i)]).collect();
    if let Some(i) = (0..n).find(|&i| d[i] < 0.0) {
        return Err(PlsKitError::InvalidInput(format!(
            "gram[{i}, {i}] = {:e} is negative; the diagonal of a Gram matrix is a sum of squares",
            d[i]
        )));
    }
    let sq: Vec<f64> = d.iter().copied().map(f64::sqrt).collect();

    let tol_sym = GRAM_CHECK_BAND * 2.0 * (pf + 1.0) * eps;
    for j in 0..n {
        for i in (j + 1)..n {
            let diff = (g[(i, j)] - g[(j, i)]).abs();
            let bound = tol_sym * sq[i] * sq[j];
            let within = diff <= bound;
            if !within {
                return Err(PlsKitError::InvalidInput(format!(
                    "gram is not symmetric: gram[{i}, {j}] and gram[{j}, {i}] differ by \
                     {diff:e}, more than the rounding of a Gram matrix allows ({bound:e})"
                )));
            }
        }
    }

    let cs_factor = 1.0 + GRAM_CHECK_BAND * 2.0 * (pf + 1.0) * eps;
    for j in 0..n {
        for i in (j + 1)..n {
            let a = g[(i, j)].abs();
            let bound = cs_factor * sq[i] * sq[j];
            // A bound that overflows to infinity rejects, as a non-finite
            // left side rejects in the other checks.
            let within = a <= bound && bound.is_finite();
            if !within {
                return Err(PlsKitError::InvalidInput(format!(
                    "|gram[{i}, {j}]| = {a:e} exceeds sqrt(gram[{i}, {i}] * gram[{j}, {j}]) = \
                     {:e} by more than the rounding of a Gram matrix allows (bound {bound:e}): \
                     gram is not the Gram matrix of real columns",
                    sq[i] * sq[j]
                )));
            }
        }
    }

    let trace: f64 = d.iter().sum();
    let trace_max = nf * pf * (1.0 + GRAM_CHECK_BAND * (pf + 2.0 * nf + 6.0) * eps);
    let within = trace <= trace_max;
    if !within {
        return Err(PlsKitError::InvalidInput(format!(
            "trace(gram) = {trace:e} exceeds n * n_features = {:e} by more than rounding: \
             the columns summed into gram have sums of squares above n, or n_features is \
             smaller than the number of columns summed into gram",
            nf * pf
        )));
    }

    let tol_row = GRAM_CHECK_BAND * (pf + 2.0 * nf + 2.0 + nf * CENTRING_KAPPA) * eps;
    let row_scale = nf * pf.sqrt();
    for i in 0..n {
        let r: f64 = (0..n).map(|j| g[(i, j)]).sum();
        let bound = tol_row * row_scale * sq[i];
        let within = r.abs() <= bound;
        if !within {
            return Err(PlsKitError::InvalidInput(format!(
                "row {i} of gram sums to {r:e}, more than rounding allows ({bound:e}): \
                 gram must be the Gram matrix of centred columns"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{FitOpts, KSpec, par_fixed, pls1_fit};
    use crate::linalg::{mat_mul, standardize, standardize1};
    use crate::test_support::{
        Layouts, assert_bits_eq, assert_rows_close, bits, col_vals, mat_vals, orthogonal_y, synth,
        unif,
    };
    use faer::linalg::matmul::matmul;
    use faer::{Accum, ColMut, ColRef, Par};

    /// `X̃ = standardize(x)` and `G = X̃X̃'`, by the product
    /// `NspaceGram::new` forms.
    fn standardized_gram(x: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
        let (xs, _, _) = standardize(x.as_ref());
        let g = mat_mul(xs.as_ref(), xs.as_ref().transpose(), par_fixed());
        (xs, g)
    }

    /// `X̃'·alpha` by one sequential GEMV, as `nspace_perm_row` forms it.
    fn coef_of(xs: MatRef<'_, f64>, alpha: ColRef<'_, f64>) -> Vec<f64> {
        let mut out = vec![0.0; xs.ncols()];
        matmul(
            ColMut::from_slice_mut(&mut out).as_mat_mut(),
            Accum::Replace,
            xs.transpose(),
            alpha.as_mat(),
            1.0,
            Par::Seq,
        );
        out
    }

    /// Four outcomes: `y` and three fixed reorderings of its rows
    /// (`gcd(3, n) = 1` for the `n` used here).
    fn outcomes(y: &Col<f64>) -> Mat<f64> {
        let n = y.nrows();
        Mat::<f64>::from_fn(n, 4, |i, j| match j {
            0 => y[i],
            1 => y[n - 1 - i],
            2 => y[(i + 7) % n],
            _ => y[(i * 3) % n],
        })
    }

    fn error(r: PlsKitResult<PermAlphaOutput>) -> PlsKitError {
        r.expect_err("the call must fail")
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn x_tilde_times_alpha_matches_pls1_fit_coef() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 1);
        let (xs, g) = standardized_gram(&x);
        let yy = outcomes(&y);
        let out = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("perm_alpha");
        assert_eq!(out.resolved, vec![true; 4]);
        assert_eq!(out.k_used, vec![1; 4]);
        assert_eq!((out.k, out.n_features), (1, p));
        for j in 0..4 {
            let yj = Col::<f64>::from_fn(n, |i| yy[(i, j)]);
            let fit = pls1_fit(
                x.as_ref(),
                yj.as_ref(),
                KSpec::Fixed(1),
                None,
                FitOpts::default(),
            )
            .expect("pls1_fit");
            assert_rows_close(
                &coef_of(xs.as_ref(), out.alpha.col(j)),
                &col_vals(fit.coef.as_ref()),
                p,
                &format!("column {j}"),
            );
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn the_observed_column_is_the_nspace_route_of_pls1_perm_null() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 2);
        let (xs, g) = standardized_gram(&x);
        let yy = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
        let out = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("perm_alpha");
        assert!(out.resolved[0]);
        assert_eq!(out.k_used, vec![1]);
        let beta = coef_of(xs.as_ref(), out.alpha.col(0));
        // The n-space row of `pls1_perm_null` for the identity permutation:
        // the same `G`, the same standardized outcome, the same GEMV.
        let block = NspaceGram::new(xs.as_ref(), par_fixed());
        let (ys, _, _) = standardize1(y.as_ref());
        let identity: Vec<usize> = (0..n).collect();
        let mut row = vec![0.0; p];
        assert!(crate::dual_route::nspace_perm_row(
            &block,
            xs.as_ref(),
            ys.as_ref(),
            &identity,
            1,
            &mut row
        ));
        assert_bits_eq(&beta, &row, "identity row");
        // The observed fit of `pls1_perm_null` is primal: equal to rounding.
        let opts = crate::perm_null::PermNullOpts {
            n_perm: 100,
            ..Default::default()
        };
        let null = crate::perm_null::pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, opts, Some(1))
            .expect("perm_null");
        assert_rows_close(&beta, &null.beta_ref, p, "beta_ref");
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn an_exactly_zero_standardized_outcome_is_the_zero_model() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 3);
        let (_, g) = standardized_gram(&x);
        // The constant 3: its 40 partial sums are exact, so its mean is.
        let three = standardize1(Col::<f64>::from_fn(n, |_| 3.0).as_ref()).0;
        assert!(
            (0..n).all(|i| three[i] == 0.0),
            "3 must standardize to exact zeros"
        );
        let yy = Mat::<f64>::from_fn(n, 3, |i, j| [y[i], 0.0, 3.0][j]);
        let out = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("perm_alpha");
        assert_eq!(out.resolved, vec![true; 3]);
        assert_eq!(out.k_used, vec![1, 0, 0]);
        for j in 1..3 {
            assert!(
                (0..n).all(|i| out.alpha[(i, j)].to_bits() == 0),
                "column {j} is +0.0"
            );
        }
        let zero = Col::<f64>::zeros(n);
        let fit = pls1_fit(
            x.as_ref(),
            zero.as_ref(),
            KSpec::Fixed(1),
            None,
            FitOpts::default(),
        )
        .expect("pls1_fit");
        assert_eq!(fit.k_used, 0);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn an_outcome_orthogonal_to_x_is_reported_unresolved() {
        let (n, p) = (40_usize, 10_usize);
        let x = unif(n, p, 4);
        let (xs, g) = standardized_gram(&x);
        let e = unif(n, 1, 5);
        let orth = orthogonal_y(&xs, 6);
        let yy = Mat::<f64>::from_fn(n, 2, |i, j| {
            if j == 0 {
                2.0 * xs[(i, 0)] + e[(i, 0)]
            } else {
                orth[i]
            }
        });
        let out = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("perm_alpha");
        assert_eq!(out.resolved, vec![true, false]);
        assert_eq!(out.k_used, vec![1, 0]);
        assert!((0..n).all(|i| out.alpha[(i, 1)].is_nan()));
        let fit = pls1_fit(
            x.as_ref(),
            orth.as_ref(),
            KSpec::Fixed(1),
            None,
            FitOpts::default(),
        )
        .expect("pls1_fit");
        assert_eq!(fit.k_used, 0, "the primal kernel stops at its floor too");
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn a_gram_summed_in_blocks_or_scaled_by_n_agrees() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 5);
        let (xs, g) = standardized_gram(&x);
        let yy = outcomes(&y);
        let reference = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("reference");
        // Four column blocks of 500, accumulated last block first: the
        // single product sums the columns forward, and a forward block sum
        // can repeat its additions exactly when the GEMM's inner chunk
        // (host dependent) divides 500. The reverse order sums in another
        // order on any host.
        let mut g_blocks = Mat::<f64>::zeros(n, n);
        for b in (0..4).rev() {
            let xb = xs.as_ref().subcols(500 * b, 500);
            matmul(
                g_blocks.as_mut(),
                Accum::Add,
                xb,
                xb.transpose(),
                1.0,
                Par::Seq,
            );
        }
        // The block sum must be a different input from `g`, or the
        // comparison below would only rerun the reference.
        assert_ne!(
            bits(&mat_vals(g_blocks.as_ref())),
            bits(&mat_vals(g.as_ref())),
            "the block sum has the bits of g"
        );
        let blocks = pls1_perm_alpha(g_blocks.as_ref(), yy.as_ref(), 1, p).expect("blocks");
        // A store of unit-norm columns D = X̃/√n, passed as n·DD'; the
        // consumer multiplies D by √n·alpha.
        let root_n = (n as f64).sqrt();
        let d = Mat::<f64>::from_fn(n, p, |i, j| xs[(i, j)] / root_n);
        let dd = mat_mul(d.as_ref(), d.as_ref().transpose(), Par::Seq);
        let g_unit = Mat::<f64>::from_fn(n, n, |i, j| n as f64 * dd[(i, j)]);
        let unit = pls1_perm_alpha(g_unit.as_ref(), yy.as_ref(), 1, p).expect("unit-norm store");
        for j in 0..4 {
            assert!(
                reference.resolved[j] && blocks.resolved[j] && unit.resolved[j],
                "column {j}"
            );
            let want = coef_of(xs.as_ref(), reference.alpha.col(j));
            assert_rows_close(
                &coef_of(xs.as_ref(), blocks.alpha.col(j)),
                &want,
                p,
                &format!("blocks, column {j}"),
            );
            let scaled = Col::<f64>::from_fn(n, |i| root_n * unit.alpha[(i, j)]);
            assert_rows_close(
                &coef_of(d.as_ref(), scaled.as_ref()),
                &want,
                p,
                &format!("unit norm, column {j}"),
            );
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn columns_far_from_zero_and_a_constant_column_pass_the_checks() {
        let (n, p) = (40_usize, 2000_usize);
        let u = unif(n, p, 10);
        let x = Mat::<f64>::from_fn(n, p, |i, j| if j == 0 { 0.1 } else { 1e4 + u[(i, j)] });
        let (xs, g) = standardized_gram(&x);
        let e = unif(n, 1, 11);
        let yy = Mat::<f64>::from_fn(n, 1, |i, _| xs[(i, 1)] + e[(i, 0)]);
        let out = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("the checks pass");
        assert_eq!(out.resolved, vec![true]);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn a_gram_centred_and_scaled_by_the_caller_passes_the_checks() {
        // The caller's own store: raw columns up to 1e7 from zero (about
        // 1.7e7 sd), each centred by its computed mean and scaled to unit
        // norm by its computed norm; every hundredth column is constant and
        // zeroed. `gram` is n·Σ_b D_b D_b' over four blocks of 500 columns,
        // and `n_features` counts every column.
        let (n, p) = (40_usize, 2000_usize);
        let u = unif(n, p, 12);
        let offsets = [0.0, 1e2, 1e4, 1e6, 1e7];
        let raw = Mat::<f64>::from_fn(n, p, |i, j| {
            let noise = if j % 100 == 99 { 0.1 } else { u[(i, j)] };
            offsets[j % 5] + noise
        });
        let mut d = Mat::<f64>::zeros(n, p);
        for j in 0..p {
            let col = raw.col(j);
            let mean = (0..n).map(|i| col[i]).sum::<f64>() / n as f64;
            let centred = Col::<f64>::from_fn(n, |i| col[i] - mean);
            let norm = centred.norm_l2();
            if norm > 1e-10 * col.norm_l2() {
                for i in 0..n {
                    d[(i, j)] = centred[i] / norm;
                }
            }
        }
        let mut dd = Mat::<f64>::zeros(n, n);
        for b in 0..4 {
            let db = d.as_ref().subcols(500 * b, 500);
            matmul(dd.as_mut(), Accum::Add, db, db.transpose(), 1.0, Par::Seq);
        }
        let gram = Mat::<f64>::from_fn(n, n, |i, j| n as f64 * dd[(i, j)]);
        // The caller's X̃ = √n·D: the outcomes and the reference fit use it.
        let root_n = (n as f64).sqrt();
        let xc = Mat::<f64>::from_fn(n, p, |i, j| root_n * d[(i, j)]);
        let e = unif(n, 1, 13);
        let y = Col::<f64>::from_fn(n, |i| xc[(i, 0)] + xc[(i, 3)] + xc[(i, 4)] + e[(i, 0)]);
        let yy = outcomes(&y);
        let out = pls1_perm_alpha(gram.as_ref(), yy.as_ref(), 1, p).expect("the checks pass");
        assert_eq!(out.resolved, vec![true; 4]);
        for j in 0..4 {
            let yj = Col::<f64>::from_fn(n, |i| yy[(i, j)]);
            let fit = pls1_fit(
                xc.as_ref(),
                yj.as_ref(),
                KSpec::Fixed(1),
                None,
                FitOpts::default(),
            )
            .expect("pls1_fit");
            // The consumer's map: its store times √n·alpha.
            let scaled = Col::<f64>::from_fn(n, |i| root_n * out.alpha[(i, j)]);
            assert_rows_close(
                &coef_of(d.as_ref(), scaled.as_ref()),
                &col_vals(fit.coef.as_ref()),
                p,
                &format!("column {j}"),
            );
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn a_column_is_the_same_bits_whatever_else_is_in_y_and_in_every_layout() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 7);
        let (_, g) = standardized_gram(&x);
        let yy = outcomes(&y);
        let all = pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("all columns");
        // NaN bits compare equal, so the bit comparisons below test something
        // only when every column resolves.
        assert_eq!(all.resolved, vec![true; 4]);
        for j in 0..4 {
            let one =
                pls1_perm_alpha(g.as_ref(), yy.as_ref().subcols(j, 1), 1, p).expect("one column");
            assert_bits_eq(
                &col_vals(all.alpha.col(j)),
                &col_vals(one.alpha.col(0)),
                &format!("column {j}"),
            );
        }
        let (lg, ly) = (Layouts::new(g.as_ref()), Layouts::new(yy.as_ref()));
        for ((gl, gv), (yl, yv)) in lg.all(&g).into_iter().zip(ly.all(&yy)) {
            let out = pls1_perm_alpha(gv, yv, 1, p).expect("layout");
            assert_bits_eq(
                &mat_vals(out.alpha.as_ref()),
                &mat_vals(all.alpha.as_ref()),
                &format!("gram {gl}, Y {yl}"),
            );
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn arguments_and_shapes_are_checked() {
        let (x, y) = synth(40, 200, 2, 1.0, 8);
        let (_, g) = standardized_gram(&x);
        let yy = Mat::<f64>::from_fn(40, 1, |i, _| y[i]);
        let (g_ref, y_ref) = (g.as_ref(), yy.as_ref());
        assert_eq!(
            error(pls1_perm_alpha(g_ref, y_ref, 0, 200)).to_string(),
            "invalid argument: k must be >= 1"
        );
        assert_eq!(
            error(pls1_perm_alpha(g_ref, y_ref, 2, 200)).to_string(),
            "invalid argument: pls1_perm_alpha supports k = 1 only, got k = 2"
        );
        assert_eq!(
            error(pls1_perm_alpha(g_ref, y_ref, 1, 0)).to_string(),
            "invalid argument: n_features must be >= 1"
        );
        assert_eq!(
            error(pls1_perm_alpha(g_ref, y_ref, 1, usize::MAX)).to_string(),
            format!(
                "invalid argument: n_features must be <= 9007199254740992, got {}",
                usize::MAX
            )
        );
        assert!(matches!(
            error(pls1_perm_alpha(g_ref.subcols(0, 39), y_ref, 1, 200)),
            PlsKitError::ShapeMismatch(_)
        ));
        assert!(matches!(
            error(pls1_perm_alpha(g_ref, y_ref.subrows(0, 39), 1, 200)),
            PlsKitError::ShapeMismatch(_)
        ));
        let one = Mat::<f64>::zeros(1, 1);
        assert!(matches!(
            error(pls1_perm_alpha(one.as_ref(), one.as_ref(), 1, 200)),
            PlsKitError::KExceedsMax { k: 1, k_max: 0 }
        ));
        let mut g_nan = g.clone();
        g_nan[(2, 5)] = f64::NAN;
        assert!(matches!(
            error(pls1_perm_alpha(g_nan.as_ref(), y_ref, 1, 200)),
            PlsKitError::NonFiniteInput
        ));
        let mut y_inf = yy.clone();
        y_inf[(3, 0)] = f64::INFINITY;
        assert!(matches!(
            error(pls1_perm_alpha(g_ref, y_inf.as_ref(), 1, 200)),
            PlsKitError::NonFiniteInput
        ));
        let no_columns = Mat::<f64>::zeros(40, 0);
        let empty =
            pls1_perm_alpha(g_ref, no_columns.as_ref(), 1, 200).expect("an empty Y is valid");
        assert_eq!((empty.alpha.nrows(), empty.alpha.ncols()), (40, 0));
        assert!(empty.k_used.is_empty() && empty.resolved.is_empty());
        // An all-zero gram (+0.0 and -0.0 entries) passes the checks and is
        // the zero model for every outcome.
        let zeros = Mat::<f64>::from_fn(40, 40, |i, j| if (i + j) % 2 == 0 { 0.0 } else { -0.0 });
        let flat = pls1_perm_alpha(zeros.as_ref(), y_ref, 1, 200).expect("an all-zero gram passes");
        assert_eq!(flat.resolved, vec![true]);
        assert_eq!(flat.k_used, vec![0]);
        assert!(
            (0..40).all(|i| flat.alpha[(i, 0)].to_bits() == 0),
            "alpha is +0.0"
        );
        // One tiny non-zero diagonal entry passes the checks and is not
        // the zero model: the trigger is exact zeros, not a tolerance.
        let mut tiny = Mat::<f64>::zeros(40, 40);
        tiny[(0, 0)] = 1e-100;
        let t = pls1_perm_alpha(tiny.as_ref(), y_ref, 1, 200).expect("a tiny gram passes");
        assert!(
            !t.resolved[0] || t.k_used[0] != 0,
            "a gram with a non-zero entry is not the zero model"
        );
        // The constant 0.1: its 40-term sum rounds, so it standardizes to
        // noise, which is fitted as any outcome and is not the zero model.
        let tenth = Mat::<f64>::from_fn(40, 1, |_, _| 0.1);
        let c =
            pls1_perm_alpha(g_ref, tenth.as_ref(), 1, 200).expect("a rounded constant is valid");
        assert!(
            !c.resolved[0] || c.k_used[0] == 1,
            "0.1 is not the zero model"
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn a_wrong_gram_is_rejected() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 9);
        let (xs, g) = standardized_gram(&x);
        let yy = outcomes(&y);
        let rejected = |g: &Mat<f64>, p: usize| match pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p)
        {
            Err(PlsKitError::InvalidInput(m)) => m,
            other => panic!("expected InvalidInput, got {other:?}"),
        };
        pls1_perm_alpha(g.as_ref(), yy.as_ref(), 1, p).expect("the correct gram passes");
        let mut negative = g.clone();
        negative[(3, 3)] = -1.0;
        assert!(rejected(&negative, p).contains("is negative"));
        let mut asymmetric = g.clone();
        asymmetric[(0, 1)] += 1e-6 * (g[(0, 0)] * g[(1, 1)]).sqrt();
        assert!(rejected(&asymmetric, p).contains("not symmetric"));
        let scaled = Mat::<f64>::from_fn(n, n, |i, j| 1.01 * g[(i, j)]);
        assert!(rejected(&scaled, p).contains("trace(gram)"));
        assert!(rejected(&g, p / 2).contains("trace(gram)"));
        // Every column shifted by 0.5 sd, then rescaled to sum of squares n:
        // the trace is right and the columns are not centred.
        let shift = 1.25_f64.sqrt();
        let xu = Mat::<f64>::from_fn(n, p, |i, j| (xs[(i, j)] + 0.5) / shift);
        let gu = mat_mul(xu.as_ref(), xu.as_ref().transpose(), Par::Seq);
        assert!(rejected(&gu, p).contains("centred"));
        // A raw `X X'`, uncentred and with mean square above 1: the trace
        // check runs first and reports the scale.
        let raw = Mat::<f64>::from_fn(n, p, |i, j| x[(i, j)] + 2.0);
        let g_raw = mat_mul(raw.as_ref(), raw.as_ref().transpose(), Par::Seq);
        assert!(rejected(&g_raw, p).contains("trace(gram)"));
    }

    #[test]
    fn a_hollow_gram_is_rejected_by_the_cauchy_schwarz_check() {
        // A hollow symmetric matrix with zero row sums: it passes the
        // diagonal, symmetry, trace and centring checks, so only the
        // Cauchy-Schwarz check rejects it.
        let rows = [
            [0.0, 1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 1.0, 0.0],
        ];
        let hollow = Mat::<f64>::from_fn(4, 4, |i, j| rows[i][j]);
        let z = Mat::<f64>::from_fn(4, 1, |i, _| if i < 2 { 1.0 } else { -1.0 });
        match pls1_perm_alpha(hollow.as_ref(), z.as_ref(), 1, 1) {
            Err(PlsKitError::InvalidInput(m)) => assert_eq!(
                m,
                "|gram[1, 0]| = 1e0 exceeds sqrt(gram[1, 1] * gram[0, 0]) = 0e0 by more than \
                 the rounding of a Gram matrix allows (bound 0e0): gram is not the Gram matrix \
                 of real columns"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn an_entry_beyond_the_cauchy_schwarz_bound_is_rejected_and_parallel_rows_pass() {
        let (n, p) = (40_usize, 2000_usize);
        let (x, y) = synth(n, p, 2, 1.0, 9);
        let (_, g) = standardized_gram(&x);
        let yy = outcomes(&y);
        // One off-diagonal pair raised to 1.01·√(d_i d_j), symmetrically.
        let mut raised = g.clone();
        let v = 1.01 * (g[(7, 7)] * g[(3, 3)]).sqrt();
        raised[(7, 3)] = v;
        raised[(3, 7)] = v;
        match pls1_perm_alpha(raised.as_ref(), yy.as_ref(), 1, p) {
            Err(PlsKitError::InvalidInput(m)) => {
                assert!(m.starts_with("|gram[7, 3]| = "), "{m}");
                assert!(m.ends_with("is not the Gram matrix of real columns"), "{m}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // Duplicate observations: rows 2i and 2i + 1 of X are equal, so
        // |G_ij| = √(d_i d_j) up to rounding for each pair, and it passes.
        let xd = Mat::<f64>::from_fn(n, p, |i, j| x[(i - i % 2, j)]);
        let (_, gd) = standardized_gram(&xd);
        pls1_perm_alpha(gd.as_ref(), yy.as_ref(), 1, p).expect("parallel rows pass the checks");
    }
}
