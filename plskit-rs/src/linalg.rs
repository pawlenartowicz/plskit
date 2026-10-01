//! Linear-algebra and small-stat helpers shared across the core.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatMut, MatRef, Par};

// ── Explicit-`Par` products and decompositions ──
//
// faer's operator `*` and its high-level decompositions (`thin_svd`,
// `self_adjoint_eigen`, `PartialPivLu`) take their parallelism from
// `faer::get_global_parallelism()`, which defaults to `Par::rayon(0)`: a
// degree equal to the size of the Rayon pool the call runs in. The degree
// sets how faer splits a reduction (a column-major GEMV sums one partial
// product per piece), so those calls round differently under different
// `RAYON_NUM_THREADS`. The crate calls the functions below instead, with
// `Par::Seq` inside replicate workers and `fit::par_fixed()` (or
// `resample::block_par`) elsewhere, so no result depends on the pool size.

/// `a · b` with an explicit `par`: the call operator `*` makes (zeroed
/// destination, `Accum::Replace`, `alpha = 1`), minus the global read.
pub(crate) fn mat_mul(a: MatRef<'_, f64>, b: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let mut out = Mat::<f64>::zeros(a.nrows(), b.ncols());
    matmul(out.as_mut(), Accum::Replace, a, b, 1.0, par);
    out
}

/// `a · v` with an explicit `par`: see [`mat_mul`].
pub(crate) fn mat_vec(a: MatRef<'_, f64>, v: ColRef<'_, f64>, par: Par) -> Col<f64> {
    let mut out = Col::<f64>::zeros(a.nrows());
    matmul(
        out.as_mut().as_mat_mut(),
        Accum::Replace,
        a,
        v.as_mat(),
        1.0,
        par,
    );
    out
}

/// Thin SVD `a = U diag(s) V'` with an explicit `par`: the computation of
/// faer's `thin_svd`, minus the global read. `U` is `m × min(m, n)`, `s`
/// has `min(m, n)` entries in non-increasing order, `V` is
/// `n × min(m, n)`.
pub(crate) struct ThinSvd {
    /// Left singular vectors.
    pub(crate) u: Mat<f64>,
    /// Singular values, non-increasing.
    pub(crate) s: Col<f64>,
    /// Right singular vectors.
    pub(crate) v: Mat<f64>,
}

/// See [`ThinSvd`].
#[allow(clippy::many_single_char_names)]
pub(crate) fn thin_svd(
    a: MatRef<'_, f64>,
    par: Par,
) -> Result<ThinSvd, faer::linalg::svd::SvdError> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
    let (m, n) = a.shape();
    let size = m.min(n);
    let mut u = Mat::<f64>::zeros(m, size);
    let mut v = Mat::<f64>::zeros(n, size);
    let mut s = faer::diag::Diag::<f64>::zeros(size);
    let mut mem = MemBuffer::new(svd_scratch::<f64>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        faer::Spec::default(),
    ));
    svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut mem),
        faer::Spec::default(),
    )?;
    Ok(ThinSvd {
        u,
        s: s.column_vector().to_owned(),
        v,
    })
}

/// Singular values of `a`, non-increasing, with an explicit `par`: faer's
/// `singular_values`, minus the global read.
pub(crate) fn singular_values(
    a: MatRef<'_, f64>,
    par: Par,
) -> Result<Col<f64>, faer::linalg::svd::SvdError> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
    let (m, n) = a.shape();
    let mut s = faer::diag::Diag::<f64>::zeros(m.min(n));
    let mut mem = MemBuffer::new(svd_scratch::<f64>(
        m,
        n,
        ComputeSvdVectors::No,
        ComputeSvdVectors::No,
        par,
        faer::Spec::default(),
    ));
    svd(
        a,
        s.as_mut(),
        None,
        None,
        par,
        MemStack::new(&mut mem),
        faer::Spec::default(),
    )?;
    Ok(s.column_vector().to_owned())
}

/// Eigendecomposition of the self-adjoint `a` with an explicit `par`:
/// faer's `self_adjoint_eigen(Side::Lower)`, minus the global read.
/// Returns `(λ, U)` with the eigenvalues ascending and `U`'s columns the
/// matching eigenvectors. Only the lower triangle of `a` is read.
pub(crate) fn self_adjoint_eigen(
    a: MatRef<'_, f64>,
    par: Par,
) -> Result<(Col<f64>, Mat<f64>), faer::linalg::evd::EvdError> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
    let n = a.nrows();
    let mut u = Mat::<f64>::zeros(n, n);
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        par,
        faer::Spec::default(),
    ));
    self_adjoint_evd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        par,
        MemStack::new(&mut mem),
        faer::Spec::default(),
    )?;
    Ok((s.column_vector().to_owned(), u))
}

/// The rotation `procrustes::orthogonal(a, reference, false)` returns:
/// `R = U V'` from the SVD `a' reference = U Σ V'`, the orthogonal `K × K`
/// matrix minimizing `‖a R − reference‖_F`, formed the same way (`Par::Seq`
/// products, `U V'`). That crate takes the SVD through faer's `svd()`,
/// which reads the global parallelism; this one runs it on `Par::Seq`.
/// That crate asks for the full SVD and this for the thin one; faer's
/// `svd` distinguishes the two only in the size of `U` and `V` (`m` vs
/// `min(m, n)` columns), which coincide for the square `K × K` input, so
/// the computation is the same.
/// NaN-filled when the SVD fails to converge, as there. The caller
/// guarantees `a` and `reference` have the same non-zero shape.
pub(crate) fn orthogonal_rotation(a: MatRef<'_, f64>, reference: MatRef<'_, f64>) -> Mat<f64> {
    let k = a.ncols();
    let m = mat_mul(a.transpose(), reference, Par::Seq);
    let Ok(svd) = thin_svd(m.as_ref(), Par::Seq) else {
        return Mat::<f64>::from_fn(k, k, |_, _| f64::NAN);
    };
    mat_mul(svd.u.as_ref(), svd.v.transpose(), Par::Seq)
}

/// Solve `a x = b` in place of `b` (square `a`, any number of right-hand
/// sides) by LU with partial pivoting, sequentially: faer's
/// `PartialPivLu::new(a).solve_in_place(b)`, minus the global read. The
/// factorizations here are `k × k`; `Par::Seq` costs nothing at that size
/// and is what faer's own LU threshold (`128²`) picks below it.
pub(crate) fn lu_solve_in_place(a: MatRef<'_, f64>, b: MatMut<'_, f64>) {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::lu::partial_pivoting::{factor, solve};
    let n = a.nrows();
    let mut lu = a.to_owned();
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(
        factor::lu_in_place_scratch::<usize, f64>(n, n, Par::Seq, faer::Spec::default()).or(
            solve::solve_in_place_scratch::<usize, f64>(n, b.ncols(), Par::Seq),
        ),
    );
    let stack = MemStack::new(&mut mem);
    let (_, p) = factor::lu_in_place(
        lu.as_mut(),
        &mut perm,
        &mut perm_inv,
        Par::Seq,
        stack,
        faer::Spec::default(),
    );
    // The unit-lower solve reads only the strict lower triangle of `lu`
    // and the upper solve only its upper triangle, so the packed factor
    // stands in for faer's split `L` and `U`.
    solve::solve_in_place_with_conj(
        lu.as_ref(),
        lu.as_ref(),
        p,
        faer::Conj::No,
        b,
        Par::Seq,
        stack,
    );
}

/// Row subset (`row_subset(x, &idx)`). Replaces ndarray's `x.select(Axis(0), &idx)`.
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - `idx`: indices in `0..n_samples`
/// - returns: `(idx.len(), n_features)`
#[must_use]
pub fn row_subset(x: MatRef<'_, f64>, idx: &[usize]) -> Mat<f64> {
    Mat::<f64>::from_fn(idx.len(), x.ncols(), |i, j| x[(idx[i], j)])
}

/// Column-vector row subset.
#[must_use]
pub fn col_row_subset(y: ColRef<'_, f64>, idx: &[usize]) -> Col<f64> {
    Col::<f64>::from_fn(idx.len(), |i| y[idx[i]])
}

/// Column-wise z-score. Returns (`X_standardized`, mean, scale).
/// A column that is constant to rounding (`constant_to_rounding` on its
/// centered and uncentered sums of squares), or whose std is below the
/// normal range, gets scale `max(1, |mean|)` and is only centered, so it
/// standardizes to zeros up to rounding at any magnitude. The rule is
/// relative to the column's own magnitude, so rescaling a column by a
/// positive constant never changes whether it is rescaled. `ddof = 0`
/// (population std, like numpy default).
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - returns `(xs: (n_samples, n_features), mean: (n_features,), scale: (n_features,))`
#[must_use]
pub fn standardize(x: MatRef<'_, f64>) -> (Mat<f64>, Col<f64>, Col<f64>) {
    standardize_weighted(x, None)
}

/// Weighted column-wise z-score. `weights = None` matches `standardize` bit-for-bit.
/// `weights = Some(w)` renormalizes to `w' = w · n / Σw` (mean 1), then uses
/// `mean = Σ w'x / n` and `var = Σ w'(x − mean)² / n` (population, ddof=0).
/// A column is classified as constant, and gets scale `max(1, |mean|)`
/// (as does one whose std is below the normal range), by
/// `constant_to_rounding` applied to `Σ w'(x − mean)²` and `Σ w'x²`,
/// both formed on the column divided by a power of two near its largest
/// entry so that neither overflows nor underflows at any magnitude (see
/// `scaled_moments`; the division is exact and leaves every output bit
/// unchanged wherever the unscaled sums were in range).
/// Caller must ensure weights are non-negative, finite, and Σw > 0
/// (validation lives in `validate_and_normalize_weights` / `preprocess`).
#[must_use]
pub fn standardize_weighted(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> (Mat<f64>, Col<f64>, Col<f64>) {
    let w_prime: Option<Col<f64>> = weights.map(|w| renormalize_mean_one_n(w, x.nrows()));
    let (xs, mean, scale, _) =
        standardize_columns(x, None, w_prime.as_ref().map(owned_col_slice), None, true);
    (xs, mean, scale)
}

/// Apply previously-computed (mean, scale) to a fresh matrix. A
/// column-major `x` is written column by column over contiguous slices;
/// any other layout element by element. The element expression
/// `(x - mean) / scale` is the same either way.
#[must_use]
pub fn standardize_apply(
    x: MatRef<'_, f64>,
    mean: ColRef<'_, f64>,
    scale: ColRef<'_, f64>,
) -> Mat<f64> {
    let Some(xc) = x.try_as_col_major() else {
        return Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| {
            standardized(x[(i, j)], mean[j], scale[j])
        });
    };
    let mut out = Mat::<f64>::zeros(x.nrows(), x.ncols());
    for j in 0..x.ncols() {
        write_standardized(
            out.col_as_slice_mut(j),
            xc.col(j).as_slice(),
            mean[j],
            scale[j],
        );
    }
    out
}

/// `standardize_weighted(row_subset(x, idx), w)`, computed in one pass:
/// the rows are read through `idx` by the same moment kernel and written
/// with the same element expression, so only one block of at most eight
/// gathered columns is held at a time and no bit changes. `w` is indexed
/// by output row (length `idx.len()`) and is renormalized to mean one for
/// the moments, as `standardize_weighted` does. `row_scale`, when given,
/// multiplies output row `i` as `v * row_scale[i]` on the write: the √w
/// row scaling of the split statistics, fused.
pub(crate) fn standardize_rows(
    x: MatRef<'_, f64>,
    idx: &[usize],
    w: Option<ColRef<'_, f64>>,
    row_scale: Option<ColRef<'_, f64>>,
) -> (Mat<f64>, Col<f64>, Col<f64>) {
    let w_prime: Option<Col<f64>> = w.map(|w| renormalize_mean_one_n(w, idx.len()));
    let row_scale: Option<Vec<f64>> = row_scale.map(|r| (0..r.nrows()).map(|i| r[i]).collect());
    let (xs, mean, scale, _) = standardize_columns(
        x,
        Some(idx),
        w_prime.as_ref().map(owned_col_slice),
        row_scale.as_deref(),
        true,
    );
    (xs, mean, scale)
}

/// `standardize_apply(row_subset(x, idx), mean, scale)` in one pass, with
/// the optional per-row factor of [`standardize_rows`].
pub(crate) fn standardize_apply_rows(
    x: MatRef<'_, f64>,
    idx: &[usize],
    mean: ColRef<'_, f64>,
    scale: ColRef<'_, f64>,
    row_scale: Option<ColRef<'_, f64>>,
) -> Mat<f64> {
    match row_scale {
        None => Mat::<f64>::from_fn(idx.len(), x.ncols(), |i, j| {
            standardized(x[(idx[i], j)], mean[j], scale[j])
        }),
        Some(r) => Mat::<f64>::from_fn(idx.len(), x.ncols(), |i, j| {
            standardized(x[(idx[i], j)], mean[j], scale[j]) * r[i]
        }),
    }
}

/// `Σw` in index order: the sum [`normalize_weights`] checks for zero
/// before renormalizing.
fn weight_sum(w: ColRef<'_, f64>) -> f64 {
    (0..w.nrows()).map(|i| w[i]).sum()
}

/// `w · n / Σw`, the mean-one renormalization `standardize_weighted`,
/// `standardize1_weighted` and `normalize_weights` perform, factored out.
/// Same expression, same summation order.
pub(crate) fn renormalize_mean_one(w: ColRef<'_, f64>) -> Col<f64> {
    renormalize_mean_one_n(w, w.nrows())
}

/// `w[0..n] · n / Σ w[0..n]`, in index order: the same mean-one
/// renormalization as [`renormalize_mean_one`], but restricted to the
/// first `n` entries of `w` and using `n` (not `w.nrows()`) as both the
/// summation bound and the count. `standardize_weighted` and
/// `standardize1_weighted` sum and rebuild only their own `n_rows` / `n`
/// elements of `weights`, so a `weights` column longer than the rows being
/// standardized (reachable from a Rust caller that passes a longer
/// `weights` to the public `standardize_weighted`) gives the same bits as
/// one cut to `n` rows, whereas [`renormalize_mean_one`] would read the
/// rest of `w` and divide by a different count.
fn renormalize_mean_one_n(w: ColRef<'_, f64>, n: usize) -> Col<f64> {
    let s: f64 = (0..n).map(|i| w[i]).sum();
    let n_f = n as f64;
    Col::<f64>::from_fn(n, |i| w[i] * n_f / s)
}

/// `√w[i]` for every `i`: the row factor of the √w row-scaling convention.
///
/// The √w row-scaling convention is how `pls1_fit` handles weights:
/// standardize X and y with weighted moments (mean-one weights `w'`), then
/// multiply row `i` of both by `√w'[i]` and run the unweighted algorithm on
/// the result.
pub(crate) fn sqrt_col(w: ColRef<'_, f64>) -> Col<f64> {
    Col::<f64>::from_fn(w.nrows(), |i| w[i].sqrt())
}

/// A column-major copy of `x` when `x` is not column-major (or is
/// column-major with a negative column stride); `None` when it is
/// column-major with a nonnegative column stride, and the caller then reads
/// `x` itself.
///
/// Borrowing is value-identical to copying, and on a column-major,
/// nonnegative-column-stride view it is also bit-identical downstream: the kernels that read the
/// caller's view directly (faer's `norm_l2` for the NIPALS floor, and the
/// products of the score test) take the same path on such a view as on an
/// owned copy, whatever its column stride or offset; the per-module
/// layout-invariance tables over `test_support::Layouts` assert it (the
/// padded-submatrix view is borrowed), e.g. the score `pre_standardized`
/// rows of
/// `signal_test::tests::confirmatory_output_is_bit_identical_across_x_layouts`.
/// A row-major or strided view takes other paths there (`norm_l2` transposes a row-major view and walks a strided
/// one with a scalar `hypot` loop), which could move a last bit, so it is
/// still copied. A column-major view with a *negative* column stride
/// (e.g. `reverse_cols()`) passes `try_as_col_major` (which only checks
/// `row_stride() == 1`) but is not safe to borrow: faer's matmul reverses
/// the k order for a left operand with negative column stride, which can
/// move a last bit in `xs_eff * xs_eff.transpose()` (`signal_test::run_score`).
#[must_use]
pub(crate) fn col_major_or_copy(x: MatRef<'_, f64>) -> Option<Mat<f64>> {
    if x.try_as_col_major().is_some() && x.col_stride() >= 0 {
        None
    } else {
        Some(Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)]))
    }
}

/// Standardize a 1-D vector. Returns (z, mean, scale): the same result as
/// standardizing it as a one-column matrix.
#[must_use]
pub fn standardize1(y: ColRef<'_, f64>) -> (Col<f64>, f64, f64) {
    standardize1_weighted(y, None)
}

/// Weighted scalar-standardize for y. None ⇒ unweighted. Same moments and
/// the same constant rule as one column of [`standardize_weighted`].
/// Caller is responsible for weight validation; see `validate_and_normalize_weights`.
#[must_use]
pub fn standardize1_weighted(
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> (Col<f64>, f64, f64) {
    let n = y.nrows();
    let w_prime: Option<Col<f64>> = weights.map(|w| renormalize_mean_one_n(w, n));
    let wpref = w_prime.as_ref().map(Col::as_ref);
    let (mean, scale) = mean_and_scale(n, |i| y[i], wpref);
    let z = Col::<f64>::from_fn(n, |i| (y[i] - mean) / scale);
    (z, mean, scale)
}

/// `(mean, scale)` of one column `v(0)..v(n−1)` under mean-one weights `w`
/// (`None` for unit weights): the kernel of [`standardize1_weighted`].
/// [`standardize_weighted`] (and [`standardize_rows`]) forms the same bits
/// for each of its columns through [`scaled_moments_block`] and
/// [`mean_and_scale_from`].
///
/// The moments are formed on the column divided by a power of two `s` near
/// its largest absolute value ([`pow2_scale`]), so the squares inside
/// [`constant_to_rounding`] and the variance stay finite and normal at any
/// magnitude. Summed on the raw column, `Σx²` is `inf` once `|x|` exceeds
/// about `1e154` (and a varying column there would be classified as
/// constant, with scale `inf`), and is `0` once `|x|` is below about
/// `1e-162` (a varying column there would be classified as constant too).
///
/// Dividing by a power of two is exact, and IEEE rounding commutes with it
/// while no intermediate leaves the normal range, so on such a column every
/// sum here is the raw-column sum times `2^-e` (squares times `2^-2e`), the
/// constant test takes the same branch, and `s·mean`, `s·sqrt(ss/n)` are
/// the bits the raw-column formulas give. Outputs differ from the raw
/// formulas only where those overflowed or lost precision to underflow.
#[allow(clippy::cast_precision_loss)]
fn mean_and_scale(n: usize, v: impl Fn(usize) -> f64, w: Option<ColRef<'_, f64>>) -> (f64, f64) {
    mean_and_scale_from(&scaled_moments(n, v, w), n)
}

/// `(mean, scale)` of a length-`n` column from its [`ScaledMoments`]:
/// mean `s·mean`, and scale `sd = s·sqrt(ss/n)`, or `max(1, |mean|)` when
/// the column carries no information: [`constant_to_rounding`] reads it as
/// constant, or `sd` is below the normal range (`0` or subnormal, e.g. a
/// single `5e-324` among zeros), where it cannot be formed accurately and
/// `1/sd` may overflow. The tail of [`mean_and_scale`], shared with
/// [`standardize_columns`] and [`row_major_moments`], so every standardizer
/// and both fit routes see the same scale.
///
/// Such a column is only centered, and its centered entries are rounding
/// noise of size up to about `2n·ε·|mean|`. Dividing by `max(1, |mean|)`
/// keeps them below about `2n·ε` at any magnitude: with scale 1 a constant
/// near `1e300` would standardize to entries near `1e285`, which would
/// dominate `‖Xs‖_F` (and the fit's truncation floor) or overflow the fit
/// to NaN.
/// A constant of magnitude at most 1, the all-zero column included, keeps
/// scale 1. A constant column's `|mean| / scale` is at most 1, so one of
/// magnitude above `1e3` does not push `pls1_fit` past
/// `IMPLICIT_MAX_MEAN_RATIO` onto the materialized-copy route
/// ([`FitX::max_mean_ratio`]).
#[allow(clippy::cast_precision_loss)]
fn mean_and_scale_from(m: &ScaledMoments, n: usize) -> (f64, f64) {
    let mean = m.mean * m.s;
    let sd = (m.ss / n as f64).sqrt() * m.s;
    // `sd < MIN_POSITIVE` is false for a NaN `sd` (a non-finite column),
    // which keeps its NaN scale.
    let scale = if m.is_constant(n) || sd < f64::MIN_POSITIVE {
        mean.abs().max(1.0)
    } else {
        sd
    };
    (mean, scale)
}

/// The body of [`standardize_weighted`], [`standardize_rows`] and the
/// column-major arm of [`fit_x_moments`] once the weights are mean-one (`w`, one per
/// output row; `None` for unit weights). The input is `x` itself
/// (`rows = None`) or its rows `idx` in that order (`rows = Some(idx)`,
/// repeats allowed). Columns go through [`scaled_moments_block`]
/// `STD_BLOCK` at a time (a shorter last block one column at a time), and
/// each block is then appended to the output as `(v - mean) / scale`,
/// times `row_scale[i]` when given, while it is still in cache; every
/// output entry is written once. A column-major `x` read without `rows` is
/// used in place; otherwise each block is copied, exactly, into a
/// column-major scratch buffer first: from a row-major `x` (such as a view
/// of a C-ordered host array) row by row over contiguous slices, from any
/// other layout column by column. Sharing this body is what makes the row
/// helpers byte-identical to standardizing a gathered copy, and what makes
/// every layout of `x` give the same bits. The fourth value is
/// [`fro_from_terms`] of the columns' [`fro_term`]s: the Frobenius norm of
/// the result when `row_scale` is `None` or `row_scale[i]² = w[i]`. With
/// `write` false only the moments are formed, and the matrix returned is
/// `n_rows × 0`.
fn standardize_columns(
    x: MatRef<'_, f64>,
    rows: Option<&[usize]>,
    w: Option<&[f64]>,
    row_scale: Option<&[f64]>,
    write: bool,
) -> (Mat<f64>, Col<f64>, Col<f64>, f64) {
    let n_rows = rows.map_or(x.nrows(), <[usize]>::len);
    let n_cols = x.ncols();
    // Read in place only when every row is taken, in order, from a column-major x.
    let xc = if rows.is_none() {
        x.try_as_col_major()
    } else {
        None
    };
    let xr = if xc.is_none() {
        x.try_as_row_major()
    } else {
        None
    };
    let mut scratch: Vec<f64> = Vec::new();
    if xr.is_some() {
        scratch.resize(STD_BLOCK * n_rows, 0.0);
    }
    let mut block_moments: Vec<ScaledMoments> = Vec::with_capacity(STD_BLOCK);
    // Columns are appended one block at a time, each written once: no
    // zero-fill pass. Same capacity request, so the same column stride and
    // alignment as `Mat::zeros(n_rows, n_cols)`.
    let mut xs = Mat::<f64>::with_capacity(n_rows, if write { n_cols } else { 0 });
    // Set the row count now (no entry is created), so that an input with
    // no columns still gives an `n_rows × 0` result.
    xs.resize_with(n_rows, 0, |_, _| 0.0);
    let mut mean = Col::<f64>::zeros(n_cols);
    let mut scale = Col::<f64>::zeros(n_cols);
    let mut fro = Vec::with_capacity(n_cols);
    for j0 in (0..n_cols).step_by(STD_BLOCK) {
        let width = STD_BLOCK.min(n_cols - j0);
        if let Some(xr) = xr {
            for i in 0..n_rows {
                let r = rows.map_or(i, |idx| idx[i]);
                let src = &xr.row(r).as_slice()[j0..j0 + width];
                for (b, &v) in src.iter().enumerate() {
                    scratch[b * n_rows + i] = v;
                }
            }
        } else if xc.is_none() {
            scratch.clear();
            for j in j0..j0 + width {
                let col = x.col(j);
                match rows {
                    None => scratch.extend(col.iter().copied()),
                    Some(idx) => scratch.extend(idx.iter().map(|&r| col[r])),
                }
            }
        }
        let cols: [&[f64]; STD_BLOCK] = core::array::from_fn(|b| {
            if b >= width {
                &[][..]
            } else if let Some(xc) = xc {
                xc.col(j0 + b).as_slice()
            } else {
                &scratch[b * n_rows..(b + 1) * n_rows]
            }
        });
        block_moments.clear();
        if width == STD_BLOCK {
            block_moments.extend(scaled_moments_block(cols, w));
        } else {
            for &col in &cols[..width] {
                block_moments.extend(scaled_moments_block([col], w));
            }
        }
        let mut stats = [(0.0_f64, 1.0_f64); STD_BLOCK];
        for (b, moments) in block_moments.iter().enumerate() {
            let (col_mean, col_scale) = mean_and_scale_from(moments, n_rows);
            mean[j0 + b] = col_mean;
            scale[j0 + b] = col_scale;
            stats[b] = (col_mean, col_scale);
            fro.push(fro_term(moments, col_scale));
        }
        if !write {
            continue;
        }
        // The element expression of every standardizer, times the row
        // factor when given.
        match row_scale {
            None => xs.resize_with(n_rows, j0 + width, |i, j| {
                let (col_mean, col_scale) = stats[j - j0];
                standardized(cols[j - j0][i], col_mean, col_scale)
            }),
            Some(factor) => xs.resize_with(n_rows, j0 + width, |i, j| {
                let (col_mean, col_scale) = stats[j - j0];
                standardized(cols[j - j0][i], col_mean, col_scale) * factor[i]
            }),
        }
    }
    (xs, mean, scale, fro_from_terms(&fro))
}

/// `(v - mean) / scale`: the element expression of every standardizer,
/// in one place so that its copies cannot drift apart.
#[inline]
fn standardized(v: f64, mean: f64, scale: f64) -> f64 {
    (v - mean) / scale
}

/// `dst[i] = standardized(src[i], mean, scale)` for every `i`, over one
/// contiguous column.
fn write_standardized(dst: &mut [f64], src: &[f64], mean: f64, scale: f64) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = standardized(v, mean, scale);
    }
}

/// Contribution `r` of one standardized column to `‖Xs‖_F`, with
/// `r² = Σ w'ᵢ((xᵢ − mean)/scale)² = s²·ss / scale²`, formed as
/// `√ss · (s / scale)` so that no intermediate overflows. `√n` (to
/// rounding) for a non-constant column; for a constant one the rounding
/// noise of its centering, at most about `2n·ε·√n` (its scale is
/// `max(1, |mean|)`).
fn fro_term(m: &ScaledMoments, scale: f64) -> f64 {
    m.ss.sqrt() * (m.s / scale)
}

/// `√(Σ rⱼ²)` over the [`fro_term`]s, each divided by the largest `|rⱼ|`
/// before it is squared, so that no `r²` can overflow (with scale 1, a
/// constant column of large entries would give `r` near `1e285`).
/// Non-finite when a term is.
fn fro_from_terms(r: &[f64]) -> f64 {
    let big = r.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if big == 0.0 || !big.is_finite() {
        return big;
    }
    let sq: f64 = r
        .iter()
        .map(|&v| {
            let q = v / big;
            q * q
        })
        .sum();
    big * sq.sqrt()
}

/// The standardization of `X` for `pls1_fit`: its moments, and the
/// standardized matrix itself once [`FitX::materialize`] has written it, in the
/// input's own layout.
pub(crate) struct FitX {
    /// The `n × d` result; for a row-major result its `d × n` transpose, so
    /// that row `i` of the result is the contiguous storage column `i`.
    /// `None` until materialized.
    storage: Option<Mat<f64>>,
    row_major: bool,
    /// Column means, the bits of [`standardize_weighted`]'s.
    pub(crate) mean: Col<f64>,
    /// Column scales, the bits of [`standardize_weighted`]'s.
    pub(crate) scale: Col<f64>,
    /// `‖result‖_F` from the moments ([`fro_from_terms`] of the columns'
    /// [`fro_term`]s): equal to the norm of the stored result, to rounding, when
    /// `row_scale` is `None` or `row_scale[i]² = w'ᵢ`, as `pls1_fit` passes
    /// it. The fit uses it only for its truncation floor.
    pub(crate) fro: f64,
}

impl FitX {
    /// The `n × d` result.
    ///
    /// # Panics
    /// When the result has not been materialized.
    pub(crate) fn xs(&self) -> MatRef<'_, f64> {
        let storage = self.storage.as_ref().expect("FitX::xs before materialize");
        if self.row_major {
            storage.as_ref().transpose()
        } else {
            storage.as_ref()
        }
    }

    /// `max_j |mean_j| / scale_j`: how far the columns sit from zero in units
    /// of their scale, which bounds the cancellation in products formed from
    /// the raw `x` instead of the standardized matrix.
    pub(crate) fn max_mean_ratio(&self) -> f64 {
        let (mean, scale) = (owned_col_slice(&self.mean), owned_col_slice(&self.scale));
        mean.iter()
            .zip(scale)
            .fold(0.0_f64, |m, (&mu, &s)| m.max(mu.abs() / s))
    }

    /// Writes the standardized matrix from `x` (the matrix the moments were
    /// taken of): each entry is `(x − mean) / scale · row_scale[i]`, the
    /// element expression of every standardizer, so the bits do not depend
    /// on the order it is written in. A row-major `x` gives a row-major
    /// result, written one contiguous row at a time; any other layout gives
    /// a column-major one.
    pub(crate) fn materialize(&mut self, x: MatRef<'_, f64>, row_scale: Option<ColRef<'_, f64>>) {
        let (n, d) = (x.nrows(), x.ncols());
        let factor: Option<Vec<f64>> = row_scale.map(|r| (0..r.nrows()).map(|i| r[i]).collect());
        let (mean, scale) = (owned_col_slice(&self.mean), owned_col_slice(&self.scale));
        let storage = if self.row_major {
            let xr = x
                .try_as_row_major()
                .expect("the moments were read row-major");
            let mut storage = Mat::<f64>::zeros(d, n);
            for i in 0..n {
                let terms = storage
                    .col_as_slice_mut(i)
                    .iter_mut()
                    .zip(xr.row(i).as_slice())
                    .zip(mean)
                    .zip(scale);
                match &factor {
                    None => {
                        for (((o, &v), &m), &s) in terms {
                            *o = standardized(v, m, s);
                        }
                    }
                    Some(f) => {
                        let fi = f[i];
                        for (((o, &v), &m), &s) in terms {
                            *o = standardized(v, m, s) * fi;
                        }
                    }
                }
            }
            storage
        } else {
            // Same capacity request as `standardize_columns`, so the same
            // column stride and alignment.
            let mut xs = Mat::<f64>::with_capacity(n, d);
            match &factor {
                None => xs.resize_with(n, d, |i, j| standardized(x[(i, j)], mean[j], scale[j])),
                Some(f) => {
                    xs.resize_with(n, d, |i, j| {
                        standardized(x[(i, j)], mean[j], scale[j]) * f[i]
                    });
                }
            }
            xs
        };
        self.storage = Some(storage);
    }
}

/// The moments of `standardize_weighted(x, weights)`: its `mean` and `scale`,
/// bit for bit, and `fro`, with nothing written. A row-major `x` (a C-ordered
/// host array read in place) is read over its contiguous rows, one at a time
/// or in blocks on a wide `x` ([`row_major_moments`]); any other layout through
/// [`standardize_columns`].
/// [`FitX::materialize`] then writes the standardized matrix in `x`'s layout
/// when the caller needs it. The kernel's products round differently on the
/// two layouts, so a fit's last bits depend on the layout of `x`, within the
/// crate's tolerance.
pub(crate) fn fit_x_moments(x: MatRef<'_, f64>, weights: Option<ColRef<'_, f64>>) -> FitX {
    let w_prime: Option<Col<f64>> = weights.map(|w| renormalize_mean_one_n(w, x.nrows()));
    let w = w_prime.as_ref().map(owned_col_slice);
    if x.try_as_col_major().is_none() && x.try_as_row_major().is_some() {
        return row_major_moments(x, w);
    }
    let (_, mean, scale, fro) = standardize_columns(x, None, w, None, false);
    FitX {
        storage: None,
        row_major: false,
        mean,
        scale,
        fro,
    }
}

/// Rows the row-major kernels ([`row_major_moments`],
/// [`row_major_t_mul`]) advance together on a wide matrix: each pass then
/// reads and writes its per-column running sums once per block of this many
/// rows, not once per row. The rows of a block are read in order, so every
/// column's sum still adds its terms one row at a time.
const ROW_BLOCK: usize = 8;

/// The row width from which the row-major kernels block rows
/// ([`ROW_BLOCK`]). One row at a time, a pass loads and stores its per-column
/// sums for every entry; that is cheap while they stay in the L1 cache and
/// dominates once they spill (with faer's `X'·t`, which does the same, a
/// C-ordered `1000 × 10000` X fitted about 1.4× slower than its column-major
/// copy). Blocked, the rows of a block are read as that many streams, which
/// costs more than it saves on narrow rows (measured on an Apple M4: slower
/// at 1000 columns, even at 2000, faster from 4000). Below it the product
/// [`row_major_t_mul`] declines and the caller keeps faer's.
pub(crate) const ROW_BLOCK_MIN_COLS: usize = 2048;

/// One pass of a row-major kernel: the rows of `x`, in order, `K` at a time
/// (a last, shorter block one row at a time), each with its per-row factor
/// (a weight, or an entry of the vector `x` is multiplied by).
trait RowPass {
    /// Adds the rows `rows` (with factors `f`, `None` for unit factors) to
    /// the pass's per-column state.
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>);
}

/// Runs `pass` over `rows` (each `d` long) and their factors `f` (one per
/// row), [`ROW_BLOCK`] rows at a time when `d >= ROW_BLOCK_MIN_COLS`,
/// otherwise one at a time. Either way the rows reach each column in order,
/// so the blocking changes no bit.
fn run_rows(rows: &[&[f64]], f: Option<&[f64]>, d: usize, pass: &mut impl RowPass) {
    fn blocks<const R: usize>(rows: &[&[f64]], f: Option<&[f64]>, pass: &mut impl RowPass) {
        let mut chunks = rows.chunks_exact(R);
        for (b, chunk) in (&mut chunks).enumerate() {
            let blk: [&[f64]; R] = core::array::from_fn(|k| chunk[k]);
            pass.rows(blk, f.map(|f| core::array::from_fn(|k| f[b * R + k])));
        }
        let i0 = rows.len() - chunks.remainder().len();
        for (t, &r) in chunks.remainder().iter().enumerate() {
            pass.rows([r], f.map(|f| [f[i0 + t]]));
        }
    }
    if d >= ROW_BLOCK_MIN_COLS {
        blocks::<ROW_BLOCK>(rows, f, pass);
    } else {
        blocks::<1>(rows, f, pass);
    }
}

/// The contiguous rows of a row-major `x`, each `ncols` long.
fn row_slices(x: MatRef<'_, f64>) -> Vec<&[f64]> {
    let xr = x.try_as_row_major().expect("the caller checked the layout");
    let d = x.ncols();
    (0..x.nrows()).map(|i| &xr.row(i).as_slice()[..d]).collect()
}

/// `max |x|` per column. `f64::max` ignores a NaN operand, as the fold in
/// `scaled_moments` does; the order of a max does not matter.
struct MaxAbs<'a> {
    max_abs: &'a mut [f64],
}

impl RowPass for MaxAbs<'_> {
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], _f: Option<[f64; K]>) {
        let d = self.max_abs.len();
        let rows = rows.map(|r| &r[..d]);
        for (j, m) in self.max_abs.iter_mut().enumerate() {
            let mut v = *m;
            for r in &rows {
                v = v.max(r[j].abs());
            }
            *m = v;
        }
    }
}

/// The mean sum `Σ w·u` and `Σ w·(u·u)` per column, `u = x·inv`: the terms
/// of [`scaled_moments_block`]'s second pass.
struct SumSq<'a> {
    sum: &'a mut [f64],
    sq: &'a mut [f64],
    inv: &'a [f64],
}

#[allow(clippy::many_single_char_names)]
impl RowPass for SumSq<'_> {
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>) {
        let d = self.sum.len();
        let (sum, sq, inv) = (&mut *self.sum, &mut self.sq[..d], &self.inv[..d]);
        let rows = rows.map(|r| &r[..d]);
        match f {
            None => {
                for j in 0..d {
                    let (mut s, mut q, iv) = (sum[j], sq[j], inv[j]);
                    for r in &rows {
                        let u = r[j] * iv;
                        s += u;
                        q += u * u;
                    }
                    sum[j] = s;
                    sq[j] = q;
                }
            }
            Some(w) => {
                for j in 0..d {
                    let (mut s, mut q, iv) = (sum[j], sq[j], inv[j]);
                    for (r, &wi) in rows.iter().zip(&w) {
                        let u = r[j] * iv;
                        s += wi * u;
                        q += wi * (u * u);
                    }
                    sum[j] = s;
                    sq[j] = q;
                }
            }
        }
    }
}

/// `Σ w·(u − mean)²` per column, `u = x·inv`: the terms of
/// [`scaled_moments_block`]'s third pass.
struct CenteredSq<'a> {
    ss: &'a mut [f64],
    inv: &'a [f64],
    mean: &'a [f64],
}

impl RowPass for CenteredSq<'_> {
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>) {
        let d = self.ss.len();
        let (ss, inv, mean) = (&mut *self.ss, &self.inv[..d], &self.mean[..d]);
        let rows = rows.map(|r| &r[..d]);
        match f {
            None => {
                for j in 0..d {
                    let (mut acc, iv, m) = (ss[j], inv[j], mean[j]);
                    for r in &rows {
                        let dv = r[j] * iv - m;
                        acc += dv * dv;
                    }
                    ss[j] = acc;
                }
            }
            Some(w) => {
                for j in 0..d {
                    let (mut acc, iv, m) = (ss[j], inv[j], mean[j]);
                    for (r, &wi) in rows.iter().zip(&w) {
                        let dv = r[j] * iv - m;
                        acc += wi * (dv * dv);
                    }
                    ss[j] = acc;
                }
            }
        }
    }
}

/// The row-major arm of [`fit_x_moments`]. The moments are those of
/// [`scaled_moments_block`] (the same terms, in the same row order, from the
/// same start, so the same bits), each column's sums advancing together over
/// contiguous rows ([`run_rows`]: one row at a time, or a block of
/// [`ROW_BLOCK`] rows on a wide `x`); change the two together. Three passes
/// over `x`: `max |x|`, the mean sum and `Σu²` together, and `Σ(u − mean)²`,
/// with `u = x·(1/s)`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn row_major_moments(x: MatRef<'_, f64>, w: Option<&[f64]>) -> FitX {
    let (n, d) = (x.nrows(), x.ncols());
    let rows = row_slices(x);
    let zero = sum_start();

    let mut max_abs = vec![0.0_f64; d];
    run_rows(
        &rows,
        None,
        d,
        &mut MaxAbs {
            max_abs: &mut max_abs,
        },
    );
    let scales: Vec<(f64, f64)> = max_abs.iter().map(|&m| pow2_scale(m)).collect();
    let inv: Vec<f64> = scales.iter().map(|&(_, iv)| iv).collect();

    let mut sum = vec![zero; d];
    let mut sq = vec![zero; d];
    run_rows(
        &rows,
        w,
        d,
        &mut SumSq {
            sum: &mut sum,
            sq: &mut sq,
            inv: &inv,
        },
    );
    let n_f = n as f64;
    let mean_u: Vec<f64> = sum.iter().map(|&s| s / n_f).collect();

    let mut ss = vec![zero; d];
    run_rows(
        &rows,
        w,
        d,
        &mut CenteredSq {
            ss: &mut ss,
            inv: &inv,
            mean: &mean_u,
        },
    );

    let mut mean = Col::<f64>::zeros(d);
    let mut scale = Col::<f64>::zeros(d);
    let mut fro = Vec::with_capacity(d);
    for j in 0..d {
        let m = ScaledMoments {
            s: scales[j].0,
            inv: inv[j],
            mean: mean_u[j],
            ss: ss[j],
            sq: sq[j],
        };
        let (col_mean, col_scale) = mean_and_scale_from(&m, n);
        mean[j] = col_mean;
        scale[j] = col_scale;
        fro.push(fro_term(&m, col_scale));
    }
    FitX {
        storage: None,
        row_major: true,
        mean,
        scale,
        fro: fro_from_terms(&fro),
    }
}

/// `x'·t` accumulated into `out` (one sum per column, rows in order).
struct TMul<'a> {
    out: &'a mut [f64],
}

impl RowPass for TMul<'_> {
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>) {
        let t = f.expect("TMul reads its rows' entries of t as factors");
        let d = self.out.len();
        let rows = rows.map(|r| &r[..d]);
        for (j, o) in self.out.iter_mut().enumerate() {
            let mut acc = *o;
            for (r, &ti) in rows.iter().zip(&t) {
                acc += ti * r[j];
            }
            *o = acc;
        }
    }
}

/// `x'·t` for a row-major `x` (a C-ordered host array read in place) at
/// least [`ROW_BLOCK_MIN_COLS`] wide, read one contiguous block of rows at a
/// time ([`run_rows`]). faer forms this product as a column-major GEMV of
/// `x'`, which loads and stores the whole output once per row: on a wide `x`
/// the output leaves the L1 cache and the product ran about 1.6× slower than
/// on the column-major copy. `None` for any other layout or a narrower `x`:
/// the caller then multiplies with faer.
///
/// Column `j` is `Σᵢ tᵢ·xᵢⱼ` from `0.0` in row order. Under `Par::Rayon` the
/// rows are cut into [`crate::fit::PAR_DEGREE`] contiguous pieces, each
/// summed on its own and the pieces' sums then added in piece order, so the
/// bits depend on the shape and on `par`, never on the pool size; `Par::Seq`
/// and `Par::Rayon` can differ in the last bits, as faer's two arms do.
#[allow(clippy::many_single_char_names)]
pub(crate) fn row_major_t_mul(
    x: MatRef<'_, f64>,
    t: ColRef<'_, f64>,
    par: Par,
) -> Option<Col<f64>> {
    let (n, d) = (x.nrows(), x.ncols());
    if d < ROW_BLOCK_MIN_COLS || x.try_as_col_major().is_some() || x.try_as_row_major().is_none() {
        return None;
    }
    let rows = row_slices(x);
    let t: Vec<f64> = (0..n).map(|i| t[i]).collect();
    let piece = |lo: usize, hi: usize| {
        let mut out = vec![0.0_f64; d];
        run_rows(
            &rows[lo..hi],
            Some(&t[lo..hi]),
            d,
            &mut TMul { out: &mut out },
        );
        out
    };
    let out = match par {
        Par::Seq => piece(0, n),
        Par::Rayon(_) => {
            use rayon::prelude::*;
            let p = crate::fit::PAR_DEGREE;
            let parts: Vec<Vec<f64>> = (0..p)
                .into_par_iter()
                .map(|k| piece(k * n / p, (k + 1) * n / p))
                .collect();
            let mut parts = parts.into_iter();
            let mut out = parts.next().expect("PAR_DEGREE >= 1");
            for part in parts {
                for (o, v) in out.iter_mut().zip(&part) {
                    *o += v;
                }
            }
            out
        }
    };
    Some(Col::<f64>::from_fn(d, |j| out[j]))
}

/// `Σ xᵢⱼ²` with no overflow guard, four interleaved partial sums over each
/// contiguous column (column-major) or row (row-major), entry by entry for
/// any other layout. Non-finite when an entry is non-finite or a square
/// overflows (`|x|` above about `1e154`). The summation order depends on the
/// layout; `pls1_fit` uses it only for its truncation floor and as a
/// finiteness screen.
pub(crate) fn sum_of_squares(x: MatRef<'_, f64>) -> f64 {
    #[allow(clippy::needless_range_loop)]
    fn add_slice(v: &[f64], acc: &mut [f64; 4]) {
        let mut chunks = v.chunks_exact(4);
        for c in &mut chunks {
            for l in 0..4 {
                acc[l] += c[l] * c[l];
            }
        }
        for (l, &e) in chunks.remainder().iter().enumerate() {
            acc[l] += e * e;
        }
    }
    let mut acc = [0.0_f64; 4];
    if let Some(xc) = x.try_as_col_major() {
        for j in 0..x.ncols() {
            add_slice(xc.col(j).as_slice(), &mut acc);
        }
    } else if let Some(xr) = x.try_as_row_major() {
        for i in 0..x.nrows() {
            add_slice(xr.row(i).as_slice(), &mut acc);
        }
    } else {
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let e = x[(i, j)];
                acc[0] += e * e;
            }
        }
    }
    acc.iter().sum()
}

/// An owned column's entries as a slice (an owned `Col` is contiguous).
pub(crate) fn owned_col_slice(c: &Col<f64>) -> &[f64] {
    c.try_as_col_major()
        .expect("an owned Col is contiguous")
        .as_slice()
}

/// The [`centered_moments`] of one vector divided by a power of two `s`
/// near its largest absolute value ([`pow2_scale`]), together with `s`.
/// See [`scaled_moments`].
#[derive(Clone, Copy)]
pub(crate) struct ScaledMoments {
    /// The power of two the vector was divided by.
    pub(crate) s: f64,
    /// `1/s`, also a power of two.
    pub(crate) inv: f64,
    /// Mean of `v/s` (weighted by `w` when given).
    pub(crate) mean: f64,
    /// `Σ(vᵢ/s − mean)²` (each term times `wᵢ` when weighted).
    pub(crate) ss: f64,
    /// `Σ(vᵢ/s)²` (each term times `wᵢ` when weighted).
    pub(crate) sq: f64,
}

impl ScaledMoments {
    /// [`constant_to_rounding`] on these moments. The rule is a ratio of
    /// the two sums, so it reads the same on `v/s` as on `v`.
    pub(crate) fn is_constant(&self, n: usize) -> bool {
        constant_to_rounding(self.ss, self.sq, n)
    }
}

/// Moments of `v(0)..v(n−1)` formed on `v/s`, `s` the power of two
/// [`pow2_scale`] picks for the largest `|vᵢ|`, under mean-one weights `w`
/// (`None` for unit weights). The one kernel behind the standardizers
/// ([`mean_and_scale`]; [`standardize_weighted`] and [`standardize_rows`]
/// run its lockstep form [`scaled_moments_block`], which returns the same
/// bits) and the
/// split-half correlations
/// (`signal_test::guarded_pearson`, `signal_test::split_perm_nr_zbars`,
/// PLS3's `pls3_signal_test::pearson_r_guarded`), so a column and a
/// held-out vector are judged constant by literally the same sums. The sums
/// of squares neither overflow nor underflow at any magnitude of `v`:
/// formed on `v` itself, `Σv²` is `inf` once `|v|` exceeds about `1e154`
/// and `0` once it is below about `1e-162`, and `constant_to_rounding` would
/// then read every such vector as constant.
///
/// Dividing by a power of two is exact, and rounding commutes with it while
/// no intermediate leaves the normal range, so where the raw sums are in
/// range these are the raw sums times `1/s` (`mean`) or `1/s²` (`ss`,
/// `sq`), bit for bit: `constant_to_rounding` takes the same branch, and
/// `signal_test::pearson_scaled` returns the bits
/// `signal_test::pearson_from_moments` gives on the raw vectors (the
/// numerator and the square root in its denominator carry the same
/// power-of-two factor).
///
/// Every sum runs in index order. A weighted term is `wᵢ·(d·d)`, not
/// `(wᵢ·d)·d`; the two round differently.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn scaled_moments(
    n: usize,
    v: impl Fn(usize) -> f64,
    w: Option<ColRef<'_, f64>>,
) -> ScaledMoments {
    let max_abs = (0..n).map(|i| v(i).abs()).fold(0.0_f64, f64::max);
    let (s, inv) = pow2_scale(max_abs);
    let u = |i: usize| v(i) * inv;
    let (mean, ss, sq) = match w {
        None => centered_moments(n, u),
        Some(w) => {
            let mean: f64 = (0..n).map(|i| w[i] * u(i)).sum::<f64>() / n as f64;
            let ss: f64 = (0..n)
                .map(|i| {
                    let d = u(i) - mean;
                    w[i] * (d * d)
                })
                .sum();
            let sq: f64 = (0..n).map(|i| w[i] * (u(i) * u(i))).sum();
            (mean, ss, sq)
        }
    };
    ScaledMoments {
        s,
        inv,
        mean,
        ss,
        sq,
    }
}

/// Columns the standardizer hands [`scaled_moments_block`] at a time: the
/// moment sums of this many columns advance together, one row at a time,
/// so the processor overlaps their additions. Each sum is still one chain
/// in row order.
const STD_BLOCK: usize = 8;

/// [`scaled_moments`] of `B` equal-length columns, bit for bit: for each
/// column, the same terms in the same (row) order from the same start as
/// [`scaled_moments`] with `v(i) = col[i]`. The columns advance together,
/// row by row, so their sums (independent chains) overlap in the pipeline;
/// no sum is split or reordered. Three passes over the block: `max |x|`,
/// then the mean sum and `Σu²` together, then `Σ(u − mean)²`, with
/// `u = x·(1/s)`. `w` holds the mean-one weights (one per row), `None`
/// for unit weights; a weighted term is `wᵢ·u` / `wᵢ·(u·u)` /
/// `wᵢ·(d·d)` as in [`scaled_moments`]. [`row_major_moments`] repeats
/// these loops over the rows of a row-major matrix for the same bits:
/// change the two together.
// Row `i` addresses all `B` columns, so the loops index rather than iterate.
#[allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
fn scaled_moments_block<const B: usize>(
    cols: [&[f64]; B],
    w: Option<&[f64]>,
) -> [ScaledMoments; B] {
    let n = cols[0].len();
    let n_f = n as f64;
    let cols: [&[f64]; B] = core::array::from_fn(|b| &cols[b][..n]);
    let w = w.map(|w| &w[..n]);

    let mut max_abs = [0.0_f64; B];
    for i in 0..n {
        for b in 0..B {
            max_abs[b] = max_abs[b].max(cols[b][i].abs());
        }
    }
    let scales: [(f64, f64); B] = core::array::from_fn(|b| pow2_scale(max_abs[b]));
    let inv: [f64; B] = core::array::from_fn(|b| scales[b].1);

    let zero = sum_start();
    let mut sum = [zero; B];
    let mut sq = [zero; B];
    let mut ss = [zero; B];
    let mean: [f64; B];
    match w {
        None => {
            for i in 0..n {
                for b in 0..B {
                    let u = cols[b][i] * inv[b];
                    sum[b] += u;
                    sq[b] += u * u;
                }
            }
            mean = core::array::from_fn(|b| sum[b] / n_f);
            for i in 0..n {
                for b in 0..B {
                    let d = cols[b][i] * inv[b] - mean[b];
                    ss[b] += d * d;
                }
            }
        }
        Some(w) => {
            for (i, &wi) in w.iter().enumerate() {
                for b in 0..B {
                    let u = cols[b][i] * inv[b];
                    sum[b] += wi * u;
                    sq[b] += wi * (u * u);
                }
            }
            mean = core::array::from_fn(|b| sum[b] / n_f);
            for (i, &wi) in w.iter().enumerate() {
                for b in 0..B {
                    let d = cols[b][i] * inv[b] - mean[b];
                    ss[b] += wi * (d * d);
                }
            }
        }
    }

    core::array::from_fn(|b| ScaledMoments {
        s: scales[b].0,
        inv: inv[b],
        mean: mean[b],
        ss: ss[b],
        sq: sq[b],
    })
}

/// The value an `f64` `Iterator::sum` starts from (`-0.0` on current
/// toolchains). [`scaled_moments_block`] starts its sums here so that
/// each equals the corresponding `Iterator::sum` bit for bit, down to the
/// sign of a sum whose every term is `-0.0`.
fn sum_start() -> f64 {
    core::iter::empty::<f64>().sum()
}

/// `(mean, Σ(vᵢ − mean)², Σvᵢ²)` of `v(0)..v(n−1)`, summed in index order:
/// the unweighted arm of [`scaled_moments`].
#[allow(clippy::cast_precision_loss)]
pub(crate) fn centered_moments(n: usize, v: impl Fn(usize) -> f64) -> (f64, f64, f64) {
    let mean: f64 = (0..n).map(&v).sum::<f64>() / n as f64;
    let ss: f64 = (0..n)
        .map(|i| {
            let d = v(i) - mean;
            d * d
        })
        .sum();
    let sq: f64 = (0..n).map(|i| v(i) * v(i)).sum();
    (mean, ss, sq)
}

/// `(s, 1/s)` with `s = 2^e` and `e` the binary exponent of `max_abs`
/// (so `max_abs / s` is in `[1, 2)`), clamped to `e ∈ [−1022, 1022]` so
/// that both `s` and `1/s` are normal numbers. The clamp only bites at the
/// ends of the range: a zero or subnormal `max_abs` gets `e = −1022`, and
/// one of binary exponent `1023` gets `e = 1022`, so a column divided by
/// `s` always has entries of magnitude below `4`.
pub(crate) fn pow2_scale(max_abs: f64) -> (f64, f64) {
    // Biased exponent field of an f64: bits 52..63, bias 1023.
    let biased = i64::try_from((max_abs.to_bits() >> 52) & 0x7ff).expect("11-bit field");
    let e = (biased - 1023).clamp(-1022, 1022);
    let pow2 = |exp: i64| f64::from_bits(u64::try_from(exp + 1023).expect("normal exponent") << 52);
    (pow2(e), pow2(-e))
}

/// `true` when a length-`n` vector whose centered sum of squares is `ss`
/// and uncentered sum of squares is `sq` is constant to rounding: `ss` is
/// no larger than what centering leaves of an exactly constant vector.
/// The computed mean of `n` equal values `c` is off by at most about
/// `n·ε·|c|`, so each centered entry of such a vector is at most about
/// `(n + 1)·ε·|c|` and `ss ≤ ((n + 1)·ε)²·sq`; the rule uses `(2n·ε)²`.
/// With mean-one weights `w'` (`Σ w' = n`), the weighted sums
/// `Σ w'(x − mean)²` and `Σ w'x²` obey the same bound.
///
/// It is relative to the vector's own magnitude, so it is invariant to
/// rescaling the vector (as long as the squares neither overflow nor
/// underflow): multiplying it by any positive constant cannot switch it.
/// The standardizers keep the squares in range at any magnitude by
/// forming the sums on the column divided by a power of two near its
/// largest entry, and the split-half correlations do the same to each
/// held-out vector: both go through [`scaled_moments`].
/// An absolute threshold would switch: with `sd ≤ 1e-12`, a column of
/// standard deviation `1e-13` would be centered but not rescaled (and
/// `pls1_fit` on it would truncate to `k_used = 0`), while a constant
/// column of magnitude `1e6`, whose centered entries are rounding noise of
/// order `1e-10` to `1e-9`, would be rescaled to unit variance. An
/// absolute guard `ss < 1e-15` on a split-half correlation has the same
/// flaw. An all-zero vector (`ss = sq = 0`) is constant.
///
/// The price of a bound that holds for every constant vector is that it
/// grows with `n`: a vector whose `sd/|mean|` is at most about `2n·ε` is
/// treated as constant even if it genuinely varies (at `n = 1e6`, below
/// about `4e-10`, e.g. `1e6 + 1e-4·noise`). That band is deliberate. A
/// constant column of `1e6 + 0.1` at `n = 1e6` already shows `sd/|mean|`
/// of about `1.7e-11` from the rounding of its mean alone (a constant whose
/// partial sums are all exact, such as `1e6` itself, shows `0`), so a
/// variation within a few tens of that is not separable from rounding by
/// any rule on these sums; center such data (e.g. subtract a baseline)
/// before fitting.
///
/// Used by [`standardize_weighted`] / [`standardize1_weighted`] to decide
/// which columns get scale `max(1, |mean|)`, and by the split-half statistics of PLS1 and
/// PLS3 to decide which held-out vectors give a correlation of `0`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn constant_to_rounding(ss: f64, sq: f64, n: usize) -> bool {
    let tol = 2.0 * n as f64 * f64::EPSILON;
    ss <= tol * tol * sq
}

/// Kish's effective sample size: `(Σw)² / Σw²`.
/// Caller is responsible for ensuring `Σw > 0`.
#[must_use]
pub fn compute_n_eff(w: ColRef<'_, f64>) -> f64 {
    let n = w.nrows();
    let s: f64 = (0..n).map(|i| w[i]).sum();
    let s2: f64 = (0..n).map(|i| w[i].powi(2)).sum();
    (s * s) / s2
}

/// Stable rank: `‖A‖²_F / ‖A‖²_2` — Frobenius norm squared over the largest
/// singular value squared. Equivalently `Σσᵢ² / σ₁²`. Does NOT standardize
/// `a`; the caller standardizes first (intended use: `stable_rank(standardized X)`,
/// where it equals the reciprocal of the first principal component's variance
/// share — VE1 in Aquino, Fulcher, Parkes, Sabaroedin & Fornito, 2020,
/// "Identifying and removing widespread signal deflections from fMRI data",
/// NeuroImage 212:116614, doi:10.1016/j.neuroimage.2020.116614).
/// Zero matrix (σ₁ = 0) returns 0.0.
///
/// The SVD runs on the crate's fixed-degree Rayon split
/// (`fit::par_fixed`), so the result does not depend on the size of the
/// Rayon pool it is called from.
///
/// # Panics
/// Panics if faer's SVD fails to converge (only expected on pathological input).
#[must_use]
#[allow(clippy::doc_markdown)] // "NeuroImage"
pub fn stable_rank(a: MatRef<'_, f64>) -> f64 {
    let s = singular_values(a, crate::fit::par_fixed()).expect("SVD failed to converge");
    let sigma1 = if s.nrows() == 0 { 0.0 } else { s[0] };
    if sigma1 <= 0.0 {
        return 0.0;
    }
    let frob_sq: f64 = s.iter().map(|v| v * v).sum();
    frob_sq / (sigma1 * sigma1)
}

/// Normalize weights so Σw' = n (mean 1). Returns `None` if `Σw == 0`.
/// Validation (negative, NaN) is the caller's responsibility.
#[must_use]
pub fn normalize_weights(w: ColRef<'_, f64>) -> Option<Col<f64>> {
    if weight_sum(w) == 0.0 {
        return None;
    }
    Some(renormalize_mean_one(w))
}

/// Split shuffled indices into `n_folds` (almost-equal) groups.
/// Equivalent to numpy's `np.array_split(arr, n_folds)`: first
/// `len % n_folds` groups have one extra element.
#[must_use]
pub fn fold_split(shuffled: &[usize], n_folds: usize) -> Vec<Vec<usize>> {
    let n_total = shuffled.len();
    let base = n_total / n_folds;
    let extra = n_total % n_folds;
    let mut out = Vec::with_capacity(n_folds);
    let mut cursor = 0;
    for i in 0..n_folds {
        let len = base + usize::from(i < extra);
        out.push(shuffled[cursor..cursor + len].to_vec());
        cursor += len;
    }
    out
}

/// Regularized incomplete beta I_x(a, b) via Lentz continued fraction.
#[allow(clippy::doc_markdown)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::shadow_unrelated)]
fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - betainc(b, a, 1.0 - x);
    }
    let lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta_ab).exp() / a;
    let tiny = 1e-30;
    let eps = 1e-14;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1_i32..201 {
        let m2 = f64::from(2 * m);
        let mf = f64::from(m);
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    front * h
}

/// Stable log-gamma via Lanczos coefficients (g = 7).
pub(crate) fn lgamma(x: f64) -> f64 {
    lanczos_lgamma(x)
}

#[allow(clippy::many_single_char_names)]
fn lanczos_lgamma(x: f64) -> f64 {
    // Lanczos approximation, g = 7 (the standard 9-coefficient set below); accurate to ~1e-13 over x > 0.
    let g = 7.0;
    let p: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: lgamma(x) = ln(pi / sin(pi x)) - lgamma(1 - x)
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln()
            - lanczos_lgamma(1.0 - x);
    }
    let x_minus_one = x - 1.0;
    let mut a = p[0];
    let t = x_minus_one + g + 0.5;
    for (i, pi) in p.iter().enumerate().skip(1) {
        a += pi / (x_minus_one + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x_minus_one + 0.5) * t.ln() - t + a.ln()
}

/// Per-row leverage `diag(W (W'W)^-1 W')`.
///
/// Computes one LU on `W'W` (cost dominated by `O(K^3)` decomposition + `O(N K^2)`
/// for the row sweep), instead of recomputing the full inverse matmul. Callers
/// must ensure `W'W` is non-singular (W is full column rank).
#[allow(clippy::many_single_char_names)]
pub(crate) fn leverage_diag(w: faer::MatRef<'_, f64>) -> Vec<f64> {
    let n = w.nrows();
    let k = w.ncols();
    let mut wtw = Mat::<f64>::zeros(k, k);
    matmul(
        wtw.as_mut(),
        Accum::Replace,
        w.transpose(),
        w,
        1.0,
        Par::Seq,
    );
    let mut m = Mat::<f64>::identity(k, k);
    lu_solve_in_place(wtw.as_ref(), m.as_mut());
    let mut tmp = vec![0.0_f64; k];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        for jj in 0..k {
            let mut s = 0.0;
            for kk in 0..k {
                s += m[(jj, kk)] * w[(i, kk)];
            }
            tmp[jj] = s;
        }
        out.push((0..k).map(|jj| w[(i, jj)] * tmp[jj]).sum());
    }
    out
}

/// Survival function P(T > t) for Student's t-distribution with `df`.
#[must_use]
pub fn t_sf(t_val: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    if t_val == 0.0 {
        return 0.5;
    }
    let x_val = df / (df + t_val * t_val);
    let half_p = 0.5 * betainc(df / 2.0, 0.5, x_val);
    if t_val > 0.0 {
        half_p
    } else {
        1.0 - half_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn mat(rows: usize, cols: usize, data: &[f64]) -> Mat<f64> {
        Mat::<f64>::from_fn(rows, cols, |i, j| data[i * cols + j])
    }

    #[test]
    fn standardize_centers_and_scales() {
        let x = mat(4, 2, &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
        let (xs, mean, scale) = standardize(x.as_ref());
        assert_relative_eq!(mean[0], 2.5, epsilon = 1e-12);
        assert_relative_eq!(mean[1], 25.0, epsilon = 1e-12);
        // var = mean of squared deviations (ddof=0): for [1,2,3,4], var=1.25
        assert_relative_eq!(scale[0], 1.25_f64.sqrt(), epsilon = 1e-12);
        // After standardize, column mean ≈ 0.
        for c in 0..2 {
            let col_mean: f64 =
                (0..xs.nrows()).map(|i| xs[(i, c)]).sum::<f64>() / xs.nrows() as f64;
            assert_relative_eq!(col_mean, 0.0, epsilon = 1e-12);
        }
    }

    /// Weights for the weighted variants: uneven, mean not one (the
    /// standardizers renormalize them).
    fn uneven_weights(n: usize) -> Col<f64> {
        Col::<f64>::from_fn(n, |i| 0.25 + (i % 7) as f64 * 0.5)
    }

    /// Every standardizer on one column: (scale, standardized column) from
    /// `standardize_weighted` and `standardize1_weighted`, each unweighted and
    /// weighted.
    #[allow(clippy::many_single_char_names)]
    fn all_standardizers(v: &[f64]) -> Vec<(&'static str, f64, Vec<f64>)> {
        let n = v.len();
        let x = Mat::<f64>::from_fn(n, 1, |i, _| v[i]);
        let y = Col::<f64>::from_fn(n, |i| v[i]);
        let w = uneven_weights(n);
        let mut out = Vec::new();
        for (label, wref) in [("unweighted", None), ("weighted", Some(w.as_ref()))] {
            let (xs, _, s) = standardize_weighted(x.as_ref(), wref);
            out.push((label, s[0], (0..n).map(|i| xs[(i, 0)]).collect()));
            let (z, _, s1) = standardize1_weighted(y.as_ref(), wref);
            out.push((label, s1, (0..n).map(|i| z[i]).collect()));
        }
        out
    }

    /// Constant columns get scale `max(1, |mean|)` and standardize to zeros
    /// up to rounding relative to one, whatever their magnitude: an all-zero
    /// column, a constant whose computed mean is off by rounding (`0.1`), a
    /// large constant whose centered entries are rounding noise of standard
    /// deviation about `1e-9` (which an absolute `sd ≤ 1e-12` floor would
    /// rescale to unit variance), and a column varying by a few ulps.
    /// With scale 1 at every magnitude, the centering noise of a constant
    /// near `1e300` would stay near `1e285` (squares overflowing to NaN fits,
    /// or a `‖Xs‖_F` that truncates `pls1_fit` to `k_used = 0`).
    #[test]
    fn constant_columns_to_rounding_standardize_to_zeros_at_any_magnitude() {
        let n = 50;
        let big = 1e6 + 0.1;
        // The large constant's rounding noise is far above an absolute 1e-12 floor.
        let mean = (0..n).map(|_| big).sum::<f64>() / n as f64;
        let noise_sd = ((0..n).map(|_| (big - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(noise_sd > 1e-10, "noise sd = {noise_sd:e}");
        let few_ulps = |c: f64| -> Vec<f64> {
            (0..n)
                .map(|i| c * (1.0 + (i % 4) as f64 * f64::EPSILON))
                .collect()
        };
        // The magnitudes past 1e154 and below 1e-162 are where the squares
        // of the raw column overflow or underflow; the standardizers
        // pre-scale, so the rule reaches them unchanged.
        let cases: [(&str, Vec<f64>); 9] = [
            ("zero", vec![0.0; n]),
            ("0.1", vec![0.1; n]),
            ("1e6 + 0.1", vec![big; n]),
            ("few ulps", few_ulps(3.7)),
            ("1.1e200", vec![1.1e200; n]),
            ("-1.1e300", vec![-1.1e300; n]),
            ("1.1e-200", vec![1.1e-200; n]),
            ("few ulps at 3.7e200", few_ulps(3.7e200)),
            ("few ulps at 3.7e-200", few_ulps(3.7e-200)),
        ];
        // `|z| ≤ 2nε·√(n/w'_min)` for a column constant to rounding; the
        // uneven weights have `w'_min ≈ 0.14`.
        let bound = 1e-12;
        for (name, v) in &cases {
            let magnitude = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            for (label, scale, z) in all_standardizers(v) {
                assert!(
                    scale >= 1.0 && scale <= magnitude.max(1.0) * (1.0 + 1e-12),
                    "{name} {label}: scale = {scale:e}"
                );
                for zi in z {
                    assert!(zi.abs() <= bound, "{name} {label}: z = {zi:e}");
                }
            }
        }
        // Scale 1 wherever the magnitude is at most 1.
        for v in [vec![0.0; n], vec![0.1; n], vec![-1.0; n]] {
            for (label, scale, _) in all_standardizers(&v) {
                assert_eq!(scale.to_bits(), 1.0_f64.to_bits(), "{:e} {label}", v[0]);
            }
        }
    }

    /// A column whose spread is below the normal range (so that `sd` would
    /// be `0` or subnormal) though its entries are not all equal is
    /// treated as constant: a normal scale, finite standardized entries.
    /// Without this rule it would get scale `0` (a single `5e-324` among
    /// zeros) or a subnormal scale (`1e-310` on half the rows), and
    /// `pls1_fit` would return NaN coefficients without an error.
    #[test]
    fn columns_whose_spread_underflows_are_only_centered() {
        let n = 40;
        let cases: [(&str, Vec<f64>); 3] = [
            (
                "one 5e-324",
                (0..n).map(|i| if i == 0 { 5e-324 } else { 0.0 }).collect(),
            ),
            (
                "1e-310 on half",
                (0..n)
                    .map(|i| if i % 2 == 0 { 1e-310 } else { 0.0 })
                    .collect(),
            ),
            (
                "-3e-308 on a third",
                (0..n)
                    .map(|i| if i % 3 == 0 { -3e-308 } else { 0.0 })
                    .collect(),
            ),
        ];
        for (name, v) in &cases {
            for (label, scale, z) in all_standardizers(v) {
                assert_eq!(scale.to_bits(), 1.0_f64.to_bits(), "{name} {label}");
                assert!(
                    z.iter().all(|zi| zi.abs() < 1e-300),
                    "{name} {label}: {z:?}"
                );
            }
        }
        // Just inside the normal range the column is standardized as usual.
        let v: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1e-300 } else { 0.0 })
            .collect();
        for (label, scale, _) in all_standardizers(&v) {
            assert!(scale < 1e-299 && scale.is_normal(), "{label}: {scale:e}");
        }
    }

    /// A genuinely varying column is rescaled however small its spread, in
    /// absolute terms or relative to its mean: its standardized values match
    /// those of the column at unit scale.
    #[test]
    fn small_but_varying_columns_are_rescaled() {
        let base: Vec<f64> = (0..40).map(|i| f64::from(i * 7 % 11).sin()).collect();
        let reference = all_standardizers(&base);
        // 1e±200 and 1e±300 put the squares of the raw column past the f64
        // range (`inf` above about 1e154, `0` below about 1e-162).
        for factor in [1e-13, 1e-100, 1e100, 1e-200, 1e200, 1e-300, 1e300] {
            let v: Vec<f64> = base.iter().map(|b| b * factor).collect();
            for ((label, scale, z), (_, ref_scale, ref_z)) in
                all_standardizers(&v).into_iter().zip(&reference)
            {
                assert_relative_eq!(scale / factor, *ref_scale, max_relative = 1e-12);
                for (a, b) in z.iter().zip(ref_z) {
                    assert!((a - b).abs() < 1e-12, "×{factor:e} {label}: {a} vs {b}");
                }
            }
        }
        // Spread 1e-4 on an offset of 1e6: tiny relative to the mean, but
        // about 1e6 ulps, so not rounding noise.
        let offset: Vec<f64> = base.iter().map(|b| 1e6 + 1e-4 * b).collect();
        for (label, scale, _) in all_standardizers(&offset) {
            assert!((scale - 1.0).abs() > 0.5, "{label}: scale = {scale:e}");
        }
    }

    /// A column far outside the range where its squares are finite and
    /// normal, with a small spread relative to its magnitude, is still
    /// rescaled, and standardizes to the same values as the same column
    /// near 1. Summed on the raw values, `Σx²` is `inf` at `1e200` (raw sums
    /// would classify the column as constant) and `0` at `1e-200`
    /// (likewise).
    #[test]
    fn huge_and_tiny_columns_with_small_relative_spread_are_rescaled() {
        let base: Vec<f64> = (0..40).map(|i| f64::from(i * 7 % 11).sin()).collect();
        let unit: Vec<f64> = base.iter().map(|b| 1.0 + 1e-4 * b).collect();
        let reference = all_standardizers(&unit);
        for magnitude in [1e200, 1e-200, 1e300, 1e-300, -1e200] {
            let v: Vec<f64> = base.iter().map(|b| magnitude * (1.0 + 1e-4 * b)).collect();
            for ((label, scale, z), (_, ref_scale, ref_z)) in
                all_standardizers(&v).into_iter().zip(&reference)
            {
                assert!(
                    scale.is_finite(),
                    "{magnitude:e} {label}: scale = {scale:e}"
                );
                assert_relative_eq!(scale / magnitude.abs(), *ref_scale, max_relative = 1e-9);
                // Spread 1e-4 turns the ~1e-16 rounding of the inputs into
                // ~1e-12 in the standardized values.
                let sign = magnitude.signum();
                for (a, b) in z.iter().zip(ref_z) {
                    assert!(
                        (a - sign * b).abs() < 1e-9,
                        "{magnitude:e} {label}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// Raw-column standardization formulas (no power-of-two pre-scaling):
    /// the bit-level reference for the claim that pre-scaling changes
    /// nothing where those formulas stay in range.
    fn raw_mean_and_scale(v: &[f64], w: Option<&[f64]>) -> (f64, f64) {
        let n = v.len();
        let n_f = n as f64;
        let w_prime: Option<Vec<f64>> = w.map(|w| {
            let s: f64 = w.iter().sum();
            w.iter().map(|wi| wi * n_f / s).collect()
        });
        let mean: f64 = match &w_prime {
            None => v.iter().sum::<f64>() / n_f,
            Some(w) => (0..n).map(|i| w[i] * v[i]).sum::<f64>() / n_f,
        };
        let (ss, sq): (f64, f64) = match &w_prime {
            None => (
                v.iter().map(|x| (x - mean).powi(2)).sum(),
                v.iter().map(|x| x.powi(2)).sum(),
            ),
            Some(w) => (
                (0..n).map(|i| w[i] * (v[i] - mean).powi(2)).sum(),
                (0..n).map(|i| w[i] * v[i].powi(2)).sum(),
            ),
        };
        let scale = if constant_to_rounding(ss, sq, n) {
            mean.abs().max(1.0)
        } else {
            (ss / n_f).sqrt()
        };
        (mean, scale)
    }

    /// Wherever the raw-column sums stay finite and normal, the pre-scaled
    /// standardizers reproduce the raw formulas bit for bit: mean, scale,
    /// every standardized entry, and the constant/non-constant decision.
    /// So the frozen corpus and `tests/byte_parity.rs` are unaffected by
    /// the pre-scaling.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn prescaling_is_bit_identical_in_the_normal_range() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let n = 37;
        let w: Vec<f64> = (0..n).map(|i| 0.25 + (i % 7) as f64 * 0.5).collect();
        let mut columns: Vec<Vec<f64>> = Vec::new();
        for magnitude in [1e-120, 1e-30, 1e-3, 0.7, 1.0, 3.0, 1e3, 1e30, 1e120] {
            for offset in [0.0, 1.0, 1e6] {
                columns.push(
                    (0..n)
                        .map(|_| magnitude * (offset + rng.random_range(-1.0..1.0)))
                        .collect(),
                );
            }
            columns.push(vec![magnitude; n]);
            columns.push(vec![magnitude * (1.0 + 0.1); n]);
        }
        columns.push(vec![0.0; n]);
        for v in &columns {
            let x = Mat::<f64>::from_fn(n, 1, |i, _| v[i]);
            let y = Col::<f64>::from_fn(n, |i| v[i]);
            let wc = Col::<f64>::from_fn(n, |i| w[i]);
            for (wref, wraw) in [(None, None), (Some(wc.as_ref()), Some(w.as_slice()))] {
                let (ref_mean, ref_scale) = raw_mean_and_scale(v, wraw);
                let (xs, mean, scale) = standardize_weighted(x.as_ref(), wref);
                let (z, mean1, scale1) = standardize1_weighted(y.as_ref(), wref);
                for (m, s) in [(mean[0], scale[0]), (mean1, scale1)] {
                    assert_eq!(m.to_bits(), ref_mean.to_bits(), "mean of {:e}", v[0]);
                    assert_eq!(s.to_bits(), ref_scale.to_bits(), "scale of {:e}", v[0]);
                }
                for i in 0..n {
                    let r = (v[i] - ref_mean) / ref_scale;
                    assert_eq!(xs[(i, 0)].to_bits(), r.to_bits());
                    assert_eq!(z[i].to_bits(), r.to_bits());
                }
            }
        }
    }

    #[test]
    fn pow2_scale_brackets_the_largest_entry() {
        for m in [1.0, 1.5, 3.0, 1e-200, 1e200, 1e300, f64::MIN_POSITIVE] {
            let (s, inv) = pow2_scale(m);
            assert_eq!((s * inv).to_bits(), 1.0_f64.to_bits());
            assert!((1.0..2.0).contains(&(m * inv)), "{m:e}");
        }
        // The clamped ends: zero and subnormal get 2^-1022, binary exponent
        // 1023 gets 2^1022. Both factors stay normal, entries stay below 4.
        for (m, s_expected) in [
            (0.0, f64::MIN_POSITIVE),
            (1e-310, f64::MIN_POSITIVE),
            (f64::MAX, 2.0_f64.powi(1022)),
        ] {
            let (s, inv) = pow2_scale(m);
            assert_eq!(s.to_bits(), s_expected.to_bits(), "{m:e}");
            assert!(s.is_normal() && inv.is_normal(), "{m:e}");
            assert!(m * inv < 4.0, "{m:e}");
        }
    }

    #[test]
    fn constant_to_rounding_is_scale_free_and_counts_zero_as_constant() {
        assert!(constant_to_rounding(0.0, 0.0, 10));
        assert!(constant_to_rounding(0.0, 5.0, 10));
        // Just under and just over (2nε)²·sq.
        let n = 10;
        let tol = 2.0 * n as f64 * f64::EPSILON;
        for sq in [1e-200, 1.0, 1e200] {
            assert!(constant_to_rounding(0.5 * tol * tol * sq, sq, n));
            assert!(!constant_to_rounding(2.0 * tol * tol * sq, sq, n));
        }
    }

    #[test]
    fn fold_split_matches_numpy_array_split() {
        // numpy.array_split(arange(10), 3): the first n % k folds get one extra.
        let idx: Vec<usize> = (0..10).collect();
        assert_eq!(
            fold_split(&idx, 3),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        );
        // numpy.array_split(arange(2), 3): more folds than indices leaves the
        // last fold empty.
        assert_eq!(fold_split(&[0, 1], 3), vec![vec![0], vec![1], vec![]]);
    }

    #[test]
    fn t_sf_matches_scipy() {
        // scipy.stats.t.sf(t, df), scipy 1.18.1. (0.5, 3) takes betainc's
        // reflection branch (x = df/(df+t²) > (a+1)/(a+b+2)); (6, 5) is a
        // far tail; -2 is the lower half.
        for (t, df, want) in [
            (2.228, 10.0, 0.025_005_885_908_555_66),
            (2.0, 10.0, 0.036_694_017_385_370_18),
            (-2.0, 10.0, 0.963_305_982_614_629_9),
            (6.0, 5.0, 0.000_923_069_144_797_006_8),
            (0.5, 3.0, 0.325_723_982_424_075_5),
        ] {
            assert_relative_eq!(t_sf(t, df), want, max_relative = 1e-10);
        }
    }

    /// `orthogonal_rotation` is `procrustes::orthogonal(a, reference,
    /// false).rotation` to the bit (its contract: the same products and SVD,
    /// minus the global-parallelism read), on the shapes `rotation_stability`
    /// forms (`a` = `W_b`, `reference` = `W_ref`, both d × k), a near-degenerate
    /// pair (reference = `a` with two columns swapped, one sign flipped and
    /// 1e-12 noise) and a rank-deficient `a'·reference` (two equal
    /// reference columns).
    #[test]
    #[allow(clippy::disallowed_methods)] // procrustes::orthogonal is the oracle
    fn orthogonal_rotation_matches_procrustes_orthogonal() {
        use crate::test_support::{assert_bits_eq, mat_vals};
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        let mut uniform =
            |d: usize, k: usize| Mat::<f64>::from_fn(d, k, |_, _| rng.random_range(-1.0..1.0));
        let mut cases: Vec<(String, Mat<f64>, Mat<f64>)> = [(7, 2), (7, 3), (400, 5)]
            .into_iter()
            .map(|(d, k)| (format!("uniform {d}x{k}"), uniform(d, k), uniform(d, k)))
            .collect();
        let a = uniform(7, 3);
        let noise = uniform(7, 3);
        let near = Mat::<f64>::from_fn(7, 3, |i, j| {
            let v = match j {
                0 => a[(i, 1)],
                1 => -a[(i, 0)],
                _ => a[(i, 2)],
            };
            v + 1e-12 * noise[(i, j)]
        });
        cases.push(("near-degenerate".into(), a.clone(), near));
        let r = uniform(7, 3);
        let rank_def = Mat::<f64>::from_fn(7, 3, |i, j| r[(i, j.min(1))]);
        cases.push(("rank-deficient".into(), a, rank_def));
        for (what, a, reference) in &cases {
            let got = orthogonal_rotation(a.as_ref(), reference.as_ref());
            let want = procrustes::orthogonal(a.as_ref(), reference.as_ref(), false)
                .unwrap()
                .rotation;
            assert_eq!(
                (got.nrows(), got.ncols()),
                (want.nrows(), want.ncols()),
                "{what}"
            );
            assert_bits_eq(&mat_vals(got.as_ref()), &mat_vals(want.as_ref()), what);
        }
    }

    #[test]
    fn stable_rank_rank1_is_one() {
        // Outer product u v^T: single nonzero singular value.
        let x = mat(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        assert_relative_eq!(stable_rank(x.as_ref()), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn stable_rank_mixed_spectrum_matches_formula() {
        // diag(2, 1, 1): singular values (2, 1, 1) → (4+1+1)/4 = 1.5
        let x = mat(3, 3, &[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_relative_eq!(stable_rank(x.as_ref()), 1.5, epsilon = 1e-10);
    }

    #[test]
    fn stable_rank_zero_matrix_is_zero() {
        let x = Mat::<f64>::zeros(4, 3);
        assert_relative_eq!(stable_rank(x.as_ref()), 0.0, epsilon = 1e-15);
    }
}

#[cfg(test)]
mod weighted_tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::{Col, Mat};

    fn small_x() -> Mat<f64> {
        Mat::from_fn(4, 2, |i, j| (i as f64) - (j as f64) * 0.5)
    }

    #[test]
    fn standardize_uniform_weights_matches_unweighted() {
        let x = small_x();
        let w = Col::<f64>::from_fn(x.nrows(), |_| 3.7); // any positive constant
        let (xs_a, m_a, s_a) = standardize(x.as_ref());
        let (xs_b, m_b, s_b) = standardize_weighted(x.as_ref(), Some(w.as_ref()));
        for j in 0..x.ncols() {
            assert_relative_eq!(m_a[j], m_b[j], epsilon = 1e-12);
            assert_relative_eq!(s_a[j], s_b[j], epsilon = 1e-12);
            for i in 0..x.nrows() {
                assert_relative_eq!(xs_a[(i, j)], xs_b[(i, j)], epsilon = 1e-12);
            }
        }
        for j in 0..x.ncols() {
            let (za, ma, sa) = standardize1(x.col(j));
            let (zb, mb, sb) = standardize1_weighted(x.col(j), Some(w.as_ref()));
            assert_relative_eq!(ma, mb, epsilon = 1e-12);
            assert_relative_eq!(sa, sb, epsilon = 1e-12);
            for i in 0..x.nrows() {
                assert_relative_eq!(za[i], zb[i], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn n_eff_kish() {
        let w = Col::<f64>::from_fn(10, |_| 1.0);
        assert_relative_eq!(compute_n_eff(w.as_ref()), 10.0, epsilon = 1e-12);
        // Single positive row, n-1 zero rows: n_eff = 1
        let w2 = Col::<f64>::from_fn(10, |i| if i == 0 { 1.0 } else { 0.0 });
        assert_relative_eq!(compute_n_eff(w2.as_ref()), 1.0, epsilon = 1e-12);
        // Hand-computed: w = [1, 2, 3]. Σw=6, Σw²=14. n_eff = 36/14 ≈ 2.571
        let w3 = Col::<f64>::from_fn(3, |i| (i + 1) as f64);
        assert_relative_eq!(compute_n_eff(w3.as_ref()), 36.0 / 14.0, epsilon = 1e-12);
    }

    #[test]
    fn normalize_weights_to_mean_one() {
        let w = Col::<f64>::from_fn(4, |_| 5.0);
        let wn = normalize_weights(w.as_ref()).unwrap();
        for i in 0..4 {
            assert_relative_eq!(wn[i], 1.0, epsilon = 1e-12);
        }
        // Total stays at n.
        let s: f64 = (0..4).map(|i| wn[i]).sum();
        assert_relative_eq!(s, 4.0, epsilon = 1e-12);
        assert!(normalize_weights(Col::<f64>::zeros(4).as_ref()).is_none());
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod leverage_diag_tests {
    use super::*;
    use faer::Mat;

    /// Textbook two-matmul reference for `diag(W (W'W)^-1 W')`.
    #[allow(clippy::many_single_char_names)]
    fn leverage_diag_naive(w: faer::MatRef<'_, f64>) -> Vec<f64> {
        use faer::linalg::matmul::matmul;
        use faer::linalg::solvers::{PartialPivLu, Solve};
        use faer::Accum;
        let n = w.nrows();
        let k = w.ncols();
        let mut wtw = Mat::<f64>::zeros(k, k);
        matmul(
            wtw.as_mut(),
            Accum::Replace,
            w.transpose(),
            w,
            1.0,
            faer::Par::Seq,
        );
        let lu = PartialPivLu::new(wtw.as_ref());
        let mut m = Mat::<f64>::identity(k, k);
        lu.solve_in_place(m.as_mut());
        (0..n)
            .map(|i| {
                let mut h = 0.0;
                for jj in 0..k {
                    for kk in 0..k {
                        h += w[(i, jj)] * m[(jj, kk)] * w[(i, kk)];
                    }
                }
                h
            })
            .collect()
    }

    #[test]
    fn leverage_diag_matches_naive_reference() {
        use rand::{rngs::StdRng, RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let (n, k) = (20, 4);
        let w = Mat::<f64>::from_fn(n, k, |_, _| rng.random::<f64>() - 0.5);
        let h_fast = leverage_diag(w.as_ref());
        let h_naive = leverage_diag_naive(w.as_ref());
        for i in 0..n {
            assert!(
                (h_fast[i] - h_naive[i]).abs() < 1e-10,
                "i={i}: {} vs {}",
                h_fast[i],
                h_naive[i]
            );
        }
    }
}

/// R type-7 / numpy-default linear-interpolation quantile
/// (Hyndman & Fan 1996, "Sample Quantiles in Statistical Packages",
/// The American Statistician 50(4), definition 7).
///
/// Serves the Politis–Romano centered-scaled subsampling CIs in
/// `subsample` and the percentile bootstrap CIs in `rotation_stability`.
///
/// Caller is responsible for sorting `sorted` ascending before calling.
/// Empty slice ⇒ `f64::NAN`. `q` is clamped to `[0, 1]`.
pub(crate) fn empirical_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    // R type 7 / numpy: h = (n − 1) · q; floor index ⌊h⌋, fractional part h − ⌊h⌋.
    let h = (n - 1) as f64 * q;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod empirical_quantile_tests {
    use super::empirical_quantile;

    /// numpy.quantile(v, q) (method "linear", R type 7); 0.1 and 0.975
    /// interpolate between order statistics.
    #[test]
    #[allow(clippy::float_cmp)] // n == 1 returns the element itself
    fn matches_numpy_linear_quantiles() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (q, want) in [
            (0.0, 1.0),
            (0.1, 1.4),
            (0.25, 2.0),
            (0.5, 3.0),
            (0.975, 4.9),
            (1.0, 5.0),
        ] {
            let got = empirical_quantile(&v, q);
            assert!((got - want).abs() < 1e-12, "q={q}: {got} vs {want}");
        }
        assert_eq!(empirical_quantile(&[3.0], 0.3), 3.0);
        assert!(empirical_quantile(&[], 0.5).is_nan());
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod row_standardize_tests {
    use super::*;
    use crate::test_support::{assert_bits_eq, col_vals, mat_vals, test_weights, Layouts};

    /// 24 × 5 with a constant column (2), a column constant on the even
    /// rows only (3) and a huge-offset column (4).
    fn data() -> Mat<f64> {
        let (x, _) = crate::test_support::signal_data(24, 5, 91);
        Mat::<f64>::from_fn(24, 5, |i, j| match j {
            2 => 3.0,
            3 => {
                if i % 2 == 0 {
                    -7.5
                } else {
                    x[(i, 0)]
                }
            }
            4 => 1e6 + x[(i, 1)],
            _ => x[(i, j)],
        })
    }

    fn index_sets() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("all", (0..24).collect()),
            ("reversed", (0..24).rev().collect()),
            ("even rows", (0..24).step_by(2).collect()),
            (
                "with repeats",
                vec![3, 3, 0, 17, 17, 17, 5, 23, 0, 11, 11, 2],
            ),
            ("single row", vec![9]),
        ]
    }

    /// A `weights` column longer than the rows being standardized: only
    /// its first `n` entries count, renormalized over `n` rows
    /// (`renormalize_mean_one_n`). Reachable
    /// from a Rust caller that passes a longer `weights` to the public
    /// `standardize_weighted`; also through the row gather of
    /// `standardize_rows`.
    #[test]
    fn weights_longer_than_rows_use_their_first_n_entries() {
        let x = data();
        let w = test_weights(30);
        let w24 = w.subrows(0, 24).to_owned();
        let (a, am, as_) = standardize_weighted(x.as_ref(), Some(w.as_ref()));
        let (b, bm, bs) = standardize_weighted(x.as_ref(), Some(w24.as_ref()));
        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()), "xs shape");
        assert_bits_eq(&mat_vals(a.as_ref()), &mat_vals(b.as_ref()), "xs");
        assert_bits_eq(&col_vals(am.as_ref()), &col_vals(bm.as_ref()), "mean");
        assert_bits_eq(&col_vals(as_.as_ref()), &col_vals(bs.as_ref()), "scale");
        for j in 0..x.ncols() {
            let col = x.col(j);
            let (z, m, s) = standardize1_weighted(col, Some(w.as_ref()));
            let (zr, mr, sr) = standardize1_weighted(col, Some(w24.as_ref()));
            assert_bits_eq(&col_vals(z.as_ref()), &col_vals(zr.as_ref()), "z");
            assert_bits_eq(&[m, s], &[mr, sr], "moments");
        }
        for (label, idx) in index_sets() {
            if !matches!(label, "all" | "reversed" | "with repeats") {
                continue;
            }
            let m = idx.len();
            let w_long = test_weights(m + 6);
            let w_m = w_long.subrows(0, m).to_owned();
            let rs = Col::<f64>::from_fn(m, |i| 0.5 + (i % 3) as f64);
            for rsref in [None, Some(rs.as_ref())] {
                let (a, am, as_) = standardize_rows(x.as_ref(), &idx, Some(w_long.as_ref()), rsref);
                let (b, bm, bs) = standardize_rows(x.as_ref(), &idx, Some(w_m.as_ref()), rsref);
                let what = format!("{label} weights-longer-than-rows rs={}", rsref.is_some());
                assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()), "xs shape");
                assert_bits_eq(&mat_vals(a.as_ref()), &mat_vals(b.as_ref()), &what);
                assert_bits_eq(&col_vals(am.as_ref()), &col_vals(bm.as_ref()), &what);
                assert_bits_eq(&col_vals(as_.as_ref()), &col_vals(bs.as_ref()), &what);
            }
        }
    }

    #[test]
    fn standardize_rows_is_standardize_weighted_of_the_row_subset() {
        let x = data();
        let lay = Layouts::new(x.as_ref());
        for (view, xv) in lay.all(&x) {
            for (label, idx) in index_sets() {
                let m = idx.len();
                let w = test_weights(m + 1).subrows(1, m).to_owned();
                let rs = Col::<f64>::from_fn(m, |i| 0.5 + (i % 3) as f64);
                for wref in [None, Some(w.as_ref())] {
                    for rsref in [None, Some(rs.as_ref())] {
                        let (a, am, as_) = standardize_rows(xv, &idx, wref, rsref);
                        let (b0, bm, bs) =
                            standardize_weighted(row_subset(xv, &idx).as_ref(), wref);
                        let b = match rsref {
                            None => b0,
                            Some(r) => Mat::<f64>::from_fn(m, x.ncols(), |i, j| b0[(i, j)] * r[i]),
                        };
                        let what =
                            format!("{view} {label} w={} rs={}", wref.is_some(), rsref.is_some());
                        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()), "xs shape");
                        assert_bits_eq(&mat_vals(a.as_ref()), &mat_vals(b.as_ref()), &what);
                        assert_bits_eq(&col_vals(am.as_ref()), &col_vals(bm.as_ref()), &what);
                        assert_bits_eq(&col_vals(as_.as_ref()), &col_vals(bs.as_ref()), &what);
                        let scale_c = as_[2].to_bits();
                        assert_eq!(scale_c, 3.0_f64.to_bits(), "{what}: constant column");
                    }
                }
            }
        }
        // Column 3 is constant on the even rows only: the constant rule
        // (scale `max(1, |mean|)`) is decided on the gathered rows.
        let even: Vec<usize> = (0..24).step_by(2).collect();
        let (_, _, s) = standardize_rows(x.as_ref(), &even, None, None);
        assert_eq!(s[3].to_bits(), 7.5_f64.to_bits());
        // On every row it varies, and its scale is its population sd.
        let (_, _, s_all) = standardize_rows(x.as_ref(), &(0..24).collect::<Vec<_>>(), None, None);
        let c3: Vec<f64> = (0..24).map(|i| x[(i, 3)]).collect();
        let m3 = c3.iter().sum::<f64>() / 24.0;
        let sd3 = (c3.iter().map(|v| (v - m3).powi(2)).sum::<f64>() / 24.0).sqrt();
        assert!(sd3 > 1.0, "premise: sd = {sd3}");
        approx::assert_relative_eq!(s_all[3], sd3, max_relative = 1e-14);
    }

    #[test]
    fn standardize_apply_rows_is_standardize_apply_of_the_row_subset() {
        let x = data();
        let (_, mean, scale) = standardize(x.as_ref());
        let lay = Layouts::new(x.as_ref());
        for (view, xv) in lay.all(&x) {
            for (label, idx) in index_sets() {
                let rs = Col::<f64>::from_fn(idx.len(), |i| 0.25 + (i % 4) as f64);
                for rsref in [None, Some(rs.as_ref())] {
                    let a = standardize_apply_rows(xv, &idx, mean.as_ref(), scale.as_ref(), rsref);
                    let b0 = standardize_apply(
                        row_subset(xv, &idx).as_ref(),
                        mean.as_ref(),
                        scale.as_ref(),
                    );
                    let b = match rsref {
                        None => b0,
                        Some(r) => {
                            Mat::<f64>::from_fn(idx.len(), x.ncols(), |i, j| b0[(i, j)] * r[i])
                        }
                    };
                    assert_bits_eq(
                        &mat_vals(a.as_ref()),
                        &mat_vals(b.as_ref()),
                        &format!("{view} {label}"),
                    );
                }
            }
        }
    }

    #[test]
    fn col_major_or_copy_borrows_only_column_major_views() {
        let x = data();
        let lay = Layouts::new(x.as_ref());
        assert!(col_major_or_copy(x.as_ref()).is_none());
        assert!(col_major_or_copy(lay.submatrix()).is_none());
        let copy = col_major_or_copy(lay.row_major()).expect("row-major view is copied");
        assert!(copy.as_ref().try_as_col_major().is_some());
        assert_bits_eq(
            &mat_vals(copy.as_ref()),
            &mat_vals(x.as_ref()),
            "copy values",
        );
        // Column-major with a negative column stride passes
        // `try_as_col_major` but is not safe to borrow (faer's matmul
        // reverses the k order for such a left operand); it must still be
        // copied.
        let rev_copy =
            col_major_or_copy(lay.reversed_cols()).expect("negative-column-stride view is copied");
        assert!(rev_copy.as_ref().try_as_col_major().is_some());
        assert!(rev_copy.as_ref().col_stride() >= 0);
        assert_bits_eq(
            &mat_vals(rev_copy.as_ref()),
            &mat_vals(x.as_ref()),
            "reversed_cols copy values",
        );
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod fast_standardize_tests {
    use super::*;
    use crate::test_support::Layouts;
    use rand::{RngExt, SeedableRng};

    /// [`fit_x_moments`] then [`FitX::materialize`]: `standardize_weighted(x,
    /// weights)` with output row `i` multiplied by `row_scale[i]`, in `x`'s layout.
    fn standardize_fit_x(
        x: MatRef<'_, f64>,
        weights: Option<ColRef<'_, f64>>,
        row_scale: Option<ColRef<'_, f64>>,
    ) -> FitX {
        let mut f = fit_x_moments(x, weights);
        f.materialize(x, row_scale);
        f
    }

    /// Per-column scalar path (`mean_and_scale`), the reference for the
    /// block driver.
    fn standardize_weighted_reference(
        x: MatRef<'_, f64>,
        weights: Option<ColRef<'_, f64>>,
    ) -> (Mat<f64>, Col<f64>, Col<f64>) {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        let n_f = n_rows as f64;

        // w': normalize so Σ w'_i = n (mean 1). For weights=None, treat w'_i = 1.
        let w_prime: Option<Col<f64>> = weights.map(|w| {
            let s: f64 = (0..n_rows).map(|i| w[i]).sum();
            Col::<f64>::from_fn(n_rows, |i| w[i] * n_f / s)
        });
        let wpref = w_prime.as_ref().map(Col::as_ref);

        let moments: Vec<(f64, f64)> = (0..n_cols)
            .map(|j| mean_and_scale(n_rows, |i| x[(i, j)], wpref))
            .collect();
        let mean = Col::<f64>::from_fn(n_cols, |j| moments[j].0);
        let scale = Col::<f64>::from_fn(n_cols, |j| moments[j].1);
        let xs = Mat::<f64>::from_fn(n_rows, n_cols, |i, j| (x[(i, j)] - mean[j]) / scale[j]);
        (xs, mean, scale)
    }

    /// The elementwise definition `(x − mean) / scale`, the reference for
    /// the column-slice path (`write_standardized`).
    fn standardize_apply_reference(
        x: MatRef<'_, f64>,
        mean: ColRef<'_, f64>,
        scale: ColRef<'_, f64>,
    ) -> Mat<f64> {
        Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| {
            (x[(i, j)] - mean[j]) / scale[j]
        })
    }

    /// `(nrows, ncols, bits in column order)`: comparing these compares
    /// the shape too, which the bits alone do not (an `n × 0` and a
    /// `0 × 0` matrix both flatten to nothing).
    fn mat_bits(m: MatRef<'_, f64>) -> (usize, usize, Vec<u64>) {
        let mut v = Vec::with_capacity(m.nrows() * m.ncols());
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                v.push(bits(m[(i, j)]));
            }
        }
        (m.nrows(), m.ncols(), v)
    }

    fn col_bits(c: ColRef<'_, f64>) -> Vec<u64> {
        (0..c.nrows()).map(|i| bits(c[i])).collect()
    }

    /// `x.to_bits()`, with every NaN read as `f64::NAN`. Rust leaves a
    /// NaN's sign and payload unspecified: the zero-row mean `-0.0 / 0.0`
    /// is `0xFFF8…` computed at run time on x86 but `0x7FF8…` when LLVM
    /// folds it at compile time (and on ARM), so two paths that agree on
    /// every other bit can disagree on a NaN's.
    fn bits(x: f64) -> u64 {
        if x.is_nan() {
            f64::NAN.to_bits()
        } else {
            x.to_bits()
        }
    }

    /// `n × p` test matrix whose column `j` is of kind `j % 10`: noise,
    /// noise on an offset, noise on `1e6`, the constant `0.1`, the constant
    /// `1e6 + 0.1`, all `+0.0`, all `-0.0`, noise at `1e200`, noise at
    /// `1e-200`, and a column constant on the even rows only.
    fn test_matrix(n: usize, p: usize, seed: u64) -> Mat<f64> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Mat::<f64>::from_fn(n, p, |i, j| {
            let r: f64 = rng.random_range(-1.0..1.0);
            match j % 10 {
                0 => r,
                1 => 3.0 + r,
                2 => 1e6 + r,
                3 => 0.1,
                4 => 1e6 + 0.1,
                5 => 0.0,
                6 => -0.0,
                7 => 1e200 * r,
                8 => 1e-200 * r,
                _ => {
                    if i % 2 == 0 {
                        -7.5
                    } else {
                        r
                    }
                }
            }
        })
    }

    /// Raw weights (the standardizers renormalize them): uneven with a
    /// zero every third row, all equal, and one nonzero row.
    fn test_weight_sets(n: usize) -> Vec<(&'static str, Col<f64>)> {
        vec![
            (
                "uneven with zeros",
                Col::<f64>::from_fn(n, |i| {
                    if i % 3 == 2 {
                        0.0
                    } else {
                        0.25 + (i % 7) as f64 * 0.5
                    }
                }),
            ),
            ("all equal", Col::<f64>::from_fn(n, |_| 2.5)),
            (
                "one nonzero",
                Col::<f64>::from_fn(n, |i| if i == 0 { 1.0 } else { 0.0 }),
            ),
        ]
    }

    /// Shapes around the block width: empty, one row, fewer columns than a
    /// block, exactly one block, blocks plus a tail, and tall cases with
    /// many blocks, and a wide one (p ≫ n) with several blocks.
    const SHAPES: [(usize, usize); 14] = [
        (0, 3),
        (3, 0),
        (1, 1),
        (1, 11),
        (2, 3),
        (5, 8),
        (13, 16),
        (37, 19),
        (64, 41),
        (300, 10),
        (600, 100),
        (2000, 37),
        (6, 40),
        // At least `ROW_BLOCK_MIN_COLS` wide: a row-major view's moments
        // read their rows in blocks of `ROW_BLOCK`, here one block and three
        // single rows.
        (11, 2050),
    ];

    /// Every standardizer returns an `n × p` matrix and `p`-long moments
    /// for an `n × p` input, including the degenerate shapes: no rows, no
    /// columns, neither, and a single entry, on every layout.
    #[test]
    fn standardizers_keep_the_input_shape_at_the_edges() {
        let shape = |m: &Mat<f64>| (m.nrows(), m.ncols());
        for (k, &(n, p)) in [(0, 0), (0, 3), (3, 0), (1, 1), (1, 4), (4, 1)]
            .iter()
            .enumerate()
        {
            let x = test_matrix(n, p, 900 + k as u64);
            let lay = Layouts::new(x.as_ref());
            let w = Col::<f64>::from_fn(n, |i| 1.0 + i as f64);
            let rs = Col::<f64>::from_fn(n, |i| 0.5 + i as f64);
            let all_rows: Vec<usize> = (0..n).collect();
            for (view, xv) in lay.all(&x) {
                let what = format!("{n}x{p} {view}");
                let mut wcases: Vec<Option<ColRef<'_, f64>>> = vec![None];
                if n > 0 {
                    wcases.push(Some(w.as_ref()));
                }
                for wref in wcases {
                    let what = format!("{what} w={}", wref.is_some());
                    let (a, am, asc) = standardize_weighted(xv, wref);
                    assert_eq!(shape(&a), (n, p), "{what}: standardize_weighted");
                    assert_eq!((am.nrows(), asc.nrows()), (p, p), "{what}: moments");
                    let fb = standardize_fit_x(xv, wref, Some(rs.as_ref()));
                    assert_eq!(
                        (fb.xs().nrows(), fb.xs().ncols()),
                        (n, p),
                        "{what}: standardize_fit_x"
                    );
                    let fc = standardize_fit_x(xv, wref, None);
                    assert_eq!(
                        (fc.xs().nrows(), fc.xs().ncols()),
                        (n, p),
                        "{what}: unscaled"
                    );
                    let (d, _, _) = standardize_rows(xv, &all_rows, wref, Some(rs.as_ref()));
                    assert_eq!(shape(&d), (n, p), "{what}: standardize_rows");
                }
                let (e, mean, scale) = standardize(xv);
                assert_eq!(shape(&e), (n, p), "{what}: standardize");
                let (f, _, _) = standardize_rows(xv, &[], None, None);
                assert_eq!(shape(&f), (0, p), "{what}: standardize_rows, no rows");
                let g = standardize_apply(xv, mean.as_ref(), scale.as_ref());
                assert_eq!(shape(&g), (n, p), "{what}: standardize_apply");
                // Zero rows are refused: `preprocess_basic::zero_rows_error_one_row_is_finite`.
                if n > 0 {
                    let out = crate::preprocess::preprocess(crate::preprocess::PreprocessInput {
                        x: Some(xv),
                        y: None,
                        weights: None,
                    })
                    .unwrap();
                    let (h, _, _) = out.x_std.unwrap();
                    assert_eq!(shape(&h), (n, p), "{what}: preprocess");
                }
            }
        }
    }

    /// `standardize_fit_x` gives the entries, mean and scale of
    /// `standardize_weighted` followed by `fit::scale_rows`, bit for bit, on
    /// every layout, and keeps a row-major input row-major (no transpose).
    /// With the row factor `√w'` (as `pls1_fit` passes it), `fro` is the
    /// Frobenius norm of the result.
    #[test]
    fn standardize_fit_x_is_standardize_then_scale_rows() {
        for (k, &(n, p)) in SHAPES.iter().enumerate() {
            let x = test_matrix(n, p, 700 + k as u64);
            let lay = Layouts::new(x.as_ref());
            let weights = test_weight_sets(n);
            let rs = Col::<f64>::from_fn(n, |i| (0.5 + (i % 5) as f64 * 0.37).sqrt());
            for (view, xv) in lay.all(&x) {
                let row_major = xv.try_as_col_major().is_none() && xv.try_as_row_major().is_some();
                let mut cases: Vec<(&str, Option<ColRef<'_, f64>>)> = vec![("unweighted", None)];
                if n > 0 {
                    cases.extend(weights.iter().map(|(l, w)| (*l, Some(w.as_ref()))));
                }
                for (label, wref) in cases {
                    let what = format!("{n}x{p} {view} {label}");
                    let (b0, bm, bsc) = standardize_weighted(xv, wref);
                    let b = crate::fit::scale_rows(b0.as_ref(), rs.as_ref());
                    let a = standardize_fit_x(xv, wref, Some(rs.as_ref()));
                    assert_eq!(mat_bits(a.xs()), mat_bits(b.as_ref()), "{what}: xs");
                    assert_eq!(
                        col_bits(a.mean.as_ref()),
                        col_bits(bm.as_ref()),
                        "{what}: mean"
                    );
                    assert_eq!(
                        col_bits(a.scale.as_ref()),
                        col_bits(bsc.as_ref()),
                        "{what}: scale"
                    );
                    assert_eq!(a.row_major, row_major, "{what}: storage layout");
                    let c = standardize_fit_x(xv, wref, None);
                    assert_eq!(mat_bits(c.xs()), mat_bits(b0.as_ref()), "{what}: no rs");

                    let sw = wref.map(|w| sqrt_col(renormalize_mean_one_n(w, n).as_ref()));
                    let f = standardize_fit_x(xv, wref, sw.as_ref().map(Col::as_ref));
                    let norm = f.xs().norm_l2();
                    assert!(
                        (f.fro - norm).abs() <= 1e-12 * norm.max(1.0),
                        "{what}: fro {} vs norm {norm}",
                        f.fro
                    );
                }
            }
        }
    }

    /// A non-finite entry makes its column's mean non-finite on every
    /// layout, weighted or not, a zero weight on its row included (`0·inf`
    /// and `0·NaN` are NaN). `pls1_fit` reads X's finiteness from this.
    #[test]
    fn standardize_fit_x_flags_non_finite_columns() {
        // The second shape takes the blocked row-major moments.
        for (n, p) in [(37, 19), (11, ROW_BLOCK_MIN_COLS + 2)] {
            flags_non_finite_columns(n, p);
        }
    }

    fn flags_non_finite_columns(n: usize, p: usize) {
        let w = Col::<f64>::from_fn(n, |i| if i == 5 { 0.0 } else { 1.0 + (i % 3) as f64 });
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut x = test_matrix(n, p, 900);
            x[(5, 4)] = bad;
            let lay = Layouts::new(x.as_ref());
            for (view, xv) in lay.all(&x) {
                for wref in [None, Some(w.as_ref())] {
                    let f = standardize_fit_x(xv, wref, None);
                    for j in 0..p {
                        assert_eq!(
                            f.mean[j].is_finite(),
                            j != 4,
                            "{view} {bad} weighted={} col {j}",
                            wref.is_some()
                        );
                    }
                }
            }
        }
    }

    /// `sum_of_squares` agrees with `norm_l2` squared on every layout, and
    /// is non-finite for a NaN entry and for squares that overflow.
    #[test]
    fn sum_of_squares_matches_norm_on_every_layout() {
        for (k, &(n, p)) in SHAPES.iter().enumerate() {
            let x = Mat::<f64>::from_fn(n, p, |i, j| ((i * 31 + j * 17 + k) % 13) as f64 - 6.0);
            let lay = Layouts::new(x.as_ref());
            let norm = x.norm_l2();
            for (view, xv) in lay.all(&x) {
                let ss = sum_of_squares(xv);
                assert!(
                    (ss.sqrt() - norm).abs() <= 1e-13 * norm.max(1.0),
                    "{n}x{p} {view}: {} vs {norm}",
                    ss.sqrt()
                );
            }
        }
        let mut x = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
        x[(2, 1)] = f64::NAN;
        assert!(!sum_of_squares(x.as_ref()).is_finite());
        let big = Mat::<f64>::from_fn(4, 3, |_, _| 1e200);
        assert!(!sum_of_squares(big.as_ref()).is_finite());
    }

    #[test]
    fn standardize_weighted_is_bit_identical_to_reference() {
        for (k, &(n, p)) in SHAPES.iter().enumerate() {
            let x = test_matrix(n, p, 100 + k as u64);
            let lay = Layouts::new(x.as_ref());
            let weights = test_weight_sets(n);
            for (view, xv) in lay.all(&x) {
                let mut cases: Vec<(&str, Option<ColRef<'_, f64>>)> = vec![("unweighted", None)];
                if n > 0 {
                    cases.extend(weights.iter().map(|(l, w)| (*l, Some(w.as_ref()))));
                }
                for (label, wref) in cases {
                    let what = format!("{n}x{p} {view} {label}");
                    let (a, am, asc) = standardize_weighted(xv, wref);
                    let (b, bm, bsc) = standardize_weighted_reference(xv, wref);
                    assert_eq!(mat_bits(a.as_ref()), mat_bits(b.as_ref()), "{what}: xs");
                    assert_eq!(col_bits(am.as_ref()), col_bits(bm.as_ref()), "{what}: mean");
                    assert_eq!(
                        col_bits(asc.as_ref()),
                        col_bits(bsc.as_ref()),
                        "{what}: scale"
                    );
                }
            }
        }
    }

    #[test]
    fn standardize_apply_is_bit_identical_to_reference() {
        for (k, &(n, p)) in SHAPES.iter().enumerate() {
            let x = test_matrix(n, p, 200 + k as u64);
            let fit_rows = test_matrix(n + 5, p, 300 + k as u64);
            let (_, mean, scale) = standardize_weighted_reference(fit_rows.as_ref(), None);
            let lay = Layouts::new(x.as_ref());
            for (view, xv) in lay.all(&x) {
                let a = standardize_apply(xv, mean.as_ref(), scale.as_ref());
                let b = standardize_apply_reference(xv, mean.as_ref(), scale.as_ref());
                assert_eq!(mat_bits(a.as_ref()), mat_bits(b.as_ref()), "{n}x{p} {view}");
            }
        }
    }

    /// `standardize1_weighted` reads the same moments as a column of
    /// `standardize_weighted`, though the two go through different loops.
    #[test]
    fn standardize1_weighted_matches_each_column_of_standardize_weighted() {
        let (n, p) = (37, 19);
        let x = test_matrix(n, p, 500);
        let weights = test_weight_sets(n);
        let mut cases: Vec<Option<ColRef<'_, f64>>> = vec![None];
        cases.extend(weights.iter().map(|(_, w)| Some(w.as_ref())));
        for wref in cases {
            let (xs, mean, scale) = standardize_weighted(x.as_ref(), wref);
            for j in 0..p {
                let (z, m, s) = standardize1_weighted(x.col(j), wref);
                assert_eq!(col_bits(z.as_ref()), col_bits(xs.col(j)), "column {j}");
                assert_eq!(
                    [m.to_bits(), s.to_bits()],
                    [mean[j].to_bits(), scale[j].to_bits()]
                );
            }
        }
    }

    /// `standardize_rows` shares the block driver; pin it on full blocks
    /// and row-major views too (the row-standardization tests above use five columns).
    #[test]
    fn standardize_rows_is_bit_identical_to_reference_of_the_row_subset() {
        for (k, &(n, p)) in SHAPES.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let x = test_matrix(n, p, 600 + k as u64);
            let lay = Layouts::new(x.as_ref());
            let idx: Vec<usize> = (0..n).rev().chain([0, n - 1, n / 2]).collect();
            let m = idx.len();
            let weights = test_weight_sets(m);
            let rs = Col::<f64>::from_fn(m, |i| 0.5 + (i % 3) as f64);
            for (view, xv) in lay.all(&x) {
                let sub = row_subset(xv, &idx);
                let mut cases: Vec<(&str, Option<ColRef<'_, f64>>)> = vec![("unweighted", None)];
                cases.extend(weights.iter().map(|(l, w)| (*l, Some(w.as_ref()))));
                for (label, wref) in cases {
                    for rsref in [None, Some(rs.as_ref())] {
                        let what = format!("{n}x{p} {view} {label} rs={}", rsref.is_some());
                        let (a, am, asc) = standardize_rows(xv, &idx, wref, rsref);
                        let (b0, bm, bsc) = standardize_weighted_reference(sub.as_ref(), wref);
                        let b = match rsref {
                            None => b0,
                            Some(r) => Mat::<f64>::from_fn(m, p, |i, j| b0[(i, j)] * r[i]),
                        };
                        assert_eq!(mat_bits(a.as_ref()), mat_bits(b.as_ref()), "{what}: xs");
                        assert_eq!(col_bits(am.as_ref()), col_bits(bm.as_ref()), "{what}: mean");
                        assert_eq!(
                            col_bits(asc.as_ref()),
                            col_bits(bsc.as_ref()),
                            "{what}: scale"
                        );
                    }
                }
            }
        }
    }

    /// The block kernel is `scaled_moments` column by column, at every
    /// block width the driver uses (1 and `STD_BLOCK`) and at an odd one.
    #[test]
    fn scaled_moments_block_matches_scaled_moments() {
        fn check<const B: usize>(x: &Mat<f64>, w: Option<&Col<f64>>) {
            let n = x.nrows();
            let cols: [&[f64]; B] = core::array::from_fn(|b| x.col_as_slice(b));
            let ws: Option<Vec<f64>> = w.map(|w| (0..n).map(|i| w[i]).collect());
            let got = scaled_moments_block(cols, ws.as_deref());
            for (b, g) in got.iter().enumerate() {
                let r = scaled_moments(n, |i| x[(i, b)], w.map(Col::as_ref));
                let gb = [g.s, g.inv, g.mean, g.ss, g.sq].map(bits);
                let rb = [r.s, r.inv, r.mean, r.ss, r.sq].map(bits);
                assert_eq!(gb, rb, "B={B} n={n} column {b} weighted={}", w.is_some());
            }
        }
        for n in [0, 1, 2, 7, 37, 50, 64, 300] {
            let x = test_matrix(n, STD_BLOCK.max(10), 400 + n as u64);
            // Mean-one weights with zeros, as standardize_weighted passes them.
            let w = Col::<f64>::from_fn(n, |i| if i % 3 == 2 { 0.0 } else { 1.5 });
            for wref in [None, Some(&w)] {
                check::<1>(&x, wref);
                check::<3>(&x, wref);
                check::<STD_BLOCK>(&x, wref);
                check::<10>(&x, wref);
            }
        }
    }
}
