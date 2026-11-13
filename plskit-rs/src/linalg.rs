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
// pool sizes. The crate calls the functions below instead, with `Par::Seq`
// inside replicate workers and `fit::par_fixed()` elsewhere, so no result
// depends on the pool size.

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
///
/// `pls3::lv1_bytes` counts what this allocates (change together).
#[allow(clippy::many_single_char_names)]
pub(crate) fn thin_svd(
    a: MatRef<'_, f64>,
    par: Par,
) -> Result<ThinSvd, faer::linalg::svd::SvdError> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::svd::{ComputeSvdVectors, svd, svd_scratch};
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

/// The `R` factor of the Householder QR of `a` (`min(m, n) × n`, upper
/// trapezoidal, `a'a = R'R` up to the QR's backward error), computed in
/// place on `a` under `Par::Seq`, so its bits do not depend on the pool.
pub(crate) fn qr_r_seq(mut a: Mat<f64>) -> Mat<f64> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::qr::no_pivoting::factor::{
        qr_in_place, qr_in_place_scratch, recommended_block_size,
    };
    let (m, n) = a.shape();
    let size = m.min(n);
    if size == 0 {
        return Mat::<f64>::zeros(0, n);
    }
    let block = recommended_block_size::<f64>(m, n);
    let mut q_coeff = Mat::<f64>::zeros(block, size);
    let mut mem = MemBuffer::new(qr_in_place_scratch::<f64>(
        m,
        n,
        block,
        Par::Seq,
        faer::Spec::default(),
    ));
    qr_in_place(
        a.as_mut(),
        q_coeff.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
        faer::Spec::default(),
    );
    Mat::<f64>::from_fn(size, n, |i, j| if i <= j { a[(i, j)] } else { 0.0 })
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
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
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
///
/// # Panics
/// When `weights` is `Some` and its length differs from `x.nrows()`.
#[must_use]
pub fn standardize_weighted(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> (Mat<f64>, Col<f64>, Col<f64>) {
    if let Some(w) = weights {
        assert_eq!(
            w.nrows(),
            x.nrows(),
            "standardize_weighted: weights has {} rows, x has {}",
            w.nrows(),
            x.nrows()
        );
    }
    let w_prime: Option<Col<f64>> = weights.map(renormalize_mean_one);
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
        let mut out = Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| {
            standardized(x[(i, j)], mean[j], scale[j])
        });
        for j in 0..x.ncols() {
            requarter(
                x.nrows(),
                mean[j],
                scale[j],
                |i| x[(i, j)],
                |i, v| out[(i, j)] = v,
            );
        }
        return out;
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

/// The column scales the held-out tests divide a block by, once for all
/// rows: those the canonical standardization gives it (weighted moments,
/// mean-one `w`, when `w` is given), as `pls1_fit` scales X and the PLS3
/// tests scale X and Y; or, when `pre_standardized`, where the block is
/// used as given, all ones, except that a block whose largest entry (on a
/// row of positive weight) is outside `[2^-256, 2^256]` gets one common
/// power of two near it ([`pre_standardized_scale`]). Every held-out
/// result is unchanged by a common positive factor on the block.
pub(crate) fn held_out_scale(
    x: MatRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> Col<f64> {
    if pre_standardized {
        let c = pre_standardized_scale(x, w).unwrap_or(1.0);
        return Col::<f64>::from_fn(x.ncols(), |_| c);
    }
    fit_x_moments(x, w, crate::fit::par_fixed()).scale
}

/// The power of two a block used as given (`pre_standardized`) is divided
/// by: `Some(`[`pow2_scale`]` of its largest entry)` when that entry is
/// outside `[2^-256, 2^256]`, else `None`. Products of such entries leave
/// the `f64` range (the Gram matrix overflows from about 1e154), and a
/// power of two moves no bit of an entry, so a fit or test on the divided
/// block is the one on the block, up to that factor. With weights `w`
/// (any non-negative weights), the entries are read times [`row_keep`],
/// so a row of weight 0, which no weighted result reads, does not pick
/// the power of two.
pub(crate) fn pre_standardized_scale(
    x: MatRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
) -> Option<f64> {
    let m = match w {
        None => x.norm_max(),
        Some(w) => {
            let mut m = 0.0_f64;
            for j in 0..x.ncols() {
                for i in 0..x.nrows() {
                    m = m.max((x[(i, j)] * row_keep(w[i])).abs());
                }
            }
            m
        }
    };
    (!(2f64.powi(-256)..=2f64.powi(256)).contains(&m)).then(|| pow2_scale(m).0)
}

/// `x` divided by its [`pre_standardized_scale`] `c` under `w`, as a
/// column-major copy, with `c`; `None` when `x` is used as given. A row of
/// weight 0 is written as 0, as the moment kernels read it: divided by a
/// `c` the other rows picked, a far one could overflow.
pub(crate) fn pre_standardized_div(
    x: MatRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
) -> Option<(Mat<f64>, f64)> {
    pre_standardized_scale(x, w).map(|c| {
        let inv = 1.0 / c;
        let m = match w {
            None => Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * inv),
            Some(w) => Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| {
                x[(i, j)] * row_keep(w[i]) * inv
            }),
        };
        (m, c)
    })
}

/// A block used as given, as the paths that read it in place take it:
/// [`pre_standardized_div`]'s copy and `c`, or [`col_major_or_copy`] and 1.
pub(crate) fn pre_standardized_input(
    x: MatRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
) -> (Option<Mat<f64>>, f64) {
    pre_standardized_div(x, w).map_or_else(|| (col_major_or_copy(x), 1.0), |(m, c)| (Some(m), c))
}

/// Rows `idx` of `x` centred by their own means and divided by the given
/// column `scale`: `(x[idx[i], j] - mean[j]) / scale[j]`, times
/// `row_scale[i]` when given ([`times_row_factor`]). The means are those of
/// `standardize_weighted(row_subset(x, idx), w)`, bit for bit (`w` indexed
/// by output row, renormalized to mean one); only the
/// scale is not the part's own. Returns the matrix, the means and
/// `‖result‖_F` from the column moments ([`fro_from_terms`]), to rounding
/// when `row_scale[i]²` equals `w[i]` renormalized to mean one, or when `w`
/// and `row_scale` are both `None`.
///
/// # Panics
/// When `scale` does not have one entry per column of `x`.
pub(crate) fn center_rows_fro(
    x: MatRef<'_, f64>,
    idx: &[usize],
    w: Option<ColRef<'_, f64>>,
    row_scale: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
) -> (Mat<f64>, Col<f64>, f64) {
    assert_eq!(
        scale.nrows(),
        x.ncols(),
        "center_rows_fro: scale has {} entries, x has {} columns",
        scale.nrows(),
        x.ncols()
    );
    let w_prime: Option<Col<f64>> = w.map(renormalize_mean_one);
    let row_scale: Option<Vec<f64>> = row_scale.map(|r| (0..r.nrows()).map(|i| r[i]).collect());
    let scale: Vec<f64> = (0..scale.nrows()).map(|j| scale[j]).collect();
    let (xs, mean, _, terms) = standardize_columns_terms(
        x,
        Some(idx),
        w_prime.as_ref().map(owned_col_slice),
        row_scale.as_deref(),
        Some(&scale),
        true,
    );
    (xs, mean, fro_from_terms(&terms))
}

/// Bytes one block of [`centered_blocks`] may hold per half: small
/// enough to stay in cache between the centring pass and the products
/// formed from the block.
pub(crate) const STREAM_BLOCK_BYTES: usize = 2 << 20;

/// Columns per block of [`centered_blocks`] for halves of at most
/// `n_rows` rows: [`STREAM_BLOCK_BYTES`] worth, rounded down to a multiple
/// of [`STD_BLOCK`], at least one `STD_BLOCK`. A function of the row count
/// alone, so the block boundaries, and with them the bits of every product
/// summed over blocks, do not depend on the thread count.
/// [`fit_x_moments`] reads X in strips of this width, a row-major X only
/// while the width is at least [`ROW_MAJOR_MIN_STRIP`].
pub(crate) fn stream_block_width(n_rows: usize) -> usize {
    let cols = STREAM_BLOCK_BYTES / (8 * n_rows.max(1));
    (cols / STD_BLOCK * STD_BLOCK).max(STD_BLOCK)
}

/// Bytes of the two blocks [`centered_blocks`] holds at once for halves
/// of `n_a` and `n_b` rows at width `stream_block_width(max(n_a, n_b))`,
/// the width every batched loop passes it (change together): two owned
/// matrices, each padded as `resample::copy_bytes` counts. A block can
/// differ from [`STREAM_BLOCK_BYTES`]: the smaller half's is smaller, row
/// padding can push a block slightly over it, and past 32,768 rows, where
/// the width stays at [`STD_BLOCK`] columns, a block is larger. The full
/// width is counted even when X has fewer columns. Mirrored by
/// `stream_blocks` in `tests/memory_budget_parity.rs` (change together).
pub(crate) fn stream_blocks_bytes(n_a: usize, n_b: usize) -> usize {
    use crate::resample::copy_bytes;
    let w = stream_block_width(n_a.max(n_b));
    copy_bytes(n_a, w).saturating_add(copy_bytes(n_b, w))
}

/// `center_rows_fro(x, tr, w_tr, sw_tr, scale)` and
/// `standardize_apply_rows(x, te, mean_tr, scale, sw_te)`, `width` columns
/// at a time: `f(j0, a, c)` receives the two blocks of columns
/// `j0 .. j0 + a.ncols()`, in column order, and nothing the size of `x` is
/// written. Each column's moments are one sum chain in row order whatever
/// columns it is grouped with ([`scaled_moments_block`]), so every entry of
/// `a` and `c` has the bits the two full calls give, and the blocks are
/// column-major for every layout of `x`. `te` may be empty.
///
/// Returns `‖X̃_tr‖_F` from the column moments, the block values combined
/// by [`fro_from_terms`]: valid under [`center_rows_fro`]'s condition
/// on `w_tr` and `sw_tr`.
///
/// # Panics
/// When `width` is zero, or `scale` does not have one entry per column of
/// `x`.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
pub(crate) fn centered_blocks(
    x: MatRef<'_, f64>,
    tr: &[usize],
    te: &[usize],
    w_tr: Option<ColRef<'_, f64>>,
    sw_tr: Option<ColRef<'_, f64>>,
    sw_te: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    width: usize,
    mut f: impl FnMut(usize, MatRef<'_, f64>, MatRef<'_, f64>),
) -> f64 {
    assert!(width > 0, "centered_blocks: width must be positive");
    assert_eq!(
        scale.nrows(),
        x.ncols(),
        "centered_blocks: scale has {} entries, x has {} columns",
        scale.nrows(),
        x.ncols()
    );
    let p = x.ncols();
    let mut fro = Vec::with_capacity(p.div_ceil(width));
    let mut j0 = 0;
    while j0 < p {
        let w = width.min(p - j0);
        let xb = x.subcols(j0, w);
        let sb = scale.subrows(j0, w);
        let (a, mean, fro_b) = center_rows_fro(xb, tr, w_tr, sw_tr, sb);
        let c = standardize_apply_rows(xb, te, mean.as_ref(), sb, sw_te);
        fro.push(fro_b);
        f(j0, a.as_ref(), c.as_ref());
        j0 += w;
    }
    fro_from_terms(&fro)
}

/// `standardize_apply(row_subset(x, idx), mean, scale)` in one pass, with
/// the optional per-row factor of [`center_rows_fro`].
pub(crate) fn standardize_apply_rows(
    x: MatRef<'_, f64>,
    idx: &[usize],
    mean: ColRef<'_, f64>,
    scale: ColRef<'_, f64>,
    row_scale: Option<ColRef<'_, f64>>,
) -> Mat<f64> {
    let mut out = match row_scale {
        None => Mat::<f64>::from_fn(idx.len(), x.ncols(), |i, j| {
            standardized(x[(idx[i], j)], mean[j], scale[j])
        }),
        Some(r) => Mat::<f64>::from_fn(idx.len(), x.ncols(), |i, j| {
            times_row_factor(standardized(x[(idx[i], j)], mean[j], scale[j]), r[i])
        }),
    };
    for j in 0..x.ncols() {
        requarter(
            idx.len(),
            mean[j],
            scale[j],
            |i| x[(idx[i], j)],
            |i, v| out[(i, j)] = row_scale.map_or(v, |r| times_row_factor(v, r[i])),
        );
    }
    out
}

/// `X̃'v` for `X̃ = (x − 1·mean')/scale`, entry `j` the sum over the rows
/// in index order of `standardized(x[i, j], mean[j], scale[j])·v(i)`: the
/// entries [`standardize_apply_rows`] writes, so a part's sums and the
/// all-row sum are formed on the same bits. Nothing the size of `x` is
/// written.
pub(crate) fn centered_xt_vec(
    x: MatRef<'_, f64>,
    mean: ColRef<'_, f64>,
    scale: ColRef<'_, f64>,
    v: impl Fn(usize) -> f64,
) -> Col<f64> {
    Col::<f64>::from_fn(x.ncols(), |j| {
        if mean_overflows(mean[j]) {
            (0..x.nrows())
                .map(|i| standardized_quartered(x[(i, j)], mean[j], scale[j]) * v(i))
                .sum()
        } else {
            (0..x.nrows())
                .map(|i| standardized(x[(i, j)], mean[j], scale[j]) * v(i))
                .sum()
        }
    })
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
    let n = w.nrows();
    let s = weight_sum(w);
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
///
/// # Panics
/// When `weights` is `Some` and its length differs from `y.nrows()`.
#[must_use]
pub fn standardize1_weighted(
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> (Col<f64>, f64, f64) {
    let n = y.nrows();
    if let Some(w) = weights {
        assert_eq!(
            w.nrows(),
            n,
            "standardize1_weighted: weights has {} rows, y has {n}",
            w.nrows()
        );
    }
    let w_prime: Option<Col<f64>> = weights.map(renormalize_mean_one);
    let wpref = w_prime.as_ref().map(Col::as_ref);
    let (mean, scale) = mean_and_scale(n, |i| y[i], wpref);
    let z = Col::<f64>::from_fn(n, |i| (y[i] - mean) / scale);
    (z, mean, scale)
}

/// `(mean, scale)` of one column `v(0)..v(n−1)` under mean-one weights `w`
/// (`None` for unit weights): the kernel of [`standardize1_weighted`].
/// [`standardize_weighted`] forms the same bits
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
/// [`standardize_columns_terms`] and [`row_major_moments`], so every
/// standardizer and both fit routes see the same scale.
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

/// [`standardize_columns_terms`] with its fourth value combined by
/// [`fro_from_terms`]: the Frobenius norm of the result, to rounding, when
/// `w` and `row_scale` are both `None` or `row_scale[i]² = w[i]`.
fn standardize_columns(
    x: MatRef<'_, f64>,
    rows: Option<&[usize]>,
    w: Option<&[f64]>,
    row_scale: Option<&[f64]>,
    write: bool,
) -> (Mat<f64>, Col<f64>, Col<f64>, f64) {
    let (xs, mean, scale, terms) = standardize_columns_terms(x, rows, w, row_scale, None, write);
    (xs, mean, scale, fro_from_terms(&terms))
}

/// The body of [`standardize_weighted`] once the
/// weights are mean-one (`w`, one per output row; `None` for unit weights),
/// of [`fit_x_moments`] on one strip of columns of any layout but
/// row-major, and of [`centered_part_moments`] on one strip of any layout.
/// The input is `x` itself (`rows = None`) or its rows `idx` in that order
/// (`rows = Some(idx)`, repeats allowed). Columns go through
/// [`scaled_moments_block`] `STD_BLOCK` at a time (a shorter last block one
/// column at a time), and each block is then appended to the output as
/// `(v - mean) / scale`, times `row_scale[i]` when given, while it is still
/// in cache; every output entry is written once. A column-major `x` read
/// without `rows` is used in place; otherwise each block is copied, exactly,
/// into a column-major scratch buffer first: from a row-major `x` (such as a
/// view of a C-ordered host array) row by row over contiguous slices, from
/// any other layout column by column. Sharing this body is what makes the
/// row helpers byte-identical to standardizing a gathered copy, and what
/// makes every layout of `x` give the same bits. The fourth value is the
/// columns' [`fro_term`]s in column order; [`standardize_columns`] combines
/// them. With `write` false only the moments are formed, and the matrix
/// returned is `n_rows × 0`. With `fixed_scale` (one entry per column),
/// column `j` is divided by `fixed_scale[j]` in place of its own scale,
/// which is also the scale returned and the one its `fro_term` uses; the
/// moments still give the mean ([`center_rows_fro`]).
fn standardize_columns_terms(
    x: MatRef<'_, f64>,
    rows: Option<&[usize]>,
    w: Option<&[f64]>,
    row_scale: Option<&[f64]>,
    fixed_scale: Option<&[f64]>,
    write: bool,
) -> (Mat<f64>, Col<f64>, Col<f64>, Vec<f64>) {
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
            let (col_mean, own_scale) = mean_and_scale_from(moments, n_rows);
            let col_scale = fixed_scale.map_or(own_scale, |s| s[j0 + b]);
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
                times_row_factor(
                    standardized(cols[j - j0][i], col_mean, col_scale),
                    factor[i],
                )
            }),
        }
        for (b, &(col_mean, col_scale)) in stats[..width].iter().enumerate() {
            requarter(
                n_rows,
                col_mean,
                col_scale,
                |i| cols[b][i],
                |i, v| xs[(i, j0 + b)] = row_scale.map_or(v, |f| times_row_factor(v, f[i])),
            );
        }
    }
    (xs, mean, scale, fro)
}

/// `(v - mean) / scale`: the element expression of every standardizer,
/// in one place so that its copies cannot drift apart. Every writer runs
/// it as is, then [`requarter`]s a column whose mean lets `v − mean`
/// overflow.
#[inline]
fn standardized(v: f64, mean: f64, scale: f64) -> f64 {
    (v - mean) / scale
}

/// `z·f`, a standardized entry `z` times its row's factor `f` (`√w`), as
/// every row-scaled writer forms it. A row of factor 0 is written as 0,
/// as `HeldOutGram::with_row_weights` writes it: a finite entry far out on
/// a column of small scale can give `z = inf`, and `inf·0` would be NaN.
#[inline]
fn times_row_factor(z: f64, f: f64) -> f64 {
    if f == 0.0 { 0.0 } else { z * f }
}

/// Whether `v − mean` can round past `f64::MAX` for a finite `v`: only
/// when `|mean|` is at least `2^970` (half an ulp of `MAX`); the test uses
/// `2^969`.
fn mean_overflows(mean: f64) -> bool {
    mean.abs() >= 2f64.powi(969)
}

/// [`standardized`] from the quartered operands, for a column whose mean
/// [`mean_overflows`]: exact unless an operand is below about 8.9e-308, so
/// it is what [`standardized`] gives wherever that is finite, and finite
/// wherever the true value is.
fn standardized_quartered(v: f64, mean: f64, scale: f64) -> f64 {
    (0.25 * v - 0.25 * mean) / (0.25 * scale)
}

/// Rewrites a column of `n` entries, `set(i, ·)` from `src(i)`, with
/// [`standardized_quartered`] when its mean [`mean_overflows`]; any other
/// column is left as written. The test reads `mean` alone, once per column,
/// so the writers' own loops carry no test.
fn requarter(
    n: usize,
    mean: f64,
    scale: f64,
    src: impl Fn(usize) -> f64,
    mut set: impl FnMut(usize, f64),
) {
    if mean_overflows(mean) {
        for i in 0..n {
            set(i, standardized_quartered(src(i), mean, scale));
        }
    }
}

/// `dst[i] = standardized(src[i], mean, scale)` for every `i`, over one
/// contiguous column.
fn write_standardized(dst: &mut [f64], src: &[f64], mean: f64, scale: f64) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = standardized(v, mean, scale);
    }
    requarter(dst.len(), mean, scale, |i| src[i], |i, v| dst[i] = v);
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
/// NaN when a term is NaN, otherwise `+∞` when a term is infinite.
pub(crate) fn fro_from_terms(r: &[f64]) -> f64 {
    // `f64::max` drops a NaN operand, so the fold below cannot carry one.
    if r.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
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
    /// [`fro_term`]s): equal to the norm of the stored result, to rounding,
    /// with no weights and no `row_scale`, or with `row_scale[i]² = w'ᵢ`.
    /// `pls1_fit` always passes one of those two: no weights and no row
    /// scale, or `√w` of the mean-one weights it took the moments with. The
    /// fit uses it only for its truncation floor.
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
                            *o = times_row_factor(standardized(v, m, s), fi);
                        }
                    }
                }
            }
            for j in 0..d {
                requarter(
                    n,
                    mean[j],
                    scale[j],
                    |i| xr[(i, j)],
                    |i, v| {
                        storage[(j, i)] = factor.as_ref().map_or(v, |f| times_row_factor(v, f[i]));
                    },
                );
            }
            storage
        } else {
            // Same capacity request as `standardize_columns_terms`, so the same
            // column stride and alignment.
            let mut xs = Mat::<f64>::with_capacity(n, d);
            match &factor {
                None => xs.resize_with(n, d, |i, j| standardized(x[(i, j)], mean[j], scale[j])),
                Some(f) => {
                    xs.resize_with(n, d, |i, j| {
                        times_row_factor(standardized(x[(i, j)], mean[j], scale[j]), f[i])
                    });
                }
            }
            for j in 0..d {
                requarter(
                    n,
                    mean[j],
                    scale[j],
                    |i| x[(i, j)],
                    |i, v| xs[(i, j)] = factor.as_ref().map_or(v, |f| times_row_factor(v, f[i])),
                );
            }
            xs
        };
        self.storage = Some(storage);
    }
}

/// The moments of `standardize_weighted(x, weights)`: its `mean` and `scale`,
/// bit for bit, and `fro`, with nothing written. `x` is read in strips of columns,
/// and each strip's sweeps run while it is in cache. A column-major or strided
/// `x` is read in strips of [`stream_block_width`]`(n)` columns, so `x` is read
/// from memory once while a strip fits in [`STREAM_BLOCK_BYTES`], for `n` up to
/// 32,768 (the width does not go below [`STD_BLOCK`] columns), through
/// [`standardize_columns_terms`]. A row-major `x` (a C-ordered host array read
/// in place) is read over the strip's segments of its contiguous rows
/// ([`row_major_moments`]), in strips of [`row_major_strip_width`] columns:
/// `stream_block_width(n)` while that is at least [`ROW_MAJOR_MIN_STRIP`]
/// columns (`n` up to 1,024), otherwise wider strips, about one per thread the
/// strips can use, so that each row segment stays long. Under `Par::Rayon` the
/// strips map in parallel. Each column's sums run in row order whatever strip
/// holds it, and `fro` is one [`fro_from_terms`] over every column's
/// [`fro_term`] in column order, so no bit depends on the strips, on `par` or
/// on the pool size. [`FitX::materialize`] then writes the standardized matrix
/// in `x`'s layout when the caller needs it. The kernel's products round
/// differently on the two layouts, so a fit's last bits depend on the layout of
/// `x`, within the crate's tolerance.
pub(crate) fn fit_x_moments(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    par: Par,
) -> FitX {
    let w_prime: Option<Col<f64>> = weights.map(renormalize_mean_one);
    let w = w_prime.as_ref().map(owned_col_slice);
    let row_major = x.try_as_col_major().is_none() && x.try_as_row_major().is_some();
    let (n, d) = (x.nrows(), x.ncols());
    let width = if row_major {
        row_major_strip_width(n, d, par)
    } else {
        stream_block_width(n)
    };
    let strip = |j0: usize| {
        let xb = x.subcols(j0, width.min(d - j0));
        if row_major {
            row_major_moments(xb, w)
        } else {
            let (_, mean, scale, terms) = standardize_columns_terms(xb, None, w, None, None, false);
            (mean, scale, terms)
        }
    };
    // Each strip writes into the full-width outputs as soon as it is done,
    // so at most one strip's results per thread are alive. Holding every
    // strip's results to the end (47 small blocks at 200 × 60,000) pushed
    // glibc's heap past its trim threshold: each fit then returned and
    // page-faulted back about 2 MB, 8% of a 1-thread raw fit at that shape.
    let mut mean = Col::<f64>::zeros(d);
    let mut scale = Col::<f64>::zeros(d);
    let mut terms = vec![0.0_f64; d];
    {
        let ms = mean
            .try_as_col_major_mut()
            .expect("an owned Col is contiguous")
            .as_slice_mut();
        let ss = scale
            .try_as_col_major_mut()
            .expect("an owned Col is contiguous")
            .as_slice_mut();
        let fill = |c: usize, mo: &mut [f64], so: &mut [f64], to: &mut [f64]| {
            let (sm, sc, st) = strip(c * width);
            for b in 0..mo.len() {
                mo[b] = sm[b];
                so[b] = sc[b];
            }
            to.copy_from_slice(&st);
        };
        match par {
            Par::Seq => {
                for (c, ((mo, so), to)) in ms
                    .chunks_mut(width)
                    .zip(ss.chunks_mut(width))
                    .zip(terms.chunks_mut(width))
                    .enumerate()
                {
                    fill(c, mo, so, to);
                }
            }
            Par::Rayon(_) => {
                use rayon::prelude::*;
                ms.par_chunks_mut(width)
                    .zip(ss.par_chunks_mut(width))
                    .zip(terms.par_chunks_mut(width))
                    .enumerate()
                    .for_each(|(c, ((mo, so), to))| fill(c, mo, so, to));
            }
        }
    }
    FitX {
        storage: None,
        row_major,
        mean,
        scale,
        fro: fro_from_terms(&terms),
    }
}

/// The means of rows `idx` of `x` and `‖√w'·(X_idx − 1·mean')/scale‖_F`
/// (`w` indexed by output row, renormalized to mean one), the second and
/// third values of [`center_rows_fro`]`(x, idx, w, None, scale)` bit for
/// bit, without writing a block. `x` is read in strips of
/// [`stream_block_width`]`(idx.len())` columns through
/// [`standardize_columns_terms`] (every layout, the row-major one
/// included); under `Par::Rayon` the strips map in parallel. No bit
/// depends on the strips or on `par`.
///
/// # Panics
/// When `scale` does not have one entry per column of `x`.
pub(crate) fn centered_part_moments(
    x: MatRef<'_, f64>,
    idx: &[usize],
    w: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    par: Par,
) -> (Col<f64>, f64) {
    assert_eq!(
        scale.nrows(),
        x.ncols(),
        "centered_part_moments: scale has {} entries, x has {} columns",
        scale.nrows(),
        x.ncols()
    );
    let w_prime: Option<Col<f64>> = w.map(renormalize_mean_one);
    let w = w_prime.as_ref().map(owned_col_slice);
    let scale: Vec<f64> = (0..scale.nrows()).map(|j| scale[j]).collect();
    let d = x.ncols();
    let width = stream_block_width(idx.len());
    let mut mean = Col::<f64>::zeros(d);
    let mut terms = vec![0.0_f64; d];
    {
        let ms = mean
            .try_as_col_major_mut()
            .expect("an owned Col is contiguous")
            .as_slice_mut();
        let fill = |c: usize, mo: &mut [f64], to: &mut [f64]| {
            let (j0, b) = (c * width, mo.len());
            let (_, sm, _, st) = standardize_columns_terms(
                x.subcols(j0, b),
                Some(idx),
                w,
                None,
                Some(&scale[j0..j0 + b]),
                false,
            );
            for (o, j) in mo.iter_mut().zip(0..b) {
                *o = sm[j];
            }
            to.copy_from_slice(&st);
        };
        match par {
            Par::Seq => {
                for (c, (mo, to)) in ms
                    .chunks_mut(width)
                    .zip(terms.chunks_mut(width))
                    .enumerate()
                {
                    fill(c, mo, to);
                }
            }
            Par::Rayon(_) => {
                use rayon::prelude::*;
                ms.par_chunks_mut(width)
                    .zip(terms.par_chunks_mut(width))
                    .enumerate()
                    .for_each(|(c, (mo, to))| fill(c, mo, to));
            }
        }
    }
    (mean, fro_from_terms(&terms))
}

/// The narrowest [`stream_block_width`] at which [`fit_x_moments`] reads a
/// row-major `x` in strips of that width. Narrower, each strip row is a
/// short segment of a long row (8 columns at 20,000 rows), visited once per
/// row and sweep, and the strided reads cost more than the cache saves.
const ROW_MAJOR_MIN_STRIP: usize = 256;

/// Columns per strip of a row-major `n × d` `x` in [`fit_x_moments`]:
/// [`stream_block_width`]`(n)` when that is at least
/// [`ROW_MAJOR_MIN_STRIP`]; otherwise `max(stream_block_width(n), ⌈d / t⌉)`
/// rounded up to a multiple of [`STD_BLOCK`] and capped at `d`, with `t`
/// the threads the strips can use (1 under `Par::Seq`, the pool size under
/// `Par::Rayon`). At least one column. The width only schedules the work:
/// no bit depends on it.
fn row_major_strip_width(n: usize, d: usize, par: Par) -> usize {
    let width = stream_block_width(n);
    if width >= ROW_MAJOR_MIN_STRIP {
        return width;
    }
    let t = match par {
        Par::Seq => 1,
        Par::Rayon(_) => rayon::current_num_threads().max(1),
    };
    width
        .max(d.div_ceil(t))
        .next_multiple_of(STD_BLOCK)
        .min(d)
        .max(1)
}

/// Rows the row-major kernels ([`row_major_moments`],
/// [`row_major_t_mul`]) advance together: each pass reads and writes its
/// per-column running sums once per block of this many rows, not once per
/// row. The rows of a block are read in order, so every column's sum still
/// adds its terms one row at a time.
const ROW_BLOCK: usize = 8;

/// The row width from which [`row_major_t_mul`] forms `x'·t` itself, its
/// rows in blocks of [`ROW_BLOCK`]; below it the product declines and the
/// caller keeps faer's. One row at a time, a pass loads and stores its
/// per-column sums for every entry; that is cheap while they stay in the L1
/// cache and dominates once they spill (with faer's `X'·t`, which does the
/// same, a C-ordered `1000 × 10000` X fitted about 1.4× slower than its
/// column-major copy). Blocked, the full rows of a block are streamed from
/// memory as that many streams, which costs more than it saves on narrow
/// rows (measured on an Apple M4: slower at 1000 columns, even at 2000,
/// faster from 4000). The moment passes of [`row_major_moments`] do not use
/// this threshold: they block their rows on a strip of any width, because
/// on a strip's row segments blocking measured faster at every width tried
/// (8 to 60,000 columns, on an x86 core), short segments included.
pub(crate) const ROW_BLOCK_MIN_COLS: usize = 2048;

/// One pass of a row-major kernel: the rows of `x`, in order, `K` at a time
/// (a last, shorter block one row at a time), each with its per-row factor
/// (a weight, or an entry of the vector `x` is multiplied by).
trait RowPass {
    /// Adds the rows `rows` (with factors `f`, `None` for unit factors) to
    /// the pass's per-column state.
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>);
}

/// Runs `pass` over `rows` and their factors `f` (one per row),
/// [`ROW_BLOCK`] rows at a time (a last, shorter block one row at a time),
/// whatever the rows' width. Every caller blocks: the moment passes of
/// [`row_major_moments`] on a strip of any width, and [`row_major_t_mul`]
/// on rows at least [`ROW_BLOCK_MIN_COLS`] wide (it declines narrower
/// ones). The rows reach each column in order, so the blocking changes no
/// bit.
fn run_rows(rows: &[&[f64]], f: Option<&[f64]>, pass: &mut impl RowPass) {
    let mut chunks = rows.chunks_exact(ROW_BLOCK);
    for (b, chunk) in (&mut chunks).enumerate() {
        let blk: [&[f64]; ROW_BLOCK] = core::array::from_fn(|k| chunk[k]);
        pass.rows(
            blk,
            f.map(|f| core::array::from_fn(|k| f[b * ROW_BLOCK + k])),
        );
    }
    let i0 = rows.len() - chunks.remainder().len();
    for (t, &r) in chunks.remainder().iter().enumerate() {
        pass.rows([r], f.map(|f| [f[i0 + t]]));
    }
}

/// The contiguous rows of a row-major `x`, each `ncols` long, in order.
fn row_slices(x: MatRef<'_, f64>) -> Vec<&[f64]> {
    let xr = x.try_as_row_major().expect("the caller checked the layout");
    let d = x.ncols();
    (0..x.nrows()).map(|i| &xr.row(i).as_slice()[..d]).collect()
}

/// `max |x|` per column, each row's entries times [`row_keep`] of its
/// weight when weighted. `f64::max` ignores a NaN operand, as the fold in
/// `scaled_moments` does; the order of a max does not matter.
struct MaxAbs<'a> {
    max_abs: &'a mut [f64],
}

#[allow(clippy::many_single_char_names)]
impl RowPass for MaxAbs<'_> {
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>) {
        let d = self.max_abs.len();
        let rows = rows.map(|r| &r[..d]);
        match f {
            None => {
                for (j, m) in self.max_abs.iter_mut().enumerate() {
                    let mut v = *m;
                    for r in &rows {
                        v = v.max(r[j].abs());
                    }
                    *m = v;
                }
            }
            Some(w) => {
                let k = w.map(row_keep);
                for (j, m) in self.max_abs.iter_mut().enumerate() {
                    let mut v = *m;
                    for (r, &ki) in rows.iter().zip(&k) {
                        v = v.max((r[j] * ki).abs());
                    }
                    *m = v;
                }
            }
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
                let k = w.map(row_keep);
                for j in 0..d {
                    let (mut s, mut q, iv) = (sum[j], sq[j], inv[j]);
                    for ((r, &wi), &ki) in rows.iter().zip(&w).zip(&k) {
                        let u = r[j] * ki * iv;
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

#[allow(clippy::many_single_char_names)]
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
                let k = w.map(row_keep);
                for j in 0..d {
                    let (mut acc, iv, m) = (ss[j], inv[j], mean[j]);
                    for ((r, &wi), &ki) in rows.iter().zip(&w).zip(&k) {
                        let dv = r[j] * ki * iv - m;
                        acc += wi * (dv * dv);
                    }
                    ss[j] = acc;
                }
            }
        }
    }
}

/// The row-major arm of [`fit_x_moments`] on one strip of columns (a row-major
/// view, each row a contiguous segment), every row in order: the columns'
/// means, scales and
/// [`fro_term`]s, in column order. The moments are those of
/// [`scaled_moments_block`] (the same terms, in the same row order, from the
/// same start, so the same bits), each column's sums advancing together over
/// the rows ([`run_rows`]: blocks of [`ROW_BLOCK`] rows, on a strip of any
/// width); change the two together. Three passes over the strip: `max |x|`, the
/// mean sum and `Σu²` together, and `Σ(u − mean)²`, with `u = x·(1/s)`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn row_major_moments(x: MatRef<'_, f64>, w: Option<&[f64]>) -> (Col<f64>, Col<f64>, Vec<f64>) {
    let rows = row_slices(x);
    let (n, d) = (rows.len(), x.ncols());
    let zero = sum_start();

    let mut max_abs = vec![0.0_f64; d];
    run_rows(
        &rows,
        w,
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
    (mean, scale, fro)
}

/// `x'·t` over the columns `j0 .. j0 + out.len()` of the rows, accumulated
/// into `out` (one sum per column, rows in order).
struct TMul<'a> {
    out: &'a mut [f64],
    j0: usize,
}

impl RowPass for TMul<'_> {
    fn rows<const K: usize>(&mut self, rows: [&[f64]; K], f: Option<[f64; K]>) {
        let t = f.expect("TMul reads its rows' entries of t as factors");
        let (j0, d) = (self.j0, self.out.len());
        let rows = rows.map(|r| &r[j0..j0 + d]);
        for (j, o) in self.out.iter_mut().enumerate() {
            let mut acc = *o;
            for (r, &ti) in rows.iter().zip(&t) {
                acc += ti * r[j];
            }
            *o = acc;
        }
    }
}

/// `x'·t` for a row-major `x` at least [`ROW_BLOCK_MIN_COLS`] wide (a
/// C-ordered host array read in place, raw or pre-standardized, or the
/// row-major copy [`FitX::materialize`] writes), read one contiguous block of
/// rows at a time ([`run_rows`]). faer forms this product as a column-major GEMV of
/// `x'`, which loads and stores the whole output once per row: on a wide `x`
/// the output leaves the L1 cache and the product ran about 1.6× slower than
/// on the column-major copy. `None` for any other layout or a narrower `x`:
/// the caller then multiplies with faer.
///
/// Column `j` is `Σᵢ tᵢ·xᵢⱼ` from `0.0` in row order. Under `Par::Rayon` the
/// rows are cut into [`crate::fit::PAR_DEGREE`] contiguous pieces, each
/// summed on its own from `0.0` and the pieces' sums then added in piece
/// order, `((p₀ + p₁) + p₂) + …` up to the last piece, so the bits depend
/// on the shape and on `par`, never on the pool size; `Par::Seq` and
/// `Par::Rayon` can differ in the last bits, as faer's two arms do. The
/// parallel arm maps chunks of columns in parallel ([`t_mul_chunk_width`]:
/// about `PAR_DEGREE` chunks up to `PAR_DEGREE · T_MUL_CHUNK` columns,
/// chunks of [`T_MUL_CHUNK`] columns on wider rows, the last chunk
/// shorter); each chunk forms the pieces for its columns, the first in
/// place and the others in one chunk-sized scratch, so no partial vector of
/// length `d` is allocated. The chunks never change a piece's rows or the
/// order of the pieces, so no bit depends on them.
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
    let mut out = vec![0.0_f64; d];
    match par {
        Par::Seq => run_rows(
            &rows,
            Some(&t),
            &mut TMul {
                out: &mut out,
                j0: 0,
            },
        ),
        Par::Rayon(_) => {
            use rayon::prelude::*;
            let p = crate::fit::PAR_DEGREE;
            let chunk = t_mul_chunk_width(d);
            out.par_chunks_mut(chunk).enumerate().for_each(|(c, o)| {
                let j0 = c * chunk;
                let piece = |k: usize, out: &mut [f64]| {
                    let (lo, hi) = (k * n / p, (k + 1) * n / p);
                    run_rows(&rows[lo..hi], Some(&t[lo..hi]), &mut TMul { out, j0 });
                };
                piece(0, o);
                let mut part = vec![0.0_f64; o.len()];
                for k in 1..p {
                    part.fill(0.0);
                    piece(k, &mut part);
                    for (a, v) in o.iter_mut().zip(&part) {
                        *a += v;
                    }
                }
            });
        }
    }
    Some(Col::<f64>::from_fn(d, |j| out[j]))
}

/// The most columns per chunk of [`row_major_t_mul`]'s parallel arm: a
/// chunk's output and scratch (at most 32 KiB each) stay in cache while its
/// pieces run.
const T_MUL_CHUNK: usize = 4096;

/// Columns per chunk of [`row_major_t_mul`]'s parallel arm on rows `d`
/// long: `⌈d / PAR_DEGREE⌉` rounded up to a multiple of [`STD_BLOCK`], at
/// least `STD_BLOCK` and at most [`T_MUL_CHUNK`]. Every `d` the product
/// takes (at least [`ROW_BLOCK_MIN_COLS`]) then gets at least `PAR_DEGREE`
/// chunks, so narrow rows keep the arm's parallelism, and wide rows get
/// cache-sized chunks. A function of `d` alone, never of the pool size.
fn t_mul_chunk_width(d: usize) -> usize {
    d.div_ceil(crate::fit::PAR_DEGREE)
        .next_multiple_of(STD_BLOCK)
        .clamp(STD_BLOCK, T_MUL_CHUNK)
}

/// `Σ xᵢⱼ²` with no overflow guard. The storage lines (contiguous columns
/// of a column-major `x`, contiguous rows of a row-major one, columns of any
/// other layout) are cut into [`crate::fit::PAR_DEGREE`] contiguous ranges;
/// each range is summed with four interleaved partial sums over each of its
/// lines (entry by entry on a strided view, into the first sum), and the
/// ranges' sums are added in range order. Under `Par::Rayon` the ranges run
/// in parallel; the bits are the same under either arm and at any pool
/// size. Non-finite when an entry is non-finite or a square overflows (`|x|`
/// above about `1e154`). The summation order depends on the layout;
/// `pls1_fit` uses it only for its truncation floor and as a finiteness
/// screen.
pub(crate) fn sum_of_squares(x: MatRef<'_, f64>, par: Par) -> f64 {
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
    let xc = x.try_as_col_major();
    let xr = if xc.is_none() {
        x.try_as_row_major()
    } else {
        None
    };
    let lines = if xr.is_some() { x.nrows() } else { x.ncols() };
    let pieces = crate::fit::PAR_DEGREE;
    let piece = |k: usize| -> f64 {
        let mut acc = [0.0_f64; 4];
        for l in k * lines / pieces..(k + 1) * lines / pieces {
            if let Some(xc) = xc {
                add_slice(xc.col(l).as_slice(), &mut acc);
            } else if let Some(xr) = xr {
                add_slice(xr.row(l).as_slice(), &mut acc);
            } else {
                for i in 0..x.nrows() {
                    let e = x[(i, l)];
                    acc[0] += e * e;
                }
            }
        }
        acc.iter().sum::<f64>()
    };
    let parts: Vec<f64> = match par {
        Par::Seq => (0..pieces).map(piece).collect(),
        Par::Rayon(_) => {
            use rayon::prelude::*;
            (0..pieces).into_par_iter().map(piece).collect()
        }
    };
    parts.iter().sum()
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

/// The factor the moment kernels read row `i`'s entries with, from its
/// mean-one weight: 0 for a row of weight 0, 1 otherwise. Such a row adds
/// nothing to the weighted sums, and read as `x·0` it does not set their
/// power-of-two scale either: a far one (an entry near 1e200 among entries
/// near 1) would push the other rows' squares below the normal range, or
/// its own past it, where its term `0·inf` is NaN. `x·0` is 0 for a finite
/// entry at any magnitude and NaN for a non-finite one, so the sums still
/// carry a non-finite entry. A factor of 1 leaves every bit as it was.
#[inline]
fn row_keep(wi: f64) -> f64 {
    if wi == 0.0 { 0.0 } else { 1.0 }
}

/// Moments of `v(0)..v(n−1)` formed on `v/s`, `s` the power of two
/// [`pow2_scale`] picks for the largest `|vᵢ|`, under mean-one weights `w`
/// (`None` for unit weights). The one kernel behind the standardizers
/// ([`mean_and_scale`]; [`standardize_weighted`] runs its lockstep form [`scaled_moments_block`], which returns the same
/// bits) and the
/// split-half correlations
/// (`signal_test::guarded_pearson`, `signal_test::nr_split_r`,
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
/// `(wᵢ·d)·d`; the two round differently. Weighted, every entry is read
/// times [`row_keep`]`(wᵢ)`, so a row of weight 0 does not set `s`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn scaled_moments(
    n: usize,
    v: impl Fn(usize) -> f64,
    w: Option<ColRef<'_, f64>>,
) -> ScaledMoments {
    let v = |i: usize| w.map_or_else(|| v(i), |w| v(i) * row_keep(w[i]));
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
/// `wᵢ·(d·d)` as in [`scaled_moments`], and every pass reads `x` times
/// [`row_keep`]`(wᵢ)`. [`row_major_moments`] repeats
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
    match w {
        None => {
            for i in 0..n {
                for b in 0..B {
                    max_abs[b] = max_abs[b].max(cols[b][i].abs());
                }
            }
        }
        Some(w) => {
            for (i, &wi) in w.iter().enumerate() {
                let k = row_keep(wi);
                for b in 0..B {
                    max_abs[b] = max_abs[b].max((cols[b][i] * k).abs());
                }
            }
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
                let k = row_keep(wi);
                for b in 0..B {
                    let u = cols[b][i] * k * inv[b];
                    sum[b] += wi * u;
                    sq[b] += wi * (u * u);
                }
            }
            mean = core::array::from_fn(|b| sum[b] / n_f);
            for (i, &wi) in w.iter().enumerate() {
                let k = row_keep(wi);
                for b in 0..B {
                    let d = cols[b][i] * k * inv[b] - mean[b];
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
/// Zero matrix (σ₁ = 0) returns 0.0. NaN when the Gram matrix below is not
/// finite: a non-finite entry of `a`, or a product that overflows.
///
/// Computed from the Gram matrix on `a`'s shorter side (`AA'` or `A'A`),
/// whose trace is `‖A‖²_F` and whose largest eigenvalue is `σ₁²`: it takes
/// `min(m, n)²` memory, where an SVD of `a` needs a working copy of `a`.
/// The product and the eigendecomposition run on the crate's fixed-degree
/// Rayon split (`fit::par_fixed`), so the result does not depend on the
/// size of the Rayon pool it is called from.
///
/// # Panics
/// Panics if faer's eigendecomposition fails to converge (only expected on
/// pathological input).
#[must_use]
#[allow(clippy::doc_markdown)] // "NeuroImage"
pub fn stable_rank(a: MatRef<'_, f64>) -> f64 {
    let short = if a.nrows() <= a.ncols() {
        a
    } else {
        a.transpose()
    };
    if short.nrows() == 0 {
        return 0.0;
    }
    let g = mat_mul(short, short.transpose(), crate::fit::par_fixed());
    stable_rank_from_gram(g.as_ref())
}

/// [`stable_rank`] of a matrix whose Gram matrix on its shorter side is
/// `g`: `trace(g) / λ_max(g)`, `0.0` for an empty or non-positive `g`, NaN
/// for a non-finite `g`.
///
/// # Panics
/// When the eigendecomposition fails to converge on a finite `g`, as in
/// [`stable_rank`].
pub(crate) fn stable_rank_from_gram(g: MatRef<'_, f64>) -> f64 {
    let d = g.nrows();
    if d == 0 {
        return 0.0;
    }
    // faer's eigendecomposition can fail to converge on a non-finite `g`,
    // as it does on a standardized X whose centering overflows.
    if !g.is_all_finite() {
        return f64::NAN;
    }
    let (lam, _) = self_adjoint_eigen(g, crate::fit::par_fixed())
        .expect("eigendecomposition failed to converge");
    // Ascending, so the last one is σ₁².
    let sigma1_sq = lam[d - 1];
    if sigma1_sq <= 0.0 {
        return 0.0;
    }
    let frob_sq: f64 = (0..d).map(|i| g[(i, i)]).sum();
    frob_sq / sigma1_sq
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
#[allow(clippy::similar_names)]
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
    if t_val > 0.0 { half_p } else { 1.0 - half_p }
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
    /// So the frozen corpus is unaffected by the pre-scaling.
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

    #[test]
    fn stable_rank_from_gram_is_stable_rank_of_the_short_side() {
        let x = mat(3, 3, &[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let g = mat_mul(x.as_ref(), x.transpose(), crate::fit::par_fixed());
        assert_eq!(
            stable_rank_from_gram(g.as_ref()).to_bits(),
            stable_rank(x.as_ref()).to_bits()
        );
        assert_relative_eq!(stable_rank_from_gram(Mat::<f64>::zeros(0, 0).as_ref()), 0.0);
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

    /// A `weights` column of another length than the data is a caller
    /// bug: the public standardizers panic instead of reading a prefix.
    #[test]
    #[should_panic(expected = "standardize_weighted: weights has 5 rows, x has 4")]
    fn standardize_weighted_rejects_weights_longer_than_the_rows() {
        let x = small_x();
        let w = Col::<f64>::from_fn(x.nrows() + 1, |i| 1.0 + i as f64);
        let _ = standardize_weighted(x.as_ref(), Some(w.as_ref()));
    }

    #[test]
    #[should_panic(expected = "standardize1_weighted: weights has 5 rows, y has 4")]
    fn standardize1_weighted_rejects_weights_longer_than_the_rows() {
        let x = small_x();
        let w = Col::<f64>::from_fn(x.nrows() + 1, |i| 1.0 + i as f64);
        let _ = standardize1_weighted(x.col(0), Some(w.as_ref()));
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
        // Hand-computed: w = [1, 2, 3]. Σw=6, n=3, so w·n/Σw = [0.5, 1, 1.5].
        let w = Col::<f64>::from_fn(3, |i| (i + 1) as f64);
        let wn = normalize_weights(w.as_ref()).unwrap();
        for (i, want) in [0.5, 1.0, 1.5].into_iter().enumerate() {
            assert_relative_eq!(wn[i], want, epsilon = 1e-12);
        }
        // Total stays at n.
        let s: f64 = (0..3).map(|i| wn[i]).sum();
        assert_relative_eq!(s, 3.0, epsilon = 1e-12);
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
        use faer::Accum;
        use faer::linalg::matmul::matmul;
        use faer::linalg::solvers::{PartialPivLu, Solve};
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
        use rand::{RngExt, SeedableRng, rngs::StdRng};
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
/// `subsample`.
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
    use crate::test_support::{
        Layouts, assert_bits_eq, assert_mat_bits_eq, col_vals, mat_vals, test_weights, unif,
    };

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

    /// 12 × 5: columns at scales 1e-3, 1 and 1e3, the constant 7.0, and
    /// one with an outlier (row 4).
    fn held_out_x() -> Mat<f64> {
        let u = unif(12, 5, 33);
        Mat::<f64>::from_fn(12, 5, |i, j| match j {
            0 => 1e-3 * u[(i, 0)],
            2 => 1e3 * u[(i, 2)],
            3 => 7.0,
            4 if i == 4 => 50.0,
            _ => u[(i, j)],
        })
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn held_out_scale_is_the_fit_scale_or_ones() {
        let x = held_out_x();
        for w in [None, Some(test_weights(12))] {
            let w = w.as_ref().map(Col::as_ref);
            let what = format!("weighted={}", w.is_some());
            let s = held_out_scale(x.as_ref(), w, false);
            let (_, _, s_ref) = standardize_weighted(x.as_ref(), w);
            assert_bits_eq(&col_vals(s.as_ref()), &col_vals(s_ref.as_ref()), &what);
            let ones = held_out_scale(x.as_ref(), w, true);
            assert_eq!(col_vals(ones.as_ref()), vec![1.0; 5], "{what}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn center_rows_divides_part_centred_rows_by_the_given_scale() {
        let x = held_out_x();
        let idx = [0, 2, 3, 5, 8, 11];
        for w in [None, Some(test_weights(12))] {
            let what = format!("weighted={}", w.is_some());
            let scale = held_out_scale(x.as_ref(), w.as_ref().map(Col::as_ref), false);
            // Part weights, and the row factor `√` of them renormalized to mean one.
            let w_part = w.map(|w| Col::<f64>::from_fn(idx.len(), |i| w[idx[i]]));
            let w_part = w_part.as_ref().map(Col::as_ref);
            let sw = w_part.map(|w| sqrt_col(renormalize_mean_one(w).as_ref()));
            let sw = sw.as_ref().map(Col::as_ref);
            let (xc, mean, fro) = center_rows_fro(x.as_ref(), &idx, w_part, sw, scale.as_ref());
            let (_, mean_ref, _) =
                standardize_weighted(row_subset(x.as_ref(), &idx).as_ref(), w_part);
            assert_bits_eq(
                &col_vals(mean.as_ref()),
                &col_vals(mean_ref.as_ref()),
                &what,
            );
            for j in 0..5 {
                for i in 0..idx.len() {
                    let r = sw.map_or(1.0, |s| s[i]);
                    let expect = ((x[(idx[i], j)] - mean[j]) / scale[j]) * r;
                    approx::assert_relative_eq!(xc[(i, j)], expect, max_relative = 1e-14);
                }
            }
            approx::assert_relative_eq!(fro, xc.norm_l2(), max_relative = 1e-12);
            assert!(
                (0..idx.len()).all(|i| xc[(i, 3)] == 0.0),
                "{what}: constant column"
            );
        }
    }

    /// `centered_part_moments` over a list of rows gives `center_rows_fro`'s
    /// means and norm at the all-row scale, bit for bit, and its means are
    /// `standardize_weighted`'s of those rows, on every layout, with and without
    /// weights, under `Par::Seq` and the fixed split, over the copy-free
    /// families and a 40 × 13,200 X whose 27 listed rows read two strips of
    /// `stream_block_width(27)` columns, the last one partial.
    #[test]
    fn centered_part_moments_match_center_rows_fro() {
        use crate::test_support::{
            Agree, Family, assert_agree, assert_layout_agree, copy_free_families, signal_data,
        };
        let (xs, ys) = signal_data(40, 13_200, 106);
        let mut families = copy_free_families();
        for (name, w) in [
            ("strips", None),
            ("weighted_strips", Some(test_weights(40))),
        ] {
            families.push(Family {
                name,
                x: xs.clone(),
                y: ys.clone(),
                w,
            });
        }
        for f in families {
            let n = f.x.nrows();
            // Out of order, with rows left out, as a fold's training rows.
            let idx: Vec<usize> = (0..n).rev().filter(|i| i % 3 != 1).collect();
            if f.name.ends_with("strips") {
                let width = stream_block_width(idx.len());
                assert!(
                    f.x.ncols() > width && f.x.ncols() < 2 * width && f.x.ncols() % width != 0,
                    "{}: not two strips with a partial last one",
                    f.name
                );
            }
            let scale = held_out_scale(f.x.as_ref(), f.w.as_ref().map(Col::as_ref), false);
            let w_rows: Option<Col<f64>> =
                f.w.as_ref()
                    .map(|w| Col::<f64>::from_fn(idx.len(), |i| w[idx[i]]));
            let wr = w_rows.as_ref().map(Col::as_ref);
            let (_, mean, fro) = center_rows_fro(f.x.as_ref(), &idx, wr, None, scale.as_ref());
            let (_, mean_rows, _) =
                standardize_weighted(row_subset(f.x.as_ref(), &idx).as_ref(), wr);
            assert_bits_eq(
                &col_vals(mean.as_ref()),
                &col_vals(mean_rows.as_ref()),
                f.name,
            );
            let mut want = col_vals(mean.as_ref());
            want.push(fro);
            for par in [Par::Seq, crate::fit::par_fixed()] {
                let what = format!("{} {par:?}", f.name);
                let got = assert_layout_agree(&f.x, &what, Agree::Bits, |xv| {
                    let (m, fro) = centered_part_moments(xv, &idx, wr, scale.as_ref(), par);
                    let mut v = col_vals(m.as_ref());
                    v.push(fro);
                    Ok(v)
                })
                .unwrap();
                assert_agree(&got, &want, Agree::Bits, &what);
            }
        }
    }

    /// Training and held-out blocks of `centered_blocks`, concatenated.
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn collect_centered_blocks(
        x: MatRef<'_, f64>,
        tr: &[usize],
        te: &[usize],
        w: Option<ColRef<'_, f64>>,
        sw_tr: Option<ColRef<'_, f64>>,
        sw_te: Option<ColRef<'_, f64>>,
        scale: ColRef<'_, f64>,
        width: usize,
    ) -> (Mat<f64>, Mat<f64>, f64, Vec<usize>) {
        let mut a_all = Mat::<f64>::zeros(tr.len(), x.ncols());
        let mut c_all = Mat::<f64>::zeros(te.len(), x.ncols());
        let mut starts = Vec::new();
        let fro = centered_blocks(x, tr, te, w, sw_tr, sw_te, scale, width, |j0, a, c| {
            starts.push(j0);
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    a_all[(i, j0 + j)] = a[(i, j)];
                }
                for i in 0..c.nrows() {
                    c_all[(i, j0 + j)] = c[(i, j)];
                }
            }
        });
        (a_all, c_all, fro, starts)
    }

    #[test]
    #[allow(clippy::similar_names, clippy::cast_precision_loss)]
    fn centered_blocks_match_full_calls_at_every_width() {
        // 24 × 37: wider than one STD_BLOCK and not a multiple of it.
        let x = unif(24, 37, 5);
        let lay = Layouts::new(x.as_ref());
        let tr: Vec<usize> = (0..24).step_by(2).collect();
        let te: Vec<usize> = (1..24).step_by(2).collect();
        let w = renormalize_mean_one(test_weights(tr.len()).as_ref());
        let sw_tr = sqrt_col(w.as_ref());
        let sw_te = Col::<f64>::from_fn(te.len(), |i| 0.5 + (i % 3) as f64);
        let scale = Col::<f64>::from_fn(37, |j| 0.5 + j as f64 * 0.25);
        for (view, xv) in lay.all(&x) {
            for weighted in [false, true] {
                let (wr, sr, se) = if weighted {
                    (Some(w.as_ref()), Some(sw_tr.as_ref()), Some(sw_te.as_ref()))
                } else {
                    (None, None, None)
                };
                let (full_a, mean, full_fro) = center_rows_fro(xv, &tr, wr, sr, scale.as_ref());
                let full_c = standardize_apply_rows(xv, &te, mean.as_ref(), scale.as_ref(), se);
                for width in [1, 3, 8, 64] {
                    let what = format!("{view} weighted={weighted} width={width}");
                    let (a, c, fro, starts) =
                        collect_centered_blocks(xv, &tr, &te, wr, sr, se, scale.as_ref(), width);
                    assert_mat_bits_eq(a.as_ref(), full_a.as_ref(), &what);
                    assert_mat_bits_eq(c.as_ref(), full_c.as_ref(), &what);
                    let expect: Vec<usize> = (0..37).step_by(width).collect();
                    assert_eq!(starts, expect, "{what}: block starts");
                    // One term per block (width 1) or one block: the same
                    // `fro_from_terms` sum. Otherwise block norms are combined.
                    if width == 1 || width >= 37 {
                        assert_eq!(fro.to_bits(), full_fro.to_bits(), "{what}: fro");
                    } else {
                        approx::assert_relative_eq!(fro, full_fro, max_relative = 1e-13);
                    }
                    approx::assert_relative_eq!(fro, full_a.norm_l2(), max_relative = 1e-13);
                }
            }
        }
    }

    #[test]
    fn centered_blocks_take_an_empty_held_out_set() {
        let x = data();
        let rows: Vec<usize> = (0..24).collect();
        let scale = held_out_scale(x.as_ref(), None, false);
        let (full, _, _) = center_rows_fro(x.as_ref(), &rows, None, None, scale.as_ref());
        let (a, c, _, _) =
            collect_centered_blocks(x.as_ref(), &rows, &[], None, None, None, scale.as_ref(), 2);
        assert_mat_bits_eq(a.as_ref(), full.as_ref(), "all rows, no held-out");
        assert_eq!((c.nrows(), c.ncols()), (0, x.ncols()));
    }

    /// `fro_from_terms` is NaN when a term is NaN, also when every term is:
    /// `f64::max` drops a NaN, so the largest-term fold alone would read an
    /// all-NaN slice as the zero norm.
    /// `centered_blocks` combines its block norms with it, so one
    /// all-NaN block makes the whole norm NaN next to a finite block.
    #[test]
    #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
    fn fro_from_terms_is_nan_when_a_term_is_nan() {
        assert!(fro_from_terms(&[f64::NAN, f64::NAN]).is_nan());
        assert!(fro_from_terms(&[f64::NAN, 2.0]).is_nan());
        assert!(fro_from_terms(&[f64::INFINITY, f64::NAN]).is_nan());
        assert_eq!(fro_from_terms(&[f64::INFINITY, 2.0]), f64::INFINITY);
        assert_eq!(fro_from_terms(&[3.0, 4.0]), 5.0);
        assert_eq!(fro_from_terms(&[]), 0.0);

        // Width 2: block 0 (columns 0 and 1) is all NaN, block 1 finite.
        let x = Mat::<f64>::from_fn(6, 4, |i, j| {
            if j < 2 {
                f64::NAN
            } else {
                (i * (j + 1)) as f64
            }
        });
        let rows: Vec<usize> = (0..6).collect();
        let ones = Col::<f64>::from_fn(4, |_| 1.0);
        let fro = centered_blocks(
            x.as_ref(),
            &rows,
            &[],
            None,
            None,
            None,
            ones.as_ref(),
            2,
            |_, _, _| {},
        );
        assert!(fro.is_nan(), "fro = {fro}");
    }

    #[test]
    fn stream_block_width_is_a_multiple_of_the_moment_block() {
        for n in [1, 2, 127, 203, 254, 6000, 10_000_000] {
            let w = stream_block_width(n);
            assert!(w >= STD_BLOCK && w % STD_BLOCK == 0, "n={n}: width {w}");
            assert!(
                w == STD_BLOCK || 8 * n * w <= STREAM_BLOCK_BYTES,
                "n={n}: width {w}"
            );
        }
        assert_eq!(stream_block_width(127), 2064);
        assert_eq!(stream_block_width(0), stream_block_width(1));
    }

    /// `stream_blocks_bytes` is the most the two blocks of
    /// `centered_blocks` allocate at the width the batched loops pass,
    /// over a full block and a partial one: halves equal and unequal, not
    /// multiples of 8, and past 32,768 rows, where the width stays at
    /// `STD_BLOCK` columns.
    #[test]
    fn stream_blocks_bytes_is_what_centered_blocks_allocates() {
        let bytes = |m: MatRef<'_, f64>| m.col_stride().unsigned_abs() * m.ncols() * 8;
        for (n_a, n_b) in [(20, 20), (21, 19), (32, 8), (127, 127), (40_000, 1)] {
            let n = n_a + n_b;
            let width = stream_block_width(n_a.max(n_b));
            let p = width + 3;
            let x = Mat::<f64>::from_fn(n, p, |i, j| ((i * 7 + j * 3) % 11) as f64);
            let tr: Vec<usize> = (0..n_a).collect();
            let te: Vec<usize> = (n_a..n).collect();
            let ones = Col::<f64>::from_fn(p, |_| 1.0);
            let mut most = 0;
            centered_blocks(
                x.as_ref(),
                &tr,
                &te,
                None,
                None,
                None,
                ones.as_ref(),
                width,
                |_, a, c| {
                    most = most.max(bytes(a) + bytes(c));
                },
            );
            assert_eq!(stream_blocks_bytes(n_a, n_b), most, "{n_a} + {n_b} rows");
        }
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

    /// 12 × 3 under [`test_weights`] (row 10 has weight 0), row 10 set to `row10`.
    fn with_row10(row10: [f64; 3]) -> Mat<f64> {
        let u = unif(12, 3, 41);
        Mat::<f64>::from_fn(12, 3, |i, j| if i == 10 { row10[j] } else { u[(i, j)] })
    }

    /// A row of weight 0 enters no weighted moment however far it lies, in
    /// every moment kernel and layout; a non-finite entry there still makes
    /// its column's mean non-finite, which is how the fit detects it.
    #[test]
    #[allow(clippy::float_cmp)]
    fn a_far_zero_weight_row_leaves_the_weighted_moments_unchanged() {
        let w = test_weights(12);
        assert_eq!(w[10], 0.0);
        let wn = normalize_weights(w.as_ref()).unwrap();
        let near = with_row10([0.5; 3]);
        for row10 in [[1e200, -1e300, 0.5], [f64::NAN, f64::INFINITY, 0.5]] {
            let far = with_row10(row10);
            let finite = row10[0].is_finite();
            let (ln, lf) = (Layouts::new(near.as_ref()), Layouts::new(far.as_ref()));
            for ((name, a), (_, b)) in ln.all(&near).into_iter().zip(lf.all(&far)) {
                let fa = fit_x_moments(a, Some(w.as_ref()), Par::Seq);
                let fb = fit_x_moments(b, Some(w.as_ref()), Par::Seq);
                if finite {
                    assert_bits_eq(
                        &col_vals(fb.mean.as_ref()),
                        &col_vals(fa.mean.as_ref()),
                        name,
                    );
                    assert_bits_eq(
                        &col_vals(fb.scale.as_ref()),
                        &col_vals(fa.scale.as_ref()),
                        name,
                    );
                    assert_eq!(fb.fro.to_bits(), fa.fro.to_bits(), "{name}");
                } else {
                    assert!(!fb.mean[0].is_finite() && !fb.mean[1].is_finite(), "{name}");
                }
            }
            for j in 0..3 {
                let a = mean_and_scale(12, |i| near[(i, j)], Some(wn.as_ref()));
                let b = mean_and_scale(12, |i| far[(i, j)], Some(wn.as_ref()));
                if finite || j == 2 {
                    assert_eq!(
                        (b.0.to_bits(), b.1.to_bits()),
                        (a.0.to_bits(), a.1.to_bits()),
                        "col {j}"
                    );
                } else {
                    assert!(!b.0.is_finite(), "col {j}");
                }
            }
        }
    }

    /// Used as given, a block's power of two comes from its rows of positive
    /// weight: a zero-weight row at 1e300 among rows near 1e-200 does not
    /// pick it, and is 0 in the divided copy, where dividing it would
    /// overflow.
    #[test]
    #[allow(clippy::float_cmp)]
    fn pre_standardized_scale_ignores_zero_weight_rows() {
        let w = test_weights(12);
        let u = unif(12, 3, 45);
        let x = |far: f64| {
            Mat::<f64>::from_fn(12, 3, |i, j| if i == 10 { far } else { 1e-200 * u[(i, j)] })
        };
        let (near, far) = (x(1e-200), x(1e300));
        let wr = Some(w.as_ref());
        assert_bits_eq(
            &col_vals(held_out_scale(far.as_ref(), wr, true).as_ref()),
            &col_vals(held_out_scale(near.as_ref(), wr, true).as_ref()),
            "held_out_scale",
        );
        let (a, ca) = pre_standardized_input(near.as_ref(), wr);
        let (b, cb) = pre_standardized_input(far.as_ref(), wr);
        assert_eq!(cb.to_bits(), ca.to_bits());
        assert!(cb < 1.0);
        let b = b.expect("divided copy");
        assert_mat_bits_eq(
            b.as_ref(),
            a.expect("divided copy").as_ref(),
            "divided copy",
        );
        assert!((0..3).all(|j| b[(10, j)] == 0.0));
    }

    /// Every row-scaled writer writes a zero-weight row as 0, also where its
    /// standardized entry overflows: 1e300 on a column of scale near 3e-11.
    #[test]
    #[allow(clippy::float_cmp)]
    fn row_scaled_writers_write_a_zero_weight_row_as_zero() {
        let wn = normalize_weights(test_weights(12).as_ref()).unwrap();
        let sw = sqrt_col(wn.as_ref());
        let x = with_row10([1e300, 0.5, 0.5]);
        let x = Mat::<f64>::from_fn(12, 3, |i, j| {
            if j == 0 && i != 10 {
                1e-10 * x[(i, 0)]
            } else {
                x[(i, j)]
            }
        });
        let rows: Vec<usize> = (0..12).collect();
        let scale = held_out_scale(x.as_ref(), Some(wn.as_ref()), false);
        let (a, mean, _) = center_rows_fro(
            x.as_ref(),
            &rows,
            Some(wn.as_ref()),
            Some(sw.as_ref()),
            scale.as_ref(),
        );
        let c = standardize_apply_rows(
            x.as_ref(),
            &rows,
            mean.as_ref(),
            scale.as_ref(),
            Some(sw.as_ref()),
        );
        let mut f = fit_x_moments(x.as_ref(), Some(wn.as_ref()), Par::Seq);
        f.materialize(x.as_ref(), Some(sw.as_ref()));
        for (name, m) in [
            ("center_rows_fro", a.as_ref()),
            ("standardize_apply_rows", c.as_ref()),
            ("materialize", f.xs()),
        ] {
            assert!(crate::fit::check_finite_mat(m).is_ok(), "{name}");
            assert!((0..3).all(|j| m[(10, j)] == 0.0), "{name}");
        }
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
        let mut f = fit_x_moments(x, weights, Par::Seq);
        f.materialize(x, row_scale);
        f
    }

    /// `standardize1_weighted` on each column (the scalar path), the
    /// reference for the block driver.
    fn standardize_weighted_reference(
        x: MatRef<'_, f64>,
        weights: Option<ColRef<'_, f64>>,
    ) -> (Mat<f64>, Col<f64>, Col<f64>) {
        let cols: Vec<(Col<f64>, f64, f64)> = (0..x.ncols())
            .map(|j| standardize1_weighted(x.col(j), weights))
            .collect();
        let xs = Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| cols[j].0[i]);
        let mean = Col::<f64>::from_fn(x.ncols(), |j| cols[j].1);
        let scale = Col::<f64>::from_fn(x.ncols(), |j| cols[j].2);
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
    const SHAPES: [(usize, usize); 15] = [
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
        // One strip of rows wider than `ROW_BLOCK_MIN_COLS`, with a partial
        // block: a row-major view's moments read one block of `ROW_BLOCK`
        // rows and three single rows.
        (11, 2050),
        // Neither rows nor columns.
        (0, 0),
    ];

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

                    let sw = wref.map(|w| sqrt_col(renormalize_mean_one(w).as_ref()));
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

    /// Asserts that `f`'s mean, scale and `fro` have the bits of `mean`,
    /// `scale` and `fro`.
    fn assert_moments_bits(f: &FitX, mean: &Col<f64>, scale: &Col<f64>, fro: f64, what: &str) {
        assert_eq!(
            col_bits(f.mean.as_ref()),
            col_bits(mean.as_ref()),
            "{what}: mean"
        );
        assert_eq!(
            col_bits(f.scale.as_ref()),
            col_bits(scale.as_ref()),
            "{what}: scale"
        );
        assert_eq!(bits(f.fro), bits(fro), "{what}: fro");
    }

    /// Rayon pools of one and of four threads.
    fn pools() -> Vec<(usize, rayon::ThreadPool)> {
        [1_usize, 4]
            .into_iter()
            .map(|t| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("thread pool");
                (t, pool)
            })
            .collect()
    }

    /// `fit_x_moments` reads X in strips of columns, on one thread or in
    /// parallel. On shapes with three strips of `stream_block_width(n)`
    /// columns or more (the last one partial; strips narrower and wider than
    /// `ROW_BLOCK_MIN_COLS`), every layout, weighted or not, under `Par::Seq`
    /// and `fit::par_fixed()`, and on tall row-major shapes, whose strips
    /// are wider (one strip under `Par::Seq`, several with a partial last
    /// one under `fit::par_fixed()` in a pool of four threads), in pools of
    /// one and four threads, the mean, scale and `fro` are those of one
    /// full-width `standardize_columns` on the owned column-major matrix,
    /// bit for bit.
    #[test]
    fn fit_x_moments_in_strips_are_the_full_width_moments() {
        let shapes = [(40_usize, 13_200_usize), (1000, 700), (203, 3000)];
        assert!(
            shapes
                .iter()
                .any(|&(n, _)| stream_block_width(n) >= ROW_BLOCK_MIN_COLS),
            "no shape has strips at least ROW_BLOCK_MIN_COLS wide"
        );
        for (k, &(n, p)) in shapes.iter().enumerate() {
            assert!(
                p > 2 * stream_block_width(n),
                "{n}x{p}: fewer than three strips"
            );
            assert!(
                p % stream_block_width(n) != 0,
                "{n}x{p}: the last strip is not partial"
            );
            let x = test_matrix(n, p, 950 + k as u64);
            let lay = Layouts::new(x.as_ref());
            // Neither stride is one: the column body's strided arm.
            let rows_rev = x.as_ref().reverse_rows().to_owned();
            let strided = rows_rev.as_ref().reverse_rows();
            assert!(strided.try_as_col_major().is_none() && strided.try_as_row_major().is_none());
            let weights = test_weight_sets(n);
            let mut cases: Vec<(&str, Option<ColRef<'_, f64>>)> = vec![("unweighted", None)];
            cases.extend(weights.iter().map(|(l, w)| (*l, Some(w.as_ref()))));
            for (label, wref) in cases {
                let w_prime = wref.map(renormalize_mean_one);
                let (_, mean, scale, fro) = standardize_columns(
                    x.as_ref(),
                    None,
                    w_prime.as_ref().map(owned_col_slice),
                    None,
                    false,
                );
                for (view, xv) in lay.all(&x).into_iter().chain([("strided", strided)]) {
                    for par in [Par::Seq, crate::fit::par_fixed()] {
                        let what = format!("{n}x{p} {view} {label} {par:?}");
                        let f = fit_x_moments(xv, wref, par);
                        assert_moments_bits(&f, &mean, &scale, fro, &what);
                    }
                }
            }
        }

        let pools = pools();
        for (k, &(n, p)) in [(20_000_usize, 200_usize), (5000, 900)].iter().enumerate() {
            assert!(
                stream_block_width(n) < ROW_MAJOR_MIN_STRIP,
                "{n}x{p}: row-major strips of stream_block_width(n)"
            );
            let x = test_matrix(n, p, 970 + k as u64);
            let x_t = x.transpose().to_owned();
            let xv = x_t.as_ref().transpose();
            assert!(xv.try_as_col_major().is_none() && xv.try_as_row_major().is_some());
            let weights = test_weight_sets(n);
            let mut cases: Vec<(&str, Option<ColRef<'_, f64>>)> = vec![("unweighted", None)];
            cases.extend(weights.iter().map(|(l, w)| (*l, Some(w.as_ref()))));
            for (label, wref) in cases {
                let w_prime = wref.map(renormalize_mean_one);
                let (_, mean, scale, fro) = standardize_columns(
                    x.as_ref(),
                    None,
                    w_prime.as_ref().map(owned_col_slice),
                    None,
                    false,
                );
                for (threads, pool) in &pools {
                    pool.install(|| {
                        for par in [Par::Seq, crate::fit::par_fixed()] {
                            let width = row_major_strip_width(n, p, par);
                            let what = format!(
                                "{n}x{p} row-major {label} {par:?} pool {threads}: strips of {width}"
                            );
                            if *threads == 4 && matches!(par, Par::Rayon(_)) {
                                assert!(
                                    width < p && p % width != 0,
                                    "{what}: not several strips with a partial last one"
                                );
                            } else {
                                assert_eq!(width, p, "{what}: not one strip");
                            }
                            let f = fit_x_moments(xv, wref, par);
                            assert_moments_bits(&f, &mean, &scale, fro, &what);
                        }
                    });
                }
            }
        }
    }

    /// `row_major_t_mul`'s parallel arm, which maps chunks of
    /// `t_mul_chunk_width(d)` columns, has the bits of `PAR_DEGREE` row
    /// pieces each formed over the full width from `0.0` in row order and
    /// added in piece order (a plain loop below), on row-major shapes with
    /// several chunks and a partial last one (chunks narrower than
    /// `T_MUL_CHUNK` and chunks of exactly `T_MUL_CHUNK`), in pools of one
    /// and four threads.
    #[test]
    fn row_major_t_mul_in_column_chunks_is_the_full_width_pieces() {
        fn full_width_pieces(x: MatRef<'_, f64>, t: ColRef<'_, f64>) -> Vec<u64> {
            let (n, d) = (x.nrows(), x.ncols());
            let p = crate::fit::PAR_DEGREE;
            let piece = |k: usize| {
                let mut out = vec![0.0_f64; d];
                for i in k * n / p..(k + 1) * n / p {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o += t[i] * x[(i, j)];
                    }
                }
                out
            };
            let mut out = piece(0);
            for k in 1..p {
                for (o, v) in out.iter_mut().zip(&piece(k)) {
                    *o += v;
                }
            }
            out.into_iter().map(bits).collect()
        }

        let shapes = [
            (203_usize, 9000_usize),
            (37, 4097),
            (203, 2100),
            (11, 40_000),
        ];
        assert!(
            shapes
                .iter()
                .any(|&(_, d)| t_mul_chunk_width(d) == T_MUL_CHUNK)
        );
        assert!(
            shapes
                .iter()
                .any(|&(_, d)| t_mul_chunk_width(d) < T_MUL_CHUNK && d < T_MUL_CHUNK)
        );
        let pools = pools();
        for (k, &(n, d)) in shapes.iter().enumerate() {
            let chunk = t_mul_chunk_width(d);
            assert!(
                d > chunk && d % chunk != 0,
                "{n}x{d}: chunks of {chunk}, not several with a partial last one"
            );
            let x = test_matrix(n, d, 990 + k as u64);
            let x_t = x.transpose().to_owned();
            let xv = x_t.as_ref().transpose();
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(995 + k as u64);
            let t = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
            let want = full_width_pieces(xv, t.as_ref());
            // The one-chain sum of `Par::Seq` differs in some column: the
            // comparison tells the two arithmetics apart.
            let seq = row_major_t_mul(xv, t.as_ref(), Par::Seq).expect("row-major and wide");
            assert_ne!(col_bits(seq.as_ref()), want, "{n}x{d}: Seq");
            for (threads, pool) in &pools {
                let got = pool.install(|| {
                    row_major_t_mul(xv, t.as_ref(), crate::fit::par_fixed())
                        .expect("row-major and wide")
                });
                assert_eq!(col_bits(got.as_ref()), want, "{n}x{d} pool {threads}");
            }
        }
    }

    /// `row_major_strip_width` and `t_mul_chunk_width` at their edges:
    /// `stream_block_width(n)` up to 1,024 rows, no columns, fewer columns
    /// than threads, the cap at `d`, and the `STD_BLOCK` floor and
    /// `T_MUL_CHUNK` cap of the chunks (narrow rows the product itself
    /// declines included).
    #[test]
    fn strip_and_chunk_widths_at_their_edges() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("thread pool");
        let par = crate::fit::par_fixed();
        assert_eq!(stream_block_width(1024), ROW_MAJOR_MIN_STRIP);
        assert_eq!(row_major_strip_width(1024, 9000, Par::Seq), 256);
        assert_eq!(pool.install(|| row_major_strip_width(1024, 9000, par)), 256);
        assert_eq!(stream_block_width(1025), 248);
        assert_eq!(row_major_strip_width(1025, 0, Par::Seq), 1);
        assert_eq!(pool.install(|| row_major_strip_width(1025, 0, par)), 1);
        // Fewer columns than threads, and the cap at `d`.
        assert_eq!(pool.install(|| row_major_strip_width(2000, 3, par)), 3);
        assert_eq!(row_major_strip_width(20_000, 100, Par::Seq), 100);
        assert_eq!(pool.install(|| row_major_strip_width(20_000, 100, par)), 32);
        assert_eq!(pool.install(|| row_major_strip_width(2000, 600, par)), 152);
        assert_eq!(row_major_strip_width(2000, 600, Par::Seq), 600);

        for (d, chunk) in [
            (0, 8),
            (11, 8),
            (40, 8),
            (64, 8),
            (65, 16),
            (2048, 256),
            (2100, 264),
            (32_768, 4096),
            (40_000, 4096),
        ] {
            assert_eq!(t_mul_chunk_width(d), chunk, "d = {d}");
        }
    }

    /// A non-finite entry makes its column's mean non-finite on every
    /// layout, weighted or not, a zero weight on its row included (`0·inf`
    /// and `0·NaN` are NaN). `pls1_fit` reads X's finiteness from this.
    #[test]
    fn standardize_fit_x_flags_non_finite_columns() {
        // Row-major views take the blocked row-major moments; the second
        // shape is one strip of rows wider than `ROW_BLOCK_MIN_COLS`, with a
        // partial block.
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
                let ss = sum_of_squares(xv, Par::Seq);
                assert!(
                    (ss.sqrt() - norm).abs() <= 1e-13 * norm.max(1.0),
                    "{n}x{p} {view}: {} vs {norm}",
                    ss.sqrt()
                );
            }
        }
        let mut x = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
        x[(2, 1)] = f64::NAN;
        assert!(!sum_of_squares(x.as_ref(), Par::Seq).is_finite());
        let big = Mat::<f64>::from_fn(4, 3, |_, _| 1e200);
        assert!(!sum_of_squares(big.as_ref(), Par::Seq).is_finite());
    }

    /// `sum_of_squares` has the same bits under `Par::Seq` and
    /// `fit::par_fixed()` on every layout, also with fewer storage lines
    /// than pieces.
    #[test]
    fn sum_of_squares_is_the_same_on_both_arms() {
        for (k, &(n, p)) in SHAPES.iter().enumerate() {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1300 + k as u64);
            let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
            let lay = Layouts::new(x.as_ref());
            // Neither stride is one (from two rows on): the entry-by-entry arm.
            let rows_rev = x.as_ref().reverse_rows().to_owned();
            let strided = rows_rev.as_ref().reverse_rows();
            for (view, xv) in lay.all(&x).into_iter().chain([("strided", strided)]) {
                let seq = sum_of_squares(xv, Par::Seq);
                let par = sum_of_squares(xv, crate::fit::par_fixed());
                assert_eq!(
                    seq.to_bits(),
                    par.to_bits(),
                    "{n}x{p} {view}: {seq:e} vs {par:e}"
                );
            }
        }
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
}
