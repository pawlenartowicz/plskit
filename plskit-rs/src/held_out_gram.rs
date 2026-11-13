//! The all-row Gram matrix `G = X̃X̃'` (`n × n`) of the held-out tests, and
//! the per-split kernels taken from its blocks, and [`gram_matrix`], which
//! returns that matrix to the caller.
//!
//! `X̃` is X centred over all rows (weighted means when weighted) and
//! divided by the all-row column scale (`linalg::held_out_scale`). Centring
//! a part by its own means cancels any common shift of the rows, so the
//! training kernel `C_A G_AA C_A` and the cross-kernel `G_BA C_A`, shifted
//! by the training means, equal the products of the halves centred by the
//! training half's means in exact arithmetic. Centring over all rows first
//! keeps that cancellation from losing digits on an uncentred X.

use faer::linalg::matmul::triangular::{BlockStructure, matmul};
use faer::{Accum, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};

/// `G = X̃X̃'` (`n × n`, exactly symmetric) and the column count `p` of the
/// `X̃` it was built from.
#[derive(Debug, Clone)]
pub(crate) struct HeldOutGram {
    g: Mat<f64>,
    n_features: usize,
}

#[cfg(test)]
thread_local! {
    static GRAMS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl HeldOutGram {
    /// `G` of `X̃`: `x` centred over all rows (the weighted means of
    /// `w_norm` when given) and divided by `scale`, with no row weights.
    /// `X̃` streams in `stream_block_width(n)` column blocks
    /// (`linalg::center_rows_fro`, as `linalg::centered_blocks` forms them),
    /// each adding its product to the lower triangle with
    /// `fit::par_fixed()`, which is mirrored once at the end: `n²p/2`
    /// multiply-adds, and the bits do not depend on the pool size.
    ///
    /// # Panics
    /// When `scale` does not have one entry per column of `x`.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn from_x(
        x: MatRef<'_, f64>,
        w_norm: Option<ColRef<'_, f64>>,
        scale: ColRef<'_, f64>,
    ) -> Self {
        #[cfg(test)]
        GRAMS_BUILT.with(|c| c.set(c.get() + 1));
        let (n, p) = (x.nrows(), x.ncols());
        assert_eq!(
            scale.nrows(),
            p,
            "HeldOutGram::from_x: scale has {} entries, x has {p} columns",
            scale.nrows()
        );
        let all: Vec<usize> = (0..n).collect();
        let mut g = Mat::<f64>::zeros(n, n);
        let width = crate::linalg::stream_block_width(n);
        let mut j0 = 0;
        while j0 < p {
            let w = width.min(p - j0);
            let (a, _, _) = crate::linalg::center_rows_fro(
                x.subcols(j0, w),
                &all,
                w_norm,
                None,
                scale.subrows(j0, w),
            );
            matmul(
                g.as_mut(),
                BlockStructure::TriangularLower,
                Accum::Add,
                a.as_ref(),
                BlockStructure::Rectangular,
                a.transpose(),
                BlockStructure::Rectangular,
                1.0,
                crate::fit::par_fixed(),
            );
            j0 += w;
        }
        mirror_lower(&mut g);
        Self { g, n_features: p }
    }

    /// `D·G·D` with `D = diag(√w_norm)`: the Gram matrix of `X̃` with the √w
    /// row scaling of `pls1_fit`. A row or column of weight 0 is written as
    /// 0, so an overflowed entry of a zero-weight row gives no `inf·0 = NaN`.
    pub(crate) fn with_row_weights(&self, w_norm: ColRef<'_, f64>) -> Self {
        let sw = crate::linalg::sqrt_col(w_norm);
        let n = self.g.nrows();
        Self {
            g: Mat::<f64>::from_fn(n, n, |i, j| {
                if sw[i] == 0.0 || sw[j] == 0.0 {
                    0.0
                } else {
                    self.g[(i, j)] * sw[i] * sw[j]
                }
            }),
            n_features: self.n_features,
        }
    }

    /// A caller's `g = XX'` of a block with `n_features` columns, centred
    /// over all rows (`C·g·C`, `C = I − 11'/n`, in `O(n²)`), so `g` need not
    /// be centred: a row subset `G[S, S]` of a centred `G` is accepted. The
    /// centred matrix must pass `perm_alpha::check_gram_entries` (diagonal,
    /// symmetry, Cauchy-Schwarz); its lower triangle is kept and mirrored.
    ///
    /// # Errors
    /// `ShapeMismatch` when `g` is not square; `NonFiniteInput` when `g` is
    /// not finite; `InvalidInput` when the centred matrix fails a check.
    pub(crate) fn from_gram(mut g: Mat<f64>, n_features: usize) -> PlsKitResult<Self> {
        if g.nrows() != g.ncols() {
            return Err(PlsKitError::ShapeMismatch(format!(
                "gram must be square, got {} x {}",
                g.nrows(),
                g.ncols()
            )));
        }
        crate::fit::check_finite_mat(g.as_ref())?;
        #[cfg(test)]
        GRAMS_BUILT.with(|c| c.set(c.get() + 1));
        let n = g.nrows();
        let nf = n as f64;
        let row: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| g[(i, j)]).sum::<f64>() / nf)
            .collect();
        let col: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| g[(i, j)]).sum::<f64>() / nf)
            .collect();
        let all = col.iter().sum::<f64>() / nf;
        for j in 0..n {
            for i in 0..n {
                g[(i, j)] = g[(i, j)] - row[i] - col[j] + all;
            }
        }
        crate::perm_alpha::check_gram_entries(g.as_ref(), n_features)?;
        mirror_lower(&mut g);
        Ok(Self { g, n_features })
    }

    /// `G`.
    pub(crate) fn g(&self) -> MatRef<'_, f64> {
        self.g.as_ref()
    }

    /// The column count of the `X̃` behind `G`.
    pub(crate) fn n_features(&self) -> usize {
        self.n_features
    }

    /// `linalg::stable_rank` of `X̃`, from `G`.
    pub(crate) fn stable_rank(&self) -> f64 {
        crate::linalg::stable_rank_from_gram(self.g.as_ref())
    }

    /// For training rows `tr` (A) and test rows `te` (B), with `u` the
    /// roots `√w` of the row weights over all rows (ones when `sw` is
    /// `None`) and `P_A = I − u_A u_A'/(u_A'u_A)`: the training kernel
    /// `K_A = P_A G_AA P_A` (`n_A × n_A`) and the cross-kernel
    /// `M = (G_BA − u_B u_A'G_AA/(u_A'u_A))·P_A` (`n_B × n_A`). Unweighted,
    /// these are `X̃_A X̃_A'` and `X̃_B X̃_A'` with both halves centred by the
    /// training half's means, in exact arithmetic; on the `G` of
    /// `with_row_weights(w)` with `sw = √w`, the same with weighted
    /// training means and the √w row scaling. With `r = G_AA u_A/(u_A'u_A)`,
    /// `r̄ = u_A'r/(u_A'u_A)` and `s = G_BA u_A/(u_A'u_A)`:
    /// `K_A[i, j] = G_AA[i, j] − r_i u_j − u_i r_j + u_i u_j r̄` and
    /// `M[i, j] = G_BA[i, j] − s_i u_j − u_i (r_j − r̄ u_j)`.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn part_blocks(
        &self,
        tr: &[usize],
        te: &[usize],
        sw: Option<ColRef<'_, f64>>,
    ) -> (Mat<f64>, Mat<f64>) {
        let g = self.g.as_ref();
        let u = |i: usize| sw.map_or(1.0, |s| s[i]);
        let uu = tr.iter().map(|&a| u(a) * u(a)).sum::<f64>();
        let r: Vec<f64> = tr
            .iter()
            .map(|&aj| tr.iter().map(|&ai| g[(ai, aj)] * u(ai)).sum::<f64>() / uu)
            .collect();
        let r_bar = tr.iter().zip(&r).map(|(&a, r)| u(a) * r).sum::<f64>() / uu;
        // Column `bi` of `G_AB`: the bits of row `bi` of `G_BA`, by exact symmetry.
        let s: Vec<f64> = te
            .iter()
            .map(|&bi| tr.iter().map(|&aj| g[(aj, bi)] * u(aj)).sum::<f64>() / uu)
            .collect();
        let k_a = Mat::<f64>::from_fn(tr.len(), tr.len(), |i, j| {
            let (ui, uj) = (u(tr[i]), u(tr[j]));
            g[(tr[i], tr[j])] - r[i] * uj - ui * r[j] + ui * uj * r_bar
        });
        let m = Mat::<f64>::from_fn(te.len(), tr.len(), |i, j| {
            let uj = u(tr[j]);
            g[(te[i], tr[j])] - s[i] * uj - u(te[i]) * (r[j] - r_bar * uj)
        });
        (k_a, m)
    }
}

/// Output of [`gram_matrix`].
#[derive(Debug, Clone)]
pub struct GramMatrixOutput {
    /// `X̃X̃'`, shape `(n, n)`, exactly symmetric. `X̃` is `X` centred over
    /// all rows and divided by its column standard deviations (population),
    /// or only centred when `pre_standardized` (and then, when X's largest
    /// entry is outside `[2^-256, 2^256]`, divided by a power of two `c`
    /// that is not stored, so `gram` is the Gram matrix of `X/c`).
    pub gram: Mat<f64>,
    /// The column count `p` of `X`, constant columns included.
    pub n_features: usize,
    /// Whether `X` was read as given (only centred).
    pub pre_standardized: bool,
}

impl GramMatrixOutput {
    /// `gram[rows, rows]`, in the order given (a repeated row is repeated),
    /// with the same `n_features` and `pre_standardized`: the Gram matrix of
    /// the rows `rows` of `X̃`. The rows keep the all-row centring and
    /// scale; they are not re-centred.
    ///
    /// # Errors
    /// [`PlsKitError::InvalidArgument`] when a row is not below `n`, or
    /// `rows` has fewer than 2 entries.
    pub fn subset(&self, rows: &[usize]) -> PlsKitResult<Self> {
        let n = self.gram.nrows();
        if rows.len() < 2 {
            return Err(PlsKitError::InvalidArgument(format!(
                "rows must hold at least 2 rows, got {}",
                rows.len()
            )));
        }
        if let Some(&r) = rows.iter().find(|&&r| r >= n) {
            return Err(PlsKitError::InvalidArgument(format!(
                "row {r} is out of range for a Gram matrix of {n} rows"
            )));
        }
        Ok(Self {
            gram: Mat::<f64>::from_fn(rows.len(), rows.len(), |i, j| self.gram[(rows[i], rows[j])]),
            n_features: self.n_features,
            pre_standardized: self.pre_standardized,
        })
    }
}

/// The all-row Gram matrix `X̃X̃'` the held-out tests compute from `X`, bit
/// for bit: `X` centred over all rows and divided by its column standard
/// deviations (population), or only centred when `pre_standardized`
/// (and divided by one power of two `c` near its largest entry when that
/// entry is outside `[2^-256, 2^256]`, see `linalg::held_out_scale`; `c` is
/// not returned, so `pls1_perm_alpha_gram`'s `alpha` and the `score`
/// statistic from this matrix are those of `X/c`, while p-values are not
/// affected). Computing it once lets a caller reuse it across calls on the same `X`.
///
/// # Errors
/// [`PlsKitError::InvalidArgument`] when `x` has fewer than 2 rows;
/// [`PlsKitError::InvalidInput`] when `x` has no columns;
/// [`PlsKitError::NonFiniteInput`] when `x` holds NaN or infinity.
pub fn gram_matrix(x: MatRef<'_, f64>, pre_standardized: bool) -> PlsKitResult<GramMatrixOutput> {
    if x.nrows() < 2 {
        return Err(PlsKitError::InvalidArgument(format!(
            "X must have at least 2 rows, got {}",
            x.nrows()
        )));
    }
    if x.ncols() == 0 {
        return Err(PlsKitError::InvalidInput(
            "X must have at least 1 column".into(),
        ));
    }
    crate::fit::check_finite_mat(x)?;
    crate::fit::with_thread_limit(|| {
        let scale = crate::linalg::held_out_scale(x, None, pre_standardized);
        let g = HeldOutGram::from_x(x, None, scale.as_ref());
        Ok(GramMatrixOutput {
            gram: g.g,
            n_features: g.n_features,
            pre_standardized,
        })
    })
}

/// Copies the strict lower triangle of the square `g` onto its upper one.
fn mirror_lower(g: &mut Mat<f64>) {
    for j in 0..g.ncols() {
        for i in (j + 1)..g.nrows() {
            g[(j, i)] = g[(i, j)];
        }
    }
}

/// How many [`HeldOutGram`]s this thread has built with `from_x` or
/// `from_gram` (test builds). With `PLSKIT_NUM_THREADS` set to a cap, a
/// public call runs on a plskit pool worker, which this thread-local does
/// not reach, so this panics when the variable sets a cap.
#[cfg(test)]
pub(crate) fn grams_built() -> usize {
    assert!(
        std::env::var_os(crate::fit::NUM_THREADS_ENV).is_none_or(|v| v == "0"),
        "unset PLSKIT_NUM_THREADS or set it to 0: the Gram counter does not reach plskit's pool"
    );
    GRAMS_BUILT.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{held_out_scale, mat_mul, normalize_weights};
    use crate::signal_test::{SplitIdx, prepare_split};
    use crate::test_support::{assert_mat_bits_eq, signal_data, test_weights};
    use faer::{Col, Par};

    /// `n × p` uniform X with column `j` scaled by `1 + j/10` and shifted by
    /// `1 + j/20`: uncentred, with unequal column scales.
    #[allow(clippy::cast_precision_loss)]
    fn raw_x(n: usize, p: usize, seed: u64) -> Mat<f64> {
        let (x, _) = signal_data(n, p, seed);
        Mat::<f64>::from_fn(n, p, |i, j| {
            x[(i, j)] * (1.0 + j as f64 / 10.0) + 1.0 + j as f64 / 20.0
        })
    }

    /// Every entry of `got` within `tol·max|want|` of `want`.
    fn assert_close(got: MatRef<'_, f64>, want: MatRef<'_, f64>, tol: f64, what: &str) {
        assert_eq!(
            (got.nrows(), got.ncols()),
            (want.nrows(), want.ncols()),
            "{what}: shape"
        );
        let mut big = 0.0_f64;
        for j in 0..want.ncols() {
            for i in 0..want.nrows() {
                big = big.max(want[(i, j)].abs());
            }
        }
        for j in 0..want.ncols() {
            for i in 0..want.nrows() {
                let (a, b) = (got[(i, j)], want[(i, j)]);
                assert!(
                    (a - b).abs() <= tol * big,
                    "{what}[({i}, {j})]: {a:e} vs {b:e} (scale {big:e})"
                );
            }
        }
    }

    /// `x` centred by the `w`-weighted column means (plain means when
    /// `None`) and divided by `scale`, by hand.
    #[allow(clippy::cast_precision_loss)]
    fn centred_scaled(x: &Mat<f64>, w: Option<&Col<f64>>, scale: &Col<f64>) -> Mat<f64> {
        let n = x.nrows();
        let wt = |i: usize| w.map_or(1.0, |w| w[i]);
        let sw: f64 = (0..n).map(wt).sum();
        let mean: Vec<f64> = (0..x.ncols())
            .map(|j| (0..n).map(|i| wt(i) * x[(i, j)]).sum::<f64>() / sw)
            .collect();
        Mat::<f64>::from_fn(n, x.ncols(), |i, j| (x[(i, j)] - mean[j]) / scale[j])
    }

    /// `a·a'` by hand.
    fn outer(a: &Mat<f64>) -> Mat<f64> {
        Mat::<f64>::from_fn(a.nrows(), a.nrows(), |i, j| {
            (0..a.ncols()).map(|l| a[(i, l)] * a[(j, l)]).sum()
        })
    }

    /// `from_x` equals `X̃X̃'` by hand, with X centred over all rows
    /// (weighted means when weighted) and divided by the all-row scale, or
    /// by ones as under `pre_standardized`. `p` spans two stream blocks, the
    /// second partial.
    #[test]
    fn gram_from_x_is_the_product_of_the_centred_scaled_x() {
        let (n, p) = (256, 1100);
        assert!(crate::linalg::stream_block_width(n) < p);
        let x = raw_x(n, p, 11);
        let wn = normalize_weights(test_weights(n).as_ref()).unwrap();
        for w in [None, Some(&wn)] {
            for flag in [false, true] {
                let what = format!("weighted={}, pre_standardized={flag}", w.is_some());
                let wr = w.map(Col::as_ref);
                let scale = held_out_scale(x.as_ref(), wr, flag);
                let hg = HeldOutGram::from_x(x.as_ref(), wr, scale.as_ref());
                let xt = centred_scaled(&x, w, &scale);
                assert_close(hg.g(), outer(&xt).as_ref(), 1e-13, &what);
                assert_eq!(hg.n_features(), p, "{what}");
                approx::assert_relative_eq!(
                    hg.stable_rank(),
                    crate::linalg::stable_rank(xt.as_ref()),
                    max_relative = 1e-12
                );
            }
        }
    }

    /// Column 0 at `f64::MAX, f64::MAX, -f64::MAX` overflows `x − mean`;
    /// its block is centred from its quarter, so `G` is finite and has the
    /// bits of `G` of `x/8`, where nothing overflows.
    #[test]
    fn gram_from_x_centres_entries_near_max_without_overflow() {
        let mut x = Mat::<f64>::from_fn(12, 5, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
        x[(0, 0)] = f64::MAX;
        x[(1, 0)] = f64::MAX;
        x[(2, 0)] = -f64::MAX;
        let x8 = Mat::<f64>::from_fn(12, 5, |i, j| x[(i, j)] / 8.0);
        let g = HeldOutGram::from_x(
            x.as_ref(),
            None,
            held_out_scale(x.as_ref(), None, false).as_ref(),
        );
        let g8 = HeldOutGram::from_x(
            x8.as_ref(),
            None,
            held_out_scale(x8.as_ref(), None, false).as_ref(),
        );
        assert!(g.g().is_all_finite(), "{:?}", g.g());
        for i in 0..12 {
            for j in 0..12 {
                assert_eq!(
                    g.g()[(i, j)].to_bits(),
                    g8.g()[(i, j)].to_bits(),
                    "({i}, {j})"
                );
            }
        }
    }

    /// `G` is exactly symmetric, its bits are the same in pools of 1 and 4
    /// threads, and each `from_x` counts once.
    #[test]
    fn gram_is_symmetric_and_thread_count_invariant() {
        let (n, p) = (512, 1100);
        let x = raw_x(n, p, 12);
        let scale = held_out_scale(x.as_ref(), None, false);
        let before = grams_built();
        let g = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
        assert_eq!(grams_built(), before + 1);
        for j in 0..n {
            for i in (j + 1)..n {
                assert_eq!(
                    g.g()[(i, j)].to_bits(),
                    g.g()[(j, i)].to_bits(),
                    "({i}, {j})"
                );
            }
        }
        for threads in [1_usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let gt = pool.install(|| HeldOutGram::from_x(x.as_ref(), None, scale.as_ref()));
            assert_mat_bits_eq(gt.g(), g.g(), &format!("{threads} threads"));
        }
    }

    /// `K_A` and `M` equal the products of `prepare_split`'s halves (test
    /// half centred by the training means, all-row scale), on unequal
    /// interleaved halves, from a `G` centred with plain or weighted means.
    #[test]
    fn part_blocks_equal_products_of_centred_halves() {
        let (n, p) = (45, 70);
        let x = raw_x(n, p, 13);
        let sp = SplitIdx {
            tr: (0..n).filter(|i| i % 3 != 1).collect(),
            te: (0..n).filter(|i| i % 3 == 1).collect(),
        };
        let scale = held_out_scale(x.as_ref(), None, false);
        let prep = prepare_split(x.as_ref(), &sp, None, scale.as_ref());
        let k_ref = mat_mul(prep.xs_tr.as_ref(), prep.xs_tr.transpose(), Par::Seq);
        let m_ref = mat_mul(prep.xs_te.as_ref(), prep.xs_tr.transpose(), Par::Seq);
        let wn = normalize_weights(test_weights(n).as_ref()).unwrap();
        for w in [None, Some(wn.as_ref())] {
            let what = format!("G weighted={}", w.is_some());
            let hg = HeldOutGram::from_x(x.as_ref(), w, scale.as_ref());
            let (k_a, m) = hg.part_blocks(&sp.tr, &sp.te, None);
            assert_close(k_a.as_ref(), k_ref.as_ref(), 1e-12, &format!("K_A, {what}"));
            assert_close(m.as_ref(), m_ref.as_ref(), 1e-12, &format!("M, {what}"));
        }
    }

    /// `from_gram` centres an uncentred `G`: the `G` of the scaled,
    /// uncentred X gives `from_x`'s `G` and blocks, and a row subset
    /// `G[S, S]` passes and gives `from_x` on `X[S]`. An asymmetric `G` is
    /// refused.
    #[test]
    fn from_gram_centres_the_given_gram() {
        let (n, p) = (40, 60);
        let x = raw_x(n, p, 14);
        let scale = held_out_scale(x.as_ref(), None, false);
        let xs = Mat::<f64>::from_fn(n, p, |i, j| x[(i, j)] / scale[j]);
        let g_raw = mat_mul(xs.as_ref(), xs.transpose(), Par::Seq);
        let want = HeldOutGram::from_x(x.as_ref(), None, scale.as_ref());
        let got = HeldOutGram::from_gram(g_raw.clone(), p).unwrap();
        assert_eq!(got.n_features(), p);
        assert_close(got.g(), want.g(), 1e-12, "G");
        let tr: Vec<usize> = (0..n).step_by(2).collect();
        let te: Vec<usize> = (1..n).step_by(2).collect();
        let (k_got, m_got) = got.part_blocks(&tr, &te, None);
        let (k_want, m_want) = want.part_blocks(&tr, &te, None);
        assert_close(k_got.as_ref(), k_want.as_ref(), 1e-12, "K_A");
        assert_close(m_got.as_ref(), m_want.as_ref(), 1e-12, "M");

        let rows: Vec<usize> = (0..n).filter(|i| i % 4 != 3).collect();
        let g_sub =
            Mat::<f64>::from_fn(rows.len(), rows.len(), |i, j| want.g()[(rows[i], rows[j])]);
        let x_sub = Mat::<f64>::from_fn(rows.len(), p, |i, j| x[(rows[i], j)]);
        let sub = HeldOutGram::from_gram(g_sub, p).unwrap();
        let sub_want = HeldOutGram::from_x(x_sub.as_ref(), None, scale.as_ref());
        assert_close(sub.g(), sub_want.g(), 1e-12, "G[S, S]");

        let mut bad = g_raw.clone();
        bad[(1, 0)] += 1.0;
        let err = HeldOutGram::from_gram(bad, p).expect_err("asymmetric");
        assert!(err.to_string().contains("not symmetric"), "{err}");
    }

    /// `with_row_weights` equals `diag(√w)·X̃X̃'·diag(√w)` by hand, with X
    /// centred by the weighted means.
    #[test]
    fn with_row_weights_is_the_weighted_product() {
        let (n, p) = (60, 90);
        let x = raw_x(n, p, 15);
        let wn = normalize_weights(test_weights(n).as_ref()).unwrap();
        let scale = held_out_scale(x.as_ref(), Some(wn.as_ref()), false);
        let hg = HeldOutGram::from_x(x.as_ref(), Some(wn.as_ref()), scale.as_ref());
        let got = hg.with_row_weights(wn.as_ref());
        let xt = centred_scaled(&x, Some(&wn), &scale);
        let xw = Mat::<f64>::from_fn(n, p, |i, j| xt[(i, j)] * wn[i].sqrt());
        assert_close(got.g(), outer(&xw).as_ref(), 1e-13, "D G D");
        assert_eq!(got.n_features(), p);
    }

    #[test]
    fn gram_matrix_rejects_non_finite_x() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut x = raw_x(10, 5, 17);
            x[(3, 2)] = bad;
            let err = gram_matrix(x.as_ref(), false).expect_err("non-finite X");
            assert!(matches!(err, PlsKitError::NonFiniteInput), "{bad}: {err}");
        }
    }

    #[test]
    fn gram_matrix_rejects_too_few_rows_or_no_columns() {
        for n in [0, 1] {
            let x = Mat::<f64>::from_fn(n, 3, |_, j| j as f64);
            let err = gram_matrix(x.as_ref(), false).expect_err("n < 2");
            assert!(
                matches!(err, PlsKitError::InvalidArgument(_)),
                "n = {n}: {err}"
            );
        }
        let err = gram_matrix(Mat::<f64>::zeros(5, 0).as_ref(), false).expect_err("p = 0");
        assert!(matches!(err, PlsKitError::InvalidInput(_)), "p = 0: {err}");
    }

    /// Under the flag, X far from unit scale (`1e200·X`, `1e-200·X`)
    /// gives the Gram matrix of `X·2^-e`, with `2^e` near its largest entry:
    /// finite, and in bits that of the same X scaled to its largest entry
    /// in `[1, 2)`.
    #[test]
    fn gram_matrix_divides_a_far_block_by_a_power_of_two() {
        let x = raw_x(6, 10, 20);
        for f in [1e200, 1e-200] {
            let far = Mat::<f64>::from_fn(6, 10, |i, j| x[(i, j)] * f);
            let s = crate::linalg::pow2_scale(far.norm_max()).0;
            let unit = Mat::<f64>::from_fn(6, 10, |i, j| far[(i, j)] / s);
            let got = gram_matrix(far.as_ref(), true).unwrap().gram;
            let want = gram_matrix(unit.as_ref(), true).unwrap().gram;
            assert!(got.is_all_finite(), "{f}");
            assert_mat_bits_eq(got.as_ref(), want.as_ref(), "far block");
        }
    }

    /// `subset(rows)` is `gram[rows, rows]` in the order given, repeats
    /// included, with `n_features` and `pre_standardized` kept.
    #[test]
    fn subset_keeps_the_record() {
        let x = raw_x(12, 7, 18);
        let full = gram_matrix(x.as_ref(), true).unwrap();
        let rows = [9, 0, 4, 4, 11];
        let sub = full.subset(&rows).unwrap();
        let want = Mat::<f64>::from_fn(5, 5, |i, j| full.gram[(rows[i], rows[j])]);
        assert_mat_bits_eq(sub.gram.as_ref(), want.as_ref(), "gram[rows, rows]");
        assert_eq!((sub.n_features, sub.pre_standardized), (7, true));
    }

    #[test]
    fn subset_rejects_bad_rows() {
        let full = gram_matrix(raw_x(12, 7, 19).as_ref(), false).unwrap();
        for rows in [&[0, 12][..], &[3, usize::MAX], &[5], &[]] {
            let err = full.subset(rows).expect_err("bad rows");
            assert!(
                matches!(err, PlsKitError::InvalidArgument(_)),
                "{rows:?}: {err}"
            );
        }
    }
}
