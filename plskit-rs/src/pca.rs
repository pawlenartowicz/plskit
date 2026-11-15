//! PCA basis of standardized `X`: the right singular vectors with nonzero
//! singular value, in component order. Used by
//! [`crate::factors::factors`] as the basis it splits a coefficient vector on.

use faer::{Col, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::linalg::{scaled_moments, standardize, tall_svd_v};

/// Output of [`pca_basis`].
#[derive(Debug)]
pub struct PcaBasisOutput {
    /// Right singular vectors, shape `(D, K)`, orthonormal columns. Column
    /// signs are arbitrary.
    pub v: Mat<f64>,
    /// Singular values, length `K`, non-increasing and positive.
    pub s: Col<f64>,
}

/// Right singular vectors and singular values of standardized `X`, keeping
/// `s > 1e-10 s[0]`, so `K = rank(Xs)`.
///
/// `Xs = X` standardized with [`crate::linalg::standardize`] (column means
/// removed, `ddof = 0` scales), or `x` itself when `pre_standardized`. Then
/// `Xs = U diag(s) V'` and `V` has orthonormal columns.
///
/// Tall full-rank `Xs` takes `V` and `s` from the eigendecomposition of
/// `Xs'Xs` instead of the SVD, which changes results by rounding only.
///
/// # Shapes
/// - `x`: `(N, D)`, both at least 1
/// - returns `v: (D, K)`, `s: (K,)`, with `K ≤ min(N, D)`
///
/// # Errors
///
/// - [`PlsKitError::InvalidInput`] — `x` has no rows or no columns, has
///   non-finite values, has every column constant (when standardized), or
///   has all singular values zero.
/// - [`PlsKitError::Internal`] — the SVD failed.
pub fn pca_basis(x: MatRef<'_, f64>, pre_standardized: bool) -> PlsKitResult<PcaBasisOutput> {
    crate::fit::with_thread_limit(|| pca_basis_impl(x, pre_standardized, crate::fit::par_fixed()))
}

/// Body of [`pca_basis`], on the caller's pool; `par` threads the Gram
/// product of [`tall_svd_v`] (`Par::Seq` inside replicate workers).
pub(crate) fn pca_basis_impl(
    x: MatRef<'_, f64>,
    pre_standardized: bool,
    par: Par,
) -> PlsKitResult<PcaBasisOutput> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(PlsKitError::InvalidInput(format!(
            "X has shape ({}, {}); both dimensions must be at least 1",
            x.nrows(),
            x.ncols()
        )));
    }
    if !x.is_all_finite() {
        return Err(PlsKitError::InvalidInput(
            "X contains non-finite values".into(),
        ));
    }
    // An `x` used as given that is far from unit scale is divided by a
    // power of two `c`, and its singular values multiplied back.
    let mut c = 1.0;
    let (sv, v) = if pre_standardized {
        match crate::linalg::pre_standardized_div(x, None) {
            Some((xd, cd)) => {
                c = cd;
                tall_svd_v(xd.as_ref(), par)
            }
            None => tall_svd_v(x, par),
        }
    } else {
        // A constant column standardizes to rounding noise, not to zeros, so
        // an all-constant X would give a basis of noise.
        let n = x.nrows();
        if (0..x.ncols()).all(|j| scaled_moments(n, |i| x[(i, j)], None).is_constant(n)) {
            return Err(PlsKitError::InvalidInput(
                "every column of X is constant".into(),
            ));
        }
        tall_svd_v(standardize(x).0.as_ref(), par)
    }
    .map_err(|e| PlsKitError::Internal(format!("SVD of X failed: {e:?}")))?;

    // Singular values are non-increasing, so the kept ones are a prefix.
    // The shape check above makes `min(N, D) ≥ 1`, so `s[0]` exists.
    let s0 = sv[0];
    let k = (0..sv.nrows()).take_while(|&i| sv[i] > 1e-10 * s0).count();
    if k == 0 {
        return Err(PlsKitError::InvalidInput(
            "X has all singular values zero after standardization".into(),
        ));
    }
    Ok(PcaBasisOutput {
        v: v.as_ref().subcols(0, k).to_owned(),
        s: Col::<f64>::from_fn(k, |i| sv[i] * c),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{GRAM_TAU, mat_mul, self_adjoint_eigen, tall_svd_v, thin_svd, thin_svd_v};
    use crate::test_support::{assert_bits_eq, assert_mat_bits_eq, col_vals, unif};

    fn assert_orthonormal(v: MatRef<'_, f64>) {
        let g = mat_mul(v.transpose(), v, Par::Seq);
        for i in 0..g.nrows() {
            for j in 0..g.ncols() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (g[(i, j)] - want).abs() < 1e-12,
                    "V'V[{i},{j}]={}",
                    g[(i, j)]
                );
            }
        }
    }

    #[test]
    fn pca_keeps_the_rank_and_orthonormal_columns() {
        let mut x = unif(40, 5, 1);
        for i in 0..40 {
            x[(i, 2)] = x[(i, 0)] + x[(i, 1)];
        }
        let out = pca_basis(x.as_ref(), false).unwrap();
        assert_eq!(out.v.shape(), (5, 4));
        assert_eq!(out.s.nrows(), 4);
        assert_orthonormal(out.v.as_ref());
        for i in 1..4 {
            assert!(out.s[i - 1] >= out.s[i] && out.s[i] > 0.0);
        }
    }

    #[test]
    fn pca_matches_the_svd_with_u() {
        // tall, square, wide, and rank 3 at both orientations.
        let low = |n, d, seed| {
            mat_mul(
                unif(n, 3, seed).as_ref(),
                unif(3, d, seed + 1).as_ref(),
                Par::Seq,
            )
        };
        for (name, x) in [
            ("tall", unif(200, 12, 5)),
            ("square", unif(40, 40, 6)),
            ("wide", unif(20, 90, 7)),
            ("rank 3 tall", low(60, 10, 8)),
            ("rank 3 wide", low(10, 60, 10)),
        ] {
            let out = pca_basis(x.as_ref(), false).unwrap();
            let full = thin_svd(standardize(x.as_ref()).0.as_ref(), Par::Seq).unwrap();
            let k = (0..full.s.nrows())
                .take_while(|&i| full.s[i] > 1e-10 * full.s[0])
                .count();
            assert_eq!(out.s.nrows(), k, "{name}: K");
            // Tall full-rank input takes the Gram route, accurate to 1e-10.
            let tol = if name == "tall" { 1e-10 } else { 1e-12 };
            for i in 0..k {
                assert!(
                    (out.s[i] - full.s[i]).abs() <= tol * full.s[0],
                    "{name}: s[{i}]"
                );
            }
            // Column signs are arbitrary, so compare the projectors V V'.
            let vk = full.v.as_ref().subcols(0, k);
            let a = mat_mul(out.v.as_ref(), out.v.transpose(), Par::Seq);
            let b = mat_mul(vk, vk.transpose(), Par::Seq);
            assert!((&a - &b).norm_max() < 1e-10, "{name}: V V'");
        }
    }

    #[test]
    fn pca_on_wide_data() {
        let x = unif(30, 100, 2);
        let out = pca_basis(x.as_ref(), false).unwrap();
        assert_eq!(out.v.shape(), (100, 29));
        assert_orthonormal(out.v.as_ref());
    }

    #[test]
    fn pca_error_table() {
        let mut nan = unif(10, 4, 3);
        nan[(2, 1)] = f64::NAN;
        let cases: [(&str, Mat<f64>, bool); 5] = [
            ("nan", nan, false),
            ("all zero", Mat::zeros(10, 4), true),
            ("all zero standardized", Mat::zeros(10, 4), false),
            ("no rows", Mat::zeros(0, 4), false),
            ("no cols", Mat::zeros(10, 0), false),
        ];
        for (name, x, pre) in cases {
            assert!(
                matches!(
                    pca_basis(x.as_ref(), pre),
                    Err(PlsKitError::InvalidInput(_))
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn all_constant_columns_are_invalid_input() {
        // 0.1 is not exact in binary, so its column standardizes to rounding
        // noise rather than zeros.
        let x = Mat::<f64>::from_fn(20, 3, |_, j| [0.1, 3.7, -1e5][j]);
        let r = pca_basis(x.as_ref(), false);
        assert!(
            matches!(&r, Err(PlsKitError::InvalidInput(m)) if m.contains("constant")),
            "{r:?}"
        );
    }

    #[test]
    fn pca_pre_standardized_skips_standardization() {
        let x = unif(25, 6, 4);
        let a = pca_basis(standardize(x.as_ref()).0.as_ref(), true).unwrap();
        let b = pca_basis(x.as_ref(), false).unwrap();
        assert_bits_eq(&col_vals(a.s.as_ref()), &col_vals(b.s.as_ref()), "s");
        assert_mat_bits_eq(a.v.as_ref(), b.v.as_ref(), "v");
    }

    /// The Gram route computed by hand: eigenvalues and eigenvectors of
    /// `a'a`, reversed to non-increasing order, `s = sqrt(λ)`.
    fn gram_by_hand(a: MatRef<'_, f64>) -> (Col<f64>, Mat<f64>) {
        let d = a.ncols();
        let (lam, u) =
            self_adjoint_eigen(mat_mul(a.transpose(), a, Par::Seq).as_ref(), Par::Seq).unwrap();
        (
            Col::<f64>::from_fn(d, |i| lam[d - 1 - i].sqrt()),
            Mat::<f64>::from_fn(d, d, |i, j| u[(i, d - 1 - j)]),
        )
    }

    /// `Q1 diag(s) Q2'` (200 × 12) with `s` geometric from 1 to `s_min`.
    fn with_spectrum(s_min: f64) -> Mat<f64> {
        let q1 = thin_svd(unif(200, 12, 21).as_ref(), Par::Seq).unwrap().u;
        let q2 = thin_svd(unif(12, 12, 22).as_ref(), Par::Seq).unwrap().v;
        let r = s_min.powf(1.0 / 11.0);
        let s: Vec<f64> = (0..12).map(|j| r.powi(j)).collect();
        let q1s = Mat::<f64>::from_fn(200, 12, |i, j| q1[(i, j)] * s[j]);
        mat_mul(q1s.as_ref(), q2.transpose(), Par::Seq)
    }

    fn assert_same_as_gram(a: MatRef<'_, f64>, what: &str) {
        let (s, v) = tall_svd_v(a, Par::Seq).unwrap();
        let (ws, wv) = gram_by_hand(a);
        assert_bits_eq(&col_vals(s.as_ref()), &col_vals(ws.as_ref()), what);
        assert_mat_bits_eq(v.as_ref(), wv.as_ref(), what);
    }

    #[test]
    fn tall_full_rank_takes_the_gram_route() {
        let a = standardize(unif(200, 12, 5).as_ref()).0;
        assert_same_as_gram(a.as_ref(), "tall");
    }

    #[test]
    fn just_above_tau_takes_the_gram_route() {
        assert_same_as_gram(with_spectrum(1.1e-3).as_ref(), "s_min 1.1e-3");
    }

    #[test]
    fn fallbacks_are_bit_identical_to_the_svd() {
        let mut dup = standardize(unif(200, 12, 5).as_ref()).0;
        for i in 0..200 {
            dup[(i, 3)] = dup[(i, 0)];
        }
        let mut constant = unif(200, 12, 9);
        for i in 0..200 {
            constant[(i, 4)] = 0.1;
        }
        for (name, a) in [
            ("duplicated column", dup),
            ("near collinear", with_spectrum(0.9e-3)),
            ("constant column", standardize(constant.as_ref()).0),
            ("square", standardize(unif(40, 40, 6).as_ref()).0),
            ("wide", standardize(unif(20, 90, 7).as_ref()).0),
        ] {
            let (s, v) = tall_svd_v(a.as_ref(), Par::Seq).unwrap();
            let (ws, wv) = thin_svd_v(a.as_ref(), Par::Seq).unwrap();
            assert_bits_eq(&col_vals(s.as_ref()), &col_vals(ws.as_ref()), name);
            assert_mat_bits_eq(v.as_ref(), wv.as_ref(), name);
        }
    }

    #[test]
    fn gram_route_matches_the_svd() {
        let signal = mat_mul(
            unif(2000, 5, 31).as_ref(),
            unif(5, 30, 32).as_ref(),
            Par::Seq,
        );
        let noise = unif(2000, 30, 33);
        let low_rank = Mat::<f64>::from_fn(2000, 30, |i, j| signal[(i, j)] + 0.01 * noise[(i, j)]);
        // A projector onto all D columns is the identity and cannot fail,
        // so the full-width cases check only K and s.
        for (name, x, gap) in [
            ("2000x30 rank 5 + noise", low_rank, Some(5)),
            ("N = D + 1", unif(13, 12, 11), None),
            ("one column", unif(50, 1, 12), None),
        ] {
            let out = pca_basis(x.as_ref(), false).unwrap();
            let (ws, wv) = thin_svd_v(standardize(x.as_ref()).0.as_ref(), Par::Seq).unwrap();
            let d = ws.nrows();
            // Every case is full rank and well inside τ, so it takes the Gram route.
            assert!(
                ws[d - 1] > GRAM_TAU * ws[0],
                "{name}: not on the Gram route"
            );
            assert_eq!(out.s.nrows(), d, "{name}: K");
            let tol = if d == 1 { 1e-12 } else { 1e-10 };
            for i in 0..d {
                assert!((out.s[i] - ws[i]).abs() <= tol * ws[0], "{name}: s[{i}]");
            }
            if let Some(gap) = gap {
                let vk = out.v.as_ref().subcols(0, gap);
                let wk = wv.as_ref().subcols(0, gap);
                let a = mat_mul(vk, vk.transpose(), Par::Seq);
                let b = mat_mul(wk, wk.transpose(), Par::Seq);
                assert!((&a - &b).norm_max() < 1e-8, "{name}: projector");
            }
        }
    }

    #[test]
    fn pre_standardized_far_from_unit_scale() {
        // 2^300 is past `pre_standardized_div`'s range, so `x` is divided by
        // a power of two before the Gram product, which would overflow.
        let xs = standardize(unif(300, 10, 13).as_ref()).0;
        let b = pca_basis(xs.as_ref(), true).unwrap();
        for c in [2f64.powi(20), 2f64.powi(300)] {
            let big = Mat::<f64>::from_fn(300, 10, |i, j| xs[(i, j)] * c);
            let a = pca_basis(big.as_ref(), true).unwrap();
            assert_eq!(a.s.nrows(), b.s.nrows(), "{c:e}: K");
            for i in 0..a.s.nrows() {
                assert!(
                    (a.s[i] / (b.s[i] * c) - 1.0).abs() <= 1e-10,
                    "{c:e}: s[{i}]"
                );
            }
        }
    }
}
