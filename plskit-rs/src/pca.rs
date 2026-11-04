//! PCA basis of standardized `X`: the right singular vectors with nonzero
//! singular value, in component order. Used by
//! [`crate::factors::factors`] as the basis it splits a coefficient vector on.

use faer::{Col, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::linalg::{scaled_moments, standardize, thin_svd};

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
    let svd = if pre_standardized {
        thin_svd(x, Par::Seq)
    } else {
        // A constant column standardizes to rounding noise, not to zeros, so
        // an all-constant X would give a basis of noise.
        let n = x.nrows();
        if (0..x.ncols()).all(|j| scaled_moments(n, |i| x[(i, j)], None).is_constant(n)) {
            return Err(PlsKitError::InvalidInput(
                "every column of X is constant".into(),
            ));
        }
        thin_svd(standardize(x).0.as_ref(), Par::Seq)
    }
    .map_err(|e| PlsKitError::Internal(format!("SVD of X failed: {e:?}")))?;

    // Singular values are non-increasing, so the kept ones are a prefix.
    // The shape check above makes `min(N, D) ≥ 1`, so `s[0]` exists.
    let s0 = svd.s[0];
    let k = (0..svd.s.nrows())
        .take_while(|&i| svd.s[i] > 1e-10 * s0)
        .count();
    if k == 0 {
        return Err(PlsKitError::InvalidInput(
            "X has all singular values zero after standardization".into(),
        ));
    }
    Ok(PcaBasisOutput {
        v: svd.v.as_ref().subcols(0, k).to_owned(),
        s: Col::<f64>::from_fn(k, |i| svd.s[i]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::mat_mul;
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
}
