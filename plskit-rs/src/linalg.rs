//! Linear-algebra and small-stat helpers shared across the core.

use faer::{Col, ColRef, Mat, MatRef};

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
/// Zero-variance columns get scale=1.
/// `ddof = 0` (population std, like numpy default).
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - returns `(xs: (n_samples, n_features), mean: (n_features,), scale: (n_features,))`
#[must_use]
pub fn standardize(x: MatRef<'_, f64>) -> (Mat<f64>, Col<f64>, Col<f64>) {
    standardize_weighted(x, None)
}

/// Weighted column-wise z-score. `weights = None` matches `standardize` bit-for-bit.
/// `weights = Some(w)` uses spec §3.2 weighted mean/var (population, ddof=0).
/// Caller must ensure weights are non-negative, finite, and Σw > 0
/// (validation lives in `validate_and_normalize_weights` / `preprocess`).
#[must_use]
pub fn standardize_weighted(
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

    let mean = Col::<f64>::from_fn(n_cols, |j| match wpref {
        None => (0..n_rows).map(|i| x[(i, j)]).sum::<f64>() / n_f,
        Some(w) => (0..n_rows).map(|i| w[i] * x[(i, j)]).sum::<f64>() / n_f,
    });
    let scale = Col::<f64>::from_fn(n_cols, |j| {
        let col_mean = mean[j];
        let var: f64 = match wpref {
            None => {
                (0..n_rows)
                    .map(|i| (x[(i, j)] - col_mean).powi(2))
                    .sum::<f64>()
                    / n_f
            }
            Some(w) => {
                (0..n_rows)
                    .map(|i| w[i] * (x[(i, j)] - col_mean).powi(2))
                    .sum::<f64>()
                    / n_f
            }
        };
        let std = var.sqrt();
        if std > 1e-12 {
            std
        } else {
            1.0
        }
    });
    let xs = Mat::<f64>::from_fn(n_rows, n_cols, |i, j| (x[(i, j)] - mean[j]) / scale[j]);
    (xs, mean, scale)
}

/// Apply previously-computed (mean, scale) to a fresh matrix.
#[must_use]
pub fn standardize_apply(
    x: MatRef<'_, f64>,
    mean: ColRef<'_, f64>,
    scale: ColRef<'_, f64>,
) -> Mat<f64> {
    Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| {
        (x[(i, j)] - mean[j]) / scale[j]
    })
}

/// Standardize a 1-D vector. Returns (z, mean, scale). Mirrors the
/// reshape→standardize→ravel pattern used by the prototype.
#[must_use]
pub fn standardize1(y: ColRef<'_, f64>) -> (Col<f64>, f64, f64) {
    standardize1_weighted(y, None)
}

/// Weighted scalar-standardize for y. None ⇒ unweighted.
/// Caller is responsible for weight validation; see `validate_and_normalize_weights`.
#[must_use]
pub fn standardize1_weighted(
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> (Col<f64>, f64, f64) {
    let n = y.nrows();
    let n_f = n as f64;
    let w_prime: Option<Col<f64>> = weights.map(|w| {
        let s: f64 = (0..n).map(|i| w[i]).sum();
        Col::<f64>::from_fn(n, |i| w[i] * n_f / s)
    });
    let wpref = w_prime.as_ref().map(Col::as_ref);
    let mean: f64 = match wpref {
        None => (0..n).map(|i| y[i]).sum::<f64>() / n_f,
        Some(w) => (0..n).map(|i| w[i] * y[i]).sum::<f64>() / n_f,
    };
    let var: f64 = match wpref {
        None => (0..n).map(|i| (y[i] - mean).powi(2)).sum::<f64>() / n_f,
        Some(w) => (0..n).map(|i| w[i] * (y[i] - mean).powi(2)).sum::<f64>() / n_f,
    };
    let scale = if var.sqrt() > 1e-12 { var.sqrt() } else { 1.0 };
    let z = Col::<f64>::from_fn(n, |i| (y[i] - mean) / scale);
    (z, mean, scale)
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

/// Normalize weights so Σw' = n (mean 1). Returns `None` if `Σw == 0`.
/// Validation (negative, NaN) is the caller's responsibility.
#[must_use]
pub fn normalize_weights(w: ColRef<'_, f64>) -> Option<Col<f64>> {
    let n = w.nrows();
    let n_f = n as f64;
    let s: f64 = (0..n).map(|i| w[i]).sum();
    if s == 0.0 {
        return None;
    }
    Some(Col::<f64>::from_fn(n, |i| w[i] * n_f / s))
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

/// Stable log-gamma via Lanczos coefficients.
pub(crate) fn lgamma(x: f64) -> f64 {
    libm_lgamma(x)
}

#[allow(clippy::many_single_char_names)]
fn libm_lgamma(x: f64) -> f64 {
    // Stirling-with-correction; accurate to ~1e-13 over x > 0.
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
            - libm_lgamma(1.0 - x);
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
    use faer::linalg::matmul::matmul;
    use faer::linalg::solvers::{PartialPivLu, Solve};
    use faer::{Accum, Mat, Par};
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
    let lu = PartialPivLu::new(wtw.as_ref());
    let mut m = Mat::<f64>::identity(k, k);
    lu.solve_in_place(m.as_mut());
    (0..n)
        .map(|i| {
            let mut tmp = vec![0.0_f64; k];
            for jj in 0..k {
                let mut s = 0.0;
                for kk in 0..k {
                    s += m[(jj, kk)] * w[(i, kk)];
                }
                tmp[jj] = s;
            }
            (0..k).map(|jj| w[(i, jj)] * tmp[jj]).sum()
        })
        .collect()
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

    #[test]
    fn standardize_zero_variance_column_uses_scale_one() {
        let x = mat(3, 2, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]);
        let (_, _, scale) = standardize(x.as_ref());
        assert_relative_eq!(scale[0], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn fold_split_matches_numpy_array_split() {
        // n=10, 3 folds → sizes [4, 3, 3] per numpy.array_split semantics
        let idx: Vec<usize> = (0..10).collect();
        let folds = fold_split(&idx, 3);
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0].len(), 4);
        assert_eq!(folds[1].len(), 3);
        assert_eq!(folds[2].len(), 3);
        // No index lost
        let total: usize = folds.iter().map(Vec::len).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn t_sf_symmetric_around_zero() {
        let p_pos = t_sf(2.0, 10.0);
        let p_neg = t_sf(-2.0, 10.0);
        assert_relative_eq!(p_pos + p_neg, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn t_sf_matches_known_value() {
        // scipy.stats.t.sf(2.228, 10) ≈ 0.025 (two-tailed 0.05 critical)
        let p = t_sf(2.228, 10.0);
        assert_relative_eq!(p, 0.025, epsilon = 1e-3);
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
    #[allow(clippy::float_cmp)] // intentional bit-exact: weights=None must be identical to unweighted
    fn standardize_none_matches_unweighted() {
        let x = small_x();
        let (xs_a, m_a, s_a) = standardize(x.as_ref());
        let (xs_b, m_b, s_b) = standardize_weighted(x.as_ref(), None);
        for j in 0..x.ncols() {
            assert_eq!(m_a[j], m_b[j]);
            assert_eq!(s_a[j], s_b[j]);
            for i in 0..x.nrows() {
                assert_eq!(xs_a[(i, j)], xs_b[(i, j)]);
            }
        }
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
    }

    #[test]
    fn standardize_weighted_mean_is_weighted() {
        // 4 rows; weight first row heavily so weighted mean ≠ arithmetic mean.
        let x = Mat::from_fn(4, 1, |i, _| i as f64); // 0, 1, 2, 3
        let w_raw = Col::<f64>::from_fn(4, |i| if i == 0 { 9.0 } else { 1.0 / 3.0 });
        // After normalization w' has Σ = n = 4 and mean = 1.
        let n = 4.0_f64;
        let sum_w: f64 = (0..4).map(|i| if i == 0 { 9.0 } else { 1.0 / 3.0 }).sum();
        let w_prime: Vec<f64> = (0..4)
            .map(|i| (if i == 0 { 9.0 } else { 1.0 / 3.0 }) * n / sum_w)
            .collect();
        let expected_mean: f64 = (0..4).map(|i| w_prime[i] * (i as f64)).sum::<f64>() / n;
        let (_xs, m, _s) = standardize_weighted(x.as_ref(), Some(w_raw.as_ref()));
        assert_relative_eq!(m[0], expected_mean, epsilon = 1e-12);
        // Heavy weight on row 0 pulls weighted mean far below arithmetic mean (1.5).
        assert!(m[0] < 1.0);
    }

    #[test]
    fn standardize1_weighted_matches_unweighted_under_uniform() {
        let y = Col::<f64>::from_fn(5, |i| i as f64 - 2.0);
        let w = Col::<f64>::from_fn(5, |_| 1.0);
        let (z_a, m_a, s_a) = standardize1(y.as_ref());
        let (z_b, m_b, s_b) = standardize1_weighted(y.as_ref(), Some(w.as_ref()));
        assert_relative_eq!(m_a, m_b, epsilon = 1e-12);
        assert_relative_eq!(s_a, s_b, epsilon = 1e-12);
        for i in 0..y.nrows() {
            assert_relative_eq!(z_a[i], z_b[i], epsilon = 1e-12);
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
    }
}

#[cfg(test)]
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

/// R type-7 / numpy-default linear-interpolation quantile.
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

    #[test]
    fn matches_known_values() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // numpy.quantile(v, 0.5) = 3.0; (0.0) = 1.0; (1.0) = 5.0; (0.25) = 2.0.
        // We caller-sort, so pass already-sorted input.
        assert!((empirical_quantile(&v, 0.5) - 3.0).abs() < 1e-12);
        assert!((empirical_quantile(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((empirical_quantile(&v, 1.0) - 5.0).abs() < 1e-12);
        assert!((empirical_quantile(&v, 0.25) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn handles_empty() {
        assert!(empirical_quantile(&[], 0.5).is_nan());
    }
}
