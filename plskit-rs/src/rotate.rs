//! Simple-structure rotation of PLS weights. v0.1.1 ships varimax via
//! pairwise Kaiser sweeps; the `RotationMethod` enum is parameterized so
//! promax / oblimin / geomin can land later without changing the surface.
//!
//! Algorithm reference: `SSD/SSDLite/ssdiff/backends/multipls.py:20-128`
//! (Kaiser closed-form 2D angle + lexicographic pair sweeps). The Rust
//! port reproduces SSDLite's numerical output for the SSDLite migration
//! to be bit-exact.
//!
// "SSDLite" is a tool name, not a Rust item — suppress backtick warning.
#![allow(clippy::doc_markdown)]

use faer::{Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};

/// Method-axis dispatch payload. The variant carries its resolved
/// args; wrappers translate `(method: str, args: dict)` to the
/// matching variant at the FFI seam.
#[derive(Debug, Clone)]
pub enum RotationMethod {
    /// Varimax via pairwise Kaiser sweeps.
    Varimax(VarimaxArgs),
    // Promax(PromaxArgs), Oblimin(ObliminArgs), Geomin(GeominArgs) — v0.2+.
}

/// Resolved varimax parameters. Defaults match SSDLite's
/// `varimax_kaiser_sweep` so migration is numerically faithful.
#[derive(Debug, Clone, Copy)]
pub struct VarimaxArgs {
    /// Maximum Kaiser sweeps. SSDLite default: 50.
    pub max_iter: usize,
    /// Convergence tolerance on the varimax criterion `V`. SSDLite default: 1e-8.
    pub tol: f64,
    /// Row-normalize the simple-structure target before sweeping. SSDLite default: true.
    pub kaiser_normalize: bool,
}

impl Default for VarimaxArgs {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-8,
            kaiser_normalize: true,
        }
    }
}

/// Output of `rotate`. The caller already holds the args and passes them
/// through to populate `RotationSpec.args` on the wrapper side; we don't
/// echo them back.
#[derive(Debug)]
pub struct RotateOutput {
    /// Rotated weights `W @ R`, shape `(D, K)`.
    pub w_rot: Mat<f64>,
    /// Orthogonal rotation, shape `(K, K)`. `w_rot = w @ r`.
    pub r: Mat<f64>,
    /// Number of Kaiser sweeps actually run (≤ `args.max_iter`).
    pub sweeps: usize,
    /// Final value of the varimax criterion `V = Σ_j Var(target[:, j]²)`.
    pub v_converged: f64,
}

/// Rotate weights `w` so that simple structure (per `method`) is
/// maximized, optionally on a different loading basis `l`.
///
/// # Errors
///
/// - [`PlsKitError::InvalidInput`] — K=0, or non-finite values in `w` / `l`.
/// - [`PlsKitError::ShapeMismatch`] — `l.ncols() != w.ncols()`.
///
/// `RotationMethodNotImplemented` and `InvalidArgs` are produced at the
/// wrapper layer (when parsing `(method, args)` strings/dicts) and never
/// here — by the time the core sees a `RotationMethod`, the method is
/// known and the args are typed.
// The public signature takes `method` by value; the enum is small
// and the by-value form keeps wrappers (FFI, fixture gen) from threading
// references through the dispatch.
#[allow(clippy::needless_pass_by_value)]
pub fn rotate(
    w: MatRef<'_, f64>,
    method: RotationMethod,
    l: Option<MatRef<'_, f64>>,
) -> PlsKitResult<RotateOutput> {
    // Shape / finiteness gates.
    let (d_rows, k) = (w.nrows(), w.ncols());
    if k == 0 {
        return Err(PlsKitError::InvalidInput("W has K=0".into()));
    }
    if !mat_is_finite(w) {
        return Err(PlsKitError::InvalidInput(
            "W contains non-finite values".into(),
        ));
    }
    if let Some(ll) = l.as_ref() {
        if ll.ncols() != k {
            return Err(PlsKitError::ShapeMismatch(format!(
                "L.ncols={} but W.ncols={}",
                ll.ncols(),
                k
            )));
        }
        if !mat_is_finite(*ll) {
            return Err(PlsKitError::InvalidInput(
                "L contains non-finite values".into(),
            ));
        }
    }

    let _ = d_rows; // shape sanity reserved for future variants
    match method {
        RotationMethod::Varimax(args) => varimax_rotate(w, l, args, k),
    }
}

// varimax_rotate always returns Ok today; the Result return is forward-
// compatibility for future variants (promax, oblimin) that may error.
#[allow(clippy::unnecessary_wraps)]
// Single-letter math vars (w, l, k, r, c, s, p, q) are domain-correct —
// see project memory rule. Clippy allow is localized here, not global.
#[allow(clippy::many_single_char_names)]
fn varimax_rotate(
    w: MatRef<'_, f64>,
    l: Option<MatRef<'_, f64>>,
    args: VarimaxArgs,
    k: usize,
) -> PlsKitResult<RotateOutput> {
    // K=1 no-op short-circuit: rotation of a 1-D subspace is
    // identity. v_converged uses the chosen target (L if provided, else W).
    if k == 1 {
        let r = identity(1);
        let w_rot = mat_clone(w);
        let target = l.unwrap_or(w);
        let v_converged = sum_var_squared_columns(target);
        return Ok(RotateOutput {
            w_rot,
            r,
            sweeps: 0,
            v_converged,
        });
    }

    // Step 1: build T_simp (the simple-structure target).
    let basis = l.unwrap_or(w);
    let n_rows = basis.nrows();
    let mut t_simp: Mat<f64> = if args.kaiser_normalize {
        row_normalize(basis)
    } else {
        mat_clone(basis)
    };

    // Step 2: R = I_k.
    let mut r = identity(k);

    // Step 3: pairwise Kaiser sweeps.
    let mut v_prev = sum_var_squared_columns(t_simp.as_ref());
    let mut sweeps_done = 0_usize;
    for sweep in 1..=args.max_iter {
        sweeps_done = sweep;
        for p in 0..(k - 1) {
            for q in (p + 1)..k {
                // Closed-form 2D angle on columns (p, q) of T_simp.
                let pair = Mat::<f64>::from_fn(n_rows, 2, |i, j| {
                    if j == 0 {
                        t_simp[(i, p)]
                    } else {
                        t_simp[(i, q)]
                    }
                });
                let theta = varimax_angle_2d(pair.as_ref());
                let c = theta.cos();
                let s = theta.sin();
                // Apply 2D rotation to columns (p, q) of T_simp and R.
                rotate_columns_inplace(&mut t_simp, p, q, c, s);
                rotate_columns_inplace(&mut r, p, q, c, s);
            }
        }
        let v_new = sum_var_squared_columns(t_simp.as_ref());
        if v_new - v_prev < args.tol {
            // Bit-exact match with SSDLite multipls.py:118-120: break
            // BEFORE updating v_prev, so v_converged reports the V at
            // the sweep that converged (the OLD value at break time).
            // Reassigning v_prev = v_new here would shift v_converged
            // by one sweep relative to the SSDLite reference.
            break;
        }
        v_prev = v_new;
    }

    // Step 4: w_rot = w @ r. Note: when kaiser_normalize=true the SSDLite
    // reference (multipls.py:120) computes L_rot = L @ R from the
    // *original* L (not row-normalized) at the end — i.e. row magnitudes
    // are restored. We do the analogous thing for W: w_rot = w @ r is
    // computed from the original w, regardless of kaiser_normalize.
    let w_rot = matmul(w, r.as_ref());
    Ok(RotateOutput {
        w_rot,
        r,
        sweeps: sweeps_done,
        v_converged: v_prev,
    })
}

// ── small helpers ────────────────────────────────────────────────

fn mat_is_finite(m: MatRef<'_, f64>) -> bool {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}

fn mat_clone(m: MatRef<'_, f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)])
}

fn identity(k: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 })
}

#[allow(clippy::many_single_char_names)]
fn row_normalize(m: MatRef<'_, f64>) -> Mat<f64> {
    // Row-norm with floor 1e-12 (matches SSDLite `varimax_kaiser_sweep`,
    // multipls.py:91-93: zero rows are left at norm=1 to avoid div-by-zero).
    let n = m.nrows();
    let k = m.ncols();
    let mut norms = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..k {
            let v = m[(i, j)];
            s += v * v;
        }
        let nrm = s.sqrt();
        norms[i] = if nrm > 1e-12 { nrm } else { 1.0 };
    }
    Mat::<f64>::from_fn(n, k, |i, j| m[(i, j)] / norms[i])
}

fn sum_var_squared_columns(m: MatRef<'_, f64>) -> f64 {
    let n = m.nrows();
    let k = m.ncols();
    let n_f = n as f64;
    let mut total = 0.0_f64;
    for j in 0..k {
        let mut sum_sq = 0.0_f64;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let v = m[(i, j)];
            let vv = v * v;
            sum += vv;
            sum_sq += vv * vv;
        }
        let mean = sum / n_f;
        // Population variance, matching numpy.var default (ddof=0) used in SSDLite.
        total += sum_sq / n_f - mean * mean;
    }
    total
}

#[allow(clippy::many_single_char_names)]
fn rotate_columns_inplace(m: &mut Mat<f64>, p: usize, q: usize, c: f64, s: f64) {
    let n = m.nrows();
    for i in 0..n {
        let mp = m[(i, p)];
        let mq = m[(i, q)];
        m[(i, p)] = c * mp + s * mq;
        m[(i, q)] = -s * mp + c * mq;
    }
}

fn matmul(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    debug_assert_eq!(a.ncols(), b.nrows());
    Mat::<f64>::from_fn(a.nrows(), b.ncols(), |i, j| {
        let mut s = 0.0_f64;
        for k in 0..a.ncols() {
            s += a[(i, k)] * b[(k, j)];
        }
        s
    })
}

/// Closed-form Kaiser 2D varimax rotation angle for a `(n, 2)` loadings
/// slice. Maximizes `Σ_j Var(L_rot[:, j]²)` over rotations.
///
/// Reference: SSDLite `varimax_angle_2d` (multipls.py:20-44).
#[allow(clippy::many_single_char_names)]
fn varimax_angle_2d(l: MatRef<'_, f64>) -> f64 {
    debug_assert_eq!(l.ncols(), 2, "varimax_angle_2d requires exactly 2 columns");
    let n = l.nrows();
    let n_f = n as f64;

    let mut u_sum = 0.0_f64;
    let mut v_sum = 0.0_f64;
    let mut uu = 0.0_f64;
    let mut vv = 0.0_f64;
    let mut uv = 0.0_f64;
    for i in 0..n {
        let a = l[(i, 0)];
        let b = l[(i, 1)];
        let u = a * a - b * b;
        let v = 2.0 * a * b;
        u_sum += u;
        v_sum += v;
        uu += u * u;
        vv += v * v;
        uv += u * v;
    }
    let big_a = (uu - vv) - (u_sum * u_sum - v_sum * v_sum) / n_f;
    let big_b = 2.0 * (uv - u_sum * v_sum / n_f);
    big_b.atan2(big_a) / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn varimax_angle_2d_zero_for_already_simple() {
        // A target with one all-positive column and one all-zero
        // column already has perfect simple structure — angle is 0.
        let l = Mat::<f64>::from_fn(5, 2, |i, j| if j == 0 { (i + 1) as f64 } else { 0.0 });
        let theta = varimax_angle_2d(l.as_ref());
        assert!(theta.abs() < 1e-12, "expected ~0, got {theta}");
    }

    #[test]
    #[allow(clippy::unnested_or_patterns)] // tabular layout is clearer here
    fn varimax_angle_2d_known_value() {
        // Drift lock: reference value computed against SSDLite's
        // `varimax_angle_2d` on this exact input.
        // Input:
        //   L = [[ 1, 1], [ 1,-1], [-1, 1], [-1,-1], [ 2, 0]] / 2
        // The (2, 0) row breaks symmetry: big_b = 0, big_a = -0.2,
        // so theta = atan2(0, -0.2)/4 = π/4.
        let l = Mat::<f64>::from_fn(5, 2, |i, j| match (i, j) {
            (0, 0) | (0, 1) | (1, 0) | (2, 1) => 0.5,
            (1, 1) | (2, 0) | (3, 0) | (3, 1) => -0.5,
            (4, 0) => 1.0,
            _ => 0.0,
        });
        let theta = varimax_angle_2d(l.as_ref());
        // Reference Python output: theta ≈ π/4 ≈ 0.7853981633974483.
        // Verified against SSDLite varimax_angle_2d on this exact input (2026-04-27).
        let expected = std::f64::consts::PI / 4.0;
        assert!(
            (theta - expected).abs() < 1e-12,
            "expected π/4, got {theta}"
        );
    }

    fn random_w(rng_seed: u64, n: usize, k: usize) -> Mat<f64> {
        // Tiny deterministic LCG so tests don't depend on rand crate state.
        let mut s = rng_seed;
        Mat::<f64>::from_fn(n, k, |_, _| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to roughly U(-1, 1).
            (f64::from((s >> 33) as u32) / f64::from(u32::MAX)) * 2.0 - 1.0
        })
    }

    fn approx_eq_mat(a: MatRef<'_, f64>, b: MatRef<'_, f64>, tol: f64) -> bool {
        if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
            return false;
        }
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                if (a[(i, j)] - b[(i, j)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn rotate_k1_is_noop() {
        let w = random_w(1, 20, 1);
        let out = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )
        .unwrap();
        assert_eq!(out.sweeps, 0);
        assert_eq!(out.r.nrows(), 1);
        assert_eq!(out.r.ncols(), 1);
        assert!((out.r[(0, 0)] - 1.0).abs() < 1e-15);
        assert!(approx_eq_mat(out.w_rot.as_ref(), w.as_ref(), 1e-15));
    }

    #[test]
    fn rotate_k0_errors() {
        let w = Mat::<f64>::from_fn(5, 0, |_, _| 0.0);
        let res = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        );
        assert!(matches!(res, Err(PlsKitError::InvalidInput(_))));
    }

    #[test]
    fn rotate_l_shape_mismatch_errors() {
        let w = random_w(2, 10, 3);
        let l = random_w(3, 8, 2); // ncols=2 ≠ w.ncols=3
        let res = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            Some(l.as_ref()),
        );
        assert!(matches!(res, Err(PlsKitError::ShapeMismatch(_))));
    }

    #[test]
    fn rotate_non_finite_w_errors() {
        let mut w = random_w(4, 5, 2);
        w[(0, 0)] = f64::NAN;
        let res = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        );
        assert!(matches!(res, Err(PlsKitError::InvalidInput(_))));
    }

    #[test]
    fn rotate_non_finite_l_errors() {
        let w = random_w(11, 5, 2);
        let mut l = random_w(12, 8, 2);
        l[(0, 1)] = f64::INFINITY;
        let res = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            Some(l.as_ref()),
        );
        assert!(matches!(res, Err(PlsKitError::InvalidInput(_))));
    }

    #[test]
    fn rotate_r_is_orthogonal() {
        let w = random_w(5, 30, 4);
        let out = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )
        .unwrap();
        // R'R should be I.
        let rt_r = matmul(out.r.transpose(), out.r.as_ref());
        let eye = identity(4);
        assert!(approx_eq_mat(rt_r.as_ref(), eye.as_ref(), 1e-10));
    }

    #[test]
    fn rotate_idempotent_on_converged_solution() {
        // R-precision tracks V-precision quadratically near the optimum:
        // V-tol=1e-8 (default) → R within ~5e-5 of I on a second pass;
        // V-tol=1e-12 → R within ~3e-7. We tighten V-tol here so the
        // idempotency property can be asserted at 1e-6.
        let tight = VarimaxArgs {
            tol: 1e-12,
            ..VarimaxArgs::default()
        };
        let w = random_w(6, 40, 3);
        let out1 = rotate(w.as_ref(), RotationMethod::Varimax(tight), None).unwrap();
        let out2 = rotate(out1.w_rot.as_ref(), RotationMethod::Varimax(tight), None).unwrap();
        let eye = identity(3);
        assert!(approx_eq_mat(out2.r.as_ref(), eye.as_ref(), 1e-6));
    }

    #[test]
    fn rotate_w_at_r_equals_w_rot() {
        let w = random_w(7, 25, 3);
        let out = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )
        .unwrap();
        let recomputed = matmul(w.as_ref(), out.r.as_ref());
        assert!(approx_eq_mat(
            out.w_rot.as_ref(),
            recomputed.as_ref(),
            1e-15
        ));
    }

    #[test]
    fn rotate_with_l_uses_loading_basis() {
        // Different L should produce a different R than L=None.
        let w = random_w(8, 10, 3);
        let l = random_w(9, 40, 3);
        let out_none = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )
        .unwrap();
        let out_l = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            Some(l.as_ref()),
        )
        .unwrap();
        assert!(
            !approx_eq_mat(out_none.r.as_ref(), out_l.r.as_ref(), 1e-6),
            "R should differ when L is provided"
        );
    }

    #[test]
    fn rotate_kaiser_normalize_off_differs() {
        let w = random_w(10, 30, 3);
        let on = rotate(
            w.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )
        .unwrap();
        let off_args = VarimaxArgs {
            kaiser_normalize: false,
            ..VarimaxArgs::default()
        };
        let off = rotate(w.as_ref(), RotationMethod::Varimax(off_args), None).unwrap();
        assert!(
            !approx_eq_mat(on.r.as_ref(), off.r.as_ref(), 1e-6),
            "R should differ when kaiser_normalize is toggled"
        );
    }
}
