//! Tests of the PLS1 kernel (`pls1_kernel`, `pls1_component_loop`, the X
//! backend) against an explicit-deflation reference kernel,
//! `super::tests::nipals_pls1_reference`.
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use super::tests::nipals_pls1_reference;
use super::tests::uniform_mat;
use super::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `(T, P, W, Q)` as both kernels return them.
type Parts = (Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>);

// ── designs ─────────────────────────────────────────────────────────────

// `uniform_mat` is reused from `fit::tests` (`pub(super)`); no local copy here.

/// `x` uniform on `[-1, 1)`, `y = Σ_{j<m} b_j x_j + sigma·noise`.
fn signal_design(n: usize, d: usize, m: usize, sigma: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let b: Vec<f64> = (0..m.min(d)).map(|_| rng.random_range(0.5..1.5)).collect();
    let y = Col::<f64>::from_fn(n, |i| {
        b.iter()
            .enumerate()
            .map(|(j, bj)| x[(i, j)] * bj)
            .sum::<f64>()
            + sigma * rng.random_range(-1.0..1.0)
    });
    (x, y)
}

/// Weights in `[0.25, 2.65]`, every fifth row (from row 3) exactly zero.
fn mixed_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| {
        if i % 5 == 3 {
            0.0
        } else {
            0.25 + (i % 7) as f64 * 0.4
        }
    })
}

/// `n × 40`, last column the sum of the first two (rank 39), `y` independent
/// of `X`. With `seed = 11` it is `tests::exhausted_y_design` at `n = 2000`.
fn exhausted_design(n: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
    let d = 40;
    let mut x = uniform_mat(n, d, seed);
    for i in 0..n {
        x[(i, d - 1)] = x[(i, 0)] + x[(i, 1)];
    }
    let yc = uniform_mat(n, 1, seed + 1);
    (x, Col::<f64>::from_fn(n, |i| yc[(i, 0)]))
}

// ── kernel inputs, exactly as `pls1_fit` builds them ─────────────────────

/// The arrays `pls1_fit` hands its kernel, and the scales it back-projects with.
struct KernelInputs {
    xs: Mat<f64>,
    ys: Col<f64>,
    x_scale: Col<f64>,
    y_scale: f64,
}

fn kernel_inputs(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
) -> KernelInputs {
    let wn = w.map(|w| crate::linalg::normalize_weights(w).expect("sum of weights > 0"));
    let wref = wn.as_ref().map(Col::as_ref);
    let (xs, _, x_scale) = crate::linalg::standardize_weighted(x, wref);
    let (zs, _, y_scale) = crate::linalg::standardize1_weighted(y, wref);
    match w {
        None => KernelInputs {
            xs,
            ys: zs,
            x_scale,
            y_scale,
        },
        Some(w) => {
            let sqw = fit_row_scale(w);
            KernelInputs {
                xs: scale_rows(xs.as_ref(), sqw.as_ref()),
                ys: scale_col(zs.as_ref(), sqw.as_ref()),
                x_scale,
                y_scale,
            }
        }
    }
}

/// `s_1 = Xs'ys` (sequential GEMV) and the relative floor, as `pls1_kernel` forms them.
fn loop_inputs(inp: &KernelInputs) -> (Col<f64>, f64) {
    let (n, d) = (inp.xs.nrows(), inp.xs.ncols());
    let mut s0 = Col::<f64>::zeros(d);
    matmul(
        s0.as_mut().as_mat_mut(),
        Accum::Replace,
        inp.xs.as_ref().transpose(),
        inp.ys.as_ref().as_mat(),
        1.0,
        Par::Seq,
    );
    (s0, w_rel_floor(n, d, inp.xs.norm_l2(), inp.ys.norm_l2()))
}

// ── rebuilt cross-products and tolerances ────────────────────────────────

/// `s_1 … s_{k_used+1}` (`s_a = X_a'y_a`), rebuilt from a kernel's outputs
/// with the loop's own update `s ← s − (q·t't)·p`.
fn s_sequence(inp: &KernelInputs, parts: &Parts) -> Vec<Col<f64>> {
    let (t, p, _, q) = parts;
    let d = inp.xs.ncols();
    let (mut s, _) = loop_inputs(inp);
    let mut out = vec![s.clone()];
    for a in 0..q.nrows() {
        let tt = t.col(a).to_owned().squared_norm_l2();
        let qtt = q[a] * tt;
        for j in 0..d {
            s[j] -= qtt * p[(j, a)];
        }
        out.push(s.clone());
    }
    out
}

/// `‖s‖` after keep-selection, the quantity the floor tests.
fn selected_norm(s: &Col<f64>, keep: Option<usize>) -> f64 {
    let mut v = s.clone();
    if let Some(kp) = keep {
        if kp < v.nrows() {
            hard_select_keep(&mut v, kp);
        }
    }
    v.norm_l2()
}

/// `C` in the per-component factor tolerance
/// `max(base, C·w_rel_floor / ‖s_a‖)` (see `factor_tolerances`).
const FACTOR_TOL_C: f64 = 10.0;

/// Per-component tolerance on `T`, `P`, `W`, `Q`, `a = 1..=k_used`:
/// `max(base, FACTOR_TOL_C · w_rel_floor(n, d, ‖Xs‖_F, ‖ys‖) / ‖s_a‖)`,
/// with `‖s_a‖` the selected norm of the reference's cross-product, rebuilt
/// from `reference` (the kernel under test never sets its own tolerance).
/// On both kernels the rounding of `s_a` sits at a small multiple of
/// `w_rel_floor` (its doc comment), so the direction `w_a`, and with it
/// `r_a`, `t_a`, `p_a` and `q_a`, is determined only to about
/// `w_rel_floor / ‖s_a‖` on either kernel. A component whose cross-product
/// has not decayed toward the floor gets `base` exactly. `coef` and `beta`
/// keep an unscaled tolerance (`coef_drift`): a decayed component's `q_a`,
/// and with it its share of `coef`, shrinks by the same factor.
fn factor_tolerances(
    inp: &KernelInputs,
    reference: &Parts,
    keep: Option<usize>,
    base: f64,
) -> Vec<f64> {
    let (_, floor) = loop_inputs(inp);
    let k_used = reference.3.nrows();
    s_sequence(inp, reference)[..k_used]
        .iter()
        .map(|s| base.max(FACTOR_TOL_C * floor / selected_norm(s, keep)))
        .collect()
}

/// Equal `k_used` and shapes, and every column of `T`, `P`, `W` (and entry
/// of `Q`) within `tols[a]`. Returns the worst `diff / tols[a]` (at most 1).
fn assert_factors_close(new: &Parts, old: &Parts, tols: &[f64], what: &str) -> f64 {
    let k_used = new.2.ncols();
    assert_eq!(k_used, old.2.ncols(), "{what}: k_used");
    for (name, a, b) in [
        ("T", &new.0, &old.0),
        ("P", &new.1, &old.1),
        ("W", &new.2, &old.2),
    ] {
        assert_eq!(
            (a.nrows(), a.ncols()),
            (b.nrows(), b.ncols()),
            "{what}: {name} shape"
        );
    }
    assert_eq!(new.3.nrows(), old.3.nrows(), "{what}: Q length");
    assert_eq!(tols.len(), k_used, "{what}: one tolerance per component");
    let mut worst = 0.0_f64;
    for (a, &tol) in tols.iter().enumerate() {
        let col_diff = |x: &Mat<f64>, y: &Mat<f64>| {
            (0..x.nrows())
                .map(|i| (x[(i, a)] - y[(i, a)]).abs())
                .fold(0.0_f64, f64::max)
        };
        let diff = col_diff(&new.0, &old.0)
            .max(col_diff(&new.1, &old.1))
            .max(col_diff(&new.2, &old.2))
            .max((new.3[a] - old.3[a]).abs());
        assert!(
            diff <= tol,
            "{what}: component {} differs by {diff:e} > {tol:e}",
            a + 1
        );
        worst = worst.max(diff / tol);
    }
    worst
}

fn assert_mat_bits(a: &Mat<f64>, b: &Mat<f64>, what: &str) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{what}: shape"
    );
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_eq!(
                a[(i, j)].to_bits(),
                b[(i, j)].to_bits(),
                "{what}[{i},{j}]: {} vs {}",
                a[(i, j)],
                b[(i, j)]
            );
        }
    }
}

fn assert_col_bits(a: &Col<f64>, b: &Col<f64>, what: &str) {
    assert_eq!(a.nrows(), b.nrows(), "{what}: length");
    for i in 0..a.nrows() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "{what}[{i}]: {} vs {}",
            a[i],
            b[i]
        );
    }
}

fn max_abs_diff_col(a: &Col<f64>, b: &Col<f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows(), "length");
    (0..a.nrows())
        .map(|i| (a[i] - b[i]).abs())
        .fold(0.0_f64, f64::max)
}

/// `max|a − b| / max(1, max|b|)`, the `coef` and `beta` drift. Against
/// `1e-10` it is relative for coefficients of order one and above and
/// absolute below, so a weak-signal fit (coefficients near `1e-6`) is not
/// held to `1e-16` absolute.
fn coef_drift(a: &Col<f64>, b: &Col<f64>) -> f64 {
    let scale = (0..b.nrows()).map(|i| b[i].abs()).fold(1.0_f64, f64::max);
    max_abs_diff_col(a, b) / scale
}

fn coef_of(parts: &Parts, par: Par) -> Col<f64> {
    pls1_coef_at_k(&parts.2, &parts.1, &parts.3, parts.2.ncols(), par)
}

/// `beta[j] = coef[j]·y_scale / x_scale[j]`, `pls1_fit`'s back-projection.
fn beta_of(coef: &Col<f64>, inp: &KernelInputs) -> Col<f64> {
    Col::<f64>::from_fn(coef.nrows(), |j| coef[j] * inp.y_scale / inp.x_scale[j])
}

// ── the loop's contract with a backend (a Gram backend relies on it) ──

/// The X backend with a gate that returns `Unresolved` at a chosen
/// component, logging every gate call. It stands in for the Gram backend,
/// the only backend that ever returns `Unresolved`.
struct GateAt<'a> {
    inner: XBackend<'a>,
    s_at: Option<usize>,
    tt_at: Option<usize>,
    log: Vec<(&'static str, usize)>,
}

impl ComponentBackend for GateAt<'_> {
    fn gate_s(&mut self, a: usize, s: &Col<f64>, keep: Option<usize>) -> Gate {
        self.log.push(("s", a));
        if self.s_at == Some(a) {
            Gate::Unresolved
        } else {
            self.inner.gate_s(a, s, keep)
        }
    }
    fn score(&mut self, r: &Col<f64>) -> Scored {
        self.inner.score(r)
    }
    fn gate_tt(&mut self, a: usize, r: &Col<f64>, tt: f64) -> Gate {
        self.log.push(("tt", a));
        if self.tt_at == Some(a) {
            Gate::Unresolved
        } else {
            self.inner.gate_tt(a, r, tt)
        }
    }
    fn loading(&mut self, r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        self.inner.loading(r, t, inv_tt)
    }
    fn q(&mut self, s: &Col<f64>, w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        self.inner.q(s, w, t, inv_tt)
    }
}

fn gate_at(inp: &KernelInputs, s_at: Option<usize>, tt_at: Option<usize>) -> GateAt<'_> {
    GateAt {
        inner: XBackend {
            xs: inp.xs.as_ref(),
            ys: inp.ys.as_ref(),
            par: Par::Seq,
        },
        s_at,
        tt_at,
        log: Vec::new(),
    }
}

#[test]
fn loop_returns_unresolved_when_a_gate_says_so() {
    let (x, y) = signal_design(100, 20, 4, 0.1, 21);
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    let (s0, w_floor) = loop_inputs(&inp);
    for (s_at, tt_at) in [
        (Some(1), None),
        (Some(3), None),
        (None, Some(1)),
        (None, Some(4)),
    ] {
        let mut b = gate_at(&inp, s_at, tt_at);
        let out = pls1_component_loop(&mut b, s0.clone(), w_floor, 5, None);
        assert!(
            matches!(out, LoopOutcome::Unresolved),
            "s_at={s_at:?} tt_at={tt_at:?}"
        );
    }
    // No gate fires: the stub is the X backend, to the bit.
    let mut b = gate_at(&inp, None, None);
    let LoopOutcome::Done { t, p, w, q } = pls1_component_loop(&mut b, s0, w_floor, 5, None) else {
        panic!("expected Done");
    };
    let direct = pls1_kernel(
        inp.xs.as_ref(),
        inp.ys.as_ref(),
        5,
        None,
        Par::Seq,
        inp.xs.norm_l2(),
    )
    .unwrap();
    assert_mat_bits(&t.expect("the X backend forms T"), &direct.0, "T");
    assert_mat_bits(&p, &direct.1, "P");
    assert_mat_bits(&w, &direct.2, "W");
    assert_col_bits(&q, &direct.3, "Q");
}

#[test]
fn gates_see_one_based_component_indices_in_order() {
    let (x, y) = signal_design(100, 20, 4, 0.1, 22);
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    let (s0, w_floor) = loop_inputs(&inp);
    let mut b = gate_at(&inp, None, None);
    let out = pls1_component_loop(&mut b, s0, w_floor, 3, None);
    assert!(matches!(out, LoopOutcome::Done { .. }));
    assert_eq!(
        b.log,
        vec![
            ("s", 1),
            ("tt", 1),
            ("s", 2),
            ("tt", 2),
            ("s", 3),
            ("tt", 3)
        ]
    );

    // On a design that truncates before reaching the requested k, gate_s
    // fires once more (at k_used + 1) to see the stop, and no gate_tt call
    // follows it: a downstream Gram backend relies on gate_s alone signaling
    // the stopping selection.
    let (x, y) = exhausted_design(2000, 11);
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    let (s0, w_floor) = loop_inputs(&inp);
    let mut b = gate_at(&inp, None, None);
    let out = pls1_component_loop(&mut b, s0, w_floor, 40, None);
    let LoopOutcome::Done { w, .. } = out else {
        panic!("expected Done");
    };
    let k_used = w.ncols();
    assert_eq!(b.log.last(), Some(&("s", k_used + 1)), "log: {:?}", b.log);
    assert!(
        !b.log.contains(&("tt", k_used + 1)),
        "no gate_tt call after the stopping gate_s: {:?}",
        b.log
    );

    // When a gate itself returns Unresolved, nothing runs after it either.
    let (x, y) = signal_design(100, 20, 4, 0.1, 22);
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    let (s0, w_floor) = loop_inputs(&inp);
    let mut b = gate_at(&inp, Some(2), None);
    let out = pls1_component_loop(&mut b, s0, w_floor, 3, None);
    assert!(matches!(out, LoopOutcome::Unresolved));
    assert_eq!(b.log.last(), Some(&("s", 2)), "log: {:?}", b.log);
}

// ── empty and truncated fits ──────────────────────────────────────────────

#[test]
fn zero_component_fit_returns_empty_factor_matrices() {
    // A constant y standardizes to zero, so s_1 = 0 and no component exists.
    let x = uniform_mat(40, 5, 71);
    let y = Col::<f64>::from_fn(40, |_| 3.0);
    let m = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )
    .unwrap();
    assert_eq!(m.k_used, 0);
    assert_eq!((m.t_scores.nrows(), m.t_scores.ncols()), (40, 0), "T");
    assert_eq!((m.p_loadings.nrows(), m.p_loadings.ncols()), (5, 0), "P");
    assert_eq!((m.w_star.nrows(), m.w_star.ncols()), (5, 0), "W");
    assert_eq!(m.q_loadings.nrows(), 0, "Q");
    // Straight from the kernel, the same shapes as the reference's.
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    let new = pls1_kernel(
        inp.xs.as_ref(),
        inp.ys.as_ref(),
        2,
        None,
        Par::Seq,
        inp.xs.norm_l2(),
    )
    .unwrap();
    let old = nipals_pls1_reference(inp.xs.as_ref(), inp.ys.as_ref(), 2, None, Par::Seq).unwrap();
    assert_factors_close(&new, &old, &[], "zero-component");
}

#[test]
fn truncated_fit_matches_reference_on_the_exhausted_design() {
    let (x, y) = exhausted_design(2000, 11);
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    let new = pls1_kernel(
        inp.xs.as_ref(),
        inp.ys.as_ref(),
        40,
        None,
        Par::Seq,
        inp.xs.norm_l2(),
    )
    .unwrap();
    let old = nipals_pls1_reference(inp.xs.as_ref(), inp.ys.as_ref(), 40, None, Par::Seq).unwrap();
    let k_used = new.2.ncols();
    assert!((5..=20).contains(&k_used), "k_used = {k_used}");
    assert_eq!(
        (new.0.nrows(), new.0.ncols()),
        (2000, k_used),
        "T truncated"
    );
    assert_eq!((new.1.nrows(), new.1.ncols()), (40, k_used), "P truncated");
    let tols = factor_tolerances(&inp, &old, None, 1e-10);
    assert_factors_close(&new, &old, &tols, "exhausted");
    let (cn, co) = (coef_of(&new, Par::Seq), coef_of(&old, Par::Seq));
    assert!(
        coef_drift(&cn, &co) <= 1e-10,
        "coef drift {:e}",
        coef_drift(&cn, &co)
    );
}

// ── property test: random designs against the reference ──────────────────

#[derive(Clone, Copy, Debug)]
enum Family {
    Dense,
    Sparse,
    Weighted,
    WeightedSparse,
}

struct Design {
    label: String,
    x: Mat<f64>,
    y: Col<f64>,
    w: Option<Col<f64>>,
    k: usize,
    keep: Option<usize>,
}

/// Log-uniform integer in `[lo, hi]`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn log_uniform(rng: &mut ChaCha8Rng, lo: usize, hi: usize) -> usize {
    if lo >= hi {
        return lo;
    }
    let v = rng
        .random_range((lo as f64).ln()..=(hi as f64).ln())
        .exp()
        .round();
    (v as usize).clamp(lo, hi)
}

/// Makes every column of `x` orthogonal to `e − ē` in the `w`-weighted
/// inner product (`ē` the `w`-weighted mean of `e`, `w = 1` when `None`)
/// and returns `e − ē`. `pls1_fit`'s weighted centering and scaling of the
/// columns and its √w row scaling keep the orthogonality, so the kernel
/// sees a `ys` orthogonal to every column of `Xs` up to rounding.
fn orthogonalize_columns(x: &mut Mat<f64>, e: &[f64], w: Option<&Col<f64>>) -> Vec<f64> {
    let n = x.nrows();
    let wt = |i: usize| w.map_or(1.0, |w| w[i]);
    let sw: f64 = (0..n).map(wt).sum();
    let mean = (0..n).map(|i| wt(i) * e[i]).sum::<f64>() / sw;
    let ec: Vec<f64> = e.iter().map(|v| v - mean).collect();
    let ee: f64 = (0..n).map(|i| wt(i) * ec[i] * ec[i]).sum();
    for j in 0..x.ncols() {
        let c = (0..n).map(|i| wt(i) * ec[i] * x[(i, j)]).sum::<f64>() / ee;
        for i in 0..n {
            x[(i, j)] -= c * ec[i];
        }
    }
    ec
}

/// One random design. Families cycle dense, sparse, weighted, weighted
/// sparse; one design in three is rank-deficient (last column the sum of
/// the first two); noise cycles `1e-3`, `0.1`, `1.0`; weighted designs zero
/// about 15% of the rows past the first `k + 2`. One design in five is
/// weak-signal: `y` orthogonal to every (weighted, centered) column of `X`
/// plus `1e-6·Σ_{j<m} b_j x_j`, so the cross-products `s_a` start a few
/// orders of magnitude above the rounding floor and decay to it within the
/// fit.
fn random_design(rng: &mut ChaCha8Rng, idx: usize, n_max: usize, d_max: usize) -> Design {
    let family = [
        Family::Dense,
        Family::Sparse,
        Family::Weighted,
        Family::WeightedSparse,
    ][idx % 4];
    let n = log_uniform(rng, 5, n_max);
    let d = log_uniform(rng, 1, d_max);
    let k = rng.random_range(1..=20_usize.min(d).min(n - 1));
    let mut x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let rank_deficient = d >= 3 && idx % 3 == 1;
    let weak = idx % 5 == 2;
    let m = rng.random_range(1..=d.min(10));
    let sigma = [1e-3, 0.1, 1.0][(idx / 4) % 3];
    let b: Vec<f64> = (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();
    let e: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let sparse = matches!(family, Family::Sparse | Family::WeightedSparse);
    let weighted = matches!(family, Family::Weighted | Family::WeightedSparse);
    let keep = if sparse {
        Some(rng.random_range(1..=d))
    } else {
        None
    };
    let w = if weighted {
        Some(Col::<f64>::from_fn(n, |i| {
            if i >= k + 2 && rng.random_range(0.0..1.0) < 0.15 {
                0.0
            } else {
                rng.random_range(0.2..3.0)
            }
        }))
    } else {
        None
    };
    // Every random draw above; the rest is arithmetic on them.
    let e = if weak {
        orthogonalize_columns(&mut x, &e, w.as_ref())
    } else {
        e
    };
    if rank_deficient {
        for i in 0..n {
            x[(i, d - 1)] = x[(i, 0)] + x[(i, 1)];
        }
    }
    let signal = |i: usize| {
        b.iter()
            .enumerate()
            .map(|(j, bj)| x[(i, j)] * bj)
            .sum::<f64>()
    };
    let y = if weak {
        Col::<f64>::from_fn(n, |i| e[i] + 1e-6 * signal(i))
    } else {
        Col::<f64>::from_fn(n, |i| signal(i) + sigma * e[i])
    };
    Design {
        label: format!("design {idx}: {family:?} n={n} d={d} k={k} keep={keep:?} rank_deficient={rank_deficient} weak={weak} sigma={sigma}"),
        x,
        y,
        w,
        k,
        keep,
    }
}

/// Distance between the `keep`-th and `(keep+1)`-th largest `|s|`.
fn keep_gap(s: &Col<f64>, keep: usize) -> f64 {
    let mut a: Vec<f64> = (0..s.nrows()).map(|i| s[i].abs()).collect();
    a.sort_by(|u, v| v.total_cmp(u));
    a[keep - 1] - a[keep]
}

/// The rows where column `a` of `W` is non-zero: the support `keep` selected.
#[allow(clippy::float_cmp)] // exact zeros of hard_select_keep define the support
fn support(w: &Mat<f64>, a: usize) -> Vec<usize> {
    (0..w.nrows()).filter(|&j| w[(j, a)] != 0.0).collect()
}

/// The first component (0-based) at which the two kernels selected
/// different supports, if any.
fn first_support_split(new: &Parts, old: &Parts) -> Option<usize> {
    (0..new.2.ncols().min(old.2.ncols())).find(|&a| support(&new.2, a) != support(&old.2, a))
}

/// One comparison. `None` when the design is excluded: the kernels selected
/// different supports at some component, which is allowed only where the
/// reference's keep-boundary gap at that component is within rounding
/// (`max(n, d)·ε·‖Xs‖_F·‖ys‖`, the relative floor); a split at a wider gap
/// fails. Otherwise the worst `diff / tol` of the factors, and the `coef`
/// and `beta` drift.
fn compare_with_reference(ds: &Design) -> Option<[f64; 3]> {
    let inp = kernel_inputs(ds.x.as_ref(), ds.y.as_ref(), ds.w.as_ref().map(Col::as_ref));
    let new = pls1_kernel(
        inp.xs.as_ref(),
        inp.ys.as_ref(),
        ds.k,
        ds.keep,
        Par::Seq,
        inp.xs.norm_l2(),
    )
    .expect("kernel");
    let old = nipals_pls1_reference(inp.xs.as_ref(), inp.ys.as_ref(), ds.k, ds.keep, Par::Seq)
        .expect("reference");
    if let Some(keep) = ds.keep {
        if let Some(a) = first_support_split(&new, &old) {
            let (_, floor) = loop_inputs(&inp);
            let gap = keep_gap(&s_sequence(&inp, &old)[a], keep);
            assert!(
                gap <= floor,
                "{}: supports differ at component {} although the keep-boundary gap {gap:e} is above the rounding scale {floor:e}",
                ds.label,
                a + 1
            );
            return None;
        }
    }
    let tols = factor_tolerances(&inp, &old, ds.keep, 1e-10);
    let factors = assert_factors_close(&new, &old, &tols, &ds.label);
    let (cn, co) = (coef_of(&new, Par::Seq), coef_of(&old, Par::Seq));
    let coef = coef_drift(&cn, &co);
    let beta = coef_drift(&beta_of(&cn, &inp), &beta_of(&co, &inp));
    assert!(coef <= 1e-10, "{}: coef drift {coef:e}", ds.label);
    assert!(beta <= 1e-10, "{}: beta drift {beta:e}", ds.label);
    Some([factors, coef, beta])
}

fn run_property(seed: u64, n_designs: usize, n_max: usize, d_max: usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (mut sparse, mut excluded) = (0_usize, 0_usize);
    let mut worst = [0.0_f64; 3];
    for idx in 0..n_designs {
        let ds = random_design(&mut rng, idx, n_max, d_max);
        if ds.keep.is_some() {
            sparse += 1;
        }
        match compare_with_reference(&ds) {
            None => excluded += 1,
            Some(v) => {
                for (w, x) in worst.iter_mut().zip(v) {
                    *w = w.max(x);
                }
            }
        }
    }
    eprintln!(
        "ikpls_vs_nipals designs={n_designs} sparse={sparse} excluded_support_split={excluded} \
         worst_factor_ratio={:.2} worst_coef_drift={:.1e} worst_beta_drift={:.1e}",
        worst[0], worst[1], worst[2]
    );
    assert!(
        excluded * 10 <= sparse,
        "{excluded} of {sparse} sparse designs excluded: the support-split exclusion would hide real disagreements"
    );
}

#[test]
fn kernel_matches_reference_on_random_designs_full() {
    // Ranges n to 2000, d to 5000, k to 20; a few seconds in release.
    run_property(20_260_926, 400, 2000, 5000);
}

// ── K = 1 through the public path, bit for bit ────────────────────────────

#[test]
fn k1_public_fit_is_bit_identical_to_reference() {
    // (2000, 600): n·d·k ≥ 1e6, so ParChoice::Auto resolves to `par_fixed()`.
    for (n, d) in [(60, 12), (2000, 600)] {
        let (x, y) = signal_design(n, d, 3, 0.1, 17);
        // The offset puts |mean|/scale past `IMPLICIT_MAX_MEAN_RATIO`, so the
        // public fit runs the kernel on the standardized copy the reference
        // gets. The implicit route is checked against the kernel in
        // `half_zero_weights_match_reference_dense_and_sparse` and pinned by
        // the corpus.
        let x = Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)] + 1e6);
        let w = mixed_weights(n);
        for weights in [None, Some(w.as_ref())] {
            let inp = kernel_inputs(x.as_ref(), y.as_ref(), weights);
            for keep in [None, Some(4), Some(d)] {
                for choice in [ParChoice::Seq, ParChoice::Auto] {
                    let what = format!(
                        "n={n} d={d} weighted={} keep={keep:?} {choice:?}",
                        weights.is_some()
                    );
                    let opts = FitOpts {
                        par: choice,
                        keep,
                        ..FitOpts::default()
                    };
                    let m =
                        pls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(1), weights, opts).unwrap();
                    let par = resolve_par(choice, n, d, 1);
                    let old = nipals_pls1_reference(inp.xs.as_ref(), inp.ys.as_ref(), 1, keep, par)
                        .unwrap();
                    let coef = coef_of(&old, par);
                    assert_mat_bits(&m.t_scores, &old.0, &format!("{what} T"));
                    assert_mat_bits(&m.p_loadings, &old.1, &format!("{what} P"));
                    assert_mat_bits(&m.w_star, &old.2, &format!("{what} W"));
                    assert_col_bits(&m.q_loadings, &old.3, &format!("{what} Q"));
                    assert_col_bits(&m.coef, &coef, &format!("{what} coef"));
                    assert_col_bits(&m.beta, &beta_of(&coef, &inp), &format!("{what} beta"));
                }
            }
        }
    }
}

// ── views, ties, zero weights ─────────────────────────────────────────────

#[test]
fn non_column_major_views_agree_with_reference() {
    let (n, d) = (80, 15);
    let (x, y) = signal_design(n, d, 3, 0.1, 11);
    let (xs, _, _) = crate::linalg::standardize(x.as_ref());
    let (ys, _, _) = crate::linalg::standardize1(y.as_ref());
    let xt = xs.transpose().to_owned();
    let xsr = xs.as_ref();
    let padded: Vec<f64> = (0..d)
        .flat_map(|j| (0..n + 3).map(move |i| if i < n { xsr[(i, j)] } else { 0.0 }))
        .collect();
    let ys_rev = Col::<f64>::from_fn(n, |i| ys[n - 1 - i]);
    let views: [(&str, MatRef<'_, f64>, ColRef<'_, f64>); 3] = [
        (
            "row-major (transposed)",
            xt.as_ref().transpose(),
            ys.as_ref(),
        ),
        (
            "padded column-major (strided)",
            MatRef::from_column_major_slice_with_stride(&padded, n, d, n + 3),
            ys.as_ref(),
        ),
        (
            "reversed rows (negative stride)",
            xsr.reverse_rows(),
            ys_rev.as_ref(),
        ),
    ];
    for (label, xv, yv) in views {
        for k in [1_usize, 3] {
            let what = format!("{label}, k={k}");
            let opts = FitOpts {
                pre_standardized: true,
                check_n_eff: false,
                par: ParChoice::Seq,
                keep: None,
            };
            let m = pls1_fit(xv, yv, KSpec::Fixed(k), None, opts).expect("fit");
            let old = nipals_pls1_reference(xv, yv, k, None, Par::Seq).expect("reference");
            let new: Parts = (
                m.t_scores.clone(),
                m.p_loadings.clone(),
                m.w_star.clone(),
                m.q_loadings.clone(),
            );
            let inp = KernelInputs {
                xs: xv.to_owned(),
                ys: yv.to_owned(),
                x_scale: Col::<f64>::from_fn(d, |_| 1.0),
                y_scale: 1.0,
            };
            // Within 1e-12 at k = 1; the corpus array tolerance above.
            let tol = if k == 1 { 1e-12 } else { 1e-10 };
            let tols = factor_tolerances(&inp, &old, None, tol);
            assert_factors_close(&new, &old, &tols, &what);
            let drift = coef_drift(&m.coef, &coef_of(&old, Par::Seq));
            assert!(drift <= tol, "{what}: coef drift {drift:e}");
            assert_col_bits(
                &m.beta,
                &m.coef,
                &format!("{what}: beta = coef when pre_standardized"),
            );
        }
    }
}

#[test]
fn sparse_exact_ties_pick_the_reference_support_at_every_component() {
    // Columns 1 and 3 duplicate columns 0 and 2 exactly, so their |s_a| tie
    // exactly at every component and the lowest-index rule decides.
    let n = 120;
    let base = uniform_mat(n, 6, 51);
    let src = [0_usize, 0, 2, 2, 1, 3, 4, 5];
    let x = Mat::<f64>::from_fn(n, src.len(), |i, j| base[(i, src[j])]);
    let noise = uniform_mat(n, 1, 52);
    let y = Col::<f64>::from_fn(n, |i| {
        x[(i, 0)] + 0.5 * x[(i, 2)] + 0.3 * x[(i, 4)] + 0.05 * noise[(i, 0)]
    });
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
    for keep in [1_usize, 3, 5] {
        let new = pls1_kernel(
            inp.xs.as_ref(),
            inp.ys.as_ref(),
            3,
            Some(keep),
            Par::Seq,
            inp.xs.norm_l2(),
        )
        .unwrap();
        let old = nipals_pls1_reference(inp.xs.as_ref(), inp.ys.as_ref(), 3, Some(keep), Par::Seq)
            .unwrap();
        assert_eq!(new.2.ncols(), old.2.ncols(), "keep={keep}: k_used");
        assert_eq!(
            first_support_split(&new, &old),
            None,
            "keep={keep}: supports"
        );
        let tols = factor_tolerances(&inp, &old, Some(keep), 1e-10);
        assert_factors_close(&new, &old, &tols, &format!("ties keep={keep}"));
    }
}

#[test]
fn half_zero_weights_match_reference_dense_and_sparse() {
    let n = 200;
    let (x, y) = signal_design(n, 30, 4, 0.2, 61);
    let w = Col::<f64>::from_fn(n, |i| {
        if i % 2 == 0 {
            0.0
        } else {
            0.5 + (i % 5) as f64 * 0.3
        }
    });
    let inp = kernel_inputs(x.as_ref(), y.as_ref(), Some(w.as_ref()));
    for keep in [None, Some(7)] {
        let what = format!("half-zero weights keep={keep:?}");
        let new = pls1_kernel(
            inp.xs.as_ref(),
            inp.ys.as_ref(),
            6,
            keep,
            Par::Seq,
            inp.xs.norm_l2(),
        )
        .unwrap();
        let old =
            nipals_pls1_reference(inp.xs.as_ref(), inp.ys.as_ref(), 6, keep, Par::Seq).unwrap();
        let tols = factor_tolerances(&inp, &old, keep, 1e-10);
        assert_factors_close(&new, &old, &tols, &what);
        let (cn, co) = (coef_of(&new, Par::Seq), coef_of(&old, Par::Seq));
        assert!(coef_drift(&cn, &co) <= 1e-10, "{what}: coef");
        assert!(
            coef_drift(&beta_of(&cn, &inp), &beta_of(&co, &inp)) <= 1e-10,
            "{what}: beta"
        );
        // The public path forms the same products from the raw X (the
        // implicit backend), so it agrees to rounding.
        // n·d·k = 36000 < 1e6, so ParChoice::Auto resolves to Par::Seq.
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(6),
            Some(w.as_ref()),
            FitOpts {
                keep,
                ..FitOpts::default()
            },
        )
        .unwrap();
        assert!(coef_drift(&m.coef, &cn) <= 1e-10, "{what}: pls1_fit coef");
        // The factors the model returns, not only `coef`: T, P and Q of the
        // implicit backend against the copy kernel, per component.
        let public: Parts = (
            m.t_scores.clone(),
            m.p_loadings.clone(),
            new.2.clone(),
            m.q_loadings.clone(),
        );
        assert_factors_close(&public, &new, &tols, &format!("{what}: pls1_fit factors"));
    }
}

// ── floor calibration (`w_rel_floor`) ─────────────────────────────────────

/// A component contributes when its share of `‖ys‖` in the fitted values,
/// `|q_a|·‖t_a‖ / ‖ys‖`, is at least this. The threshold `w_rel_floor`'s doc uses.
const CONTRIB_REAL: f64 = 1e-9;

/// The selected `‖s_a‖` and the contribution of each kept component of an
/// unfloored run (`w_floor = 0`: the `1e-14` absolute floor only), and the
/// relative floor the kernel would have applied.
struct Trace {
    w_norm: Vec<f64>,
    contrib: Vec<f64>,
    floor: f64,
}

fn unfloored_trace(inp: &KernelInputs, k: usize, keep: Option<usize>) -> Trace {
    let (s0, floor) = loop_inputs(inp);
    let mut backend = XBackend {
        xs: inp.xs.as_ref(),
        ys: inp.ys.as_ref(),
        par: Par::Seq,
    };
    let LoopOutcome::Done { t, p, w, q } = pls1_component_loop(&mut backend, s0, 0.0, k, keep)
    else {
        unreachable!("the X backend never returns Unresolved")
    };
    let n = inp.xs.nrows();
    let parts: Parts = (t.unwrap_or_else(|| Mat::<f64>::zeros(n, 0)), p, w, q);
    let seq = s_sequence(inp, &parts);
    let y_norm = inp.ys.norm_l2();
    let k_used = parts.3.nrows();
    let w_norm = seq[..k_used]
        .iter()
        .map(|s| selected_norm(s, keep))
        .collect();
    let contrib = (0..k_used)
        .map(|a| parts.3[a].abs() * parts.0.col(a).to_owned().norm_l2() / y_norm)
        .collect();
    Trace {
        w_norm,
        contrib,
        floor,
    }
}

/// What a design's construction says about its components, in exact arithmetic.
#[derive(Clone, Copy)]
enum Structure {
    /// `s = 0` exactly from 0-based component `a0` on: the components before
    /// it are real, component `a0` is the first noise component. `y` in the
    /// span of `m` singular directions of `Xs` (`a0 = m`), `y` orthogonal to
    /// `Xs` (`a0 = 0`), or `Xs` fully deflated at its rank (`a0 = rank`).
    /// Records all three sides.
    NoiseAt(usize),
    /// The real components decay into rounding long before any structural
    /// noise index (the `y`-exhausted rank-39 design, the weak-signal
    /// design). Records only the contributing side.
    ContributingOnly,
}

struct Margins {
    first_noise_max: f64,
    contributing_min: f64,
    dropped_max: f64,
    designs: usize,
    with_noise: usize,
}

impl Margins {
    fn new() -> Self {
        Self {
            first_noise_max: 0.0,
            contributing_min: f64::INFINITY,
            dropped_max: 0.0,
            designs: 0,
            with_noise: 0,
        }
    }

    fn record(&mut self, inp: &KernelInputs, k: usize, keep: Option<usize>, structure: Structure) {
        let tr = unfloored_trace(inp, k, keep);
        let k_unfloored = tr.w_norm.len();
        // Real components end at the known noise index (or with the run).
        let real_end = match structure {
            Structure::NoiseAt(a0) => a0.min(k_unfloored),
            Structure::ContributingOnly => k_unfloored,
        };
        // Contributing side: the leading real components that contribute at
        // least CONTRIB_REAL must clear the floor.
        let lead = tr.contrib[..real_end]
            .iter()
            .position(|&c| c < CONTRIB_REAL)
            .unwrap_or(real_end);
        for wn in &tr.w_norm[..lead] {
            self.contributing_min = self.contributing_min.min(wn / tr.floor);
        }
        if let Structure::NoiseAt(a0) = structure {
            // Noise side, only when every real component contributes: after
            // a real component that has decayed into rounding, the unfloored
            // run continues on noise and `‖s‖` regrows, so the value at `a0`
            // is no longer the first noise component's.
            if a0 < k_unfloored && lead == a0 {
                self.first_noise_max = self.first_noise_max.max(tr.w_norm[a0] / tr.floor);
                self.with_noise += 1;
            }
            // Real components the floor drops: the floored run shares the
            // unfloored run's first `k_floored` components exactly. Counted
            // down to the smallest unfloored `‖s_a‖` before the noise index,
            // past which the unfloored run is rounding noise.
            let k_floored = pls1_kernel(
                inp.xs.as_ref(),
                inp.ys.as_ref(),
                k,
                keep,
                Par::Seq,
                inp.xs.norm_l2(),
            )
            .expect("kernel")
            .3
            .nrows();
            if k_floored < real_end {
                let bottom = (k_floored..real_end)
                    .min_by(|&a, &b| tr.w_norm[a].total_cmp(&tr.w_norm[b]))
                    .expect("k_floored < real_end");
                for c in &tr.contrib[k_floored..=bottom] {
                    self.dropped_max = self.dropped_max.max(*c);
                }
            }
        }
        self.designs += 1;
    }
}

/// `y` an exact combination of `m` left singular vectors of the
/// standardized `X`: `m` real components, then noise (unweighted, dense).
fn span_design(n: usize, d: usize, m: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
    let x = uniform_mat(n, d, seed);
    let (xs, _, _) = crate::linalg::standardize(x.as_ref());
    let svd = xs.thin_svd().expect("svd");
    let u = svd.U();
    let c = uniform_mat(m, 1, seed + 1);
    let y = Col::<f64>::from_fn(n, |i| {
        (0..m)
            .map(|j| (1.0 + c[(j, 0)]) * u[(i, 2 * j)])
            .sum::<f64>()
    });
    (x, y)
}

/// A `y` orthogonal to the standardized `X` plus `signal` times `X b`:
/// with `signal = 0` the first component is already noise, with
/// `signal = 1e-6` the real components are weak (the design of
/// `nipals_floor_is_relative_*`).
fn orthogonal_design(n: usize, d: usize, signal: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    let x = uniform_mat(n, d, seed);
    let (xs, _, _) = crate::linalg::standardize(x.as_ref());
    let svd = xs.thin_svd().expect("svd");
    let u = svd.U();
    let e = uniform_mat(n, 1, seed + 1);
    let mut e_perp = Col::<f64>::from_fn(n, |i| e[(i, 0)]);
    for j in 0..d {
        let c: f64 = (0..n).map(|i| u[(i, j)] * e_perp[i]).sum();
        for i in 0..n {
            e_perp[i] -= c * u[(i, j)];
        }
    }
    let b = uniform_mat(d, 1, seed + 2);
    let y = Col::<f64>::from_fn(n, |i| {
        e_perp[i] + signal * (0..d).map(|j| xs[(i, j)] * b[(j, 0)]).sum::<f64>()
    });
    (x, y)
}

#[test]
#[allow(clippy::too_many_lines)]
fn floor_calibration_sweep() {
    // Re-run when the PLS1 kernel changes, then update w_rel_floor's doc comment.
    let row_weights = |n: usize| Col::<f64>::from_fn(n, |i| 0.5 + (i % 7) as f64 * 0.25);
    let random_y = |n: usize, seed: u64| {
        let yc = uniform_mat(n, 1, seed);
        Col::<f64>::from_fn(n, |i| yc[(i, 0)])
    };
    let mut m = Margins::new();
    for seed in 0..4_u64 {
        let base = 1000 * seed;
        // Rank-deficient, y exhausted (rank 39 of 40): dense and sparse, weighted and not.
        for n in [200, 2000, 20_000] {
            let (x, y) = exhausted_design(n, base + 11);
            for w in [None, Some(row_weights(n))] {
                let inp = kernel_inputs(x.as_ref(), y.as_ref(), w.as_ref().map(Col::as_ref));
                m.record(&inp, 40, None, Structure::ContributingOnly);
                m.record(&inp, 40, Some(10), Structure::ContributingOnly);
            }
        }
        // y in the span of `dirs` singular directions: noise from component
        // `dirs + 1` on the unweighted dense fit. Weighting or `keep` breaks
        // that structure, and X (rank d, n > d) is then exhausted only at d.
        for (n, d) in [
            (100, 10),
            (100, 30),
            (500, 10),
            (500, 30),
            (5000, 10),
            (5000, 30),
        ] {
            for dirs in [1_usize, 2, 5] {
                let (x, y) = span_design(n, d, dirs, base + 21);
                let k = d.min(dirs + 5);
                let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
                m.record(&inp, k, None, Structure::NoiseAt(dirs));
                m.record(&inp, k, Some(d / 2), Structure::NoiseAt(d));
                let inp = kernel_inputs(x.as_ref(), y.as_ref(), Some(row_weights(n).as_ref()));
                m.record(&inp, k, None, Structure::NoiseAt(d));
            }
        }
        // y orthogonal to X (noise from the first component), and weak signal.
        for n in [1000, 10_000] {
            let (x, y) = orthogonal_design(n, 20, 0.0, base + 31);
            m.record(
                &kernel_inputs(x.as_ref(), y.as_ref(), None),
                3,
                None,
                Structure::NoiseAt(0),
            );
            let (x, y) = orthogonal_design(n, 20, 1e-6, base + 31);
            m.record(
                &kernel_inputs(x.as_ref(), y.as_ref(), None),
                20,
                None,
                Structure::ContributingOnly,
            );
        }
        // p ≫ n: X centered has rank n − 1 and spans y, so components past
        // n − 1 are noise; dense and sparse.
        for n in [5_usize, 20, 60] {
            for d in [200_usize, 3000] {
                let x = uniform_mat(n, d, base + 41);
                let y = random_y(n, base + 42);
                let inp = kernel_inputs(x.as_ref(), y.as_ref(), None);
                let k = (n + 2).min(20);
                m.record(&inp, k, None, Structure::NoiseAt(n - 1));
                m.record(&inp, k, Some(d / 10), Structure::NoiseAt(n - 1));
            }
        }
        // Tiny: X exhausted at its rank min(d, n − 1).
        for n in [5_usize, 8] {
            for d in [3_usize, 6] {
                let x = uniform_mat(n, d, base + 51);
                let y = random_y(n, base + 52);
                m.record(
                    &kernel_inputs(x.as_ref(), y.as_ref(), None),
                    d,
                    None,
                    Structure::NoiseAt(d.min(n - 1)),
                );
            }
        }
    }
    // The largest n, where the floor is highest.
    let (x, y) = span_design(200_000, 10, 3, 61);
    m.record(
        &kernel_inputs(x.as_ref(), y.as_ref(), None),
        10,
        None,
        Structure::NoiseAt(3),
    );

    // The doc comment's illustration: the exhausted design at n = 2000, unfloored.
    let (x, y) = exhausted_design(2000, 11);
    let tr = unfloored_trace(&kernel_inputs(x.as_ref(), y.as_ref(), None), 40, None);
    let (min_at, min_w) =
        tr.w_norm
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |acc, (a, v)| if v < acc.1 { (a, v) } else { acc },
            );
    let regrow = tr.w_norm[min_at..].iter().copied().fold(0.0_f64, f64::max);

    eprintln!(
        "floor_sweep first_noise_max={:.3} contributing_min={:.3} dropped_max={:.1e} designs={} with_noise={}",
        m.first_noise_max, m.contributing_min, m.dropped_max, m.designs, m.with_noise
    );
    eprintln!(
        "floor_sweep exhausted_min_w_norm={min_w:.1e} exhausted_min_at={} exhausted_regrow_max={regrow:.3e} exhausted_k_kept={}",
        min_at + 1,
        tr.w_norm.len()
    );
    assert!(
        m.with_noise >= 50,
        "only {} designs reached a noise component",
        m.with_noise
    );
    assert!(
        m.first_noise_max <= 0.1,
        "a noise component approached the floor: {}× (documented value: 0.020×)",
        m.first_noise_max
    );
    assert!(
        m.contributing_min >= 10.0,
        "a real component drifted toward the floor: {}× (documented value: 38.8×)",
        m.contributing_min
    );
}
