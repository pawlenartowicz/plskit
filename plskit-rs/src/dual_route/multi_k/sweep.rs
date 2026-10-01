//! The n-space kernel's validation sweep: gate soundness, per-replicate
//! agreement with the primal kernel, fallback rates, and the score
//! discrepancy `SCORE_BAND` is set from. `cargo test -p plskit --release
//! multi_k::sweep -- --nocapture` prints the report.
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]

use std::collections::BTreeMap;

use super::*;
use crate::dual_route::SCORE_BAND;
use crate::fit::{pls1_fit_prepared_fro, ParChoice};
use crate::linalg::{standardize, standardize1, standardize_apply};
use crate::test_support::{orthonormal_basis, project_off};
use faer::{Col, Mat, Par};
use rand::{RngExt, SeedableRng};

/// Soundness and agreement are checked for every `k` up to this, whatever
/// `K_DUAL_MAX` is: the gates must fail closed at every `k`.
const K_SWEEP_MAX: usize = 10;

struct Design {
    family: &'static str,
    xs: Mat<f64>,
    z: Col<f64>,
    k_min: usize,
}

#[derive(Default)]
struct Tally {
    resolved: usize,
    zero: usize,
    unresolved: usize,
    worst_rel: f64,
}

fn uniform(n: usize, p: usize, seed: u64) -> Mat<f64> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0))
}

fn noise(n: usize, seed: u64) -> Col<f64> {
    let e = uniform(n, 1, seed);
    Col::<f64>::from_fn(n, |i| e[(i, 0)])
}

fn std_x(x: &Mat<f64>) -> Mat<f64> {
    standardize(x.as_ref()).0
}

fn std_y(y: &Col<f64>) -> Col<f64> {
    standardize1(y.as_ref()).0
}

/// Rank-`r` matrix `A·B` with uniform factors.
fn low_rank(n: usize, p: usize, r: usize, seed: u64) -> Mat<f64> {
    let a = uniform(n, r, seed);
    let b = uniform(r, p, seed + 1);
    a.as_ref() * b.as_ref()
}

fn signal_outcome(xs: &Mat<f64>, snr: f64, seed: u64) -> Col<f64> {
    let e = noise(xs.nrows(), seed);
    std_y(&Col::<f64>::from_fn(xs.nrows(), |i| {
        snr * xs[(i, 0)] + e[i]
    }))
}

/// `Σ_{j<m} c_j·u_j` over the first `m` left singular vectors of `xs`,
/// together with `U` and the singular values.
fn in_span(xs: &Mat<f64>, m: usize) -> (Col<f64>, Mat<f64>, Col<f64>) {
    let svd = xs.thin_svd().expect("svd");
    let u = svd.U().to_owned();
    let s = svd.S().column_vector().to_owned();
    let coefs = [1.3, -0.7, 0.4];
    let z = Col::<f64>::from_fn(xs.nrows(), |i| {
        (0..m).map(|j| coefs[j] * u[(i, j)]).sum::<f64>()
    });
    (z, u, s)
}

/// Component `m` of an outcome placed at `c·√(BOUND_BAND·E_w(m))·‖z‖`, just
/// past gate 1 for `c` near 1, after `m − 1` strong components and before
/// exhaustion (the outcome's `X̃` part lies in `m` directions). The kernel's
/// own trace supplies `E_w(m)` and `wn_m`, which is linear in the weight
/// `δ` of the `m`-th direction while `δ` is small.
fn gate_edge_designs() -> Vec<Design> {
    let mut out = Vec::new();
    // Rank 6 with p ≫ n: an outcome part orthogonal to X̃ exists, so
    // component m can be weak relative to ‖z‖.
    let xs = std_x(&low_rank(40, 800, 6, 17));
    let (_, u, _) = in_span(&xs, 1);
    let ones = Col::<f64>::from_fn(40, |_| 1.0);
    let basis = orthonormal_basis(ones.as_ref(), xs.as_ref(), 1e-8);
    let mut e_perp = noise(40, 117);
    project_off(&basis, &mut e_perp);
    let gram = NspaceGram::new(xs.as_ref(), Par::Seq);
    let block = gram.block();
    let strong = [1.3, -0.7];
    for m in 1..=3_usize {
        let build = |delta: f64| {
            Col::<f64>::from_fn(40, |i| {
                (0..m - 1).map(|j| strong[j] * u[(i, j)]).sum::<f64>()
                    + delta * u[(i, m - 1)]
                    + e_perp[i]
            })
        };
        let delta0 = 1e-3;
        let z0 = build(delta0);
        let (_, trace) = pls1_nspace_kernel_traced(&block, z0.as_ref(), m);
        let tr = trace
            .get(m - 1)
            .unwrap_or_else(|| panic!("gate_edge m={m}: the kernel stopped before component m"));
        let wn0 = tr.wn2.max(0.0).sqrt();
        for c in [1.0, 1.1, 1.5, 3.0, 10.0, 30.0, 100.0] {
            let target = c * (BOUND_BAND * tr.e_w).sqrt() * z0.norm_l2();
            let z = build(delta0 * target / wn0);
            out.push(Design {
                family: "gate_edge",
                xs: xs.clone(),
                z,
                k_min: m,
            });
        }
    }
    out
}

#[allow(clippy::too_many_lines)]
fn designs() -> Vec<Design> {
    let mut out = Vec::new();
    for (n, p, seed) in [(30_usize, 2000_usize, 2_u64), (60, 3000, 3), (60, 3000, 4)] {
        let xs = std_x(&uniform(n, p, seed));
        let z = signal_outcome(&xs, 2.0, seed + 100);
        out.push(Design {
            family: "ordinary",
            xs,
            z,
            k_min: 1,
        });
    }
    {
        // n = 12: k reaches n − 2, so the last components sit near
        // exhaustion; reported, not held to zero fallbacks.
        let xs = std_x(&uniform(12, 400, 1));
        let z = signal_outcome(&xs, 2.0, 101);
        out.push(Design {
            family: "small_n",
            xs,
            z,
            k_min: 1,
        });
    }
    for seed in [5_u64, 6] {
        let xs = std_x(&uniform(30, 2000, seed));
        let z = signal_outcome(&xs, 1e-6, seed + 100);
        out.push(Design {
            family: "weak_signal",
            xs,
            z,
            k_min: 1,
        });
    }
    {
        let xs = std_x(&low_rank(40, 1500, 10, 7));
        let z = signal_outcome(&xs, 1.0, 107);
        out.push(Design {
            family: "rank_deficient",
            xs,
            z,
            k_min: 1,
        });
    }
    {
        let base = uniform(20, 1500, 8);
        let xs = std_x(&Mat::<f64>::from_fn(40, 1500, |i, j| base[(i % 20, j)]));
        let z = signal_outcome(&xs, 1.0, 108);
        out.push(Design {
            family: "duplicated_rows",
            xs,
            z,
            k_min: 1,
        });
    }
    for (n, seed) in [(6_usize, 9_u64), (8, 10)] {
        let xs = std_x(&uniform(n, 300, seed));
        let z = signal_outcome(&xs, 1.0, seed + 100);
        out.push(Design {
            family: "tiny_n",
            xs,
            z,
            k_min: 1,
        });
    }
    for seed in [11_u64, 12] {
        // Rank 5, so an outcome orthogonal to every column of X̃ exists.
        let xs = std_x(&low_rank(40, 800, 5, seed));
        let ones = Col::<f64>::from_fn(40, |_| 1.0);
        let basis = orthonormal_basis(ones.as_ref(), xs.as_ref(), 1e-8);
        let mut z = noise(40, seed + 100);
        project_off(&basis, &mut z);
        out.push(Design {
            family: "y_orthogonal",
            xs,
            z,
            k_min: 1,
        });
    }
    {
        // Rank 4: z's projection on the span of X̃ is fitted within four
        // components, after which ‖X_a'y_a‖ is rounding.
        let xs = std_x(&low_rank(40, 800, 4, 13));
        let z = std_y(&noise(40, 113));
        out.push(Design {
            family: "y_exhausted",
            xs,
            z,
            k_min: 1,
        });
    }
    for m in 1..=3 {
        let xs = std_x(&uniform(30, 1500, 14));
        let (z, _, _) = in_span(&xs, m);
        out.push(Design {
            family: "y_in_span",
            xs,
            z,
            k_min: 1,
        });
    }
    for c in [0.1, 0.3, 1.0, 3.0, 10.0] {
        // Component 3 carries about c × w_rel_floor.
        let xs = std_x(&uniform(30, 1500, 15));
        let (z0, u, s) = in_span(&xs, 2);
        let floor = crate::fit::w_rel_floor(30, 1500, xs.norm_l2(), z0.norm_l2());
        let delta = c * floor / s[2];
        let z = Col::<f64>::from_fn(30, |i| z0[i] + delta * u[(i, 2)]);
        out.push(Design {
            family: "boundary",
            xs,
            z,
            k_min: 3,
        });
    }
    for s in [1e-4, 1e-8, 1e-10, 1e-12] {
        // Tiny-scale X̃: the one input that reaches the absolute-floor gates.
        let base = std_x(&uniform(30, 1500, 16));
        let xs = Mat::<f64>::from_fn(30, 1500, |i, j| base[(i, j)] * s);
        let z = signal_outcome(&base, 1.0, 116);
        out.push(Design {
            family: "tiny_scale",
            xs,
            z,
            k_min: 1,
        });
    }
    out.extend(gate_edge_designs());
    out
}

#[test]
fn nspace_kernel_is_sound_and_matches_the_primal_on_every_design_family() {
    let mut tally: BTreeMap<(&'static str, usize), Tally> = BTreeMap::new();
    for d in designs() {
        let (n, p) = (d.xs.nrows(), d.xs.ncols());
        let gram = NspaceGram::new(d.xs.as_ref(), Par::Seq);
        let block = gram.block();
        for k in d.k_min..=K_SWEEP_MAX.min(n - 1) {
            let fit = pls1_fit_prepared_fro(
                d.xs.as_ref(),
                d.z.as_ref(),
                k,
                None,
                ParChoice::Seq,
                d.xs.norm_l2(),
            )
            .expect("primal fit");
            let t = tally.entry((d.family, k)).or_default();
            match pls1_nspace_kernel(&block, d.z.as_ref(), k) {
                NspaceOutcome::Resolved { alpha, k_used } => {
                    assert_eq!(k_used, k);
                    assert_eq!(
                        fit.k_used, k,
                        "gate soundness: {} n={n} p={p} k={k}: Resolved, but the primal keeps {}",
                        d.family, fit.k_used
                    );
                    let coef = seq_gemv(d.xs.transpose(), &alpha);
                    let scale = (0..p).map(|j| fit.coef[j].abs()).fold(1.0_f64, f64::max);
                    let diff = (0..p)
                        .map(|j| (coef[j] - fit.coef[j]).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(
                        diff <= 1e-10 * scale,
                        "equivalence: {} n={n} p={p} k={k}: |Δcoef| = {diff:e} (scale {scale:e})",
                        d.family
                    );
                    t.resolved += 1;
                    t.worst_rel = t.worst_rel.max(diff / scale);
                }
                NspaceOutcome::ZeroModel => {
                    assert_eq!(fit.k_used, 0, "zero model: {} k={k}", d.family);
                    t.zero += 1;
                }
                NspaceOutcome::Unresolved => t.unresolved += 1,
            }
        }
    }
    eprintln!("family             k  resolved  zero  to_primal  worst |Δcoef|/max(1,|coef|)");
    for ((family, k), t) in &tally {
        eprintln!(
            "{family:<17} {k:>2}  {:>8}  {:>4}  {:>9}  {:.1e}",
            t.resolved, t.zero, t.unresolved, t.worst_rel
        );
    }
    let (res, zero, unres) = tally.values().fold((0, 0, 0), |a, t| {
        (a.0 + t.resolved, a.1 + t.zero, a.2 + t.unresolved)
    });
    let served = |family: &str, k: usize| family == "ordinary" && k <= K_DUAL_MAX;
    let ordinary: Vec<&Tally> = tally
        .iter()
        .filter(|((family, k), _)| served(family, *k))
        .map(|(_, t)| t)
        .collect();
    let ordinary_fallbacks: usize = ordinary.iter().map(|t| t.unresolved).sum();
    let worst = tally.values().map(|t| t.worst_rel).fold(0.0_f64, f64::max);
    eprintln!(
        "/// Measured on the validation sweep ({} calls, k = 1..={K_SWEEP_MAX}, \
         `K_DUAL_MAX` = {K_DUAL_MAX}): {res} resolved, {zero} zero-model, {unres} sent to the \
         primal kernel; {ordinary_fallbacks} fallbacks on ordinary data at k ≤ `K_DUAL_MAX`; \
         worst |Δcoef| / max(1, ‖coef‖_∞) = {worst:.1e}.",
        res + zero + unres
    );
    assert_eq!(
        ordinary_fallbacks, 0,
        "ordinary data must not fall back at k ≤ K_DUAL_MAX"
    );
    assert!(!ordinary.is_empty() && ordinary.iter().all(|t| t.resolved > 0));
}

// The band is measured for the `k` the n-space route serves on the
// `split_exact` score path, `2..=K_DUAL_MAX`; larger `k` are measured and
// printed only, so raising `K_DUAL_MAX` moves them into the band. The checks
// on `SCORE_BAND` stay runtime assertions: a `const` assertion would stop the
// whole test crate from compiling instead of failing this test.
#[test]
#[allow(clippy::assertions_on_constants)]
fn score_band_covers_the_measured_score_discrepancy() {
    let mut eta_max = 0.0_f64;
    let mut measured = 0_usize;
    // Diagnostic only: `k > K_DUAL_MAX`, which the route never serves.
    let mut eta_beyond = 0.0_f64;
    let mut measured_beyond = 0_usize;
    for d in designs() {
        let n = d.xs.nrows();
        if n < 12 {
            continue;
        }
        let n_tr = n / 2;
        let n_te = n - n_tr;
        let (xs_tr, mean, scale) = standardize(d.xs.subrows(0, n_tr));
        let xs_te = standardize_apply(d.xs.subrows(n_tr, n_te), mean.as_ref(), scale.as_ref());
        let z = std_y(&Col::<f64>::from_fn(n_tr, |i| d.z[i]));
        let gram = NspaceGram::new(xs_tr.as_ref(), Par::Seq);
        let m = cross_of(xs_te.as_ref(), xs_tr.as_ref(), Par::Seq);
        for k in 2..=K_SWEEP_MAX.min(n_tr - 1) {
            let NspaceOutcome::Resolved { alpha, .. } =
                pls1_nspace_kernel(&gram.block(), z.as_ref(), k)
            else {
                continue;
            };
            let fit = pls1_fit_prepared_fro(
                xs_tr.as_ref(),
                z.as_ref(),
                k,
                None,
                ParChoice::Seq,
                xs_tr.norm_l2(),
            )
            .expect("primal fit");
            let s_gram = seq_gemv(m.as_ref(), &alpha);
            let s_primal = seq_gemv(xs_te.as_ref(), &fit.coef);
            let norm = s_primal.norm_l2();
            if norm > 0.0 {
                let diff = Col::<f64>::from_fn(n_te, |i| s_gram[i] - s_primal[i]).norm_l2();
                if k <= K_DUAL_MAX {
                    eta_max = eta_max.max(diff / norm);
                    measured += 1;
                } else {
                    eta_beyond = eta_beyond.max(diff / norm);
                    measured_beyond += 1;
                }
            }
        }
    }
    let required = 10.0 * eta_max / 1e-10;
    eprintln!(
        "score discrepancy over {measured} resolved calls at k = 2..={K_DUAL_MAX}: \
         eta_max = {eta_max:.2e}; SCORE_BAND must be at least {required:.2e}"
    );
    eprintln!(
        "diagnostic, not in the band: {measured_beyond} resolved calls at \
         k = {}..={K_SWEEP_MAX}: eta_max = {eta_beyond:.2e}",
        K_DUAL_MAX + 1
    );
    assert!(measured > 0);
    assert!(
        eta_max > 0.0,
        "eta_max must be a positive measured discrepancy"
    );
    let target = 10f64.powf(required.log10().ceil());
    // One-sided: SCORE_BAND is shared with the p-space Gram route (gram_p),
    // which may raise it above this n-space requirement to cover its own
    // score discrepancy, up to the ceiling asserted below. So this only
    // checks that the band still covers the n-space measurement; it does
    // not pin SCORE_BAND to this exact value.
    assert!(
        SCORE_BAND >= target,
        "SCORE_BAND is {SCORE_BAND:e}; it must be at least the required value {target:e} \
         (10x the measured maximum, rounded up to a power of ten), \
         so raise it if the measurement grows"
    );
    assert!(
        SCORE_BAND <= 1e-2,
        "SCORE_BAND = {SCORE_BAND:e} would gate ordinary scores"
    );
    assert!(SCORE_BAND > 2.0 * (crate::dual_route::DUAL_ROUTE_MAX_N_TR as f64) * f64::EPSILON);
}
