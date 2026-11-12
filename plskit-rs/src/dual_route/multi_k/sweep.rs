//! The n-space kernel's accuracy sweep: per design family and component
//! count `k = 1..=10`, the coefficients and the split-half `r` of the
//! n-space route against the p-space fit on the same scaled X, which set
//! [`K_DUAL_MAX`]. `cargo test -p plskit --release multi_k::sweep --
//! --nocapture` prints the table.
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]

use std::collections::BTreeMap;

use super::*;
use crate::fit::{ParChoice, pls1_fit_prepared_fro};
use crate::held_out_gram::HeldOutGram;
use crate::linalg::{standardize, standardize1};
use crate::signal_test::{SplitIdx, prepare_split, split_column_r};
use crate::test_support::{orthonormal_basis, project_off};
use faer::{Col, Mat};
use rand::{RngExt, SeedableRng};

/// The component counts the sweep measures.
const K_SWEEP_MAX: usize = 10;

/// The agreement tolerances: `|Δcoef| ≤ COEF_TOL·max(1, ‖coef‖_∞)` and
/// `|Δr| ≤ R_TOL`.
const COEF_TOL: f64 = 1e-10;
const R_TOL: f64 = 1e-12;

struct Design {
    family: &'static str,
    xs: Mat<f64>,
    z: Col<f64>,
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
        });
    }
    for seed in [5_u64, 6] {
        let xs = std_x(&uniform(30, 2000, seed));
        let z = signal_outcome(&xs, 1e-6, seed + 100);
        out.push(Design {
            family: "weak_signal",
            xs,
            z,
        });
    }
    {
        let xs = std_x(&low_rank(40, 1500, 10, 7));
        let z = signal_outcome(&xs, 1.0, 107);
        out.push(Design {
            family: "rank_deficient",
            xs,
            z,
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
        });
    }
    for (n, seed) in [(6_usize, 9_u64), (8, 10)] {
        let xs = std_x(&uniform(n, 300, seed));
        let z = signal_outcome(&xs, 1.0, seed + 100);
        out.push(Design {
            family: "tiny_n",
            xs,
            z,
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
        });
    }
    for m in 1..=3 {
        let xs = std_x(&uniform(30, 1500, 14));
        let (z, _, _) = in_span(&xs, m);
        out.push(Design {
            family: "y_in_span",
            xs,
            z,
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
        });
    }
    out
}

/// Worst errors of one (k, family) cell on the fits where both routes keep
/// the same component count: `|Δcoef| / max(1, ‖coef‖_∞)` on all rows and
/// `|Δr|` on the split.
#[derive(Default)]
struct Cell {
    coef: f64,
    r: f64,
}

/// One fit where the two routes keep different component counts.
struct Flip {
    family: &'static str,
    design: usize,
    k: usize,
    /// `"coef"` (all rows) or `"r"` (the split's training half).
    on: &'static str,
    nspace: usize,
    pspace: usize,
    /// `|Δcoef| / max(1, ‖coef‖_∞)` or `|Δr|`.
    err: f64,
}

/// The fits where the n-space route may keep a different component count
/// than the p-space fit: `z'Gz` resolves `‖X̃'z‖` only to `G`'s rounding,
/// far above the relative floor, so an outcome orthogonal to `X̃` and the
/// components past the rank of `X̃` (rank 4 in `y_exhausted`) sit where the
/// two routes can decide differently.
fn flip_expected(family: &str, k: usize) -> bool {
    family == "y_orthogonal" || (family == "y_exhausted" && k > 4)
}

/// `coef = X̃_tr'·alpha` of an n-space outcome and its component count.
fn nspace_coef(xs_tr: MatRef<'_, f64>, out: NspaceOutcome) -> (Col<f64>, usize) {
    match out {
        NspaceOutcome::Resolved { alpha, k_used } => (seq_gemv(xs_tr.transpose(), &alpha), k_used),
        NspaceOutcome::ZeroModel => (Col::<f64>::zeros(xs_tr.ncols()), 0),
    }
}

/// The n-space component count for `z` on `block`.
fn nspace_k_used(block: &NspaceBlock<'_>, z: ColRef<'_, f64>, k: usize) -> usize {
    match pls1_nspace_kernel(block, z, k) {
        NspaceOutcome::Resolved { k_used, .. } => k_used,
        NspaceOutcome::ZeroModel => 0,
    }
}

/// Every design family, `k = 1..=10`: the block the route reads (the all-row
/// Gram matrix's, `NspaceSplit::from_gram`) against `pls1_fit` on the same
/// scaled X (`prepare_split`'s halves, scale one, the designs being
/// standardized). Coefficients on all rows as the training block; `r` on
/// the split of the first half against the second, `split_columns_r_nspace`
/// against `split_column_r`. On the fits where both routes keep the same
/// component count, every family meets `|Δcoef| ≤ 1e-10·max(1, ‖coef‖_∞)`
/// and `|Δr| ≤ 1e-12` at every `k ≤ K_DUAL_MAX`; a fit where the counts
/// differ is listed, and is allowed only where [`flip_expected`] says.
#[test]
#[allow(clippy::too_many_lines)]
fn nspace_matches_pspace_up_to_k_dual_max() {
    let mut cells: BTreeMap<(usize, &'static str), Cell> = BTreeMap::new();
    let mut flips: Vec<Flip> = Vec::new();
    let mut seen: BTreeMap<&'static str, usize> = BTreeMap::new();
    for d in designs() {
        let design = {
            let c = seen.entry(d.family).or_default();
            *c += 1;
            *c - 1
        };
        let (n, p) = (d.xs.nrows(), d.xs.ncols());
        let ones = Col::<f64>::from_fn(p, |_| 1.0);
        let g = HeldOutGram::from_x(d.xs.as_ref(), None, ones.as_ref());
        let z = std_y(&d.z);
        let all = SplitIdx {
            tr: (0..n).collect(),
            te: Vec::new(),
        };
        let prep = prepare_split(d.xs.as_ref(), &all, None, ones.as_ref());
        let ns = NspaceSplit::from_gram(&g, &all);
        let n_tr = n - n / 2;
        let half = SplitIdx {
            tr: (0..n_tr).collect(),
            te: (n_tr..n).collect(),
        };
        let prep_h = prepare_split(d.xs.as_ref(), &half, None, ones.as_ref());
        let ns_h = NspaceSplit::from_gram(&g, &half);
        let z_h = std_y(&Col::<f64>::from_fn(n_tr, |i| d.z[i]));
        let primal = |xs: MatRef<'_, f64>, z: &Col<f64>, x_fro: f64, k: usize| {
            pls1_fit_prepared_fro(xs, z.as_ref(), k, None, ParChoice::Seq, x_fro)
                .expect("p-space fit")
        };
        for k in 1..=K_SWEEP_MAX.min(n - 1) {
            let cell = cells.entry((k, d.family)).or_default();
            let fit = primal(prep.xs_tr.as_ref(), &z, prep.x_fro, k);
            let (coef, k_used) = nspace_coef(
                prep.xs_tr.as_ref(),
                pls1_nspace_kernel(&ns.gram.block(), z.as_ref(), k),
            );
            let scale = (0..p).map(|j| fit.coef[j].abs()).fold(1.0_f64, f64::max);
            let diff = (0..p)
                .map(|j| (coef[j] - fit.coef[j]).abs())
                .fold(0.0_f64, f64::max)
                / scale;
            if k_used == fit.k_used {
                cell.coef = cell.coef.max(diff);
            } else {
                flips.push(Flip {
                    family: d.family,
                    design,
                    k,
                    on: "coef",
                    nspace: k_used,
                    pspace: fit.k_used,
                    err: diff,
                });
            }
            if k < n_tr {
                let r_n = split_columns_r_nspace(&half, &ns_h, 1, &|_, i| d.z[i], &[k])[0][0];
                let r_p = split_column_r(&prep_h, &half, &|i| d.z[i], &[k], None)[0];
                let ku_p = primal(prep_h.xs_tr.as_ref(), &z_h, prep_h.x_fro, k).k_used;
                let ku_n = nspace_k_used(&ns_h.gram.block(), z_h.as_ref(), k);
                if ku_n == ku_p {
                    cell.r = cell.r.max((r_n - r_p).abs());
                } else {
                    flips.push(Flip {
                        family: d.family,
                        design,
                        k,
                        on: "r",
                        nspace: ku_n,
                        pspace: ku_p,
                        err: (r_n - r_p).abs(),
                    });
                }
            }
        }
    }
    let meets = |k: usize| {
        cells
            .range((k, "")..(k + 1, ""))
            .all(|(_, c)| c.coef <= COEF_TOL && c.r <= R_TOL)
    };
    let k_max = (1..=K_SWEEP_MAX)
        .take_while(|&k| meets(k))
        .last()
        .unwrap_or(0);
    eprintln!("k  family             |dcoef|/max(1,|coef|)  |dr|   (equal component counts)");
    for ((k, family), c) in &cells {
        eprintln!("{k:>2} {family:<17} {:>21.1e}  {:.1e}", c.coef, c.r);
    }
    eprintln!("/// | k | worst |Δcoef|/max(1, ‖coef‖_∞) | worst |Δr| |");
    for k in 1..=K_SWEEP_MAX {
        let row = cells.range((k, "")..(k + 1, ""));
        let worst = |f: &dyn Fn(&Cell) -> f64| {
            row.clone().fold(
                ("", 0.0_f64),
                |a, ((_, fam), c)| {
                    if f(c) > a.1 { (*fam, f(c)) } else { a }
                },
            )
        };
        let (fc, wc) = worst(&|c| c.coef);
        let (fr, wr) = worst(&|c| c.r);
        eprintln!("/// | {k} | {wc:.1e} ({fc}) | {wr:.1e} ({fr}) |");
    }
    eprintln!("/// Flips (family, design, k, on, kept n-space / p-space, error):");
    for f in &flips {
        eprintln!(
            "/// - {} {}, k = {}, {}: {} / {}, {:.1e}",
            f.family, f.design, f.k, f.on, f.nspace, f.pspace, f.err
        );
    }
    eprintln!("largest k meeting the tolerances at every k up to it: {k_max}");
    let unexpected: Vec<String> = flips
        .iter()
        .filter(|f| !flip_expected(f.family, f.k))
        .map(|f| format!("{} {} k={} {}", f.family, f.design, f.k, f.on))
        .collect();
    assert!(
        unexpected.is_empty(),
        "component counts differ outside the expected families: {unexpected:?}"
    );
    assert!(
        k_max >= K_DUAL_MAX,
        "K_DUAL_MAX = {K_DUAL_MAX}, but the sweep meets the tolerances only up to k = {k_max}"
    );
}
