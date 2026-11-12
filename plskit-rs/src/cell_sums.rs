//! Every split's training sums from the cells of the balanced split design
//! ([`crate::resample::SplitDesign`]), for the tall held-out routes.
//!
//! With `H = 2^L` cells, `T` the sum of the H cell rows and `W` the rows
//! after [`fwht_rows`] (`W[s] = Σ_c (−1)^{|c & s|}·row_c`, so `W[0] = T`),
//! split `j` trains on the cells where `|c & s_j| + b_j` is even, so its
//! sum over its training cells is `(T + σ_j·W[s_j])/2`, `σ_j = −1` when
//! `flips[j]`, else `+1`. Its leftover picks are added directly. One pass
//! over the rows and one transform give the sums of all J splits: about
//! `n·w + H·L·w + J·(r/2)·w` additions for `w` sums per row, against `J·n_A·w`
//! for J separate passes.

use faer::{Col, Mat, MatMut, MatRef, Par};

use crate::linalg::{mat_mul, row_subset};
use crate::resample::SplitDesign;

/// Outcome columns one [`cell_sums`] call takes at most. The batches are
/// fixed by the column count alone, so the bits do not depend on the pool.
pub(crate) const CELL_BATCH: usize = 32;

/// The column batch width the routes use for `n_cols` columns.
pub(crate) fn cell_batch(n_cols: usize) -> usize {
    n_cols.clamp(1, CELL_BATCH)
}

#[cfg(test)]
thread_local! {
    static CELL_GRAMS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many times [`cell_grams`] ran on the calling thread (once per call
/// of a route that sums from the cells).
#[cfg(test)]
pub(crate) fn cell_grams_built() -> usize {
    CELL_GRAMS_BUILT.with(std::cell::Cell::get)
}

/// The in-place fast Walsh–Hadamard transform over the rows of `a`: row `s`
/// becomes `Σ_c (−1)^{|c & s|}·row_c` (`|·|` the bit count), in `L`
/// butterfly stages of `H/2` sums and differences per column, for
/// `H = 2^L` rows.
///
/// # Panics
/// When the row count is not a power of two.
pub(crate) fn fwht_rows(mut a: MatMut<'_, f64>) {
    let h = a.nrows();
    assert!(h.is_power_of_two(), "fwht_rows needs 2^L rows, got {h}");
    for col in 0..a.ncols() {
        let mut half = 1;
        while half < h {
            for start in (0..h).step_by(2 * half) {
                for i in start..start + half {
                    let (u, v) = (a[(i, col)], a[(i + half, col)]);
                    a[(i, col)] = u + v;
                    a[(i + half, col)] = u - v;
                }
            }
            half *= 2;
        }
    }
}

/// The sums of one batch of `b` outcome columns, in groups of `g`
/// consecutive columns, over every split's training rows, and over all
/// rows, from [`cell_sums`].
pub(crate) struct CellSums {
    /// `X̃_A'y`, `p × (J·b)`: split `j`, column `c` at `j·b + c`.
    pub(crate) xa_y: Mat<f64>,
    /// `1'y_A`, `J × b`.
    pub(crate) ya_sum: Mat<f64>,
    /// `Y_A'Y_A` of each group, `J × (b·g)`: group `G`, entry `(k, l)` at
    /// `G·g² + k·g + l` (at `g = 1`, column `c`'s `y_A'y_A` at `c`).
    pub(crate) ya_ss: Mat<f64>,
    /// `X̃'y` over all rows, `p × b`.
    pub(crate) x_y: Mat<f64>,
    /// `1'y` over all rows, length `b`.
    pub(crate) y_sum: Col<f64>,
    /// `Y'Y` of each group over all rows, length `b·g`, laid out as `ya_ss`.
    pub(crate) y_ss: Col<f64>,
}

/// `(X̃_R'Y_R, 1'Y_R, Y_R'Y_R)` over the rows `rows` of `xs` and `cols`
/// (`p × b`, then `b` sums and the `b·g` within-group products in
/// [`CellSums::ya_ss`]'s layout), the leftover part of a split's sums.
fn direct_sums(
    xs: MatRef<'_, f64>,
    cols: MatRef<'_, f64>,
    rows: &[usize],
    g: usize,
) -> (Mat<f64>, Vec<f64>, Vec<f64>) {
    let (xr, yr) = (row_subset(xs, rows), row_subset(cols, rows));
    let s = mat_mul(xr.transpose(), yr.as_ref(), Par::Seq);
    let sum = (0..cols.ncols())
        .map(|c| (0..rows.len()).map(|i| yr[(i, c)]).sum())
        .collect();
    let ss = (0..cols.ncols() * g)
        .map(|e| {
            let (k, l) = (e / (g * g) * g + e % (g * g) / g, e / (g * g) * g + e % g);
            (0..rows.len()).map(|i| yr[(i, k)] * yr[(i, l)]).sum()
        })
        .collect();
    (s, sum, ss)
}

/// Every split's training sums of the `b` columns `cols` (`n × b`, read as
/// given, so a caller's null columns enter as they are built), in groups of
/// `g` consecutive columns (`g` divides `b`), with `xs` the `n × p` X̃ the
/// route works in, rows in their original order: per cell one product
/// `X̃_c'Y_c`, the sums of `Y_c` and each group's `Y_c'Y_c`, the transform
/// over the H cells, then for split `j` `(T + σ_j·W[s_j])/2` plus its
/// leftover picks (its training rows past the `H·m/2` cell rows). The
/// totals are `T` plus every leftover row.
#[allow(clippy::many_single_char_names)]
pub(crate) fn cell_sums(
    xs: MatRef<'_, f64>,
    d: &SplitDesign,
    cols: MatRef<'_, f64>,
    g: usize,
) -> CellSums {
    let (p, b, n_splits) = (xs.ncols(), cols.ncols(), d.splits.len());
    debug_assert!(g > 0 && b % g == 0);
    let (h, m) = (d.cells(), d.m);
    // Row c: `X̃_c'Y_c` column by column (`p·b`), then `1'Y_c` and the
    // groups' `Y_c'Y_c` (`b·g`).
    let (sum_at, ss_at, n_ss) = (p * b, (p + 1) * b, b * g);
    let mut a = Mat::<f64>::zeros(h, ss_at + n_ss);
    if m > 0 {
        for c in 0..h {
            let (s, sum, ss) = direct_sums(xs, cols, &d.cell_rows[c * m..(c + 1) * m], g);
            for col in 0..b {
                for f in 0..p {
                    a[(c, col * p + f)] = s[(f, col)];
                }
                a[(c, sum_at + col)] = sum[col];
            }
            for (e, v) in ss.into_iter().enumerate() {
                a[(c, ss_at + e)] = v;
            }
        }
    }
    fwht_rows(a.as_mut());
    let half = h * m / 2;
    let mut out = CellSums {
        xa_y: Mat::<f64>::zeros(p, n_splits * b),
        ya_sum: Mat::<f64>::zeros(n_splits, b),
        ya_ss: Mat::<f64>::zeros(n_splits, n_ss),
        x_y: Mat::<f64>::zeros(p, b),
        y_sum: Col::<f64>::zeros(b),
        y_ss: Col::<f64>::zeros(n_ss),
    };
    for (j, sp) in d.splits.iter().enumerate() {
        let s = d.masks[j] as usize;
        let sigma = if d.flips[j] { -1.0 } else { 1.0 };
        let train = |e: usize| f64::midpoint(a[(0, e)], sigma * a[(s, e)]);
        let (ls, lsum, lss) = direct_sums(xs, cols, &sp.tr[half..], g);
        for col in 0..b {
            for f in 0..p {
                out.xa_y[(f, j * b + col)] = train(col * p + f) + ls[(f, col)];
            }
            out.ya_sum[(j, col)] = train(sum_at + col) + lsum[col];
        }
        for (e, v) in lss.into_iter().enumerate() {
            out.ya_ss[(j, e)] = train(ss_at + e) + v;
        }
    }
    let (ls, lsum, lss) = direct_sums(xs, cols, &d.leftovers, g);
    for col in 0..b {
        for f in 0..p {
            out.x_y[(f, col)] = a[(0, col * p + f)] + ls[(f, col)];
        }
        out.y_sum[col] = a[(0, sum_at + col)] + lsum[col];
    }
    for (e, v) in lss.into_iter().enumerate() {
        out.y_ss[e] = a[(0, ss_at + e)] + v;
    }
    out
}

/// Every split's training-half cross-product matrix and column sums from
/// the cells of `d`: `S_A = Σ_{c∈A} X̃_c'X̃_c + Σ_picks x̃_i x̃_i' − n_A·μ_A μ_A'`
/// (`p × p`, `X̃_A` centred by its own means) and `X̃_A'1 = n_A·μ_A`, both
/// through the transform over the H cell matrices `X̃_c'X̃_c` and sums
/// `X̃_c'1`, so the call reads each row once. Without `with_s_a` only the
/// sums are formed and the `S_A` list is empty.
#[allow(clippy::many_single_char_names)]
pub(crate) fn cell_grams(
    xs: MatRef<'_, f64>,
    d: &SplitDesign,
    with_s_a: bool,
) -> (Vec<Mat<f64>>, Vec<Col<f64>>) {
    #[cfg(test)]
    CELL_GRAMS_BUILT.with(|c| c.set(c.get() + 1));
    let p = xs.ncols();
    let (h, m) = (d.cells(), d.m);
    let sum_at = if with_s_a { p * p } else { 0 };
    let mut a = Mat::<f64>::zeros(h, sum_at + p);
    let block = |rows: &[usize]| {
        let xr = row_subset(xs, rows);
        let g = if with_s_a {
            mat_mul(xr.transpose(), xr.as_ref(), Par::Seq)
        } else {
            Mat::<f64>::zeros(0, 0)
        };
        let sum: Vec<f64> = (0..p)
            .map(|f| (0..rows.len()).map(|i| xr[(i, f)]).sum())
            .collect();
        (g, sum)
    };
    if m > 0 {
        for c in 0..h {
            let (g, sum) = block(&d.cell_rows[c * m..(c + 1) * m]);
            for f2 in 0..p {
                for f1 in 0..g.nrows() {
                    a[(c, f2 * p + f1)] = g[(f1, f2)];
                }
                a[(c, sum_at + f2)] = sum[f2];
            }
        }
    }
    fwht_rows(a.as_mut());
    let half = h * m / 2;
    let (s_a, xa_1): (Vec<_>, Vec<_>) = d
        .splits
        .iter()
        .enumerate()
        .map(|(j, sp)| {
            let s = d.masks[j] as usize;
            let sigma = if d.flips[j] { -1.0 } else { 1.0 };
            let train = |e: usize| f64::midpoint(a[(0, e)], sigma * a[(s, e)]);
            let (lg, lsum) = block(&sp.tr[half..]);
            let xa_1 = Col::<f64>::from_fn(p, |f| train(sum_at + f) + lsum[f]);
            let n_a = sp.tr.len() as f64;
            let s_a = with_s_a.then(|| {
                Mat::<f64>::from_fn(p, p, |f1, f2| {
                    train(f2 * p + f1) + lg[(f1, f2)] - xa_1[f1] * xa_1[f2] / n_a
                })
            });
            (s_a, xa_1)
        })
        .unzip();
    (s_a.into_iter().flatten().collect(), xa_1)
}

/// The bytes [`cells_eligible`] counts, for `n` rows of which `n_b` test
/// rows per split, `h` cells, `n_splits` splits, `p` columns and column
/// batches of width `b`: the H cell matrices and the J training matrices
/// (`(H + J)·p²`), one batch's H cell sums (`H·p·b`), the route's `n × p`
/// X̃ (`n·p`), the J test-half factors (`J·p²`) and one test half's
/// standardized copy while it is factored (`n_B·p`; the halves are factored
/// in budget units, as many at once as the budget allows), all at 8 bytes.
pub(crate) fn cells_bytes(
    n: usize,
    n_b: usize,
    h: usize,
    n_splits: usize,
    p: usize,
    b: usize,
) -> usize {
    (h + 2 * n_splits)
        .saturating_mul(p)
        .saturating_mul(p)
        .saturating_add(h.saturating_mul(p).saturating_mul(b))
        .saturating_add(n.saturating_mul(p))
        .saturating_add(n_b.saturating_mul(p))
        .saturating_mul(8)
}

/// Whether a tall route sums its training rows from the cells of `d`, for
/// X with `p` columns and column batches of width `b`: unweighted only,
/// only when the design has cells (`m > 0`), and only when the buffers
/// [`cells_bytes`] counts fit `fit::memory_budget`.
pub(crate) fn cells_eligible(d: &SplitDesign, p: usize, b: usize, weighted: bool) -> bool {
    let n = d.cell_rows.len() + d.leftovers.len();
    let n_b = d.splits.first().map_or(0, |sp| sp.te.len());
    let bytes = cells_bytes(n, n_b, d.cells(), d.splits.len(), p, b);
    !weighted && d.m > 0 && u64::try_from(bytes).is_ok_and(|v| v <= crate::fit::memory_budget())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::{balanced_splits, design_cells, split_sizes};
    use rand::{RngExt as _, SeedableRng as _};

    fn rng(seed: u64) -> rand_chacha::ChaCha8Rng {
        rand_chacha::ChaCha8Rng::seed_from_u64(seed)
    }

    fn design(n: usize, n_splits: usize, seed: u64) -> SplitDesign {
        let (_, mut r) = crate::rng::resolve_seed(Some(seed)).unwrap();
        balanced_splits(n, split_sizes(n, 1).0, n_splits, &mut r)
    }

    /// At the default budget the cells serve a large tall call: 10^6 × 20 at
    /// J = 32 (H = 64, `n_B = 5·10^5`, b = 32) counts about 241 MB.
    #[test]
    fn a_large_tall_shape_fits_the_default_budget() {
        let bytes = cells_bytes(1_000_000, 500_000, 64, 32, 20, CELL_BATCH);
        assert_eq!(
            bytes,
            (128 * 400 + 64 * 20 * 32 + 20_000_000 + 10_000_000) * 8
        );
        assert!(u64::try_from(bytes).unwrap() <= crate::fit::DEFAULT_MEMORY_BUDGET);
    }

    /// The transform equals the product with the explicit ±1 Walsh matrix
    /// `(−1)^{|c & s|}`, exactly on small integers.
    #[test]
    fn fwht_matches_the_walsh_matrix() {
        for h in [2_usize, 8, 128] {
            let mut r = rng(h as u64);
            let a = Mat::<f64>::from_fn(h, 3, |_, _| f64::from(r.random_range(-50_i32..50)));
            let mut w = a.clone();
            fwht_rows(w.as_mut());
            for s in 0..h {
                for col in 0..3 {
                    let explicit: f64 = (0..h)
                        .map(|c| {
                            let sign = if (c & s).count_ones() % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            sign * a[(c, col)]
                        })
                        .sum();
                    assert_eq!(w[(s, col)], explicit, "H={h} s={s} col={col}");
                }
            }
        }
    }

    /// `|u − v| ≤ 1e-12·max(1, |v|)`.
    fn close(u: f64, v: f64, what: &str) {
        assert!(
            (u - v).abs() <= 1e-12 * v.abs().max(1.0),
            "{what}: {u} vs direct {v}"
        );
    }

    /// Every split's sums from the cells against direct sums over its
    /// training rows, at the design's edges: `n = H·m`, `H − 1` (no cells),
    /// `H + 1`, odd `n`, leftover rows (`r > 0`), and `J = 1, 64, 65`; the
    /// products of the columns in groups of 1 and of 3.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn cell_sums_equal_direct_sums_at_design_edges() {
        let (p, b) = (5, 3);
        for (n_splits, g) in [1_usize, 64, 65].into_iter().flat_map(|j| [(j, 1), (j, 3)]) {
            let h = design_cells(n_splits);
            for n in [6 * h, h - 1, h + 1, 2 * h + 5, 3 * h + 7] {
                if n < 6 {
                    continue;
                }
                let d = design(n, n_splits, n as u64);
                let at = format!("J={n_splits} H={h} n={n} m={}", d.m);
                let r_left = n - h * d.m;
                if n % h == 0 {
                    assert!(d.m > 0, "{at}: premise, cells");
                }
                let mut r = rng(7 + n as u64);
                let xs = Mat::<f64>::from_fn(n, p, |_, _| r.random_range(-1.0..1.0) + 0.3);
                let ys = Mat::<f64>::from_fn(n, b, |_, _| r.random_range(-2.0..2.0));
                let cs = cell_sums(xs.as_ref(), &d, ys.as_ref(), g);
                let (s_a, xa_1) = cell_grams(xs.as_ref(), &d, true);
                // Entry `e` of a group product is columns `(k, l)`.
                let pair = |e: usize| (e / (g * g) * g + e % (g * g) / g, e / (g * g) * g + e % g);
                for (j, sp) in d.splits.iter().enumerate() {
                    let tr = &sp.tr;
                    let na = tr.len() as f64;
                    let mu: Vec<f64> = (0..p)
                        .map(|f| tr.iter().map(|&i| xs[(i, f)]).sum::<f64>() / na)
                        .collect();
                    for f in 0..p {
                        close(xa_1[j][f], mu[f] * na, &format!("{at} split {j} X'1[{f}]"));
                        for g in 0..p {
                            let direct: f64 = tr
                                .iter()
                                .map(|&i| (xs[(i, f)] - mu[f]) * (xs[(i, g)] - mu[g]))
                                .sum();
                            close(s_a[j][(f, g)], direct, &format!("{at} split {j} S_A"));
                        }
                    }
                    for c in 0..b {
                        let w = format!("{at} split {j} col {c} (r={r_left})");
                        for f in 0..p {
                            let direct: f64 = tr.iter().map(|&i| xs[(i, f)] * ys[(i, c)]).sum();
                            close(cs.xa_y[(f, j * b + c)], direct, &format!("{w} X'y"));
                        }
                        let sum: f64 = tr.iter().map(|&i| ys[(i, c)]).sum();
                        close(cs.ya_sum[(j, c)], sum, &format!("{w} 1'y"));
                    }
                    for e in 0..b * g {
                        let (k, l) = pair(e);
                        let ss: f64 = tr.iter().map(|&i| ys[(i, k)] * ys[(i, l)]).sum();
                        close(
                            cs.ya_ss[(j, e)],
                            ss,
                            &format!("{at} split {j} y'y {e} (g={g})"),
                        );
                    }
                }
                for c in 0..b {
                    for f in 0..p {
                        let direct: f64 = (0..n).map(|i| xs[(i, f)] * ys[(i, c)]).sum();
                        close(cs.x_y[(f, c)], direct, &format!("{at} total X'y"));
                    }
                    close(cs.y_sum[c], (0..n).map(|i| ys[(i, c)]).sum(), &at);
                }
                for e in 0..b * g {
                    let (k, l) = pair(e);
                    let ss: f64 = (0..n).map(|i| ys[(i, k)] * ys[(i, l)]).sum();
                    close(cs.y_ss[e], ss, &format!("{at} total y'y {e} (g={g})"));
                }
            }
        }
    }
}
