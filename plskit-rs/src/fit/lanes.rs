//! PLS1 fits of a batch of outcome columns ("lanes") on one standardized
//! `X̃` (`n × p`), in lockstep, one component at a time.
//!
//! Each lane runs the float steps of `pls1_kernel` (the selection, floors,
//! rotation and update of `s` live in [`LaneState`]), but every product with
//! `X̃` is one matrix product for the whole batch, summed over column strips
//! of `X̃` ([`strip_mul`], [`strip_t_mul`]). A lane that stops, or was never
//! entered, holds a zero column in every product from then on, so the batch
//! keeps its width and a lane's bits depend on its batch, not on which other
//! lanes stop. The batched products sum in another order than the single
//! fit's matrix-vector products, so a lane agrees with `pls1_kernel` to
//! rounding, not to the bit.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatMut, MatRef, Par};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use super::{LaneState, check_finite_mat, pls1_coef_at_k, pls1_fit_bytes, tt_stops, w_rel_floor};
use crate::linalg::{centered_blocks, standardize_apply_rows_into};
use crate::resample::copy_bytes;

/// Lanes fitted together: a caller fits its columns in batches of
/// `min(n_cols, LANE_BATCH)` consecutive columns, so the batch, and with it
/// every bit, is fixed by the column count alone.
pub(crate) const LANE_BATCH: usize = 32;

/// Columns of `X̃` per strip of [`strip_mul`] and [`strip_t_mul`]; the last
/// strip is shorter. 64 was the fastest on 1 GB tall and wide splits.
pub(crate) const STRIP_COLS: usize = 64;

/// Where the strips of `X̃` come from.
pub(crate) enum StripSource<'a> {
    /// A standardized copy of the half; a strip is a view of its columns.
    Copy(MatRef<'a, f64>),
    /// No copy: each strip (or its rows of one thread's range) is gathered
    /// from X and standardized into a scratch strip of that thread's. Its
    /// entries have the bits of the copy's.
    Gather(Gathered<'a>),
}

/// A half of X standardized as it is read: rows `rows` of `x`, less the
/// training means `mean`, divided by the all-row `scale`, times
/// `row_scale` (one per row of the half) when given, which is
/// `center_rows_fro`'s training half (on `rows = tr`, `row_scale = √w_tr`)
/// or `standardize_apply_rows`'s test half, bit for bit.
pub(crate) struct Gathered<'a> {
    pub(crate) x: MatRef<'a, f64>,
    pub(crate) rows: &'a [usize],
    pub(crate) mean: Col<f64>,
    pub(crate) scale: ColRef<'a, f64>,
    pub(crate) row_scale: Option<ColRef<'a, f64>>,
}

impl StripSource<'_> {
    /// `(n, p)` of `X̃`.
    fn shape(&self) -> (usize, usize) {
        match self {
            StripSource::Copy(x) => (x.nrows(), x.ncols()),
            StripSource::Gather(g) => (g.rows.len(), g.x.ncols()),
        }
    }

    /// The scratch [`Self::block`] writes up to `rows` rows of a strip
    /// into: empty for `Copy`.
    fn scratch(&self, rows: usize) -> Mat<f64> {
        match self {
            StripSource::Copy(_) => Mat::new(),
            StripSource::Gather(g) => Mat::zeros(rows, STRIP_COLS.min(g.x.ncols())),
        }
    }

    /// Rows `i0 .. i0 + h` of columns `j0 .. j0 + w` of `X̃`: a view of the
    /// copy, or gathered into the top-left of `scratch` (from
    /// [`Self::scratch`]).
    fn block<'s>(
        &'s self,
        (i0, h): (usize, usize),
        (j0, w): (usize, usize),
        scratch: &'s mut Mat<f64>,
    ) -> MatRef<'s, f64> {
        match self {
            StripSource::Copy(x) => x.submatrix(i0, j0, h, w),
            StripSource::Gather(g) => {
                standardize_apply_rows_into(
                    scratch.as_mut().submatrix_mut(0, 0, h, w),
                    g.x.subcols(j0, w),
                    &g.rows[i0..i0 + h],
                    g.mean.as_ref().subrows(j0, w),
                    g.scale.subrows(j0, w),
                    g.row_scale.map(|r| r.subrows(i0, h)),
                );
                scratch.as_ref().submatrix(0, 0, h, w)
            }
        }
    }
}

/// What a [`StripSource::Gather`] split needs before its products, from
/// one `centered_blocks` pass at width [`STRIP_COLS`]: the training means,
/// `‖X̃_tr‖_F` (the bits of `center_rows_fro`'s), and whether each
/// standardized half (after its row factor) is finite.
pub(crate) struct GatherPass {
    pub(crate) mean: Col<f64>,
    pub(crate) x_fro: f64,
    pub(crate) finite_tr: bool,
    pub(crate) finite_te: bool,
}

/// [`GatherPass`] of the split `tr`/`te` of `x`: training weights `w_tr`,
/// row factors `sw_tr` and `sw_te`, all-row `scale`, as `centered_blocks`
/// takes them. Holds two blocks of [`STRIP_COLS`] columns at a time.
#[allow(clippy::similar_names)]
pub(crate) fn gather_pass(
    x: MatRef<'_, f64>,
    tr: &[usize],
    te: &[usize],
    w_tr: Option<ColRef<'_, f64>>,
    sw_tr: Option<ColRef<'_, f64>>,
    sw_te: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
) -> GatherPass {
    let mut mean = Col::<f64>::zeros(x.ncols());
    let (mut finite_tr, mut finite_te) = (true, true);
    let x_fro = centered_blocks(
        x,
        tr,
        te,
        w_tr,
        sw_tr,
        sw_te,
        scale,
        STRIP_COLS,
        |j0, a, c, m| {
            mean.subrows_mut(j0, m.nrows()).copy_from(m);
            finite_tr &= check_finite_mat(a).is_ok();
            finite_te &= check_finite_mat(c).is_ok();
        },
    );
    GatherPass {
        mean,
        x_fro,
        finite_tr,
        finite_te,
    }
}

/// Calls `f(i0, rows, scratch)` on each of up to one contiguous range of
/// `out`'s rows per thread of `par` (`i0` its first row), each range a
/// multiple of `align` rows but the last, with its own
/// `scratch(rows.nrows())`. The ranges write disjoint rows and depend on
/// `par`'s degree only, not the pool size, and a faer product gives each
/// row the bits of one product over all rows, so `Par::Seq` and every
/// degree give the same bits.
fn row_ranges(
    out: MatMut<'_, f64>,
    align: usize,
    par: Par,
    scratch: impl Fn(usize) -> Mat<f64> + Sync,
    f: impl Fn(usize, MatMut<'_, f64>, &mut Mat<f64>) + Sync,
) {
    let threads = match par {
        Par::Seq => 1,
        Par::Rayon(t) => t.get(),
    };
    let chunk = align * out.nrows().div_ceil(align).div_ceil(threads).max(1);
    let run = |i0: usize, rows: MatMut<'_, f64>| {
        let mut s = scratch(rows.nrows());
        f(i0, rows, &mut s);
    };
    if threads == 1 {
        run(0, out);
    } else {
        out.par_row_chunks_mut(chunk)
            .enumerate()
            .for_each(|(c, rows)| run(c * chunk, rows));
    }
}

/// The `Gather` sources of a training half (`tr`, row factors `sw_tr`) and
/// its test half (`te`, `sw_te`), both on the training means `mean`.
#[allow(clippy::similar_names)]
pub(crate) fn gather_sources<'a>(
    x: MatRef<'a, f64>,
    tr: &'a [usize],
    te: &'a [usize],
    sw_tr: Option<ColRef<'a, f64>>,
    sw_te: Option<ColRef<'a, f64>>,
    scale: ColRef<'a, f64>,
    mean: Col<f64>,
) -> (StripSource<'a>, StripSource<'a>) {
    let half_te = Gathered {
        x,
        rows: te,
        mean: mean.clone(),
        scale,
        row_scale: sw_te,
    };
    let half_tr = Gathered {
        x,
        rows: tr,
        mean,
        scale,
        row_scale: sw_tr,
    };
    (StripSource::Gather(half_tr), StripSource::Gather(half_te))
}

/// `X̃·rhs`, `n × m` for `rhs` `p × m`: `Σ_s X̃[:, s]·rhs[s, :]` over strips
/// `s` of [`STRIP_COLS`] columns in ascending order, the first strip
/// writing the result and each later one adding to it, on one range of
/// rows per thread. A strip's product alone is too small to split well.
pub(crate) fn strip_mul(src: &StripSource<'_>, rhs: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let (n, p) = src.shape();
    let mut out = Mat::<f64>::zeros(n, rhs.ncols());
    row_ranges(
        out.as_mut(),
        STRIP_COLS,
        par,
        |h| src.scratch(h),
        |i0, mut rows, scratch| {
            for j0 in (0..p).step_by(STRIP_COLS) {
                let w = STRIP_COLS.min(p - j0);
                let accum = if j0 == 0 { Accum::Replace } else { Accum::Add };
                let x = src.block((i0, rows.nrows()), (j0, w), scratch);
                matmul(rows.as_mut(), accum, x, rhs.subrows(j0, w), 1.0, Par::Seq);
            }
        },
    );
    out
}

/// `X̃'·lhs`, `p × m` for `lhs` `n × m`: rows `s` of the result are
/// `X̃[:, s]'·lhs`, one strip `s` of [`STRIP_COLS`] columns at a time, on
/// one range of strips per thread.
pub(crate) fn strip_t_mul(src: &StripSource<'_>, lhs: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let (n, p) = src.shape();
    let mut out = Mat::<f64>::zeros(p, lhs.ncols());
    row_ranges(
        out.as_mut(),
        STRIP_COLS,
        par,
        |_| src.scratch(n),
        |j0, mut rows, scratch| {
            for s0 in (0..rows.nrows()).step_by(STRIP_COLS) {
                let w = STRIP_COLS.min(rows.nrows() - s0);
                let x = src.block((0, n), (j0 + s0, w), scratch);
                let dst = rows.as_mut().subrows_mut(s0, w);
                matmul(dst, Accum::Replace, x.transpose(), lhs, 1.0, Par::Seq);
            }
        },
    );
    out
}

/// One lane's fit: raw weights `W` and loadings `P` (`p × k_used`), the
/// y-loadings `q` (`k_used`), and `coef = W (P'W)⁻¹ q` (`p`), with
/// `k_used = w.ncols()`.
pub(crate) struct LaneFit {
    pub(crate) w: Mat<f64>,
    pub(crate) p: Mat<f64>,
    pub(crate) q: Col<f64>,
    pub(crate) coef: Col<f64>,
}

/// PLS1 fits of the lanes `ys` (each `n`, standardized as the kernel takes
/// them) on `X̃` to up to `k` components, one entry per lane; a `None` lane
/// is not entered and gives `None`. `x_fro` is `‖X̃‖_F`; `keep` and the
/// floors act per lane as in `pls1_kernel`; `par` runs every product.
///
/// Steps, for the batch `Y` (`n × B`, zero columns for lanes not entered):
/// - `S₀ = X̃'Y`; each lane starts a [`LaneState`] from its column, with
///   the floor `w_rel_floor(n, p, x_fro, ‖y‖)`.
/// - Per component: each live lane's [`LaneState::weight`] gives its `w`
///   and rotation `r` (a `None` stops the lane); the rotations form `R`
///   (`p × B`, zero columns for lanes not live), and `T = X̃R`.
/// - Per lane `tt = ‖t‖²`; [`tt_stops`] stops the lane (its `T` column is
///   zeroed). Then `P = X̃'T`, and per lane `p = P[:, l]/tt` and
///   `q = (Σᵢ yᵢ tᵢ)/tt` in ascending `i` (the expression of the X
///   backend's `q`), and [`LaneState::accept`].
/// - `coef = pls1_coef_at_k` on each lane's truncated `W`, `P`, `q`.
///
/// `lane_batch_bytes` counts what this holds besides `X̃`, with the
/// caller's lanes `ys` (change together).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub(crate) fn pls1_fit_lanes(
    src: &StripSource<'_>,
    ys: &[Option<Col<f64>>],
    x_fro: f64,
    k: usize,
    keep: Option<usize>,
    par: Par,
) -> Vec<Option<LaneFit>> {
    let (n, p) = src.shape();
    let b = ys.len();
    let mut lanes: Vec<Option<LaneState>> = {
        let y = Mat::<f64>::from_fn(n, b, |i, l| ys[l].as_ref().map_or(0.0, |y| y[i]));
        let s0 = strip_t_mul(src, y.as_ref(), par);
        ys.iter()
            .enumerate()
            .map(|(l, y)| {
                let y = y.as_ref()?;
                let floor = w_rel_floor(n, p, x_fro, y.norm_l2());
                Some(LaneState::new(s0.col(l).to_owned(), floor, k))
            })
            .collect()
    };
    let mut live: Vec<bool> = lanes.iter().map(Option::is_some).collect();

    for _ in 0..k {
        let mut r_mat = Mat::<f64>::zeros(p, b);
        let mut steps: Vec<Option<(Col<f64>, Col<f64>)>> = vec![None; b];
        for l in 0..b {
            let Some(lane) = lanes[l].as_ref().filter(|_| live[l]) else {
                continue;
            };
            match lane.weight(keep) {
                Some((w, r)) => {
                    r_mat.col_mut(l).copy_from(&r);
                    steps[l] = Some((w, r));
                }
                None => live[l] = false,
            }
        }
        if steps.iter().all(Option::is_none) {
            break;
        }
        let mut t_mat = strip_mul(src, r_mat.as_ref(), par);
        drop(r_mat);
        let mut tts = vec![0.0; b];
        for l in 0..b {
            if steps[l].is_none() {
                continue;
            }
            let tt = t_mat.col(l).squared_norm_l2();
            if tt_stops(tt) {
                live[l] = false;
                steps[l] = None;
                t_mat.col_mut(l).fill(0.0);
            } else {
                tts[l] = tt;
            }
        }
        let p_mat = strip_t_mul(src, t_mat.as_ref(), par);
        for (l, step) in steps.into_iter().enumerate() {
            let (Some((w, r)), Some(lane), Some(y)) = (step, lanes[l].as_mut(), ys[l].as_ref())
            else {
                continue;
            };
            let tt = tts[l];
            let inv_tt = 1.0 / tt;
            let pl = Col::<f64>::from_fn(p, |j| p_mat[(j, l)] * inv_tt);
            let q = (0..n).map(|i| y[i] * t_mat[(i, l)]).sum::<f64>() * inv_tt;
            lane.accept(w, r, pl, q, tt);
        }
    }

    lanes
        .into_iter()
        .map(|lane| {
            let (p_mat, w_mat, q) = lane?.finish();
            let coef = pls1_coef_at_k(&w_mat, &p_mat, &q, w_mat.ncols(), par);
            Some(LaneFit {
                w: w_mat,
                p: p_mat,
                q,
                coef,
            })
        })
        .collect()
}

/// Bytes a caller of [`pls1_fit_lanes`] holds at once for a batch of `b`
/// lanes on `p` columns to `k` components, scored at each of `nk`
/// component counts on `n_te` test rows: the batch's outcome columns
/// (`(n_tr + n_te) × b`, every row of the unit) and training targets
/// (`n_tr × b`, the lanes `ys`), `b` lane states (`pls1_fit_bytes`), `T`
/// (`n_tr × b`), the `R` and `P` blocks of one component (`p × b` each),
/// the coefficients (`p × b·nk`) and the test scores (`n_te × b·nk`). The
/// outcome block `Y` (`n_tr × b`) and `S₀ = X̃'Y` (`p × b`) are freed
/// before the first component forms `T`, `R` and `P`, and are no larger
/// than `T` and `R`, so they add nothing to the peak. Mirrors
/// `pls1_fit_lanes` (change together).
#[allow(clippy::similar_names)] // n_tr, n_te
pub(crate) fn lane_batch_bytes(
    n_tr: usize,
    n_te: usize,
    p: usize,
    k: usize,
    nk: usize,
    b: usize,
) -> usize {
    let bnk = b.saturating_mul(nk);
    copy_bytes(n_tr.saturating_add(n_te), b)
        // the training targets and `T`
        .saturating_add(copy_bytes(n_tr, b).saturating_mul(2))
        .saturating_add(b.saturating_mul(pls1_fit_bytes(p, k)))
        .saturating_add(copy_bytes(p, b).saturating_mul(2))
        .saturating_add(copy_bytes(p, bnk))
        .saturating_add(copy_bytes(n_te, bnk))
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: designs may use faer's global-parallelism APIs
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::fit::{ParChoice, check_finite_mat, par_fixed, pls1_fit_prepared_fro, resolve_par};
    use crate::linalg::{
        center_rows_fro, renormalize_mean_one, sqrt_col, standardize, standardize_apply_rows,
        standardize1,
    };
    use crate::test_support::{
        Layouts, assert_bits_eq, assert_mat_bits_eq, col_vals, linear_y, test_weights, unif,
        unif_col,
    };

    /// Standardized `X` (`n × p`) and `m` standardized outcome columns:
    /// even columns `X b + noise`, odd columns noise alone.
    fn design(n: usize, p: usize, m: usize, seed: u64) -> (Mat<f64>, Vec<Col<f64>>) {
        let x = unif(n, p, seed);
        let ys = (0..m)
            .map(|c| {
                let s = seed + 100 + c as u64;
                let y = if c % 2 == 0 {
                    linear_y(&x, 0.5, s)
                } else {
                    unif_col(n, s)
                };
                standardize1(y.as_ref()).0
            })
            .collect();
        (standardize(x.as_ref()).0, ys)
    }

    /// Every lane against its single fit on the same view: same `k_used`,
    /// `max|Δcoef| / max(1, ‖coef‖_∞) ≤ 1e-12`. Returns the largest ratio.
    fn check_against_single(xs: &Mat<f64>, ys: &[Col<f64>], k: usize, keep: Option<usize>) -> f64 {
        let (n, p) = (xs.nrows(), xs.ncols());
        let x_fro = xs.norm_l2();
        let par = resolve_par(ParChoice::Auto, n, p, k);
        let lanes: Vec<Option<Col<f64>>> = ys.iter().cloned().map(Some).collect();
        let fits = pls1_fit_lanes(&StripSource::Copy(xs.as_ref()), &lanes, x_fro, k, keep, par);
        assert_eq!(fits.len(), ys.len());
        let mut worst = 0.0_f64;
        for (c, (fit, y)) in fits.iter().zip(ys).enumerate() {
            let fit = fit.as_ref().expect("an entered lane gives a fit");
            let one =
                pls1_fit_prepared_fro(xs.as_ref(), y.as_ref(), k, keep, ParChoice::Auto, x_fro);
            let what = format!("{n}x{p} k {k} keep {keep:?} lane {c}");
            assert_eq!(fit.w.ncols(), one.k_used, "{what}: k_used");
            let scale = (0..p).map(|j| one.coef[j].abs()).fold(1.0_f64, f64::max);
            let diff = (0..p)
                .map(|j| (fit.coef[j] - one.coef[j]).abs())
                .fold(0.0_f64, f64::max);
            let rel = diff / scale;
            assert!(rel <= 1e-12, "{what}: relative coef difference {rel:e}");
            worst = worst.max(rel);
        }
        worst
    }

    #[test]
    fn lanes_match_single_fits() {
        let mut worst = 0.0_f64;
        for (n, p, seed) in [(200, 600, 1), (600, 40, 2)] {
            let (xs, ys) = design(n, p, 20, seed);
            for k in 1..=5 {
                for keep in [None, Some(7)] {
                    worst = worst.max(check_against_single(&xs, &ys, k, keep));
                }
            }
        }
        eprintln!("lanes_match_single_fits: largest relative coef difference {worst:e}");
    }

    #[test]
    fn lanes_match_single_fits_odd_shapes() {
        let k = 3;
        let mut worst = 0.0_f64;
        for m in [1, 31, 33] {
            for p in [1, k, 255, 257] {
                let (xs, ys) = design(120, p, m, 10 + p as u64);
                for keep in [None, Some(7)] {
                    worst = worst.max(check_against_single(&xs, &ys, k, keep));
                }
            }
        }
        eprintln!("lanes_match_single_fits_odd_shapes: largest relative coef difference {worst:e}");
    }

    /// Every bit of two batches' outputs, lanes other than `skip`.
    fn assert_lanes_bits_eq(a: &[Option<LaneFit>], b: &[Option<LaneFit>], skip: usize, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lane count");
        for (l, (fa, fb)) in a.iter().zip(b).enumerate() {
            if l == skip {
                continue;
            }
            let (Some(fa), Some(fb)) = (fa, fb) else {
                assert!(
                    fa.is_none() && fb.is_none(),
                    "{what}: lane {l} entered in one run only"
                );
                continue;
            };
            let w = format!("{what}: lane {l}");
            assert_mat_bits_eq(fa.w.as_ref(), fb.w.as_ref(), &format!("{w} w"));
            assert_mat_bits_eq(fa.p.as_ref(), fb.p.as_ref(), &format!("{w} p"));
            assert_bits_eq(
                &col_vals(fa.q.as_ref()),
                &col_vals(fb.q.as_ref()),
                &format!("{w} q"),
            );
            assert_bits_eq(
                &col_vals(fa.coef.as_ref()),
                &col_vals(fb.coef.as_ref()),
                &format!("{w} coef"),
            );
        }
    }

    #[test]
    fn stopped_lane_leaves_neighbours_bits() {
        let (n, p, k) = (200, 300, 4);
        let (xs, ys) = design(n, p, 8, 21);
        let x_fro = xs.norm_l2();
        let par = resolve_par(ParChoice::Auto, n, p, k);
        let src = StripSource::Copy(xs.as_ref());
        let lane3 = [
            Some(Col::<f64>::from_fn(n, |_| 2.5)),
            Some(standardize1(linear_y(&unif(n, p, 22), 0.5, 23).as_ref()).0),
            None,
        ];
        let runs: Vec<Vec<Option<LaneFit>>> = lane3
            .iter()
            .map(|y3| {
                let mut batch: Vec<Option<Col<f64>>> = ys.iter().cloned().map(Some).collect();
                batch[3].clone_from(y3);
                pls1_fit_lanes(&src, &batch, x_fro, k, None, par)
            })
            .collect();
        assert_eq!(
            runs[0][3].as_ref().unwrap().w.ncols(),
            0,
            "constant y stops at component 1"
        );
        assert_eq!(
            runs[1][3].as_ref().unwrap().w.ncols(),
            k,
            "random y lives to k"
        );
        assert!(runs[2][3].is_none(), "a lane not entered gives None");
        for l in (0..8).filter(|&l| l != 3) {
            assert_eq!(
                runs[0][l].as_ref().unwrap().w.ncols(),
                k,
                "lane {l} lives to k"
            );
        }
        assert_lanes_bits_eq(&runs[0], &runs[1], 3, "constant vs live");
        assert_lanes_bits_eq(&runs[0], &runs[2], 3, "constant vs not entered");
    }

    #[test]
    fn lanes_threads_parity() {
        let (n, p, k) = (400, 600, 3);
        let (xs, ys) = design(n, p, 20, 31);
        let x_fro = xs.norm_l2();
        let lanes: Vec<Option<Col<f64>>> = ys.into_iter().map(Some).collect();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                pls1_fit_lanes(
                    &StripSource::Copy(xs.as_ref()),
                    &lanes,
                    x_fro,
                    k,
                    None,
                    par_fixed(),
                )
            })
        };
        assert_lanes_bits_eq(&run(1), &run(4), usize::MAX, "1 vs 4 threads");
        let seq = pls1_fit_lanes(
            &StripSource::Copy(xs.as_ref()),
            &lanes,
            x_fro,
            k,
            None,
            Par::Seq,
        );
        assert_lanes_bits_eq(&seq, &run(4), usize::MAX, "Seq vs row ranges");
    }

    /// Every block of `STRIP_COLS` rows of every strip of `src`, in place.
    fn strips(src: &StripSource<'_>) -> Mat<f64> {
        let (n, p) = src.shape();
        let mut scratch = src.scratch(STRIP_COLS);
        let mut out = Mat::<f64>::zeros(n, p);
        for i0 in (0..n).step_by(STRIP_COLS) {
            let h = STRIP_COLS.min(n - i0);
            for j0 in (0..p).step_by(STRIP_COLS) {
                let w = STRIP_COLS.min(p - j0);
                out.submatrix_mut(i0, j0, h, w).copy_from(src.block(
                    (i0, h),
                    (j0, w),
                    &mut scratch,
                ));
            }
        }
        out
    }

    #[test]
    #[allow(clippy::similar_names, clippy::cast_precision_loss)]
    fn gather_strips_match_copy() {
        // Three row blocks and three strips of each half, the last ones shorter.
        let (n, p) = (300, 150);
        let x = unif(n, p, 41);
        let lay = Layouts::new(x.as_ref());
        let tr: Vec<usize> = (0..n).step_by(2).collect();
        let te: Vec<usize> = (1..n).step_by(2).collect();
        let scale = Col::<f64>::from_fn(p, |j| 0.5 + (j % 7) as f64 * 0.25);
        let w_pos = Col::<f64>::from_fn(tr.len(), |i| 0.25 + (i % 7) as f64 * 0.5);
        // Zero at training row 10.
        let w_zero = test_weights(tr.len());
        let sw_te = Col::<f64>::from_fn(te.len(), |i| 0.5 + (i % 3) as f64);
        for (view, xv) in lay.all(&x) {
            for (label, w) in [
                ("unweighted", None),
                ("weighted", Some(&w_pos)),
                ("zero-weight row", Some(&w_zero)),
            ] {
                let what = format!("{view} {label}");
                let sw_tr = w.map(|w| sqrt_col(renormalize_mean_one(w.as_ref()).as_ref()));
                let (wr, sr) = (w.map(Col::as_ref), sw_tr.as_ref().map(Col::as_ref));
                let se = w.map(|_| sw_te.as_ref());
                let (copy_tr, mean, fro) =
                    center_rows_fro(xv, &tr, wr, sr, scale.as_ref(), Par::Seq);
                let copy_te = standardize_apply_rows(xv, &te, mean.as_ref(), scale.as_ref(), se);
                let gp = gather_pass(xv, &tr, &te, wr, sr, se, scale.as_ref());
                assert_bits_eq(
                    &col_vals(gp.mean.as_ref()),
                    &col_vals(mean.as_ref()),
                    &format!("{what}: means"),
                );
                assert_eq!(gp.x_fro.to_bits(), fro.to_bits(), "{what}: x_fro");
                let (g_tr, g_te) =
                    gather_sources(xv, &tr, &te, sr, se, scale.as_ref(), gp.mean.clone());
                assert_mat_bits_eq(
                    strips(&g_tr).as_ref(),
                    copy_tr.as_ref(),
                    &format!("{what}: training"),
                );
                assert_mat_bits_eq(
                    strips(&g_te).as_ref(),
                    copy_te.as_ref(),
                    &format!("{what}: test"),
                );
            }
        }
    }

    #[test]
    #[allow(clippy::similar_names, clippy::cast_precision_loss)]
    fn gather_matches_copy_large_means() {
        // n_tr = 150 rows and 300 columns: several row ranges per product.
        let (n, p, k) = (300, 300, 5);
        let s = Col::<f64>::from_fn(p, |j| 0.5 + (j % 5) as f64 * 0.5);
        let u = unif(n, p, 51);
        let x = Mat::<f64>::from_fn(n, p, |i, j| s[j] * (1e6 + u[(i, j)]));
        let scale = standardize(x.as_ref()).2;
        let tr: Vec<usize> = (0..n).step_by(2).collect();
        let te: Vec<usize> = (1..n).step_by(2).collect();
        let n_tr = tr.len();
        let w = test_weights(n_tr);
        let sw_tr = sqrt_col(renormalize_mean_one(w.as_ref()).as_ref());
        let ys: Vec<Option<Col<f64>>> = (0..6)
            .map(|c| {
                let y = if c % 2 == 0 {
                    let ut = Mat::<f64>::from_fn(n_tr, p, |i, j| u[(tr[i], j)]);
                    linear_y(&ut, 0.5, 52 + c)
                } else {
                    unif_col(n_tr, 52 + c)
                };
                Some(standardize1(y.as_ref()).0)
            })
            .collect();
        let par = par_fixed();
        for weighted in [false, true] {
            let what = format!("weighted={weighted}");
            let (wr, sr) = if weighted {
                (Some(w.as_ref()), Some(sw_tr.as_ref()))
            } else {
                (None, None)
            };
            let (copy_tr, _, fro) =
                center_rows_fro(x.as_ref(), &tr, wr, sr, scale.as_ref(), Par::Seq);
            let gp = gather_pass(x.as_ref(), &tr, &te, wr, sr, None, scale.as_ref());
            assert_eq!(gp.x_fro.to_bits(), fro.to_bits(), "{what}: x_fro");
            let (g_tr, _) = gather_sources(
                x.as_ref(),
                &tr,
                &te,
                sr,
                None,
                scale.as_ref(),
                gp.mean.clone(),
            );
            let copy = pls1_fit_lanes(&StripSource::Copy(copy_tr.as_ref()), &ys, fro, k, None, par);
            let gather = pls1_fit_lanes(&g_tr, &ys, gp.x_fro, k, None, par);
            assert!(
                copy.iter()
                    .all(|f| f.as_ref().is_some_and(|f| f.w.ncols() == k)),
                "{what}: every lane lives to k"
            );
            assert_lanes_bits_eq(
                &copy,
                &gather,
                usize::MAX,
                &format!("{what}: copy vs gather"),
            );
        }
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn gather_pass_flags_nonfinite() {
        let (n, p) = (40, 10);
        let x = unif(n, p, 61);
        let scale = standardize(x.as_ref()).2;
        let tr: Vec<usize> = (0..n).step_by(2).collect();
        let te: Vec<usize> = (1..n).step_by(2).collect();
        // Zero at training row 10, X row 20.
        let w = test_weights(tr.len());
        let sw_tr = sqrt_col(renormalize_mean_one(w.as_ref()).as_ref());
        let sw_te = Col::<f64>::from_fn(te.len(), |_| 1.0);
        // (X row set far, weighted, expected (finite_tr, finite_te)).
        for (far, weighted, expect) in [
            (20, true, (true, true)),
            (21, true, (true, false)),
            (20, false, (false, true)),
        ] {
            let what = format!("far row {far} weighted={weighted}");
            let mut xf = x.clone();
            xf[(far, 3)] = 1.5e308;
            let (wr, sr, se) = if weighted {
                (Some(w.as_ref()), Some(sw_tr.as_ref()), Some(sw_te.as_ref()))
            } else {
                (None, None, None)
            };
            let (a, mean, _) = center_rows_fro(xf.as_ref(), &tr, wr, sr, scale.as_ref(), Par::Seq);
            let c = standardize_apply_rows(xf.as_ref(), &te, mean.as_ref(), scale.as_ref(), se);
            let gp = gather_pass(xf.as_ref(), &tr, &te, wr, sr, se, scale.as_ref());
            assert_eq!(
                gp.finite_tr,
                check_finite_mat(a.as_ref()).is_ok(),
                "{what}: training"
            );
            assert_eq!(
                gp.finite_te,
                check_finite_mat(c.as_ref()).is_ok(),
                "{what}: test"
            );
            assert_eq!((gp.finite_tr, gp.finite_te), expect, "{what}");
        }
    }
}
