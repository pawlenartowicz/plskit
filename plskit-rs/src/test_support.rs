//! Helpers shared by the unit tests of several modules. Test-only.

use std::fmt::Debug;

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::PlsKitResult;
use crate::perm_null::PermNullOutput;

/// Remove from `v` its component in the span of the orthonormal `basis`,
/// with one reorthogonalization pass (each projection applied twice).
pub(crate) fn project_off(basis: &[Col<f64>], v: &mut Col<f64>) {
    for _ in 0..2 {
        for b in basis {
            let d: f64 = (0..v.nrows()).map(|i| v[i] * b[i]).sum();
            for i in 0..v.nrows() {
                v[i] -= d * b[i];
            }
        }
    }
}

/// Orthonormal basis of `span{lead, columns of m}` by Gram-Schmidt with
/// [`project_off`]. A candidate whose norm after projection is not above
/// `drop_rel` times its norm before is taken as dependent on the earlier
/// ones and left out; `drop_rel = 0.0` keeps every nonzero candidate.
pub(crate) fn orthonormal_basis(
    lead: ColRef<'_, f64>,
    m: MatRef<'_, f64>,
    drop_rel: f64,
) -> Vec<Col<f64>> {
    let n = lead.nrows();
    let mut basis: Vec<Col<f64>> = Vec::new();
    for j in 0..=m.ncols() {
        let mut v = if j == 0 {
            lead.to_owned()
        } else {
            m.col(j - 1).to_owned()
        };
        let n0 = v.norm_l2();
        project_off(&basis, &mut v);
        let nv = v.norm_l2();
        if nv > drop_rel * n0 {
            basis.push(Col::<f64>::from_fn(n, |i| v[i] / nv));
        }
    }
    basis
}

/// `to_bits` equality of two `f64` slices, naming the first index that
/// differs.
pub(crate) fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x:e} vs {y:e}");
    }
}

/// The bit patterns of `v`, for `assert_eq!` / `assert_ne!` on whole slices.
pub(crate) fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Column-major entries of `m`, for [`assert_bits_eq`].
pub(crate) fn mat_vals(m: MatRef<'_, f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * m.ncols());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            v.push(m[(i, j)]);
        }
    }
    v
}

/// Entries of `c`, for [`assert_bits_eq`].
pub(crate) fn col_vals(c: ColRef<'_, f64>) -> Vec<f64> {
    (0..c.nrows()).map(|i| c[i]).collect()
}

/// Same shape and `to_bits` equality of every entry, naming the first
/// `(row, column)` that differs.
pub(crate) fn assert_mat_bits_eq(a: MatRef<'_, f64>, b: MatRef<'_, f64>, what: &str) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{what}: shape"
    );
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let (x, y) = (a[(i, j)], b[(i, j)]);
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}[({i}, {j})]: {x:e} vs {y:e}"
            );
        }
    }
}

/// On `d`-wide rows: NaN in the same places, and every row within
/// `1e-10·max(1, ‖reference row‖∞)`.
pub(crate) fn assert_rows_close(a: &[f64], b: &[f64], d: usize, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (r, (ra, rb)) in a.chunks(d).zip(b.chunks(d)).enumerate() {
        let scale = rb
            .iter()
            .filter(|v| v.is_finite())
            .fold(1.0_f64, |m, v| m.max(v.abs()));
        for (j, (x, y)) in ra.iter().zip(rb).enumerate() {
            assert!(
                (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-10 * scale,
                "{what}: row {r}, column {j}: {x} vs {y}"
            );
        }
    }
}

/// The reduced outputs of two `pls1_perm_null` runs: same lengths, NaN in
/// the same places, `1e-10` absolute.
pub(crate) fn assert_summaries_close(a: &PermNullOutput, b: &PermNullOutput, what: &str) {
    for (name, va, vb) in [
        ("beta_perm_mean", &a.beta_perm_mean, &b.beta_perm_mean),
        ("beta_perm_sd", &a.beta_perm_sd, &b.beta_perm_sd),
        ("beta_perm_z", &a.beta_perm_z, &b.beta_perm_z),
    ] {
        assert_eq!(va.len(), vb.len(), "{what}: {name} length");
        for (j, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert!(
                x.is_nan() == y.is_nan() && (x.is_nan() || (x - y).abs() <= 1e-10),
                "{what}: {name}[{j}]: {x:e} vs {y:e}"
            );
        }
    }
}

/// `(x, y)` with `x` uniform on `[-1, 1)` and `y = 2·x₀ − x₁ + noise`
/// (`d >= 2`).
pub(crate) fn signal_data(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| {
        2.0 * x[(i, 0)] - x[(i, 1)] + 0.5 * rng.random_range(-1.0..1.0)
    });
    (x, y)
}

/// `(x, y)` with `x` uniform on `[-1, 1)` and
/// `y = snr·Σ_{j < k_signal} x_j + noise`, noise uniform on `[-1, 1)`.
/// `k_signal = 0` gives `y` independent of `X`.
pub(crate) fn synth(
    n: usize,
    d: usize,
    k_signal: usize,
    snr: f64,
    seed: u64,
) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let beta = Col::<f64>::from_fn(d, |j| if j < k_signal { 1.0 } else { 0.0 });
    let y_signal: Col<f64> = x.as_ref() * beta.as_ref();
    let y = Col::<f64>::from_fn(n, |i| y_signal[i] * snr + rng.random_range(-1.0..1.0));
    (x, y)
}

/// `n × d` uniform on `[-1, 1)`.
pub(crate) fn unif(n: usize, d: usize, seed: u64) -> Mat<f64> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0))
}

/// Length-`n` uniform on `[-1, 1)`.
pub(crate) fn unif_col(n: usize, seed: u64) -> Col<f64> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0))
}

/// `y = X b + noise·e`, `b_j = 1/(j + 1)`, `e` uniform.
pub(crate) fn linear_y(x: &Mat<f64>, noise: f64, seed: u64) -> Col<f64> {
    let e = unif_col(x.nrows(), seed);
    Col::<f64>::from_fn(x.nrows(), |i| {
        (0..x.ncols())
            .map(|j| x[(i, j)] / (j as f64 + 1.0))
            .sum::<f64>()
            + noise * e[i]
    })
}

/// The arrays a fixed-X loop hands its kernel: X and y standardized
/// (weighted moments under `normalize_weights(w)` when weighted), both
/// then scaled by `fit_row_scale` of those weights.
pub(crate) fn prepared(x: &Mat<f64>, y: &Col<f64>, w: Option<&Col<f64>>) -> (Mat<f64>, Col<f64>) {
    use crate::linalg::{
        normalize_weights, standardize, standardize1, standardize1_weighted, standardize_weighted,
    };
    match w {
        None => {
            let (xs, _, _) = standardize(x.as_ref());
            let (ys, _, _) = standardize1(y.as_ref());
            (xs, ys)
        }
        Some(w) => {
            let wn = normalize_weights(w.as_ref()).expect("some weight is nonzero");
            let (xs, _, _) = standardize_weighted(x.as_ref(), Some(wn.as_ref()));
            let (ys, _, _) = standardize1_weighted(y.as_ref(), Some(wn.as_ref()));
            let sqw = crate::fit::fit_row_scale(wn.as_ref());
            (
                crate::fit::scale_rows(xs.as_ref(), sqw.as_ref()),
                crate::fit::scale_col(ys.as_ref(), sqw.as_ref()),
            )
        }
    }
}

/// `√n · U diag(σ) V'` with `σ_j = κ^(−j/(p−1))`: condition number
/// exactly `kappa` up to rounding. Columns are not centered; the kernel
/// comparison does not need them to be.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
pub(crate) fn conditioned(n: usize, p: usize, kappa: f64, seed: u64) -> Mat<f64> {
    let a = unif(n, p, seed);
    let b = unif(p, p, seed + 1);
    let ua = a.thin_svd().expect("svd");
    let vb = b.thin_svd().expect("svd");
    let (u, v) = (ua.U(), vb.U());
    let sig = |j: usize| {
        if p == 1 {
            1.0
        } else {
            kappa.powf(-(j as f64) / (p as f64 - 1.0))
        }
    };
    let scale = (n as f64).sqrt();
    Mat::<f64>::from_fn(n, p, |i, j| {
        (0..p).map(|l| u[(i, l)] * sig(l) * v[(j, l)]).sum::<f64>() * scale
    })
}

/// A `y` orthogonal to `1` and to every column of `xs` (Gram-Schmidt,
/// each projection applied twice): `X'y` is rounding noise.
pub(crate) fn orthogonal_y(xs: &Mat<f64>, seed: u64) -> Col<f64> {
    let n = xs.nrows();
    let basis = orthonormal_basis(Col::<f64>::from_fn(n, |_| 1.0).as_ref(), xs.as_ref(), 0.0);
    let e = unif_col(n, seed);
    let mut y = Col::<f64>::from_fn(n, |i| 5.0 + e[i]);
    project_off(&basis, &mut y);
    y
}

/// Uneven observation weights with a mean other than one and a zero on
/// every eleventh row.
pub(crate) fn test_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| {
        if i % 11 == 10 {
            0.0
        } else {
            0.25 + (i % 7) as f64 * 0.5
        }
    })
}

/// Uneven observation weights, zero on every ninth row from row 0.
pub(crate) fn gram_p_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| {
        if i % 9 == 0 {
            0.0
        } else {
            0.5 + (i % 5) as f64 * 0.3
        }
    })
}

/// The values of one matrix in three more layouts: a column-major submatrix
/// of a larger NaN-padded buffer (non-zero offset, column stride above the
/// row count), a row-major view (the transpose of a stored transpose,
/// which is not column-major), and a column-major view with a *negative*
/// column stride (`reverse_cols()` of a matrix whose columns are stored in
/// reverse order, so the values read back in original order). Degenerate
/// shapes (no rows, no columns, one row) are accepted: the self-checks that
/// cannot hold for them (a one-row view is both row- and column-major; an
/// empty view has no meaningful stride) are skipped.
pub(crate) struct Layouts {
    padded: Mat<f64>,
    transposed: Mat<f64>,
    col_reversed: Mat<f64>,
    n: usize,
    d: usize,
}

impl Layouts {
    pub(crate) fn new(x: MatRef<'_, f64>) -> Self {
        let (n, d) = (x.nrows(), x.ncols());
        let padded = Mat::<f64>::from_fn(n + 3, d + 2, |i, j| {
            if (1..=n).contains(&i) && (1..=d).contains(&j) {
                x[(i - 1, j - 1)]
            } else {
                f64::NAN
            }
        });
        let transposed = Mat::<f64>::from_fn(d, n, |j, i| x[(i, j)]);
        let col_reversed = Mat::<f64>::from_fn(n, d, |i, j| x[(i, d - 1 - j)]);
        let l = Self {
            padded,
            transposed,
            col_reversed,
            n,
            d,
        };
        assert!(l.submatrix().try_as_col_major().is_some());
        if n >= 2 {
            assert!(l.row_major().try_as_col_major().is_none());
        }
        if n >= 1 && d >= 1 {
            assert!(l.reversed_cols().try_as_col_major().is_some());
            assert!(l.reversed_cols().col_stride() < 0);
        }
        l
    }

    pub(crate) fn submatrix(&self) -> MatRef<'_, f64> {
        self.padded.as_ref().submatrix(1, 1, self.n, self.d)
    }

    pub(crate) fn row_major(&self) -> MatRef<'_, f64> {
        self.transposed.as_ref().transpose()
    }

    /// A column-major view with negative column stride: `try_as_col_major`
    /// returns `Some` (it only checks `row_stride() == 1`), but its values
    /// come from column `d - 1 - j`, not `j`, at storage offset `j`.
    pub(crate) fn reversed_cols(&self) -> MatRef<'_, f64> {
        self.col_reversed.as_ref().reverse_cols()
    }

    /// `x` itself, the submatrix view, the row-major view and the
    /// negative-column-stride view, labelled. `"owned"` comes first; the
    /// layout-invariance helpers below rely on that order.
    pub(crate) fn all<'a>(&'a self, x: &'a Mat<f64>) -> [(&'static str, MatRef<'a, f64>); 4] {
        [
            ("owned", x.as_ref()),
            ("submatrix", self.submatrix()),
            ("row_major", self.row_major()),
            ("reversed_cols", self.reversed_cols()),
        ]
    }
}

/// One input family of the layout-invariance tests.
pub(crate) struct Family {
    pub(crate) name: &'static str,
    pub(crate) x: Mat<f64>,
    pub(crate) y: Col<f64>,
    /// Raw observation weights (not renormalized); `None` when unweighted.
    pub(crate) w: Option<Col<f64>>,
}

impl Family {
    /// `(x, y)` as a layout test feeds them: raw, or for `pre = true`
    /// standardized without weights (`linalg::standardize` /
    /// `standardize1`). Standardizing with unweighted moments keeps
    /// `pre_standardized` observable on the weighted families.
    pub(crate) fn inputs(&self, pre: bool) -> (Mat<f64>, Col<f64>) {
        if pre {
            let (xs, _, _) = crate::linalg::standardize(self.x.as_ref());
            let (ys, _, _) = crate::linalg::standardize1(self.y.as_ref());
            (xs, ys)
        } else {
            (self.x.clone(), self.y.clone())
        }
    }
}

/// Dense (48 x 7), weighted dense, p >> n (30 x 400) and weighted p >> n.
/// Sparse fits (`keep`) and the layouts of [`Layouts`] are crossed with
/// these inside each test; tests that have a `pre_standardized` axis cross
/// that too.
pub(crate) fn copy_free_families() -> Vec<Family> {
    let (xd, yd) = signal_data(48, 7, 101);
    let (xw, yw) = signal_data(30, 400, 102);
    vec![
        Family {
            name: "dense",
            x: xd.clone(),
            y: yd.clone(),
            w: None,
        },
        Family {
            name: "weighted",
            x: xd,
            y: yd,
            w: Some(test_weights(48)),
        },
        Family {
            name: "wide",
            x: xw.clone(),
            y: yw.clone(),
            w: None,
        },
        Family {
            name: "weighted_wide",
            x: xw,
            y: yw,
            w: Some(test_weights(30)),
        },
    ]
}

// ---------------------------------------------------------------------------
// Layout invariance. One core primitive, `for_each_layout`,
// and three thin assertions on top:
// - `assert_layout_invariant`: one matrix, whole output bit-identical (Debug
//   rendering);
// - `assert_layout_agree`: one matrix, flat values per `Agree` (bits or a
//   relative tolerance), errors compared by code and message;
// - `assert_families_layout_invariant`: `copy_free_families()` x pre x
//   layouts, flat values bit-identical, every run `Ok`.
// ---------------------------------------------------------------------------

/// The core: runs `run` on `x` itself (owned, column-major, first) and on
/// the other three views of [`Layouts::new`]`(x)` (padded submatrix,
/// row-major, negative column stride), calls `check(view, &view_result,
/// &owned_result)` for each of the three, and returns the owned result.
/// `run` gets the view label with the view. It is called four times, so a
/// run that draws from an RNG must build it from its seed inside `run`.
pub(crate) fn for_each_layout<T>(
    x: &Mat<f64>,
    mut run: impl FnMut(&'static str, MatRef<'_, f64>) -> T,
    mut check: impl FnMut(&'static str, &T, &T),
) -> T {
    let lay = Layouts::new(x.as_ref());
    let views = lay.all(x);
    let (owned_label, owned_view) = views[0];
    let owned = run(owned_label, owned_view);
    for &(view, xv) in &views[1..] {
        let got = run(view, xv);
        check(view, &got, &owned);
    }
    owned
}

/// Bit identity of a whole output across layouts: `run` on every view of
/// `x` renders (`{:?}`) exactly like `run` on `x` itself. `{:?}` of an
/// `f64` is its shortest round-trip string, so equal renderings mean equal
/// bits (and equal `-0.0` signs); every NaN renders as `NaN`, whatever its
/// sign and payload, which Rust leaves unspecified. A `PlsKitResult`
/// output compares `Ok` values and error variants (with their fields)
/// alike; unwrap inside `run` when an error must fail the test. Panics
/// naming the view and the first differing position. Returns the owned
/// run's output.
pub(crate) fn assert_layout_invariant<T: Debug>(
    x: &Mat<f64>,
    what: &str,
    mut run: impl FnMut(MatRef<'_, f64>) -> T,
) -> T {
    for_each_layout(
        x,
        |_, xv| run(xv),
        |view, got, want| {
            let (g, w) = (format!("{got:?}"), format!("{want:?}"));
            if g != w {
                let (g, w): (Vec<char>, Vec<char>) = (g.chars().collect(), w.chars().collect());
                let i = g.iter().zip(&w).take_while(|(a, b)| a == b).count();
                let lo = i.saturating_sub(60);
                let snip = |c: &[char]| {
                    c[lo.min(c.len())..(i + 60).min(c.len())]
                        .iter()
                        .collect::<String>()
                };
                panic!(
                    "{what} [{view}]: output differs from the owned run at char {i}\n  \
                     {view}: ...{}...\n  owned: ...{}...",
                    snip(&g),
                    snip(&w)
                );
            }
        },
    )
}

/// How values from another layout must agree with the owned run's.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Agree {
    /// `to_bits` equality, any NaN matching any NaN (Rust leaves a NaN's
    /// sign and payload unspecified). The contract of the resampling and
    /// inference engines, which copy X column-major, so their output is layout-independent.
    Bits,
    /// `u == v` or `|u - v| <= atol + 1e-14 · |v|` per entry (`v` the owned
    /// run's value), NaN matching NaN: the corpus rule at the given `atol`,
    /// owned by `plskit-testdata-gen/src/settle.rs::close` (its `rtol` is the
    /// `1e-14` in `assert_agree`: change together). For the entries that form their
    /// products in X's own layout and so agree across layouts only to
    /// rounding (`pls1_fit` / `spls1_fit`, which form products in X's own
    /// layout, and what reads them: `pls1_predict`, `pls1_rotation_stability`). Lengths are
    /// still exact, and an integer (a count, `k_used`) that differs by one
    /// fails any `atol` below 1.
    Corpus(f64),
}

/// `got` and `want` have the same length and agree entry by entry per
/// `agree`; panics naming `what` and the first failing index.
#[allow(clippy::float_cmp)] // `u == v` first: equal infinities (inf - inf is NaN)
pub(crate) fn assert_agree(got: &[f64], want: &[f64], agree: Agree, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, (&u, &v)) in got.iter().zip(want).enumerate() {
        let ok = if u.is_nan() || v.is_nan() {
            u.is_nan() && v.is_nan()
        } else {
            match agree {
                Agree::Bits => u.to_bits() == v.to_bits(),
                // `1e-14` is the corpus rtol; the rule is owned by
                // `plskit-testdata-gen/src/settle.rs::close`: change together.
                Agree::Corpus(atol) => u == v || (u - v).abs() <= atol + 1e-14 * v.abs(),
            }
        };
        assert!(ok, "{what}[{i}]: {u:e} vs {v:e} ({agree:?})");
    }
}

/// Flat values across layouts: `run` on every view of `x` either returns
/// `Ok` values that agree with the owned run's per `agree`
/// ([`assert_agree`]), or fails with the owned run's error (same `code()`
/// and message). `Ok` on one layout and `Err` on another panics. Returns
/// the owned run's result (e.g. to compare two owned configurations with
/// [`assert_agree`]). Flatten outputs with [`mat_vals`] / [`col_vals`].
pub(crate) fn assert_layout_agree(
    x: &Mat<f64>,
    what: &str,
    agree: Agree,
    mut run: impl FnMut(MatRef<'_, f64>) -> PlsKitResult<Vec<f64>>,
) -> PlsKitResult<Vec<f64>> {
    for_each_layout(
        x,
        |_, xv| run(xv),
        |view, got, want| {
            let what = format!("{what} [{view}]");
            match (got, want) {
                (Ok(g), Ok(w)) => assert_agree(g, w, agree, &what),
                (Err(g), Err(w)) => {
                    assert_eq!(g.code(), w.code(), "{what}: error code");
                    assert_eq!(g.to_string(), w.to_string(), "{what}: error message");
                }
                (g, w) => panic!("{what}: {view} gave {g:?}, owned gave {w:?}"),
            }
        },
    )
}

/// One run of [`assert_families_layout_invariant`].
pub(crate) struct LayoutCase<'a> {
    /// Whether `x_owned` / `y` are the unweighted-standardized inputs
    /// ([`Family::inputs`]); pass it on as `pre_standardized`.
    pub(crate) pre: bool,
    /// The owned column-major X this case's views show: fit references here
    /// so that only the code under test sees the layout.
    pub(crate) x_owned: &'a Mat<f64>,
    /// The view of `x_owned` under test.
    pub(crate) x: MatRef<'a, f64>,
    /// `y` matching `x_owned` (standardized when `pre`).
    pub(crate) y: ColRef<'a, f64>,
    /// The family's raw weights (not renormalized), `None` when unweighted.
    pub(crate) w: Option<ColRef<'a, f64>>,
}

/// [`copy_free_families`] × `pre` in {false, true} × the four layouts of
/// X: `run` must be `Ok` on every case, and on each view `flat` of its
/// output must equal `flat` of the owned run's to the bit
/// ([`Agree::Bits`]). Labels read `"{what} {family} pre={pre} [{view}]"`.
/// Only X varies; `y` and the weights stay owned.
pub(crate) fn assert_families_layout_invariant<T>(
    what: &str,
    run: impl Fn(&LayoutCase<'_>) -> PlsKitResult<T>,
    flat: impl Fn(&T) -> Vec<f64>,
) {
    for f in copy_free_families() {
        let w = f.w.as_ref().map(Col::as_ref);
        for pre in [false, true] {
            let (x_owned, y) = f.inputs(pre);
            let label = format!("{what} {} pre={pre}", f.name);
            let ok = |view: &str, r: &PlsKitResult<T>| match r {
                Ok(v) => flat(v),
                Err(e) => panic!("{label} [{view}]: {e:?}"),
            };
            let owned = for_each_layout(
                &x_owned,
                |_view, xv| {
                    run(&LayoutCase {
                        pre,
                        x_owned: &x_owned,
                        x: xv,
                        y: y.as_ref(),
                        w,
                    })
                },
                |view, got, want| {
                    let want = ok("owned", want);
                    assert_agree(
                        &ok(view, got),
                        &want,
                        Agree::Bits,
                        &format!("{label} [{view}]"),
                    );
                },
            );
            ok("owned", &owned);
        }
    }
}

/// Negative controls: every layout helper panics on a run whose output
/// depends on the view's strides, so no helper edit can make the per-module
/// layout tables pass vacuously.
mod helper_controls {
    use super::{
        assert_agree, assert_families_layout_invariant, assert_layout_agree,
        assert_layout_invariant, signal_data, Agree,
    };

    #[test]
    #[should_panic(expected = "control")]
    fn assert_layout_invariant_catches_a_layout_dependent_run() {
        let (x, _) = signal_data(12, 4, 1);
        assert_layout_invariant(&x, "control", |xv| (xv.row_stride(), xv.col_stride()));
    }

    #[test]
    #[should_panic(expected = "control")]
    #[allow(clippy::cast_precision_loss)]
    fn assert_layout_agree_catches_a_layout_dependent_run() {
        let (x, _) = signal_data(12, 4, 1);
        let _ = assert_layout_agree(&x, "control", Agree::Corpus(1e-10), |xv| {
            Ok(vec![xv.row_stride() as f64, xv.col_stride() as f64])
        });
    }

    #[test]
    #[should_panic(expected = "control")]
    #[allow(clippy::cast_precision_loss)]
    fn assert_families_layout_invariant_catches_a_layout_dependent_run() {
        assert_families_layout_invariant(
            "control",
            |c| Ok((c.x.row_stride(), c.x.col_stride())),
            |&(r, s)| vec![r as f64, s as f64],
        );
    }

    #[test]
    #[should_panic(expected = "control")]
    fn agree_corpus_rejects_a_difference_above_its_tolerance() {
        assert_agree(&[1.0], &[1.0 + 1e-8], Agree::Corpus(1e-10), "control");
    }
}
