//! Helpers shared by the unit tests of several modules. Test-only.

use faer::{Col, ColRef, Mat, MatRef};

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
/// differs. The byte-identity tests of the copy-free refactor use it.
pub(crate) fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x:e} vs {y:e}");
    }
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

/// The values of one matrix in three more layouts: a column-major submatrix
/// of a larger NaN-padded buffer (non-zero offset, column stride above the
/// row count), a row-major view (the transpose of a stored transpose,
/// which is not column-major), and a column-major view with a *negative*
/// column stride (`reverse_cols()` of a matrix whose columns are stored in
/// reverse order, so the values read back in original order).
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
        assert!(l.row_major().try_as_col_major().is_none());
        assert!(l.reversed_cols().try_as_col_major().is_some());
        assert!(l.reversed_cols().col_stride() < 0);
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
    /// negative-column-stride view, labelled.
    pub(crate) fn all<'a>(&'a self, x: &'a Mat<f64>) -> [(&'static str, MatRef<'a, f64>); 4] {
        [
            ("owned", x.as_ref()),
            ("submatrix", self.submatrix()),
            ("row_major", self.row_major()),
            ("reversed_cols", self.reversed_cols()),
        ]
    }
}

/// One input family of the copy-free byte-identity tests.
pub(crate) struct Family {
    pub(crate) name: &'static str,
    pub(crate) x: Mat<f64>,
    pub(crate) y: Col<f64>,
    /// Raw observation weights (not renormalized); `None` when unweighted.
    pub(crate) w: Option<Col<f64>>,
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
