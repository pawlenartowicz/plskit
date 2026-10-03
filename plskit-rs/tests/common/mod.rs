//! Helpers shared by the integration tests.
#![allow(dead_code)]

use faer::{Col, ColRef, Mat, MatRef};
use plskit::PermNullOpts;

/// `x` uniform on `[-1, 1)` and `y = (x₀ + 0.5·x₁)·snr + noise` (`d >= 2`).
pub fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| (x[(i, 0)] + 0.5 * x[(i, 1)]) * snr + noise[i]);
    (x, y)
}

pub fn perm_opts() -> PermNullOpts {
    PermNullOpts {
        n_perm: 100,
        return_perm_matrix: true,
        pre_standardized: false,
        verbose: false,
    }
}

/// Expand `(x, y)` by integer weights into row-duplicated `(x_dup, y_dup)`.
pub fn duplicate_rows(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w_int: &[u32],
) -> (Mat<f64>, Col<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let total: usize = w_int.iter().map(|&w| w as usize).sum();
    let mut x_dup = Mat::<f64>::zeros(total, p);
    let mut y_dup = Col::<f64>::zeros(total);
    let mut row = 0;
    for i in 0..n {
        for _ in 0..w_int[i] {
            for j in 0..p {
                x_dup[(row, j)] = x[(i, j)];
            }
            y_dup[row] = y[i];
            row += 1;
        }
    }
    (x_dup, y_dup)
}
