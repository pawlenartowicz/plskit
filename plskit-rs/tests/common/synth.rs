// Included by `tests/common/mod.rs` and `src/memory_budget_cases.rs`.

/// `x` uniform on `[-1, 1)` and `y = (x₀ + 0.5·x₁)·snr + noise` (`d >= 2`).
pub fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| (x[(i, 0)] + 0.5 * x[(i, 1)]) * snr + noise[i]);
    (x, y)
}
