//! The fitted model records `n_eff` and echoes non-uniform weights, and
//! echoes all-equal weights as `None`. That all-equal weights are the same
//! call as absent weights at every entry is pinned by
//! `fit::boundary_tests::equal_weights_are_absent_weights_at_every_entry`.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)] // intentional bit-exact: uniform weights must be identical to None

use faer::{Col, Mat};
use plskit::fit::{pls1_fit, FitOpts, KSpec};

fn fixture(seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::RngExt;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let n = 50;
    let p = 6;
    let x = Mat::from_fn(n, p, |_, _| rng.random_range(0.0..1.0_f64));
    let y = Col::<f64>::from_fn(n, |i| {
        (0..p).map(|j| x[(i, j)]).sum::<f64>() + rng.random_range(-0.1..0.1_f64)
    });
    (x, y)
}

#[test]
fn pls1_model_records_n_eff_and_weights() {
    let (x, y) = fixture(6);
    let n = x.nrows();
    let w = Col::<f64>::from_fn(n, |i| (i + 1) as f64);
    let m = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    )
    .unwrap();
    // The echoed weights are the normalized (mean-one) input, to the bit.
    let echoed = m.weights.as_ref().expect("non-uniform weights are echoed");
    let want = plskit::linalg::normalize_weights(w.as_ref()).unwrap();
    assert_eq!(echoed.nrows(), n);
    for i in 0..n {
        assert_eq!(echoed[i].to_bits(), want[i].to_bits(), "weights[{i}]");
    }
    assert!((m.n_eff - plskit::linalg::compute_n_eff(w.as_ref())).abs() < 1e-12);

    // Uniform weights → echo as None (uniform-weight invariance: all-equal weights fit as none).
    let w_uniform = Col::<f64>::from_fn(n, |_| 1.0);
    let m_u = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w_uniform.as_ref()),
        FitOpts::default(),
    )
    .unwrap();
    assert!(m_u.weights.is_none());
    assert!((m_u.n_eff - n as f64).abs() < 1e-12);
}
