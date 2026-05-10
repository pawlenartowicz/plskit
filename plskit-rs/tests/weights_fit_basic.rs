//! Basic weighted fit correctness tests.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)] // intentional bit-exact comparisons for determinism tests

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
fn weights_none_self_consistent() {
    let (x, y) = fixture(0);
    let m_a = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )
    .unwrap();
    let m_b = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )
    .unwrap();
    for j in 0..m_a.beta.nrows() {
        assert_eq!(m_a.beta[j], m_b.beta[j]);
    }
    assert_eq!(m_a.intercept, m_b.intercept);
}

#[test]
fn weights_uniform_invariance() {
    let (x, y) = fixture(1);
    let n = x.nrows();
    let w = Col::<f64>::from_fn(n, |_| 7.5);
    let m_w = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    )
    .unwrap();
    let m_n = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )
    .unwrap();
    for j in 0..m_w.beta.nrows() {
        assert!((m_w.beta[j] - m_n.beta[j]).abs() < 1e-12);
    }
    assert!((m_w.intercept - m_n.intercept).abs() < 1e-12);
}

#[test]
fn weights_nan_rejected() {
    let (x, y) = fixture(2);
    let n = x.nrows();
    let mut w = Col::<f64>::from_fn(n, |_| 1.0);
    w[0] = f64::NAN;
    let r = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    );
    assert!(matches!(r, Err(plskit::error::PlsKitError::NonFiniteInput)));
}

#[test]
fn weights_negative_rejected() {
    let (x, y) = fixture(3);
    let n = x.nrows();
    let mut w = Col::<f64>::from_fn(n, |_| 1.0);
    w[0] = -0.5;
    let r = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    );
    assert!(matches!(
        r,
        Err(plskit::error::PlsKitError::InvalidWeights { reason: "negative" })
    ));
}

#[test]
fn weights_all_zero_rejected() {
    let (x, y) = fixture(4);
    let n = x.nrows();
    let w = Col::<f64>::from_fn(n, |_| 0.0);
    let r = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    );
    assert!(matches!(
        r,
        Err(plskit::error::PlsKitError::InvalidWeights { reason: "all_zero" })
    ));
}

#[test]
fn weights_insufficient_n_eff_rejected() {
    let (x, y) = fixture(5);
    let n = x.nrows();
    // Concentrate weight on 1 row → n_eff ≈ 1. k=2 needs n_eff ≥ 3.
    let w = Col::<f64>::from_fn(n, |i| if i == 0 { 1.0 } else { 1e-6 });
    let r = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    );
    assert!(matches!(
        r,
        Err(plskit::error::PlsKitError::InvalidWeights {
            reason: "insufficient_effective_n"
        })
    ));
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
    assert!(m.weights.is_some());
    assert!((m.n_eff - plskit::linalg::compute_n_eff(w.as_ref())).abs() < 1e-12);

    // Uniform weights → echo as None (per spec §3.6 / open issue #2 recommended resolution).
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
