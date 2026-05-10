//! Uniform weights c·1 must produce the same fit as weights=None.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)] // intentional bit-exact: uniform weights must be identical to None

use faer::{Col, Mat};
use plskit::fit::{pls1_fit, FitOpts, KSpec};

#[test]
fn weights_c_times_one_equals_none() {
    let n = 50;
    let p = 5;
    let x = Mat::<f64>::from_fn(n, p, |i, j| ((i * 7 + j * 13) % 23) as f64 / 23.0);
    let y = Col::<f64>::from_fn(n, |i| (i as f64 * 0.1).sin());

    for c in [0.5, 1.0, 7.5, 1e6] {
        let w = Col::<f64>::from_fn(n, |_| c);
        let m_w = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            Some(w.as_ref()),
            FitOpts::default(),
        )
        .unwrap();
        let m_n = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap();
        for j in 0..p {
            assert_eq!(m_w.beta[j], m_n.beta[j], "beta[{j}] differs at c={c}");
        }
        assert_eq!(m_w.intercept, m_n.intercept);
    }
}
