//! Integer weights match row duplication parity test.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]

use faer::{Col, Mat};
use plskit::fit::{pls1_fit, FitOpts, KSpec};

#[test]
fn integer_weights_match_row_duplication() {
    let n = 20;
    let p = 3;
    let x = Mat::<f64>::from_fn(n, p, |i, j| ((i + j) as f64).sin());
    let y = Col::<f64>::from_fn(n, |i| (i as f64) * 0.5);
    let w_int = vec![
        1u32, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1,
    ];

    let w = Col::<f64>::from_fn(n, |i| f64::from(w_int[i]));
    let m_w = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    )
    .unwrap();

    let total: usize = w_int.iter().sum::<u32>() as usize;
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
    let m_d = pls1_fit(
        x_dup.as_ref(),
        y_dup.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )
    .unwrap();

    for j in 0..p {
        assert!(
            (m_w.beta[j] - m_d.beta[j]).abs() < 1e-12,
            "β[{j}] differs: weighted={} dup={}",
            m_w.beta[j],
            m_d.beta[j]
        );
    }
    assert!((m_w.intercept - m_d.intercept).abs() < 1e-12);
}
