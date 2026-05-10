//! Spec §9 test 5: weighted fit ≡ unweighted fit on row-scaled standardized inputs
//! when both go through `pre_standardized=true` (the cache pattern of §5.5).

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]

use faer::{Col, Mat};
use plskit::fit::{pls1_fit, FitOpts, KSpec};
use plskit::preprocess::{preprocess, PreprocessInput};

#[test]
fn cache_pattern_round_trip_parity() {
    let n = 40;
    let p = 5;
    let x = Mat::<f64>::from_fn(n, p, |i, j| ((i * 11 + j * 17) % 31) as f64 / 31.0);
    let y = Col::<f64>::from_fn(n, |i| (i as f64).cos());
    let w = Col::<f64>::from_fn(n, |i| (i + 1) as f64 * 0.7);

    let m_raw = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(3),
        Some(w.as_ref()),
        FitOpts::default(),
    )
    .unwrap();

    let pre = preprocess(PreprocessInput {
        x: Some(x.as_ref()),
        y: Some(y.as_ref()),
        weights: Some(w.as_ref()),
    })
    .unwrap();
    let (xs, _, _) = pre.x_std.unwrap();
    let (ys, _, _) = pre.y_std.unwrap();
    let wn = pre.weights_normalized.unwrap();
    let m_cached = pls1_fit(
        xs.as_ref(),
        ys.as_ref(),
        KSpec::Fixed(3),
        Some(wn.as_ref()),
        FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        },
    )
    .unwrap();

    // Cache pattern: m_cached.coef (in standardized space) should equal m_raw.coef.
    // Because m_cached uses pre_standardized=true: m_cached.beta = m_cached.coef and m_cached.intercept = 0.
    // Whereas m_raw.beta = m_raw.coef * y_scale / x_scale (back-projected).
    // So the comparable invariant is `coef`, not `beta`.
    for j in 0..p {
        assert!(
            (m_raw.coef[j] - m_cached.coef[j]).abs() < 1e-12,
            "coef[{j}] differs raw={} cached={}",
            m_raw.coef[j],
            m_cached.coef[j]
        );
    }
}
