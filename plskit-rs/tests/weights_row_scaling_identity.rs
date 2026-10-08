//! The √w row-scaling convention of the weighted fit, checked two ways. A
//! weighted fit of the raw `(X, y)` and a weighted fit of `preprocess`'s
//! standardized outputs with `pre_standardized = true` give the same
//! standardized coefficients (the `preprocess` cache pattern; see
//! `_docs/concepts/PLS1/weights.md`). At full rank the weighted fit is
//! weighted least squares.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs

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

#[test]
fn full_rank_weighted_fit_is_weighted_least_squares() {
    // At k = p the PLS1 scores span the column space of X, so beta and the
    // intercept are the weighted least-squares solution: least squares on the
    // √w-scaled rows of [1 X], solved here by faer's QR.
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let n = 30;
    let p = 4;
    let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
    let b = Col::<f64>::from_fn(p, |_| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| {
        (0..p).map(|j| x[(i, j)] * b[j]).sum::<f64>() + 0.2 * noise[i]
    });
    let w = Col::<f64>::from_fn(n, |_| rng.random_range(0.5..2.0));

    let sw = Col::<f64>::from_fn(n, |i| w[i].sqrt());
    let a = Mat::<f64>::from_fn(n, p + 1, |i, j| {
        let v = if j == 0 { 1.0 } else { x[(i, j - 1)] };
        sw[i] * v
    });
    let rhs = Col::<f64>::from_fn(n, |i| sw[i] * y[i]);
    // params[0] is the intercept, params[1..] the slopes.
    let params = {
        use faer::prelude::SolveLstsq;
        faer::linalg::solvers::Qr::new(a.as_ref()).solve_lstsq(rhs.as_ref())
    };

    // Largest absolute miss of (intercept, beta) against the oracle.
    let miss = |k: usize, weights: Option<&Col<f64>>| {
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(k),
            weights.map(Col::as_ref),
            FitOpts::default(),
        )
        .unwrap();
        (0..p)
            .map(|j| (m.beta[j] - params[j + 1]).abs())
            .fold((m.intercept - params[0]).abs(), f64::max)
    };

    let full = miss(p, Some(&w));
    assert!(full < 1e-10, "k = p weighted fit misses WLS by {full}");
    // The oracle separates the full-rank weighted fit from its neighbours.
    let reduced = miss(p - 1, Some(&w));
    assert!(reduced > 1e-6, "k = p - 1 misses WLS by only {reduced}");
    let plain = miss(p, None);
    assert!(plain > 1e-4, "unweighted fit misses WLS by only {plain}");
}
