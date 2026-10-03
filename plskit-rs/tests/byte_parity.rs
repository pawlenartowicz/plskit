//! `ParChoice::Seq` vs `ParChoice::Auto` byte-parity.
//! On the same binary, same platform, fixed seed: the sequential and the
//! parallel arm of a fit must produce byte-identical output when the
//! problem is large enough for `Auto` to take the Rayon path.

use faer::Mat;
use plskit::{ParChoice, Pls3FitOpts};

/// Two-block generator for `spls3_fit`'s serial-vs-parallel parity test.
/// Sized so `resolve_par`'s `n * n_features * n_targets >= 1_000_000`
/// threshold actually selects the rayon path on the `Auto` arm: see
/// `spls3_fit_is_byte_identical_serial_vs_parallel` for the arithmetic.
#[allow(clippy::many_single_char_names)]
fn wide_two_block_data(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
    let y = Mat::<f64>::from_fn(n, q, |_, _| rng.random_range(-1.0..1.0));
    (x, y)
}

#[test]
fn spls3_fit_is_byte_identical_serial_vs_parallel() {
    // n = 200, p = 2000, q = 6: `resolve_par` is called inside `spls3_fit`
    // as `resolve_par(opts.par, x.nrows(), n_features, n_targets)`, so the
    // work product it thresholds against is n * n_features * n_targets =
    // 200 * 2000 * 6 = 2_400_000, comfortably over the 1_000_000 cutoff, so
    // the `Auto` arm genuinely takes the rayon path here (verified by
    // reading `resolve_par` in `plskit-rs/src/fit.rs` and the matmul that
    // forms `A = Xs' Ys` inside `spls3_fit`, whose M*N*K = 2000*6*200 also
    // clears faer's own internal parallel-dispatch threshold).
    let (x, y) = wide_two_block_data(200, 2000, 6, 31);
    let seq = plskit::spls3_fit(
        x.as_ref(),
        y.as_ref(),
        3,
        25,
        3,
        None,
        Pls3FitOpts {
            par: ParChoice::Seq,
            ..Pls3FitOpts::default()
        },
    )
    .unwrap();
    let par = plskit::spls3_fit(
        x.as_ref(),
        y.as_ref(),
        3,
        25,
        3,
        None,
        Pls3FitOpts {
            par: ParChoice::Auto,
            ..Pls3FitOpts::default()
        },
    )
    .unwrap();

    // Pin k_used to its expected value first: a fit that silently returned
    // zero components must not pass this test with every loop below
    // skipped.
    assert_eq!(seq.k_used, 3, "seq.k_used");
    assert_eq!(seq.k_used, par.k_used, "k_used");
    assert_eq!(seq.n_iter, par.n_iter, "n_iter");
    assert_eq!(seq.converged, par.converged, "converged");
    for a in 0..seq.k_used {
        assert_eq!(
            seq.singular_values[a].to_bits(),
            par.singular_values[a].to_bits(),
            "singular_values[{a}]"
        );
        for i in 0..seq.u_saliences.nrows() {
            assert_eq!(
                seq.u_saliences[(i, a)].to_bits(),
                par.u_saliences[(i, a)].to_bits(),
                "u_saliences[({i}, {a})]"
            );
        }
        for i in 0..seq.v_saliences.nrows() {
            assert_eq!(
                seq.v_saliences[(i, a)].to_bits(),
                par.v_saliences[(i, a)].to_bits(),
                "v_saliences[({i}, {a})]"
            );
        }
    }
}
