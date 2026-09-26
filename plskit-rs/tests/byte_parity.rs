//! Single-thread vs multi-thread byte-parity.
//! On the same binary, same platform, fixed seed: parallel and serial
//! execution must produce byte-identical output. The pre-computed child
//! seeds in `resample::parallel_for_each_seeded` make this structural,
//! not aspirational. This test guards the structural property.

use faer::{Col, Mat};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, pls1_perm_null,
    pls1_rotation_stability, pls3_confirmatory_test, CIOpts, ConfirmatoryArgs, ConfirmatoryMethod,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, FindKOptimalOpts, FindKSequenceOpts, ParChoice,
    PermNullOpts, Pls3ConfirmatoryTestOpts, Pls3FitOpts, RotationStabilityMethod,
    RotationStabilityOpts, Selector, VarimaxArgs,
};

fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| x[(i, 0)] * snr + noise[i]);
    (x, y)
}

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

fn assert_confirmatory_byte_eq(
    a: &plskit::ConfirmatoryTestOutput,
    b: &plskit::ConfirmatoryTestOutput,
    name: &str,
) {
    assert_eq!(a.pvalue.to_bits(), b.pvalue.to_bits(), "{name}.pvalue");
    assert_eq!(
        a.statistic.to_bits(),
        b.statistic.to_bits(),
        "{name}.statistic"
    );
    assert_eq!(a.seed, b.seed, "{name}.seed");
    assert_eq!(a.k, b.k, "{name}.k");
}

#[test]
fn confirmatory_raw_perm_byte_parity() {
    let (x, y) = synth(40, 5, 3.0, 1);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::RawPerm {
            n_perm: 100,
            n_folds: 5,
        },
        seed: Some(7),
        disable_parallelism: dp,
        ..Default::default()
    };
    // weights_none_parity: the assertions below are the byte-for-byte parity
    // contract between pls1_fit(weights=None) and the unweighted fit that
    // predates observation weights. Do not relax these tolerances.
    let serial = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 1,
            weights: None,
        },
        opts(true),
    )
    .unwrap();
    let par = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 1,
            weights: None,
        },
        opts(false),
    )
    .unwrap();
    assert_confirmatory_byte_eq(&serial, &par, "confirmatory_raw_perm");
}

#[test]
fn confirmatory_split_exact_byte_parity() {
    // k = 2, dense (no `keep`): takes the refit route.
    let (x, y) = synth(40, 5, 3.0, 1);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitExact {
            n_perm: 50,
            n_splits: 20,
        },
        seed: Some(11),
        disable_parallelism: dp,
        ..Default::default()
    };
    let serial = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 2,
            weights: None,
        },
        opts(true),
    )
    .unwrap();
    let par = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 2,
            weights: None,
        },
        opts(false),
    )
    .unwrap();
    assert_confirmatory_byte_eq(&serial, &par, "confirmatory_split_exact");
}

#[test]
fn confirmatory_split_exact_no_refit_byte_parity() {
    // k = 1, dense (no `keep`): takes the no-refit route (see signal_test.rs
    // for the K = 1 linear-map identity that makes this an exact shortcut).
    let (x, y) = synth(40, 5, 3.0, 1);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitExact {
            n_perm: 50,
            n_splits: 20,
        },
        seed: Some(12),
        disable_parallelism: dp,
        ..Default::default()
    };
    let serial = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 1,
            weights: None,
        },
        opts(true),
    )
    .unwrap();
    let par = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 1,
            weights: None,
        },
        opts(false),
    )
    .unwrap();
    assert_confirmatory_byte_eq(&serial, &par, "confirmatory_split_exact_no_refit");
}

#[test]
fn confirmatory_ci_bundle_byte_parity() {
    // The `ci` resampling bundle: the per-replicate subsample and bootstrap
    // fits run under Rayon, the reduction (leverage CI and clamp, β CIs, κ̂
    // and the β z) runs serially over rows collected in iteration order, so
    // every array must match bit for bit.
    let (x, y) = synth(120, 6, 3.0, 5);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitExact {
            n_perm: 50,
            n_splits: 20,
        },
        ci: Some(CIOpts {
            n_boot: 200,
            ..CIOpts::default()
        }),
        seed: Some(13),
        disable_parallelism: dp,
        ..Default::default()
    };
    let run = |dp: bool| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            opts(dp),
        )
        .unwrap()
    };
    let serial = run(true);
    let par = run(false);
    assert_confirmatory_byte_eq(&serial, &par, "confirmatory_ci");
    let (a, b) = (serial.ci.unwrap(), par.ci.unwrap());
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    for (name, va, vb) in [
        ("beta_sign_z", &a.beta_sign_z, &b.beta_sign_z),
        (
            "beta_sign_z_signed",
            &a.beta_sign_z_signed,
            &b.beta_sign_z_signed,
        ),
        (
            "leverage_ci_lower",
            &a.leverage_ci_lower,
            &b.leverage_ci_lower,
        ),
        (
            "leverage_ci_upper",
            &a.leverage_ci_upper,
            &b.leverage_ci_upper,
        ),
        ("leverage_se", &a.leverage_se, &b.leverage_se),
        ("beta_ci_lower", &a.beta_ci_lower, &b.beta_ci_lower),
        ("beta_ci_upper", &a.beta_ci_upper, &b.beta_ci_upper),
        ("beta_se", &a.beta_se, &b.beta_se),
    ] {
        assert_eq!(bits(va), bits(vb), "confirmatory_ci.{name}");
    }
    let hc = |c: &plskit::CIScalar| bits(&[c.point, c.lower, c.upper, c.sd]);
    assert_eq!(
        hc(&a.holdout_corr),
        hc(&b.holdout_corr),
        "confirmatory_ci.holdout_corr"
    );
}

#[test]
fn sequence_byte_parity() {
    // Drives sequential.rs through the public API. stop-early may fire,
    // but it fires identically in serial vs parallel for the same seed,
    // so the byte-parity assertion still holds for whichever k_max steps
    // produced p-values.
    let (x, y) = synth(40, 5, 3.0, 1);
    let opts = |dp: bool| FindKSequenceOpts {
        test_method: ConfirmatoryMethod::SplitNb,
        n_splits: 30,
        alpha: 0.05,
        seed: Some(13),
        disable_parallelism: dp,
        ..Default::default()
    };
    let serial = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 3, None, opts(true)).unwrap();
    let par = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 3, None, opts(false)).unwrap();
    assert_eq!(
        serial.pvalues.nrows(),
        par.pvalues.nrows(),
        "pvalues length"
    );
    for i in 0..serial.pvalues.nrows() {
        // Both serial and parallel must agree bitwise — including NaN
        // positions past the early-stop point.
        assert_eq!(
            serial.pvalues[i].to_bits(),
            par.pvalues[i].to_bits(),
            "sequence.pvalues[{i}]"
        );
    }
    assert_eq!(serial.k_star, par.k_star, "k_star");
    assert_eq!(serial.seed, par.seed, "seed");
}

#[test]
fn find_k_optimal_byte_parity() {
    // Larger n + n_folds so the fold loop actually has something to
    // parallelize. cv_scores / cv_scores_se are checked at f64-bit level
    // so any reduction-order divergence between serial and rayon paths
    // would surface immediately.
    let (x, y) = synth(80, 6, 3.0, 1);
    let opts = |dp: bool| FindKOptimalOpts {
        selector: Selector::R2Se,
        n_folds: 10,
        seed: Some(17),
        disable_parallelism: dp,
        ..Default::default()
    };
    let serial = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, None, opts(true)).unwrap();
    let par = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, None, opts(false)).unwrap();
    assert_eq!(serial.k_star, par.k_star, "k_star");
    assert_eq!(serial.seed, par.seed, "seed");

    let s_cv = serial.cv_scores.as_ref().expect("serial cv_scores");
    let p_cv = par.cv_scores.as_ref().expect("par cv_scores");
    assert_eq!(s_cv.len(), p_cv.len(), "cv_scores length");
    for (k, sv) in s_cv {
        let pv = p_cv.get(k).unwrap_or_else(|| panic!("missing key {k}"));
        assert_eq!(sv.to_bits(), pv.to_bits(), "cv_scores[{k}]");
    }

    let s_se = serial.cv_scores_se.as_ref().expect("serial cv_scores_se");
    let p_se = par.cv_scores_se.as_ref().expect("par cv_scores_se");
    for (k, sv) in s_se {
        let pv = p_se.get(k).unwrap_or_else(|| panic!("missing se key {k}"));
        assert_eq!(sv.to_bits(), pv.to_bits(), "cv_scores_se[{k}]");
    }
}

#[test]
fn confirmatory_split_nb_byte_parity() {
    // split_nb path: parallel_for_each_seeded over n_splits is now
    // gated by disable_parallelism; this test pins the equivalence.
    let (x, y) = synth(60, 5, 3.0, 1);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitNb {
            n_splits: 40,
            force: false,
        },
        seed: Some(23),
        disable_parallelism: dp,
        ..Default::default()
    };
    let serial = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 2,
            weights: None,
        },
        opts(true),
    )
    .unwrap();
    let par = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 2,
            weights: None,
        },
        opts(false),
    )
    .unwrap();
    assert_confirmatory_byte_eq(&serial, &par, "confirmatory_split_nb");
}

#[test]
#[allow(clippy::too_many_lines)]
fn rotation_stability_byte_parity() {
    // Serial vs parallel bootstrap stream must be bit-identical after the
    // paired-bootstrap re-derivation (boot_seed drawn post-subsample from
    // the same parent RNG state in both paths).
    let (x, y) = synth(80, 6, 3.0, 1);
    let opts = |dp: bool| RotationStabilityOpts {
        n_boot: 200,
        m_rate: 0.7,
        level: 0.95,
        seed: Some(29),
        disable_parallelism: dp,
        ..Default::default()
    };
    let serial = pls1_rotation_stability(
        x.as_ref(),
        y.as_ref(),
        2,
        RotationStabilityMethod::Varimax(VarimaxArgs::default()),
        None,
        None,
        opts(true),
    )
    .unwrap();
    let par = pls1_rotation_stability(
        x.as_ref(),
        y.as_ref(),
        2,
        RotationStabilityMethod::Varimax(VarimaxArgs::default()),
        None,
        None,
        opts(false),
    )
    .unwrap();

    // Aggregate ratio CI (point, lower, upper, sd).
    assert_eq!(
        serial.variance_ratio.point.to_bits(),
        par.variance_ratio.point.to_bits(),
        "variance_ratio.point"
    );
    assert_eq!(
        serial.variance_ratio.lower.to_bits(),
        par.variance_ratio.lower.to_bits(),
        "variance_ratio.lower"
    );
    assert_eq!(
        serial.variance_ratio.upper.to_bits(),
        par.variance_ratio.upper.to_bits(),
        "variance_ratio.upper"
    );
    assert_eq!(
        serial.variance_ratio.sd.to_bits(),
        par.variance_ratio.sd.to_bits(),
        "variance_ratio.sd"
    );

    // Per-axis ratio CIs.
    assert_eq!(
        serial.variance_ratio_per_axis.len(),
        par.variance_ratio_per_axis.len(),
        "variance_ratio_per_axis length"
    );
    for (k, (s, p)) in serial
        .variance_ratio_per_axis
        .iter()
        .zip(par.variance_ratio_per_axis.iter())
        .enumerate()
    {
        assert_eq!(
            s.point.to_bits(),
            p.point.to_bits(),
            "variance_ratio_per_axis[{k}].point"
        );
        assert_eq!(
            s.lower.to_bits(),
            p.lower.to_bits(),
            "variance_ratio_per_axis[{k}].lower"
        );
        assert_eq!(
            s.upper.to_bits(),
            p.upper.to_bits(),
            "variance_ratio_per_axis[{k}].upper"
        );
        assert_eq!(
            s.sd.to_bits(),
            p.sd.to_bits(),
            "variance_ratio_per_axis[{k}].sd"
        );
    }

    // Aggregate variance components.
    assert_eq!(
        serial.variance_unrot.to_bits(),
        par.variance_unrot.to_bits(),
        "variance_unrot"
    );
    assert_eq!(
        serial.variance_rot.to_bits(),
        par.variance_rot.to_bits(),
        "variance_rot"
    );

    // Per-axis variance components.
    assert_eq!(
        serial.variance_unrot_per_axis.len(),
        par.variance_unrot_per_axis.len(),
        "variance_unrot_per_axis length"
    );
    for (k, (sv, pv)) in serial
        .variance_unrot_per_axis
        .iter()
        .zip(par.variance_unrot_per_axis.iter())
        .enumerate()
    {
        assert_eq!(sv.to_bits(), pv.to_bits(), "variance_unrot_per_axis[{k}]");
    }
    assert_eq!(
        serial.variance_rot_per_axis.len(),
        par.variance_rot_per_axis.len(),
        "variance_rot_per_axis length"
    );
    for (k, (sv, pv)) in serial
        .variance_rot_per_axis
        .iter()
        .zip(par.variance_rot_per_axis.iter())
        .enumerate()
    {
        assert_eq!(sv.to_bits(), pv.to_bits(), "variance_rot_per_axis[{k}]");
    }

    // Scalar metadata.
    assert_eq!(serial.seed, par.seed, "seed");
    assert_eq!(serial.n_boot_finite, par.n_boot_finite, "n_boot_finite");
    assert_eq!(serial.n_eff.to_bits(), par.n_eff.to_bits(), "n_eff");
}

#[test]
fn perm_null_byte_parity() {
    let (x, y) = synth(60, 5, 1.0, 41);
    let opts = |dp: bool| PermNullOpts {
        n_perm: 200,
        return_perm_matrix: false,
        pre_standardized: false,
        disable_parallelism: dp,
        verbose: false,
    };
    let r1 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts(true), Some(2026)).unwrap();
    let r2 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts(false), Some(2026)).unwrap();
    assert_eq!(r1.beta_ref, r2.beta_ref, "beta_ref");
    assert_eq!(r1.beta_perm_mean, r2.beta_perm_mean, "beta_perm_mean");
    assert_eq!(r1.beta_perm_sd, r2.beta_perm_sd, "beta_perm_sd");
    assert_eq!(r1.beta_perm_z, r2.beta_perm_z, "beta_perm_z");
    assert_eq!(r1.seed, r2.seed, "seed");

    // Retained-matrix path.
    let opts_m = |dp: bool| PermNullOpts {
        return_perm_matrix: true,
        ..opts(dp)
    };
    let m1 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_m(true), Some(2027)).unwrap();
    let m2 = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, opts_m(false), Some(2027)).unwrap();
    assert_eq!(m1.beta_perm_matrix, m2.beta_perm_matrix, "beta_perm_matrix");
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn perm_null_weighted_byte_parity() {
    // Weighted permutation loop: the √w-scaled X is built once and shared
    // read-only by every worker.
    let (x, y) = synth(60, 5, 1.0, 43);
    let w = Col::<f64>::from_fn(60, |i| {
        if i % 9 == 8 {
            0.0
        } else {
            0.5 + (i % 5) as f64 * 0.25
        }
    });
    for pre in [false, true] {
        let opts = |dp: bool| PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: pre,
            disable_parallelism: dp,
            verbose: false,
        };
        let r1 = pls1_perm_null(
            x.as_ref(),
            y.as_ref(),
            2,
            Some(w.as_ref()),
            opts(true),
            Some(2028),
        )
        .unwrap();
        let r2 = pls1_perm_null(
            x.as_ref(),
            y.as_ref(),
            2,
            Some(w.as_ref()),
            opts(false),
            Some(2028),
        )
        .unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(&r1.beta_ref), bits(&r2.beta_ref), "beta_ref pre={pre}");
        assert_eq!(
            bits(&r1.beta_perm_mean),
            bits(&r2.beta_perm_mean),
            "beta_perm_mean pre={pre}"
        );
        assert_eq!(
            bits(&r1.beta_perm_sd),
            bits(&r2.beta_perm_sd),
            "beta_perm_sd pre={pre}"
        );
        assert_eq!(
            bits(&r1.beta_perm_z),
            bits(&r2.beta_perm_z),
            "beta_perm_z pre={pre}"
        );
        assert_eq!(
            r1.beta_perm_matrix.as_deref().map(bits),
            r2.beta_perm_matrix.as_deref().map(bits),
            "beta_perm_matrix pre={pre}"
        );
        assert_eq!(r1.seed, r2.seed, "seed pre={pre}");
    }
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
    // skipped (a vacuous-pass defect seen before in this suite).
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

#[test]
#[allow(clippy::cast_precision_loss)]
fn confirmatory_raw_perm_weighted_byte_parity() {
    // Weighted primal raw_perm: folds prepared once, √w-scaled once per
    // fold, replicate columns mapped in parallel inside each fold.
    let (x, y) = synth(40, 5, 3.0, 2);
    let w = Col::<f64>::from_fn(40, |i| {
        if i % 7 == 6 {
            0.0
        } else {
            0.5 + (i % 3) as f64 * 0.5
        }
    });
    let run = |dp: bool| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: Some(w.as_ref()),
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 100,
                    n_folds: 5,
                },
                seed: Some(9),
                disable_parallelism: dp,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_confirmatory_byte_eq(&run(true), &run(false), "confirmatory_raw_perm_weighted");
}

#[test]
fn pls3_split_exact_wide_byte_parity() {
    // p ≫ n: the Gram route. Splits run one at a time, the replicate
    // columns of a split in parallel.
    let (x, y) = wide_two_block_data(60, 3000, 3, 37);
    let run = |dp: bool| {
        pls3_confirmatory_test(
            x.as_ref(),
            y.as_ref(),
            1,
            Pls3ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 50,
                    n_splits: 6,
                },
                seed: Some(19),
                disable_parallelism: dp,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_confirmatory_byte_eq(&run(true), &run(false), "pls3_split_exact_wide");
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().copied().map(f64::to_bits).collect()
}

#[test]
fn perm_null_wide_nspace_byte_parity() {
    // n = 40, d = 2000, n_perm = 100 at k = 1 and 2: dense and unweighted,
    // so both take the n-space Gram route (pinned by
    // `dual_route::multi_k::tests::byte_parity_wide_shapes_take_the_nspace_route`).
    let (x, y) = synth(40, 2000, 1.0, 43);
    let opts = |dp: bool| PermNullOpts {
        n_perm: 100,
        return_perm_matrix: true,
        pre_standardized: false,
        disable_parallelism: dp,
        verbose: false,
    };
    for k in [1_usize, 2] {
        let r1 = pls1_perm_null(x.as_ref(), y.as_ref(), k, None, opts(true), Some(2028)).unwrap();
        let r2 = pls1_perm_null(x.as_ref(), y.as_ref(), k, None, opts(false), Some(2028)).unwrap();
        assert_eq!(
            bits(r1.beta_perm_matrix.as_deref().unwrap()),
            bits(r2.beta_perm_matrix.as_deref().unwrap()),
            "k={k}: beta_perm_matrix"
        );
        assert_eq!(
            bits(&r1.beta_perm_mean),
            bits(&r2.beta_perm_mean),
            "k={k}: mean"
        );
        assert_eq!(bits(&r1.beta_perm_sd), bits(&r2.beta_perm_sd), "k={k}: sd");
        assert_eq!(bits(&r1.beta_perm_z), bits(&r2.beta_perm_z), "k={k}: z");
    }
}

fn raw_perm_parity(n_folds: usize, what: &str) {
    let (x, y) = synth(40, 2000, 1.0, 3);
    let run = |dp: bool| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 50,
                    n_folds,
                },
                seed: Some(21),
                disable_parallelism: dp,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_confirmatory_byte_eq(&run(true), &run(false), what);
}

#[test]
fn confirmatory_raw_perm_wide_k2_byte_parity() {
    // n = 40, d = 2000, n_folds = 5, n_perm = 50, k = 2: dense and
    // unweighted, so raw_perm takes the n-space Gram route (pinned by
    // `dual_route::multi_k::tests::byte_parity_wide_shapes_take_the_nspace_route`).
    raw_perm_parity(5, "confirmatory_raw_perm_wide_k2");
}

#[test]
fn confirmatory_raw_perm_wide_many_folds_k2_byte_parity() {
    // n = 40, d = 2000, n_folds = 20, n_perm = 50, k = 2: n_tr = n - n/n_folds
    // = 38, so n_tr*d = 76 000 >= 65 536, where faer's matrix-vector kernels
    // turn parallel, without n_folds == n. `nspace_eligible_raw_perm` still
    // admits this shape (k = 2 < n_tr), so it also keeps the n-space route
    // under test, the same way `confirmatory_raw_perm_wide_k2_byte_parity`
    // does at n_folds = 5. `n_folds == n` (leave-one-out) is rejected by
    // `pls1_confirmatory_test` and covered directly by
    // `signal_test::tests::raw_perm_rejects_leave_one_out_folds` /
    // `raw_perm_rejects_n_folds_greater_than_n`, not by byte parity: a
    // rejected call has no result to compare bit for bit.
    raw_perm_parity(20, "confirmatory_raw_perm_wide_many_folds_k2");
}

#[test]
fn confirmatory_split_exact_wide_k2_byte_parity() {
    // n = 40, d = 2000, n_splits = 20, n_perm = 50, k = 2: dense and
    // unweighted, so the refit route takes the n-space Gram route (pinned by
    // `dual_route::multi_k::tests::byte_parity_wide_shapes_take_the_nspace_route`).
    let (x, y) = synth(40, 2000, 1.0, 5);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitExact {
            n_perm: 50,
            n_splits: 20,
        },
        seed: Some(23),
        disable_parallelism: dp,
        ..Default::default()
    };
    let run = |dp: bool| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            opts(dp),
        )
        .unwrap()
    };
    assert_confirmatory_byte_eq(&run(true), &run(false), "confirmatory_split_exact_wide_k2");
}

/// Bit-for-bit comparison of two `perm_null` outputs, NaN included.
fn assert_perm_null_bits_eq(a: &plskit::PermNullOutput, b: &plskit::PermNullOutput, name: &str) {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    assert_eq!(bits(&a.beta_ref), bits(&b.beta_ref), "{name}.beta_ref");
    assert_eq!(
        bits(&a.beta_perm_mean),
        bits(&b.beta_perm_mean),
        "{name}.beta_perm_mean"
    );
    assert_eq!(
        bits(&a.beta_perm_sd),
        bits(&b.beta_perm_sd),
        "{name}.beta_perm_sd"
    );
    assert_eq!(
        bits(&a.beta_perm_z),
        bits(&b.beta_perm_z),
        "{name}.beta_perm_z"
    );
    assert_eq!(
        a.beta_perm_matrix.as_deref().map(bits),
        b.beta_perm_matrix.as_deref().map(bits),
        "{name}.beta_perm_matrix"
    );
    assert_eq!(a.seed, b.seed, "{name}.seed");
}

/// n = 2000, d = 50, k = 2, `n_perm = 300` takes the p-space Gram route,
/// dense and weighted (pinned by
/// `gram_p::tests_site_route::byte_parity_shapes_take_the_gram_p_route`).
#[test]
#[allow(clippy::cast_precision_loss)]
fn perm_null_gram_p_byte_parity() {
    let (x, y) = synth(2000, 50, 0.3, 61);
    let w = Col::<f64>::from_fn(2000, |i| {
        if i % 9 == 0 {
            0.0
        } else {
            0.5 + (i % 5) as f64 * 0.3
        }
    });
    for (name, wopt) in [("dense", None), ("weighted", Some(w.as_ref()))] {
        let opts = |dp: bool| PermNullOpts {
            n_perm: 300,
            return_perm_matrix: true,
            pre_standardized: false,
            disable_parallelism: dp,
            verbose: false,
        };
        let serial =
            pls1_perm_null(x.as_ref(), y.as_ref(), 2, wopt, opts(true), Some(3031)).unwrap();
        let par = pls1_perm_null(x.as_ref(), y.as_ref(), 2, wopt, opts(false), Some(3031)).unwrap();
        assert_perm_null_bits_eq(&serial, &par, &format!("perm_null_gram_p_{name}"));
    }
}

/// `raw_perm` at K = 2, n = 2000, d = 50, `n_perm = 300` takes the p-space
/// Gram route (pinned by
/// `gram_p::tests_site_route::byte_parity_shapes_take_the_gram_p_route`).
#[test]
fn confirmatory_raw_perm_gram_p_byte_parity() {
    let (x, y) = synth(2000, 50, 0.3, 62);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::RawPerm {
            n_perm: 300,
            n_folds: 5,
        },
        seed: Some(3032),
        disable_parallelism: dp,
        ..Default::default()
    };
    let run = |dp: bool| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            opts(dp),
        )
        .unwrap()
    };
    assert_confirmatory_byte_eq(&run(true), &run(false), "confirmatory_raw_perm_gram_p");
}

/// `split_exact` refit route with `keep`, k = 1, n = 2000, d = 200,
/// `n_perm = 400` takes the p-space Gram route (pinned by
/// `gram_p::tests_site_route::byte_parity_shapes_take_the_gram_p_route`).
#[test]
fn confirmatory_split_exact_gram_p_keep_byte_parity() {
    let (x, y) = synth(2000, 200, 0.3, 63);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitExact {
            n_perm: 400,
            n_splits: 5,
        },
        seed: Some(3033),
        disable_parallelism: dp,
        keep: Some(10),
        ..Default::default()
    };
    let run = |dp: bool| {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            opts(dp),
        )
        .unwrap()
    };
    assert_confirmatory_byte_eq(
        &run(true),
        &run(false),
        "confirmatory_split_exact_gram_p_keep",
    );
}
