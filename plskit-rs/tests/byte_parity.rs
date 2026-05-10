//! Single-thread vs multi-thread byte-parity.
//! On the same binary, same platform, fixed seed: parallel and serial
//! execution must produce byte-identical output. The pre-computed child
//! seeds in `resample::parallel_for_each_seeded` make this structural,
//! not aspirational. This test guards the structural property.

use faer::{Col, Mat};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, ConfirmatoryArgs,
    ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts, FindKOptimalOpts,
    FindKSequenceOpts, Selector,
};

fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| x[(i, 0)] * snr + noise[i]);
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
    // Spec §9 test 1 (weights_none_parity): the assertions below are the
    // byte-for-byte parity contract between pls1_fit(weights=None) and the
    // pre-spec unweighted fit. Do not relax these tolerances.
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
fn confirmatory_split_perm_byte_parity() {
    let (x, y) = synth(40, 5, 3.0, 1);
    let opts = |dp: bool| ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitPerm {
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
    assert_confirmatory_byte_eq(&serial, &par, "confirmatory_split_perm");
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
        args: ConfirmatoryArgs::SplitNb { n_splits: 40 },
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
