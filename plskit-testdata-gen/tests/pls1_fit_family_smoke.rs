//! Smoke tests for the four additional `pls1_fit` cases.

use plskit_testdata_gen::cases::pls1_fit::{
    small_n50_d10_k3, small_n50_d10_sequence, wide_n30_d100_k1, wide_n30_d100_k3,
};
use tempfile::tempdir;

#[test]
fn small_n50_d10_k3_emits_files_and_hashes() {
    let dir = tempdir().unwrap();
    let case = small_n50_d10_k3(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_fit_small_n50_d10_k3");
    assert_eq!(case.function, "pls1_fit");
    assert_eq!(case.inputs, "inputs/pls1_fit_small_n50_d10_k3.npz");
    assert_eq!(
        case.outputs,
        "outputs/pls1_fit/pls1_fit_small_n50_d10_k3.npz"
    );
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
}

#[test]
fn wide_n30_d100_k1_emits_files() {
    let dir = tempdir().unwrap();
    let case = wide_n30_d100_k1(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_fit_wide_n30_d100_k1");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
}

#[test]
fn wide_n30_d100_k3_emits_files() {
    let dir = tempdir().unwrap();
    let case = wide_n30_d100_k3(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_fit_wide_n30_d100_k3");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
}

#[test]
fn small_n50_d10_sequence_emits_files_and_hashes() {
    // The sequence case must detect a signal (k_star > 0) at the chosen seed;
    // the function hard-errors if k_star == 0.
    let dir = tempdir().unwrap();
    let case = small_n50_d10_sequence(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_fit_small_n50_d10_sequence");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
}
