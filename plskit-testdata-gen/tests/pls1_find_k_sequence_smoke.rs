//! Smoke tests for `find_k_sequence` cases.

use plskit_testdata_gen::cases::pls1_find_k_sequence::{e, raw_perm, split_nb, split_perm};
use tempfile::tempdir;

fn check_basic(
    case_fn: fn(&std::path::Path) -> anyhow::Result<plskit_testdata_gen::manifest::Case>,
    expected_name: &str,
    expected_outputs: &str,
) {
    let dir = tempdir().unwrap();
    let case = case_fn(dir.path()).unwrap();
    assert_eq!(case.name, expected_name);
    assert_eq!(case.function, "pls1_find_k_sequence");
    assert_eq!(case.inputs, "inputs/pls1_find_k_sequence_inputs.npz");
    assert_eq!(case.outputs, expected_outputs);
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
}

#[test]
fn raw_perm_emits_files() {
    check_basic(
        raw_perm,
        "pls1_find_k_sequence_raw_perm",
        "outputs/pls1_find_k_sequence/pls1_find_k_sequence_raw_perm.npz",
    );
}

#[test]
fn split_nb_emits_files() {
    check_basic(
        split_nb,
        "pls1_find_k_sequence_split_nb",
        "outputs/pls1_find_k_sequence/pls1_find_k_sequence_split_nb.npz",
    );
}

#[test]
fn split_perm_emits_files() {
    check_basic(
        split_perm,
        "pls1_find_k_sequence_split_perm",
        "outputs/pls1_find_k_sequence/pls1_find_k_sequence_split_perm.npz",
    );
}

#[test]
fn e_emits_files() {
    check_basic(
        e,
        "pls1_find_k_sequence_e",
        "outputs/pls1_find_k_sequence/pls1_find_k_sequence_e.npz",
    );
}

#[test]
fn shared_input_file_is_byte_identical_across_cases() {
    let dir = tempdir().unwrap();
    let c1 = raw_perm(dir.path()).unwrap();
    let c2 = e(dir.path()).unwrap();
    assert_eq!(c1.hashes.inputs_sha256, c2.hashes.inputs_sha256);
}
