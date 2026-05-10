//! Smoke tests for `find_k_optimal` cases.

use ndarray_npy::NpzReader;
use plskit_testdata_gen::cases::pls1_find_k_optimal::{bic, r2_max, r2_se, r2_se_diagnostic};
use tempfile::tempdir;

#[test]
fn r2_se_emits_files_and_hashes() {
    let dir = tempdir().unwrap();
    let case = r2_se(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_find_k_optimal_r2_se");
    assert_eq!(case.function, "pls1_find_k_optimal");
    assert_eq!(case.inputs, "inputs/pls1_find_k_optimal_inputs.npz");
    assert_eq!(
        case.outputs,
        "outputs/pls1_find_k_optimal/pls1_find_k_optimal_r2_se.npz"
    );
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
}

#[test]
fn r2_max_emits_files() {
    let dir = tempdir().unwrap();
    let case = r2_max(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_find_k_optimal_r2_max");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
}

#[test]
fn bic_emits_files() {
    let dir = tempdir().unwrap();
    let case = bic(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_find_k_optimal_bic");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
}

#[test]
fn r2_se_diagnostic_emits_files_and_includes_diagnostic_fields() {
    let dir = tempdir().unwrap();
    let case = r2_se_diagnostic(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_find_k_optimal_r2_se_diagnostic");
    assert!(dir.path().join(&case.outputs).exists());

    // Spot-check that diagnostic fields are present in the output file by reading it.
    let f = std::fs::File::open(dir.path().join(&case.outputs)).unwrap();
    let mut npz = NpzReader::new(f).unwrap();
    let names: Vec<String> = npz.names().unwrap();
    assert!(names.contains(&"pvalues".to_string()));
    assert!(names.contains(&"diagnostic".to_string()));
}

#[test]
fn shared_input_file_is_byte_identical_across_cases() {
    let dir = tempdir().unwrap();
    let c1 = r2_se(dir.path()).unwrap();
    let c2 = bic(dir.path()).unwrap(); // overwrites the same input file
    assert_eq!(c1.inputs, c2.inputs);
    // Both should produce the same inputs_sha256 since the inputs are deterministic.
    assert_eq!(c1.hashes.inputs_sha256, c2.hashes.inputs_sha256);
}
