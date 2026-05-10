//! Smoke tests for `preprocess` fixture cases.

use plskit_testdata_gen::cases::preprocess::n50_d10_with_weights;
use tempfile::tempdir;

#[test]
fn preprocess_with_weights_emits_files_and_hashes() {
    let dir = tempdir().unwrap();
    let case = n50_d10_with_weights(dir.path()).unwrap();
    assert_eq!(case.name, "preprocess_n50_d10_with_weights");
    assert_eq!(case.function, "preprocess");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
    assert!(case.tolerance.is_some());
}
