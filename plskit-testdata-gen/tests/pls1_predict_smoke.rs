//! Smoke tests for `pls1_predict` fixture cases.

use plskit_testdata_gen::cases::pls1_predict::basic_n80_d6_k2;
use tempfile::tempdir;

#[test]
fn pls1_predict_basic_emits_files_and_hashes() {
    let dir = tempdir().unwrap();
    let case = basic_n80_d6_k2(dir.path()).unwrap();
    assert_eq!(case.name, "pls1_predict_basic_n80_d6_k2");
    assert_eq!(case.function, "pls1_predict");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
    assert!(case.tolerance.is_some());
}
