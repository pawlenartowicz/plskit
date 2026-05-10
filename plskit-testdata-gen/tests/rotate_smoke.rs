//! Smoke tests for `rotate` fixture cases.

use plskit_testdata_gen::cases::rotate::varimax_d6_k2;
use tempfile::tempdir;

#[test]
fn rotate_varimax_emits_files_and_hashes() {
    let dir = tempdir().unwrap();
    let case = varimax_d6_k2(dir.path()).unwrap();
    assert_eq!(case.name, "rotate_varimax_d6_k2");
    assert_eq!(case.function, "rotate");
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
    assert!(case.tolerance.is_some());
}
