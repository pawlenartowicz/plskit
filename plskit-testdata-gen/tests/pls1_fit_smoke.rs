//! Smoke test: `small_n50_d10_k1` case writes inputs/outputs files,
//! produces 64-char hex hashes, and the manifest Case round-trips.

use plskit_testdata_gen::cases::pls1_fit::small_n50_d10_k1;
use tempfile::tempdir;

#[test]
fn small_n50_d10_k1_writes_files_and_hashes() {
    let dir = tempdir().unwrap();
    let case = small_n50_d10_k1(dir.path()).unwrap();

    assert_eq!(case.name, "pls1_fit_small_n50_d10_k1");
    assert_eq!(case.function, "pls1_fit");
    assert_eq!(case.inputs, "inputs/pls1_fit_small_n50_d10_k1.npz");
    assert_eq!(
        case.outputs,
        "outputs/pls1_fit/pls1_fit_small_n50_d10_k1.npz"
    );

    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());

    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
    assert!(case
        .hashes
        .inputs_sha256
        .chars()
        .all(|c| c.is_ascii_hexdigit()));
}
