//! Integration tests for [`plskit_testdata_gen::manifest`] JSON schema.

use plskit_testdata_gen::manifest::{Case, Hashes, Manifest};

#[test]
fn manifest_v2_round_trips_through_json() {
    let m = Manifest {
        schema_version: 2,
        producing_version: "0.1.0".into(),
        cases: vec![Case {
            name: "pls1_fit_small_n50_d10_k1".into(),
            function: "pls1_fit".into(),
            inputs: "inputs/pls1_fit_small_n50_d10_k1.npz".into(),
            outputs: "outputs/pls1_fit/pls1_fit_small_n50_d10_k1.npz".into(),
            kwargs: serde_json::json!({"k": 1, "seed": 42}),
            hashes: Hashes {
                inputs_sha256: "0".repeat(64),
                outputs_sha256: "1".repeat(64),
            },
            tolerance: None,
        }],
    };
    let s = serde_json::to_string_pretty(&m).unwrap();
    let m2: Manifest = serde_json::from_str(&s).unwrap();
    assert_eq!(m, m2);
}
