//! The generator reproduces the committed corpus index. `cases::all_cases`
//! runs every case into a scratch directory, and its manifest entries must
//! equal `testdata/manifest.json` case for case (name, function, inputs,
//! outputs, kwargs, tolerance): a case added, removed or edited without
//! regenerating the corpus fails here. Content hashes are not compared with
//! the committed ones (fixture bytes may differ across hosts within
//! tolerance, `testdata/README.md` "Regenerating: settle mode");
//! `scripts/check_corpus_hash.py` owns those and `plskit-rs/tests/corpus.rs`
//! the values.

use plskit_testdata_gen::cases::all_cases;
use plskit_testdata_gen::npz::sha256_of_file;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use tempfile::tempdir;

fn without_hashes(mut case: Value) -> Value {
    case.as_object_mut()
        .expect("case is an object")
        .remove("hashes");
    case
}

#[test]
fn all_cases_matches_the_committed_manifest() {
    let dir = tempdir().unwrap();
    let cases = all_cases(dir.path()).unwrap();

    let mut generated = BTreeMap::new();
    for c in &cases {
        // A case's recorded hashes must be those of its files as the whole
        // run left them: a later case that rewrites a shared inputs file
        // ("last writer wins") with other bytes leaves this one stale.
        for (rel, recorded) in [
            (&c.inputs, &c.hashes.inputs_sha256),
            (&c.outputs, &c.hashes.outputs_sha256),
        ] {
            let path = dir.path().join(rel);
            assert!(path.exists(), "{}: {rel} was not written", c.name);
            assert_eq!(
                &sha256_of_file(&path).unwrap(),
                recorded,
                "{}: {rel} changed after its case hashed it",
                c.name
            );
        }
        let entry = without_hashes(serde_json::to_value(c).unwrap());
        assert!(
            generated.insert(c.name.clone(), entry).is_none(),
            "duplicate case name {}",
            c.name
        );
    }

    let manifest =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/manifest.json");
    let committed: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    let committed: BTreeMap<String, Value> = committed["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| {
            let name = c["name"].as_str().expect("name").to_owned();
            (name, without_hashes(c.clone()))
        })
        .collect();

    let names = |m: &BTreeMap<String, Value>| m.keys().cloned().collect::<BTreeSet<_>>();
    let (gen, com) = (names(&generated), names(&committed));
    assert_eq!(
        gen,
        com,
        "the generator and testdata/manifest.json list different cases (manifest only: {:?}; \
         generator only: {:?}); regenerate: cargo run -p plskit-testdata-gen -- --testdata-root testdata",
        com.difference(&gen).collect::<Vec<_>>(),
        gen.difference(&com).collect::<Vec<_>>()
    );
    for (name, entry) in &generated {
        assert_eq!(
            entry, &committed[name],
            "{name}: the manifest entry differs from the generator's; regenerate"
        );
    }
}
