//! Regenerator binary for `plskit/testdata/`. Calls `cases::all_cases` into a
//! staging directory, then settles the result onto `--testdata-root`: a
//! fixture file is written only when it is new or moved beyond the corpus
//! tolerance (see `settle`), with a case's input going in whenever its output
//! does, so a regeneration on any host leaves unrelated fixtures
//! byte-identical. `manifest.json` is rewritten with the settled
//! files' hashes.

use anyhow::{anyhow, Result};
use plskit_testdata_gen::cases::all_cases;
use plskit_testdata_gen::manifest::Manifest;
use plskit_testdata_gen::npz::sha256_of_file;
use plskit_testdata_gen::settle::{settle_cases, Outcome, Tolerance};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut root: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--testdata-root" => root = args.next().map(PathBuf::from),
            other => return Err(anyhow!("unknown arg: {other}")),
        }
    }
    let root = root.ok_or_else(|| anyhow!("--testdata-root <path> required"))?;

    let staging =
        std::env::temp_dir().join(format!("plskit-testdata-stage-{}", std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(staging.join("inputs"))?;
    std::fs::create_dir_all(staging.join("outputs"))?;
    let result = settle(&staging, &root);
    std::fs::remove_dir_all(&staging)?;
    result
}

fn settle(staging: &std::path::Path, root: &std::path::Path) -> Result<()> {
    let mut cases = all_cases(staging)?;

    let files: Vec<(&str, &str)> = cases
        .iter()
        .map(|c| (c.inputs.as_str(), c.outputs.as_str()))
        .collect();
    let settled = settle_cases(staging, root, &files, Tolerance::DEFAULT)?;
    let changed: Vec<String> = settled
        .iter()
        .filter(|(_, outcome)| *outcome != Outcome::Kept)
        .map(|(rel, outcome)| format!("{outcome:?}: {rel}"))
        .collect();
    for c in &mut cases {
        c.hashes.inputs_sha256 = sha256_of_file(&root.join(&c.inputs))?;
        c.hashes.outputs_sha256 = sha256_of_file(&root.join(&c.outputs))?;
    }

    // A settle that moved no file and kept the case list keeps the committed
    // `producing_version`: the fixtures still come from that version.
    let manifest_path = root.join("manifest.json");
    let committed: Option<Manifest> =
        std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|s| match serde_json::from_str(&s) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!(
                        "warning: committed {} does not parse ({e}); producing_version is reset",
                        manifest_path.display()
                    );
                    None
                }
            });
    let producing_version = match &committed {
        Some(m) if changed.is_empty() && m.cases == cases => m.producing_version.clone(),
        _ => env!("CARGO_PKG_VERSION").to_string(),
    };
    let manifest = Manifest {
        schema_version: 2,
        producing_version,
        cases,
    };
    let json = serde_json::to_string_pretty(&manifest)? + "\n";
    std::fs::write(&manifest_path, json)?;

    for line in &changed {
        eprintln!("{line}");
    }
    eprintln!(
        "settled {} cases into {}: {} file(s) written, {} kept",
        manifest.cases.len(),
        root.display(),
        changed.len(),
        settled.len() - changed.len()
    );
    Ok(())
}
