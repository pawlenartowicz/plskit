//! Regenerator binary for `plskit/testdata/`. Calls `cases::all_cases` and
//! materializes 25 input/output NPZ files plus a v2 `manifest.json`.

use anyhow::{anyhow, Result};
use plskit_testdata_gen::cases::all_cases;
use plskit_testdata_gen::manifest::Manifest;
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

    std::fs::create_dir_all(root.join("inputs"))?;
    std::fs::create_dir_all(root.join("outputs"))?;

    let cases = all_cases(&root)?;

    let manifest = Manifest {
        schema_version: 2,
        producing_version: env!("CARGO_PKG_VERSION").to_string(),
        cases,
    };
    let json = serde_json::to_string_pretty(&manifest)? + "\n";
    std::fs::write(root.join("manifest.json"), json)?;
    eprintln!("wrote {} cases to {}", manifest.cases.len(), root.display());
    Ok(())
}
