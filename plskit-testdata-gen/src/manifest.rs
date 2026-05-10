//! Manifest v2: index for `plskit/testdata/`. See `TESTDATA.md` for governance.

use serde::{Deserialize, Serialize};

/// Top-level manifest file (`testdata/manifest.json`).
///
/// Lists every fixture case; consumed by cross-language test runners
/// to locate input/output `.npz` files and verify content hashes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest format version (currently `2`).
    pub schema_version: u32,
    /// Version of `plskit` that produced this manifest (e.g. `"0.1.0"`).
    pub producing_version: String,
    /// Ordered list of fixture cases.
    pub cases: Vec<Case>,
}

/// A single fixture case entry in the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    /// Unique name for this case (also used as the `.npz` file stem).
    pub name: String,
    /// Public API function exercised by this case (e.g. `"pls1_fit"`).
    pub function: String,
    /// Relative path to the inputs `.npz` file under `testdata/`.
    pub inputs: String,
    /// Relative path to the outputs `.npz` file under `testdata/`.
    pub outputs: String,
    /// Keyword arguments passed to the function (for documentation/replay).
    pub kwargs: serde_json::Value,
    /// SHA-256 content hashes of the input and output files.
    pub hashes: Hashes,
    /// Optional per-case numerical tolerances (absolute/relative).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<serde_json::Value>,
}

/// SHA-256 content hashes for a case's input and output files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hashes {
    /// Lowercase hex SHA-256 of the inputs `.npz` file.
    pub inputs_sha256: String,
    /// Lowercase hex SHA-256 of the outputs `.npz` file.
    pub outputs_sha256: String,
}
