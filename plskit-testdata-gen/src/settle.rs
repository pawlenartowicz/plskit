//! Tolerance-aware settling of a freshly generated corpus onto the committed
//! one, so a regeneration on any host rewrites only the fixtures it moved.
//!
//! The generator writes every case into a staging root. A staged file then
//! replaces its committed counterpart only when the two are not equivalent
//! under the corpus contract (`testdata/README.md` "Tolerance"): same entry
//! names, dtypes and shapes; `f64` entries within `atol_scalar` (0-D) or
//! `atol_array` (otherwise) plus `rtol` times the committed value, NaN equal
//! to NaN, an infinity equal only to itself; integer and string entries
//! exactly equal. Cross-host rounding drift stays inside that contract, so it
//! never reaches the committed bytes. A case's input settles with its output
//! (`settle_cases`): when the output moved, the staged input goes in too.

use anyhow::{anyhow, Result};
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::{NpzReader, ReadableElement};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::path::Path;

/// Tolerances for one case, as recorded in its manifest entry: an `f64`
/// entry `a` matches the committed `e` when `|a − e| ≤ atol + rtol·|e|`.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Absolute tolerance for 0-D `f64` entries.
    pub atol_scalar: f64,
    /// Absolute tolerance for `f64` entries of rank 1 or more.
    pub atol_array: f64,
    /// Relative tolerance for every `f64` entry.
    pub rtol: f64,
}

impl Tolerance {
    /// The corpus defaults: scalars `1e-12`, arrays `1e-10`, `rtol = 1e-14`.
    pub const DEFAULT: Self = Self {
        atol_scalar: 1e-12,
        atol_array: 1e-10,
        rtol: 1e-14,
    };

    /// Read `atol_scalar` / `atol_array` / `rtol` from a manifest `tolerance` value,
    /// falling back to [`Tolerance::DEFAULT`] for a missing key.
    #[must_use]
    pub fn from_manifest(v: Option<&serde_json::Value>) -> Self {
        let get = |key: &str, default: f64| {
            v.and_then(|t| t.get(key))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(default)
        };
        Self {
            atol_scalar: get("atol_scalar", Self::DEFAULT.atol_scalar),
            atol_array: get("atol_array", Self::DEFAULT.atol_array),
            rtol: get("rtol", Self::DEFAULT.rtol),
        }
    }
}

/// Whether the `.npz` files at `committed` and `staged` hold the same fixture
/// under `tol`. Byte-identical files short-circuit to `true`.
///
/// # Errors
/// Returns an error if either file cannot be read or is not a valid `.npz`.
pub fn equivalent(committed: &Path, staged: &Path, tol: Tolerance) -> Result<bool> {
    let a = std::fs::read(committed)?;
    let b = std::fs::read(staged)?;
    if a == b {
        return Ok(true);
    }
    let mut ra = NpzReader::new(Cursor::new(a))?;
    let mut rb = NpzReader::new(Cursor::new(b))?;
    let mut names = ra.names()?;
    let mut names_b = rb.names()?;
    names.sort();
    names_b.sort();
    if names != names_b {
        return Ok(false);
    }
    for name in &names {
        if let (Some(x), Some(y)) = (read::<f64>(&mut ra, name), read::<f64>(&mut rb, name)) {
            let atol = if x.ndim() == 0 {
                tol.atol_scalar
            } else {
                tol.atol_array
            };
            if x.shape() != y.shape()
                || !x.iter().zip(&y).all(|(&c, &s)| close(s, c, atol, tol.rtol))
            {
                return Ok(false);
            }
        } else if let (Some(x), Some(y)) = (read::<i64>(&mut ra, name), read::<i64>(&mut rb, name))
        {
            if x != y {
                return Ok(false);
            }
        } else if let (Some(x), Some(y)) = (read::<u8>(&mut ra, name), read::<u8>(&mut rb, name)) {
            if x != y {
                return Ok(false);
            }
        } else {
            // Different dtypes on the two sides, or a dtype the corpus never writes.
            return Ok(false);
        }
    }
    Ok(true)
}

fn read<T: ReadableElement>(r: &mut NpzReader<Cursor<Vec<u8>>>, name: &str) -> Option<ArrayD<T>> {
    r.by_name::<OwnedRepr<T>, IxDyn>(name).ok()
}

/// Staged `a` against committed `e`: `|a − e| ≤ atol + rtol·|e|` for a
/// finite `e`, NaN equal to NaN, an infinity equal only to itself (without
/// the finiteness guard `rtol·|inf|` would accept anything). The comparison
/// `plskit-rs/tests/corpus.rs` and the wrappers apply.
#[allow(clippy::float_cmp)]
fn close(a: f64, e: f64, atol: f64, rtol: f64) -> bool {
    a == e
        || (a.is_nan() && e.is_nan())
        || (e.is_finite() && (a - e).abs() <= atol + rtol * e.abs())
}

/// What settling did to one fixture file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Not in the committed corpus; the staged file was copied in.
    Added,
    /// Moved beyond tolerance; the staged file replaced the committed one.
    Replaced,
    /// Equivalent under tolerance; the committed bytes were kept.
    Kept,
}

/// Settle the staged file `rel` (a path relative to both roots) onto
/// `committed_root`, copying it only when it is new or not [`equivalent`].
///
/// # Errors
/// Returns an error on I/O failure or an unreadable `.npz`.
pub fn settle_file(
    staged_root: &Path,
    committed_root: &Path,
    rel: &str,
    tol: Tolerance,
) -> Result<Outcome> {
    let committed = committed_root.join(rel);
    if committed.exists() && equivalent(&committed, &staged_root.join(rel), tol)? {
        return Ok(Outcome::Kept);
    }
    install(staged_root, committed_root, rel)
}

/// Copy the staged file `rel` over its committed counterpart whatever the
/// tolerance says; byte-identical files are kept.
fn install(staged_root: &Path, committed_root: &Path, rel: &str) -> Result<Outcome> {
    let staged = staged_root.join(rel);
    let committed = committed_root.join(rel);
    let outcome = if !committed.exists() {
        Outcome::Added
    } else if std::fs::read(&committed)? == std::fs::read(&staged)? {
        return Ok(Outcome::Kept);
    } else {
        Outcome::Replaced
    };
    let parent = committed
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", committed.display()))?;
    std::fs::create_dir_all(parent)?;
    std::fs::copy(&staged, &committed)?;
    Ok(outcome)
}

/// Settle every case's `(inputs, outputs, tolerance)` files onto
/// `committed_root`. Outputs settle first, each under its case's tolerance.
/// A case's input then settles with its output: when the output moved, the
/// staged input goes in with it even if that input alone is within
/// tolerance, so no committed output is paired with an input it was not
/// computed from. Kept outputs are within tolerance of f(staged input), so an
/// input replaced for one case stays valid for the others that read it. Any
/// other input settles on its own, under the tightest tolerance of the cases
/// that read it. Returns every file with its outcome.
///
/// # Errors
/// Returns an error when an output belongs to more than one case or a path
/// is both an input and an output, and on I/O failure or an unreadable `.npz`.
pub fn settle_cases(
    staged_root: &Path,
    committed_root: &Path,
    cases: &[(&str, &str, Tolerance)],
) -> Result<Vec<(String, Outcome)>> {
    let mut inputs: BTreeMap<&str, Tolerance> = BTreeMap::new();
    let mut outputs: BTreeMap<&str, Tolerance> = BTreeMap::new();
    for &(input, output, tol) in cases {
        if outputs.insert(output, tol).is_some() {
            return Err(anyhow!("{output} is the output of more than one case"));
        }
        inputs
            .entry(input)
            .and_modify(|t| {
                t.atol_scalar = t.atol_scalar.min(tol.atol_scalar);
                t.atol_array = t.atol_array.min(tol.atol_array);
                t.rtol = t.rtol.min(tol.rtol);
            })
            .or_insert(tol);
    }
    if let Some(rel) = inputs.keys().find(|rel| outputs.contains_key(*rel)) {
        return Err(anyhow!("{rel} is both an input and an output"));
    }

    let mut settled = Vec::with_capacity(outputs.len() + inputs.len());
    let mut moved_inputs: BTreeSet<&str> = BTreeSet::new();
    for &(input, output, tol) in cases {
        let outcome = settle_file(staged_root, committed_root, output, tol)?;
        if outcome != Outcome::Kept {
            moved_inputs.insert(input);
        }
        settled.push((output.to_owned(), outcome));
    }
    for (&rel, &tol) in &inputs {
        let outcome = if moved_inputs.contains(rel) {
            install(staged_root, committed_root, rel)?
        } else {
            settle_file(staged_root, committed_root, rel, tol)?
        };
        settled.push((rel.to_owned(), outcome));
    }
    Ok(settled)
}
