//! plskit — PLS regression with modern inference.
//!
//! Single Rust crate; the same engine is consumed by the Python, R, and Julia
//! wrappers (which call into this crate through their respective FFI layers).

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)] // sample-size usize → f64 casts are routine in math kernels

/// Dual (Gram) execution route for resampling loops. Crate-internal — the
/// route is chosen by the engine from `(n_tr, p, B, q)` and is not
/// observable from any public surface.
pub(crate) mod dual_route;
/// Error types for plskit.
pub mod error;
/// K selection (CV / BIC / sequence).
pub mod find_k;
/// PLS1 model fitting (NIPALS loop).
pub mod fit;
/// Low-level linear algebra helpers (standardize, row-subset, etc.).
pub mod linalg;
/// Permutation-null engine for signed per-voxel z statistics.
/// Powers the TFCE / cluster-mass / max-stat downstream pipeline.
pub mod perm_null;
/// PLS3 / PLSSVD fit and transform (SVD on X'Y, no deflation).
pub mod pls3;
/// Confirmatory PLS3 / PLSSVD omnibus test at LV1.
pub mod pls3_signal_test;
/// PLS1 prediction from a fitted model.
pub mod predict;
/// Public preprocess helper: validates/normalizes weights and standardizes (X, y).
pub mod preprocess;
/// Resampling utilities (permutation indices, split-half indices).
pub mod resample;
/// Seeded RNG construction.
pub mod rng;
/// Simple-structure rotation of PLS weights (varimax in v0.1.1).
pub mod rotate;
/// Rotation-stability diagnostic for PLS1.
pub mod rotation_stability;
/// Sequential component tests (step-down inference). Crate-internal helper
/// used by `pls1_find_k_sequence` and the diagnostic path of `pls1_find_k_optimal`.
pub(crate) mod sequential;
/// Confirmatory PLS1 omnibus test at fixed K.
pub mod signal_test;
/// Subsampling engine (Politis–Romano). Public for the engines that drive
/// the `CI` path of `pls1_confirmatory_test` and `pls1_rotation_stability`,
/// but the engine entry points themselves are not on the public surface.
pub(crate) mod subsample;

/// Re-export of faer's owned and borrowed matrix/column types.
///
/// `Pls1Model`, `FindKOptimalOutput`, and `FindKSequenceOutput` expose
/// `Mat<f64>` / `Col<f64>` fields directly — keeping the canonical faer
/// representation avoids round-trip allocations for Rust callers, who
/// typically want to pass these straight back into linear-algebra code.
/// Downstream users can `use plskit::{Mat, Col}` instead of adding faer
/// to their `Cargo.toml`. The R, Python, and Julia wrappers translate
/// these types to their host array representations at the FFI seam — see
/// `_docs/internals/api-contract.md`.
pub use faer::{Col, ColRef, Mat, MatRef};

pub use error::{PlsKitError, PlsKitResult};
pub use find_k::{
    pls1_find_k_optimal, pls1_find_k_sequence, spls1_find_k_optimal, spls1_find_k_sequence,
    spls1_find_keep_optimal, FindKOptimalOpts, FindKOptimalOutput, FindKSequenceOpts,
    FindKSequenceOutput, FindKeepOptimalOpts, FindKeepOptimalOutput, Selector,
};
pub use fit::{pls1_fit, spls1_fit, FitOpts, KSpec, ParChoice, Pls1Model};
pub use perm_null::{pls1_perm_null, PermNullOpts, PermNullOutput};
pub use pls3::{
    pls3_fit, pls3_transform, plssvd_fit, plssvd_transform, Pls3FitOpts, Pls3Model, Pls3Scores,
    TransformWhich,
};
pub use pls3_signal_test::{pls3_confirmatory_test, Pls3ConfirmatoryTestOpts};
pub use predict::pls1_predict;
pub use preprocess::{preprocess, PreprocessInput, PreprocessResult};
pub use rotate::{rotate, RotateOutput, RotationMethod, VarimaxArgs};
pub use rotation_stability::{
    pls1_rotation_stability, RotationStabilityMethod, RotationStabilityOpts,
    RotationStabilityOutput,
};
pub use signal_test::{
    pls1_confirmatory_test, split_nb_gate, CIOpts, ConfirmatoryArgs, ConfirmatoryMethod,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput, SplitNbGateOutput,
};
// subsample is pub(crate) — only these two result types are on the public surface.
// See subsample.rs module header for the full engine contract.
pub use subsample::{CIScalar, ConfirmatoryCI};

/// Returns the `CARGO_PKG_VERSION` string (e.g. `"0.0.1"`).
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
