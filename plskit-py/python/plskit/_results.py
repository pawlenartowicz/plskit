"""Frozen dataclass result types for plskit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RotationSpec:
    """Spec stamped onto rotation outputs."""
    method: str
    args: Mapping[str, Any]
    R: np.ndarray
    sweeps: int
    V_converged: float
    L_was_provided: bool


@dataclass(frozen=True)
class RotateResult:
    """Return type of `rotate(W: np.ndarray, ...)`."""
    W_rot: np.ndarray
    spec: RotationSpec


@dataclass(frozen=True)
class PLS1Result:
    T: np.ndarray
    P: np.ndarray
    W: np.ndarray
    Q: np.ndarray
    coef: np.ndarray
    beta: np.ndarray
    intercept: float
    k_used: int
    pre_standardized: bool
    weights: np.ndarray | None
    n_eff: float
    rotation_spec: RotationSpec | None = None
    selection_result: FindKOptimalResult | FindKSequenceResult | None = None


@dataclass(frozen=True)
class CIScalar:
    """Subsampling CI for a scalar functional, plus its SE.

    `point`, `lower`, `upper` are on the natural scale of the statistic.
    `sd` is on the *inference scale* used to build the CI:

    * leverage / variance_ratio (centered-scaled): inference scale = natural scale,
      so ``point ± Φ⁻¹(1−α/2) · sd`` reconstructs the CI.
    * holdout_corr (Fisher-transformed NB-Wald): inference scale = atanh(r)
      (z-scale). The r-scale CI is asymmetric and cannot be reconstructed from
      ``point ± sd``; bounds are guaranteed to lie in (−1, 1).
    """
    point: float
    lower: float
    upper: float
    sd: float


@dataclass(frozen=True)
class ConfirmatoryCI:
    """Optional CI bundle returned by `pls1_confirmatory_test(ci=True)`."""
    n_boot: int
    m: int
    m_rate: float
    level: float

    # per-variable
    beta_sign_z: np.ndarray            # shape (D,); folded — canonical for hypothesis tests
    beta_sign_z_signed: np.ndarray     # shape (D,); = sign(β_ref) · |beta_sign_z|; descriptive directional map
    leverage_ci_lower: np.ndarray      # shape (D,)
    leverage_ci_upper: np.ndarray      # shape (D,)
    leverage_se: np.ndarray            # shape (D,)
    beta_ci_lower: np.ndarray          # shape (D,); per-coordinate centered-scaled CI (PLS1 only); diagnostic — see RESULTS_FORMAT.md caveats
    beta_ci_upper: np.ndarray          # shape (D,)
    beta_se: np.ndarray                # shape (D,); = √(m/n) · sd(β_b[j])

    # composite
    holdout_corr: CIScalar             # Fisher z-transformed NB-Wald CI; bounds in (−1, 1), asymmetric on r-scale

    n_boot_finite: int                 # workers that succeeded; ≤ n_boot
    n_boot_finite_holdout_corr: int    # subset whose holdout_corr is finite; ≤ n_boot_finite


@dataclass(frozen=True)
class RotationStabilityResult:
    """Output of `pls1_rotation_stability`.

    All variance and ratio quantities are computed on signed-permutation-
    aligned squared Frobenius residuals (rotated and unrotated weights
    aligned to their respective references via 2^K · K! brute-force
    enumeration).

    Interpretation:
        ratio < 1   → rotation reduced axis variance (rotated axes
                       are more replicable than unrotated ones)
        ratio ≈ 1   → rotation did not change axis replicability
        ratio > 1   → rotation increased axis variance (rotated axes
                       are less replicable; suspect a local-optimum
                       varimax convergence issue)
    """
    method: str
    n_boot: int
    m: int
    m_rate: float
    level: float
    seed: int

    variance_ratio: CIScalar
    variance_ratio_per_axis: list[CIScalar]

    variance_unrot: float
    variance_rot: float
    variance_unrot_per_axis: np.ndarray
    variance_rot_per_axis: np.ndarray

    degenerate_baseline: bool
    n_boot_finite: int
    n_eff: float = float("nan")


@dataclass(frozen=True)
class ConfirmatoryTestResult:
    pvalue: float
    statistic: float
    method: str               # "raw_perm" | "split_nb" | "split_perm" | "score" | "e"
    k: int                    # the K tested
    n_perm: int | None
    n_splits: int | None
    seed: int
    n_eff: float = float("nan")
    ci: ConfirmatoryCI | None = None


@dataclass(frozen=True)
class FindKOptimalResult:
    k_star: int
    selector: str             # "r2_se" | "r2_max" | "bic"
    cv_scores: dict[int, float] | None
    cv_scores_se: dict[int, float] | None
    bic_scores: dict[int, float] | None
    pvalues: np.ndarray | None
    diagnostic: str | None
    seed: int
    n_eff: float = float("nan")


@dataclass(frozen=True)
class FindKSequenceResult:
    k_star: int
    pvalues: np.ndarray
    test_method: str
    alpha: float
    seed: int
    n_eff: float = float("nan")


@dataclass(frozen=True)
class PermNullResult:
    """Output of `pls1_perm_null`.

    Per-voxel signed z statistic suitable for downstream TFCE / cluster-mass /
    max-stat FWER pipelines (PALM, FSL randomise, nltools). The `beta_perm_matrix`
    is opt-in; when present, hand it directly to those tools as the permutation
    map for cluster-based correction.

    Under H0 (`y ⊥ X`), `beta_perm_mean ≈ 0` (calibration diagnostic) and
    `beta_perm_z ~ N(0, 1)` asymptotically. Multiplicity correction is downstream.
    """
    n_perm: int
    k: int
    seed: int
    beta_ref: np.ndarray            # shape (D,); full-data β
    beta_perm_mean: np.ndarray      # shape (D,); ≈ 0 under H0
    beta_perm_sd: np.ndarray        # shape (D,); SD of β under permuted y
    beta_perm_z: np.ndarray         # shape (D,); signed = β_ref / β_perm_sd
    beta_perm_matrix: np.ndarray | None  # shape (n_perm, D) when return_perm_matrix=True
    n_eff: float = float("nan")


@dataclass(frozen=True)
class PreprocessResult:
    """Return type of `plskit.preprocess(...)`. Spec §5.2.

    Each field is populated only if the matching input was passed.
    ``Y_std`` is shape-polymorphic — matches the input shape (1-D or 2-D).
    ``Y_mean`` and ``Y_scale`` are scalar floats for 1-D Y and 1-D
    ``np.ndarray`` for 2-D Y (one entry per Y column).
    """
    X_std: np.ndarray | None
    X_mean: np.ndarray | None
    X_scale: np.ndarray | None
    Y_std: np.ndarray | None
    Y_mean: float | np.ndarray | None
    Y_scale: float | np.ndarray | None
    weights_normalized: np.ndarray | None
    n_eff: float | None
