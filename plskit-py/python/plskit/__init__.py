"""plskit — PLS regression with modern inference."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("plskit")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from plskit._api import (
    pls1_confirmatory_test,
    pls1_find_k_optimal,
    pls1_find_k_sequence,
    pls1_fit,
    pls1_perm_null,
    pls1_predict,
    pls1_rotation_stability,
    preprocess,
    rotate,
)
from plskit._errors import PlsKitError, PlsKitInvalidWeights, PlsKitResamplingDegenerate
from plskit._results import (
    CIScalar,
    ConfirmatoryCI,
    ConfirmatoryTestResult,
    FindKOptimalResult,
    FindKSequenceResult,
    PermNullResult,
    PLS1Result,
    PreprocessResult,
    RotateResult,
    RotationSpec,
    RotationStabilityResult,
)

__all__ = [
    "__version__",
    "preprocess",
    "pls1_fit", "pls1_predict",
    "pls1_confirmatory_test",
    "pls1_find_k_optimal", "pls1_find_k_sequence",
    "pls1_perm_null",
    "pls1_rotation_stability",
    "rotate",
    "PlsKitError",
    "PlsKitInvalidWeights",
    "PlsKitResamplingDegenerate",
    "PreprocessResult",
    "PLS1Result",
    "ConfirmatoryTestResult",
    "FindKOptimalResult", "FindKSequenceResult",
    "PermNullResult",
    "RotateResult", "RotationSpec",
    "CIScalar", "ConfirmatoryCI", "RotationStabilityResult",
]
