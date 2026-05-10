"""Python exception type for plskit."""

from plskit import _plskit


class PlsKitError(Exception):
    """Raised by plskit. `code` holds the variant name."""

    def __init__(self, message: str = "", code: str = "") -> None:
        super().__init__(message)
        self.code = code


# The PyO3 cdylib raises _plskit.PlsKitException; @_convert_errors in _api.py
# catches it on every public function and re-raises as PlsKitError. The cdylib
# type is exposed as `_PlsKitException` for advanced introspection only —
# user code should always catch `plskit.PlsKitError`.
_PlsKitException = _plskit.PlsKitException


class PlsKitInvalidWeights(PlsKitError):
    """Raised when the weight vector fails validation. `reason` is one of
    'negative', 'all_zero', 'insufficient_effective_n'."""
    def __init__(self, message: str = "", reason: str = "") -> None:
        super().__init__(message, code="invalid_weights")
        self.reason = reason


class PlsKitResamplingDegenerate(PlsKitError):
    """Raised when too many resamples failed validation. Spec §6.3."""
    def __init__(self, message: str = "",
                 skipped: int = 0, total: int = 0,
                 skip_rate: float = 0.0, threshold: float = 0.0) -> None:
        super().__init__(message, code="resampling_degenerate")
        self.skipped = skipped
        self.total = total
        self.skip_rate = skip_rate
        self.threshold = threshold
