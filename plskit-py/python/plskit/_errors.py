"""Python exception type for plskit."""


class PlsKitError(Exception):
    """Raised by plskit. `code` holds the variant name."""

    def __init__(self, message: str = "", code: str = "") -> None:
        super().__init__(message)
        self.code = code


class PlsKitInvalidWeights(PlsKitError):
    """Raised when the weight vector fails validation. `reason` is one of
    'negative', 'all_zero', 'insufficient_effective_n', 'length_mismatch'."""
    def __init__(self, message: str = "", reason: str = "") -> None:
        super().__init__(message, code="invalid_weights")
        self.reason = reason


class PlsKitResamplingDegenerate(PlsKitError):
    """Raised when too many resamples failed validation. See `_docs/python/api.md` (Errors)."""
    def __init__(self, message: str = "",
                 skipped: int = 0, total: int = 0,
                 skip_rate: float = 0.0, threshold: float = 0.0) -> None:
        super().__init__(message, code="resampling_degenerate")
        self.skipped = skipped
        self.total = total
        self.skip_rate = skip_rate
        self.threshold = threshold
