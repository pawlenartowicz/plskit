"""Python exception and warning types for plskit."""


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
    """Raised when too many resamples failed validation. See `docs/python/api.md` (Errors)."""
    def __init__(self, message: str = "",
                 skipped: int = 0, total: int = 0,
                 skip_rate: float = 0.0, threshold: float = 0.0) -> None:
        super().__init__(message, code="resampling_degenerate")
        self.skipped = skipped
        self.total = total
        self.skip_rate = skip_rate
        self.threshold = threshold


class PlsKitRerouted(UserWarning):
    """Warned when the `split_nb` auto-gate reroutes an explicit `split_nb`
    request (a test, a `find_k_optimal` diagnostic or a sequence test).
    `requested` is the method asked for and `actual` the one that ran;
    `n_perm` (the fallback's permutations), `stable_rank` and `n_eff` (what
    the gate read) are `None` when not computed."""

    def __init__(self, message: str = "", requested=None, actual=None,
                 n_perm=None, stable_rank=None, n_eff=None) -> None:
        super().__init__(message)
        self.requested = requested
        self.actual = actual
        self.n_perm = n_perm
        self.stable_rank = stable_rank
        self.n_eff = n_eff


class PlsKitRotatedSparse(UserWarning):
    """Warned when `rotate` gets a `PLS3Result` from a sparse fit: the
    rotation spreads its zeros."""
