"""User-facing Python functions; thin wrapper over the PyO3 cdylib."""

from __future__ import annotations

import functools
import warnings
from types import MappingProxyType
from typing import Literal

import numpy as np

from plskit import _plskit
from plskit._errors import PlsKitError, PlsKitInvalidWeights, PlsKitResamplingDegenerate
from plskit._results import (
    CIScalar,
    ConfirmatoryCI,
    ConfirmatoryTestResult,
    FindKOptimalResult,
    FindKeepOptimalResult,
    FindKSequenceResult,
    PermNullResult,
    PLS1Result,
    PreprocessResult,
    RotateResult,
    RotationSpec,
    RotationStabilityResult,
    SplitNbGateResult,
)


def _ciscalar_from_dict(d: dict) -> CIScalar:
    return CIScalar(
        point=d["point"], lower=d["lower"], upper=d["upper"], sd=d["sd"],
    )


def _confirmatory_ci_from_dict(d: dict) -> ConfirmatoryCI:
    return ConfirmatoryCI(
        n_boot=d["n_boot"], m=d["m"],
        m_rate=d["m_rate"], level=d["level"],
        beta_sign_z=np.asarray(d["beta_sign_z"], dtype=np.float64),
        beta_sign_z_signed=np.asarray(d["beta_sign_z_signed"], dtype=np.float64),
        leverage_ci_lower=np.asarray(d["leverage_ci_lower"], dtype=np.float64),
        leverage_ci_upper=np.asarray(d["leverage_ci_upper"], dtype=np.float64),
        leverage_se=np.asarray(d["leverage_se"], dtype=np.float64),
        beta_ci_lower=np.asarray(d["beta_ci_lower"], dtype=np.float64),
        beta_ci_upper=np.asarray(d["beta_ci_upper"], dtype=np.float64),
        beta_se=np.asarray(d["beta_se"], dtype=np.float64),
        holdout_corr=_ciscalar_from_dict(d["holdout_corr"]),
        n_boot_finite=int(d["n_boot_finite"]),
        n_boot_finite_holdout_corr=int(d["n_boot_finite_holdout_corr"]),
    )


def _convert_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _plskit.PlsKitException as e:
            code = getattr(e, "code", "")
            msg = str(e)
            if code == "invalid_weights":
                reason = getattr(e, "reason", "")
                raise PlsKitInvalidWeights(msg, reason=reason) from e
            if code == "resampling_degenerate":
                raise PlsKitResamplingDegenerate(
                    msg,
                    skipped=getattr(e, "skipped", 0),
                    total=getattr(e, "total", 0),
                    skip_rate=getattr(e, "skip_rate", 0.0),
                    threshold=getattr(e, "threshold", 0.0),
                ) from e
            raise PlsKitError(msg, code=code) from e
    return wrapper


def _ensure_array(x: np.ndarray, name: str, ndim: int) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != ndim:
        raise PlsKitError(
            f"{name} must be {ndim}-D, got {a.ndim}-D",
            code="invalid_argument",
        )
    return a


_FIND_K_ALLOWED: dict[str, tuple[str, ...]] = {
    "optimal": ("selector", "diagnostic", "args"),
    "sequence": ("test_method", "alpha", "args"),
}


def _validate_find_k_args(fk_args: dict, allowed: tuple[str, ...]) -> None:
    for key in fk_args:
        if key not in allowed:
            raise PlsKitError(
                f"find_k_args does not accept arg {key!r}; "
                f"allowed: {list(allowed)}",
                code="invalid_args",
            )


# Permutation budget the split_nb → split_exact reroute spends when the result
# object carries no n_perm of its own (neither FindKSequenceResult nor
# FindKOptimalResult has such a field). It must equal the n_perm the engine
# writes into the reroute itself — the literal 1000 in
# plskit-rs/src/sequential.rs (SequentialArgs::SplitExact) and its mirror in
# plskit-rs/src/signal_test.rs (ConfirmatoryArgs::SplitExact); all three change
# together.
_REROUTE_FALLBACK_N_PERM = 1000


def _warn_if_rerouted(
    requested, actual, *, n_perm, stable_rank=None, n_eff=None,
):
    """Tell the caller when the split_nb auto-gate sent the run elsewhere.

    The gate rule lives in Rust and only there — this reports the values the
    engine already returned (which condition fired is read off them), never
    the thresholds and never a recomputed stable rank.

    ``stacklevel=4`` lands on user code: warn → this helper → the public API
    function → its ``_convert_errors`` wrapper → the caller.
    """
    # A caller that requested nothing (`diagnostic=None`) gets nothing back
    # (`result.diagnostic is None`), so the equality test covers that case too.
    if requested == actual:
        return
    saw = []
    if stable_rank is not None:
        saw.append(f"stable rank of the standardized X = {stable_rank:.4g}")
    if n_eff is not None:
        saw.append(f"n_eff = {n_eff:.4g}")
    seen = f" ({'; '.join(saw)})" if saw else ""
    override = f" Pass args={{'force': True}} to run {requested} anyway."
    warnings.warn(
        f"{requested!r} was rerouted to {actual!r}: the {requested} auto-gate "
        f"flagged this design{seen}. The fallback runs n_perm={n_perm} "
        f"permutations, so it costs more than {requested}.{override}",
        UserWarning,
        stacklevel=4,
    )


@_convert_errors
def preprocess(
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> PreprocessResult:
    """Standardize X / Y and normalize weights using plskit's canonical recipe.

    All arguments optional; only the fields matching passed inputs are populated.
    See spec §5.1–5.2 and the preprocessing guide for the cache pattern.
    """
    if X is not None:
        X = _ensure_array(X, "X", 2)
    if Y is not None:
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        if Y.ndim not in (1, 2):
            raise PlsKitError("Y must be 1-D or 2-D", code="invalid_argument")
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.preprocess(x=X, y=Y, weights=weights)
    return PreprocessResult(
        X_std=raw["X_std"],
        X_mean=raw["X_mean"],
        X_scale=raw["X_scale"],
        Y_std=raw["Y_std"],
        Y_mean=raw["Y_mean"],
        Y_scale=raw["Y_scale"],
        weights_normalized=raw["weights_normalized"],
        n_eff=raw["n_eff"],
    )


@_convert_errors
def pls1_fit(
    X: np.ndarray,
    y: np.ndarray,
    k: int | Literal["optimal", "sequence"] = 1,
    *,
    k_max: int | None = None,
    find_k_args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    weights: np.ndarray | None = None,
) -> PLS1Result:
    """Fit a PLS1 model.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix. Standardized internally unless `pre_standardized=True`.
    y : np.ndarray, shape (n,)
        Response vector.
    k : int | 'optimal' | 'sequence', default 1
        Number of PLS components. Pass ``'optimal'`` or ``'sequence'`` to
        select k automatically (requires ``k_max``).
    k_max : int | None
        Maximum k to try when ``k='optimal'`` or ``k='sequence'``.
    find_k_args : dict | None
        Extra kwargs forwarded to `pls1_find_k_optimal` / `pls1_find_k_sequence`.
        Allowed keys are the public params of the target function *except*
        ``seed``, ``pre_standardized``, ``weights``, ``disable_parallelism``,
        and ``verbose`` — pass those on ``pls1_fit`` directly. Unknown keys
        raise ``PlsKitError(code="invalid_args")`` listing the allowed set.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance. See spec §3.5 readings table and the preprocessing guide
        (``plskit.preprocess``) for the cache pattern.
    seed : int | None
        RNG seed forwarded to ``pls1_find_k_optimal`` / ``pls1_find_k_sequence``
        when ``k`` is a string.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See spec §3.5.

    Returns
    -------
    PLS1Result
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    _sel = None
    if isinstance(k, str):
        if k_max is None:
            raise PlsKitError(
                f"k={k!r} requires k_max",
                code="invalid_argument",
            )
        fk_args = dict(find_k_args or {})
        if k == "optimal":
            _validate_find_k_args(fk_args, _FIND_K_ALLOWED["optimal"])
            _sel = pls1_find_k_optimal(
                X, y, k_max,
                pre_standardized=pre_standardized,
                seed=seed,
                weights=weights,
                **fk_args,
            )
            k_int = _sel.k_star
        elif k == "sequence":
            _validate_find_k_args(fk_args, _FIND_K_ALLOWED["sequence"])
            _sel = pls1_find_k_sequence(
                X, y, k_max,
                pre_standardized=pre_standardized,
                seed=seed,
                weights=weights,
                **fk_args,
            )
            if _sel.k_star == 0:
                raise PlsKitError(
                    f"pls1_find_k_sequence rejected no component at alpha "
                    f"{_sel.alpha:.4g} (all pvalues >= alpha). "
                    f"No valid K to fit. Call pls1_find_k_sequence() directly "
                    f"and pass an explicit int k if you want to fit anyway.",
                    code="sequence_no_rejection",
                )
            k_int = _sel.k_star
        else:
            raise PlsKitError(
                f"unknown k mode {k!r}; use int, 'optimal', or 'sequence'",
                code="invalid_argument",
            )
    else:
        k_int = int(k)
    raw = _plskit.pls1_fit(
        X, y, k_int,
        pre_standardized=pre_standardized,
        weights=weights,
    )
    return PLS1Result(**raw, selection_result=_sel)


@_convert_errors
def pls1_predict(model: PLS1Result, X_new: np.ndarray) -> np.ndarray:
    X_new = _ensure_array(X_new, "X_new", 2)
    model_dict = {
        "T": np.ascontiguousarray(model.T, dtype=np.float64),
        "P": np.ascontiguousarray(model.P, dtype=np.float64),
        "W": np.ascontiguousarray(model.W, dtype=np.float64),
        "Q": np.ascontiguousarray(model.Q, dtype=np.float64),
        "coef": np.ascontiguousarray(model.coef, dtype=np.float64),
        "beta": np.ascontiguousarray(model.beta, dtype=np.float64),
        "intercept": float(model.intercept),
        "k_used": int(model.k_used),
        "keep": (int(model.keep) if model.keep is not None else None),
        "pre_standardized": bool(model.pre_standardized),
        "weights": (np.ascontiguousarray(model.weights, dtype=np.float64) if model.weights is not None else None),
        "n_eff": float(model.n_eff),
    }
    return _plskit.pls1_predict(model_dict, X_new)


@_convert_errors
def pls1_confirmatory_test(
    X: np.ndarray, y: np.ndarray, k: int = 1,
    *,
    method: Literal["raw_perm", "split_nb", "split_exact", "score", "e"],
    args: dict | None = None,
    ci: bool = False,
    n_boot: int = 1000,
    m_rate: float = 0.7,
    level: float = 0.95,
    max_failure_rate: float = 0.01,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
    max_skip_rate: float = 0.01,
) -> ConfirmatoryTestResult:
    """Run the confirmatory PLS1 omnibus test at fixed K.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    k : int, default 1
        Number of components to test.
    method : str
        Test method: ``'raw_perm'``, ``'split_nb'``, ``'split_exact'``,
        ``'score'``, or ``'e'``.

        ``'split_exact'`` is the recommended default: a split-half test
        (statistic ``tanh(z̄)``, the mean Fisher-z of held-out correlations)
        calibrated by permutation, so it holds its level on any design.
        ``'split_nb'`` uses the same statistic with an asymptotic correction
        instead of permutations — much cheaper, and appropriate when n ≫ p
        with a flat X spectrum. Designs outside that regime are auto-gated:
        a flagged ``'split_nb'`` request runs ``'split_exact'`` instead
        (``result.method`` says so, and Python warns). Pass
        ``args={'force': True}`` to run ``'split_nb'`` anyway.

        ``rho_hat`` is populated for ``'split_nb'`` only; ``None`` for every
        other method, including ``'split_exact'``, and ``None`` for
        ``'split_nb'`` itself when the input is weighted or the test half
        is too small. ``stable_rank`` is populated whenever ``'split_nb'``
        was requested — it is what the auto-gate saw.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_perm': 500}`` for ``raw_perm``,
        ``{'force': True}`` for ``split_nb``).
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See spec §3.5.
    max_skip_rate : float, default 0.01
        Subsample-loop skip threshold for the ``ci`` branch (spec §6.3).
        The CI loop fails with ``PlsKitResamplingDegenerate`` if
        ``skipped / total > max_skip_rate``.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed.
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_confirmatory_test_raw(
        X, y, k,
        method=method, args=args,
        ci=ci, n_boot=n_boot, m_rate=m_rate, level=level,
        max_failure_rate=max_failure_rate,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
        max_skip_rate=max_skip_rate,
    )
    ci_dict = raw.pop("ci", None)
    ci_obj = _confirmatory_ci_from_dict(ci_dict) if ci_dict is not None else None
    result = ConfirmatoryTestResult(ci=ci_obj, **raw)
    _warn_if_rerouted(
        method, result.method,
        n_perm=result.n_perm,
        stable_rank=result.stable_rank,
        n_eff=result.n_eff,
    )
    return result


@_convert_errors
def split_nb_gate(
    X: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> SplitNbGateResult:
    """Ask whether the ``split_nb`` auto-gate flags a design, without testing.

    Reports the decision ``pls1_confirmatory_test`` and the ``find_k``
    functions make internally, so you can see it before paying for a run.
    ``fires=True`` means a ``'split_nb'`` request on this X reroutes to
    ``'split_exact'`` unless you pass ``args={'force': True}``.

    Standardizes its own copy of X (weighted moments when ``weights`` is
    given), as the embedded gates do. Only X and the weights enter the rule —
    y does not.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.

    Returns
    -------
    SplitNbGateResult
    """
    X = _ensure_array(X, "X", 2)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    return SplitNbGateResult(**_plskit.split_nb_gate(X, weights=weights))


@_convert_errors
def pls1_find_k_optimal(
    X: np.ndarray, y: np.ndarray, k_max: int,
    *,
    selector: Literal["r2_se", "r2_max", "bic"] = "r2_se",
    diagnostic: Literal["raw_perm", "split_nb", "split_exact", "e"] | None = None,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> FindKOptimalResult:
    """Select the optimal number of PLS1 components K*.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    k_max : int
        Maximum number of components to consider.
    selector : str, default 'r2_se'
        Selection criterion: ``'r2_se'`` (1-SE rule), ``'r2_max'``, or ``'bic'``.
    diagnostic : str | None, default None
        Optional same-sample sequential diagnostic to attach to K*. One of
        ``'raw_perm'`` / ``'split_nb'`` / ``'split_exact'`` / ``'e'``, or
        ``None`` (no diagnostic). Selection and test share data, so the
        resulting ``pvalues`` are a robustness check, not honest inference.
        A ``'split_nb'`` diagnostic the auto-gate flags runs ``'split_exact'``
        instead; ``result.diagnostic`` says so and Python warns. Pass
        ``args={'force': True}`` to run ``'split_nb'`` anyway.
    args : dict | None
        Method-specific kwargs. Selector keys: ``n_folds``. Diagnostic
        keys: ``n_perm`` (for ``raw_perm``/``split_exact``), ``n_splits``
        (for ``split_nb``/``split_exact``), ``force`` (for ``split_nb``).
        Diagnostic keys require ``diagnostic`` to be set.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed.
    disable_parallelism : bool, default False
        Force serial execution.
    verbose : bool, default False
        Print progress to stderr.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See spec §3.5.

    Returns
    -------
    FindKOptimalResult
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_find_k_optimal(
        X, y, k_max,
        selector=selector,
        diagnostic=diagnostic,
        args=args,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
    )
    result = FindKOptimalResult(**raw)
    # The diagnostic runs through the same hoisted sequence gate, so it can be
    # rerouted the same way.
    _warn_if_rerouted(
        diagnostic, result.diagnostic,
        n_perm=_REROUTE_FALLBACK_N_PERM,
        stable_rank=result.stable_rank,
        n_eff=result.n_eff,
    )
    return result


@_convert_errors
def pls1_find_k_sequence(
    X: np.ndarray, y: np.ndarray, k_max: int,
    *,
    test_method: Literal["raw_perm", "split_nb", "split_exact", "e"] = "split_nb",
    alpha: float = 0.05,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> FindKSequenceResult:
    """Select K* via a sequential closed test on the PLS1 component chain.

    Closed testing on nested H is exact, so the per-step pvalues form an
    honest FWER-controlled sequence. To recover the path-max p-value
    along the rejected chain, compute ``np.nanmax(r.pvalues[:r.k_star])``.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    k_max : int
        Maximum number of components to test.
    test_method : str, default 'split_nb'
        Per-step test method: ``'raw_perm'``, ``'split_nb'``,
        ``'split_exact'``, or ``'e'``. ``'split_exact'`` is the recommended
        default (permutation-calibrated, holds its level on any design);
        ``'split_nb'`` is the cheaper asymptotic alternative for n ≫ p with a
        flat X spectrum. The auto-gate is evaluated once for the whole
        sequence: a flagged ``'split_nb'`` request runs ``'split_exact'``
        for every step and ``result.test_method`` says so.
    alpha : float, default 0.05
        Significance threshold for rejection.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_splits': 50}``, or
        ``{'force': True}`` to run ``split_nb`` past the auto-gate).
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed.
    disable_parallelism : bool, default False
        Force serial execution.
    verbose : bool, default False
        Print progress to stderr.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See spec §3.5.

    Returns
    -------
    FindKSequenceResult
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_find_k_sequence(
        X, y, k_max,
        test_method=test_method,
        alpha=alpha,
        args=args,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
    )
    result = FindKSequenceResult(**raw)
    _warn_if_rerouted(
        test_method, result.test_method,
        n_perm=_REROUTE_FALLBACK_N_PERM,
        stable_rank=result.stable_rank,
        n_eff=result.n_eff,
    )
    return result


@_convert_errors
def spls1_fit(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    keep: int,
    *,
    pre_standardized: bool = False,
    weights: np.ndarray | None = None,
) -> PLS1Result:
    """Fit a sparse PLS1 model (hard keep-count selection on the weight vector).

    Each latent direction loads on exactly ``keep`` X variables (Chun & Keleş
    2010 lineage, keep-count formulation). ``keep = n_features`` reproduces
    ``pls1_fit`` bit-exactly. Tune ``keep`` with ``spls1_find_keep_optimal``;
    tune ``k`` at fixed ``keep`` with ``spls1_find_k_optimal`` /
    ``spls1_find_k_sequence`` — one axis is always fixed (no joint search).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix. Standardized internally unless `pre_standardized=True`.
    y : np.ndarray, shape (n,)
        Response vector.
    k : int
        Number of PLS components (no ``'optimal'``/``'sequence'`` string modes —
        use the ``spls1_find_*`` entry points directly).
    keep : int
        Keep-count in ``[1, n_features]``; scalar, broadcast to all components.
    pre_standardized : bool, default False
        If True, skip standardization.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights; ``None`` means uniform.

    Returns
    -------
    PLS1Result
        With ``keep`` set; exactly ``keep`` nonzeros per ``W`` column.
        Per-coordinate β CIs are NOT offered under selection (post-selection
        inference) — see the spls1 spec.
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_fit(
        X, y, int(k), int(keep),
        pre_standardized=pre_standardized,
        weights=weights,
    )
    return PLS1Result(**raw)


@_convert_errors
def spls1_find_keep_optimal(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    args: dict | None = None,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> FindKeepOptimalResult:
    """Tune the spls1 keep-count at fixed ``k`` (CV R² with the 1-SE rule).

    Sweeps a logged geometric keep grid over ``[1, n_features]`` (powers of
    two, endpoints always included) and returns the SPARSEST keep whose mean
    CV R² is within 1 SE of the best. The swept grid is reported on
    ``result.keep_grid``. Sparsity is tuned inside the training split, never
    on test data.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    k : int
        Fixed component count for every fit in the sweep.
    args : dict | None
        ``{'n_folds': int}`` (default 5).
    seed, disable_parallelism, verbose, weights
        As on ``pls1_find_k_optimal``.

    Returns
    -------
    FindKeepOptimalResult
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_find_keep_optimal(
        X, y, int(k),
        args=args,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
    )
    return FindKeepOptimalResult(**raw)


@_convert_errors
def spls1_find_k_optimal(
    X: np.ndarray, y: np.ndarray, k_max: int, keep: int,
    *,
    selector: Literal["r2_se", "r2_max", "bic"] = "r2_se",
    diagnostic: Literal["raw_perm", "split_nb", "split_exact", "e"] | None = None,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> FindKOptimalResult:
    """`pls1_find_k_optimal` with the inner fitter swapped to the sparse fit
    at the caller's fixed ``keep``. Same selectors, diagnostic, and result
    shape. ``keep = n_features`` reproduces the dense function exactly.

    Note: ``selector='bic'`` reuses the dense complexity penalty (not a
    keep-aware sparse BIC) — it under-penalizes added components and biases
    the selected k upward under sparsity. Deliberate v1 simplification.
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_find_k_optimal(
        X, y, k_max, int(keep),
        selector=selector,
        diagnostic=diagnostic,
        args=args,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
    )
    result = FindKOptimalResult(**raw)
    # Same reroute as pls1_find_k_optimal.
    _warn_if_rerouted(
        diagnostic, result.diagnostic,
        n_perm=_REROUTE_FALLBACK_N_PERM,
        stable_rank=result.stable_rank,
        n_eff=result.n_eff,
    )
    return result


@_convert_errors
def spls1_find_k_sequence(
    X: np.ndarray, y: np.ndarray, k_max: int, keep: int,
    *,
    test_method: Literal["raw_perm", "split_nb", "split_exact", "e"] = "split_nb",
    alpha: float = 0.05,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> FindKSequenceResult:
    """`pls1_find_k_sequence` with the inner fitter swapped to the sparse fit
    at the caller's fixed ``keep``: each step deflates on the sparse residual
    AND tests the sparse marginal component (coherent sequential test).
    ``keep = n_features`` reproduces the dense function exactly.
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_find_k_sequence(
        X, y, k_max, int(keep),
        test_method=test_method,
        alpha=alpha,
        args=args,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
    )
    result = FindKSequenceResult(**raw)
    _warn_if_rerouted(
        test_method, result.test_method,
        n_perm=_REROUTE_FALLBACK_N_PERM,
        stable_rank=result.stable_rank,
        n_eff=result.n_eff,
    )
    return result


@_convert_errors
def rotate(
    model_or_W,
    *,
    method: Literal["varimax"] = "varimax",
    L: np.ndarray | None = None,
    args: dict | None = None,
):
    if args is None:
        resolved_args: dict = {}
    else:
        resolved_args = dict(args)
    if method == "varimax":
        resolved_args.setdefault("max_iter", 50)
        resolved_args.setdefault("tol", 1e-8)
        resolved_args.setdefault("kaiser_normalize", True)

    if isinstance(model_or_W, PLS1Result):
        if model_or_W.rotation_spec is not None:
            raise PlsKitError(
                "model already has a rotation_spec; v0.1.1 does not support re-rotation",
                code="already_rotated",
            )
        return _rotate_model(model_or_W, method, L, resolved_args)
    if isinstance(model_or_W, np.ndarray):
        return _rotate_array(model_or_W, method, L, resolved_args)
    raise TypeError(
        "rotate() first arg must be a PLS1Result or numpy ndarray, "
        f"got {type(model_or_W).__name__}"
    )


def _rotate_array(W, method, L, resolved_args) -> RotateResult:
    W = _ensure_array(W, "W", 2)
    L_was_provided = L is not None
    L_arr = _ensure_array(L, "L", 2) if L_was_provided else None
    raw = _plskit.rotate(W, method=method, args=resolved_args, l=L_arr)
    spec = RotationSpec(
        method=method,
        args=MappingProxyType(dict(resolved_args)),
        R=raw["r"],
        sweeps=raw["sweeps"],
        V_converged=raw["v_converged"],
        L_was_provided=L_was_provided,
    )
    return RotateResult(W_rot=raw["w_rot"], spec=spec)


def _rotate_model(model: PLS1Result, method, L, resolved_args) -> PLS1Result:
    rot = _rotate_array(model.W, method, L, resolved_args)
    R = rot.spec.R
    return PLS1Result(
        T=model.T @ R,
        P=model.P @ R,
        W=rot.W_rot,
        Q=R.T @ model.Q,
        coef=model.coef,
        beta=model.beta,
        intercept=model.intercept,
        k_used=model.k_used,
        pre_standardized=model.pre_standardized,
        weights=model.weights,
        n_eff=model.n_eff,
        rotation_spec=rot.spec,
    )


@_convert_errors
def pls1_rotation_stability(
    X: np.ndarray, y: np.ndarray, k: int,
    *,
    rotation_method: Literal["varimax"] = "varimax",
    rotation_args: dict | None = None,
    L: np.ndarray | None = None,
    n_boot: int = 1000,
    m_rate: float = 0.7,
    level: float = 0.95,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
    max_skip_rate: float = 0.01,
) -> RotationStabilityResult:
    """PLS1 rotation-stability diagnostic.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    k : int
        Number of components (2 ≤ k ≤ 7).
    rotation_method : str
        Rotation method; currently only ``"varimax"`` is implemented.
    rotation_args : dict or None
        Method-specific kwargs (e.g. ``{"max_iter": 100}`` for varimax).
    L : np.ndarray or None
        Optional fixed loading matrix for constrained rotation.
    n_boot : int
        Number of subsampling resamples.
    m_rate : float
        Subsample-size exponent; ``m = ceil(n ** m_rate)``.
    level : float
        Nominal CI level (e.g. 0.95).
    pre_standardized : bool
        Set ``True`` when ``X`` is already column-standardized.
    seed : int or None
        RNG seed for reproducibility.
    disable_parallelism : bool
        Disable Rayon parallelism (useful for tests).
    verbose : bool
        Reserved for future progress reporting.
    weights : np.ndarray or None, shape (n,)
        Optional per-observation weights. ``None`` is equivalent to all-ones.
    max_skip_rate : float
        Maximum fraction of subsamples that may be skipped (due to weight
        degeneracy) before raising ``PlsKitResamplingDegenerate``.
        Default ``0.01``.
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    L_arr = _ensure_array(L, "L", 2) if L is not None else None
    w_arr = _ensure_array(weights, "weights", 1) if weights is not None else None
    raw = _plskit.pls1_rotation_stability_raw(
        X, y, k,
        rotation_method=rotation_method,
        rotation_args=rotation_args,
        l=L_arr,
        n_boot=n_boot, m_rate=m_rate, level=level,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=w_arr,
        max_skip_rate=max_skip_rate,
    )
    return RotationStabilityResult(
        method=raw["method"],
        n_boot=raw["n_boot"], m=raw["m"],
        m_rate=raw["m_rate"], level=raw["level"],
        seed=raw["seed"],
        variance_ratio=_ciscalar_from_dict(raw["variance_ratio"]),
        variance_ratio_per_axis=[
            _ciscalar_from_dict(d) for d in raw["variance_ratio_per_axis"]
        ],
        variance_unrot=raw["variance_unrot"],
        variance_rot=raw["variance_rot"],
        variance_unrot_per_axis=raw["variance_unrot_per_axis"],
        variance_rot_per_axis=raw["variance_rot_per_axis"],
        degenerate_baseline=raw["degenerate_baseline"],
        n_boot_finite=raw["n_boot_finite"],
        n_eff=raw["n_eff"],
    )


@_convert_errors
def pls1_perm_null(
    X: np.ndarray, y: np.ndarray, k: int,
    *,
    n_perm: int = 1000,
    return_perm_matrix: bool = False,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> PermNullResult:
    """Permutation-null engine for PLS1 β. Signed per-voxel z + optional perm matrix.

    Pair with `pls1_confirmatory_test(method="split_nb")` as an omnibus gate
    before spending the `n_perm` permutation budget at fMRI scale.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix. Standardized internally unless `pre_standardized=True`.
    y : np.ndarray, shape (n,)
        Response vector.
    k : int
        Number of PLS components.
    n_perm : int, default 1000
        Number of permutations (must be ≥ 100).
    return_perm_matrix : bool, default False
        If True, return the full `(n_perm, d)` β matrix. Memory-intensive at
        fMRI scale; use only when needed for cluster-based correction.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed for reproducibility.
    disable_parallelism : bool, default False
        Force serial execution (useful for deterministic tests).
    verbose : bool, default False
        Reserved for future progress reporting.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are NOT permuted — `w[i]` stays tied to row `i` regardless
        of which `y` value lands there under the permutation.
    """
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_perm_null_raw(
        X, y, k,
        n_perm=n_perm,
        return_perm_matrix=return_perm_matrix,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
    )
    matrix = raw["beta_perm_matrix"]
    return PermNullResult(
        n_perm=raw["n_perm"],
        k=raw["k"],
        seed=raw["seed"],
        beta_ref=np.asarray(raw["beta_ref"], dtype=np.float64),
        beta_perm_mean=np.asarray(raw["beta_perm_mean"], dtype=np.float64),
        beta_perm_sd=np.asarray(raw["beta_perm_sd"], dtype=np.float64),
        beta_perm_z=np.asarray(raw["beta_perm_z"], dtype=np.float64),
        beta_perm_matrix=(
            np.asarray(matrix, dtype=np.float64) if matrix is not None else None
        ),
        n_eff=float(raw["n_eff"]),
    )


__all__ = [
    "preprocess",
    "pls1_fit",
    "pls1_predict",
    "pls1_confirmatory_test",
    "pls1_find_k_optimal",
    "pls1_find_k_sequence",
    "spls1_fit",
    "spls1_find_keep_optimal",
    "spls1_find_k_optimal",
    "spls1_find_k_sequence",
    "pls1_perm_null",
    "pls1_rotation_stability",
    "rotate",
]
