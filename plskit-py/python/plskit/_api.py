"""User-facing Python functions; thin wrapper over the PyO3 cdylib."""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
import warnings
from collections.abc import Mapping
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
    PLS3Result,
    PLS3Scores,
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


# numpy dtype kinds read as real numbers: bool, signed and unsigned int, float.
_REAL_KINDS = "biuf"


def _aligned_f64(x, name: str, what: str = "array") -> np.ndarray:
    """`x` as an aligned float64 array, in its own memory layout.

    A C- or F-contiguous float64 array passes through uncopied, and so does
    any other strided view (the extension copies that one itself); a copy
    is made only to convert the dtype or to align a misaligned array (a
    byte-offset view of a buffer, which Rust may not read in place), and it
    keeps the input's order. Results for different layouts of the same
    values agree to rounding, not bit for bit.

    Anything numpy cannot read as real numbers (strings, numeric ones
    included, ragged nested lists, complex values, objects that are not
    numbers) raises `PlsKitError(code="invalid_argument")`, worded as
    plskit-bind words it. In an object array, `None` and pandas' `pd.NA`
    both become NaN, so the engine reports `non_finite_input`.
    """
    try:
        a = np.asarray(x)
        if a.dtype.kind == "O":
            a = _object_to_f64(a, name, what)
    except (ValueError, TypeError) as exc:
        raise PlsKitError(
            f"{name} must be a numeric {what}, got a value numpy cannot read as "
            f"one ({exc})",
            code="invalid_argument",
        ) from None
    if a.dtype.kind not in _REAL_KINDS:
        got = _describe(x) if isinstance(x, str) else f"an array of dtype {a.dtype}"
        raise PlsKitError(
            f"{name} must be a numeric {what}, got {got}", code="invalid_argument"
        )
    a = np.asarray(a, dtype=np.float64)
    if not a.flags.aligned:
        a = a.copy(order="K")
    return a


def _object_to_f64(a: np.ndarray, name: str, what: str) -> np.ndarray:
    """An object array as float64. A string element is refused (numpy would
    parse `"1.5"` here, but not in a list of strings, and plskit-bind takes
    no strings); `pd.NA` becomes NaN, as `None` does."""
    for v in a.flat:
        if isinstance(v, (str, bytes)):
            raise PlsKitError(
                f"{name} must be a numeric {what}, got an object array holding "
                f"{_describe(v)}",
                code="invalid_argument",
            )
    na = getattr(sys.modules.get("pandas"), "NA", None)
    if na is not None:
        a = a.copy(order="K")
        for i, v in enumerate(a.flat):
            if v is na:
                a.flat[i] = np.nan
    return a.astype(np.float64)


def _ensure_array(x: np.ndarray, name: str, ndim: int) -> np.ndarray:
    a = _aligned_f64(x, name, {1: "vector", 2: "matrix"}[ndim])
    if a.ndim != ndim:
        raise PlsKitError(
            f"{name} must be {ndim}-D, got {a.ndim}-D",
            code="invalid_argument",
        )
    return a


def _describe(value) -> str:
    """How an error message quotes a rejected value (plskit-bind's
    `describe`): a string as `the string "x"`, so `"7"` and `7` read
    differently, anything else by its repr."""
    if isinstance(value, str):
        return f"the string {json.dumps(value, ensure_ascii=False)}"
    return repr(value)


def _type_phrase(value) -> str:
    """How a wrong-type error names what it got: `None`, or the type name
    with its article ("a PLS3Result", "an int", and "an ndarray", which is
    read letter by letter)."""
    if value is None:
        return "None"
    t = type(value).__name__
    an = t[:1].lower() in "aeiou" or t == "ndarray"
    return f"{'an' if an else 'a'} {t}"


def _require_result(value, cls: type, name: str = "model") -> None:
    """Refuse a `model` argument that is not a `cls` result (a
    `PLS3Result` passed to `pls1_predict`, say) with
    `PlsKitError(code="invalid_argument")`, worded as plskit-bind words it,
    instead of the `AttributeError` reading its fields would raise."""
    if not isinstance(value, cls):
        raise PlsKitError(
            f"{name} must be a {cls.__name__}, got {_type_phrase(value)}",
            code="invalid_argument",
        )


_INT_MAX = 2**63 - 1  # bind receives an int as i64
_FLOAT_MAX = 2**53  # F64_EXACT_MAX: largest float exactly representable as an int


def _whole(value, name: str) -> int:
    """`value` as an int. A `bool`, a negative value, a float with a
    fractional part (this also excludes NaN/inf, whose `is_integer()` is
    False), or a magnitude outside what plskit-bind accepts is rejected;
    this mirrors the rule plskit-bind applies for R and Julia. `int()`
    would truncate 2.7 and accept `True` and `-1`. A 0-d NumPy array
    (`np.array(3)`) is unwrapped with `.item()` first, then checked by the
    same rules."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool):
        pass
    elif isinstance(value, (int, np.integer)) and 0 <= value <= _INT_MAX:
        return int(value)
    elif (
        isinstance(value, (float, np.floating))
        and float(value).is_integer()
        and 0 <= value <= _FLOAT_MAX
    ):
        return int(value)
    raise PlsKitError(
        f"{name} must be a non-negative whole number, got {_describe(value)}",
        code="invalid_argument",
    )


def _whole_or_none(value, name: str) -> int | None:
    """`_whole` for an optional count (`None` passes through)."""
    return None if value is None else _whole(value, name)


_SEED_MAX = 2**64 - 1  # the engine takes a u64 seed


def _seed(value) -> int | None:
    """`seed` as an int in ``[0, 2^64)``, or None. The same rules as
    `_whole` otherwise (no bool; a float must be whole and at most 2^53),
    mirroring plskit-bind's seed rule for R and Julia."""
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool):
        pass
    elif isinstance(value, (int, np.integer)) and 0 <= value <= _SEED_MAX:
        return int(value)
    elif (
        isinstance(value, (float, np.floating))
        and float(value).is_integer()
        and 0 <= value <= _FLOAT_MAX
    ):
        return int(value)
    raise PlsKitError(
        f"seed must be a whole number in [0, 2^64), got {_describe(value)}",
        code="invalid_argument",
    )


def _number_or_none(value, name: str) -> float | None:
    """An optional float argument: an int or a float (not a bool), or None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not isinstance(value, bool) and isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        try:
            return float(value)
        except OverflowError:  # an int beyond the float range, e.g. 10**400
            pass
    raise PlsKitError(
        f"{name} must be a number, got {_describe(value)}",
        code="invalid_argument",
    )


def _flag(value, name: str, default: bool = False) -> bool:
    """A bool argument (Python or NumPy bool; nothing converts). `None`
    means the default, as a null does in plskit-bind."""
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise PlsKitError(
        f"{name} must be a bool, got {_describe(value)}",
        code="invalid_argument",
    )


def _string(value, name: str, default: str | None = None) -> str:
    """A string argument such as a method name. `None` means `default`;
    with no default the argument is required."""
    if value is None and default is not None:
        return default
    if isinstance(value, str):
        return value
    raise PlsKitError(
        f"{name} must be a string, got {_describe(value)}",
        code="invalid_argument",
    )


def _args_dict(value, name: str) -> dict | None:
    """A method-args mapping (`args`, `rotation_args`, `find_k_args`) as a
    plain dict, or None. Its keys and values are checked where the method
    is known (the extension, or `_validate_find_k_args`)."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise PlsKitError(
        f"{name} must be a dict of named values, got {_describe(value)}",
        code="invalid_argument",
    )


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
# FindKOptimalResult has such a field). Read straight off the engine constant
# (SPLIT_NB_REROUTE_N_PERM in plskit-rs/src/signal_test.rs, which every Rust
# reroute site reads) instead of keeping a copy here that could drift.
_REROUTE_FALLBACK_N_PERM = _plskit.SPLIT_NB_REROUTE_N_PERM


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
    See _docs/concepts/preprocessing.md for the recipe and the cache pattern.
    """
    if X is not None:
        X = _ensure_array(X, "X", 2)
    if Y is not None:
        Y = _aligned_f64(Y, "Y")
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
        select k automatically (requires ``k_max``). A float must be whole
        (``2.0``, not ``2.7``).
    k_max : int | None
        Maximum k to try when ``k='optimal'`` or ``k='sequence'``. Must be
        ``None`` when ``k`` is an int.
    find_k_args : dict | None
        Extra kwargs forwarded to `pls1_find_k_optimal` / `pls1_find_k_sequence`.
        Allowed keys are the public params of the target function *except*
        ``seed``, ``pre_standardized``, ``weights``, ``disable_parallelism``,
        and ``verbose`` — pass those on ``pls1_fit`` directly. Unknown keys
        raise ``PlsKitError(code="invalid_args")`` listing the allowed set.
        Must be ``None`` when ``k`` is an int.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance. See _docs/concepts/preprocessing.md (the
        ``pre_standardized`` decision table) and ``plskit.preprocess`` for the
        cache pattern.
    seed : int | None
        RNG seed forwarded to ``pls1_find_k_optimal`` / ``pls1_find_k_sequence``
        when ``k`` is a string.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        _docs/concepts/PLS1/weights.md.

    Returns
    -------
    PLS1Result
    """
    seed = _seed(seed)
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    find_k_args = _args_dict(find_k_args, "find_k_args")
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
        # `for_fit=True` hands the "no K to fit" decision to the engine
        # (`k_to_fit` on the selection result), which raises
        # `optimal_no_component` / `sequence_no_rejection` itself.
        if k == "optimal":
            _validate_find_k_args(fk_args, _FIND_K_ALLOWED["optimal"])
            _sel = _pls1_find_k_optimal(
                X, y, k_max,
                pre_standardized=pre_standardized,
                seed=seed,
                weights=weights,
                for_fit=True,
                **fk_args,
            )
            k_int = _sel.k_star
        elif k == "sequence":
            _validate_find_k_args(fk_args, _FIND_K_ALLOWED["sequence"])
            _sel = _pls1_find_k_sequence(
                X, y, k_max,
                pre_standardized=pre_standardized,
                seed=seed,
                weights=weights,
                for_fit=True,
                **fk_args,
            )
            k_int = _sel.k_star
        else:
            raise PlsKitError(
                f"unknown k mode {k!r}; use int, 'optimal', or 'sequence'",
                code="invalid_argument",
            )
    else:
        if k_max is not None or find_k_args is not None:
            raise PlsKitError(
                "k_max and find_k_args apply only when k is 'optimal' or 'sequence'",
                code="invalid_argument",
            )
        k_int = _whole(1 if k is None else k, "k")
    raw = _plskit.pls1_fit(
        X, y, k_int,
        pre_standardized=pre_standardized,
        weights=weights,
    )
    return PLS1Result(**raw, selection_result=_sel)


@_convert_errors
def pls1_predict(model: PLS1Result, X_new: np.ndarray) -> np.ndarray:
    _require_result(model, PLS1Result)
    X_new = _ensure_array(X_new, "X_new", 2)
    model_dict = {
        "T": _aligned_f64(model.T, "model.T"),
        "P": _aligned_f64(model.P, "model.P"),
        "W": _aligned_f64(model.W, "model.W"),
        "Q": _aligned_f64(model.Q, "model.Q"),
        "coef": _aligned_f64(model.coef, "model.coef"),
        "beta": _aligned_f64(model.beta, "model.beta"),
        "intercept": float(model.intercept),
        "k_used": int(model.k_used),
        "keep": (int(model.keep) if model.keep is not None else None),
        "pre_standardized": bool(model.pre_standardized),
        "weights": (_aligned_f64(model.weights, "model.weights") if model.weights is not None else None),
        "n_eff": float(model.n_eff),
    }
    return _plskit.pls1_predict(model_dict, X_new)


def _ensure_2d_Y(Y: np.ndarray) -> np.ndarray:
    """PLS3 takes a Y matrix. A 1-D Y is a PLS1 problem, so say that.

    Thin wrapper over `_ensure_array` — the shape check and the error code
    are already there; the only thing added is the pls1_fit hint, worth a
    sentence because a 1-D Y is the mistake this family invites. Scoped to
    genuinely 1-D input: a 3-D array isn't a PLS1 problem either, and the
    hint is nonsense there.
    """
    try:
        return _ensure_array(Y, "Y", 2)
    except PlsKitError as exc:
        if np.ndim(Y) == 1:
            raise PlsKitError(
                f"{exc}. A single-column outcome is a PLS1 problem — use pls1_fit.",
                code="invalid_argument",
            ) from exc
        raise


@_convert_errors
def pls3_fit(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    *,
    pre_standardized_X: bool = False,
    pre_standardized_Y: bool = False,
    weights: np.ndarray | None = None,
) -> PLS3Result:
    """Fit PLS3 / PLSSVD — SVD of the standardized cross-covariance ``X'Y``.

    Symmetric analysis: neither block is the outcome. The question is which
    pattern of X covaries with which pattern of Y. In psychology and
    neuroimaging this method is called PLSC.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        First block. Standardized internally unless ``pre_standardized_X=True``.
    Y : np.ndarray, shape (n, q)
        Second block. Must be 2-D.
    k : int, default 1
        Number of latent variables to keep; ``k <= min(p, q)``. All k come
        out of one SVD — there is no deflation, so the components are
        orthogonal by construction.
    pre_standardized_X, pre_standardized_Y : bool, default False
        Skip centering/scaling of that block. The returned moments are then
        the identity, so ``pls3_transform`` will not re-apply any transform.
    weights : np.ndarray | None, default None
        Not implemented for this family; anything other than ``None`` raises
        ``PlsKitError(code="invalid_argument")``.

    Returns
    -------
    PLS3Result
    """
    pre_standardized_X = _flag(pre_standardized_X, "pre_standardized_X")
    pre_standardized_Y = _flag(pre_standardized_Y, "pre_standardized_Y")
    X = _ensure_array(X, "X", 2)
    Y = _ensure_2d_Y(Y)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls3_fit(
        X, Y, _whole(1 if k is None else k, "k"),
        pre_standardized_X=pre_standardized_X,
        pre_standardized_Y=pre_standardized_Y,
        weights=weights,
    )
    return PLS3Result(**raw)


@_convert_errors
def plssvd_fit(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    *,
    pre_standardized_X: bool = False,
    pre_standardized_Y: bool = False,
    weights: np.ndarray | None = None,
) -> PLS3Result:
    """Alias for :func:`pls3_fit` under the SVD-PLS name. Same function."""
    return pls3_fit(
        X, Y, k,
        pre_standardized_X=pre_standardized_X,
        pre_standardized_Y=pre_standardized_Y,
        weights=weights,
    )


@_convert_errors
def spls3_fit(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    keep_X: int,
    keep_Y: int,
    *,
    pre_standardized_X: bool = False,
    pre_standardized_Y: bool = False,
    max_iter: int | None = None,
    tol: float | None = None,
    weights: np.ndarray | None = None,
) -> PLS3Result:
    """Sparse PLS3 / PLSSVD: keep-count selection on both salience sides.

    Each component loads on at most ``keep_X`` of the X variables and
    ``keep_Y`` of the Y variables, selected by magnitude with exact ties
    broken toward the lower index. ``keep_Y < q`` is the point of the
    method: it forces each latent dimension onto a few outcomes, so the
    outcomes separate into groups instead of every dimension loading a
    little on everything.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        First block. Standardized internally unless ``pre_standardized_X=True``.
    Y : np.ndarray, shape (n, q)
        Second block. Must be 2-D.
    k : int
        Number of latent variables. Components come out of a deflating
        alternation, not one SVD, so they are **not** orthogonal.
    keep_X : int
        Non-zeros per column of ``U``; ``1 <= keep_X <= p``.
    keep_Y : int
        Non-zeros per column of ``V``; ``1 <= keep_Y <= q``.
    pre_standardized_X, pre_standardized_Y : bool, default False
        Skip centering/scaling of that block.
    max_iter : int | None, default None
        Cap on alternation sweeps per component. ``None`` uses the engine
        default, 100. Hitting it sets ``result.converged[a] = False`` for
        that component; it is not an error.
    tol : float | None, default None
        Convergence tolerance, read as the max absolute change in ``v``
        across one sweep. ``None`` uses the engine default, 1e-8.
    weights : np.ndarray | None, default None
        Not implemented for this family; anything other than ``None``
        raises ``PlsKitError(code="invalid_argument")``.

    Returns
    -------
    PLS3Result
        With ``keep_X``, ``keep_Y``, ``converged`` and ``n_iter``
        populated. ``keep_X = p`` and ``keep_Y = q`` together reproduce
        ``pls3_fit`` bit for bit.

    Notes
    -----
    ``singular_values`` on a sparse fit are ``u'Av`` on the deflated ``A``,
    not singular values of ``X'Y``. Do not read an explained-variance share
    off them. Rotating a sparse fit with ``rotate`` destroys the zeros:
    choose sparsity or rotation, not both. That warning is reachable only
    through ``rotate``'s array overload applied to ``U``; ``rotate``
    refuses a ``PLS3Result`` outright (it takes a ``PLS1Result`` or an
    ``np.ndarray``), so there is no model overload to guard.
    """
    pre_standardized_X = _flag(pre_standardized_X, "pre_standardized_X")
    pre_standardized_Y = _flag(pre_standardized_Y, "pre_standardized_Y")
    X = _ensure_array(X, "X", 2)
    Y = _ensure_2d_Y(Y)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls3_fit(
        X, Y, _whole(k, "k"), _whole(keep_X, "keep_X"), _whole(keep_Y, "keep_Y"),
        pre_standardized_X=pre_standardized_X,
        pre_standardized_Y=pre_standardized_Y,
        max_iter=_whole_or_none(max_iter, "max_iter"),
        tol=_number_or_none(tol, "tol"),
        weights=weights,
    )
    return PLS3Result(**raw)


@_convert_errors
def pls3_transform(
    model: PLS3Result,
    X_new: np.ndarray | None = None,
    Y_new: np.ndarray | None = None,
    *,
    which: Literal["x_scores", "y_scores", "both"] = "both",
) -> PLS3Scores:
    """Project new data onto a fitted PLS3's latent-variable scores.

    New data is standardized with the *fit's* moments. PLS3 has no
    regression ``predict`` — it is symmetric, so there is nothing to predict.

    Parameters
    ----------
    model : PLS3Result
    X_new : np.ndarray | None, shape (n_new, p)
    Y_new : np.ndarray | None, shape (n_new, q)
    which : {'x_scores', 'y_scores', 'both'}, default 'both'
        Which block(s) to project. A block ``which`` asks for must be
        supplied, or ``PlsKitError(code="invalid_argument")`` is raised.

    Returns
    -------
    PLS3Scores
    """
    _require_result(model, PLS3Result)
    model_dict = {
        "U": _aligned_f64(model.U, "model.U"),
        "V": _aligned_f64(model.V, "model.V"),
        "singular_values": _aligned_f64(model.singular_values, "model.singular_values"),
        "x_scores": _aligned_f64(model.x_scores, "model.x_scores"),
        "y_scores": _aligned_f64(model.y_scores, "model.y_scores"),
        "X_mean": _aligned_f64(model.X_mean, "model.X_mean"),
        "X_scale": _aligned_f64(model.X_scale, "model.X_scale"),
        "Y_mean": _aligned_f64(model.Y_mean, "model.Y_mean"),
        "Y_scale": _aligned_f64(model.Y_scale, "model.Y_scale"),
        "k_used": int(model.k_used),
        "pre_standardized_X": bool(model.pre_standardized_X),
        "pre_standardized_Y": bool(model.pre_standardized_Y),
    }
    if X_new is not None:
        X_new = _ensure_array(X_new, "X_new", 2)
    if Y_new is not None:
        Y_new = _ensure_array(Y_new, "Y_new", 2)
    which = _string(which, "which", "both")
    raw = _plskit.pls3_transform(model_dict, X_new, Y_new, which=which)
    return PLS3Scores(x_scores=raw["x_scores"], y_scores=raw["y_scores"])


@_convert_errors
def plssvd_transform(
    model: PLS3Result,
    X_new: np.ndarray | None = None,
    Y_new: np.ndarray | None = None,
    *,
    which: Literal["x_scores", "y_scores", "both"] = "both",
) -> PLS3Scores:
    """Alias for :func:`pls3_transform` under the SVD-PLS name."""
    return pls3_transform(model, X_new, Y_new, which=which)


@_convert_errors
def pls3_confirmatory_test(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    *,
    test_method: Literal["split_exact", "split_nb"],
    args: dict | None = None,
    pre_standardized_X: bool = False,
    pre_standardized_Y: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
) -> ConfirmatoryTestResult:
    """Confirmatory PLS3 omnibus test at LV1.

    Statistic: the held-out latent-variable correlation
    ``r = cor(X_te @ u1, Y_te @ v1)``, where ``(u1, v1)`` come from a PLS3
    fit on the training half only. Fisher-z averaged across ``n_splits``
    splits and reported as ``tanh(z_bar)``. Both methods report that same
    statistic and differ only in what they compare it against.

    Sign indeterminacy costs nothing here: an SVD fixes ``(u1, v1)`` only up
    to a simultaneous flip, and a flip negates both held-out score vectors
    at once, leaving ``r`` unchanged.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    Y : np.ndarray, shape (n, q)
    k : int, default 1
        Must be 1. Above LV1 the training-half component ordering need not
        survive to the test half, and whether the statistic should then be
        per-component or subspace-level is not settled.
    test_method : {'split_exact', 'split_nb'}
        ``'split_exact'`` is recommended (``test_method`` has no default): the p-value comes from
        a permutation reference built by shuffling the rows of Y against X,
        with the splits held fixed across all permutations, so it is exact
        whenever the rows are exchangeable under the null. Neither method is
        valid on clustered rows (e.g. repeated scans per subject): random
        splits leak subjects across halves and row permutation breaks
        within-subject exchangeability. ``'split_nb'`` compares the same statistic
        against an asymptotic t reference instead, costing ``n_splits``
        fits in total rather than ``n_perm * n_splits``.

        Both sides of the correlation are estimated on the training half,
        where PLS1 has an observed outcome on one side. The per-split null
        still carries over: conditional on the training half the two
        held-out score vectors are fixed linear combinations of independent
        test-half rows, so under the null ``r`` on one split follows the
        ordinary null correlation law. The between-split correction the
        p-value uses is PLS1's Nadeau-Bengio heuristic, which is not derived
        for two blocks; its transfer is supported empirically only. Measured
        on Gaussian, heavy-tailed, low-stable-rank and real two-block
        designs, ``'split_nb'`` came out conservative, never
        anti-conservative.

        ``'split_nb'`` requests are auto-gated on X exactly as in
        ``pls1_confirmatory_test``: a flagged design runs ``'split_exact'``
        instead (``result.test_method`` says so, and Python warns). Pass
        ``args={'force': True}`` to run ``'split_nb'`` anyway. Y never
        enters the gate — q is small by construction in PLSC, so a
        stable-rank floor on Y would flag almost every design. The gate
        thresholds are the PLS1 ones and have not been re-derived for a
        two-block design.

        ``'raw_perm'`` needs a CV statistic that a method with no
        ``predict`` does not have. ``'score'`` is not implemented: its
        symmetric analog is an RV-type test on ``||X'Y||_F^2``, which tests
        a different estimand from the LV1 held-out correlation. ``'e'``
        needs a generative model that symmetric cross-decomposition does
        not supply.
    args : dict | None
        ``'split_exact'``: ``{'n_perm': int, 'n_splits': int}``, defaults
        1000 and 50. ``'split_nb'``: ``{'n_splits': int, 'force': bool}``,
        defaults 50 and False.
    pre_standardized_X, pre_standardized_Y : bool, default False
        Accepted but have no effect on either method: each training half is
        re-standardized with its own moments regardless, and the gate
        standardizes its own copy of X. Kept on the signature so it does not
        change when a method that reads them lands.
    seed : int | None
        RNG seed. ``None`` draws from OS entropy and records the value on
        ``result.seed``.

    Returns
    -------
    ConfirmatoryTestResult
        ``ci`` is always ``None`` for this family, and ``n_eff`` equals ``n``
        (weights are not implemented). ``rho_hat`` is populated for
        ``'split_nb'`` only, and only when the test half has at least 4 rows.
        ``stable_rank`` is populated whenever ``'split_nb'`` was requested —
        it is what the auto-gate saw. ``n_perm`` is ``None`` for
        ``'split_nb'``, which runs no permutations.
    """
    test_method = _string(test_method, "test_method")
    args = _args_dict(args, "args")
    pre_standardized_X = _flag(pre_standardized_X, "pre_standardized_X")
    pre_standardized_Y = _flag(pre_standardized_Y, "pre_standardized_Y")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    Y = _ensure_2d_Y(Y)
    raw = _plskit.pls3_confirmatory_test_raw(
        X, Y, _whole(1 if k is None else k, "k"),
        test_method=test_method, args=args,
        pre_standardized_X=pre_standardized_X,
        pre_standardized_Y=pre_standardized_Y,
        seed=_seed(seed),
        disable_parallelism=disable_parallelism,
        verbose=verbose,
    )
    raw.pop("ci", None)
    result = ConfirmatoryTestResult(ci=None, **raw)
    _warn_if_rerouted(
        test_method, result.test_method,
        n_perm=result.n_perm,
        stable_rank=result.stable_rank,
        n_eff=result.n_eff,
    )
    return result


@_convert_errors
def pls1_confirmatory_test(
    X: np.ndarray, y: np.ndarray, k: int = 1,
    *,
    test_method: Literal["raw_perm", "split_nb", "split_exact", "score", "e"],
    args: dict | None = None,
    ci: bool = False,
    n_boot: int | None = None,
    m_rate: float | None = None,
    level: float | None = None,
    max_failure_rate: float | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
    max_skip_rate: float | None = None,
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
    test_method : str
        Test method: ``'raw_perm'``, ``'split_nb'``, ``'split_exact'``,
        ``'score'``, or ``'e'``.

        ``'split_exact'`` is recommended (``test_method`` has no default): a split-half test
        (statistic ``tanh(z̄)``, the mean Fisher-z of held-out correlations)
        calibrated by permutation, so it holds its level on any design.
        ``'split_nb'`` uses the same statistic with an asymptotic correction
        instead of permutations: much cheaper, and meant for n large
        relative to p with a flat X spectrum. The auto-gate checks only a
        coarse version of that: it flags a design whose X has 4 columns or
        fewer, whose effective sample size ``n_eff`` is below 25, or whose
        standardized X (weighted, when ``weights`` is given) has a stable
        rank below 3. A flagged ``'split_nb'`` request runs ``'split_exact'``
        (``n_perm=1000``) instead (``result.test_method`` says so, and Python
        warns). A design that passes the gate is not thereby shown to be in
        the regime above. Pass ``args={'force': True}`` to run
        ``'split_nb'`` anyway; ``split_nb_gate`` reports the decision
        without running a test.

        ``rho_hat`` is populated for ``'split_nb'`` only; ``None`` for every
        other method, including ``'split_exact'``, and ``None`` for
        ``'split_nb'`` itself when the weights are non-uniform (all-equal
        weights count as none) or the test half is too small.
        ``stable_rank`` is populated whenever ``'split_nb'`` was requested:
        it is what the auto-gate saw.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_perm': 500}`` for ``raw_perm``,
        ``{'force': True}`` for ``split_nb``).
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        _docs/concepts/PLS1/weights.md.
    n_boot : int | None, default None
        Number of subsampling resamples for the ``ci`` branch. ``None`` uses
        the engine default, 1000. Inert when ``ci=False``.
    m_rate : float | None, default None
        Subsample-size exponent; ``m = ceil(n ** m_rate)``. ``None`` uses the
        engine default, 0.7. Inert when ``ci=False``.
    level : float | None, default None
        Nominal CI level. ``None`` uses the engine default, 0.95. Inert when
        ``ci=False``.
    max_failure_rate : float | None, default None
        Maximum tolerable combined per-resample failure rate for the ``ci``
        branch. ``None`` uses the engine default, 0.01. Inert when
        ``ci=False``.
    max_skip_rate : float | None, default None
        Subsample-loop skip threshold for the ``ci`` branch (see
        _docs/concepts/effective-sample-size.md). ``None`` uses the engine
        default, 0.01. The CI loop fails with ``PlsKitResamplingDegenerate``
        if ``skipped / total > max_skip_rate``.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed.
    """
    test_method = _string(test_method, "test_method")
    args = _args_dict(args, "args")
    ci = _flag(ci, "ci")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_confirmatory_test_raw(
        X, y, _whole(1 if k is None else k, "k"),
        test_method=test_method, args=args,
        ci=ci,
        n_boot=_whole_or_none(n_boot, "n_boot"),
        m_rate=_number_or_none(m_rate, "m_rate"),
        level=_number_or_none(level, "level"),
        max_failure_rate=_number_or_none(max_failure_rate, "max_failure_rate"),
        pre_standardized=pre_standardized,
        seed=_seed(seed),
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
        max_skip_rate=_number_or_none(max_skip_rate, "max_skip_rate"),
    )
    ci_dict = raw.pop("ci", None)
    ci_obj = _confirmatory_ci_from_dict(ci_dict) if ci_dict is not None else None
    result = ConfirmatoryTestResult(ci=ci_obj, **raw)
    _warn_if_rerouted(
        test_method, result.test_method,
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
        Run the replicate loops serially; single top-level products keep
        the fixed parallel split, so results are bit-identical to the
        parallel run.
    verbose : bool, default False
        Print progress to stderr.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        _docs/concepts/PLS1/weights.md.

    Returns
    -------
    FindKOptimalResult
    """
    return _pls1_find_k_optimal(
        X, y, k_max,
        selector=selector,
        diagnostic=diagnostic,
        args=args,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
        for_fit=False,
    )


def _pls1_find_k_optimal(
    X, y, k_max, *, selector="r2_se", diagnostic=None, args=None,
    pre_standardized=False, seed=None, disable_parallelism=False,
    verbose=False, weights=None, for_fit,
) -> FindKOptimalResult:
    """Body of `pls1_find_k_optimal`; `for_fit=True` is `pls1_fit(k="optimal")`."""
    selector = _string(selector, "selector", "r2_se")
    diagnostic = None if diagnostic is None else _string(diagnostic, "diagnostic")
    args = _args_dict(args, "args")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_find_k_optimal(
        X, y, _whole(k_max, "k_max"),
        selector=selector,
        diagnostic=diagnostic,
        args=args,
        pre_standardized=pre_standardized,
        seed=_seed(seed),
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
        for_fit=for_fit,
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
    alpha: float | None = None,
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
        method (permutation-calibrated, holds its level on any design);
        ``'split_nb'``, the default, is the cheaper asymptotic alternative,
        meant for n large relative to p with a flat X spectrum. Its auto-gate
        flags only X with 4 columns or fewer, ``n_eff`` below 25, or a stable
        rank of the standardized X below 3 (see ``pls1_confirmatory_test``),
        and is evaluated once for the whole sequence: a flagged
        ``'split_nb'`` request runs ``'split_exact'`` for every step and
        ``result.test_method`` says so.
    alpha : float | None, default None
        Significance threshold for rejection. ``None`` uses the engine
        default, 0.05.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_splits': 50}``, or
        ``{'force': True}`` to run ``split_nb`` past the auto-gate).
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed.
    disable_parallelism : bool, default False
        Run the replicate loops serially; single top-level products keep
        the fixed parallel split, so results are bit-identical to the
        parallel run.
    verbose : bool, default False
        Print progress to stderr.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        _docs/concepts/PLS1/weights.md.

    Returns
    -------
    FindKSequenceResult
    """
    return _pls1_find_k_sequence(
        X, y, k_max,
        test_method=test_method,
        alpha=alpha,
        args=args,
        pre_standardized=pre_standardized,
        seed=seed,
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
        for_fit=False,
    )


def _pls1_find_k_sequence(
    X, y, k_max, *, test_method="split_nb", alpha=None, args=None,
    pre_standardized=False, seed=None, disable_parallelism=False,
    verbose=False, weights=None, for_fit,
) -> FindKSequenceResult:
    """Body of `pls1_find_k_sequence`; `for_fit=True` is `pls1_fit(k="sequence")`."""
    test_method = _string(test_method, "test_method", "split_nb")
    args = _args_dict(args, "args")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_find_k_sequence(
        X, y, _whole(k_max, "k_max"),
        test_method=test_method,
        alpha=_number_or_none(alpha, "alpha"),
        args=args,
        pre_standardized=pre_standardized,
        seed=_seed(seed),
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=weights,
        for_fit=for_fit,
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
        With ``keep`` set; each ``W`` column is zero outside its ``keep``
        selected rows. Per-coordinate β CIs are NOT offered under selection
        (post-selection inference); see ``_docs/concepts/sPLS1/keep-and-selection.md``.
    """
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_fit(
        X, y, _whole(k, "k"), _whole(keep, "keep"),
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
    args = _args_dict(args, "args")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_find_keep_optimal(
        X, y, _whole(k, "k"),
        args=args,
        seed=_seed(seed),
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
    selector = _string(selector, "selector", "r2_se")
    diagnostic = None if diagnostic is None else _string(diagnostic, "diagnostic")
    args = _args_dict(args, "args")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_find_k_optimal(
        X, y, _whole(k_max, "k_max"), _whole(keep, "keep"),
        selector=selector,
        diagnostic=diagnostic,
        args=args,
        pre_standardized=pre_standardized,
        seed=_seed(seed),
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
    alpha: float | None = None,
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
    test_method = _string(test_method, "test_method", "split_nb")
    args = _args_dict(args, "args")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.spls1_find_k_sequence(
        X, y, _whole(k_max, "k_max"), _whole(keep, "keep"),
        test_method=test_method,
        alpha=_number_or_none(alpha, "alpha"),
        args=args,
        pre_standardized=pre_standardized,
        seed=_seed(seed),
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
    method = _string(method, "method", "varimax")
    args = _args_dict(args, "args")

    if isinstance(model_or_W, PLS1Result):
        if model_or_W.rotation_spec is not None:
            raise PlsKitError(
                "model already has a rotation_spec; re-rotation is not supported",
                code="already_rotated",
            )
        return _rotate_model(model_or_W, method, L, args)
    # Another result (a PLS3Result, say) or None is refused in plskit-bind's
    # words; anything else is W, read like every other array argument (an
    # ndarray, a nested list, ...), so a non-array fails as `W` does there.
    if dataclasses.is_dataclass(model_or_W) and not isinstance(model_or_W, type):
        _require_result(model_or_W, PLS1Result, "rotate() first arg")
    if model_or_W is None:
        raise PlsKitError(
            "rotate() first arg must be a PLS1Result or a matrix, got None",
            code="invalid_argument",
        )
    return _rotate_array(model_or_W, method, L, args)


def _rotate_array(W, method, L, args) -> RotateResult:
    W = _ensure_array(W, "W", 2)
    L_was_provided = L is not None
    L_arr = _ensure_array(L, "L", 2) if L_was_provided else None
    raw = _plskit.rotate(W, method=method, args=args, l=L_arr)
    spec = RotationSpec(
        method=method,
        # The args the engine resolved (defaults filled, values typed), not
        # the caller's dict: `{"max_iter": None}` records 50, `30.0` records 30.
        args=MappingProxyType(raw["args"]),
        R=raw["r"],
        sweeps=raw["sweeps"],
        V_converged=raw["v_converged"],
        L_was_provided=L_was_provided,
    )
    return RotateResult(W_rot=raw["w_rot"], spec=spec)


def _rotate_model(model: PLS1Result, method, L, args) -> PLS1Result:
    rot = _rotate_array(model.W, method, L, args)
    R = rot.spec.R
    # replace() carries every field rotation doesn't touch (e.g. keep,
    # selection_result) so sparse/optimal-k models don't silently lose them
    return dataclasses.replace(
        model,
        T=model.T @ R,
        P=model.P @ R,
        W=rot.W_rot,
        Q=R.T @ model.Q,
        rotation_spec=rot.spec,
    )


@_convert_errors
def pls1_rotation_stability(
    X: np.ndarray, y: np.ndarray, k: int,
    *,
    rotation_method: Literal["varimax"] = "varimax",
    rotation_args: dict | None = None,
    L: np.ndarray | None = None,
    n_boot: int | None = None,
    m_rate: float | None = None,
    level: float | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
    max_skip_rate: float | None = None,
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
    n_boot : int | None, default None
        Number of subsampling resamples. ``None`` uses the engine default,
        1000.
    m_rate : float | None, default None
        Subsample-size exponent; ``m = ceil(n ** m_rate)``. ``None`` uses the
        engine default, 0.7.
    level : float | None, default None
        Nominal CI level (e.g. 0.95). ``None`` uses the engine default, 0.95.
    pre_standardized : bool
        Set ``True`` when ``X`` is already column-standardized.
    seed : int or None
        RNG seed for reproducibility.
    disable_parallelism : bool
        Run the replicate loops serially; single top-level products keep
        the fixed parallel split, so results are bit-identical to the
        parallel run.
    verbose : bool
        Reserved for future progress reporting.
    weights : np.ndarray or None, shape (n,)
        Optional per-observation weights. ``None`` is equivalent to all-ones.
    max_skip_rate : float | None, default None
        Maximum fraction of subsamples that may be skipped (due to weight
        degeneracy) before raising ``PlsKitResamplingDegenerate``. ``None``
        uses the engine default, 0.01.
    """
    rotation_method = _string(rotation_method, "rotation_method", "varimax")
    rotation_args = _args_dict(rotation_args, "rotation_args")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    L_arr = _ensure_array(L, "L", 2) if L is not None else None
    w_arr = _ensure_array(weights, "weights", 1) if weights is not None else None
    raw = _plskit.pls1_rotation_stability_raw(
        X, y, _whole(k, "k"),
        rotation_method=rotation_method,
        rotation_args=rotation_args,
        l=L_arr,
        n_boot=_whole_or_none(n_boot, "n_boot"),
        m_rate=_number_or_none(m_rate, "m_rate"),
        level=_number_or_none(level, "level"),
        pre_standardized=pre_standardized,
        seed=_seed(seed),
        disable_parallelism=disable_parallelism,
        verbose=verbose,
        weights=w_arr,
        max_skip_rate=_number_or_none(max_skip_rate, "max_skip_rate"),
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
    n_perm: int | None = None,
    return_perm_matrix: bool = False,
    pre_standardized: bool = False,
    seed: int | None = None,
    disable_parallelism: bool = False,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> PermNullResult:
    """Permutation-null engine for PLS1 β. Signed per-voxel z + optional perm matrix.

    Pair with `pls1_confirmatory_test(test_method="split_exact")` as an omnibus gate
    before spending the `n_perm` permutation budget at fMRI scale.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix. Standardized internally unless `pre_standardized=True`.
    y : np.ndarray, shape (n,)
        Response vector.
    k : int
        Number of PLS components.
    n_perm : int | None, default None
        Number of permutations (must be ≥ 100). ``None`` uses the engine
        default, 1000.
    return_perm_matrix : bool, default False
        If True, return the full `(n_perm, d)` β matrix. Memory-intensive at
        fMRI scale; use only when needed for cluster-based correction.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed for reproducibility.
    disable_parallelism : bool, default False
        Run the replicate loops serially; single top-level products keep
        the fixed parallel split, so results are bit-identical to the
        parallel run.
    verbose : bool, default False
        Reserved for future progress reporting.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are NOT permuted — `w[i]` stays tied to row `i` regardless
        of which `y` value lands there under the permutation.
    """
    return_perm_matrix = _flag(return_perm_matrix, "return_perm_matrix")
    pre_standardized = _flag(pre_standardized, "pre_standardized")
    disable_parallelism = _flag(disable_parallelism, "disable_parallelism")
    verbose = _flag(verbose, "verbose")
    X = _ensure_array(X, "X", 2)
    y = _ensure_array(y, "y", 1)
    if weights is not None:
        weights = _ensure_array(weights, "weights", 1)
    raw = _plskit.pls1_perm_null_raw(
        X, y, _whole(k, "k"),
        n_perm=_whole_or_none(n_perm, "n_perm"),
        return_perm_matrix=return_perm_matrix,
        pre_standardized=pre_standardized,
        seed=_seed(seed),
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
