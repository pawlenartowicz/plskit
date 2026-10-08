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

from plskit import _plskit, _results
from plskit._errors import PlsKitError, PlsKitInvalidWeights, PlsKitResamplingDegenerate
from plskit._results import (
    ConfirmatoryTestResult,
    FindKOptimalResult,
    FindKeepOptimalResult,
    FindKSequenceResult,
    PermNullResult,
    PLS1Result,
    PLS3Result,
    PLS3Scores,
    PreprocessResult,
    RotationStabilityResult,
    SplitNbGateResult,
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
# Mirrors `REAL_KINDS` in the extension's `lib.rs` — change together.
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


def _describe(value) -> str:
    """How an error message quotes a rejected value (plskit-bind's
    `describe`): a string as `the string "x"`, so `"7"` and `7` read
    differently, anything else by its repr."""
    if isinstance(value, str):
        return f"the string {json.dumps(value, ensure_ascii=False)}"
    return repr(value)


# Parameter kinds by function name, then parameter name, as plskit-bind's
# registry declares them.
_KINDS = {
    f["name"]: {p["name"]: p["kind"] for p in f["params"]}
    for f in json.loads(_plskit.registry_json())["functions"]
}

# Per array parameter kind: what a message calls the value, and the
# dimensions it may have. The keys, and the `"model"` / `"model_or_mat"`
# kinds `_call` tests, mirror `ParamKind::as_str` in plskit-bind's registry
# — change together.
_ARRAY_KINDS = {
    "mat": ("matrix", "2-D"),
    "vec": ("vector", "1-D"),
    "vec_or_mat": ("array", "1-D or 2-D"),
}


def _array_argument(value, name: str, kind: str) -> np.ndarray:
    """An array argument of `kind` (a key of `_ARRAY_KINDS`) as the
    extension takes it: aligned float64, at most 2-D. A 0-D value is refused
    where a vector fits, because plskit-bind would widen it to a one-element
    vector."""
    what, dims = _ARRAY_KINDS[kind]
    a = _aligned_f64(value, name, what)
    if a.ndim > 2 or (a.ndim == 0 and kind != "mat"):
        raise PlsKitError(
            f"{name} must be {dims}, got {a.ndim}-D", code="invalid_argument"
        )
    return a


def _is_record(value) -> bool:
    """Whether `value` is a model in one of its two forms: a result
    dataclass instance, or a mapping of a result's fields."""
    return isinstance(value, Mapping) or (
        dataclasses.is_dataclass(value) and not isinstance(value, type)
    )


# How deep `_aligned_model` converts; anything deeper goes to the extension
# as it is. Mirrors `MAX_DEPTH` in the extension's `lib.rs`, which refuses
# anything deeper — change together.
_MAX_DEPTH = 32


def _aligned_model(value, path: str = "model", depth: int = 0):
    """A model argument with every array of one or more dimensions in it,
    nested results included, passed through `_aligned_f64`. The caller's
    object is left as it is: a result dataclass is rebuilt with
    `dataclasses.replace`, which keeps its class (the extension tags the
    record with the class name), and a mapping becomes a new dict. Anything
    else is returned as it is, for the extension to read or refuse: a 0-D
    array (a scalar field, read as its item), a dataclass that is not one of
    plskit's result classes, and anything nested deeper than `_MAX_DEPTH`."""
    if isinstance(value, np.ndarray):
        return _aligned_f64(value, path) if value.ndim else value
    if depth > _MAX_DEPTH:
        return value
    if isinstance(value, Mapping):
        return {
            k: _aligned_model(v, f"{path}.{k}", depth + 1)
            for k, v in value.items()
        }
    if getattr(_results, type(value).__name__, None) is type(value):
        return dataclasses.replace(value, **{
            f.name: _aligned_model(
                getattr(value, f.name), f"{path}.{f.name}", depth + 1
            )
            for f in dataclasses.fields(value)
        })
    return value


def _make_result(type_name: str, fields: dict):
    """The result class `type_name` built from the fields the extension
    returns. The extension hands over `PLS3Result.converged` and `n_iter` as
    lists and `RotationSpec.args` as a dict; they become a bool array, an
    int64 array and a read-only mapping."""
    if type_name == "PLS3Result":
        if fields["converged"] is not None:
            fields["converged"] = np.asarray(fields["converged"], dtype=bool)
        if fields["n_iter"] is not None:
            fields["n_iter"] = np.asarray(fields["n_iter"], dtype=np.int64)
    elif type_name == "RotationSpec":
        fields["args"] = MappingProxyType(fields["args"])
    return getattr(_results, type_name)(**fields)


def _call(name: str, **arguments):
    """Run the public function `name` in plskit-bind and return its result.

    `arguments` are the function's parameters by name. plskit-bind
    validates them; here array-likes are read with numpy first, because the
    extension takes only aligned arrays of a real dtype and at most two
    dimensions.

    `stacklevel=4` lands a warning on user code: warn, this helper, the
    public function, its `_convert_errors` wrapper, the caller.
    """
    kinds = _KINDS[name]
    for key, value in arguments.items():
        kind = kinds[key]
        if value is None:
            continue
        if kind in ("model", "model_or_mat") and _is_record(value):
            arguments[key] = _aligned_model(value)
        elif kind == "model_or_mat":
            arguments[key] = _array_argument(value, key, "mat")
        elif kind in _ARRAY_KINDS:
            arguments[key] = _array_argument(value, key, kind)
    result, emitted = _plskit.call(name, arguments, _make_result)
    for warning in emitted:
        warnings.warn(warning["message"], UserWarning, stacklevel=4)
    return result


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
    return _call("preprocess", X=X, Y=Y, weights=weights)


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
        ``seed``, ``pre_standardized``, ``weights``, and ``verbose`` — pass
        those on ``pls1_fit`` directly. Unknown keys
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
    return _call("pls1_fit", X=X, y=y, k=k, k_max=k_max, find_k_args=find_k_args,
                 pre_standardized=pre_standardized, seed=seed, weights=weights)


@_convert_errors
def pls1_predict(model: PLS1Result, X_new: np.ndarray) -> np.ndarray:
    return _call("pls1_predict", model=model, X_new=X_new)


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
    return _call("pls3_fit", X=X, Y=Y, k=k, pre_standardized_X=pre_standardized_X,
                 pre_standardized_Y=pre_standardized_Y, weights=weights)


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
    return _call("plssvd_fit", X=X, Y=Y, k=k, pre_standardized_X=pre_standardized_X,
                 pre_standardized_Y=pre_standardized_Y, weights=weights)


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
    return _call("spls3_fit", X=X, Y=Y, k=k, keep_X=keep_X, keep_Y=keep_Y,
                 pre_standardized_X=pre_standardized_X,
                 pre_standardized_Y=pre_standardized_Y, max_iter=max_iter, tol=tol,
                 weights=weights)


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
    return _call("pls3_transform", model=model, X_new=X_new, Y_new=Y_new, which=which)


@_convert_errors
def plssvd_transform(
    model: PLS3Result,
    X_new: np.ndarray | None = None,
    Y_new: np.ndarray | None = None,
    *,
    which: Literal["x_scores", "y_scores", "both"] = "both",
) -> PLS3Scores:
    """Alias for :func:`pls3_transform` under the SVD-PLS name."""
    return _call("plssvd_transform", model=model, X_new=X_new, Y_new=Y_new, which=which)


@_convert_errors
def pls3_confirmatory_test(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    *,
    test_method: Literal["auto", "split_exact", "split_nb"] = "auto",
    args: dict | None = None,
    pre_standardized_X: bool = False,
    pre_standardized_Y: bool = False,
    seed: int | None = None,
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
    test_method : {'auto', 'split_exact', 'split_nb'}, default 'auto'
        ``'auto'`` runs ``'split_exact'`` or ``'split_nb'``, chosen once per
        call from X (never Y), and ``result.test_method`` says which ran. It
        picks ``'split_exact'`` when the ``'split_nb'`` auto-gate flags the
        design (validity), or when ``n`` is below 250 (power); otherwise
        ``'split_nb'``. Unlike ``pls1_confirmatory_test``, a very wide X does
        not pick ``'split_exact'``: in PLS3 it refits for every permutation,
        so there it costs more than ``'split_nb'``. The thresholds may change between versions. See
        _docs/concepts/PLS1/inference.md.

        ``'split_exact'``: the p-value comes from
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

        Explicit ``'split_nb'`` requests are auto-gated on X exactly as in
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
        defaults 50 and False. ``'auto'``: ``{'n_perm': int, 'n_splits': int}``,
        defaults 1000 and 50; ``n_perm`` is ignored when ``'split_nb'`` runs.
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
        ``stable_rank`` is not ``None`` on an explicit ``'split_nb'``
        request, and on an ``"auto"`` request that reached the stable-rank
        check (p > 4, n_eff ≥ 250); ``None`` otherwise. ``n_perm``
        is ``None`` for ``'split_nb'``, which runs no permutations.
    """
    return _call("pls3_confirmatory_test", X=X, Y=Y, k=k, test_method=test_method,
                 args=args, pre_standardized_X=pre_standardized_X,
                 pre_standardized_Y=pre_standardized_Y, seed=seed, verbose=verbose)


@_convert_errors
def pls1_confirmatory_test(
    X: np.ndarray, y: np.ndarray, k: int = 1,
    *,
    test_method: Literal[
        "auto", "raw_perm", "split_nb", "split_exact", "score", "e"
    ] = "auto",
    args: dict | None = None,
    ci: bool = False,
    n_boot: int | None = None,
    m_rate: float | None = None,
    level: float | None = None,
    max_failure_rate: float | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
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
    test_method : str, default 'auto'
        Test method: ``'auto'``, ``'raw_perm'``, ``'split_nb'``,
        ``'split_exact'``, ``'score'``, or ``'e'``.

        ``'auto'`` runs ``'split_exact'`` or ``'split_nb'``, chosen once per
        call from X and ``weights`` (never y), and ``result.test_method`` says
        which ran. It picks ``'split_exact'`` when the ``'split_nb'``
        auto-gate flags the design (validity), when ``n_eff`` is below 250
        (power), or, at ``k=1`` without ``keep``, when p exceeds 100 times n
        (cost: there ``'split_exact'`` costs about as much as ``'split_nb'``
        on very wide X; with ``k >= 2`` or a sparse ``keep`` it refits for
        every permutation and costs more, so this clause does not apply);
        otherwise ``'split_nb'``. The
        thresholds may change between versions. See
        _docs/concepts/PLS1/inference.md.

        ``'split_exact'``: a split-half test
        (statistic ``tanh(z̄)``, the mean Fisher-z of held-out correlations)
        calibrated by permutation, so it holds its level on any design.
        ``'split_nb'`` uses the same statistic with an asymptotic correction
        instead of permutations: much cheaper, and meant for n large
        relative to p with a flat X spectrum. The auto-gate checks only a
        coarse version of that: it flags a design whose X has 4 columns or
        fewer, whose effective sample size ``n_eff`` is below 25, or whose
        standardized X (weighted, when ``weights`` is given) has a stable
        rank below 3. A flagged explicit ``'split_nb'`` request runs
        ``'split_exact'`` (``n_perm=1000``) instead (``result.test_method`` says so, and Python
        warns). A design that passes the gate is not thereby shown to be in
        the regime above. Pass ``args={'force': True}`` to run
        ``'split_nb'`` anyway; ``split_nb_gate`` reports the decision
        without running a test.

        ``rho_hat`` is populated for ``'split_nb'`` only; ``None`` for every
        other method, including ``'split_exact'``, and ``None`` for
        ``'split_nb'`` itself when the weights are non-uniform (all-equal
        weights count as none) or the test half is too small.
        ``stable_rank`` is not ``None`` on an explicit ``'split_nb'``
        request, and on an ``"auto"`` request that reached the stable-rank
        check (p > 4, n_eff ≥ 250, and p ≤ 100·n at ``k=1`` without
        ``keep``); ``None`` otherwise.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_perm': 500}`` for ``raw_perm``,
        ``{'force': True}`` for ``split_nb``; ``'auto'`` takes ``n_perm`` and
        ``n_splits``, with ``n_perm`` ignored when ``'split_nb'`` runs).
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
    return _call("pls1_confirmatory_test", X=X, y=y, k=k, test_method=test_method,
                 args=args, ci=ci, n_boot=n_boot, m_rate=m_rate, level=level,
                 max_failure_rate=max_failure_rate, pre_standardized=pre_standardized,
                 seed=seed, verbose=verbose, weights=weights,
                 max_skip_rate=max_skip_rate)


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
    return _call("split_nb_gate", X=X, weights=weights)


@_convert_errors
def pls1_find_k_optimal(
    X: np.ndarray, y: np.ndarray, k_max: int,
    *,
    selector: Literal["r2_se", "r2_max", "bic"] = "r2_se",
    diagnostic: Literal["raw_perm", "split_nb", "split_exact", "e"] | None = None,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
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
        ``None`` (no diagnostic); ``'auto'`` is not accepted. Selection and test share data, so the
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
    return _call("pls1_find_k_optimal", X=X, y=y, k_max=k_max, selector=selector,
                 diagnostic=diagnostic, args=args, pre_standardized=pre_standardized,
                 seed=seed, verbose=verbose, weights=weights)


@_convert_errors
def pls1_find_k_sequence(
    X: np.ndarray, y: np.ndarray, k_max: int,
    *,
    test_method: Literal["auto", "raw_perm", "split_nb", "split_exact", "e"] = "auto",
    alpha: float | None = None,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
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
    test_method : str, default 'auto'
        Per-step test method: ``'auto'``, ``'raw_perm'``, ``'split_nb'``,
        ``'split_exact'``, or ``'e'``. ``'auto'``, the default, resolves once
        on the full X to ``'split_exact'`` or ``'split_nb'`` by the rule in
        ``pls1_confirmatory_test`` (see _docs/concepts/PLS1/inference.md) and
        runs that method for every step; ``result.test_method`` says which.
        ``'split_exact'`` is permutation-calibrated and holds its level on any
        design; ``'split_nb'`` is the cheaper asymptotic alternative, meant
        for n large relative to p with a flat X spectrum. Its auto-gate flags
        only X with 4 columns or fewer, ``n_eff`` below 25, or a stable rank
        of the standardized X below 3 (see ``pls1_confirmatory_test``), and
        is evaluated once for the whole sequence: a flagged explicit
        ``'split_nb'`` request runs ``'split_exact'`` for every step and
        ``result.test_method`` says so.
    alpha : float | None, default None
        Significance threshold for rejection. ``None`` uses the engine
        default, 0.05.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_splits': 50}``, or
        ``{'force': True}`` to run ``split_nb`` past the auto-gate;
        ``'auto'`` takes ``n_perm`` and ``n_splits`` only, and ``n_perm`` is
        ignored when ``'split_nb'`` runs).
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance.
    seed : int | None
        RNG seed.
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
    return _call("pls1_find_k_sequence", X=X, y=y, k_max=k_max, test_method=test_method,
                 alpha=alpha, args=args, pre_standardized=pre_standardized, seed=seed,
                 verbose=verbose, weights=weights)


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
    return _call("spls1_fit", X=X, y=y, k=k, keep=keep,
                 pre_standardized=pre_standardized, weights=weights)


@_convert_errors
def spls1_find_keep_optimal(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    args: dict | None = None,
    seed: int | None = None,
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
    seed, verbose, weights
        As on ``pls1_find_k_optimal``.

    Returns
    -------
    FindKeepOptimalResult
    """
    return _call("spls1_find_keep_optimal", X=X, y=y, k=k, args=args, seed=seed,
                 verbose=verbose, weights=weights)


@_convert_errors
def spls1_find_k_optimal(
    X: np.ndarray, y: np.ndarray, k_max: int, keep: int,
    *,
    selector: Literal["r2_se", "r2_max", "bic"] = "r2_se",
    diagnostic: Literal["raw_perm", "split_nb", "split_exact", "e"] | None = None,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
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
    return _call("spls1_find_k_optimal", X=X, y=y, k_max=k_max, keep=keep,
                 selector=selector, diagnostic=diagnostic, args=args,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights)


@_convert_errors
def spls1_find_k_sequence(
    X: np.ndarray, y: np.ndarray, k_max: int, keep: int,
    *,
    test_method: Literal["auto", "raw_perm", "split_nb", "split_exact", "e"] = "auto",
    alpha: float | None = None,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    weights: np.ndarray | None = None,
) -> FindKSequenceResult:
    """`pls1_find_k_sequence` with the inner fitter swapped to the sparse fit
    at the caller's fixed ``keep``: each step deflates on the sparse residual
    AND tests the sparse marginal component (coherent sequential test).
    ``keep = n_features`` reproduces the dense function exactly.
    """
    return _call("spls1_find_k_sequence", X=X, y=y, k_max=k_max, keep=keep,
                 test_method=test_method, alpha=alpha, args=args,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights)


@_convert_errors
def rotate(
    model_or_W,
    *,
    method: Literal["varimax"] = "varimax",
    L: np.ndarray | None = None,
    args: dict | None = None,
):
    return _call("rotate", model_or_W=model_or_W, method=method, L=L, args=args)


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
    verbose : bool
        Reserved for future progress reporting.
    weights : np.ndarray or None, shape (n,)
        Optional per-observation weights. ``None`` is equivalent to all-ones.
    max_skip_rate : float | None, default None
        Maximum fraction of subsamples that may be skipped (due to weight
        degeneracy) before raising ``PlsKitResamplingDegenerate``. ``None``
        uses the engine default, 0.01.
    """
    return _call("pls1_rotation_stability", X=X, y=y, k=k,
                 rotation_method=rotation_method, rotation_args=rotation_args, L=L,
                 n_boot=n_boot, m_rate=m_rate, level=level,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights, max_skip_rate=max_skip_rate)


@_convert_errors
def pls1_perm_null(
    X: np.ndarray, y: np.ndarray, k: int,
    *,
    n_perm: int | None = None,
    return_perm_matrix: bool = False,
    pre_standardized: bool = False,
    seed: int | None = None,
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
    verbose : bool, default False
        Reserved for future progress reporting.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are NOT permuted — `w[i]` stays tied to row `i` regardless
        of which `y` value lands there under the permutation.
    """
    return _call("pls1_perm_null", X=X, y=y, k=k, n_perm=n_perm,
                 return_perm_matrix=return_perm_matrix,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights)


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
