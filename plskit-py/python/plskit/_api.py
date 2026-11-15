"""User-facing Python functions; thin wrapper over the PyO3 cdylib."""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import numpy as np

from plskit import _plskit, _results
from plskit._errors import PlsKitError, PlsKitInvalidWeights, PlsKitResamplingDegenerate
from plskit._results import (
    ConfirmatoryTestResult,
    FactorsResult,
    FindKOptimalResult,
    FindKeepOptimalResult,
    FindKSequenceResult,
    GramMatrixResult,
    PCABasisResult,
    PermAlphaResult,
    PermNullResult,
    PLS1Result,
    PLS3Result,
    PLS3Scores,
    PreprocessResult,
    ProfilesFindNResult,
    ProfilesResult,
    ProfilesSearchResult,
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
# dimensions it may have. The keys, and the `"model"` / `"model_or_mat"` /
# `"model_or_vec"` kinds `_call` tests, mirror `ParamKind::as_str` in plskit-bind's registry
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
    returns. The extension hands over `PLS3Result.converged` and `n_iter` and
    `PermAlphaResult.k_used` as lists and `RotationSpec.args`
    as a dict; they become bool and int64 arrays and a read-only mapping."""
    if type_name == "PLS3Result":
        if fields["converged"] is not None:
            fields["converged"] = np.asarray(fields["converged"], dtype=bool)
        if fields["n_iter"] is not None:
            fields["n_iter"] = np.asarray(fields["n_iter"], dtype=np.int64)
    elif type_name == "PermAlphaResult":
        fields["k_used"] = np.asarray(fields["k_used"], dtype=np.int64)
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
        if kind in ("model", "model_or_mat", "model_or_vec") and _is_record(value):
            arguments[key] = _aligned_model(value)
        elif kind == "model_or_mat":
            arguments[key] = _array_argument(value, key, "mat")
        elif kind == "model_or_vec":
            arguments[key] = _array_argument(value, key, "vec")
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
    See docs/concepts/preprocessing.md for the recipe and the cache pattern.
    """
    return _call("preprocess", X=X, Y=Y, weights=weights)


@_convert_errors
def pls1_fit(
    X: np.ndarray,
    y: np.ndarray,
    k: int | Literal["optimal", "sequence"] = "optimal",
    *,
    k_max: int | None = None,
    find_k_args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
    weights: np.ndarray | None = None,
    covariates: np.ndarray | None = None,
) -> PLS1Result:
    """Fit a PLS1 model.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Predictor matrix. Standardized internally unless `pre_standardized=True`.
    y : np.ndarray, shape (n,)
        Response vector.
    k : int | 'optimal' | 'sequence', default 'optimal'
        Number of PLS components, or ``'optimal'`` / ``'sequence'`` to
        select it by `pls1_find_k_optimal` (selector ``'r2_se'`` unless
        ``find_k_args`` says otherwise) / `pls1_find_k_sequence` and fit at
        the selected k. A y with no first component (constant, or
        orthogonal to X) raises ``PlsKitError(code="optimal_no_component")``
        under ``'optimal'``; an int k fits the ``k_used = 0`` zero model.
        A float must be whole (``2.0``, not ``2.7``).
    k_max : int | None
        Maximum k to try when ``k='optimal'`` or ``k='sequence'``. ``None``
        means ``min(9, p)``, lowered further so that the weights' effective
        n and the covariates' residual degrees of freedom both stay at least
        ``k_max + 1``, and never below 1. Must be ``None`` when ``k`` is an
        int.
    find_k_args : dict | None
        Extra kwargs forwarded to `pls1_find_k_optimal` / `pls1_find_k_sequence`.
        Allowed keys are the public params of the target function *except*
        ``seed``, ``pre_standardized``, ``weights`` and ``covariates`` — pass those on
        ``pls1_fit`` directly — and ``verbose``, which the selection run
        does not take. Unknown keys
        raise ``PlsKitError(code="invalid_args")`` listing the allowed set.
        Must be ``None`` when ``k`` is an int.
    pre_standardized : bool, default False
        If True, skip standardization — X and y are assumed already zero-mean,
        unit-variance. See docs/concepts/preprocessing.md (the
        ``pre_standardized`` decision table) and ``plskit.preprocess`` for the
        cache pattern.
    seed : int | None
        RNG seed forwarded to ``pls1_find_k_optimal`` / ``pls1_find_k_sequence``
        when ``k`` is a string. Without a seed (the default call included)
        the selection draws its folds from OS entropy, so the chosen k can
        differ between runs; ``selection_result.seed`` repeats a run.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        docs/concepts/PLS1/weights.md.
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        Covariates to adjust for; a 1-D array is one covariate. An intercept
        is always added, and a column that is constant or a linear
        combination of the intercept and earlier columns is dropped. X is
        standardized first, then X and y are replaced by their (weighted)
        least-squares residuals on ``[1, Z]`` over all rows. See
        docs/concepts/PLS1/covariates.md.

    Returns
    -------
    PLS1Result
        With covariates, ``covariate_coef`` holds their coefficients (0 at a
        dropped column), ``intercept`` and ``beta`` give
        ``y_hat = intercept + X @ beta + Z @ covariate_coef``, and the other
        fields describe the adjusted data.
    """
    return _call("pls1_fit", X=X, y=y, k=k, k_max=k_max, find_k_args=find_k_args,
                 pre_standardized=pre_standardized, seed=seed, weights=weights,
                 covariates=covariates)


@_convert_errors
def pls1_predict(
    model: PLS1Result,
    X_new: np.ndarray,
    *,
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    """Predict y for new rows: ``intercept + X_new @ beta``, plus
    ``covariates @ covariate_coef`` for a model fitted with covariates.

    Parameters
    ----------
    model : PLS1Result
    X_new : np.ndarray, shape (m, d)
    covariates : np.ndarray | None, shape (m, p_z) or (m,), default None
        The covariates of the new rows, with the columns the model was
        fitted with. Required for a model fitted with covariates, refused
        otherwise.
    """
    return _call("pls1_predict", model=model, X_new=X_new, covariates=covariates)


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
    choose sparsity or rotation, not both. ``rotate`` warns when it is
    given a sparse ``PLS3Result``.
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
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    k: int = 1,
    *,
    test_method: Literal["auto", "split_exact", "split_nb", "sv_perm"] = "auto",
    args: dict | None = None,
    pre_standardized_X: bool = False,
    pre_standardized_Y: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    gram: GramMatrixResult | np.ndarray | None = None,
    n_features: int | None = None,
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
    X : np.ndarray, shape (n, p), or None
        Pass ``X`` or ``gram``, not both.
    Y : np.ndarray, shape (n, q)
        Required.
    k : int, default 1
        Must be 1. Above LV1 the training-half component ordering need not
        survive to the test half, and whether the statistic should then be
        per-component or subspace-level is not settled.
    test_method : {'auto', 'split_exact', 'split_nb', 'sv_perm'}, default 'auto'
        ``'auto'`` runs ``'split_exact'`` on every design, and
        ``result.test_method`` says so; ``'split_nb'`` runs only when
        requested. This may change between versions. See
        docs/concepts/PLS1/inference.md.

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
        still carries over: conditional on X, the training half and the
        all-row scales, one held-out score vector is fixed and the rows of
        the other are exchangeable, so under the null ``r`` on one split
        follows the ordinary null correlation law to the same approximation
        as in PLS1. With both ``pre_standardized_X`` and
        ``pre_standardized_Y`` set the statement is exact. The
        between-split correction the
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
        a different estimand from the LV1 held-out correlation.

        ``'sv_perm'``: a permutation test on the first singular value of
        ``X̃ᵀỸ`` (McIntosh and Lobaugh 2004). The rows of Y are shuffled
        against X ``n_perm`` times, and ``p = (1 + #{null >= observed}) /
        (n_perm + 1)``. It runs at ``k=1`` only. With ``rotate=True`` each
        permuted singular vector set is Procrustes-rotated to the observed
        one before its singular value is taken.

        ``'e'`` needs a generative model that symmetric
        cross-decomposition does not supply.
    args : dict | None
        ``'split_exact'``: ``{'n_perm': int, 'n_splits': int}``, defaults
        1000 and 32. ``'split_nb'``: ``{'n_splits': int, 'force': bool}``,
        defaults 32 and False. ``'auto'``: ``{'n_perm': int, 'n_splits': int}``,
        defaults 1000 and 32; ``n_perm`` is ignored when ``'split_nb'`` runs.
        ``'sv_perm'``: ``{'n_perm': int, 'rotate': bool}``, defaults 1000 and
        False. ``rotate`` is a PLS3 option, recorded on ``result.rotate``;
        ``rotate=True`` is refused in ``pls1_confirmatory_test``.
    pre_standardized_X, pre_standardized_Y : bool, default False
        Skip the one standardization of that block over all rows. Without
        the flag, the block's columns are divided once by their standard
        deviations over all ``n`` rows, as in ``pls3_fit``; each split then
        centres its training rows with their own means and its held-out
        rows with those same means, and nothing is scaled per split. With
        the flag, the block is used as given and each split only centres.
        Y's scale comes from its rows before any permutation, so permuting
        rows does not change it. The ``'split_nb'`` gate reads X under the
        same rule.
    seed : int | None
        RNG seed. ``None`` draws from OS entropy and records the value on
        ``result.seed``.
    gram : GramMatrixResult | np.ndarray | None
        The Gram matrix of X in place of ``X``: ``gram_matrix(X)``'s result,
        or an ``(n, n)`` array ``X̃ @ X̃.T`` with ``n_features``. It is centred
        over its rows (so ``result.subset(rows)`` is accepted) and checked on
        its entries, and every method runs from it: on ``gram_matrix(X, ...)``
        the result equals the call on ``X`` at the same seed, to rounding.
        ``pre_standardized_X`` is read from the result (passing it raises
        ``invalid_argument``); with an array it says how ``X̃`` was scaled.
    n_features : int | None
        The column count of ``X̃``, required with an array ``gram`` and
        refused with a ``GramMatrixResult``, which records it.

    Returns
    -------
    ConfirmatoryTestResult
        ``ci`` is always ``None`` for this family, and ``n_eff`` equals ``n``
        (weights are not implemented). ``rho_hat`` is populated for
        ``'split_nb'`` only, and only when the test half has at least 4 rows.
        ``stable_rank`` is not ``None`` on an explicit ``'split_nb'``
        request and ``None`` on every ``"auto"`` request. ``n_perm``
        is ``None`` for ``'split_nb'``, which runs no permutations.
    """
    return _call("pls3_confirmatory_test", X=X, Y=Y, k=k, test_method=test_method,
                 args=args, pre_standardized_X=pre_standardized_X,
                 pre_standardized_Y=pre_standardized_Y, seed=seed, verbose=verbose,
                 gram=gram, n_features=n_features)


@_convert_errors
def pls1_confirmatory_test(
    X: np.ndarray | None = None, y: np.ndarray | None = None,
    k: int | Sequence[int] | range | None = None,
    *,
    test_method: Literal[
        "auto", "raw_perm", "split_nb", "split_exact", "score", "sv_perm", "e"
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
    covariates: np.ndarray | None = None,
    null_y: np.ndarray | None = None,
    gram: GramMatrixResult | np.ndarray | None = None,
    n_features: int | None = None,
) -> ConfirmatoryTestResult:
    """Run the confirmatory PLS1 omnibus test at one K or over a set of K.

    With a set, the test asks whether X predicts y at some k in the set and
    returns one p-value, combined over the set by min-p (Westfall and Young):
    each permutation's smallest per-k p is compared with the observed one.
    The set is fixed before the test sees y, so the test stays
    confirmatory. The test names no k to fit at: fit with ``pls1_fit(X, y)``,
    which chooses its own k.

    Parameters
    ----------
    X : np.ndarray, shape (n, d), or None
        Predictor matrix. Pass ``X`` or ``gram``, not both.
    y : np.ndarray, shape (n,)
        Response vector. Required.
    k : int | Sequence[int] | range | None, default None
        The component counts to test: one integer, or a set (a list, a
        ``range`` or an integer array). A set is sorted; an empty set, a
        duplicate or a k of 0 raises ``invalid_argument``, and a set past a
        limit of the method (``k <= d``, the split sizes, ``k <= 10`` with
        ``gram``) is an error, never cut. ``None`` is ``1..8``, cut at
        ``min(8, d, n // 2 - 2)`` and below the weights' effective n
        (``n_eff >= k + 1``) and the covariates' residual df, never below 1;
        it is ``{1}`` for ``'split_nb'``, ``'score'``, ``'sv_perm'``, ``'e'`` and
        ``ci=True``, which refuse a set of more than one k. ``'auto'`` with
        more than one k runs ``'split_exact'``, which refits for every
        permutation: on data where ``'auto'`` would pick ``'split_nb'`` the
        default call costs much more than ``k=1, test_method='split_nb'``.
    test_method : str, default 'auto'
        Test method: ``'auto'``, ``'raw_perm'``, ``'split_nb'``,
        ``'split_exact'``, ``'score'``, ``'sv_perm'``, or ``'e'``.

        ``'sv_perm'`` is a permutation test on the largest singular value of
        ``X̃ᵀỹ``, at ``k=1`` only. ``'score'`` takes ``{'n_perm': int | None}``:
        ``None`` (the default) is the closed form, and an integer runs the
        same permutation test as ``'sv_perm'`` and reports ``T = ||X̃ᵀỹ||²``.

        ``'auto'`` runs ``'split_exact'`` or ``'split_nb'``, chosen once per
        call from X and ``weights`` (never y), and ``result.test_method`` says
        which ran. It picks ``'split_exact'`` when the ``'split_nb'``
        auto-gate flags the design (validity), when ``n_eff`` is at or below
        1000 at ``k=1`` without ``keep`` or below 250 otherwise (power), or,
        at ``k=1`` without ``keep``, when p exceeds 100 times n (cost: there ``'split_exact'`` costs about as much as ``'split_nb'``
        on very wide X; with ``k >= 2`` or a sparse ``keep`` it refits for
        every permutation and costs more, so this clause does not apply);
        otherwise ``'split_nb'``. The
        thresholds may change between versions. See
        docs/concepts/PLS1/inference.md.

        ``'split_exact'``: a split-half test
        (statistic ``tanh(z̄)``, the mean Fisher-z of held-out correlations)
        calibrated by permutation, so it holds its level on any design.
        ``'split_nb'`` uses the same statistic with an asymptotic correction
        instead of permutations: much cheaper, and meant for n large
        relative to p with a flat X spectrum. The auto-gate checks only a
        coarse version of that: it flags a design whose X has 4 columns or
        fewer, whose effective sample size ``n_eff`` is below 25, or whose
        X, as the tests read it (standardized once over all rows, weighted
        when ``weights`` is given, or only centred under
        ``pre_standardized``), has a stable rank below 3. A flagged explicit
        ``'split_nb'`` request runs
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
        check (p > 4, ``n_eff`` above 1000 at ``k=1`` without ``keep``, at
        least 250 otherwise, and p ≤ 100·n at ``k=1`` without ``keep``);
        ``None`` otherwise.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_perm': 500}`` for ``raw_perm``,
        ``{'force': True}`` for ``split_nb``; ``'auto'`` takes ``n_perm`` and
        ``n_splits``, with ``n_perm`` ignored when ``'split_nb'`` runs;
        ``'score'`` takes ``n_perm`` (default ``None``: the closed form);
        ``'sv_perm'`` takes ``n_perm`` (default 1000), and ``rotate`` must stay
        False).
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        docs/concepts/PLS1/weights.md.
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        Covariates to adjust for; a 1-D array is one covariate. An intercept
        is always added, and a column that is constant or a linear
        combination of the intercept and earlier columns is dropped. X is
        standardized first, then X and y are replaced by their (weighted)
        least-squares residuals on ``[1, Z]`` over all rows. See
        docs/concepts/PLS1/covariates.md.
        Taken by ``'split_exact'``, ``'raw_perm'``, ``'sv_perm'`` and
        ``'score'`` with ``n_perm`` set, whose null columns
        are then Freedman–Lane residuals (approximately exact, not exact);
        ``'auto'`` runs ``'split_exact'``, and ``'split_nb'``, ``'score'``
        without ``n_perm`` and ``'e'`` raise ``invalid_argument``. With ``gram`` the Gram
        matrix is adjusted as ``M_Z @ G @ M_Z``, Z having one row per row of
        ``gram``.
    null_y : np.ndarray | None, shape (n, B), default None
        Your own null outcomes, one per column, for ``'split_exact'`` only:
        the statistic of each column, on the same splits as ``y``, makes the
        null distribution, and ``p = (#{null >= observed} + 1) / (B + 1)``
        at one ``k``; over a set the reported p is the min-p p over the set
        (per-k p in ``pvalues_by_k``).
        ``args`` then takes no ``n_perm`` (``invalid_args``), and
        ``covariates`` are refused.
    ci : bool, default False
        If True, run an independent resampling pass after the test and fill
        ``result.ci`` (a ``ConfirmatoryCI``). ``ci`` is beta, not to be
        trusted yet. It scales X and y once over all rows, as the held-out
        tests do (not under ``pre_standardized``). Its ``holdout_corr`` half
        split centres the training half with its own means and the held-out
        half with the training means, as the held-out tests do. The β and
        leverage resamples are centred with their own means too, except
        under ``pre_standardized``, where they use X and y as given, as the
        reference fit does.
    n_boot : int | None, default None
        Number of subsampling resamples for the ``ci`` branch. ``None`` uses
        the engine default, 1000. Inert when ``ci=False``.
    m_rate : float | None, default None
        Subsample-size exponent for β; ``m = ceil(n ** m_rate)``.
        ``holdout_corr`` uses half splits and does not depend on it. ``None``
        uses the engine default, 0.7. Inert when ``ci=False``.
    level : float | None, default None
        Nominal CI level. ``None`` uses the engine default, 0.95. Inert when
        ``ci=False``.
    max_failure_rate : float | None, default None
        Maximum tolerable combined per-resample failure rate for the ``ci``
        branch. ``None`` uses the engine default, 0.01. Inert when
        ``ci=False``.
    max_skip_rate : float | None, default None
        Subsample-loop skip threshold for the ``ci`` branch (see
        docs/concepts/effective-sample-size.md). ``None`` uses the engine
        default, 0.01. The CI loop fails with ``PlsKitResamplingDegenerate``
        if ``skipped / total > max_skip_rate``.
    pre_standardized : bool, default False
        If True, X and y are assumed already zero-mean, unit-variance:
        ``'score'``, the reference fit and the β and leverage resamples of
        the ``ci`` branch use them exactly as given, with no centring or
        scaling; its ``holdout_corr`` half split is only centred, as below. In ``'split_exact'``,
        ``'split_nb'`` (and its gate), ``'e'`` and ``'raw_perm'`` the flag
        skips the one standardization of X over all rows. Without it, X is
        divided once by its column standard deviations over all ``n`` rows
        (weighted when ``weights`` is given). Either way each split or fold
        centres its training part with its own means and its held-out part
        with the training means, and nothing is scaled per split.
    seed : int | None
        RNG seed.
    gram : GramMatrixResult | np.ndarray | None
        The Gram matrix of X in place of ``X``: ``gram_matrix(X)``'s result,
        or an ``(n, n)`` array ``X̃ @ X̃.T`` with ``n_features``. It is centred
        over its rows (so ``result.subset(rows)`` is accepted) and checked on
        its entries, and every method runs from it: on
        ``gram_matrix(X, pre_standardized=flag)`` the result equals the call
        on ``X`` with ``pre_standardized=flag`` at the same seed, to
        rounding. ``'score'`` reads the centred matrix, so there it equals
        the call on ``X`` centred over all rows. ``pre_standardized`` is read
        from the result (passing ``True`` raises ``invalid_argument``); with
        an array it says how ``X̃`` was scaled. ``weights``, ``ci=True`` and
        ``k`` above 10 raise ``invalid_argument``.
    n_features : int | None
        The column count of ``X̃``, required with an array ``gram`` and
        refused with a ``GramMatrixResult``, which records it.
    """
    if isinstance(k, range):
        k = list(k)
    return _call("pls1_confirmatory_test", X=X, y=y, k=k, test_method=test_method,
                 args=args, ci=ci, n_boot=n_boot, m_rate=m_rate, level=level,
                 max_failure_rate=max_failure_rate, pre_standardized=pre_standardized,
                 seed=seed, verbose=verbose, weights=weights,
                 max_skip_rate=max_skip_rate, covariates=covariates,
                 null_y=null_y, gram=gram, n_features=n_features)


@_convert_errors
def split_nb_gate(
    X: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    pre_standardized: bool = False,
) -> SplitNbGateResult:
    """Ask whether the ``split_nb`` auto-gate flags a design, without testing.

    Reports the decision ``pls1_confirmatory_test`` and the ``find_k``
    functions make on a call with the same ``pre_standardized``, so you can
    see it before paying for a run. ``fires=True`` means a ``'split_nb'``
    request on this X reroutes to ``'split_exact'`` unless you pass
    ``args={'force': True}``. Only X and the weights enter the rule; y does
    not.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
    pre_standardized : bool, default False
        If True, X is read as given and only centred, as the tests read it
        under the same flag. Without it, X is standardized once over all
        rows (weighted moments when ``weights`` is given).

    Returns
    -------
    SplitNbGateResult
    """
    return _call("split_nb_gate", X=X, weights=weights,
                 pre_standardized=pre_standardized)


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
    covariates: np.ndarray | None = None,
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
        ``'bic'`` counts each component as one degree of freedom. On wide X
        (p much larger than n) the effective degrees of freedom of a PLS fit
        are much larger, so BIC tends to select ``k_max``; ``'r2_se'`` is
        recommended there.
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
        If True, the cross-validation folds of ``'r2_se'`` and ``'r2_max'``
        read X as given and only centre each part, the training rows with
        their own means and the held-out rows with those same means.
        Without the flag the folds read X divided once by its column
        standard deviations over all rows (weighted when ``weights`` is
        given), and centre each part. Each fold standardizes y either way,
        and ``'bic'`` standardizes X itself either way. For the
        ``diagnostic`` the flag governs the ``'split_nb'`` gate and step 1;
        steps from 2 on test the deflated residual on the scale step 1
        used, as in ``pls1_find_k_sequence``.
    seed : int | None
        RNG seed.
    verbose : bool, default False
        Print progress to stderr.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        docs/concepts/PLS1/weights.md.
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        Covariates removed from X and y before selection, as in
        ``pls1_fit``; the diagnostic follows the ``pls1_confirmatory_test``
        rules for covariates.

    Returns
    -------
    FindKOptimalResult
    """
    return _call("pls1_find_k_optimal", X=X, y=y, k_max=k_max, selector=selector,
                 diagnostic=diagnostic, args=args, pre_standardized=pre_standardized,
                 seed=seed, verbose=verbose, weights=weights, covariates=covariates)


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
    covariates: np.ndarray | None = None,
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
        ``pls1_confirmatory_test`` (see docs/concepts/PLS1/inference.md) and
        runs that method for every step; ``result.test_method`` says which.
        ``'split_exact'`` is permutation-calibrated and holds its level on any
        design; ``'split_nb'`` is the cheaper asymptotic alternative, meant
        for n large relative to p with a flat X spectrum. Its auto-gate flags
        only X with 4 columns or fewer, ``n_eff`` below 25, or a stable rank
        of X as the tests read it below 3 (see ``pls1_confirmatory_test``), and
        is evaluated once for the whole sequence: a flagged explicit
        ``'split_nb'`` request runs ``'split_exact'`` for every step and
        ``result.test_method`` says so.
    alpha : float | None, default None
        Significance threshold for rejection. ``None`` uses the engine
        default, 0.05.
    args : dict | None
        Method-specific kwargs (e.g. ``{'n_splits': 32}``, or
        ``{'force': True}`` to run ``split_nb`` past the auto-gate;
        ``'auto'`` takes ``n_perm`` and ``n_splits`` only, and ``n_perm`` is
        ignored when ``'split_nb'`` runs).
    pre_standardized : bool, default False
        Governs the ``'split_nb'`` gate and step 1. Without the flag, X and y
        are standardized once over all rows (weighted when ``weights`` is
        given); with it, they are used as given. Either way each test only
        centres its training part with its own means and its held-out part
        with the training means. Steps from 2 on test the deflated
        residual on the scale step 1 used, each part only centred.
    seed : int | None
        RNG seed.
    verbose : bool, default False
        Print progress to stderr.
    weights : np.ndarray | None, shape (n,), default None
        Non-negative observation weights. ``None`` means uniform weights.
        Weights are normalized to mean 1 before use. See
        docs/concepts/PLS1/weights.md.
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        Covariates removed from X and y before selection, as in
        ``pls1_fit``; each step's test follows the ``pls1_confirmatory_test``
        rules for covariates.

    Returns
    -------
    FindKSequenceResult
    """
    return _call("pls1_find_k_sequence", X=X, y=y, k_max=k_max, test_method=test_method,
                 alpha=alpha, args=args, pre_standardized=pre_standardized, seed=seed,
                 verbose=verbose, weights=weights, covariates=covariates)


@_convert_errors
def spls1_fit(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    keep: int,
    *,
    pre_standardized: bool = False,
    weights: np.ndarray | None = None,
    covariates: np.ndarray | None = None,
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
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        As on ``pls1_fit``.

    Returns
    -------
    PLS1Result
        With ``keep`` set; each ``W`` column is zero outside its ``keep``
        selected rows. Per-coordinate β CIs are NOT offered under selection
        (post-selection inference); see ``docs/concepts/PLS1/sparse.md``.
    """
    return _call("spls1_fit", X=X, y=y, k=k, keep=keep,
                 pre_standardized=pre_standardized, weights=weights,
                 covariates=covariates)


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
    covariates: np.ndarray | None = None,
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
    seed, verbose, weights, covariates
        As on ``pls1_find_k_optimal``.

    Returns
    -------
    FindKeepOptimalResult
    """
    return _call("spls1_find_keep_optimal", X=X, y=y, k=k, args=args, seed=seed,
                 verbose=verbose, weights=weights, covariates=covariates)


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
    covariates: np.ndarray | None = None,
) -> FindKOptimalResult:
    """`pls1_find_k_optimal` with the inner fitter swapped to the sparse fit
    at the caller's fixed ``keep``. Same selectors, diagnostic, and result
    shape. ``keep = n_features`` reproduces the dense function exactly.

    Note: ``selector='bic'`` reuses the dense complexity penalty (not a
    keep-aware sparse BIC) — it under-penalizes added components and biases
    the selected k upward under sparsity. Deliberate v1 simplification.
    BIC also counts each component as one degree of freedom; on wide X
    (p much larger than n) the effective degrees of freedom are much larger,
    so BIC tends to select ``k_max``, and ``'r2_se'`` is recommended there.
    """
    return _call("spls1_find_k_optimal", X=X, y=y, k_max=k_max, keep=keep,
                 selector=selector, diagnostic=diagnostic, args=args,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights, covariates=covariates)


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
    covariates: np.ndarray | None = None,
) -> FindKSequenceResult:
    """`pls1_find_k_sequence` with the inner fitter swapped to the sparse fit
    at the caller's fixed ``keep``: each step deflates on the sparse residual
    AND tests the sparse marginal component (coherent sequential test).
    ``keep = n_features`` reproduces the dense function exactly.
    """
    return _call("spls1_find_k_sequence", X=X, y=y, k_max=k_max, keep=keep,
                 test_method=test_method, alpha=alpha, args=args,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights, covariates=covariates)


@_convert_errors
def rotate(
    model_or_W,
    X: np.ndarray | None = None,
    *,
    method: Literal["varimax"] = "varimax",
    L: np.ndarray | None = None,
    args: dict | None = None,
    pre_standardized: bool | None = None,
    covariates: np.ndarray | None = None,
):
    """Rotate a `FactorsResult`, a `PLS3Result` or a matrix.

    Parameters
    ----------
    model_or_W : FactorsResult | PLS3Result | np.ndarray
        What gets rotated. A `PLS1Result` is refused: use `profiles`, or
        ``rotate(m.W, ...)`` for the weights alone.
    X : np.ndarray, shape (n, d), optional
        The data that gives the target: the row scores. Give `X` or `L`,
        not both. For a `FactorsResult`, `X` must be the data it was made
        from. An `X` with the right columns but other rows is not checked,
        and the rotation is then wrong. For a `PLS3Result` the target is its
        stored ``x_scores``; `X` is checked for its row count only.
    method : str
        Only ``'varimax'`` is implemented.
    L : np.ndarray, optional
        Any other target of the caller's.
    args : dict | None
        Options of the rotation method.
    pre_standardized : bool | None
        ``True`` uses `X` as given, ``False`` standardizes it. ``None``
        follows the result, or is ``False`` for an array. A mismatch with
        the fit is the caller's responsibility. Refused for a `PLS3Result`.
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        Covariates removed from the target: ``M_Z @ Xs @ W``, the
        least-squares residuals of ``Xs @ W`` on ``[1, Z]``. Required with
        `X` for a `FactorsResult` made with covariates, refused for one made
        without them and for a `PLS3Result`; optional for a matrix. Refused
        without `X`.

    Returns
    -------
    FactorsResult | PLS3Result | RotateResult
        The type follows the first argument. A sparse `PLS3Result` warns
        that rotation spreads its zeros.
    """
    return _call("rotate", model_or_W=model_or_W, X=X, method=method, L=L,
                 args=args, pre_standardized=pre_standardized, covariates=covariates)


@_convert_errors
def pca_basis(X: np.ndarray, *, pre_standardized: bool = False) -> PCABasisResult:
    """Right singular vectors and singular values of standardized `X`.

    Keeps the components with ``s > 1e-10 * s[0]``, so ``V`` has shape
    ``(D, K)`` with ``K`` the rank of `X`. Column signs are arbitrary, and
    `factors` does not depend on them.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    pre_standardized : bool, default False
        If True, `X` is used as given.
    """
    return _call("pca_basis", X=X, pre_standardized=pre_standardized)


@_convert_errors
def factors(
    model_or_coef, X: np.ndarray, n: int,
    *,
    pre_standardized: bool | None = None,
    covariates: np.ndarray | None = None,
) -> FactorsResult:
    """Split a coefficient vector into `n` orthogonal parts that sum to it.

    A factor is a part of one coefficient vector. It is not a latent
    variable, and `n` is not a count of dimensions.

    Parameters
    ----------
    model_or_coef : PLS1Result | np.ndarray, shape (D,)
        A fit from `pls1_fit` or `spls1_fit`, or a coefficient vector on the
        standardized scale: ``model.coef``, not ``model.beta``. A weighted
        fit is refused.
    X : np.ndarray, shape (N, D)
        The data that gives the basis. Variables whose `coef` is exactly
        zero are left out.
    n : int
        Number of factors, ``1 <= n <= K``, with `K` the rank of `X` after
        the variables are left out, plus one when `coef` has a part outside
        its row space.
    pre_standardized : bool | None
        ``True`` uses `X` as given, ``False`` standardizes it. ``None``
        follows the fit, or is ``False`` for a vector. A mismatch with the
        fit is the caller's responsibility.
    covariates : np.ndarray | None, shape (N, p_z) or (N,), default None
        Covariates removed from `X` (standardized without weights, then
        least-squares residuals on ``[1, Z]``) before the basis is taken.
        Required for a fit with covariates, refused for a fit without them;
        optional for a vector, whose source the caller knows.

    Returns
    -------
    FactorsResult
        Not rotated, in block order. ``size`` is nonnegative.
        ``covariate_adjusted`` says whether covariates were removed.
    """
    return _call("factors", model_or_coef=model_or_coef, X=X, n=n,
                 pre_standardized=pre_standardized, covariates=covariates)


@_convert_errors
def profiles_find_n(
    X: np.ndarray, y: np.ndarray, k: int, n_max: int = 9,
    *,
    args: dict | None = None,
    pre_standardized: bool = False,
    seed: int | None = None,
) -> ProfilesFindNResult:
    """Choose the number of factors by repeated cross-validation.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    k : int
        PLS components used in every fold.
    n_max : int, default 9
        Upper limit of the search. It does not limit `n` in `profiles`.
    args : dict | None
        ``n_folds`` (default 5), ``n_repeats`` (default 2) and ``n_perm``
        (default 20, shuffles for the parallel analysis). A fold with a
        constant training column, or a held-out fold with no finite R², is
        not used and is counted in ``n_failed_folds``. Fewer than two usable
        folds raises ``invalid_input``.
    pre_standardized : bool
        If True, `X` is used as given: the parallel analysis does not
        standardize it, and the folds only centre it. Otherwise the folds
        read `X` divided once by its column standard deviations over all
        rows, each fold centred by its training rows' means.
    seed : int | None
        Recorded on the result.
    """
    return _call("profiles_find_n", X=X, y=y, k=k, n_max=n_max, args=args,
                 pre_standardized=pre_standardized, seed=seed)


@_convert_errors
def profiles(
    model_or_coef, X: np.ndarray, n: int,
    *,
    rotation: Literal["varimax"] = "varimax",
    args: dict | None = None,
    pre_standardized: bool | None = None,
) -> ProfilesResult:
    """Split the coefficient vector of a fit into `n` factors and rotate them.

    A factor is a part of one coefficient vector. It is not a latent
    variable, and `n` is not a count of dimensions.

    Parameters
    ----------
    model_or_coef : PLS1Result | np.ndarray, shape (D,)
        A fit from `pls1_fit` or `spls1_fit`, or a coefficient vector on the
        standardized scale: ``model.coef``, not ``model.beta``. A weighted
        fit, and a fit with covariates, are refused (use `factors` and
        `rotate` with ``covariates``). With a vector,
        ``ProfilesResult.model`` is ``None``.
    X : np.ndarray, shape (N, D)
        The data of the fit.
    n : int
        Number of factors.
    rotation : str
        Passed to `rotate` as ``method``. Only ``'varimax'`` is implemented.
    args : dict | None
        Passed to `rotate`.
    pre_standardized : bool | None
        As in `factors`.
    """
    return _call("profiles", model_or_coef=model_or_coef, X=X, n=n,
                 rotation=rotation, args=args,
                 pre_standardized=pre_standardized)


@_convert_errors
def profiles_search(
    model: PLS1Result, X: np.ndarray, y: np.ndarray,
    *,
    n_max: int = 9,
    rotation: Literal["varimax"] = "varimax",
    args: dict | None = None,
    find_n_args: dict | None = None,
    seed: int | None = None,
) -> ProfilesSearchResult:
    """`profiles` at the `n` chosen by `profiles_find_n`, with the search kept.

    Parameters
    ----------
    model, X, rotation, args :
        As in `profiles`. A model fitted with covariates is refused.
    y : np.ndarray, shape (n,)
        The outcome of the fit.
    n_max : int, default 9
        Upper limit of the search.
    find_n_args : dict | None
        Passed to `profiles_find_n` as ``args``. Its ``k`` is
        ``model.k_used``. When it returns ``n_star = 0``, this runs with
        ``n = 1``.
    seed : int | None
        Recorded on the result.
    """
    return _call("profiles_search", model=model, X=X, y=y, n_max=n_max,
                 rotation=rotation, args=args, find_n_args=find_n_args, seed=seed)


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
    covariates: np.ndarray | None = None,
) -> PermNullResult:
    """Permutation-null engine for PLS1 β. Signed per-voxel z + optional perm matrix.

    Pair with `pls1_confirmatory_test(test_method="split_exact")` as an omnibus gate
    before spending the `n_perm` permutation budget at fMRI scale.

    It takes X, not a Gram matrix: its per-feature outputs need X. A caller
    who holds only the Gram matrix gets the dual coefficients of the fit from
    `pls1_perm_alpha`.

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
    covariates : np.ndarray | None, shape (n, p_z) or (n,), default None
        Covariates removed from X and y as in ``pls1_fit``; the null
        columns are Freedman–Lane residuals and the β maps are computed on
        the adjusted X.
    """
    return _call("pls1_perm_null", X=X, y=y, k=k, n_perm=n_perm,
                 return_perm_matrix=return_perm_matrix,
                 pre_standardized=pre_standardized, seed=seed, verbose=verbose,
                 weights=weights, covariates=covariates)


@_convert_errors
def pls1_perm_alpha(
    gram: GramMatrixResult | np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    *,
    n_features: int | None = None,
) -> PermAlphaResult:
    """PLS1 dual coefficients for many outcomes, from a Gram matrix.

    For each column of ``Y`` returns the n-vector ``alpha`` with
    ``coef = X̃.T @ alpha``, where ``coef`` is the PLS1 coefficient at ``k``
    components on the standardized scale (``PLS1Result.coef``) and ``X̃`` is
    the standardized block whose Gram matrix is ``gram``. plskit never sees
    ``X̃``. Nothing is random, so there is no ``seed``.

    Parameters
    ----------
    gram : GramMatrixResult | np.ndarray, shape (n, n)
        A ``GramMatrixResult`` (``gram_matrix(X)``, or its
        ``subset(rows)``): centred over its rows first, then checked for a
        non-negative diagonal, symmetry and
        ``|gram[i, j]| <= sqrt(gram[i, i] * gram[j, j])``, so a subset and
        a matrix built under ``pre_standardized`` are accepted; ``X̃`` is
        then its rows of ``X̃`` centred by their own means. Or an array
        ``X̃ @ X̃.T`` for a block whose columns are centred and scaled as
        ``plskit.preprocess`` scales them (sum of squares ``n``), computed in
        float64, which is not centred here and gets every check: a
        non-negative diagonal (exactly), symmetry,
        ``|gram[i, j]| <= sqrt(gram[i, i] * gram[j, j])``,
        ``trace(gram) <= n * n_features`` and row sums near zero, each to
        rounding. ``PlsKitError(code="invalid_input")`` when a check fails.
    Y : np.ndarray, shape (n, m)
        One raw outcome per column; each column is standardized inside.
    k : int, default 1
        Number of components, 1 to 10.
    n_features : int | None
        Number of columns of ``X̃`` summed into ``gram``, constant ones
        included. Required with an array ``gram``, refused with a
        ``GramMatrixResult``, which records it.

    Returns
    -------
    PermAlphaResult
        Each fit, its truncation included, is decided from ``gram`` alone.
        The zero model (a zero column with ``k_used[j] = 0``) comes
        from an outcome that standardizes to exact zeros (all zeros, or a
        constant whose computed mean is exact), and from a ``gram`` whose
        every entry is exactly zero (every outcome then gets it).
    """
    return _call("pls1_perm_alpha", gram=gram, Y=Y, k=k, n_features=n_features)


@_convert_errors
def gram_matrix(X: np.ndarray, *, pre_standardized: bool = False) -> GramMatrixResult:
    """The all-row Gram matrix of `X` as the held-out tests read it.

    ``gram = X̃ @ X̃.T``, where ``X̃`` is `X` centred over all rows and
    divided by its column standard deviations (population, as
    `preprocess` scales them), or only centred when ``pre_standardized``.
    These are the bits the held-out tests compute from `X` without
    weights, so the matrix can be computed once and reused across calls.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        At least 2 rows and 1 column; NaN or infinity raises
        ``PlsKitError(code="non_finite_input")``. No columns raises
        ``PlsKitError(code="invalid_input")``.
    pre_standardized : bool, default False
        If True, `X` is read as given and only centred. An `X` whose
        largest entry is outside ``2**-256 .. 2**256`` is first divided by
        one power of two near that entry, as the held-out tests divide it.

    Returns
    -------
    GramMatrixResult
        ``gram`` ``(n, n)``, ``n_features`` (``d``) and
        ``pre_standardized``. ``result.subset(rows)`` gives the Gram matrix
        of a row subsample on the same all-row centring and scale.
    """
    return _call("gram_matrix", X=X, pre_standardized=pre_standardized)


@_convert_errors
def gram_matrix_blocks(
    blocks: Iterable[np.ndarray], *, pre_standardized: bool = False
) -> GramMatrixResult:
    """The `gram_matrix` of `X` given as column blocks, summed block by block.

    ``X`` is the blocks side by side, and its Gram matrix is the sum of the
    blocks' Gram matrices, so with a generator only one block is in memory
    at a time (plus the ``(n, n)`` sum). Each block is read as
    `gram_matrix` reads `X`, with the same `pre_standardized`.

    Parameters
    ----------
    blocks : iterable of np.ndarray, each shape (n, d_i)
        Every block has the same ``n`` rows as the first. No blocks raises
        ``PlsKitError(code="invalid_input")``; a different row count raises
        ``PlsKitError(code="shape_mismatch")``. Anything else a block gets
        wrong is raised as `gram_matrix` raises it.
    pre_standardized : bool, default False
        As in `gram_matrix`.

    Returns
    -------
    GramMatrixResult
        As `gram_matrix` returns, with ``n_features`` the sum of the
        blocks' column counts.

    Notes
    -----
    The result equals ``gram_matrix(X)`` to rounding, not bit for bit.
    Under ``pre_standardized`` each block is scaled on its own, so data
    outside ``2**-256 .. 2**256`` must use `gram_matrix` on the whole ``X``.
    """
    g, p = None, 0
    for block in blocks:
        r = gram_matrix(block, pre_standardized=pre_standardized)
        if g is None:
            g = r.gram
        elif r.gram.shape != g.shape:
            raise PlsKitError(
                f"block has {r.gram.shape[0]} rows, the first block has {g.shape[0]}",
                code="shape_mismatch",
            )
        else:
            g += r.gram
        p += r.n_features
    if g is None:
        raise PlsKitError("blocks holds no block", code="invalid_input")
    return GramMatrixResult(gram=g, n_features=p, pre_standardized=pre_standardized)


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
    "pls1_perm_alpha",
    "gram_matrix",
    "gram_matrix_blocks",
    "rotate",
    "factors",
    "pca_basis",
    "profiles_find_n",
    "profiles",
    "profiles_search",
    "pls3_fit",
    "plssvd_fit",
    "spls3_fit",
    "pls3_transform",
    "plssvd_transform",
    "pls3_confirmatory_test",
    "split_nb_gate",
]
