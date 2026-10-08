"""The comparison table: every plskit fit that has an outside reference.

A group holds the plskit routines and the reference routines that fit the same
model on the same data. validate.py compares each plskit routine with each
reference; bench.py times every routine. A routine's `out` returns the arrays
that are compared, on plskit's scale.

The sparse fits appear only at their dense endpoint (keep = every variable):
no outside package computes a fixed-count selection.
"""
from importlib.metadata import version
from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np
import plskit
from sklearn.cross_decomposition import PLSSVD, PLSRegression

from common import fit_data

try:
    from ikpls.numpy import PLS as IKPLS
except ImportError:  # older ikpls layouts
    from ikpls.numpy_ikpls import PLS as IKPLS

TOL = 1e-8
Q = 10      # Y columns of the PLS3 designs
N_NEW = 20  # rows the PLS1 predictions are compared on


class Routine(NamedTuple):
    lib: str
    version: str
    routine: str
    fit: Callable       # (d, k) -> model
    out: Callable       # (model, d, k) -> {field: array}
    runs: Callable = lambda n, p: True  # (n, p) -> bool: bench.py times it at this shape


class Group(NamedTuple):
    plskit: list
    refs: list
    sign_by: str | None = None  # field whose columns fix each component's sign


def data(group, n, p):
    X, y = fit_data(n, p)
    rng = np.random.default_rng(1)
    d = SimpleNamespace(X=X, y=y, X_new=rng.standard_normal((N_NEW, p)))
    if group == "pls1":
        d.pre = plskit.preprocess(X, y)  # for pls1_fit_prestd, outside the timed call
    elif group == "pls1_weighted":
        d.w = rng.uniform(0.2, 3.0, n)
    elif group == "pls3":
        m = min(3, p)
        d.Y = X[:, :m] @ rng.standard_normal((m, Q)) + rng.standard_normal((n, Q))
    return d


def max_diff(got, ref, sign_by=None):
    """Largest |got - ref| over the fields. With sign_by, each component of ref is first
    flipped to the sign of got's sign_by column (every field of a component flips together)."""
    s = np.sign(np.sum(got[sign_by] * ref[sign_by], axis=0)) if sign_by else 1.0
    return max(float(np.max(np.abs(got[f] - s * ref[f]))) for f in got)


def _pred(m, d, k):
    return {"pred": np.asarray(plskit.pls1_predict(m, d.X_new)).ravel()}


def _pred_prestd(m, d, k):
    z = plskit.pls1_predict(m, (d.X_new - d.pre.X_mean) / d.pre.X_scale)
    return {"pred": d.pre.Y_mean + d.pre.Y_scale * np.asarray(z).ravel()}


def _pred_sklearn(m, d, k):
    return {"pred": m.predict(d.X_new).ravel()}


def _pred_ikpls(m, d, k):
    return {"pred": np.asarray(m.predict(d.X_new, n_components=k)).ravel()}


def _svd(m, d, k):
    return {f: np.asarray(getattr(m, f)) for f in ("U", "V", "x_scores", "y_scores")}


def _svd_sklearn(m, d, k):
    xs, ys = m.transform(d.X, d.Y)
    # scikit-learn scales by the n - 1 standard deviation, plskit by the n one.
    c = np.sqrt(len(d.X) / (len(d.X) - 1))
    return {"U": m.x_weights_, "V": m.y_weights_, "x_scores": c * xs, "y_scores": c * ys}


def _alg2(n, p):
    """The ikpls algorithm to time: Algorithm 2 when N >> K, else Algorithm 1 (Engstrøm et al.
    2024, J. Open Source Softw. 9(99), 6533). ">>" is taken as 100 times."""
    return n >= 100 * p


PK, SK, IK = plskit.__version__, version("scikit-learn"), version("ikpls")

GROUPS = {
    "pls1": Group(
        plskit=[
            Routine("plskit-py", PK, "pls1_fit",
                    lambda d, k: plskit.pls1_fit(d.X, d.y, k=k), _pred),
            Routine("plskit-py", PK, "pls1_fit_prestd",
                    lambda d, k: plskit.pls1_fit(d.pre.X_std, d.pre.Y_std, k=k, pre_standardized=True),
                    _pred_prestd),
            Routine("plskit-py", PK, "spls1_fit_endpoint",
                    lambda d, k: plskit.spls1_fit(d.X, d.y, k, d.X.shape[1]), _pred),
        ],
        refs=[
            Routine("sklearn", SK, "PLSRegression",
                    lambda d, k: PLSRegression(n_components=k, scale=True).fit(d.X, d.y), _pred_sklearn),
            Routine("ikpls", IK, "alg1",
                    lambda d, k: IKPLS(algorithm=1).fit(d.X, d.y, k), _pred_ikpls,
                    lambda n, p: not _alg2(n, p)),
            Routine("ikpls", IK, "alg2",
                    lambda d, k: IKPLS(algorithm=2).fit(d.X, d.y, k), _pred_ikpls, _alg2),
        ],
    ),
    "pls1_weighted": Group(
        plskit=[
            Routine("plskit-py", PK, "pls1_fit",
                    lambda d, k: plskit.pls1_fit(d.X, d.y, k=k, weights=d.w), _pred),
            Routine("plskit-py", PK, "spls1_fit_endpoint",
                    lambda d, k: plskit.spls1_fit(d.X, d.y, k, d.X.shape[1], weights=d.w), _pred),
        ],
        refs=[
            Routine("ikpls", IK, "alg1",
                    lambda d, k: IKPLS(algorithm=1).fit(d.X, d.y, k, sample_weight=d.w), _pred_ikpls,
                    lambda n, p: not _alg2(n, p)),
            Routine("ikpls", IK, "alg2",
                    lambda d, k: IKPLS(algorithm=2).fit(d.X, d.y, k, sample_weight=d.w), _pred_ikpls,
                    _alg2),
        ],
    ),
    "pls3": Group(
        plskit=[
            Routine("plskit-py", PK, "pls3_fit",
                    lambda d, k: plskit.pls3_fit(d.X, d.Y, k), _svd),
            Routine("plskit-py", PK, "spls3_fit_endpoint",
                    lambda d, k: plskit.spls3_fit(d.X, d.Y, k, keep_X=d.X.shape[1], keep_Y=Q), _svd),
        ],
        refs=[
            Routine("sklearn", SK, "PLSSVD",
                    lambda d, k: PLSSVD(n_components=k, scale=True).fit(d.X, d.Y), _svd_sklearn),
        ],
        sign_by="U",
    ),
}
