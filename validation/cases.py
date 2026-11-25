"""The comparison table: every plskit routine that has an outside reference.

A group holds the plskit routines and the reference routines that fit the same
model on the same data. validate.py compares each plskit routine with each
reference; bench.py times every routine of GROUPS. CHECKS holds the comparisons
that are not timed fits. A routine's `out` returns the arrays that are compared,
on plskit's scale.

The sparse fits appear only at their dense endpoint (keep = every variable):
no outside package computes a fixed-count selection.
"""
from importlib.metadata import version
from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np
import plskit
from factor_analyzer import Rotator
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
    elif group == "pls1_covariates":
        d.Z = rng.standard_normal((n, 2))
        d.Z_new = rng.standard_normal((N_NEW, 2))
    elif group in ("pls3", "pls3_transform", "pls3_singular_values", "pls1_perm_alpha"):
        m = min(3, p)
        B = rng.standard_normal((m, Q))
        d.Y = X[:, :m] @ B + rng.standard_normal((n, Q))
        d.Y_new = d.X_new[:, :m] @ B + rng.standard_normal((N_NEW, Q))
    return d


def _std(A):
    """Columns centred and divided by their n standard deviation, as plskit standardizes."""
    return (A - A.mean(axis=0)) / A.std(axis=0)


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


def _covariates_sklearn(d, k):
    """scikit-learn on standardized X and on y, both residualized on [1, Z] by least squares."""
    A = np.column_stack([np.ones(len(d.X)), d.Z])
    Xs = _std(d.X)
    Pi = np.linalg.lstsq(A, Xs, rcond=None)[0]
    g = np.linalg.lstsq(A, d.y, rcond=None)[0]
    m = PLSRegression(n_components=k, scale=False).fit(Xs - A @ Pi, d.y - A @ g)
    return SimpleNamespace(m=m, Pi=Pi, g=g, mean=d.X.mean(axis=0), sd=d.X.std(axis=0))


def _pred_covariates_sklearn(f, d, k):
    A = np.column_stack([np.ones(N_NEW), d.Z_new])
    return {"pred": f.m.predict((d.X_new - f.mean) / f.sd - A @ f.Pi).ravel() + A @ f.g}


def _transform(m, d, k):
    s = plskit.pls3_transform(m, d.X_new, d.Y_new)
    return {"U": np.asarray(m.U), "x_new": np.asarray(s.x_scores), "y_new": np.asarray(s.y_scores)}


def _transform_sklearn(m, d, k):
    xs, ys = m.transform(d.X_new, d.Y_new)
    c = np.sqrt(len(d.X) / (len(d.X) - 1))  # the n - 1 standard deviation, as in _svd_sklearn
    return {"U": m.x_weights_, "x_new": c * xs, "y_new": c * ys}


def _pca_numpy(d, k):
    _, s, Vt = np.linalg.svd(_std(d.X), full_matrices=False)
    K = int(np.sum(s > 1e-10 * s[0]))  # the rank cut pca_basis documents
    return SimpleNamespace(V=Vt[:K].T, s=s[:K])


def _bic_sklearn(d, k):
    """n log(SSR / n) + j log n at each j <= k, on y divided by its n standard deviation."""
    n = len(d.y)
    bic = []
    for j in range(1, k + 1):
        r = (d.y - PLSRegression(n_components=j, scale=True).fit(d.X, d.y).predict(d.X).ravel()) / d.y.std()
        bic.append(n * np.log(np.sum(r**2) / n) + j * np.log(n))
    return np.array(bic)


def _perm_alpha_sklearn(d, k):
    Xs, Ys = _std(d.X), _std(d.Y)
    return np.column_stack([PLSRegression(n_components=k, scale=False).fit(Xs, Ys[:, j]).coef_.ravel()
                            for j in range(Ys.shape[1])])


VARIMAX_ARGS = {"tol": 1e-15, "max_iter": 1000, "kaiser_normalize": True}


def _varimax_fa(d, k):
    """factor_analyzer's rotation of the same W. Its varimax returns a bare matrix for one
    column, so k = 1 takes the identity without calling it."""
    if k == 1:
        return np.array([[1.0]])
    # It stops when `d < old_d * (1 + tol)`, d a sum of singular values, so tol = -1 never stops
    # it early. A tol of 0 or more stops it short of the optimum on a flat pair of columns.
    r = Rotator(method="varimax", normalize=True, tol=-1.0, max_iter=1000)
    r.fit_transform(np.asarray(plskit.pls1_fit(d.X, d.y, k=k).W))
    return r.rotation_


def _alg2(n, p):
    """The ikpls algorithm to time: Algorithm 2 when N >> K, else Algorithm 1 (Engstrøm et al.
    2024, J. Open Source Softw. 9(99), 6533). ">>" is taken as 100 times."""
    return n >= 100 * p


PK, SK, IK = plskit.__version__, version("scikit-learn"), version("ikpls")
NP, FA = version("numpy"), version("factor_analyzer")

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

CHECKS = {
    "pls1_covariates": Group(
        plskit=[Routine("plskit-py", PK, "pls1_fit_covariates",
                        lambda d, k: plskit.pls1_fit(d.X, d.y, k=k, covariates=d.Z),
                        lambda m, d, k: {"pred": np.asarray(
                            plskit.pls1_predict(m, d.X_new, covariates=d.Z_new)).ravel()})],
        refs=[Routine("sklearn", SK, "PLSRegression_residualized", _covariates_sklearn,
                      _pred_covariates_sklearn)],
    ),
    "pls3_transform": Group(
        plskit=[Routine("plskit-py", PK, "pls3_transform",
                        lambda d, k: plskit.pls3_fit(d.X, d.Y, k), _transform)],
        refs=[Routine("sklearn", SK, "PLSSVD.transform",
                      lambda d, k: PLSSVD(n_components=k, scale=True).fit(d.X, d.Y), _transform_sklearn)],
        sign_by="U",
    ),
    "pls3_singular_values": Group(
        plskit=[Routine("plskit-py", PK, "pls3_fit",
                        lambda d, k: plskit.pls3_fit(d.X, d.Y, k),
                        lambda m, d, k: {"sv": np.asarray(m.singular_values)})],
        refs=[Routine("numpy", NP, "linalg.svd",
                      lambda d, k: np.linalg.svd(_std(d.X).T @ _std(d.Y), compute_uv=False)[:k],
                      lambda sv, d, k: {"sv": sv})],
    ),
    "pca_basis_V": Group(
        plskit=[Routine("plskit-py", PK, "pca_basis", lambda d, k: plskit.pca_basis(d.X),
                        lambda m, d, k: {"V": np.asarray(m.V)})],
        refs=[Routine("numpy", NP, "linalg.svd", _pca_numpy, lambda m, d, k: {"V": m.V})],
        sign_by="V",
    ),
    "pca_basis_s": Group(
        plskit=[Routine("plskit-py", PK, "pca_basis", lambda d, k: plskit.pca_basis(d.X),
                        lambda m, d, k: {"s": np.asarray(m.s)})],
        refs=[Routine("numpy", NP, "linalg.svd", _pca_numpy, lambda m, d, k: {"s": m.s})],
    ),
    "pls1_bic": Group(
        plskit=[Routine("plskit-py", PK, "pls1_find_k_optimal_bic",
                        lambda d, k: plskit.pls1_find_k_optimal(d.X, d.y, k_max=k, selector="bic"),
                        lambda m, d, k: {"bic": np.array([m.bic_scores[j] for j in range(1, k + 1)])})],
        refs=[Routine("sklearn", SK, "PLSRegression", _bic_sklearn, lambda bic, d, k: {"bic": bic})],
    ),
    "pls1_perm_alpha": Group(
        plskit=[Routine("plskit-py", PK, "pls1_perm_alpha",
                        lambda d, k: plskit.pls1_perm_alpha(plskit.gram_matrix(d.X), d.Y, k),
                        lambda m, d, k: {"coef": _std(d.X).T @ np.asarray(m.alpha)})],
        refs=[Routine("sklearn", SK, "PLSRegression", _perm_alpha_sklearn, lambda c, d, k: {"coef": c})],
    ),
    "varimax": Group(
        plskit=[Routine("plskit-py", PK, "rotate_varimax",
                        lambda d, k: plskit.rotate(plskit.pls1_fit(d.X, d.y, k=k).W, method="varimax",
                                                   args=VARIMAX_ARGS),
                        lambda m, d, k: {"R": np.asarray(m.spec.R)})],
        refs=[Routine("factor_analyzer", FA, "Rotator_varimax", _varimax_fa, lambda R, d, k: {"R": R})],
        sign_by="R",
    ),
}
