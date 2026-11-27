# validation

plskit's fits compared with other packages, for numbers and for speed. A local tool: CI does not run it. The pre-tag checklist runs it.

`cases.py` holds the comparisons: the fits in `GROUPS` and the other routines in `CHECKS`. `validate.py` checks the numbers of both, and `run.sh` times the fits in `GROUPS`.

| Group | plskit | Reference | Compared |
|---|---|---|---|
| `pls1` | `pls1_fit`, `pls1_fit` with `pre_standardized=True`, `spls1_fit` at the dense endpoint | scikit-learn `PLSRegression`, ikpls (Algorithms 1 and 2) | predictions on new rows |
| `pls1_weighted` | `pls1_fit`, `spls1_fit` at the dense endpoint, both with `weights` | ikpls `sample_weight` | predictions on new rows |
| `pls3` | `pls3_fit`, `spls3_fit` at the dense endpoint | scikit-learn `PLSSVD` | `U`, `V`, `x_scores`, `y_scores`, up to sign |
| `pls1_covariates` | `pls1_fit` and `pls1_predict` with `covariates` | scikit-learn `PLSRegression(scale=False)` on X standardized and then residualized on `[1, Z]` by `numpy.linalg.lstsq`, and on y residualized the same way | predictions on new rows |
| `pls3_transform` | `pls3_transform` on new rows | scikit-learn `PLSSVD.transform` | `U` and the new rows' X and Y scores, up to sign |
| `pls3_singular_values` | `pls3_fit` | `numpy.linalg.svd` of `X̃'Ỹ`, both blocks standardized as plskit does | singular values |
| `pca_basis_V`, `pca_basis_s` | `pca_basis` | `numpy.linalg.svd` of standardized X | `V` up to sign; `s` in its own comparison, since the sign alignment flips every field of a component |
| `pls1_bic` | `pls1_find_k_optimal` with `selector="bic"` | `n·log(SSR/n) + j·log n` from scikit-learn `PLSRegression` fits at each j up to k, on y divided by its standard deviation | BIC scores |
| `pls1_perm_alpha` | `pls1_perm_alpha` on `gram_matrix(X)`, as `X̃'α` | scikit-learn `PLSRegression(scale=False)` on standardized X and each standardized column of Y | coefficients |
| `varimax` | `rotate` with varimax and `kaiser_normalize` on `pls1_fit`'s `W`, at `tol` 1e-15 and `max_iter` 1000, since at its default `tol` 1e-8 it stops up to about 2e-6 from a fully converged varimax, more than `1e-8` | factor_analyzer `Rotator(method="varimax", normalize=True)`, run to its 1000-iteration cap: its own tolerance stops it before it converges on a flat pair of columns | rotation `R`, up to sign; at k = 1, `[[1]]` without the package |
| `pyls_sv_perm.py` (standalone) | `pls3_confirmatory_test` and `pls1_confirmatory_test`, `sv_perm` | pyls `behavioral_pls` | statistic to `1e-8`; p-value within Monte Carlo error. Last pass: plskit 0.11.1, 2026-11-27 |

- **Sparse fits** have no outside reference for the selection. R `mixOmics::spls` also keeps a fixed count of X variables per component (`keepX`), but it soft-thresholds the weight vector: each kept weight shrinks by the largest dropped one, where `spls1_fit` keeps the kept weights as they are. Both keep the same variables on the first component; from there the weights, scores and later selections differ, so the two agree only at `keep` = 1 and at the dense endpoint. Sparse fits appear here only at the dense endpoint (`keep` = every variable), where they must equal the dense fit. That row says nothing about the selection itself.
- **PLS3 scores**: scikit-learn standardizes with the `n − 1` standard
  deviation and plskit with the `n` one, so its scores are multiplied by
  `sqrt(n / (n − 1))` before the comparison.
- **PLS3 weights** are not implemented in plskit, so there is no weighted
  PLS3 group.

```bash
./setup.sh                      # once per host (NICE="nice -n 19" ./setup.sh on a shared box)
.venv/bin/python validate.py [--save]
./run.sh [--save]
.venv/bin/python pyls_sv_perm.py   # sv_perm against pyls; about 10 minutes
.venv/bin/python report.py [--host NAME]   # tables from a saved results/<host>/
```

Both commands compute everything on every run and print their results.
`--save` also writes them under `results/`.

## validate.py

Every plskit routine against every reference of its group, at k = 1 and 3 on
a small (50 × 10), a wide (30 × 100) and a tall (2000 × 20) design. Prints the
largest absolute difference per comparison and exits non-zero when one exceeds
`1e-8`. `--save` writes `results/validation.csv`.

## run.sh

Single-core timings of every routine in `GROUPS`, plus **plskit-rs** (the
native Rust core) for unweighted PLS1.

- **Single core**: `single.env` pins Rayon, Accelerate, OpenBLAS, OMP and MKL
  to one thread; the runners refuse to start without it.
- **Wall vs CPU**: both are recorded. On a loaded host, read the CPU tables;
  single-core CPU time mostly survives contention, wall time does not.
  `machine.txt` records the load average at start and end.
- **Grid**: n ∈ {100, 1k, 10k} × p ∈ {10, 100, 1k, 10k, 100k} × k ∈ {1, 5, 10},
  n·p ≤ 2e7; median of 5 batches of ≥ 0.2 s. PLS3 designs have 10 Y columns.
  ikpls is timed with Algorithm 2 when n ≥ 100·p and with Algorithm 1
  otherwise: its authors prefer Algorithm 2 when N ≫ K. Every routine is checked against
  its group's first plskit routine (`1e-8`).
- **Timed call**: the fit only. `pls1_fit` with `pre_standardized=True` is
  timed on X and y standardized by `plskit.preprocess` outside the timed call;
  every other fit standardizes inside the fit. `PLSSVD.fit` returns weights
  only, while `pls3_fit` also returns the scores.
- plskit-rs vs plskit-py: both link the same engine with the same release profile. plskit-rs is timed on a column-major `Mat` from its own generator, plskit-py on a C-ordered numpy array, which the extension copies column-major before the call. The difference is the wrapper cost (result dicts, dataclasses, that copy) plus the data.
- `--save` writes `py.csv`, `rs.csv`, `machine.txt` and `report.md` to
  `results/<host>/`. `BENCH_HOST` overrides the host name.
