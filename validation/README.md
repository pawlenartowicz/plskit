# validation

plskit's fits compared with other packages, for numbers and for speed. A local
tool: CI does not run it.

`cases.py` holds one table of comparisons. `validate.py` checks the numbers
and `run.sh` times the same routines.

| Group | plskit | Reference | Compared |
|---|---|---|---|
| `pls1` | `pls1_fit`, `pls1_fit` with `pre_standardized=True`, `spls1_fit` at the dense endpoint | scikit-learn `PLSRegression`, ikpls (Algorithms 1 and 2) | predictions on new rows |
| `pls1_weighted` | `pls1_fit`, `spls1_fit` at the dense endpoint, both with `weights` | ikpls `sample_weight` | predictions on new rows |
| `pls3` | `pls3_fit`, `spls3_fit` at the dense endpoint | scikit-learn `PLSSVD` | `U`, `V`, `x_scores`, `y_scores`, up to sign |

- **Sparse fits** have no outside reference: no other package keeps a fixed
  count of variables per component. They appear only at the dense endpoint
  (`keep` = every variable), where they must equal the dense fit. That row
  says nothing about the selection itself.
- **PLS3 singular values** are not compared: `PLSSVD` does not expose them.
- **PLS3 scores**: scikit-learn standardizes with the `n − 1` standard
  deviation and plskit with the `n` one, so its scores are multiplied by
  `sqrt(n / (n − 1))` before the comparison.
- **PLS3 weights** are not implemented in plskit, so there is no weighted
  PLS3 group.

```bash
./setup.sh                      # once per host (NICE="nice -n 19" ./setup.sh on a shared box)
.venv/bin/python validate.py [--save]
./run.sh [--save]
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

Single-core timings of every routine in the table, plus **plskit-rs** (the
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
- plskit-rs vs plskit-py: both link the same engine with the same release profile.
  plskit-rs is timed on a column-major `Mat` from its own generator, plskit-py on a
  C-ordered numpy array that the engine reads row-major in place. The difference is
  the wrapper cost (result dicts, dataclasses) plus the layout and the data.
- `--save` writes `py.csv`, `rs.csv`, `machine.txt` and `report.md` to
  `results/<host>/`. `BENCH_HOST` overrides the host name.
