# PLS1

PLS1 is the asymmetric single-response variant: `X (n × p)` predicting a
single response `y (n,)` via NIPALS deflation. It is the only PLS family
implemented in `plskit` today — all other families (PLS2, PLS3/PLSSVD,
PLS1 robust, MBPLS) are planned.

## Pages

- [Fit and predict](fit-and-predict.md) — the core `pls1_fit` / `pls1_predict` workflow
- [Find K](find-k.md) — choosing the number of components (`pls1_find_k_optimal`, `pls1_find_k_sequence`)
- [Inference](inference.md) — confirmatory tests (`split_nb`, `split_perm`, `score`, `e`, `raw_perm`) — *placeholder, paper-pending*
- [Confidence intervals](ci.md) — rotation-invariant subsample CIs (`pls1_confirmatory_test(ci=True)`) — *placeholder, paper-pending*
- [Weights](weights.md) — observation weights (WLS-style precision / sampling weights)
- [Rotations and stability](rotations-and-stability.md) — varimax, `pls1_rotation_stability` — *placeholder, paper-pending*
