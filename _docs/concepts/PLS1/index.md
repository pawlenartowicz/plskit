# PLS1

PLS1 is the asymmetric single-response variant: `X (n × p)` predicting a
single response `y (n,)` through the PLS1 (NIPALS) model. It is one of three
families `plskit` ships, alongside [sparse PLS1](../sPLS1/index.md)
(`spls1_*`) and the symmetric [PLS3 / PLSSVD](../PLS3/index.md)
(`pls3_*`, `plssvd_*`). PLS2, robust PLS1, and MBPLS are *(planned)*.

## Pages

- [Fit and predict](fit-and-predict.md): the core `pls1_fit` / `pls1_predict` workflow
- [Find K](find-k.md): choosing the number of components (`pls1_find_k_optimal`, `pls1_find_k_sequence`)
- [Inference](inference.md): confirmatory tests (`split_exact`, `split_nb`, `score`, `e`, `raw_perm`) (*placeholder, paper-pending*)
- [Confidence intervals](ci.md): rotation-invariant subsample CIs (`pls1_confirmatory_test(ci=True)`) (*placeholder, paper-pending*)
- [Weights](weights.md): observation weights (WLS-style precision / sampling weights)
- [Rotations and stability](rotations-and-stability.md): varimax, `pls1_rotation_stability` (*placeholder, paper-pending*)
