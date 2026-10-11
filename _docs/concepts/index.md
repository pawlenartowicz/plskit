# Concepts

Methods-driven background. The pages here describe what `plskit`
implements and the inference contracts it offers, organized by PLS
family. Argument-naming conventions and contributor docs live under
[internals](../internals/index.md).

## Pages

- [Why plskit?](whyplskit.md): when to reach for this library
- [PLS1](PLS1/index.md): fit, find K, inference, CIs, weights, rotations
- [sPLS1](PLS1/sparse.md): sparse PLS1, the `keep` count and its selection
- [PLS3](PLS3/index.md): symmetric PLS3 / PLSSVD (PLSC), [fit and transform](PLS3/fit-and-transform.md), [inference](PLS3/inference.md)
- [sPLS3](PLS3/sparse.md): sparse PLS3, the two keep-counts and [what sparsity costs](PLS3/sparse.md#what-sparsity-costs)
- [Preprocessing](preprocessing.md): canonical standardize recipe, `pre_standardized` flag, chemometrics scalings
- [Limitations](limitations.md): sample-size regimes, conditioning, open questions
- [Effective sample size](effective-sample-size.md): `n_eff` definition, where the check fires, error taxonomy
- [Threads](threads.md): capping the thread count with `PLSKIT_NUM_THREADS`, and bounding resampling memory with `PLSKIT_MEMORY_BUDGET`
- [Citation & reproducibility](citation.md): how to cite, how to pin
