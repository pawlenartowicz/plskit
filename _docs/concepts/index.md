# Concepts

Methods-driven background. The pages here describe what `plskit`
implements and the inference contracts it offers, organized by PLS
family. Argument-naming conventions and contributor docs live under
[internals](../internals/index.md).

## Pages

- [Why plskit?](whyplskit.md): when to reach for this library *(placeholder)*
- [PLS1](PLS1/index.md): fit, find K, inference, CIs, weights, rotations
- [sPLS1](sPLS1/index.md): sparse PLS1, the `keep` count and its selection ([keep and selection](sPLS1/keep-and-selection.md))
- [PLS3](PLS3/index.md): symmetric PLS3 / PLSSVD (PLSC), [fit and transform](PLS3/fit-and-transform.md), [inference](PLS3/inference.md)
- [sPLS3](sPLS3/index.md): sparse PLS3, the two keep-counts and [what sparsity costs](sPLS3/sparse-saliences.md)
- [Preprocessing](preprocessing.md): canonical standardize recipe, `pre_standardized` flag, chemometrics scalings
- [Limitations](limitations.md): sample-size regimes, conditioning, open questions
- [Effective sample size](effective-sample-size.md): `n_eff` definition, where the check fires, error taxonomy
- [Citation & reproducibility](citation.md): how to cite, how to pin
