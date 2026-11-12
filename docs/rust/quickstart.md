# Quickstart

Three steps on small synthetic data: fit PLS1, run a confirmatory test with `split_exact` over the default set of `k`, and fit PLS3. The same steps in Python are in the
[Python quickstart](../python/quickstart.md).

Add `plskit` as a dependency (see [Installation](installation.md)). Build
and run in release mode (`cargo run --release`): the numerical kernels are
far slower under the dev profile.

## The program

```rust
use plskit::{
    pls1_confirmatory_test, pls1_fit, pls1_predict, pls3_fit, pls3_transform, Col,
    ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, FitOpts, KSpec, Mat,
    Pls3FitOpts, TransformWhich,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic data: 60 observations, 8 predictors. `y` depends on the
    // first two columns plus noise. A deterministic pseudo-noise keeps the
    // example free of an RNG dependency.
    let (n, p) = (60, 8);
    let noise = |i: usize, j: usize| ((i * 7919 + j * 104_729) as f64).sin();
    let x = Mat::<f64>::from_fn(n, p, |i, j| noise(i, j));
    let y = Col::<f64>::from_fn(n, |i| x[(i, 0)] - 0.5 * x[(i, 1)] + 0.5 * noise(i, 99));

    // 1. Fit PLS1 with two components and predict on the training rows.
    let model = pls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(2), None, None, FitOpts::default())?;
    println!("k_used = {}", model.k_used);
    println!("beta[0..2] = [{:.3}, {:.3}]", model.beta[0], model.beta[1]);
    let y_hat = pls1_predict(&model, x.as_ref(), None)?;
    println!("y_hat[0] = {:.3}", y_hat[0]);

    // 2. Confirmatory test with split_exact, the recommended method, over
    //    the default set of k (k: None). Small n_perm / n_splits keep the
    //    example fast; use the defaults (1000 / 32) for real analyses.
    let test = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: None,
            weights: None,
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 199,
                n_splits: 20,
            },
            seed: Some(42),
            ..ConfirmatoryTestOpts::default()
        },
    )?;
    println!("{}: k = {:?}, p = {:.4}", test.test_method, test.k, test.pvalue);

    // 3. Fit PLS3 (PLSSVD) on two blocks: X and a 3-column Y block.
    let q = 3;
    let y_block = Mat::<f64>::from_fn(n, q, |i, j| x[(i, j)] + 0.5 * noise(i, 50 + j));
    let pls3 = pls3_fit(x.as_ref(), y_block.as_ref(), 2, None, Pls3FitOpts::default())?;
    println!(
        "singular values = [{:.3}, {:.3}]",
        pls3.singular_values[0], pls3.singular_values[1]
    );
    let scores = pls3_transform(&pls3, Some(x.as_ref()), None, TransformWhich::XScores)?;
    let x_scores = scores.x_scores.expect("XScores requested");
    println!("x_scores: {} x {}", x_scores.nrows(), x_scores.ncols());

    Ok(())
}
```

Output (Linux x86-64; other platforms can differ in the last
printed digits):

```text
k_used = 2
beta[0..2] = [0.092, -0.112]
y_hat[0] = -0.812
split_exact: k = [1, 2, 3, 4, 5, 6, 7, 8], p = 0.0050
singular values = [163.845, 127.909]
x_scores: 60 x 2
```

## What each step does

**1. Fit.** `pls1_fit(x, y, k, weights, covariates, opts)` takes borrowed views
(`MatRef` / `ColRef`, obtained with `.as_ref()`), a component count as
`KSpec::Fixed(k)`, optional observation weights (`None` for uniform),
optional covariates (`None` for none), and a
`FitOpts` struct. The returned `Pls1Model` carries raw-scale coefficients
(`beta`, `intercept`), the standardized-scale fit (`coef`, `w_star`,
`t_scores`, ...), and `k_used`, which can be smaller than the requested `k`
when `X` or `y` runs out of informative directions. `pls1_predict(&model, x_new, covariates)`
applies the fitted model to new rows.

**2. Test.** `pls1_confirmatory_test` asks whether there is any `X`-`y` signal at some `k` in a set fixed before the test sees `y`. `k: None` is the default set `1..=8`, cut by the design (here `p = 8` and `n = 60` leave all eight); `Some(&[1])` tests one `k`. The test returns one p-value for the whole set, `test.k` lists the tested set, and `test.pvalues_by_k` / `test.statistics_by_k` hold each `k`'s own p and statistic; `test.statistic` is `None` for a set of more than one `k` ([A set of `k`](../concepts/PLS1/inference.md#a-set-of-k)). The test names no `k` to fit at. The method is the `ConfirmatoryArgs` variant; `ConfirmatoryTestOpts::default()` fills `args` with `Auto`, which runs `split_exact` over a set of more than one `k` and picks `split_exact` or `split_nb` from the design at one `k` (see [`"auto"`](../concepts/PLS1/inference.md#test_methodauto) and [Choosing the method in Rust](api.md#choosing-the-method-in-rust)). `split_exact` is the recommended method. A fixed `seed` makes the result reproducible, and the same seed gives byte-identical output at any thread count. `test.test_method` reports the method that actually ran.

**3. PLS3.** `pls3_fit(x, y, k, weights, opts)` takes two matrices: neither
block is the outcome, and the model describes which pattern of `X` covaries
with which pattern of `Y`. `weights` must be `None` (observation weights
are not implemented for this family and are refused with an error).
There is no `pls3_predict`; `pls3_transform` projects new rows of either
block onto the fitted latent variables, using the fit's standardization.

## Next steps

- [API reference](api.md): orientation, Rust signatures, and the Rust-only options
- [Result objects](results.md): every field of `Pls1Model`, `Pls3Model` and the test outputs
- <https://docs.rs/plskit>: generated reference for every public item
