# Profiles

A PLS1 fit gives one coefficient vector `coef`. `factors`, `rotate` and `profiles` split that vector into `n` parts, called factors, that add up to it exactly. Each factor is a vector over the same variables, so you can read which variables carry which part of the prediction.

A factor is a part of one coefficient vector. It is not a latent variable, and `n` is not a count of dimensions.

## The two steps

1. `factors` bins the betas into `n` consecutive blocks. It writes `coef` in the principal-component basis of `X`, cuts those entries into `n` consecutive blocks with about equal shares of the squared coefficient, and gives each block one factor. Variables whose coefficient is exactly zero are left out first and get zeros in every factor. When `coef` has a part outside the row space of the standardized kept columns (for example a sparse fit with more kept columns than `N − 1`), that part, normalized, is one extra last direction of the basis, so it lands in the last factor.
2. `rotate` rotates the factors. A rotation changes how `coef` is shared between the factors, not the total. With `X`, and always in `profiles`, it is varimax on the row scores `Xs W`. Without `X` the target is `W`. The factors are then sorted by size, largest first.

`profiles` runs both steps in one call and also returns the scores `Xs F`, with `Xs` the standardized `X`. Use `factors` and `rotate` separately when you want to look at the blocks before rotating.

## Example

```python
m = plskit.pls1_fit(X, y, k=3)

r = plskit.profiles(m, X, n=3)
r.factors.F        # (D, 3), one factor per column
r.factors.size     # size of each factor, largest first
r.scores           # (N, 3), the score of each row on each factor

# the same in two calls
f = plskit.factors(m, X, n=3)
f = plskit.rotate(f, X)
f.F.sum(axis=1)    # equals m.coef
```

## Choosing n

`k` and `n` are independent. `k` is the number of predictive components of the fit. `n` is how many parts the coefficient is cut into. The only bound on `n` in the code is `K`, the rank of `X` after the variables with a zero coefficient are left out, plus one when the coefficient has a part outside the row space of those columns.

To choose `n` by cross-validation, call `profiles_find_n(X, y, k)`. It returns `n_star`, the smallest `n` whose held-out R² is within 2 SE of the best, or `0` when the best `n` has a held-out R² that is not above 0 by 2 SE. `profiles_search(m, X, y)` runs the search at `k = m.k_used`, then runs `profiles` at `n_star` and keeps the search on the result. When `n_star` is `0` it uses `n = 1`.

```python
r = plskit.profiles_search(m, X, y, seed=1)
r.n_search.n_star                    # the n it chose
r.n_search.r2_mean, r.n_search.r2_se
r.factors.F                          # the rotated factors at that n
```

## Fit form and coefficient form

Pass a `PLS1Result` and `pre_standardized` is taken from the fit. Pass a bare vector and there is no fit to read, so `ProfilesResult.model` is `None` and `pre_standardized` defaults to `False`.

```python
Xs = plskit.preprocess(X).X_std
r = plskit.profiles(m.coef, Xs, n=3, pre_standardized=True)
```

`pre_standardized=True` uses `X` as given. `False` standardizes it. Whatever you set is used, and a mismatch with the fit is your responsibility.

## What you have to get right

- Pass `model.coef`, not `model.beta`. `coef` is on the standardized scale that `factors` works in.
- A coefficient from a weighted fit is refused for a `PLS1Result`, and not checked for a vector. Do not pass one.
- `X` must be the rows the fit came from. `rotate(f, X)` needs the same rows the factors came from. An `X` with the right columns but other rows is not detected, and the rotation is then wrong.
- `rotate` refuses a `PLS1Result`. Use `profiles`, or `rotate(m.W, ...)` for the weights alone.
- `rotate` also takes a `PLS3Result`, where `X` selects the fit's stored `x_scores` as the target and `pre_standardized` is refused. It warns on a sparse fit, whose rotated columns are not unit-norm. See [After `rotate`](../PLS3/fit-and-transform.md#after-rotate).

The call signatures are in the [Python API](../../python/api.md#41-rotation), and the result fields in [results](../../python/results.md).
