# Shared test helpers. Data comes from NumPy's default_rng, so it is the
# same on every Julia version (Julia's own RNG streams are not stable).
const np = pyimport("numpy")
const inspect = pyimport("inspect")
const pk = pyimport("plskit")

randn_np(seed, dims...) = pyconvert(Array{Float64}, np.random.default_rng(seed).normal(size=dims))

"(X, y) with signal on the first column."
function pls1_data(seed=1; n=60, d=8)
    X = randn_np(seed, n, d)
    y = X[:, 1] .+ 0.5 .* randn_np(seed + 1, n)
    return X, y
end

"A design the split_nb auto-gate flags (ncols <= 4), so split_nb reroutes."
function gated_data(seed=3)
    X = randn_np(seed, 40, 3)
    y = X[:, 1] .+ randn_np(seed + 1, 40)
    return X, y
end

"`f()`'s PlsKitError, or fail the test with what happened instead."
function plskit_error(f)
    try
        f()
    catch e
        e isa PlsKitError && return e
        rethrow()
    end
    error("expected a PlsKitError, but the call returned normally")
end
