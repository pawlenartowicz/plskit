# ---- Julia -> Python arguments -------------------------------------------

_to_py(x::PlsKitResult) = getfield(x, :py)      # round-trip: Python's own object
_to_py(x::Py) = x
_to_py(::Nothing) = pybuiltins.None
_to_py(x::Symbol) = Py(String(x))
_to_py(x::AbstractString) = Py(String(x))
_to_py(x::Bool) = Py(x)
_to_py(x::Integer) = Py(x)
_to_py(x::AbstractFloat) = Py(Float64(x))
# A NumPy view of a dense Julia array (no copy here); any other array is
# collected first. plskit-py applies its own layout and dtype rules.
_to_py(x::Array) = _np.asarray(x)
# `missing` becomes NaN, so the engine answers :non_finite_input as it
# does for NaN (and as R's NA does).
_to_py(x::Array{Union{Missing,T}}) where {T<:Real} =
    _np.asarray(map(v -> v === missing ? NaN : Float64(v), x))
# An Array{Any} (a DataFrame column mix, say) that happens to hold `missing`
# goes through the same path; without this NumPy raises an opaque
# "setting an array element with a sequence" instead of a plskit error.
_to_py(x::Array{Any}) =
    any(ismissing, x) ? _to_py(map(v -> v === missing ? NaN : Float64(v), x)) :
                         _to_py(collect(Float64, x))
_to_py(x::AbstractArray) = _to_py(collect(x))
_to_py(x::NamedTuple) = pydict(Pair{String,Py}[String(k) => _to_py(v) for (k, v) in pairs(x)])
_to_py(x::AbstractDict) = pydict(Pair{String,Py}[_key(k) => _to_py(v) for (k, v) in x])
_to_py(x) = Py(x)

_key(k::Union{Symbol,AbstractString}) = String(k)
_key(k) = throw(ArgumentError("PLSKit: args keys must be Symbols or Strings, got $(repr(k))"))

# ---- the one call path every stub uses ----------------------------------

# PythonCall cannot be used from a thread other than Julia's main thread;
# calling it there segfaults the process instead of raising an error, so
# `_call` guards against it explicitly, before any Python object is touched.
function _check_main_thread(tid=Threads.threadid())
    tid == 1 && return nothing
    error("PLSKit: call plskit functions from Julia's main thread " *
          "(PythonCall cannot be used from other threads); " *
          "see the Julia differences page")
end

function _call(fname::Symbol; kwargs...)
    _check_main_thread()
    return _pycall(pygetattr(_plskit, String(fname)); kwargs...)
end

# Call the Python callable `fn`: a Python PlsKitError becomes a Julia one,
# and Python warnings become Julia warnings. Callers check the thread first.
function _pycall(fn::Py, args...; kwargs...)
    pyargs = [_to_py(a) for a in args]
    pykw = [k => _to_py(v) for (k, v) in kwargs]
    catcher = _warnings.catch_warnings(record=true)
    caught = catcher.__enter__()
    local out
    try
        _warnings.simplefilter("always")
        out = fn(pyargs...; pykw...)
    catch e
        if e isa PyException && pyisinstance(e.v, _plskit.PlsKitError)
            throw(_plskit_error(e.v))
        end
        rethrow()
    finally
        catcher.__exit__(nothing, nothing, nothing)
        for w in caught
            @warn pyconvert(String, pystr(w.message)) _group = :plskit
        end
    end
    return _to_julia(out)
end

# `X` and `y` / `Y` of the two confirmatory tests have a `None` default in
# Python, so that `gram=` can replace `X`, and the stubs in generated.jl take
# them by keyword. These methods keep them positional, as they were.
"""
    pls1_confirmatory_test(X, y; kwargs...) -> ConfirmatoryTestResult

The same call with `X` and `y` positional; every other argument is a keyword, as
in the keyword form. A `gram=` call uses the keyword form, without `X`:
`pls1_confirmatory_test(; gram=g, y=y)`.
"""
pls1_confirmatory_test(X, y; kwargs...) = pls1_confirmatory_test(; X, y, kwargs...)

"""
    pls3_confirmatory_test(X, Y; kwargs...) -> ConfirmatoryTestResult

The same call with `X` and `Y` positional; every other argument is a keyword, as
in the keyword form. A `gram=` call uses the keyword form, without `X`:
`pls3_confirmatory_test(; gram=g, Y=Y)`.
"""
pls3_confirmatory_test(X, Y; kwargs...) = pls3_confirmatory_test(; X, Y, kwargs...)

"""
    PLSKit.subset(g::GramMatrixResult, rows) -> GramMatrixResult

Python's `g.subset(rows)`, with 1-based `rows`: `g.gram[rows, rows]` with the
same `n_features` and `pre_standardized`, the Gram matrix of those rows on
the all-row centring and scale. Not exported, because DataFrames exports a
`subset` of its own.
"""
function subset(g::PlsKitResult{:GramMatrixResult}, rows::AbstractVector{<:Union{Signed,Unsigned}})
    _check_main_thread()
    any(<(1), rows) && throw(PlsKitError(:invalid_argument, "rows must be row numbers, at least 1", NamedTuple()))
    if any(>(typemax(Int)), rows)
        # Past every row; `collect(Int, rows)` would throw InexactError.
        n = pyconvert(Int, getfield(g, :py).gram.shape[0])
        throw(PlsKitError(:invalid_argument, "rows must index the $n rows of the Gram matrix", NamedTuple()))
    end
    return _pycall(getfield(g, :py).subset, collect(Int, rows) .- 1)
end

"""
    gram_matrix_blocks(blocks; pre_standardized=false) -> GramMatrixResult

The `gram_matrix` of `X` given as column blocks, summed block by block. `blocks`
is any iterable of matrices (a generator keeps one block in memory at a time,
plus the `(n, n)` sum), all with the same number of rows. Each block goes through
`gram_matrix`, so its conversion rules and errors apply. Equal to
`gram_matrix(hcat(blocks...))` up to rounding. With `pre_standardized=true`, data
beyond `2^±256` must use `gram_matrix` on the whole `X`. The calls that take the
result rescale it themselves, by a power of four, when its scale would reach their
absolute floors, and map their outputs back. Throws `:invalid_input` for no
blocks and `:shape_mismatch` for a block with another row count.
"""
function gram_matrix_blocks(blocks; pre_standardized=false)
    _check_main_thread()
    acc = nothing
    p = 0
    for b in blocks
        r = gram_matrix(b; pre_standardized)
        g = r.gram
        if acc === nothing
            acc = g
        elseif size(g, 1) != size(acc, 1)
            throw(PlsKitError(:shape_mismatch,
                "block has $(size(g, 1)) rows, the first block has $(size(acc, 1))",
                NamedTuple()))
        else
            acc .+= g
        end
        p += r.n_features
    end
    acc === nothing && throw(PlsKitError(:invalid_input, "blocks holds no block", NamedTuple()))
    return _pycall(_plskit.GramMatrixResult; gram=acc, n_features=p, pre_standardized)
end
