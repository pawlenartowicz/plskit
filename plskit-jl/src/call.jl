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
    pykw = [k => _to_py(v) for (k, v) in kwargs]
    fn = pygetattr(_plskit, String(fname))
    catcher = _warnings.catch_warnings(record=true)
    caught = catcher.__enter__()
    local out
    try
        _warnings.simplefilter("always")
        out = fn(; pykw...)
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
