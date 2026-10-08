"""
    PlsKitResult{T}

Every plskit result. `T` is the Python type name (`:PLS1Result`, ...);
the exported aliases (`PLS1Result`, `ConfirmatoryTestResult`, ...) name
each one. Fields read as properties (`fit.W`, `fit.seed`) and are Julia
copies of the Python values; editing one does not change what a later
call receives, because a result passed back as an argument hands Python
its own object.
"""
struct PlsKitResult{T}
    fields::NamedTuple
    py::Py
end

function Base.getproperty(r::PlsKitResult{T}, name::Symbol) where {T}
    fields = getfield(r, :fields)
    hasfield(typeof(fields), name) ||
        error("PLSKit: $(T) has no field $(name)")
    return getfield(fields, name)
end
Base.propertynames(r::PlsKitResult, private::Bool=false) = keys(getfield(r, :fields))

# ---- Python -> Julia, by the runtime type of the Python value ------------

# `name` is the field the value sits in: an `int` in a field called
# `seed` becomes a UInt64 (seeds span the full u64 range).
function _to_julia(x::Py, name::Symbol=:_)
    if pyisinstance(x, _np.generic)
        # A NumPy scalar (np.bool_, np.int64, np.float64, ...): unwrap to
        # its native Python value first, as `_array` already does for 0-D
        # arrays, then dispatch on that.
        return _to_julia(x.item(), name)
    elseif pyis(x, pybuiltins.None)
        return nothing
    elseif pyisinstance(x, pybuiltins.bool)
        return pyconvert(Bool, x)
    elseif pyisinstance(x, pybuiltins.int)
        return _to_int(x, name)
    elseif pyisinstance(x, pybuiltins.float)
        return pyconvert(Float64, x)
    elseif pyisinstance(x, pybuiltins.str)
        return pyconvert(String, x)
    elseif pyisinstance(x, _np.ndarray)
        return _array(x, name)
    elseif pytruth(_dataclasses.is_dataclass(x)) && !pyisinstance(x, pybuiltins.type)
        return _result(x)
    elseif pyhasattr(x, "keys")
        return _mapping(x)
    elseif pyisinstance(x, pybuiltins.list) || pyisinstance(x, pybuiltins.tuple)
        return _list(x)
    end
    error("PLSKit: no Julia conversion for a Python $(pytype(x)) in field `$(name)`")
end

# A Python `int` to Julia; the target width depends on the field (`seed`
# spans the full u64 range). A seed overflow error names the field,
# matching the ndarray dtype/dimension errors below.
function _to_int(x::Py, name::Symbol)
    if name === :seed
        try
            return pyconvert(UInt64, x)
        catch e
            error("PLSKit: seed value in field `$(name)` does not fit in UInt64 " *
                  "($(sprint(showerror, e)))")
        end
    end
    return pyconvert(Int, x)
end

function _array(x::Py, name::Symbol)
    ndim = pyconvert(Int, x.ndim)
    ndim == 0 && return _to_julia(x.item(), name)
    kind = pyconvert(String, x.dtype.kind)
    T = kind == "f" ? Float64 : kind == "b" ? Bool : kind == "i" ? Int :
        error("PLSKit: unsupported ndarray dtype $(x.dtype) in field `$(name)`")
    dtype = T === Float64 ? _np.float64 : T === Bool ? _np.bool_ : _np.int64
    # A single Julia-owned copy, laid out F-order (Julia's own column-major
    # order) so `pyconvert` does not need to transpose it; `copy=true`
    # forces a fresh array even when `x` is already F-ordered, so the
    # result never aliases NumPy's memory.
    c = x.astype(dtype, order="F", copy=true)
    ndim == 1 && return pyconvert(Vector{T}, c)
    ndim == 2 && return pyconvert(Matrix{T}, c)
    error("PLSKit: unsupported $(ndim)-D ndarray in field `$(name)`")
end

# dict[int, float] (cv_scores, ...) -> Dict{Int,Float64}; a mapping with
# string keys (RotationSpec.args) -> NamedTuple. An empty mapping is a
# Dict{Int,Float64}: only the int-keyed score maps can be empty.
function _mapping(x::Py)
    ks = collect(x.keys())
    if !isempty(ks) && all(k -> pyisinstance(k, pybuiltins.str), ks)
        names = Tuple(Symbol(pyconvert(String, k)) for k in ks)
        vals = Tuple(_to_julia(x[k], n) for (k, n) in zip(ks, names))
        return NamedTuple{names}(vals)
    end
    return Dict{Int,Float64}(pyconvert(Int, k) => pyconvert(Float64, x[k]) for k in ks)
end

# list[int] (keep_grid) -> Vector{Int}; list[bool] -> Vector{Bool} (checked
# before the int case since bool is an int subtype in Python); a list of results (variance_ratio_per_axis) -> Vector{PlsKitResult}.
# An empty list is a Vector{Int}: only keep_grid can be empty.
function _list(x::Py)
    items = collect(x)
    if !isempty(items) && all(i -> pytruth(_dataclasses.is_dataclass(i)), items)
        return PlsKitResult[_result(i) for i in items]
    elseif !isempty(items) && all(i -> pyisinstance(i, pybuiltins.bool), items)
        return Bool[pyconvert(Bool, i) for i in items]
    end
    return Int[pyconvert(Int, i) for i in items]
end

function _result(x::Py)
    T = Symbol(pyconvert(String, pytype(x).__name__))
    names = get(_RESULT_FIELDS, T) do
        [Symbol(pyconvert(String, f.name)) for f in _dataclasses.fields(x)]
    end
    vals = Tuple(_to_julia(pygetattr(x, String(n)), n) for n in names)
    return PlsKitResult{T}(NamedTuple{Tuple(names)}(vals), x)
end

# ---- display (plumbing only: no statistical methods) --------------------

# `io` is the caller's IOContext (`:compact`, `:limit`, ...); arrays and
# dicts always get their short `summary` regardless, but anything else
# (strings, numbers, NamedTuples, nested results in the generic fallback)
# is shown through that context, so a caller asking for compact or limited
# output gets it.
_brief(io::IO, v::AbstractArray) = summary(v)
_brief(io::IO, v::AbstractDict) = summary(v)
_brief(io::IO, v::PlsKitResult{T}) where {T} = string(T)
_brief(io::IO, v) = sprint(show, v; context=io)

function Base.show(io::IO, r::PlsKitResult{T}) where {T}
    print(io, T, "(")
    pairs_str = (string(k, "=", _brief(io, v)) for (k, v) in pairs(getfield(r, :fields)))
    join(io, pairs_str, ", ")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::PlsKitResult{T}) where {T}
    fields = getfield(r, :fields)
    print(io, T)
    width = maximum(length ∘ string, keys(fields); init=0)
    for (k, v) in pairs(fields)
        print(io, "\n  ", rpad(string(k), width), "  ", _brief(io, v))
    end
end
