"""
    PlsKitError(code, msg, details)

Raised by every plskit function. `code` is Python's `PlsKitError.code` as
a `Symbol` (`:invalid_argument`, `:invalid_weights`, ...). `details`
carries `(reason=...,)` for `:invalid_weights` and
`(skipped=..., total=..., skip_rate=..., threshold=...)` for
`:resampling_degenerate`; it is empty otherwise. Dispatch on `e.code`.
Any other Python exception reaches you unchanged as a `PyException`.
"""
struct PlsKitError <: Exception
    code::Symbol
    msg::String
    details::NamedTuple
end

Base.showerror(io::IO, e::PlsKitError) = print(io, "PlsKitError(:", e.code, "): ", e.msg)

# Build a PlsKitError from a Python plskit.PlsKitError instance.
function _plskit_error(v::Py)
    code = pyconvert(String, v.code)
    msg = pyconvert(String, pystr(v))
    details = if code == "invalid_weights"
        (reason=Symbol(pyconvert(String, v.reason)),)
    elseif code == "resampling_degenerate"
        (skipped=pyconvert(Int, v.skipped), total=pyconvert(Int, v.total),
         skip_rate=pyconvert(Float64, v.skip_rate), threshold=pyconvert(Float64, v.threshold))
    else
        NamedTuple()
    end
    return PlsKitError(Symbol(code), msg, details)
end
