"""
    PLSKit

Partial Least Squares with modern inference, from Julia. PLSKit runs the
Python `plskit` package through PythonCall.jl: every function, argument
name and result field is the Python one (see `_docs/julia/`).
"""
module PLSKit

using PythonCall

export PlsKitResult, PlsKitError

include("python.jl")
include("results.jl")
include("errors.jl")
include("call.jl")
include("generated.jl")

end # module PLSKit
