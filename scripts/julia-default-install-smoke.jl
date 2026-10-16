# Installs PLSKit the way a user does (fresh environment, CondaPkg left on,
# no JULIA_PYTHONCALL_EXE) and fits once.
#
#   julia scripts/julia-default-install-smoke.jl path <plskit-jl dir>
#   julia scripts/julia-default-install-smoke.jl rev <git tag or sha>
using Pkg

const USAGE = "usage: julia-default-install-smoke.jl (path <plskit-jl dir> | rev <tag>)"
length(ARGS) == 2 || error(USAGE)
for k in keys(ENV)
    if startswith(k, "JULIA_CONDAPKG_") || startswith(k, "JULIA_PYTHONCALL_")
        error("$k is set; the smoke test needs the default install")
    end
end

Pkg.activate(; temp=true)
if ARGS[1] == "path"
    Pkg.develop(path=abspath(ARGS[2]))
elseif ARGS[1] == "rev"
    Pkg.add(url="https://github.com/pawlenartowicz/plskit", subdir="plskit-jl", rev=ARGS[2])
else
    error(USAGE)
end

using PLSKit
const PC = PLSKit.PythonCall  # not a direct dependency of the temporary environment
X = randn(40, 5); y = randn(40)
pls1_fit(X, y; k=2).k_used == 2 || error("k_used != 2")

println("julia ", VERSION)
println("python ", PC.pyconvert(String, PC.pyimport("sys").version))
println("plskit ", PC.pyconvert(String, PC.pyimport("plskit").__version__))
