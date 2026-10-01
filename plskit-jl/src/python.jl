# Python-side handles, filled in by __init__ (PythonCall's precompile-safe
# pattern: pynew() at load time, pycopy! at runtime).

# The Python plskit version this PLSKit.jl release runs. Equals the
# `version` in Project.toml and the pin in CondaPkg.toml (spec §10).
const PLSKIT_PY_VERSION = "0.6.2"

const _plskit = PythonCall.pynew()
const _np = PythonCall.pynew()
const _warnings = PythonCall.pynew()
const _dataclasses = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(_plskit, pyimport("plskit"))
    PythonCall.pycopy!(_np, pyimport("numpy"))
    PythonCall.pycopy!(_warnings, pyimport("warnings"))
    PythonCall.pycopy!(_dataclasses, pyimport("dataclasses"))
    _check_version(pyconvert(String, _plskit.__version__))
    return nothing
end

function _check_version(found::AbstractString)
    found == PLSKIT_PY_VERSION && return nothing
    error("PLSKit.jl needs the Python package plskit $(PLSKIT_PY_VERSION), " *
          "but the Python it runs has plskit $(found). With your own Python " *
          "(JULIA_CONDAPKG_BACKEND=\"Null\"), install plskit==$(PLSKIT_PY_VERSION) " *
          "into it; otherwise unset JULIA_CONDAPKG_BACKEND and " *
          "JULIA_PYTHONCALL_EXE so CondaPkg installs the pinned version.")
end
