# __init__ version check and the three version numbers that must agree
# (spec §5.6, §10).
@test PLSKit._check_version(PLSKit.PLSKIT_PY_VERSION) === nothing
@test_throws ErrorException PLSKit._check_version("0.0.0")
@test string(pkgversion(PLSKit)) == PLSKit.PLSKIT_PY_VERSION
condapkg = read(joinpath(@__DIR__, "..", "CondaPkg.toml"), String)
@test match(r"plskit\s*=\s*\"==([^\"]+)\"", condapkg).captures[1] == PLSKit.PLSKIT_PY_VERSION
@test pyconvert(String, pk.__version__) == PLSKit.PLSKIT_PY_VERSION
