# __init__ version check.
@test PLSKit._check_version(PLSKit.PLSKIT_PY_VERSION) === nothing
@test_throws ErrorException PLSKit._check_version("0.0.0")
