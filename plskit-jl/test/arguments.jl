# Julia -> Python argument conversion.

@testset "_to_py" begin
    X, _ = pls1_data()
    a = PLSKit._to_py(X)
    @test pyisinstance(a, np.ndarray)
    @test pyconvert(Tuple, a.shape) == (60, 8)
    X[1, 2] = 99.0                                  # a view: no copy on the Julia side
    @test pyconvert(Float64, a[0, 1]) == 99.0
    t = PLSKit._to_py(permutedims(X)')
    @test pyisinstance(t, np.ndarray) && pyconvert(Float64, t[0, 1]) == 99.0
    @test pyisinstance(PLSKit._to_py(view(X, :, 1)), np.ndarray)
    m = PLSKit._to_py(Union{Missing,Float64}[1.0, missing])
    @test pyconvert(Vector{Float64}, m)[1] == 1.0 && isnan(pyconvert(Vector{Float64}, m)[2])
    # a Matrix{Any} holding `missing` (a DataFrame column mix, say) takes
    # the same missing -> NaN path
    ma = Matrix{Any}(undef, 2, 2)
    ma[1, 1] = 1.0; ma[1, 2] = 2; ma[2, 1] = missing; ma[2, 2] = 4.0
    ca = PLSKit._to_py(ma)
    @test pyisinstance(ca, np.ndarray)
    v = pyconvert(Matrix{Float64}, ca)
    @test v[1, 1] == 1.0 && isnan(v[2, 1])
    va = Any[1.0, 2, missing]
    cv = PLSKit._to_py(va)
    @test isnan(pyconvert(Vector{Float64}, cv)[3])
    @test PLSKit._to_py(nothing) === pybuiltins.None
    @test pyconvert(String, PLSKit._to_py(:raw_perm)) == "raw_perm"
    d = PLSKit._to_py((n_perm=100, inner=(selector=:bic,)))
    @test pyisinstance(d, pybuiltins.dict)
    @test pyconvert(Int, d["n_perm"]) == 100
    @test pyconvert(String, d["inner"]["selector"]) == "bic"
    @test pyconvert(Int, PLSKit._to_py(Dict(:n_perm => 7))["n_perm"]) == 7
    @test pyconvert(Int, PLSKit._to_py(Dict("n_perm" => 7))["n_perm"]) == 7
    @test pyconvert(UInt64, PLSKit._to_py(typemax(UInt64))) === typemax(UInt64)
    @test_throws ArgumentError PLSKit._to_py(Dict(1 => 2))
end

@testset "_check_main_thread guards PythonCall against non-main threads" begin
    # PythonCall segfaults the process when used off the main thread, so
    # this is checked directly against the small guard function rather than
    # by crashing the test process.
    @test PLSKit._check_main_thread(1) === nothing
    @test_throws ErrorException PLSKit._check_main_thread(2)
    err = try
        PLSKit._check_main_thread(2)
        nothing
    catch e
        e
    end
    @test occursin("main thread", err.msg)
    if Threads.nthreads() > 1
        # A spawned task usually lands off the main thread; when it happens
        # to land back on thread 1 the guard correctly lets it through, so
        # that outcome is not a test failure either.
        t = Threads.@spawn begin
            PLSKit._check_main_thread()
            :ran_on_main_thread
        end
        result = try
            fetch(t)
        catch e
            e
        end
        @test result === :ran_on_main_thread || result isa TaskFailedException ||
              result isa ErrorException
    else
        @test_skip "Threads.nthreads() == 1: the off-main-thread path is untested here"
    end
end

@testset "a stub call returns a result and hands it back" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k=2)
    @test fit isa PLS1Result
    @test PLSKit._to_py(fit) === getfield(fit, :py)
    @test plskit_error(() -> pls1_fit(X, y[1:59])).code === :dimension_mismatch
end

@testset "arguments through the public API" begin
    X, y = pls1_data()
    base = pls1_confirmatory_test(X, y; test_method="raw_perm", args=(n_perm=100,), seed=5)
    # args as NamedTuple, Dict{String}, Dict{Symbol}; test_method as Symbol
    for args in ((n_perm=100,), Dict("n_perm" => 100), Dict(:n_perm => 100))
        r = pls1_confirmatory_test(X, y; test_method=:raw_perm, args=args, seed=5)
        @test r.pvalue == base.pvalue && r.statistic == base.statistic
    end
    # nested args inside find_k_args, Symbol values inside args
    opt = pls1_fit(X, y; k="optimal", k_max=3, seed=2,
                   find_k_args=(selector=:r2_max, args=(n_folds=5,)))
    @test opt.selection_result.selector == "r2_max"
    # adjoint, view, integer, Float32, BitVector and whole-float inputs
    fit = pls1_fit(X, y; k=2)
    @test pls1_fit(permutedims(X)', y; k=2).coef == fit.coef
    @test pls1_fit(view(hcat(X, X), :, 1:8), y; k=2).coef == fit.coef
    @test pls1_fit(X, y; k=2.0).coef == fit.coef
    Xi = round.(Int, 10 .* X)
    @test pls1_fit(Xi, y; k=2).coef == pls1_fit(Float64.(Xi), y; k=2).coef
    @test pls1_fit(Float32.(X), y; k=2).k_used == 2
    w = BitVector(isodd.(1:60)) .| BitVector(1:60 .<= 30)
    @test pls1_fit(X, y; k=2, weights=w).coef == pls1_fit(X, y; k=2, weights=Float64.(w)).coef
    # Bool data (BitMatrix, Matrix{Bool}, BitVector y) reads as 1/0, as in
    # Python and R; `missing` in it is NaN; a Bool flag stays a flag
    Xb = X .> 0
    fit01 = pls1_fit(Float64.(Xb), Float64.(y .> 0); k=2)
    @test Xb isa BitMatrix
    @test pls1_fit(Xb, y .> 0; k=2).coef == fit01.coef
    @test pls1_fit(Matrix{Bool}(Xb), Vector{Bool}(y .> 0); k=2).coef == fit01.coef
    @test pls1_predict(fit01, Xb) == pls1_predict(fit01, Float64.(Xb))
    Xbm = Array{Union{Missing,Bool}}(Xb); Xbm[2, 3] = missing
    @test plskit_error(() -> pls1_fit(Xbm, y)).code === :non_finite_input
    @test plskit_error(() -> pls1_fit(X, y; pre_standardized=1)).code === :invalid_argument
    # `missing` reaches the engine as NaN
    Xm = Array{Union{Missing,Float64}}(X); Xm[1, 1] = missing
    @test plskit_error(() -> pls1_fit(Xm, y)).code === :non_finite_input
    # same for a Matrix{Any} column mix
    Xa = Matrix{Any}(X); Xa[1, 1] = missing
    @test plskit_error(() -> pls1_fit(Xa, y)).code === :non_finite_input
end
