# Python -> Julia conversion by runtime type.

@testset "ndarrays keep orientation, dtype and ndim" begin
    m = PLSKit._to_julia(np.arange(6.0).reshape(2, 3))
    @test m isa Matrix{Float64}
    @test m == [0.0 1.0 2.0; 3.0 4.0 5.0]
    f = PLSKit._to_julia(np.asfortranarray(np.arange(6.0).reshape(2, 3)))
    @test f == [0.0 1.0 2.0; 3.0 4.0 5.0]
    @test PLSKit._to_julia(np.array([true, false])) == [true, false]
    @test PLSKit._to_julia(np.array([true, false])) isa Vector{Bool}
    @test PLSKit._to_julia(np.array([3, 4], dtype=np.int32)) isa Vector{Int}
    @test PLSKit._to_julia(np.zeros((5, 0))) == zeros(5, 0)
    @test PLSKit._to_julia(np.float64(2.5)) === 2.5
    @test PLSKit._to_julia(np.array(7)) === 7
    @test PLSKit._to_julia(np.array([true])[0]) === true
end

@testset "scalars, seeds, maps, lists" begin
    @test PLSKit._to_julia(pybuiltins.None) === nothing
    @test PLSKit._to_julia(Py(true)) === true
    @test PLSKit._to_julia(Py(3)) === 3
    @test PLSKit._to_julia(Py(typemax(UInt64)), :seed) === typemax(UInt64)
    @test PLSKit._to_julia(Py(42), :seed) === UInt64(42)
    @test PLSKit._to_julia(np.int64(7), :seed) === UInt64(7)
    @test PLSKit._to_julia(Py("r2_se")) == "r2_se"
    d = PLSKit._to_julia(pyeval("{1: 0.5, 3: 0.25}", Main))
    @test d isa Dict{Int,Float64} && d == Dict(1 => 0.5, 3 => 0.25)
    @test PLSKit._to_julia(pydict()) == Dict{Int,Float64}()
    nt = PLSKit._to_julia(pyeval(
        "__import__('types').MappingProxyType({'max_iter': 50, 'tol': 1e-8})", Main))
    @test nt === (max_iter=50, tol=1e-8)
    @test PLSKit._to_julia(pylist([1, 2, 4])) == [1, 2, 4]
    @test PLSKit._to_julia(pylist()) == Int[]
    v = PLSKit._to_julia(pylist([true, false, true]))
    @test v isa Vector{Bool} && v == [true, false, true]
end

@testset "a Python result converts by its dataclass fields" begin
    X, y = pls1_data()
    pyfit = pk.pls1_fit(np.asarray(X), np.asarray(y); k=2)
    fit = PLSKit._to_julia(pyfit)
    @test fit isa PlsKitResult{:PLS1Result}
    @test propertynames(fit) == Tuple(PLSKit._RESULT_FIELDS[:PLS1Result])
    @test fit.W isa Matrix{Float64} && size(fit.W) == (8, 2)
    @test pytruth(np.array_equal(np.asarray(fit.W), pyfit.W))
    @test fit.k_used === 2 && fit.weights === nothing
    @test getfield(fit, :py) === pyfit
end

@testset "show" begin
    X, y = pls1_data()
    fit = PLSKit._to_julia(pk.pls1_fit(np.asarray(X), np.asarray(y); k=2))
    txt = sprint(show, MIME"text/plain"(), fit)
    @test startswith(txt, "PLS1Result\n")
    @test occursin("60×2 Matrix{Float64}", txt)
    @test occursin("\n  k_used            2", txt)
    @test startswith(sprint(show, fit), "PLS1Result(T=60×2 Matrix{Float64}, P=")
    ci = PLSKit._to_julia(pk.CIScalar(point=0.5, lower=0.25, upper=0.75, sd=0.1))
    @test sprint(show, ci) == "CIScalar(point=0.5, lower=0.25, upper=0.75, sd=0.1)"
    # scalar fields are shown through the caller's IOContext
    third = PLSKit._to_julia(pk.CIScalar(point=1 / 3, lower=0.25, upper=0.75, sd=0.1))
    @test sprint(show, third; context=:compact => true) ==
          "CIScalar(point=0.333333, lower=0.25, upper=0.75, sd=0.1)"
    seq = PLSKit._to_julia(pk.FindKSequenceResult(k_star=1, pvalues=np.array([0.01]),
                                                  test_method="split_nb", alpha=0.05,
                                                  seed=42))
    @test seq.seed === UInt64(42)
    @test occursin("seed         0x000000000000002a", sprint(show, MIME"text/plain"(), seq))
end

@testset "array conversion does not alias NumPy's memory" begin
    a = np.arange(6.0).reshape(2, 3)
    m = PLSKit._to_julia(a)
    a[0, 0] = 99.0
    @test m[1, 1] == 0.0
    f = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    mf = PLSKit._to_julia(f)
    f[1, 1] = -1.0
    @test mf[2, 2] == 4.0
end

@testset "a typo'd field name names the result type and the field" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k=2)
    err = try
        fit.coeff
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("PLS1Result", err.msg) && occursin("coeff", err.msg)
    # unaffected: real fields, and the escape hatches to the raw Python
    # object and the underlying fields NamedTuple
    @test fit.coef isa Vector{Float64}
    @test getfield(fit, :py) isa Py
    @test getfield(fit, :fields) isa NamedTuple
end
