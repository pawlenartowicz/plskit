# Python -> Julia conversion by runtime type.

# A probe type whose `show` method reads the IOContext it is called with,
# so a test can assert that `_brief` forwards `io` rather than dropping it.
struct _IOContextProbe
    v::Int
end
Base.show(io::IO, p::_IOContextProbe) = print(io, get(io, :probe, "no"))

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
end

@testset "scalars, seeds, maps, lists" begin
    @test PLSKit._to_julia(pybuiltins.None) === nothing
    @test PLSKit._to_julia(Py(true)) === true
    @test PLSKit._to_julia(Py(3)) === 3
    @test PLSKit._to_julia(Py(typemax(UInt64)), :seed) === typemax(UInt64)
    @test PLSKit._to_julia(Py(42), :seed) === UInt64(42)
    @test PLSKit._to_julia(Py("r2_se")) == "r2_se"
    d = PLSKit._to_julia(pyeval("{1: 0.5, 3: 0.25}", Main))
    @test d isa Dict{Int,Float64} && d == Dict(1 => 0.5, 3 => 0.25)
    @test PLSKit._to_julia(pydict()) == Dict{Int,Float64}()
    nt = PLSKit._to_julia(pyeval(
        "__import__('types').MappingProxyType({'max_iter': 50, 'tol': 1e-8})", Main))
    @test nt === (max_iter=50, tol=1e-8)
    @test PLSKit._to_julia(pylist([1, 2, 4])) == [1, 2, 4]
    @test PLSKit._to_julia(pylist()) == Int[]
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
    # field order of `ConfirmatoryTestResult` in _docs/python/results.md, shared by plskit-bind and the Python dataclass
    ct = PLSKit._to_julia(pk.pls1_confirmatory_test(np.asarray(X), np.asarray(y);
                                                    test_method="score", seed=1))
    @test collect(propertynames(ct))[8:10] == [:n_eff, :rho_hat, :stable_rank]
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
    seq = PLSKit._to_julia(pk.FindKSequenceResult(k_star=1, pvalues=np.array([0.01]),
                                                  test_method="split_nb", alpha=0.05,
                                                  seed=42))
    @test seq.seed === UInt64(42)
    @test occursin("seed         0x000000000000002a", sprint(show, MIME"text/plain"(), seq))
end

# Edge cases of the conversion: NumPy scalars, unsigned overflow, bool lists,
# overflow error messages, F-order copies without aliasing, IOContext.

@testset "NumPy scalars unwrap through .item() before the type ladder" begin
    @test PLSKit._to_julia(np.array([true])[0]) === true
    @test PLSKit._to_julia(np.int64(7), :seed) === UInt64(7)
    @test PLSKit._to_julia(np.float64(2.5)) isa Float64
    @test PLSKit._to_julia(np.float64(2.5)) === 2.5
end

@testset "an out-of-range unsigned array raises, naming the field" begin
    big = np.array([typemax(UInt64)], dtype="uint64")
    err = try
        PLSKit._to_julia(big, :big_uints)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("big_uints", err.msg)
end

@testset "a Python list of bools becomes Vector{Bool}" begin
    v = PLSKit._to_julia(pylist([true, false, true]))
    @test v isa Vector{Bool}
    @test v == [true, false, true]
end

@testset "a non-seed int overflow names the field" begin
    huge = pyeval("2**63", Main)
    err = try
        PLSKit._to_julia(huge, :count)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("count", err.msg)
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

@testset "_brief forwards the caller's IOContext to the generic fallback" begin
    io = IOBuffer()
    ioctx = IOContext(io, :probe => "yes")
    @test PLSKit._brief(ioctx, _IOContextProbe(1)) == "yes"
    @test PLSKit._brief(devnull, _IOContextProbe(1)) == "no"
end
