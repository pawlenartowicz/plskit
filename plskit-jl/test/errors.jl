# Every PlsKitError code maps to PlsKitError(code::Symbol, msg, details).
# Codes the public API can raise are triggered for real;
# the rest (reserved or not reachable on demand) go through the same
# translation from a Python PlsKitError built by hand.

# The code set of `_docs/python/api.md` ("All codes"), which equals the
# plskit-bind registry's error_codes.
const ALL_CODES = [
    "dimension_mismatch", "k_exceeds_max", "non_finite_input", "convergence_failure",
    "invalid_argument", "internal", "rotation_method_not_implemented", "invalid_args",
    "invalid_input", "shape_mismatch", "already_rotated", "invalid_weights",
    "resampling_degenerate", "resample_failure_rate_exceeded", "perm_null_degenerate",
    "optimal_no_component", "sequence_no_rejection",
]

@testset "translation of every code: $(c)" for c in ALL_CODES
    v = if c == "invalid_weights"
        pk.PlsKitInvalidWeights("bad weights"; reason="all_zero")
    elseif c == "resampling_degenerate"
        pk.PlsKitResamplingDegenerate("too many skips"; skipped=3, total=10,
                                      skip_rate=0.3, threshold=0.01)
    else
        pk.PlsKitError("boom"; code=c)
    end
    e = PLSKit._plskit_error(v)
    @test e isa PlsKitError
    @test e.code === Symbol(c)
    @test e.msg == pyconvert(String, pystr(v))
    if c == "invalid_weights"
        @test e.details === (reason=:all_zero,)
    elseif c == "resampling_degenerate"
        @test e.details === (skipped=3, total=10, skip_rate=0.3, threshold=0.01)
    else
        @test e.details === NamedTuple()
    end
end

@testset "raised through the public API" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k=2)
    Xnan = copy(X); Xnan[1, 1] = NaN
    yconst = fill(3.0, 60)
    cases = [
        :dimension_mismatch => () -> pls1_fit(X, y[1:59]),
        :k_exceeds_max => () -> pls1_fit(X, y; k=20),
        :non_finite_input => () -> pls1_fit(Xnan, y),
        :invalid_argument => () -> pls1_fit(X, y; k="optimal"),
        :invalid_args => () -> pls1_confirmatory_test(X, y; test_method="raw_perm", args=(bogus=1,)),
        :invalid_input => () -> rotate(zeros(10, 0)),
        :shape_mismatch => () -> rotate(randn_np(1, 10, 3); L=randn_np(2, 40, 2)),
        :rotation_method_not_implemented => () -> rotate(fit; method="promax"),
        :already_rotated => () -> rotate(rotate(fit)),
        :invalid_weights => () -> pls1_fit(X, y; weights=-ones(60)),
        :optimal_no_component => () -> pls1_fit(X, yconst; k="optimal", k_max=3, seed=3),
        :sequence_no_rejection => () -> pls1_fit(X, yconst; k="sequence", k_max=3, seed=3),
    ]
    @testset "$(code)" for (code, f) in cases
        @test plskit_error(f).code === code
    end
    e = plskit_error(() -> pls1_fit(X, y; weights=-ones(60)))
    @test e.details === (reason=:negative,)
    @test occursin("PlsKitError(:invalid_weights)", sprint(showerror, e))
end

@testset "resampling_degenerate carries its four fields" begin
    X = randn_np(13, 60, 5)
    y = X[:, 1] .+ 0.2 .* randn_np(14, 60)
    w = fill(1e-8, 60); w[1:5] .= 1.0
    e = plskit_error(() -> pls1_rotation_stability(X, y, 3; weights=w, n_boot=500, seed=0))
    @test e.code === :resampling_degenerate
    @test keys(e.details) == (:skipped, :total, :skip_rate, :threshold)
    @test e.details.total == 500 && e.details.threshold == 0.01
    @test e.details.skipped > 0 && e.details.skip_rate > 0.01
end

@testset "unknown method string" begin
    X, y = pls1_data()
    e = plskit_error(() -> pls1_confirmatory_test(X, y; test_method="bogus"))
    @test occursin("bogus", e.msg)
    @test e.code === :invalid_args
end

@testset "other Python exceptions propagate unchanged" begin
    # Every plskit function raises PlsKitError for a bad value, so drive the
    # one call path with a keyword Python itself refuses (a stub never
    # passes one): its TypeError must come through as a PyException.
    X, y = pls1_data()
    e = try
        PLSKit._call(:pls1_fit; X=X, y=y, ncomp=2)
    catch err
        err
    end
    @test e isa PyException
    @test pyisinstance(e.v, pybuiltins.TypeError)
end

@testset "a result of the wrong type as a model is an error" begin
    X, y = pls1_data()
    m3 = pls3_fit(X, hcat(y, randn_np(9, 60)); k=1)
    e = try
        pls1_predict(m3, X)
        nothing
    catch err
        err
    end
    @test e isa PlsKitError && e.code === :invalid_argument
    @test e.msg == "model must be a PLS1Result, got a PLS3Result"
    e = plskit_error(() -> rotate(m3))
    @test e.code === :invalid_argument
    @test e.msg == "rotate() first arg must be a PLS1Result, got a PLS3Result"
    @test plskit_error(() -> rotate("not a matrix")).code === :invalid_argument
end

@testset "a missing required keyword is Julia's own error" begin
    X, y = pls1_data()
    @test_throws UndefKeywordError pls1_confirmatory_test(X, y)
end
