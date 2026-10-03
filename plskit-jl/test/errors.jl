# `_plskit_error` has three branches: a generic Python PlsKitError, the
# `invalid_weights` subclass (its `reason` becomes a Symbol) and the
# `resampling_degenerate` subclass (four typed fields). Each is built by
# hand from the real Python classes; which code the engine raises for which
# input is owned by the engine and Python tests.

@testset "translation of Python errors: $(name)" for (name, v, code, details) in [
    ("generic", pk.PlsKitError("boom"; code="dimension_mismatch"),
     :dimension_mismatch, NamedTuple()),
    ("invalid_weights", pk.PlsKitInvalidWeights("bad weights"; reason="all_zero"),
     :invalid_weights, (reason=:all_zero,)),
    ("resampling_degenerate",
     pk.PlsKitResamplingDegenerate("too many skips"; skipped=3, total=10,
                                   skip_rate=0.3, threshold=0.01),
     :resampling_degenerate, (skipped=3, total=10, skip_rate=0.3, threshold=0.01)),
]
    e = PLSKit._plskit_error(v)
    @test e isa PlsKitError
    @test e.code === code
    @test e.msg == pyconvert(String, pystr(v))
    @test e.details === details
end

@testset "raised through the public API" begin
    X, y = pls1_data()
    @test plskit_error(() -> pls1_fit(X, y[1:59])).code === :dimension_mismatch
    e = plskit_error(() -> pls1_fit(X, y; weights=-ones(60)))
    @test e.code === :invalid_weights
    @test e.details === (reason=:negative,)
    @test occursin("PlsKitError(:invalid_weights)", sprint(showerror, e))
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
