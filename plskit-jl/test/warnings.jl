# Python warnings raised during a call are re-emitted with @warn, message
# verbatim, _group=:plskit (spec §5.5).

const REROUTE = r"^'split_nb' was rerouted to 'split_exact'"

@testset "reroute notice is re-emitted" begin
    X, y = gated_data()
    r = @test_logs (:warn, REROUTE, PLSKit, :plskit) pls1_confirmatory_test(
        X, y; method="split_nb", args=(n_splits=10,), seed=1)
    @test r.method == "split_exact"
end

@testset "every call re-emits (the \"always\" filter)" begin
    X, y = gated_data()
    call() = pls1_confirmatory_test(X, y; method="split_nb", args=(n_splits=10,), seed=1)
    @test_logs (:warn, REROUTE) (:warn, REROUTE) (call(); call())
end

@testset "message equals Python's" begin
    X, y = gated_data()
    logs, _ = Test.collect_test_logs() do
        pls1_confirmatory_test(X, y; method="split_nb", args=(n_splits=10,), seed=1)
    end
    w = pyimport("warnings")
    ctx = w.catch_warnings(record=true)
    rec = ctx.__enter__()
    w.simplefilter("always")
    pk.pls1_confirmatory_test(np.asarray(X), np.asarray(y); method="split_nb",
                              args=pydict(Dict("n_splits" => 10)), seed=1)
    ctx.__exit__(nothing, nothing, nothing)
    @test only(logs).message == pyconvert(String, pystr(rec[0].message))
end

@testset "no warning, no log; Python's filters are restored" begin
    X, y = pls1_data()
    before = pylen(pyimport("warnings").filters)
    @test_logs pls1_fit(X, y; k=2)
    @test pylen(pyimport("warnings").filters) == before
end

@testset "a failing call restores Python's filters too" begin
    X, y = pls1_data()
    before = pylen(pyimport("warnings").filters)
    @test plskit_error(() -> pls1_fit(X, y[1:59])).code === :dimension_mismatch
    @test pylen(pyimport("warnings").filters) == before
end
