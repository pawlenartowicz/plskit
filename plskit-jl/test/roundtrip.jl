# A result passed back as an argument hands Python its own object;
# seeds reproduce runs exactly, including seeds above 2^63.

@testset "pls1_predict and rotate take the result back" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k=2)
    ŷ = pls1_predict(fit, X)
    @test ŷ isa Vector{Float64} && length(ŷ) == 60
    @test pytruth(np.array_equal(np.asarray(ŷ), pk.pls1_predict(getfield(fit, :py), np.asarray(X))))
    # fields are copies: editing one does not change what Python receives
    fit.W .= 0.0
    @test pls1_predict(fit, X) == ŷ
    rot = rotate(fit)
    @test rot isa PLS1Result
    @test rot.rotation_spec isa RotationSpec
    @test rot.rotation_spec.args === (max_iter=50, tol=1e-8, kaiser_normalize=true)
    W = pls1_fit(X, y; k=2).W
    rw = rotate(W; method=:varimax)
    @test rw isa RotateResult
    @test rw.W_rot == rot.W
end

@testset "pls3_transform takes the model back" begin
    X, y = pls1_data()
    Y = hcat(y, randn_np(9, 60), randn_np(10, 60))
    m = pls3_fit(X, Y; k=2)
    sc = pls3_transform(m; X_new=X, Y_new=Y)
    @test sc isa PLS3Scores
    @test pls3_transform(m; X_new=X, which=:x_scores).y_scores === nothing
    @test plssvd_transform(m; Y_new=Y, which="y_scores").x_scores === nothing
end

@testset "zero-component model round-trips" begin
    X, _ = pls1_data()
    z = pls1_fit(X, fill(3.0, 60); k=2)
    @test z.k_used == 0 && size(z.T) == (60, 0)
    @test length(pls1_predict(z, X)) == 60
end

@testset "seed: drawn, passed back, reproduced" begin
    X, y = pls1_data()
    a = pls1_confirmatory_test(X, y; test_method="raw_perm", args=(n_perm=100,))
    @test a.seed isa UInt64
    b = pls1_confirmatory_test(X, y; test_method="raw_perm", args=(n_perm=100,), seed=a.seed)
    @test b.seed === a.seed
    @test b.pvalue == a.pvalue && b.statistic == a.statistic
end

@testset "seed above 2^63: $(s)" for s in (typemax(UInt64), UInt64(2)^63 + 5)
    X, y = pls1_data()
    a = pls1_confirmatory_test(X, y; test_method="raw_perm", args=(n_perm=100,), seed=s)
    @test a.seed === s
    b = pls1_confirmatory_test(X, y; test_method="raw_perm", args=(n_perm=100,), seed=a.seed)
    @test b.pvalue == a.pvalue && b.statistic == a.statistic
end

@testset "seed in a nested result" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k="sequence", k_max=3, seed=typemax(UInt64),
                   find_k_args=(args=(n_splits=10,),))
    @test fit.selection_result isa FindKSequenceResult
    @test fit.selection_result.seed === typemax(UInt64)
end
