# A result passed back as an argument hands Python its own object;
# seeds reproduce runs exactly, including seeds above 2^63.

@testset "pls1_predict takes the result back" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k=2)
    ŷ = pls1_predict(fit, X)
    @test ŷ isa Vector{Float64} && length(ŷ) == 60
    @test pytruth(np.array_equal(np.asarray(ŷ), pk.pls1_predict(getfield(fit, :py), np.asarray(X))))
    # fields are copies: editing one does not change what Python receives
    fit.W .= 0.0
    @test pls1_predict(fit, X) == ŷ
    rw = rotate(fit.W; method=:varimax)
    @test rw isa RotateResult
    @test rw.spec.args === (max_iter=50, tol=1e-8, kaiser_normalize=true)
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

@testset "PLSKit.subset takes 1-based rows" begin
    X, _ = pls1_data()
    g = gram_matrix(X; pre_standardized=true)
    @test g isa GramMatrixResult
    rows = [8, 1, 4, 4, 60]
    s = PLSKit.subset(g, rows)
    @test s isa GramMatrixResult
    @test s.gram == g.gram[rows, rows]
    @test (s.n_features, s.pre_standardized) == (g.n_features, true)
    for bad in ([0, 2], [1, 61], [3])
        e = try
            PLSKit.subset(g, bad)
        catch err
            err
        end
        @test e isa PlsKitError && e.code === :invalid_argument
    end
    for (bad, msg) in (([1, 61], "rows must index the 60 rows of the Gram matrix"),
                       (UInt[typemax(UInt), 1], "rows must index the 60 rows of the Gram matrix"),
                       ([0, 2], "rows must be row numbers, at least 1"),
                       (UInt[0, 2], "rows must be row numbers, at least 1"))
        e = try PLSKit.subset(g, bad); nothing catch err; err end
        @test e isa PlsKitError && e.code === :invalid_argument && e.msg == msg
    end
    # a Bool mask is not a list of row numbers
    @test_throws MethodError PLSKit.subset(g, [true, true])
end

@testset "gram= takes the GramMatrixResult and a subset of it" begin
    X, y = pls1_data(5; n=30, d=80)
    kw = (k=1, test_method="split_exact", args=(n_perm=19, n_splits=8), seed=3)
    a = pls1_confirmatory_test(X, y; kw...)
    @test pls1_confirmatory_test(; X, y, kw...).statistic == a.statistic
    b = pls1_confirmatory_test(; gram=gram_matrix(X), y, kw...)
    @test b.statistic ≈ a.statistic rtol = 1e-12
    @test b.pvalue == a.pvalue
    s = PLSKit.subset(gram_matrix(X), collect(1:25))
    r = pls1_perm_alpha(s, reshape(y[1:25], :, 1); k=2)
    @test size(r.alpha) == (25, 1)
end

@testset "gram_matrix_blocks sums the block Gram matrices" begin
    X, y = pls1_data(5; n=30, d=80)
    cuts = [0, 5, 13, size(X, 2)]
    blocks() = (X[:, a+1:b] for (a, b) in zip(cuts, cuts[2:end]))
    whole = gram_matrix(X)
    g = gram_matrix_blocks(blocks())
    @test g isa GramMatrixResult
    @test g.n_features == whole.n_features
    @test isapprox(g.gram, whole.gram; atol=1e-12 * maximum(abs, whole.gram))
    gs = gram_matrix_blocks(blocks(); pre_standardized=true)
    @test gs.pre_standardized
    @test isapprox(gs.gram, gram_matrix(X; pre_standardized=true).gram;
                   atol=1e-12 * maximum(abs, whole.gram))
    one = gram_matrix_blocks([X])
    @test one.gram == whole.gram && one.n_features == whole.n_features
    kw = (k=1, test_method="split_exact", args=(n_perm=19, n_splits=8), seed=1)
    @test pls1_confirmatory_test(; gram=g, y, kw...).pvalue ==
          pls1_confirmatory_test(; gram=whole, y, kw...).pvalue
    e = try gram_matrix_blocks(Matrix{Float64}[]); nothing catch err; err end
    @test e isa PlsKitError && e.code == :invalid_input && e.msg == "blocks holds no block"
    e = try gram_matrix_blocks([X, X[1:20, :]]); nothing catch err; err end
    @test e isa PlsKitError && e.code == :shape_mismatch
    e = try gram_matrix_blocks([X[1:4, :], X[1:3, :]]); nothing catch err; err end
    @test e isa PlsKitError && e.code == :shape_mismatch &&
          e.msg == "block has 3 rows, the first block has 4"
end

@testset "k takes a set" begin
    X, y = pls1_data()
    r = pls1_confirmatory_test(X, y; k=1:3, test_method="split_exact",
                               args=(n_perm=19, n_splits=5), seed=1)
    @test r.k isa Vector{Int} && r.k == [1, 2, 3]
    @test r.pvalues_by_k isa Dict{Int,Float64}
    @test sort(collect(keys(r.pvalues_by_k))) == [1, 2, 3]
    @test r.statistic === nothing
end

@testset "seed: drawn, passed back, reproduced" begin
    X, y = pls1_data()
    a = pls1_confirmatory_test(X, y; k=1, test_method="raw_perm", args=(n_perm=100,))
    @test a.seed isa UInt64
    b = pls1_confirmatory_test(X, y; k=1, test_method="raw_perm", args=(n_perm=100,), seed=a.seed)
    @test b.seed === a.seed
    @test b.pvalue == a.pvalue && b.statistic == a.statistic
end

@testset "seed above 2^63: $(s)" for s in (typemax(UInt64), UInt64(2)^63 + 5)
    X, y = pls1_data()
    a = pls1_confirmatory_test(X, y; k=1, test_method="raw_perm", args=(n_perm=100,), seed=s)
    @test a.seed === s
    b = pls1_confirmatory_test(X, y; k=1, test_method="raw_perm", args=(n_perm=100,), seed=a.seed)
    @test b.pvalue == a.pvalue && b.statistic == a.statistic
end

@testset "seed in a nested result" begin
    X, y = pls1_data()
    fit = pls1_fit(X, y; k="sequence", k_max=3, seed=typemax(UInt64),
                   find_k_args=(args=(n_splits=10,),))
    @test fit.selection_result isa FindKSequenceResult
    @test fit.selection_result.seed === typemax(UInt64)
end

@testset "factors and profiles results take rotate" begin
    X, y = pls1_data()
    m = pls1_fit(X, y; k=2)
    coef = m.coef
    f = factors(coef, X, 3)
    @test f isa FactorsResult
    @test vec(sum(f.F; dims=2)) ≈ f.coef atol = 1e-10
    @test factors(m, X, 3).F == f.F
    r = rotate(f; X=X)
    @test r isa FactorsResult
    @test vec(sum(r.F; dims=2)) ≈ r.coef atol = 1e-10
    Y = hcat(y, randn_np(9, 60))
    m3 = pls3_fit(X, Y; k=2)
    r3 = rotate(m3; X=X)
    @test r3 isa PLS3Result
    @test r3.rotation_spec !== nothing
    s3 = spls3_fit(X, Y, 2, 3, 1)
    logs, rs = Test.collect_test_logs() do
        rotate(s3; X=X)
    end
    @test occursin("spreads its zeros", only(logs).message)
    @test rs.converged isa Vector{Bool}
    @test rs.n_iter isa Vector{<:Integer}
    Xc = X .- sum(X; dims=1) ./ size(X, 1)
    Xs = Xc ./ sqrt.(sum(abs2, Xc; dims=1) ./ (size(X, 1) - 1))
    ps = profiles(coef, Xs, 3; pre_standardized=true)
    @test ps isa ProfilesResult
    @test ps.scores ≈ Xs * ps.factors.F atol = 1e-10
    p = profiles(coef, X, 2)
    @test p isa ProfilesResult
    @test p.factors isa FactorsResult
    m1 = pls1_fit(X, y; k=1)
    @test profiles(m1, X, 2).model isa PLS1Result
    @test profiles_search(m1, X, y; seed=1) isa ProfilesSearchResult

    # pre_standardized overrides what the fit recorded: scores are X F, not Xs F
    Xu = X .* reshape(1.0:size(X, 2), 1, :)
    mu = pls1_fit(Xu, y; k=2)
    po = profiles(mu, Xu, 2; pre_standardized=true)
    @test po.scores ≈ Xu * po.factors.F atol = 1e-8
    @test po.scores != profiles(mu, Xu, 2).scores
    # refused inputs
    @test plskit_error(() -> rotate(m1)).code === :invalid_argument
    @test plskit_error(() -> rotate(f; X=X, L=f.W)).code === :invalid_argument
    @test plskit_error(() -> rotate(f; pre_standardized=true)).code === :invalid_argument
    @test plskit_error(() -> factors(m3, X, 2)).code === :invalid_argument
end

@testset "failed split counts reach the result" begin
    # One nonzero row leaves one half of every split with a constant outcome.
    X = randn(30, 4)
    y = [1.0; zeros(29)]
    r = pls1_confirmatory_test(X, y; k=1, test_method="split_exact",
                               args=(n_perm=19, n_splits=8), seed=1)
    @test r.n_failed_splits == 8
    @test r.n_failed_perms == 0
    @test r.n_failed_folds === nothing
end
