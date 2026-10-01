using PLSKit
using PythonCall
using Test

include("helpers.jl")

@testset "PLSKit" begin
    @testset "package" include("package.jl")
    @testset "registry parity" include("registry.jl")
    @testset "conversion" include("conversion.jl")
    @testset "arguments" include("arguments.jl")
    @testset "errors" include("errors.jl")
    @testset "round trip and seeds" include("roundtrip.jl")
    @testset "warnings" include("warnings.jl")
    @testset "corpus" include("corpus.jl")
end
