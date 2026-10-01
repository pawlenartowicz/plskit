# RULE 1 parity against the Python package PLSKit actually runs: the same
# public functions and result types, and per function the same argument
# names. Julia positionals are Python's required, not keyword-only
# parameters, in order; every other parameter is a Julia keyword, in
# Python's order (spec §5.2). Base.method_argnames / Base.kwarg_decl are
# unexported internals: if a Julia upgrade breaks them, this test fails,
# not the package.

py_names = [pyconvert(String, n) for n in pk.__all__]
py_functions = [n for n in py_names if pytruth(inspect.isfunction(pygetattr(pk, n)))]
py_types = [n for n in py_names
            if pytruth(inspect.isclass(pygetattr(pk, n))) &&
               pytruth(pyimport("dataclasses").is_dataclass(pygetattr(pk, n)))]

exported = setdiff(names(PLSKit), [:PLSKit])
jl_functions = [n for n in exported if getfield(PLSKit, n) isa Function]
jl_types = [n for n in exported
            if getfield(PLSKit, n) isa Type && getfield(PLSKit, n) <: PlsKitResult &&
               n !== :PlsKitResult]

@testset "same exported functions" begin
    @test Set(jl_functions) == Set(Symbol.(py_functions))
end

@testset "same result types" begin
    @test Set(jl_types) == Set(Symbol.(py_types))
    for t in py_types
        T = Symbol(t)
        @test getfield(PLSKit, T) === PlsKitResult{T}
        py_fields = [pyconvert(String, f.name)
                     for f in pyimport("dataclasses").fields(pygetattr(pk, t))]
        @test Set(PLSKit._RESULT_FIELDS[T]) == Set(Symbol.(py_fields))
    end
end

@testset "argument names: $(name)" for name in py_functions
    params = collect(inspect.signature(pygetattr(pk, name)).parameters.values())
    kinds = [pyconvert(String, p.kind.name) for p in params]
    # §5.2: _api.py has no positional-only parameters, so Julia may pass
    # every argument by name.
    @test !("POSITIONAL_ONLY" in kinds)
    required = [pyis(p.default, inspect.Parameter.empty) for p in params]
    pnames = [Symbol(pyconvert(String, p.name)) for p in params]
    positional = [pnames[i] for i in eachindex(params)
                  if required[i] && kinds[i] == "POSITIONAL_OR_KEYWORD"]
    keywords = [pnames[i] for i in eachindex(params)
                if !(required[i] && kinds[i] == "POSITIONAL_OR_KEYWORD")]
    m = only(methods(getfield(PLSKit, Symbol(name))))
    @test Base.method_argnames(m)[2:end] == positional
    @test Base.kwarg_decl(m) == keywords
end
