# One testdata/ fixture per function family, at the tolerances
# (scalars 1e-12, arrays 1e-10, plus rtol 1e-14 for finite expected values;
# integers and strings exact). The numbers
# come from the Python wheel, which passes the whole corpus in
# plskit-py/tests/test_corpus.py, so this checks conversion.
#
# Key resolution: a fixture key maps
# onto the result by name, exact, then ASCII case-insensitive (`y_std` is
# `Y_std`), then into nested results, recursively. `{field}__keys` /
# `{field}__values` encode an integer-keyed map; `{field}_{part}` and
# `{field}_k{i}_{part}` (part in point, lower, upper, sd) encode a CIScalar
# and the i-th CIScalar of a list. Every fixture key must resolve and match.
# Fixtures are read with NumPy through PythonCall (no test dependency).

# The corpus lives in the monorepo checkout (`../../testdata` from here),
# which is not present for an installed copy (`Pkg.test("PLSKit")` outside
# the workspace). `PLSKIT_TESTDATA` points at a corpus directory explicitly;
# `PLSKIT_REQUIRE_CORPUS` ("1"/"true", case-insensitive) turns a missing
# corpus into a failure instead of a skip. The R wrapper uses the same two
# variable names.
const _TESTDATA_FALLBACK = normpath(joinpath(@__DIR__, "..", "..", "testdata"))
const TESTDATA = get(ENV, "PLSKIT_TESTDATA", _TESTDATA_FALLBACK)
const _REQUIRE_CORPUS = lowercase(get(ENV, "PLSKIT_REQUIRE_CORPUS", "")) in ("1", "true")
const _HAVE_CORPUS = isfile(joinpath(TESTDATA, "manifest.json"))

if !_HAVE_CORPUS && _REQUIRE_CORPUS
    error("PLSKit: PLSKIT_REQUIRE_CORPUS is set but no corpus was found at " *
          "\"$(TESTDATA)\" (set PLSKIT_TESTDATA to the testdata directory)")
end

# One case per function in the manifest, the cheapest that still covers
# nested results (sequence fit, CI bundle), maps, lists and bool/int vectors.
const CORPUS_CASES = [
    "preprocess_n50_d10_with_weights",
    "pls1_fit_small_n50_d10_sequence",
    "pls1_predict_basic_n80_d6_k2",
    "pls1_find_k_optimal_r2_se",
    "pls1_find_k_sequence_split_nb",
    "spls1_fit_small_n50_d10_k2_keep3",
    "spls1_find_keep_optimal_k1",
    "spls1_find_k_optimal_r2_se_keep3",
    "spls1_find_k_sequence_split_nb_keep3",
    "pls3_fit_small_n50_p10_q4_k3",
    "pls3_transform_basic_n80_p6_q3_k2",
    "spls3_fit_small_n50_p10_q4_keep3_2_k2",
    "pls3_confirmatory_split_nb",
    "pls1_confirmatory_split_nb_ci",
    "pls1_confirmatory_auto_split_exact",
    "pls1_perm_null_basic_n80_d6_k2",
    "pls1_rotation_stability_n80_d6_k2",
    "rotate_varimax_d6_k2",
]

const ATOL_SCALAR = 1e-12
const ATOL_ARRAY = 1e-10
const RTOL = 1e-14

# JSON value -> Julia (objects become Dict{String,Any}, so `args` exercises
# the Dict path of the argument conversion).
function jsonval(x::Py)
    pyis(x, pybuiltins.None) && return nothing
    pyisinstance(x, pybuiltins.bool) && return pyconvert(Bool, x)
    pyisinstance(x, pybuiltins.int) && return pyconvert(Int, x)
    pyisinstance(x, pybuiltins.float) && return pyconvert(Float64, x)
    pyisinstance(x, pybuiltins.str) && return pyconvert(String, x)
    pyisinstance(x, pybuiltins.dict) &&
        return Dict{String,Any}(pyconvert(String, k) => jsonval(x[k]) for k in x)
    error("unexpected JSON value $(x)")
end

function manifest_cases()
    json = pyimport("json")
    m = json.loads(read(joinpath(TESTDATA, "manifest.json"), String))
    return [Dict{String,Any}("name" => pyconvert(String, c["name"]),
                             "function" => pyconvert(String, c["function"]),
                             "inputs" => pyconvert(String, c["inputs"]),
                             "outputs" => pyconvert(String, c["outputs"]),
                             "kwargs" => jsonval(c["kwargs"]))
            for c in m["cases"]]
end

# npz entry -> Float64 / Int scalar or array, or String (1-D uint8 bytes,
# plskit-testdata-gen/src/npz.rs::add_string).
function fixture_value(a::Py)
    kind = pyconvert(String, a.dtype.kind)
    ndim = pyconvert(Int, a.ndim)
    if pyconvert(String, a.dtype.name) == "uint8" && ndim == 1
        return String(pyconvert(Vector{UInt8}, a))
    end
    T = kind == "f" ? Float64 : kind in ("i", "u", "b") ? Int : error("dtype $(a.dtype)")
    c = np.ascontiguousarray(a, dtype=T === Float64 ? np.float64 : np.int64)
    ndim == 0 && return pyconvert(T, c.item())
    return pyconvert(Array{T,ndim}, c)
end

function load_npz(rel)
    out = Dict{String,Any}()
    f = np.load(joinpath(TESTDATA, rel); allow_pickle=false)
    try
        for k in f.files
            out[pyconvert(String, k)] = fixture_value(f[k])
        end
    finally
        f.close()
    end
    return out
end

describes_data(fn, key) =
    key in ("d", "n", "n_new", "n_train", "seed_new", "seed_train") ||
    (key == "seed" && fn in ("spls1_fit", "pls3_fit"))

function corpus_weights(case, inputs)
    descriptor = get(case["kwargs"], "weights", nothing) == "nonuniform"
    @assert descriptor == haskey(inputs, "weights") "$(case["name"]): weights descriptor and NPZ disagree"
    return get(inputs, "weights", nothing)
end

# Call `f` with every argument by name, splitting positionals from
# keywords the way the rendered stub declares them.
function call_by_name(fname::AbstractString, named::Dict{Symbol,Any})
    f = getfield(PLSKit, Symbol(fname))
    pos = Base.method_argnames(only(methods(f)))[2:end]
    args = [named[p] for p in pos]
    kws = [k => v for (k, v) in named if !(k in pos)]
    return f(args...; kws...)
end

# (name, value) pairs of a result, the unit key resolution walks.
record(r::PlsKitResult) = [(String(k), getproperty(r, k)) for k in propertynames(r)]

function run_case(case, inputs)
    fn = case["function"]
    kw = case["kwargs"]
    if fn == "pls1_predict"
        fit = pls1_fit(inputs["X_train"], inputs["y_train"]; k=kw["k"])
        return vcat(record(fit), [("y_pred", pls1_predict(fit, inputs["X_new"]))])
    elseif fn == "rotate"
        fit = pls1_fit(inputs["X"], inputs["y"]; k=kw["k"])
        return record(rotate(fit.W; method=kw["method"]))
    elseif fn == "preprocess"
        return record(preprocess(; X=inputs["X"], Y=inputs["y"],
                                 weights=corpus_weights(case, inputs)))
    elseif fn == "pls3_transform"
        model = pls3_fit(inputs["X"], inputs["Y"]; k=kw["k"])
        return record(pls3_transform(model; X_new=inputs["X_new"], Y_new=inputs["Y_new"],
                                     which=kw["which"]))
    end
    named = Dict{Symbol,Any}(:X => inputs["X"])
    haskey(inputs, "y") && (named[:y] = inputs["y"])
    haskey(inputs, "Y") && (named[:Y] = inputs["Y"])
    w = corpus_weights(case, inputs)
    w === nothing || (named[:weights] = w)
    for (k, v) in kw
        (k == "weights" || describes_data(fn, k)) && continue
        named[Symbol(k)] = v
    end
    return record(call_by_name(fn, named))
end

function lookup(rec, key)
    for (k, v) in rec
        k == key && return v
    end
    for (k, v) in rec
        lowercase(k) == lowercase(key) && return v
    end
    for (_, v) in rec
        if v isa PlsKitResult
            found = lookup(record(v), key)
            found === nothing || return found
        end
    end
    return nothing
end

const CI_PARTS = ("point", "lower", "upper", "sd")

# The value a fixture key names, or `nothing`.
function resolve(rec, key, expected)
    if endswith(key, "__keys")
        m = lookup(rec, key[1:end-6])
        return m isa Dict{Int,Float64} ? sort(collect(keys(m))) : nothing
    elseif endswith(key, "__values")
        base = key[1:end-8]
        m = lookup(rec, base)
        m isa Dict{Int,Float64} || return nothing
        return [m[k] for k in expected[base * "__keys"]]
    end
    v = lookup(rec, key)
    v === nothing || return v
    i = findlast('_', key)
    i === nothing && return nothing
    base, part = key[1:i-1], key[i+1:end]
    part in CI_PARTS || return nothing
    ci = lookup(rec, base)
    if ci === nothing
        j = findlast("_k", base)
        j === nothing && return nothing
        list = lookup(rec, base[1:first(j)-1])
        idx = tryparse(Int, base[last(j)+1:end])
        (list isa Vector && idx !== nothing && idx < length(list)) || return nothing
        ci = list[idx+1]
    end
    return ci isa PlsKitResult ? getproperty(ci, Symbol(part)) : nothing
end

# NaN on exactly one side is a mismatch, never a crash or a silent pass:
# `a == e` is false, `isnan(a) && isnan(e)` is false, and `abs(a - e)` is
# itself NaN so the `<=` comparison is false too. Named in the enclosing
# `@testset "$(name)"` and by `key` in the `@info` line below.
close_enough(a, e, atol) =
    a == e || (isnan(a) && isnan(e)) || (isfinite(e) && abs(a - e) <= atol + RTOL * abs(e))

function matches(found, want)
    if want isa String
        return found == want
    elseif want isa Float64
        return found isa Real && close_enough(Float64(found), want, ATOL_SCALAR)
    elseif want isa Int
        return (found isa Integer || found isa Bool) && Int(found) == want
    elseif want isa Array{Float64}
        return found isa AbstractArray{<:Real} && size(found) == size(want) &&
               all(close_enough.(Float64.(found), want, ATOL_ARRAY))
    elseif want isa Array{Int}
        return found isa AbstractArray && size(found) == size(want) && Int.(found) == want
    end
    return false
end

if !_HAVE_CORPUS
    @info "PLSKit: skipping the corpus testset, no corpus found" TESTDATA PLSKIT_TESTDATA = get(
        ENV, "PLSKIT_TESTDATA", nothing) PLSKIT_REQUIRE_CORPUS = get(
        ENV, "PLSKIT_REQUIRE_CORPUS", nothing)
else

cases = Dict(c["name"] => c for c in manifest_cases())

@testset "the chosen cases cover every function in the manifest" begin
    @test Set(c["function"] for c in values(cases)) ==
          Set(cases[n]["function"] for n in CORPUS_CASES)
end

@testset "$(name)" for name in CORPUS_CASES
    case = cases[name]
    inputs = load_npz(case["inputs"])
    expected = load_npz(case["outputs"])
    rec = run_case(case, inputs)
    for key in sort(collect(keys(expected)))
        found = resolve(rec, key, expected)
        ok = found !== nothing && matches(found, expected[key])
        ok || @info "corpus mismatch" name key found expected[key]
        @test ok
    end
end

end # _HAVE_CORPUS
