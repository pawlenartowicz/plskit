# Runs the Julia examples in README.md as written. From the first ```julia
# block that loads PLSKit, every later ```julia block runs in one fresh
# module, so a block may use names an earlier one defined; blocks before it
# (install lines) are skipped. Same rule as plskit-py/tests/test_readme.py:
# change together.
readme = joinpath(@__DIR__, "..", "README.md")
blocks = [m.captures[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, read(readme, String))]
start = findfirst(b -> occursin("using PLSKit", b), blocks)
@test start !== nothing
sandbox = Module(:README)
for block in blocks[start:end]
    include_string(sandbox, block, readme)
end
