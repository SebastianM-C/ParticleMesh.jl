# Regression microbenchmarks (BenchmarkTools). Run from the repository root with
#   julia +1.13 --project=benchmark benchmark/benchmarks.jl
# Results go to benchmark/results/ (ignored) and are recorded as run records in trace/.
using BenchmarkTools
using ParticleMesh

const SUITE = BenchmarkGroup()
SUITE["umbrella"] = BenchmarkGroup()
SUITE["umbrella"]["load"] = @benchmarkable isdefined(ParticleMesh, :ParticleMesh)

if abspath(PROGRAM_FILE) == @__FILE__
    results = run(SUITE; verbose = true)
    mkpath(joinpath(@__DIR__, "results"))
    BenchmarkTools.save(joinpath(@__DIR__, "results", "latest.json"), results)
end
