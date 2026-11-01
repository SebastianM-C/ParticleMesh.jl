using Test

include("setup.jl")          # engine test setup, fd_scan, tolerances, case files
include("em_helpers.jl")
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))   # the case files' observables
using .Observables: damping_from_peaks, growth_from_window
include("theory.jl")
include("determinism_cases.jl")

@testset "ParticleMeshElectromagnetic" begin
    try
        include("yee.jl")
        include("boris.jl")
        include("gauss.jl")
        include("physics.jl")
        include("determinism.jl")
        include("gradient.jl")
        include("allocations.jl")
        include("harness.jl")       # after gradient.jl: T-105 reuses its run
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
