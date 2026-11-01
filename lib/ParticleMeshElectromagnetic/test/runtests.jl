using Test

include("setup.jl")          # engine test setup, fd_scan, tolerances, case files
include("em_helpers.jl")
include(joinpath(CORE_TEST, "es_reference.jl"))   # the electrostatic edge-case reference (test support)
using .ESReference: ESReference, landau_case, two_stream_case, field_energy_history
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
        include("gpu.jl")           # WP-013: after physics.jl (T-197 compares its CPU runs); or one line
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
