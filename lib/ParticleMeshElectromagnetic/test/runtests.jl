using Test

include("setup.jl")          # engine test setup, fd_scan, tolerances, case files
include("em_helpers.jl")
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))   # the case files' observables
using .Observables: damping_from_peaks, growth_from_window
include("theory.jl")
include("determinism_cases.jl")
include("yee_nd_cases.jl")     # WP-022: grids, random fields and digests of yee_nd.jl and its worker
include("em2d_cases.jl")       # WP-024: the 2D problems and digests of em2d.jl and the determinism worker
include("antenna_cases.jl")    # WP-026: the antenna's runs and their analysis
include("filter_cases.jl")     # WP-029: the filtered runs of filter.jl and the determinism worker
include("boundaries_cases.jl") # WP-027: the open boundary's runs and the layer's discrete theory
include("pml_cases.jl")        # WP-034: the PML's runs and its discrete theory

@testset "ParticleMeshElectromagnetic" begin
    try
        include("yee.jl")
        include("yee_nd.jl")        # WP-022: T-320 to T-329, the solver in N dimensions
        include("boris.jl")
        include("pushers.jl")
        include("pushers_sciml.jl")
        include("pushers_resolution.jl")   # WP-016: after pushers.jl, whose helpers it uses
        include("gauss.jl")
        include("initialize.jl")    # T-420, the time of the loaded momenta; T-522, where the density is sampled
        include("subluminal.jl")    # T-480, T-481: a load whose drawn speeds reach c is refused (#257); after em2d_cases.jl
        include("physics.jl")
        include("determinism.jl")
        include("gradient.jl")
        include("allocations.jl")
        include("em2d.jl")          # WP-024: T-350 to T-358, after determinism.jl (T-355 reuses its worker runs)
        include("em2d_step.jl")     # WP-023: T-360 to T-369, after em2d.jl (its helpers) and determinism.jl (T-363's worker runs)
        include("antenna.jl")       # WP-026: T-400 to T-409
        include("filter.jl")        # WP-029: T-440 to T-445, after em2d.jl and em2d_step.jl (their helpers)
        include("boundaries.jl")    # WP-027: T-450 to T-457, after em2d_step.jl (its helpers) and determinism.jl (T-454's worker runs)
        include("pml.jl")           # WP-034: T-540 to T-548, after boundaries.jl (its helpers)
        include("harness.jl")       # after gradient.jl: T-105 reuses its run
        include("gpu.jl")           # WP-013: after physics.jl (T-197 compares its CPU runs); or one line
        include("qa.jl")            # T-372 to T-374: Aqua and JET
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
