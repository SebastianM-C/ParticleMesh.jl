const TESTDIR = normpath(joinpath(@__DIR__, "..", "..", "..", "lib", "ParticleMeshElectromagnetic", "test"))
using Test

include(joinpath(TESTDIR, "setup.jl"))          # engine test setup, fd_scan, tolerances, case files
include(joinpath(TESTDIR, "em_helpers.jl"))
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))   # the case files' observables
using .Observables: damping_from_peaks, growth_from_window
include(joinpath(TESTDIR, "theory.jl"))
include(joinpath(TESTDIR, "determinism_cases.jl"))
include(joinpath(TESTDIR, "yee_nd_cases.jl"))     # WP-022: grids, random fields and digests of yee_nd.jl and its worker
include(joinpath(TESTDIR, "em2d_cases.jl"))       # WP-024: the 2D problems and digests of em2d.jl and the determinism worker
include(joinpath(TESTDIR, "antenna_cases.jl"))    # WP-026: the antenna's runs and their analysis
include(joinpath(TESTDIR, "filter_cases.jl"))     # WP-029: the filtered runs of filter.jl and the determinism worker
include(joinpath(TESTDIR, "boundaries_cases.jl")) # WP-027: the open boundary's runs and the layer's discrete theory
include(joinpath(TESTDIR, "window_cases.jl"))     # WP-028: the moving window's problems

@testset "ParticleMeshElectromagnetic" begin
    try
        include(joinpath(TESTDIR, "yee.jl"))
        include(joinpath(TESTDIR, "yee_nd.jl"))        # WP-022: T-320 to T-329, the solver in N dimensions
        include(joinpath(TESTDIR, "boris.jl"))
        include(joinpath(TESTDIR, "pushers.jl"))
        # pushers_sciml.jl skipped: SciMLBase is not warm on the box
        include(joinpath(TESTDIR, "pushers_resolution.jl"))   # WP-016: after pushers.jl, whose helpers it uses
        include(joinpath(TESTDIR, "gauss.jl"))
        include(joinpath(TESTDIR, "initialize.jl"))    # T-420, the time of the loaded momenta
        include(joinpath(TESTDIR, "subluminal.jl"))    # T-480, T-481: a load whose drawn speeds reach c is refused (#257); after em2d_cases.jl
        include(joinpath(TESTDIR, "physics.jl"))
        include(joinpath(TESTDIR, "determinism.jl"))
        include(joinpath(TESTDIR, "gradient.jl"))
        include(joinpath(TESTDIR, "allocations.jl"))
        include(joinpath(TESTDIR, "em2d.jl"))          # WP-024: T-350 to T-358, after determinism.jl (T-355 reuses its worker runs)
        include(joinpath(TESTDIR, "em2d_step.jl"))     # WP-023: T-360 to T-369, after em2d.jl (its helpers) and determinism.jl (T-363's worker runs)
        include(joinpath(TESTDIR, "antenna.jl"))       # WP-026: T-400 to T-409
        include(joinpath(TESTDIR, "filter.jl"))        # WP-029: T-440 to T-445, after em2d.jl and em2d_step.jl (their helpers)
        include(joinpath(TESTDIR, "boundaries.jl"))    # WP-027: T-450 to T-457, after em2d_step.jl (its helpers) and determinism.jl (T-454's worker runs)
        include(joinpath(TESTDIR, "window.jl"))        # WP-028: T-470 to T-479
        include(joinpath(TESTDIR, "harness.jl"))       # after gradient.jl: T-105 reuses its run
        include(joinpath(TESTDIR, "gpu.jl"))           # WP-013: after physics.jl (T-197 compares its CPU runs); or one line
        include(joinpath(TESTDIR, "qa.jl"))            # T-372 to T-374: Aqua and JET
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
