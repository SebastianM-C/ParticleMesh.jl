using Test

include("setup.jl")          # engine test setup, fd_scan, tolerances, case files
include("em_helpers.jl")
include(joinpath(CORE_TEST, "groups.jl"))   # the CI test groups (PM_TEST_GROUP)
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
include("window_cases.jl")     # WP-028: the moving window's problems
include("pml_back_cases.jl")   # WP-034: the back-edge PML's problems and runs, after window_cases.jl

# The suite's files in their order; each note says what a file reuses from earlier ones. CI runs
# them in the groups of groups.toml, in parallel jobs (ParticleMeshCore's test/groups.jl).
const TEST_FILES = [
    "yee.jl",
    "yee_nd.jl",              # WP-022: T-320 to T-329, the solver in N dimensions
    "boris.jl",
    "pushers.jl",
    "pushers_sciml.jl",
    "pushers_resolution.jl",  # WP-016: after pushers.jl, whose helpers it uses
    "gauss.jl",
    "fast_reference.jl",      # WP-037: T-290, Fast against Reference histories
    "initialize.jl",          # T-420, the time of the loaded momenta; T-522, where the density is sampled
    "subluminal.jl",          # T-480, T-481: a Gaussian-velocity load whose drawn speeds reach c is refused (#257); after em2d_cases.jl
    "momenta.jl",             # WP-033: T-537, T-539, the momentum distributions in the module; after subluminal.jl (its helpers)
    "physics.jl",
    "determinism.jl",
    "gradient.jl",
    "allocations.jl",
    "em2d.jl",                # WP-024: T-350 to T-358, after determinism.jl (T-355 reuses its worker runs)
    "em2d_step.jl",           # WP-023: T-360 to T-369, after em2d.jl (its helpers) and determinism.jl (T-363's worker runs)
    "antenna.jl",             # WP-026: T-400 to T-409
    "filter.jl",              # WP-029: T-440 to T-445, after em2d.jl and em2d_step.jl (their helpers)
    "boundaries.jl",          # WP-027: T-450 to T-457, after em2d_step.jl (its helpers) and determinism.jl (T-454's worker runs)
    "pml.jl",                 # WP-034: T-540 to T-549, after boundaries.jl (its helpers)
    "window.jl",              # WP-028: T-470 to T-479
    "pml_back.jl",            # WP-034: T-570 to T-576, the PML on the window's back edge, after window.jl
    "lwfa_gradient.jl",       # WP-043: T-580 to T-582, a gradient through a small LWFA run, after window.jl
    "witness.jl",             # WP-047: T-640 to T-645, the witness bunch and its mean energy gain, after lwfa_gradient.jl
    "harness.jl",             # after gradient.jl: T-105 reuses its run
    "params.jl",              # WP-055: T-730 to T-735, the step as a function of the state and the parameters, after harness.jl
    "restart.jl",             # WP-056: T-740 to T-746 and T-748, the closed state and the restart, after params.jl (its helpers)
    "modifiers.jl",           # WP-057: T-750 to T-754 and T-759, the one field solve and its tuple of modifiers, after restart.jl (its helpers)
    "one_step.jl",            # WP-057: T-755 to T-757, the one step body over N, after modifiers.jl (its helpers)
    "gpu.jl",                 # WP-013: after physics.jl (T-197 compares its CPU runs); or one line
    "qa.jl",                  # T-372 to T-374: Aqua and JET
]

@testset "ParticleMeshElectromagnetic" begin
    try
        include_test_files(@__MODULE__, @__DIR__, TEST_FILES)
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
