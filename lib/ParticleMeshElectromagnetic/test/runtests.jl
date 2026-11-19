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
include("ionization_cases.jl") # WP-059: the field ionization's problems, after em2d_cases.jl
include("em3d_cases.jl")       # WP-042: the 3D problems and digests of em3d_step.jl and its worker, after em2d_cases.jl

# The suite's files in their order; each note says what a file reuses from earlier ones. CI runs
# them in the groups of groups.toml, in parallel jobs (ParticleMeshCore's test/groups.jl).
const TEST_FILES = [
    "yee.jl",
    "yee_nd.jl",              # WP-022: T-320 to T-329, the solver in N dimensions
    "em3d.jl",                # WP-042: T-880 to T-883, the layout, the initial field and the recorder at N = 3, after yee_nd.jl (its helpers)
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
    "shaped_antenna.jl",      # WP-066: T-900 to T-908, shaped pulses (PulseShapes) on the antenna, after antenna.jl (antenna_cases.jl's analysis)
    "filter.jl",              # WP-029: T-440 to T-445, after em2d.jl and em2d_step.jl (their helpers)
    "boundaries.jl",          # WP-027: T-450 to T-457, after em2d_step.jl (its helpers) and determinism.jl (T-454's worker runs)
    "pml.jl",                 # WP-034: T-540 to T-549, after boundaries.jl (its helpers)
    "window.jl",              # WP-028: T-470 to T-479
    "pml_back.jl",            # WP-034: T-570 to T-576, the PML on the window's back edge, after window.jl
    "lwfa_gradient.jl",       # WP-043: T-580 to T-582, a gradient through a small LWFA run, after window.jl
    "witness.jl",             # WP-047: T-640 to T-645, the witness bunch and its mean energy gain, after lwfa_gradient.jl
    "tunables.jl",            # WP-060: T-790 to T-793, the parameters as one vector and as a SciML structure, after lwfa_gradient.jl (its case and T-581's gradient)
    "harness.jl",             # after gradient.jl: T-105 reuses its run
    "params.jl",              # WP-055: T-730 to T-735, the step as a function of the state and the parameters, after harness.jl
    "recorder.jl",            # WP-062: T-810 to T-812 and T-814, the recorder's cadence and selection; its own helpers
    "restart.jl",             # WP-056: T-740 to T-746 and T-748, the closed state and the restart, after params.jl (its helpers)
    "modifiers.jl",           # WP-057: T-750 to T-754 and T-759, the one field solve and its tuple of modifiers, after restart.jl (its helpers)
    "one_step.jl",            # WP-057: T-755 to T-757, the one step body over N, after modifiers.jl (its helpers)
    "ionization.jl",          # WP-059: T-770 to T-779 and T-781 to T-785, the field ionization of a dopant; its own helpers
    "barrier_suppression.jl", # WP-063: T-820 to T-826, the rate by the level's type and the Tong–Lin level, after ionization.jl (its helpers)
    "resort_step.jl",         # WP-041: T-852, the step's sort from the last order, after ionization.jl (its helpers)
    "gather.jl",              # WP-067: T-960 to T-963, the shape-compensated gather; after ionization.jl (its helpers)
    "transposes.jl",          # WP-069: T-985 to T-987, the transposed pairs of the step; yee_nd_cases.jl's helpers
    "em3d_step.jl",           # WP-042: T-950 to T-957, the periodic 3D3V step; its own helpers and worker (T-952)
    "em3d_open.jl",           # WP-072: T-1010 to T-1018, the open transverse boundaries of the 3D step, after em3d_step.jl (its helpers and worker)
    "em3d_antenna.jl",        # WP-073: T-1020 to T-1028, the laser antenna of the 3D step, after em3d_step.jl (its helpers and worker)
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
