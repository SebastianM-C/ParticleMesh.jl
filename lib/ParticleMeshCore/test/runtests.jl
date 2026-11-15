using Test

include("setup.jl")          # ParticleMeshCore, the backend, the A8 matrix, tolerances, comparisons
include("helpers.jl")
include("digest.jl")
include("engine_cases.jl")
include("flux_cases.jl")
include("n1_digests.jl")
include("engine2d_cases.jl")
include("flux2d_cases.jl")
include("recorded_2d.jl")     # WP-037: the 2D Reference digests recorded on a commit (T-266, T-303)
include("spectral_cases.jl")
include("momenta_cases.jl")
include("workers.jl")         # the thread-count worker runs and the helpers that read them
include("groups.jl")          # the CI test groups (PM_TEST_GROUP)

# The suite's files in their order. CI runs them in the groups of groups.toml, in parallel jobs (groups.jl).
const TEST_FILES = [
    "transfers.jl",
    "particles.jl",
    "density_at.jl",    # T-520, T-521: where load_quiet! evaluates the density (#226)
    "allocations.jl",
    "determinism.jl",
    "engine.jl",
    "flux.jl",
    "engine2d.jl",
    "flux2d.jl",        # WP-020: T-300 to T-309, the flux deposition at N = 2
    "spectral.jl",      # WP-021: T-310 to T-316, the periodic solve
    "momenta.jl",       # WP-033: T-530 to T-536, T-538, the momentum distributions (after engine.jl, whose worker it reads)
    "input_checks.jl",  # T-340 to T-342, T-344, T-345: input checks, the Float32 draw, the Fast() fallback sort (#194, #195, #196, #201)
    "launch.jl",        # T-500, T-501: a fixed workgroupsize a kernel cannot take is refused by name (#297); T-502, the tuned one (#375)
    "switch.jl",        # WP-013: T-199, the GPU switch
    "gpu.jl",           # WP-013: the GPU testsets, or one line saying why they do not run
    "qa.jl",            # T-370, T-371: Aqua and JET
]
@testset "ParticleMeshCore" begin
    try
        include_test_files(@__MODULE__, @__DIR__, TEST_FILES)
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
