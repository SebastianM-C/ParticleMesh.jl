using Test

include("setup.jl")          # ParticleMeshCore, the backend, the A8 matrix, tolerances, comparisons
include("helpers.jl")
include("digest.jl")
include("engine_cases.jl")
include("flux_cases.jl")
include("n1_digests.jl")
include("engine2d_cases.jl")
include("flux2d_cases.jl")
include("spectral_cases.jl")

@testset "ParticleMeshCore" begin
    try
        include("transfers.jl")
        include("particles.jl")
        include("allocations.jl")
        include("determinism.jl")
        include("engine.jl")
        include("flux.jl")
        include("engine2d.jl")
        include("flux2d.jl")        # WP-020: T-300 to T-309, the flux deposition at N = 2
        include("spectral.jl")      # WP-021: T-310 to T-316, the periodic solve
        include("input_checks.jl")  # T-340 to T-342, T-344, T-345: input checks, the Float32 draw, the Fast() fallback sort (#194, #195, #196, #201)
        include("switch.jl")        # WP-013: T-199, the GPU switch
        include("gpu.jl")           # WP-013: the GPU testsets, or one line saying why they do not run
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
