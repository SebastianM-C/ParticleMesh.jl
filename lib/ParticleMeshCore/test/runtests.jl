using Test

include("setup.jl")          # ParticleMeshCore, the backend, the A8 matrix, tolerances, comparisons
include("helpers.jl")
include("digest.jl")
include("engine_cases.jl")
include("flux_cases.jl")
include("n1_digests.jl")
include("engine2d_cases.jl")

@testset "ParticleMeshCore" begin
    try
        include("transfers.jl")
        include("particles.jl")
        include("allocations.jl")
        include("determinism.jl")
        include("engine.jl")
        include("flux.jl")
        include("engine2d.jl")
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
