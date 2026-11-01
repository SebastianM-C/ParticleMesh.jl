using Test

include("setup.jl")          # ParticleMeshCore, the backend, the A8 matrix, tolerances, comparisons
include("es_reference.jl")   # the electrostatic edge-case reference (test support, not engine code)
using .ESReference
include("cases.jl")
include("theory.jl")
include("helpers.jl")
include("digest.jl")
include("engine_cases.jl")
include("flux_cases.jl")
include("n1_digests.jl")

@testset "ParticleMeshCore" begin
    try
        include("transfers.jl")
        include("particles.jl")
        include("electrostatic.jl")
        include("physics.jl")
        include("gradient.jl")
        include("determinism.jl")
        include("engine.jl")
        include("split.jl")
        include("flux.jl")
        include("engine2d.jl")
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
