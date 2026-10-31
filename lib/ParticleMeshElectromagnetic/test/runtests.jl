using Test

include("setup.jl")

@testset "ParticleMeshElectromagnetic" begin
    try
        include("yee.jl")
        include("boris.jl")
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
