using ParticleMesh
using Test

@testset "T-000 umbrella package loads [A6]" begin
    @test isdefined(ParticleMesh, :ParticleMesh)
end
