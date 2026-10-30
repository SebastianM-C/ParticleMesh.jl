using ParticleMesh
using Test

@testset "T-000 umbrella package loads [R-008]" begin
    @test isdefined(ParticleMesh, :ParticleMesh)
end
