using ParticleMesh
using Test

@testset "T-000 umbrella package loads [A6]" begin
    @test isdefined(ParticleMesh, :ParticleMesh)
end

# The lib/ subpackages' own suites run in CI's per-package step, not from here.
@testset "T-145 umbrella re-exports ParticleMeshCore [A6, WP-010]" begin
    @test ParticleMesh.ParticleMeshCore === ParticleMesh.ParticleMeshCore.ParticleMeshCore
    @test isdefined(ParticleMesh, :PeriodicGrid) && ParticleMesh.PeriodicGrid === ParticleMesh.ParticleMeshCore.PeriodicGrid
    @test isdefined(ParticleMesh, :deposit!) && ParticleMesh.deposit! === ParticleMesh.ParticleMeshCore.deposit!
end
