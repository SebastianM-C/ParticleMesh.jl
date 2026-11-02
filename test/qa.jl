# Aqua and JET on the umbrella package; the lib/ packages run theirs in their own suites.
using Aqua

@testset "T-375 Aqua: no ambiguities, stale dependencies, missing compat entries or piracy in ParticleMesh [A6]" begin
    Aqua.test_all(ParticleMesh; ambiguities = (recursive = false,))
end

if isempty(VERSION.prerelease)
    @eval using JET
    @testset "T-376 JET finds no undefined names or impossible calls in ParticleMesh [A6]" begin
        JET.test_package(ParticleMesh; target_modules = (ParticleMesh,), mode = :typo)
    end
else
    println("T-376 skipped: JET runs on released Julia versions only (this is $(VERSION))")
end
