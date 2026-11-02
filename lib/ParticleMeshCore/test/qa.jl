# Programmatic quality checks: Aqua (ambiguities, stale dependencies, compat entries, piracy, ...)
# and JET (undefined names and impossible calls). JET follows the compiler's internals, so it runs on
# released Julia versions only; the `Julia pre` job skips it.
using Aqua

@testset "T-370 Aqua: no ambiguities, stale dependencies, missing compat entries or piracy in ParticleMeshCore [A6]" begin
    # Unbound type parameters: seven NTuple{N,T} signatures leave T unbound for an empty tuple (#252).
    Aqua.test_all(ParticleMeshCore; ambiguities = (recursive = false,), unbound_args = (broken = true,))
end

if isempty(VERSION.prerelease)
    @eval using JET
    @testset "T-371 JET finds no undefined names or impossible calls in ParticleMeshCore [A6]" begin
        JET.test_package(ParticleMeshCore; target_modules = (ParticleMeshCore,), mode = :typo)
    end
else
    println("T-371 skipped: JET runs on released Julia versions only (this is $(VERSION))")
end
