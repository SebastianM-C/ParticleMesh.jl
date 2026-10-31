using ParticleMesh
using Test

@testset "T-000 umbrella package loads [A6]" begin
    @test isdefined(ParticleMesh, :ParticleMesh)
end

@testset "T-145 umbrella re-exports ParticleMeshCore and runs its test suite [A6, WP-010]" begin
    @test ParticleMesh.ParticleMeshCore === ParticleMesh.ParticleMeshCore.ParticleMeshCore
    @test isdefined(ParticleMesh, :PeriodicGrid) && ParticleMesh.PeriodicGrid === ParticleMesh.ParticleMeshCore.PeriodicGrid
    # CI runs only the root test suite, so the subpackage suites run from here, each in its own
    # process with its own test environment (a workspace member of the root manifest).
    root = dirname(@__DIR__)
    for lib in ("ParticleMeshCore",)
        project = joinpath(root, "lib", lib)
        # Pkg.test restricts JULIA_LOAD_PATH to the test environment; the child needs Pkg.
        cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$project -e "using Pkg; Pkg.test()"`,
                     "JULIA_LOAD_PATH" => "@:@stdlib")
        @test success(pipeline(cmd; stdout, stderr))
    end
end
