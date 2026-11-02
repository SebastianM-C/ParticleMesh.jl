# Programmatic quality checks: Aqua and JET, as in ParticleMeshCore's qa.jl. JET runs on released Julia
# versions only; the `Julia pre` job skips it.
using Aqua

@testset "T-372 Aqua: no ambiguities, stale dependencies, missing compat entries or piracy in ParticleMeshElectromagnetic [A6]" begin
    Aqua.test_all(ParticleMeshElectromagnetic; ambiguities = (recursive = false,))
end

if isempty(VERSION.prerelease)
    @eval using JET
    @testset "T-373 JET finds no undefined names or impossible calls in ParticleMeshElectromagnetic [A6]" begin
        JET.test_package(ParticleMeshElectromagnetic; target_modules = (ParticleMeshElectromagnetic,), mode = :typo)
    end
    # A planted `Ref{Any}(k)[] + 1` in the 2D step! gives three reports here; the step is free of them.
    @testset "T-374 no runtime dispatch in step! at N = 1 and N = 2 (with and without current), Float32, Float64 and Dual{Float32,3} [A8, WP-023]" begin
        targets = (ParticleMeshCore, ParticleMeshElectromagnetic)
        weibel = case_state("weibel", Float64; ppc = 4, t_end = 1)
        states = [("N = 1, Float64", weibel[1], weibel[2]),
                  (("N = 2, $(type_label(T))", state_2d(plasma_2d(T)), params(plasma_2d(T))) for T in (Float32, D3))...,
                  (("N = 2 with current, $(type_label(T))", state_2d(plasma_step_2d(T); current = Esirkepov()), params(plasma_step_2d(T)))
                   for T in (Float32, D3))...]
        for (label, st, p) in states
            step!(st, p, 1)
            r = JET.report_opt(step!, (typeof(st), typeof(p), Int); target_modules = targets)
            @test isempty(JET.get_reports(r))
            isempty(JET.get_reports(r)) || (println(label); show(stdout, MIME"text/plain"(), r))
        end
    end
else
    println("T-373, T-374 skipped: JET runs on released Julia versions only (this is $(VERSION))")
end
