# WP-077: the pullbacks of the module's deposited quantities (T-1065), which the engine's reverse
# rules of deposit! and deposit_flux! call through `ParticleMeshCore.pullback_quantity!`.

using ParticleMeshElectromagnetic: CurrentY, CurrentZ, KineticEnergy

@testset "T-1065 pullback_quantity! of the currents and the kinetic energy against duals [A7, WP-077]" begin
    C = emC("T-1065")
    D4 = Dual{Nothing,Float64,4}
    np = 64
    s = Species(CPU(), Float64, np; q = -1, m = 1, N = 2)
    sD = Species(CPU(), D4, np; q = -1, m = 1, N = 2)
    rng = Xoshiro(1065)
    for i in 1:np
        # momenta from 10⁻³ to 30, so γ runs from 1 to the LWFA's tens, and both signs
        u = ntuple(_ -> (2rand(rng) - 1) * 10.0^(4rand(rng) - 3) * 3, 3)
        w = 0.5 + rand(rng)
        s.u[1][i], s.u[2][i], s.u[3][i], s.w[i] = u..., w
        sD.w[i] = Dual{Nothing}(w, 1.0, 0.0, 0.0, 0.0)
        for k in 1:3
            sD.u[k][i] = Dual{Nothing}(u[k], ntuple(l -> Float64(l == k + 1), 4)...)
        end
        s.status[i] = sD.status[i] = ParticleMeshCore.ALIVE
    end
    for Q in (CurrentY(), CurrentZ(), KineticEnergy())
        s̄ = Species(CPU(), Float64, np; q = -1, m = 1, N = 2)
        for i in 1:np
            q̄ = cos(0.7i)
            ParticleMeshCore.pullback_quantity!(s̄, Q, s, i, q̄)
            ∂ = q̄ .* Tuple(partials(Q(sD, i)))
            got = (s̄.w[i], s̄.u[1][i], s̄.u[2][i], s̄.u[3][i])
            scale = maximum(abs, ∂) + abs(q̄ * value(Q(sD, i)))
            r = record!("T-1065", Float64, maximum(abs.(got .- ∂)) / (eps(Float64) * scale))
            r <= C || @info "T-1065" Q i got ∂ r
            @test r <= C
        end
    end
end
