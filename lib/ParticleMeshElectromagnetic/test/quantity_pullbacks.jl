# WP-077: the pullbacks of the module's deposited quantities (T-1065), which the engine's reverse
# rules of deposit! and deposit_flux! call through `ParticleMeshCore.pullback_quantity!`.

using ParticleMeshElectromagnetic: CurrentY, CurrentZ, KineticEnergy

@testset "T-1065 pullback_quantity! of the currents and the kinetic energy against duals, electrons and ions [A7, WP-077]" begin
    C = emC("T-1065")
    D4 = Dual{Nothing,Float64,4}
    # an electron and an ion of charge 2 and mass 7.5: the pullbacks carry q and m
    for (q, m) in ((-1, 1), (2, 7.5))
        np = 64
        s = Species(CPU(), Float64, np; q, m, N = 2)
        sD = Species(CPU(), D4, np; q, m, N = 2)
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
            s̄ = Species(CPU(), Float64, np; q, m, N = 2)
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
    # one large momentum along the current: ∂(u_K/γ)/∂u_K = (1 + Σ_{l≠K} u_l²)/γ³ is small and must not
    # come from a cancellation (duals cancel the same way, so this compares with BigFloat)
    for T in (Float32, Float64), (Q, K) in ((CurrentY(), 2), (CurrentZ(), 3)), big_u in (3000, 5000)
        s = Species(CPU(), T, 1; q = -1, m = 1, N = 2)
        u = ntuple(k -> T(k == K ? big_u : 0.25k), 3)
        s.u[1][1], s.u[2][1], s.u[3][1], s.w[1] = u..., T(0.75)
        s.status[1] = ParticleMeshCore.ALIVE
        s̄ = Species(CPU(), T, 1; q = -1, m = 1, N = 2)
        ParticleMeshCore.pullback_quantity!(s̄, Q, s, 1, one(T))
        ub = big.(u)
        γ = sqrt(1 + sum(abs2, ub))
        exact = -big(0.75) * (1 + sum(abs2, ub) - ub[K]^2) / γ^3
        r = record!("T-1065", T, Float64(abs(s̄.u[K][1] - exact) / (eps(T) * abs(exact))))
        r <= C || @info "T-1065 large momentum" T Q big_u s̄.u[K][1] exact r
        @test r <= C
    end
end
