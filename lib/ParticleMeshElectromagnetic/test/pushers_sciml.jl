# T-237: the pushers through SciML (the OrdinaryDiffEqCore extension). A `particle_problem` is a
# plain ODEProblem with the true right-hand side `lorentz_rhs!`; `pusher_algorithm(P)` steps it
# with `push_particle`, and `Vern9` integrates the same problem. The figure-8 conventions are those
# of the header of pushers.jl (an electron at rest at η = 0 in A_x = a₀ sin(t − z)).
# The element-type matrix of T-231, T-234 and T-235 (Float32, Float64, Dual{Float32,3}, at
# a₀ = 2 for T-235) runs in pushers.jl; the duals through the exact rotation near B = 0 are here.

using SciMLBase: SciMLBase, solve
using OrdinaryDiffEqCore: OrdinaryDiffEqCore        # loads the pusher extension
using OrdinaryDiffEqVerner: Vern9

figure8_fields(a0) = ((x, t) -> (-a0 * cos(t - x[3]), zero(a0), zero(a0)),
                      (x, t) -> (zero(a0), -a0 * cos(t - x[3]), zero(a0)))

@testset "T-237 the SciML interface [A7, A8, A10, WP-015]" begin
    @testset "solve reproduces push_particle bit for bit, $(pname(P)), $(type_label(T))" for P in PUSHERS, T in TYPES
        V = basetype(T)
        a0 = seeded(T, V(2))
        E, B = figure8_fields(a0)
        dt = V(1) / 16                                    # binary: t = k·dt is exact, no last-step clamp
        nsteps = 160
        prob = particle_problem((zero(T), zero(T), zero(T)), (zero(T), zero(T), zero(T)), (zero(V), nsteps * dt);
                                qm = -one(T), E, B)
        sol = solve(prob, pusher_algorithm(P); dt)
        @test SciMLBase.successful_retcode(sol)
        @test length(sol.t) == nsteps + 1
        x = (zero(T), zero(T), zero(T))
        u = x
        t = zero(V)
        ok = true
        for k in 1:nsteps
            x, u = push_particle(P, x, u, convert(T, t), convert(T, dt), -one(T), E, B)
            t += dt
            ok &= sol.t[k + 1] === t
            ok &= same_bits(sol.u[k + 1], T[x..., u...])
        end
        @test ok
    end

    @testset "Vern9 through the same problem matches the closed form" begin
        a0 = 2.0
        E, B = figure8_fields(a0)
        Tp = 2π * (1 + a0^2 / 4)
        prob = particle_problem((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, Tp); qm = -1.0, E, B)
        tol = 1e-12
        sol = solve(prob, Vern9(); abstol = tol, reltol = tol)
        s = sol.u[end]
        exact = [0.0, 0.0, π * a0^2 / 2, 0.0, 0.0, 0.0]
        err = maximum(abs, s .- exact)
        # the invariants along the dense output
        inv = maximum(range(0, Tp; length = 1001)) do t
            v = sol(t)
            γ = sqrt(1 + v[4]^2 + v[5]^2 + v[6]^2)
            max(abs(γ - v[6] - 1), abs(v[4] - a0 * sin(t - v[3])))
        end
        @info "T-237 Vern9 figure-8 at a₀ = 2, tolerance $tol" err inv
        record_named!("T-237 Vern9 closure error / tolerance", "a0=2", err / tol)
        record_named!("T-237 Vern9 invariant error / tolerance", "a0=2", inv / tol)
        @test err <= EMTOL["T-237"]["vern9"] * tol
        @test inv <= EMTOL["T-237"]["vern9"] * tol
    end

    @testset "duals through BorisExactRotation near B = 0" begin
        # At B = 0 the exact rotation and Boris have the same derivative in B, u⁻×(2qm_h/γ)·∂B
        # (both rotations are u⁻ + u⁻×w + O(w²)); the series branch and the trigonometric branch
        # agree across the switch at θ² = √eps(T).
        D = D3
        z = zero(D)
        qm_h = D(-1.0f0 / 20)
        u = (D(1.0f0), D(0.5f0), D(-0.25f0))
        E = (D(0.1f0), z, z)
        for Bv in ((z, z, mkdual(0, 1, 0, 0)), (mkdual(0, 0, 1, 0), z, mkdual(0, 0, 0, 1)))
            ue = push_momentum(BorisExactRotation(), u, E, Bv, qm_h)
            ub = push_momentum(Boris(), u, E, Bv, qm_h)
            @test all(c -> all(isfinite, components(c)), ue)
            @test maximum(i -> absdiff(ue[i], ub[i]), 1:3) <= 4 * eps(Float32)
        end
        # across the switch: θ² just below and just above √eps(T), against BigFloat
        for T in (Float32, Float64)
            xc = sqrt(eps(T))
            for x in (prevfloat(xc), xc, nextfloat(xc), 2xc, 4xc)
                c, s, k = ParticleMeshElectromagnetic.rotation_coefficients(x)
                θ = sqrt(big(x))
                @test abs(c - cos(θ)) <= 2 * eps(T)
                @test abs(s - sin(θ) / θ) <= 2 * eps(T)
                @test abs(k - (1 - cos(θ)) / θ^2) <= 2 * eps(T)
            end
            # and the derivative of the coefficients is finite at θ² = 0
            d = ForwardDiff.derivative(x -> ParticleMeshElectromagnetic.rotation_coefficients(x)[2], zero(T))
            @test d ≈ -T(1) / 6
        end
    end
end
