# T-211: the relativistic Boris pusher against exact gyration and the E×B drift.
#
# A1 units: u = γv per unit mass in c, du/dt = (q/m)(E + v×B), Ω = |q|B/m. In a uniform B with
# E = 0, γ̄ (after the zero half kick) equals γ, so each step rotates u about B by exactly
# θ = 2·atan(ΩΔt/(2γ)) (the rotation by t = (q/m)(Δt/2)B/γ and s = 2t/(1 + t²)), in the sense of
# q·(v×B): counter-clockwise in (uₓ, u_y) for q < 0 and B along +z. In crossed uniform fields with
# E < B (c = 1) the guiding centre drifts at v_d = E×B/B² in the lab frame, relativistic or not;
# E = E₀ŷ, B = B₀ẑ give v_d = (E₀/B₀)x̂. The average over a finite run differs from v_d by the
# unfinished gyration, about r_L/(N·Δt); that relative error is measured and recorded.

using KernelAbstractions: @kernel, @index

@kernel function boris_steps_kernel!(U, @Const(u0s), E, Bf, h)
    j = @index(Global)
    u = u0s[j]
    @inbounds for s in axes(U, 1)
        u = push_momentum(Boris(), u, E, Bf, h)
        U[s, j] = u
    end
end

"""
    boris_steps(be, u0s, E, Bf, h, nsteps) -> Matrix

`U[s, j]`: the momentum after step `s` of `nsteps` Boris pushes from `u0s[j]` in the uniform
fields `E`, `Bf` with `h = (q/m)·Δt/2`. On the CPU the host loop, on another backend one work
item per start (the same sequence of `push_momentum` calls).
"""
function boris_steps(::CPU, u0s, E, Bf, h, nsteps)
    U = Matrix{eltype(u0s)}(undef, nsteps, length(u0s))
    for (j, u0) in enumerate(u0s)
        u = u0
        for s in 1:nsteps
            u = push_momentum(Boris(), u, E, Bf, h)
            U[s, j] = u
        end
    end
    return U
end
function boris_steps(be, u0s, E, Bf, h, nsteps)
    U = KernelAbstractions.allocate(be, eltype(u0s), nsteps, length(u0s))
    boris_steps_kernel!(be)(U, todev(be, u0s), E, Bf, h; ndrange = length(u0s))
    KernelAbstractions.synchronize(be)
    return Array(U)
end

function body_T211(be)
    C = emC("T-211")
    Cφ = emC("T-211", "C_phase")
    @testset "uniform Bz, $(type_label(T))" for T in TYPES
        B = basetype(T)
        qm = -one(B)                                        # electron, q/m = -1
        B0 = B(2)
        dt = B(1) / 10
        nsteps = 2000
        worstu = 0.0
        worstφ = 0.0
        u0vals = (B(1) / 10, B(1), B(10))
        E = (zero(T), zero(T), zero(T))
        Bf = (zero(T), zero(T), T(B0))
        U = boris_steps(be, [(seeded(T, u0), zero(T), zero(T)) for u0 in u0vals], E, Bf, T(qm * dt / 2), nsteps)
        for (j, u0) in enumerate(u0vals)
            γ = sqrt(1 + Float64(u0)^2)
            θ = 2 * atan(big(Float64(B0)) * big(Float64(dt)) / (2 * big(γ)))   # Ω = |q|B/m = B0
            u = (seeded(T, u0), zero(T), zero(T))
            umag0 = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
            for s in 1:nsteps
                u = U[s, j]
                umag = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
                worstu = max(worstu, absdiff(umag, umag0) / (base_eps(T) * absmax(umag0) * s))
                φ = atan(Float64(value_of(u[2])), Float64(value_of(u[1])))
                Δ = mod(φ - Float64(mod(s * θ, 2 * big(π))) + π, 2π) - π
                worstφ = max(worstφ, abs(Δ) / (base_eps(T) * s))
                @test u[3] === zero(T)
            end
        end
        record!("T-211 |u|", T, worstu, be)
        record!("T-211 phase", T, worstφ, be)
        @test worstu <= C
        @test worstφ <= Cφ
    end

    @testset "crossed E and B: the E×B drift, $(type_label(T))" for T in (Float32, Float64)
        E0, B0 = T(1) / 2, one(T)
        dt = T(1) / 20
        nsteps = 40_000
        x = zero(Float64)                                   # positions accumulated in Float64 on the host
        E = (zero(T), E0, zero(T))
        Bf = (zero(T), zero(T), B0)
        U = boris_steps(be, [(zero(T), zero(T), zero(T))], E, Bf, -dt / 2, nsteps)
        for s in 1:nsteps
            u = U[s, 1]
            x += Float64(dt) * Float64(u[1] / lorentz_factor(u))
        end
        vd = x / (nsteps * Float64(dt))
        rel = (vd - Float64(E0 / B0)) / Float64(E0 / B0)
        @info "T-211 E×B drift" T E0 B0 dt nsteps vd rel
        d = get!(MEASURED, "T-211 drift relative error", Dict{String,Float64}())
        d[be isa CPU ? type_label(T) : "$(type_label(T)) on $(backend_label(be))"] = abs(rel)
        @test abs(rel) <= emC("T-211", "drift_rel")
    end
end

@testset "T-211 Boris: exact gyration and the E×B drift [A2, A7, A8, WP-014]" begin
    body_T211(CPU())
end
