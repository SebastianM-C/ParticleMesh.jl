# T-240 to T-245: the quadratic-unitary pusher of WP-016 (Gordon and Hafizi, arXiv:2006.10146
# (2020), read in v1), the figure-8 scan of the five pushers over a₀ and ω₀Δt, and Float32
# rounding at weak rotation (#203). The conventions are those of the header of pushers.jl, whose
# helpers (PUSHERS, pname, norm3, orders, record_named!, the figure-8 fields) this file uses.
#
# The quadratic-unitary update (`quadratic_unitary_step`): with e = κE, h = κB, κ = (q/m)Δs/2,
# Ψ = e + ih and the spinor ζ = γ + σ·u, ζ′ = ΛζΛ† with Λ = (1 + σ·Ψ + Ψ·Ψ/4)/(1 − Ψ·Ψ/4)
# (their Eqs. 3, 4, 18), and Δs = (Δt/γ)(1 − (q/m)(E·u)Δt/(2γ²)) from their Eq. 22, so
# κ = (qm_h/γ)(1 − qm_h(E·u)/γ²) in the variables of `push_momentum`.
#
# T-241, uniform B, E = 0: Ψ = ih, Ψ·Ψ = −h², and Λ = ((1 − h²/4) + iσ·h)/(1 + h²/4) is the SU(2)
# element of a rotation about h by θ with cos(θ/2) = (1 − h²/4)/(1 + h²/4) and
# sin(θ/2) = |h|/(1 + h²/4), so tan(θ/4) = |h|/2 and
#     θ = 4·atan(|h|/2) = 4·atan(ΩΔt/(4γ)),
# since Δs = Δt/γ exactly at E = 0 and |h| = |qm_h|B/γ = ΩΔt/(2γ). The sense of rotation is the
# Lorentz force's, as for the other pushers. Boris turns by 2·atan(ΩΔt/(2γ)): both are ΩΔt/γ
# to first order, with third-order errors (ΩΔt/γ)³/48 and (ΩΔt/γ)³/12.
#
# T-241, the plane wave: the fields at the particle are E = (E_x, 0, 0), B = (0, E_x, 0), a null
# field with E×B along ẑ. Then Ψ·Ψ = 0, Λ = 1 + σ·Ψ is the exact operator, and tr((1 − σ_z)ζ)/2 =
# γ − u_z is preserved by every step (their Eq. 14) whatever Δs is: γ − u_z = 1 holds to roundoff
# along the discrete orbit, not to second order.
#
# T-242, crossed fields with E·B = 0: Ψ·Ψ = κ²(E² − B²) is real, Λ(2) = exp(a·σ·Ψ) with a real
# scalar a (Λ(2) is an even function of σ·Ψ times 1 plus an odd one, both real functions of Ψ·Ψ),
# so the step is the exact operator of the same field over a rescaled proper time. The four-
# velocity at E×B/B² is annihilated by the field tensor, so every such step leaves it unchanged.
# In uniform E, B = 0, Λ(2) is the Cayley form of the boost: the rapidity per step is
# 4·artanh(|e|/2) instead of 2|e|, and u is not exact.
#
# T-243, the figure-8 scan (WP-016): one period of the orbit of T-235 for a₀ ∈ {1, 2, 5, 10} and
# ω₀Δt ∈ {0.2, 0.1, 0.05, 0.025}, extended to 0.2/2⁷ for the orders of the orbit errors, Float64. Four relative errors, each defined in
# `figure8_scan_errors`: γ − u_z (exact value 1), u_x + qA_x (relative to a₀), the drift
# velocity over the period against a₀²/(4 + a₀²), and the position after one period against
# (0, 0, πa₀²/2), relative to πa₀²/2. Over one period the drift error is the z component of the
# position error. `scripts/pushers/resolution/scan.jl` runs the same errors on a finer grid.
#
# T-245, Float32 at weak rotation (#203): uniform B = Bẑ, E = 0, u ⊥ B, t² = (qm_h·B/γ)² =
# 6.25e-8 < eps(Float32) for both setups. The drift of |u| over N steps, |Δ|u||/(eps·|u₀|),
# grows like √N for unbiased rounding and like N for a biased one; the test checks it against
# C·√N at N = 10⁴ and at 4·10⁴, so a linear drift that passes at 10⁴ fails at 4·10⁴.

using ParticleMeshElectromagnetic: quadratic_unitary_step, dot3

const QU = QuadraticUnitary()
const PUSHERS5 = (PUSHERS..., QU)

@testset "T-240 the quadratic-unitary pusher in the slot: a Landau run, bitwise reproducible [A2, WP-016]" begin
    case = load_case("landau")
    prob = problem_from_case(case, Float64; ppc = 64, t_end = 1)
    digest(num) = (st = EMState(prob, num; backend); initialize!(st, prob); run!(st, params(prob), prob.nsteps); st)
    st = digest(numerics_from_case(case; pusher = QU))
    @test st.numerics.pusher === QU
    @test all(isfinite, st.hist.K) && all(isfinite, st.fields.Ex)
    @test state_digest(st) == state_digest(digest(numerics_from_case(case; pusher = QU)))
end

"""
    t241_checks(T) -> (; minkowski, energy, phase, planewave)

The worst ratios of T-241 in element type `T` (the header gives the closed forms); T-241 runs
Float64 and T-244 the A8 matrix.
"""
function t241_checks(::Type{T}) where {T}
    V = basetype(T)
    # Minkowski norm: (γ, u) carried by quadratic_unitary_step, γ never recomputed from u, in
    # random uniform fields (|E|, |B| ≤ 2), 200 steps of qm_h = −1/20 per sample
    rng = Xoshiro(20261102)
    unit() = (v = randn(rng, 3); Tuple(v ./ sqrt(sum(abs2, v))))
    qm_h = T(-V(1) / 20)
    minkowski = 0.0
    for γ0 in (1.1, 10.0, 100.0), _ in 1:20
        u = seeded.(T, V.(sqrt(γ0^2 - 1) .* unit()))
        E = T.(V.(2 .* rand(rng) .* unit()))
        B = T.(V.(2 .* rand(rng) .* unit()))
        γ = lorentz_factor(u)
        for s in 1:200
            ig = one(T) / γ
            κ = qm_h * ig * (one(T) - qm_h * dot3(E, u) * ig * ig)
            γ, u = quadratic_unitary_step(γ, u, κ .* E, κ .* B)
            minkowski = max(minkowski, absdiff(γ^2 - dot3(u, u), one(T)) / (base_eps(T) * absmax(γ)^2 * s))
        end
    end
    # energy and gyrophase for E = 0: the setup of T-231
    qm = -one(V)
    B0 = V(2)
    dt = V(1) / 10
    energy = 0.0
    phase = 0.0
    for u0 in (V(1) / 10, V(1), V(10)), along in (false, true)
        γ = sqrt(1 + big(Float64(u0))^2 * (along ? 5 // 4 : 1))
        β = big(Float64(B0)) * big(Float64(dt)) / 2
        θ = 4 * atan(β / (2γ))
        u = (seeded(T, u0), zero(T), along ? seeded(T, u0 / 2) : zero(T))
        umag0 = norm3(u)
        E = (zero(T), zero(T), zero(T))
        Bf = (zero(T), zero(T), T(B0))
        for s in 1:2000
            u = push_momentum(QU, u, E, Bf, T(qm * dt / 2))
            energy = max(energy, absdiff(norm3(u), umag0) / (base_eps(T) * absmax(umag0) * s))
            if !along
                φ = atan(Float64(value_of(u[2])), Float64(value_of(u[1])))
                Δ = mod(φ - Float64(mod(s * θ, 2 * big(π))) + π, 2π) - π
                phase = max(phase, abs(Δ) / (base_eps(T) * s))
            end
        end
    end
    # γ − u_z in the plane wave: one figure-8 period at Δt = 1/10, scale eps·γ_max·steps
    planewave = 0.0
    for a0v in (V(2), V(10))
        a0 = seeded(T, a0v)
        Ef(x, t) = (-a0 * cos(t - x[3]), zero(T), zero(T))
        Bf(x, t) = (zero(T), -a0 * cos(t - x[3]), zero(T))
        h = T(V(1) / 10)
        N = floor(Int, value_of(2 * T(π) * (1 + a0^2 / 4) / h))
        γmax = 1 + Float64(a0v)^2 / 2
        x = (zero(T), zero(T), zero(T))
        u = x
        for n in 1:N
            x, u = push_particle(QU, x, u, (n - 1) * h, h, -one(T), Ef, Bf)
            planewave = max(planewave, absdiff(lorentz_factor(u) - u[3], one(T)) / (base_eps(T) * γmax * n))
        end
    end
    return (; minkowski, energy, phase, planewave)
end

function t241_assert(T)
    r = t241_checks(T)
    for (k, v) in pairs(r)
        record!("T-241 $(k) QuadraticUnitary", T, v)
        @test v <= EMTOL["T-241"]["C_$(k)"]
    end
end

@testset "T-241 quadratic-unitary invariants: Minkowski norm, energy and gyrophase for E = 0, γ − u_z in the plane wave [A2, A7, A8, WP-016]" begin
    t241_assert(Float64)
end

@testset "T-242 quadratic-unitary in uniform E and in crossed fields [A2, A7, WP-016]" begin
    # T-233's setup: E = E₀ŷ, B = ẑ, started at E×B/B² (+ v∥ẑ)
    @testset "crossed fields, $(type_label(T))" for T in (Float32, Float64)
        dt = T(1) / 10
        worst = 0.0
        for E0 in T.((0.1, 0.5, 0.9, 0.99)), parallel in (false, true)
            vz = parallel ? sqrt(1 - E0^2) / 2 : zero(T)
            γ = 1 / sqrt(1 - E0^2 - vz^2)
            u = (γ * E0, zero(T), γ * vz)
            u0 = u
            vx = 0.0
            for s in 1:10_000
                u = push_momentum(QU, u, (zero(T), E0, zero(T)), (zero(T), zero(T), one(T)), -dt / 2)
                vx += Float64(u[1] / lorentz_factor(u))
                worst = max(worst, norm3(u .- u0) / (eps(T) * norm3(u0) * s))
            end
            rel = (vx / 10_000 - Float64(E0)) / Float64(E0)
            record_named!("T-242 drift relative error $(type_label(T)) Δt=0.1", "QuadraticUnitary E0=$(Float64(E0)) $(parallel ? "v∥" : "v⊥")", abs(rel))
            @test isfinite(rel)
        end
        record!("T-242 stays at E×B QuadraticUnitary", T, worst)
        @test worst <= EMTOL["T-242"]["C_exb"]
    end
    # T-232's setup: E = E₀x̂ with (q/m)E₀ = −1, u₀ = 1/2 along x, to t ≈ 10; u is not exact
    @testset "uniform E: u and x errors against Δt" begin
        a = -1.0
        u0 = 0.5
        γ0 = sqrt(1 + u0^2)
        xexact(t) = (sqrt(1 + (u0 + a * t)^2) - γ0) / a
        uerrs = Float64[]
        xerrs = Float64[]
        for dt in (0.2, 0.1, 0.05, 0.025)
            N = round(Int, 10 / dt)
            x = xexact(dt / 2)
            u = (u0, 0.0, 0.0)
            for _ in 1:N
                u = push_momentum(QU, u, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), -dt / 2)
                x += dt * u[1] / lorentz_factor(u)
            end
            push!(uerrs, abs(u[1] - (u0 + a * N * dt)) / abs(u0 + a * N * dt))
            push!(xerrs, abs(x - xexact((N + 0.5) * dt)))
        end
        @info "T-242 uniform E, QuadraticUnitary: relative u error and x error at t ≈ 10 for Δt = 0.2 … 0.025" uerrs orders(uerrs) xerrs orders(xerrs)
        record_named!("T-242 uniform E relative u error at Δt = 0.025", "QuadraticUnitary", uerrs[end])
        record_named!("T-242 uniform E u error order, smallest pair", "QuadraticUnitary", orders(uerrs)[end])
        record_named!("T-242 uniform E x error at Δt = 0.025", "QuadraticUnitary", xerrs[end])
        record_named!("T-242 uniform E x error order, smallest pair", "QuadraticUnitary", orders(xerrs)[end])
        @test all(isfinite, uerrs) && all(isfinite, xerrs)
    end
end

"""
    figure8_scan_errors(P, a0, dt) -> (; gamma, canonical, drift, position, steps)

One period of the figure-8 orbit (header of pushers.jl) stepped with `push_particle`: the
largest |γ − u_z − 1| and |u_x − a₀ sin η|/a₀ over the steps, and at t = T_p (N steps of Δt and
one of T_p − NΔt) the drift error |z/T_p − v_d|/v_d with v_d = a₀²/(4 + a₀²) and the position
error |x − (0, 0, πa₀²/2)|/(πa₀²/2). All relative, in Float64 from the state of type T.
"""
function figure8_scan_errors(P, a0::T, dt::T) where {T}
    E(x, t) = (-a0 * cos(t - x[3]), zero(T), zero(T))
    B(x, t) = (zero(T), -a0 * cos(t - x[3]), zero(T))
    Tp = 2 * T(π) * (1 + a0^2 / 4)
    N = floor(Int, Tp / dt)
    x = (zero(T), zero(T), zero(T))
    u = x
    eh = 0.0
    ep = 0.0
    for n in 1:N
        x, u = push_particle(P, x, u, (n - 1) * dt, dt, -one(T), E, B)
        eh = max(eh, abs(Float64(lorentz_factor(u) - u[3] - 1)))
        ep = max(ep, abs(Float64(u[1] - a0 * sin(n * dt - x[3]))))
    end
    x, u = push_particle(P, x, u, N * dt, Tp - N * dt, -one(T), E, B)
    zp = Float64(T(π) * a0^2 / 2)
    vd = Float64(a0^2 / (4 + a0^2))
    drift = abs(Float64(x[3]) / Float64(Tp) - vd) / vd
    position = norm3((Float64(x[1]), Float64(x[2]), Float64(x[3]) - zp)) / zp
    return (; gamma = eh, canonical = ep / Float64(a0), drift, position, steps = N + 1)
end

const SCAN_A0 = (1.0, 2.0, 5.0, 10.0)
const SCAN_DT = (0.2, 0.1, 0.05, 0.025)
# the drift and the position error reach order 2 only below a₀·ω₀Δt ≈ 1/16 for the four
# Boris-type pushers (at a₀ = 10, from ω₀Δt ≈ 0.006 on), so their orders are asserted on Δt
# extended to 0.2/2⁷ where a₀·ω₀Δt ≤ 1/16 (WP-016 assumed 1/2; reports/pushers.md)
const SCAN_DT_EXT = ntuple(i -> 0.2 / 2^(i - 1), 8)
const SCAN_INVARIANTS = (:gamma, :canonical)
const SCAN_ORBIT = (:drift, :position)

@testset "T-243 the figure-8 scan over a₀ and ω₀Δt: every error falls with Δt, second order in the asymptotic range [A1, A2, A7, WP-016]" begin
    @testset "$(pname(P)), a₀ = $a0" for P in PUSHERS5, a0 in SCAN_A0
        res = [figure8_scan_errors(P, a0, dt) for dt in SCAN_DT_EXT]
        for k in (SCAN_INVARIANTS..., SCAN_ORBIT...)
            e = [getfield(r, k) for r in res]
            for (dt, v) in zip(SCAN_DT_EXT, e)
                record_named!("T-243 $k relative error", "$(pname(P)) a0=$a0 dt=$dt a0·dt=$(a0 * dt)", v)
            end
            if P isa QuadraticUnitary && k === :gamma
                # exact in the null field: roundoff, no order (header); the bound is T-241's scale
                bound = EMTOL["T-241"]["C_planewave"] * eps(Float64) * (1 + a0^2 / 2) * res[end].steps
                @test maximum(e) <= bound
                continue
            end
            o = orders(e)
            for i in eachindex(o)
                record_named!("T-243 $k order", "$(pname(P)) a0=$a0 dt=$(SCAN_DT_EXT[i])→$(SCAN_DT_EXT[i + 1])", o[i])
            end
            @test all(isfinite, e)
            @test all(>(0), o[1:(length(SCAN_DT) - 1)])          # falls over the WP's Δt
            if k in SCAN_INVARIANTS
                # the WP's Δt, pairs whose larger step has a₀·ω₀Δt ≤ 1/2
                asym = [i for i in 1:(length(SCAN_DT) - 1) if a0 * SCAN_DT_EXT[i] <= 0.5 + 1e-12]
                key, tol = "T-243 |order − 2|, invariants, a₀·ω₀Δt ≤ 1/2", EMTOL["T-243"]["order_tol"]
            else
                asym = [i for i in eachindex(o) if a0 * SCAN_DT_EXT[i] <= 1 / 16 + 1e-12]
                key, tol = "T-243 |order − 2|, drift and position, a₀·ω₀Δt ≤ 1/16", EMTOL["T-243"]["order_tol_orbit"]
            end
            dev = maximum(abs.(o[asym] .- 2))
            record_named!(key, "$(pname(P)) $k", dev)
            @test dev <= tol
        end
    end
end

@testset "T-244 quadratic-unitary invariants in Float32, Float64 and Dual{Float32,3} [A8, WP-016]" begin
    @testset "$(type_label(T))" for T in TYPES
        t241_assert(T)
    end
end

@testset "T-245 Float32 at weak rotation: |u| drifts like unbiased rounding, t² < eps(Float32) [A2, A7, A8, WP-016]" begin
    # #203's reproducer and |u| = 10 with B = 0.05, both at qm_h = −0.05 (t² = 6.25e-8)
    setups = (("|u| = 10⁴, B = 50", 1.0f4, 50.0f0), ("|u| = 10, B = 0.05", 10.0f0, 0.05f0))
    @testset "$(pname(P)), $label" for P in PUSHERS5, (label, um, Bz) in setups
        u = (um, 0.0f0, 0.0f0)
        E = (0.0f0, 0.0f0, 0.0f0)
        B = (0.0f0, 0.0f0, Bz)
        t = -0.05f0 * Bz / lorentz_factor(u)
        @test t^2 < eps(Float32)
        m0 = norm3(Float64.(u))
        worst = 0.0
        for n in 1:40_000
            u = push_momentum(P, u, E, B, -0.05f0)
            if n == 10_000 || n == 40_000
                r = abs(norm3(Float64.(u)) - m0) / (eps(Float32) * m0 * sqrt(n))
                record_named!("T-245 |Δ|u||/(eps·|u₀|·√N) at N = $n", "$(pname(P)) $label", r)
                worst = max(worst, r)
            end
        end
        record!("T-245 $(pname(P))", Float32, worst)
        @test worst <= EMTOL["T-245"]["C"]
    end
end
