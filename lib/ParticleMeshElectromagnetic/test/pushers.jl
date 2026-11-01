# T-230 to T-236: the four pushers of WP-015 (Boris, Vay, Higuera–Cary, exact-rotation Boris)
# against exact orbits and the structural properties the literature claims
# (docs/plan/pushers-literature.md). Single particles in prescribed fields, no mesh, except T-230.
#
# Conventions (A1 units, c = 1): u = γv per unit mass, du/dt = (q/m)(E + v×B), Ω = |q|B/m,
# qm_h = (q/m)Δt/2 and τ = qm_h·B in the arguments of `push_momentum`. The leapfrog has u at
# integer steps and x at half steps, uⁿ⁺¹ = push(uⁿ, E(xⁿ⁺¹ᐟ²), B(xⁿ⁺¹ᐟ²)), xⁿ⁺³ᐟ² = xⁿ⁺¹ᐟ² + Δt·uⁿ⁺¹/γ;
# `push_particle` is the same scheme written with two half drifts, so x and u come out
# synchronized at integer times, xⁿ = (xⁿ⁻¹ᐟ² + xⁿ⁺¹ᐟ²)/2.
#
# T-231, uniform B, E = 0, u ⊥ B: the electric kicks vanish, γ is conserved, and each step rotates
# u about B by a fixed angle. Boris rotates by the Cayley transform of t = τ/γ, the angle
# 2·atan(|t|) = 2·atan(ΩΔt/(2γ)). Vay solves u⁺ = u + u×t + u⁺×t with the γ of Higuera and Cary's
# Eq. 20 applied to u′ = u + u×t; for u ⊥ B that γ is γ itself (σ = γ² − τ²/γ², so
# (σ + √(σ² + 4τ²))/2 = γ²), and the map is the Boris Cayley rotation, as Vay's App. B states.
# Higuera–Cary rotate by 2·atan(|τ|/γ_new) with γ_new² = (σ + √(σ² + 4|τ|²))/2, σ = γ² − |τ|²
# (u⁻·τ = 0); the exact rotation turns by θ = ΩΔt/γ.
#
# T-232, uniform E = E₀x̂, B = 0, start at u₀ along x: all four pushers reduce to u⁺ = u + 2qm_h·E,
# so uⁿ = u₀ + (q/m)E₀·nΔt exactly in exact arithmetic. The exact position for a = (q/m)E₀ is
# x(t) = (γ(t) − γ₀)/a with γ(t) = √(1 + (u₀ + at)²). Two starts of the stagger: x¹ᐟ² from the
# exact solution, and x¹ᐟ² = x₀ + (Δt/2)·u₀/γ₀ (a start at t = 0, the first half drift of
# `push_particle`).
#
# T-233, crossed fields E = E₀ŷ, B = ẑ (B₀ = 1), E₀ < 1: v_d = E×B/B² = E₀x̂, and a particle with
# v = v_d + v_∥ẑ feels no force (v×B = E₀ x̂×ẑ = −E₀ŷ). Vay and Higuera–Cary keep it there; for
# Boris and the exact rotation the mean velocity over the run is recorded against v_d.
#
# T-234, the Jacobian of the momentum map at fixed E and B. Boris, Higuera–Cary and the exact
# rotation are kicks (determinant 1) around a rotation about τ whose angle depends only on
# rotation invariants of u⁻, so the determinant is 1. Vay solves F(u⁺) = G(u⁻) with
# F(u) = u − u×τ/γ(u) and G(u) = u + u×τ/γ(u) + 2qm_h·E, so det(∂u⁺/∂u⁻) = det G′(u⁻)/det F′(u⁺).
# With t = τ/γ and ∂(u×τ/γ)/∂u = −[t]ₓ − (u×t)uᵀ/γ², the matrix determinant lemma and
# (I ∓ [t]ₓ)⁻¹ = (I ± [t]ₓ + ttᵀ)/(1 + t²) give, for either sign,
#     J(u) = 1 + t² − (u²t² − (u·t)²)/γ² = 1 + (τ² + (u·τ)²)/γ⁴,
# so det = J(u⁻)/J(u⁺): Higuera and Cary's Eqs. 40–41, with their β = qBΔt/(2m) (Eq. 16) our τ
# (checked against arXiv:1701.05605v1). Over N steps in constant fields the product telescopes to
# J(u⁰)/J(uᴺ) (their Eq. 42).
#
# T-235, the figure-8 orbit (Sarachik and Schappert, Phys. Rev. D 1, 2738 (1970)): an electron
# (q/m = −1) in the plane wave A_x = a₀ sin η, η = t − z, so E_x = −∂ₜA_x = −a₀ cos η and
# B_y = ∂_z A_x = −a₀ cos η = E_x. Along any orbit d(γ − u_z)/dt = (q/m)v_x(E_x − B_y) = 0 and
# d(u_x + (q/m)A_x)/dt = 0. Starting at rest at η = 0, a zero of A_x: γ − u_z = 1,
# u_x = a₀ sin η, u_z = u_x²/2, dη/dt = 1/γ, so η is the proper time and
#     x = a₀(1 − cos η),  z = (a₀²/4)(η − sin(2η)/2),  t = η + z.
# One period, Δη = 2π, takes T_p = 2π(1 + a₀²/4) and moves z by πa₀²/2; x and u return to 0. The
# closure error is the distance of the state at t = T_p (N steps of Δt and one last step of
# T_p − NΔt) from that point. The invariants are evaluated at the synchronized zⁿ.

const PUSHERS = (Boris(), Vay(), HigueraCary(), BorisExactRotation())
pname(p) = string(nameof(typeof(p)))
"The per-pusher constant of a test, from tolerances.toml (a table keyed by the pusher's name)."
pC(id, p, key = "C") = EMTOL[id][key][pname(p)]

norm3(a) = sqrt(a[1]^2 + a[2]^2 + a[3]^2)
"Observed orders of a sequence of errors at Δt halving each time."
orders(errs) = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

function record_named!(id, key, v)
    d = get!(MEASURED, id, Dict{String,Float64}())
    d[key] = max(get(d, key, 0.0), Float64(v))
    return v
end

@testset "T-230 the pusher slot: a Landau run with each pusher, bitwise reproducible [A2, A4, A10, WP-015]" begin
    case = load_case("landau")
    prob = problem_from_case(case, Float64; ppc = 64, t_end = 1)
    digest(num) = (st = EMState(prob, num; backend); initialize!(st, prob); run!(st, params(prob), prob.nsteps); st)
    ref = digest(numerics_from_case(case))                 # WP-014's numerics: Boris by default
    for P in PUSHERS
        st = digest(numerics_from_case(case; pusher = P))
        @test st.numerics.pusher === P
        @test all(isfinite, st.hist.K) && all(isfinite, st.fields.Ex)
        @test state_digest(st) == state_digest(digest(numerics_from_case(case; pusher = P)))   # Reference mode, rerun
        P isa Boris && @test state_digest(st) == state_digest(ref)
    end
end

@testset "T-231 uniform B: |u| and the gyrophase per step [A2, A7, A8, WP-015]" begin
    @testset "$(pname(P)), $(type_label(T))" for P in PUSHERS, T in TYPES
        V = basetype(T)
        qm = -one(V)
        B0 = V(2)
        dt = V(1) / 10
        nsteps = 2000
        worstu = 0.0
        worstφ = 0.0
        for u0 in (V(1) / 10, V(1), V(10)), along in (false, true)
            γ = sqrt(1 + big(Float64(u0))^2 * (along ? 5 // 4 : 1))
            β = big(Float64(B0)) * big(Float64(dt)) / 2        # |τ| = |qm_h|·B
            θ = if P isa Boris || P isa Vay
                2 * atan(β / γ)
            elseif P isa BorisExactRotation
                2β / γ
            else
                σ = γ^2 - β^2
                2 * atan(β / sqrt((σ + sqrt(σ^2 + 4β^2)) / 2))
            end
            u = (seeded(T, u0), zero(T), along ? seeded(T, u0 / 2) : zero(T))
            umag0 = norm3(u)
            E = (zero(T), zero(T), zero(T))
            Bf = (zero(T), zero(T), T(B0))
            for s in 1:nsteps
                u = push_momentum(P, u, E, Bf, T(qm * dt / 2))
                worstu = max(worstu, absdiff(norm3(u), umag0) / (base_eps(T) * absmax(umag0) * s))
                if !along
                    φ = atan(Float64(value_of(u[2])), Float64(value_of(u[1])))
                    Δ = mod(φ - Float64(mod(s * θ, 2 * big(π))) + π, 2π) - π
                    worstφ = max(worstφ, abs(Δ) / (base_eps(T) * s))
                    @test u[3] === zero(T)
                end
            end
        end
        record!("T-231 |u| $(pname(P))", T, worstu)
        record!("T-231 phase $(pname(P))", T, worstφ)
        @test worstu <= pC("T-231", P)
        @test worstφ <= pC("T-231", P, "C_phase")
    end

    @testset "Eq. 20 γ against BigFloat, |τ| up to 10³ γ" begin
        # Vay and Higuera–Cary share eq20_gamma; for |τ| > γ the direct form of Eq. 20 cancels
        # (σ < 0), so the code switches to the rationalised form there
        eq20 = ParticleMeshElectromagnetic.eq20_gamma
        @testset "$(type_label(T))" for T in TYPES
            V = basetype(T)
            worst = 0.0
            for τm in (1 // 2, 3, 10, 100, 1000), along in (0, 1 // 2)
                u = (seeded(T, V(3) / 10), zero(T), T(V(along)))
                τ = (zero(T), T(V(τm) / 7), T(V(τm)))
                ex = eq20(big.(Float64.(value_of.(u))), big.(Float64.(value_of.(τ))))
                g = eq20(u, τ)
                @test all(isfinite, components(g))
                worst = max(worst, Float64(abs(value_of(g) - ex) / ex) / base_eps(T))
            end
            record!("T-231 Eq. 20 γ", T, worst)
            @test worst <= EMTOL["T-231"]["C_eq20"]
        end
    end
end

@testset "T-232 uniform E, B = 0: exact u, recorded position error [A1, A2, A7, WP-015]" begin
    @testset "u exact, $(pname(P)), $(type_label(T))" for P in PUSHERS, T in TYPES
        V = basetype(T)
        qm = -one(V)
        E0 = one(V)
        dt = V(1) / 10
        nsteps = 200                                      # to u = −20, γ ≈ 20
        u0 = seeded(T, V(1) / 2)
        u = (u0, zero(T), zero(T))
        E = (T(E0), zero(T), zero(T))
        Bf = (zero(T), zero(T), zero(T))
        worst = 0.0
        for n in 1:nsteps
            u = push_momentum(P, u, E, Bf, T(qm * dt / 2))
            exact = u0 + T(qm * E0) * (n * T(dt))
            scale = absmax(u0) + n * Float64(dt * E0)       # the largest |u| so far
            worst = max(worst, absdiff(u[1], exact) / (base_eps(T) * scale * n))
            @test u[2] === zero(T) && u[3] === zero(T)
        end
        record!("T-232 u $(pname(P))", T, worst)
        @test worst <= pC("T-232", P)
    end

    @testset "position error and its order, $(pname(P))" for P in PUSHERS
        a = -1.0                                          # (q/m)E₀
        u0 = 0.5
        γ0 = sqrt(1 + u0^2)
        xexact(t) = (sqrt(1 + (u0 + a * t)^2) - γ0) / a
        tend = 10.0
        for start in (:exact, :t0)
            errs = Float64[]
            for dt in (0.2, 0.1, 0.05, 0.025)
                N = round(Int, tend / dt)
                x = start === :exact ? xexact(dt / 2) : dt / 2 * u0 / γ0
                u = (u0, 0.0, 0.0)
                for _ in 1:N
                    u = push_momentum(P, u, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), -dt / 2)
                    x += dt * u[1] / lorentz_factor(u)
                end
                push!(errs, abs(x - xexact((N + 0.5) * dt)))
            end
            o = orders(errs)
            @info "T-232 position error ($(pname(P)), x½ from $start)" errs o
            record_named!("T-232 x error at Δt = 0.025 ($start)", pname(P), errs[end])
            record_named!("T-232 x order, smallest pair ($start)", pname(P), o[end])
            @test all(isfinite, errs)
        end
    end
end

@testset "T-233 crossed fields: the E×B drift [A2, A7, WP-015]" begin
    @testset "$(pname(P)), $(type_label(T))" for P in PUSHERS, T in (Float32, Float64)
        dt = T(1) / 10
        nsteps = 10_000
        worst = 0.0
        for E0 in T.((0.1, 0.5, 0.9, 0.99)), parallel in (false, true)
            vz = parallel ? sqrt(1 - E0^2) / 2 : zero(T)
            γ = 1 / sqrt(1 - E0^2 - vz^2)
            u = (γ * E0, zero(T), γ * vz)
            u0 = u
            E = (zero(T), E0, zero(T))
            Bf = (zero(T), zero(T), one(T))
            vx = 0.0                                       # mean vₓ, accumulated in Float64 on the host
            for s in 1:nsteps
                u = push_momentum(P, u, E, Bf, -dt / 2)
                vx += Float64(u[1] / lorentz_factor(u))
                (P isa Vay || P isa HigueraCary) && (worst = max(worst, norm3(u .- u0) / (eps(T) * norm3(u0) * s)))
            end
            rel = (vx / nsteps - Float64(E0)) / Float64(E0)
            record_named!("T-233 drift relative error $(type_label(T)) Δt=0.1", "$(pname(P)) E0=$(Float64(E0)) $(parallel ? "v∥" : "v⊥")", abs(rel))
            @test isfinite(rel)
        end
        if P isa Vay || P isa HigueraCary
            record!("T-233 stays at E×B $(pname(P))", T, worst)
            @test worst <= pC("T-233", P)
        end
    end

    @testset "drift error against ΩΔt, $(pname(P))" for P in (Boris(), BorisExactRotation())
        E0 = 0.1
        errs = Float64[]
        for dt in (0.4, 0.2, 0.1)
            nsteps = round(Int, 4000 / dt)
            u = (E0 / sqrt(1 - E0^2), 0.0, 0.0)
            vx = 0.0
            for _ in 1:nsteps
                u = push_momentum(P, u, (0.0, E0, 0.0), (0.0, 0.0, 1.0), -dt / 2)
                vx += u[1] / lorentz_factor(u)
            end
            push!(errs, abs(vx / nsteps - E0) / E0)
        end
        @info "T-233 drift error at E0 = 0.1 against ΩΔt = 0.4, 0.2, 0.1 ($(pname(P)))" errs orders(errs)
        record_named!("T-233 drift error order in ΩΔt at E0 = 0.1", pname(P), orders(errs)[end])
        @test all(isfinite, errs)
    end
end

widen64(x::AbstractFloat) = Float64(x)
widen64(d::Dual{Nothing}) = Dual{Nothing}(Float64(value(d)), Float64.(Tuple(partials(d)))...)
"Higuera and Cary's volume factor J(u) = 1 + (τ² + (u·τ)²)/γ⁴ of Vay's map (header)."
vay_J(u, τ) = 1 + (norm3(τ)^2 + (u[1] * τ[1] + u[2] * τ[2] + u[3] * τ[3])^2) / lorentz_factor(u)^4

@testset "T-234 phase-space volume: det ∂u′/∂u [A2, A7, A8, WP-015]" begin
    rng = Xoshiro(20261101)
    unit() = (v = randn(rng, 3); Tuple(v ./ sqrt(sum(abs2, v))))
    samples = [(γ, unit(), 2 .* rand(rng) .* unit(), 2 .* rand(rng) .* unit()) for γ in (1.1, 10.0, 100.0, 1000.0) for _ in 1:50]
    @testset "$(pname(P)), $(type_label(T))" for P in PUSHERS, T in TYPES
        V = basetype(T)
        qm_h = T(-V(1) / 20)
        worst = Dict{Float64,Float64}()
        for (γ, d, Ev, Bv) in samples
            u = seeded.(T, V.(sqrt(γ^2 - 1) .* d))
            E = T.(V.(Ev))
            B = T.(V.(Bv))
            Jm = ForwardDiff.jacobian(collect(u)) do v
                S = eltype(v)
                collect(push_momentum(P, (v[1], v[2], v[3]), S.(E), S.(B), S(qm_h)))
            end
            dJ = det(widen64.(Jm))
            target = if P isa Vay
                τ = widen64.(qm_h .* B)
                vay_J(widen64.(u), τ) / vay_J(widen64.(push_momentum(P, u, E, B, qm_h)), τ)
            else
                one(dJ)
            end
            r = absdiff(dJ, target) / (base_eps(T) * absmax(target))
            worst[γ] = max(get(worst, γ, 0.0), r)
        end
        for (γ, r) in sort(collect(worst))
            record!("T-234 $(pname(P)) γ=$(Int(round(γ, RoundDown)))", T, r)
            @test r <= pC("T-234", P)
        end
    end

    @testset "Vay: the N-step product telescopes, $(type_label(T))" for T in (Float32, Float64)
        u = T.((3, -1, 2))
        E = T.((1 // 2, 1 // 5, -1 // 3))
        B = T.((0, 1 // 3, 1))
        qm_h = -T(1) / 20
        u0 = u
        prod = 1.0
        for _ in 1:1000
            Jm = ForwardDiff.jacobian(v -> collect(push_momentum(Vay(), (v[1], v[2], v[3]), eltype(v).(E), eltype(v).(B), eltype(v)(qm_h))), collect(u))
            prod *= det(Float64.(Jm))
            u = push_momentum(Vay(), u, E, B, qm_h)
        end
        τ = Float64.(qm_h .* B)
        tele = vay_J(Float64.(u0), τ) / vay_J(Float64.(u), τ)
        @info "T-234 Vay over 1000 steps" T prod tele
        record_named!("T-234 Vay 1000-step product: |prod − J(u⁰)/J(uᴺ)|/eps", type_label(T), abs(prod - tele) / eps(T))
        record_named!("T-234 Vay 1000-step product: |prod − 1|", type_label(T), abs(prod - 1))
        @test isapprox(prod, tele; rtol = 1000 * eps(T))
    end
end

# The figure-8 run of one period: the errors in γ − u_z and in u_x + (q/m)A_x of largest value
# over the period, and the closure error component of largest value, all of type T.
function figure8_errors(P, a0::T, dt::T) where {T}
    E(x, t) = (-a0 * cos(t - x[3]), zero(T), zero(T))
    B(x, t) = (zero(T), -a0 * cos(t - x[3]), zero(T))
    qm = -one(T)
    Tp = 2 * T(π) * (1 + a0^2 / 4)
    N = floor(Int, value_of(Tp / dt))
    x = (zero(T), zero(T), zero(T))
    u = (zero(T), zero(T), zero(T))
    eh = zero(T)
    ep = zero(T)
    errh(u, x, t) = lorentz_factor(u) - u[3] - 1
    errp(u, x, t) = u[1] - a0 * sin(t - x[3])
    larger = (a, b) -> abs(value_of(a)) >= abs(value_of(b)) ? a : b
    for n in 1:N
        x, u = push_particle(P, x, u, (n - 1) * dt, dt, qm, E, B)
        eh = larger(eh, errh(u, x, n * dt))
        ep = larger(ep, errp(u, x, n * dt))
    end
    x, u = push_particle(P, x, u, N * dt, Tp - N * dt, qm, E, B)
    zp = T(π) * a0^2 / 2
    ec = foldl(larger, (x[1], x[2], x[3] - zp, u[1], u[2], u[3]))
    return eh, ep, ec
end

@testset "T-235 the figure-8 orbit: second order in γ − u_z, u_x + qA_x and closure [A1, A2, A7, WP-015]" begin
    # the WP's ω₀Δt and one more for the closure: the leading closure error of Vay at a₀ = 1/2
    # changes sign near Δt ≈ 0.07, so its order over the WP's range is pre-asymptotic (0.0125 is
    # where it reaches 2); the invariants are asserted over the WP's range, every order recorded
    dts = (0.2, 0.1, 0.05, 0.025, 0.0125)
    configs = [(Float64, 0.5), (Float64, 2.0), (Float32, 2.0), (D3, 2.0)]
    @testset "$(pname(P)), a₀ = $a0, $(type_label(T))" for P in PUSHERS, (T, a0) in configs
        V = basetype(T)
        errs = [figure8_errors(P, seeded(T, V(a0)), T(V(dt))) for dt in dts]
        for (k, what) in enumerate(("γ − u_z", "u_x + qA_x", "closure"))
            e = [abs(Float64(value_of(x[k]))) for x in errs]
            o = orders(e)
            key = "$(pname(P)) a0=$a0 $(type_label(T))"
            record_named!("T-235 $what error constant e/Δt² at Δt = 0.025", key, e[4] / 0.025^2)
            record_named!("T-235 $what |order − 2| over the WP's Δt", key, maximum(abs.(o[1:3] .- 2)))
            if what == "closure"
                # Float64: the pair 0.025 → 0.0125; Float32 reaches its roundoff floor at 0.0125, so
                # there the pair 0.05 → 0.025
                k′ = basetype(T) === Float64 ? 4 : 3
                record_named!("T-235 closure |order − 2| on the asserted pair", key, abs(o[k′] - 2))
                @test abs(o[k′] - 2) <= EMTOL["T-235"]["order_tol_closure"]
            else
                @test maximum(abs.(o[1:3] .- 2)) <= EMTOL["T-235"]["order_tol"]
            end
            if T === D3
                # the derivatives of the errors with respect to a₀ are finite and fall with Δt, but
                # not at a clean order: the last step T_p − NΔt depends on a₀ (recorded only)
                ∂ = [maximum(abs, Float64.(Tuple(partials(x[k])))) for x in errs]
                @test all(isfinite, ∂)
                record_named!("T-235 $what ∂/∂a₀ orders (Dual)", key, minimum(orders(∂)))
            end
        end
    end
end

# T-236, energy over long runs in static fields, two systems; Δt and the step counts give at
# least 10⁴ periods of the slowest oscillation. Only finiteness is asserted (A7: a bound is
# proposed after measurement); the largest deviations and the least-squares drift rate of ΔH/H
# per unit time are recorded.
#
# Higuera and Cary's integrable system (Phys. Plasmas 24, 052104 (2017), Sec. VI, Eqs. 43–51; read
# in arXiv:1701.05605v1, whose equation numbers are used here): q = m = c = 1, E = −a x x̂,
# B = b y x̂, a = 1, b = 2, so A = (b y²/2) ẑ and φ = a x²/2. With u = p − A the invariants are
#     H = γ + a x²/2,   p_z = u_z + b y²/2,   I_y = u_y² + u_z²   (their Eqs. 46–47).
# H = 4, p_z = 0, Δt = 1/10 (about 20π steps per period of the x oscillation, 2π) are theirs; the
# paper names no other initial condition (its Fig. 2 marks Vay's resonance at p_y ≈ 1.7). Assumed
# here: start on their section plane x = 0 at y = 0 with p_y,0 ∈ {1.625, 1.775}
# (docs/plan/pushers-literature.md), so u_z = 0 and p_x = u_x = √(H² − 1 − p_y,0²) > 0;
# 6.3·10⁵ steps (10⁴ periods of x).
#
# A static non-uniform B with a static non-quadratic potential: B = r ẑ, φ = 10⁻²/r
# (r = √(x² + y²)), the field of Qin et al., Phys. Plasmas 20, 084503 (2013), with q/m = 1 and
# H = γ + φ. The initial conditions are chosen here, not taken from that paper: x₀ = (0, −1, 0)
# and u₀ = (0.1, 0.01, 0) (nonrelativistic) or u₀ = (1, 0.1, 0) (γ₀ ≈ 1.42), Δt = π/10, 2·10⁵ steps
# (10⁴ gyroperiods 2π/Ω at r = 1).
const QIN_E = (x, t) -> (r = sqrt(x[1]^2 + x[2]^2); c = (one(r) / 100) / r^3; (c * x[1], c * x[2], zero(r)))
const QIN_B = (x, t) -> (zero(x[1]), zero(x[1]), sqrt(x[1]^2 + x[2]^2))
qin_H(x, u) = lorentz_factor(u) + (one(x[1]) / 100) / sqrt(x[1]^2 + x[2]^2)

const HC_E = (x, t) -> (-x[1], zero(x[1]), zero(x[1]))                  # a = 1
const HC_B = (x, t) -> (2 * x[2], zero(x[1]), zero(x[1]))               # b = 2
hc_H(x, u) = lorentz_factor(u) + x[1]^2 / 2
hc_pz(x, u) = u[3] + x[2]^2
hc_Iy(x, u) = u[2]^2 + u[3]^2
hc_start(py) = (zero(py), zero(py), zero(py)), (sqrt(15 - py^2), py, zero(py))   # H = 4: γ = 4 at x = 0

"""
    long_run(P, x, u, dt, nsteps, E, B, H, others; every = 100)

Step `push_particle` with q/m = 1 and return the largest |ΔH/H|, the least-squares slope of ΔH/H
against t over samples every `every` steps, and the largest |Δf| of each function in `others`
(all evaluated in Float64 from the state of type T).
"""
function long_run(P, x::NTuple{3,T}, u::NTuple{3,T}, dt::T, nsteps, E, B, H, others = (); every = 100) where {T}
    H0 = Float64(H(x, u))
    o0 = map(f -> Float64(f(x, u)), others)
    omax = map(_ -> 0.0, others)
    maxrel = 0.0
    ts = Float64[]
    dH = Float64[]
    for n in 1:nsteps
        x, u = push_particle(P, x, u, (n - 1) * dt, dt, one(T), E, B)
        r = (Float64(H(x, u)) - H0) / H0
        maxrel = max(maxrel, abs(r))
        omax = map((m, f, v0) -> max(m, abs(Float64(f(x, u)) - v0)), omax, others, o0)
        if n % every == 0
            push!(ts, n * Float64(dt))
            push!(dH, r)
        end
    end
    tm = sum(ts) / length(ts)
    slope = sum((ts .- tm) .* dH) / sum((ts .- tm) .^ 2)
    return (; maxrel, slope, others = omax)
end

@testset "T-236 long-run energy in static non-uniform fields [A2, A7, WP-015]" begin
    @testset "Higuera–Cary's integrable system, $(pname(P)), $(type_label(T))" for P in PUSHERS, T in (Float32, Float64)
        for py0 in (1.625, 1.775)
            x, u = hc_start(T(py0))
            r = long_run(P, x, u, T(1) / 10, 630_000, HC_E, HC_B, hc_H, (hc_pz, hc_Iy))
            key = "$(pname(P)) $(type_label(T))"
            record_named!("T-236 HC system p_y0=$py0: max |ΔH/H|", key, r.maxrel)
            record_named!("T-236 HC system p_y0=$py0: drift rate |d(ΔH/H)/dt|", key, abs(r.slope))
            record_named!("T-236 HC system p_y0=$py0: max |Δp_z|", key, r.others[1])
            record_named!("T-236 HC system p_y0=$py0: max |ΔI_y|", key, r.others[2])
            @test isfinite(r.maxrel) && isfinite(r.slope) && all(isfinite, r.others)
        end
    end
    @testset "B = r ẑ, φ = 10⁻²/r, $(pname(P)), $(type_label(T))" for P in PUSHERS, T in (Float32, Float64)
        for (label, u0) in (("nonrelativistic", (0.1, 0.01, 0.0)), ("γ₀ ≈ 1.42", (1.0, 0.1, 0.0)))
            r = long_run(P, (zero(T), -one(T), zero(T)), T.(u0), T(π) / 10, 200_000, QIN_E, QIN_B, qin_H)
            record_named!("T-236 max |ΔH/H| $label", "$(pname(P)) $(type_label(T))", r.maxrel)
            record_named!("T-236 drift rate |d(ΔH/H)/dt| $label", "$(pname(P)) $(type_label(T))", abs(r.slope))
            @test isfinite(r.maxrel) && isfinite(r.slope)
        end
    end
end
