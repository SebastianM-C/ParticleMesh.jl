# WP-059: the deterministic field ionization of a dopant, T-770 to T-779 and T-781 to T-785 (T-780
# is their rerun on the GPU, gpu.jl). The atoms' bound populations follow the rate equations of an
# ADK rate in the field at each atom, and what they release is born by weight on a fixed lattice,
# with no random number. The problems are in ionization_cases.jl.

using ParticleMeshElectromagnetic: ionization_rate_sq, dopant_density, dopant_density_params, level_as, injection_column, window_frame, state_view
using ForwardDiff: ForwardDiff

"The alive electrons of a state's ionization, on the host."
function born(st)
    e = ionization_electrons(st)
    alive = Array(e.status) .== ParticleMeshCore.ALIVE
    pick(a) = Array(a)[alive]
    return (; n = count(alive), cell = map(pick, e.cell), offset = map(pick, e.offset), u = map(pick, e.u), w = pick(e.w))
end

"True when the value and every partial of every entry is zero."
all_zero(a) = all(x -> absmax(x) == 0, a)

"The C¹ rise of a dopant section, from its definition: sin²(π·s/(2·ramp)) between 0 for s ≤ 0 and 1 for s ≥ ramp."
rise(s, ramp) = s <= 0 ? zero(s) : s >= ramp ? one(s) : sin(oftype(s, π) * s / (2 * ramp))^2

# --- T-770 ----------------------------------------------------------------------------------

# The rate as a function. For the ground state of hydrogen in atomic units (I = 1/2, Z = 1, so
# n* = 1, C² = 4: prefactor 2, field 2, power 1) the form is W = (4/E)·exp(−2/(3E)), the textbook
# static-field ionization rate of hydrogen (Landau and Lifshitz, Quantum Mechanics, the problem on
# the ionization of the ground state in the section on the Stark effect in hydrogen; cited from
# memory, and FBPIC's coefficient lines give the same closed form for hydrogen, T-779). The
# cutoff: exactly zero for x = field/(3E) ≥ cutoff, the bare form for x ≤ 7·cutoff/8, between them
# below it, with a derivative that goes to zero at the cutoff and is continuous at 7·cutoff/8. Zero
# field: a zero rate, and for a dual field of value zero zero partials, not a NaN. The derivatives
# in E and in the level's three numbers are the closed forms of the bare rate.
@testset "T-770 the ADK rate: hydrogen's closed form, the smooth cutoff, zero at zero field, and its derivatives [A1, A2, A8, WP-059]" begin
    for T in TYPES
        h = ADKLevel(T(2), T(2), T(1))
        for E in (1 // 20, 1 // 10, 3 // 10, 1, 4)
            W = adk_rate(h, T(E))
            @test W isa T
            @test value_of(W) ≈ 4 / E * exp(-2 / (3 * E)) rtol = 8 * base_eps(T)
        end
        l = ADKLevel(T(3) / 4, T(3), T(3) / 2)                      # x = 1/E, cutoff 64
        bare(E) = value_of(l.prefactor) * (3 / E)^1.5 * exp(-1 / E)
        @test adk_rate(l, T(1) / 64) === zero(T) && adk_rate(l, T(1) / 100) === zero(T) && adk_rate(l, zero(T)) === zero(T)
        @test value_of(adk_rate(l, T(1) / 56)) ≈ bare(1 / 56) rtol = 64 * base_eps(T)
        @test value_of(adk_rate(l, T(1) / 20)) ≈ bare(1 / 20) rtol = 64 * base_eps(T)
        @test 0 < value_of(adk_rate(l, T(1) / 60)) < bare(1 / 60)
        @test adk_rate(l, 1 // 2) === adk_rate(l, T(1) / 2)        # any real field is taken in the level's type
        @test ionization_rate_sq(l, T(1) / 4) === adk_rate(l, T(1) / 2)
        @test isnan(value_of(adk_rate(l, T(NaN))))                 # a field that is not a number is not a zero rate
    end
    # a dual field of value zero, with partials: the rate and its partials are zero
    z = ionization_rate_sq(ADKLevel(mkdual(2, 1, 0, 0), mkdual(2, 0, 1, 0), mkdual(1, 0, 0, 1)), mkdual(0, 1, 2, 3))
    @test z isa D3 && absmax(z) == 0
    # the cutoff is C¹ (Float64): the bare rate's value and slope at 7·cutoff/8, zero slope at the cutoff
    l = ADKLevel(0.75, 3.0, 1.5; cutoff = 16)
    bare64(E) = 0.75 * (3 / E)^1.5 * exp(-1 / E)
    dW(E) = ForwardDiff.derivative(e -> adk_rate(l, e), E)
    Ec, Ej = 1 / 16, 1 / 14
    @test adk_rate(l, Ec) == 0 && adk_rate(l, Ec * (1 + 1e-12)) < 1e-20 * bare64(Ec)
    @test dW(Ec * (1 + 1e-6)) / (bare64(Ec) / Ec) < 1e-3 && dW(Ec * (1 - 1e-6)) == 0
    @test adk_rate(l, Ej * (1 - 1e-9)) ≈ bare64(Ej) rtol = 1e-7
    @test dW(Ej * (1 - 1e-7)) ≈ dW(Ej * (1 + 1e-7)) rtol = 1e-4
    @test dW(Ej * (1 + 1e-7)) ≈ ForwardDiff.derivative(bare64, Ej) rtol = 1e-4
    @test all(E -> adk_rate(l, E) <= adk_rate(l, nextfloat(E)), range(Ec, Ej; length = 200))     # rising through the taper
    # the derivatives of the bare rate: in E, and in the prefactor, the field and the power
    for E in (0.1, 0.5, 2.0)
        W = bare64(E)
        @test dW(E) ≈ W * (1 / E^2 - 1.5 / E) rtol = 1e-12
        g = ForwardDiff.gradient(v -> adk_rate(ADKLevel(v[1], v[2], v[3]; cutoff = 16), E), [0.75, 3.0, 1.5])
        @test g ≈ [W / 0.75, W * (1.5 / 3 - 1 / (3 * E)), W * log(3 / E)] rtol = 1e-12
    end
    # the constructor: integers are taken as Float64, mixed types promote, and what is not a rate is refused
    @test ADKLevel(2, 3, 1) === ADKLevel(2.0, 3.0, 1.0; cutoff = 64.0)
    @test ADKLevel(2.0f0, 3, 1) isa ADKLevel{Float32} && ADKLevel(mkdual(2, 1, 0, 0), 3, 1) isa ADKLevel{D3}
    @test level_as(Float32, ADKLevel(2, 3, 1; cutoff = 8)) === ADKLevel(2.0f0, 3.0f0, 1.0f0; cutoff = 8.0f0)
    for bad in ((-1, 3, 1), (Inf, 3, 1), (2, 0, 1), (2, -3, 1), (2, Inf, 1), (2, 3, NaN), (2, 3, Inf))
        @test_throws ArgumentError ADKLevel(bad...)
    end
    @test adk_rate(ADKLevel(0, 3, 1), 1.0) === 0.0              # a level that never ionizes is a level
    @test_throws ArgumentError ADKLevel(2, 3, 1; cutoff = 0)
    @test_throws ArgumentError ADKLevel(2, 3, 1; cutoff = -1)
    @test_throws ArgumentError ADKLevel(2, 3, 1; cutoff = Inf)
    # the section's profile, from its definition, and with no section the density everywhere
    for T in TYPES
        sec = DopantSection(; start = T(2), stop = T(5), ramp = T(3) / 2)
        p = dopant_density_params(IonizationParams{T}(T(3) / 100, (ADKLevel(T(2), T(3), T(1)),), sec))
        for x in (-1, 2, 2.3, 3.4, 3.5, 4.2, 5, 5.6, 6.5, 7, 100)
            n = 0.03 * rise(x - 2, 1.5) * (1 - rise(x - 5, 1.5))
            @test value_of(dopant_density(T(x), p)) ≈ n atol = 16 * base_eps(T) * 0.03
        end
        @test dopant_density(T(1), p) === zero(T) && dopant_density(T(7), p) === zero(T)
        @test value_of(dopant_density(T(4), p)) ≈ 0.03 rtol = 4 * base_eps(T)
        flat = dopant_density_params(IonizationParams{T}(T(3) / 100, (ADKLevel(T(2), T(3), T(1)),)))
        @test all(x -> dopant_density(T(x), flat) === T(3) / 100, (-1.0e6, 0, 17.5, 1.0e6))
    end
    @test DopantSection(; start = 1, stop = 2, ramp = 1) === DopantSection(; start = 1.0, stop = 2.0, ramp = 1.0)
    @test DopantSection(; start = 1, stop = 1, ramp = 1).stop == 1          # no flat top: the rise, then the fall
    @test_throws ArgumentError DopantSection(; start = 1, stop = 2, ramp = 0)
    @test_throws ArgumentError DopantSection(; start = 2, stop = 1, ramp = 1)
end

# --- T-771 ----------------------------------------------------------------------------------

# One level in a constant field, where the scheme is the exact solution. A dopant of 2 × 1 atoms
# per cell on the four rows with centres in (1.1, 2.9), in the prescribed uniform field Ex = 1/2
# (no current), with the rate W = prefactor·6·e⁻² (field/E = 6, field/(3E) = 2), born every 3
# steps, 20 steps. With q = exp(−W·Δt):
# - the atoms are the lattice of the definition, in the engine's sorted order, each of the weight
#   density/2; after 2 steps nothing is born and `pending` is 1 − q²;
# - after 20 steps the bound population is q²⁰, `emitted` 1 − q¹⁸ (the births at steps 3, 6, …, 18)
#   and `pending` q¹⁸ − q²⁰, to roundoff over the steps, and they sum to one;
# - birth m put one electron per atom of the weight (density/2)·q^{3(m−1)}·(1 − q³) into the run,
#   at rest at the start of step 3m: with q/m = −1 and no B, Boris gives it
#   u_x = −Ex·Δt·(20 − 3m + ½) after the last push (the half step back at birth, then 20 − 3m + 1
#   pushes) and no other momentum, and it stays on its atom's row;
# - with the density and the prefactor duals, the populations' and the weights' partials are those
#   of these closed forms, while the atoms' and the electrons' positions and the momenta carry
#   none (A3: the parameters enter through the weights);
# - a second `initialize!` puts every atom back into its ground state, with no electron.
# In a magnetic field too, the momentum after the birth step is the pusher's full step from its
# half step back from rest, which differs from the half step forward from rest by O(Δt³).
function body_T771(be)
    C = emC("T-771")
    for T in TYPES
        dens, pref = seeded(T, 1) / 100, seeded(T, 1) / 2
        n, every, natoms = 20, 3, 16 * 4 * 2
        prob = ion_in_E(T; levels = (ADKLevel(pref, T(3), one(T)),), density = dens, ppc = (2, 1), every, rows = (1.1, 2.9), nsteps = n)
        st = ion_state(prob; current = NoCurrent(), be)
        p = params(prob)
        dt = prob.dt
        W = pref * 6 * exp(-T(2))
        q(k) = exp(-W * dt * k)
        gone(k) = -expm1(-W * dt * k)
        tol(x) = C * n * base_eps(T) * absmax(x)
        s = ionization_state(st)
        pts = [((c1 - 1 + (k - 0.5) / 2) / 2, (c2 - 0.5) / 2) for c2 in 3:6 for c1 in 1:16 for k in 1:2]
        @test length(s.x) == natoms == length(pts) && all(s.alive)
        @test value_of.(s.x) ≈ first.(pts) rtol = 4 * base_eps(T)
        @test value_of.(s.y) ≈ last.(pts) rtol = 4 * base_eps(T)
        @test all(w -> absdiff(w, dens / 2) <= 2 * base_eps(T) * absmax(dens), s.w)
        @test all(==(one(T)), s.bound[1]) && all_zero(s.pending) && all_zero(s.emitted)
        run!(st, p, 2)
        s = ionization_state(st)
        @test born(st).n == 0 && all_zero(s.emitted)
        @test maximum(x -> absdiff(x, gone(2)), s.pending) <= tol(gone(2))
        run!(st, p, n - 2; first = 3)
        s = ionization_state(st)
        err = maximum(x -> absdiff(x, q(n)), s.bound[1]) / (n * base_eps(T) * absmax(q(n)))
        record!("T-771", T, err, be)
        @test err <= C
        @test maximum(x -> absdiff(x, gone(18)), s.emitted) <= tol(gone(18))
        @test maximum(x -> absdiff(x, q(18) * gone(2)), s.pending) <= tol(q(18) * gone(2))
        @test maximum(absmax, s.bound[1] .+ s.pending .+ s.emitted .- one(T)) <= C * n * base_eps(T)
        el = born(st)
        nb = n ÷ every
        @test el.n == nb * natoms && ionization_lost(st) == 0
        order = sortperm(value_of.(el.u[1]))                 # the earliest birth has the most negative u_x
        worst_u = worst_w = 0.0
        for m in 1:nb
            idx = order[((m - 1) * natoms + 1):(m * natoms)]
            ux = -dt / 2 * (n - every * m + one(T) / 2)
            wm = dens / 2 * q(every * (m - 1)) * gone(every)
            worst_u = max(worst_u, maximum(i -> absdiff(el.u[1][i], ux), idx) / (n * base_eps(T) * absmax(ux)))
            worst_w = max(worst_w, maximum(i -> absdiff(el.w[i], wm), idx) / (n * base_eps(T) * absmax(wm)))
            @test sort(el.cell[2][idx]) == repeat(Int32.(3:6); inner = 32)      # 16 columns × 2 atoms on each row
        end
        record!("T-771 momenta", T, worst_u, be)
        record!("T-771 weights", T, worst_w, be)
        @test worst_u <= C && worst_w <= C
        @test all_zero(el.u[2]) && all_zero(el.u[3])
        @test all(==(one(T) / 2), el.offset[2])
        total = sum(s.w .* s.emitted)
        @test absdiff(sum(el.w), total) <= tol(total)
        if T === D3
            fr = ionization_frame(st)
            no_partials(a) = all(x -> all(iszero, partials(x)), Array(a))
            @test all(no_partials, (fr.atoms.offset..., fr.electrons.offset..., fr.electrons.u...))
            @test all(x -> all(!iszero, partials(x)), el.w) && all(x -> all(!iszero, partials(x)), s.bound[1])
        end
        initialize!(st, prob)
        s = ionization_state(st)
        @test all(==(one(T)), s.bound[1]) && all_zero(s.pending) && all_zero(s.emitted) && born(st).n == 0 && ionization_lost(st) == 0
        @test all(Array(ionization_electrons(st).count) .== 0)
    end
    for T in TYPES
        E, B = (T(1) / 2, T(1) / 10, zero(T)), (zero(T), zero(T), T(3) / 10)
        prob = ion_in_E(T; E0 = E, B0 = B, nsteps = 1)
        st = ion_state(prob; current = NoCurrent(), be)
        run!(st, params(prob), 1)
        dt = prob.dt
        back = push_momentum(Boris(), zero3(T), E, B, dt / 4)          # −(q/m)·Δt/4 with q/m = −1
        u = push_momentum(Boris(), back, E, B, -dt / 2)
        half = push_momentum(Boris(), zero3(T), E, B, -dt / 4)
        el = born(st)
        @test el.n == 128
        @test all(d -> maximum(x -> absdiff(x, u[d]), el.u[d]) <= 8 * base_eps(T), 1:3)
        @test maximum(d -> abs(value_of(u[d] - half[d])), 1:3) <= value_of(dt)^3 / 8
        @test maximum(d -> abs(value_of(u[d] - half[d])), 1:3) > 0
    end
end

@testset "T-771 one level in a constant field: the populations are exp(−W·t), the births carry the released weight at rest, the parameters' partials are in the weights only [A2, A3, A7, A8, WP-059]" begin
    body_T771(CPU())
end

# --- T-772 ----------------------------------------------------------------------------------

"Three levels whose rates in |E| = 1/2 are of one order (about 0.41, 1.57 and 2.64 with `p2 = 1`); `p2` is the second's prefactor."
chain_levels(::Type{T}; p2 = one(T)) where {T} = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(p2, T(4), T(3) / 2), ADKLevel(one(T), T(6), T(2)))

"The populations of a chain of three levels at time `t` from all of them in the first, the exact solution of the rate equations for constant, distinct rates `W`."
function chain_exact(W, t)
    e = exp.(-t .* W)
    f1 = e[1]
    f2 = W[1] / (W[2] - W[1]) * (e[1] - e[2])
    f3 = W[1] * W[2] * (e[1] / ((W[2] - W[1]) * (W[3] - W[1])) + e[2] / ((W[1] - W[2]) * (W[3] - W[2])) + e[3] / ((W[1] - W[3]) * (W[2] - W[3])))
    return (f1, f2, f3)
end

# A chain of three levels in the constant field. Per step, level j loses f_j·(1 − exp(−W_j·Δt)) to
# level j + 1, all from the populations at the start of the step: the populations are this
# recurrence, computed here on the host in the element type (to roundoff over the steps, with the
# second level's prefactor and the density duals); an atom passes one level per step (after the
# first step the third level is exactly empty, after the second it is not); the released fraction
# is the conserved count 3 − 3·f₁ − 2·f₂ − f₃ (level j has released j − 1 electrons), all of it
# born (`every = 1`: nothing pending) and equal to the electrons' weight. Against the exact
# solution of the rate equations the first level is exact and the others converge at first order
# in Δt (the gain of a level starts to decay a step late): halving Δt twice at a fixed end time
# halves the error twice (Float64).
function body_T772(be)
    C = emC("T-772")
    for T in TYPES
        dens = seeded(T, 1) / 100
        levels = chain_levels(T; p2 = seeded(T, 1))
        n = 12
        prob = ion_in_E(T; levels, density = dens, nsteps = n)
        st = ion_state(prob; current = NoCurrent(), be)
        p = params(prob)
        dt = prob.dt
        W = map(l -> adk_rate(l, T(1) / 2), levels)
        f, released = [one(T), zero(T), zero(T)], zero(T)
        worst = 0.0
        for k in 1:n
            d = [f[j] * -expm1(-W[j] * dt) for j in 1:3]
            f = [f[j] * exp(-W[j] * dt) + (j == 1 ? zero(T) : d[j - 1]) for j in 1:3]
            released += sum(d)
            step!(st, p, k)
            s = ionization_state(st)
            worst = max(worst, maximum(j -> maximum(x -> absdiff(x, f[j]), s.bound[j]), 1:3) / (k * base_eps(T)))
            @test all_zero(s.pending)
            @test maximum(x -> absdiff(x, released), s.emitted) <= C * k * base_eps(T) * absmax(released)
            count3 = 3 .- 3 .* s.bound[1] .- 2 .* s.bound[2] .- s.bound[3]
            @test maximum(absmax, s.emitted .- count3) <= C * 8 * base_eps(T)
            k == 1 && @test all_zero(s.bound[3]) && all(x -> value_of(x) > 0, s.bound[2])
            k == 2 && @test all(x -> value_of(x) > 0, s.bound[3])
        end
        record!("T-772", T, worst, be)
        @test worst <= C
        s, el = ionization_state(st), born(st)
        total = sum(s.w .* s.emitted)
        @test el.n == n * 128 && ionization_lost(st) == 0
        @test absdiff(sum(el.w), total) <= C * n * base_eps(T) * absmax(total)
        @test value_of(released) > 1                    # more than one electron per atom: the chain is under way
    end
    levels = chain_levels(Float64)
    W = map(l -> adk_rate(l, 0.5), levels)
    errs = map((1, 2, 4)) do r
        prob = ion_in_E(Float64; levels, nsteps = 12 * r, courant = 9 // (10 * r))
        st = ion_state(prob; current = NoCurrent(), be)
        run!(st, params(prob), prob.nsteps)
        s = ionization_state(st)
        exact = chain_exact(W, prob.nsteps * prob.dt)
        @test maximum(abs, s.bound[1] .- exact[1]) <= 64 * eps() * prob.nsteps
        maximum(j -> maximum(abs, s.bound[j] .- exact[j]), 2:3)
    end
    ratios = (errs[1] / errs[2], errs[2] / errs[3])
    @info "T-772 the chain against the exact solution" errs ratios
    d = get!(MEASURED, "T-772 error ratio per halving of Δt" * (be isa CPU ? "" : " on $(nameof(typeof(be)))"), Dict{String,Float64}())
    d["first"], d["second"] = ratios
    @test errs[3] < errs[2] < errs[1] < emC("T-772", "error_max")
    @test all(r -> emC("T-772", "ratio_min") <= r <= emC("T-772", "ratio_max"), ratios)
end

@testset "T-772 a chain of levels: the per-step recurrence, one level per step, the conserved electron count, first order in Δt against the exact solution [A2, A7, A8, WP-059]" begin
    body_T772(CPU())
end

# --- T-773 ----------------------------------------------------------------------------------

"""
    ion_gauss(st, p, k) -> (; res, bare, scale, escale, ions)

Gauss's law of a state with ionization after step `k`, per interior node: `res`, the residual
`D⁻ₓEx + D⁻_yEy − ρ` with ρ the charge of every particle ([`window_charge!`](@ref): the species and
the born electrons) plus the dopant ions' ([`ionization_charge!`](@ref)), less the neutralizing
background (the mean of a periodic box; a window's `window_background!`; none in a box with an open
edge, whose runs here start without charge); `bare`, the same without the ions; `scale` as T-360,
`(n₁ + n₂)·max|ρ|` of the particles; `escale`, the divergence's own roundoff scale
`(2/Δx + 2/Δy)·max|E|`, which is the larger where a strong field stands over little charge;
`ions`, the largest ion charge density. `absmax` takes a dual's partials.
"""
function ion_gauss(st, p, k)
    T = eltype(st.ρ)
    grid = st.grid
    be = get_backend(st.ρ)
    ρ, ρi, ρ0 = field(be, T, grid), field(be, T, grid), field(be, T, grid)
    window_charge!(ρ, st)
    ionization_charge!(ρi, st)
    window = st.aux.window
    window === nothing || window_background!(ρ0, st, p, window_shifts(st, k))
    ρ, ρi, ρ0 = Array(ρ), Array(ρi), Array(ρ0)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    mean(a) = window === nothing && st.numerics.absorber === nothing ? sum(a[I1, I2]) / prod(n) : zero(T)
    m, mi = mean(ρ), mean(ρi)
    Ex, Ey = Array(st.fields.Ex), Array(st.fields.Ey)
    div(i, j) = (Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2]
    res = [absmax(div(i, j) - (ρ[i, j] - m + ρi[i, j] - mi - ρ0[i, j])) for i in I1, j in I2]
    bare = [absmax(div(i, j) - (ρ[i, j] - m - ρ0[i, j])) for i in I1, j in I2]
    hv = map(value_of, h)
    escale = (2 / hv[1] + 2 / hv[2]) * max(maximum(absmax, Ex[I1, I2]), maximum(absmax, Ey[I1, I2]))
    return (; res, bare, scale = sum(n) * maximum(absmax, ρ[I1, I2]), escale, ions = maximum(absmax, ρi[I1, I2]))
end

# Gauss's law with ionization, periodic box. An electron is born on top of the immobile ion it
# leaves, so its birth changes no charge density; from then on its current is the charge-conserving
# one, so D⁻·E − ρ stays what it was to roundoff (T-360) when ρ counts the ions' charge, the atoms'
# weight times their emitted fraction at their places. A thermal plasma (its particles cross cells
# and both seams) in the uniform field Ex = 1/2, which the plasma's current then changes, with a
# dopant of two levels at 0.05 n_c (a dual), two atoms per cell along x on four of the eight rows
# (an electron born anywhere but at its atom's own offset would show), born every 2 steps, both
# deposition modes: the residual with the ions stays at roundoff after every step, scaled as T-360,
# while without them it is the ions' charge less its mean, about half its largest value; no birth
# is lost, and every atom has released more than one electron.
function body_T773(be)
    C = emC("T-773")
    for T in TYPES, mode in (Reference(), Fast())
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(one(T) / 2, T(4), T(3) / 2))
        prob = ion_in_E(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1), every = 2, nsteps = 30, plasma = true, rows = (1.1, 2.9))
        st = ion_state(prob; mode, be)
        p = params(prob)
        r = ion_gauss(st, p, 0)
        worst = maximum(r.res) / (base_eps(T) * r.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss(st, p, k)
            worst = max(worst, maximum(r.res) / (base_eps(T) * r.scale * sqrt(k + 1)))
        end
        s = ionization_state(st)
        bare = maximum(r.bare) / r.ions
        @info "T-773" T mode worst bare released = extrema(value_of, s.emitted) born = born(st).n ions = r.ions
        record!("T-773 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test bare > 1 / 4 && r.ions > 1 / 100
        @test ionization_lost(st) == 0 && born(st).n == 15 * 128
        @test minimum(value_of, s.emitted) > 1
        @test maximum(abs ∘ value_of, Array(st.fields.Jx)) > 0
    end
end

@testset "T-773 Gauss's law to roundoff with the ions' charge after every step of a plasma with a dopant being ionized, all three types, both modes [A2, A4, A7, A8, WP-059]" begin
    body_T773(CPU())
end

# --- T-774 ----------------------------------------------------------------------------------

# The dopant in the moving window. Gauss's law as T-473 (off nodes 1 and 2 and the antenna's two
# columns), with the ions' charge: a laser ionizes a doped section of the plasma (two atoms per
# cell along x), the window moves on until the section and its electrons have left through the
# back edge (150 steps, about 58 shifts), both modes; while the section is in the window the
# residual without the ions is the ions' charge (between a half and twice its largest density,
# wherever that is above 10⁻⁴ n_c). The plasma's refill loses no particle (a box that
# does fails Gauss's law with or without a dopant).
#
# The lattice through the shifts (vacuum, so that a level with the cutoff at |E| = 1/8 is exactly
# idle behind the pulse; 2 × 2 atoms per cell): the lattice's columns are the window's up to its
# injection column, and after any number of shifts the atoms' buffer is the full lattice in sorted
# order, each atom of the weight of the section's profile at its global coordinate, from the
# definition (`ion_lattice`; with the density and the section's start, stop and ramp duals, their
# partials too, while the positions carry none); and an atom keeps its populations as it moves
# back through the buffer: between step 112, when the pulse has left the section, and step 124, 8
# shifts later, every atom of the section still in the window has the bits of its bound, pending
# and emitted fractions, at its new slot (the atoms ahead of the section have no weight and
# release nothing, but their populations follow the pulse they sit in). A section that lies ahead
# of the window at the start (no atom has weight) is loaded by the refill alone, column by column
# at the window's position then: after 100 steps its weights are the profile's.
"""
    ion_lattice(st, k, ppc, (density, start, stop, ramp)) -> (ok, werr)

The atoms' buffer of a window state after step `k` against the definition: `ok` when it is the
full lattice of `ppc` atoms per cell in the engine's sorted order (columns fastest, then rows, the
in-cell index fastest within a cell, x the faster of its two) with the in-cell offsets
`((k₁ − ½)/p₁, (k₂ − ½)/p₂)`; `werr` the largest difference of an atom's weight from the section's
profile at its global coordinate over the points per cell, in eps of the density's share.
"""
function ion_lattice(st, k, ppc, prof)
    T = eltype(st.ρ)
    fr = ionization_frame(st)
    a = fr.atoms
    ncols, nrows, K = Int(fr.ncols), Int(fr.nrows), prod(ppc)
    dens, start, stop, ramp = prof
    h = cellsize(st.grid)[1]
    sw = window_shifts(st, k)
    s = ionization_state(st)
    δ1 = [(T(kk % ppc[1]) + one(T) / 2) / ppc[1] for kk in 0:(K - 1)]
    δ2 = [(T(kk ÷ ppc[1]) + one(T) / 2) / ppc[2] for kk in 0:(K - 1)]
    ok = all(s.alive) && length(s.w) == ncols * nrows * K
    ok &= Array(a.cell[1]) == repeat(Int32.(1:ncols); inner = K, outer = nrows) && Array(a.cell[2]) == repeat(Int32.(fr.row0 .+ (0:(nrows - 1))); inner = ncols * K)
    ok &= Array(a.offset[1]) == repeat(δ1; outer = ncols * nrows) && Array(a.offset[2]) == repeat(δ2; outer = ncols * nrows)
    werr = 0.0
    for c in 1:ncols, kk in 1:K
        x = (T(c + sw - 1) + δ1[kk]) * h
        w = dens * rise(x - start, ramp) * (1 - rise(x - stop, ramp)) / K
        werr = max(werr, maximum(r -> absdiff(s.w[((r - 1) * ncols + c - 1) * K + kk], w), 1:nrows))
    end
    return ok, werr * K / (base_eps(T) * value_of(dens))
end

function body_T774(be)
    C = emC("T-774")
    for T in TYPES, mode in (Reference(), Fast())
        prob = ion_window(T; nsteps = 150, density = seeded(T, 1) / 200, ppc = (2, 1), capacity = 8192)
        st = ion_state(prob; mode, be)
        p = params(prob)
        ia = antenna_planes(st)[1].ia
        worst = bare = released = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 3 == 0 || continue
            s_win = window_shifts(st, k)
            r = ion_gauss(st, p, k)
            cols = [i for i in axes(r.res, 1) if !(i in (1, 2, ia - 1 - s_win, ia - s_win))]
            worst = max(worst, Float64(maximum(r.res[cols, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
            r.ions > 1.0e-4 && (bare = max(bare, Float64(maximum(r.bare[cols, :]) / r.ions)))
            released = max(released, maximum(value_of, ionization_state(st).emitted))
        end
        @info "T-774" T mode worst bare released shifts = window_shifts(st, prob.nsteps)
        record!("T-774 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test 1 / 2 < bare < 2
        @test released > 3 / 10 && ionization_lost(st) == 0 && all(==(0), window_lost(st))
        @test all_zero(ionization_state(st).emitted) && born(st).n == 0        # the section and its electrons have left
    end
    for T in TYPES
        dens, start, stop, ramp = seeded(T, 1) / 200, seeded(T, 28), T(34) + seeded(T, 1) / 2 - T(1) / 2, T(3) + seeded(T, 1) / 4 - T(1) / 4
        ppc = (2, 2)
        K = prod(ppc)
        prob = ion_window(T; nsteps = 124, density = dens, start, stop, ramp, plasma = false, cutoff = 8, ppc, capacity = 16384)
        st = ion_state(prob; be)
        p = params(prob)
        fr = ionization_frame(st)
        ncols, nrows = Int(fr.ncols), Int(fr.nrows)
        @test ncols == injection_column(window_frame(st)) < 64 && Array(fr.atoms.count)[1] == ncols * nrows * K
        slots(c, r) = ((r - 1) * ncols + c - 1) * K .+ (1:K)
        prof = (dens, start, stop, ramp)
        ok, werr = ion_lattice(st, 0, ppc, prof)
        @test ok && werr <= emC("T-774", "C_weight")
        run!(st, p, 112)
        ok, w1 = ion_lattice(st, 112, ppc, prof)
        @test ok
        s1, sw1 = ionization_state(st), window_shifts(st, 112)
        run!(st, p, 12; first = 113)
        ok, w2 = ion_lattice(st, 124, ppc, prof)
        @test ok
        @test max(w1, w2) <= emC("T-774", "C_weight")
        s2 = ionization_state(st)
        moved = window_shifts(st, 124) - sw1
        @test moved >= 5 && sw1 > 20
        kept = [(i, j) for r in 1:nrows for c in (moved + 1):ncols for (i, j) in zip(slots(c, r), slots(c - moved, r)) if value_of(s1.w[i]) > 0]
        @test length(kept) > 40 * K
        @test all(((i, j),) -> same_bits([s1.emitted[i], s1.bound[1][i], s1.pending[i]], [s2.emitted[j], s2.bound[1][j], s2.pending[j]]), kept)
        @test count(((i, j),) -> value_of(s2.emitted[j]) > 1 / 10, kept) > 20 * K        # ionized atoms among them
        fresh = [i for r in 1:nrows for c in (ncols - moved + 1):ncols for i in slots(c, r)]
        @test all(i -> absmax(s2.emitted[i]) == 0 && s2.bound[1][i] == one(T), fresh)
        if T === D3
            @test all(a -> all(x -> all(iszero, partials(x)), Array(a)), fr.atoms.offset)
            @test any(x -> any(!iszero, partials(x)), s2.w)
        end
        # a section ahead of the window: only the refill loads it
        ppc, prof = (1, 2), (dens, start + 22, stop + 21, ramp)
        prob = ion_window(T; nsteps = 100, density = dens, start = prof[2], stop = prof[3], ramp, plasma = false, cutoff = 8, ppc)
        st = ion_state(prob; be)
        @test all_zero(ionization_state(st).w)
        run!(st, params(prob), prob.nsteps)
        ok, w3 = ion_lattice(st, prob.nsteps, ppc, prof)
        @test ok && w3 <= emC("T-774", "C_weight")
        @test count(w -> value_of(w) > value_of(dens) / 8, ionization_state(st).w) > 50 && window_shifts(st, prob.nsteps) > 20
        record!("T-774 weights", T, max(werr, w1, w2, w3), be)
    end
end

@testset "T-774 the dopant in the moving window: Gauss's law with the ions across shifts, the refill at the global column with the section's weights, populations that move with their atoms [A2, A3, A4, A8, WP-059, WP-028]" begin
    body_T774(CPU())
end

# --- T-775 ----------------------------------------------------------------------------------

# Reference mode is bitwise with ionization: the births are one work item per atom into a slot
# fixed by the alive count and the atom's place, the populations one item per atom, the refill one
# per fresh atom, and the sorts fix the order the deposition sees. The window run of
# `ion_digests` (everything the state evolves, hashed) is the same across reruns and launch
# shapes, and on the CPU at 1 and 4 threads (the subprocess worker); the two deposition modes
# agree to roundoff in the fields and in the atoms' released fractions (Float64, 110 steps).
function body_T775(be)
    ref = ion_digests(; be)
    @test ref == ion_digests(; be)
    @test length(unique(last.(ref))) == length(TYPES)
    for wgs in (be isa CPU ? (32, 256) : (64, 256))
        @test ion_digests(; workgroupsize = wgs, be) == ref
    end
    prob = ion_window(Float64)
    a, b = ion_state(prob; mode = Reference(), be), ion_state(prob; mode = Fast(), be)
    run!(a, params(prob), prob.nsteps)
    run!(b, params(prob), prob.nsteps)
    comps = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
    m = maximum(c -> maximum(abs, Array(getfield(a.fields, c))), comps)
    d = maximum(c -> maximum(abs, Array(getfield(a.fields, c)) .- Array(getfield(b.fields, c))), comps)
    sa, sb = ionization_state(a), ionization_state(b)
    de = maximum(abs, sa.emitted .- sb.emitted)
    @info "T-775 Reference against Fast" fields = d / m emitted = de born = (born(a).n, born(b).n)
    record!("T-775", Float64, max(d / m, de) / eps(Float64), be)
    @test max(d / m, de) <= emC("T-775") * eps(Float64)
    @test born(a).n == born(b).n > 1000
    return ref
end

@testset "T-775 the step with ionization is bitwise across reruns, launch shapes and thread counts in Reference mode, and the modes agree [A4, WP-059]" begin
    ref = body_T775(CPU())
    for nt in DETERMINISM_THREADS
        lines = filter(startswith("EMION "), split(worker_output(nt), '\n'))
        got = Dict(String(f[3]) => parse(UInt64, f[4]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(ref)
        for (T, d) in ref
            @test get(got, type_label(T), nothing) == d
        end
    end
end

# --- T-776 ----------------------------------------------------------------------------------

# The electrons' buffer is fixed (§0.4): a birth step offers atom slot i the electron slot
# count + i, and an atom that finds none keeps what it has released and is counted. In the
# constant field of T-771 with 128 atoms, a birth in every step and 300 slots: steps 1 and 2 fill
# 256 slots, in step 3 the first 44 atoms find a slot and 84 do not, and from step 4 on none does,
# so after 6 steps 84 + 3·128 births are counted as lost, 300 electrons are alive, the first 44
# atoms have emitted 1 − q³ and hold q³ − q⁶ pending, the others 1 − q² and q² − q⁶, and the
# electrons' weight is still the atoms' emitted weight: no charge is lost; a new `initialize!` zeroes
# the counter. With the charge-conserving current and a plasma (T-773's run with 400 slots) Gauss's
# law holds to roundoff through the overflow.
#
# What a birth step needs is the electrons alive and the highest atom slot that releases, in a
# step that sorts the electrons and in one that does not (the birth sorts them first): a section
# gives 48 of the 128 atoms a weight (columns 5 to 10 of 16, the last of them atom slot 122), so
# four births need 3·48 + 122 slots; with exactly those none is lost, and with one fewer exactly
# one is.
function body_T776(be)
    C = emC("T-771")
    for T in TYPES
        n = 6
        prob = ion_in_E(T; nsteps = n, capacity = 300)
        st = ion_state(prob; current = NoCurrent(), be)
        run!(st, params(prob), n)
        W = adk_rate(prob.ionization.levels[1], T(1) / 2)
        q(k) = exp(-W * prob.dt * k)
        s, el = ionization_state(st), born(st)
        @test ionization_lost(st) == 84 + 3 * 128 && el.n == 300
        tol = C * n * base_eps(T)
        @test maximum(x -> absdiff(x, 1 - q(3)), s.emitted[1:44]) <= tol && maximum(x -> absdiff(x, q(3) - q(6)), s.pending[1:44]) <= tol
        @test maximum(x -> absdiff(x, 1 - q(2)), s.emitted[45:end]) <= tol && maximum(x -> absdiff(x, q(2) - q(6)), s.pending[45:end]) <= tol
        @test maximum(absmax, s.bound[1] .+ s.pending .+ s.emitted .- one(T)) <= tol
        @test absdiff(sum(el.w), sum(s.w .* s.emitted)) <= tol * absmax(sum(el.w))
        initialize!(st, prob)
        @test ionization_lost(st) == 0 && born(st).n == 0
    end
    for T in TYPES, (current, mode) in ((NoCurrent(), Reference()), (Esirkepov(), Reference()), (Esirkepov(), Fast()))
        nrel, imax, nb = 48, 122, 4
        for lost in (0, 1)
            prob = ion_in_E(T; every = 2, nsteps = 2 * nb, capacity = (nb - 1) * nrel + imax - lost,
                            section = DopantSection(; start = T(2), stop = T(4), ramp = one(T)))
            st = ion_state(prob; current, mode, be)
            @test count(w -> value_of(w) > 0, ionization_state(st).w) == nrel
            run!(st, params(prob), prob.nsteps)
            @test ionization_lost(st) == lost && born(st).n == nb * nrel - lost
        end
    end
    for T in TYPES, mode in (Reference(), Fast())
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(one(T) / 2, T(4), T(3) / 2))
        prob = ion_in_E(T; levels, density = seeded(T, 1) / 20, every = 2, nsteps = 30, plasma = true, capacity = 400)
        st = ion_state(prob; mode, be)
        p = params(prob)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss(st, p, k)
            worst = max(worst, maximum(r.res) / (base_eps(T) * r.scale * sqrt(k + 1)))
        end
        record!("T-776 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= emC("T-773")
        @test ionization_lost(st) > 128 && born(st).n == 400
        @test maximum(value_of, ionization_state(st).pending) > 1 / 10
    end
end

@testset "T-776 a full electron buffer: the births that find no slot are counted and keep their weight, and no charge is lost [A2, A4, A8, WP-059]" begin
    body_T776(CPU())
end

# --- T-777 ----------------------------------------------------------------------------------

"Fill with a value that would show: NaN for floats, 2 for integers."
ion_junk!(a::AbstractArray{<:Integer}) = fill!(a, eltype(a)(2))
ion_junk!(a::AbstractArray) = fill!(a, eltype(a)(NaN))
ion_junk!(x::Union{Tuple,NamedTuple}) = foreach(ion_junk!, x)

"Overwrite what the ionization holds besides its evolved arrays: the birth flags, the electrons' extras and both workspaces' buffers."
function scramble_ionization!(st)
    fr = ionization_frame(st)
    ion_junk!((fr.flags, fr.electrons.extras))
    for ws in (fr.atoms_ws, fr.electrons_ws)
        ion_junk!((ws.key, ws.tmp_i32, ws.tmp_T, ws.tmp_u8, ws.tmp_extras, ws.flux_patch))
        fill!(ws.perm, 1)
    end
    return st
end

"The bits of everything a state with ionization evolves, of `K` and of the histories from entry `k + 1` on."
ion_bits(st, k = 0) = (map(bits, evolved(st))..., bits(st.K), map(h -> bits(Array(h)[(k + 1):end]), values(st.recorder.hist))...)

# The closed state with ionization (WP-056): the atoms (their cells, offsets, weights, status and
# count, and the extras `pending`, `emitted` and each level's bound population), the electrons and
# the counter of lost births are in `evolved`, each once, so a snapshot after any step restored
# into a new state, or into a used one whose scratch has been overwritten, continues with the bits
# of the uninterrupted run: saved before the first birth, at a birth step and the step after it,
# and after shifts, with electrons leaving and births that found no slot (`ion_window` with 2048
# slots, 130 steps, Reference mode). A `state_view` over copies of a state's evolved arrays holds
# the ionization in those copies and continues with the same bits.
function body_T777(be)
    for T in TYPES
        prob = ion_window(T; nsteps = 130, density = seeded(T, 1) / 200, capacity = 2048)
        p = params(prob)
        full = ion_state(prob; be)
        fr = ionization_frame(full)
        ev = evolved(full)
        mine = (fr.atoms.cell..., fr.atoms.offset..., fr.atoms.w, fr.atoms.status, fr.atoms.count, values(fr.atoms.extras)...,
                fr.electrons.cell..., fr.electrons.offset..., fr.electrons.u..., fr.electrons.w, fr.electrons.status, fr.electrons.count, fr.lost)
        @test all(a -> count(x -> x === a, ev) == 1, mine)
        @test !any(a -> any(x -> x === a, ev), (fr.flags, values(fr.electrons.extras)...))
        ks = (0, 60, 80, 81, 118)
        snaps = EMSnapshot[snapshot(full, 0)]
        done = 0
        for k in ks[2:end]
            run!(full, p, k - done; first = done + 1)
            done = k
            push!(snaps, snapshot(full, k))
        end
        run!(full, p, prob.nsteps - done; first = done + 1)
        @test ionization_lost(full) > 0 && born(full).n > 0          # the counter counts and electrons are in the window at the end
        used = ion_state(prob; be)
        run!(used, p, 97)
        for (k, snap) in zip(ks, snaps)
            fresh = EMState(prob, EMNumerics(; current = Esirkepov()); backend = be)
            restore!(fresh, snap)
            run!(fresh, p, prob.nsteps - k; first = k + 1)
            @test ion_bits(fresh, k) == ion_bits(full, k)
            restore!(used, snap)
            scramble_ionization!(used)
            run!(used, p, prob.nsteps - k; first = k + 1)
            @test ion_bits(used, k) == ion_bits(full, k)
        end
        ran = ion_state(prob; be)
        run!(ran, p, 81)
        arrays = map(copy, evolved(ran))
        at81 = map(bits, evolved(ran))
        sv = state_view(ran, arrays)
        @test all(map(===, evolved(sv), arrays))
        run!(sv, p, prob.nsteps - 81; first = 82)
        @test map(bits, evolved(sv)) == map(bits, evolved(full)) && bits(sv.K) == bits(full.K)
        @test map(bits, evolved(ran)) == at81                       # the state it was taken from is where it was
    end
end

@testset "T-777 the atoms, their electrons and the lost-birth counter are evolved state: a restart at any step continues with the uninterrupted run's bits [A4, A8, WP-059, WP-056]" begin
    body_T777(CPU())
end

# --- T-778, T-779 ---------------------------------------------------------------------------

"The parameters of T-778: the laser's a₀, the dopant's density, the level's prefactor and the section's start (c/ω₀)."
const ION_X0 = (1.0, 0.005, 0.2, 28.0)
const ION_NAMES = ("a0", "density", "prefactor", "start")

"""
The two objectives of T-778 at `x = (a0, density, prefactor, start)`, after `ion_window`'s 110
steps: the charge born (the electrons' summed weight, n_c·cells) and the born electrons' kinetic
energy Σ w·(γ − 1). No electron has left the window. `gather` is the particles' gather (WP-067).
"""
function ion_objectives(x, mode = Reference(); be = CPU(), gather = nothing)
    T = promote_type(map(typeof, x)...)
    prob = ion_window(T; a0 = x[1], density = x[2], prefactor = x[3], start = x[4])
    st = ion_state(prob; mode, be, gather)
    run!(st, params(prob), prob.nsteps)
    el = born(st)
    ionization_lost(st) == 0 || error("a birth found no slot")
    γ = sqrt.(1 .+ el.u[1] .^ 2 .+ el.u[2] .^ 2 .+ el.u[3] .^ 2)
    return [sum(el.w), sum(el.w .* (γ .- 1))]
end

# The gradient through the ionization (A7): ForwardDiff of both objectives in all four parameters
# (one Jacobian, chunk 4, Float64, Reference) against the finite-difference floor on each partial.
# The laser's amplitude reaches the born charge through the rates in the field at each atom, the
# prefactor through the rates, the density and the section's start through the atoms' weights
# (A3), and all of them reach the energy through the fields along the electrons' orbits too. No
# random number and no threshold stands between a parameter and a weight: the released weight is a
# smooth function of the field, and where the rate reaches zero it does so with its derivative.
const T778_AD = Ref{Any}(nothing)
@testset "T-778 ForwardDiff of the born charge and of the born electrons' energy through a small ionization-injection run against the finite-difference floor, four parameters [A0, A3, A7, WP-059]" begin
    x0 = collect(ION_X0)
    J0 = ion_objectives(ION_X0)
    @info "T-778 objectives" charge = J0[1] energy = J0[2]
    @test J0[1] > 0.05 && J0[2] > 1.0e-3
    ad = ForwardDiff.jacobian(x -> ion_objectives(x), x0)
    T778_AD[] = ad
    for (o, name) in enumerate(("charge", "energy")), i in eachindex(x0)
        f(xi, mode, seed) = ion_objectives(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode)[o]
        hs = x0[i] .* 2.0 .^ (-4:-2:-20)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[o, i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-778 $name, $(ION_NAMES[i])" ad = ad[o, i] fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-778 rel/floor, $name", Dict{String,Float64}())[ION_NAMES[i]] = ratio
        get!(MEASURED, "T-778 floor, $name", Dict{String,Float64}())[ION_NAMES[i]] = r.floor_rel
        @test isfinite(ad[o, i]) && ad[o, i] != 0
        @test r.rel[r.istar] <= emC("T-778") * r.floor_rel
        @test r.floor_rel <= emC("T-778", "floor_max")
    end
    # the born charge is linear in the density up to the electrons' own fields, which are weak here
    @test ad[1, 2] * x0[2] ≈ J0[1] rtol = 0.05
end

# The production element type: Dual{Float32,3} seeded in a₀, the density and the prefactor,
# against T-778's Float64 Jacobian, measured as in T-582.
@testset "T-779 Dual{Float32,3} partials of the born charge and energy against the Float64 gradient [A7, A8, WP-059]" begin
    x0 = ION_X0
    ad = T778_AD[] === nothing ? ForwardDiff.jacobian(x -> ion_objectives(x), collect(x0)) : T778_AD[]
    xd = (mkdual(x0[1], 1, 0, 0), mkdual(x0[2], 0, 1, 0), mkdual(x0[3], 0, 0, 1), mkdual(x0[4], 0, 0, 0))
    J = ion_objectives(xd)
    J0 = ion_objectives(x0)
    for (o, name) in enumerate(("charge", "energy"))
        pd = collect(Float64.(Tuple(partials(J[o]))))
        rel = maximum(abs.(pd .- ad[o, 1:3]) ./ abs.(ad[o, 1:3]))
        primal = abs(value(J[o]) - J0[o]) / abs(J0[o])
        @info "T-779 $name" partials = pd ref = ad[o, 1:3] rel primal
        get!(MEASURED, "T-779 rel", Dict{String,Float64}())[name] = rel
        get!(MEASURED, "T-779 primal rel", Dict{String,Float64}())[name] = primal
        @test J[o] isa D3 && all(isfinite, pd) && all(!iszero, pd)
        @test rel <= emC("T-779") && primal <= emC("T-779")
    end
end

# --- T-784 ----------------------------------------------------------------------------------

# The field at the atom's place, and the electron at it. A lattice of 2 × 3 atoms per cell on four
# rows of a box of 16 × 8 cells of 1/2 × 3/4 (so that neither the axes' spacings nor their points
# per cell can be exchanged unnoticed), in a prescribed static field with all three components:
# Ex = 0.3 + x/20, linear along x (curl-free, so it stays; the quadratic gather reproduces a linear
# field exactly), Ey = 1/10 and Ez = 1/5 uniform. Born every 2 steps, 8 steps:
# - the atoms are the lattice of the definition, x the faster in-cell index, each of the weight
#   density/6;
# - each atom's bound population is exp(−W·t) of the rate in |E| at its own x, with the prefactor's
#   partials (off the two columns whose stencil reaches across the periodic seam, where the linear
#   field jumps), and the populations differ along x by more than 0.1;
# - the fourth birth, in the last step, is one electron per atom, in the atoms' order behind the
#   three births before it (the electrons' count is the 4 × 384 slots offered), of the weight
#   (density/6)·q⁶·(1 − q²) of its atom's rate, and it started at its atom: its position less the
#   step's displacement u/γ·Δt is the atom's, on both axes;
# - the state's kinetic energy is the born electrons' Σ w·(γ − 1) (the species has no weight);
# - the ions' charge deposited twice into one array is the ions' charge once.
function body_T784(be)
    C = emC("T-784")
    for T in TYPES
        dens, pref = seeded(T, 1) / 100, seeded(T, 1) / 2
        n, every, ppc, box = 8, 2, (2, 3), (8, 6)
        a, slope, Ey0, Ez0 = T(3) / 10, T(1) / 20, T(1) / 10, T(1) / 5
        level = ADKLevel(pref, T(3), one(T))
        prob = ion_in_E(T; E0 = (a, Ey0, Ez0), levels = (level,), density = dens, ppc, every, rows = (1.2, 4.8), nsteps = n, box)
        st = ion_state(prob; current = NoCurrent(), be)
        p = params(prob)
        g, h, dt = guards(st.grid), cellsize(st.grid), prob.dt
        Ex = Array(st.fields.Ex)                                    # (Center, Node): index i at x = (i − g − ½)·Δx
        for i in axes(Ex, 1)
            Ex[i, :] .= a + slope * (T(i - g) - one(T) / 2) * h[1]
        end
        copyto!(st.fields.Ex, Ex)
        natoms = 16 * 4 * 6
        pts = [((c1 - 1 + (k1 - 0.5) / 2) / 2, (c2 - 1 + (k2 - 0.5) / 3) * 0.75) for c2 in 3:6 for c1 in 1:16 for k2 in 1:3 for k1 in 1:2]
        s = ionization_state(st)
        @test length(s.x) == natoms == length(pts) && all(s.alive)
        @test value_of.(s.x) ≈ first.(pts) rtol = 4 * base_eps(T)
        @test value_of.(s.y) ≈ last.(pts) rtol = 4 * base_eps(T)
        @test all(w -> absdiff(w, dens / 6) <= 2 * base_eps(T) * absmax(dens), s.w)
        run!(st, p, n)
        s = ionization_state(st)
        W = [adk_rate(level, sqrt((a + slope * T(x))^2 + Ey0^2 + Ez0^2)) for x in first.(pts)]
        inner = [i for i in 1:natoms if 1 < (i - 1) ÷ 6 % 16 + 1 < 16]
        err = maximum(i -> absdiff(s.bound[1][i], exp(-W[i] * dt * n)), inner) / (n * base_eps(T))
        record!("T-784", T, err, be)
        @test err <= C
        @test maximum(value_of, s.bound[1][inner]) - minimum(value_of, s.bound[1][inner]) > 1 / 10
        e = ionization_electrons(st)
        @test Array(e.count)[1] == 4 * natoms && born(st).n == 4 * natoms && ionization_lost(st) == 0
        fourth = (3 * natoms + 1):(4 * natoms)
        w4 = Array(e.w)[fourth]
        werr = maximum(i -> absdiff(w4[i], dens / 6 * exp(-W[i] * dt * 6) * -expm1(-W[i] * dt * 2)), inner) / (n * base_eps(T) * value_of(dens) / 6)
        record!("T-784 weights", T, werr, be)
        @test werr <= C
        u = map(c -> value_of.(Array(c)[fourth]), e.u)
        γ = sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2)
        start(d) = (Array(e.cell[d])[fourth] .- 1 .+ value_of.(Array(e.offset[d])[fourth])) .* value_of(h[d]) .- u[d] ./ γ .* value_of(dt)
        away(x, L) = abs(mod(x + L / 2, L) - L / 2)
        @test maximum(away.(start(1) .- first.(pts), box[1])) <= 64 * base_eps(T)
        @test maximum(away.(start(2) .- last.(pts), box[2])) <= 64 * base_eps(T)
        el = born(st)
        u2 = el.u[1] .^ 2 .+ el.u[2] .^ 2 .+ el.u[3] .^ 2
        Kb = sum(el.w .* u2 ./ (sqrt.(1 .+ u2) .+ 1))
        @test value_of(Kb) > 0 && absdiff(sum(Array(st.K)), Kb) <= 64 * base_eps(T) * absmax(Kb)
        ρi = field(be, T, st.grid)
        ionization_charge!(ρi, st)
        once = bits(ρi)
        ionization_charge!(ρi, st)
        @test bits(ρi) == once && any(!iszero, once)
    end
end

@testset "T-784 the rate is of the field at the atom's own place, all three components, and an electron starts at its atom, on a lattice of 2 × 3 per cell of unequal sides [A2, A3, A8, WP-059]" begin
    body_T784(CPU())
end

# --- T-785 ----------------------------------------------------------------------------------

# The born electrons at an open edge (WP-027). A box of 16 × 16 cells open along y, with an
# absorbing layer of three cells, in the uniform field Ey = 1/2 (normal to the edges and uniform
# along x, so it stays where the layer does not damp it), and a dopant on the four rows about the
# middle, born every 2 steps with the charge-conserving current: the field drives the electrons
# towards the lower edge, which the first of them pass within the 40 steps. They are retired there
# as the problem's particles are (fewer are alive than were born, and the alive ones' weight is
# below the weight the atoms have emitted), no birth is lost, and on the node rows off the layer
# Gauss's law with the ions' charge holds to roundoff after every step, both modes (the box starts
# without charge; the scale is T-360's of the charge plus the divergence's own of the field, which
# is the larger here: a field of 1/2 over a charge density of a few 10⁻³), while without the
# ions' charge the residual there is more than a quarter of it.
function body_T785(be)
    C = emC("T-785")
    for T in TYPES, mode in (Reference(), Fast())
        layer, n2, natoms = 3, 16, 16 * 4
        prob = ion_in_E(T; E0 = (zero(T), one(T) / 2, zero(T)), box = (8, 8), cells = (16, n2), boundaries = (:periodic, :open), rows = (3.1, 4.9),
                        every = 2, nsteps = 40, density = seeded(T, 1) / 20)
        st = ion_state(prob; mode, absorber = AbsorbingLayer(; cells = layer, strength = 1, order = 2), be)
        p = params(prob)
        free = (layer + 2):(n2 - layer)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss(st, p, k)
            worst = max(worst, Float64(maximum(r.res[:, free]) / (base_eps(T) * (r.scale + r.escale) * sqrt(k + 1))))
        end
        r = ion_gauss(st, p, prob.nsteps)
        @test maximum(r.bare[:, free]) > r.ions / 4 > 0
        s, el = ionization_state(st), born(st)
        births = (prob.nsteps ÷ 2) * natoms
        emitted = sum(s.w .* s.emitted)
        @info "T-785" T mode worst alive = el.n births left = 1 - value_of(sum(el.w)) / value_of(emitted)
        record!("T-785 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test count(x -> value_of(x) > 0, s.emitted) == natoms && ionization_lost(st) == 0
        @test 0 < el.n < births - natoms
        @test value_of(sum(el.w)) < 0.95 * value_of(emitted)
    end
end

@testset "T-785 born electrons leave through an open edge as the plasma's do: retired there, and Gauss's law with the ions' charge on the free rows [A2, A4, A8, WP-059, WP-027]" begin
    body_T785(CPU())
end

# --- T-781 ----------------------------------------------------------------------------------

"Bytes allocated by four steps with births (and a shift or more) of `ion_window` with `ppc` atoms per cell and `capacity` electron slots, after 76 steps."
function ion_step_bytes(; ppc, capacity)
    prob = ion_window(Float64; ppc, capacity)
    st = ion_state(prob)
    p = params(prob)
    run!(st, p, 76)
    return @allocated run!(st, p, 4; first = 77)
end

@testset "T-781 allocation per step with ionization does not grow with the number of atoms or the electrons' capacity [A7, WP-059]" begin
    small = ion_step_bytes(; ppc = (1, 1), capacity = 4096)
    more_atoms = ion_step_bytes(; ppc = (2, 2), capacity = 4 * 4096)
    larger_buffer = ion_step_bytes(; ppc = (1, 1), capacity = 4 * 4096)
    @info "T-781 bytes per four steps" small more_atoms larger_buffer threads = Threads.nthreads()
    if Threads.nthreads() == 1
        # each step's two sorts (the plasma's and the born electrons') may each add Base's radix count buffer
        bound = small + 4 * 2 * SORT_COUNTS_BYTES
        @test_alloc more_atoms <= bound
        @test_alloc larger_buffer <= bound
    end
end

# --- T-782 ----------------------------------------------------------------------------------

# What is refused: a description that is not one, at N = 1, with no level, a negative density, no
# capacity, a lattice or a cadence that is none; a state asked to run a problem with another
# lattice, cadence, capacity or number of levels (or with ionization where it has none, and the
# reverse), and parameters whose ionization does not fit the state's. Parameters with other rates,
# another density or another section are the state's to run (A3).
@testset "T-782 input checks of the ionization: the description, the state's configuration and the parameters [A2, A10, WP-059]" begin
    lv = ADKLevel(0.5, 3.0, 1.0)
    ok = (; density = 0.01, levels = (lv,), capacity = 100)
    @test FieldIonization(; ok...) isa FieldIonization{Float64,1}
    @test FieldIonization(; ok..., levels = lv).levels === (lv,)
    @test FieldIonization(; density = 1, levels = (lv, ADKLevel(1.0f0, 2, 3)), capacity = 5).levels isa NTuple{2,ADKLevel{Float64}}
    for bad in ((; levels = ()), (; levels = (1.0,)), (; density = -1.0), (; density = Inf), (; capacity = 0), (; ppc = (0, 1)), (; ppc = (1, 1, 1)),
                (; every = 0), (; rows = (2.0, 1.0)), (; rows = (1.0, 1.0)), (; rows = (1.0,)), (; section = (1.0, 2.0, 3.0)))
        @test_throws ArgumentError FieldIonization(; ok..., bad...)
    end
    @test FieldIonization(; ok..., density = 0).density == 0        # a dopant of no density is a dopant
    prob = ion_in_E(Float64)
    @test prob.ionization isa FieldIonization{Float64,1} && params(prob).ionization isa IonizationParams{Float64,1,Nothing}
    @test ion_in_E(Float32).ionization isa FieldIonization{Float32}
    e1 = SpeciesDescription{Float64}(:electron, -1.0, 1.0, 1.0, 0.0, 1, 4, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1)
    @test_throws ArgumentError EMProblem("1d", 8.0, 16, 0.1, 4, (e1,), 1, 2π / 8; ionization = prob.ionization)
    @test_throws ArgumentError EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species, prob.seed, prob.kmode; ionization = lv)
    st = ion_state(prob; current = NoCurrent())
    plain = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species, prob.seed, prob.kmode; E0 = prob.E0)
    @test_throws ArgumentError initialize!(st, plain)
    @test_throws ArgumentError initialize!(state_2d(plain), prob)
    cap = prob.ionization.capacity                              # the state's, so that each check below is of the one thing named
    for other in (ion_in_E(Float64; ppc = (2, 1), capacity = cap), ion_in_E(Float64; every = 2, capacity = cap), ion_in_E(Float64; rows = (1.1, 2.9)),
                  ion_in_E(Float64; capacity = 77), ion_in_E(Float64; levels = chain_levels(Float64)))
        @test other.ionization.capacity == cap || other.ionization.capacity == 77
        @test_throws ArgumentError initialize!(st, other)
    end
    @test initialize!(st, ion_in_E(Float64; density = 0.5, levels = (ADKLevel(7.0, 1.0, 2.0),), section = DopantSection(; start = 1, stop = 2, ramp = 1))) === st
    p = params(prob)
    @test_throws ArgumentError step!(st, EMParams(p; ionization = nothing), 1)
    @test_throws ArgumentError step!(st, EMParams(p; ionization = IonizationParams{Float64}(0.01, chain_levels(Float64))), 1)
    @test_throws ArgumentError step!(state_2d(plain), EMParams(params(plain); ionization = p.ionization), 1)
    @test step!(st, EMParams(p; ionization = IonizationParams{Float64}(0.5, (ADKLevel(7.0, 1.0, 2.0),))), 1) === st
    # parameters rebuilt from parameters keep the ionization, its section too
    pw = params(ion_window(Float64))
    @test pw.ionization.section === DopantSection(; start = 28.0, stop = 34.0, ramp = 3.0)
    @test EMParams(pw; seed = pw.seed).ionization === pw.ionization
    @test EMParams{Float64}(; species = pw.species, seed = 1, ionization = pw.ionization).ionization === pw.ionization
    for f in (ionization_electrons, ionization_state, ionization_lost)
        @test_throws ArgumentError f(state_2d(plain))
    end
    @test_throws ArgumentError ionization_charge!(similar(st.ρ), state_2d(plain))
end

# --- T-783 ----------------------------------------------------------------------------------

# The user layer (A1): `scripts/ionization/atomic.jl` makes a level's three numbers in normalized
# units from an ionization energy in eV and a laser wavelength. Against another code: FBPIC 0.27.1's
# three coefficient lines (`scripts/ionization/fbpic_reference.py` holds them, with the table of
# ionization energies FBPIC ships) give, for every level of H, He, N and Ar at 0.8 µm and
# a₀ = 0.25 to 8, the rates of `fbpic_reference.toml`, in two variants. With the atomic unit of
# energy the formula is written in (half a Hartree, the "rydberg" variant) the rates here agree
# with them to the rounding of the constants' digits (the exponent, up to several hundred,
# multiplies a relative 1e-10 of the atomic field). FBPIC itself takes hydrogen's measured
# ionization energy as that unit, which is 5.3e-4 smaller: n* differs by 2.7e-4 and the rates by
# what that does to the exponent and the power, measured here for every rate above 1e-12 ω₀ and
# held below a bound; hydrogen is where the two conventions differ in kind (FBPIC's n* is exactly
# 1). In atomic units the numbers of hydrogen with I = half a Hartree are the closed form of T-770:
# n* = 1, C² = 4, field 2, prefactor 2.
const ION_ATOMS = Module(:IonAtoms)
Base.include(ION_ATOMS, joinpath(REPO, "scripts", "ionization", "atomic.jl"))

@testset "T-783 the ADK level from an ionization energy and a wavelength: FBPIC's coefficients, hydrogen's closed form, the atomic data [A1, A7, WP-059]" begin
    A = ION_ATOMS
    ref = TOML.parsefile(joinpath(REPO, "scripts", "ionization", "fbpic_reference.toml"))
    λ, a0s = ref["wavelength_m"], ref["a0"]
    @test λ == 0.8e-6 && length(ref["levels"]) == 1 + 2 + 7 + 18
    worst = Dict("n* rydberg" => 0.0, "C² rydberg" => 0.0, "rate rydberg" => 0.0, "n* fbpic" => 0.0, "rate fbpic" => 0.0)
    for lv in ref["levels"]
        el, Z = Symbol(lv["element"]), lv["charge"]
        @test A.ionization_energy(el, Z) ≈ lv["energy_eV"] rtol = 4 * eps()
        num = A.adk_numbers(A.ionization_energy(el, Z), Z; wavelength = λ, ground_eV = A.ionization_energy(el, 1))
        level = A.adk_level(Float64, el, Z; wavelength = λ, cutoff = 1.0e6)
        rates = [adk_rate(level, a) for a in a0s]
        worst["n* rydberg"] = max(worst["n* rydberg"], abs(num.nstar / lv["rydberg"]["n_eff"] - 1))
        worst["C² rydberg"] = max(worst["C² rydberg"], abs(num.C2 / lv["rydberg"]["C2"] - 1))
        worst["n* fbpic"] = max(worst["n* fbpic"], abs(num.nstar / lv["fbpic"]["n_eff"] - 1))
        for (r, rr, rf) in zip(rates, lv["rydberg"]["rate"], lv["fbpic"]["rate"])
            @test isapprox(r, rr; rtol = emC("T-783", "rate"), atol = 1.0e-300)
            rr > 1.0e-300 && (worst["rate rydberg"] = max(worst["rate rydberg"], abs(r / rr - 1)))
            rf > 1.0e-12 && (worst["rate fbpic"] = max(worst["rate fbpic"], abs(r / rf - 1)))
        end
    end
    @info "T-783 against FBPIC's coefficients" worst
    merge!(get!(MEASURED, "T-783 relative", Dict{String,Float64}()), worst)
    @test worst["n* rydberg"] <= emC("T-783", "nstar") && worst["C² rydberg"] <= emC("T-783", "nstar")
    @test 2.6e-4 < worst["n* fbpic"] < 2.7e-4
    @test 0 < worst["rate fbpic"] <= emC("T-783", "rate_fbpic")
    # hydrogen in atomic units, with half a Hartree as its ionization energy
    hartree = A.FINE_STRUCTURE^2 * A.ELECTRON_REST_ENERGY_EV
    @test hartree ≈ 27.211386245981 rtol = 1.0e-9                 # CODATA 2022's Hartree energy in eV, as scipy 1.18.1 lists it
    h = A.adk_numbers(hartree / 2, 1; wavelength = λ)
    au_field, au_rate = A.field_unit_au(λ), A.photon_energy(λ) / A.FINE_STRUCTURE^2      # mₑcω₀/e and ω₀ in atomic units
    @test h.nstar ≈ 1 rtol = 1.0e-14
    @test h.C2 ≈ 4 rtol = 1.0e-13
    @test h.lstar ≈ 0 atol = 1.0e-14
    @test h.power ≈ 1 rtol = 1.0e-13
    @test h.field * au_field ≈ 2 rtol = 1.0e-13
    @test h.prefactor * au_rate ≈ 2 rtol = 1.0e-13
    @test A.photon_energy(λ) * A.ELECTRON_REST_ENERGY_EV ≈ 1.5498024804150033 rtol = 1.0e-14     # h·c/λ in eV
    # the level's type, the other convention of l*, and the quantum numbers' factor
    @test A.adk_level(Float32, :N, 6; wavelength = λ) isa ADKLevel{Float32}
    @test A.adk_level(Float64, :N, 6; wavelength = λ).cutoff == 64
    n6 = A.adk_numbers(A.ionization_energy(:N, 6), 6; wavelength = λ, lstar = :level)
    @test n6.lstar == n6.nstar - 1
    @test n6.C2 ≈ 2^(2 * n6.nstar) / (n6.nstar * A.gamma(2 * n6.nstar)) rtol = 1.0e-14
    @test A.adk_numbers(20.0, 1; wavelength = λ, l = 1, m = 0).prefactor ≈ 3 * A.adk_numbers(20.0, 1; wavelength = λ).prefactor rtol = 1.0e-14
    p1 = A.adk_numbers(20.0, 1; wavelength = λ, l = 1, m = 1)
    @test p1.prefactor ≈ 3 * A.adk_numbers(20.0, 1; wavelength = λ).prefactor rtol = 1.0e-14      # (2l+1)(l+|m|)!/(2^|m|·|m|!·(l−|m|)!) = 3·2/2
    @test p1.power ≈ A.adk_numbers(20.0, 1; wavelength = λ).power - 1 rtol = 1.0e-14
    @test A.adk_numbers(20.0, 1; wavelength = λ, l = 2, m = 1).prefactor ≈ 15 * A.adk_numbers(20.0, 1; wavelength = λ).prefactor rtol = 1.0e-14   # 5·3!/(2·1·1)
    # the atomic data (Mendeleev.jl): nitrogen's seven energies rise, the K shell far above the L shell; an element the
    # FBPIC comparison above does not cover has one energy per charge; and what the package does not hold is refused
    N = [A.ionization_energy(:N, Z) for Z in 1:7]
    @test N == A.ionization_energies(:N) && N isa Vector{Float64}
    @test issorted(N) && N[5] == 97.89013 && N[6] == 552.06731 && N[6] / N[5] > 5
    @test A.nuclear_charge(:Kr) == 36 == length(A.ionization_energies(:Kr)) && issorted(A.ionization_energies(:Kr))
    @test A.ionization_energy("N", 6) == N[6]
    @test_throws ArgumentError A.ionization_energy(:N, 8)
    @test_throws ArgumentError A.ionization_energy(:N, 0)
    @test_throws ArgumentError A.ionization_energy(:Xx, 1)
    @test_throws ArgumentError A.ionization_energies(:Og)          # an element without measured energies
    # what a run's manifest keeps: the package's version and the energies of the elements it used
    rec = A.atomic_record(("N", :He))
    @test rec["source"] == "Mendeleev.jl $(pkgversion(A.Mendeleev))" && startswith(rec["source"], "Mendeleev.jl 1.")
    @test rec["ionization_energies_eV"] == Dict("N" => N, "He" => [A.ionization_energy(:He, 1), A.ionization_energy(:He, 2)])
    @test_throws ArgumentError A.adk_numbers(20.0, 0; wavelength = λ)
    @test_throws ArgumentError A.adk_numbers(20.0, 1; wavelength = λ, l = 0, m = 1)
    @test_throws ArgumentError A.adk_numbers(20.0, 1; wavelength = λ, lstar = :other)
end
