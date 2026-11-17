# WP-063: the ionization rate as an interface of the level's type, and the level with the
# barrier-suppression factor of Tong and Lin, T-820 to T-826 (T-827 is their rerun on the GPU,
# gpu.jl). After ionization.jl, whose helpers (`born`, `all_zero`, the user layer in `ION_ATOMS`,
# `ion_objectives`' pattern) it uses; the problems are in ionization_cases.jl.

using ParticleMeshElectromagnetic: ionization_rate_sq, level_numbers, with_level_numbers, level_eltype, tunables, tunable_names, with_tunables
import ParticleMeshElectromagnetic as PME

# --- T-820 ----------------------------------------------------------------------------------

# The Tong–Lin rate as a function: the ADK level's rate times exp(−suppression·E/field). For the
# ground state of hydrogen in atomic units (prefactor 2, field 2, power 1, T-770) with α = 6 the
# suppression is 4α·n*² = 24 and the factor exp(−12E), as Remme et al. (arXiv:2502.02697) write
# Tong and Lin's factor for hydrogen: W = (4/E)·exp(−2/(3E) − 12E). With a zero suppression the
# rate is the ADK level's, bit for bit in the plain element types (the exponent −x − 0/r is −x).
# The cutoff is the ADK level's: exactly zero for field/(3E) ≥ cutoff, zero partials for a dual
# field of value zero, C¹ through the taper. The derivatives in E and in the four numbers are the
# closed forms of the bare rate; the rate is never above the ADK level's; and it has the one
# maximum the docstring warns of, where field/(3E²) − power/E − suppression/field = 0, beyond
# which it falls.
@testset "T-820 the Tong–Lin rate: hydrogen's closed form, the ADK level's bits at zero suppression, the cutoff, the derivatives and the maximum [A1, A2, A8, WP-063]" begin
    for T in TYPES
        h = TongLinLevel(T(2), T(2), T(1), T(24))
        for E in (1 // 20, 1 // 16, 1 // 10, 3 // 10, 1)
            W = ionization_rate(h, T(E))
            @test W isa T
            @test value_of(W) ≈ 4 / E * exp(-2 / (3 * E) - 12 * E) rtol = 64 * base_eps(T)      # an exponent of up to 14, rounded twice
        end
        a = ADKLevel(T(3) / 4, T(3), T(3) / 2)                          # x = 1/E, cutoff 64
        t0 = TongLinLevel(T(3) / 4, T(3), T(3) / 2, zero(T))
        l = TongLinLevel(T(3) / 4, T(3), T(3) / 2, T(5))
        for E in (0, 1 // 100, 1 // 64, 1 // 60, 1 // 56, 1 // 20, 1 // 2, 1, 7, 300)
            Wa, W0, W = adk_rate(a, T(E)), ionization_rate(t0, T(E)), ionization_rate(l, T(E))
            T <: AbstractFloat ? (@test W0 === Wa) : (@test W0 == Wa)
            @test value_of(W) ≈ value_of(Wa) * exp(-5 * E / 3) rtol = 64 * base_eps(T) atol = floatmin(Float32)
            @test value_of(W) <= value_of(Wa)
        end
        @test ionization_rate(l, T(1) / 64) === zero(T) && ionization_rate(l, T(1) / 100) === zero(T) && ionization_rate(l, zero(T)) === zero(T)
        @test 0 < value_of(ionization_rate(l, T(1) / 60)) < value_of(l.prefactor) * 180^1.5 * exp(-60)
        @test ionization_rate(l, 1 // 2) === ionization_rate(l, T(1) / 2)       # any real field is taken in the level's type
        @test ionization_rate_sq(l, T(1) / 4) === ionization_rate(l, T(1) / 2)
        @test isnan(value_of(ionization_rate(l, T(NaN))))                       # a field that is not a number is not a zero rate
        @test adk_rate(a, T(1) / 2) === ionization_rate(a, T(1) / 2)            # the ADK level's rate by the interface
    end
    # a dual field of value zero, with partials: the rate and its partials are zero
    z = ionization_rate_sq(TongLinLevel(mkdual(2, 1, 0, 0), mkdual(2, 0, 1, 0), mkdual(1, 0, 0, 1), mkdual(3, 1, 1, 1)), mkdual(0, 1, 2, 3))
    @test z isa D3 && absmax(z) == 0
    # the derivatives of the bare rate: in E, and in the prefactor, the field, the power and the suppression
    l = TongLinLevel(0.75, 3.0, 1.5, 5.0; cutoff = 16)
    bare(E) = 0.75 * (3 / E)^1.5 * exp(-1 / E - 5 * E / 3)
    dW(E) = ForwardDiff.derivative(e -> ionization_rate(l, e), E)
    for E in (0.1, 0.5, 2.0)
        W = bare(E)
        @test ionization_rate(l, E) ≈ W rtol = 1e-13
        @test dW(E) ≈ W * (1 / E^2 - 1.5 / E - 5 / 3) rtol = 1e-12
        g = ForwardDiff.gradient(v -> ionization_rate(TongLinLevel(v[1], v[2], v[3], v[4]; cutoff = 16), E), [0.75, 3.0, 1.5, 5.0])
        @test g ≈ [W / 0.75, W * (1.5 / 3 - 1 / (3 * E) + 5 * E / 9), W * log(3 / E), -W * E / 3] rtol = 1e-12
    end
    # the taper is C¹, as the ADK level's: zero slope at the cutoff, the bare rate's value and slope at 7·cutoff/8
    Ec, Ej = 1 / 16, 1 / 14
    @test ionization_rate(l, Ec) == 0 && dW(Ec * (1 - 1e-6)) == 0 && dW(Ec * (1 + 1e-6)) / (bare(Ec) / Ec) < 1e-3
    @test ionization_rate(l, Ej * (1 - 1e-9)) ≈ bare(Ej) rtol = 1e-7
    @test dW(Ej * (1 - 1e-7)) ≈ dW(Ej * (1 + 1e-7)) rtol = 1e-4
    @test dW(Ej * (1 + 1e-7)) ≈ ForwardDiff.derivative(bare, Ej) rtol = 1e-4
    # the maximum: (5/3)·E² + (3/2)·E − 1 = 0, rising below it and falling above
    Em = (-1.5 + sqrt(1.5^2 + 4 * 5 / 3)) / (2 * 5 / 3)
    @test abs(dW(Em)) <= 1e-13 * bare(Em) / Em
    @test dW(Em / 2) > 0 && dW(2 * Em) < 0 && ionization_rate(l, 20 * Em) < 1e-6 * ionization_rate(l, Em)
    # the constructor and the interface's other three methods
    @test TongLinLevel(2, 3, 1, 4) === TongLinLevel(2.0, 3.0, 1.0, 4.0; cutoff = 64.0)
    @test TongLinLevel(2.0f0, 3, 1, 4) isa TongLinLevel{Float32} && TongLinLevel(2, 3, 1, mkdual(4, 1, 0, 0)) isa TongLinLevel{D3}
    @test TongLinLevel{Float32} <: IonizationLevel{Float32} && ADKLevel{Float64} <: IonizationLevel{Float64}
    @test level_eltype(TongLinLevel(2.0f0, 3, 1, 4)) === Float32 && level_eltype(ADKLevel(2, 3, 1)) === Float64
    @test level_as(Float32, TongLinLevel(2, 3, 1, 4; cutoff = 8)) === TongLinLevel(2.0f0, 3.0f0, 1.0f0, 4.0f0; cutoff = 8.0f0)
    @test level_numbers(TongLinLevel(2, 3, 1, 4)) === (; prefactor = 2.0, field = 3.0, power = 1.0, suppression = 4.0)
    @test level_numbers(ADKLevel(2, 3, 1)) === (; prefactor = 2.0, field = 3.0, power = 1.0)
    @test with_level_numbers(TongLinLevel(2, 3, 1, 4; cutoff = 8), Float32, (1.0f0, 2.0f0, 3.0f0, 5.0f0)) === TongLinLevel(1.0f0, 2.0f0, 3.0f0, 5.0f0; cutoff = 8.0f0)
    @test with_level_numbers(ADKLevel(2, 3, 1; cutoff = 8), Float32, (1.0f0, 2.0f0, 3.0f0)) === ADKLevel(1.0f0, 2.0f0, 3.0f0; cutoff = 8.0f0)
    @test_throws ArgumentError with_level_numbers(TongLinLevel(2, 3, 1, 4), Float64, (1.0, -2.0, 3.0, 5.0))    # the checking constructor
    for bad in ((-1, 3, 1, 4), (Inf, 3, 1, 4), (2, 0, 1, 4), (2, Inf, 1, 4), (2, 3, NaN, 4), (2, 3, 1, -1), (2, 3, 1, Inf), (2, 3, 1, NaN))
        @test_throws ArgumentError TongLinLevel(bad...)
    end
    @test_throws ArgumentError TongLinLevel(2, 3, 1, 4; cutoff = 0)
    @test ionization_rate(TongLinLevel(0, 3, 1, 4), 1.0) === 0.0
end

# --- T-821 ----------------------------------------------------------------------------------

# The user layer (A1): `scripts/ionization/atomic.jl` makes the level's four numbers from an
# ionization energy, a wavelength and the fitted α of `tong_lin.toml`.
# - Hydrogen in atomic units with half a Hartree: suppression 24, the barrier gone at 1/16 of the
#   atomic field, and the rate (4/F)·exp(−2/(3F) − 12F) back in atomic units at any wavelength's
#   conversion.
# - Nitrogen's K shell at 0.8 µm: the barrier of N⁵⁺ is gone at I²/(4·6) in atomic units, a laser
#   amplitude of about 2.2.
# - Against another method: the static-field rates of helium-like ions that Lötstedt, Ciappina and
#   Yamanouchi computed with the multiconfiguration time-dependent Hartree–Fock method (their
#   supplementary table, `tong_lin.toml`), for He, Li⁺, N⁵⁺, O⁶⁺, Ne⁸⁺ and Ar¹⁶⁺. With their α the
#   level's rate differs from theirs by the measured amounts: least in the band of fields about
#   the barrier's where an atom ionizes within a few laser cycles (0.7 to 1.75 of it), more below
#   (the factor acts where the tunnelling rate was right) and above (the fit's end). The rate
#   without the factor, the same level as an `ADKLevel`, is about twice theirs at the barrier's
#   field and more above. The energies here are measured ones and theirs a non-relativistic
#   expansion, which is part of the difference at Z = 18.
@testset "T-821 the Tong–Lin level from an ionization energy and a wavelength: hydrogen's closed form, the barrier's field, helium-like ions against published MCTDHF rates [A1, A7, WP-063]" begin
    A = ION_ATOMS
    λ = 0.8e-6
    au_field, au_rate = A.field_unit_au(λ), A.photon_energy(λ) / A.FINE_STRUCTURE^2      # mₑcω₀/e and ω₀ in atomic units
    hartree = A.FINE_STRUCTURE^2 * A.ELECTRON_REST_ENERGY_EV
    h = A.tong_lin_numbers(hartree / 2, 1; alpha = 6, wavelength = λ)
    @test h.suppression ≈ 24 rtol = 1.0e-13
    @test h.barrier_field * au_field ≈ 1 / 16 rtol = 1.0e-13
    @test (h.prefactor, h.field, h.power) === values(A.adk_numbers(hartree / 2, 1; wavelength = λ))[1:3]
    hl = TongLinLevel(h.prefactor, h.field, h.power, h.suppression)
    for F in (0.03, 1 / 16, 0.1, 0.15)
        @test ionization_rate(hl, F / au_field) * au_rate ≈ 4 / F * exp(-2 / (3 * F) - 12 * F) rtol = 1.0e-11
    end
    # the tabulated α, and what the table does not hold
    @test A.tong_lin_alpha(:H, 1) == 6 == A.tong_lin_alpha(:He, 2) == A.tong_lin_alpha(:N, 7) == A.tong_lin_alpha(:Ar, 18)
    @test A.tong_lin_alpha(:He, 1) == 7.12 && A.tong_lin_alpha(:N, 6) == 6.38 && A.tong_lin_alpha(:Ar, 17) == 5.62
    @test A.nuclear_charge(:N) == 7 && A.nuclear_charge(:Ar) == 18
    @test_throws ArgumentError A.tong_lin_alpha(:N, 5)
    @test_throws ArgumentError A.tong_lin_alpha(:N, 1)
    @test_throws ArgumentError A.tong_lin_alpha(:Xx, 1)
    @test_throws ArgumentError A.tong_lin_numbers(20.0, 1; alpha = -1, wavelength = λ)
    @test_throws ArgumentError A.tong_lin_numbers(20.0, 1; alpha = Inf, wavelength = λ)
    @test_throws ArgumentError A.adk_numbers(20.0, 1; wavelength = λ, electrons = 0)
    # nitrogen's K shell: the barrier's field, the level's defaults (the fit's conventions), its type
    I6 = A.ionization_energy(:N, 6) / hartree
    n6 = A.tong_lin_numbers(A.ionization_energy(:N, 6), 6; alpha = 6.38, wavelength = λ)
    @test n6.barrier_field * au_field ≈ I6^2 / (4 * 6) rtol = 1.0e-12
    @test 2.15 < n6.barrier_field < 2.25
    @test n6.suppression ≈ 4 * 6.38 * 6^2 / (2 * I6) rtol = 1.0e-12
    lv = A.tong_lin_level(Float64, :N, 6; wavelength = λ)
    same = A.adk_level(Float64, :N, 6; wavelength = λ, lstar = 0, electrons = 2)
    @test lv isa TongLinLevel{Float64} && (lv.prefactor, lv.field, lv.power, lv.cutoff) === (same.prefactor, same.field, same.power, same.cutoff)
    @test lv.suppression ≈ n6.suppression rtol = 1.0e-14
    @test A.adk_level(Float64, :N, 6; wavelength = λ, electrons = 2).prefactor ≈ 2 * A.adk_level(Float64, :N, 6; wavelength = λ).prefactor rtol = 1.0e-15
    @test A.tong_lin_level(Float32, :N, 7; wavelength = λ) isa TongLinLevel{Float32}
    @test A.tong_lin_level(Float64, :N, 7; wavelength = λ).prefactor == A.adk_level(Float64, :N, 7; wavelength = λ, lstar = 0).prefactor
    @test_throws ArgumentError A.tong_lin_level(Float64, :N, 5; wavelength = λ)
    l5 = A.tong_lin_level(Float64, :N, 5; wavelength = λ, alpha = 8, cutoff = 32)
    @test l5.cutoff == 32 && l5.prefactor == A.adk_level(Float64, :N, 5; wavelength = λ, lstar = 0).prefactor
    # the reference rates
    ions = A.TONG_LIN["mctdhf"]["ions"]
    @test [ion["element"] for ion in ions] == ["He", "Li", "N", "O", "Ne", "Ar"]
    worst = Dict("0.7 to 1.75 E_bs" => 0.0, "below 0.7 E_bs" => 0.0, "above 1.75 E_bs" => 0.0)
    adk_least = Inf
    for ion in ions
        el, Z = Symbol(ion["element"]), ion["Z"]
        @test A.nuclear_charge(el) == Z && length(ion["scaled_field"]) == length(ion["rate"])
        level = A.tong_lin_level(Float64, el, Z - 1; wavelength = λ, cutoff = 1.0e6)
        plain = A.adk_level(Float64, el, Z - 1; wavelength = λ, lstar = 0, electrons = 2, cutoff = 1.0e6)
        Ebs = A.tong_lin_numbers(A.ionization_energy(el, Z - 1), Z - 1; alpha = 0, wavelength = λ).barrier_field
        for (Fs, Wref) in zip(ion["scaled_field"], ion["rate"])
            E = Fs * Z^3 / 8 / au_field
            W, Wadk = ionization_rate(level, E) * au_rate, adk_rate(plain, E) * au_rate
            band = E < 0.7 * Ebs ? "below 0.7 E_bs" : E <= 1.75 * Ebs ? "0.7 to 1.75 E_bs" : "above 1.75 E_bs"
            worst[band] = max(worst[band], abs(W / Wref - 1))
            E >= 0.95 * Ebs && (adk_least = min(adk_least, Wadk / Wref))
        end
    end
    @info "T-821 against the MCTDHF rates of Lötstedt et al.: |W/W_ref − 1| per band of fields, and the least ADK rate over theirs from 0.95 E_bs" worst adk_least
    merge!(get!(MEASURED, "T-821 relative", Dict{String,Float64}()), worst)
    MEASURED["T-821 relative"]["least ADK/reference from 0.95 E_bs"] = adk_least
    @test 0 < worst["0.7 to 1.75 E_bs"] <= emC("T-821", "rate")
    @test 0 < worst["below 0.7 E_bs"] <= emC("T-821", "rate_below")
    @test 0 < worst["above 1.75 E_bs"] <= emC("T-821", "rate_above")
    @test adk_least >= emC("T-821", "adk_over_min")
end

# --- T-822 ----------------------------------------------------------------------------------

"""
A level type of the tests' own, to show that a rate law plugs in by its type: the rate
`rate·E²/(E² + knee²)`, which is smooth, zero with zero slope at zero field and saturates at
`rate`. Both numbers are parameters.
"""
struct SaturatingLevel{T} <: IonizationLevel{T}
    rate::T
    knee::T
end
PME.ionization_rate_sq(l::SaturatingLevel{T}, E2::T) where {T} = l.rate * E2 / (E2 + l.knee * l.knee)
PME.level_as(::Type{T}, l::SaturatingLevel) where {T} = SaturatingLevel{T}(T(l.rate), T(l.knee))
PME.level_numbers(l::SaturatingLevel) = (; rate = l.rate, knee = l.knee)
PME.with_level_numbers(::SaturatingLevel, ::Type{S}, x::NTuple{2,Any}) where {S} = SaturatingLevel{S}(S(x[1]), S(x[2]))

"Three levels of three types whose rates in |E| = 1/2 are of one order: the tests' own (0.4), a Tong–Lin level (about 1.1 with `s = 2`) and an ADK level (about 2.6)."
mixed_levels(::Type{T}; s = T(2), knee = one(T) / 2) where {T} =
    (SaturatingLevel(T(4) / 5, knee), TongLinLevel(one(T), T(4), T(3) / 2, s), ADKLevel(one(T), T(6), T(2)))

# A chain of three levels of three types in the constant field |E| = 1/2 (T-772's run and
# recurrence, each level's rate now its own type's): the description holds the levels as they are,
# in the problem's element type; the populations after each of 12 steps are the per-step
# recurrence with the rates `ionization_rate` gives on the host, value and partials, with the
# Tong–Lin level's suppression, the tests' level's knee and the density duals; the released
# fraction is the conserved count and all of it is born.
function body_T822(be)
    C = emC("T-822")
    for T in TYPES
        dens = seeded(T, 1) / 100
        levels = mixed_levels(T; s = 2 * seeded(T, 1), knee = seeded(T, 1) / 2)
        n = 12
        prob = ion_in_E(T; levels, density = dens, nsteps = n)
        @test prob.ionization.levels === levels && prob.ionization isa FieldIonization{T,3}
        @test params(prob).ionization.levels === levels
        st = ion_state(prob; current = NoCurrent(), be)
        p = params(prob)
        dt = prob.dt
        W = map(l -> ionization_rate(l, T(1) / 2), levels)
        @test value_of(W[1]) ≈ 0.4 rtol = 4 * base_eps(T)
        @test value_of(W[2]) ≈ 8^1.5 * exp(-8 / 3 - 2 / 8) rtol = 16 * base_eps(T)
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
        end
        record!("T-822", T, worst, be)
        @test worst <= C
        s, el = ionization_state(st), born(st)
        total = sum(s.w .* s.emitted)
        @test el.n == n * 128 && ionization_lost(st) == 0
        @test absdiff(sum(el.w), total) <= C * n * base_eps(T) * absmax(total)
        @test value_of(released) > 1
        T === D3 && @test all(x -> all(!iszero, partials(x)), s.bound[2]) && all(x -> all(!iszero, partials(x)), s.bound[3])
    end
end

@testset "T-822 levels of three types in one chain, one of them defined outside the module: each level's own rate in the per-step recurrence, value and partials [A2, A3, A8, A10, WP-063]" begin
    body_T822(CPU())
    # the description: one level or a tuple, of any types, promoted to one element type; what is not a level is refused
    lv = mixed_levels(Float64)
    d = FieldIonization(; density = 1, levels = (lv[2], SaturatingLevel(1.0f0, 2.0f0)), capacity = 5)
    @test d.levels isa Tuple{TongLinLevel{Float64},SaturatingLevel{Float64}} && d isa FieldIonization{Float64,2}
    @test FieldIonization(; density = 1.0f0, levels = TongLinLevel(1.0f0, 2, 3, 4), capacity = 5).levels isa Tuple{TongLinLevel{Float32}}
    @test FieldIonization(; density = 1, levels = [lv[3], lv[2]], capacity = 5).levels === (lv[3], lv[2])
    @test IonizationParams{Float32}(0.5, lv).levels isa Tuple{SaturatingLevel{Float32},TongLinLevel{Float32},ADKLevel{Float32}}
    @test_throws ArgumentError FieldIonization(; density = 1, levels = (lv[2], 3.0), capacity = 5)
    @test_throws ArgumentError FieldIonization(; density = 1, levels = (), capacity = 5)
    # a state steps with levels of other types than it was built with, when their number is the same
    prob = ion_in_E(Float64; levels = chain_levels(Float64))
    st = ion_state(prob; current = NoCurrent())
    p = params(prob)
    @test step!(st, EMParams(p; ionization = IonizationParams{Float64}(0.01, lv)), 1) === st
    @test_throws ArgumentError step!(st, EMParams(p; ionization = IonizationParams{Float64}(0.01, lv[1:2])), 2)
end

# --- T-823 ----------------------------------------------------------------------------------

# One Tong–Lin level in a constant field, where the scheme is the exact solution (T-771's run): in
# Ex = 1/2 the level of field 3 and power 1 has the rate W = prefactor·6·exp(−2 − suppression/6).
# After 20 steps, born every 3, the bound population is exp(−W·20·Δt), the births carry
# (density/2)·q^{3(m−1)}·(1 − q³), and with the density, the prefactor and the suppression duals
# (one partial each) the partials are those of the closed forms: the suppression reaches the
# weights through the rate alone, and no position or momentum carries a partial (A3).
function body_T823(be)
    C = emC("T-823")
    for T in TYPES
        dens, pref, supp = T === D3 ? (mkdual(0.01, 1, 0, 0), mkdual(0.5, 0, 1, 0), mkdual(1.5, 0, 0, 1)) : (T(1) / 100, T(1) / 2, T(3) / 2)
        n, every, natoms = 20, 3, 16 * 4 * 2
        prob = ion_in_E(T; levels = (TongLinLevel(pref, T(3), one(T), supp),), density = dens, ppc = (2, 1), every, rows = (1.1, 2.9), nsteps = n)
        st = ion_state(prob; current = NoCurrent(), be)
        run!(st, params(prob), n)
        dt = prob.dt
        W = pref * 6 * exp(-T(2) - supp / 6)
        q(k) = exp(-W * dt * k)
        gone(k) = -expm1(-W * dt * k)
        s, el = ionization_state(st), born(st)
        err = maximum(x -> absdiff(x, q(n)), s.bound[1]) / (n * base_eps(T) * absmax(q(n)))
        record!("T-823", T, err, be)
        @test err <= C
        @test maximum(x -> absdiff(x, gone(18)), s.emitted) <= C * n * base_eps(T) * absmax(gone(18))
        nb = n ÷ every
        @test el.n == nb * natoms && ionization_lost(st) == 0
        order = sortperm(value_of.(el.u[1]))                 # the earliest birth has the most negative u_x
        worst_w = 0.0
        for m in 1:nb
            wm = dens / 2 * q(every * (m - 1)) * gone(every)
            worst_w = max(worst_w, maximum(i -> absdiff(el.w[i], wm), order[((m - 1) * natoms + 1):(m * natoms)]) / (n * base_eps(T) * absmax(wm)))
        end
        record!("T-823 weights", T, worst_w, be)
        @test worst_w <= C
        if T === D3
            fr = ionization_frame(st)
            no_partials(a) = all(x -> all(iszero, partials(x)), Array(a))
            @test all(no_partials, (fr.atoms.offset..., fr.electrons.offset..., fr.electrons.u...))
            @test all(x -> partials(x)[3] > 0, s.bound[1])          # a stronger suppression keeps more bound
            @test all(x -> partials(x)[3] < 0 && partials(x)[2] > 0 && partials(x)[1] > 0, el.w[order[1:natoms]])
        end
    end
end

@testset "T-823 one Tong–Lin level in a constant field: the populations and the births by the suppressed rate, the suppression's partial in the weights only [A2, A3, A7, A8, WP-063]" begin
    body_T823(CPU())
end

# --- T-824 ----------------------------------------------------------------------------------

# What the interface costs the runs that do not use the new level: nothing, in bits. `ion_window`'s
# Reference run (150 steps: births, window shifts, atoms and electrons leaving) with its ADK level
# and with the Tong–Lin level of the same numbers and a zero suppression gives the same digest of
# everything the state evolves, in Float32 and in Float64, and the same born weights at step 110
# in Dual{Float32,3} too (value and partials; a zero partial's sign is not compared). With a suppression
# the run differs and its atoms have released less by step 110, when the pulse has passed them.
function body_T824(be)
    for T in TYPES
        runs = map((nothing, zero(T), T(3) / 2)) do s
            levels = s === nothing ? (ADKLevel(T(1) / 5, T(3), one(T)),) : (TongLinLevel(T(1) / 5, T(3), one(T), s),)
            prob = ion_window(T; nsteps = 150, levels)
            st = ion_state(prob; be)
            run!(st, params(prob), 110)                      # the pulse has passed the section, which is still in the window
            emitted, w = sum(ionization_state(st).emitted), born(st).w
            run!(st, params(prob), prob.nsteps - 110; first = 111)
            (; digest = ion_digest(st), w, emitted)
        end
        T <: AbstractFloat && @test runs[1].digest == runs[2].digest
        @test length(runs[1].w) > 1000 && runs[1].w == runs[2].w && runs[1].emitted == runs[2].emitted
        @test runs[3].digest != runs[1].digest
        @test 0 < value_of(runs[3].emitted) < 0.9 * value_of(runs[1].emitted)
    end
end

@testset "T-824 a Tong–Lin level of zero suppression runs the ADK level's bits through an ionization-injection run [A4, A8, WP-063]" begin
    body_T824(CPU())
end

# --- T-825 ----------------------------------------------------------------------------------

"The parameters of T-825: the laser's a₀ and the level's suppression."
const BSI_X0 = (1.0, 1.5)
const BSI_NAMES = ("a0", "suppression")

"The born charge and the born electrons' kinetic energy of `ion_window` (110 steps) with a Tong–Lin level, at `x = (a0, suppression)`."
function bsi_objectives(x, mode = Reference(); be = CPU())
    T = promote_type(map(typeof, x)...)
    prob = ion_window(T; a0 = x[1], levels = (TongLinLevel(T(1) / 5, T(3), one(T), x[2]),))
    st = ion_state(prob; mode, be)
    run!(st, params(prob), prob.nsteps)
    el = born(st)
    ionization_lost(st) == 0 || error("a birth found no slot")
    γ = sqrt.(1 .+ el.u[1] .^ 2 .+ el.u[2] .^ 2 .+ el.u[3] .^ 2)
    return [sum(el.w), sum(el.w .* (γ .- 1))]
end

# The gradient through the suppressed rate (A7): ForwardDiff of the born charge and of the born
# electrons' energy of the small ionization-injection run in the laser's amplitude and in the
# level's suppression (Float64, Reference), against the finite-difference floor on each partial
# as in T-778. The suppression enters through the rates alone (A3), and a stronger one releases
# less charge. In the production element type, Dual{Float32,3} seeded in both, the partials are
# the Float64 gradient's to T-779's bound.
@testset "T-825 ForwardDiff of the born charge and energy in the laser's amplitude and the level's suppression against the finite-difference floor, and Dual{Float32,3} against it [A0, A3, A7, A8, WP-063]" begin
    x0 = collect(BSI_X0)
    J0 = bsi_objectives(BSI_X0)
    plain = ion_objectives(ION_X0)
    @info "T-825 objectives" charge = J0[1] energy = J0[2] charge_without_suppression = plain[1]
    @test 0.02 < J0[1] < 0.9 * plain[1]
    ad = ForwardDiff.jacobian(x -> bsi_objectives(x), x0)
    for (o, name) in enumerate(("charge", "energy")), i in eachindex(x0)
        f(xi, mode, seed) = bsi_objectives(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode)[o]
        hs = x0[i] .* 2.0 .^ (-4:-2:-20)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[o, i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-825 $name, $(BSI_NAMES[i])" ad = ad[o, i] fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-825 rel/floor, $name", Dict{String,Float64}())[BSI_NAMES[i]] = ratio
        get!(MEASURED, "T-825 floor, $name", Dict{String,Float64}())[BSI_NAMES[i]] = r.floor_rel
        @test isfinite(ad[o, i]) && ad[o, i] != 0
        @test r.rel[r.istar] <= emC("T-825") * r.floor_rel
        @test r.floor_rel <= emC("T-825", "floor_max")
    end
    @test ad[1, 2] < 0 && ad[1, 1] > 0
    J = bsi_objectives((mkdual(x0[1], 1, 0, 0), mkdual(x0[2], 0, 1, 0)))
    for (o, name) in enumerate(("charge", "energy"))
        pd = collect(Float64.(Tuple(partials(J[o]))))[1:2]
        rel = maximum(abs.(pd .- ad[o, :]) ./ abs.(ad[o, :]))
        primal = abs(value(J[o]) - J0[o]) / abs(J0[o])
        @info "T-825 Dual{Float32,3}, $name" partials = pd ref = ad[o, :] rel primal
        get!(MEASURED, "T-825 Dual{Float32,3} rel", Dict{String,Float64}())[name] = rel
        get!(MEASURED, "T-825 Dual{Float32,3} primal rel", Dict{String,Float64}())[name] = primal
        @test J[o] isa D3 && all(isfinite, pd) && all(!iszero, pd)
        @test rel <= emC("T-825", "dual") && primal <= emC("T-825", "dual")
    end
end

# --- T-826 ----------------------------------------------------------------------------------

# The levels' numbers as tunables (WP-060), by the level's type: a Tong–Lin level adds
# `ionization_level<j>_suppression` after the ADK level's three, a level type from outside the
# module its own names; the vector holds the numbers in that order and written back gives the same
# description, a cutoff kept; a vector the level's constructor refuses is its ArgumentError; and the
# derivative of a run's released charge with respect to the suppression, taken through
# `with_tunables`, is the closed form's (the one level of T-823 in its constant field).
@testset "T-826 the levels' tunables by type: names, the round trip, and a gradient in the suppression through with_tunables [A3, A7, WP-060, WP-063]" begin
    for T in (Float32, Float64)
        levels = (TongLinLevel(T(1) / 2, T(3), one(T), T(3) / 2; cutoff = T(40)), ADKLevel(T(1) / 7, T(4), T(3) / 2; cutoff = T(48)),
                  SaturatingLevel(T(4) / 5, T(1) / 2))
        prob = ion_in_E(T; levels)
        for p in (prob, params(prob))
            names = tunable_names(p)
            i0 = findfirst(==(:ionization_density), names)
            @test names[i0:end] == [:ionization_density, :ionization_level1_prefactor, :ionization_level1_field, :ionization_level1_power,
                                    :ionization_level1_suppression, :ionization_level2_prefactor, :ionization_level2_field,
                                    :ionization_level2_power, :ionization_level3_rate, :ionization_level3_knee]
            v = tunables(p)
            @test v isa Vector{T} && v[(i0 + 1):end] == T[1 // 2, 3, 1, 3 // 2, 1 // 7, 4, 3 // 2, 4 // 5, 1 // 2]
            @test with_tunables(p, v) === p
            v2 = copy(v)
            v2[i0 + 4] = 2
            q = with_tunables(p, v2)
            @test q.ionization.levels[1] === TongLinLevel(T(1) / 2, T(3), one(T), T(2); cutoff = T(40)) && q.ionization.levels[2:3] === levels[2:3]
            v2[i0 + 4] = -1
            @test_throws ArgumentError with_tunables(p, v2)
        end
    end
    n = 20
    prob = ion_in_E(Float64; levels = (TongLinLevel(0.5, 3.0, 1.0, 1.5),), nsteps = n)
    i = findfirst(==(:ionization_level1_suppression), tunable_names(prob))
    function released(v)
        q = with_tunables(prob, v)
        st = ion_state(q; current = NoCurrent())
        run!(st, params(q), n)
        return sum(born(st).w)
    end
    g = ForwardDiff.gradient(released, tunables(prob))
    W = 0.5 * 6 * exp(-2 - 1.5 / 6)
    t = n * prob.dt
    @test released(tunables(prob)) ≈ 128 * 0.01 * -expm1(-W * t) rtol = 1.0e-12
    @test g[i] ≈ 128 * 0.01 * exp(-W * t) * t * W * (-1 / 6) rtol = 1.0e-11
end
