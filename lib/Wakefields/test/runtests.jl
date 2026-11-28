# Tests of Wakefields (WP-092). The suite is small enough to run whole; T-783 and T-821 moved here
# from ParticleMeshElectromagnetic's suite with the unit layer they test.
using Test
using TOML: TOML
using Wakefields
using ParticleMeshElectromagnetic: ADKLevel, TongLinLevel, adk_rate, ionization_rate

const TOL = TOML.parsefile(joinpath(@__DIR__, "tolerances.toml"))
emC(id, key = "C") = TOL[id][key]
const MEASURED = Dict{String,Dict{String,Float64}}()
const ION_ATOMS = Wakefields

# --- T-783 ----------------------------------------------------------------------------------

# The unit layer (A1): `adk_numbers` and `adk_level` make a level's three numbers in normalized
# units from an ionization energy in eV and a laser wavelength. Against another code: FBPIC 0.27.1's
# three coefficient lines (`scripts/ionization/fbpic_reference.py` holds them, with the table of
# ionization energies FBPIC ships, and wrote `fbpic_reference.toml` next to this file) give, for every level of H, He, N and Ar at 0.8 µm and
# a₀ = 0.25 to 8, the rates of `fbpic_reference.toml`, in two variants. With the atomic unit of
# energy the formula is written in (half a Hartree, the "rydberg" variant) the rates here agree
# with them to the rounding of the constants' digits (the exponent, up to several hundred,
# multiplies a relative 1e-10 of the atomic field). FBPIC itself takes hydrogen's measured
# ionization energy as that unit, which is 5.3e-4 smaller: n* differs by 2.7e-4 and the rates by
# what that does to the exponent and the power, measured here for every rate above 1e-12 ω₀ and
# held below a bound; hydrogen is where the two conventions differ in kind (FBPIC's n* is exactly
# 1). In atomic units the numbers of hydrogen with I = half a Hartree are the closed form of T-770:
# n* = 1, C² = 4, field 2, prefactor 2.

@testset "T-783 the ADK level from an ionization energy and a wavelength: FBPIC's coefficients, hydrogen's closed form, the atomic data [A1, A7, WP-059]" begin
    A = ION_ATOMS
    ref = TOML.parsefile(joinpath(@__DIR__, "fbpic_reference.toml"))
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

# --- T-821 ----------------------------------------------------------------------------------

# The unit layer (A1): `tong_lin_numbers` and `tong_lin_level` make the level's four numbers from an
# ionization energy, a wavelength and the fitted α of `data/tong_lin.toml`.
# - Hydrogen in atomic units with half a Hartree: suppression 24, the barrier gone at 1/16 of the
#   atomic field, and the rate (4/F)·exp(−2/(3F) − 12F) back in atomic units at any wavelength's
#   conversion.
# - Nitrogen's K shell at 0.8 µm: the barrier of N⁵⁺ is gone at I²/(4·6) in atomic units, a laser
#   amplitude of about 2.2.
# - Against another method: the static-field rates of helium-like ions that Lötstedt, Ciappina and
#   Yamanouchi computed with the multiconfiguration time-dependent Hartree–Fock method (their
#   supplementary table, `data/tong_lin.toml`), for He, Li⁺, N⁵⁺, O⁶⁺, Ne⁸⁺ and Ar¹⁶⁺. With their α the
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

using Aqua
@testset "T-1243 Wakefields patches nothing: no piracy, no eval, no stale dependencies or missing compat [A6, A9, WP-092]" begin
    Aqua.test_all(Wakefields)
    for (root, _, files) in walkdir(dirname(pathof(Wakefields))), f in files
        endswith(f, ".jl") || continue
        src = read(joinpath(root, f), String)
        @test !occursin(r"@eval|\beval\(|Core\.eval|Base\.eval|include_string|Symbol\(\"ev", src)
    end
end

println("\nMeasured (compare with tolerances.toml):")
for id in sort(collect(keys(MEASURED)))
    println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
end
