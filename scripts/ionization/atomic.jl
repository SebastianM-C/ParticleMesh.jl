# The user layer of the field ionization (WP-059, WP-063, A1): atomic data and the conversion of a
# level to the engine's normalized units, where the laser frequency ω₀ is the reference. The
# engine's `ADKLevel` holds three numbers in those units and its `TongLinLevel` a fourth; this file
# makes them from an ionization energy in eV and a laser wavelength in metres.
#
# In normalized units an atom needs two more numbers than classical electrodynamics has: the fine
# structure constant α and the photon energy ε_γ = ħω₀/(mₑc²). With the ionization energy
# ε_I = I/(mₑc²), the atomic unit of field is α³/ε_γ in mₑcω₀/e and the atomic unit of frequency
# α²/ε_γ in ω₀, so that for the charge Z of the ion left behind
#
#     n*        = Z·α/√(2ε_I)                         the effective principal quantum number
#     field     = 2·(2ε_I)^{3/2}/ε_γ                  2(2I)^{3/2} in atomic units; α cancels
#     power     = 2n* − |m| − 1
#     prefactor = g·C²·B·ε_I/ε_γ                      g·C²·B·I in atomic units, g equivalent electrons
#     C²        = 2^{2n*}/(n*·Γ(n* + l* + 1)·Γ(n* − l*))
#     B         = (2l + 1)·(l + |m|)!/(2^{|m|}·|m|!·(l − |m|)!)
#
# and the rate is `prefactor·(field/E)^power·exp(−field/(3E))`, the static-field rate of Ammosov,
# Delone and Krainov as PIC codes that resolve the laser cycle use it. The form and the choice
# l* = n*₀ − 1 (n*₀ that of the neutral atom's first ionization) are those of FBPIC 0.27.1
# (`fbpic/particles/elementary_process/ionization/ionizer.py`, which cites Chen et al., J. Comput.
# Phys. 236 (2013), eq. 2, with l = m = 0), read in its source; `fbpic_reference.toml` holds rates
# computed with FBPIC's lines and T-783 compares. `lstar = :level` takes l* = n* − 1 per level
# instead, the convention the author remembers Smilei's documentation to state (not checked: its
# site could not be reached from the session that wrote this).
#
# Past the field at which the barrier is gone, E_bs = I²/(4Z) in atomic units, which is
# `field/(32n*)`, the tunnelling rate is too high. `tong_lin_level` makes the level with the
# empirical factor of Tong and Lin, exp(−α·(Z²/I)·E/(2I)^{3/2}) in atomic units, which in the
# engine's form exp(−suppression·E/field) is
#
#     suppression = 4α·n*²                            a pure number
#
# with the fitted α of `tong_lin.toml`, which names where the factor and the numbers were read.

using TOML: TOML
using SpecialFunctions: gamma
using ParticleMeshElectromagnetic: ADKLevel, TongLinLevel

"CODATA 2022, as scipy 1.18.1's `scipy.constants` lists them: α, mₑc² in eV, and h·c in eV·m (exact since 2019)."
const FINE_STRUCTURE = 7.2973525643e-3
const ELECTRON_REST_ENERGY_EV = 510998.95069
const HC_EV_M = 1.2398419843320026e-6

"Ionization energies in eV per element symbol (`ionization_energies.toml`, which names its source)."
const IONIZATION_ENERGIES_EV = TOML.parsefile(joinpath(@__DIR__, "ionization_energies.toml"))

"""
    ionization_energy(element, charge) -> eV

The energy that takes `element` (a symbol such as `:N`) from the charge `charge − 1` to `charge`:
`ionization_energy(:N, 6)` is that of N⁵⁺ → N⁶⁺.
"""
function ionization_energy(element, charge::Integer)
    key = String(element)
    haskey(IONIZATION_ENERGIES_EV, key) || throw(ArgumentError("no ionization energies for $key; the table has $(sort(collect(keys(IONIZATION_ENERGIES_EV))))"))
    e = IONIZATION_ENERGIES_EV[key]
    1 <= charge <= length(e) || throw(ArgumentError("$key has the charges 1 to $(length(e)) to reach, got $charge"))
    return Float64(e[charge])
end

"The photon energy ħω₀/(mₑc²) of a laser of vacuum wavelength `wavelength` (m)."
photon_energy(wavelength) = HC_EV_M / wavelength / ELECTRON_REST_ENERGY_EV

"The laser field mₑcω₀/e in atomic units of field, for a vacuum wavelength (m): `E_au = a·field_unit_au(λ)`."
field_unit_au(wavelength) = photon_energy(wavelength) / FINE_STRUCTURE^3

"""
    adk_numbers(energy_eV, charge; wavelength, ground_eV = energy_eV, l = 0, m = 0, lstar = :ground, electrons = 1)
        -> (; prefactor, field, power, nstar, lstar, C2)

The three numbers of an [`ADKLevel`](@ref) in the engine's units (the header's formulas), with the
effective quantum numbers and C²: for the ionization energy `energy_eV`, the charge `charge` of
the ion left behind, and a laser of vacuum wavelength `wavelength` (m). `lstar = :ground` takes
l* = n*₀ − 1 with n*₀ from `ground_eV`, the neutral atom's first ionization energy (FBPIC's
choice); `:level` takes l* = n* − 1; a number is taken as it is. `electrons` is the number of
equivalent electrons of the shell the level takes one from, a factor on the rate (FBPIC has none,
so the default is 1; the two 1s electrons of a helium-like ion are `electrons = 2`).
"""
function adk_numbers(energy_eV::Real, charge::Integer; wavelength::Real, ground_eV::Real = energy_eV, l::Integer = 0, m::Integer = 0,
                     lstar = :ground, electrons::Integer = 1)
    charge >= 1 || throw(ArgumentError("the charge of the ion left behind is at least 1, got $charge"))
    electrons >= 1 || throw(ArgumentError("a level has at least one electron to release, got electrons = $electrons"))
    0 <= abs(m) <= l || l == 0 == m || throw(ArgumentError("quantum numbers l = $l, m = $m need |m| ≤ l"))
    εI = energy_eV / ELECTRON_REST_ENERGY_EV
    εγ = photon_energy(wavelength)
    nstar = charge * FINE_STRUCTURE / sqrt(2 * εI)
    ls = lstar === :ground ? FINE_STRUCTURE / sqrt(2 * ground_eV / ELECTRON_REST_ENERGY_EV) - 1 :
         lstar === :level ? nstar - 1 :
         lstar isa Real ? Float64(lstar) : throw(ArgumentError("lstar is :ground, :level or a number, got $lstar"))
    C2 = 2^(2 * nstar) / (nstar * gamma(nstar + ls + 1) * gamma(nstar - ls))
    am = abs(m)
    B = (2 * l + 1) * factorial(l + am) / (2^am * factorial(am) * factorial(l - am))
    return (; prefactor = electrons * C2 * B * εI / εγ, field = 2 * (2 * εI)^(3 / 2) / εγ, power = 2 * nstar - am - 1, nstar, lstar = ls, C2)
end

"""
    adk_level(T, element, charge; wavelength, l = 0, m = 0, lstar = :ground, cutoff = 64) -> ADKLevel{T}

The [`ADKLevel`](@ref) of the transition that leaves `element` with the charge `charge`
(`adk_level(Float64, :N, 6; wavelength = 0.8e-6)` is N⁵⁺ → N⁶⁺), in the engine's units for a laser
of vacuum wavelength `wavelength` (m), from the table's ionization energies.
"""
function adk_level(::Type{T}, element, charge::Integer; wavelength::Real, cutoff = 64, kw...) where {T}
    a = adk_numbers(ionization_energy(element, charge), charge; wavelength, ground_eV = ionization_energy(element, 1), kw...)
    return ADKLevel(T(a.prefactor), T(a.field), T(a.power); cutoff = T(cutoff))
end

"The factor of Tong and Lin: its fitted numbers α and the reference rates of helium-like ions (`tong_lin.toml`, which names its sources)."
const TONG_LIN = TOML.parsefile(joinpath(@__DIR__, "tong_lin.toml"))

"The nuclear charge of `element`: the number of ionization energies the table holds for it."
function nuclear_charge(element)
    ionization_energy(element, 1)
    return length(IONIZATION_ENERGIES_EV[String(element)])
end

"""
    tong_lin_alpha(element, charge) -> α

The fitted number α of Tong and Lin's factor for the transition that leaves `element` with the
charge `charge`, where `tong_lin.toml` has one: 6 for a hydrogen-like ion (`charge` the nuclear
charge Z) and the value of Lötstedt et al. for a helium-like one (`charge = Z − 1`, the Z of their
table). Any other level is an `ArgumentError`: its α is the caller's to give.
"""
function tong_lin_alpha(element, charge::Integer)
    Z = nuclear_charge(element)
    charge == Z && return Float64(TONG_LIN["alpha"]["hydrogen_like"])
    table = TONG_LIN["alpha"]["helium_like"]
    charge == Z - 1 && haskey(table, string(Z)) && return Float64(table[string(Z)])
    throw(ArgumentError("no α of Tong and Lin's factor for $element → charge $charge: tong_lin.toml has hydrogen-like ions and the helium-like ions of Z = $(join(sort(parse.(Int, keys(table))), ", ")); pass `alpha`"))
end

"""
    tong_lin_numbers(energy_eV, charge; alpha, wavelength, kw...)
        -> (; prefactor, field, power, suppression, barrier_field, nstar, lstar, C2)

The four numbers of a [`TongLinLevel`](@ref) in the engine's units: those of [`adk_numbers`](@ref)
(which takes `kw`), `suppression = 4·alpha·n*²`, and with them the field `barrier_field =
field/(32n*)` at which the barrier is gone (mₑcω₀/e: the a₀ of a laser whose peak field reaches it).
"""
function tong_lin_numbers(energy_eV::Real, charge::Integer; alpha::Real, wavelength::Real, kw...)
    alpha >= 0 && isfinite(alpha) || throw(ArgumentError("the α of Tong and Lin's factor must be finite and ≥ 0, got $alpha"))
    a = adk_numbers(energy_eV, charge; wavelength, kw...)
    return (; a.prefactor, a.field, a.power, suppression = 4 * alpha * a.nstar^2, barrier_field = a.field / (32 * a.nstar), a.nstar, a.lstar, a.C2)
end

"""
    tong_lin_level(T, element, charge; wavelength, alpha = tong_lin_alpha(element, charge), lstar = 0,
                   electrons = (2 for a helium-like ion, else 1), cutoff = 64, kw...) -> TongLinLevel{T}

The [`TongLinLevel`](@ref) of the transition that leaves `element` with the charge `charge`
(`tong_lin_level(Float64, :N, 6; wavelength = 0.8e-6)` is N⁵⁺ → N⁶⁺), in the engine's units for a
laser of vacuum wavelength `wavelength` (m): the tunnelling rate of [`adk_numbers`](@ref) with the
barrier-suppression factor of the number `alpha`. The defaults are the conventions the tabulated
α were fitted with, which are not those of [`adk_level`](@ref): l* = 0 in the coefficient, and
both 1s electrons of a helium-like ion counted. With them the rate is the one `tong_lin.toml`'s
reference rates are held against; `adk_level(T, element, charge; wavelength, lstar = 0, electrons)`
is the same level without the factor.
"""
function tong_lin_level(::Type{T}, element, charge::Integer; wavelength::Real, alpha::Real = tong_lin_alpha(element, charge), lstar = 0,
                        electrons::Integer = charge == nuclear_charge(element) - 1 ? 2 : 1, cutoff = 64, kw...) where {T}
    a = tong_lin_numbers(ionization_energy(element, charge), charge; alpha, wavelength, ground_eV = ionization_energy(element, 1), lstar, electrons,
                         kw...)
    return TongLinLevel(T(a.prefactor), T(a.field), T(a.power), T(a.suppression); cutoff = T(cutoff))
end
