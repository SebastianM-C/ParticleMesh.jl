# The user layer of the field ionization (WP-059, A1): atomic data and the conversion of an ADK
# level to the engine's normalized units, where the laser frequency ω₀ is the reference. The
# engine's `ADKLevel` holds three numbers in those units; this file makes them from an ionization
# energy in eV and a laser wavelength in metres.
#
# In normalized units an atom needs two more numbers than classical electrodynamics has: the fine
# structure constant α and the photon energy ε_γ = ħω₀/(mₑc²). With the ionization energy
# ε_I = I/(mₑc²), the atomic unit of field is α³/ε_γ in mₑcω₀/e and the atomic unit of frequency
# α²/ε_γ in ω₀, so that for the charge Z of the ion left behind
#
#     n*        = Z·α/√(2ε_I)                         the effective principal quantum number
#     field     = 2·(2ε_I)^{3/2}/ε_γ                  2(2I)^{3/2} in atomic units; α cancels
#     power     = 2n* − |m| − 1
#     prefactor = C²·B·ε_I/ε_γ                        C²·B·I in atomic units
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

using TOML: TOML
using SpecialFunctions: gamma
using ParticleMeshElectromagnetic: ADKLevel

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
    adk_numbers(energy_eV, charge; wavelength, ground_eV = energy_eV, l = 0, m = 0, lstar = :ground)
        -> (; prefactor, field, power, nstar, lstar, C2)

The three numbers of an [`ADKLevel`](@ref) in the engine's units (the header's formulas), with the
effective quantum numbers and C²: for the ionization energy `energy_eV`, the charge `charge` of
the ion left behind, and a laser of vacuum wavelength `wavelength` (m). `lstar = :ground` takes
l* = n*₀ − 1 with n*₀ from `ground_eV`, the neutral atom's first ionization energy (FBPIC's
choice); `:level` takes l* = n* − 1; a number is taken as it is.
"""
function adk_numbers(energy_eV::Real, charge::Integer; wavelength::Real, ground_eV::Real = energy_eV, l::Integer = 0, m::Integer = 0,
                     lstar = :ground)
    charge >= 1 || throw(ArgumentError("the charge of the ion left behind is at least 1, got $charge"))
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
    return (; prefactor = C2 * B * εI / εγ, field = 2 * (2 * εI)^(3 / 2) / εγ, power = 2 * nstar - am - 1, nstar, lstar = ls, C2)
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
