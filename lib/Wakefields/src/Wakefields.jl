"""
    Wakefields

The laser-wakefield user layer of ParticleMesh (WP-092), on top of `ParticleMeshElectromagnetic`.
The engine and the module work in normalized units with the laser frequency ω₀ as reference (A1);
this package holds what is not normalized and what every LWFA study needs around a run.

So far it holds the unit layer of the field ionization: the ionization energies of an element
(Mendeleev.jl's table) and their conversion to the normalized numbers of the module's
`ADKLevel` and `TongLinLevel` for a laser of a given vacuum wavelength ([`adk_level`](@ref),
[`tong_lin_level`](@ref), with [`adk_numbers`](@ref) and [`tong_lin_numbers`](@ref) under them),
and what a run's manifest keeps of that data ([`atomic_record`](@ref)).

The package uses only names that `ParticleMeshElectromagnetic` makes public, and adds no method to
another package's function for another package's types.
"""
module Wakefields

using TOML: TOML
using SpecialFunctions: gamma
using Mendeleev: Mendeleev, chem_elements
using Unitful: @u_str, ustrip
using ParticleMeshElectromagnetic: ADKLevel, TongLinLevel

export FINE_STRUCTURE, ELECTRON_REST_ENERGY_EV, HC_EV_M
export ionization_energies, ionization_energy, nuclear_charge, atomic_record, photon_energy, field_unit_au
export adk_numbers, adk_level, tong_lin_alpha, tong_lin_numbers, tong_lin_level
public TONG_LIN

include("atomic.jl")

end
