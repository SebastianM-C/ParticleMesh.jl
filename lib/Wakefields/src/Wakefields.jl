"""
    Wakefields

The laser-wakefield user layer of ParticleMesh (WP-092), on top of `ParticleMeshElectromagnetic`.
The engine and the module work in normalized units with the laser frequency ω₀ as reference (A1);
this package holds what is not normalized and what every LWFA study needs around a run.

It holds the unit layer of the field ionization: the ionization energies of an element
(Mendeleev.jl's table) and their conversion to the normalized numbers of the module's
`ADKLevel` and `TongLinLevel` for a laser of a given vacuum wavelength ([`adk_level`](@ref),
[`tong_lin_level`](@ref), with [`adk_numbers`](@ref) and [`tong_lin_numbers`](@ref) under them),
and what a run's manifest keeps of that data ([`atomic_record`](@ref)).

It also holds the 2D optimization cases of WP-064 and WP-065: their fixed parameters
(`BCASE`, `TCASE`), their problems and numerics ([`born_problem`](@ref),
[`trapped_problem`](@ref)), the fixed laser energy ([`pulse_energy`](@ref),
[`a0_for_energy`](@ref)), and their objectives and tables ([`born_energy`](@ref),
[`trapped_energy`](@ref), [`born_summary`](@ref), [`trapped_summary`](@ref)).

The package uses only names that `ParticleMeshElectromagnetic` makes public, and adds no method to
another package's function for another package's types.
"""
module Wakefields

using TOML: TOML
using SpecialFunctions: gamma
using Mendeleev: Mendeleev, chem_elements
using Unitful: @u_str, ustrip
using ForwardDiff: ForwardDiff
using KernelAbstractions: CPU
using ParticleMeshCore
using ParticleMeshCore: ALIVE
using ParticleMeshElectromagnetic

export FINE_STRUCTURE, ELECTRON_REST_ENERGY_EV, HC_EV_M
export ionization_energies, ionization_energy, nuclear_charge, atomic_record, photon_energy, field_unit_au
export adk_numbers, adk_level, tong_lin_alpha, tong_lin_numbers, tong_lin_level
export born_problem, born_numerics, born_start, born_state, born_energy, born_objective, born_summary
export smooth_rise, back_band_end, plasma_peak_energy, plasma_fast_weight, plasma_fast_energy
export pulse_energy, a0_for_energy, trapped_point, trapped_capacity
export trapped_problem, trapped_state, trapped_parts, trapped_energy, trapped_objective, trapped_summary
public TONG_LIN, BCASE, TCASE, TRAPPED_BOX, budget, start_of, run_to_end, check_fade, check_rise, check_from
public slot_energy, slot_trapped, species_trapped

include("atomic.jl")
include("cases2d.jl")

end
