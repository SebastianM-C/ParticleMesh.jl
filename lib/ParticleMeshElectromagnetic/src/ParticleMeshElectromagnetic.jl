"""
    ParticleMeshElectromagnetic

The electromagnetic particle-in-cell module, the only physics module of ParticleMesh (A9), for
`N = 1` periodic. It owns the physics the engine (`ParticleMeshCore`) leaves out: the field
layout (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz with their Yee staggers), the 1D Yee curl updates in
the mesh-solve slot, the relativistic Boris pusher behind a pusher type, the step composition
and the diagnostics. The current comes from the engine's charge-conserving `deposit_flux!`, so
Gauss's law holds to roundoff from the initial `cumsum_periodic!` field onwards. The
electrostatic cases (Landau damping, two-stream) run as edge cases with no transverse momenta
or fields, as Smilei runs them.

Units (A1): normalized with the laser frequency ω₀ as reference and c = 1: time in 1/ω₀, length
in c/ω₀, fields in mₑcω₀/e, density in units of n_c, current in e·n_c·c, momenta `u = γv` in
units of mₑc (per unit mass), charge in e, mass in mₑ. Maxwell's equations then read
∂ₜE = ∇×B − J, ∂ₜB = −∇×E, ∇·E = ρ, and the equation of motion du/dt = (q/m)(E + v×B).

Problem description and numerics are separate (A10, `docs/plan/architecture-draft.md` §0.5):
an [`EMProblem`](@ref) holds the grid extent, the species and the parameters in A1 units and
names no kernel; an [`EMNumerics`](@ref) chooses, by type, the shape, the deposition mode, the
pusher ([`Boris`](@ref)) and the mesh solver ([`Yee`](@ref)). Every kernel is a
KernelAbstractions kernel, generic in the element type and within the Reactant rules of §0.4.
"""
module ParticleMeshElectromagnetic

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend
import Adapt
using ParticleMeshCore
using ParticleMeshCore: ALIVE, gather_at, advance, launcher, Stagger

export FIELD_LAYOUT, RHO_STAGGER, em_fields, em_guard_width
export Yee, yee_b_half!, yee_e!, solve_fields!
export Pusher, Boris, push_momentum, lorentz_factor
export CosineProfile, SpeciesDescription, EMProblem, EMNumerics, EMParams, problem_from_case, numerics_from_case
export EMState, initialize!, step!, run!, params
export CurrentY, CurrentZ, KineticEnergy
export histories, field_energy, kinetic_energy_history, total_energy_history, mode_amplitude_history

include("layout.jl")
include("yee.jl")
include("pushers.jl")
include("problem.jl")
include("step.jl")
include("diagnostics.jl")

end
