"""
    ParticleMeshElectromagnetic

The electromagnetic particle-in-cell module, the only physics module of ParticleMesh (A9), for
`N = 1` and `N = 2` (1D3V and 2D3V; periodic, and at `N = 2` with an open transverse axis
through an absorbing layer, WP-027, or a convolutional PML, WP-034, and a moving window along axis 1, WP-028, and a witness bunch of test electrons whose mean energy gain is the
optimization objective, WP-047), with the current a choice of the numerics
(`Esirkepov()`, charge-conserving, or `NoCurrent()` for prescribed fields at `N = 2`). It owns the physics
the engine (`ParticleMeshCore`) leaves out: the field layout (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz
with their Yee staggers, [`field_layout`](@ref)), the Yee curl updates in the mesh-solve slot with
their Courant limit, the relativistic Boris pusher behind a pusher type, the step composition
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
pusher ([`Boris`](@ref)), the mesh solver ([`Yee`](@ref)) and the current filter
([`BinomialFilter`](@ref)). Every kernel is a
KernelAbstractions kernel, generic in the element type and within the Reactant rules of §0.4.

A run is `state = EMState(problem, numerics)`, `initialize!(state, p)` and `step!(state, p, k)`
(or `run!`) with `p = params(problem)`: the step is a function of the state and the parameters
([`EMParams`](@ref): the lasers, the densities and the profile, the witness bunch, the seed; A3),
and the state keeps what a run evolves and the configuration it was sized for, no parameter. The
diagnostics are an observer's ([`observe!`](@ref), [`HistoryRecorder`](@ref)), called at the start
of each step. The state is closed: [`evolved`](@ref) lists what a run evolves, and with the step
index it is a restart, [`snapshot`](@ref) after any step and [`restore!`](@ref) into a state built
for the same problem (to a file with [`save_snapshot`](@ref), [`load_snapshot`](@ref)). There is
one mesh solve, [`solve_fields!`](@ref), which walks the state's tuple of field modifiers
([`FieldModifier`](@ref): the absorbers, the antennas, a moving window's edges) around the
solver's sub-steps.

With SciMLBase and RecursiveArrayTools loaded, a package extension wraps the step as a
`DiscreteProblem` (WP-008): `DiscreteProblem(problem, numerics)`, solved with `FunctionMap`,
with [`state_partition`](@ref), [`remake_reload`](@ref) and [`harness_state`](@ref).
"""
module ParticleMeshElectromagnetic

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend
import Adapt
using ParticleMeshCore
using ParticleMeshCore: ALIVE, gather_at, advance, launcher, Stagger, particle_cell, particle_offset

export FIELD_LAYOUT, RHO_STAGGER, field_layout, em_fields, em_guard_width
export Yee, yee_b_half!, yee_e!, solve_fields!, courant_limit, check_courant
export FieldModifier, BHalf, EStep, modifier_params, before_substep!, after_substep!, evolved_arrays, on_arrays, field_modifiers
export open_absorber, back_absorber, antenna_planes, with_modifiers
export Pusher, Boris, Vay, HigueraCary, BorisExactRotation, QuadraticUnitary, push_momentum, push_particle, lorentz_factor
export lorentz_rhs!, particle_problem, pusher_algorithm
export CosineProfile, SpeciesDescription, EMProblem, EMNumerics, Esirkepov, NoCurrent, problem_from_case, case_seed, numerics_from_case, momentum_distribution
export EMParams, SpeciesParams, WitnessParams, params, check_params
export EMState, initialize!, step!, run!, check_setup, reset_convolution!
export HistoryRecorder, observe!
export CurrentY, CurrentZ, KineticEnergy
export histories, field_energy, kinetic_energy_history, total_energy_history, mode_amplitude_history
export evolved, EMSnapshot, snapshot, restore!, save_snapshot, load_snapshot
export state_partition, remake_reload, harness_state
export GaussianLaser, AxisTime, WavefrontTime, Antenna, AntennaPlane, AntennaPlanes, laser_energy, laser_field
export BinomialFilter, binomial!, filter_current!, binomial_transfer
export AbsorbingLayer, LayerFactors, PML, PMLFields, BackPMLFields
export WitnessBunch, witness_gains, witness_energy_gain, witness_alive
export MovingWindow, DensityRamp, WindowFrame, WindowEdges, WindowClock, window_load, window_shifts, window_back_band, window_lost, window_superluminal, window_charge!, window_background!

include("layout.jl")
include("yee.jl")
include("modifiers.jl")
include("pushers.jl")
include("test_particles.jl")
include("filter.jl")
include("problem.jl")
include("step.jl")
include("step2d.jl")
include("antenna.jl")
include("boundaries.jl")
include("pml.jl")
include("window.jl")
include("witness.jl")
include("restart.jl")
include("diagnostics.jl")
include("harness.jl")

end
