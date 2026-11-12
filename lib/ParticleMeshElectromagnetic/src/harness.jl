# The SciML harness (WP-008): the functions the extension ParticleMeshElectromagneticSciMLExt
# implements when SciMLBase and RecursiveArrayTools are loaded. The module itself depends on
# neither; it declares the names here so that they have one home and one docstring.

"""
    state_partition(state::EMState) -> ArrayPartition

The SciML state of an [`EMState`](@ref): a zero-copy `RecursiveArrayTools.ArrayPartition` over
everything a run evolves, the arrays of [`evolved`](@ref)`(state)` in its order (per species the
cells, offsets, momenta, weights, status and count; the fields; at N = 2 a PML's convolution
fields, the moving window's counters and the witness bunch). The element types stay mixed
(`Int32`, `UInt8` and `T`). The workspaces, the extras, the scratch `ρ`, the kinetic energy `K`
(a cache of the momenta, rebuilt where a partition is written into a state) and the recorder with
its histories are not in it.

`copyto!(state, u)` is [`restore!`](@ref)`(state, u.x)`: it writes a partition back into the
state's arrays and rebuilds `K`, so a run from the step after the one `u` was saved at is the run
`u` came from.

Needs `using SciMLBase, RecursiveArrayTools` (the package extension).
"""
function state_partition end

"""
    remake_reload(prob; p, kwargs...) -> DiscreteProblem

The `remake` path that builds the harness problem of another problem description `p` (an
[`EMProblem`](@ref)): a new cache, sized for `p`, and a new `u0`, loaded by [`initialize!`](@ref),
with the numerics, the backend, the time span and the problem keywords of `prob` unless `kwargs`
replaces them. The element type follows `p`, so a `p` of duals gets a dual cache.

A plain `remake(prob; p = p′)` keeps the cache and `u0`. The steps then take the parameters of
`p′` (`p′` an `EMProblem`, whose `params` are taken, or an [`EMParams`](@ref)): its lasers, its
witness bunch, and the densities, profile and seed of the moving window's refill. That is the whole
change for a parameter that acts only during the steps (a laser's). It is not for one that also
enters the initial load (a density, a drift, a thermal speed, the seed, the background fields),
whose `u0` holds the old weights and momenta, nor for a `p′` of another configuration or element
type: a problem of another configuration is refused at the first step ([`check_setup`](@ref)),
parameters that do not fit by the step ([`check_params`](@ref)), while an `EMParams` cannot say
what configuration it was made for. `remake_reload` is for those.

Needs `using SciMLBase, RecursiveArrayTools` (the package extension).
"""
function remake_reload end

"""
    harness_state(sol) -> EMState
    harness_state(prob) -> EMState
    harness_state(prob_or_sol, u) -> EMState

Of a solution: the state at the end of the solve, as the plain loop leaves it. Its evolved arrays
([`evolved`](@ref): particles, fields, a PML's convolution fields, the window's counters, the
witness bunch) are those of the last saved state `sol.u[end]` (no copy), its kinetic energy `K` is
computed from them, and its recorder is the solve's: `histories(harness_state(sol))`,
`witness_energy_gain(harness_state(sol), params)`, `window_lost(harness_state(sol))`. The recorder
is the cache's, one per problem, so it holds the histories of the last solve of that problem.

Of a problem: its cache, the `EMState` whose workspaces and recorder the steps use. Its evolved
arrays are the problem's `u0`, with their own `K`, whatever has been solved since.

With a saved state `u` (`sol.u[i]`): a state over the arrays of `u` with a `K` computed from them
and an empty recorder of its own, so observing it or running on from it
(`run!(state, params, n; first = k + 1)`) touches neither the solve's histories nor its cache.
There is no copy: a run continued from it advances the arrays of `u`, so hand it `copy(u)` to
keep the solution's saved state.

The states of one problem share their scratch (workspaces, extras, `ρ`): use one at a time, and
not while an integrator of the problem is between two steps.

Needs `using SciMLBase, RecursiveArrayTools` (the package extension).
"""
function harness_state end
