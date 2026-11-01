# The SciML harness (WP-008): the functions the extension ParticleMeshElectromagneticSciMLExt
# implements when SciMLBase and RecursiveArrayTools are loaded. The module itself depends on
# neither; it declares the names here so that they have one home and one docstring.

"""
    state_partition(state::EMState) -> ArrayPartition

The SciML state of an [`EMState`](@ref): a zero-copy `RecursiveArrayTools.ArrayPartition`
over its arrays, in this order:

1. per species: the cells and offsets per axis, the momenta `u[1:3]`, the weights, the status
   and the alive count;
2. the fields of [`FIELD_LAYOUT`](@ref), in its order;
3. the kinetic energy density `K` at n − 1/2, which the next step's diagnostics read.

The element types stay mixed (`Int32`, `UInt8` and `T`). The workspaces, the extras, the
scratch `ρ` and the diagnostic histories are not in the state: they stay in the state's
cache. `copyto!(state, u)` writes a partition back into the state's arrays.

Needs `using SciMLBase, RecursiveArrayTools` (the package extension).
"""
function state_partition end

"""
    remake_reload(prob; p, kwargs...) -> DiscreteProblem

The `remake` path that reloads the state for new parameters (A3): a new cache and a new `u0`,
loaded by [`initialize!`](@ref) from the problem description `p` (an [`EMProblem`](@ref)),
with the numerics, the backend, the time span and the problem keywords of `prob` unless
`kwargs` replaces them. The element type follows `p`, so a `p` of duals gets a dual cache.

A plain `remake(prob; p = p′)` does not reload: it keeps `u0`, and with it the weights that
`initialize!` loaded from the old parameters. Only the time step and the diagnostic
wavenumber of `p′` would take effect. Use `remake_reload` whenever `p` changes.

Needs `using SciMLBase, RecursiveArrayTools` (the package extension).
"""
function remake_reload end

"""
    harness_state(prob_or_sol) -> EMState

The cache of a harness problem, or of a solution of one: the `EMState` whose workspaces and
diagnostic histories the steps use. Read the histories with `histories(harness_state(sol))`.
Its particle and field arrays hold the problem's `u0`, not the solution.

Needs `using SciMLBase, RecursiveArrayTools` (the package extension).
"""
function harness_state end
