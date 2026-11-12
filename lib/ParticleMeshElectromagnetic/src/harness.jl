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
3. the kinetic energy density `K` at n − 1/2, which the next step's observer reads.

The element types stay mixed (`Int32`, `UInt8` and `T`). The workspaces, the extras, the
scratch `ρ` and the recorder with its histories are not in the partition: they stay in the
state's cache, and so does what a 2D run evolves next to the particles and the fields (a PML's
convolution fields, the moving window's counters, the witness bunch's particles), so the
partition is the whole evolved state only for a periodic run without a witness.
`copyto!(state, u)` writes a partition back into the state's arrays.

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
type, which the step refuses ([`check_params`](@ref)). `remake_reload` is for those.

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
