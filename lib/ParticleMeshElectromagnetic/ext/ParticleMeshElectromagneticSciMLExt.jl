"""
    ParticleMeshElectromagneticSciMLExt

The SciML harness of the electromagnetic module (WP-008): the module's step as a
`SciMLBase.DiscreteProblem`, solved with OrdinaryDiffEq's `FunctionMap`. It is an adapter
around the module's own `initialize!` and `step!`, with no second implementation: a solve gives
the same bits as `initialize!` plus `run!` in `Reference()` mode (T-100).

- The state `u` is [`state_partition`](@ref): the particle and field arrays and `K`, with mixed
  element types.
- The cache is an `EMState` owned by the problem's map. It holds the workspaces, the extras and
  the diagnostic histories, which stay out of `u` because `FunctionMap` copies the state every
  step.
- `p` is the problem description, an `EMProblem`. `remake_reload` reloads the state for a new
  `p`; a plain `remake` does not (A3).
- Time counts steps: `t = k` after step `k`, with `dt = 1`, the `DiscreteProblem` convention.
  The physical time is `t·Δt` in 1/ω₀ (A1), and `saveat` takes whole step numbers.
"""
module ParticleMeshElectromagneticSciMLExt

using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: EMState, EMProblem, EMNumerics, params, initialize!, step!
import ParticleMeshElectromagnetic: state_partition, remake_reload, harness_state
using ParticleMeshCore: Species
using KernelAbstractions: CPU
using SciMLBase: SciMLBase, DiscreteProblem, DiscreteFunction, AbstractSciMLProblem, AbstractTimeseriesSolution
using RecursiveArrayTools: ArrayPartition

# --- the state: a flat tuple of arrays, and back ---------------------------------------------

species_arrays(s::Species) = (s.cell..., s.offset..., s.u..., s.w, s.status, s.count)
species_length(::Species{N}) where {N} = 2N + 6

flat_species() = ()
flat_species(s, rest...) = (species_arrays(s)..., flat_species(rest...)...)

state_arrays(st::EMState) = (flat_species(st.species...)..., values(st.fields)..., st.K)

state_partition(st::EMState) = ArrayPartition(state_arrays(st)...)

# copyto! over two tuples of arrays, unrolled (a foreach over a heterogeneous tuple dispatches
# dynamically and allocates)
copy_arrays!(::Tuple{}, ::Tuple{}) = nothing
@inline function copy_arrays!(dst::Tuple, src::Tuple)
    copyto!(first(dst), first(src))
    return copy_arrays!(Base.tail(dst), Base.tail(src))
end

@inline take(a::Tuple, ::Val{n}) where {n} = ntuple(i -> a[i], Val(n))
@inline drop(a::Tuple, ::Val{n}) where {n} = ntuple(i -> a[n + i], Val(length(a) - n))

# A species of the same type as `s` over the first arrays of `a`: same charge, mass and extras.
function species_on(s::Species{N}, a::Tuple) where {N}
    cell = ntuple(i -> a[i], Val(N))
    offset = ntuple(i -> a[N + i], Val(N))
    u = (a[2N + 1], a[2N + 2], a[2N + 3])
    return typeof(s)(cell, offset, u, a[2N + 4], a[2N + 5], a[2N + 6], s.q, s.m, s.extras)
end

species_on(::Tuple{}, a::Tuple) = ((), a)
function species_on(sp::Tuple, a::Tuple)
    s = first(sp)
    n = Val(species_length(s))
    rest, a′ = species_on(Base.tail(sp), drop(a, n))
    return (species_on(s, take(a, n)), rest...), a′
end

"""
    state_on(state, u) -> EMState

An `EMState` that shares the cache of `state` (numerics, grid, workspaces, extras, `ρ`, the
histories) and whose particle and field arrays and `K` are the arrays of the partition `u`.
Zero-copy; the array types of `u` must be those of the state.
"""
function state_on(st::EMState, u::ArrayPartition)
    species, a = species_on(st.species, u.x)
    nf = Val(length(st.fields))
    fields = NamedTuple{keys(st.fields)}(take(a, nf))
    K = only(drop(a, nf))
    return EMState(st.numerics, st.grid, species, st.ws, fields, st.ρ, K, st.hist)
end

function check_layout(st::EMState, u::ArrayPartition)
    a = state_arrays(st)
    length(u.x) == length(a) && all(map((x, y) -> typeof(x) == typeof(y) && size(x) == size(y), u.x, a)) ||
        throw(ArgumentError("the partition does not have the layout of the state (state_partition)"))
    return nothing
end

function Base.copyto!(st::EMState, u::ArrayPartition)
    check_layout(st, u)
    copy_arrays!(state_arrays(st), u.x)
    return st
end

# --- the map ------------------------------------------------------------------------------

"""
    EMMap(state, backend)

The in-place map of the harness, `m(u, uprev, p, t)`: copy `uprev` into `u`, then the module's
`step!` on `u` with the cache of `state`, as step `k = t` (`FunctionMap` passes the time at the
end of the step). `p` is the `EMProblem`, read for the time step and the diagnostic wavenumber.
`done` counts the steps of the current solve and `owner` holds the first array of the state
it stepped (a reference, so `===` cannot match a later array at a reused address): a solve must start at step 1 and take every step in order (`t0 = 0`, `dt = 1`), and
only the same integrator's state may continue, so that the histories and the state are those
of `run!`.
"""
struct EMMap{S<:EMState,B}
    state::S
    backend::B
    done::Base.RefValue{Int}
    owner::Base.RefValue{Any}
end

EMMap(state::EMState, backend) = EMMap(state, backend, Ref(0), Ref{Any}(nothing))

function (m::EMMap)(u, uprev, p, t)
    # The module's step sorts the particle buffers in place, so the next state starts as a
    # copy of the previous one (sciml-reuse.md §1).
    copy_arrays!(u.x, uprev.x)
    k = round(Int, t)
    # host-side checks of the host time and of which arrays are stepped: step 1 starts a solve,
    # every later step follows the last one on the same integrator's state
    a = first(u.x)
    (k == t && (k == 1 || (k == m.done[] + 1 && a === m.owner[]))) ||
        throw(ArgumentError("the harness takes every step from step 1 (t0 = 0, dt = 1) on one integrator; got t = $t after step $(m.done[])"))
    step!(state_on(m.state, u), params(p), k)
    m.done[] = k
    m.owner[] = a
    return nothing
end

harness_map(prob::AbstractSciMLProblem) = harness_map(prob.f.f)
harness_map(sol::AbstractTimeseriesSolution) = harness_map(sol.prob)
harness_map(m::EMMap) = m
harness_map(f) = throw(ArgumentError("not a problem of the ParticleMeshElectromagnetic harness: map $(typeof(f))"))

harness_state(x::Union{AbstractSciMLProblem,AbstractTimeseriesSolution}) = harness_map(x).state

# --- the problem --------------------------------------------------------------------------

function harness_problem(problem::EMProblem, numerics::EMNumerics, backend, tspan; kwargs...)
    iszero(first(tspan)) || throw(ArgumentError("tspan must start at step 0, got $(first(tspan))"))
    last(tspan) <= problem.nsteps ||
        throw(ArgumentError("tspan ends at step $(last(tspan)), past the $(problem.nsteps) steps of the problem"))
    get(kwargs, :dt, 1) == 1 || throw(ArgumentError("the harness steps with dt = 1, got dt = $(kwargs[:dt])"))
    st = EMState(problem, numerics; backend)
    initialize!(st, problem)
    return DiscreteProblem(DiscreteFunction{true}(EMMap(st, backend)), state_partition(st), tspan, problem; merge((; dt = 1), kwargs)...)
end

"""
    DiscreteProblem(problem::EMProblem, numerics::EMNumerics; backend = CPU(), nsteps = problem.nsteps, kwargs...)

The module's step as a `DiscreteProblem`, to be solved with `FunctionMap()`:

- `u0 = state_partition(state)` of a new `EMState`, loaded by `initialize!(state, problem)`;
- the map runs the module's `step!` (see `EMMap`), with that state as its cache;
- `p = problem`; `tspan = (0, nsteps)` in steps, `dt = 1`. A solve takes every step from
  step 1, so a `tspan` that starts elsewhere or a `dt ≠ 1` is refused with an `ArgumentError`:
  here, or at the first step when it comes from `remake` or a `solve` keyword. A run is
  continued with the integrator interface (`init`, `step!`), not by a second solve.

`kwargs` are stored as problem keywords and passed to `solve`, for example
`saveat = [0, 10, 20]` (whole steps; the saved states are copies of the particle and field
arrays) or `save_everystep = false`. The diagnostic histories are in the cache, entry `k`
written by step `k`: `histories(harness_state(sol))`.

To change the parameters use [`remake_reload`](@ref): a plain `remake(prob; p = p′)` keeps the
weights loaded from the old `p`. One problem holds one cache, so solves of the same problem
must not run concurrently; `remake_reload` gives each problem of an ensemble its own.
"""
function SciMLBase.DiscreteProblem(problem::EMProblem, numerics::EMNumerics; backend = CPU(),
                                   nsteps::Integer = problem.nsteps, kwargs...)
    return harness_problem(problem, numerics, backend, (0.0, Float64(nsteps)); kwargs...)
end

function remake_reload(prob::DiscreteProblem; p::EMProblem, tspan = prob.tspan, kwargs...)
    m = harness_map(prob)
    return harness_problem(p, m.state.numerics, m.backend, tspan; merge(NamedTuple(prob.kwargs), kwargs)...)
end

end
