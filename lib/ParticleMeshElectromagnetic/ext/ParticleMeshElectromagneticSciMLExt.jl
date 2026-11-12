"""
    ParticleMeshElectromagneticSciMLExt

The SciML harness of the electromagnetic module (WP-008): the module's step as a
`SciMLBase.DiscreteProblem`, solved with OrdinaryDiffEq's `FunctionMap`. It is an adapter
around the module's own `initialize!` and `step!`, with no second implementation: a solve gives
the same bits as `initialize!` plus `run!` in `Reference()` mode (T-100).

- The state `u` is [`state_partition`](@ref): everything a run evolves (`evolved`: the particle
  and field arrays, a PML's convolution fields, the moving window's counters, the witness
  bunch), with mixed element types. A saved `u` with its step is a restart.
- The cache is an `EMState` owned by the problem's map. It holds the configuration, the
  workspaces, the extras and the recorder with its histories, which stay out of `u` because
  `FunctionMap` copies the state every step. Its own evolved arrays are the problem's `u0`, with
  their kinetic energy `K`; the kinetic energy of the integrator's state, which a step leaves
  for the next observation, is a buffer of the map's, apart from it.
- `p` is the problem description, an `EMProblem`, whose `params` the steps read; an `EMParams`
  is taken as it is. A plain `remake(prob; p = p′)` changes the parameters of the steps and
  keeps `u0`; `remake_reload` builds the cache and `u0` of another problem (A3).
- Time counts steps: `t = k` after step `k`, with `dt = 1`, the `DiscreteProblem` convention.
  The physical time is `t·Δt` in 1/ω₀ (A1), and `saveat` takes whole step numbers.
"""
module ParticleMeshElectromagneticSciMLExt

using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: EMState, EMProblem, EMNumerics, EMParams, params, check_setup, initialize!, step!
using ParticleMeshElectromagnetic: evolved, on_evolved, state_view, restore!, refresh_caches!, copy_arrays!, check_evolved
import ParticleMeshElectromagnetic: state_partition, remake_reload, harness_state
using KernelAbstractions: CPU
using SciMLBase: SciMLBase, DiscreteProblem, DiscreteFunction, AbstractSciMLProblem, AbstractTimeseriesSolution
using RecursiveArrayTools: ArrayPartition

# --- the state: the evolved arrays as a partition, and back -------------------------------------

state_partition(st::EMState) = ArrayPartition(evolved(st)...)

"""
    state_on(map, u) -> EMState

The integrator's state as an `EMState`: the configuration, the scratch and the recorder of the
map's cache state, the arrays of the partition `u` as its evolved arrays (`on_evolved`), and the
map's own kinetic-energy buffer, which the steps of a solve write and read in order. Zero-copy;
the array types of `u` must be those of the cache.
"""
state_on(m, u::ArrayPartition) = on_evolved(m.state, u.x, m.K, m.state.recorder)

Base.copyto!(st::EMState, u::ArrayPartition) = restore!(st, u.x)

# --- the map ------------------------------------------------------------------------------

"""
    EMMap(state, backend)

The in-place map of the harness, `m(u, uprev, p, t)`: copy `uprev` into `u`, then the module's
`step!` on `u` with the cache of `state`, as step `k = t` (`FunctionMap` passes the time at the
end of the step). `p` gives the parameters of the step: the `params` of an `EMProblem`, or an
`EMParams` itself. `done` counts the steps of the current solve and `owner` holds the first array
of the state it stepped (a reference, so `===` cannot match a later array at a reused address):
a solve must start at step 1 and take every step in order (`t0 = 0`, `dt = 1`), and only the same
integrator's state may continue, so that the histories and the state are those of `run!`. Step 1
starts from `u0` without `initialize!`: `u0` holds everything a run evolves, so a solve is the run
of its `u0`, a second solve the same run. `K` is the kinetic energy of the integrator's state, a
buffer of the map's that the last solve left, so step 1 first checks that `u0` has the layout of the cache's evolved arrays (their
number, element types and sizes: a `u0` of another problem is an `ArgumentError`, not a silent
conversion), computes `K` from it (`refresh_caches!`, as `restore!` does) and checks that a
problem given as `p` has the cache's configuration ([`check_setup`](@ref)). The cache state's own
`K` stays that of its own arrays, the problem's `u0`.
"""
struct EMMap{S<:EMState,B,KB}
    state::S
    backend::B
    K::KB
    done::Base.RefValue{Int}
    owner::Base.RefValue{Any}
end

EMMap(state::EMState, backend) = EMMap(state, backend, similar(state.K), Ref(0), Ref{Any}(nothing))

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
    pk = step_params(p)
    # a solve's first step: `u` is a copy of a `u0` that may come from anywhere (`remake`)
    k == 1 && check_evolved(m.state, u.x)
    st = state_on(m, u)
    k == 1 && (check_fit(m.state, p); refresh_caches!(st))
    step!(st, pk, k)
    m.done[] = k
    m.owner[] = a
    return nothing
end

step_params(p::EMProblem) = params(p)
step_params(p::EMParams) = p
check_fit(st::EMState, p::EMProblem) = check_setup(st, p)
check_fit(::EMState, ::EMParams) = nothing

harness_map(prob::AbstractSciMLProblem) = harness_map(prob.f.f)
harness_map(sol::AbstractTimeseriesSolution) = harness_map(sol.prob)
harness_map(m::EMMap) = m
harness_map(f) = throw(ArgumentError("not a problem of the ParticleMeshElectromagnetic harness: map $(typeof(f))"))

harness_state(prob::AbstractSciMLProblem) = harness_map(prob).state
function harness_state(sol::AbstractTimeseriesSolution)
    st = harness_map(sol).state
    tend = last(sol.prob.tspan)
    # host values of the solution: the state at the end exists only if the solve saved it
    !isempty(sol.t) && sol.t[end] == tend ||
        throw(ArgumentError("harness_state(sol) is the state at the end of the solve (step $tend), and the solution's last saved state is at step " *
                            "$(isempty(sol.t) ? "none" : sol.t[end]): save the end (save_end = true, or a saveat that includes $tend), " *
                            "or take a saved state with harness_state(sol, u); the histories are also in harness_state(sol.prob)"))
    return state_view(st, sol.u[end].x, st.recorder)
end
harness_state(x::Union{AbstractSciMLProblem,AbstractTimeseriesSolution}, u::ArrayPartition) = state_view(harness_map(x).state, u.x)

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
- `p = problem`, whose `params` the steps read; `tspan = (0, nsteps)` in steps, `dt = 1`. A solve takes every step from
  step 1, so a `tspan` that starts elsewhere or a `dt ≠ 1` is refused with an `ArgumentError`:
  here, or at the first step when it comes from `remake` or a `solve` keyword. A run is
  continued with the integrator interface (`init`, `step!`), not by a second solve.

`kwargs` are stored as problem keywords and passed to `solve`, for example
`saveat = [0, 10, 20]` (whole steps; the saved states are copies of the evolved arrays, each a
restart of its step: `restore!(state, sol.u[i].x)`) or `save_everystep = false`. The diagnostic
histories are in the cache's recorder, entry `k` written at step `k`:
`histories(harness_state(sol))`, and `harness_state(sol)` is the state at the end of the solve for
every read-out (the witness bunch's gain, the window's counters), which the solve must have
saved (it does unless `save_end = false` or a `saveat` leaves the end out);
`harness_state(sol, sol.u[i])` is that of another saved state, and the histories are also in the
problem's cache, `histories(harness_state(prob))`.

A plain `remake(prob; p = p′)` changes the parameters the steps read and keeps `u0`, with the
weights and momenta loaded from the old `p`; for a parameter of the initial load, or another
configuration, use [`remake_reload`](@ref). One problem holds one cache, so solves of the same
problem must not run concurrently; `remake_reload` gives each problem of an ensemble its own.
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
