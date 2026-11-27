# Checkpointed reverse mode (WP-080, WP-086): Enzyme's reverse pass over a run of many steps
# through Checkpointing.jl's `Revolve` scheme, with checkpoints that hold the evolved arrays of the
# states the loop steps, not copies of the whole loop body, and (with a `StepBody` as the loop body)
# a working copy for the reverse sweep that holds the evolved arrays alone too. The storage type and
# the working copy's methods live in the extension on Checkpointing
# (`ParticleMeshElectromagneticCheckpointingExt`), so this package does not depend on Checkpointing
# or Enzyme.

"""
    checkpoint_storage(; host = false)

The storage to hand to a Checkpointing.jl scheme, `Revolve(c; storage = checkpoint_storage())`,
when its loop steps `EMState`s: a checkpoint holds copies of the [`evolved`](@ref) arrays of every
state the loop body captures (on the state's backend, or in host memory with `host = true`), and a
restore writes them back with [`restore!`](@ref). Defined once Checkpointing is loaded.

    using Checkpointing, Enzyme
    function run_steps!(st, pr, n, scheme)
        checkpoint_for(StepBody(st, pr), scheme, 1:n)
        return nothing
    end
    scheme = Revolve(8; storage = checkpoint_storage())

(`@ad_checkpoint scheme for k in 1:n step!(st, pr[], k; observer = ()) end` is the same loop with a
closure as its body, which gets the whole-body working copy described below.)

Enzyme's reverse pass of `run_steps!` with `Duplicated(st, dst)` then keeps at most 8 checkpoints
and the tape of one step, and recomputes the rest of the run from the checkpoints, as the scheme
schedules it. Checkpointing.jl differentiates the loop one step per Enzyme call, so the state's
adjoint is carried from step to step in its shadow `dst`; the parameters' adjoint accumulates in
the shadow of a `Ref` (`pr = Ref(p)`, `Duplicated(pr, Ref(Enzyme.make_zero(p)))`), as a captured
immutable value could not hold it.

What the checkpoints are: each holds `sum(sizeof, evolved(state))` bytes per captured state (the
particles, the fields, the PMLs' convolution fields, the window's counters, the witness bunch, the
ionization), against the whole state that Checkpointing.jl's default `ArrayStorage` copies
(workspaces and scratch included). On top of them Checkpointing.jl copies the loop body once, at the
first u-turn, for the reverse sweep's working state (`Checkpointing.checkpoint_alloc`): of a
[`StepBody`](@ref) that copy is [`working_state`](@ref)'s, the evolved arrays and the kinetic-energy
cache, sharing the configuration and the scratch with the stepped state (WP-086); of any other
body, such as the closure that `@ad_checkpoint` makes, a `deepcopy` of every state it captures,
scratch and workspaces included. A restore is [`restore!`](@ref)'s: in
`Reference()` mode the recomputed steps are those of the first forward run bit for bit, so the
reverse pass reads the states the forward pass produced (A4); in `Fast()` mode they differ at the
atomics' roundoff.

Only `Revolve` keeps all its checkpoints in this storage. Checkpointing.jl 0.13's `Periodic`
takes it for the period's boundaries but holds the steps inside a period in its own `ArrayStorage`
(whole-body copies), so with `Periodic` the memory is not that of the evolved arrays.

The loop body must evolve nothing but the captured states: another array it writes is neither
saved nor restored. A body that captures no `EMState` is refused when the first checkpoint is
taken.
"""
function checkpoint_storage end

"""
    StepBody(state, params::Ref)
    StepBody(f, state, params)

A loop body over the steps of `state` for Checkpointing.jl's `checkpoint_for`: `body(k)` calls
`f(state, params, k)`, by default `step!(state, params[], k; observer = ())` (no observer: the
reverse sweep recomputes steps, which would record them again). The parameters sit in a `Ref`, so
that Enzyme's reverse pass gives them an adjoint (`Duplicated(pr, Ref(Enzyme.make_zero(p)))`).
`f` must evolve nothing but the state's evolved arrays ([`evolved`](@ref)), and the state must be
the only one the body steps.

With Checkpointing loaded, the reverse sweep's working copy of a `StepBody` (`checkpoint_alloc`,
which `Revolve` calls once, at the first u-turn) is a `StepBody` of the same `f` and `params` over
[`working_state`](@ref)`(state)`: new memory for the evolved arrays alone, where a closure's working
copy is a `deepcopy` of the whole state (WP-086). Its `checkpoint_copy!` is [`restore!`](@ref) of
the evolved arrays, so Checkpointing.jl's own `ArrayStorage` holds the evolved arrays per slot too.
"""
struct StepBody{F,S<:EMState,P} <: Function
    f::F
    state::S
    params::P
end
StepBody(st::EMState, pr::Ref) = StepBody(step_params_ref!, st, pr)
(b::StepBody)(k) = (b.f(b.state, b.params, k); nothing)

"`step!(state, pr[], k; observer = ())`, the default step of a [`StepBody`](@ref)."
step_params_ref!(st, pr, k) = step!(st, pr[], k; observer = ())

"""
    working_state(state) -> EMState

A state of the type of `state` over new copies of its evolved arrays ([`evolved`](@ref)), with a
kinetic-energy cache and a recorder of its own ([`state_view`](@ref)), that shares the
configuration and the scratch (workspaces, the particles' push extras, `ρ`, the sharpened fields)
with `state`: the memory of one checkpoint instead of a whole state. Stepping it continues the run
of `state` from where it was with the bits of `state`'s own continuation (Reference mode, A4;
T-1170), and leaves the evolved arrays of `state` as they are, while its scratch is the last
step's of whichever of the two stepped last. Its species do not own their arrays, so a sort writes
its permutation back into them instead of swapping them for the workspace's spare set, which
`state` keeps (`sort_by_cell!`). Use one of the two at a time.
"""
function working_state(st::EMState)
    w = state_view(st, map(copy, evolved(st)))
    w isa typeof(st) || throw(ArgumentError("working_state: the view of the state is of another type ($(typeof(w)))"))
    return w
end
