# Checkpointed reverse mode (WP-080): Enzyme's reverse pass over a run of many steps through
# Checkpointing.jl's schemes (`Revolve`, `Periodic`, `Online_r2`), with checkpoints that hold the
# evolved arrays of the states the loop steps, not copies of the whole loop body. The storage type
# lives in the extension on Checkpointing (`ParticleMeshElectromagneticCheckpointingExt`), so this
# package does not depend on Checkpointing or Enzyme.

"""
    checkpoint_storage(; host = false)

The storage to hand to a Checkpointing.jl scheme, `Revolve(c; storage = checkpoint_storage())`,
when its loop steps `EMState`s: a checkpoint holds copies of the [`evolved`](@ref) arrays of every
state the loop body captures (on the state's backend, or in host memory with `host = true`), and a
restore writes them back with [`restore!`](@ref). Defined once Checkpointing is loaded.

    using Checkpointing, Enzyme
    function run_steps!(st, pr, n, scheme)
        @ad_checkpoint scheme for k in 1:n
            step!(st, pr[], k; observer = ())
        end
        return nothing
    end
    scheme = Revolve(8; storage = checkpoint_storage())

Enzyme's reverse pass of `run_steps!` with `Duplicated(st, dst)` then keeps at most 8 checkpoints
and the tape of one step, and recomputes the rest of the run from the checkpoints, as the scheme
schedules it. Checkpointing.jl differentiates the loop one step per Enzyme call, so the state's
adjoint is carried from step to step in its shadow `dst`; the parameters' adjoint accumulates in
the shadow of a `Ref` (`pr = Ref(p)`, `Duplicated(pr, Ref(Enzyme.make_zero(p)))`), as a captured
immutable value could not hold it.

What the checkpoints are: each holds `sum(sizeof, evolved(state))` bytes per captured state (the
particles, the fields, the PMLs' convolution fields, the window's counters, the witness bunch, the
ionization), against the whole state that Checkpointing.jl's default `ArrayStorage` copies
(workspaces and scratch included). On top of them Checkpointing.jl copies the whole loop body once
(`deepcopy`), for the reverse sweep's working state. A restore is [`restore!`](@ref)'s: in
`Reference()` mode the recomputed steps are those of the first forward run bit for bit, so the
reverse pass reads the states the forward pass produced (A4); in `Fast()` mode they differ at the
atomics' roundoff.

The loop body must evolve nothing but the captured states: another array it writes is neither
saved nor restored. A body that captures no `EMState` is refused when the first checkpoint is
taken.
"""
function checkpoint_storage end
