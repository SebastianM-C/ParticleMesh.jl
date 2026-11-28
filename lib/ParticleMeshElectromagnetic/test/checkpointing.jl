# WP-080: checkpointed reverse mode through Checkpointing.jl with the storage of the evolved arrays
# (`checkpoint_storage`, the extension on Checkpointing): T-1112 to T-1114.
#
# Enzyme cannot differentiate the step's own kernels on the CPU backend (PoCL, #509), so the CPU
# checks split the work: T-1112 runs the real 2D and 3D steps through Revolve's forward schedule and
# checks every checkpoint it stores and the working copy it leaves for the reverse sweep against the
# uninterrupted run, bit for bit; T-1113 differentiates a loop of the engine's transfers whose
# reverse rules run on the CPU (WP-077, WP-080: the sort, the gather and the deposition) through
# `@ad_checkpoint` and checks the gradient against the plain reverse pass of the same loop. T-1150
# (WP-084) runs Revolve's whole schedule over a small 3D trapped-beam case, its reverse sweep with
# each step's adjoint replaced by a check of the state it sees. The GPU checks through the full
# step are `scripts/enzyme/checkpointed.jl` and `scripts/enzyme/trapped3d.jl`. T-1170 and T-1171
# (WP-086): the reverse sweep's working copy of a `StepBody`, the evolved arrays alone.

using Checkpointing: Checkpointing, Revolve, @ad_checkpoint, checkpoint_for
using Enzyme: Enzyme, autodiff, Reverse, Active, Const, Duplicated
using ParticleMeshElectromagnetic: kinetic!

const CKPT_EXT = Base.get_extension(ParticleMeshElectromagnetic, :ParticleMeshElectromagneticCheckpointingExt)

"Copies of a state's evolved arrays on the host."
evolved_copy(st) = map(Array, evolved(st))
"Whether two tuples of arrays have the same bits."
same_arrays(a, b) = length(a) == length(b) && all(map((x, y) -> bits(Array(x)) == bits(Array(y)), a, b))

"A value that no evolved array holds in these runs: NaN, or the largest integer of the type."
poison(::Type{T}) where {T<:AbstractFloat} = T(NaN)
poison(::Type{T}) where {T<:Integer} = typemax(T)
"Overwrites a state's evolved arrays, so that a restore that leaves one out shows even where the run keeps it constant (a counter that stays 0)."
poison!(st) = foreach(a -> fill!(a, poison(eltype(a))), evolved(st))

"""
`(label, () -> (problem, numerics))` of T-1112: a 2D window over a ramp whose window shifts, with a
PML on the window's back edge, and a 3D window run; Reference mode.
"""
const CKPT_CASES = (
    ("2D window with the back-edge PML",
     () -> (window_plasma(Float64; cells = (24, 6), nsteps = 24, start = 0, drift = (1 // 2, 0, 0)), EMNumerics(; back_absorber = PML(; cells = 4)))),
    ("3D window", () -> (window_plasma_3d(Float64; cells = (16, 6, 6), nsteps = 16, start = 0, drift = (1 // 2, 0, 0)), EMNumerics())),
)

"""
Revolve's forward schedule with `c` checkpoints over the run of `build()`'s problem and numerics,
against the uninterrupted run (T-1112, T-1150): the schedule ends on the run's last state, its working
copy is the state before the last step, every checkpoint holds the evolved arrays of a step of the
run and, restored, continues to the run's end bit for bit. Returns the checkpoints' steps and the
uninterrupted run's last state.
"""
function revolve_forward_check(label, build, host, c)
    prob, num = build()
    n = prob.nsteps
    p = params(prob)
    ref = EMState(prob, num)
    initialize!(ref, prob)
    run_bits = [evolved_copy(ref)]
    for k in 1:n
        step!(ref, p, k; observer = ())
        push!(run_bits, evolved_copy(ref))
    end
    @test window_shifts(ref, n) > 0
    st = EMState(prob, num)
    initialize!(st, prob)
    body = StepBody(st, Ref(p))
    scheme = Checkpointing.instantiate(typeof(body), Revolve(c; storage = checkpoint_storage(; host)), n)
    last, _, stored = Checkpointing.fwd_checkpoint_for(body, scheme, 1:n)
    @test same_arrays(evolved(st), run_bits[end])                  # the forward schedule ends on the run's last state
    @test same_arrays(evolved(last.state), run_bits[n])               # the reverse sweep's working copy is the state before the last step
    slots = filter(!isnothing, scheme.storage.slots)
    @test length(slots) == stored >= 1
    steps = Int[]
    for slot in slots
        arrays = only(slot)
        @test sum(sizeof, arrays) == sum(sizeof, evolved(st))      # the evolved arrays, nothing more
        @test all(a -> a isa Array, arrays) || !host
        k = findfirst(b -> same_arrays(arrays, b), run_bits)
        @test k !== nothing
        k === nothing && continue
        push!(steps, k - 1)
        # restored into the used state through the storage and continued, it gives the run's end;
        # the restore is restore!'s, which refreshes the caches (marks K current)
        st.fresh[] = false
        poison!(st)
        Checkpointing.load!(body, scheme.storage, findfirst(x -> x === slot, scheme.storage.slots))
        @test same_arrays(evolved(st), run_bits[k])
        @test st.fresh[]
        for j in k:n
            step!(st, p, j; observer = ())
        end
        d = findall(map((x, y) -> bits(Array(x)) != bits(Array(y)), evolved(st), run_bits[end]))
        isempty(d) || @info "Revolve forward check: the continued run differs" label host c k d
        @test isempty(d)
    end
    @test 0 in steps                                               # the first checkpoint is the start
    isempty(steps) || @info "Revolve forward check" label host c steps
    return (; steps, ref, run_bits)
end

"""
Revolve's whole schedule with `c` checkpoints over the run of `build()`'s problem, the reverse sweep
as `Checkpointing.rev_checkpoint_for` runs it but with each step's adjoint replaced by a check
(T-1150): the forward schedule, then from the working copy it leaves the u-turns from the last step
down to the first, the stores (which write slots again that a u-turn freed) and the restores. At
every u-turn the working copy holds the state before that step of the uninterrupted run
(`run_bits`), bit for bit, and is then stepped, as the adjoint's own forward sweep steps it; every
restore, into a state whose evolved arrays were overwritten, gives the stored step. Returns the
u-turns' steps, the restores' steps (the steps run before the restored state) and whether a slot
was written again.
"""
function revolve_reverse_check(build, host, c, run_bits; storage = c -> checkpoint_storage(; host))
    prob, num = build()
    n = prob.nsteps
    p = params(prob)
    st = EMState(prob, num)
    initialize!(st, prob)
    b0 = StepBody(st, Ref(p))
    scheme = Checkpointing.instantiate(typeof(b0), Revolve(c; storage = storage(c)), n)
    body, storemap, check = Checkpointing.fwd_checkpoint_for(b0, scheme, 1:n)
    written = Set(values(storemap))
    rewritten = false
    turns, restores, wrong_turns, wrong_restores = Int[], Int[], Int[], Int[]
    turn!(k) = (push!(turns, k); same_arrays(evolved(body.state), run_bits[k]) || push!(wrong_turns, k); body(k))
    step = n
    turn!(step)                                                      # the first u-turn, from the working copy
    step -= 1
    while true
        a = Checkpointing.next_action!(scheme)
        if a.actionflag == Checkpointing.store
            check += 1
            storemap[a.iteration - 1] = check
            rewritten |= check in written
            push!(written, check)
            Checkpointing.save!(scheme.storage, body, check)
        elseif a.actionflag == Checkpointing.forward
            for j in a.startiteration:(a.iteration - 1)
                body(j + 1)
            end
        elseif a.actionflag == Checkpointing.uturn
            turn!(step)
            step -= 1
            haskey(storemap, a.iteration - 2) && (delete!(storemap, a.iteration - 2); check -= 1)
        elseif a.actionflag == Checkpointing.restore
            # the slot of the state after `iteration` steps (Revolve's keys count from -1)
            k = a.iteration
            push!(restores, k)
            poison!(body.state)
            Checkpointing.load!(body, scheme.storage, storemap[k - 1])
            same_arrays(evolved(body.state), run_bits[k + 1]) || push!(wrong_restores, k)
        elseif a.actionflag == Checkpointing.done
            break
        else
            error("unexpected action $(a.actionflag)")
        end
    end
    isempty(wrong_turns) && isempty(wrong_restores) || @info "Revolve reverse check: states that differ" host c wrong_turns wrong_restores
    @test isempty(wrong_turns)
    @test isempty(wrong_restores)
    @test turns == n:-1:1                                            # every step's adjoint, last first
    return (; turns, restores, rewritten)
end

@testset "T-1112 Revolve's forward schedule with checkpoint_storage: every checkpoint is a step of the uninterrupted run and continues to its end bit for bit, 2D and 3D, device and host slots [A4, WP-080]" begin
    for (label, build) in CKPT_CASES, host in (false, true), c in (2, 4)
        revolve_forward_check(label, build, host, c)
    end
    # a body without a state and a slot never stored are refused
    st = EMState(plasma_step_2d(Float64), EMNumerics())
    s = CKPT_EXT.EvolvedStorage{typeof(identity),false}(2)
    @test_throws r"holds no EMState" Checkpointing.save!(s, identity, 1)
    sb = CKPT_EXT.EvolvedStorage{typeof(StepBody(st, Ref(nothing))),false}(2)
    @test_throws r"restored before it was stored" Checkpointing.load!(StepBody(st, Ref(nothing)), sb, 1)
    @test CKPT_EXT.body_states(StepBody(st, Ref(nothing))) === (st,)
    @test CKPT_EXT.body_states((a = 1, b = (st, Ref(st)))) === (st,)
    # a state in an array or nested deep is found too, each state once
    st2 = EMState(plasma_step_2d(Float64), EMNumerics())
    @test CKPT_EXT.body_states((x = [st2], y = (a = (b = (c = (d = Ref(st),),),),), z = Any[1.0, st2])) === (st2, st)
    @test_throws r"more than 32 levels" CKPT_EXT.body_states(foldl((x, _) -> (x,), 1:40; init = st))
end

"""
T-1150's case: the 3D trapped-beam case of WP-084 (`scripts/optim/trapped3d_case.jl`) in small, in
Float32: [`window_lwfa_3d`](@ref)'s window with its antenna, open along y and z with 4-cell PMLs
there and on the window's back edge, a dopant section that the pulse ionizes while the window
moves, and the particles reading the fields through the compensated gather.
"""
function trapped3d_small()
    T = Float32
    ion = FieldIonization(; density = T(1) / 200, levels = (ADKLevel(T(1) / 5, T(3), one(T); cutoff = T(64)),), capacity = 8192, ppc = (1, 1, 1),
                          every = 1, rows = ((9.4, 15.8), (9.4, 15.8)), section = DopantSection(; start = T(28), stop = T(34), ramp = T(3)))
    prob = window_lwfa_3d(T; nsteps = 96, cells = (72, 16, 16), bd = (:open, :open), ionization = ion)
    return prob, EMNumerics(; absorber = PML(; cells = 4), back_absorber = PML(; cells = 4), gather = CompensatedGather(; k = 1))
end

@testset "T-1150 Revolve's schedule with checkpoint_storage over the 3D trapped-beam case: antenna, window, PMLs on the open faces and the back edge, ionization and the compensated gather; every checkpoint continues to the run's end bit for bit, and the reverse sweep, its adjoints replaced by checks, meets every step of the run bit for bit [A4, WP-084]" begin
    # the forward schedule into device slots, the whole schedule into host slots (on the CPU both
    # are Arrays, through the two copy paths)
    r = revolve_forward_check("3D trapped-beam case", trapped3d_small, false, 3)
    # the run does what the case does: the window moves, the pulse ionizes the dopant, and the
    # checkpoints hold the PMLs' convolution fields beside the particles and the fields
    st = r.ref
    @test window_shifts(st, 96) > 0
    @test alive_count(ionization_electrons(st)) > 0
    @test length(open_absorbers(st)) == 2 && back_absorber(st) !== nothing
    @test length(evolved(st)) > length(evolved(EMState(window_lwfa_3d(Float32; nsteps = 1), EMNumerics())))
    @test length(r.steps) >= 2
    # the counters that stay 0 in this run are evolved too (the window's, the births without a slot)
    ev = evolved(st)
    wf, fr = ParticleMeshElectromagnetic.window_frame(st), ParticleMeshElectromagnetic.ionization_frame(st)
    counters = (wf.lost..., wf.fast..., fr.lost)
    @test all(x -> any(y -> y === x, ev), counters)
    rv = revolve_reverse_check(trapped3d_small, true, 3, r.run_bits)
    @test rv.rewritten                                               # the reverse sweep wrote slots again
    @test any(isodd, rv.restores) && any(iseven, rv.restores)
    @info "T-1150 reverse schedule" restored_steps = sort(unique(rv.restores)) turns = length(rv.turns)
end

# --- T-1170 (WP-086) -----------------------------------------------------------------------------

"""
Every `Array` that `x` reaches through its fields, tuples, references and the entries of arrays of
non-bits elements, each once (by identity): the memory a value holds. Views count as their parents.
"""
reachable_arrays(x) = reachable_arrays!(Base.IdSet{Any}(), Base.IdSet{Any}(), x)
function reachable_arrays!(out, seen, x)
    if x isa Array
        push!(out, x)
        isbitstype(eltype(x)) || foreach(k -> isassigned(x, k) && reachable_arrays!(out, seen, x[k]), eachindex(x))
        return out
    end
    (isbits(x) || x isa Union{Module,Type,AbstractString,Symbol,Function}) && return out
    if ismutable(x)
        x in seen && return out
        push!(seen, x)
    end
    for k in 1:nfields(x)
        isdefined(x, k) && reachable_arrays!(out, seen, getfield(x, k))
    end
    return out
end
array_bytes(as) = sum(sizeof, as; init = 0)

"The uninterrupted run of `build()`'s problem: its problem, numerics, parameters and the evolved arrays after every step (host copies, 0 to n)."
function uninterrupted_bits(build)
    prob, num = build()
    p = params(prob)
    ref = EMState(prob, num)
    initialize!(ref, prob)
    bits_ = [evolved_copy(ref)]
    for k in 1:prob.nsteps
        step!(ref, p, k; observer = ())
        push!(bits_, evolved_copy(ref))
    end
    return prob, num, p, bits_
end

const T1170_CASES = (CKPT_CASES..., ("3D trapped-beam case", trapped3d_small))

@testset "T-1170 the reverse sweep's working copy of a StepBody holds new memory for the evolved arrays alone (and its cache and recorder), continues the run bit for bit from where it was taken, and leaves the stepped state's evolved arrays as they are, so that the state continues after it; Revolve's whole schedule through it, with checkpoint_storage and with Checkpointing.jl's ArrayStorage [A4, WP-086]" begin
    for (label, build) in T1170_CASES
        prob, num, p, run_bits = uninterrupted_bits(build)
        n = prob.nsteps
        m = n ÷ 2
        st = EMState(prob, num)
        initialize!(st, prob)
        for k in 1:m
            step!(st, p, k; observer = ())
        end
        body = StepBody(st, Ref(p))
        @test parentmodule(which(Checkpointing.checkpoint_alloc, Tuple{typeof(body)})) === CKPT_EXT
        w = Checkpointing.checkpoint_alloc(body)
        @test w isa typeof(body) && w.f === body.f && w.params === body.params && w.state !== st
        # its kinetic-energy cache is computed from its arrays and marked so, and its recorder is its own
        K_m = bits(kinetic!(deepcopy(st)).K)
        @test bits(w.state.K) == K_m && w.state.fresh[]
        @test w.state.recorder !== st.recorder
        # the memory: what the working copy holds that the state does not is the evolved arrays,
        # the kinetic-energy cache and the recorder's histories
        old = reachable_arrays(st)
        new = [a for a in reachable_arrays(w.state) if !(a in old)]
        ev = evolved(w.state)
        @test all(a -> any(b -> b === a, new), ev)                          # every evolved array is the copy's own
        @test !any(a -> any(b -> b === a, old), ev)
        rec = [a for a in reachable_arrays(w.state.recorder) if !(a in old)]
        @test array_bytes(new) == array_bytes(ev) + sizeof(w.state.K) + array_bytes(rec)
        whole = array_bytes(reachable_arrays(deepcopy(st)))
        @info "T-1170 working copy" label new_bytes = array_bytes(new) evolved_bytes = array_bytes(ev) deepcopy_bytes = whole
        @test array_bytes(new) < whole
        @test all(s -> !ParticleMeshCore.owns_arrays(s), w.state.species) && all(ParticleMeshCore.owns_arrays, st.species)
        # it is the state after m steps, and stepped it continues the run
        @test same_arrays(evolved(w.state), run_bits[m + 1])
        wrong = Int[]
        for k in (m + 1):n
            w(k)
            same_arrays(evolved(w.state), run_bits[k + 1]) || push!(wrong, k)
        end
        isempty(wrong) || @info "T-1170 the working copy's steps that differ" label wrong
        @test isempty(wrong)
        # the stepped state's evolved arrays are untouched, and it continues too, through the shared scratch
        @test same_arrays(evolved(st), run_bits[m + 1])
        # the body steps with no observer: the recorder's histories stay as they are
        recorded = [bits(a) for a in reachable_arrays(st.recorder)]
        for k in (m + 1):n
            body(k)
        end
        @test same_arrays(evolved(st), run_bits[n + 1])
        @test [bits(a) for a in reachable_arrays(st.recorder)] == recorded
        # checkpoint_copy! is a restore of the evolved arrays, after which the cache is that of them
        K_n = bits(kinetic!(deepcopy(st)).K)
        poison!(w.state)
        fill!(w.state.K, poison(eltype(w.state.K)))
        w.state.fresh[] = false
        @test Checkpointing.checkpoint_copy!(w, body) === w
        @test same_arrays(evolved(w.state), run_bits[n + 1])
        @test bits(w.state.K) == K_n && w.state.fresh[]
        @test Checkpointing.checkpoint_copy!(w, w) === w
    end
    # Revolve's whole schedule through the working copy, with Checkpointing.jl's own storage, whose
    # slots are working copies too: every u-turn and every restore meets the uninterrupted run
    prob, num, p, run_bits = uninterrupted_bits(trapped3d_small)
    rv = revolve_reverse_check(trapped3d_small, false, 3, run_bits; storage = c -> Checkpointing.ArrayStorage)
    @test rv.rewritten
end

# --- T-1113 -------------------------------------------------------------------------------------

# One step of a map made of the engine's transfers alone, whose reverse rules run on the CPU: the
# sort, gathers of Ex and Ey into u₁ and u₂, and depositions of the currents q·w·u₂/γ and
# q·w·u₃/γ into Ey and Ex. Nonlinear in the momenta and the weights, linear in the fields, and every
# step reads what the last one wrote, so a checkpoint restored wrong changes the gradient.
function transfer_step!(st::EMState{N}) where {N}
    s, ws, g = st.species[1], st.ws[1], st.grid
    nodes = ntuple(_ -> Node(), Val(N))
    sort_by_cell!(s, g, ws)
    gather!(s.u[1], st.fields.Ex, s, Quadratic, nodes, g)
    gather!(s.u[2], st.fields.Ey, s, Quadratic, nodes, g)
    deposit!(Reference(), st.fields.Ey, s, Quadratic, nodes, g, ws; quantity = CurrentY())
    deposit!(Reference(), st.fields.Ex, s, Quadratic, nodes, g, ws; quantity = CurrentZ())
    return nothing
end

"`Σ W·Ex + Σ W·Ey` of the state, written as a loop for Enzyme."
function field_loss(st, W)
    acc = 0.0
    Ex, Ey = st.fields.Ex, st.fields.Ey
    for j in eachindex(Ex)
        acc += W[j] * Ex[j] + W[j] * Ey[j]
    end
    return acc
end

function transfers_plain(st, W, n)
    for k in 1:n
        transfer_step!(st)
    end
    return field_loss(st, W)
end

function transfers_checkpointed(st, W, n, scheme)
    @ad_checkpoint scheme for k in 1:n
        transfer_step!(st)
    end
    return field_loss(st, W)
end

"The floating-point evolved arrays of a state's shadow, by place in `evolved`."
float_evolved(st) = [a for a in evolved(st) if eltype(a) <: AbstractFloat]

"Deterministic values in (-1, 1) in the shape of `a`, from `seed`."
function lcg_like(a, seed)
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    return reshape([(state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; 2 * Float64(state >> 11) / 2.0^53 - 1) for _ in 1:length(a)], size(a))
end

transfer_state(N) = (prob = N == 2 ? plasma_step_2d(Float64; nsteps = 6) : plasma_step_3d(Float64; nsteps = 6);
                     st = EMState(prob, EMNumerics(; mode = Reference())); initialize!(st, prob); st)

@testset "T-1113 Enzyme through @ad_checkpoint with checkpoint_storage: the gradient of a loop of transfers is the plain reverse pass's, 2D and 3D, Revolve with 1 to 3 checkpoints, device and host slots [A4, A7, WP-080]" begin
    C = emC("T-1113")
    n = 6
    for N in (2, 3)
        st0 = transfer_state(N)
        W = lcg_like(st0.fields.Ex, 31)
        plain = deepcopy(st0)
        dplain = Enzyme.make_zero(plain)
        autodiff(Reverse, transfers_plain, Active, Duplicated(plain, dplain), Const(W), Const(n))
        ref = float_evolved(dplain)
        scale = maximum(a -> maximum(abs, a), ref)
        @test scale > 0
        @test count(a -> any(!iszero, a), ref) >= 6                   # the weights, offsets, momenta and fields all get adjoints
        for host in (false, true), c in (1, 2, 3)
            st = deepcopy(st0)
            dst = Enzyme.make_zero(st)
            scheme = Revolve(c; storage = checkpoint_storage(; host))
            autodiff(Reverse, transfers_checkpointed, Active, Duplicated(st, dst), Const(W), Const(n), Const(scheme))
            @test same_arrays(evolved(st), evolved(plain))            # the primal ends where the plain run does
            err = maximum(map((a, b) -> maximum(abs.(a .- b); init = 0.0), float_evolved(dst), ref))
            r = record!("T-1113", Float64, err / (eps(Float64) * scale))
            r <= C || @info "T-1113" N host c r
            @test r <= C
        end
    end
end

# --- T-1171 (WP-086) -----------------------------------------------------------------------------

transfer_body(st, _, k) = transfer_step!(st)

function transfers_stepbody(st, W, n, scheme)
    checkpoint_for(StepBody(transfer_body, st, nothing), scheme, 1:n)
    return field_loss(st, W)
end

@testset "T-1171 Enzyme through checkpoint_for with a StepBody, whose reverse sweep steps the working copy over the evolved arrays: the gradient of a loop of transfers is the plain reverse pass's, 2D and 3D, Revolve with 1 to 3 checkpoints, device and host slots [A4, A7, WP-086]" begin
    C = emC("T-1113")
    n = 6
    for N in (2, 3)
        st0 = transfer_state(N)
        W = lcg_like(st0.fields.Ex, 37)
        plain = deepcopy(st0)
        dplain = Enzyme.make_zero(plain)
        autodiff(Reverse, transfers_plain, Active, Duplicated(plain, dplain), Const(W), Const(n))
        ref = float_evolved(dplain)
        scale = maximum(a -> maximum(abs, a), ref)
        @test scale > 0
        for host in (false, true), c in (1, 2, 3)
            st = deepcopy(st0)
            dst = Enzyme.make_zero(st)
            scheme = Revolve(c; storage = checkpoint_storage(; host))
            autodiff(Reverse, transfers_stepbody, Active, Duplicated(st, dst), Const(W), Const(n), Const(scheme))
            @test same_arrays(evolved(st), evolved(plain))            # the primal ends where the plain run does
            err = maximum(map((a, b) -> maximum(abs.(a .- b); init = 0.0), float_evolved(dst), ref))
            r = record!("T-1171", Float64, err / (eps(Float64) * scale))
            r <= C || @info "T-1171" N host c r
            @test r <= C
        end
    end
end
