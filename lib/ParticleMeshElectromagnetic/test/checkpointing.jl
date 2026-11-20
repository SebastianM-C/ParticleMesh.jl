# WP-080: checkpointed reverse mode through Checkpointing.jl with the storage of the evolved arrays
# (`checkpoint_storage`, the extension on Checkpointing): T-1112 to T-1114.
#
# Enzyme cannot differentiate the step's own kernels on the CPU backend (PoCL, #509), so the CPU
# checks split the work: T-1112 runs the real 2D and 3D steps through Revolve's forward schedule and
# checks every checkpoint it stores and the working copy it leaves for the reverse sweep against the
# uninterrupted run, bit for bit; T-1113 differentiates a loop of the engine's transfers whose
# reverse rules run on the CPU (WP-077, WP-080: the sort, the gather and the deposition) through
# `@ad_checkpoint` and checks the gradient against the plain reverse pass of the same loop. The
# GPU check through the full step is `scripts/enzyme/checkpointed.jl`.

using Checkpointing: Checkpointing, Revolve, @ad_checkpoint
using Enzyme: Enzyme, autodiff, Reverse, Active, Const, Duplicated

const CKPT_EXT = Base.get_extension(ParticleMeshElectromagnetic, :ParticleMeshElectromagneticCheckpointingExt)

"Copies of a state's evolved arrays on the host."
evolved_copy(st) = map(Array, evolved(st))
"Whether two tuples of arrays have the same bits."
same_arrays(a, b) = length(a) == length(b) && all(map((x, y) -> bits(Array(x)) == bits(Array(y)), a, b))

struct StepBody{S,P} <: Function
    st::S
    p::P
end
(b::StepBody)(k) = (step!(b.st, b.p, k; observer = ()); nothing)

"""
`(label, () -> (problem, numerics))` of T-1112: a 2D window over a ramp whose window shifts, with a
PML on the window's back edge, and a 3D window run; Reference mode.
"""
const CKPT_CASES = (
    ("2D window with the back-edge PML",
     () -> (window_plasma(Float64; cells = (24, 6), nsteps = 24, start = 0, drift = (1 // 2, 0, 0)), EMNumerics(; back_absorber = PML(; cells = 4)))),
    ("3D window", () -> (window_plasma_3d(Float64; cells = (16, 6, 6), nsteps = 16, start = 0, drift = (1 // 2, 0, 0)), EMNumerics())),
)

@testset "T-1112 Revolve's forward schedule with checkpoint_storage: every checkpoint is a step of the uninterrupted run and continues to its end bit for bit, 2D and 3D, device and host slots [A4, WP-080]" begin
    for (label, build) in CKPT_CASES, host in (false, true), c in (2, 4)
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
        body = StepBody(st, p)
        scheme = Checkpointing.instantiate(typeof(body), Revolve(c; storage = checkpoint_storage(; host)), n)
        last, _, stored = Checkpointing.fwd_checkpoint_for(body, scheme, 1:n)
        @test same_arrays(evolved(st), run_bits[end])                  # the forward schedule ends on the run's last state
        @test same_arrays(evolved(last.st), run_bits[n])               # the reverse sweep's working copy is the state before the last step
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
            # restored into the used state through the storage and continued, it gives the run's end
            Checkpointing.load!(body, scheme.storage, findfirst(x -> x === slot, scheme.storage.slots))
            @test same_arrays(evolved(st), run_bits[k])
            for j in k:n
                step!(st, p, j; observer = ())
            end
            d = findall(map((x, y) -> bits(Array(x)) != bits(Array(y)), evolved(st), run_bits[end]))
            isempty(d) || @info "T-1112 continued run differs" label host c k d
            @test isempty(d)
        end
        @test 0 in steps                                               # the first checkpoint is the start
        isempty(steps) || @info "T-1112" label host c steps
    end
    # a body without a state and a slot never stored are refused
    st = EMState(plasma_step_2d(Float64), EMNumerics())
    s = CKPT_EXT.EvolvedStorage{typeof(identity),false}(2)
    @test_throws r"holds no EMState" Checkpointing.save!(s, identity, 1)
    sb = CKPT_EXT.EvolvedStorage{StepBody{typeof(st),Nothing},false}(2)
    @test_throws r"restored before it was stored" Checkpointing.load!(StepBody(st, nothing), sb, 1)
    @test CKPT_EXT.body_states(StepBody(st, nothing)) === (st,)
    @test CKPT_EXT.body_states((a = 1, b = (st, Ref(st)))) === (st, st)
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
