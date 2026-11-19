# WP-050 spike: Enzyme reverse mode against ForwardDiff for each part of the 2D step on its own.
# For a part g (a mutation of the state) the identity ⟨W, J·V⟩ = ⟨Jᵀ·W, V⟩ holds with J its
# Jacobian over every floating-point array of the state: V a random tangent on every array of a
# Dual state (forward), W a random seed on every array's final value (reverse). Each part is
# checked from the same initialized state, which differs from the one that part sees inside a step.
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=... julia +1.13 --project=lib/ParticleMeshElectromagnetic/test \
#         scripts/enzyme/spike/pieces.jl [part ...]
using ParticleMeshCore, ParticleMeshElectromagnetic, KernelAbstractions, ForwardDiff, Enzyme
using ParticleMeshCore: Fast, Reference
using Random: Xoshiro, randn
using Printf
foreach(f -> isempty(f) || include(abspath(f)), split(get(ENV, "PM_ENZYME_PATCH", ""), ';'))
const ON_GPU = get(ENV, "PM_BACKEND", "cpu") == "cuda"
if ON_GPU
    using CUDACore
    CUDACore.allowscalar(false)
    # PM_CUDA_HEAP=<bytes>: Enzyme's dynamic tapes live on the device heap (8 MB by default) from the
    # augmented forward to the reverse kernels, so more than about one step needs a larger heap (Enzyme.jl#2324)
    haskey(ENV, "PM_CUDA_HEAP") && CUDACore.limit!(CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE, parse(Int, ENV["PM_CUDA_HEAP"]))
end
const BE = ON_GPU ? CUDACore.CUDABackend() : CPU()
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "em2d_cases.jl"))
const PME = ParticleMeshElectromagnetic
const MODE = get(ENV, "PM_MODE", "fast") == "reference" ? Reference() : Fast()
const T = Float64

function problem(::Type{S}, α) where {S}
    prob = plasma_step_2d(S; amplitude = α, nsteps = 2, seed = 7)
    return EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species[1:1], prob.seed, prob.kmode)
end
# The Dual state runs Reference() (no atomics for Duals) and `initialize!` sorts it by cell; the
# Float state is sorted the same way here, so that slot i holds the same particle on both sides.
function initialized(prob)
    st = EMState(prob, EMNumerics(; mode = MODE, current = Esirkepov()); backend = BE)
    initialize!(st, prob)
    if !PME.sorts(st, st.species[1])
        for (s, ws) in zip(PME.deposited_species(st), PME.deposited_workspaces(st))
            sort_by_cell!(s, st.grid, ws)
        end
    end
    return st
end
leaves!(out, path, x::AbstractArray{<:Union{AbstractFloat,ForwardDiff.Dual}}) = (any(y -> y === x, values(out)) || (out[path] = x); out)
leaves!(out, path, x::AbstractArray) = out
leaves!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> leaves!(out, "$path.$k", x[k]), keys(x)); out)
function leaves!(out, path, x)
    isstructtype(typeof(x)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && leaves!(out, "$path.$f", getfield(x, f))
    end
    return out
end
# Each array once, by identity: workspace temporaries alias other arrays of the state, and counting
# an array under two paths would count its share of ⟨∇f, v⟩ twice (paths are visited in field
# order, so the same path wins in a state, its Dual twin and its shadow).
leaves(x) = leaves!(Dict{String,Any}(), "st", x)
dev(a) = (b = KernelAbstractions.allocate(BE, eltype(a), size(a)); copyto!(b, a); b)

const PROB = problem(T, T(1) / 10)
const P = params(PROB)
const DUALPROB = problem(ForwardDiff.Dual{:v,T,1}, ForwardDiff.Dual{:v}(T(1) / 10, zero(T)))
const DUALP = params(DUALPROB)

function push_part!(st, p)
    s = st.species[1]
    PME.launcher(PME.push_kernel!, get_backend(s), nothing)(s, PME.particle_fields(st), st.numerics.pusher, st.numerics.shape,
                                                         Int32(guards(st.grid)), s.q / s.m * st.dt / 2, map(h -> st.dt / h, st.grid.dx),
                                                         Int32.(st.grid.n); ndrange = capacity(s))
    return nothing
end
@kernel function gather_probe_kernel!(out, s, f, shape, g::Int32)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == 0x01
        E, B = PME.fields_at(f, ParticleMeshCore.particle_cell(s, i, Val(2)), ParticleMeshCore.particle_offset(s, i, Val(2)), shape, g)
        out[i] = E[1] * E[1] + E[2] + E[3] + B[1] + B[2] * B[3]
    end
end
function gather_part!(st, p)
    s = st.species[1]
    gather_probe_kernel!(get_backend(s))(st.ws[1].tmp_T, s, PME.particle_fields(st), st.numerics.shape, Int32(guards(st.grid)); ndrange = capacity(s))
    return nothing
end
const PMC = ParticleMeshCore
# the same probe from plain arrays, without the species struct
@kernel function gather_arrays_kernel!(out, c1, c2, o1, o2, status, f, shape, g::Int32)
    i = @index(Global, Linear)
    @inbounds if status[i] == 0x01
        E, B = PME.fields_at(f, (c1[i], c2[i]), (o1[i], o2[i]), shape, g)
        out[i] = E[1] * E[1] + E[2] + E[3] + B[1] + B[2] * B[3]
    end
end
function gather_arrays_part!(st, p)
    s = st.species[1]
    gather_arrays_kernel!(get_backend(s))(st.ws[1].tmp_T, s.cell[1], s.cell[2], s.offset[1], s.offset[2], s.status, PME.particle_fields(st),
                                          st.numerics.shape, Int32(guards(st.grid)); ndrange = capacity(s))
    return nothing
end
# one component, one gather_at
@kernel function gather_ex_kernel!(out, c1, c2, o1, o2, status, Ex, shape, g::Int32)
    i = @index(Global, Linear)
    @inbounds if status[i] == 0x01
        out[i] = PMC.gather_at(Ex, (c1[i], c2[i]), (o1[i], o2[i]), shape, PME.gather_layout(Val(2)).Ex, g)
    end
end
function gather_ex_part!(st, p)
    s = st.species[1]
    gather_ex_kernel!(get_backend(s))(st.ws[1].tmp_T, s.cell[1], s.cell[2], s.offset[1], s.offset[2], s.status, st.fields.Ex,
                                      st.numerics.shape, Int32(guards(st.grid)); ndrange = capacity(s))
    return nothing
end
# one component, the stencil written out with the shape weights
@kernel function gather_ex_manual_kernel!(out, c1, c2, o1, o2, status, Ex, shape, g::Int32)
    i = @index(Global, Linear)
    @inbounds if status[i] == 0x01
        (f1, w1), (f2, w2) = PMC.shape_weights(shape, PME.gather_layout(Val(2)).Ex, (o1[i], o2[i]))
        b1 = c1[i] + f1 + g
        b2 = c2[i] + f2 + g
        acc = zero(eltype(out))
        acc += w1[1] * w2[1] * Ex[b1, b2]; acc += w1[2] * w2[1] * Ex[b1 + 1, b2]; acc += w1[3] * w2[1] * Ex[b1 + 2, b2]
        acc += w1[1] * w2[2] * Ex[b1, b2 + 1]; acc += w1[2] * w2[2] * Ex[b1 + 1, b2 + 1]; acc += w1[3] * w2[2] * Ex[b1 + 2, b2 + 1]
        acc += w1[1] * w2[3] * Ex[b1, b2 + 2]; acc += w1[2] * w2[3] * Ex[b1 + 1, b2 + 2]; acc += w1[3] * w2[3] * Ex[b1 + 2, b2 + 2]
        out[i] = acc
    end
end
function gather_ex_manual_part!(st, p)
    s = st.species[1]
    gather_ex_manual_kernel!(get_backend(s))(st.ws[1].tmp_T, s.cell[1], s.cell[2], s.offset[1], s.offset[2], s.status, st.fields.Ex,
                                             st.numerics.shape, Int32(guards(st.grid)); ndrange = capacity(s))
    return nothing
end
current_part!(st, p) = (PME.current!(st.numerics.current, st); nothing)
solve_part!(st, p) = (PME.solve_fields!(st.numerics.solver, st.fields, st.grid, st.dt, st.modifiers, p, 1); nothing)
kinetic_part!(st, p) = (PME.kinetic!(st); nothing)
push_current_part!(st, p) = (push_part!(st, p); current_part!(st, p); nothing)
step_part!(st, p) = (step!(st, p, 1; observer = ()); nothing)
step2_part!(st, p) = (step!(st, p, 1; observer = ()); step!(st, p, 2; observer = ()); nothing)
const PARTS = Dict("gather" => gather_part!, "gather arrays" => gather_arrays_part!, "gather Ex" => gather_ex_part!,
                   "gather Ex manual" => gather_ex_manual_part!, "push" => push_part!, "current" => current_part!, "solve" => solve_part!, "kinetic" => kinetic_part!,
                   "push+current" => push_current_part!, "step" => step_part!, "two steps" => step2_part!)

# PM_ONE_PARTICLE=1: only the first particle alive, so no two work items touch the same node
const KEEP = parse(Int, get(ENV, "PM_KEEP", get(ENV, "PM_ONE_PARTICLE", "0") == "1" ? "1" : "0"))   # PM_KEEP=n: the first n particles alive
const one_particle = KEEP > 0
const SEED_FIELDS = get(ENV, "PM_SEED_FIELDS", "0") == "1"
function keep_one!(st)
    s = st.species[1]
    st_ = Array(s.status); st_[(KEEP + 1):end] .= 0x00
    copyto!(s.status, st_)
    return st
end
function check(name)
    g! = PARTS[name]
    rng = Xoshiro(11)
    st0 = initialized(PROB)
    one_particle && keep_one!(st0)
    L0 = leaves(st0)
    keys0 = sort!(collect(keys(L0)))
    # forward: a Dual state with random tangents V on every array
    std = initialized(DUALPROB)
    one_particle && keep_one!(std)
    Ld = leaves(std)
    common = [k for k in keys0 if haskey(Ld, k)]
    V = Dict(k => dev(randn(rng, T, size(L0[k]))) for k in common)
    for k in common
        Ld[k] .= ForwardDiff.Dual{:v}.(L0[k], V[k])
    end
    g!(std, DUALP)
    JV = Dict(k => ForwardDiff.partials.(Ld[k], 1) for k in common)
    # reverse: seeds W on every array's final value
    # PM_SEED_FIELDS=1: seeds on the fields and K only. A part that deposits sorts the Dual state
    # (Reference) but not the Float one (Fast), so its particle outputs sit in other slots on the
    # two sides; the fields do not depend on the slot order.
    W = Dict(k => (SEED_FIELDS && !(startswith(k, "st.fields.") || k == "st.K")) ? dev(zeros(T, size(L0[k]))) : dev(randn(rng, T, size(L0[k]))) for k in common)
    st = deepcopy(st0)
    dst = Enzyme.make_zero(st)
    Lr = leaves(dst)
    for k in common
        copyto!(Lr[k], W[k])
    end
    t = @elapsed Enzyme.autodiff(set_runtime_activity(Reverse), g!, Const, Duplicated(st, dst), Const(P))
    KernelAbstractions.synchronize(BE)
    G = leaves(dst)
    lhs = Dict(k => sum(W[k] .* JV[k]) for k in common)
    rhs = Dict(k => sum(G[k] .* V[k]) for k in common)
    a, b = sum(values(lhs)), sum(values(rhs))
    scale = sum(abs, values(lhs)) + sum(abs, values(rhs))
    rel = abs(a - b) / scale
    @printf("[%s] %s: ⟨W, J·V⟩ = %.16g, ⟨Jᵀ·W, V⟩ = %.16g, difference relative to Σ|terms| %.2e (reverse %.1f s)\n",
            name, rel < 1e-12 ? "agree" : "DISAGREE", a, b, rel, t)
    # per array: the forward side groups by output array, the reverse by input array, so compare
    # the reverse gradient of each input with a forward pass seeded on that input alone where they disagree
    if rel >= 1e-12
        for k in common
            all(iszero, G[k]) && all(iszero, JV[k]) && continue
            @printf("    %-36s out ⟨W,JV⟩ %+.6e   in ⟨G,V⟩ %+.6e\n", k, lhs[k], rhs[k])
        end
        # isolate the inputs: forward pass with a tangent on one input array at a time
        for k in common
            all(iszero, G[k]) && continue
            stk = initialized(DUALPROB); one_particle && keep_one!(stk); Lk = leaves(stk)
            for j in common
                Lk[j] .= ForwardDiff.Dual{:v}.(L0[j], j == k ? V[j] : zero(V[j]))
            end
            g!(stk, DUALP)
            fwd = sum(j -> sum(W[j] .* ForwardDiff.partials.(Lk[j], 1)), common)
            r = abs(fwd - rhs[k]) / (abs(fwd) + abs(rhs[k]) + eps(T))
            @printf("    input %-30s forward %+.10e reverse %+.10e %s\n", k, fwd, rhs[k], r < 1e-10 ? "" : "<-- DIFFERS")
        end
    end
    flush(stdout)
end

println("pieces: backend ", BE, ", mode ", MODE, ", one particle ", one_particle, ", patches ", get(ENV, "PM_ENZYME_PATCH", ""))
for name in (isempty(ARGS) ? ["solve", "push", "kinetic", "current", "push+current", "step", "two steps"] : ARGS)
    try
        check(name)
    catch e
        println("[$name] FAILED: ", replace(first(sprint(showerror, e), 12000), "\n" => " | "))
        flush(stdout)
    end
end
isdefined(Main, :ACTIVE_SEEN) && println("adjoints dropped by the patch: ", sort!(collect(ACTIVE_SEEN)))
