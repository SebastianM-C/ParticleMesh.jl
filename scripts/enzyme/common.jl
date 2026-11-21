# WP-050: what the Enzyme scripts share (chain_rule.jl, cost.jl): the GPU switch, the WP-023
# case, the state, the differentiated function and the state's floating-point arrays.
using ParticleMeshCore, ParticleMeshElectromagnetic, KernelAbstractions, ForwardDiff, Enzyme
# PM_LOAD_CUDA=1: the CUDA package as well, before the patches (a stacked environment provides it),
# so that Enzyme's CUDA extension loads with its device-copy rules (#552); CUDACore alone leaves it unloaded
get(ENV, "PM_LOAD_CUDA", "0") == "1" && @eval using CUDA
# PM_ENZYME_PATCH: spike-only patches (`spike/active_args.jl`, `spike/unroll.jl`), a `;`-separated list
foreach(f -> isempty(f) || include(abspath(f)), split(get(ENV, "PM_ENZYME_PATCH", ""), ';'))
# PM_BACKEND=cuda runs the state on the GPU (CUDACore of the test environment)
const ON_GPU = get(ENV, "PM_BACKEND", "cpu") == "cuda"
if ON_GPU
    using CUDACore
    CUDACore.allowscalar(false)
    # PM_CUDA_HEAP=<bytes>: Enzyme's dynamic tapes live on the device heap (8 MB by default) from the
    # augmented forward to the reverse kernels, so more than about one step needs a larger heap (Enzyme.jl#2324)
    haskey(ENV, "PM_CUDA_HEAP") && CUDACore.limit!(CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE, parse(Int, ENV["PM_CUDA_HEAP"]))
end
backend() = ON_GPU ? CUDACore.CUDABackend() : CPU()
sync() = KernelAbstractions.synchronize(backend())
using ParticleMeshCore: Reference, Fast
using Printf, TOML, Dates
using SHA: sha256

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASES = joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "em2d_cases.jl")
include(CASES)    # plasma_step_2d, the WP-023 case

function problem(::Type{S}, α; nsteps = 1, nspecies = 1, seed = 7, cells = (10, 8), ppc = (3, 3)) where {S}
    prob = plasma_step_2d(S; amplitude = α, nsteps = max(nsteps, 1), seed, cells, ppc)
    nspecies == 1 || return prob
    return EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species[1:1], prob.seed, prob.kmode)
end
# A Dual has no atomics, so its state runs Reference() (`effective_mode`) and `initialize!` sorts
# its particles by cell; a Float state in Fast() mode is not sorted. The two sides of the chain
# rule pair arrays slot by slot, so the Float state is sorted the same way here: otherwise slot i
# holds another particle on each side (the "primal gap" below shows it).
function initialized(prob, mode; wgs = nothing)
    st = EMState(prob, EMNumerics(; mode, current = Esirkepov(), workgroupsize = wgs); backend = backend())
    initialize!(st, prob)
    if !ParticleMeshElectromagnetic.sorts(st, st.species[1])
        for (s, ws) in zip(ParticleMeshElectromagnetic.deposited_species(st), ParticleMeshElectromagnetic.deposited_workspaces(st))
            sort_by_cell!(s, st.grid, ws; workgroupsize = wgs)
        end
    end
    return st
end
energy(st) = sum(a -> sum(abs2, a), (st.fields.Ex, st.fields.Ey, st.fields.Ez, st.fields.Bx, st.fields.By, st.fields.Bz))
# The reverse pass differentiates `steps_density!`: the steps, then the energy density per node
# into `E2`, returning nothing. A reduction to a scalar inside it would synchronize the GPU
# stream, which Enzyme cannot differentiate, so the adjoint of E2 is seeded with ones instead and
# the gradient is that of f = sum(E2).
@kernel function energy_density_kernel!(E2, @Const(Ex), @Const(Ey), @Const(Ez), @Const(Bx), @Const(By), @Const(Bz))
    i = @index(Global, Linear)
    @inbounds E2[i] = Ex[i] * Ex[i] + Ey[i] * Ey[i] + Ez[i] * Ez[i] + Bx[i] * Bx[i] + By[i] * By[i] + Bz[i] * Bz[i]
end
function steps_density!(E2, st, p, k)
    for i in 1:k
        step!(st, p, i; observer = ())
    end
    f = st.fields
    energy_density_kernel!(get_backend(E2))(E2, f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz; ndrange = length(E2))
    return nothing
end
function reverse_gradient(st, p, k)
    @assert allequal(map(size, (st.fields.Ex, st.fields.Ey, st.fields.Ez, st.fields.Bx, st.fields.By, st.fields.Bz)))
    E2 = similar(st.fields.Ex); fill!(E2, 0)
    dE2 = similar(E2); fill!(dE2, 1)
    dst = Enzyme.make_zero(st)
    Enzyme.autodiff(set_runtime_activity(Reverse), steps_density!, Const, Duplicated(E2, dE2), Duplicated(st, dst), Const(p), Const(k))
    KernelAbstractions.synchronize(get_backend(E2))
    return dst
end
function run_energy!(st, p, k)
    for i in 1:k
        step!(st, p, i; observer = ())
    end
    return energy(st)
end

# every floating-point array of a state, by its path
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

