# WP-050: Enzyme reverse mode through the 2D electromagnetic step, checked against ForwardDiff
# by the chain rule. Runs only in the side environment of `setup.jl`.
#
#     julia +1.13 --project=<dir>/env scripts/enzyme/chain_rule.jl [mode] [steps] [species] [T] [seed] [workgroupsize]
#
# mode `fast` or `reference`, `steps` steps (default 1), `species` 1 or 2 (2 needs the
# `--foreach` environment), `T` Float64 or Float32, the load's `seed` (default 7), the
# kernels' `workgroupsize` (default: KernelAbstractions', 1024 on the CPU in 0.9, which puts
# each kernel of this case in one workgroup and so runs it serially; give 16 with
# `julia -t 4` to run the kernels on several threads).
#
# The case is WP-023's `plasma_step_2d` (Esirkepov current, Quadratic shape, Boris, Yee,
# 10 × 8 cells, 3 × 3 particles per cell) with the electrons' density perturbation amplitude α
# as the parameter (it enters through the weights, A3). The functional is the field energy
# f = Σ (Ex² + Ey² + Ez² + Bx² + By² + Bz²) over every field node, guards included, after
# `steps` steps of the initialized state.
#
# The check: Enzyme gives ∇f, the gradient of f with respect to every floating-point array of
# the initialized state x (fields, weights, momenta, offsets, workspaces) in one reverse pass;
# ForwardDiff gives the tangent dx/dα of the initialized state (a Dual{T,1} run of
# `initialize!`) and df/dα of the whole run. The chain rule df/dα = ⟨∇f, dx/dα⟩ is an exact
# identity of the two derivatives of the same code, so the two sides agree to roundoff when
# both are right. The α tangent is zero on several arrays (Bx, By, Bz, Ez, the offsets, u_z),
# so a second forward pass seeds a random tangent on every array and checks ⟨∇f, v⟩ too. The
# verdict is "pass" when both agree within 64·eps(T)·steps·(Σ|c_k|/|Σc_k|), the cancellation
# of the per-array contributions c_k. Both checks compare two AD modes of
# the same code: they do not see a physics defect (a sign in the push, a dropped current
# component), which the test suite's T-tests guard. The run writes `result.toml` and `manifest.toml` to
# `scripts/runs/enzyme-<stamp>/`.

using ParticleMeshCore, ParticleMeshElectromagnetic, KernelAbstractions, ForwardDiff, Enzyme
# PM_ENZYME_PATCH: spike-only patches (`spike/active_args.jl`, `spike/unroll.jl`), a `;`-separated list
foreach(f -> isempty(f) || include(abspath(f)), split(get(ENV, "PM_ENZYME_PATCH", ""), ';'))
# PM_BACKEND=cuda runs the state on the GPU (CUDACore of the test environment)
const ON_GPU = get(ENV, "PM_BACKEND", "cpu") == "cuda"
if ON_GPU
    using CUDACore
    CUDACore.allowscalar(false)
end
backend() = ON_GPU ? CUDACore.CUDABackend() : CPU()
using ParticleMeshCore: Reference, Fast
using Printf, TOML, Dates
using SHA: sha256

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASES = joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "em2d_cases.jl")
include(CASES)    # plasma_step_2d, the WP-023 case

mode_name = get(ARGS, 1, "fast")
nsteps = parse(Int, get(ARGS, 2, "1"))
nspecies = parse(Int, get(ARGS, 3, "1"))
T = Dict("Float64" => Float64, "Float32" => Float32)[get(ARGS, 4, "Float64")]
seed = parse(Int, get(ARGS, 5, "7"))
wgs = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : nothing
mode = Dict("fast" => Fast(), "reference" => Reference())[mode_name]
α0 = T(1) / 10

function problem(::Type{S}, α) where {S}
    prob = plasma_step_2d(S; amplitude = α, nsteps = max(nsteps, 1), seed)
    nspecies == 1 || return prob
    return EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species[1:1], prob.seed, prob.kmode)
end
function initialized(prob)
    st = EMState(prob, EMNumerics(; mode, current = Esirkepov(), workgroupsize = wgs); backend = backend())
    initialize!(st, prob)
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
        step!(st, p, i; observer = nothing)
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
        step!(st, p, i; observer = nothing)
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

prob = problem(T, α0)
p = params(prob)
st0 = initialized(prob)

# forward mode
D = ForwardDiff.Dual{:α}(α0, one(T))
probd = problem(typeof(D), D)
std = initialized(probd)
Ld, L = leaves(std), leaves(st0)
common = sort!(collect(intersect(keys(Ld), keys(L))))
only_float = sort!(collect(setdiff(keys(L), keys(Ld))))
primal_gap = maximum(k -> maximum(abs.(ForwardDiff.value.(Ld[k]) .- L[k]); init = zero(T)), common)
tangent = Dict(k => ForwardDiff.partials.(Ld[k], 1) for k in common)
t_fd = @elapsed fd = ForwardDiff.partials(run_energy!(std, params(probd), nsteps), 1)

# reverse mode
st = deepcopy(st0)
t_first = @elapsed dst = reverse_gradient(st, p, nsteps)
G = leaves(dst)
contrib = Dict(k => sum(G[k] .* tangent[k]) for k in common)
ad = sum(values(contrib))
stray = [k for k in only_float if any(!iszero, G[k])]
rel = abs(ad - fd) / abs(fd)
# Which part of ∇f the α-direction tests: arrays with a nonzero tangent and a zero gradient
# (the step overwrites or never reads them), and arrays with a nonzero gradient and a zero
# tangent (untested by the α-direction).
zero_grad = [k for k in common if all(iszero, G[k]) && any(!iszero, tangent[k])]
untested = [k for k in common if any(!iszero, G[k]) && all(iszero, tangent[k])]

# The second check covers every direction: a Dual state with a random tangent v on every
# floating-point array (the initialized values, seeded counter-free from `seed`), whose
# forward derivative must equal ⟨∇f, v⟩. It tests the gradient on the arrays the α tangent
# leaves at zero (Bx, By, Bz, Ez, the offsets, u_z) and on those it never reads.
using Random: Xoshiro
rng = Xoshiro(seed)
stv = initialized(probd)
Lv = leaves(stv)
V = Dict{String,Any}()
for k in common
    v = similar(L[k])
    copyto!(v, randn(rng, T, size(Lv[k])))
    V[k] = v
    Lv[k] .= ForwardDiff.Dual{:α}.(L[k], v)
end
# the scalars of the Dual state (q, m, Δt, Δx, khat) carry a zero tangent: none depends on α
@assert all(k -> all(iszero, Lv[k]), setdiff(keys(Lv), keys(L)))
fdv = ForwardDiff.partials(run_energy!(stv, params(probd), nsteps), 1)
adv = sum(k -> sum(G[k] .* V[k]), common)
relv = abs(adv - fdv) / abs(fdv)
# Roundoff in the sum grows with the steps and with its cancellation (Σ|c_k|/|Σc_k|), so the
# tolerance scales with both.
cancel = sum(abs, values(contrib)) / abs(ad)
tol = 64 * eps(T) * max(nsteps, 1) * max(cancel, 1)
verdict = rel <= tol && relv <= tol ? "pass" : "FAIL"

# timings once compiled: the primal, one reverse pass, one forward pass
sync() = KernelAbstractions.synchronize(backend())
st1 = deepcopy(st0); t_primal = @elapsed (f0 = run_energy!(st1, p, nsteps); sync())
st1 = deepcopy(st0); t_primal = @elapsed (run_energy!(st1, p, nsteps); sync())
st2 = deepcopy(st0)
t_rev = @elapsed dst2 = reverse_gradient(st2, p, nsteps)
std2 = initialized(probd); t_fwd = @elapsed (run_energy!(std2, params(probd), nsteps); sync())
same = all(k -> G[k] == leaves(dst2)[k], keys(G))
ninputs = sum(k -> length(G[k]), keys(G))

@printf("mode=%s steps=%d species=%d T=%s seed=%d backend=%s\n", mode_name, nsteps, nspecies, T, seed, backend())
@printf("  f = %.17g\n  forward df/dα = %.17g\n  reverse ⟨∇f, dx/dα⟩ = %.17g\n  relative difference %.3e (eps(T) = %.3e)\n",
        f0, fd, ad, rel, eps(T))
@printf("  random direction on every array: forward %.17g, reverse %.17g, relative difference %.3e\n", fdv, adv, relv)
@printf("  verdict %s (tolerance 64·eps(T)·steps·cancellation = %.3e, cancellation %.2f)\n", verdict, tol, cancel)
println("  nonzero tangent, zero gradient: ", zero_grad, "\n  nonzero gradient, zero α tangent: ", untested)
@printf("  primal gap between the T and Dual initial states %.3e; Float64-only arrays with a gradient: %s\n", primal_gap, stray)
@printf("  times: Enzyme first call %.1f s, reverse pass %.4f s, primal %.4f s, one forward (Dual) pass %.4f s; repeat bitwise equal: %s; %d gradient entries\n",
        t_first, t_rev, t_primal, t_fwd, same, ninputs)
for k in sort!(collect(keys(contrib)); by = k -> -abs(contrib[k]))
    contrib[k] == 0 && break
    @printf("    %-28s %+.6e\n", k, contrib[k])
end

stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
out = joinpath(REPO, "scripts", "runs", "enzyme-$mode_name-$nsteps-$nspecies-$T-$seed-$stamp")
mkpath(out)
open(io -> TOML.print(io, Dict(
    "case" => Dict("mode" => mode_name, "steps" => nsteps, "species" => nspecies, "eltype" => string(T), "seed" => seed, "alpha" => Float64(α0)),
    "result" => Dict("f" => Float64(f0), "forward" => Float64(fd), "reverse" => Float64(ad), "relative_difference" => Float64(rel),
                     "random_forward" => Float64(fdv), "random_reverse" => Float64(adv), "random_relative_difference" => Float64(relv),
                     "tolerance" => Float64(tol), "cancellation" => Float64(cancel), "workgroupsize" => string(wgs), "verdict" => verdict, "zero_gradient_nonzero_tangent" => zero_grad,
                     "untested_by_alpha" => untested,
                     "eps" => Float64(eps(T)), "primal_gap" => Float64(primal_gap), "stray_gradients" => stray,
                     "repeat_bitwise_equal" => same, "gradient_entries" => ninputs,
                     "contributions" => Dict(k => Float64(v) for (k, v) in contrib if v != 0)),
    "seconds" => Dict("enzyme_first_call" => t_first, "reverse" => t_rev, "primal" => t_primal, "forward" => t_fwd))), joinpath(out, "result.toml"), "w")
commit = try readchomp(`git -C $REPO rev-parse HEAD`) catch; "unknown" end
dirty = try !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`)) catch; missing end
hash(f) = bytes2hex(open(sha256, f))
inputs = [@__FILE__, CASES, joinpath(@__DIR__, "setup.jl"), Base.active_project(), joinpath(REPO, "Manifest.toml")]
step2d = joinpath(pkgdir(ParticleMeshElectromagnetic), "src", "step2d.jl")
open(io -> TOML.print(io, Dict(
    "commit" => commit, "dirty" => string(dirty), "julia" => string(VERSION), "threads" => Threads.nthreads(),
    "cpu" => Sys.cpu_info()[1].model, "args" => ARGS, "backend" => string(backend()),
    "gpu" => ON_GPU ? CUDACore.name(CUDACore.device()) : "none", "patches" => get(ENV, "PM_ENZYME_PATCH", ""),
    "versions" => Dict(string(m) => string(pkgversion(m)) for m in (Enzyme, ForwardDiff, KernelAbstractions)),
    "inputs" => Dict(relpath(f, REPO) => hash(f) for f in inputs if startswith(f, REPO)),
    "environment" => Dict("Project.toml" => hash(inputs[4]), "Manifest.toml" => hash(inputs[5]), "step2d.jl" => hash(step2d)))),
    joinpath(out, "manifest.toml"), "w")
println("  wrote ", out)
println("  kernel arguments whose adjoints the spike patch dropped: ", isdefined(Main, :ACTIVE_SEEN) ? sort!(collect(ACTIVE_SEEN)) : "none (no patch)")
