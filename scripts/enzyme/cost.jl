# WP-050: what one Enzyme gradient costs on the GPU, against the plain run and dual numbers.
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/spike/active_args.jl \
#       julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/cost.jl [T] [cells] [ppc] [steps]
#
# `T` Float64 or Float32 (default Float32), `cells` and `ppc` as `NxM` (default 10x8 and 3x3, the
# WP-023 case; its electrons only), `steps` a comma-separated list of step counts (default
# 1,2,4,8,16,32). Fast() mode, `observer = ()`. For each step count k, once compiled, the median of
# three timings of: the plain run of k steps, a Dual run with 1 and with 4 partials (the same k
# steps with α as a dual number: one Dual pass gives that many directional derivatives), and
# Enzyme's reverse pass of the same k steps (the gradient with respect to every floating-point
# array of the state at once). The first reverse call, which compiles, is timed on its own. A
# step count whose reverse pass fails (on the device heap, for instance) is recorded with its
# error and ends the list, since a CUDA error leaves the context unusable. Writes `result.toml`
# and `manifest.toml` to `scripts/runs/enzyme-cost-<stamp>/`. A measurement, no threshold (A7).

include(joinpath(@__DIR__, "common.jl"))
using Statistics: median

const T = Dict("Float64" => Float64, "Float32" => Float32)[get(ARGS, 1, "Float32")]
parse_pair(s) = Tuple(parse.(Int, split(s, 'x')))
const CELLS = parse_pair(get(ARGS, 2, "10x8"))
const PPC = parse_pair(get(ARGS, 3, "3x3"))
const STEPS = parse.(Int, split(get(ARGS, 4, "1,2,4,8,16,32"), ','))
const α0 = T(1) / 10

case(S, α) = problem(S, α; nsteps = maximum(STEPS), cells = CELLS, ppc = PPC)
prob = case(T, α0)
p = params(prob)
st0 = initialized(prob, Fast())
nparticles = sum(alive_count, st0.species)
dual(N) = ForwardDiff.Dual{:α}(α0, ntuple(i -> T(i == 1), N)...)
std0 = Dict(N => initialized(case(typeof(dual(N)), dual(N)), Fast()) for N in (1, 4))
pd = Dict(N => params(case(typeof(dual(N)), dual(N))) for N in (1, 4))

run_steps!(st, p, k) = (for i in 1:k; step!(st, p, i; observer = ()); end; sync(); st)
function timed(f, make)
    ts = map(1:3) do _
        x = make()
        sync()
        @elapsed f(x)
    end
    return median(ts)
end

# compile everything once, with one step
run_steps!(deepcopy(st0), p, 1)
foreach(N -> run_steps!(deepcopy(std0[N]), pd[N], 1), (1, 4))
t_compile = @elapsed (reverse_gradient(deepcopy(st0), p, 1); sync())
println("T = $T, cells $CELLS, ppc $PPC, $nparticles particles, backend $(backend()); first reverse call (compiles) $(round(t_compile; digits = 1)) s")

rows = Dict{String,Any}[]
for k in STEPS
    row = Dict{String,Any}("steps" => k)
    row["primal"] = timed(st -> run_steps!(st, p, k), () -> deepcopy(st0))
    for N in (1, 4)
        row["dual$N"] = timed(st -> run_steps!(st, pd[N], k), () -> deepcopy(std0[N]))
    end
    try
        row["reverse"] = timed(st -> (reverse_gradient(st, p, k); sync()), () -> deepcopy(st0))
    catch e
        row["error"] = sprint(showerror, e)[1:min(end, 400)]
    end
    push!(rows, row)
    if haskey(row, "reverse")
        @printf("steps %4d: primal %.4f s, Dual{1} %.4f s, Dual{4} %.4f s, reverse %.4f s (%.1f× primal, %.2f ms per step)\n",
                k, row["primal"], row["dual1"], row["dual4"], row["reverse"], row["reverse"] / row["primal"], 1e3 * row["reverse"] / k)
    else
        println("steps $k: reverse failed: ", row["error"])
        break
    end
end

stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
out = joinpath(REPO, "scripts", "runs", "enzyme-cost-$T-$(join(CELLS, 'x'))-$stamp")
mkpath(out)
open(io -> TOML.print(io, Dict("case" => Dict("eltype" => string(T), "cells" => collect(CELLS), "ppc" => collect(PPC), "particles" => nparticles,
                                               "mode" => "Fast", "species" => 1),
                               "seconds" => Dict("first_reverse_call" => t_compile), "rows" => rows)), joinpath(out, "result.toml"), "w")
commit = try readchomp(`git -C $REPO rev-parse HEAD`) catch; "unknown" end
dirty = try !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`)) catch; missing end
hashfile(f) = bytes2hex(open(sha256, f))
inputs = [@__FILE__, joinpath(@__DIR__, "common.jl"), CASES, Base.active_project(), joinpath(REPO, "Manifest.toml")]
open(io -> TOML.print(io, Dict(
    "commit" => commit, "dirty" => string(dirty), "julia" => string(VERSION), "threads" => Threads.nthreads(),
    "cpu" => Sys.cpu_info()[1].model, "args" => ARGS, "backend" => string(backend()),
    "gpu" => ON_GPU ? CUDACore.name(CUDACore.device()) : "none", "patches" => get(ENV, "PM_ENZYME_PATCH", ""),
    "cuda_heap" => get(ENV, "PM_CUDA_HEAP", "default"),
    "versions" => Dict(string(m) => string(pkgversion(m)) for m in (Enzyme, ForwardDiff, KernelAbstractions)),
    "inputs" => Dict(relpath(f, REPO) => hashfile(f) for f in inputs if startswith(f, REPO)))), joinpath(out, "manifest.toml"), "w")
println("wrote ", relpath(out, REPO))
