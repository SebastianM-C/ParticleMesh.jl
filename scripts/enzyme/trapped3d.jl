# WP-084: Enzyme's reverse mode through the whole 3D trapped-beam case (`scripts/optim/trapped3d_case.jl`:
# the antenna, the moving window, PMLs on the four open faces and at the window's back, field
# ionization, the compensated gather, Reference mode, Float32), on a GPU, over the full run with
# Checkpointing.jl's Revolve and the engine's `checkpoint_storage` (WP-080), checked against
# forward-mode duals with one partial per parameter, with the time and the device memory of each.
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH="scripts/enzyme/active_reduce.jl;scripts/enzyme/spike/unrolled_species.jl;scripts/enzyme/spike/unrolled_current.jl" \
#       JULIA_LOAD_PATH=@:<repo>:@stdlib \
#       julia +1.13 -t 2 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped3d.jl grad [key=value ...]
#
# The module's test environment carries Enzyme, Checkpointing, CUDA and CUDACore; the repository's
# root project is stacked behind it for `ParticleMesh`. The patches: WP-078's two that WP-083 kept,
# Active kernel arguments with their adjoints (#546) and the one-thread flag sum and the sinpi rule
# (#551, #545), and `current!` with its species loops unrolled, without which Enzyme's type analysis
# of the 3D step fails (#384). A second thread samples the device memory.
#
# Keys: the case `ppl=16 nt=1 rows=128 ppc=2x1x1 ion_ppc=1x1x1 cap=1` (the born electrons' capacity
# factor, `trapped3d_capacity`) `t_end=300 a0E=1.9 theta=duration,waist,focus,density` (default the
# base point); the reverse pass `c=<checkpoints>` (default 12) `host=0|1` (checkpoints on the device
# or in host memory); the comparison `dual=1,2,3,4` (the parameters whose Dual{1} run is made, by
# position; `dual=` for none) `dual4=0|1` (one Dual{4} run as well); `primal=1|0` (a plain run for
# its time and memory); `warm=1|0` (first the whole sequence on a small box of the same types, so
# that the timed calls do not compile; its own gradient check against Dual{4} is reported too;
# `warm_t_end=300`); `run=0` stops after the warm-up; `strict_aliasing=0` turns off Enzyme's strict
# aliasing in its type analysis.
#
# What `grad` does, in this order, each phase timed and its peak of device memory recorded:
#
# 1. the plain run (J);
# 2. the reverse pass: `Enzyme.autodiff` of `run_J!` (the steps under `@ad_checkpoint`, then a
#    kernel that writes J's summand per particle slot of both electron species, whose adjoints are
#    seeded with ones), with `Duplicated` state and a `Ref` of the parameters; it gives the adjoint
#    of the start state and of the parameters;
# 3. the gradient dJ/dθ: the start's adjoint contracted with the tangent of `initialize!` (a Dual{1}
#    start per parameter) and the parameters' with dp/dθ of the problem's parameters with Dual{4}
#    (no simulation);
# 4. the Dual{1} runs of the listed parameters to the end (dJ/dθ_j each), and Dual{4} if asked.
#
# Writes result.toml and manifest.toml into scripts/runs/enzyme-trapped3d-grad-<stamp>/. The numbers
# are measurements of single runs, not independently reviewed.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "scripts", "optim", "trapped3d_case.jl"))
include(joinpath(@__DIR__, "contract.jl"))
using Checkpointing: Checkpointing, Revolve, @ad_checkpoint
using ParticleMeshElectromagnetic: ionization_electrons, ionization_lost, CompensatedGather, Yee

const COMMAND = get(ARGS, 1, "grad")
const OPTS = Dict(String(first(split(a, "="))) => String(last(split(a, "="; limit = 2))) for a in ARGS[2:end] if occursin("=", a))
opt(k, d) = get(OPTS, k, d)
parse_tuple(s) = Tuple(parse.(Int, split(s, 'x')))
parse_list(s) = isempty(s) ? Int[] : parse.(Int, split(s, ','))
const F = Float32
struct Trap3Tag end

"""
The case's sizes from the keys; `warm` for the small box of the warm-up: the same types, 8 cells per
λ₀ and 64 transverse cells of π/4 (the 12-cell PMLs leave the dopant's tube and the waist free),
the whole run to t = 300 (past the window's start at 108 and through the dopant's section, so that
every kernel of the run compiles, and with electrons born and trapped, so that its gradient check
against Dual{4} means something).
"""
function case_opts(; warm = false)
    warm && return (; ppl = 8, nt = 1, rows = 64, ppc = parse_tuple(opt("ppc", "2x1x1")), ion_ppc = parse_tuple(opt("ion_ppc", "1x1x1")),
                    cap = 1.0, t_end = parse(Float64, opt("warm_t_end", "300")), a0E = parse(Float64, opt("a0E", "1.9")))
    return (; ppl = parse(Int, opt("ppl", "16")), nt = parse(Int, opt("nt", "1")), rows = parse(Int, opt("rows", "128")),
            ppc = parse_tuple(opt("ppc", "2x1x1")), ion_ppc = parse_tuple(opt("ion_ppc", "1x1x1")), cap = parse(Float64, opt("cap", "1")),
            t_end = parse(Float64, opt("t_end", "300")), a0E = parse(Float64, opt("a0E", "1.9")))
end
const THETA = haskey(OPTS, "theta") ? parse.(Float64, split(OPTS["theta"], ",")) : [TCASE.x0[i] for i in TCASE.free]
const PNAMES = [TCASE.names[i] for i in TCASE.free]

stamp() = Dates.format(now(UTC), "HH:MM:SS")
reclaim() = (GC.gc(true); ON_GPU && CUDACore.reclaim())
used_bytes() = ON_GPU ? Int(CUDACore.MemoryInfo().pool_used_bytes) : 0
gib(b) = b / 2^30
function mark(what)
    @printf("[%s] %s (pool used %.2f GiB)\n", stamp(), what, gib(used_bytes()))
    flush(stdout)
end

"""
Run `f()` and return its value, the seconds, the peak of device memory in use during it (sampled
every 2 ms from another thread, from a reclaimed pool) and the memory in use before it.
"""
function measured(f)
    reclaim(); sync()
    base = used_bytes()
    peak = Threads.Atomic{Int}(base)
    stop = Threads.Atomic{Bool}(false)
    sampler = Threads.@spawn while !stop[]
        Threads.atomic_max!(peak, used_bytes())
        sleep(0.002)
    end
    t = @elapsed (r = f(); sync())
    stop[] = true
    wait(sampler)
    Threads.atomic_max!(peak, used_bytes())
    return r, t, peak[], base
end

"The case's start at `θ` (Float32 or duals of it) with the sizes `o`, on this script's backend."
function start3(θ, o)
    capacity = trapped3d_capacity(o.ppl; o.nt, o.ion_ppc, factor = o.cap)
    return trapped3d_start(θ; backend = backend(), ppl = o.ppl, nt = o.nt, rows = o.rows, ppc = o.ppc, ion_ppc = o.ion_ppc, capacity,
                           t_end = o.t_end, energy = F(budget3(o.a0E)), float = F)
end
"Duals of the four free parameters with `N` partials, θ_j's tangent the unit vector `dirs[j]` (0: none)."
dual_theta(θ, N, dirs = 1:4) = [ForwardDiff.Dual{Trap3Tag}(F(θ[j]), ntuple(i -> F(i == dirs[j]), N)...) for j in eachindex(θ)]
"The Dual{1} parameters with the tangent of parameter `j` alone."
dual1_theta(θ, j) = dual_theta(θ, 1, [i == j ? 1 : 0 for i in eachindex(θ)])

volume(st) = prod(cellsize(st.grid))
summand_buffers(st) = (similar(st.species[1].w), similar(ionization_electrons(st).w))

"The `n` steps under Revolve's checkpointing, then J's summands of the plasma's electrons into `Je` and of the born electrons into `Jb`."
function run_J!(Je, Jb, st, pr, n, scheme, fade, rise)
    @ad_checkpoint scheme for k in 1:n
        step!(st, pr[], k; observer = ())
    end
    h1 = cellsize(st.grid)[1]
    summands!(Je, st.species[1], h1, fade, rise)
    summands!(Jb, ionization_electrons(st), h1, fade, rise)
    return nothing
end

"""
The reverse pass of the whole run from the start `st` with `c` checkpoints (device or `host`
memory): the start's adjoint `dst`, the parameters' `dp` and J from the summands of the reverse
pass's own forward sweep.
"""
function reverse_run(st, p, n; c, host, fade = TCASE.fade, rise = TCASE.rise)
    Je, Jb = summand_buffers(st)
    fill!(Je, 0); fill!(Jb, 0)
    dJe, dJb = similar(Je), similar(Jb)
    fill!(dJe, 1); fill!(dJb, 1)
    dst = Enzyme.make_zero(st)
    pr, dpr = Ref(p), Ref(Enzyme.make_zero(p))
    scheme = Revolve(c; storage = checkpoint_storage(; host))
    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), run_J!, Const, Duplicated(Je, dJe), Duplicated(Jb, dJb), Duplicated(st, dst),
                    Duplicated(pr, dpr), Const(n), Const(scheme), Const(fade), Const(rise))
    sync()
    J = Float64(volume(st)) * (Float64(foldl(+, Array(Je))) + Float64(foldl(+, Array(Jb))))
    return (; dst, dp = dpr[], J)
end

"J of a run of `n` steps from `st` (any element type)."
run_to(st, p, n) = (run!(st, p, n; observer = ()); sync(); ionization_lost(st) == 0 || error("$(ionization_lost(st)) births found no free slot: raise `cap`"); trapped_energy3(st))

"""
One pass of the whole sequence at the sizes `o`: the plain run, the reverse pass, the gradient
from it, the Dual{1} runs of `duals` and Dual{4} with `dual4`. Returns a Dict of the results.
"""
function sequence(o; c, host, duals, dual4, primal, label)
    res = Dict{String,Any}("case" => Dict{String,Any}("ppl" => o.ppl, "nt" => o.nt, "rows" => o.rows, "ppc" => collect(o.ppc), "ion_ppc" => collect(o.ion_ppc),
                                                      "capacity_factor" => o.cap, "t_end" => o.t_end, "a0E" => o.a0E, "theta" => THETA, "eltype" => string(F),
                                                      "gather" => "CompensatedGather(k = 1)", "solver" => "Yee", "mode" => "Reference"),
                           "checkpoints" => c, "checkpoint_memory" => host ? "host" : "device")
    st0, p, prob = start3(F.(THETA), o)
    n = prob.nsteps
    v = Float64(volume(st0))
    ckpt_bytes = sum(sizeof, evolved(st0))
    # the floating-point arrays (what the shadow mirrors) and every array of the state
    state_bytes = sum(sizeof, values(leaves(st0)); init = 0)
    all_bytes = sum(sizeof, values(all_arrays(st0)); init = 0)
    merge!(res["case"], Dict{String,Any}("nsteps" => n, "cells" => collect(prob.cells), "plasma_slots" => length(st0.species[1].w),
                                         "born_slots" => length(ionization_electrons(st0).w), "dt" => Float64(prob.dt)))
    res["bytes"] = Dict{String,Any}("state_float" => state_bytes, "state_all" => all_bytes, "checkpoint" => ckpt_bytes)
    mark("$label: $(prob.cells) cells, $n steps, $(length(st0.species[1].w)) plasma slots, $(length(ionization_electrons(st0).w)) born slots; " *
         @sprintf("state %.2f GiB in floating-point arrays (%.2f GiB in all), one checkpoint %.2f GiB, %d checkpoints in %s memory", gib(state_bytes), gib(all_bytes), gib(ckpt_bytes), c, host ? "host" : "device"))
    if primal
        J, t, peak, base = measured(() -> run_to(deepcopy(st0), p, n))
        res["J"] = Float64(J)
        res["primal"] = Dict{String,Any}("seconds" => t, "peak_bytes" => peak, "base_bytes" => base)
        mark(@sprintf("%s: plain run %.1f s, J = %.8g, peak %.2f GiB", label, t, J, gib(peak)))
    end
    # the reverse pass on the start itself (a copy would hold one state more; the Dual starts below
    # are built afresh); a failure is recorded and the duals still run
    g = fill(NaN, 4)
    try
        sref = Ref{Any}(st0)
        st0 = nothing
        r, t, peak, base = measured(() -> (s = sref[]; sref[] = nothing; reverse_run(s, p, n; c, host)))
        res["J_reverse_forward"] = r.J
        res["reverse"] = Dict{String,Any}("seconds" => t, "peak_bytes" => peak, "base_bytes" => base)
        mark(@sprintf("%s: reverse pass %.1f s (%.1f ms per step), J from its forward sweep %.8g, peak %.2f GiB", label, t, 1e3 * t / n, r.J, gib(peak)))
        dst, dp = r.dst, r.dp
        r = nothing
        reclaim()
        # the gradient: the start's adjoint with the tangent of initialize! per parameter, and the parameters' with dp/dθ
        pd = params(trapped3d_problem(trapped_point(dual_theta(THETA, 4))...; ppl = o.ppl, nt = o.nt, rows = o.rows, ppc = o.ppc, ion_ppc = o.ion_ppc,
                                      capacity = trapped3d_capacity(o.ppl; o.nt, o.ion_ppc, factor = o.cap), t_end = o.t_end, energy = F(budget3(o.a0E)), float = F))
        g_params = [v * scalar_dot(dp, pd, j) for j in 1:4]
        g_state = zeros(4)
        t_tan = @elapsed for j in 1:4
            sd, _, _ = start3(dual1_theta(THETA, j), o)
            g_state[j] = v * (array_dot(dst, sd, 1) + scalar_dot(dst, sd, 1))
            sd = nothing; reclaim()
        end
        g .= g_state .+ g_params
        res["gradient_reverse"] = g; res["gradient_reverse_start"] = g_state; res["gradient_reverse_params"] = g_params
        res["seconds_start_tangents"] = t_tan
        res["nonfinite_adjoints"] = Dict(k => v for (k, v) in NONFINITE); empty!(NONFINITE)
        for j in 1:4
            @printf("    dJ/d%-8s reverse %+.8e (start %+.3e, parameters %+.3e)\n", PNAMES[j], g[j], g_state[j], g_params[j])
        end
        dst = nothing; reclaim()
    catch e
        res["reverse_error"] = sprint(showerror, e)[1:min(end, 3000)]
        mark("$label: reverse pass FAILED: " * res["reverse_error"][1:min(end, 600)])
        reclaim()
    end
    # the duals
    rows = Dict{String,Any}[]
    for j in duals
        sd, pdj, _ = start3(dual1_theta(THETA, j), o)
        Jd, t, peak, base = try
            measured(() -> run_to(sd, pdj, n))
        catch e
            push!(rows, Dict{String,Any}("parameter" => PNAMES[j], "error" => sprint(showerror, e)[1:min(end, 2000)]))
            mark("$label: Dual{1} in $(PNAMES[j]) FAILED"); sd = nothing; reclaim(); continue
        end
        d = Float64(ForwardDiff.partials(Jd, 1))
        rel = abs(g[j] - d) / abs(d)
        push!(rows, Dict{String,Any}("parameter" => PNAMES[j], "J" => Float64(ForwardDiff.value(Jd)), "derivative" => d, "reverse" => g[j],
                                     "relative_difference" => rel, "seconds" => t, "peak_bytes" => peak, "base_bytes" => base))
        mark(@sprintf("%s: Dual{1} in %s %.1f s, dJ/d%s %+.8e, reverse %+.8e, relative difference %.2e, peak %.2f GiB", label, PNAMES[j], t, PNAMES[j], d, g[j], rel, gib(peak)))
        sd = nothing; reclaim()
    end
    res["dual1"] = rows
    if dual4
        sd, pdj, _ = start3(dual_theta(THETA, 4), o)
        Jd, t, peak, base = measured(() -> run_to(sd, pdj, n))
        d = [Float64(ForwardDiff.partials(Jd, j)) for j in 1:4]
        rel = abs.(g .- d) ./ abs.(d)
        res["dual4"] = Dict{String,Any}("J" => Float64(ForwardDiff.value(Jd)), "gradient" => d, "relative_difference" => rel, "seconds" => t,
                                        "peak_bytes" => peak, "base_bytes" => base)
        mark(@sprintf("%s: Dual{4} %.1f s, peak %.2f GiB; relative differences %s", label, t, gib(peak), string(round.(rel; sigdigits = 3))))
        sd = nothing; reclaim()
    end
    return res
end

function write_run(result)
    out = joinpath(REPO, "scripts", "runs", "enzyme-trapped3d-$COMMAND-$(Dates.format(now(UTC), "yyyymmddTHHMMSS"))")
    mkpath(out)
    open(io -> TOML.print(io, result), joinpath(out, "result.toml"), "w")
    commit = try readchomp(`git -C $REPO rev-parse HEAD`) catch; "unknown" end
    dirty = try !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`)) catch; missing end
    hashfile(f) = bytes2hex(open(sha256, f))
    # the patches the run loaded (PM_ENZYME_PATCH, relative to the repository) are hashed with the rest
    patches = [isabspath(f) ? f : joinpath(REPO, f) for f in split(get(ENV, "PM_ENZYME_PATCH", ""), ';'; keepempty = false)]
    inputs = [@__FILE__, joinpath(@__DIR__, "common.jl"), joinpath(@__DIR__, "contract.jl"), patches...,
              joinpath(REPO, "scripts", "optim", "trapped3d_case.jl"),
              joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"), joinpath(REPO, "scripts", "optim", "born2d_case.jl"),
              Base.active_project(), joinpath(REPO, "Manifest.toml")]
    open(io -> TOML.print(io, Dict(
        "commit" => commit, "dirty" => string(dirty), "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "cpu" => Sys.cpu_info()[1].model, "args" => ARGS, "backend" => string(backend()), "work_package" => "WP-084",
        "gpu" => ON_GPU ? CUDACore.name(CUDACore.device()) : "none", "patches" => get(ENV, "PM_ENZYME_PATCH", ""),
        "cuda_heap" => get(ENV, "PM_CUDA_HEAP", "default"),
        "versions" => Dict(string(m) => string(pkgversion(m)) for m in (Enzyme, ForwardDiff, KernelAbstractions, Checkpointing)),
        "inputs" => Dict(relpath(f, REPO) => hashfile(f) for f in inputs if isfile(f) && startswith(f, REPO)))), joinpath(out, "manifest.toml"), "w")
    println("wrote ", relpath(out, REPO))
    return out
end

function main_grad()
    Threads.nthreads() >= 2 || @warn "run with -t 2: the device memory is sampled from a second thread"
    c = parse(Int, opt("c", "12"))
    host = opt("host", "0") == "1"
    duals = parse_list(opt("dual", "1,2,3,4"))
    dual4 = opt("dual4", "0") == "1"
    primal = opt("primal", "1") == "1"
    mark("start: θ $THETA, keys $OPTS, backend $(backend())")
    result = Dict{String,Any}()
    # strict_aliasing=0: Enzyme's type analysis without strict aliasing (its advice on an IllegalTypeAnalysisException)
    opt("strict_aliasing", "1") == "0" && Enzyme.API.strictAliasing!(false)
    if opt("warm", "1") == "1"
        # the small box compiles every kernel and every reverse pass the run takes (the same types); its gradient is checked against Dual{4}
        w = sequence(case_opts(; warm = true); c = 4, host, duals = Int[], dual4 = true, primal = true, label = "warm-up")
        result["warm"] = w
    end
    opt("run", "1") == "1" && (result["run"] = sequence(case_opts(); c, host, duals, dual4, primal, label = "run"))
    write_run(result)
    # a reverse pass that failed for another reason than the card's memory exits non-zero, so that
    # run_trapped3d.sh retries it at a 1 GiB device heap (running out of memory is not retried: a
    # larger heap only takes more of the card)
    failed = [r["reverse_error"] for r in values(result) if haskey(r, "reverse_error")]
    any(e -> !occursin("Out of GPU memory", e), failed) && exit(2)
end

COMMAND == "grad" ? main_grad() : error("trapped3d.jl grad [key=value ...]")
