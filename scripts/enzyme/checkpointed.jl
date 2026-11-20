# WP-080: Enzyme's reverse pass through the full 2D and 3D step with checkpointing
# (Checkpointing.jl's Revolve over `checkpoint_storage()`), on a GPU: its gradient against the
# plain reverse pass and against forward-mode duals, and its cost against the plain run, the duals
# and the plain reverse pass, in time and device memory. Runs in the module's test environment:
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/spike/active_args.jl \
#       julia +1.13 -t 2 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/checkpointed.jl <command> key=value ...
#
# Keys: dim=2|3, T=Float32|Float64, mode=fast|reference, cells=NxM[xL], ppc=NxM[xL], c (the
# checkpoints), seed. The case is WP-023's `plasma_step_2d` (WP-050's, electrons only) or its 3D
# twin `plasma_step_3d` (electrons only), Esirkepov current, Quadratic shape, Boris, Yee,
# periodic; α is the electrons' density perturbation amplitude (through the weights, A3) and the
# functional f the field energy after the steps, as in `chain_rule.jl`.
#
# `check steps=<n> [plain=1|0|only]`: the checkpointed reverse pass of n steps (with `plain=only`, the
# plain reverse pass in its place, checked against the duals alone); with `plain=1` also the
# plain one (Enzyme's tape of all n steps in device memory), and their difference per array; and
# the chain rule against a Dual run: ⟨∇f, dx/dα⟩ against df/dα and ⟨∇f, v⟩ against the derivative
# along a random direction v on every array (both from the same initialized state), each within
# 64·eps(T)·n·c, c the direction's condition number Σ|∇f·t| / |⟨∇f, t⟩| over every term.
#
# `cost steps=<k1,k2,...> [plain=<kmax>] [dual=<kmax>]`: for each k, the median of three timings
# (each on a fresh copy of the start state made outside the timing)
# of the plain run, Dual{1} and Dual{4} passes (one Dual pass gives that many directional
# derivatives; up to `dual` steps), the plain reverse pass (up to `plain` steps, since its tape
# grows with k) and the checkpointed one, and each one's peak of device memory in use beyond its
# input state (the memory pool's `used` bytes sampled every 2 ms from a second thread, from a
# reclaimed pool; it counts tapes freed but not yet collected, so it is an upper bound; the plain
# run and the Dual passes allocate nothing, and the Float and Dual states' sizes are recorded). The first call of each
# reverse pass, which compiles, is timed on its own.
#
# Writes result.toml and manifest.toml to scripts/runs/enzyme-checkpointed-<command>-<stamp>/.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "window_cases.jl"))
include(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "em3d_cases.jl"))
using Checkpointing: Checkpointing, Revolve, @ad_checkpoint
using Statistics: median
using Random: Xoshiro

const COMMAND = get(ARGS, 1, "check")
const OPTS = Dict(String(first(split(a, "="))) => String(last(split(a, "="; limit = 2))) for a in ARGS[2:end] if occursin("=", a))
opt(k, d) = get(OPTS, k, d)
const DIM = parse(Int, opt("dim", "2"))
const T = Dict("Float64" => Float64, "Float32" => Float32)[opt("T", "Float32")]
const MODE = Dict("fast" => Fast(), "reference" => Reference())[opt("mode", "reference")]
parse_tuple(s) = Tuple(parse.(Int, split(s, 'x')))
const CELLS = parse_tuple(opt("cells", DIM == 2 ? "10x8" : "8x7x6"))
const PPC = parse_tuple(opt("ppc", DIM == 2 ? "3x3" : "2x2x2"))
const NCKPT = parse(Int, opt("c", "4"))
const SEED = parse(Int, opt("seed", "7"))
const α0 = T(1) / 10

"The case with the electrons alone (two species meet #384 under Enzyme), α the electrons' perturbation amplitude."
function case(::Type{S}, α, nsteps) where {S}
    prob = DIM == 2 ? plasma_step_2d(S; amplitude = α, nsteps = max(nsteps, 1), seed = SEED, cells = CELLS, ppc = PPC) :
           plasma_step_3d(S; amplitude = α, nsteps = max(nsteps, 1), seed = SEED, cells = CELLS, ppc = PPC)
    return EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species[1:1], prob.seed, prob.kmode)
end
start(S, α, nsteps) = (prob = case(S, α, nsteps); (initialized(prob, MODE), params(prob)))

function steps_density_ckpt!(E2, st, pr, k, scheme)
    @ad_checkpoint scheme for i in 1:k
        step!(st, pr[], i; observer = ())
    end
    f = st.fields
    energy_density_kernel!(get_backend(E2))(E2, f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz; ndrange = length(E2))
    return nothing
end

"The checkpointed reverse pass of `k` steps with `c` checkpoints: the state's adjoint (the parameters' is dropped by the spike patch)."
function checkpointed_gradient(st, p, k, c; host = false)
    E2 = similar(st.fields.Ex); fill!(E2, 0)
    dE2 = similar(E2); fill!(dE2, 1)
    dst = Enzyme.make_zero(st)
    pr = Ref(p)
    scheme = Revolve(c; storage = checkpoint_storage(; host))
    Enzyme.autodiff(set_runtime_activity(Reverse), steps_density_ckpt!, Const, Duplicated(E2, dE2), Duplicated(st, dst),
                    Duplicated(pr, Ref(Enzyme.make_zero(p))), Const(k), Const(scheme))
    sync()
    return dst
end

reclaim() = (GC.gc(true); ON_GPU && CUDACore.reclaim())
used_bytes() = ON_GPU ? Int(CUDACore.MemoryInfo().pool_used_bytes) : 0
"Run `f()` and return its value, the seconds and the peak of device memory in use above the start (sampled from another thread)."
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
    return r, t, peak[] - base
end
snapshot_bytes(st) = sum(sizeof, evolved(st))
state_bytes(st) = sum(a -> sizeof(a), values(leaves(st)); init = 0)

function write_run(result)
    out = joinpath(REPO, "scripts", "runs", "enzyme-checkpointed-$COMMAND-$(DIM)d-$(Dates.format(now(UTC), "yyyymmddTHHMMSS"))")
    mkpath(out)
    open(io -> TOML.print(io, result), joinpath(out, "result.toml"), "w")
    commit = try readchomp(`git -C $REPO rev-parse HEAD`) catch; "unknown" end
    dirty = try !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`)) catch; missing end
    hashfile(f) = bytes2hex(open(sha256, f))
    inputs = [@__FILE__, joinpath(@__DIR__, "common.jl"), CASES, joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "em3d_cases.jl"),
              Base.active_project(), joinpath(REPO, "Manifest.toml")]
    open(io -> TOML.print(io, Dict(
        "commit" => commit, "dirty" => string(dirty), "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "cpu" => Sys.cpu_info()[1].model, "args" => ARGS, "backend" => string(backend()), "work_package" => "WP-080",
        "gpu" => ON_GPU ? CUDACore.name(CUDACore.device()) : "none", "patches" => get(ENV, "PM_ENZYME_PATCH", ""),
        "versions" => Dict(string(m) => string(pkgversion(m)) for m in (Enzyme, ForwardDiff, KernelAbstractions, Checkpointing)),
        "inputs" => Dict(relpath(f, REPO) => hashfile(f) for f in inputs if startswith(f, REPO)))), joinpath(out, "manifest.toml"), "w")
    println("wrote ", relpath(out, REPO))
end

const CASE_INFO = Dict{String,Any}("dim" => DIM, "eltype" => string(T), "mode" => string(MODE), "cells" => collect(CELLS), "ppc" => collect(PPC),
                                   "checkpoints" => NCKPT, "seed" => SEED, "species" => 1)

function main_check()
    n = parse(Int, opt("steps", "32"))
    only_plain = opt("plain", "1") == "only"
    do_plain = opt("plain", "1") == "1"
    st0, p = start(T, α0, n)
    CASE_INFO["particles"] = sum(alive_count, st0.species)
    println("check: $(DIM)D $T $MODE cells $CELLS ppc $PPC, $(CASE_INFO["particles"]) particles, $n steps, $NCKPT checkpoints, backend $(backend())")
    if only_plain
        # the plain reverse pass alone, checked against the duals below (tells a failure of the step from one of the checkpointed path)
        t_pl = @elapsed (dpl = reverse_gradient(deepcopy(st0), p, n))
        println(@sprintf("  plain reverse only, first call %.1f s", t_pl))
        G = leaves(dpl)
        res = Dict{String,Any}("first_call_plain" => t_pl, "plain_only" => true)
    else
        t_ck = @elapsed (dck = checkpointed_gradient(deepcopy(st0), p, n, NCKPT))
        println(@sprintf("  checkpointed reverse, first call %.1f s", t_ck))
        G = leaves(dck)
        res = Dict{String,Any}("first_call_checkpointed" => t_ck)
    end
    if do_plain
        t_pl = @elapsed (dpl = reverse_gradient(deepcopy(st0), p, n))
        P = leaves(dpl)
        diffs = Dict(k => Float64(maximum(abs.(Array(G[k]) .- Array(P[k])); init = zero(T)) / max(maximum(abs, Array(P[k]); init = zero(T)), floatmin(T)))
                     for k in keys(P) if haskey(G, k))
        worst = sort!(collect(diffs); by = x -> -x[2])
        println(@sprintf("  plain reverse, first call %.1f s; largest relative difference per array (max|Δ|/max|plain|): ", t_pl), worst[1:min(end, 5)])
        res["first_call_plain"] = t_pl
        res["plain_relative_difference"] = Dict(worst)
        # the checkpointed host-storage pass and a second device pass, for the spread between reruns
        dck2 = checkpointed_gradient(deepcopy(st0), p, n, NCKPT; host = true)
        res["host_vs_device"] = maximum(k -> Float64(maximum(abs.(Array(leaves(dck2)[k]) .- Array(G[k])); init = zero(T)) /
                                                     max(maximum(abs, Array(G[k]); init = zero(T)), floatmin(T))), keys(G))
        dpl2 = reverse_gradient(deepcopy(st0), p, n)
        res["plain_rerun"] = maximum(k -> Float64(maximum(abs.(Array(leaves(dpl2)[k]) .- Array(P[k])); init = zero(T)) /
                                                  max(maximum(abs, Array(P[k]); init = zero(T)), floatmin(T))), keys(P))
        # the checkpointed gradient against the plain one is the check of the checkpointing itself: a
        # wrong restore moves it by far more than the rounding spread between two plain passes
        res["plain_tolerance"] = 8 * max(res["plain_rerun"], res["host_vs_device"], Float64(eps(T)))
        println(@sprintf("  host storage against device storage %.2e; plain reverse rerun against itself %.2e; tolerance %.2e",
                         res["host_vs_device"], res["plain_rerun"], res["plain_tolerance"]))
    end
    # the chain rule against a Dual run
    D = ForwardDiff.Dual{:α}(α0, one(T))
    std, pd = start(typeof(D), D, n)
    L, Ld = leaves(st0), leaves(std)
    common = sort!(collect(intersect(keys(L), keys(Ld))))
    tangent = Dict(k => ForwardDiff.partials.(Ld[k], 1) for k in common)
    fd = Float64(ForwardDiff.partials(run_energy!(deepcopy(std), pd, n), 1))
    contrib = Dict(k => Float64(sum(Array(G[k]) .* Array(tangent[k]))) for k in common)
    ad = sum(values(contrib))
    # a random direction on every array, from the Float state's values (the Dual and Float runs part with the steps)
    rng = Xoshiro(SEED)
    stv = deepcopy(std)
    Lv = leaves(stv)
    V = Dict{String,Any}()
    for k in common
        v = similar(L[k]); copyto!(v, randn(rng, T, size(L[k]))); V[k] = v
        Lv[k] .= ForwardDiff.Dual{:α}.(L[k], v)
    end
    fdv = Float64(ForwardDiff.partials(run_energy!(stv, pd, n), 1))
    adv = sum(k -> Float64(sum(Array(G[k]) .* Array(V[k]))), common)
    # each direction's own condition number: Σ|∇f·t| over every term of the dot product against
    # |⟨∇f, t⟩| (a direction whose terms cancel amplifies the gradient's relative error)
    cancel = sum(k -> Float64(sum(abs.(Array(G[k]) .* Array(tangent[k])))), common) / abs(ad)
    cancelv = sum(k -> Float64(sum(abs.(Array(G[k]) .* Array(V[k])))), common) / abs(adv)
    tol, tolv = 64 * eps(T) * n * max(cancel, 1), 64 * eps(T) * n * max(cancelv, 1)
    rel, relv = abs(ad - fd) / abs(fd), abs(adv - fdv) / abs(fdv)
    # the chain-rule tolerance is a loose bound (it lets the gradient's error grow linearly in the steps
    # and assumes no cancellation of rounding errors between terms); the measured difference is the evidence
    plain_ok = !haskey(res, "plain_tolerance") || maximum(values(res["plain_relative_difference"])) <= res["plain_tolerance"]
    verdict = rel <= tol && relv <= tolv && plain_ok ? "pass" : "FAIL"
    println(@sprintf("  chain rule: α direction forward %.10e reverse %.10e relative %.2e tolerance %.2e (cancellation %.1f); random direction forward %.10e reverse %.10e relative %.2e tolerance %.2e (cancellation %.1f): %s",
                     fd, ad, rel, tol, cancel, fdv, adv, relv, tolv, cancelv, verdict))
    merge!(res, Dict("forward" => fd, "reverse" => ad, "relative_difference" => rel, "random_forward" => fdv, "random_reverse" => adv,
                     "random_relative_difference" => relv, "tolerance" => tol, "cancellation" => cancel, "random_tolerance" => tolv,
                     "random_cancellation" => cancelv, "verdict" => verdict))
    println("  kernel arguments whose adjoints the spike patch dropped: ", isdefined(Main, :ACTIVE_SEEN) ? sort!(collect(ACTIVE_SEEN)) : "none (no patch)")
    write_run(Dict("case" => merge(CASE_INFO, Dict{String,Any}("steps" => n)), "result" => res))
end

function main_cost()
    ks = parse.(Int, split(opt("steps", "16,64,256"), ','))
    plain_max = parse(Int, opt("plain", "64"))
    dual_max = parse(Int, opt("dual", string(maximum(ks))))
    st0, p = start(T, α0, maximum(ks))
    CASE_INFO["particles"] = sum(alive_count, st0.species)
    # the Dual states are built when a row times them, one at a time (at 1024², a Dual{4} state is
    # about five Float states, and two of them besides st0 and a copy do not fit a 16 GB card)
    dualα(N) = ForwardDiff.Dual{:α}(α0, ntuple(i -> T(i == 1), N)...)
    dual_case(N) = case(typeof(dualα(N)), dualα(N), maximum(ks))
    dual_params = Dict(N => params(dual_case(N)) for N in (1, 4))
    dual_bytes = Dict(N => (sd = initialized(dual_case(N), MODE); b = state_bytes(sd); sd = nothing; reclaim(); b) for N in (1, 4))
    snap = snapshot_bytes(st0)
    println("cost: $(DIM)D $T $MODE cells $CELLS ppc $PPC, $(CASE_INFO["particles"]) particles, $NCKPT checkpoints, snapshot $(round(snap / 2^20; digits = 1)) MiB, state $(round(state_bytes(st0) / 2^20; digits = 1)) MiB, backend $(backend())")
    run_steps!(st, p, k) = (for i in 1:k; step!(st, p, i; observer = ()); end; nothing)
    run_steps!(deepcopy(st0), p, 1); foreach(N -> run_steps!(initialized(dual_case(N), MODE), dual_params[N], 1), (1, 4)); reclaim(); sync()
    _, t_first_ck, m_first_ck = measured(() -> (checkpointed_gradient(deepcopy(st0), p, 2, min(NCKPT, 2)); nothing))
    _, t_first_pl, m_first_pl = measured(() -> (reverse_gradient(deepcopy(st0), p, 1); nothing))
    println(@sprintf("  first calls (compile): checkpointed %.1f s, plain %.1f s; plain reverse of 1 step %.1f MiB peak", t_first_ck, t_first_pl, m_first_pl / 2^20))
    rows = Dict{String,Any}[]
    # three timings of `f(x)`, each on a fresh input `x = setup()` made outside the timing and the
    # peak (so the peak is the memory `f` needs beyond its input); results are dropped at once
    function timed(setup, f)
        ts, peaks = Float64[], Int[]
        for _ in 1:3
            x = Ref{Any}(setup()); sync()
            _, t, m = measured(() -> (f(x[]); nothing))
            x[] = nothing
            push!(ts, t); push!(peaks, m)
        end
        reclaim()
        return median(ts), maximum(peaks)
    end
    copy0() = deepcopy(st0)
    for k in ks
        row = Dict{String,Any}("steps" => k)
        row["primal"], row["primal_peak"] = timed(copy0, st -> run_steps!(st, p, k))
        if k <= dual_max
            for N in (1, 4)
                row["dual$N"], row["dual$(N)_peak"] = timed(() -> initialized(dual_case(N), MODE), st -> run_steps!(st, dual_params[N], k))
            end
        end
        if k <= plain_max
            try
                row["plain_reverse"], row["plain_reverse_peak"] = timed(copy0, st -> reverse_gradient(st, p, k))
            catch e
                row["plain_error"] = sprint(showerror, e)[1:min(end, 300)]
            end
        end
        try
            row["checkpointed"], row["checkpointed_peak"] = timed(copy0, st -> checkpointed_gradient(st, p, k, NCKPT))
            row["checkpointed_host"], row["checkpointed_host_peak"] = timed(copy0, st -> checkpointed_gradient(st, p, k, NCKPT; host = true))
        catch e
            row["checkpointed_error"] = sprint(showerror, e)[1:min(end, 300)]
        end
        push!(rows, row)
        g(key) = haskey(row, key) ? @sprintf("%.4f s (%.0f MiB)", row[key], row[key * "_peak"] / 2^20) : "-"
        println("  steps $k: primal $(g("primal")), Dual{1} $(g("dual1")), Dual{4} $(g("dual4")), plain reverse $(g("plain_reverse")), ",
                "checkpointed $(g("checkpointed")), host checkpoints $(g("checkpointed_host"))",
                haskey(row, "plain_error") ? "; plain failed: " * row["plain_error"] : "", haskey(row, "checkpointed_error") ? "; checkpointed failed: " * row["checkpointed_error"] : "")
        flush(stdout)
    end
    write_run(Dict("case" => CASE_INFO, "bytes" => Dict("snapshot" => snap, "state" => state_bytes(st0), "dual1_state" => dual_bytes[1], "dual4_state" => dual_bytes[4],
                                    "plain_reverse_one_step_peak" => m_first_pl),
                   "seconds" => Dict("first_call_checkpointed" => t_first_ck, "first_call_plain" => t_first_pl), "rows" => rows))
end

COMMAND == "check" ? main_check() : COMMAND == "cost" ? main_cost() : error("checkpointed.jl check|cost [key=value ...]")
