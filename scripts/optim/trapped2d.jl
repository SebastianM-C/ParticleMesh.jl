# WP-065: the objective over every trapped electron of the first bucket (trapped2d_case.jl) at a fixed laser energy,
# under refinement, and its ForwardDiff gradient in the five parameters (duration, waist, focus,
# delay, density) and in the energy budget against central differences. No optimizer runs here.
#
# A campaign is a list of independent runs ("jobs"); each job is one run of the case at one point
# and one resolution, plain or with duals in all six directions, and writes `<id>.json` into the
# campaign's directory. Jobs run as `PM_JOBS` subprocesses at a time with `PM_JOB_THREADS`
# threads each (default 1 and the threads of this process), and a job whose JSON exists is not
# run again, so a campaign that was stopped continues where it stopped.
#
#   julia +1.13 --project=scripts scripts/optim/trapped2d.jl campaign <kind> <resolutions> [dir] [jobs=N] [threads=M]
#       kind: converge (the five points, with duals), xonly (plain runs with Δy kept at π/4),
#             fine (17 plain runs 2⁻¹⁰ apart in the budget at two points and in the delay at one: roughness),
#             fd (central differences at the points), energy (a line in the energy budget, with
#             duals), density (a line in the density, with duals);
#       resolutions: ppl×ny pairs, e.g. 16x1,32x2 (Δx = λ₀/ppl, Δy = π/(4·ny));
#       dir: the run directory (made if missing; by default scripts/runs/optim-trapped2d-<kind>-<stamp>);
#       jobs=N, threads=M: as PM_JOBS and PM_JOB_THREADS (for a request of scripts/cpu-box.sh, which sets no environment);
#       float=32 runs every job in Float32 (default 64), backend=cuda on the GPU (default cpu)
#   julia +1.13 --project=scripts -t 4 scripts/optim/trapped2d.jl job <dir> <job JSON>
#   julia +1.13 --project=scripts scripts/optim/trapped2d.jl report <out dir> <run dir>...
#
# The report writes trapped2d.png + .json (convergence, gradients, finite differences, lines) and
# tables.md into <out dir>.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "trapped2d_case.jl"))

"""
The study's points: the three energy budgets at the base pulse and density, and two densities at
the middle budget. `x` is (duration, waist, focus, delay, density).
"""
const POINTS = let x0 = collect(TCASE.x0)
    withdensity(n) = [x0[1:4]; n]
    [(name = "E1.9", a0E = 1.9, x = x0), (name = "E2.175", a0E = 2.175, x = x0), (name = "E2.4", a0E = 2.4, x = x0),
     (name = "n0.06", a0E = 2.175, x = withdensity(0.06)), (name = "n0.085", a0E = 2.175, x = withdensity(0.085))]
end

"The six directions of a gradient job: the five parameters, then the budget's a₀ (`a0E`)."
const DIRECTIONS = (TCASE.names..., "a0E")
"Relative central-difference steps 2^-k of the fd campaign."
const FD_POWERS = (5, 9, 13, 17)
"The read-out's sensitivity to the weights' edges: J again from the same state with these."
const RISES = ((2.0, 4.0), (4.0, 6.0))

job_id(parts...) = join(string.(parts), "-")

"The backend of a job: the CPU, or CUDA (loaded on first use; the scripts environment has CUDACore)."
function job_backend(name)
    name == "cpu" && return CPU()
    name == "cuda" || error("a job's backend is cpu or cuda, got $name")
    @eval Main using CUDACore: CUDACore, CUDABackend
    return Base.invokelatest(() -> Main.CUDABackend())
end

"The GPU's model, for a CUDA campaign's manifest."
gpu_name() = (job_backend("cuda"); Base.invokelatest(() -> Main.CUDACore.name(Main.CUDACore.device())))

function campaign_jobs(kind, res)
    jobs = Dict{String,Any}[]
    add!(id, a0E, x, ppl, ny, mode; extra...) =
        push!(jobs, Dict{String,Any}("id" => id, "a0E" => a0E, "x" => collect(Float64, x), "ppl" => ppl, "ny" => ny, "mode" => mode,
                                     (string(k) => v for (k, v) in extra)...))
    for (ppl, ny) in res
        r = "r$(ppl)x$(ny)"
        if kind == "converge"
            for p in POINTS
                add!(job_id("conv", p.name, r), p.a0E, p.x, ppl, ny, "grad"; point = p.name)
            end
        elseif kind == "xonly"
            for p in POINTS
                add!(job_id("xonly", p.name, r), p.a0E, p.x, ppl, ny, "plain"; point = p.name)
            end
        elseif kind == "fd"
            for p in POINTS[1:3], i in eachindex(DIRECTIONS), k in FD_POWERS, sgn in (1, -1)
                x, a0E = copy(p.x), p.a0E
                if i <= 5
                    h = x[i] * 2.0^-k
                    x[i] += sgn * h
                else
                    h = a0E * 2.0^-k
                    a0E += sgn * h
                end
                add!(job_id("fd", p.name, r, DIRECTIONS[i], k, sgn > 0 ? "p" : "m"), a0E, x, ppl, ny, "plain";
                     point = p.name, direction = i, power = k, sign = sgn, h = h)
            end
        elseif kind == "fine"
            # roughness: 17 runs a relative 2⁻¹⁰ apart in the budget's a₀ at E2.175 and E2.4, and in the delay at E2.175
            for (p, i) in ((POINTS[2], 6), (POINTS[3], 6), (POINTS[2], 4)), k in -8:8
                x, a0E = copy(p.x), p.a0E
                i == 6 ? (a0E += k * a0E * 2.0^-10) : (x[i] += k * x[i] * 2.0^-10)
                add!(job_id("fine", p.name, DIRECTIONS[i], k, r), a0E, x, ppl, ny, "plain"; point = p.name, direction = i, step = k)
            end
        elseif kind == "energy"
            for a in range(1.8, 2.6; length = 17)
                add!(job_id("energy", round(a; digits = 3), r), a, collect(TCASE.x0), ppl, ny, "grad")
            end
        elseif kind == "density"
            for n in range(0.05, 0.1; length = 11)
                add!(job_id("density", round(n; digits = 4), r), 2.175, [collect(TCASE.x0)[1:4]; n], ppl, ny, "grad")
            end
        else
            error("unknown campaign kind $kind: converge, xonly, fine, fd, energy or density")
        end
    end
    # the longest first, so that the last jobs to finish are short ones
    return sort(jobs; by = j -> -(j["ppl"]^2 * j["ny"] * (j["mode"] == "grad" ? 3 : 1)))
end

"""
    run_job(job) -> Dict

One run of the case at the job's point and resolution. A plain job gives J and the summary; a
grad job runs with duals in the five parameters and the budget's a₀ and adds the gradient (its J
is the plain run's J bit for bit, T-912).
"""
function run_job(job)
    x, a0E, ppl, ny = Float64.(job["x"]), Float64(job["a0E"]), Int(job["ppl"]), Int(job["ny"])
    F = get(job, "float", 64) == 32 ? Float32 : Float64
    backend = job_backend(get(job, "backend", "cpu"))
    if job["mode"] == "grad"
        D(v, i) = ForwardDiff.Dual{Nothing}(F(v), ntuple(j -> F(i == j), 6))
        args = [D(x[i], i) for i in 1:5]
        energy = budget(D(a0E, 6))
    else
        args, energy = F.(x), F(budget(a0E))
    end
    t = @elapsed st = trapped_state(args...; energy, ppl, ny, float = F, backend)
    J = trapped_energy(st)
    out = Dict{String,Any}(job)
    merge!(out, Dict{String,Any}(string(k) => v for (k, v) in pairs(trapped_summary(st))))
    out["J"] = Float64(value_of(J))
    out["seconds"] = t
    out["threads"] = Threads.nthreads()
    out["eltype"] = string(F)
    out["cells"] = collect(ncells(st.grid))
    out["a0"] = a0_for_energy(x[1], x[2]; energy = budget(a0E))
    for r in RISES
        out["J_rise_$(r[1])_$(r[2])"] = Float64(value_of(trapped_energy(st; rise = r)))
    end
    out["J_first_bucket"] = Float64(value_of(trapped_energy(st; fade = BCASE.fade)))
    out["J_from_4"] = Float64(value_of(trapped_energy(st; fade = (4.0, 10.0))))
    out["J_born_label"] = Float64(value_of(born_energy(st)))
    if job["mode"] == "grad"
        out["gradient"] = collect(ForwardDiff.partials(J))
        p = trapped_parts(st)
        out["gradient_born"] = collect(ForwardDiff.partials(p.born))
        out["gradient_plasma"] = collect(ForwardDiff.partials(p.plasma))
    end
    return out
end

"Run `jobs` into `dir`, `PM_JOBS` subprocesses at a time; a job with a JSON already there is skipped."
function run_campaign(jobs, dir)
    n = parse(Int, get(ENV, "PM_JOBS", "1"))
    th = parse(Int, get(ENV, "PM_JOB_THREADS", string(Threads.nthreads())))
    todo = filter(j -> !isfile(joinpath(dir, j["id"] * ".json")), jobs)
    println("$(length(todo)) of $(length(jobs)) jobs to run, $n at a time with $th threads each, into $dir"); flush(stdout)
    failed = String[]
    asyncmap(todo; ntasks = n) do j
        log = joinpath(dir, j["id"] * ".log")
        cmd = `$(Base.julia_cmd()) --project=$(joinpath(REPO, "scripts")) -t $th $(@__FILE__) job $dir $(JSON.json(j))`
        t = @elapsed ok = open(io -> success(pipeline(cmd; stdout = io, stderr = io)), log, "w")
        ok || push!(failed, j["id"])
        println(ok ? "done " : "FAILED ", j["id"], " in ", round(t; digits = 1), " s"); flush(stdout)
    end
    isempty(failed) || error("$(length(failed)) jobs failed: $(join(failed, ", ")); their logs are in $dir")
    return dir
end

function main_campaign(kind, resarg, dir = nothing)
    res = [Tuple(parse.(Int, split(s, "x"))) for s in split(resarg, ",")]
    jobs = campaign_jobs(kind, res)
    # the element type and the backend, for every job; a Float32 or CUDA job's id says so
    F, be = parse(Int, get(ENV, "PM_FLOAT", "64")), get(ENV, "PM_BACKEND", "cpu")
    for j in jobs
        j["float"], j["backend"] = F, be
        F == 32 && (j["id"] *= "-f32")
        be == "cuda" && (j["id"] *= "-cuda")
    end
    params = Dict{String,Any}("kind" => kind, "resolutions" => [collect(r) for r in res], "jobs" => length(jobs),
                              "case" => Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v isa NamedTuple ? Dict(string(a) => b for (a, b) in pairs(v)) : v) for (k, v) in pairs(BCASE)),
                              "trapped" => Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(TCASE)),
                              "pm_jobs" => get(ENV, "PM_JOBS", "1"), "float" => get(ENV, "PM_FLOAT", "64"), "job_backend" => get(ENV, "PM_BACKEND", "cpu"), "pm_job_threads" => get(ENV, "PM_JOB_THREADS", string(Threads.nthreads())))
    d = run_dir("trapped2d-$kind", @__FILE__; prefix = "optim", work_package = "WP-065", dir,
                backend = be == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                extra = be == "cuda" ? Dict{String,Any}("gpu" => gpu_name()) : Dict{String,Any}(),
                extra_inputs = [joinpath(@__DIR__, "trapped2d_case.jl"), joinpath(@__DIR__, "born2d_case.jl"),
                                joinpath(@__DIR__, "..", "ionization", "atomic.jl")], params)
    open(io -> JSON.print(io, jobs, 2), joinpath(d, "jobs.json"), "w")
    t = @elapsed run_campaign(jobs, d)
    println("campaign $kind done in $(round(t; digits = 1)) s: $d")
end

if length(ARGS) >= 1 && ARGS[1] == "job"
    job = JSON.parse(ARGS[3])
    out = run_job(job)
    open(io -> JSON.print(io, out, 2), joinpath(ARGS[2], job["id"] * ".json"), "w")
    println(job["id"], ": J = ", out["J"], " in ", round(out["seconds"]; digits = 1), " s")
elseif length(ARGS) >= 3 && ARGS[1] == "campaign"
    opts = filter(a -> occursin(r"^(jobs|threads|float|backend)=\w+$", a), ARGS[4:end])
    rest = setdiff(ARGS[4:end], opts)
    for o in opts
        k, v = split(o, "=")
        ENV[Dict("jobs" => "PM_JOBS", "threads" => "PM_JOB_THREADS", "float" => "PM_FLOAT", "backend" => "PM_BACKEND")[k]] = v
    end
    main_campaign(ARGS[2], ARGS[3], isempty(rest) ? nothing : rest[1])
elseif length(ARGS) >= 3 && ARGS[1] == "report"
    include(joinpath(@__DIR__, "trapped2d_report.jl"))
    report(ARGS[2], ARGS[3:end])
elseif !isempty(ARGS)
    error("usage: trapped2d.jl campaign <kind> <resolutions> [dir] | job <dir> <job JSON> | report <out dir> <run dir>...")
end
