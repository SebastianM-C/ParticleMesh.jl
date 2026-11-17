# What the recorder's options save: milliseconds per step of the WP-045 case (wake2d_case.jl) in
# Reference mode, per element type, at its own size (S) and at 448 × 128 cells with 16 particles
# per cell (M), under the default recorder, a recorder of every 16th step, one of the field
# energies only at every 16th step, and one that records nothing (WP-062).
#
#   julia +1.13 -t 4 --project=scripts scripts/optim/recorder_timing.jl [cpu|cuda]
#
# Every case runs three times in one process and the least of the last two is reported, so that
# no compilation is in the numbers. In a tree without the options (copy this file into it) only
# the default recorder runs: that is the comparison with the tree before WP-062.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
using ForwardDiff: Dual

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
elseif BACKEND_NAME == "cpu"
    const BE = CPU()
else
    error("recorder_timing.jl: the argument is cpu or cuda, got $BACKEND_NAME")
end

const HAS_OPTIONS = isdefined(ParticleMeshElectromagnetic, :RECORD_GROUPS)
d2(T) = (Dual{Nothing}(T(7), one(T), zero(T)), Dual{Nothing}(T(100), zero(T), one(T)))
const ELEMENTS = (("Float32", (7.0f0, 100.0f0)), ("Float64", (7.0, 100.0)), ("Dual{Float64,2}", d2(Float64)))
const SIZES = (("S", (;)), ("M", (; ppl = 32, dy_cells = 4, ppc = (4, 4), nsteps = 1024)))
const CONFIGS = HAS_OPTIONS ? (("default", (;)), ("every = 16", (; every = 16)), ("field energy, every = 16", (; every = 16, record = :field_energy)),
                               ("record = ()", (; record = ()))) : (("default", (;)),)

"Seconds of `run!` over the problem's steps with the state's own recorder: the least of the last two of three runs."
function run_seconds(prob, kw)
    best = Inf
    for i in 1:3
        st = EMState(prob, EMNumerics(; mode = Reference()); backend = BE, kw...)
        initialize!(st, prob)
        KernelAbstractions.synchronize(BE)
        t0 = time_ns()
        run!(st, params(prob), prob.nsteps)
        KernelAbstractions.synchronize(BE)
        i > 1 && (best = min(best, (time_ns() - t0) / 1e9))
    end
    return best
end

fmt(x) = string(round(x; sigdigits = 3))

"The Markdown table of the rows: milliseconds per step per size and element type, one column per recorder."
function timing_table(rows)
    configs = unique(r["config"] for r in rows)
    io = IOBuffer()
    println(io, "| size | element type | ", join(("`$c`" for c in configs), " | "), " |")
    println(io, "|---|---|", join(("---" for _ in configs), "|"), "|")
    for key in unique((r["size"], r["T"]) for r in rows)
        cells = map(configs) do c
            r = only(filter(r -> (r["size"], r["T"]) == key && r["config"] == c, rows))
            fmt(r["ms_per_step"])
        end
        println(io, "| ", key[1], " | `", key[2], "` | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

function timing_figure(rows, title)
    configs = unique(r["config"] for r in rows)
    keys_ = unique((r["size"], r["T"]) for r in rows)
    fig = Figure(; size = (900, 420))
    ax = Axis(fig[1, 1]; title, ylabel = "ms per step, relative to the default recorder", xticks = (1:length(keys_), ["$(k[1])\n$(k[2])" for k in keys_]))
    colors = Makie.wong_colors()
    for (i, key) in enumerate(keys_), (j, c) in enumerate(configs)
        base = only(filter(r -> (r["size"], r["T"]) == key && r["config"] == "default", rows))["ms_per_step"]
        v = only(filter(r -> (r["size"], r["T"]) == key && r["config"] == c, rows))["ms_per_step"]
        barplot!(ax, [i], [v / base]; dodge = [j], n_dodge = length(configs), color = colors[j], label = c)
    end
    axislegend(ax; unique = true, position = :rb)
    return fig
end

function main()
    gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
    extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
    dir = run_dir("recorder-timing-$BACKEND_NAME", @__FILE__; prefix = "optim", work_package = "WP-062",
                  backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                  params = Dict{String,Any}("case" => "wake_problem(7, 100)", "sizes" => [s for (s, _) in SIZES], "configs" => [c for (c, _) in CONFIGS],
                                            "options" => HAS_OPTIONS),
                  extra_inputs = [joinpath(@__DIR__, "wake2d_case.jl")], extra)
    rows = Dict{String,Any}[]
    for (size, skw) in SIZES, (label, x) in ELEMENTS, (config, kw) in CONFIGS
        prob = wake_problem(x...; skw...)
        t = run_seconds(prob, kw)
        push!(rows, Dict{String,Any}("size" => size, "T" => label, "config" => config, "steps" => prob.nsteps, "cells" => collect(prob.cells),
                                     "seconds" => t, "ms_per_step" => 1e3 * t / prob.nsteps))
        println(size, " ", label, " ", config, ": ", fmt(rows[end]["ms_per_step"]), " ms per step")
    end
    where_ = gpu === nothing ? "$(Threads.nthreads()) threads, $(cpu_model())" : gpu
    fig = timing_figure(rows, "WP-045 case, Reference mode, $where_")
    save_plot(dir, "recorder_timing", fig, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => cpu_model(), "gpu" => gpu,
                                                "backend" => BACKEND_NAME, "options" => HAS_OPTIONS))
    table = "Milliseconds per step, Reference mode ($where_):\n\n" * timing_table(rows)
    write(joinpath(dir, "tables.md"), table)
    print(table)
    println("recorder_timing: outputs in ", relpath(dir, REPO))
    return dir
end

main()
