# Time per mobile particle-step of analyzed Smilei and PIConGPU runs and of ParticleMesh's 2D
# timing, side by side (WP-030, A5): one table (throughput.json) and one bar chart
# (throughput.png with its sidecar).
#
#   julia +1.13 --project=scripts scripts/crosscode/throughput.jl <out dir> <label>=<run dir> ... [pm=<timing-2d.json>]
#
# A run directory is an analyzed run of either code (analyze.jl): a single run (its
# observables.json `performance`) or a seeded one (each seed's wall time from container.txt
# and, for Smilei, its own time-loop time from smilei.log). Per run: the median and range over
# seeds of the wall time per mobile particle-step and, for Smilei, of the time loop's. The
# ParticleMesh rows are the `rows` of scripts/em2d/timing.jl's sidecar. A measurement, no
# threshold (A7); the hardware of every row is recorded next to it.

include(joinpath(@__DIR__, "analyze.jl"))
using Statistics: median

"""
    run_throughput(dir) -> Dict

The time per mobile particle-step of one analyzed run directory: ns per mobile particle-step
of the wall time and (Smilei) of the time loop, the median, minimum and maximum over its seeds
(one value for a single run), with the case, the code, the build and the hardware.
"""
function run_throughput(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    seeded = haskey(case["numerics"], "seeds")
    dirs = seeded ? [joinpath(dir, "seed-$s") for s in run_seeds(dir, case)] : [dir]
    ran = code_of(first(dirs)) == "smilei" ? smilei_steps(first(dirs)) : nothing
    size_ = run_size(case; steps = something(ran, round(Int, case["time"]["t_end"] / case["time"]["dt"])))
    ps = size_["mobile_particle_steps"]
    wall = [wall_seconds(d) for d in dirs]
    any(isnothing, wall) && error("$dir: a run without its wall time")
    ns(v) = 1e9 .* v ./ ps
    summary(v) = Dict{String,Any}("median" => median(v), "min" => minimum(v), "max" => maximum(v), "n" => length(v))
    r = Dict{String,Any}("run" => basename(abspath(dir)), "case" => case["name"], "code" => code_of(first(dirs)),
        "mobile_particle_steps_per_run" => ps, "wall_ns_per_mobile_particle_step" => summary(ns(wall)),
        "hardware" => hardware(first(dirs)))
    if r["code"] == "smilei"
        r["build"] = smilei_build(first(dirs))
        loop = [smilei_time_loop(d) for d in dirs]
        all(!isnothing, loop) && (r["time_loop_ns_per_mobile_particle_step"] = summary(ns(loop)))
    end
    return r
end

function main(out, args)
    mkpath(out)
    rows = Dict{String,Any}[]
    pm = nothing
    for a in args
        label, path = split(a, "="; limit = 2)
        if label == "pm"
            pm = JSON.parsefile(path)
            continue
        end
        push!(rows, merge(run_throughput(path), Dict{String,Any}("label" => label)))
    end
    pmrows = pm === nothing ? Dict{String,Any}[] : [merge(Dict{String,Any}(r), Dict{String,Any}("gpu" => get(pm, "gpu", "unknown")))
                                                      for r in pm["rows"]]
    table = Dict{String,Any}("runs" => rows, "particlemesh" => pmrows,
        "note" => "ns per mobile particle-step; Smilei's time loop excludes its start-up (Python, load, Poisson); the wall time includes it and the output. A measurement, not a threshold (A7)")
    open(io -> JSON.print(io, table, 2), joinpath(out, "throughput.json"), "w")
    # one bar per row: the median, the range as an error bar; a hatched second bar for the time loop
    labels = String[]
    vals, lo, hi, colors = Float64[], Float64[], Float64[], Symbol[]
    for r in rows
        w = r["wall_ns_per_mobile_particle_step"]
        push!(labels, r["label"] * " (wall)"); push!(vals, w["median"]); push!(lo, w["min"]); push!(hi, w["max"])
        push!(colors, r["code"] == "picongpu" ? :darkorange : get(r, "build", "cpu") == "gpu" ? :seagreen : :steelblue)
        if haskey(r, "time_loop_ns_per_mobile_particle_step")
            t = r["time_loop_ns_per_mobile_particle_step"]
            push!(labels, r["label"] * " (loop)"); push!(vals, t["median"]); push!(lo, t["min"]); push!(hi, t["max"])
            push!(colors, get(r, "build", "cpu") == "gpu" ? :seagreen : :steelblue)
        end
    end
    for r in pmrows
        push!(labels, "ParticleMesh $(r["case"]) $(r["type"]) $(r["mode"])"); v = r["ns_per_mobile_particle_step"]
        push!(vals, v); push!(lo, v); push!(hi, v); push!(colors, :purple)
    end
    fig = Figure(size = (900, 120 + 28 * length(labels)))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "ns per mobile particle-step", yticks = (1:length(labels), labels),
              yreversed = true, title = "Time per mobile particle-step", xticks = [1, 2, 3, 5, 10, 20, 30, 50, 100])
    base = 0.5 * minimum(vals)
    barplot!(ax, 1:length(labels), vals; direction = :x, color = colors, fillto = base)
    xlims!(ax, base, 1.5 * maximum(hi))
    rangebars!(ax, 1:length(labels), lo, hi; direction = :x, color = :black, whiskerwidth = 6)
    save_plot(out, "throughput", fig, merge(Dict{String,Any}("plot" => "ns per mobile particle-step per run, median and range over seeds",
        "labels" => labels, "values" => vals, "min" => lo, "max" => hi), table))
    return table
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || (println(stderr, "usage: throughput.jl <out dir> <label>=<run dir> ... [pm=<timing-2d.json>]"); exit(2))
    JSON.print(stdout, main(ARGS[1], ARGS[2:end]), 2)
    println()
end
