# Figures and tables of the hardware-counter study (WP-091) from the `kernels.json` of each cell
# of one or more collection runs (`study.jl collect`, then `study.jl reduce`).
#
#   julia +1.13 --project=scripts scripts/counters/report.jl <out dir> <label>=<run dir> ...
#
# Each <run dir> holds one directory per cell; the label names the GPU ("5090", "4080"). Writes
# into <out dir>, each PNG next to its JSON sidecar (F3):
#
# - `shares.png`: the time share of each kernel group in each cell (from the `time` collection:
#   one pass, no cache flush), coloured by the group's phase of the step;
# - `sol.png`: speed of light per kernel group, SM against memory throughput as fractions of
#   peak (from the `detail` collection), the area by time share, the colour by limiter label;
# - `stalls.png`: the stall reasons of the groups that hold the most time, in cycles per issued
#   instruction;
# - `occupancy.png`: achieved and theoretical occupancy and the registers per thread of the
#   same groups;
#
# and `tables.md`, one table per cell. The phase of a kernel group is a match on its name
# (`PHASES`); a group no pattern matches is `other`.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))

"The phase of the step a kernel group belongs to, by the first pattern its name matches."
const PHASES = [
    "push" => r"push"i,
    "gather" => r"gather|sharpen|compensat"i,
    "sort" => r"sort|rank|scan|perm|merge|bin|count|key|histo|radix|tile"i,
    "deposit" => r"depos|flux|current|patch|segment|reduce_j|accum"i,
    "field solve" => r"yee|curl|solve|faraday|ampere|update_[eb]|m4|field"i,
    "absorbers" => r"pml|absorb|damp|retire_open"i,
    "window" => r"window|shift|retire|refill|inject"i,
    "ionization" => r"ioniz|adk|birth|born"i,
]
phase_of(group) = something(findfirst(p -> occursin(last(p), group), PHASES) |> i -> i === nothing ? nothing : first(PHASES[i]), "other")
const PHASE_NAMES = [first.(PHASES); "other"]
const PHASE_COLORS = Dict(zip(PHASE_NAMES, Makie.wong_colors()[[1, 2, 3, 4, 5, 6, 7]] |> c -> [c; Makie.RGBf(0.35, 0.35, 0.35); Makie.RGBf(0.7, 0.7, 0.7)]))
const LIMITERS = ["memory_bound", "compute_bound", "latency_bound", "occupancy_limited", "mixed", "unknown"]
const LIMITER_COLORS = Dict(zip(LIMITERS, [Makie.wong_colors()[1], Makie.wong_colors()[6], Makie.wong_colors()[2], Makie.wong_colors()[4],
                                           Makie.wong_colors()[3], Makie.RGBf(0.6, 0.6, 0.6)]))

num(x) = x === nothing ? NaN : Float64(x)
share(r) = num(something(r["time_share"], r["share"], NaN))
lim(r) = something(r["limiter"], "unknown")
"The first raw counter of `names` the row has (a duration-weighted mean), or NaN."
raw(r, names...) = (i = findfirst(n -> haskey(r["raw"], n), names); i === nothing ? NaN : Float64(r["raw"][names[i]]))

"The cells of all runs as (label, cell, rows), in run order and cell order."
function load_cells(runs)
    cells = Tuple{String,String,Vector{Any},Dict{String,Any}}[]
    for (label, dir) in runs, c in sort(readdir(dir))
        f = joinpath(dir, c, "kernels.json")
        isfile(f) || continue
        d = JSON.parsefile(f)
        push!(cells, (label, c, d["rows"], d))
    end
    return cells
end

"The groups holding the most time, until `cover` of the cell's time or `n` groups."
function top_rows(rows; n = 8, cover = 0.9)
    rs = sort(filter(r -> isfinite(share(r)), rows); by = share, rev = true)
    out, acc = Any[], 0.0
    for r in rs
        (length(out) >= n || acc >= cover) && break
        push!(out, r)
        acc += share(r)
    end
    return out
end

function shares_figure(cells)
    fig = Figure(size = (900, 120 + 70 * length(cells)))
    ax = Axis(fig[1, 1]; xlabel = "share of the kernel time in the profiled steps", yticks = (1:length(cells), ["$l $c" for (l, c, _, _) in cells]),
              yreversed = true, xticks = 0:0.2:1, xtickformat = v -> string.(round.(Int, 100 .* v), " %"))
    side = Dict{String,Any}()
    for (i, (label, cell, rows, _)) in enumerate(cells)
        byphase = Dict(p => 0.0 for p in PHASE_NAMES)
        for r in rows
            isfinite(share(r)) && (byphase[phase_of(r["group"])] += share(r))
        end
        left = 0.0
        for p in PHASE_NAMES
            w = byphase[p]
            w > 0 || continue
            barplot!(ax, [i], [w]; offset = left, direction = :x, color = PHASE_COLORS[p], width = 0.7)
            w > 0.06 && text!(ax, left + w / 2, i; text = p, align = (:center, :center), fontsize = 11, color = :white)
            left += w
        end
        side["$label $cell"] = byphase
    end
    Legend(fig[2, 1], [PolyElement(color = PHASE_COLORS[p]) for p in PHASE_NAMES], PHASE_NAMES; orientation = :horizontal, framevisible = false, nbanks = 1)
    return fig, Dict("share_by_phase" => side, "phase_patterns" => Dict(p => r.pattern for (p, r) in PHASES))
end

function sol_figure(cells)
    nc = min(2, length(cells))
    fig = Figure(size = (520 * nc, 440 * cld(length(cells), nc) + 60))
    side = Dict{String,Any}()
    for (i, (label, cell, rows, _)) in enumerate(cells)
        ax = Axis(fig[cld(i, nc), mod1(i, nc)]; title = "$label $cell", xlabel = "memory throughput (fraction of peak)",
                  ylabel = "SM throughput (fraction of peak)", limits = (0, 1, 0, 1), aspect = 1)
        lines!(ax, [0, 1], [0, 1]; color = (:gray, 0.4), linestyle = :dash)
        pts = Any[]
        for r in sort(filter(r -> isfinite(share(r)), rows); by = share)
            x, y = num(get(r, "nvidia_memory_throughput", nothing)), num(get(r, "nvidia_sm_throughput", nothing))
            (isfinite(x) && isfinite(y)) || continue
            scatter!(ax, [x], [y]; markersize = 6 + 60 * sqrt(share(r)), color = (LIMITER_COLORS[lim(r)], 0.7), strokewidth = 0.5)
            share(r) > 0.05 && text!(ax, x, y; text = r["group"], fontsize = 10, offset = (6, 4))
            push!(pts, Dict("group" => r["group"], "memory" => x, "sm" => y, "share" => share(r), "limiter" => lim(r)))
        end
        side["$label $cell"] = pts
    end
    Legend(fig[cld(length(cells), nc) + 1, 1:nc], [MarkerElement(marker = :circle, color = LIMITER_COLORS[l], markersize = 12) for l in LIMITERS],
           LIMITERS; orientation = :horizontal, framevisible = false)
    return fig, side
end

const STALLS = ["long_scoreboard", "lg_throttle", "short_scoreboard", "mio_throttle", "math_pipe_throttle", "wait", "barrier", "membar",
                "branch_resolving", "no_instruction", "dispatch_stall", "tex_throttle", "drain", "sleeping", "misc"]

function stalls_figure(cells)
    nc = min(2, length(cells))
    fig = Figure(size = (560 * nc, 380 * cld(length(cells), nc) + 80))
    colors = Makie.cgrad(:tab20, length(STALLS); categorical = true)
    side = Dict{String,Any}()
    for (i, (label, cell, rows, _)) in enumerate(cells)
        top = top_rows(rows)
        ax = Axis(fig[cld(i, nc), mod1(i, nc)]; title = "$label $cell", xlabel = "cycles per issued instruction",
                  yticks = (1:length(top), [string(r["group"], " (", round(Int, 100 * share(r)), " %)") for r in top]), yreversed = true)
        rowsout = Any[]
        for (j, r) in enumerate(top)
            left = 0.0
            st = Dict{String,Float64}()
            for (k, s) in enumerate(STALLS)
                v = num(get(r, "nvidia_stall_" * s, nothing))
                (isfinite(v) && v > 0) || continue
                barplot!(ax, [j], [v]; offset = left, direction = :x, color = colors[k], width = 0.7)
                left += v
                st[s] = v
            end
            push!(rowsout, Dict("group" => r["group"], "share" => share(r), "stalls" => st))
        end
        side["$label $cell"] = rowsout
    end
    Legend(fig[cld(length(cells), nc) + 1, 1:nc], [PolyElement(color = colors[k]) for k in eachindex(STALLS)], STALLS;
           orientation = :horizontal, nbanks = 3, framevisible = false)
    return fig, side
end

function occupancy_figure(cells)
    nc = min(2, length(cells))
    fig = Figure(size = (560 * nc, 360 * cld(length(cells), nc)))
    side = Dict{String,Any}()
    for (i, (label, cell, rows, _)) in enumerate(cells)
        top = top_rows(rows)
        ax = Axis(fig[cld(i, nc), mod1(i, nc)]; title = "$label $cell", xlabel = "occupancy (fraction of the SM's warp slots)", limits = ((0, 1), nothing),
                  yticks = (1:length(top), [string(r["group"], " · ", r["registers"] === nothing ? "?" : r["registers"], " regs") for r in top]), yreversed = true)
        th = [num(get(r, "nvidia_theoretical_occupancy", nothing)) for r in top]
        ac = [num(get(r, "nvidia_active_occupancy", nothing)) for r in top]
        barplot!(ax, 1:length(top), th; direction = :x, color = (:gray, 0.35), width = 0.75)
        barplot!(ax, 1:length(top), ac; direction = :x, color = Makie.wong_colors()[1], width = 0.45)
        side["$label $cell"] = [Dict("group" => r["group"], "theoretical" => t, "achieved" => a, "registers" => r["registers"]) for (r, t, a) in zip(top, th, ac)]
    end
    return fig, side
end

pct(x) = isfinite(x) ? string(round(100x; digits = 1)) : "–"
fmt(x; d = 2) = isfinite(x) ? string(round(x; digits = d)) : "–"

"One Markdown table per cell: the groups holding 95 % of the time."
function tables(cells)
    io = IOBuffer()
    for (label, cell, rows, d) in cells
        w = get(get(d, "workload", Dict()), "time", Dict())
        println(io, "### $label $cell\n")
        isempty(w) || println(io, "Cells ", join(get(w, "cells", []), " × "), ", ", get(w, "particles_alive", "?"), " particles alive, ",
                              fmt(1e3 * get(w, "seconds_per_step", NaN)), " ms per step unprofiled, window at step ", get(w, "window_first_step", "?"),
                              " of ", get(w, "nsteps", "?"), ".\n")
        println(io, "| kernel group | phase | share % | SM % | memory % | DRAM % | L1 hit % | L2 hit % | occupancy achieved / theoretical % | registers | top stalls (cycles per issue) | limiter |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in top_rows(rows; n = 25, cover = 0.95)
            st = join(["$(s["reason"]) $(fmt(s["cycles_per_issue"]; d = 1))" for s in r["top_stalls"]], ", ")
            println(io, "| ", r["group"], " | ", phase_of(r["group"]), " | ", pct(share(r)), " | ", pct(num(get(r, "nvidia_sm_throughput", nothing))),
                    " | ", pct(num(get(r, "nvidia_memory_throughput", nothing))), " | ", pct(num(get(r, "nvidia_dram_throughput", nothing))),
                    " | ", fmt(raw(r, "l1tex__t_sector_hit_rate.pct"); d = 1), " | ", fmt(raw(r, "lts__t_sector_hit_rate.pct"); d = 1),
                    " | ", pct(num(get(r, "nvidia_active_occupancy", nothing))), " / ", pct(num(get(r, "nvidia_theoretical_occupancy", nothing))),
                    " | ", something(r["registers"], "–"), " | ", st, " | ", lim(r), " |")
        end
        println(io)
    end
    return String(take!(io))
end

function main(args)
    length(args) >= 2 || error("usage: report.jl <out dir> <label>=<run dir> ...")
    out = mkpath(abspath(args[1]))
    runs = [String(first(split(a, "="; limit = 2))) => abspath(String(last(split(a, "="; limit = 2)))) for a in args[2:end]]
    cells = load_cells(runs)
    isempty(cells) && error("report.jl: no kernels.json under $(last.(runs))")
    for (name, f) in (("shares", shares_figure), ("sol", sol_figure), ("stalls", stalls_figure), ("occupancy", occupancy_figure))
        fig, side = f(cells)
        save_plot(out, name, fig, Dict("runs" => Dict(runs), "data" => side))
    end
    write(joinpath(out, "tables.md"), tables(cells))
    println("report.jl: wrote ", out)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
