# WP-078: the tables and the figure of reports/wp-078-reverse-mode-at-production-size.md from the
# runs of trapped.jl `grad` (the reverse gradient through the λ₀/32 trapped-beam case) and of
# trapped2d.jl's `gathergrad` campaign (J and its Dual gradient with the compensated and the
# Galerkin-compensated gather).
#
#     julia +1.13 --project=scripts scripts/enzyme/wp078_report.jl <out dir> <grad run dir> <gathergrad run dir>
#
# <grad run dir> holds the `enzyme-trapped-grad-*` result directories and `logs/b2/` (the run and
# nvidia-smi logs); <gathergrad run dir> the campaign's `gg-*.json`. Writes `gradient.md`,
# `gradient.json`, `gathergrad.md`, `gathergrad.png` and its sidecar `gathergrad.json` (every number
# of the tables and the figure) into <out dir>, and prints the two tables. The optimization's table is
# `scripts/optim/trapped2d_opt.jl report`'s, and the B1 cost table `scripts/enzyme/table.jl`'s.

using CairoMakie, JSON, TOML, Printf, Statistics

"""
    reverse_seconds(segments) -> (steady, compiling)

The reverse pass's seconds without compilation. The first reverse call of each segment length
compiles (about 170 s and 260 s on the RTX 5090: the last, shorter segment and the first full
one); a segment more than 10 times the median is counted as compiling and replaced by the median.
"""
function reverse_seconds(segments)
    m = median(segments)
    compiling = count(>(10m), segments)
    return (; steady = sum(s for s in segments if s <= 10m) + compiling * m, compiling)
end

"""
    peak_mib(logdir) -> Dict(result dir name => MiB)

The largest memory in use on the card while each run ran: nvidia-smi's `memory.used` of the
whole card, sampled every 0.5 s into `mem-<run>.log`, matched to its result directory by the
`wrote scripts/runs/<dir>` line of `run-<run>.log`. It includes CUDA.jl's memory pool, which keeps
freed blocks, so it bounds the need from above and is not the need.
"""
function peak_mib(logdir)
    out = Dict{String,Int}()
    for f in readdir(logdir)
        m = match(r"^run-(.+)\.log$", f)
        m === nothing && continue
        w = match(r"wrote scripts/runs/(enzyme-trapped-grad-[0-9T]+)", read(joinpath(logdir, f), String))
        mem = joinpath(logdir, "mem-$(m[1]).log")
        (w === nothing || !isfile(mem)) && continue
        mib = [parse(Int, x[1]) for x in eachmatch(r", (\d+) MiB", read(mem, String))]
        isempty(mib) || (out[w[1]] = maximum(mib))
    end
    return out
end

const FREE = ("duration", "waist", "focus", "density")

function gradient_rows(dir)
    peaks = peak_mib(joinpath(dir, "logs", "b2"))
    rows = Dict{String,Any}[]
    for d in sort(filter(startswith("enzyme-trapped-grad-"), readdir(dir)))
        r = TOML.parsefile(joinpath(dir, d, "result.toml"))
        c, s = r["case"], r["seconds"]
        rs = reverse_seconds(s["segments"])
        push!(rows, Dict{String,Any}("run" => d, "C" => c["checkpoint_every"], "checkpoints" => get(c, "checkpoint_memory", "host"),
                                     "J" => r["J"], "J_dual" => r["J_dual"], "gradient_reverse" => r["gradient_reverse"],
                                     "gradient_reverse_start" => r["gradient_reverse_start"],
                                     "gradient_reverse_params" => r["gradient_reverse_params"], "gradient_dual4" => r["gradient_dual4"],
                                     "relative_difference" => r["relative_difference"], "nsteps" => c["nsteps"],
                                     "primal_with_checkpoints" => s["primal_with_checkpoints"], "start_tangent" => s["start_tangent"],
                                     "reverse_steady" => rs.steady, "reverse_compiling_segments" => rs.compiling,
                                     "reverse_with_compilation" => s["reverse"], "dual4" => s["dual4"],
                                     "peak_mib" => get(peaks, d, nothing)))
    end
    sort!(rows; by = r -> (r["checkpoints"], r["C"], r["run"]))
    return rows
end

function gradient_md(rows)
    io = IOBuffer()
    d = rows[1]
    println(io, "| gradient | checkpoints | every C steps | dJ/dτ | dJ/dw | dJ/dfocus | dJ/dn | largest relative difference from Dual{4} | primal with checkpoints (s) | reverse, no compilation (s) | reverse per step (ms) | start tangent (s) | total (s) | card memory in use, peak (GiB) |")
    println(io, "|", repeat("---|", 14))
    g = d["gradient_dual4"]
    @printf(io, "| Dual{4} | | | %.6g | %.6g | %.6g | %.6g | | | | | | %.1f | |\n", g..., d["dual4"])
    for r in rows
        g = r["gradient_reverse"]
        pk = r["peak_mib"] === nothing ? "" : @sprintf("%.1f", r["peak_mib"] / 1024)
        total = r["primal_with_checkpoints"] + r["reverse_steady"] + r["start_tangent"]
        @printf(io, "| reverse | %s | %d | %.6g | %.6g | %.6g | %.6g | %.1e | %.1f | %.1f | %.1f | %.1f | %.1f | %s |\n", r["checkpoints"], r["C"], g...,
                maximum(r["relative_difference"]), r["primal_with_checkpoints"], r["reverse_steady"], 1e3 * r["reverse_steady"] / r["nsteps"],
                r["start_tangent"], total, pk)
    end
    return String(take!(io))
end

const POINT_ORDER = ("E1.9", "E2.175", "E2.4", "n0.06", "n0.085")
const GATHERS = ("compensated", "galerkin-compensated")
"The scales of z = x ./ scale of the five parameters (duration, waist, focus, delay, density), `TCASE.scale`."
const SCALE = (6.0, 10.0, 80.0, 54.0, 0.072)
const FREE_IDX = (1, 2, 3, 5)

function gathergrad_rows(dir)
    jobs = [JSON.parsefile(joinpath(dir, f)) for f in readdir(dir) if startswith(f, "gg-") && endswith(f, ".json")]
    grids = sort(unique((j["ppl"], j["ny"]) for j in jobs))
    finest = grids[end]
    rows = Dict{String,Any}[]
    for p in POINT_ORDER
        pick(g, r) = only(j for j in jobs if j["point"] == p && j["gather"] == g && (j["ppl"], j["ny"]) == r)
        ref = pick("compensated", finest)
        for r in grids, g in GATHERS
            j = pick(g, r)
            J = j["J_first_bucket"]
            lg = [j["gradient"][i] * SCALE[i] / J for i in FREE_IDX]
            push!(rows, Dict{String,Any}("point" => p, "gather" => g, "ppl" => r[1], "ny" => r[2], "J" => J,
                                         "J_relative_to_reference" => J / ref["J_first_bucket"] - 1, "gradient" => j["gradient"],
                                         "dlnJ_dz" => lg, "delay_gradient" => j["gradient"][4], "seconds" => j["seconds"]))
        end
    end
    return rows, grids, finest
end

function gathergrad_md(rows, grids)
    io = IOBuffer()
    println(io, "| point | grid | J, compensated | J, galerkin-compensated | J from the reference, compensated | J from the reference, galerkin-compensated | d ln J/dz (τ, w, focus, n), compensated | d ln J/dz (τ, w, focus, n), galerkin-compensated | seconds, compensated | seconds, galerkin-compensated |")
    println(io, "|", repeat("---|", 10))
    for p in POINT_ORDER, r in grids
        a, b = (only(x for x in rows if x["point"] == p && x["gather"] == g && (x["ppl"], x["ny"]) == r) for g in GATHERS)
        lg(x) = join((@sprintf("%.3g", v) for v in x["dlnJ_dz"]), ", ")
        @printf(io, "| %s | λ₀/%d, Δy = π/%d | %.5g | %.5g | %.2f %% | %.2f %% | %s | %s | %.0f | %.0f |\n", p, r[1], 4r[2], a["J"], b["J"],
                100a["J_relative_to_reference"], 100b["J_relative_to_reference"], lg(a), lg(b), a["seconds"], b["seconds"])
    end
    return String(take!(io))
end

function gathergrad_figure(rows, grids, finest)
    fig = Figure(; size = (1500, 620))
    colors = Makie.wong_colors()
    ppls = [r[1] for r in grids]
    styles = Dict("compensated" => :solid, "galerkin-compensated" => :dash)
    for (k, p) in enumerate(POINT_ORDER)
        a1 = Axis(fig[1, k]; title = p, xlabel = "λ₀/Δx", ylabel = k == 1 ? "J / J(compensated, λ₀/$(finest[1]))" : "", xticks = ppls)
        a2 = Axis(fig[2, k]; xlabel = "λ₀/Δx", ylabel = k == 1 ? "d ln J / dz" : "", xticks = ppls)
        for g in GATHERS
            pr = sort([r for r in rows if r["point"] == p && r["gather"] == g]; by = r -> r["ppl"])
            scatterlines!(a1, ppls, [1 + r["J_relative_to_reference"] for r in pr]; color = :black, linestyle = styles[g],
                          label = g)
            for (i, name) in enumerate(FREE)
                scatterlines!(a2, ppls, [r["dlnJ_dz"][i] for r in pr]; color = colors[i], linestyle = styles[g],
                              label = g == "compensated" ? name : nothing)
            end
        end
        hlines!(a1, [1.0]; color = :gray, linestyle = :dot)
        k == 1 && axislegend(a1; position = :rb, framevisible = false)
        k == 1 && axislegend(a2; position = :rt, framevisible = false, merge = true)
    end
    Label(fig[0, :], "J of the first bucket and d ln J/dz with the compensated (solid) and the Galerkin-compensated (dashed) gather, Dual{6}, Float32, RTX 4080 SUPER")
    return fig
end

function main(out, graddir, ggdir)
    mkpath(out)
    grows = gradient_rows(graddir)
    gmd = gradient_md(grows)
    rows, grids, finest = gathergrad_rows(ggdir)
    ggmd = gathergrad_md(rows, grids)
    fig = gathergrad_figure(rows, grids, finest)
    save(joinpath(out, "gathergrad.png"), fig; px_per_unit = 1)
    open(io -> JSON.print(io, Dict("rows" => rows, "finest" => collect(finest), "run_dir" => basename(normpath(ggdir))), 2),
         joinpath(out, "gathergrad.json"), "w")
    open(io -> JSON.print(io, Dict("rows" => grows, "run_dir" => basename(normpath(graddir))), 2), joinpath(out, "gradient.json"), "w")
    write(joinpath(out, "gradient.md"), gmd)
    write(joinpath(out, "gathergrad.md"), ggmd)
    print(gmd, "\n", ggmd)
end

length(ARGS) == 3 || error("usage: wp078_report.jl <out dir> <grad run dir> <gathergrad run dir>")
main(ARGS...)
