# The figure and table of reports/wp-070-galerkin-gather.md from the jobs of trapped2d.jl's `gathers`
# campaign: `trapped2d.jl report-gathers <out dir> <run dir>...` (included by trapped2d.jl, after
# trapped2d_report.jl, whose `load_jobs` it uses).

"The gathers in the order the table and the figure show them."
const GATHER_ORDER = ("plain", "compensated", "galerkin", "galerkin-compensated")

"""
    gathers_table(jobs) -> (rows, md)

J of the first bucket (`jval`) per point, resolution and gather, and its relative difference from
the compensated gather's J on the finest grid of the campaign at that point (the reference), as
rows and as a Markdown table.
"""
function gathers_table(jobs)
    gj = [j for j in values(jobs) if haskey(j, "gather")]
    points = sort(unique(j["point"] for j in gj))
    res = sort(unique((j["ppl"], j["ny"]) for j in gj))
    finest = res[end]
    rows = Dict{String,Any}[]
    io = IOBuffer()
    println(io, "| point | grid | ", join(("J, $g" for g in GATHER_ORDER), " | "), " | ",
            join(("from the reference, $g" for g in GATHER_ORDER), " | "), " |")
    println(io, "|", repeat("---|", 2 + 2 * length(GATHER_ORDER)))
    for p in points
        pick(r, g) = (k = findfirst(j -> j["point"] == p && (j["ppl"], j["ny"]) == r && j["gather"] == g, gj); k === nothing ? nothing : gj[k])
        ref = pick(finest, "compensated")
        Jref = ref === nothing ? NaN : jval(ref)
        for r in res
            js = [pick(r, g) for g in GATHER_ORDER]
            J = [j === nothing ? NaN : jval(j) for j in js]
            rel = (J .- Jref) ./ Jref
            push!(rows, Dict{String,Any}("point" => p, "ppl" => r[1], "ny" => r[2], "gathers" => collect(GATHER_ORDER), "J" => J,
                                         "relative_to_reference" => rel, "reference" => Jref,
                                         "seconds" => [j === nothing ? NaN : j["seconds"] for j in js]))
            fmt(x) = isnan(x) ? "" : string(round(x; sigdigits = 4))
            pct(x) = isnan(x) ? "" : string(round(100x; digits = 1), " %")
            println(io, "| ", p, " | λ₀/", r[1], ", Δy = π/", 4 * r[2], " | ", join(fmt.(J), " | "), " | ", join(pct.(rel), " | "), " |")
        end
    end
    return rows, String(take!(io)), finest
end

"Write `gathers.png`, `gathers.json` and `gathers.md` into `out` from the jobs in `dirs`."
function report_gathers(out, dirs)
    mkpath(out)
    jobs = load_jobs(dirs)
    rows, md, finest = gathers_table(jobs)
    points = sort(unique(r["point"] for r in rows))
    fig = Figure(; size = (1000, 420))
    colors = Makie.wong_colors()
    for (i, p) in enumerate(points)
        ax = Axis(fig[1, i]; title = "$p: J of the first bucket against the grid", xlabel = "cells per laser wavelength (λ₀/Δx)",
                  ylabel = "J relative to the compensated gather at λ₀/$(finest[1])", xticks = sort(unique(r["ppl"] for r in rows)))
        pr = sort([r for r in rows if r["point"] == p]; by = r -> r["ppl"])
        for (k, g) in enumerate(GATHER_ORDER)
            x = [r["ppl"] for r in pr]
            y = [1 + r["relative_to_reference"][k] for r in pr]
            scatterlines!(ax, x, y; label = g, color = colors[k])
        end
        hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
        i == 1 && axislegend(ax; position = :rb)
    end
    save_plot(out, "gathers", fig, Dict{String,Any}("rows" => rows, "finest" => collect(finest), "run_dirs" => basename.(dirs)))
    write(joinpath(out, "gathers.md"), md)
    print(md)
    return out
end
