# The figure and tables of reports/wp-065-trapped-resolution.md from the job results of
# trapped2d.jl's campaigns: `trapped2d.jl report <out dir> <run dir>...` (included by trapped2d.jl).

"Every job result (`<id>.json`) in the run directories, by id."
function load_jobs(dirs)
    out = Dict{String,Dict{String,Any}}()
    for d in dirs, f in readdir(d)
        endswith(f, ".json") && f ∉ ("jobs.json", "trapped2d.json") || continue
        j = JSON.parsefile(joinpath(d, f))
        haskey(j, "id") && haskey(j, "J") && (out[j["id"]] = j)
    end
    return out
end

res_label(j) = "λ₀/$(j["ppl"]), Δy = π/$(4 * j["ny"])"
res_key(j) = (j["ppl"], j["ny"])
"The gradient in z = x ./ scale (the five parameters, then the budget's a₀ at scale 1), over J: d ln J / dz."
log_gradient(j) = j["gradient"] .* [collect(TCASE.scale); 1.0] ./ j["J"]

"Central differences of the fd jobs against the gradient of the conv job at the same point and resolution."
function fd_rows(jobs)
    rows = Dict{String,Any}[]
    for (id, j) in jobs
        startswith(id, "fd-") && j["sign"] == 1 || continue
        m = jobs[replace(id, r"-p$" => "-m")]
        c = get(jobs, "conv-$(j["point"])-r$(j["ppl"])x$(j["ny"])", nothing)
        c === nothing && continue
        fd = (j["J"] - m["J"]) / (2 * j["h"])
        ad = c["gradient"][j["direction"]]
        push!(rows, Dict{String,Any}("point" => j["point"], "ppl" => j["ppl"], "ny" => j["ny"], "direction" => DIRECTIONS[j["direction"]],
                                     "power" => j["power"], "h" => j["h"], "fd" => fd, "ad" => ad, "relative_difference" => abs(fd - ad) / abs(ad)))
    end
    return sort(rows; by = r -> (r["point"], r["ppl"], r["direction"], r["power"]))
end

"The jobs of a line campaign (`energy` or `density`) grouped by resolution, sorted along the line, with the secant-against-tangent mismatch at the interior points."
function line_rows(jobs, kind)
    sel = [j for (id, j) in jobs if startswith(id, kind * "-")]
    k = kind == "energy" ? 6 : 5
    coord(j) = kind == "energy" ? j["a0E"] : j["x"][5]
    out = Dict{Tuple{Int,Int},Any}()
    for r in unique(res_key.(sel))
        js = sort([j for j in sel if res_key(j) == r]; by = coord)
        x = coord.(js)
        J = [j["J"] for j in js]
        g = [j["gradient"][k] for j in js]
        mismatch = [abs((J[i + 1] - J[i - 1]) / (x[i + 1] - x[i - 1]) - g[i]) / abs(g[i]) for i in 2:(length(js) - 1)]
        out[r] = Dict{String,Any}("x" => x, "J" => J, "dJ" => g, "J_born" => [j["J_born"] for j in js], "J_plasma" => [j["J_plasma"] for j in js],
                                  "mismatch_x" => x[2:(end - 1)], "mismatch" => mismatch, "label" => res_label(js[1]))
    end
    return out
end

function report(outdir, dirs)
    mkpath(outdir)
    jobs = load_jobs(dirs)
    conv = sort([j for (id, j) in jobs if startswith(id, "conv-")]; by = j -> (j["point"], j["ppl"]))
    xonly = sort([j for (id, j) in jobs if startswith(id, "xonly-")]; by = j -> (j["point"], j["ppl"]))
    fd = fd_rows(jobs)
    lines = Dict(k => line_rows(jobs, k) for k in ("energy", "density"))
    side = Dict{String,Any}("conv" => conv, "xonly" => xonly, "fd" => fd,
                            "lines" => Dict(k => Dict(string(r) => v for (r, v) in l) for (k, l) in lines), "dirs" => basename.(dirs))
    fig = Figure(size = (1600, 1350))
    colors = Makie.wong_colors()
    pts = [p.name for p in POINTS]

    ax = Axis(fig[1, 1]; xlabel = "cells per λ₀ along x", ylabel = "J (n_c·mₑc²·(c/ω₀)²)", yscale = log10, xticks = [16, 32, 48, 64],
              title = "J under refinement (solid: x and y; dashed: x only)")
    for (c, p) in enumerate(pts)
        js = filter(j -> j["point"] == p, conv)
        isempty(js) || scatterlines!(ax, [j["ppl"] for j in js], [j["J"] for j in js]; color = colors[c], label = p)
        xs = filter(j -> j["point"] == p, xonly)
        isempty(xs) || scatterlines!(ax, [j["ppl"] for j in xs], [j["J"] for j in xs]; color = colors[c], linestyle = :dash)
    end
    axislegend(ax; position = :rb, labelsize = 11)

    ax = Axis(fig[1, 2]; xlabel = "cells per λ₀ along x", ylabel = "share of J", xticks = [16, 32, 48, 64],
              title = "The plasma's own electrons' share of J (solid) and the first bucket's (dashed)")
    for (c, p) in enumerate(pts)
        js = filter(j -> j["point"] == p, conv)
        isempty(js) && continue
        scatterlines!(ax, [j["ppl"] for j in js], [j["J_plasma"] / j["J"] for j in js]; color = colors[c], label = p)
        scatterlines!(ax, [j["ppl"] for j in js], [j["J_first_bucket"] / j["J"] for j in js]; color = colors[c], linestyle = :dash)
    end
    axislegend(ax; position = :lc, labelsize = 11)

    ax = Axis(fig[1, 3]; xlabel = "cells per λ₀ along x", ylabel = "∂ ln J / ∂z", xticks = [16, 32, 48, 64],
              title = "Gradient in z = x/scale at E2.175 (ForwardDiff)")
    js = filter(j -> j["point"] == "E2.175", conv)
    for (c, name) in enumerate(DIRECTIONS)
        isempty(js) || scatterlines!(ax, [j["ppl"] for j in js], [log_gradient(j)[c] for j in js]; color = colors[c], label = name)
    end
    axislegend(ax; position = :lb, labelsize = 11)

    ax = Axis(fig[2, 1]; xlabel = "relative step h/x", ylabel = "|FD − AD| / |AD|", xscale = log2, yscale = log10,
              title = "Central differences against ForwardDiff (marker: resolution)")
    markers = Dict(16 => :circle, 32 => :rect, 48 => :diamond, 64 => :utriangle)
    for (c, name) in enumerate(DIRECTIONS), r in unique(res_key.(conv))
        q = filter(x -> x["direction"] == name && (x["ppl"], x["ny"]) == r, fd)
        isempty(q) && continue
        scatter!(ax, [2.0^-x["power"] for x in q], [max(x["relative_difference"], 1e-16) for x in q]; color = colors[c], marker = markers[r[1]],
                 label = r == res_key(conv[1]) ? name : nothing)
    end
    axislegend(ax; position = :rb, labelsize = 11, unique = true)

    for (col, kind, xl) in ((2, "energy", "a₀ of the energy budget (at τ = 6, w = 10)"), (3, "density", "plasma density (n_c), budget of a₀ = 2.175"))
        l = lines[kind]
        ax = Axis(fig[2, col]; xlabel = xl, ylabel = "J", yscale = log10, title = "J along the $kind line with ForwardDiff tangents")
        ax2 = Axis(fig[3, col]; xlabel = xl, ylabel = "|secant − tangent| / |tangent|", yscale = log10,
                   title = "Smoothness: neighbours' secant against the tangent")
        for (c, r) in enumerate(sort(collect(keys(l))))
            v = l[r]
            scatterlines!(ax, v["x"], v["J"]; color = colors[c], label = v["label"], markersize = 6)
            d = length(v["x"]) > 1 ? (v["x"][2] - v["x"][1]) / 3 : 0.0
            for (x, J, g) in zip(v["x"], v["J"], v["dJ"])
                yt = J .+ g .* [-d, d]
                all(>(0), yt) && lines!(ax, [x - d, x + d], yt; color = :black, linewidth = 1.5)
            end
            pj = v["J_plasma"]
            any(>(0), pj) && scatter!(ax, v["x"][pj .> 0], pj[pj .> 0]; color = colors[c], marker = :xcross, markersize = 9)
            isempty(v["mismatch"]) || scatterlines!(ax2, v["mismatch_x"], max.(v["mismatch"], 1e-16); color = colors[c], label = v["label"])
        end
        axislegend(ax; position = :rb, labelsize = 10)
    end
    Label(fig[3, 1], "×: the plasma's own electrons' part of J\nblack: ForwardDiff tangents"; tellwidth = false)
    save_plot(outdir, "trapped2d", fig, side)
    open(io -> print(io, tables(conv, xonly, fd, lines)), joinpath(outdir, "tables.md"), "w")
    println(tables(conv, xonly, fd, lines))
    return outdir
end

sig(x) = string(round(x; sigdigits = 4))

function tables(conv, xonly, fd, lines)
    io = IOBuffer()
    println(io, "| point | grid | J | change from the coarser grid | born part | plasma part | first bucket | trapped weight (born, plasma) | a₀ | seconds (threads) |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for p in POINTS
        js = filter(j -> j["point"] == p.name, conv)
        for (i, j) in enumerate(js)
            ch = i == 1 ? "" : sig(100 * (j["J"] / js[i - 1]["J"] - 1)) * " %"
            println(io, "| ", p.name, " | ", res_label(j), " | ", sig(j["J"]), " | ", ch, " | ", sig(j["J_born"]), " | ", sig(j["J_plasma"]), " | ",
                    sig(j["J_first_bucket"]), " | ", sig(j["weight_born"]), ", ", sig(j["weight_plasma"]), " | ", sig(j["a0"]), " | ",
                    round(Int, j["seconds"]), " (", j["threads"], ") |")
        end
    end
    if !isempty(xonly)
        println(io, "\n| point | grid (x only) | J | against x and y at the same Δx |")
        println(io, "|---|---|---|---|")
        for j in xonly
            c = findfirst(q -> q["point"] == j["point"] && q["ppl"] == j["ppl"], conv)
            println(io, "| ", j["point"], " | ", res_label(j), " | ", sig(j["J"]), " | ", c === nothing ? "" : sig(100 * (j["J"] / conv[c]["J"] - 1)) * " %", " |")
        end
    end
    println(io, "\n| point | grid | ", join(["∂ln J/∂z: " * d for d in DIRECTIONS], " | "), " |")
    println(io, "|---|---|", repeat("---|", length(DIRECTIONS)))
    for j in conv
        println(io, "| ", j["point"], " | ", res_label(j), " | ", join(sig.(log_gradient(j)), " | "), " |")
    end
    if !isempty(fd)
        println(io, "\n| point | grid | direction | |FD − AD|/|AD| at h/x = ", join(["2^-$k" for k in FD_POWERS], ", "), " |")
        println(io, "|---|---|---|---|")
        for p in unique(r["point"] for r in fd), r in unique((x["ppl"], x["ny"]) for x in fd), d in DIRECTIONS
            q = sort(filter(x -> x["point"] == p && (x["ppl"], x["ny"]) == r && x["direction"] == d, fd); by = x -> x["power"])
            isempty(q) && continue
            println(io, "| ", p, " | λ₀/", r[1], ", Δy = π/", 4r[2], " | ", d, " | ", join(sig.([x["relative_difference"] for x in q]), ", "), " |")
        end
    end
    for (kind, l) in lines
        isempty(l) && continue
        println(io, "\n| $kind line, grid | points | J from … to | largest |secant − tangent|/|tangent| | where the plasma's part first exceeds 1 % of J |")
        println(io, "|---|---|---|---|---|")
        for r in sort(collect(keys(l)))
            v = l[r]
            k = findfirst(i -> v["J_plasma"][i] > 0.01 * v["J"][i], eachindex(v["J"]))
            println(io, "| ", v["label"], " | ", length(v["x"]), " | ", sig(v["J"][1]), " … ", sig(v["J"][end]), " | ",
                    isempty(v["mismatch"]) ? "" : sig(maximum(v["mismatch"])), " | ", k === nothing ? "nowhere" : sig(v["x"][k]), " |")
        end
    end
    return String(take!(io))
end
