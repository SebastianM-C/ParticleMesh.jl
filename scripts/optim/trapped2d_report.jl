# The figure and tables of reports/wp-065-trapped-resolution.md from the job results of
# trapped2d.jl's campaigns: `trapped2d.jl report <out dir> <run dir>...` (included by trapped2d.jl).

"Every job result (`<id>.json`) in the run directories, by id; an id in two directories is an error."
function load_jobs(dirs)
    out = Dict{String,Dict{String,Any}}()
    from = Dict{String,String}()
    for d in dirs, f in readdir(d)
        endswith(f, ".json") && f ∉ ("jobs.json", "trapped2d.json") || continue
        j = JSON.parsefile(joinpath(d, f))
        haskey(j, "id") && haskey(j, "J") || continue
        haskey(out, j["id"]) && error("job $(j["id"]) is in both $(from[j["id"]]) and $d")
        out[j["id"]], from[j["id"]] = j, d
    end
    return out
end

# J is the first bucket's (`TCASE.fade`). Runs made while the case's weight reached back to x = 4
# hold that J under "J" and the first bucket's under "J_first_bucket". Where the two agree (the
# plasma traps nothing behind x = 12) the run's gradient is the first bucket's; where they do not
# (the plasma's charge behind the first bucket), the born part's gradient is, to 2.3·10⁻⁴ (no
# born electron is trapped behind x = 12), as long as no plasma electron is trapped in the first
# bucket, which the density lines' runs of that kind show (their J_first_bucket is the born J).
jval(j) = j["J_first_bucket"]
jgrad(j) = haskey(j, "J_from_4") || abs(j["J"] - jval(j)) <= 1e-3 * jval(j) ? j["gradient"] : j["gradient_born"]
res_label(j) = "λ₀/$(j["ppl"]), Δy = π/$(4 * j["ny"])"
"Where a job ran: the RTX 5090 in Float32 or the CPU in Float64 (the runs of the study are one or the other)."
source(j) = get(j, "backend", "cpu") == "cuda" ? "GPU, $(get(j, "eltype", "Float32"))" : "CPU, $(get(j, "eltype", "Float64"))"
on_gpu(j) = get(j, "backend", "cpu") == "cuda"
"The directions an optimizer moves (`TCASE.free`) and the budget's a₀, as indices into `DIRECTIONS`."
const SHOWN = (TCASE.free..., length(DIRECTIONS))
res_key(j) = (j["ppl"], j["ny"])
"The gradient in z = x ./ scale (the five parameters, then the budget's a₀ at scale 1), over J: d ln J / dz."
log_gradient(j) = jgrad(j) .* [collect(TCASE.scale); 1.0] ./ jval(j)

"Central differences of the fd jobs against the gradient of the conv job at the same point and resolution."
function fd_rows(jobs)
    rows = Dict{String,Any}[]
    # pairs and gradients matched by their fields, so that a GPU or Float32 job's id suffix does not matter
    key(j) = (j["point"], j["ppl"], j["ny"], source(j))
    minus = Dict((key(j), j["direction"], j["power"]) => j for (id, j) in jobs if startswith(id, "fd-") && j["sign"] == -1)
    grads = Dict(key(j) => j for (id, j) in jobs if startswith(id, "conv-"))
    for (id, j) in jobs
        startswith(id, "fd-") && j["sign"] == 1 || continue
        m = get(minus, (key(j), j["direction"], j["power"]), nothing)
        c = get(grads, key(j), nothing)
        (m === nothing || c === nothing) && continue
        fd = (jval(j) - jval(m)) / (2 * j["h"])
        ad = jgrad(c)[j["direction"]]
        push!(rows, Dict{String,Any}("point" => j["point"], "ppl" => j["ppl"], "ny" => j["ny"], "source" => source(j), "direction" => DIRECTIONS[j["direction"]],
                                     "power" => j["power"], "h" => j["h"], "fd" => fd, "ad" => ad, "relative_difference" => abs(fd - ad) / abs(ad),
                                     "log_difference" => abs(fd - ad) * [collect(TCASE.scale); 1.0][j["direction"]] / jval(c),
                                     "ad_log" => ad * [collect(TCASE.scale); 1.0][j["direction"]] / jval(c)))
    end
    return sort(rows; by = r -> (r["point"], r["ppl"], r["direction"], r["power"]))
end

"""
The fine scans: for each point, direction and grid, J relative to the scan's middle run against
the step in units of 2⁻¹⁰ of the coordinate, with the ForwardDiff slope of the conv job at the
same point and grid (and source) where there is one, as d(J/J₀)/d(relative step).
"""
function fine_rows(jobs)
    sel = [j for (id, j) in jobs if startswith(id, "fine-")]
    out = Dict{String,Any}()
    for key in unique((j["point"], j["direction"], j["ppl"], j["ny"], source(j)) for j in sel)
        js = sort([j for j in sel if (j["point"], j["direction"], j["ppl"], j["ny"], source(j)) == key]; by = j -> j["step"])
        i0 = findfirst(j -> j["step"] == 0, js)
        i0 === nothing && continue
        J0 = jval(js[i0])
        x0 = key[2] == length(DIRECTIONS) ? js[i0]["a0E"] : js[i0]["x"][key[2]]
        c = [j for (id, j) in jobs if startswith(id, "conv-") && (j["point"], j["ppl"], j["ny"], source(j)) == (key[1], key[3], key[4], key[5])]
        tangent = isempty(c) ? nothing : jgrad(c[1])[key[2]] * x0 / J0
        label = "$(key[1]) $(DIRECTIONS[key[2]]), λ₀/$(key[3]) $(key[5])"
        out[label] = Dict{String,Any}("k" => [j["step"] for j in js], "dJ" => [jval(j) / J0 - 1 for j in js], "tangent" => tangent,
                                      "scatter" => rough(js, J0))
    end
    return out
end

"The RMS scatter of a scan about its least-squares parabola, over J at its middle: the roughness on the scale of 2⁻¹⁰ (a parabola takes out J's curvature, which a line would count)."
function rough(js, J0)
    k = Float64[j["step"] for j in js]
    y = [jval(j) / J0 for j in js]
    A = [ones(length(k)) k k .^ 2]
    r = y .- A * (A \ y)
    return sqrt(sum(abs2, r) / max(1, length(r) - 3))
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
        J = [jval(j) for j in js]
        g = [jgrad(j)[k] for j in js]
        mismatch = [abs((J[i + 1] - J[i - 1]) / (x[i + 1] - x[i - 1]) - g[i]) / abs(g[i]) for i in 2:(length(js) - 1)]
        # the plasma's part of J where the run's J is the first bucket's (older runs' J read from x = 4)
        share = [abs(j["J"] - jval(j)) <= 1e-3 * jval(j) ? j["J_plasma"] / jval(j) : nothing for j in js]
        out[r] = Dict{String,Any}("x" => x, "J" => J, "dJ" => g, "dark" => [j["fast_plasma_window"] for j in js], "born_trapped" => [j["weight_born"] for j in js],
                                  "plasma_share" => share,
                                  "mismatch_x" => x[2:(end - 1)], "mismatch" => mismatch, "label" => res_label(js[1]))
    end
    return out
end

function report(outdir, dirs)
    mkpath(outdir)
    jobs = load_jobs(dirs)
    # the GPU's Float32 runs make the convergence series where they exist; the others check them
    allconv = [j for (id, j) in jobs if startswith(id, "conv-")]
    main(j) = source(j) == "GPU, Float32"
    gpu = Set((j["point"], j["ppl"], j["ny"]) for j in allconv if main(j))
    conv = sort([j for j in allconv if main(j) || (!on_gpu(j) && get(j, "eltype", "Float64") == "Float64" && (j["point"], j["ppl"], j["ny"]) ∉ gpu)];
                by = j -> (j["point"], j["ppl"]))
    check = sort([(g, c) for g in allconv for c in allconv if main(g) && !main(c) && (g["point"], g["ppl"], g["ny"]) == (c["point"], c["ppl"], c["ny"])];
                 by = p -> (p[1]["point"], p[1]["ppl"], source(p[2])))
    fine = fine_rows(jobs)
    xonly = sort([j for (id, j) in jobs if startswith(id, "xonly-")]; by = j -> (j["point"], j["ppl"]))
    fd = fd_rows(jobs)
    lines = Dict(k => line_rows(jobs, k) for k in ("energy", "density"))
    side = Dict{String,Any}("conv" => conv, "xonly" => xonly, "fd" => fd, "fine" => fine,
                            "lines" => Dict(k => Dict(string(r) => v for (r, v) in l) for (k, l) in lines), "dirs" => basename.(dirs))
    fig = Figure(size = (1600, 1350))
    colors = Makie.wong_colors()
    pts = [p.name for p in POINTS]

    ax = Axis(fig[1, 1]; xlabel = "cells per λ₀ along x", ylabel = "J (n_c·mₑc²·(c/ω₀)²)", yscale = log10, xticks = [16, 32, 48, 64],
              title = "J under refinement (solid: x and y; dashed: x only)")
    for (c, p) in enumerate(pts)
        js = filter(j -> j["point"] == p, conv)
        isempty(js) || scatterlines!(ax, [j["ppl"] for j in js], [jval(j) for j in js]; color = colors[c], label = p)
        xs = filter(j -> j["point"] == p, xonly)
        isempty(xs) || scatterlines!(ax, [j["ppl"] for j in xs], [jval(j) for j in xs]; color = colors[c], linestyle = :dash)
    end
    axislegend(ax; position = :rb, labelsize = 11)

    ax = Axis(fig[1, 2]; xlabel = "cells per λ₀ along x", ylabel = "weight (n_c·(c/ω₀)²)", yscale = log10, xticks = [16, 32, 48, 64],
              title = "Trapped weight: first bucket (solid), dark current behind it (dashed)")
    for (c, p) in enumerate(pts)
        js = filter(j -> j["point"] == p, conv)
        isempty(js) && continue
        scatterlines!(ax, [j["ppl"] for j in js], [j["weight_born"] for j in js]; color = colors[c], label = p)
        d = [j["fast_plasma_window"] for j in js]
        any(>(0), d) && scatterlines!(ax, [j["ppl"] for j in js][d .> 0], d[d .> 0]; color = colors[c], linestyle = :dash)
    end
    axislegend(ax; position = :lc, labelsize = 11)

    ax = Axis(fig[1, 3]; xlabel = "cells per λ₀ along x", ylabel = "∂ ln J / ∂z", xticks = [16, 32, 48, 64],
              title = "Gradient in z = x/scale at E2.175 (ForwardDiff)")
    js = filter(j -> j["point"] == "E2.175", conv)
    for (c, k) in enumerate(SHOWN)
        isempty(js) || scatterlines!(ax, [j["ppl"] for j in js], [log_gradient(j)[k] for j in js]; color = colors[c], label = DIRECTIONS[k])
    end
    axislegend(ax; position = :lb, labelsize = 11)

    ax = Axis(fig[2, 1]; xlabel = "relative step h/x", ylabel = "|FD − AD| in ∂ln J/∂z", xscale = log2, yscale = log10,
              title = "Central differences against ForwardDiff (marker: grid)")
    markers = Dict(16 => :circle, 32 => :rect, 48 => :diamond, 64 => :utriangle)
    labelled = Set{String}()
    for (c, name) in enumerate(DIRECTIONS), r in unique((x["ppl"], x["ny"]) for x in fd)
        q = filter(x -> x["direction"] == name && (x["ppl"], x["ny"]) == r, fd)
        isempty(q) && continue
        scatter!(ax, [2.0^-x["power"] * (1 + 0.04 * (c - 3.5)) for x in q], [max(x["log_difference"], 1e-6) for x in q]; color = colors[c],
                 marker = markers[r[1]], markersize = 10, label = name in labelled ? nothing : name)
        push!(labelled, name)
    end
    isempty(fd) || axislegend(ax; position = :cb, orientation = :horizontal, nbanks = 2, labelsize = 11, unique = true)

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
            pj = v["dark"]
            any(>(0), pj) && scatter!(ax, v["x"][pj .> 0], pj[pj .> 0]; color = colors[c], marker = :xcross, markersize = 9)
            isempty(v["mismatch"]) || scatterlines!(ax2, v["mismatch_x"], max.(v["mismatch"], 1e-16); color = colors[c], label = v["label"])
        end
        isempty(l) || axislegend(ax; position = :rb, labelsize = 10)
    end
    ax = Axis(fig[3, 1]; xlabel = "relative change of the delay (units of 2⁻¹⁰)", ylabel = "J / J(E2.175) − 1",
              title = "The delay: plain runs 2⁻¹⁰ apart (dashed: ForwardDiff's tangent)")
    delay = sort([(k, v) for (k, v) in fine if occursin(" delay,", k)]; by = first)
    for (c, (key, v)) in enumerate(delay)
        scatterlines!(ax, v["k"], v["dJ"]; color = colors[c], label = key, markersize = 5)
        v["tangent"] === nothing || lines!(ax, [-8.0, 8.0], v["tangent"] .* [-8.0, 8.0] .* 2.0^-10; color = colors[c], linestyle = :dash)
    end
    ylims!(ax, -0.012, 0.012)
    isempty(delay) || axislegend(ax; position = :lb, labelsize = 10)
    save_plot(outdir, "trapped2d", fig, side)
    t = tables(conv, xonly, fd, lines, check, fine)
    open(io -> print(io, t), joinpath(outdir, "tables.md"), "w")
    println(t)
    return outdir
end

sig(x) = string(round(x; sigdigits = 4))

function tables(conv, xonly, fd, lines, check = [], fine = Dict{String,Any}())
    io = IOBuffer()
    println(io, "| point | grid | J (first bucket) | change from the coarser grid | its trapped weight | dark current: plasma weight above u_x = 5 in the window | a₀ | seconds (threads) |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for p in POINTS
        js = filter(j -> j["point"] == p.name, conv)
        for (i, j) in enumerate(js)
            ch = i == 1 ? "" : sig(100 * (jval(j) / jval(js[i - 1]) - 1)) * " %"
            println(io, "| ", p.name, " | ", res_label(j), " | ", sig(jval(j)), " | ", ch, " | ", sig(j["weight_born"]), " | ", sig(j["fast_plasma_window"]), " | ", sig(j["a0"]), " | ",
                    round(Int, j["seconds"]), " (", j["threads"], ") |")
        end
    end
    if !isempty(xonly)
        println(io, "\n| point | grid (x only) | J | against x and y at the same Δx |")
        println(io, "|---|---|---|---|")
        for j in xonly
            c = findfirst(q -> q["point"] == j["point"] && q["ppl"] == j["ppl"], conv)
            println(io, "| ", j["point"], " | ", res_label(j), " | ", sig(jval(j)), " | ", c === nothing ? "" : sig(100 * (jval(j) / jval(conv[c]) - 1)) * " %", " |")
        end
    end
    if !isempty(check)
        println(io, "\n| point | grid | GPU Float32 against | |ΔJ|/J | largest |Δ ∂ln J/∂z| over the free directions and the budget |")
        println(io, "|---|---|---|---|---|")
        for (g, c) in check
            dg = maximum(abs(log_gradient(g)[k] - log_gradient(c)[k]) for k in SHOWN)
            println(io, "| ", g["point"], " | ", res_label(g), " | ", source(c), " | ", sig(abs(jval(g) / jval(c) - 1)), " | ", sig(dg), " |")
        end
    end
    println(io, "\n| point | grid | ", join(["∂ln J/∂z: " * d * (k in SHOWN ? "" : " (fixed)") for (k, d) in enumerate(DIRECTIONS)], " | "), " |")
    println(io, "|---|---|", repeat("---|", length(DIRECTIONS)))
    for j in conv
        println(io, "| ", j["point"], " | ", res_label(j), " | ", join(sig.(log_gradient(j)), " | "), " |")
    end
    if !isempty(fd)
        powers = sort(unique(r["power"] for r in fd))
        println(io, "\n| point | grid | direction | ForwardDiff's ∂ln J/∂z | |FD − AD| in ∂ln J/∂z at h/x = ", join(["2^-$k" for k in powers], ", "), " |")
        println(io, "|---|---|---|---|---|")
        for p in unique(r["point"] for r in fd), r in unique((x["ppl"], x["ny"]) for x in fd), d in DIRECTIONS
            q = sort(filter(x -> x["point"] == p && (x["ppl"], x["ny"]) == r && x["direction"] == d, fd); by = x -> x["power"])
            isempty(q) && continue
            println(io, "| ", p, " | λ₀/", r[1], ", Δy = π/", 4r[2], " | ", d, " | ", sig(q[1]["ad_log"]), " | ",
                    join([(k = findfirst(x -> x["power"] == pw, q); k === nothing ? "–" : sig(q[k]["log_difference"])) for pw in powers], ", "), " |")
        end
    end
    if !isempty(fine)
        println(io, "\n| fine scan | runs | RMS scatter about a parabola, over J | J's change over the scan (±8·2⁻¹⁰), over J | ForwardDiff's prediction of it |")
        println(io, "|---|---|---|---|---|")
        for (key, v) in sort(collect(fine); by = first)
            println(io, "| ", key, " | ", length(v["k"]), " | ", sig(v["scatter"]), " | ", sig(v["dJ"][end] - v["dJ"][1]), " | ",
                    v["tangent"] === nothing ? "" : sig(v["tangent"] * (v["k"][end] - v["k"][1]) * 2.0^-10), " |")
        end
    end
    for (kind, l) in lines
        isempty(l) && continue
        println(io, "\n| $kind line, grid | points | J from … to | largest |secant − tangent|/|tangent| | where the dark current first exceeds the trapped weight's 1 % | where the plasma's own electrons first make 1 % of J |")
        println(io, "|---|---|---|---|---|---|")
        for r in sort(collect(keys(l)))
            v = l[r]
            k = findfirst(i -> v["dark"][i] > 0.01 * v["born_trapped"][i], eachindex(v["J"]))
            println(io, "| ", v["label"], " | ", length(v["x"]), " | ", sig(v["J"][1]), " … ", sig(v["J"][end]), " | ",
                    isempty(v["mismatch"]) ? "" : sig(maximum(v["mismatch"])), " | ", k === nothing ? "nowhere" : sig(v["x"][k]), " | ",
                    any(isnothing, v["plasma_share"]) ? "not recorded" : (q = findfirst(>(0.01), v["plasma_share"]); q === nothing ? "nowhere" : sig(v["x"][q])), " |")
        end
    end
    return String(take!(io))
end
