# The figure and tables of reports/wp-068-first-gpu-optimizations.md from a run directory of
# trapped2d_opt.jl: `trapped2d_opt.jl report <out dir> <run dir>` (included by trapped2d_opt.jl).

"The point records of a run directory, by key."
load_points(dir) = PointStore(joinpath(dir, "points"); run = _ -> error("the report runs nothing")).records

"Every point a start's solver reported, in order, with the point's record."
start_path(res, points) = [merge(points[p["key"]], Dict{String,Any}("seconds_since_start" => p["seconds"])) for p in res["path"]]

"The log-gradient d ln J/dz of a record (z = x ./ scale)."
log_gradient(r) = r["gradient"] .* TRAPPED_BOX.scale ./ r["J"]

"Which of a record's free parameters sit on a bound of the box, as `name=lo` or `name=hi`."
function on_bounds(free)
    out = String[]
    for i in eachindex(free)
        n = TCASE.names[TCASE.free[i]]
        free[i] <= TRAPPED_BOX.lo[i] && push!(out, "$n=lo")
        free[i] >= TRAPPED_BOX.hi[i] && push!(out, "$n=hi")
    end
    return isempty(out) ? "none" : join(out, ", ")
end

fmt(x; d = 4) = x === nothing ? "–" : x isa AbstractString ? x : x isa Integer ? string(x) : string(round(x; sigdigits = d))

"Central differences of the check's fd jobs against its Float64 gradient run, per free direction."
function check_rows(cdir)
    isdir(cdir) || return Dict{String,Any}[]
    jobs = Dict(splitext(f)[1] => JSON.parsefile(joinpath(cdir, f)) for f in readdir(cdir) if endswith(f, ".json"))
    g = get(jobs, "f64-grad", nothing)
    rows = Dict{String,Any}[]
    g === nothing && return rows
    for i in eachindex(TCASE.free)
        n = TCASE.names[TCASE.free[i]]
        p, m = get(jobs, "fd-$n-p", nothing), get(jobs, "fd-$n-m", nothing)
        (p === nothing || m === nothing) && continue
        fd = (p["J"] - m["J"]) / (2 * p["h"])
        push!(rows, Dict{String,Any}("direction" => n, "fd" => fd, "ad" => g["gradient"][i], "fd_log" => fd * TRAPPED_BOX.scale[i] / g["J"],
                                     "ad_log" => g["gradient"][i] * TRAPPED_BOX.scale[i] / g["J"], "h" => p["h"]))
    end
    return rows
end

function opt_report(out, dir)
    mkpath(out)
    points = load_points(dir)
    results = load_results(dir)
    names = sort(collect(keys(results)); by = n -> findfirst(==(Symbol(n)), keys(STARTS)))
    paths = Dict(n => start_path(results[n], points) for n in names)
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))
    params = m["parameters"]

    fig = Figure(; size = (1500, 720))
    colors = Makie.wong_colors()
    axs = [Axis(fig[1, 1]; xlabel = "point of the solver", ylabel = "J", yscale = log10, title = "objective"),
           Axis(fig[1, 2]; xlabel = "point of the solver", ylabel = "a₀", title = "a₀ (band: own injection at 0.072 n_c)"),
           Axis(fig[1, 3]; xlabel = "point of the solver", ylabel = "dark current weight", title = "dark current (u_x > 5, behind bucket 1)"),
           Axis(fig[1, 4]; xlabel = "point of the solver", ylabel = "J_plasma / J", title = "the plasma's own share of J")]
    paxs = [Axis(fig[2, i]; xlabel = "point of the solver", ylabel = TCASE.names[TCASE.free[i]], title = "$(TCASE.names[TCASE.free[i]]) (dashed: the box)") for i in 1:4]
    hspan!(axs[2], 2.45, 2.55; color = (:gray, 0.3))
    sidecar = Dict{String,Any}("run" => basename(dir), "box" => Dict("lo" => TRAPPED_BOX.lo, "hi" => TRAPPED_BOX.hi), "starts" => Dict{String,Any}())
    for (k, n) in enumerate(names)
        P = paths[n]
        it = collect(1:length(P))
        c = colors[k]
        J = [p["J"] for p in P]
        lines!(axs[1], it, J; color = c, label = n); scatter!(axs[1], it, J; color = c, markersize = 6)
        lines!(axs[2], it, [p["a0"] for p in P]; color = c)
        lines!(axs[3], it, [p["dark_weight"] for p in P]; color = c)
        lines!(axs[4], it, [p["J"] > 0 ? p["J_plasma"] / p["J"] : 0.0 for p in P]; color = c)
        for i in 1:4
            lines!(paxs[i], it, [p["free"][i] for p in P]; color = c)
        end
        sidecar["starts"][n] = Dict{String,Any}("J" => J, "a0" => [p["a0"] for p in P], "dark_weight" => [p["dark_weight"] for p in P],
                                                "J_plasma" => [p["J_plasma"] for p in P], "free" => [p["free"] for p in P],
                                                "log_gradient" => [log_gradient(p) for p in P])
    end
    for i in 1:4
        hlines!(paxs[i], [TRAPPED_BOX.lo[i], TRAPPED_BOX.hi[i]]; color = :black, linestyle = :dash)
    end
    axislegend(axs[1]; position = :rb)
    path_png = save_plot(out, "trapped2d_opt", fig, sidecar)

    io = IOBuffer()
    println(io, "Run `", basename(dir), "`: λ₀/", params["ppl"], ", Δy = π/", 4 * params["ny"], ", Float", params["float"], ", ", params["backend"],
            ", gather ", params["gather"], ", budget of a₀ = ", params["a0E"], " at the base pulse; L-BFGS-B with maxiters = ", params["maxiters"], ".\n")
    println(io, "| start | from (τ, w, focus, n) | a₀ | J there | to (τ, w, focus, n) | a₀ | J there | J ratio | on the box's bounds | stop | iterations | points run | largest dark current weight on the path | largest J_plasma/J on the path |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names
        r, P = results[n], paths[n]
        s = P[1]
        e = points[r["key"]]
        println(io, "| ", n, " | ", join(fmt.(s["free"]), ", "), " | ", fmt(s["a0"]), " | ", fmt(s["J"]), " | ", join(fmt.(e["free"]), ", "), " | ", fmt(e["a0"]), " | ",
                fmt(e["J"]), " | ", fmt(e["J"] / s["J"]), " | ", on_bounds(e["free"]), " | ", r["retcode"], " | ", r["iterations"], " | ", r["points_run"], " | ",
                fmt(maximum(p -> p["dark_weight"], P)), " | ", fmt(maximum(p -> p["J"] > 0 ? p["J_plasma"] / p["J"] : 0.0, P)), " |")
    end
    for n in names
        println(io, "\nStart `", n, "`, every point the solver reported:\n")
        println(io, "| point | τ | w | focus | n | a₀ | J | J_plasma/J | dark weight | d ln J/dz (τ, w, focus, n) | seconds of the run |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        for (k, p) in enumerate(paths[n])
            println(io, "| ", k, " | ", join(fmt.(p["free"]), " | "), " | ", fmt(p["a0"]), " | ", fmt(p["J"]), " | ", fmt(p["J"] > 0 ? p["J_plasma"] / p["J"] : 0.0), " | ",
                    fmt(p["dark_weight"]), " | ", join(fmt.(log_gradient(p); d = 3), ", "), " | ", fmt(p["seconds"]; d = 3), " |")
        end
    end
    secs = [p["seconds"] for p in values(points)]
    println(io, "\nCost: ", length(secs), " runs with duals in the four free directions, ", fmt(sum(secs)), " s in all, median ", fmt(Statistics.median(secs)),
            " s per run (first run, with compilation: ", fmt(maximum(secs)), " s).")

    cdir = joinpath(dir, "check")
    if isdir(cdir)
        jobs = Dict(splitext(f)[1] => JSON.parsefile(joinpath(cdir, f)) for f in readdir(cdir) if endswith(f, ".json"))
        ck = TOML.parsefile(joinpath(cdir, "check.toml"))
        best = points[point_key(Float64.(ck["free"]))]
        println(io, "\nThe check at the optimum of start `", ck["start"], "` (", join(fmt.(ck["free"]), ", "), "):\n")
        println(io, "| run | J | d ln J/dz (τ, w, focus, n) | dark weight | J_plasma/J | seconds |")
        println(io, "|---|---|---|---|---|---|")
        row(label, j) = println(io, "| ", label, " | ", fmt(j["J"]; d = 6), " | ", j["gradient"] === nothing ? "–" : join(fmt.(log_gradient(j); d = 3), ", "), " | ",
                                fmt(j["dark_weight"]), " | ", fmt(j["J"] > 0 ? j["J_plasma"] / j["J"] : 0.0), " | ", fmt(j["seconds"]; d = 3), " |")
        row("optimization: λ₀/$(params["ppl"]), Float$(params["float"]), duals", best)
        for (id, label) in (("f64-grad", "λ₀/$(ck["fdppl"]), Float64, duals"), ("fine-plain", "λ₀/$(ck["ppl"]), Float$(params["float"]), plain"),
                            ("fine-grad", "λ₀/$(ck["ppl"]), Float$(params["float"]), duals"))
            haskey(jobs, id) && row(label, jobs[id])
        end
        rows = check_rows(cdir)
        if !isempty(rows)
            println(io, "\n| direction | ForwardDiff d ln J/dz, Float64 | central difference at 2^-", ck["power"], ", Float64 | difference |")
            println(io, "|---|---|---|---|")
            for r in rows
                println(io, "| ", r["direction"], " | ", fmt(r["ad_log"]), " | ", fmt(r["fd_log"]), " | ", fmt(abs(r["ad_log"] - r["fd_log"]); d = 2), " |")
            end
        end
        sidecar["check"] = Dict{String,Any}("jobs" => jobs, "fd" => rows, "optimum" => best)
        open(f -> JSON.print(f, sidecar, 2), joinpath(out, "trapped2d_opt.json"), "w")
    end
    sdir = joinpath(dir, "scan")
    isdir(sdir) && scan_figure(out, dir, sdir, results, paths, io)
    write(joinpath(out, "tables.md"), String(take!(io)))
    println("report: ", path_png, " and ", joinpath(out, "tables.md"))
end

"""
    scan_plane(jobs) -> (; origin, e1, e2)

The plane of one start's scan from its jobs: the one they record, or, for jobs that do not, the
one their points span (z = origin + s·e1 + t·e2 at each job's grid coordinates s, t).
"""
function scan_plane(jobs)
    j0 = jobs[1]
    haskey(j0, "e1") && return (; origin = Float64.(j0["origin"]), e1 = Float64.(j0["e1"]), e2 = Float64.(j0["e2"]))
    z(j) = Float64.(j["z"])
    a = findfirst(j -> j["j"] == j0["j"] && j["i"] != j0["i"], jobs)
    b = findfirst(j -> j["i"] == j0["i"] && j["j"] != j0["j"], jobs)
    e1 = (z(jobs[a]) .- z(j0)) ./ (jobs[a]["s"] - j0["s"])
    e2 = (z(jobs[b]) .- z(j0)) ./ (jobs[b]["t"] - j0["t"])
    return (; origin = z(j0) .- j0["s"] .* e1 .- j0["t"] .* e2, e1, e2)
end

"J on the scan's grids, with each path drawn over them: the plane of every start's path, and every pair of free parameters through the best optimum."
function scan_figure(out, dir, sdir, results, paths, io)
    jobs = [JSON.parsefile(joinpath(sdir, f)) for f in readdir(sdir) if endswith(f, ".json")]
    isempty(jobs) && return
    n = maximum(j -> max(j["i"], j["j"]), jobs)
    grids = Dict{String,Any}()
    for j in jobs
        key = j["kind"] == "plane" ? "plane-$(j["start"])" : "pair-$(j["a"])$(j["b"])"
        g = get!(grids, key) do
            Dict{String,Any}("J" => fill(NaN, n, n), "dark_weight" => fill(NaN, n, n), "u" => fill(NaN, n), "v" => fill(NaN, n), "kind" => j["kind"], "spec" => j)
        end
        g["J"][j["i"], j["j"]] = j["J"]
        g["dark_weight"][j["i"], j["j"]] = j["dark_weight"]
        if j["kind"] == "plane"
            g["u"] .= j["s_grid"]; g["v"] .= j["t_grid"]
        else
            g["u"][j["i"]], g["v"][j["j"]] = j["free"][j["a"]], j["free"][j["b"]]
        end
    end
    keys_sorted = sort(collect(keys(grids)))
    fig = Figure(; size = (420 * min(5, length(keys_sorted)), 400 * cld(length(keys_sorted), 5)))
    colors = Makie.wong_colors()
    names = sort(collect(keys(results)); by = n -> findfirst(==(Symbol(n)), keys(STARTS)))
    sidecar = Dict{String,Any}()
    for (q, key) in enumerate(keys_sorted)
        g = grids[key]
        spec = g["spec"]
        if g["kind"] == "plane"
            r = results[spec["start"]]
            pl = scan_plane(filter(j -> j["kind"] == "plane" && j["start"] == spec["start"], jobs))
            zs = [Float64.(p["z"]) for p in r["path"]]
            ps = [vdot(z .- pl.origin, pl.e1) for z in zs]
            pt = [vdot(z .- pl.origin, pl.e2) for z in zs]
            residual = maximum(vnorm(z .- pl.origin .- ps[k] .* pl.e1 .- pt[k] .* pl.e2) for (k, z) in enumerate(zs))
            xl, yl, title = "s (z units, first direction)", "t (second direction)", "start $(spec["start"]): the path's plane (off it by ≤ $(fmt(residual; d = 2)) in z)"
            curves = [(spec["start"], ps, pt, [p["J"] for p in paths[spec["start"]]])]
            extra = Dict{String,Any}("origin" => pl.origin, "e1" => pl.e1, "e2" => pl.e2, "residual" => residual)
        else
            a, b = spec["a"], spec["b"]
            xl, yl = TCASE.names[TCASE.free[a]], TCASE.names[TCASE.free[b]]
            title = "$xl × $yl, the others at start $(spec["start"])'s optimum"
            curves = [(nm, [p["free"][a] for p in paths[nm]], [p["free"][b] for p in paths[nm]], [p["J"] for p in paths[nm]]) for nm in names]
            extra = Dict{String,Any}("a" => a, "b" => b, "at" => results[spec["start"]]["free"])
        end
        # J spans orders of magnitude across the box (the plasma's own injection), so the height is log₁₀ J
        ax = Axis3(fig[cld(q, 5), mod1(q, 5)]; xlabel = xl, ylabel = yl, zlabel = "log₁₀ J", title, titlesize = 11)
        surface!(ax, g["u"], g["v"], log10.(max.(g["J"], 1.0e-8)); colormap = :viridis, nan_color = (:white, 0.0))
        for (k, (nm, cu, cv, cJ)) in enumerate(curves)
            c = colors[findfirst(==(nm), names)]
            lines!(ax, cu, cv, log10.(cJ); color = c, linewidth = 2)
            scatter!(ax, cu, cv, log10.(cJ); color = c, markersize = 5)
        end
        # grid cells without a run (outside the box) are NaN in the plot and null in the sidecar
        nonan(A) = map(x -> isnan(x) ? nothing : x, A)
        sidecar[key] = merge(Dict{String,Any}("u" => g["u"], "v" => g["v"], "J" => nonan(g["J"]), "dark_weight" => nonan(g["dark_weight"]), "xlabel" => xl, "ylabel" => yl,
                                              "paths" => [Dict("start" => c[1], "u" => c[2], "v" => c[3], "J" => c[4]) for c in curves]), extra)
    end
    # ten 3D panels at save_plot's 2 px per unit make a PNG over the 1 MB a report file may hold (F3)
    save(joinpath(out, "trapped2d_opt_scan.png"), fig; px_per_unit = 1)
    open(io -> JSON.print(io, sidecar, 2), joinpath(out, "trapped2d_opt_scan.json"), "w")
    println(io, "\nScan: ", length(jobs), " plain runs on ", length(grids), " grids of ", n, "×", n, " (", fmt(sum(j -> j["seconds"], jobs)), " s of runs); ",
            "J on them from ", fmt(minimum(j -> j["J"], jobs)), " to ", fmt(maximum(j -> j["J"], jobs)), ", largest dark current weight ", fmt(maximum(j -> j["dark_weight"], jobs)), ".")
end
