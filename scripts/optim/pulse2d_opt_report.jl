# The figures and tables of reports/wp-087-pulse-shape-study.md from a run directory of
# pulse2d_opt.jl: `pulse2d_opt.jl report <out dir> <run dir>` (included by pulse2d_opt.jl).
using CairoMakie

"The point records of a run directory, by key."
pulse_points(dir) = PointStore(joinpath(dir, "points"); run = _ -> error("the report runs nothing")).records

"Every point a start's solver reported, in order, with the point's record."
pulse_path(res, points) = [points[p["key"]] for p in res["path"]]

pfmt(x; d = 4) = x === nothing ? "–" : x isa AbstractString ? x : x isa Integer ? string(x) : string(round(x; sigdigits = d))

"""
    pulse_check_rows(cdir, point) -> Vector of Dict

Central differences of the check's jobs at `point` (\"best\" or \"reference\") against its Float64
run with duals, per direction, with both in d J/dz (z = x ./ scale).
"""
function pulse_check_rows(cdir, point)
    isdir(cdir) || return Dict{String,Any}[]
    jobs = Dict(splitext(f)[1] => JSON.parsefile(joinpath(cdir, f)) for f in readdir(cdir) if endswith(f, ".json"))
    g = get(jobs, "$point-f64-grad", nothing)
    rows = Dict{String,Any}[]
    g === nothing && return rows
    for (i, n) in enumerate(PCASE.names)
        p, m = get(jobs, "$point-fd-$n-p", nothing), get(jobs, "$point-fd-$n-m", nothing)
        (p === nothing || m === nothing) && continue
        fd = (p["J"] - m["J"]) / (2 * p["h"])
        push!(rows, Dict{String,Any}("direction" => n, "J" => g["J"], "fd_z" => fd * PULSE_BOX.scale[i], "ad_z" => g["gradient"][i] * PULSE_BOX.scale[i], "h" => p["h"]))
    end
    return rows
end

"Which parameters of `x` sit on a bound of `PULSE_BOX`, as `name=lo` or `name=hi`."
function pulse_on_bounds(x)
    out = String[]
    for i in eachindex(x)
        x[i] <= PULSE_BOX.lo[i] && push!(out, "$(PCASE.names[i])=lo")
        x[i] >= PULSE_BOX.hi[i] && push!(out, "$(PCASE.names[i])=hi")
    end
    return isempty(out) ? "none" : join(out, ", ")
end

"""
    shape_curves(x) -> NamedTuple

What the shape at `x` looks like, as plain numbers: the wavefront W(ρ) over the pupil, the
near field's amplitude, the intensity |U|² across the focal plane, the on-axis intensity along
ξ around the focus, and the envelope's intensity |A(s)|².
"""
function shape_curves(x)
    sh = pulse_shape(collect(Float64, x))
    b = sh.profile
    ρ = collect(range(-1, 1; length = 201))
    y = collect(range(-40, 40; length = 401))
    ξ = collect(range(-80, 80; length = 321))
    s = collect(range(-40, 40; length = 401))
    peak(ξi) = maximum(abs2(first(PulseShapes.profile(b, ξi, yi))) for yi in range(-20, 20; length = 161))
    return (; ρ, wavefront = [PulseShapes.wavefront(b, r) for r in ρ], amplitude = [PulseShapes.pupil_amplitude(b.amplitude, r) for r in ρ],
            y, focal = [abs2(first(PulseShapes.profile(b, 0.0, yi))) for yi in y],
            ξ, peak = peak.(ξ), s, intensity = [abs2(PulseShapes.envelope(sh.envelope, si)) for si in s])
end

function pulse_report(out, dir)
    mkpath(out)
    points = pulse_points(dir)
    results = load_results(dir)
    names = sort(collect(keys(results)); by = n -> parse(Int, n[2:end]))
    paths = Dict(n => pulse_path(results[n], points) for n in names)
    ref = points[point_key(collect(PCASE.x0))]
    colors = Makie.wong_colors()

    # the paths: J, Strehl ratio, launched share and the five parameters along each start's solver
    fig = Figure(; size = (1500, 760))
    axJ = Axis(fig[1, 1]; xlabel = "point of the solver", ylabel = "J", title = "objective (dotted: the unaberrated pulse)")
    axS = Axis(fig[1, 2]; xlabel = "point of the solver", ylabel = "Strehl ratio", title = "vacuum Strehl ratio of the focus")
    axL = Axis(fig[1, 3]; xlabel = "point of the solver", ylabel = "1 − launched share", yscale = log10, title = "power outside the antenna's interior rows")
    axW = Axis(fig[1, 4]; xlabel = "point of the solver", ylabel = "envelope duration h", title = "stretched duration (6 at the transform limit)")
    paxs = [Axis(fig[2, i]; xlabel = "point of the solver", ylabel = PCASE.names[i], title = "$(PCASE.names[i]) (dashed: the box)") for i in 1:5]
    hlines!(axJ, [ref["J"]]; color = :black, linestyle = :dot)
    sidecar = Dict{String,Any}("run" => basename(dir), "reference" => Dict("J" => ref["J"], "gradient" => ref["gradient"]),
                               "box" => Dict("lo" => PULSE_BOX.lo, "hi" => PULSE_BOX.hi, "names" => collect(PCASE.names)), "starts" => Dict{String,Any}())
    for (k, n) in enumerate(names)
        P = paths[n]
        it = collect(1:length(P))
        c = colors[mod1(k, length(colors))]
        J = [p["J"] for p in P]
        lines!(axJ, it, J; color = c, label = n); scatter!(axJ, it, J; color = c, markersize = 6)
        lines!(axS, it, [p["strehl"] for p in P]; color = c)
        lines!(axL, it, [max(1 - p["launched_interior"], 1.0e-16) for p in P]; color = c)
        lines!(axW, it, [p["envelope_duration"] for p in P]; color = c)
        for i in 1:5
            lines!(paxs[i], it, [p["free"][i] for p in P]; color = c)
        end
        sidecar["starts"][n] = Dict{String,Any}("J" => J, "free" => [p["free"] for p in P], "gradient" => [p["gradient"] for p in P],
                                                "strehl" => [p["strehl"] for p in P], "launched_interior" => [p["launched_interior"] for p in P],
                                                "envelope_duration" => [p["envelope_duration"] for p in P], "dark_weight" => [p["dark_weight"] for p in P])
    end
    for i in 1:5
        hlines!(paxs[i], [PULSE_BOX.lo[i], PULSE_BOX.hi[i]]; color = :gray, linestyle = :dash)
    end
    axislegend(axJ; position = :rb)
    save_plot(out, "pulse2d_paths", fig, sidecar)

    # the shapes: the unaberrated pulse against each start's optimum
    best = argmax(n -> results[n]["J"], names)
    fig2 = Figure(; size = (1500, 420))
    a1 = Axis(fig2[1, 1]; xlabel = "pupil coordinate ρ", ylabel = "W (waves)", title = "wavefront (gray: near-field amplitude)")
    a2 = Axis(fig2[1, 2]; xlabel = "y − center (c/ω₀)", ylabel = "|U|²", title = "focal-plane intensity, vacuum")
    a3 = Axis(fig2[1, 3]; xlabel = "ξ = x − focus (c/ω₀)", ylabel = "peak |U|²", title = "peak intensity along the beam, vacuum")
    a4 = Axis(fig2[1, 4]; xlabel = "s (1/ω₀)", ylabel = "|A|²", title = "envelope intensity")
    side2 = Dict{String,Any}("best" => best)
    rc = shape_curves(PCASE.x0)
    lines!(a1, rc.ρ, rc.amplitude; color = (:gray, 0.6))
    for (lab, x, c, ls) in [("unaberrated", collect(PCASE.x0), :black, :dot); [(n, results[n]["free"], colors[mod1(k, length(colors))], n == best ? :solid : :dash) for (k, n) in enumerate(names)]]
        sc = shape_curves(x)
        lines!(a1, sc.ρ, sc.wavefront; color = c, linestyle = ls, label = lab)
        lines!(a2, sc.y, sc.focal; color = c, linestyle = ls)
        lines!(a3, sc.ξ, sc.peak; color = c, linestyle = ls)
        lines!(a4, sc.s, sc.intensity; color = c, linestyle = ls)
        side2[lab] = Dict{String,Any}(string(k) => v for (k, v) in pairs(sc))
        side2[lab]["x"] = collect(Float64, x)
    end
    axislegend(a1; position = :lt)
    save_plot(out, "pulse2d_shapes", fig2, side2)

    # tables
    io = IOBuffer()
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))
    p = m["parameters"]
    println(io, "Run `$(basename(dir))` at commit `$(get(m, "commit", get(get(m, "git", Dict()), "commit", "?")))`: ppl $(p["ppl"]), ny $(p["ny"]), Float$(p["float"]), $(p["backend"]), $(p["gather"]) gather, ",
            "maxiters $(p["maxiters"]), pgtol $(p["pgtol"]) of the reference gradient, seed $(p["seed"]), na $(pfmt(p["na"]; d = 6)).\n")
    println(io, "Reference (unaberrated, transform-limited, focus 80): J = $(pfmt(ref["J"])), dJ/dz = ", join(pfmt.(ref["gradient"] .* PULSE_BOX.scale), ", "),
            " (", join(PCASE.names, ", "), "), a₀ = $(pfmt(ref["a0"])), $(pfmt(ref["seconds"]; d = 3)) s.\n")
    println(io, "| start | start x | optimum x | J | J/J_ref | Strehl | h | on bounds | retcode | iterations | points run | max proj. grad. (z) | seconds |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names
        r = results[n]
        q = points[r["key"]]
        println(io, "| $n | ", join(pfmt.(r["start"]; d = 3), ", "), " | ", join(pfmt.(r["free"]; d = 4), ", "), " | $(pfmt(r["J"])) | $(pfmt(r["J"] / ref["J"]; d = 3)) | ",
                "$(pfmt(q["strehl"]; d = 3)) | $(pfmt(q["envelope_duration"]; d = 3)) | $(pulse_on_bounds(r["free"])) | $(r["retcode"]) | $(r["iterations"]) | $(r["points_run"]) | ",
                "$(pfmt(r["projected_gradient"]; d = 3)) | $(round(Int, r["seconds"])) |")
    end
    secs = [q["seconds"] for q in values(points)]
    println(io, "\n$(length(points)) points run with duals in five directions, median $(pfmt(sort(secs)[cld(length(secs), 2)]; d = 3)) s each; ",
            "lowest launched share in the interior rows over all points $(pfmt(minimum(q["launched_interior"] for q in values(points)); d = 6)).\n")
    cdirs = sort(filter(f -> startswith(f, "check") && isdir(joinpath(dir, f)), readdir(dir)); by = f -> TOML.parsefile(joinpath(dir, f, "check.toml"))["power"])
    for cdir in joinpath.(dir, cdirs), point in ("best", "reference")
        rows = pulse_check_rows(cdir, point)
        isempty(rows) && continue
        ck = TOML.parsefile(joinpath(cdir, "check.toml"))
        println(io, "Check at the $point point, Float64 at ppl $(ck["fdppl"]), ny $(ck["fdny"]), central differences at 2^-$(ck["power"]) of each scale (J = $(pfmt(rows[1]["J"]; d = 6))):\n")
        println(io, "| direction | dJ/dz central differences | dJ/dz ForwardDiff | relative difference |")
        println(io, "|---|---|---|---|")
        for r in rows
            rel = abs(r["fd_z"] - r["ad_z"]) / max(abs(r["ad_z"]), eps())
            println(io, "| $(r["direction"]) | $(pfmt(r["fd_z"]; d = 6)) | $(pfmt(r["ad_z"]; d = 6)) | $(pfmt(rel; d = 2)) |")
        end
        println(io)
    end
    write(joinpath(out, "tables.md"), String(take!(io)))
    println("report written to $out")
end
