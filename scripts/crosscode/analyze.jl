# Observables, the standard plot set and the sidecars of a finished Smilei run.
#
#   julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/<tag> [runner-log]
#
# Reads case.toml, scalars.txt and Fields0.h5 from the run directory; writes plots/<name>.png
# with plots/<name>.json next to each, observables.json, and adds the results, the output hashes
# and the image id (from the runner's log header or image.txt) to manifest.toml. Schema-2 cases
# (2D) go through `analyze_2d`, which evaluates every pre-registered observable of the case file
# and also records the wall time and the particle-steps of the run.

include(joinpath(@__DIR__, "read.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "theory.jl"))
using .SmileiRead, .Observables, .RunManifest, .Theory
using Statistics: median, mean
using Dates: DateTime, Millisecond
using CairoMakie
using JSON
using TOML

function save_plot(dir, name, fig, sidecar)
    mkpath(joinpath(dir, "plots"))
    save(joinpath(dir, "plots", name * ".png"), fig)
    open(joinpath(dir, "plots", name * ".json"), "w") do io
        JSON.print(io, sidecar, 2)
    end
end

finite_or_nothing(x) = isfinite(x) ? x : nothing

"The image line of the runner's log header or of run-smilei.sh's image.txt, if any."
function image_id(dir, log)
    for f in (log, joinpath(dir, "image.txt"))
        (f === nothing || !isfile(f)) && continue
        for line in eachline(f)
            m = match(r"^#\s*image:\s*(.*)$"i, line)
            m === nothing || return strip(m.captures[1])
        end
    end
    return "unknown"
end

"CPU model, thread count and CPU count from the run's container.txt; \"unknown\" where the run did not record them."
function hardware(dir)
    f = joinpath(dir, "container.txt")
    lines = isfile(f) ? readlines(f) : String[]
    cpu = something(findfirst(l -> startswith(l, "# cpu model:"), lines), 0)
    thr = something(findfirst(l -> startswith(l, "# threads:"), lines), 0)
    h = Dict{String,Any}()
    if thr > 0
        # "# threads: 8, cpus: 8"
        mt = match(r"threads:\s*(\d+)", lines[thr])
        mc = match(r"cpus:\s*(\d+)", lines[thr])
        mt === nothing || (h["threads"] = parse(Int, mt.captures[1]))
        mc === nothing || (h["cpus"] = parse(Int, mc.captures[1]))
    end
    if cpu > 0
        h["cpu"] = strip(split(lines[cpu], ":"; limit = 2)[2])
        h["cpu_source"] = "run"
    else
        h["cpu"] = "unknown"
        h["cpu_source"] = "not recorded by the run"
    end
    return h
end

function analyze(dir; log = nothing)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    get(case, "schema", 1) == 2 && return analyze_2d(dir, case; log)
    name = case["name"]
    sc = read_scalars(dir)
    fl = read_fields(dir; names = ["Ex"], timestep = case["time"]["dt"])
    t, U = sc.time, sc.columns["Uelm_Ex"]
    obs_key, obs = only(case["observables"])
    mode = only(unique(s["perturbation"]["mode"] for s in case["species"] if haskey(s, "perturbation")))
    Ek = mode_amplitude(fl.fields["Ex"], mode)

    result = Dict{String,Any}("case" => name, "observable" => obs_key, "theory" => obs["theory"])
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "field energy Uelm_Ex",
        title = "$name: field energy")
    pos = U .> 0
    lines!(ax, t[pos], U[pos]; color = :black, label = "Smilei")
    if obs["method"] == "peak_envelope"
        r = damping_from_peaks(t, U; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
        rk = damping_from_peaks(fl.time, Ek .^ 2; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
        result["measured"] = finite_or_nothing(r.gamma)
        result["measured_stderr"] = finite_or_nothing(r.gamma_stderr)
        result["measured_frequency"] = finite_or_nothing(r.omega)
        result["theory_frequency"] = obs["theory_frequency"]
        result["measured_from_mode"] = finite_or_nothing(rk.gamma)
        result["measured_from_mode_stderr"] = finite_or_nothing(rk.gamma_stderr)
        result["noise_floor"] = r.floor
        result["fit_points"] = length(r.t_fit)
        result["fit_range"] = isempty(r.t_fit) ? nothing : [first(r.t_fit), last(r.t_fit)]
        if r.fit !== nothing
            scatter!(ax, r.t_fit, r.U_fit; color = :red, markersize = 6, label = "maxima used")
            lines!(ax, r.t_fit, exp.(r.fit.intercept .+ r.fit.slope .* r.t_fit); color = :red,
                label = "fit γ = $(round(r.gamma; sigdigits = 4))")
            t0 = first(r.t_fit)
            lines!(ax, r.t_fit, r.U_fit[1] .* exp.(-2obs["theory"] .* (r.t_fit .- t0));
                color = :blue, linestyle = :dash, label = "theory γ = $(round(obs["theory"]; sigdigits = 4))")
        end
        hlines!(ax, [r.floor]; color = :gray, linestyle = :dot, label = "noise floor")
    else
        r = growth_from_window(t, U; window = obs["window"])
        rk = growth_from_window(fl.time, Ek .^ 2; window = obs["window"])
        result["measured"] = finite_or_nothing(r.gamma)
        result["measured_stderr"] = finite_or_nothing(r.gamma_stderr)
        result["measured_from_mode"] = finite_or_nothing(rk.gamma)
        result["measured_from_mode_stderr"] = finite_or_nothing(rk.gamma_stderr)
        result["theory_nonrelativistic"] = obs["theory_nonrelativistic"]
        result["fit_points"] = length(r.t_fit)
        result["fit_range"] = isempty(r.t_fit) ? nothing : [first(r.t_fit), last(r.t_fit)]
        result["saturation"] = Dict("t" => r.t_max, "U" => r.U_max)
        if r.fit !== nothing
            lines!(ax, r.t_fit, exp.(r.fit.intercept .+ r.fit.slope .* r.t_fit); color = :red,
                linewidth = 3, label = "fit γ = $(round(r.gamma; sigdigits = 4))")
            t0 = first(r.t_fit)
            lines!(ax, r.t_fit, r.U_fit[1] .* exp.(2obs["theory"] .* (r.t_fit .- t0));
                color = :blue, linestyle = :dash, label = "theory γ = $(round(obs["theory"]; sigdigits = 4))")
        end
    end
    result["relative_difference"] = result["measured"] === nothing ? nothing :
        (result["measured"] - obs["theory"]) / obs["theory"]
    axislegend(ax; position = :rt)
    save_plot(dir, "field-energy", fig, merge(Dict("plot" => "field energy Uelm_Ex versus time, log scale"), result))

    # mode amplitude of Ex from the field dumps
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|Ê_x| of mode $mode",
        title = "$name: mode $mode of Ex")
    p = Ek .> 0
    lines!(ax, fl.time[p], Ek[p]; color = :black)
    save_plot(dir, "mode-amplitude", fig, Dict("plot" => "amplitude of Fourier mode $mode of Ex versus time",
        "mode" => mode, "initial" => Ek[1], "max" => maximum(Ek), "t_max" => hasproperty(rk, :t_max) ? rk.t_max : fl.time[argmax(Ek)],
        "final" => Ek[end],
        "gamma_from_mode" => result["measured_from_mode"], "gamma_from_mode_stderr" => result["measured_from_mode_stderr"]))

    # field snapshots
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xlabel = "x [c/ω₀]", ylabel = "Ex [m_e c ω₀/e]", title = "$name: Ex snapshots")
    nt = length(fl.time)
    picks = unique(round.(Int, range(1, nt; length = 5)))
    xE = fl.x["Ex"]
    snaps = Dict{String,Any}()
    for j in picks
        lines!(ax, xE, fl.fields["Ex"][:, j]; label = "t = $(round(fl.time[j]; digits = 2))")
        snaps[string(round(fl.time[j]; digits = 4))] = maximum(abs, fl.fields["Ex"][:, j])
    end
    axislegend(ax; position = :rt)
    save_plot(dir, "field-snapshots", fig, Dict("plot" => "Ex(x) at five times", "max_abs_Ex_at_t" => snaps))

    # energy conservation
    Utot = sc.columns["Utot"]
    drift = (Utot .- Utot[1]) ./ Utot[1]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "(Utot - Utot(0)) / Utot(0)", title = "$name: total energy")
    lines!(ax, t, drift; color = :black)
    save_plot(dir, "energy-conservation", fig, Dict("plot" => "relative change of the total energy versus time",
        "max_abs_relative_change" => maximum(abs, drift), "final_relative_change" => drift[end],
        "Utot0" => Utot[1], "Ukin0" => sc.columns["Ukin"][1], "Uelm0" => sc.columns["Uelm"][1]))

    open(io -> JSON.print(io, result, 2), joinpath(dir, "observables.json"), "w")
    outputs = filter(isfile, joinpath.(dir, ["scalars.txt", "Fields0.h5", "smilei.log", "container.txt"]))
    m = read_manifest(dir)
    outhash = Dict(relpath(f, dir) => "sha256:" * file_hash(f) for f in outputs)
    image = merge(get(m, "image", Dict{String,Any}()), Dict{String,Any}("id" => image_id(dir, log)))
    write_manifest(dir; extra = Dict{String,Any}("outputs" => outhash, "image" => image,
        "hardware" => hardware(dir),
        "observables" => Dict(k => v for (k, v) in result if v !== nothing)))
    return result
end

"Wall time in seconds between the `# start:` and `# end:` lines of the run's container.txt, or nothing."
function wall_seconds(dir)
    f = joinpath(dir, "container.txt")
    isfile(f) || return nothing
    stamp(prefix) = for l in eachline(f)
        startswith(l, prefix) && return DateTime(strip(l[length(prefix)+1:end])[1:23])
    end
    t0, t1 = stamp("# start:"), stamp("# end:")
    (t0 === nothing || t1 === nothing) && return nothing
    return (t1 - t0) / Millisecond(1000)
end

"""
    run_size(case) -> Dict

Particles and particle-steps of a schema-2 case: the particles of every species
(cells × particles per cell), those of the mobile species, and the products with the step
count.
"""
function run_size(case)
    cells = prod(case["grid"]["cells"])
    steps = round(Int, case["time"]["t_end"] / case["time"]["dt"])
    n(s) = cells * prod(s["particles_per_cell"])
    total = sum(n, case["species"])
    mobile = sum(n(s) for s in case["species"] if s["mobile"])
    return Dict{String,Any}("cells" => cells, "steps" => steps, "particles" => total,
        "mobile_particles" => mobile, "particle_steps" => total * steps,
        "mobile_particle_steps" => mobile * steps)
end

"The time series (t, U) a pre-registered observable of a 2D case fits, and the mode coefficients it came from."
function observable_series(obs, sc, fl, mode, lengths)
    sig = obs["signal"]
    if sig == "mode_projection"
        a, b = obs["fields"]
        c = project_mode(fl.fields[a], fl.fields[b], fl.x[a], fl.x[b], mode, lengths;
            component = Symbol(obs["component"]))
        return fl.time, abs2.(c)
    elseif sig == "scalar_sum"
        return sc.time, sum(sc.columns[k] for k in obs["columns"])
    elseif sig == "scalar_ratio"
        a, b = obs["columns"]
        return sc.time, sc.columns[a] ./ sc.columns[b]
    elseif sig == "mode_ratio"
        a, b = obs["fields"]
        ca = mode_coefficient(fl.fields[a], mode, fl.x[a], lengths)
        cb = mode_coefficient(fl.fields[b], mode, fl.x[b], lengths)
        return fl.time, abs2.(ca) ./ abs2.(cb)
    end
    error("unknown signal $sig")
end

function analyze_2d(dir, case; log = nothing)
    name = case["name"]
    observables = case["observables"]
    sc = read_scalars(dir)
    fieldnames = unique(vcat(String[], (get(o, "fields", String[]) for o in values(observables))...))
    fl = read_fields(dir; names = fieldnames, timestep = case["time"]["dt"])
    lengths = Tuple(Float64.(case["grid"]["length"]))
    mode = Tuple(Int.(case["wavevector"]["mode"]))
    series = Dict(k => observable_series(o, sc, fl, mode, lengths) for (k, o) in observables)
    fits = Dict{String,Any}()
    results = Dict{String,Any}()
    # rate fits first, then the observables that refer to a fit (`fit_of`)
    order = sort(collect(keys(observables)); by = k -> (haskey(observables[k], "fit_of"), k))
    for key in order
        obs = observables[key]
        t, U = series[key]
        r = Dict{String,Any}("signal" => obs["signal"], "method" => obs["method"], "theory" => obs["theory"])
        if obs["method"] == "peak_envelope"
            f = damping_from_peaks(t, U; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
            fits[key] = f
            r["measured"] = finite_or_nothing(f.gamma)
            r["measured_stderr"] = finite_or_nothing(f.gamma_stderr)
            r["measured_frequency"] = finite_or_nothing(f.omega)
            r["theory_frequency"] = obs["theory_frequency"]
            r["noise_floor"] = f.floor
            r["fit_points"] = length(f.t_fit)
            r["fit_range"] = isempty(f.t_fit) ? nothing : [first(f.t_fit), last(f.t_fit)]
        elseif obs["method"] == "exponential_window"
            f = growth_from_window(t, U; window = obs["window"])
            fits[key] = f
            r["measured"] = finite_or_nothing(f.gamma)
            r["measured_stderr"] = finite_or_nothing(f.gamma_stderr)
            r["fit_points"] = length(f.t_fit)
            r["fit_range"] = isempty(f.t_fit) ? nothing : [first(f.t_fit), last(f.t_fit)]
            r["saturation"] = Dict("t" => f.t_max, "value" => f.U_max)
        elseif obs["method"] == "first_output"
            r["measured"] = finite_or_nothing(U[1])
            r["t"] = t[1]
        elseif obs["method"] == "median_over_fit"
            ref = fits[obs["fit_of"]]
            idx = [argmin(abs.(t .- tp)) for tp in ref.t_fit]
            r["measured"] = isempty(idx) ? nothing : finite_or_nothing(median(U[idx]))
            r["points"] = length(idx)
        else
            error("unknown method $(obs["method"])")
        end
        haskey(obs, "theory_continuum") && (r["theory_continuum"] = obs["theory_continuum"])
        r["relative_difference"] = r["measured"] === nothing ? nothing : (r["measured"] - obs["theory"]) / obs["theory"]
        results[key] = r
    end

    # Plots: the fitted series (field energy from the scalars, mode amplitude from the dumps).
    sign = any(o["method"] == "peak_envelope" for o in values(observables)) ? -1 : 1
    for (plotname, sig, ylabel) in (("field-energy", "scalar_sum", "field energy (sum of $(join(unique(vcat((get(o, "columns", String[]) for o in values(observables) if o["signal"] == "scalar_sum")...)), " + ")))"),
                                    ("mode-amplitude", "mode_projection", "|mode $(collect(mode))|² of the projected field"))
        keys_ = sort([k for (k, o) in observables if o["signal"] == sig])
        isempty(keys_) && continue
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel, title = "$name: $plotname")
        side = Dict{String,Any}("plot" => "$ylabel versus time, log scale")
        for key in keys_
            t, U = series[key]
            obs, f, r = observables[key], fits[key], results[key]
            pos = U .> 0
            lines!(ax, t[pos], U[pos]; color = :black, label = "Smilei ($key)")
            if f.fit !== nothing
                lines!(ax, f.t_fit, exp.(f.fit.intercept .+ f.fit.slope .* f.t_fit); color = :red,
                    linewidth = 2, label = "fit γ = $(round(f.gamma; sigdigits = 4))")
                lines!(ax, f.t_fit, f.U_fit[1] .* exp.(2sign * obs["theory"] .* (f.t_fit .- f.t_fit[1]));
                    color = :blue, linestyle = :dash, label = "theory γ = $(round(obs["theory"]; sigdigits = 4))")
                obs["method"] == "peak_envelope" && scatter!(ax, f.t_fit, f.U_fit; color = :red, markersize = 6)
            end
            hasproperty(f, :floor) && hlines!(ax, [f.floor]; color = :gray, linestyle = :dot, label = "noise floor")
            side[key] = merge(r, Dict("initial" => U[1], "max" => maximum(U), "final" => U[end]))
        end
        axislegend(ax; position = sign < 0 ? :rt : :lt)
        save_plot(dir, plotname, fig, side)
    end

    # The Ex/Ey energy ratio, from the scalars and from the mode coefficients.
    rkeys = sort([k for (k, o) in observables if o["signal"] in ("scalar_ratio", "mode_ratio")])
    if !isempty(rkeys)
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "Ex/Ey energy ratio", title = "$name: energy ratio")
        side = Dict{String,Any}("plot" => "ratio of the Ex and Ey field energies (scalars) and of the mode coefficients, versus time")
        for key in rkeys
            t, R = series[key]
            pos = R .> 0
            lines!(ax, t[pos], R[pos]; label = key)
            side[key] = merge(results[key], Dict("first" => R[1], "median" => median(R)))
        end
        o = observables[first(rkeys)]
        hlines!(ax, [o["theory"]]; color = :blue, linestyle = :dash, label = "Yee grid $(round(o["theory"]; sigdigits = 6))")
        hlines!(ax, [o["theory_continuum"]]; color = :gray, linestyle = :dot, label = "continuum $(o["theory_continuum"])")
        axislegend(ax; position = :rt)
        save_plot(dir, "energy-ratio", fig, side)
    end

    # Field snapshots: the first field of the projection at four times, and the k = 0 component
    # (the mean over the box) of every dumped field.
    fname = first(first(get(o, "fields", [""]) for o in values(observables) if o["signal"] == "mode_projection"))
    F = fl.fields[fname]
    nt = size(F, 3)
    picks = unique(round.(Int, range(1, nt; length = 4)))
    fig = Figure(size = (1000, 900))
    side = Dict{String,Any}("plot" => "$fname(x, y) at four times", "max_abs_at_t" => Dict{String,Any}(),
        "box_mean_at_first_output" => Dict(n => mean(fl.fields[n][1:end-1, 1:end-1, 1]) for n in fieldnames))
    for (p, j) in enumerate(picks)
        ax = Axis(fig[(p - 1) ÷ 2 + 1, (p - 1) % 2 + 1]; xlabel = "x [c/ω₀]", ylabel = "y [c/ω₀]",
            title = "$fname, t = $(round(fl.time[j]; digits = 2))", aspect = DataAspect())
        heatmap!(ax, fl.x[fname][1], fl.x[fname][2], F[:, :, j]; colormap = :RdBu)
        side["max_abs_at_t"][string(round(fl.time[j]; digits = 4))] = maximum(abs, F[:, :, j])
    end
    save_plot(dir, "field-snapshots", fig, side)

    # Energy conservation.
    Utot = sc.columns["Utot"]
    drift = (Utot .- Utot[1]) ./ Utot[1]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "(Utot - Utot(0)) / Utot(0)", title = "$name: total energy")
    lines!(ax, sc.time, drift; color = :black)
    save_plot(dir, "energy-conservation", fig, Dict("plot" => "relative change of the total energy versus time",
        "max_abs_relative_change" => maximum(abs, drift), "final_relative_change" => drift[end],
        "Utot0" => Utot[1], "Ukin0" => sc.columns["Ukin"][1], "Uelm0" => sc.columns["Uelm"][1]))

    size_ = run_size(case)
    wall = wall_seconds(dir)
    performance = merge(size_, Dict{String,Any}("wall_seconds" => wall === nothing ? "not recorded" : wall))
    wall === nothing || (performance["ns_per_mobile_particle_step"] = 1e9 * wall / size_["mobile_particle_steps"])
    result = Dict{String,Any}("case" => name, "observables" => results, "performance" => performance)
    open(io -> JSON.print(io, result, 2), joinpath(dir, "observables.json"), "w")
    outputs = filter(isfile, joinpath.(dir, ["scalars.txt", "Fields0.h5", "smilei.log", "container.txt"]))
    m = read_manifest(dir)
    outhash = Dict(relpath(f, dir) => "sha256:" * file_hash(f) for f in outputs)
    image = merge(get(m, "image", Dict{String,Any}()), Dict{String,Any}("id" => image_id(dir, log)))
    flat = Dict(k => Dict(kk => vv for (kk, vv) in v if vv !== nothing) for (k, v) in results)
    write_manifest(dir; extra = Dict{String,Any}("outputs" => outhash, "image" => image,
        "hardware" => hardware(dir), "performance" => performance, "observables" => flat))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    dir = ARGS[1]
    r = analyze(dir; log = get(ARGS, 2, nothing))
    JSON.print(stdout, r, 2)
    println()
end
