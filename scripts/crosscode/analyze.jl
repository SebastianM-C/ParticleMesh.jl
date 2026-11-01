# Observables, the standard plot set and the sidecars of a finished Smilei run.
#
#   julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/<tag> [runner-log]
#
# Reads case.toml, scalars.txt and Fields0.h5 from the run directory; writes plots/<name>.png
# with plots/<name>.json next to each, observables.json, and adds the results, the output hashes
# and the image id (from the runner's log header or image.txt) to manifest.toml. A seeded run
# (a case with a `seeds` list, WP-012) is analyzed per seed-<seed>/ subdirectory as above, and
# the ensemble observable goes into the top directory (scripts/crosscode/ensemble_report.jl).

include(joinpath(@__DIR__, "read.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "ensemble.jl"))
include(joinpath(@__DIR__, "ensemble_report.jl"))
using .SmileiRead, .Observables, .RunManifest, .Ensemble, .EnsembleReport
using CairoMakie
using JSON
using TOML
using Dates

function save_plot(dir, name, fig, sidecar)
    mkpath(joinpath(dir, "plots"))
    save(joinpath(dir, "plots", name * ".png"), fig)
    open(joinpath(dir, "plots", name * ".json"), "w") do io
        JSON.print(io, sidecar, 2)
    end
end

finite_or_nothing(x) = isfinite(x) ? x : nothing

"""
The provenance of the analysis (F3): the commit of the checkout and the hashes of the analysis
code, recorded in the run's manifest next to the preparation commit.
"""
function analysis_record()
    root = dirname(dirname(@__DIR__))
    c = git_commit(root)
    files = ("analyze.jl", "analysis.jl", "ensemble.jl", "ensemble_report.jl", "read.jl", "manifest.jl")
    return Dict{String,Any}("commit" => c.commit, "dirty" => coalesce(c.dirty, "unknown"),
        "inputs" => Dict{String,Any}("scripts/crosscode/" * f => "sha256:" * file_hash(joinpath(@__DIR__, f)) for f in files))
end

"The per-run observable of a case (a seeded case also has the ensemble one)."
run_observable(case) = only((k, o) for (k, o) in case["observables"] if o["method"] != "ensemble_peak_envelope")

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

function analyze(dir; log = nothing, image = nothing)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    name = case["name"]
    sc = read_scalars(dir)
    fl = read_fields(dir; names = ["Ex"], timestep = case["time"]["dt"])
    t, U = sc.time, sc.columns["Uelm_Ex"]
    obs_key, obs = run_observable(case)
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
        record_mode_fit!(result, rk; min_points = 3)
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
        record_mode_fit!(result, rk; min_points = 5)
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
    m = isfile(joinpath(dir, "manifest.toml")) ? read_manifest(dir) : Dict{String,Any}()
    outhash = Dict(relpath(f, dir) => "sha256:" * file_hash(f) for f in outputs)
    img = something(image, image_from_header(log, joinpath(dir, "image.txt")))
    image = merge(get(m, "image", Dict{String,Any}()), Dict{String,Any}("id" => img.id, "name" => img.name))
    write_manifest(dir; extra = Dict{String,Any}("outputs" => outhash, "image" => image,
        "hardware" => hardware(dir), "analysis" => analysis_record(),
        "observables" => Dict(k => v for (k, v) in result if v !== nothing)))
    return result
end

"""
The mode-amplitude estimate as it may be reported (WP-012): `measured_from_mode` only when the
fit is usable (`checked_fit`: finite, at least `min_points` points), with the point count and,
otherwise, the reason it was refused.
"""
function record_mode_fit!(result, rk; min_points)
    c = checked_fit(rk; min_points)
    result["measured_from_mode"] = c.gamma
    result["measured_from_mode_stderr"] = c.gamma_stderr
    result["measured_from_mode_points"] = c.points
    c.usable || (result["measured_from_mode_flag"] = c.flag)
    return result
end

"""
    analyze_seeded(dir; log) -> result

A seeded run: `analyze` of every seed-<seed>/ subdirectory (its own plots, observables and
manifest), then the ensemble observable of the complex mode coefficients of Ex over the seeds,
written to the top directory with its plots, and the manifest of the run updated.
"""
function analyze_seeded(dir; log = nothing)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    seeds = collect(Int, case["numerics"]["seeds"])
    img = image_from_header(log, joinpath(dir, "image.txt"))
    _, obs = run_observable(case)
    mode = only(unique(s["perturbation"]["mode"] for s in case["species"] if haskey(s, "perturbation")))
    coeffs = Vector{ComplexF64}[]
    per_run = []
    t = nothing
    outputs = Dict{String,Any}()
    for s in seeds
        sd = joinpath(dir, "seed-$s")
        analyze(sd; image = img)
        fl = read_fields(sd; names = ["Ex"], timestep = case["time"]["dt"])
        t === nothing ? (t = fl.time) : (fl.time == t || error("seed $s: field dump times differ"))
        push!(coeffs, mode_coefficient(fl.fields["Ex"], mode))
        sc = read_scalars(sd)
        push!(per_run, damping_from_peaks(sc.time, sc.columns["Uelm_Ex"]; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"]))
        for f in filter(isfile, joinpath.(sd, ["scalars.txt", "Fields0.h5", "smilei.log", "container.txt"]))
            outputs[relpath(f, dir)] = "sha256:" * file_hash(f)
        end
    end
    hw = [hardware(joinpath(dir, "seed-$s")) for s in seeds]
    walls = [wall_seconds(joinpath(dir, "seed-$s")) for s in seeds]
    nsteps = round(Int, case["time"]["t_end"] / case["time"]["dt"])
    particles = case["grid"]["cells"] * sum(s["particles_per_cell"] for s in case["species"] if s["mobile"])
    r = write_ensemble(dir, case, t, coeffs, seeds; code = "smilei", per_run,
        extra = Dict{String,Any}("wall_seconds_per_seed" => walls, "particles_per_seed" => particles,
            "steps" => nsteps, "particle_steps_per_seed" => particles * nsteps,
            "hardware" => first(hw)))
    m = read_manifest(dir)
    image = merge(get(m, "image", Dict{String,Any}()), Dict{String,Any}("id" => img.id, "name" => img.name))
    write_manifest(dir; extra = Dict{String,Any}("outputs" => outputs, "image" => image, "hardware" => first(hw),
        "analysis" => analysis_record(),
        "observables" => Dict(k => r[k] for k in ("observable", "measured", "measured_stderr", "maxima_used",
            "ensemble_floor", "observable_exists", "M") if r[k] !== nothing)))
    return r
end

"Wall time of a Smilei run from the start and end stamps in its container.txt, or nothing."
function wall_seconds(dir)
    f = joinpath(dir, "container.txt")
    isfile(f) || return nothing
    stamp(key) = (l = findfirst(x -> startswith(x, "# $key:"), readlines(f)); l === nothing ? nothing :
        strip(split(readlines(f)[l], ":"; limit = 2)[2]))
    a, b = stamp("start"), stamp("end")
    (a === nothing || b === nothing) && return nothing
    secs(x) = (d = Dates.DateTime(x[1:23], dateformat"yyyy-mm-ddTHH:MM:SS.sss"); Dates.datetime2unix(d))
    return secs(b) - secs(a)
end

if abspath(PROGRAM_FILE) == @__FILE__
    dir = ARGS[1]
    log = get(ARGS, 2, nothing)
    seeded = haskey(TOML.parsefile(joinpath(dir, "case.toml"))["numerics"], "seeds")
    r = seeded ? analyze_seeded(dir; log) : analyze(dir; log)
    JSON.print(stdout, r, 2)
    println()
end
