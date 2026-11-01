# Observables, the standard plot set and the sidecars of a finished Smilei or PIConGPU run.
#
#   julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/<tag> [runner-log]
#
# Reads case.toml and the code's output from the run directory (Smilei: scalars.txt and
# Fields0.h5; PIConGPU: openPMD/simData.h5 and the energy files, through units.toml, with the
# quasi-1D runs of schema-1 cases averaged over y, read.jl); writes plots/<name>.png
# with plots/<name>.json next to each, observables.json, and adds the results, the output hashes
# and the image id (from the runner's log header or image.txt) to manifest.toml. A seeded run
# (a case with a `seeds` list, WP-012) is analyzed per seed-<seed>/ subdirectory as above, and
# the ensemble observable goes into the top directory (scripts/crosscode/ensemble_report.jl).
# Schema-2 cases (2D) go through `analyze_2d`, which evaluates every pre-registered observable of
# the case file and also records the wall time and the particle-steps of the run.

include(joinpath(@__DIR__, "read.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "theory.jl"))
include(joinpath(@__DIR__, "ensemble.jl"))
include(joinpath(@__DIR__, "ensemble_report.jl"))
using .SmileiRead, .PIConGPURead, .Observables, .RunManifest, .Theory, .Ensemble, .EnsembleReport
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

"The code whose output `dir` holds: \"picongpu\" or \"smilei\"."
code_of(dir) = picongpu_output(dir) ? "picongpu" : "smilei"
code_label(dir) = code_of(dir) == "picongpu" ? "PIConGPU" : "Smilei"

"The scalars of a run in the schema of `read_scalars`, from either code."
run_scalars(dir) = code_of(dir) == "picongpu" ? read_picongpu_scalars(dir) : read_scalars(dir)

"""
The field dumps of a run in the schema of `read_fields`, from either code; a PIConGPU run of a
schema-1 case is quasi-1D and comes back averaged over y, in the 1D schema.
"""
function run_fields(dir; names, timestep)
    code_of(dir) == "smilei" && return read_fields(dir; names, timestep)
    q = PIConGPURead.picongpu_units(dir)["quasi1d"]
    return read_picongpu_fields(dir; names, yaverage = q)
end

"The output files of a run whose hashes go into the manifest."
function output_files(dir)
    code_of(dir) == "smilei" &&
        return filter(isfile, joinpath.(dir, ["scalars.txt", "Fields0.h5", "smilei.log", "container.txt"]))
    own = filter(f -> endswith(f, ".dat") || f == "picongpu.log", readdir(dir))
    return filter(isfile, vcat(joinpath.(dir, own), joinpath(dir, "openPMD", "simData.h5"),
        joinpath.(dir, ["container.txt", "versions.toml", "build.log"])))
end

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
    code_of(dir) == "picongpu" && return picongpu_hardware(dir)
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
    get(case, "schema", 1) == 2 && return analyze_2d(dir, case; log, image)
    name = case["name"]
    code = code_label(dir)
    sc = run_scalars(dir)
    fl = run_fields(dir; names = ["Ex"], timestep = case["time"]["dt"])
    t, U = sc.time, sc.columns["Uelm_Ex"]
    obs_key, obs = run_observable(case)
    mode = only(unique(s["perturbation"]["mode"] for s in case["species"] if haskey(s, "perturbation")))
    Ek = mode_amplitude(fl.fields["Ex"], mode)

    result = Dict{String,Any}("case" => name, "observable" => obs_key, "theory" => obs["theory"])
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "field energy Uelm_Ex",
        title = "$name: field energy")
    pos = U .> 0
    lines!(ax, t[pos], U[pos]; color = :black, label = code)
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

    result["code"] = code_of(dir)
    open(io -> JSON.print(io, result, 2), joinpath(dir, "observables.json"), "w")
    outputs = output_files(dir)
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

A seeded run of either code: `analyze` of every seed-<seed>/ subdirectory (its own plots,
observables and manifest), then the ensemble observable of the complex mode coefficients over
the seeds, written to the top directory with its plots, and the manifest of the run updated.
Schema 1: the coefficient of Ex at the perturbation's mode, the per-run fits on Uelm_Ex.
Schema 2: the coefficient the ensemble observable's `signal` names (`mode_projection` of its
`fields`), the per-run fits on the scalar sum of the case's `damping_rate_energy` columns.
"""
function analyze_seeded(dir; log = nothing)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    two = get(case, "schema", 1) == 2
    seeds = collect(Int, case["numerics"]["seeds"])
    img = image_from_header(log, joinpath(dir, "image.txt"))
    ekey, eobs = ensemble_observable(case)
    coeffs = Vector{ComplexF64}[]
    per_run = []
    t = nothing
    outputs = Dict{String,Any}()
    for s in seeds
        sd = joinpath(dir, "seed-$s")
        analyze(sd; image = img)
        if two
            a, b = eobs["fields"]
            fl = run_fields(sd; names = [a, b], timestep = case["time"]["dt"])
            mode = Tuple(Int.(case["wavevector"]["mode"]))
            lengths = Tuple(Float64.(case["grid"]["length"]))
            c = project_mode(fl.fields[a], fl.fields[b], fl.x[a], fl.x[b], mode, lengths; component = Symbol(eobs["component"]))
        else
            mode = only(unique(x["perturbation"]["mode"] for x in case["species"] if haskey(x, "perturbation")))
            fl = run_fields(sd; names = ["Ex"], timestep = case["time"]["dt"])
            c = mode_coefficient(fl.fields["Ex"], mode)
        end
        t === nothing ? (t = fl.time) : (fl.time == t || error("seed $s: field dump times differ"))
        push!(coeffs, c)
        sc = run_scalars(sd)
        if two
            ro = case["observables"]["damping_rate_energy"]
            U = sum(sc.columns[k] for k in ro["columns"])
        else
            _, ro = run_observable(case)
            U = sc.columns["Uelm_Ex"]
        end
        push!(per_run, damping_from_peaks(sc.time, U; fit_start = ro["fit_start"], floor_factor = ro["floor_factor"]))
        for f in output_files(sd)
            outputs[relpath(f, dir)] = "sha256:" * file_hash(f)
        end
    end
    hw = [hardware(joinpath(dir, "seed-$s")) for s in seeds]
    walls = [wall_seconds(joinpath(dir, "seed-$s")) for s in seeds]
    nsteps = round(Int, case["time"]["t_end"] / case["time"]["dt"])
    cells = prod(case["grid"]["cells"])
    particles = cells * sum(prod(s["particles_per_cell"]) for s in case["species"] if s["mobile"])
    code = code_of(joinpath(dir, "seed-$(first(seeds))"))
    extra = Dict{String,Any}("wall_seconds_per_seed" => walls, "particles_per_seed" => particles,
        "steps" => nsteps, "particle_steps_per_seed" => particles * nsteps, "hardware" => first(hw))
    code == "picongpu" && (extra["compile_seconds"] = compile_seconds(dir))
    r = write_ensemble(dir, case, t, coeffs, seeds; code, per_run, extra)
    m = read_manifest(dir)
    image = merge(get(m, "image", Dict{String,Any}()), Dict{String,Any}("id" => img.id, "name" => img.name))
    write_manifest(dir; extra = Dict{String,Any}("outputs" => outputs, "image" => image, "hardware" => first(hw),
        "analysis" => analysis_record(),
        "observables" => Dict(k => r[k] for k in ("observable", "measured", "measured_stderr", "maxima_used",
            "ensemble_floor", "observable_exists", "M") if r[k] !== nothing)))
    return r
end

"""
The hardware of a PIConGPU run from the top run directory's container.txt (the GPU name and
driver as nvidia-smi reports them inside the container, the CPU model and count).
"""
function picongpu_hardware(dir)
    f = joinpath(picongpu_top(dir), "container.txt")
    lines = isfile(f) ? readlines(f) : String[]
    h = Dict{String,Any}()
    for (key, prefix) in (("gpu", "# gpu:"), ("cpu", "# cpu model:"), ("cpus", "# cpus:"), ("picongpu", "# picongpu:"))
        k = findfirst(l -> startswith(l, prefix), lines)
        h[key] = k === nothing ? "unknown" : strip(lines[k][length(prefix)+1:end])
    end
    h["source"] = isempty(lines) ? "not recorded by the run" : "run"
    return h
end

"The top run directory of a PIConGPU run (the one holding run.sh), from it or a seed directory."
picongpu_top(dir) = isfile(joinpath(dir, "run.sh")) ? dir : dirname(abspath(dir))

"Seconds between two `# <what> start:`/`end:` lines of a container.txt, or nothing."
function stamp_interval(f, start_prefix, end_prefix)
    isfile(f) || return nothing
    stamp(prefix) = for l in eachline(f)
        startswith(l, prefix) && return DateTime(strip(l[length(prefix)+1:end])[1:23])
    end
    t0, t1 = stamp(start_prefix), stamp(end_prefix)
    (t0 === nothing || t1 === nothing) && return nothing
    return (t1 - t0) / Millisecond(1000)
end

"The compile time of a PIConGPU run's setup in seconds, from container.txt."
compile_seconds(dir) = stamp_interval(joinpath(picongpu_top(dir), "container.txt"), "# compile start:", "# compile end:")

"""
Wall time in seconds of a run: Smilei's between the `# start:` and `# end:` lines of the run's
container.txt; PIConGPU's between the `# run <dir> seed <seed> start:`/`end:` lines of the top
directory's container.txt (the compile is timed separately). Nothing when not recorded.
"""
function wall_seconds(dir)
    if code_of(dir) == "picongpu"
        top = picongpu_top(dir)
        d = abspath(dir) == abspath(top) ? "." : basename(abspath(dir))
        f = joinpath(top, "container.txt")
        isfile(f) || return nothing
        k = findfirst(l -> startswith(l, "# run $d seed "), readlines(f))
        k === nothing && return nothing
        pre = match(r"^(# run \S+ seed \d+) start:", readlines(f)[k]).captures[1]
        return stamp_interval(f, pre * " start:", pre * " end:")
    end
    return stamp_interval(joinpath(dir, "container.txt"), "# start:", "# end:")
end

"The step count Smilei ran (`n_time` in smilei.log), or nothing."
function smilei_steps(dir)
    f = joinpath(dir, "smilei.log")
    isfile(f) || return nothing
    for l in eachline(f)
        m = match(r"n_time = (\d+)", l)
        m === nothing || return parse(Int, m[1])
    end
    return nothing
end

"""
    run_size(case; steps) -> Dict

Particles and particle-steps of a schema-2 case: the particles of every species
(cells × particles per cell), those of the mobile species, and the products with the step
count (`steps`, by default the case's round(t_end/dt)).
"""
function run_size(case; steps = round(Int, case["time"]["t_end"] / case["time"]["dt"]))
    cells = prod(case["grid"]["cells"])
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

function analyze_2d(dir, case; log = nothing, image = nothing)
    name = case["name"]
    code = code_label(dir)
    # the per-run observables; a seeded case's ensemble observable is evaluated by analyze_seeded
    observables = filter(p -> p.second["method"] != "ensemble_peak_envelope", case["observables"])
    sc = run_scalars(dir)
    fieldnames = unique(vcat(String[], (get(o, "fields", String[]) for o in values(observables))...))
    "Rho" in case["diagnostics"]["fields"] && push!(fieldnames, "Rho")   # for the loading checks below
    fl = run_fields(dir; names = fieldnames, timestep = case["time"]["dt"])
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
            lines!(ax, t[pos], U[pos]; color = :black, label = "$code ($key)")
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
        lim = max(maximum(abs, F[:, :, j]), floatmin())
        heatmap!(ax, fl.x[fname][1], fl.x[fname][2], F[:, :, j]; colormap = :RdBu, colorrange = (-lim, lim))
        side["max_abs_at_t"][string(round(fl.time[j]; digits = 4))] = maximum(abs, F[:, :, j])
    end
    save_plot(dir, "field-snapshots", fig, side)

    # Loading checks at t = 0 (not observables): the box mean of the charge density, its largest
    # deviation from the mean, and the amplitude of its mode (the loaded perturbation of the net
    # charge; zero for opposite-phase beams).
    checks = Dict{String,Any}()
    if haskey(fl.fields, "Rho")
        ρ = fl.fields["Rho"][1:end-1, 1:end-1, 1]
        checks["rho_mean_t0"] = mean(ρ)
        checks["rho_max_deviation_t0"] = maximum(abs.(ρ .- mean(ρ)))
        checks["rho_mode_amplitude_t0"] = abs(only(mode_coefficient(fl.fields["Rho"][:, :, 1:1], mode, fl.x["Rho"], lengths)))
    end
    # Staggering checks on the dumps, which also check the reader's dual-axis positions (a dual
    # point i at (i − ½)Δ, so Ex[i+1] − Ex[i] is centred on node i): Gauss's law at t = 0,
    # max |∇·E − ρ| / max |ρ|, and the largest |∇·B| over all outputs relative to max |B|.
    hx, hy = lengths ./ Tuple(case["grid"]["cells"])
    Nx, Ny = Tuple(case["grid"]["cells"])
    if all(haskey(fl.fields, n) for n in ("Ex", "Ey", "Rho"))
        Ex, Ey, ρ = (fl.fields[n][:, :, 1] for n in ("Ex", "Ey", "Rho"))
        # The divergence at node i from the two E points around it: Smilei's dual point i sits
        # at (i − ½)Δ (points i, i + 1 around node i), PIConGPU's at (i + ½)Δ (points i − 1, i);
        # read from the positions, the periodic copy closing the seam.
        sx = round(Int, (fl.x["Ex"][1][1] - fl.x["Rho"][1][1]) / hx + 0.5)   # 0 Smilei, 1 PIConGPU
        sy = round(Int, (fl.x["Ey"][2][1] - fl.x["Rho"][2][1]) / hy + 0.5)
        P(a, i, j) = a[mod1(i, Nx), mod1(j, Ny)]
        checks["gauss_residual_t0_relative"] = maximum(abs((P(Ex, i + 1 - sx, j) - P(Ex, i - sx, j)) / hx +
            (P(Ey, i, j + 1 - sy) - P(Ey, i, j - sy)) / hy - ρ[i, j]) for i in 1:Nx, j in 1:Ny) / maximum(abs, ρ)
    end
    if all(haskey(fl.fields, n) for n in ("Bx", "By"))
        Bx, By = fl.fields["Bx"], fl.fields["By"]     # Bx: (primal x, dual y), By: (dual x, primal y)
        divB = maximum(abs((Bx[i+1, j+1, n] - Bx[i, j+1, n]) / hx + (By[i+1, j+1, n] - By[i+1, j, n]) / hy)
                       for i in 1:Nx, j in 1:Ny, n in axes(Bx, 3))
        checks["divB_max_relative"] = divB / max(maximum(abs, Bx), maximum(abs, By))
    end
    # Post-hoc sensitivity, labelled as such (not a pre-registered observable): the Ex/Ey energy
    # ratio at the first output with the box mean (k = 0 part) of each field removed. Smilei adds
    # a uniform field after its initial Poisson solve (scripts/crosscode/README.md).
    if all(haskey(fl.fields, n) for n in ("Ex", "Ey"))
        e(n) = (f = fl.fields[n][1:end-1, 1:end-1, 1]; sum(abs2, f .- mean(f)))
        checks["sensitivity_energy_ratio_t0_without_box_mean"] = e("Ex") / e("Ey")
    end

    # Energy conservation.
    Utot = sc.columns["Utot"]
    drift = (Utot .- Utot[1]) ./ Utot[1]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "(Utot - Utot(0)) / Utot(0)", title = "$name: total energy")
    lines!(ax, sc.time, drift; color = :black)
    save_plot(dir, "energy-conservation", fig, Dict("plot" => "relative change of the total energy versus time",
        "max_abs_relative_change" => maximum(abs, drift), "final_relative_change" => drift[end],
        "Utot0" => Utot[1], "Ukin0" => sc.columns["Ukin"][1], "Uelm0" => sc.columns["Uelm"][1]))

    ran = code_of(dir) == "smilei" ? smilei_steps(dir) : nothing
    size_ = run_size(case; steps = something(ran, round(Int, case["time"]["t_end"] / case["time"]["dt"])))
    size_["steps_source"] = ran === nothing ? "case (round(t_end/dt)); PIConGPU runs -s exactly that" : "smilei.log (n_time)"
    wall = wall_seconds(dir)
    performance = merge(size_, Dict{String,Any}("wall_seconds" => wall === nothing ? "not recorded" : wall))
    wall === nothing || (performance["ns_per_mobile_particle_step"] = 1e9 * wall / size_["mobile_particle_steps"])
    code_of(dir) == "picongpu" && (performance["compile_seconds"] = something(compile_seconds(dir), "not recorded"))
    result = Dict{String,Any}("case" => name, "code" => code_of(dir), "observables" => results,
        "performance" => performance, "checks" => checks)
    open(io -> JSON.print(io, result, 2), joinpath(dir, "observables.json"), "w")
    outputs = output_files(dir)
    m = isfile(joinpath(dir, "manifest.toml")) ? read_manifest(dir) : Dict{String,Any}()
    outhash = Dict(relpath(f, dir) => "sha256:" * file_hash(f) for f in outputs)
    img = something(image, image_from_header(log, joinpath(dir, "image.txt")))
    image = merge(get(m, "image", Dict{String,Any}()), Dict{String,Any}("id" => img.id, "name" => img.name))
    flat = Dict(k => Dict(kk => vv for (kk, vv) in v if vv !== nothing) for (k, v) in results)
    write_manifest(dir; extra = Dict{String,Any}("outputs" => outhash, "image" => image,
        "hardware" => hardware(dir), "performance" => performance, "checks" => checks, "observables" => flat,
        "analysis" => analysis_record()))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    dir = ARGS[1]
    log = get(ARGS, 2, nothing)
    seeded = haskey(TOML.parsefile(joinpath(dir, "case.toml"))["numerics"], "seeds")
    r = seeded ? analyze_seeded(dir; log) : analyze(dir; log)
    JSON.print(stdout, r, 2)
    println()
end
