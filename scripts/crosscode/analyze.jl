# Observables, the standard plot set and the sidecars of a finished Smilei run.
#
#   julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/<tag> [runner-log]
#
# Reads case.toml, scalars.txt and Fields0.h5 from the run directory; writes plots/<name>.png
# with plots/<name>.json next to each, observables.json, and adds the results, the output hashes
# and the image id (from the runner's log header or image.txt) to manifest.toml.

include(joinpath(@__DIR__, "read.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
using .SmileiRead, .Observables, .RunManifest
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

"CPU model and thread count: from the run's container.txt, else from the analysis host."
function hardware(dir)
    f = joinpath(dir, "container.txt")
    lines = isfile(f) ? readlines(f) : String[]
    cpu = something(findfirst(l -> startswith(l, "# cpu model:"), lines), 0)
    thr = something(findfirst(l -> startswith(l, "# threads:"), lines), 0)
    h = Dict{String,Any}()
    thr > 0 && (h["threads"] = strip(split(lines[thr], ":"; limit = 2)[2]))
    if cpu > 0
        h["cpu"] = strip(split(lines[cpu], ":"; limit = 2)[2])
        h["cpu_source"] = "run"
    else
        m = isfile("/proc/cpuinfo") ? findfirst(l -> startswith(l, "model name"), readlines("/proc/cpuinfo")) : nothing
        h["cpu"] = m === nothing ? "unknown" : strip(split(readlines("/proc/cpuinfo")[m], ":"; limit = 2)[2])
        h["cpu_source"] = "analysis host (the run did not record it; same machine for runner runs)"
    end
    return h
end

function analyze(dir; log = nothing)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
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
        "mode" => mode, "initial" => Ek[1], "max" => maximum(Ek), "t_max" => fl.time[argmax(Ek)],
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

if abspath(PROGRAM_FILE) == @__FILE__
    dir = ARGS[1]
    r = analyze(dir; log = get(ARGS, 2, nothing))
    JSON.print(stdout, r, 2)
    println()
end
