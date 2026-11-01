# ParticleMesh on the WP-003 case files (scripts/crosscode/cases/*.toml), through the
# electromagnetic module (the 1D edge cases, no transverse fields), fitted with WP-003's own
# analysis (scripts/crosscode/analysis.jl) so that the numbers compare with Smilei's, and an
# overlay with a finished Smilei run.
#   julia +1.13 --project=scripts -t 16 scripts/em1d/crosscode.jl run landau|two-stream
#   julia +1.13 --project=scripts scripts/em1d/crosscode.jl compare <pm run dir> <smilei run dir>
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "scripts", "crosscode", "read.jl"))
using .Observables: local_maxima
using .SmileiRead

"""
    build(case) -> (state, params, problem)

The module's state of a case file, initialized: `problem_from_case` and `numerics_from_case`
in `Reference()` mode, Float64 (quadratic shape, Esirkepov current, relativistic Boris).
"""
function build(case)
    @assert case["grid"]["boundaries"] == "periodic" && case["numerics"]["shape_order"] == 2
    prob = problem_from_case(case, Float64)
    st = EMState(prob, numerics_from_case(case; mode = Reference()))
    initialize!(st, prob)
    return st, params(prob), prob
end

"Fit as WP-003's analyze.jl does, on the field energy U and on the mode amplitude squared."
function observe(case, t, U, Ek2)
    obs_key, obs = only(case["observables"])
    if obs["method"] == "peak_envelope"
        r = damping_from_peaks(t, U; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
        rk = damping_from_peaks(t, Ek2; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
        extra = Dict("measured_frequency" => r.omega, "theory_frequency" => obs["theory_frequency"], "noise_floor" => r.floor)
    else
        r = growth_from_window(t, U; window = obs["window"])
        rk = growth_from_window(t, Ek2; window = obs["window"])
        extra = Dict("theory_nonrelativistic" => obs["theory_nonrelativistic"], "saturation_t" => r.t_max)
    end
    finite(x) = x isa Real && !isfinite(x) ? nothing : x     # JSON has no NaN; a failed fit is `null`
    return r, Dict{String,Any}(k => finite(v) for (k, v) in merge(extra, Dict("observable" => obs_key, "theory" => obs["theory"], "measured" => r.gamma,
        "measured_stderr" => r.gamma_stderr, "measured_from_mode" => rk.gamma, "measured_from_mode_stderr" => rk.gamma_stderr,
        "relative_difference" => (r.gamma - obs["theory"]) / obs["theory"],
        "fit_range" => isempty(r.t_fit) ? nothing : [first(r.t_fit), last(r.t_fit)], "fit_points" => length(r.t_fit))))
end

function run_case(name)
    case = load_case(name)
    st, p, prob = build(case)
    dir = run_dir("pm-$name", @__FILE__, name; params = Dict{String,Any}("case" => name), work_package = "WP-003")
    wall = @elapsed run!(st, p, prob.nsteps)
    every = case["diagnostics"]["scalars_every"]
    h = histories(st)
    W, A = h.WEx, mode_amplitude_history(h, :Ex)
    idx = 1:every:prob.nsteps                        # the case's scalar cadence; entry k is t = (k-1)dt
    t = (idx .- 1) .* prob.dt
    r, result = observe(case, t, W[idx], A[idx] .^ 2)
    result["particles"] = sum(alive_count, st.species)
    result["steps"] = prob.nsteps
    result["wall_seconds"] = wall
    result["ns_per_particle_step"] = 1e9 * wall / (result["particles"] * prob.nsteps)
    open(io -> JSON.print(io, result, 2), joinpath(dir, "observables.json"), "w")
    open(io -> JSON.print(io, Dict("t" => t, "U" => W[idx], "E1" => A[idx]), 2), joinpath(dir, "series.json"), "w")
    println("pm-$name: γ = $(r.gamma) ± $(r.gamma_stderr) (theory $(result["theory"]), rel $(result["relative_difference"])); $(round(wall, digits = 1)) s; outputs in $dir")
    return dir
end

fmt(x) = x === nothing ? "none" : string(round(x; sigdigits = 4))

"""
    compare(pmdir, smdir)

Overlay of the field energy of a ParticleMesh run and a Smilei run of the same case, with the
pre-registered fit of the case file on both. For a damping case it adds a sensitivity table:
`damping_from_peaks` with relaxed `fit_start` and `floor_factor`, applied identically to both
codes. Those are post-hoc choices, labelled as such, not the case's observable.
"""
function compare(pmdir, smdir)
    case = TOML.parsefile(joinpath(smdir, "case.toml"))
    name = case["name"]
    pm = JSON.parsefile(joinpath(pmdir, "series.json"))
    pmobs = JSON.parsefile(joinpath(pmdir, "observables.json"))
    smobs = JSON.parsefile(joinpath(smdir, "observables.json"))
    sc = read_scalars(smdir)
    tS, US = sc.time, sc.columns["Uelm_Ex"]
    tP, UP = Float64.(pm["t"]), Float64.(pm["U"])
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "field energy ∫Ex²/2 dx", title = "$name: ParticleMesh and Smilei")
    lines!(ax, tS[US .> 0], US[US .> 0]; color = :black, label = "Smilei, γ = $(fmt(smobs["measured"]))")
    lines!(ax, tP[UP .> 0], UP[UP .> 0]; color = :orange, linestyle = :dash, label = "ParticleMesh, γ = $(fmt(pmobs["measured"]))")
    _, obs = only(case["observables"])
    sensitivity = Dict{String,Any}[]
    if obs["method"] == "peak_envelope"
        for fs in (obs["fit_start"], 2.0), ff in (obs["floor_factor"], 5.0, 3.0)
            rs = damping_from_peaks(tS, US; fit_start = fs, floor_factor = ff)
            rp = damping_from_peaks(tP, UP; fit_start = fs, floor_factor = ff)
            push!(sensitivity, Dict("fit_start" => fs, "floor_factor" => ff,
                "smilei" => isfinite(rs.gamma) ? rs.gamma : nothing, "smilei_peaks" => length(rs.t_fit),
                "particlemesh" => isfinite(rp.gamma) ? rp.gamma : nothing, "particlemesh_peaks" => length(rp.t_fit),
                "smilei_frequency" => isfinite(rs.omega) ? rs.omega : nothing,
                "particlemesh_frequency" => isfinite(rp.omega) ? rp.omega : nothing))
        end
        i1 = first(local_maxima(UP))                  # anchor the theory slope at the first maximum
        lines!(ax, tP[i1:end], UP[i1] .* exp.(-2obs["theory"] .* (tP[i1:end] .- tP[i1])); color = :blue, linestyle = :dot,
               label = "theory slope γ = $(fmt(obs["theory"])), anchored at the first maximum")
        ylims!(ax, 1e-10, 2 * max(UP[1], US[1]))
    end
    axislegend(ax; position = obs["method"] == "peak_envelope" ? :rt : :rb)
    out = joinpath(pmdir, "compare-smilei")
    mkpath(out)
    pick(o) = Dict(k => get(o, k, nothing) for k in ("measured", "measured_stderr", "relative_difference", "fit_range", "fit_points"))
    both = pmobs["measured"] !== nothing && smobs["measured"] !== nothing
    sidecar = Dict("case" => name, "theory" => obs["theory"], "particlemesh" => pick(pmobs), "smilei" => pick(smobs),
        "pm_minus_smilei_relative" => both ? (pmobs["measured"] - smobs["measured"]) / smobs["measured"] : nothing,
        "sensitivity_post_hoc" => sensitivity,
        "U0" => Dict("particlemesh" => UP[1], "smilei" => US[1]),
        "smilei_run" => basename(normpath(smdir)), "particlemesh_run" => basename(normpath(pmdir)))
    save_plot(out, "$name-field-energy", fig, sidecar)
    JSON.print(stdout, sidecar, 2)
    println()
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__()
    cmd = ARGS[1]
    cmd == "run" ? run_case(ARGS[2]) : cmd == "compare" ? compare(ARGS[2], ARGS[3]) : error("usage: run <case> | compare <pm dir> <smilei dir>")
end
