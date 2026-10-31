# ParticleMesh on the WP-003 case files (scripts/crosscode/cases/*.toml), fitted with WP-003's
# own analysis (scripts/crosscode/analysis.jl) so that the numbers compare with Smilei's, and
# an overlay with a finished Smilei run.
#   julia +1.13 --project=scripts -t 16 scripts/es1d/crosscode.jl run landau|two-stream
#   julia +1.13 --project=scripts scripts/es1d/crosscode.jl compare <pm run dir> <smilei run dir>
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))
include(joinpath(REPO, "scripts", "crosscode", "read.jl"))
using .Observables
using .SmileiRead
using TOML

const CASES = joinpath(REPO, "scripts", "crosscode", "cases")

"The ParticleMesh state of a case file (electrostatic, nonrelativistic, quadratic, Reference, Float64)."
function build(case)
    @assert case["grid"]["boundaries"] == "periodic" && case["numerics"]["shape_order"] == 2
    L = case["grid"]["length"]
    n = case["grid"]["cells"]
    dt, t_end = case["time"]["dt"], case["time"]["t_end"]
    seed = case["numerics"]["random_seed"]
    mobile = [s for s in case["species"] if s["mobile"]]
    n0 = sum(s["density"] for s in mobile)          # immobile species: the neutralizing background
    ωp = sqrt(n0)
    k = 2π / L
    pert = only(unique(s["perturbation"] for s in mobile))
    @assert pert["mode"] == 1
    α = pert["amplitude"]
    common = (; n, n0, α, dt_ωp = dt * ωp, tmax_ωp = t_end * ωp, shape = Quadratic, mode = Reference(), seed)
    if length(mobile) == 1
        s = only(mobile)
        st, p, info = landau_case(Float64; common..., ppc = s["particles_per_cell"], vth = s["thermal_velocity"],
                                  kλD = k * s["thermal_velocity"] / ωp)
    else
        v0 = maximum(s["drift"] for s in mobile)
        @assert sort([s["drift"] for s in mobile]) == [-v0, v0] && all(s -> s["thermal_velocity"] == 0, mobile)
        st, p, info = two_stream_case(Float64; common..., ppc = only(unique(s["particles_per_cell"] for s in mobile)),
                                      v0, kv0_ωp = k * v0 / ωp, vth = 0)
    end
    @assert isapprox(length(st.grid), L; rtol = 1e-12) && info.nsteps == round(Int, t_end / dt)
    return st, p, info
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
    case = TOML.parsefile(joinpath(CASES, "$name.toml"))
    st, p, info = build(case)
    dir = run_dir("pm-$name", @__FILE__; params = Dict("case" => name, "case_sha256" => bytes2hex(open(sha256, joinpath(CASES, "$name.toml")))))
    initialize!(st, p)
    wall = @elapsed run!(st, p, info.nsteps)
    every = case["diagnostics"]["scalars_every"]
    W, A = field_energy_history(st), mode_amplitude_history(st)
    idx = 1:every:info.nsteps                         # the case's scalar cadence; entry k is t = (k-1)dt
    t = (idx .- 1) .* Float64(info.dt)
    r, result = observe(case, t, W[idx], A[idx] .^ 2)
    result["particles"] = info.n * info.ppc * length(st.species)
    result["steps"] = info.nsteps
    result["wall_seconds"] = wall
    result["ns_per_particle_step"] = 1e9 * wall / (result["particles"] * info.nsteps)
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
