# ParticleMesh on the cross-code case files (scripts/crosscode/cases/*.toml), fitted with the
# cross-code analysis (scripts/crosscode/analysis.jl, and ensemble.jl for a seeded case) so that
# the numbers compare with Smilei's, and an overlay with a finished Smilei run.
#   julia +1.13 --project=scripts -t 16 scripts/es1d/crosscode.jl run landau|two-stream|two-stream-256
#   julia +1.13 --project=scripts scripts/es1d/crosscode.jl analyze <pm run dir>     # recompute the ensemble
#   julia +1.13 --project=scripts scripts/es1d/crosscode.jl compare <pm run dir> <smilei run dir>
# A case with a `seeds` list (WP-012) runs once per seed, each seed the Philox key of the
# velocity draws, into seed-<seed>/ subdirectories; the ensemble observable goes into the top one.
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))
include(joinpath(REPO, "scripts", "crosscode", "read.jl"))
include(joinpath(REPO, "scripts", "crosscode", "smilei.jl"))
include(joinpath(REPO, "scripts", "crosscode", "ensemble.jl"))
include(joinpath(REPO, "scripts", "crosscode", "ensemble_report.jl"))
using .Observables
using .SmileiRead
using .SmileiDeck: load_case, case_seeds
using .Ensemble
using .EnsembleReport
using TOML

const CROSSCODE = joinpath(REPO, "scripts", "crosscode")
const CASES = joinpath(CROSSCODE, "cases")

"""
    build(case; seed) -> (state, params, info)

The ParticleMesh state of a case file (electrostatic, nonrelativistic, quadratic, Reference,
Float64), with `seed` as the Philox key of the velocity draws (default: the case's first seed).
"""
function build(case; seed = first(case_seeds(case)))
    @assert case["grid"]["boundaries"] == "periodic" && case["numerics"]["shape_order"] == 2
    L = case["grid"]["length"]
    n = case["grid"]["cells"]
    dt, t_end = case["time"]["dt"], case["time"]["t_end"]
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

"""
    simulate(case; seed, diagnostics_every = 0) -> NamedTuple

Build and run the case: the field energy at the case's scalar cadence (`tU`, `U`), the complex
mode-1 coefficient of E at its field cadence (`tC`, `C`; entry k of the histories is t = (k-1)dt),
the wall time, and, with `diagnostics_every > 0`, the host diagnostics (kinetic and field energy)
every that many steps.
"""
function simulate(case; seed = first(case_seeds(case)), diagnostics_every = 0)
    st, p, info = build(case; seed)
    initialize!(st, p)
    ts, diags = Float64[], NamedTuple[]
    wall = @elapsed if diagnostics_every > 0
        ts, diags = run_with_diagnostics!(st, p, info.nsteps; every = diagnostics_every)
    else
        run!(st, p, info.nsteps)
    end
    d = case["diagnostics"]
    iU = 1:d["scalars_every"]:info.nsteps
    iC = 1:d["fields_every"]:info.nsteps
    W, Ere, Eim = field_energy_history(st), Array(st.Ere), Array(st.Eim)
    dt = Float64(info.dt)
    return (; tU = (iU .- 1) .* dt, U = W[iU], tC = (iC .- 1) .* dt, C = complex.(Ere[iC], Eim[iC]),
              wall, info, st, ts, diags, seed, particles = info.n * info.ppc * length(st.species))
end

"The per-run observable of a case (a seeded case also has the ensemble one)."
run_observable(case) = only((k, o) for (k, o) in case["observables"] if o["method"] != "ensemble_peak_envelope")

"""
    observe(case, tU, U, tC, Ek2) -> (r, result)

Fit as the cross-code analyze.jl does, on the field energy U (at the scalar cadence, times tU)
and on the mode amplitude squared Ek2 (at the field cadence, times tC).
"""
function observe(case, tU, U, tC, Ek2)
    length(tU) == length(U) && length(tC) == length(Ek2) || error("observe: a series and its times differ in length")
    obs_key, obs = run_observable(case)
    if obs["method"] == "peak_envelope"
        r = damping_from_peaks(tU, U; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
        rk = damping_from_peaks(tC, Ek2; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
        ck = checked_fit(rk; min_points = 3)
        extra = Dict("measured_frequency" => r.omega, "theory_frequency" => obs["theory_frequency"], "noise_floor" => r.floor)
    else
        r = growth_from_window(tU, U; window = obs["window"])
        rk = growth_from_window(tC, Ek2; window = obs["window"])
        ck = checked_fit(rk; min_points = 5)
        extra = Dict("theory_nonrelativistic" => obs["theory_nonrelativistic"], "saturation_t" => r.t_max)
    end
    finite(x) = x isa Real && !isfinite(x) ? nothing : x     # JSON has no NaN; a failed fit is `null`
    res = Dict{String,Any}(k => finite(v) for (k, v) in merge(extra, Dict("observable" => obs_key, "theory" => obs["theory"], "measured" => r.gamma,
        "measured_stderr" => r.gamma_stderr, "measured_from_mode" => ck.gamma, "measured_from_mode_stderr" => ck.gamma_stderr,
        "measured_from_mode_points" => ck.points,
        "relative_difference" => (r.gamma - obs["theory"]) / obs["theory"],
        "fit_range" => isempty(r.t_fit) ? nothing : [first(r.t_fit), last(r.t_fit)], "fit_points" => length(r.t_fit))))
    ck.usable || (res["measured_from_mode_flag"] = ck.flag)
    return r, res
end

"""
The standard plot set of one ParticleMesh run (F3), each a PNG with its JSON sidecar in
`dir/plots/`: the field energy with the case's fit, the mode-1 amplitude, the relative change of
the total energy (kinetic, at t - dt/2, plus field).
"""
function standard_plots(dir, case, sim, r, result)
    out = joinpath(dir, "plots")
    mkpath(out)
    name = case["name"]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "field energy ∫Ex²/2 dx", title = "$name, ParticleMesh, seed $(sim.seed)")
    pos = sim.U .> 0
    lines!(ax, sim.tU[pos], sim.U[pos]; color = :black, label = "ParticleMesh")
    if r.fit !== nothing
        lines!(ax, r.t_fit, exp.(r.fit.intercept .+ r.fit.slope .* r.t_fit); color = :red, label = "fit γ = $(round(r.gamma; sigdigits = 4))")
    end
    hasproperty(r, :floor) && hlines!(ax, [r.floor]; color = :gray, linestyle = :dot, label = "noise floor")
    axislegend(ax; position = :rt)
    save_plot(out, "field-energy", fig, merge(Dict("plot" => "field energy versus time, log scale"), result))

    A = abs.(sim.C)
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|Ê₁|", title = "$name, ParticleMesh: mode 1 of Ex")
    lines!(ax, sim.tC[A .> 0], A[A .> 0]; color = :black)
    save_plot(out, "mode-amplitude", fig, Dict("plot" => "amplitude of Fourier mode 1 of Ex versus time", "initial" => A[1],
        "max" => maximum(A), "final" => A[end], "gamma_from_mode" => result["measured_from_mode"]))

    if !isempty(sim.ts)
        tot = [d.kinetic + d.field for d in sim.diags]
        drift = (tot .- tot[1]) ./ tot[1]
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "(U - U(0)) / U(0)", title = "$name, ParticleMesh: total energy")
        lines!(ax, sim.ts, drift; color = :black)
        save_plot(out, "energy-conservation", fig, Dict("plot" => "relative change of kinetic (t - dt/2) plus field energy versus time",
            "max_abs_relative_change" => maximum(abs, drift), "final_relative_change" => drift[end],
            "max_gauss_residual_over_scale" => maximum(d.gauss / d.gauss_scale for d in sim.diags)))
    end
end

"One run of a case into `dir`: series, observables and the standard plots; returns (sim, r, result)."
function run_one(dir, case; seed)
    mkpath(dir)
    sim = simulate(case; seed, diagnostics_every = 40)
    r, result = observe(case, sim.tU, sim.U, sim.tC, abs2.(sim.C))
    result["seed"] = seed
    result["particles"] = sim.particles
    result["steps"] = sim.info.nsteps
    result["wall_seconds"] = sim.wall
    result["ns_per_particle_step"] = 1e9 * sim.wall / (sim.particles * sim.info.nsteps)
    open(io -> JSON.print(io, result, 2), joinpath(dir, "observables.json"), "w")
    open(io -> JSON.print(io, Dict("t" => sim.tU, "U" => sim.U, "tC" => sim.tC, "E1_re" => real.(sim.C), "E1_im" => imag.(sim.C)), 2),
         joinpath(dir, "series.json"), "w")
    standard_plots(dir, case, sim, r, result)
    return sim, r, result
end

function run_case(name)
    case = load_case(name)
    seeds = case_seeds(case)
    seeded = haskey(case["numerics"], "seeds")
    inputs = [case["_path"]; joinpath.(CROSSCODE, ["analysis.jl", "ensemble.jl", "ensemble_report.jl", "smilei.jl", "read.jl"])]
    dir = run_dir("pm-$name", @__FILE__; work_package = "WP-012", extra_inputs = inputs,
        params = Dict("case" => name, "case_sha256" => bytes2hex(open(sha256, case["_path"])), "seeds" => seeds,
                      "element_type" => "Float64", "mode" => "Reference"))
    cp(case["_path"], joinpath(dir, "case.toml"))
    if !seeded
        sim, r, result = run_one(dir, case; seed = only(seeds))
        println("pm-$name: γ = $(r.gamma) ± $(r.gamma_stderr) (theory $(result["theory"])); $(round(sim.wall, digits = 1)) s; outputs in $dir")
        return dir
    end
    coeffs, per_run, walls, tC = Vector{ComplexF64}[], [], Float64[], nothing
    sim = nothing
    for s in seeds
        sim, r, result = run_one(joinpath(dir, "seed-$s"), case; seed = s)
        tC === nothing ? (tC = sim.tC) : @assert(sim.tC == tC)
        push!(coeffs, sim.C)
        push!(per_run, r)
        push!(walls, sim.wall)
        println("pm-$name seed $s: $(round(sim.wall, digits = 1)) s, single-run γ = $(result["measured"]) ($(result["fit_points"]) maxima)")
        sim = (; sim.particles, sim.info)                 # free the state before the next seed
        GC.gc()
    end
    res = write_ensemble(dir, case, tC, coeffs, seeds; code = "particlemesh", per_run,
        extra = Dict{String,Any}("wall_seconds_per_seed" => walls, "particles_per_seed" => sim.particles,
            "steps" => sim.info.nsteps, "particle_steps_per_seed" => sim.particles * sim.info.nsteps,
            "ns_per_particle_step" => 1e9 .* walls ./ (sim.particles * sim.info.nsteps),
            "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
    println("pm-$name ensemble: γ = $(res["measured"]) ± $(res["measured_stderr"]) ($(res["maxima_used"]) maxima); outputs in $dir")
    return dir
end

"""
    reanalyze(dir)

Recompute the ensemble observable of a finished seeded ParticleMesh run from the series of its
seeds (`seed-<seed>/series.json`) with the current analysis code, keeping the run's recorded
timings; the per-seed fits are rerun on the same series.
"""
function reanalyze(dir)
    old = JSON.parsefile(joinpath(dir, "observables.json"))
    case = load_case(old["case"])
    seeds = Int.(old["seeds"])
    _, obs = run_observable(case)
    coeffs, per_run, tC = Vector{ComplexF64}[], [], nothing
    for s in seeds
        sr = JSON.parsefile(joinpath(dir, "seed-$s", "series.json"))
        t = Float64.(sr["tC"])
        tC === nothing ? (tC = t) : @assert(t == tC)
        push!(coeffs, complex.(Float64.(sr["E1_re"]), Float64.(sr["E1_im"])))
        push!(per_run, damping_from_peaks(Float64.(sr["t"]), Float64.(sr["U"]); fit_start = obs["fit_start"], floor_factor = obs["floor_factor"]))
    end
    keep = ("wall_seconds_per_seed", "particles_per_seed", "steps", "particle_steps_per_seed", "ns_per_particle_step", "threads", "cpu")
    res = write_ensemble(dir, case, tC, coeffs, seeds; code = "particlemesh", per_run,
        extra = Dict{String,Any}(k => old[k] for k in keep if haskey(old, k)))
    println("$(basename(dir)) ensemble: γ = $(res["measured"]) ± $(res["measured_stderr"]) ($(res["maxima_used"]) maxima)")
    return res
end

fmt(x) = x === nothing ? "none" : string(round(x; sigdigits = 4))

"""
    compare(pmdir, smdir)

Overlay of a ParticleMesh run and a Smilei run of the same case. For a seeded case, the two
ensembles (`compare_ensembles`, with the difference in units of Smilei's seed-to-seed spread).
Otherwise the field energy with the pre-registered fit of the case file on both; for a damping
case it adds a sensitivity table: `damping_from_peaks` with relaxed `fit_start` and
`floor_factor`, applied identically to both codes. Those are post-hoc choices, labelled as such,
not the case's observable.
"""
function compare(pmdir, smdir)
    case = TOML.parsefile(joinpath(smdir, "case.toml"))
    name = case["name"]
    out = joinpath(pmdir, "compare-smilei")
    if haskey(case["numerics"], "seeds")
        series(d) = JSON.parsefile(joinpath(d, "ensemble-series.json"))
        sm, pm = series(smdir), series(pmdir)
        a = (; code = "smilei", t = Float64.(sm["t"]), U = Float64.(sm["U"]), result = JSON.parsefile(joinpath(smdir, "observables.json")))
        b = (; code = "particlemesh", t = Float64.(pm["t"]), U = Float64.(pm["U"]), result = JSON.parsefile(joinpath(pmdir, "observables.json")))
        sidecar = compare_ensembles(out, case, a, b)
        sidecar["smilei_run"] = basename(normpath(smdir))
        sidecar["particlemesh_run"] = basename(normpath(pmdir))
        open(io -> JSON.print(io, sidecar, 2), joinpath(out, "$name-ensemble-energy.json"), "w")
        JSON.print(stdout, sidecar, 2)
        println()
        return out
    end
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
    _, obs = run_observable(case)
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
    cmd == "run" ? run_case(ARGS[2]) : cmd == "compare" ? compare(ARGS[2], ARGS[3]) :
        cmd == "analyze" ? reanalyze(ARGS[2]) : error("usage: run <case> | analyze <pm dir> | compare <pm dir> <smilei dir>")
end
