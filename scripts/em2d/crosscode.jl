# ParticleMesh on the 2D cross-code cases (WP-019's `landau-oblique-2d` and `weibel-2d`), through
# the electromagnetic module's periodic 2D3V step (WP-023), evaluated with the pre-registered
# observables of the case files by the same fits the Smilei runs used (scripts/crosscode/analysis.jl,
# and ensemble.jl for the seeded Landau case), and compared with the committed Smilei runs.
#   julia +1.13 --project=scripts -t 32 scripts/em2d/crosscode.jl run landau-oblique-2d|weibel-2d [ppc1,ppc2 cells1,cells2]
#   julia +1.13 --project=scripts scripts/em2d/crosscode.jl compare <pm run dir>
# The optional sizes scale a case down (a smoke run; Δt keeps the case's Courant fraction). A case
# with a `seeds` list runs once per seed into seed-<seed>/, and the ensemble observable goes into
# the top directory; the per-run observables are evaluated for every seed.
include(joinpath(@__DIR__, "..", "em1d", "common.jl"))
include(joinpath(REPO, "scripts", "crosscode", "ensemble.jl"))
include(joinpath(REPO, "scripts", "crosscode", "ensemble_report.jl"))
using .Observables: damping_from_peaks, growth_from_window, filtered_gauss
using .Ensemble
using .EnsembleReport
using Statistics: mean, median, std

const CROSSCODE = joinpath(REPO, "scripts", "crosscode")
"The time of the loaded momenta (`initialize!`'s `momenta_at`): −Δt/2, the reference codes' convention, recorded in the manifest."
const MOMENTA_AT = :t_minus_half
"Where the density profile is sampled (`initialize!`'s `density_at`): at cell centres, as Smilei and PIConGPU sample it, recorded in the manifest."
const DENSITY_AT = :cell_centre
"""
The committed reference runs this script compares with: WP-019's single Smilei runs and, for the
seeded Landau case, the reference ensembles of Smilei and PIConGPU (FP64) on the same case file,
chosen by the first 8 hex of its SHA-256: WP-017's four seeds (`bc8fd6af`) or WP-025's 32
(`431f4192`). The reference ensembles' difference is recorded, not a threshold (A7).
"""
const SMILEI = Dict("landau-oblique-2d" => (single = joinpath(REPO, "reports", "crosscode-smilei-2d", "smilei-landau-oblique-2d-c8709f77"),
                                            ensembles = Dict(h => ("smilei" => joinpath(REPO, "reports", "crosscode-picongpu", "smilei-landau-oblique-2d-$h"),
                                                                   "picongpu" => joinpath(REPO, "reports", "crosscode-picongpu", "picongpu-landau-oblique-2d-fp64-$h"))
                                                             for h in ("bc8fd6af", "431f4192"))),
                    "weibel-2d" => (single = joinpath(REPO, "reports", "crosscode-smilei-2d", "smilei-weibel-2d-dca1230a"), ensembles = Dict()))

"The reference ensembles of a run's case file (`case.toml` in the run directory), by its hash; none for an unknown file."
reference_ensembles(name, pmdir) = get(SMILEI[name].ensembles, RunManifest.file_hash(joinpath(pmdir, "case.toml"))[1:8], ())

"The seeds of a case: its `seeds` list, or its single `random_seed`."
case_seeds(case) = haskey(case["numerics"], "seeds") ? Int.(case["numerics"]["seeds"]) : [Int(case["numerics"]["random_seed"])]

"""
    gauss_2d(state) -> (; gauss, divB)

Host checks between steps: the discrete Gauss law max|D⁻ₓEx + D⁻_yEy − (ρ − mean ρ)|, absolute
and relative to max|ρ| of the mobile charge (not to its deviation from the mean, which is
roundoff for the neutral beams of the Weibel case), with ρ deposited now in the state's mode, and
max|D⁻ₓBx + D⁻_yBy| (Bx at
(Node, Center), By at (Center, Node), so the forward differences meet at the cell centres)
relative to max(|Bx|, |By|), zero where B is zero.
"""
function gauss_2d(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    ρd = field(get_backend(st.ρ), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, (RHO_STAGGER, RHO_STAGGER), grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    ρ, Ex, Ey, Bx, By = Array(ρd), Array(st.fields.Ex), Array(st.fields.Ey), Array(st.fields.Bx), Array(st.fields.By)
    ρ̄ = sum(ρ[I1, I2]) / prod(n)
    dev = maximum(abs, ρ[I1, I2] .- ρ̄)
    res = maximum(abs((Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2] - (ρ[i, j] - ρ̄)) for i in I1, j in I2)
    divB = maximum(abs((Bx[i + 1, j] - Bx[i, j]) / h[1] + (By[i, j + 1] - By[i, j]) / h[2]) for i in I1, j in I2)
    Bmax = max(maximum(abs, Bx[I1, I2]), maximum(abs, By[I1, I2]))
    return (; gauss = res / maximum(abs, ρ[I1, I2]), divB = Bmax > 0 ? divB / Bmax : 0.0, gauss_absolute = res, rho_deviation = dev)
end

"""
    gauss_snapshot(st) -> (; r, ρ)

The plain Gauss residual r = D⁻ₓEx + D⁻_yEy − (ρ − mean ρ) and the charge density ρ (deposited
now in the state's mode, folded) at every interior node, as host arrays: the input of the
filtered Gauss check (`filtered_gauss`, WP-038).
"""
function gauss_snapshot(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    ρd = field(get_backend(st.ρ), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, (RHO_STAGGER, RHO_STAGGER), grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    ρ, Ex, Ey = Array(ρd), Array(st.fields.Ex), Array(st.fields.Ey)
    ρ̄ = sum(ρ[I1, I2]) / prod(n)
    r = [(Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2] - (ρ[i, j] - ρ̄) for i in I1, j in I2]
    return (; r = Float64.(r), ρ = Float64.(ρ[I1, I2]))
end

"Whether a case registers the filtered Gauss check (`signal = \"filtered_gauss\"`)."
has_filtered_gauss(case) = any(o -> o["signal"] == "filtered_gauss", values(case["observables"]))

"The problem of `case` at `seed`, optionally scaled to `ppc` and `cells` per axis (Δt at the case's Courant fraction)."
function problem(case, seed; ppc = nothing, cells = nothing)
    cells === nothing && return problem_from_case(case, Float64; seed, ppc)
    n0 = Tuple(case["grid"]["cells"])
    dt = case["time"]["dt"] * minimum(n0 ./ cells)
    return problem_from_case(case, Float64; seed, ppc, cells, dt)
end

"""
    simulate(case, seed; ppc, cells, momenta_at = MOMENTA_AT, density_at = DENSITY_AT) -> NamedTuple

Run one seed of a case in Reference mode, Float64, by default started as Smilei and PIConGPU start
(the loaded momenta at −Δt/2, the density sampled at cell centres: `MOMENTA_AT`, `DENSITY_AT`;
the keywords are `initialize!`'s, and its defaults are WP-023's start): the
field energies at the scalar cadence, the mode coefficients at the field cadence (entry k of the
histories is t = (k − 1)Δt, as Smilei's iteration k − 1), the total energy at integer times, Gauss's law and ∇·B before and after the
run, the field of the case's projection at the first and last step, and the wall time of the steps.
"""
function simulate(case, seed; ppc = nothing, cells = nothing, momenta_at = MOMENTA_AT, density_at = DENSITY_AT)
    prob = problem(case, seed; ppc, cells)
    st = EMState(prob, numerics_from_case(case; mode = Reference()))
    initialize!(st, prob; momenta_at, density_at)
    first_check = gauss_2d(st)
    snap(c) = (g = guards(st.grid); n = ncells(st.grid); Array(getfield(st.fields, c))[(g + 1):(g + n[1]), (g + 1):(g + n[2])])
    comp = case["name"] == "weibel-2d" ? :Bx : :Ex
    snap0 = snap(comp)
    # a case with the filtered Gauss check is run in chunks of the field cadence, with the residual
    # and ρ taken at t = 0 and after every chunk (the step does not depend on the chunking)
    gsnaps, gtimes = [], Float64[]
    if has_filtered_gauss(case)
        fe = case["diagnostics"]["fields_every"]
        push!(gsnaps, gauss_snapshot(st)); push!(gtimes, 0.0)
        wall = 0.0
        for k0 in 1:fe:prob.nsteps
            m = min(fe, prob.nsteps - k0 + 1)
            wall += @elapsed run!(st, params(prob), m; first = k0)
            push!(gsnaps, gauss_snapshot(st)); push!(gtimes, (k0 + m - 1) * Float64(prob.dt))
        end
    else
        wall = @elapsed run!(st, params(prob), prob.nsteps)
    end
    last_check = gauss_2d(st)
    h = histories(st)
    d = case["diagnostics"]
    iU = 1:d["scalars_every"]:prob.nsteps
    iC = 1:d["fields_every"]:prob.nsteps
    dt = Float64(prob.dt)
    cx(re, im) = complex.(re[iC], im[iC])
    return (; tU = (iU .- 1) .* dt, WEx = h.WEx[iU], WEy = h.WEy[iU], WBx = h.WBx[iU], WBy = h.WBy[iU], WBz = h.WBz[iU],
              tC = (iC .- 1) .* dt, El = cx(h.El_re, h.El_im), Bt = cx(h.Bt_re, h.Bt_im), Ex = cx(h.Ex_re, h.Ex_im),
              Ey = cx(h.Ey_re, h.Ey_im), total = total_energy_history(h), K = kinetic_energy_history(h), dt, wall, seed,
              nsteps = prob.nsteps, particles = sum(alive_count, st.species), checks = (initial = first_check, final = last_check),
              snapshots = (component = String(comp), first = snap0, last = snap(comp), length = Float64.(prob.length)),
              cells = prob.cells, ppc = first(prob.species).ppc, gauss = (; t = gtimes, snaps = gsnaps),
              passes = st.numerics.filter.passes)
end

"The series a pre-registered observable fits, from a simulation (the module's diagnostics in place of Smilei's scalars and dumps)."
function observable_series(obs, sim)
    sig = obs["signal"]
    if sig == "mode_projection"
        obs["fields"] == ["Ex", "Ey"] && obs["component"] == "parallel" && return sim.tC, abs2.(sim.El)
        obs["fields"] == ["Bx", "By"] && obs["component"] == "perpendicular" && return sim.tC, abs2.(sim.Bt)
    elseif sig == "scalar_sum"
        cols = Dict("Uelm_Ex" => sim.WEx, "Uelm_Ey" => sim.WEy, "Uelm_Bx_m" => sim.WBx, "Uelm_By_m" => sim.WBy)
        return sim.tU, sum(cols[c] for c in obs["columns"])
    elseif sig == "scalar_ratio" && obs["columns"] == ["Uelm_Ex", "Uelm_Ey"]
        return sim.tU, sim.WEx ./ sim.WEy
    elseif sig == "mode_ratio" && obs["fields"] == ["Ex", "Ey"]
        return sim.tC, abs2.(sim.Ex) ./ abs2.(sim.Ey)
    end
    error("observable $(obs["signal"]) of $(get(obs, "fields", get(obs, "columns", ""))) has no ParticleMesh series")
end

nothing_if_nan(x) = x isa Real && !isfinite(x) ? nothing : x

"The per-run observables of a case on one simulation, by the rules of `scripts/crosscode/analyze.jl` (`analyze_2d`)."
function observe(case, sim)
    observables = filter(p -> p.second["method"] != "ensemble_peak_envelope", case["observables"])
    order = sort(collect(keys(observables)); by = k -> (haskey(observables[k], "fit_of"), k))
    fits, results, series = Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}()
    for key in order
        obs = observables[key]
        if obs["signal"] == "filtered_gauss"
            results[key] = observe_filtered_gauss(obs, sim)
            continue
        end
        t, U = series[key] = observable_series(obs, sim)
        r = Dict{String,Any}("signal" => obs["signal"], "method" => obs["method"], "theory" => obs["theory"])
        if obs["method"] == "peak_envelope"
            f = fits[key] = damping_from_peaks(t, U; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
            merge!(r, Dict("measured" => nothing_if_nan(f.gamma), "measured_stderr" => nothing_if_nan(f.gamma_stderr),
                           "measured_frequency" => nothing_if_nan(f.omega), "theory_frequency" => obs["theory_frequency"],
                           "noise_floor" => f.floor, "fit_points" => length(f.t_fit),
                           "fit_range" => isempty(f.t_fit) ? nothing : [first(f.t_fit), last(f.t_fit)]))
        elseif obs["method"] == "exponential_window"
            f = fits[key] = growth_from_window(t, U; window = obs["window"])
            merge!(r, Dict("measured" => nothing_if_nan(f.gamma), "measured_stderr" => nothing_if_nan(f.gamma_stderr),
                           "fit_points" => length(f.t_fit), "fit_range" => isempty(f.t_fit) ? nothing : [first(f.t_fit), last(f.t_fit)],
                           "saturation" => Dict("t" => f.t_max, "value" => f.U_max)))
        elseif obs["method"] == "first_output"
            r["measured"] = U[1]
        elseif obs["method"] == "median_over_fit"
            idx = [argmin(abs.(t .- tp)) for tp in fits[obs["fit_of"]].t_fit]
            r["measured"] = isempty(idx) ? nothing : median(U[idx])
            r["points"] = length(idx)
        end
        haskey(obs, "theory_continuum") && (r["theory_continuum"] = obs["theory_continuum"])
        r["relative_difference"] = r["measured"] === nothing ? nothing : (r["measured"] - obs["theory"]) / obs["theory"]
        results[key] = r
    end
    return results, fits, series
end

"""
    observe_filtered_gauss(obs, sim) -> Dict

The filtered Gauss observable on a simulation's snapshots, by the rule of `analyze.jl`'s
`filtered_gauss_observable`: the deviation between the first and the last snapshot, its largest
over the snapshots, and the deviations for one pass fewer and one more.
"""
function observe_filtered_gauss(obs, sim)
    S, p = sim.gauss.snaps, sim.passes
    per = [filtered_gauss(S[1].r, S[1].ρ, S[k].r, S[k].ρ, p; floor = obs["floor"]) for k in 2:length(S) if S[k].ρ != S[1].ρ]
    g = per[end]
    tol = Float64(obs["tolerance"])
    return Dict{String,Any}("signal" => obs["signal"], "method" => obs["method"], "theory" => obs["theory"],
        "passes" => p, "measured" => g.deviation, "t" => sim.gauss.t[end],
        "measured_max_over_outputs" => maximum(x -> x.deviation, per),
        "deviation_other_pass_counts" => Dict(string(q) => v for (q, v) in g.deviation_other),
        "charge_change_scale" => g.charge_change_scale, "modes_above_floor" => g.modes,
        "tolerance" => tol, "within_tolerance" => g.deviation <= tol, "outputs" => length(S),
        "one_minus_H" => g.one_minus_H, "ratio" => g.ratio)
end

"The standard plot set of one seed (F3), PNG plus JSON sidecar in `dir/plots/`."
function standard_plots(dir, case, sim, results, fits, series)
    out = joinpath(dir, "plots")
    mkpath(out)
    name = case["name"]
    observables = case["observables"]
    for (plotname, sig) in (("mode-amplitude", "mode_projection"), ("field-energy", "scalar_sum"))
        keys_ = sort([k for (k, o) in observables if o["signal"] == sig && haskey(results, k)])
        isempty(keys_) && continue
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = sig == "mode_projection" ? "|mode $(case["wavevector"]["mode"])|² of the projected field" : "field energy",
                  title = "$name, ParticleMesh, seed $(sim.seed): $plotname")
        side = Dict{String,Any}("plot" => "$plotname versus time, log scale")
        for key in keys_
            t, U = series[key]
            obs, f, r = observables[key], fits[key], results[key]
            pos = U .> 0
            lines!(ax, t[pos], U[pos]; color = :black, label = "ParticleMesh ($key)")
            sgn = obs["method"] == "peak_envelope" ? -1 : 1
            if f.fit !== nothing
                lines!(ax, f.t_fit, exp.(f.fit.intercept .+ f.fit.slope .* f.t_fit); color = :red, linewidth = 2,
                       label = "fit γ = $(round(f.gamma; sigdigits = 4))")
                lines!(ax, f.t_fit, f.U_fit[1] .* exp.(2sgn * obs["theory"] .* (f.t_fit .- f.t_fit[1])); color = :blue,
                       linestyle = :dash, label = "theory γ = $(round(obs["theory"]; sigdigits = 4))")
                obs["method"] == "peak_envelope" && scatter!(ax, f.t_fit, f.U_fit; color = :red, markersize = 6)
            end
            hasproperty(f, :floor) && hlines!(ax, [f.floor]; color = :gray, linestyle = :dot, label = "noise floor")
            side[key] = merge(r, Dict("initial" => U[1], "max" => maximum(U), "final" => U[end]))
        end
        axislegend(ax; position = :rb)
        save_plot(out, plotname, fig, side)
    end
    for key in sort([k for (k, o) in observables if o["signal"] == "filtered_gauss" && haskey(results, k)])
        r = results[key]
        fig = Figure(size = (700, 600))
        ax = Axis(fig[1, 1]; xlabel = "1 − H(m), $(r["passes"]) pass(es)", ylabel = "|Δr̂(m)| / |Δρ̂(m)|",
                  title = "$name, ParticleMesh, seed $(sim.seed): Gauss residual change per mode, t = $(round(r["t"]; digits = 2))")
        scatter!(ax, r["one_minus_H"], r["ratio"]; color = :black, markersize = 4, label = "modes, |Δρ̂| > $(observables[key]["floor"]) max")
        lines!(ax, [0.0, 1.0], [0.0, 1.0]; color = :blue, linestyle = :dash, label = "filtered Gauss law")
        axislegend(ax; position = :lt)
        save_plot(out, "filtered-gauss", fig, Dict{String,Any}(
            "plot" => "per Fourier mode m above the floor, |Δr̂|/|Δρ̂| between the first and last snapshots against 1 − H(m) of the run's filter",
            "passes" => r["passes"], "one_minus_H" => r["one_minus_H"], "ratio" => r["ratio"], "deviation" => r["measured"],
            "deviation_other_pass_counts" => r["deviation_other_pass_counts"]))
    end
    drift = (sim.total .- sim.total[1]) ./ sim.total[1]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "(U − U(0)) / U(0)", title = "$name, ParticleMesh, seed $(sim.seed): total energy")
    lines!(ax, (0:(length(drift) - 1)) .* sim.dt, drift; color = :black)
    save_plot(out, "energy-conservation", fig, Dict("plot" => "relative change of field plus kinetic energy versus time (integer times)",
        "max_abs_relative_change" => maximum(abs, drift), "final_relative_change" => drift[end], "total0" => sim.total[1]))
    s = sim.snapshots
    fig = Figure(size = (1000, 480))
    side = Dict{String,Any}("plot" => "$(s.component)(x, y) at the first and the last step", "max_abs" => Dict{String,Any}())
    for (p, (lbl, F)) in enumerate((("t = 0", s.first), ("t = $(round((sim.nsteps - 1) * sim.dt; digits = 2))", s.last)))
        ax = Axis(fig[1, p]; xlabel = "x [c/ω₀]", ylabel = "y [c/ω₀]", title = "$(s.component), $lbl", aspect = DataAspect())
        lim = max(maximum(abs, F), floatmin())
        heatmap!(ax, range(0, s.length[1]; length = size(F, 1)), range(0, s.length[2]; length = size(F, 2)), F;
                 colormap = :RdBu, colorrange = (-lim, lim))
        side["max_abs"][lbl] = maximum(abs, F)
    end
    save_plot(out, "field-snapshots", fig, side)
end

"One seed of a case into `dir`: series, observables, checks and the standard plots."
function run_one(dir, case, seed; ppc = nothing, cells = nothing, momenta_at = MOMENTA_AT, density_at = DENSITY_AT)
    mkpath(dir)
    sim = simulate(case, seed; ppc, cells, momenta_at, density_at)
    results, fits, series = observe(case, sim)
    out = Dict{String,Any}("case" => case["name"], "code" => "particlemesh", "seed" => seed, "observables" => results,
        "checks" => Dict("gauss_relative_initial" => sim.checks.initial.gauss, "gauss_relative_final" => sim.checks.final.gauss,
                         "gauss_absolute_initial" => sim.checks.initial.gauss_absolute, "gauss_absolute_final" => sim.checks.final.gauss_absolute,
                         "rho_deviation_initial" => sim.checks.initial.rho_deviation,
                         "divB_relative_final" => sim.checks.final.divB,
                         "energy_max_abs_relative_change" => maximum(abs, (sim.total .- sim.total[1]) ./ sim.total[1]),
                         "kinetic_energy_initial" => sim.K[1]),
        "performance" => Dict("particles" => sim.particles, "steps" => sim.nsteps, "cells" => collect(sim.cells),
                              "ppc" => collect(sim.ppc), "wall_seconds" => sim.wall, "threads" => Threads.nthreads(),
                              "ns_per_particle_step" => 1e9 * sim.wall / (sim.particles * sim.nsteps)))
    open(io -> JSON.print(io, out, 2), joinpath(dir, "observables.json"), "w")
    open(io -> JSON.print(io, Dict("t" => sim.tU, "WEx" => sim.WEx, "WEy" => sim.WEy, "WBx" => sim.WBx, "WBy" => sim.WBy,
                                   "WBz" => sim.WBz, "tC" => sim.tC, "El_re" => real.(sim.El), "El_im" => imag.(sim.El),
                                   "Bt_re" => real.(sim.Bt), "Bt_im" => imag.(sim.Bt), "Ex_re" => real.(sim.Ex), "Ex_im" => imag.(sim.Ex),
                                   "Ey_re" => real.(sim.Ey), "Ey_im" => imag.(sim.Ey), "total" => sim.total, "dt" => sim.dt), 2),
         joinpath(dir, "series.json"), "w")
    standard_plots(dir, case, sim, results, fits, series)
    return sim, out
end

function run_case(name; ppc = nothing, cells = nothing)
    case = load_case(name)
    seeds = case_seeds(case)
    seeded = haskey(case["numerics"], "seeds")
    inputs = joinpath.(CROSSCODE, ["ensemble.jl", "ensemble_report.jl"])
    dir = run_dir("pm-$name", @__FILE__, name; work_package = "WP-023", extra_inputs = inputs, prefix = "em2d",
        params = Dict{String,Any}("case" => name, "seeds" => seeds, "element_type" => "Float64", "mode" => "Reference",
                                  "module" => "ParticleMeshElectromagnetic", "current" => "Esirkepov",
                                  "momenta_at" => string(MOMENTA_AT), "density_at" => string(DENSITY_AT),
                                  "scaled" => ppc === nothing && cells === nothing ? "no" : "ppc = $ppc, cells = $cells"))
    cp(joinpath(CASES, "$name.toml"), joinpath(dir, "case.toml"))
    coeffs, per_run, walls, tC, sims = Vector{ComplexF64}[], [], Float64[], nothing, nothing
    for s in seeds
        sdir = seeded ? joinpath(dir, "seed-$s") : dir
        sim, out = run_one(sdir, case, s; ppc, cells)
        tC === nothing ? (tC = sim.tC) : @assert(sim.tC == tC)
        push!(coeffs, sim.El)
        seeded && push!(per_run, damping_from_peaks(sim.tU, sim.WEx .+ sim.WEy; fit_start = case["observables"]["damping_rate_energy"]["fit_start"],
                                                    floor_factor = case["observables"]["damping_rate_energy"]["floor_factor"]))
        push!(walls, sim.wall)
        sims = (; particles = sim.particles, nsteps = sim.nsteps)
        println("pm-$name seed $s: $(round(sim.wall; digits = 1)) s, ",
                join(("$k = $(something(r["measured"], "none"))" for (k, r) in out["observables"]), ", "),
                "; Gauss $(sim.checks.final.gauss)")
        flush(stdout)
        sim = nothing
        GC.gc()
    end
    if seeded
        res = write_ensemble(dir, case, tC, coeffs, seeds; code = "particlemesh", per_run,
            extra = Dict{String,Any}("wall_seconds_per_seed" => walls, "particles_per_seed" => sims.particles,
                "steps" => sims.nsteps, "particle_steps_per_seed" => sims.particles * sims.nsteps,
                "ns_per_particle_step" => 1e9 .* walls ./ (sims.particles * sims.nsteps),
                "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
        println("pm-$name ensemble: γ = $(res["measured"]) ± $(res["measured_stderr"]) ($(res["maxima_used"]) maxima)")
    end
    println("outputs in $dir")
    return dir
end

fmt(x; d = 4) = x === nothing ? "none" : string(round(x; sigdigits = d))

"""
    compare(pmdir) -> sidecar

The ParticleMesh run against the committed reference runs of its case: every per-run observable
against WP-019's single Smilei run (seed 20261031) and, for the seeded Landau case, against each
reference ensemble of the same case file and seeds (`reference_ensembles`): the per-seed
observables (difference of the seed means in combined errors of the means) and the ensembles
(`compare_ensembles`, in units of the reference's seed spread and of the combined jackknife
error), with an overlay of the three ensembles. Writes `dir/compare/`. Measurements, not
thresholds (A7).
"""
function compare(pmdir)
    case = TOML.parsefile(joinpath(pmdir, "case.toml"))
    name = case["name"]
    ref = SMILEI[name]
    out = joinpath(pmdir, "compare")
    mkpath(out)
    seeded = haskey(case["numerics"], "seeds")
    sm1 = JSON.parsefile(joinpath(ref.single, "observables.json"))
    first_seed = first(case_seeds(case))
    pm1 = JSON.parsefile(joinpath(seeded ? joinpath(pmdir, "seed-$first_seed") : pmdir, "observables.json"))
    side = Dict{String,Any}("case" => name, "smilei_single_run" => relpath(ref.single, REPO), "particlemesh_run" => basename(normpath(pmdir)),
                            "note" => "measurements, not thresholds (A7)")
    single = Dict{String,Any}()
    for (k, o) in pm1["observables"]
        s = get(sm1["observables"], k, nothing)
        s === nothing && continue
        a, b = s["measured"], o["measured"]
        single[k] = Dict("smilei" => a, "smilei_stderr" => get(s, "measured_stderr", nothing), "particlemesh" => b,
                         "particlemesh_stderr" => get(o, "measured_stderr", nothing), "theory" => o["theory"],
                         "pm_minus_smilei_relative" => a === nothing || b === nothing ? nothing : (b - a) / a,
                         "combined_stderr_units" => a === nothing || b === nothing || get(s, "measured_stderr", nothing) === nothing ? nothing :
                             (b - a) / hypot(s["measured_stderr"], o["measured_stderr"]))
    end
    side["single_run_seed_$(first_seed)"] = single
    ensembles = reference_ensembles(name, pmdir)
    seeded && haskey(SMILEI[name].ensembles, "bc8fd6af") && isempty(ensembles) && error("compare: no reference ensembles for the case file of $pmdir")
    if seeded && !isempty(ensembles)
        seeds = case_seeds(case)
        series(d) = JSON.parsefile(joinpath(d, "ensemble-series.json"))
        pm = series(pmdir)
        B = (; code = "particlemesh", t = Float64.(pm["t"]), U = Float64.(pm["U"]), result = JSON.parsefile(joinpath(pmdir, "observables.json")))
        ok(v) = Float64[x for x in v if x !== nothing]
        st(v) = length(ok(v)) >= 2 ? (mean(ok(v)), std(ok(v)), length(ok(v))) : nothing
        fig = Figure(size = (900, 550))
        ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|⟨k̂·Ê⟩|² of mode $(case["wavevector"]["mode"]), seed mean",
                  title = "$name: $(length(seeds))-seed ensembles of three codes")
        overlay = Dict{String,Any}("plot" => "ensemble |⟨k̂·Ê⟩|² of each code versus time, log scale", "theory" => B.result["theory"])
        for (code, dir) in ensembles
            per = Dict{String,Any}()
            for k in ("damping_rate_mode", "energy_ratio_mode")
                sv = [JSON.parsefile(joinpath(dir, "seed-$s", "observables.json"))["observables"][k]["measured"] for s in seeds]
                pv = [JSON.parsefile(joinpath(pmdir, "seed-$s", "observables.json"))["observables"][k]["measured"] for s in seeds]
                a, b = st(sv), st(pv)
                per[k] = Dict(code => sv, "particlemesh" => pv, "$(code)_mean_std_n" => a, "particlemesh_mean_std_n" => b,
                              "difference_of_means_in_combined_errors" => a === nothing || b === nothing ? nothing :
                                  (b[1] - a[1]) / hypot(a[2] / sqrt(a[3]), b[2] / sqrt(b[3])))
            end
            r = series(dir)
            A = (; code, t = Float64.(r["t"]), U = Float64.(r["U"]), result = JSON.parsefile(joinpath(dir, "observables.json")))
            e = compare_ensembles(joinpath(out, "vs-$code"), case, A, B)
            e["reference_run"] = relpath(dir, REPO)
            side["$(code)_ensemble"] = e
            side["$(code)_per_seed"] = per
            p = A.U .> 0
            lines!(ax, A.t[p], A.U[p]; label = "$code, γ = $(fmt(A.result["measured"])) ± $(fmt(A.result["measured_stderr"]; d = 2))")
            overlay[code] = Dict("measured" => A.result["measured"], "measured_stderr" => A.result["measured_stderr"], "run" => relpath(dir, REPO))
        end
        p = B.U .> 0
        lines!(ax, B.t[p], B.U[p]; color = :black, linewidth = 2,
               label = "ParticleMesh, γ = $(fmt(B.result["measured"])) ± $(fmt(B.result["measured_stderr"]; d = 2))")
        overlay["particlemesh"] = Dict("measured" => B.result["measured"], "measured_stderr" => B.result["measured_stderr"])
        if B.result["fit_times"] !== nothing && !isempty(B.result["fit_times"])
            tf = Float64.(B.result["fit_times"])
            lines!(ax, tf, Float64(B.result["fit_energies"][1]) .* exp.(-2B.result["theory"] .* (tf .- tf[1])); color = :blue, linestyle = :dot,
                   label = "theory γ = $(fmt(B.result["theory"])), from ParticleMesh's first maximum")
        end
        axislegend(ax; position = :rt)
        save_plot(out, "$name-three-ensembles", fig, overlay)
    end
    if !seeded
        # The primary observable's series of ParticleMesh with Smilei's fitted rate drawn from the same start.
        pm = JSON.parsefile(joinpath(pmdir, "series.json"))
        t = Float64.(pm["tC"])
        U = abs2.(complex.(Float64.(pm["Bt_re"]), Float64.(pm["Bt_im"])))
        fig = Figure(size = (900, 500))
        ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|B·ê|² of mode $(case["wavevector"]["mode"])", title = "$name: ParticleMesh with Smilei's rate")
        lines!(ax, t[U .> 0], U[U .> 0]; color = :black, label = "ParticleMesh, γ = $(fmt(pm1["observables"]["growth_rate_mode"]["measured"]))")
        fr = pm1["observables"]["growth_rate_mode"]["fit_range"]
        if fr !== nothing
            i0 = argmin(abs.(t .- fr[1]))
            tt = t[i0:argmin(abs.(t .- fr[2]))]
            lines!(ax, tt, U[i0] .* exp.(2 * sm1["observables"]["growth_rate_mode"]["measured"] .* (tt .- t[i0])); color = :orange,
                   linestyle = :dash, linewidth = 2, label = "Smilei's fitted γ = $(fmt(sm1["observables"]["growth_rate_mode"]["measured"]))")
            lines!(ax, tt, U[i0] .* exp.(2 * case["observables"]["growth_rate_mode"]["theory"] .* (tt .- t[i0])); color = :blue,
                   linestyle = :dot, label = "theory γ = $(fmt(case["observables"]["growth_rate_mode"]["theory"]))")
        end
        axislegend(ax; position = :rb)
        save_plot(out, "$name-growth", fig, side)
    end
    open(io -> JSON.print(io, side, 2), joinpath(out, "$name-comparison.json"), "w")
    JSON.print(stdout, side, 2)
    println()
    return side
end

parse_pair(s) = Tuple(parse.(Int, split(s, ",")))

if abspath(PROGRAM_FILE) == @__FILE__()
    cmd = ARGS[1]
    if cmd == "run"
        ppc = length(ARGS) >= 3 ? parse_pair(ARGS[3]) : nothing
        cells = length(ARGS) >= 4 ? parse_pair(ARGS[4]) : nothing
        run_case(ARGS[2]; ppc, cells)
    elseif cmd == "compare"
        compare(ARGS[2])
    else
        error("usage: run <case> [ppc1,ppc2 cells1,cells2] | compare <pm run dir>")
    end
end
