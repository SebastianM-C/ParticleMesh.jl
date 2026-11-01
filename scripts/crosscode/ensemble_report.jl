# The ensemble observable of a seeded case, its plots and sidecars, for either code (WP-012):
# Smilei's analyze.jl and ParticleMesh's scripts/es1d/crosscode.jl call the same functions on the
# complex mode coefficients of their seeds. Include analysis.jl and ensemble.jl before this file.

module EnsembleReport

using CairoMakie
using JSON
using Statistics: mean, median, std
using ..Observables: damping_from_peaks
using ..Ensemble

export ensemble_observable, ensemble_result, write_ensemble, compare_ensembles

finite(x) = x isa Real && !isfinite(x) ? nothing : x
finite(x::AbstractVector) = finite.(x)

"The ensemble observable of a case and its per-run counterpart (the theory values live there)."
function ensemble_observable(case)
    obs = case["observables"]
    key = only(k for (k, o) in obs if o["method"] == "ensemble_peak_envelope")
    o = obs[key]
    base = obs[o["theory_from"]]
    return key, merge(o, Dict("theory" => base["theory"], "theory_frequency" => base["theory_frequency"],
                              "formula" => base["formula"]))
end

"Aggregate per-seed fits; a refusal is recorded with the failed seeds instead of numbers."
function aggregated(seeds, fits)
    a = try
        aggregate_seeds(seeds, fits)
    catch e
        e isa SeedRefused || rethrow()
        return Dict{String,Any}("refused" => true, "failed_seeds" => e.seeds, "reasons" => e.reasons,
            "succeeded" => length(seeds) - length(e.seeds),
            "gammas" => finite([f.gamma for f in fits]))
    end
    return Dict{String,Any}("refused" => false, "mean" => a.mean, "std" => finite(a.std),
        "stderr" => finite(a.stderr), "M" => a.M, "gammas" => a.gammas)
end

"""
    ensemble_result(case, t, coeffs, seeds; per_run = nothing) -> (result, e)

The pre-registered ensemble observable of `case` on the complex mode coefficients `coeffs` (one
vector per seed on the time grid `t`) and everything the report needs next to it. `per_run`
holds, if given, each seed's result of the per-run observable (`damping_from_peaks` on the
code's own field-energy series), aggregated with the same refusal rule.
"""
function ensemble_result(case, t, coeffs, seeds; per_run = nothing)
    key, obs = ensemble_observable(case)
    e = ensemble_damping(t, coeffs; fit_start = obs["fit_start"], floor_factor = obs["floor_factor"])
    nmin = obs["minimum_maxima"]
    ok = e.ok && length(e.used) >= nmin
    n = length(t)
    late = max(1, n - n ÷ 5):n
    seed_floors = [median(abs2.(c[late])) for c in coeffs]
    r = Dict{String,Any}(
        "observable" => key, "case" => case["name"], "seeds" => collect(seeds), "M" => e.M,
        "theory" => obs["theory"], "theory_frequency" => obs["theory_frequency"],
        "measured" => ok ? e.gamma : nothing,
        "measured_stderr" => ok ? finite(e.jackknife.stderr) : nothing,
        "measured_stderr_method" => "jackknife over seeds",
        "measured_fit_stderr" => ok ? finite(e.gamma_stderr) : nothing,
        "measured_frequency" => ok ? finite(e.omega) : nothing,
        "relative_difference" => ok ? (e.gamma - obs["theory"]) / obs["theory"] : nothing,
        "maxima_used" => length(e.t_fit),
        "observable_exists" => ok,
        "fit_range" => isempty(e.t_fit) ? nothing : [first(e.t_fit), last(e.t_fit)],
        "fit_times" => e.t_fit, "fit_energies" => e.U_fit,
        "ensemble_floor" => e.floor,
        "seed_floors" => seed_floors, "seed_floor_mean" => mean(seed_floors),
        "floor_ratio_seed_mean_over_ensemble" => mean(seed_floors) / e.floor,
        "initial_energy" => e.U[1],
        "jackknife_gammas" => finite(e.jackknife.gamma),
        "per_seed_at_ensemble_peaks" => aggregated(seeds, e.per_seed),
        "per_seed_single_run_rule_on_mode" => merge(aggregated(seeds, e.single_seed),
            Dict("maxima_used" => [length(s.t_fit) for s in e.single_seed])),
        "cross_seed_correlation_late" => Dict(pairs(cross_seed_correlation(coeffs, late))),
        "seed_scatter_post_hoc" => scatter_record(coeffs, t, late, e, obs),
        "fit_rules" => Dict("fit_start" => obs["fit_start"], "floor_factor" => obs["floor_factor"],
                            "minimum_maxima" => nmin, "per_seed" => obs["per_seed"], "stderr" => obs["stderr"]))
    if per_run !== nothing
        r["per_seed_single_run_rule_on_energy"] = merge(aggregated(seeds, per_run),
            Dict("maxima_used" => [length(s.t_fit) for s in per_run]))
    end
    return r, e
end

"""
The post-hoc noise diagnostics (not part of the pre-registered observable): the noise of the seed
mean from the seeds' scatter, over the floor window and after `fit_start`; the pre-registered
floor over its noise-only expectation ln 2 × noise_of_mean (≈ 1 when the window holds noise
only, larger when residual signal sits in it); the signal-to-noise ratio |⟨Ê⟩|² over the noise
of the mean at each maximum used; the correlation of disjoint seed differences.
"""
function scatter_record(coeffs, t, late, e, obs)
    M = length(coeffs)
    M >= 2 || return nothing
    after = findfirst(>=(obs["fit_start"]), t):length(t)
    sl, sa = seed_scatter(coeffs, late), seed_scatter(coeffs, after)
    m = reduce(+, coeffs) ./ M
    snr = [abs2(m[i]) / (sum(abs2(c[i] - m[i]) for c in coeffs) / (M * (M - 1))) for i in e.used]
    return Dict{String,Any}("noise_of_mean_floor_window" => sl.noise_of_mean, "noise_per_seed_floor_window" => sl.noise_per_seed,
        "noise_of_mean_after_fit_start" => sa.noise_of_mean,
        "floor_over_ln2_noise_of_mean" => e.floor / (log(2) * sl.noise_of_mean),
        "snr_at_maxima_used" => snr,
        "difference_correlation_floor_window" => Dict(pairs(sl.difference_correlation)),
        "difference_correlation_after_fit_start" => Dict(pairs(sa.difference_correlation)))
end

function save_plot(dir, name, fig, sidecar)
    save(joinpath(dir, name * ".png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, sidecar, 2), joinpath(dir, name * ".json"), "w")
end

"""
    write_ensemble(dir, case, t, coeffs, seeds; code, per_run = nothing, extra = Dict()) -> result

Write `observables.json` (the result of `ensemble_result`, merged with `extra`) and the
series `ensemble-series.json` (t and |⟨Ê⟩|²), and the ensemble plots with their JSON sidecars
into `dir/plots/`: `ensemble-energy` (|⟨Ê⟩|², the
seeds' |Ê_s|², the maxima used, the fit, the theory slope, the floor) and `per-seed-rates`
(γ of each seed at the ensemble's maxima, their mean and spread, the ensemble γ, the theory).
"""
function write_ensemble(dir, case, t, coeffs, seeds; code, per_run = nothing, extra = Dict{String,Any}())
    r, e = ensemble_result(case, t, coeffs, seeds; per_run)
    merge!(r, extra)
    r["code"] = code
    plots = joinpath(dir, "plots")
    mkpath(plots)
    name = case["name"]
    fig = Figure(size = (900, 550))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|Ê₁|² of Ex",
        title = "$name, $code: ensemble of $(e.M) seeds")
    for (i, c) in enumerate(coeffs)
        u = abs2.(c)
        p = u .> 0
        lines!(ax, t[p], u[p]; color = (:gray, 0.4), linewidth = 0.8, label = i == 1 ? "single seeds" : nothing)
    end
    p = e.U .> 0
    lines!(ax, t[p], e.U[p]; color = :black, linewidth = 2, label = "|⟨Ê₁⟩|², seed mean")
    if e.fit !== nothing
        scatter!(ax, e.t_fit, e.U_fit; color = :red, markersize = 8, label = "maxima used ($(length(e.t_fit)))")
        lines!(ax, e.t_fit, exp.(e.fit.intercept .+ e.fit.slope .* e.t_fit); color = :red,
            label = "fit γ = $(round(e.gamma; sigdigits = 4)) ± $(round(e.jackknife.stderr; sigdigits = 2))")
        lines!(ax, e.t_fit, e.U_fit[1] .* exp.(-2r["theory"] .* (e.t_fit .- e.t_fit[1])); color = :blue,
            linestyle = :dash, label = "theory γ = $(round(r["theory"]; sigdigits = 4))")
    end
    hlines!(ax, [e.floor]; color = :gray, linestyle = :dot, label = "ensemble floor")
    hlines!(ax, [case["observables"][r["observable"]]["floor_factor"] * e.floor]; color = :orange,
        linestyle = :dot, label = "floor_factor × floor")
    ylims!(ax, max(1e-4 * e.floor, minimum(e.U[p])), 3 * maximum(e.U))
    axislegend(ax; position = :rt)
    save_plot(plots, "ensemble-energy", fig, merge(Dict("plot" => "|⟨Ê₁⟩|² of the seed mean and |Ê₁|² of each seed versus time, log scale"), r))

    ps = r["per_seed_at_ensemble_peaks"]
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xlabel = "seed index", ylabel = "γ [ω_p]", title = "$name, $code: γ per seed at the ensemble's maxima")
    g = Float64[something(x, NaN) for x in ps["gammas"]]
    scatter!(ax, 1:length(g), g; color = :black, label = "per seed")
    if !ps["refused"]
        hlines!(ax, [ps["mean"]]; color = :black, label = "mean $(round(ps["mean"]; sigdigits = 4)) ± std $(round(something(ps["std"], NaN); sigdigits = 2))")
        ps["std"] === nothing || band!(ax, [0.5, length(g) + 0.5], ps["mean"] - ps["std"], ps["mean"] + ps["std"]; color = (:gray, 0.2))
    end
    r["measured"] === nothing || hlines!(ax, [r["measured"]]; color = :red, label = "ensemble γ")
    hlines!(ax, [r["theory"]]; color = :blue, linestyle = :dash, label = "theory")
    axislegend(ax; position = :rb)
    save_plot(plots, "per-seed-rates", fig, Dict("plot" => "γ of each seed at the ensemble's maxima, with mean, spread, ensemble γ and theory",
        "seeds" => r["seeds"], "per_seed_at_ensemble_peaks" => ps, "ensemble" => r["measured"],
        "ensemble_stderr" => r["measured_stderr"], "theory" => r["theory"]))

    open(io -> JSON.print(io, r, 2), joinpath(dir, "observables.json"), "w")
    open(io -> JSON.print(io, Dict("t" => t, "U" => e.U), 2), joinpath(dir, "ensemble-series.json"), "w")
    return r
end

"""
    compare_ensembles(out, case, a, b) -> sidecar

Overlay of two codes' ensemble energies (`a`, `b`: named tuples `(; code, t, U, result)`), and
the difference of their ensemble γ in units of the first code's seed-to-seed spread and of the
combined jackknife standard error. A measurement, not a threshold (A7).
"""
function compare_ensembles(out, case, a, b)
    mkpath(out)
    name = case["name"]
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|⟨Ê₁⟩|² of Ex", title = "$name: seed ensembles")
    for (x, style) in ((a, :solid), (b, :dash))
        p = x.U .> 0
        lines!(ax, x.t[p], x.U[p]; linestyle = style, label = "$(x.code), γ = $(something(x.result["measured"], "none"))")
    end
    axislegend(ax; position = :rt)
    ga, gb = a.result["measured"], b.result["measured"]
    σa = a.result["per_seed_at_ensemble_peaks"]
    spread = σa["refused"] ? nothing : σa["std"]
    se = (a.result["measured_stderr"], b.result["measured_stderr"])
    both = ga !== nothing && gb !== nothing
    d = both ? gb - ga : nothing
    sidecar = Dict{String,Any}("case" => name, "theory" => a.result["theory"],
        a.code => Dict(k => a.result[k] for k in ("measured", "measured_stderr", "maxima_used", "fit_range", "ensemble_floor", "M")),
        b.code => Dict(k => b.result[k] for k in ("measured", "measured_stderr", "maxima_used", "fit_range", "ensemble_floor", "M")),
        "difference" => d,
        "difference_relative" => both ? d / ga : nothing,
        "reference_code" => a.code,
        "difference_in_units_of_reference_seed_spread" => both && spread !== nothing ? d / spread : nothing,
        "difference_in_units_of_combined_stderr" => both && all(!isnothing, se) ? d / hypot(se...) : nothing,
        "note" => "a measurement, not a threshold (A7: thresholds rest on the Smilei–PIConGPU spread)")
    save_plot(out, "$name-ensemble-energy", fig, sidecar)
    return sidecar
end

end
