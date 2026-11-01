# Bias of the pre-registered ensemble rule at a run's measured noise (WP-012; post hoc, a
# sensitivity, not an observable). For each seeded run given, synthetic ensembles with the
# theory's damping rate and frequency, the run's time grid, its seed count and its measured
# per-seed noise (from the seeds' scatter after fit_start) go through the same
# `ensemble_damping` with the case's rules; the mean error says how far the rule alone moves γ
# at that signal-to-noise ratio. Two amplitude models: `initial`, the run's |Ê₁(0)| (all of the
# initial perturbation in the damped mode), and `first_maximum`, the first maximum the run's fit
# used, extrapolated back to t = 0 with the theory's γ (the modal part only; the non-modal
# response of the initial perturbation has phase-mixed away by fit_start).
#
#   julia +1.13 --project=scripts scripts/crosscode/ensemble_bias.jl <seeded run dir>...
#
# Writes scripts/runs/ensemble-bias-<stamp>/ with bias.png, its sidecar bias.json (every fitted
# γ), bias-summary.json and manifest.toml. The noise model is narrowband (sinusoids with frequencies in [1, 2) ω_p, as in
# T-180), stationary in time, with no early non-modal transient in the signal; not the PIC noise
# itself. Each run gets its own random numbers.

include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "ensemble.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
using .Observables, .Ensemble, .RunManifest
using CairoMakie, JSON, TOML
using Random: Xoshiro, rand, randn
using Statistics: mean, std, median
using Dates: now, UTC, format

function narrowband_noise(rng, t, σ; K = 24)
    ωk = 1 .+ rand(rng, K)
    φ = 2π .* rand(rng, K)
    a = randn(rng, ComplexF64, K) .* (σ / sqrt(K))
    return [sum(a .* cis.(ωk .* ti .+ φ)) for ti in t]
end

function calibrate(rundir; R = 400, stream = 0, amplitude = :initial)
    o = JSON.parsefile(joinpath(rundir, "observables.json"))
    ens = o["fit_rules"]
    t = Float64.(JSON.parsefile(joinpath(rundir, "ensemble-series.json"))["t"])
    γ, ω, M = o["theory"], o["theory_frequency"], o["M"]
    A = amplitude === :initial ? sqrt(o["initial_energy"]) :
        amplitude === :first_maximum ? sqrt(o["fit_energies"][1]) * exp(γ * o["fit_times"][1]) :
        error("amplitude must be :initial or :first_maximum")
    σ2 = o["seed_scatter_post_hoc"]["noise_of_mean_after_fit_start"] * M
    signal = A .* exp.(-γ .* t) .* cos.(ω .* t)
    g, se, fails = Float64[], Float64[], 0
    for r in 1:R
        rng = Xoshiro(20261101 + 100_000 * stream + r)     # independent draws per run
        e = ensemble_damping(t, [signal .+ narrowband_noise(rng, t, sqrt(σ2)) for _ in 1:M];
                             fit_start = ens["fit_start"], floor_factor = ens["floor_factor"])
        (e.ok && length(e.used) >= ens["minimum_maxima"]) || (fails += 1; continue)
        push!(g, e.gamma)
        push!(se, e.jackknife.stderr)
    end
    ok = isfinite.(se)
    return Dict{String,Any}("run" => basename(normpath(rundir)), "code" => o["code"], "amplitude_model" => String(amplitude), "theory" => γ, "M" => M,
        "realizations" => R, "failed_fits" => fails, "initial_amplitude" => A, "noise_per_seed" => σ2,
        "mean_gamma" => mean(g), "bias" => mean(g) - γ, "bias_relative" => (mean(g) - γ) / γ,
        "bias_stderr" => std(g) / sqrt(length(g)), "sd_gamma" => std(g),
        "mean_jackknife_stderr" => mean(se[ok]), "jackknife_finite" => count(ok),
        "coverage_1_jackknife" => mean(abs.(g[ok] .- γ) .<= se[ok]),
        "measured" => o["measured"], "measured_minus_theory_over_sd" => (o["measured"] - γ) / std(g),
        "measured_minus_mean_synthetic_over_sd" => (o["measured"] - mean(g)) / std(g),
        "gammas" => g)
end

if abspath(PROGRAM_FILE) == @__FILE__
    runs = ARGS
    out = joinpath(dirname(@__DIR__), "runs", "ensemble-bias-" * format(now(UTC), "yyyymmddTHHMMSS"))
    mkpath(out)
    res = [calibrate(r; stream = 2i + (a === :first_maximum), amplitude = a)
           for (i, r) in enumerate(runs) for a in (:initial, :first_maximum)]
    open(io -> JSON.print(io, res, 2), joinpath(out, "bias.json"), "w")
    fig = Figure(size = (900, 450))
    ax = Axis(fig[1, 1]; xlabel = "γ fitted on synthetic ensembles [ω_p]", ylabel = "count",
        title = "Pre-registered ensemble rule at the measured noise ($(first(res)["realizations"]) realizations each)")
    for (i, r) in enumerate(res)
        hist!(ax, r["gammas"]; bins = 40, color = (Makie.wong_colors()[i], 0.4),
              label = "$(r["code"]), $(r["amplitude_model"]): mean $(round(r["mean_gamma"]; sigdigits = 4))")
        isodd(i) && vlines!(ax, [r["measured"]]; color = Makie.wong_colors()[i], linewidth = 3, label = "$(r["code"]) measured $(round(r["measured"]; sigdigits = 4))")
    end
    vlines!(ax, [first(res)["theory"]]; color = :black, linestyle = :dash, label = "theory")
    axislegend(ax; position = :lt)
    save(joinpath(out, "bias.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, [Dict(k => v for (k, v) in r if k != "gammas") for r in res], 2), joinpath(out, "bias-summary.json"), "w")
    root = dirname(dirname(@__DIR__))
    write_manifest(out; root,
        extra = Dict{String,Any}("inputs" => merge(
            Dict{String,Any}(relpath(joinpath(@__DIR__, f), root) => "sha256:" * file_hash(joinpath(@__DIR__, f))
                             for f in ("ensemble_bias.jl", "ensemble.jl", "analysis.jl")),
            Dict{String,Any}(basename(normpath(r)) * "/" * f => "sha256:" * file_hash(joinpath(r, f))
                             for r in runs for f in ("observables.json", "ensemble-series.json")))))
    write_manifest(out;
        extra = Dict{String,Any}("runs" => basename.(normpath.(runs)), "work_package" => "WP-012",
            "kind" => "post-hoc sensitivity", "julia" => string(VERSION)))
    for r in res
        println(r["code"], " (", r["amplitude_model"], "): bias ", round(r["bias_relative"] * 100; digits = 2), " % ± ", round(r["bias_stderr"] / r["theory"] * 100; digits = 2),
            " %, sd ", round(r["sd_gamma"]; sigdigits = 3), ", mean jk ", round(r["mean_jackknife_stderr"]; sigdigits = 3),
            ", coverage ", round(r["coverage_1_jackknife"]; digits = 2), ", fails ", r["failed_fits"],
            ", measured vs synthetic mean ", round(r["measured_minus_mean_synthetic_over_sd"]; digits = 2), " sd")
    end
    println(out)
end
