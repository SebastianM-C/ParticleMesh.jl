# Seed count for a two-code comparison of a seeded ensemble observable (WP-025; a design
# calculation, not an observable). For each seeded run given, `calibrate` (ensemble_bias.jl)
# draws synthetic ensembles at the run's measured per-seed noise, for both amplitude models and
# each seed count M in the list, through the run's pre-registered rule; the standard deviation
# of the fitted γ over the realizations is the rule's spread at M. The spread of the difference
# of two codes at M seeds each is the root sum of squares of the two runs' spreads (the
# larger of the two amplitude models per run, the conservative choice). Next to it, the
# measured jackknife errors scaled as 1/√M from the run's own seed count. The seed count chosen
# is the smallest M at which the given difference is `separation` spreads of the difference by
# both estimates: the two hypotheses (no code difference, a difference of that size) are then
# `separation` spreads apart, and the midpoint rule misclassifies either with probability
# Φ(-separation/2).
#
#   julia +1.13 --project=scripts scripts/crosscode/seed_count.jl <difference> <separation> <run a> <run b>
#
# Writes scripts/runs/seed-count-<stamp>/ with seed-count.png, its sidecar seed-count.json and
# manifest.toml.

include(joinpath(@__DIR__, "ensemble_bias.jl"))
using SpecialFunctions: erfc

const SEED_COUNTS = [4, 8, 12, 16, 24, 32, 48]

"""
    seed_study(runs; counts = SEED_COUNTS, R = 400) -> Vector{Dict}

`calibrate` of each run for both amplitude models at each seed count, with independent random
streams per run, model and count.
"""
function seed_study(runs; counts = SEED_COUNTS, R = 400)
    out = Dict{String,Any}[]
    for (i, r) in enumerate(runs), (j, a) in enumerate((:initial, :first_maximum)), (k, M) in enumerate(counts)
        push!(out, calibrate(r; R, M, amplitude = a, stream = 1000 + 100i + 10j + k))
    end
    return out
end

"""
    difference_spread(res, runs, counts) -> Dict

Per seed count: each run's spread (the larger of its two amplitude models), the spread of the
difference of the two runs, and the measured jackknife errors scaled from the runs' own M.
"""
function difference_spread(res, runs, counts)
    name(r) = basename(normpath(r))
    sd(run, M) = maximum(x["sd_gamma"] for x in res if x["run"] == name(run) && x["M"] == M)
    jk(run) = (o = JSON.parsefile(joinpath(run, "observables.json")); (o["measured_stderr"], o["M"]))
    a, b = runs
    (ja, Ma), (jb, Mb) = jk(a), jk(b)
    return Dict{String,Any}(
        "M" => counts,
        "sd_a" => [sd(a, M) for M in counts], "sd_b" => [sd(b, M) for M in counts],
        "sd_difference_synthetic" => [hypot(sd(a, M), sd(b, M)) for M in counts],
        "sd_difference_measured_jackknife_scaled" => [hypot(ja * sqrt(Ma / M), jb * sqrt(Mb / M)) for M in counts],
        "runs" => [name(a), name(b)])
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 4 || (println(stderr, "usage: seed_count.jl <difference> <separation> <run a> <run b>"); exit(2))
    d, sep = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
    runs = ARGS[3:4]
    res = seed_study(runs)
    ds = difference_spread(res, runs, SEED_COUNTS)
    need(v) = (k = findfirst(s -> abs(d) / s >= sep, v); k === nothing ? nothing : SEED_COUNTS[k])
    Ms, Mj = need(ds["sd_difference_synthetic"]), need(ds["sd_difference_measured_jackknife_scaled"])
    # the M between the listed counts at which each estimate reaches the separation (spread ∝ M^(-1/2) between neighbours)
    cont(v) = (s4 = v[1] * sqrt(SEED_COUNTS[1]); (sep * s4 / abs(d))^2)
    summary = merge(ds, Dict{String,Any}("difference" => d, "separation" => sep,
        "misclassification_probability_midpoint_rule" => 0.5 * erfc(sep / 2 / sqrt(2)),
        "M_needed_synthetic" => Ms, "M_needed_measured_jackknife" => Mj,
        "M_needed_measured_jackknife_continuous" => cont(ds["sd_difference_measured_jackknife_scaled"]),
        "note" => "a design calculation from the M = 4 runs, not an observable; the synthetic noise model is narrowband and stationary (ensemble_bias.jl)",
        "calibrations" => [Dict(k => v for (k, v) in r if k != "gammas") for r in res]))
    out = joinpath(dirname(@__DIR__), "runs", "seed-count-" * format(now(UTC), "yyyymmddTHHMMSS"))
    mkpath(out)
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "seeds per code M", ylabel = "spread of γ [ω_p]",
        xticks = SEED_COUNTS, title = "Spread of the ensemble γ and of the two-code difference against the seed count")
    for (k, (key, lab, st)) in enumerate((("sd_a", "$(ds["runs"][1]), synthetic", :solid), ("sd_b", "$(ds["runs"][2]), synthetic", :solid),
            ("sd_difference_synthetic", "difference, synthetic", :solid), ("sd_difference_measured_jackknife_scaled", "difference, measured jackknife × √(M₀/M)", :dash)))
        scatterlines!(ax, SEED_COUNTS, ds[key]; linestyle = st, color = Makie.wong_colors()[k], label = lab)
    end
    hlines!(ax, [abs(d) / sep]; color = :black, linestyle = :dot, label = "|difference| / $sep = $(round(abs(d) / sep; sigdigits = 3))")
    axislegend(ax; position = :lb)
    save(joinpath(out, "seed-count.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, summary, 2), joinpath(out, "seed-count.json"), "w")
    root = dirname(dirname(@__DIR__))
    write_manifest(out; root,
        extra = Dict{String,Any}("inputs" => merge(
            Dict{String,Any}(relpath(joinpath(@__DIR__, f), root) => "sha256:" * file_hash(joinpath(@__DIR__, f))
                             for f in ("seed_count.jl", "ensemble_bias.jl", "ensemble.jl", "analysis.jl")),
            Dict{String,Any}(basename(normpath(r)) * "/" * f => "sha256:" * file_hash(joinpath(r, f))
                             for r in runs for f in ("observables.json", "ensemble-series.json")))))
    write_manifest(out; extra = Dict{String,Any}("runs" => basename.(normpath.(runs)), "work_package" => "WP-025",
        "kind" => "design calculation", "julia" => string(VERSION), "arguments" => ARGS))
    for (k, M) in enumerate(SEED_COUNTS)
        println("M = $M: sd a ", round(ds["sd_a"][k]; sigdigits = 3), ", sd b ", round(ds["sd_b"][k]; sigdigits = 3),
            ", difference synthetic ", round(ds["sd_difference_synthetic"][k]; sigdigits = 3),
            ", measured jk scaled ", round(ds["sd_difference_measured_jackknife_scaled"][k]; sigdigits = 3))
    end
    println("needed: synthetic $Ms, measured jackknife $Mj (continuous ", round(summary["M_needed_measured_jackknife_continuous"]; digits = 1), ")")
    println(out)
end
