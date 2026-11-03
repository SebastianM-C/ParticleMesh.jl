# Disjoint sub-ensembles of two seeded runs of one case (WP-025; post hoc, a check of the error
# estimates, not an observable). The seeds of each run, in the case's order, are cut into
# consecutive groups of g seeds (g = 4, 8, 16 and all of them), and the pre-registered ensemble
# rule (`ensemble_damping` with the case's fit_start and floor_factor) runs on each group. For
# each g: every group's γ, jackknife error and maxima used; the standard deviation of γ over the
# groups (the spread of a g-seed ensemble, measured), next to the mean jackknife error (its
# estimate); and the group-by-group difference between the two runs, with its standard deviation
# over the groups. Group 1 of g = 4 holds the first four seeds of the case. Also, for all seeds,
# γ fitted on the first k maxima after fit_start (k = 3 to the longer of the two runs' rule fits),
# with the jackknife on the same samples, and the two runs' difference over the windows they share.
# And, to read a single split: random splits into groups of four, a bootstrap of each run's
# seeds, and the stop points of the jackknife's leave-one-out fits.
#
#   julia +1.13 --project=scripts scripts/crosscode/subensembles.jl <run a> <run b> <out dir>
#
# Reads seed-coefficients.json and case.toml of each run (written by analyze.jl); writes
# <out dir>/subensembles.png, its sidecar subensembles.json, and manifest.toml.

@isdefined(Observables) || include(joinpath(@__DIR__, "analysis.jl"))   # the test suite has them already
@isdefined(Ensemble) || include(joinpath(@__DIR__, "ensemble.jl"))
@isdefined(RunManifest) || include(joinpath(@__DIR__, "manifest.jl"))
using .Observables, .Ensemble, .RunManifest
using CairoMakie, JSON, TOML
using Statistics: mean, std, median, quantile
using Random: Xoshiro, randperm


"""
    group_fits(t, coeffs, g; fit_start, floor_factor, minimum_maxima) -> Vector

The ensemble rule on each consecutive group of `g` seeds (the last group dropped when
incomplete): γ (nothing when the fit takes fewer than `minimum_maxima` maxima), jackknife
error, maxima used and the seed indices.
"""
function group_fits(t, coeffs, g; fit_start, floor_factor, minimum_maxima)
    out = Dict{String,Any}[]
    for k in 1:(length(coeffs) ÷ g)
        idx = (k - 1) * g + 1:k * g
        e = ensemble_damping(t, coeffs[idx]; fit_start, floor_factor)
        ok = e.ok && length(e.used) >= minimum_maxima
        push!(out, Dict{String,Any}("group" => k, "indices" => collect(idx),
            "gamma" => ok ? e.gamma : nothing,
            "jackknife_stderr" => ok && isfinite(e.jackknife.stderr) ? e.jackknife.stderr : nothing,
            "maxima_used" => length(e.used), "fit_end" => isempty(e.t_fit) ? nothing : last(e.t_fit)))
    end
    return out
end

"""
    first_maxima_fits(t, coeffs, used) -> Vector

The ensemble γ of all seeds fitted on the first k of the maxima `used` (the full ensemble's
first maxima after fit_start), k = 3 to length(used), each with its jackknife over seeds on the same
sample indices: the rate over a fixed window instead of the rule's stop point. With k = the
number the rule took, γ is the rule's; the jackknife is not, because the rule's leave-one-out fits choose their
own maxima, and this one keeps them fixed.
"""
function first_maxima_fits(t, coeffs, used)
    M = length(coeffs)
    U = ensemble_energy(coeffs)
    map(3:length(used)) do k
        u = used[1:k]
        loo = [fit_at_peaks(t, ensemble_energy(coeffs[setdiff(1:M, j)]), u).gamma for j in 1:M]
        Dict{String,Any}("maxima" => k, "t_last" => t[u[end]], "gamma" => fit_at_peaks(t, U, u).gamma,
            "jackknife_stderr" => sqrt((M - 1) / M * sum(abs2, loo .- mean(loo))))
    end
end

"""
    resampling(t, coeffs, rules; R = 300, g = 4, rng) -> Dict

How far one split into consecutive groups can be read (post hoc). Over `R` random splits of
the seeds into groups of `g`: the ratio of the measured spread of the group γ to their mean
jackknife error, and the mean group γ minus the all-seed γ. Over `R` bootstrap resamples of
the seeds (with replacement): the all-seed rule's γ and the number of maxima it takes, with
the failed fits counted. And the maxima each leave-one-out fit of the all-seed rule takes, which
the jackknife samples. A split in which a group's jackknife is not finite is left out of the
split statistics and counted (`valid_splits`).
"""
function resampling(t, coeffs, rules; R = 300, g = 4, rng)
    M = length(coeffs)
    rule(cs) = ensemble_damping(t, cs; fit_start = rules.fit_start, floor_factor = rules.floor_factor)
    full = rule(coeffs)
    ratio, shift = Float64[], Float64[]
    for _ in 1:R
        p = randperm(rng, M)
        es = [rule(coeffs[p[(k - 1) * g + 1:k * g]]) for k in 1:(M ÷ g)]
        γ = [e.gamma for e in es]
        jk = [e.jackknife.stderr for e in es]
        all(isfinite, γ) && all(isfinite, jk) || continue
        push!(ratio, std(γ) / mean(jk))
        push!(shift, mean(γ) - full.gamma)
    end
    boot = [rule(coeffs[rand(rng, 1:M, M)]) for _ in 1:R]
    bγ = [e.gamma for e in boot if isfinite(e.gamma)]
    nmax = [length(e.used) for e in boot]
    loo = [length(damping_from_peaks(t, ensemble_energy(coeffs[setdiff(1:M, j)]); fit_start = rules.fit_start,
                                     floor_factor = rules.floor_factor).used) for j in 1:M]
    q(v) = [quantile(v, 0.05), median(v), quantile(v, 0.95)]
    count_of(v) = Dict(string(k) => count(==(k), v) for k in sort(unique(v)))
    return Dict{String,Any}("random_splits" => R, "group_size" => g, "valid_splits" => length(ratio),
        "spread_over_mean_jackknife_q05_median_q95" => q(ratio),
        "mean_group_gamma_minus_all_q05_median_q95" => q(shift),
        "bootstrap_samples" => R, "bootstrap_failed_fits" => R - length(bγ), "bootstrap_gamma_sd" => std(bγ), "bootstrap_gamma_mean" => mean(bγ),
        "bootstrap_gammas" => bγ, "bootstrap_maxima_used" => count_of(nmax),
        "jackknife_maxima_used" => count_of(loo), "rule_maxima" => length(full.used),
        "gamma" => full.gamma, "jackknife_stderr" => full.jackknife.stderr)
end

vals(v) = Float64[x for x in v if x !== nothing]
sdornothing(v) = (x = vals(v); length(x) >= 2 ? std(x) : nothing)

function subensembles(a, b, out)
    case = TOML.parsefile(joinpath(a, "case.toml"))
    caseb = TOML.parsefile(joinpath(b, "case.toml"))
    case == caseb || error("the runs are of different case files")
    o = only(v for v in values(case["observables"]) if v["method"] == "ensemble_peak_envelope")
    rules = (; fit_start = o["fit_start"], floor_factor = o["floor_factor"], minimum_maxima = o["minimum_maxima"])
    (ta, sa, ca), (tb, sb, cb) = seed_coefficients(a), seed_coefficients(b)
    sa == sb || error("the runs have different seeds")
    ta == tb || error("the runs have different dump times")
    M = length(sa)
    sizes = filter(g -> g <= M, unique([4, 8, 16, M]))
    names = (basename(normpath(a)), basename(normpath(b)))
    res = Dict{String,Any}("runs" => collect(names), "seeds" => sa, "fit_rules" => Dict(pairs(rules)),
        "theory" => case["observables"][o["theory_from"]]["theory"],
        "note" => "post hoc: disjoint consecutive groups of the case's seeds; a check of the jackknife, not an observable")
    for g in sizes
        fa, fb = group_fits(ta, ca, g; rules...), group_fits(tb, cb, g; rules...)
        d = [x["gamma"] === nothing || y["gamma"] === nothing ? nothing : y["gamma"] - x["gamma"] for (x, y) in zip(fa, fb)]
        res["g$g"] = Dict{String,Any}("groups" => length(fa),
            names[1] => Dict{String,Any}("fits" => fa, "sd_gamma" => sdornothing([f["gamma"] for f in fa]),
                "mean_gamma" => isempty(vals([f["gamma"] for f in fa])) ? nothing : mean(vals([f["gamma"] for f in fa])),
                "mean_jackknife_stderr" => isempty(vals([f["jackknife_stderr"] for f in fa])) ? nothing : mean(vals([f["jackknife_stderr"] for f in fa]))),
            names[2] => Dict{String,Any}("fits" => fb, "sd_gamma" => sdornothing([f["gamma"] for f in fb]),
                "mean_gamma" => isempty(vals([f["gamma"] for f in fb])) ? nothing : mean(vals([f["gamma"] for f in fb])),
                "mean_jackknife_stderr" => isempty(vals([f["jackknife_stderr"] for f in fb])) ? nothing : mean(vals([f["jackknife_stderr"] for f in fb]))),
            "difference_b_minus_a" => d, "sd_difference" => sdornothing(d),
            "mean_difference" => isempty(vals(d)) ? nothing : mean(vals(d)))
    end
    # the full ensembles over a common window: the first k maxima of the shorter of the two fits
    ea = ensemble_damping(ta, ca; fit_start = rules.fit_start, floor_factor = rules.floor_factor)
    eb = ensemble_damping(tb, cb; fit_start = rules.fit_start, floor_factor = rules.floor_factor)
    # each run's maxima after fit_start, as many as the longer of the two fits took (beyond its
    # own fit a run's further maxima are taken in order, whatever the floor)
    K = max(length(ea.used), length(eb.used))
    pk(t, c) = (U = ensemble_energy(c); [i for i in local_maxima(U) if t[i] >= rules.fit_start][1:K])
    pa, pb = pk(ta, ca), pk(tb, cb)
    pa[1:length(ea.used)] == ea.used && pb[1:length(eb.used)] == eb.used || error("the rule's maxima are not the first maxima")
    wa, wb = first_maxima_fits(ta, ca, pa), first_maxima_fits(tb, cb, pb)
    common = Dict{String,Any}[]
    for (x, y) in zip(wa, wb)
        ta[pa[x["maxima"]]] ≈ tb[pb[y["maxima"]]] || break   # the same maxima in both runs
        push!(common, Dict{String,Any}("maxima" => x["maxima"], "t_last" => x["t_last"],
            "difference_b_minus_a" => y["gamma"] - x["gamma"],
            "combined_jackknife" => hypot(x["jackknife_stderr"], y["jackknife_stderr"]),
            "beyond_rule" => [x["maxima"] > length(ea.used), y["maxima"] > length(eb.used)]))
    end
    # random splits, bootstrap and the jackknife's stop points (independent streams per run)
    ra = resampling(ta, ca, rules; rng = Xoshiro(20261103))
    rb = resampling(tb, cb, rules; rng = Xoshiro(20261104))
    res["resampling"] = Dict{String,Any}(names[1] => ra, names[2] => rb,
        "bootstrap_difference_sd" => hypot(ra["bootstrap_gamma_sd"], rb["bootstrap_gamma_sd"]),
        "note" => "post hoc: the bootstrap resamples each run's seeds independently, so the difference's sd is the root sum of squares")
    res["first_maxima"] = Dict{String,Any}(names[1] => wa, names[2] => wb, "common" => common,
        "rule_maxima" => [length(ea.used), length(eb.used)],
        "note" => "post hoc: all seeds, fitted on the first k maxima after fit_start, k = 3 to the longer of the two rule fits; common lists the k at which both runs' k-th maxima are the same dump, beyond_rule whether k exceeds each run's own fit")
    mkpath(out)
    fig = Figure(size = (950, 500))
    ax = Axis(fig[1, 1]; xscale = log2, xticks = sizes, xlabel = "seeds per ensemble", ylabel = "ensemble γ [ω_p]",
        title = "$(case["name"]): the ensemble rule on disjoint groups of seeds")
    for (j, (n, off)) in enumerate(zip(names, (-0.06, 0.06)))
        for g in sizes
            γ = vals([f["gamma"] for f in res["g$g"][n]["fits"]])
            scatter!(ax, fill(g * 2^off, length(γ)), γ; color = (Makie.wong_colors()[j], 0.7),
                label = g == first(sizes) ? n : nothing)
        end
    end
    hlines!(ax, [res["theory"]]; color = :black, linestyle = :dash, label = "theory $(round(res["theory"]; sigdigits = 5))")
    axislegend(ax; position = :rt)
    save(joinpath(out, "subensembles.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, merge(Dict("plot" => "ensemble γ of each disjoint group of g seeds, both runs, against g"), res), 2),
        joinpath(out, "subensembles.json"), "w")
    root = dirname(dirname(@__DIR__))
    write_manifest(out; root, extra = Dict{String,Any}("inputs" => merge(
        Dict{String,Any}(relpath(joinpath(@__DIR__, f), root) => "sha256:" * file_hash(joinpath(@__DIR__, f))
                         for f in ("subensembles.jl", "ensemble.jl", "analysis.jl")),
        Dict{String,Any}(n * "/seed-coefficients.json" => "sha256:" * file_hash(joinpath(d, "seed-coefficients.json"))
                         for (n, d) in zip(names, (a, b)))),
        "runs" => collect(names), "work_package" => "WP-025", "kind" => "post-hoc check", "julia" => string(VERSION)))
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || (println(stderr, "usage: subensembles.jl <run a> <run b> <out dir>"); exit(2))
    r = subensembles(ARGS...)
    for k in sort([k for k in keys(r) if startswith(k, "g")]; by = k -> parse(Int, k[2:end]))
        x = r[k]
        println(k, ": ", join(("$n sd $(x[n]["sd_gamma"]) mean jk $(x[n]["mean_jackknife_stderr"]) mean γ $(x[n]["mean_gamma"])" for n in r["runs"]), "; "),
            "; difference sd $(x["sd_difference"]) mean $(x["mean_difference"])")
    end
end
