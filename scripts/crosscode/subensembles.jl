# Disjoint sub-ensembles of two seeded runs of one case (WP-025; post hoc, a check of the error
# estimates, not an observable). The seeds of each run, in the case's order, are cut into
# consecutive groups of g seeds (g = 4, 8, 16 and all of them), and the pre-registered ensemble
# rule (`ensemble_damping` with the case's fit_start and floor_factor) runs on each group. For
# each g: every group's γ, jackknife error and maxima used; the standard deviation of γ over the
# groups (the spread of a g-seed ensemble, measured), next to the mean jackknife error (its
# estimate); and the group-by-group difference between the two runs, with its standard deviation
# over the groups. Group 1 of g = 4 holds the first four seeds of the case.
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
using Statistics: mean, std

"The time grid, seeds and complex mode coefficients a run's analysis wrote."
function seed_coefficients(dir)
    s = JSON.parsefile(joinpath(dir, "seed-coefficients.json"))
    c = [complex.(Float64.(re), Float64.(im)) for (re, im) in zip(s["re"], s["im"])]
    return Float64.(s["t"]), collect(Int, s["seeds"]), c
end

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
