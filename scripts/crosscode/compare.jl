# Two analyzed runs of one case: two codes (Smilei and PIConGPU), or one code in two precisions.
#
#   julia +1.13 --project=scripts scripts/crosscode/compare.jl <run a> <run b> <out dir>
#
# Writes <out dir>/comparison.json and the overlay plot with its sidecar. Run a is the reference.
# For every observable: the value of each run, b − a, relative to a, and in units of each run's
# seed spread (the standard deviation of the per-seed values) and of the combined standard error.
# Seeded runs: the ensemble observable (`compare_ensembles`) and every per-run observable over the
# seeds (mean ± standard deviation per run). Single runs: each observable with its fit's standard
# error. A measurement, not a threshold (A7).

include(joinpath(@__DIR__, "analyze.jl"))
include(joinpath(@__DIR__, "smilei.jl"))
using Statistics: mean, std

const REPO = dirname(dirname(@__DIR__))

"""
The label of a run: code and precision, from its manifest; a run of the smilei-gpu image (WP-030)
also names its patch layout and, without the guide's fastmath flags, `nofastmath`; a CPU run names
its layout when it is not the generator's default.
"""
function run_label(dir)
    m = read_manifest(dir)
    code = get(m, "code", "smilei")
    code == "picongpu" && return "picongpu-" * lowercase(replace(get(m, "precision", "Float32"), "Float" => "fp"))
    if get(m, "build", "cpu") != "gpu"
        haskey(m, "patches") || return code * "-fp64"
        case = run_case(dir)
        default = get(case, "schema", 1) == 2 ? collect(SmileiDeck.patches_2d(Tuple(Int.(case["grid"]["cells"])))) : nothing
        return collect(Int, m["patches"]) == default ? code * "-fp64" : code * "-p" * join(m["patches"], "x") * "-fp64"
    end
    return "smilei-gpu-p" * join(m["patches"], "x") * (m["fastmath"] == true ? "" : "-nofastmath") * "-fp64"
end

"""
    paired_loads(a, b) -> Bool

Whether two Smilei runs load the same particles for the seeds they share (at least one): the
same patch layout (Smilei seeds patch p with random_seed + p, Patch.cpp, and creates the particles on the
host in either build). An assumption from the source; the load check (loadcheck.jl) measures it.
A CPU run prepared before WP-030 has no `patches` in its manifest; its layout is the generator's
rule then and now, `patches_2d` of the case's cells.
"""
function paired_loads(a, b)
    ma, mb = read_manifest(a), read_manifest(b)
    get(ma, "code", "smilei") == "smilei" && get(mb, "code", "smilei") == "smilei" || return false
    function patches(m, d)
        haskey(m, "patches") && return collect(Int, m["patches"])
        case = run_case(d)
        get(case, "schema", 1) == 2 || return nothing
        return collect(SmileiDeck.patches_2d(Tuple(Int.(case["grid"]["cells"]))))
    end
    pa, pb = patches(ma, a), patches(mb, b)
    return pa !== nothing && pa == pb && !isempty(intersect(ma["seeds"], mb["seeds"]))
end

"""
Per-seed differences b_s − a_s of one per-run observable over the seeds both runs measured: mean,
standard deviation and standard error of the mean of the differences, and the mean in units of
a's seed spread. Meaningful as a paired comparison only when the loads are the same.
"""
function paired_stats(va, vb, sa)
    d = [y - x for (x, y) in zip(va, vb) if x !== nothing && y !== nothing]
    n = length(d)
    n == 0 && return Dict{String,Any}("n" => 0)
    m = mean(d)
    sd = n >= 2 ? std(d) : nothing
    return Dict{String,Any}("n" => n, "differences" => d, "mean" => m, "std" => sd,
        "stderr_of_mean" => sd === nothing ? nothing : sd / sqrt(n),
        "max_abs" => maximum(abs, d), "mean_in_units_of_reference_seed_spread" => ratio(m, sa))
end

"The per-run observables of one (seed) directory as name => value (nothing when not measured)."
function per_run_values(dir, case)
    o = JSON.parsefile(joinpath(dir, "observables.json"))
    get(case, "schema", 1) == 2 && return Dict(k => v["measured"] for (k, v) in o["observables"])
    return Dict(o["observable"] => o["measured"], o["observable"] * "_from_mode" => o["measured_from_mode"])
end

stats(v) = (vals = [x for x in v if x !== nothing];
    Dict{String,Any}("values" => v, "n" => length(vals), "mean" => isempty(vals) ? nothing : mean(vals),
        "std" => length(vals) >= 2 ? std(vals) : nothing, "missing" => count(isnothing, v)))

"""
The case of a run: its case.toml, or (a report directory that kept only the manifest) the file
the manifest names, refused unless its hash is the one the manifest recorded for the run.
"""
function run_case(dir)
    f = joinpath(dir, "case.toml")
    isfile(f) && return TOML.parsefile(f)
    m = read_manifest(dir)
    src = joinpath(REPO, m["case_source"])
    "sha256:" * file_hash(src) == m["inputs"]["case.toml"] ||
        error("$dir: $(m["case_source"]) is not the case the run used")
    return TOML.parsefile(src)
end

"""
The repository path recorded for a run: its committed record under reports/ when one with the
same directory name exists (the raw run under scripts/runs/ is not committed), else its path.
"""
function recorded_path(dir)
    name = basename(abspath(dir))
    for r in (joinpath(REPO, "reports", "crosscode-picongpu", name), joinpath(REPO, "reports", "crosscode-smilei-1d", "stage-1b", name),
              joinpath(REPO, "reports", "crosscode-smilei-gpu", name))
        isdir(r) && return relpath(r, REPO)
    end
    return relpath(abspath(dir), REPO)
end

ratio(x, y) = (x === nothing || y === nothing || y == 0) ? nothing : x / y

"""
    series_difference(a, b, case) -> Dict

For two Smilei runs of a schema-2 case with their raw output: every per-run observable's
series (`observable_series`, the scalars or the double-precision field dumps) of both runs,
the largest |b − a| over the run relative to max |a|, the time it happens at, and the first
time the relative difference |b − a|/|a| exceeds 10⁻¹⁰, 10⁻⁶ and 10⁻² (nothing when it never
does). Scalars are as printed (10 significant digits). A measurement of how far two builds or
layouts drift apart, which the rate fits, taken on 10-digit scalars, can hide.
"""
function series_difference(a, b, case)
    lengths = Tuple(Float64.(case["grid"]["length"]))
    mode = Tuple(Int.(case["wavevector"]["mode"]))
    dt = case["time"]["dt"]
    observables = filter(p -> p.second["method"] != "ensemble_peak_envelope", case["observables"])
    names = unique(vcat(String[], (get(o, "fields", String[]) for o in values(observables))...))
    sa, sb = read_scalars(a), read_scalars(b)
    fa, fb = read_fields(a; names, timestep = dt), read_fields(b; names, timestep = dt)
    out = Dict{String,Any}()
    for (k, o) in observables
        o["signal"] in ("mode_projection", "scalar_sum") || continue
        ta, ua = observable_series(o, sa, fa, mode, lengths)
        tb, ub = observable_series(o, sb, fb, mode, lengths)
        ta == tb || error("$k: the two runs' series have different times")
        out[k] = merge(Dict{String,Any}("signal" => o["signal"]), series_drift(ta, ua, ub))
    end
    return out
end

"""
    series_drift(t, ua, ub) -> Dict

How far a series `ub` drifts from the reference `ua` on the times `t`: the largest |ub − ua|
relative to max |ua| and its time, the largest pointwise |ub − ua|/|ua| (where finite), and the
first times that exceeds 10⁻¹⁰, 10⁻⁶ and 10⁻².
"""
function series_drift(t, ua, ub)
    d = abs.(ub .- ua)
    rel = d ./ abs.(ua)
    first_above(x) = (i = findfirst(r -> r > x, rel); i === nothing ? nothing : t[i])
    i = argmax(d)
    return Dict{String,Any}("points" => length(t),
        "max_abs_difference_relative_to_max" => d[i] / maximum(abs, ua), "at_t" => t[i],
        "first_t_relative_above_1e-10" => first_above(1e-10), "first_t_relative_above_1e-6" => first_above(1e-6),
        "first_t_relative_above_1e-2" => first_above(1e-2),
        "max_relative_difference" => maximum(rel[isfinite.(rel)]; init = 0.0))
end

"""
    paired_values(seeds_a, va, seeds_b, vb) -> (seeds, xa, xb)

The values of two seeded runs aligned seed by seed over the seeds both ran, in a's order:
`va[i]` belongs to `seeds_a[i]`, `vb[j]` to `seeds_b[j]`.
"""
function paired_values(seeds_a, va, seeds_b, vb)
    common = [x for x in seeds_a if x in seeds_b]
    ia, ib = [findfirst(==(x), seeds_a) for x in common], [findfirst(==(x), seeds_b) for x in common]
    return common, va[ia], vb[ib]
end

"""
    paired_coefficient_difference(a, b, seeds) -> Dict

For two seeded runs with `seed-coefficients.json` (the complex mode coefficient of every seed
on the field cadence) and the same loads: per common seed, the largest |c_b − c_a| over the
run relative to max |c_a|, the time it happens at, and the first times the relative difference
exceeds 10⁻¹⁰ and 10⁻⁶; and the largest of them over the seeds. How far two builds' paired
runs drift apart, which the rate fits summarize.
"""
function paired_coefficient_difference(a, b, seeds)
    read_c(d) = (j = JSON.parsefile(joinpath(d, "seed-coefficients.json"));
                 (Dict(Int(s) => complex.(Float64.(j["re"][i]), Float64.(j["im"][i])) for (i, s) in enumerate(j["seeds"])), Float64.(j["t"])))
    (ca, ta), (cb, tb) = read_c(a), read_c(b)
    ta == tb || error("the runs' coefficients are on different times")
    per = Dict{String,Any}()
    for s in seeds
        d = abs.(cb[s] .- ca[s]) ./ maximum(abs, ca[s])
        first_above(x) = (i = findfirst(>(x), d); i === nothing ? nothing : ta[i])
        per[string(s)] = Dict{String,Any}("max_relative_to_max" => maximum(d), "at_t" => ta[argmax(d)],
            "first_t_above_1e-10" => first_above(1e-10), "first_t_above_1e-6" => first_above(1e-6))
    end
    return Dict{String,Any}("per_seed" => per, "max_over_seeds" => maximum(v["max_relative_to_max"] for v in values(per)),
        "note" => "relative to each seed's max |c| of the reference run, over the whole run")
end

function compare_runs(a, b, out)
    mkpath(out)
    case = run_case(a)
    caseb = run_case(b)
    delete!(copy(case), "_path") == delete!(copy(caseb), "_path") || error("the runs are of different cases")
    la, lb = run_label(a), run_label(b)
    la == lb && ((la, lb) = (la * " (a)", lb * " (b)"))
    seeded = haskey(case["numerics"], "seeds")
    result = Dict{String,Any}("case" => case["name"], "reference" => la, "other" => lb,
        "runs" => Dict(la => recorded_path(a), lb => recorded_path(b)),
        "note" => "a measurement, not a threshold (A7: thresholds rest on the Smilei–PIConGPU spread)")
    if seeded
        seeds_a, seeds_b = run_seeds(a, case), run_seeds(b, case)
        common = first(paired_values(seeds_a, seeds_a, seeds_b, seeds_b))
        result["seeds"] = Dict(la => seeds_a, lb => seeds_b, "common" => common)
        series(d) = (s = JSON.parsefile(joinpath(d, "ensemble-series.json")); (Float64.(s["t"]), Float64.(s["U"])))
        ra, rb = JSON.parsefile(joinpath(a, "observables.json")), JSON.parsefile(joinpath(b, "observables.json"))
        (ta, Ua), (tb, Ub) = series(a), series(b)
        e = compare_ensembles(out, case, (; code = la, t = ta, U = Ua, result = ra), (; code = lb, t = tb, U = Ub, result = rb))
        sb = rb["per_seed_at_ensemble_peaks"]
        e["difference_in_units_of_other_seed_spread"] = sb["refused"] ? nothing : ratio(e["difference"], sb["std"])
        e["seed_spread"] = Dict(la => ra["per_seed_at_ensemble_peaks"], lb => sb)
        result["ensemble"] = e
        per = Dict{String,Any}()
        va = [per_run_values(joinpath(a, "seed-$s"), case) for s in seeds_a]
        vb = [per_run_values(joinpath(b, "seed-$s"), case) for s in seeds_b]
        for k in sort(collect(keys(first(va))))
            A, B = stats([v[k] for v in va]), stats([v[k] for v in vb])
            d = A["mean"] === nothing || B["mean"] === nothing ? nothing : B["mean"] - A["mean"]
            per[k] = Dict{String,Any}(la => A, lb => B, "difference_of_means" => d,
                "difference_relative" => ratio(d, A["mean"]),
                "in_units_of_reference_seed_spread" => ratio(d, A["std"]),
                "in_units_of_other_seed_spread" => ratio(d, B["std"]),
                "in_units_of_combined_stderr_of_means" =>
                    A["std"] === nothing || B["std"] === nothing ? nothing :
                    ratio(d, sqrt(A["std"]^2 / A["n"] + B["std"]^2 / B["n"])))
        end
        result["per_run_over_seeds"] = per
        paired = paired_loads(a, b)
        result["paired_loads"] = paired
        if paired
            # over the seeds both ran, seed by seed
            _, pa_, pb_ = paired_values(seeds_a, va, seeds_b, vb)
            result["paired_per_seed"] = Dict{String,Any}(k => paired_stats([v[k] for v in pa_], [v[k] for v in pb_], per[k][la]["std"])
                                                         for k in keys(per))
            # each seed's γ at its ensemble's maxima: paired only when both ensembles are of the same seeds
            pk = JSON.parsefile(joinpath(a, "observables.json"))["per_seed_at_ensemble_peaks"]
            qk = JSON.parsefile(joinpath(b, "observables.json"))["per_seed_at_ensemble_peaks"]
            all(d -> isfile(joinpath(d, "seed-coefficients.json")), (a, b)) &&
                (result["paired_coefficient_difference"] = paired_coefficient_difference(a, b, common))
            seeds_a == seeds_b && haskey(pk, "gammas") && haskey(qk, "gammas") &&
                (result["paired_per_seed"]["per_seed_at_ensemble_peaks"] = paired_stats(pk["gammas"], qk["gammas"], get(pk, "std", nothing)))
        end
    else
        oa, ob = JSON.parsefile(joinpath(a, "observables.json")), JSON.parsefile(joinpath(b, "observables.json"))
        flat(o) = get(case, "schema", 1) == 2 ? o["observables"] :
            Dict(o["observable"] => Dict("measured" => o["measured"], "measured_stderr" => o["measured_stderr"]),
                 o["observable"] * "_from_mode" => Dict("measured" => o["measured_from_mode"], "measured_stderr" => o["measured_from_mode_stderr"]))
        fa, fb = flat(oa), flat(ob)
        per = Dict{String,Any}()
        for k in sort(collect(keys(fa)))
            x, y = fa[k]["measured"], fb[k]["measured"]
            sx, sy = get(fa[k], "measured_stderr", nothing), get(fb[k], "measured_stderr", nothing)
            d = x === nothing || y === nothing ? nothing : y - x
            per[k] = Dict{String,Any}(la => fa[k], lb => fb[k], "difference" => d, "difference_relative" => ratio(d, x),
                "in_units_of_combined_stderr" => sx === nothing || sy === nothing ? nothing : ratio(d, hypot(sx, sy)))
        end
        result["observables"] = per
        have2(d) = isfile(joinpath(d, "scalars.txt")) && isfile(joinpath(d, "Fields0.h5"))
        get(case, "schema", 1) == 2 && have2(a) && have2(b) && (result["series_difference"] = series_difference(a, b, case))
        # overlay of the fitted field-energy series, when both runs' raw output is at hand (the
        # committed report directories keep the observables and sidecars, not the dumps)
        have(d) = isfile(joinpath(d, "scalars.txt")) || isfile(joinpath(d, "fields_energy.dat"))
        if have(a) && have(b)
            sa, sb = run_scalars(a), run_scalars(b)
            fig = Figure(size = (900, 500))
            ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "Uelm_Ex", title = "$(case["name"]): $la and $lb")
            for (s, l, style) in ((sa, la, :solid), (sb, lb, :dash))
                U = s.columns["Uelm_Ex"]
                p = U .> 0
                lines!(ax, s.time[p], U[p]; linestyle = style, label = l)
            end
            axislegend(ax; position = :rb)
            save_plot(out, "field-energy-overlay", fig, merge(Dict("plot" => "Uelm_Ex of both runs versus time, log scale"), result))
        else
            result["overlay"] = "not drawn: the raw output of a run is not at hand"
        end
    end
    open(io -> JSON.print(io, result, 2), joinpath(out, "comparison.json"), "w")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || (println(stderr, "usage: compare.jl <run a> <run b> <out dir>"); exit(2))
    JSON.print(stdout, compare_runs(ARGS...), 2)
    println()
end
