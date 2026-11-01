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
using Statistics: mean, std

const REPO = dirname(dirname(@__DIR__))

"The label of a run: code and precision, from its manifest."
function run_label(dir)
    m = read_manifest(dir)
    code = get(m, "code", "smilei")
    return code == "picongpu" ? "picongpu-" * lowercase(replace(get(m, "precision", "Float32"), "Float" => "fp")) : code * "-fp64"
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
    for r in (joinpath(REPO, "reports", "crosscode-picongpu", name), joinpath(REPO, "reports", "crosscode-smilei-1d", "stage-1b", name))
        isdir(r) && return relpath(r, REPO)
    end
    return relpath(abspath(dir), REPO)
end

ratio(x, y) = (x === nothing || y === nothing || y == 0) ? nothing : x / y

function compare_runs(a, b, out)
    mkpath(out)
    case = run_case(a)
    caseb = run_case(b)
    delete!(copy(case), "_path") == delete!(copy(caseb), "_path") || error("the runs are of different cases")
    la, lb = run_label(a), run_label(b)
    seeded = haskey(case["numerics"], "seeds")
    result = Dict{String,Any}("case" => case["name"], "reference" => la, "other" => lb,
        "runs" => Dict(la => recorded_path(a), lb => recorded_path(b)),
        "note" => "a measurement, not a threshold (A7: thresholds rest on the Smilei–PIConGPU spread)")
    if seeded
        seeds = collect(Int, case["numerics"]["seeds"])
        series(d) = (s = JSON.parsefile(joinpath(d, "ensemble-series.json")); (Float64.(s["t"]), Float64.(s["U"])))
        ra, rb = JSON.parsefile(joinpath(a, "observables.json")), JSON.parsefile(joinpath(b, "observables.json"))
        (ta, Ua), (tb, Ub) = series(a), series(b)
        e = compare_ensembles(out, case, (; code = la, t = ta, U = Ua, result = ra), (; code = lb, t = tb, U = Ub, result = rb))
        sb = rb["per_seed_at_ensemble_peaks"]
        e["difference_in_units_of_other_seed_spread"] = sb["refused"] ? nothing : ratio(e["difference"], sb["std"])
        e["seed_spread"] = Dict(la => ra["per_seed_at_ensemble_peaks"], lb => sb)
        result["ensemble"] = e
        per = Dict{String,Any}()
        va = [per_run_values(joinpath(a, "seed-$s"), case) for s in seeds]
        vb = [per_run_values(joinpath(b, "seed-$s"), case) for s in seeds]
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
