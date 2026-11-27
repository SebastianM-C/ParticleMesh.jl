# WP-078: the per-method cost table of reports/wp-078-reverse-mode-at-production-size.md from the
# runs of `scripts/enzyme/wp078_cost.sh` (`trapped.jl cost` for primal, Dual{1}, Dual{4} and the
# reverse gradient on the λ₀/32 trapped-beam case, nvidia-smi sampled beside each).
#
#     julia +1.13 --project=scripts scripts/enzyme/wp078_cost_table.jl <out dir> <run dir> [<run dir> ...]
#
# The first run dir gives every method's row; the others add their reverse row (the same script at
# another Julia thread count). Writes `trapped-cost.md` and `trapped-cost.json` into <out dir> and
# prints the table. Refuses runs whose commit, case or J disagree, or whose card was not idle.
#
# Columns: the best compiled run of each method (the warm-up, which compiles, left out), from the
# start (`initialize!`) to J and its derivatives; the reverse's own two phases; the pool's peak live
# bytes above the baseline before the start (arrays allocated and not yet freed, including garbage
# not yet collected), sampled every 20 ms; the bytes of the state's arrays (for the reverse, of the
# device checkpoints too, each one Float32 state); nvidia-smi's peak memory.used above the idle
# card's first sample, over the whole process (the warm-up's compilation, the context, the pool's
# cache and the device malloc heap included); the gradient and its largest relative difference from
# the first run dir's Dual{4}.

using TOML, JSON, Printf

const METHODS = ("primal", "dual1", "dual4", "reverse")
const LABEL = Dict("primal" => "primal", "dual1" => "Dual{1}", "dual4" => "Dual{4}", "reverse" => "reverse")
const FREE = ("duration", "waist", "focus", "density")

function result_of(dir, method)
    ds = filter(d -> startswith(d, "enzyme-trapped-cost-$method-"), readdir(dir))
    length(ds) == 1 || error("$dir: $(length(ds)) result directories for $method")
    r = TOML.parsefile(joinpath(dir, ds[1], "result.toml"))
    m = TOML.parsefile(joinpath(dir, ds[1], "manifest.toml"))
    return r, m, ds[1]
end
"First and peak memory.used (MiB) of an nvidia-smi CSV (`timestamp, memory.used [MiB], utilization`)."
function smi_mib(file)
    mib = [parse(Int, m[1]) for m in eachmatch(r", (\d+) MiB", read(file, String))]
    isempty(mib) && error("$file: no samples")
    return first(mib), maximum(mib)
end
total_mib(dir) = parse(Int, match(r"/\s*(\d+)MiB", read(joinpath(dir, "nvidia-smi-start.txt"), String))[1])
"Compute processes in an nvidia-smi listing (type C, not the desktop's G or C+G)."
compute_procs(file) = [l for l in eachline(file) if occursin(r"\s+C\s+", l)]

function rows_of(dir, methods, ref)
    # the control runs' directory has no commit.txt: its manifest names the commit
    cf = joinpath(dir, "commit.txt")
    commit = isfile(cf) ? strip(read(cf, String)) : nothing
    for f in ("nvidia-smi-start.txt", "nvidia-smi-end.txt")
        isempty(compute_procs(joinpath(dir, f))) || error("$dir/$f: another compute process on the card")
    end
    total = total_mib(dir)
    rows = Dict{String,Any}[]
    for method in methods
        r, m, name = result_of(dir, method)
        commit === nothing && (commit = m["commit"])
        m["commit"] == commit || error("$name: commit $(m["commit"]) against $commit")
        m["dirty"] == "false" || error("$name: dirty tree")
        r["case"]["nsteps"] == 1799 && r["case"]["ppl"] == 32 && r["case"]["mode"] == "Reference" || error("$name: not the λ₀/32 case")
        timed = filter(x -> !x["warmup"], r["rows"])
        isempty(timed) && error("$name: no timed run")
        best = timed[argmin([x["seconds"] for x in timed])]
        first_mib, peak_mib = smi_mib(joinpath(dir, "smi-$method.csv"))
        sb = r["state_bytes"] * (method == "reverse" ? 1 + r["checkpoints"] : 1)
        row = Dict{String,Any}("method" => method, "threads" => r["threads"], "run" => basename(dir), "result" => name, "commit" => commit,
                               "seconds" => best["seconds"], "seconds_all" => [x["seconds"] for x in timed], "J" => best["J"],
                               "gradient" => best["gradient"], "pool_peak_GiB" => maximum(x["pool_peak_GiB"] for x in timed),
                               "state_GiB" => sb / 2^30, "checkpoints" => r["checkpoints"],
                               "card_above_idle_GiB" => (peak_mib - first_mib) / 1024, "card_full" => peak_mib >= total - 512,
                               "card_total_MiB" => total, "idle_MiB" => first_mib)
        haskey(best, "phases") && (row["phases"] = best["phases"])
        push!(rows, row)
    end
    return rows
end

function main(out, dirs)
    rows = rows_of(dirs[1], METHODS, nothing)
    for d in dirs[2:end]
        append!(rows, rows_of(d, ("reverse",), nothing))
    end
    by(m) = only(r for r in rows if r["method"] == m && r["run"] == basename(dirs[1]))
    primal, dual4 = by("primal"), by("dual4")
    # one case: J of every method within Float32 rounding of the primal's
    for r in rows
        abs(r["J"] - primal["J"]) <= 1e-5 * abs(primal["J"]) || error("$(r["result"]): J $(r["J"]) against the primal's $(primal["J"])")
        r["times_primal"] = r["seconds"] / primal["seconds"]
        r["times_dual4"] = r["seconds"] / dual4["seconds"]
        g = r["gradient"]
        r["relative_difference_dual4"] = isempty(g) || r["method"] == "dual4" ? nothing :
            maximum(abs(g[j] - dual4["gradient"][j]) / abs(dual4["gradient"][j]) for j in eachindex(g))
    end
    io = IOBuffer()
    println(io, "| method | Julia threads | one run, compiled (s) | of which reverse phase (s) | × primal | × Dual{4} | pool peak (GiB) | state arrays (GiB) | card peak above idle (GiB) | dJ/dτ | dJ/dw | dJ/dfocus | dJ/dn | largest relative difference from Dual{4} |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f(x) = @sprintf("%.6g", x)
    for r in rows
        g = r["gradient"]
        gs = [j <= length(g) ? f(g[j]) : "" for j in 1:4]
        rev = haskey(r, "phases") ? @sprintf("%.1f", r["phases"]["reverse"]) : ""
        card = @sprintf("%.2f", r["card_above_idle_GiB"]) * (r["card_full"] ? " (card full)" : "")
        rd = r["relative_difference_dual4"] === nothing ? "" : @sprintf("%.1e", r["relative_difference_dual4"])
        st = @sprintf("%.2f", r["state_GiB"]) * (r["method"] == "reverse" ? " (1 + $(r["checkpoints"]) checkpoints)" : "")
        @printf(io, "| %s | %d | %.2f | %s | %.2f | %.2f | %.2f | %s | %s | %s | %s | %s | %s | %s |\n", LABEL[r["method"]], r["threads"], r["seconds"], rev,
                r["times_primal"], r["times_dual4"], r["pool_peak_GiB"], st, card, gs..., rd)
    end
    table = String(take!(io))
    print(table)
    mkpath(out)
    write(joinpath(out, "trapped-cost.md"), table)
    open(io -> JSON.print(io, Dict("rows" => rows, "runs" => basename.(dirs)), 2), joinpath(out, "trapped-cost.json"), "w")
    return rows
end

length(ARGS) >= 2 || error("usage: wp078_cost_table.jl <out dir> <run dir> [<run dir> ...]")
main(ARGS[1], ARGS[2:end])
