# WP-078: the per-method cost table of reports/wp-078-reverse-mode-at-production-size.md from the
# runs of `scripts/enzyme/wp078_cost.sh` (`trapped.jl cost` for primal, Dual{1}, Dual{4} and the
# reverse gradient on the λ₀/32 trapped-beam case, nvidia-smi sampled beside each).
#
#     julia +1.13 --project=scripts scripts/enzyme/wp078_cost_table.jl <out dir> <run dir> [<run dir> ...]
#
# The first run dir gives every method's row; the others add their reverse row (the same script at
# another Julia thread count). Writes `trapped-cost.md` and `trapped-cost.json` into <out dir> and
# prints the table. Refuses: a result whose method, case (every key of `case`) or J (every timed
# run) differs from the first run dir's primal; a reverse other than C = 20 with device checkpoints;
# a dirty manifest, or commits whose difference from the first run dir's touches more than this
# script and `wp078_cost.sh` (read with git in this repository); a card with another compute process
# in nvidia-smi at the start or the end, a first nvidia-smi sample that differs between the
# processes or shows the card in use; a pool peak below the state's arrays, or a reverse whose
# phases add up to more than its run.
#
# Columns: the best compiled run of each method (the warm-up, which compiles, left out), from the
# start (`initialize!`) to J and its derivatives; the reverse's own two phases; the pool's peak live
# bytes above the baseline before the start (arrays allocated and not yet freed, including garbage
# not yet collected), sampled every 20 ms; the bytes of the state's arrays (for the reverse, of the
# device checkpoints too, each one Float32 state); nvidia-smi's peak memory.used above the idle
# card's first sample, over the whole process (the warm-up's compilation, the context, the pool's
# cache and the device malloc heap included); J's relative difference from the primal's (which
# Float32 trajectory a method's device code follows); the gradient, its largest relative difference
# from the first run dir's Dual{4} and its dJ/dτ's from the Dual{1} (Dual{1}'s J is the primal's
# bitwise on the RTX 5090, Dual{4}'s is not, so Dual{1} follows the trajectory the reverse
# differentiates).

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
"First and peak memory.used (MiB) and the first utilization (%) of an nvidia-smi CSV (`timestamp, memory.used [MiB], utilization`)."
function smi_mib(file)
    rows = [(parse(Int, m[1]), parse(Int, m[2])) for m in eachmatch(r", (\d+) MiB, (\d+) %", read(file, String))]
    isempty(rows) && error("$file: no samples")
    return first(rows)[1], maximum(first, rows), first(rows)[2]
end
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const ALLOWED = ("scripts/enzyme/wp078_cost.sh", "scripts/enzyme/wp078_cost_table.jl")
"Refuse two commits whose difference touches more than the cost scripts."
function same_code(a, b)
    a == b && return
    changed = split(readchomp(`git -C $REPO diff --name-only $a $b`))
    extra = setdiff(changed, ALLOWED)
    isempty(extra) || error("commits $a and $b differ in $(join(extra, ", "))")
end
total_mib(dir) = parse(Int, match(r"/\s*(\d+)MiB", read(joinpath(dir, "nvidia-smi-start.txt"), String))[1])
"Compute processes in an nvidia-smi listing (type C, not the desktop's G or C+G)."
compute_procs(file) = [l for l in eachline(file) if occursin(r"\s+C\s+", l)]

function rows_of(dir, methods, ref)
    # ref: the first run dir's (commit, case, idle MiB), or nothing for the first run dir itself
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
        r["method"] == method || error("$name: method $(r["method"])")
        commit === nothing && (commit = m["commit"])
        m["commit"] == commit || error("$name: commit $(m["commit"]) against $commit")
        m["dirty"] == "false" || error("$name: dirty tree")
        r["case"]["nsteps"] == 1799 && r["case"]["ppl"] == 32 && r["case"]["mode"] == "Reference" || error("$name: not the λ₀/32 case")
        if method == "reverse"
            r["checkpoint_memory"] == "device" && r["checkpoint_every"] == 20 && r["checkpoints"] == cld(r["case"]["nsteps"], 20) ||
                error("$name: not C = 20 with device checkpoints")
        end
        timed = filter(x -> !x["warmup"], r["rows"])
        isempty(timed) && error("$name: no timed run")
        best = timed[argmin([x["seconds"] for x in timed])]
        first_mib, peak_mib, first_util = smi_mib(joinpath(dir, "smi-$method.csv"))
        first_util <= 5 || error("$name: the card was in use ($first_util %) when its process started")
        if ref !== nothing
            same_code(ref.commit, commit)
            r["case"] == ref.case || error("$name: case $(r["case"]) against $(ref.case)")
            first_mib == ref.idle || error("$name: first nvidia-smi sample $first_mib MiB against $(ref.idle) MiB")
        end
        for x in timed
            abs(x["J"] - best["J"]) <= 1e-6 * abs(best["J"]) || error("$name: timed runs disagree in J")
        end
        sb = r["state_bytes"] * (method == "reverse" ? 1 + r["checkpoints"] : 1)
        row = Dict{String,Any}("method" => method, "threads" => r["threads"], "run" => basename(dir), "result" => name, "commit" => commit,
                               "seconds" => best["seconds"], "seconds_all" => [x["seconds"] for x in timed], "J" => best["J"],
                               "gradient" => best["gradient"], "pool_peak_GiB" => maximum(x["pool_peak_GiB"] for x in timed),
                               "state_GiB" => sb / 2^30, "checkpoints" => r["checkpoints"],
                               "card_above_idle_GiB" => (peak_mib - first_mib) / 1024, "card_full" => peak_mib >= total - 512,
                               "card_total_MiB" => total, "idle_MiB" => first_mib)
        haskey(best, "phases") && (row["phases"] = best["phases"])
        haskey(best, "phases") && sum(values(best["phases"])) <= best["seconds"] || method != "reverse" || error("$name: phases exceed the run")
        row["pool_peak_GiB"] >= row["state_GiB"] * 0.95 || error("$name: pool peak $(row["pool_peak_GiB"]) GiB below the state's arrays $(row["state_GiB"]) GiB")
        row["case"] = r["case"]; row["checkpoint_every"] = r["checkpoint_every"]; row["checkpoint_memory"] = r["checkpoint_memory"]
        push!(rows, row)
        ref === nothing && (ref = (; commit, case = r["case"], idle = first_mib))
    end
    return rows
end

function main(out, dirs)
    rows = rows_of(dirs[1], METHODS, nothing)
    p1 = rows[1]
    ref = (; commit = p1["commit"], case = p1["case"], idle = p1["idle_MiB"])
    for d in dirs[2:end]
        append!(rows, rows_of(d, ("reverse",), ref))
    end
    by(m) = only(r for r in rows if r["method"] == m && r["run"] == basename(dirs[1]))
    primal, dual4, dual1 = by("primal"), by("dual4"), by("dual1")
    # one case: J of every method within Float32 rounding of the primal's
    for r in rows
        abs(r["J"] - primal["J"]) <= 1e-5 * abs(primal["J"]) || error("$(r["result"]): J $(r["J"]) against the primal's $(primal["J"])")
        r["times_primal"] = r["seconds"] / primal["seconds"]
        r["times_dual4"] = r["seconds"] / dual4["seconds"]
        g = r["gradient"]
        r["J_relative_to_primal"] = (r["J"] - primal["J"]) / primal["J"]
        r["relative_difference_dual1_tau"] = isempty(g) || r["method"] == "dual1" ? nothing : abs(g[1] - dual1["gradient"][1]) / abs(dual1["gradient"][1])
        r["relative_difference_dual4"] = isempty(g) || r["method"] == "dual4" ? nothing :
            maximum(abs(g[j] - dual4["gradient"][j]) / abs(dual4["gradient"][j]) for j in eachindex(g))
    end
    io = IOBuffer()
    println(io, "| method | Julia threads | one run, compiled (s) | of which reverse phase (s) | × primal | × Dual{4} | pool peak (GiB) | state arrays (GiB) | card peak above idle (GiB) | J − J_primal, relative | dJ/dτ | dJ/dw | dJ/dfocus | dJ/dn | largest relative difference from Dual{4} | dJ/dτ's relative difference from Dual{1} |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f(x) = @sprintf("%.6g", x)
    for r in rows
        g = r["gradient"]
        gs = [j <= length(g) ? f(g[j]) : "" for j in 1:4]
        rev = haskey(r, "phases") ? @sprintf("%.1f", r["phases"]["reverse"]) : ""
        card = @sprintf("%.2f", r["card_above_idle_GiB"]) * (r["card_full"] ? " (card full)" : "")
        rd = r["relative_difference_dual4"] === nothing ? "" : @sprintf("%.1e", r["relative_difference_dual4"])
        r1 = r["relative_difference_dual1_tau"] === nothing ? "" : @sprintf("%.1e", r["relative_difference_dual1_tau"])
        jr = @sprintf("%.1e", r["J_relative_to_primal"])
        st = @sprintf("%.2f", r["state_GiB"]) * (r["method"] == "reverse" ? " (1 + $(r["checkpoints"]) checkpoints)" : "")
        label = r["method"] == "reverse" ? "reverse (C = $(r["checkpoint_every"]), $(r["checkpoint_memory"]) checkpoints)" : LABEL[r["method"]]
        @printf(io, "| %s | %d | %.2f | %s | %.2f | %.2f | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", label, r["threads"], r["seconds"], rev,
                r["times_primal"], r["times_dual4"], r["pool_peak_GiB"], st, card, jr, gs..., rd, r1)
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
