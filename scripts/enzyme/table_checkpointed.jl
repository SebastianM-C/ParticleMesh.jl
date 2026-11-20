# WP-080: the report's tables from checkpointed.jl's runs, printed from their result.toml.
#
#     julia +1.13 --project=scripts scripts/enzyme/table_checkpointed.jl <run dir>...
#
# A `check` run gives a row of the first table, a `cost` run the rows of the second (one per step
# count) and a row of the third (memory and compile time).
using TOML, Printf

fmt(x) = x isa Real ? @sprintf("%.1e", x) : "–"
ms(x) = x isa Real ? @sprintf("%.2f", 1000x) : "–"
mib(x) = x isa Real ? @sprintf("%.0f", x / 2^20) : "–"
ratio(a, b) = a isa Real && b isa Real ? @sprintf("%.2f", a / b) : "–"
cells(c) = join(c["cells"], "×")
mode(c) = replace(c["mode"], "()" => "")

function tables(dirs)
    runs = [(d, TOML.parsefile(joinpath(d, "result.toml"))) for d in dirs]
    checks = [(d, r) for (d, r) in runs if !haskey(r, "rows")]
    costs = [(d, r) for (d, r) in runs if haskey(r, "rows")]
    if !isempty(checks)
        println("| Dim | T | Mode | Particles | Steps | c | Checkpointed vs plain reverse | Host vs device slots | Plain rerun | Chain rule, α | Chain rule, random | Tolerance | Verdict | First call, checkpointed (s) | First call, plain (s) |")
        println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (_, r) in checks
            c, x = r["case"], r["result"]
            pd = get(x, "plain_relative_difference", nothing)
            worst = pd === nothing ? nothing : maximum(values(pd))
            println("| $(c["dim"]) | $(c["eltype"]) | $(mode(c)) | $(c["particles"]) | $(c["steps"]) | $(get(x, "plain_only", false) ? "–" : c["checkpoints"]) | ",
                    "$(fmt(worst)) | $(fmt(get(x, "host_vs_device", nothing))) | $(fmt(get(x, "plain_rerun", nothing))) | $(fmt(x["relative_difference"])) | ",
                    "$(fmt(x["random_relative_difference"])) | $(fmt(x["tolerance"])) | $(x["verdict"]) | ",
                    "$(something(get(x, "first_call_checkpointed", nothing) |> y -> y === nothing ? nothing : @sprintf("%.0f", y), "–")) | ",
                    "$(something(get(x, "first_call_plain", nothing) |> y -> y === nothing ? nothing : @sprintf("%.0f", y), "–")) |")
        end
        println()
    end
    if !isempty(costs)
        println("| Dim | T | Mode | Cells | Particles | c | Steps | Primal (ms) | Dual{1} (ms) | Dual{4} (ms) | Plain reverse (ms) | Checkpointed (ms) | Host slots (ms) | Checkpointed / Dual{4} | Checkpointed / plain reverse | Checkpointed per step (ms) | Peak: Dual{4}, plain reverse, checkpointed (MiB) |")
        println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (_, r) in costs, row in r["rows"]
            c = r["case"]
            g(k) = get(row, k, nothing)
            ck = g("checkpointed")
            per = ck isa Real ? @sprintf("%.2f", 1000ck / row["steps"]) : "–"
            println("| $(c["dim"]) | $(c["eltype"]) | $(mode(c)) | $(cells(c)) | $(c["particles"]) | $(c["checkpoints"]) | $(row["steps"]) | $(ms(g("primal"))) | $(ms(g("dual1"))) | $(ms(g("dual4"))) | ",
                    "$(ms(g("plain_reverse"))) | $(ms(ck)) | $(ms(g("checkpointed_host"))) | $(ratio(ck, g("dual4"))) | $(ratio(ck, g("plain_reverse"))) | $per | ",
                    "$(mib(g("dual4_peak"))), $(mib(g("plain_reverse_peak"))), $(mib(g("checkpointed_peak"))) |")
        end
        println()
        println("| Dim | T | Mode | Cells | Particles | State (MiB) | One checkpoint (MiB) | Plain reverse of one step, peak (MiB) | First call, checkpointed (s) | First call, plain (s) |")
        println("|---|---|---|---|---|---|---|---|---|---|")
        for (_, r) in costs
            c, b, s = r["case"], r["bytes"], r["seconds"]
            println("| $(c["dim"]) | $(c["eltype"]) | $(mode(c)) | $(cells(c)) | $(c["particles"]) | $(mib(b["state"])) | $(mib(b["snapshot"])) | $(mib(b["plain_reverse_one_step_peak"])) | ",
                    "$(@sprintf("%.0f", s["first_call_checkpointed"])) | $(@sprintf("%.1f", s["first_call_plain"])) |")
        end
    end
end

tables(ARGS)
