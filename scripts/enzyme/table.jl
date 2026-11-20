# WP-050: the report's tables from the `result.toml` files of chain_rule.jl and cost.jl runs.
#
#     julia +1.13 --project=scripts scripts/enzyme/table.jl <run dir or result.toml>...
#
# Prints one Markdown table of the chain-rule runs and one of the cost runs (the smallest and the
# largest step count of each), in the order given; the report pastes them unchanged (reports/README.md).

using TOML, Printf

results = map(ARGS) do a
    f = isdir(a) ? joinpath(a, "result.toml") : a
    TOML.parsefile(f)
end
chain = filter(r -> haskey(r, "result"), results)
cost = filter(r -> haskey(r, "rows"), results)

ms(t) = @sprintf("%.2f", 1e3 * t)
if !isempty(chain)
    println("| Mode | Steps | T | Seed | α direction, relative | Random direction, relative | c | Verdict | Reverse (s) | Primal (s) | First call (s) |")
    println("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in chain
        c, x, s = r["case"], r["result"], r["seconds"]
        @printf("| %s | %d | %s | %d | %.1e | %.1e | %.2f | %s | %.3f | %.4f | %.0f |\n", c["mode"], c["steps"], c["eltype"], c["seed"],
                x["relative_difference"], x["random_relative_difference"], x["cancellation"], x["verdict"], s["reverse"], s["primal"],
                s["enzyme_first_call"])
    end
    println()
end
if !isempty(cost)
    println("| T | Cells | ppc | Particles | Steps | Primal (ms) | Dual{1} (ms) | Dual{4} (ms) | Reverse (ms) | Reverse / primal | Reverse / Dual{4} | Reverse per step (ms) |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cost, row in unique([first(r["rows"]), last(r["rows"])])
        c = r["case"]
        head = @sprintf("| %s | %s | %s | %d | %d | %s | %s | %s |", c["eltype"], join(c["cells"], "×"), join(c["ppc"], "×"), c["particles"],
                        row["steps"], ms(row["primal"]), ms(row["dual1"]), ms(row["dual4"]))
        if haskey(row, "reverse")
            rv = row["reverse"]
            println(head, @sprintf(" %s | %.1f | %.2f | %.3f |", ms(rv), rv / row["primal"], rv / row["dual4"], 1e3 * rv / row["steps"]))
        else
            println(head, " failed | | | |")
        end
    end
    println()
    println("First reverse call (compiles), s: ", join((@sprintf("%s %s %.0f", r["case"]["eltype"], join(r["case"]["cells"], "×"),
                                                            r["seconds"]["first_reverse_call"]) for r in cost), "; "))
end
