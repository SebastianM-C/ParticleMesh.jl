# Trace check: julia +1.13 --project=scripts scripts/trace/build.jl [--check] [--refresh WP-nnn ...]
#
# Reads docs/goals/, the front matter of trace/wp/ and the test tags, and prints coverage,
# the stale packages and the errors. With --check, exits 1 on any error; stale packages are
# warnings only. With --refresh, first records the current version of their goals in the
# named packages (after checking them against those goals). See trace/README.md.
include(joinpath(@__DIR__, "Trace.jl"))
using .Trace

function main(args)
    root = dirname(dirname(@__DIR__))
    check_mode = false
    refresh = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--check"
            check_mode = true
        elseif a == "--refresh"
            while i < length(args) && startswith(args[i+1], "WP-")
                i += 1
                push!(refresh, args[i])
            end
            isempty(refresh) && (println(stderr, "--refresh needs package ids"); return 2)
        else
            println(stderr, "unknown argument: ", a)
            return 2
        end
        i += 1
    end
    if !isempty(refresh)
        goals, _ = load_goals(joinpath(root, "docs", "goals"))
        packages, _ = load_packages(joinpath(root, "trace", "wp"))
        for id in refresh
            haskey(packages, id) || (println(stderr, "unknown package ", id); return 2)
            refresh!(joinpath(root, "trace", "wp", packages[id].file), goals)
            println("refreshed ", id)
        end
    end
    r = check(root)
    print(summary_text(r))
    if check_mode
        for (id, gs) in sort!(collect(r.stale))
            println("::warning::$id is stale: $(join(gs, ", ")) changed since it was checked")
        end
        if !isempty(r.errors)
            println(stderr, "trace check failed with $(length(r.errors)) error(s)")
            return 1
        end
        println("trace check passed")
    end
    return 0
end

exit(main(ARGS))
