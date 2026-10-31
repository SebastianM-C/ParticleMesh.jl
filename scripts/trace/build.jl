# Build the trace graph: julia +1.13 --project=scripts scripts/trace/build.jl [--check]
#
# Reads trace/records/, verifies hashes and links, propagates staleness, matches test records
# against the testsets in the code, writes trace/build/{graph,coverage}.json and summary.txt
# and prints the summary. With --check, exits 1 on any error (hash mismatch, broken link,
# untraced test); stale accepted records are warnings only.
using JSON

include(joinpath(@__DIR__, "Trace.jl"))
using .Trace

function main(args)
    check = "--check" in args
    unknown = filter(!=("--check"), args)
    isempty(unknown) || (println(stderr, "unknown arguments: ", join(unknown, " ")); return 2)
    root = dirname(dirname(@__DIR__))
    b = build(joinpath(root, "trace", "records"), root)
    write_outputs(b, joinpath(root, "trace", "build"); json = x -> JSON.json(x, 2))
    print(summary_text(b))
    if check
        for w in b.warnings
            println("::warning::", w)
        end
        if !isempty(b.errors)
            println(stderr, "trace check failed with $(length(b.errors)) error(s)")
            return 1
        end
        println("trace check passed")
    end
    return 0
end

exit(main(ARGS))
