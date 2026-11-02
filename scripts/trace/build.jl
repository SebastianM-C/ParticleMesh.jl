# Trace check: julia +1.13 --project=scripts scripts/trace/build.jl [--check]
#
# Reads docs/goals/, the front matter of trace/wp/ and the test tags, and prints coverage and
# the errors. With --check, exits 1 on any error. See trace/README.md.
include(joinpath(@__DIR__, "Trace.jl"))
using .Trace

function main(args)
    root = dirname(dirname(@__DIR__))
    check_mode = false
    for a in args
        if a == "--check"
            check_mode = true
        else
            println(stderr, "unknown argument: ", a)
            return 2
        end
    end
    r = check(root)
    print(summary_text(r))
    if check_mode
        if !isempty(r.errors)
            println(stderr, "trace check failed with $(length(r.errors)) error(s)")
            return 1
        end
        println("trace check passed")
    end
    return 0
end

exit(main(ARGS))
