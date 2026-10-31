# Write one session record:
#   julia +1.13 --project=scripts scripts/trace/session.jl --agent implementer \
#       --platform "Claude Code, local" --wp WP-001,WP-002 --start <commit> --end <commit or PR> \
#       --outcome <merged|open|abandoned|...> [--pr <number>] [--review <status>] [--notes <text>]
#
# Assigns the next S- id, links the work packages touched, stamps the record and prints its id.
include(joinpath(@__DIR__, "Trace.jl"))
using .Trace

const FLAGS = Dict("--agent" => "agent", "--platform" => "platform", "--wp" => "work_packages",
    "--start" => "start", "--end" => "end", "--outcome" => "outcome", "--pr" => "pr",
    "--review" => "review_status", "--notes" => "notes")
const ORDER = ["id", "agent", "platform", "work_packages", "start", "end", "pr", "outcome",
    "review_status", "notes"]

function main(args)
    data = Dict{String,Any}()
    i = 1
    while i <= length(args)
        key = get(FLAGS, args[i], nothing)
        if key === nothing || i == length(args)
            println(stderr, "bad or incomplete argument $(args[i])")
            return 2
        end
        data[key] = args[i+1]
        i += 2
    end
    data["work_packages"] = String[strip(w) for w in split(get(data, "work_packages", ""), ',') if !isempty(strip(w))]
    for f in ("agent", "platform", "start", "end", "outcome")
        haskey(data, f) || (println(stderr, "missing --", f); return 2)
    end
    dir = joinpath(dirname(dirname(@__DIR__)), "trace", "records")
    records, errors = load_records(dir)
    if !isempty(errors)
        foreach(e -> println(stderr, e), errors)
        return 1
    end
    for w in data["work_packages"]
        haskey(records, w) && startswith(w, "WP-") || (println(stderr, "no work package $w"); return 1)
    end
    data["id"] = next_id(records, "S")
    path = joinpath(dir, data["id"] * ".toml")
    write(path, render_record(data; order = ORDER))
    stamp!(path, records)
    println(data["id"])
    return 0
end

exit(main(ARGS))
