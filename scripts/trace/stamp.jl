# Stamp records: julia +1.13 --project=scripts scripts/trace/stamp.jl [--review] (ID... | --all)
#
# Recomputes the `hash` of each record and fills its `[upstream]` table: new links get the
# upstream's current hash, removed links are dropped, existing entries are kept. With --review
# every entry is refreshed, which records that the record was re-reviewed against its upstream
# and clears its staleness.
include(joinpath(@__DIR__, "Trace.jl"))
using .Trace

function main(args)
    review = "--review" in args
    all_ = "--all" in args
    ids = filter(a -> !startswith(a, "--"), args)
    if all_ == !isempty(ids)
        println(stderr, "usage: stamp.jl [--review] (ID... | --all)")
        return 2
    end
    dir = joinpath(dirname(dirname(@__DIR__)), "trace", "records")
    records, errors = load_records(dir)
    if !isempty(errors)
        foreach(e -> println(stderr, e), errors)
        return 1
    end
    all_ && (ids = sort!(collect(keys(records))))
    for id in ids
        haskey(records, id) || (println(stderr, "no record $id"); return 1)
        h = stamp!(records[id].path, records; review)
        println(id, " ", h)
    end
    return 0
end

exit(main(ARGS))
