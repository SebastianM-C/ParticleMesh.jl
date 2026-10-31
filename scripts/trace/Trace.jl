# Traceability records: loading, canonical hashing, staleness, test tags, outputs.
# Specification: trace/README.md. Used by build.jl, stamp.jl, session.jl and the tests.
module Trace

using Dates, SHA, TOML

export load_records, record_hash, canonical, build, write_outputs, summary_text, stamp!,
    next_id, scan_testsets, render_record

const PREFIXES = ("R", "D", "I", "WP", "T", "RUN", "S")
const ID_RE = r"^(R|D|I|WP|T|RUN|S)-(\d{3,})$"

# Fields whose values are ids of upstream records. Every id listed in one of them must have an
# entry in the record's `upstream` table.
const LINK_FIELDS = ("derived_from", "depends_on", "implements", "satisfies", "verifies",
    "work_packages")

const REQUIRED = Dict(
    "R" => ("id", "title", "status", "source", "statement", "rationale"),
    "D" => ("id", "title", "status", "source", "decision", "alternatives_rejected",
        "rationale", "consequences", "derived_from"),
    "I" => ("id", "title", "status", "source", "contract", "derived_from"),
    "WP" => ("id", "title", "status", "source", "scope", "non_goals", "acceptance", "budgets",
        "implements", "satisfies"),
    "T" => ("id", "title", "status", "checks", "tolerance", "testset", "verifies"),
    "RUN" => ("id", "title", "status", "inputs", "commit", "verifies"),
    "S" => ("id", "agent", "platform", "work_packages", "start", "end", "outcome"),
)

# A testset whose name starts with a trace id, then the name, then the requirement tags in
# brackets (the form is in trace/README.md; it is not spelled out here so this file stays
# out of the scan).
const TESTSET_RE = r"@testset\s+\"(T-\d{3,})\s+([^\"\[]*?)\s*(?:\[([^\]\"]*)\])?\s*\""

const TEST_DIRS = ("test", "lib", "scripts", "benchmark")

struct Record
    id::String
    prefix::String
    path::String
    data::Dict{String,Any}
    stored_hash::Union{Nothing,String}
    hash::String
    upstream::Dict{String,String}
end

prefix_of(id) = match(ID_RE, id)[1]

## Canonical form and hash

"""
    canonical(x) -> String

The canonical serialization that record hashes are computed from: JSON without whitespace,
object keys sorted by code point, arrays in order, `\\r\\n` normalized to `\\n` in strings,
integers in decimal, floats as Julia's shortest round-trip `repr` of the `Float64`, dates and
times as their ISO 8601 strings.
"""
canonical(x::AbstractDict) =
    "{" * join((canonical(string(k)) * ":" * canonical(x[k]) for k in sort!(collect(keys(x)))), ",") * "}"
canonical(x::AbstractVector) = "[" * join((canonical(v) for v in x), ",") * "]"
canonical(x::Bool) = x ? "true" : "false"
canonical(x::Integer) = string(x)
function canonical(x::AbstractFloat)
    isfinite(x) || error("non-finite float in a record: $x")
    return repr(Float64(x))
end
canonical(x::Union{Date,DateTime,Time}) = canonical(string(x))
function canonical(x::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for c in replace(x, "\r\n" => "\n")
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c < ' '
            print(io, "\\u", string(UInt32(c); base = 16, pad = 4))
        else
            print(io, c)
        end
    end
    print(io, '"')
    return String(take!(io))
end

"""
    record_hash(data::AbstractDict) -> "sha256:<hex>"

SHA-256 of the canonical form of a parsed record with its `hash` and `upstream` entries removed.
"""
function record_hash(data::AbstractDict)
    content = Dict{String,Any}(string(k) => v for (k, v) in data if !(string(k) in ("hash", "upstream")))
    return "sha256:" * bytes2hex(sha256(canonical(content)))
end

## Loading

links(data::AbstractDict) =
    unique!(String[string(v) for f in LINK_FIELDS if haskey(data, f) for v in data[f]])
links(r::Record) = links(r.data)

"""
    load_records(dir) -> (records::Dict{String,Record}, errors::Vector{String})

Parse every `*.toml` file in `dir` and check the per-file rules: the file name is the id, the id
is well formed, the required fields of its type are present, link fields are lists of strings.
"""
function load_records(dir::AbstractString)
    records = Dict{String,Record}()
    errors = String[]
    isdir(dir) || return records, ["records directory $dir does not exist"]
    for file in sort!(filter(endswith(".toml"), readdir(dir)))
        path = joinpath(dir, file)
        data = try
            TOML.parsefile(path)
        catch err
            push!(errors, "$file: TOML parse error: $(sprint(showerror, err))")
            continue
        end
        id = get(data, "id", nothing)
        if !(id isa String) || match(ID_RE, id) === nothing
            push!(errors, "$file: missing or malformed id $(repr(id))")
            continue
        end
        if file != id * ".toml"
            push!(errors, "$file: file name does not match id $id")
            continue
        end
        prefix = prefix_of(id)
        for f in REQUIRED[prefix]
            haskey(data, f) || push!(errors, "$id: missing field `$f`")
        end
        for f in LINK_FIELDS
            haskey(data, f) || continue
            v = data[f]
            if !(v isa AbstractVector && all(x -> x isa String, v))
                push!(errors, "$id: field `$f` must be a list of ids")
                delete!(data, f)
            end
        end
        stored = get(data, "hash", nothing)
        stored isa Union{Nothing,String} || (stored = string(stored))
        up = get(data, "upstream", Dict{String,Any}())
        upstream = Dict{String,String}(string(k) => string(v) for (k, v) in up)
        records[id] = Record(id, prefix, path, data, stored, record_hash(data), upstream)
    end
    return records, errors
end

## Test tags

"""
    scan_testsets(root) -> Vector of named tuples (id, name, tags, file, line)

Find every testset whose name starts with a `T-` id in the Julia files under the test
directories of the repository (`test/`, `lib/`, `scripts/`, `benchmark/`).
"""
function scan_testsets(root::AbstractString)
    found = NamedTuple{(:id, :name, :tags, :file, :line),Tuple{String,String,Vector{String},String,Int}}[]
    for d in TEST_DIRS
        top = joinpath(root, d)
        isdir(top) || continue
        for (dir, _, files) in walkdir(top)
            for f in sort!(filter(endswith(".jl"), files))
                path = joinpath(dir, f)
                for (n, line) in enumerate(eachline(path))
                    for m in eachmatch(TESTSET_RE, line)
                        tags = m[3] === nothing ? String[] :
                            String[strip(t) for t in split(m[3], ',') if !isempty(strip(t))]
                        push!(found, (id = String(m[1]), name = String(m[2]), tags = tags,
                            file = relpath(path, root), line = n))
                    end
                end
            end
        end
    end
    return found
end

## Build

struct Build
    records::Dict{String,Record}
    testsets::Vector
    errors::Vector{String}
    warnings::Vector{String}
    mismatched::Set{String}
    direct::Dict{String,Vector{String}}   # id => upstream ids whose recorded hash is out of date
    via::Dict{String,Vector{String}}      # id => stale upstream ids it inherits staleness from
end

isstale(b::Build, id) = haskey(b.direct, id) || haskey(b.via, id)
trace_status(b::Build, id) = id in b.mismatched ? "hash-mismatch" : isstale(b, id) ? "stale" : "ok"

"""
    build(records_dir, test_root) -> Build

Verify hashes and links, compute staleness, and match test records against testsets.
"""
function build(records_dir::AbstractString, test_root::AbstractString)
    records, errors = load_records(records_dir)
    warnings = String[]
    mismatched = Set{String}()
    ids = sort!(collect(keys(records)))

    for id in ids
        r = records[id]
        if r.stored_hash === nothing
            push!(errors, "$id: no `hash`; run scripts/trace/stamp.jl $id")
            push!(mismatched, id)
        elseif r.stored_hash != r.hash
            push!(errors, "$id: hash mismatch (stored $(r.stored_hash), content $(r.hash)); " *
                "restamp it with scripts/trace/stamp.jl $id")
            push!(mismatched, id)
        end
        ls = links(r)
        for u in ls
            haskey(records, u) || push!(errors, "$id: links to unknown record $u")
            haskey(r.upstream, u) || push!(errors, "$id: link $u has no entry in `upstream`")
        end
        for u in keys(r.upstream)
            u in ls || push!(errors, "$id: `upstream` entry $u is not a link of this record")
        end
    end

    # Directly stale: a recorded upstream hash differs from the upstream's current content hash.
    direct = Dict{String,Vector{String}}()
    for id in ids
        r = records[id]
        changed = sort!([u for (u, h) in r.upstream if haskey(records, u) && records[u].hash != h])
        isempty(changed) || (direct[id] = changed)
    end

    # Staleness propagates downstream along links; links form a DAG.
    via = Dict{String,Vector{String}}()
    state = Dict{String,Symbol}()
    isstale_(id) = haskey(direct, id) || haskey(via, id)
    function visit(id)
        s = get(state, id, :new)
        s === :done && return isstale_(id)
        if s === :active
            push!(errors, "$id: cycle in the record links")
            return false
        end
        state[id] = :active
        inherited = String[]
        for u in links(records[id])
            haskey(records, u) || continue
            visit(u) && push!(inherited, u)
        end
        isempty(inherited) || (via[id] = sort!(inherited))
        state[id] = :done
        return isstale_(id)
    end
    foreach(visit, ids)

    # Test records versus testsets.
    testsets = scan_testsets(test_root)
    seen = Dict{String,Any}()
    for t in testsets
        if haskey(seen, t.id)
            push!(errors, "untraced test: $(t.id) names two testsets ($(seen[t.id].file):$(seen[t.id].line), $(t.file):$(t.line))")
            continue
        end
        seen[t.id] = t
        if !haskey(records, t.id)
            push!(errors, "untraced test: testset \"$(t.id) $(t.name)\" at $(t.file):$(t.line) has no record")
            continue
        end
        r = records[t.id]
        if get(r.data, "testset", nothing) != t.name
            push!(errors, "untraced test: $(t.id) record names testset $(repr(get(r.data, "testset", nothing))), " *
                "the code has $(repr(t.name)) at $(t.file):$(t.line)")
        end
        if sort(t.tags) != sort(String[get(r.data, "verifies", String[])...])
            push!(errors, "untraced test: $(t.id) tags $(t.tags) at $(t.file):$(t.line) differ from the " *
                "record's `verifies` $(get(r.data, "verifies", String[]))")
        end
    end
    for id in ids
        records[id].prefix == "T" && !haskey(seen, id) &&
            push!(errors, "untraced test: record $id has no testset in the code")
    end

    for id in ids
        isstale_(id) && get(records[id].data, "status", "") == "accepted" &&
            push!(warnings, "$id is accepted and stale")
    end
    return Build(records, testsets, errors, warnings, mismatched, direct, via)
end

## Outputs

function node_json(b::Build, r::Record)
    t = findfirst(x -> x.id == r.id, b.testsets)
    return Dict{String,Any}(
        "id" => r.id,
        "type" => r.prefix,
        "title" => get(r.data, "title", get(r.data, "agent", "")),
        "status" => get(r.data, "status", nothing),
        "trace_status" => trace_status(b, r.id),
        "stale_upstream" => get(b.direct, r.id, String[]),
        "stale_via" => get(b.via, r.id, String[]),
        "hash" => r.hash,
        "stored_hash" => r.stored_hash,
        "path" => joinpath("trace", "records", basename(r.path)),
        "testset" => t === nothing ? nothing :
            Dict("file" => b.testsets[t].file, "line" => b.testsets[t].line),
    )
end

function graph_json(b::Build)
    ids = sort!(collect(keys(b.records)))
    edges = Dict{String,Any}[]
    for id in ids, f in LINK_FIELDS
        r = b.records[id]
        for u in get(r.data, f, String[])
            push!(edges, Dict{String,Any}(
                "from" => u, "to" => id, "kind" => f,
                "recorded_hash" => get(r.upstream, u, nothing),
                "current" => haskey(b.records, u) && get(r.upstream, u, nothing) == b.records[u].hash,
            ))
        end
    end
    return Dict{String,Any}(
        "nodes" => [node_json(b, b.records[id]) for id in ids],
        "edges" => edges,
        "errors" => b.errors,
        "warnings" => b.warnings,
    )
end

function coverage_json(b::Build)
    reqs = sort!([id for (id, r) in b.records if r.prefix == "R"])
    downstream(prefix, field, req) =
        sort!([id for (id, r) in b.records if r.prefix == prefix && req in get(r.data, field, String[])])
    rows = map(reqs) do req
        tests = downstream("T", "verifies", req)
        runs = downstream("RUN", "verifies", req)
        Dict{String,Any}(
            "requirement" => req,
            "title" => get(b.records[req].data, "title", ""),
            "status" => get(b.records[req].data, "status", nothing),
            "decisions" => downstream("D", "derived_from", req),
            "work_packages" => downstream("WP", "satisfies", req),
            "tests" => tests,
            "runs" => runs,
            "covered" => !isempty(tests) || !isempty(runs),
        )
    end
    return Dict{String,Any}("requirements" => rows)
end

function summary_text(b::Build)
    io = IOBuffer()
    ids = sort!(collect(keys(b.records)))
    counts = join(("$p $(count(id -> b.records[id].prefix == p, ids))" for p in PREFIXES), ", ")
    println(io, "trace: $(length(ids)) records ($counts); $(length(b.testsets)) traced testsets")
    frontier = sort!(collect(keys(b.direct)))
    println(io, "\nstale frontier (an upstream record changed; re-review these first): ",
        isempty(frontier) ? "none" : "")
    for id in frontier
        println(io, "  $id  (upstream changed: $(join(b.direct[id], ", ")))")
    end
    inherited = sort!([id for id in keys(b.via) if !haskey(b.direct, id)])
    println(io, "stale through upstream: ", isempty(inherited) ? "none" : "")
    for id in inherited
        println(io, "  $id  (via $(join(b.via[id], ", ")))")
    end
    untraced = filter(startswith("untraced test"), b.errors)
    println(io, "untraced tests: ", isempty(untraced) ? "none" : "")
    foreach(e -> println(io, "  ", e), untraced)
    others = filter(!startswith("untraced test"), b.errors)
    println(io, "errors: ", isempty(others) ? "none" : "")
    foreach(e -> println(io, "  ", e), others)
    println(io, "warnings: ", isempty(b.warnings) ? "none" : "")
    foreach(w -> println(io, "  ", w), b.warnings)
    return String(take!(io))
end

"""
    write_outputs(b, outdir; json)

Write `graph.json`, `coverage.json` and `summary.txt` to `outdir`. `json(x)` serializes a value
(the caller passes JSON.jl's, so this module needs only the standard library).
"""
function write_outputs(b::Build, outdir::AbstractString; json)
    mkpath(outdir)
    write(joinpath(outdir, "graph.json"), json(graph_json(b)))
    write(joinpath(outdir, "coverage.json"), json(coverage_json(b)))
    write(joinpath(outdir, "summary.txt"), summary_text(b))
    return outdir
end

## Writing records

toml_string(s::AbstractString) = canonical(s)   # JSON string escapes are valid TOML basic strings
toml_value(x::AbstractString) = toml_string(x)
toml_value(x::Union{Integer,Bool}) = canonical(x)
toml_value(x::AbstractFloat) = canonical(x)
toml_value(x::AbstractVector) = "[" * join((toml_value(v) for v in x), ", ") * "]"

"""
    render_record(data; order) -> String

TOML text of a record with top-level scalar and list fields only, in the given key order
(remaining keys sorted after it), without `hash` and `upstream`.
"""
function render_record(data::AbstractDict; order = String[])
    keys_ = [k for k in order if haskey(data, k)]
    append!(keys_, sort!([k for k in keys(data) if !(k in keys_) && !(k in ("hash", "upstream"))]))
    return join(("$k = $(toml_value(data[k]))" for k in keys_), "\n") * "\n"
end

const HASH_LINE_RE = r"^hash\s*=.*$"m
const UPSTREAM_HEADER_RE = r"^\[upstream\][ \t]*$"m

"""
    stamp!(path, records; review = false) -> String

Rewrite the `hash` line and the `[upstream]` table of the record file at `path`. Links gain an
entry with the upstream's current hash; entries of removed links are dropped; with `review`,
every entry is refreshed to the current hash (the record was re-reviewed against its upstream).
Records keep their top-level fields first and `[upstream]` as the last table.
"""
function stamp!(path::AbstractString, records::Dict{String,Record}; review::Bool = false)
    text = read(path, String)
    data = TOML.parse(text)
    old = Dict{String,String}(string(k) => string(v) for (k, v) in get(data, "upstream", Dict()))
    up = Dict{String,String}()
    for u in links(data)
        haskey(records, u) || error("$(data["id"]): links to unknown record $u")
        up[u] = review || !haskey(old, u) ? records[u].hash : old[u]
    end
    m = match(UPSTREAM_HEADER_RE, text)
    body = m === nothing ? text : text[1:prevind(text, m.offset)]
    body = rstrip(replace(body, HASH_LINE_RE => ""))
    body = replace(body, r"\n{3,}" => "\n\n")
    h = record_hash(data)
    new = body * "\nhash = \"$h\"\n\n[upstream]\n" *
        join(("$u = \"$(up[u])\"\n" for u in sort!(collect(keys(up)))))
    check = TOML.parse(new)
    record_hash(check) == h && get(check, "hash", nothing) == h ||
        error("$path: stamping would change the record's content; it must keep its fields " *
              "at the top level and `[upstream]` as its last table")
    write(path, new)
    return h
end

"""
    next_id(records, prefix) -> String

The next id for `prefix`: one past the highest number in use. Records are never deleted (they
are superseded), so an id is never reused.
"""
function next_id(records::Dict{String,Record}, prefix::AbstractString)
    nums = [parse(Int, match(ID_RE, id)[2]) for (id, r) in records if r.prefix == prefix]
    return "$prefix-" * lpad(isempty(nums) ? 1 : maximum(nums) + 1, 3, '0')
end

end
