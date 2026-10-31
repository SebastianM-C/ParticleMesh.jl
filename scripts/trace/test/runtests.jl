# Tests of the trace tooling: julia +1.13 --project=scripts scripts/trace/test/runtests.jl
using Test, TOML

include(joinpath(@__DIR__, "..", "Trace.jl"))
using .Trace

const ROOT = dirname(dirname(dirname(@__DIR__)))
const RECORDS = joinpath(ROOT, "trace", "records")

function copy_records()
    dir = mktempdir()
    for f in readdir(RECORDS)
        cp(joinpath(RECORDS, f), joinpath(dir, f))
    end
    return dir
end

stale_ids(b) = Set(id for id in keys(b.records) if Trace.isstale(b, id))

@testset "T-001 trace staleness propagates [R-014]" begin
    dir = copy_records()
    before = build(dir, ROOT)
    @test isempty(before.mismatched)
    @test isempty(stale_ids(before))

    path = joinpath(dir, "R-014.toml")
    text = read(path, String)
    write(path, replace(text, "statement = \"\"\"" => "statement = \"\"\"Edited. "; count = 1))
    unstamped = build(dir, ROOT)
    @test unstamped.mismatched == Set(["R-014"])
    @test any(contains("R-014: hash mismatch"), unstamped.errors)

    stamp!(path, first(load_records(dir)))
    b = build(dir, ROOT)
    @test isempty(b.mismatched)
    @test sort(b.errors) == sort(before.errors)
    @test stale_ids(b) == Set(["D-013", "WP-001", "T-001", "T-002"])
    @test b.direct["D-013"] == ["R-014"]
    @test b.direct["WP-001"] == ["R-014"]
    @test b.via["WP-001"] == ["D-013"]
    for id in ("R-014", "R-001", "R-008", "D-001", "D-006", "I-001", "T-000")
        @test !Trace.isstale(b, id)
    end
    @test any(contains("D-013 is accepted and stale"), b.warnings)
    @test contains(summary_text(b), "D-013  (upstream changed: R-014)")

    # Re-reviewing D-013 clears its staleness; WP-001 still links R-014 directly.
    stamp!(joinpath(dir, "D-013.toml"), first(load_records(dir)); review = true)
    after = build(dir, ROOT)
    @test !Trace.isstale(after, "D-013")
    @test after.direct["WP-001"] == ["R-014"]
    @test !haskey(after.via, "WP-001")
end

@testset "T-002 trace hashes are canonical [R-014]" begin
    data = TOML.parsefile(joinpath(RECORDS, "D-004.toml"))
    h = record_hash(data)
    @test h == data["hash"]

    reversed = sort!([k for k in keys(data) if !(k in ("hash", "upstream"))]; rev = true)
    @test record_hash(TOML.parse(render_record(data; order = reversed))) == h
    io = IOBuffer()
    TOML.print(io, data; sorted = true)
    @test record_hash(TOML.parse(String(take!(io)))) == h

    @test canonical(Dict("a" => 1, "b" => [1, 2])) == canonical(Dict("b" => [1, 2], "a" => 1))
    @test record_hash(merge(data, Dict("hash" => "x", "upstream" => Dict("R-001" => "y")))) == h
    @test record_hash(merge(data, Dict("decision" => "a\r\nb"))) ==
        record_hash(merge(data, Dict("decision" => "a\nb")))
    @test record_hash(merge(data, Dict("title" => data["title"] * "!"))) != h
    @test record_hash(merge(data, Dict("derived_from" => reverse(data["derived_from"])))) != h
end
