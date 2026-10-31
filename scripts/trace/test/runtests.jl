# Tests of the trace checker: julia +1.13 --project=scripts scripts/trace/test/runtests.jl
using Test

include(joinpath(@__DIR__, "..", "Trace.jl"))
using .Trace

const ROOT = dirname(dirname(dirname(@__DIR__)))

# A small repository: two goals, two packages, one test file, a root Project.toml.
function fixture()
    root = mktempdir()
    mkpath(joinpath(root, "docs", "goals"))
    mkpath(joinpath(root, "trace", "wp"))
    mkpath(joinpath(root, "test"))
    write(joinpath(root, "docs", "goals", "A1-units.md"), "# A1. Units\n\nNormalized units.\n")
    write(joinpath(root, "docs", "goals", "A2-scheme.md"), "# A2. Scheme\n\nYee and Boris.\n")
    write(joinpath(root, "docs", "goals", "README.md"), "not a goal\n")
    for (id, goals, deps) in (("WP-001", ["A1"], String[]), ("WP-002", ["A1", "A2"], ["WP-001"]))
        deps_toml = isempty(deps) ? "[]" : "[" * join(repr.(deps), ", ") * "]"
        write(joinpath(root, "trace", "wp", "$id-x.md"),
            "+++\nid = \"$id\"\nstatus = \"proposed\"\ngoals = [$(join(repr.(goals), ", "))]\n" *
            "depends_on = $deps_toml\n+++\n# $id: x\n\nBody.\n")
    end
    write(joinpath(root, "test", "runtests.jl"),
        "@testset \"T-001 one [A1]\" begin end\n@testset \"T-002 two [WP-002]\" begin end\n")
    write(joinpath(root, "Project.toml"), "name = \"X\"\n[deps]\nTest = \"8dfed614-e22c-5e08-85e1-65c5234f0b40\"\n")
    return root
end

refresh_all(root) = for f in readdir(joinpath(root, "trace", "wp"))
    refresh!(joinpath(root, "trace", "wp", f), first(load_goals(joinpath(root, "docs", "goals"))))
end

@testset "T-001 a goal edit marks exactly the packages that serve it stale [A0]" begin
    root = fixture()
    r = check(root)
    @test any(contains("no checked entry for A1"), r.errors)
    refresh_all(root)
    r = check(root)
    @test isempty(r.errors)
    @test isempty(r.stale)

    path = joinpath(root, "docs", "goals", "A2-scheme.md")
    write(path, read(path, String) * "Esirkepov current.\n")
    r = check(root)
    @test isempty(r.errors)
    @test r.stale == Dict("WP-002" => ["A2"])

    # Checking WP-002 against the new A2 clears it; the body is kept as it was.
    wp = joinpath(root, "trace", "wp", "WP-002-x.md")
    refresh!(wp, first(load_goals(joinpath(root, "docs", "goals"))))
    @test isempty(check(root).stale)
    @test endswith(read(wp, String), "+++\n# WP-002: x\n\nBody.\n")

    # A withdrawn package is never stale.
    write(path, read(path, String) * "Quadratic shapes.\n")
    write(wp, replace(read(wp, String), "status = \"proposed\"" => "status = \"withdrawn\""))
    @test isempty(check(root).stale)
end

@testset "T-002 checked entries are git blob ids [A0]" begin
    @test blob_id(Vector{UInt8}("")) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    @test blob_id(Vector{UInt8}("hello\n")) == "ce013625030ba8dba906f756967f9e9ca394464a"
    if Sys.which("git") !== nothing
        f = joinpath(ROOT, "docs", "goals", "A1-units.md")
        @test blob_id(f) == readchomp(`git hash-object $f`)
    end
end

@testset "T-003 broken links, tags and engine dependencies are errors [A0, A9]" begin
    root = fixture()
    refresh_all(root)
    write(joinpath(root, "test", "more.jl"),
        "@testset \"T-001 again [A1]\" begin end\n@testset \"T-004 untagged\" begin end\n" *
        "@testset \"T-005 bad [A7]\" begin end\n")
    wp = joinpath(root, "trace", "wp", "WP-002-x.md")
    write(wp, replace(read(wp, String), "depends_on = [\"WP-001\"]" => "depends_on = [\"WP-009\"]"))
    write(joinpath(root, "Project.toml"), "name = \"X\"\n[deps]\nMolly = \"aa0f7f06-fcc0-5ec4-a7f3-a573f33f9c4c\"\n")
    errors = check(root).errors
    @test any(contains("T-001 is already used"), errors)
    @test any(contains("T-004 names no goal"), errors)
    @test any(contains("tags unknown id A7"), errors)
    @test any(contains("unknown package WP-009"), errors)
    @test any(contains("must not depend on Molly"), errors)
    @test length(errors) == 5
end

@testset "T-004 the repository passes its own check [A0]" begin
    r = check(ROOT)
    @test isempty(r.errors)
end
