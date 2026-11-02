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

@testset "T-003 broken links, tags, unknown keys and engine dependencies are errors [A0, A9]" begin
    root = fixture()
    write(joinpath(root, "test", "more.jl"),
        "@testset \"T-001 again [A1]\" begin end\n@testset \"T-004 untagged\" begin end\n" *
        "@testset \"T-005 bad [A7]\" begin end\n")
    wp = joinpath(root, "trace", "wp", "WP-002-x.md")
    write(wp, replace(read(wp, String), "depends_on = [\"WP-001\"]" => "depends_on = [\"WP-009\"]\nchecked = {}"))
    write(joinpath(root, "Project.toml"), "name = \"X\"\n[deps]\nMolly = \"aa0f7f06-fcc0-5ec4-a7f3-a573f33f9c4c\"\n")
    errors = check(root).errors
    @test any(contains("T-001 is already used"), errors)
    @test any(contains("T-004 names no goal"), errors)
    @test any(contains("tags unknown id A7"), errors)
    @test any(contains("unknown package WP-009"), errors)
    @test any(contains("must not depend on Molly"), errors)
    @test any(contains("unknown front-matter key checked"), errors)
    @test length(errors) == 6
end

@testset "T-004 the repository passes its own check [A0]" begin
    r = check(ROOT)
    @test isempty(r.errors)
end
