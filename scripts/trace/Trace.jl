"""
    Trace

Links between the owner's goals (`docs/goals/A<n>-<slug>.md`), the work packages
(`trace/wp/WP-<nnn>-<slug>.md` with TOML front matter) and the tests (testset names with
trace tags), and one check over them. History comes from git; this module only reads the
links. Specification: `trace/README.md`.
"""
module Trace

using TOML

export load_goals, load_packages, scan_testsets, check, summary_text

const GOAL_FILE_RE = r"^(A\d+)-[a-z0-9-]+\.md$"
const WP_FILE_RE = r"^(WP-\d{3,})-[a-z0-9-]+\.md$"
const STATUSES = ("proposed", "active", "done", "parked", "withdrawn")
const KEYS = ("id", "status", "goals", "depends_on", "adds_dependencies")
const TEST_DIRS = ("test", "lib", "scripts", "benchmark")
# A testset whose name starts with a T- id, then the name, then the tags in brackets.
const TESTSET_RE = r"@testset\s+\"(T-\d{3,})\s+([^\"\[]*?)\s*(?:\[([^\]\"]*)\])?\s*\""
# Packages the engine (the root package and lib/) must not depend on (CLAUDE.md).
const FORBIDDEN = ("ModelingToolkit", "MethodOfLines", "Molly", "Checkpointing")

struct Goal
    id::String
    file::String
    title::String
end

struct Package
    id::String
    file::String
    meta::Dict{String,Any}
end

"""
    load_goals(dir) -> (goals::Dict{String,Goal}, errors::Vector{String})
"""
function load_goals(dir::AbstractString)
    goals = Dict{String,Goal}()
    errors = String[]
    for f in sort!(readdir(dir))
        (f == "README.md" || !endswith(f, ".md")) && continue
        m = match(GOAL_FILE_RE, f)
        if m === nothing
            push!(errors, "$f: a goal file is named A<n>-<slug>.md")
            continue
        end
        id = String(m[1])
        path = joinpath(dir, f)
        first_line = something(iterate(eachline(path)), ("",))[1]
        h = match(r"^# (A\d+)\. (.+)$", first_line)
        if h === nothing || h[1] != id
            push!(errors, "$f: the first line must be \"# $id. <title>\"")
            continue
        end
        haskey(goals, id) && push!(errors, "$f: goal id $id is used twice")
        goals[id] = Goal(id, f, String(h[2]))
    end
    return goals, errors
end

"""
    front_matter(text) -> (meta, body_start) or nothing

TOML between a first line `+++` and the next line `+++`.
"""
function front_matter(text::AbstractString)
    lines = split(text, '\n')
    (isempty(lines) || rstrip(lines[1]) != "+++") && return nothing
    close = findnext(l -> rstrip(l) == "+++", lines, 2)
    close === nothing && return nothing
    return TOML.parse(join(lines[2:close-1], '\n')), close + 1
end

"""
    load_packages(dir) -> (packages::Dict{String,Package}, errors::Vector{String})
"""
function load_packages(dir::AbstractString)
    packages = Dict{String,Package}()
    errors = String[]
    for f in sort!(readdir(dir))
        endswith(f, ".md") || continue
        m = match(WP_FILE_RE, f)
        if m === nothing
            push!(errors, "$f: a work package is named WP-<nnn>-<slug>.md")
            continue
        end
        fm = try
            front_matter(read(joinpath(dir, f), String))
        catch e
            push!(errors, "$f: front matter is not valid TOML: $(sprint(showerror, e))")
            continue
        end
        if fm === nothing
            push!(errors, "$f: no front matter (a first line +++, TOML, a line +++)")
            continue
        end
        meta = fm[1]
        id = String(m[1])
        get(meta, "id", nothing) == id || push!(errors, "$f: front matter id must be \"$id\"")
        haskey(packages, id) && push!(errors, "$f: package id $id is used twice")
        packages[id] = Package(id, f, meta)
    end
    return packages, errors
end

"""
    scan_testsets(root) -> Vector of named tuples (id, name, tags, file, line)

Every testset whose name starts with a `T-` id in the Julia files under `test/`, `lib/`,
`scripts/` and `benchmark/`.
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

struct Report
    goals::Dict{String,Goal}
    packages::Dict{String,Package}
    testsets::Vector
    errors::Vector{String}
end

string_list(x) = x isa AbstractVector && all(v -> v isa AbstractString, x)

function check_package!(errors, p::Package, goals, packages)
    m, f = p.meta, p.file
    for k in keys(m)
        k in KEYS || push!(errors, "$f: unknown front-matter key $k")
    end
    status = get(m, "status", nothing)
    status in STATUSES || push!(errors, "$f: status must be one of $(join(STATUSES, ", "))")
    gs = get(m, "goals", nothing)
    if !string_list(gs) || isempty(gs)
        push!(errors, "$f: goals must be a non-empty list of goal ids")
        gs = String[]
    end
    deps = get(m, "depends_on", String[])
    string_list(deps) || (push!(errors, "$f: depends_on must be a list of package ids"); deps = String[])
    string_list(get(m, "adds_dependencies", String[])) ||
        push!(errors, "$f: adds_dependencies must be a list of package names")
    for g in gs
        haskey(goals, g) || push!(errors, "$f: unknown goal $g")
    end
    for d in deps
        haskey(packages, d) || push!(errors, "$f: depends on unknown package $d")
        d == p.id && push!(errors, "$f: depends on itself")
    end
    return
end

function check_engine_dependencies!(errors, root)
    files = [joinpath(root, "Project.toml")]
    libdir = joinpath(root, "lib")
    if isdir(libdir)
        for d in sort!(readdir(libdir))
            f = joinpath(libdir, d, "Project.toml")
            isfile(f) && push!(files, f)
        end
    end
    for f in files
        isfile(f) || continue
        proj = TOML.parsefile(f)
        for section in ("deps", "weakdeps")
            for name in keys(get(proj, section, Dict()))
                name in FORBIDDEN &&
                    push!(errors, "$(relpath(f, root)): the engine must not depend on $name ($section)")
            end
        end
    end
end

"""
    check(root) -> Report

Read the goals, the work packages and the test tags under `root` and check the links.
"""
function check(root::AbstractString;
               goals_dir = joinpath(root, "docs", "goals"),
               wp_dir = joinpath(root, "trace", "wp"))
    goals, gerr = load_goals(goals_dir)
    packages, perr = load_packages(wp_dir)
    errors = vcat(gerr, perr)
    for id in sort!(collect(keys(packages)))
        check_package!(errors, packages[id], goals, packages)
    end
    testsets = scan_testsets(root)
    seen = Dict{String,Any}()
    for t in testsets
        where = "$(t.file):$(t.line)"
        haskey(seen, t.id) && push!(errors, "$where: test id $(t.id) is already used at $(seen[t.id])")
        seen[t.id] = where
        isempty(t.tags) && push!(errors, "$where: testset $(t.id) names no goal or package in brackets")
        for tag in t.tags
            haskey(goals, tag) || haskey(packages, tag) ||
                push!(errors, "$where: testset $(t.id) tags unknown id $tag")
        end
    end
    check_engine_dependencies!(errors, root)
    return Report(goals, packages, testsets, errors)
end

goal_sort(id) = parse(Int, id[2:end])

function summary_text(r::Report)
    io = IOBuffer()
    println(io, "trace: $(length(r.goals)) goals, $(length(r.packages)) work packages, ",
        "$(length(r.testsets)) tagged testsets")
    println(io, "coverage (goal: packages; tests):")
    for g in sort!(collect(keys(r.goals)); by = goal_sort)
        wps = sort!([p.id for p in values(r.packages)
                     if g in get(p.meta, "goals", String[]) && get(p.meta, "status", "") != "withdrawn"])
        tests = sort!(unique([t.id for t in r.testsets if g in t.tags || any(in(wps), t.tags)]))
        println(io, "  $g $(r.goals[g].title): ",
            isempty(wps) ? "-" : join(wps, ", "), "; ", isempty(tests) ? "-" : join(tests, ", "))
    end
    println(io, "errors: ", isempty(r.errors) ? "none" : "")
    for e in r.errors
        println(io, "  ", e)
    end
    return String(take!(io))
end

end
