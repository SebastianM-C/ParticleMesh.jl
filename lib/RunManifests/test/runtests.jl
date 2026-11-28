using Test
using RunManifests
using TOML: TOML

@testset "T-1240 a manifest records SHA-256 test vectors, relative inputs and keeps earlier entries [A6, WP-092]" begin
    mktempdir() do d
        # FIPS 180-2's test vectors: the empty message and "abc"
        write(joinpath(d, "empty.txt"), "")
        write(joinpath(d, "abc.txt"), "abc")
        @test file_hash(joinpath(d, "empty.txt")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        @test file_hash(joinpath(d, "abc.txt")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        mkpath(joinpath(d, "run"))
        outside = joinpath(d, "abc.txt")
        inside = joinpath(d, "run", "case.toml")
        write(inside, "name = \"x\"\n")
        write_manifest(joinpath(d, "run"); inputs = [outside, "case.toml"], extra = Dict{String,Any}("step" => "prepare"))
        m = read_manifest(joinpath(d, "run"))
        # an input inside the run directory is recorded relative to it, one outside by the path given
        @test m["inputs"]["case.toml"] == "sha256:" * file_hash(inside)
        @test m["inputs"][outside] == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        @test m["step"] == "prepare" && occursin(r"^\d{4}-\d\d-\d\dT.*Z$", m["updated"])
        # a second step adds to the manifest: earlier inputs and entries stay unless overwritten
        write(joinpath(d, "run", "summary.json"), "{}")
        write_manifest(joinpath(d, "run"); inputs = ["summary.json"], extra = Dict{String,Any}("analyzed" => true))
        m2 = read_manifest(joinpath(d, "run"))
        @test m2["inputs"]["case.toml"] == m["inputs"]["case.toml"] && haskey(m2["inputs"], "summary.json")
        @test m2["step"] == "prepare" && m2["analyzed"] === true
        @test m2 == TOML.parsefile(joinpath(d, "run", "manifest.toml"))
    end
end

@testset "T-184 the manifest stores the bare image id [A5, WP-012]" begin
    id = "7030c120c62ce14dbae971cce76a3973f0a031ef63632a6c4bf4fa20e3937b4c"
    dir = mktempdir()
    log = joinpath(dir, "runner.log")
    write(log, """
    # gpu-run 2026-10-31T17:58:05Z
    # worktree: <worktree> (HEAD 1b01ce7)
    # command: smilei scripts/runs/smilei-landau-9a47a107
    # image: localhost/particlemesh-smilei:e4f0148b8952fc87 id $id
    # network: off, gpu: no, timeout: 7200s
    # gpu-run finished: exit 0 after 51s; image: $id; log: <log>
    """)
    img = image_from_header(log)
    @test img.id == id
    @test img.name == "localhost/particlemesh-smilei:e4f0148b8952fc87"
    # run-smilei.sh's image.txt writes "# image: <name> <id>" without the word id
    txt = joinpath(dir, "image.txt")
    write(txt, "# image: particlemesh-smilei:e4f0148b8952fc87 sha256:$id\n# podman run finished: exit 0\n")
    @test image_from_header(nothing, txt).id == id
    @test image_from_header(joinpath(dir, "absent.log")).id == "unknown"
    write_manifest(dir; extra = Dict{String,Any}("image" => Dict{String,Any}("id" => img.id, "name" => img.name)))
    m = read_manifest(dir)
    @test m["image"]["id"] == id
    @test occursin(r"^[0-9a-f]{64}$", m["image"]["id"])
end

@testset "T-460 a run manifest names its commit without git: the runner's PM_COMMIT, and says so [A6]" begin
    # A sandbox or container that mounts the worktree but not the repository's .git (the CPU box
    # and the GPU runner before #179's fix) left `commit = "unknown"` in manifests unnoticed.
    sha = "0123456789abcdef0123456789abcdef01234567"
    mktempdir() do d                                   # not a git repository
        withenv("PM_COMMIT" => nothing) do
            c = git_commit(d)
            @test (c.commit, c.commit_source) == ("unknown", "none")
            @test c.dirty === missing
        end
        withenv("PM_COMMIT" => "not-a-commit") do
            @test git_commit(d).commit == "unknown"
        end
        withenv("PM_COMMIT" => sha) do
            c = git_commit(d)
            @test (c.commit, c.commit_source) == (sha, "PM_COMMIT")
            @test c.dirty === missing
            write_manifest(d; root = d)
            m = read_manifest(d)
            @test m["commit"] == sha && m["commit_source"] == "PM_COMMIT" && !haskey(m, "dirty")
            # a manifest first written under git keeps no stale dirty flag next to the new commit
            write(joinpath(d, "manifest.toml"), "commit = \"old\"\ndirty = true\n")
            write_manifest(d; root = d)
            m = read_manifest(d)
            @test m["commit"] == sha && m["commit_source"] == "PM_COMMIT" && !haskey(m, "dirty")
        end
    end
    root = dirname(dirname(dirname(@__DIR__)))           # this checkout: git sees it
    if isdir(joinpath(root, ".git")) || isfile(joinpath(root, ".git"))
        withenv("PM_COMMIT" => sha) do
            c = git_commit(root)
            @test c.commit_source == "git"
            @test c.commit != sha && occursin(r"^[0-9a-f]{40}$", c.commit)
        end
    else
        println("T-460: the checkout's .git is not visible here; its git half is skipped")
    end
end

@testset "T-1241 wall times from appended container stamps: a killed start is superseded [A5, WP-092]" begin
    mktempdir() do d
        f = joinpath(d, "container.txt")
        write(f, """
            # compile start: 2026-11-02T10:00:00.000Z
            # compile end: 2026-11-02T10:01:40.000Z
            # run seed-1 seed 1 start: 2026-11-02T10:01:40.000Z
            # compile start: 2026-11-02T11:00:00.000Z
            # compile end: 2026-11-02T11:01:00.000Z
            # run seed-1 seed 1 start: 2026-11-02T11:01:00.000Z
            # run seed-1 seed 1 end: 2026-11-02T11:06:30.000Z
            """)
        @test stamp_interval(f, "# run seed-1 seed 1 start:", "# run seed-1 seed 1 end:") == 330.0
        @test stamp_intervals(f, "# compile start:", "# compile end:") == [100.0, 60.0]
        @test stamp_interval(f, "# never start:", "# never end:") === nothing
        @test stamp_intervals(joinpath(d, "absent.txt"), "a", "b") == Float64[]
    end
end

using Aqua
@testset "T-1242 RunManifests patches nothing: no piracy, no eval, no stale dependencies or missing compat [A6, WP-092]" begin
    Aqua.test_all(RunManifests)
    # no method added to another module's function from outside, and no code evaluated into one
    src = read(pathof(RunManifests), String)
    @test !occursin(r"@eval|\beval\(|Core\.eval|Base\.eval", src)
end
