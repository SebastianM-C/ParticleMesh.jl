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
        # an entry of extra overwrites the one before it, and the time stamp moves
        sleep(0.01)
        write_manifest(joinpath(d, "run"); extra = Dict{String,Any}("step" => "analyze"))
        m3 = read_manifest(joinpath(d, "run"))
        @test m3["step"] == "analyze" && m3["updated"] > m2["updated"] && m3["inputs"] == m2["inputs"]
        # the file's keys are sorted, so two runs of the same steps write the same bytes
        keys_in_file = [m.captures[1] for m in eachmatch(r"^(\w+) ="m, read(joinpath(d, "run", "manifest.toml"), String))]
        @test issorted(keys_in_file) && "step" in keys_in_file
        # the bytes as they are, against sha256sum: a final newline, a CR, and a file past 1 MiB
        write(joinpath(d, "nl.txt"), "abc\n")
        write(joinpath(d, "crlf.txt"), "a\r\nb\n")
        write(joinpath(d, "big.txt"), repeat("a", 2^21))
        @test file_hash(joinpath(d, "nl.txt")) == "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb"
        @test file_hash(joinpath(d, "crlf.txt")) == "953bba9ac9726eaea07e844abcf144a0afe998039257c7a88b6665819597f39d"
        @test file_hash(joinpath(d, "big.txt")) == "5256ec18f11624025905d057d6befb03d77b243511ac5f77ed5e0221ce6d84b5"
        # three inputs, one given relative but outside, one in a sibling directory whose name
        # starts with the run directory's: both are outside and recorded by the path given
        mkpath(joinpath(d, "run2"))
        sibling = joinpath(d, "run2", "x.txt")
        write(sibling, "x")
        write_manifest(joinpath(d, "run"); inputs = ["../nl.txt", sibling, "case.toml"])
        h = read_manifest(joinpath(d, "run"))["inputs"]
        @test h["../nl.txt"] == "sha256:edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb"
        @test h[sibling] == "sha256:" * file_hash(sibling) && !haskey(h, "../run2/x.txt")
        @test h["case.toml"] == m["inputs"]["case.toml"]
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
    # the first image line wins, an upper-case id is stored in lower case, and a short id or a
    # line that does not start with the header's mark is no image
    two = joinpath(dir, "two.txt")
    write(two, "x # image: other $id\n# image: short id 0123abcd\n# image: first ID $(uppercase(id))\n# image: second $id\n")
    @test image_from_header(two) == (; name = "first", id)
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
        for bad in ("not-a-commit", sha[1:7], "x" * sha, sha * "0")
            withenv("PM_COMMIT" => bad) do
                @test git_commit(d).commit == "unknown"
            end
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
            @test c.commit != sha && occursin(r"^[0-9a-f]{40}$", c.commit) && c.dirty isa Bool
            # under git the manifest drops a commit_source left by a run without git
            mktempdir() do d
                write(joinpath(d, "manifest.toml"), "commit = \"$sha\"\ncommit_source = \"PM_COMMIT\"\n")
                write_manifest(d; root)
                m = read_manifest(d)
                @test m["commit"] == c.commit && !haskey(m, "commit_source") && m["dirty"] == c.dirty
            end
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
        # milliseconds count; with two complete runs stamp_interval gives the first; an end line
        # without a start before it adds nothing
        g = joinpath(d, "two.txt")
        write(g, """
            # run a start: 2026-11-02T10:00:00.250Z
            # run a end: 2026-11-02T10:00:01.500Z
            # run a end: 2026-11-02T10:00:09.000Z
            # run a start: 2026-11-02T10:00:10.000Z
            # run a end: 2026-11-02T10:00:12.000Z
            """)
        @test stamp_intervals(g, "# run a start:", "# run a end:") == [1.25, 2.0]
        @test stamp_interval(g, "# run a start:", "# run a end:") == 1.25
    end
end

using Aqua
@testset "T-1242 RunManifests patches nothing: no piracy, no eval, no stale dependencies or missing compat [A6, WP-092]" begin
    Aqua.test_all(RunManifests)
    # no method added to another module's function from outside, and no code evaluated into one
    srcdir = dirname(pathof(RunManifests))
    for f in filter(endswith(".jl"), readdir(srcdir; join = true))
        @test !occursin(r"@eval|\beval\(|Core\.eval|Base\.eval|include_string|Symbol\(\"ev", read(f, String))
    end
end
