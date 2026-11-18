# Tests of the run-data tools (F3): julia +1.13 --project=scripts scripts/rundata/test/runtests.jl
# Each test builds a throwaway repository with a bare remote in a temporary directory; nothing here
# touches this repository or the network.
using Test
using TOML
using SHA: sha256

const RD = dirname(@__DIR__)
const RUNDATA = joinpath(RD, "run-data.sh")
const BUDGET = joinpath(RD, "report-budget.sh")
const GITENV = ("GIT_AUTHOR_NAME" => "test", "GIT_AUTHOR_EMAIL" => "test@example.invalid",
                "GIT_COMMITTER_NAME" => "test", "GIT_COMMITTER_EMAIL" => "test@example.invalid",
                "GIT_CONFIG_GLOBAL" => "/dev/null", "GIT_CONFIG_NOSYSTEM" => "1")

sh(dir, cmd::Cmd) = read(setenv(addenv(cmd, GITENV...); dir), String)
# exit code and combined output of a command that may fail
function trysh(dir, cmd::Cmd)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(setenv(addenv(cmd, GITENV...); dir)); stdout = out, stderr = out))
    return p.exitcode, String(take!(out))
end
hexsha(path) = bytes2hex(open(sha256, path))

# A clone `work` of a bare `remote.git` whose main is pushed, scripts/runs/ ignored as in this repository.
function fixture(root)
    remote, work = joinpath(root, "remote.git"), joinpath(root, "work")
    sh(root, `git init -q --bare -b main $remote`)
    sh(root, `git clone -q $remote $work`)
    write(joinpath(work, ".gitignore"), "scripts/runs/\n")
    write(joinpath(work, "README.md"), "fixture\n")
    sh(work, `git add .gitignore README.md`)
    sh(work, `git commit -q -m init`)
    sh(work, `git push -q origin HEAD:main`)
    sh(work, `git fetch -q origin`)
    return work
end
function run_dir(work, name)
    d = joinpath(work, "scripts", "runs", name)
    mkpath(joinpath(d, "plots"))
    write(joinpath(d, "manifest.toml"), "commit = \"abc\"\n")
    write(joinpath(d, "plots", "field.png"), rand(UInt8, 3000))
    return d
end

@testset "T-930 publish pushes runs/<tag> with the run directories and writes their published.toml [A6]" begin
    mktempdir() do root
        work = fixture(root)
        d = run_dir(work, "demo")
        out = sh(work, `bash $RUNDATA publish demo-1 scripts/runs/demo`)
        @test occursin("pushed runs/demo-1", out)
        p = TOML.parsefile(joinpath(d, "published.toml"))
        @test p["branch"] == "runs/demo-1"
        @test strip(sh(work, `git ls-remote origin refs/heads/runs/demo-1`))[1:40] == p["commit"]
        paths = sort([f["path"] for f in p["files"]])
        @test paths == ["scripts/runs/demo/manifest.toml", "scripts/runs/demo/plots/field.png"]
        for f in p["files"]
            @test f["sha256"] == hexsha(joinpath(work, f["path"]))
            @test f["bytes"] == filesize(joinpath(work, f["path"]))
        end
        # the commit is HEAD plus the run, and the worktree, its index and its branch are untouched
        @test strip(sh(work, `git rev-parse $(p["commit"])^`)) == strip(sh(work, `git rev-parse HEAD`))
        @test strip(sh(work, `git status --porcelain`)) == ""
        @test strip(sh(work, `git log -1 --format=%s`)) == "init"
    end
end

@testset "T-931 publish refuses a used tag, large or leaking files, an unpushed HEAD and other paths [A6]" begin
    mktempdir() do root
        work = fixture(root)
        run_dir(work, "a")
        sh(work, `bash $RUNDATA publish t1 scripts/runs/a`)
        run_dir(work, "b")
        code, out = trysh(work, `bash $RUNDATA publish t1 scripts/runs/b`)
        @test code == 64 && occursin("exists; pick a new tag", out)
        code, out = trysh(work, `bash $RUNDATA publish t2 scripts/runs/a`)
        @test code == 64 && occursin("published already", out)
        code, out = trysh(work, `bash $RUNDATA publish t2 README.md`)
        @test code == 64 && occursin("name a directory exactly", out)
        code, out = trysh(work, `bash $RUNDATA publish t2 scripts/runs/../x`)
        @test code == 64 && occursin("name a directory exactly", out)
        big = run_dir(work, "big")
        open(io -> truncate(io, 21 * 1024^2), joinpath(big, "dump.bin"), "w")
        code, out = trysh(work, `bash $RUNDATA publish t2 scripts/runs/big`)
        @test code == 64 && occursin("over 20 MB", out)
        leak = run_dir(work, "leak")
        write(joinpath(leak, "log.txt"), "host " * join(["10", "1", "2", "3"], ".") * "\n")
        code, out = trysh(work, `bash $RUNDATA publish t2 scripts/runs/leak`)
        @test code == 64 && occursin("leak-scan pattern", out)
        sh(work, `git commit -q --allow-empty -m local`)
        code, out = trysh(work, `bash $RUNDATA publish t2 scripts/runs/b`)
        @test code == 64 && occursin("not on any pushed branch", out)
        @test isempty(strip(sh(work, `git ls-remote origin refs/heads/runs/t2`)))
    end
end

@testset "T-932 fetch restores a published run and checks every file against its SHA-256 [A6]" begin
    mktempdir() do root
        work = fixture(root)
        d = run_dir(work, "demo")
        sh(work, `bash $RUNDATA publish demo-1 scripts/runs/demo`)
        record = joinpath(root, "published.toml")
        cp(joinpath(d, "published.toml"), record)
        # a fresh clone, as a reviewer has it
        other = joinpath(root, "other")
        sh(root, `git clone -q $(joinpath(root, "remote.git")) $other`)
        out = sh(other, `bash $RUNDATA fetch $record`)
        @test occursin("matches its SHA-256", out)
        @test read(joinpath(other, "scripts/runs/demo/plots/field.png")) == read(joinpath(d, "plots/field.png"))
        # a local file that differs is kept and reported
        write(joinpath(other, "scripts/runs/demo/manifest.toml"), "commit = \"changed\"\n")
        code, out = trysh(other, `bash $RUNDATA fetch $record`)
        @test code == 64 && occursin("MISMATCH scripts/runs/demo/manifest.toml", out)
        # a record naming a commit that is not on its branch
        p = TOML.parsefile(record)
        p["commit"] = repeat("0", 40)
        open(io -> TOML.print(io, p), joinpath(root, "forged.toml"), "w")
        code, out = trysh(other, `bash $RUNDATA fetch $(joinpath(root, "forged.toml"))`)
        @test code == 64 && occursin("is not on runs/demo-1", out)
        # by tag: the run directories of the branch, never over a local one
        third = joinpath(root, "third")
        sh(root, `git clone -q $(joinpath(root, "remote.git")) $third`)
        out = sh(third, `bash $RUNDATA fetch demo-1`)
        @test strip(out) == "scripts/runs/demo"
        @test isfile(joinpath(third, "scripts/runs/demo/manifest.toml"))
        code, out = trysh(third, `bash $RUNDATA fetch demo-1`)
        @test code == 64 && occursin("exists already", out)
        code, out = trysh(third, `bash $RUNDATA fetch no-such-tag`)
        @test code == 64 && occursin("no branch runs/no-such-tag", out)
    end
end

@testset "T-933 report-budget holds files added or changed under reports/ to 1 MB each and 5 MB in all [A6]" begin
    mktempdir() do root
        work = fixture(root)
        base = strip(sh(work, `git rev-parse HEAD`))
        commit!(msg) = (sh(work, `git add -A`); sh(work, `git commit -q -m $msg`); strip(sh(work, `git rev-parse HEAD`)))
        mkpath(joinpath(work, "reports", "r"))
        write(joinpath(work, "reports", "r", "fig.png"), rand(UInt8, 900_000))
        write(joinpath(work, "reports", "r.md"), "text\n")
        write(joinpath(work, "elsewhere.bin"), rand(UInt8, 2_000_000))   # outside reports/: not counted
        small = commit!("small")
        code, out = trysh(work, `bash $BUDGET $base $small`)
        @test code == 0 && occursin("900005 bytes", out)
        write(joinpath(work, "reports", "r", "big.h5"), rand(UInt8, 1_000_001))
        big = commit!("big")
        code, out = trysh(work, `bash $BUDGET $small $big`)
        @test code == 1 && occursin("over 1 MB: reports/r/big.h5", out)
        rm(joinpath(work, "reports", "r", "big.h5"))
        removed = commit!("removed")
        @test first(trysh(work, `bash $BUDGET $big $removed`)) == 0           # a deletion costs nothing
        for i in 1:5
            write(joinpath(work, "reports", "r", "p$i.png"), rand(UInt8, 900_000))
        end
        many = commit!("many")
        code, out = trysh(work, `bash $BUDGET $removed $many`)
        @test code == 0                                                      # 4.5 MB
        code, out = trysh(work, `bash $BUDGET $base $many`)
        @test code == 1 && occursin("over the 5 MB budget", out)              # 5.4 MB since base
        write(joinpath(work, "reports", "r", "fig.png"), rand(UInt8, 1_200_000))
        changed = commit!("changed")
        code, out = trysh(work, `bash $BUDGET $many $changed`)
        @test code == 1 && occursin("over 1 MB: reports/r/fig.png", out)     # a changed file counts too
        @test first(trysh(work, `bash $BUDGET $base`)) == 64
    end
end
