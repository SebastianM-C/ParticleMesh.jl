# Run manifests (F3): the commit, the inputs with their hashes, the image, the hardware.

module RunManifest

using SHA: sha256
using TOML
using Dates: now, UTC

export file_hash, write_manifest, read_manifest, git_commit

file_hash(path) = bytes2hex(open(sha256, path))

"""
    git_commit(root) -> (; commit, dirty)

The commit of the checkout at `root` and whether tracked files differ from it. Read through
`git` when it is available; `"unknown"` otherwise (a run inside the container sees the worktree
read-only and may have no git).
"""
function git_commit(root)
    commit = try
        readchomp(`git -C $root rev-parse HEAD`)
    catch
        "unknown"
    end
    dirty = try
        !isempty(readchomp(`git -C $root status --porcelain --untracked-files=no`))
    catch
        missing
    end
    return (; commit, dirty)
end

"""
    write_manifest(dir; inputs, extra) -> path

Write `dir/manifest.toml` with the commit, the hash of every file in `inputs` (paths relative
to `dir` or absolute; recorded relative to `dir` when inside it) and the entries of `extra`
(a dictionary, merged at the top level). Existing entries of a manifest in `dir` are kept
unless overwritten, so the prepare and analyze steps add to one manifest.
"""
function write_manifest(dir; inputs = String[], extra = Dict{String,Any}(), root = nothing)
    path = joinpath(dir, "manifest.toml")
    m = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    if root !== nothing
        c = git_commit(root)
        m["commit"] = c.commit
        c.dirty === missing || (m["dirty"] = c.dirty)
    end
    hashes = get(m, "inputs", Dict{String,Any}())
    for p in inputs
        full = isabspath(p) ? p : joinpath(dir, p)
        rel = startswith(abspath(full), abspath(dir)) ? relpath(full, dir) : p
        hashes[rel] = "sha256:" * file_hash(full)
    end
    m["inputs"] = hashes
    m["updated"] = string(now(UTC)) * "Z"
    merge!(m, extra)
    open(io -> TOML.print(io, m; sorted = true), path, "w")
    return path
end

read_manifest(dir) = TOML.parsefile(joinpath(dir, "manifest.toml"))

end
