# Run manifests (F3): the commit, the inputs with their hashes, the image, the hardware.

module RunManifest

import Dates

using SHA: sha256
using TOML
using Dates: now, UTC

export file_hash, write_manifest, read_manifest, git_commit, image_from_header, stamp_interval, stamp_intervals

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

"""
    image_from_header(files...) -> (; name, id)

The container image of a run from the first `# image:` line found in `files` (the runner's
log, whose header reads `# image: <name> id <id>`, or run-smilei.sh's `image.txt`, which reads
`# image: <name> <id>`; `nothing` entries and missing files are skipped). `id` is the bare
64-hex image id, without a `sha256:` prefix; both fields are `"unknown"` when no line is found.
"""
function image_from_header(files...)
    for f in files
        (f === nothing || !isfile(f)) && continue
        for line in eachline(f)
            m = match(r"^#\s*image:\s*(\S+)\s+(?:id\s+)?(?:sha256:)?([0-9a-f]{64})\s*$"i, line)
            m === nothing || return (; name = String(m.captures[1]), id = lowercase(m.captures[2]))
        end
    end
    return (; name = "unknown", id = "unknown")
end

"""
Seconds between a `# <what> start:` line and the first `end:` line after it in a
container.txt, from the last start line before that end (a run that was killed and run again
in a later job, which appends to the file, starts more than once), or nothing.
"""
function stamp_interval(f, start_prefix, end_prefix)
    intervals = stamp_intervals(f, start_prefix, end_prefix)
    return isempty(intervals) ? nothing : first(intervals)
end

"Every start-to-end interval of `stamp_interval` in a container.txt, in order (one per job for the compile)."
function stamp_intervals(f, start_prefix, end_prefix)
    isfile(f) || return Float64[]
    stamp(l, prefix) = Dates.DateTime(strip(l[length(prefix)+1:end])[1:23])
    out, t0 = Float64[], nothing
    for l in eachline(f)
        if startswith(l, start_prefix)
            t0 = stamp(l, start_prefix)
        elseif startswith(l, end_prefix) && t0 !== nothing
            push!(out, (stamp(l, end_prefix) - t0) / Dates.Millisecond(1000))
            t0 = nothing
        end
    end
    return out
end

end
