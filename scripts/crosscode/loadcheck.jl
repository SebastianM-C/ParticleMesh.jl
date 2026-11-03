# Do two Smilei builds load the same particles for one seed? (WP-030)
#
#   julia +1.13 --project=scripts scripts/crosscode/loadcheck.jl <run a> <run b> <out dir>
#
# Compares two load-check runs (`smilei.jl prepare <case> --load-check [--gpu] [--patches ..]`):
# the t = 0 particles of the box's lower-left corner that DiagTrackParticles dumped
# (TrackParticlesDisordered_<species>.h5), matched by position, the scalars of the first output
# and the fields at t = 0. Writes <out dir>/loadcheck.json. A measurement: the numbers say
# whether the loads are identical, bit for bit, or by how much they differ.

include(joinpath(@__DIR__, "read.jl"))
include(joinpath(@__DIR__, "manifest.jl"))

module LoadCheck

using HDF5, JSON, TOML
using ..SmileiRead, ..RunManifest

export read_track, compare_tracks, compare_loads

"""
    read_track(file; iteration = 0) -> Dict{String,Vector{Float64}}

The particles of one iteration of a Smilei `TrackParticlesDisordered_<species>.h5` (openPMD
layout of DiagnosticTrack.cpp: /data/<iteration, 10 digits>/particles/<species>/ with
position/{x,y}, momentum/{x,y,z} and weight), as "x", "y", "px", "py", "pz", "w".
"""
function read_track(file; iteration = 0)
    h5open(file, "r") do f
        it = lpad(string(iteration), 10, '0')
        g = f["data"][it]["particles"]
        sp = only(keys(g))
        s = g[sp]
        out = Dict{String,Vector{Float64}}()
        for (name, path) in (("x", ("position", "x")), ("y", ("position", "y")), ("px", ("momentum", "x")),
                             ("py", ("momentum", "y")), ("pz", ("momentum", "z")))
            out[name] = read(s[path[1]][path[2]])
        end
        out["w"] = read(s["weight"])
        return out
    end
end

"The permutation that orders particles by (x, y), with a check that no two share a position."
function position_order(t)
    p = sortperm(collect(zip(t["x"], t["y"])))
    xs, ys = t["x"][p], t["y"][p]
    any(i -> xs[i] == xs[i+1] && ys[i] == ys[i+1], 1:length(p)-1) &&
        error("two particles share a position; matching by position needs a regular load")
    return p
end

"""
    compare_tracks(a, b) -> Dict

Particle-by-particle comparison of two loads read by `read_track`, matched by position (a
`regular` load puts every particle of a cell at its own lattice point): the counts, whether the
positions are the same set exactly, and per momentum component and weight the number of
particles that differ, the largest absolute difference and the largest relative to max |p|.
"""
function compare_tracks(a, b)
    r = Dict{String,Any}("particles" => [length(a["x"]), length(b["x"])])
    length(a["x"]) == length(b["x"]) || (r["same_positions"] = false; r["identical"] = false; return r)
    pa, pb = position_order(a), position_order(b)
    same = a["x"][pa] == b["x"][pb] && a["y"][pa] == b["y"][pb]
    r["same_positions"] = same
    if !same
        r["max_abs_position_difference"] = max(maximum(abs, a["x"][pa] .- b["x"][pb]), maximum(abs, a["y"][pa] .- b["y"][pb]))
        r["identical"] = false
        return r
    end
    identical = true
    for k in ("px", "py", "pz", "w")
        u, v = a[k][pa], b[k][pb]
        d = abs.(u .- v)
        ndiff = count(u .!== v)
        identical &= ndiff == 0
        scale = maximum(abs, u)
        r[k] = Dict{String,Any}("differing" => ndiff, "max_abs_difference" => maximum(d),
            "max_relative_to_max_abs" => scale > 0 ? maximum(d) / scale : 0.0)
    end
    r["identical"] = identical
    return r
end

"The largest |a − b| of every column of the first scalar output, absolute and relative to |a|."
function compare_first_scalars(da, db)
    sa, sb = read_scalars(da), read_scalars(db)
    out = Dict{String,Any}()
    for k in sort(collect(keys(sa.columns)))
        haskey(sb.columns, k) || continue
        x, y = sa.columns[k][1], sb.columns[k][1]
        out[k] = Dict{String,Any}("a" => x, "b" => y, "difference" => y - x,
            "relative" => x == 0 ? (y == 0 ? 0.0 : nothing) : (y - x) / abs(x))
    end
    return out
end

"max |a − b| of every field dumped at t = 0, absolute and relative to max |a|."
function compare_first_fields(da, db)
    fa, fb = read_fields(da), read_fields(db)
    out = Dict{String,Any}()
    for k in sort(collect(keys(fa.fields)))
        haskey(fb.fields, k) || continue
        A, B = selectdim(fa.fields[k], ndims(fa.fields[k]), 1), selectdim(fb.fields[k], ndims(fb.fields[k]), 1)
        d = maximum(abs, A .- B)
        out[k] = Dict{String,Any}("max_abs_difference" => d, "max_abs" => maximum(abs, A),
            "relative" => maximum(abs, A) > 0 ? d / maximum(abs, A) : 0.0, "bitwise_equal" => A == B)
    end
    return out
end

"The build and patch layout of a run, from its manifest."
function run_build(dir)
    m = read_manifest(dir)
    return Dict{String,Any}("run" => basename(abspath(dir)), "build" => get(m, "build", "cpu"),
        "patches" => get(m, "patches", nothing), "fastmath" => get(m, "fastmath", nothing),
        "load_check" => get(m, "load_check", false))
end

"""
    compare_loads(a, b, out) -> Dict

The load check of two runs (see the file header), written to <out>/loadcheck.json.
"""
function compare_loads(a, b, out)
    tracks = sort(filter(f -> startswith(f, "TrackParticlesDisordered_") && endswith(f, ".h5"), readdir(a)))
    isempty(tracks) && error("$a: no TrackParticlesDisordered_*.h5; was it prepared with --load-check?")
    r = Dict{String,Any}("a" => run_build(a), "b" => run_build(b), "species" => Dict{String,Any}())
    for f in tracks
        sp = replace(f, "TrackParticlesDisordered_" => "", ".h5" => "")
        isfile(joinpath(b, f)) || error("$b: no $f")
        r["species"][sp] = compare_tracks(read_track(joinpath(a, f)), read_track(joinpath(b, f)))
    end
    r["identical"] = all(v -> v["identical"], values(r["species"]))
    r["scalars_first_output"] = compare_first_scalars(a, b)
    r["fields_t0"] = compare_first_fields(a, b)
    r["note"] = "particles of the lower-left eighth of the box per axis at t = 0, matched by position; a measurement"
    mkpath(out)
    open(io -> JSON.print(io, r, 2), joinpath(out, "loadcheck.json"), "w")
    return r
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .LoadCheck, JSON
    length(ARGS) == 3 || (println(stderr, "usage: loadcheck.jl <run a> <run b> <out dir>"); exit(2))
    JSON.print(stdout, compare_loads(ARGS...), 2)
    println()
end
