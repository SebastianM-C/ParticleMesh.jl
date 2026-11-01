# Shared plumbing of the WP-018 measurements (the engine at N = 2): run directories with a
# manifest (F3), PNG + JSON sidecar output, and a repeated timer. Used by n1_bits.jl and
# timing.jl.

using ParticleMesh
using KernelAbstractions
using CairoMakie
using JSON
using TOML
using Dates
using Statistics

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CORE_TEST = joinpath(REPO, "lib", "ParticleMeshCore", "test")
const EM_TEST = joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test")

include(joinpath(REPO, "scripts", "crosscode", "manifest.jl"))
using .RunManifest: write_manifest

"Files whose content defines a run: the engine and module sources, the test support the run includes, these scripts, the manifests."
function input_files(script; extra = String[])
    files = String[]
    for pkg in ("ParticleMeshCore", "ParticleMeshElectromagnetic")
        for (root, _, fs) in walkdir(joinpath(REPO, "lib", pkg, "src")), f in fs
            endswith(f, ".jl") && push!(files, joinpath(root, f))
        end
    end
    append!(files, extra)
    append!(files, [joinpath(@__DIR__, "common.jl"), script, joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")])
    return sort(unique(files))
end

"""
    run_dir(name, script; params, extra_inputs) -> dir

Create `scripts/runs/engine2d-<name>-<timestamp>/` and write its `manifest.toml` (F3): the
commit and dirty flag, the SHA-256 of every input (paths relative to the repository root), the
parameters, Julia version, thread count and CPU model.
"""
function run_dir(name, script; params = Dict{String,Any}(), extra_inputs = String[])
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    dir = joinpath(REPO, "scripts", "runs", "engine2d-$name-$stamp")
    mkpath(dir)
    inputs = Dict{String,Any}(relpath(f, REPO) => "sha256:" * RunManifest.file_hash(f) for f in input_files(script; extra = extra_inputs))
    write_manifest(dir; root = REPO,
                   extra = Dict{String,Any}("inputs" => inputs, "run_id" => basename(dir), "work_package" => "WP-018", "parameters" => params,
                                            "julia" => string(VERSION), "threads" => Threads.nthreads(),
                                            "cpu" => Sys.cpu_info()[1].model, "backend" => "KernelAbstractions CPU"))
    return dir
end

"Save `fig` as `<dir>/<name>.png` next to `<dir>/<name>.json` holding `sidecar` (the plotted numbers)."
function save_plot(dir, name, fig, sidecar)
    save(joinpath(dir, "$name.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, sidecar, 2), joinpath(dir, "$name.json"), "w")
    return joinpath(dir, "$name.png")
end

"""
    timed(f; reps, inner) -> (; median, min, max, samples)

Seconds per call of `f()`: one warm-up call, then `reps` samples, each the mean over `inner`
calls. The machine is a shared cloud VM, so the spread (`min`, `max`) is reported with the median.
"""
function timed(f; reps = 7, inner = 5)
    f()
    samples = map(1:reps) do _
        t = @elapsed for _ in 1:inner
            f()
        end
        t / inner
    end
    return (; median = median(samples), min = minimum(samples), max = maximum(samples), samples)
end

cpu_model() = Sys.cpu_info()[1].model
