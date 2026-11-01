# Shared plumbing of the 1D electromagnetic runs (WP-014): case loading, run directories with a
# manifest (F3), PNG + JSON sidecar output, host-side checks. Used by case.jl, timing.jl,
# crosscode.jl and flux_timing.jl.

using ParticleMesh
using ParticleMeshElectromagnetic
using KernelAbstractions
using CairoMakie
using JSON
using TOML
using Dates

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const EM_TEST = joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test")
const CASES = joinpath(REPO, "scripts", "crosscode", "cases")

include(joinpath(REPO, "scripts", "crosscode", "manifest.jl"))
using .RunManifest: write_manifest
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))
using .Observables: damping_from_peaks, growth_from_window
include(joinpath(EM_TEST, "theory.jl"))             # Theory (Landau, two-stream) and weibel_cold_growth; reads REPO

load_case(name) = TOML.parsefile(joinpath(CASES, "$name.toml"))

"Files whose content defines a run: the engine and module sources, the case (if any), the theory, the scripts, the manifests."
function input_files(script, case_name)
    files = String[]
    for pkg in ("ParticleMeshCore", "ParticleMeshElectromagnetic")
        for (root, _, fs) in walkdir(joinpath(REPO, "lib", pkg, "src")), f in fs
            endswith(f, ".jl") && push!(files, joinpath(root, f))
        end
    end
    case_name === nothing || push!(files, joinpath(CASES, "$case_name.toml"))
    append!(files, [joinpath(EM_TEST, "theory.jl"),
                    joinpath(REPO, "scripts", "crosscode", "theory.jl"), joinpath(REPO, "scripts", "crosscode", "analysis.jl"),
                    joinpath(@__DIR__, "common.jl"), script,
                    joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")])
    return sort(files)
end

"""
    run_dir(name, script, case_name; params, work_package = "WP-014") -> dir

Create `scripts/runs/em1d-<name>-<timestamp>/` and write its `manifest.toml` (F3): the commit
and dirty flag, the SHA-256 of every input (the case file among them unless `case_name` is
`nothing`), the parameters, the work package, Julia version, thread count, CPU model and
backend, plus `extra` (the GPU model of a CUDA run).
"""
function run_dir(name, script, case_name; params = Dict{String,Any}(), work_package = "WP-014",
                 backend = "KernelAbstractions CPU", extra = Dict{String,Any}())
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    dir = joinpath(REPO, "scripts", "runs", "em1d-$name-$stamp")
    mkpath(dir)
    # input paths relative to the repository root: absolute paths would carry the machine's home directory
    inputs = Dict{String,Any}(relpath(f, REPO) => "sha256:" * RunManifest.file_hash(f) for f in input_files(script, case_name))
    write_manifest(dir; root = REPO,
                   extra = merge(Dict{String,Any}("inputs" => inputs, "run_id" => basename(dir), "work_package" => work_package,
                                                  "parameters" => params, "julia" => string(VERSION), "threads" => Threads.nthreads(),
                                                  "cpu" => Sys.cpu_info()[1].model, "backend" => backend), extra))
    return dir
end

"Save `fig` as `<dir>/<name>.png` next to `<dir>/<name>.json` holding `sidecar` (the plotted scalars)."
function save_plot(dir, name, fig, sidecar)
    save(joinpath(dir, "$name.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, sidecar, 2), joinpath(dir, "$name.json"), "w")
    return joinpath(dir, "$name.png")
end

"Discrete Gauss residual max|D⁻Ex − (ρ − mean ρ)| and its scale n·max|ρ| (host check, between steps)."
function gauss_check(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    ρ = field(CPU(), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρ, s, numerics.shape, RHO_STAGGER, grid, ws)
    end
    fold_guards!(ρ, grid)
    g, n, dx = guards(grid), ncells(grid), cellsize(grid)
    r = ρ[(g + 1):(g + n)]
    E = Array(st.fields.Ex)
    ρ̄ = sum(r) / n
    res = maximum(abs((E[g + j] - E[g + j - 1]) / dx - (r[j] - ρ̄)) for j in 1:n)
    return res, n * maximum(abs, r)
end

"Phase-space samples (x, u_component) of every alive particle of the state, every `stride`-th."
function phase_space(st, comp; stride = 1)
    xs, us = Float64[], Float64[]
    dx = cellsize(st.grid)
    for s in st.species
        alive = findall(==(0x01), Array(s.status))[1:stride:end]
        append!(xs, (Array(s.cell[1])[alive] .- 1 .+ Array(s.offset[1])[alive]) .* dx)
        append!(us, Array(s.u[comp])[alive])
    end
    return xs, us
end
