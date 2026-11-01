# Shared plumbing of the 1D electrostatic runs (WP-010; the reference moved to test support in
# WP-011): run directories, manifests (F3), PNG + JSON sidecar output. Used by landau.jl,
# two_stream.jl, gradient.jl, timing.jl.

using ParticleMesh
using KernelAbstractions
using CairoMakie
using JSON
using SHA
using Dates

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CORE_TEST = joinpath(REPO, "lib", "ParticleMeshCore", "test")
include(joinpath(CORE_TEST, "es_reference.jl"))     # the electrostatic reference (WP-011: test support)
using .ESReference
include(joinpath(CORE_TEST, "cases.jl"))
include(joinpath(CORE_TEST, "theory.jl"))

"Files whose content defines a run: the core sources, the electrostatic reference, the case definitions, the theory, the script."
function input_files(script)
    src = joinpath(REPO, "lib", "ParticleMeshCore", "src")
    files = String[]
    for (root, _, fs) in walkdir(src), f in fs
        endswith(f, ".jl") && push!(files, joinpath(root, f))
    end
    append!(files, [joinpath(CORE_TEST, "es_reference.jl"), joinpath(CORE_TEST, "cases.jl"), joinpath(CORE_TEST, "theory.jl"),
                    joinpath(@__DIR__, "common.jl"),
                    script, joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")])
    return sort(files)
end

git(args...) = try
    readchomp(Cmd(`git -C $REPO $args`))
catch
    "unknown"
end

"""
    run_dir(name, script; params, work_package = "WP-011", extra_inputs = String[]) -> dir

Create `scripts/runs/<name>-<timestamp>/` and write its `manifest.json`: commit, dirty flag,
the SHA-256 of every input file (`input_files(script)` and `extra_inputs`), the parameters, Julia version, thread count and CPU model.
"""
function run_dir(name, script; params = Dict(), work_package = "WP-011", extra_inputs = String[])
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    dir = joinpath(REPO, "scripts", "runs", "$name-$stamp")
    mkpath(dir)
    manifest = Dict(
        "run_id" => "$name-$stamp",
        "work_package" => work_package,
        "commit" => git("rev-parse", "HEAD"),
        "dirty" => !isempty(git("status", "--porcelain", "--", "lib", "src", "scripts/es1d", "Manifest.toml", "scripts/Manifest.toml")),
        "inputs" => Dict(relpath(f, REPO) => bytes2hex(open(sha256, f)) for f in sort(unique([input_files(script); extra_inputs]))),
        "parameters" => params,
        "julia" => string(VERSION),
        "threads" => Threads.nthreads(),
        "cpu" => Sys.cpu_info()[1].model,
        "backend" => "KernelAbstractions CPU",
        "created_utc" => string(now(UTC)),
    )
    open(io -> JSON.print(io, manifest, 2), joinpath(dir, "manifest.json"), "w")
    return dir
end

"Save `fig` as `<dir>/<name>.png` next to `<dir>/<name>.json` holding `sidecar` (the plotted scalars)."
function save_plot(dir, name, fig, sidecar)
    save(joinpath(dir, "$name.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, sidecar, 2), joinpath(dir, "$name.json"), "w")
    return joinpath(dir, "$name.png")
end

"Host-side diagnostics of a state: kinetic and field energy, Gauss residual, total charge."
function host_diagnostics(st)
    grid = st.grid
    n, g, dx = ncells(grid), guards(grid), cellsize(grid)
    ρ = Array(st.ρ)[(g + 1):(g + n)]
    E = Array(st.Ehalf)
    ρ̄ = sum(ρ) / n
    gauss = maximum(abs((E[j] - E[mod1(j - 1, n)]) / dx - (ρ[j] - ρ̄)) for j in 1:n)
    return (; kinetic = kinetic_energy(st), field = sum(abs2, E) * dx / 2, gauss, gauss_scale = n * maximum(abs, ρ), mean_E = sum(E) / n)
end

"Phase-space samples (x, v_x) of every alive particle of the species in `st`."
function phase_space(st; stride = 1)
    xs, vs = Float64[], Float64[]
    dx = cellsize(st.grid)
    for s in st.species
        alive = findall(==(0x01), Array(s.status))[1:stride:end]
        append!(xs, (Array(s.cell[1])[alive] .- 1 .+ Array(s.offset[1])[alive]) .* dx)
        append!(vs, Array(s.u[1])[alive])
    end
    return xs, vs
end

"Run in chunks of `every` steps, collecting host diagnostics after each chunk; returns their times and values."
function run_with_diagnostics!(st, p, nsteps; every = 10, snapshots = Int[], on_snapshot = (k, st) -> nothing)
    ts, diags = Float64[], NamedTuple[]
    k = 1
    while k <= nsteps
        m = min(every, nsteps - k + 1)
        run!(st, p, m; first = k)
        k += m
        # after the chunk the fields are those of step k-1 (positions of step k-2 → see step!)
        push!(ts, (k - 2) * Float64(p.dt))
        push!(diags, host_diagnostics(st))
        (k - 1) in snapshots && on_snapshot(k - 1, st)
    end
    return ts, diags
end
