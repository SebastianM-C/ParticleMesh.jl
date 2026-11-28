# Why Fast mode (atomic deposition) does not pay off in 3D, and what would make it (#550).
#
#   julia +1.13 --project=scripts scripts/fast3d/fast3d.jl run <cpu|cuda> <tag> [key=value ...]
#   julia +1.13 --project=scripts scripts/fast3d/fast3d.jl check                      # CPU, the variants against the module's step!
#   julia +1.13 --project=scripts scripts/fast3d/fast3d.jl plot <run dir>              # figures with their sidecars
#
# The case is WP-090's `uniform` plasma (scripts/bench3d/cases/gpu-3d.toml): electrons on a quiet
# lattice in a periodic cube of λ₀/32 cells, 0.072 n_c, v_th = 0.05 c per component, Δt at 0.95 of
# the 3D Courant limit; quadratic shape, Esirkepov, Boris, Yee, no filter, no output in the loop.
# Each size of `cells` runs every variant of `variants` in one process:
#
# - `reference`: the module's step in Reference mode (the full sort of every step, the patch flux);
# - `fast`: the module's step in Fast mode (atomic flux, the particles never sorted);
# - `fast-nz`: Fast, with the atomic flux kernel skipping the adds whose value is exactly zero (in
#   the particle's 4-node frame at order 2 a particle that crosses no face along an axis has an
#   empty node along it, so up to 90 of the 144 adds are zeros); bitwise Fast's on one thread;
# - `fast-sort<K>`, `fast-nz-sort<K>`: either, with `sort_by_cell!` before the push every K steps
#   (the gather and the atomics then see the particles in cell order again).
#
# Per run: step 1 (it compiles), then `split` steps with the device synchronized around each phase
# (the early split: push with gather, sort, current deposition with fold, field solve), then the
# block-timed steps to `steps` (the device synchronized every `block` steps: the trajectory of
# seconds per step, which shows Fast's slowdown as the particles lose the load's order), then
# `split` more phase-timed steps (the late split). The fraction of the flux values that are exact
# zeros is counted on the last state. Outputs: scripts/runs/fast3d-<tag>/results.json, table.md
# and manifest.toml.
#
# Keys (defaults): cells=64,96,128 (cubes) ppc=2x2x2 steps=400 block=20 split=5 float=32
# variants=reference,fast,fast-nz,fast-sort10,fast-nz-sort10,fast-sort50

using ParticleMesh, ParticleMeshElectromagnetic, KernelAbstractions
using JSON, TOML, Dates
const PME = ParticleMeshElectromagnetic
const PMC = ParticleMesh.ParticleMeshCore
using .PMC: Atomix
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "scripts", "crosscode", "manifest.jl"))
using .RunManifest: write_manifest

const CMD = isempty(ARGS) ? "check" : ARGS[1]
if CMD == "run" && length(ARGS) >= 2 && ARGS[2] == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
end

# --- the case -------------------------------------------------------------------------------------

"WP-090's uniform plasma (`uniform_problem` of scripts/bench3d/bench3d.jl) on `cells` with `ppc`, in T."
function uniform_problem(cells::NTuple{3,Int}, ppc::NTuple{3,Int}, steps; T = Float32)
    dx = 2π / 32
    dt = 0.95 / sqrt(3 / dx^2)
    L = ntuple(k -> T(cells[k] * dx), 3)
    v = T(0.05)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(0.072), zero(T), (1, 0, 0), ppc, (zero(T), zero(T), zero(T)), (v, v, v), 1)
    return EMProblem("fast3d-$(join(cells, 'x'))", L, cells, T(dt), steps, (e,), 1, (1, 0, 0))
end

numerics(mode) = EMNumerics(; shape = Quadratic, mode, pusher = Boris(), solver = Yee(), current = Esirkepov())

"A variant's name parsed: the module mode, whether the flux skips zeros, the sort period (0: never)."
function parse_variant(v::AbstractString)
    v == "reference" && return (; name = v, mode = Reference(), nz = false, every = 0)
    m = match(r"^fast(-nz)?(?:-sort(\d+))?$", v)
    m === nothing && error("variant $v: reference, fast, fast-nz, fast-sort<K> or fast-nz-sort<K>")
    return (; name = v, mode = Fast(), nz = m[1] !== nothing, every = m[2] === nothing ? 0 : parse(Int, m[2]))
end

# --- the zero-skipping atomic flux (a prototype of a change to lib/ParticleMeshCore/src/flux3d.jl) --

@inline function atomic_add_block_nz!(J, v::NTuple{K,Any}, b::NTuple{3,Int}, ::Val{D}) where {K,D}
    ci = CartesianIndices(D)
    for k in 1:K
        m = Tuple(ci[k])
        x = v[k]
        if !iszero(x)
            @inbounds Atomix.@atomic :monotonic J[CartesianIndex(b[1] + m[1] - 1, b[2] + m[2] - 1, b[3] + m[3] - 1)] += x
        end
    end
    return nothing
end

@inline function atomic_component_nz!(J, vC::Val{C}, s, i, shape::PMC.Shape{P}, st, n, g::Int32, sx) where {C,P}
    lo, parts = PMC.flux_parts_3d(vC, s, i, shape, st, n, sx, PMC.ParticleFrame())
    D = PMC.flux_dims_3d(vC, (P + 2, P + 2, P + 2))
    v = PMC.flux_values_3d(vC, parts, Val(0), Val(prod(D)))
    c = PMC.particle_cell(s, i, Val(3))
    atomic_add_block_nz!(J, v, ntuple(d -> Int(c[d] + lo[d] + g), 3), Val(D))
    return nothing
end

@kernel function flux_atomic_nz_kernel!(Jx, Jy, Jz, s, shape, st, n::NTuple{3,Int32}, g::Int32, sx)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == PMC.ALIVE
        atomic_component_nz!(Jx, Val(1), s, i, shape, st, n, g, sx)
        atomic_component_nz!(Jy, Val(2), s, i, shape, st, n, g, sx)
        atomic_component_nz!(Jz, Val(3), s, i, shape, st, n, g, sx)
    end
end

@inline function count_nonzero(vC::Val{C}, s, i, shape::PMC.Shape{P}, st, n, sx) where {C,P}
    _, parts = PMC.flux_parts_3d(vC, s, i, shape, st, n, sx, PMC.ParticleFrame())
    D = PMC.flux_dims_3d(vC, (P + 2, P + 2, P + 2))
    v = PMC.flux_values_3d(vC, parts, Val(0), Val(prod(D)))
    k = Int32(0)
    for x in v
        k += !iszero(x)
    end
    return k
end

@kernel function count_nonzero_kernel!(out, s, shape, st, n::NTuple{3,Int32}, sx)
    i = @index(Global, Linear)
    @inbounds out[i] = s.status[i] == PMC.ALIVE ?
        count_nonzero(Val(1), s, i, shape, st, n, sx) + count_nonzero(Val(2), s, i, shape, st, n, sx) + count_nonzero(Val(3), s, i, shape, st, n, sx) :
        Int32(0)
end

"The mean count of nonzero flux values per alive particle on the state's last move, of the 3(P + 1)(P + 2)² adds of the atomic path."
function nonzero_per_particle(st)
    s = only(st.species)
    be = get_backend(s)
    out = KernelAbstractions.zeros(be, Int32, PMC.capacity(s))
    sx = map(h -> h / st.dt, PMC.cellsize(st.grid))
    count_nonzero_kernel!(be)(out, s, st.numerics.shape, PME.rho_stagger(Val(3)), Int32.(PMC.ncells(st.grid)), sx; ndrange = PMC.capacity(s))
    KernelAbstractions.synchronize(be)
    return sum(Int64, out) / alive_count(s)
end

# --- the step, phase by phase ---------------------------------------------------------------------

const PHASES = (:sort, :push, :deposit, :fields)

function phase!(f, times, phase, be)
    times === nothing && return f()
    KernelAbstractions.synchronize(be)
    t0 = time_ns()
    f()
    KernelAbstractions.synchronize(be)
    times[phase] += (time_ns() - t0) / 1e9
    return nothing
end

"""
    variant_step!(st, p, k, v; times = nothing)

The module's `step!` (lib/ParticleMeshElectromagnetic/src/step.jl) for the periodic 3D plasma (no
witness, dopant, window, open axis, modifier, filter or observer), with the variant's sort and
flux; with `times`, the device synchronized around each phase and its seconds added there.
`check` holds it to `step!`'s bits for `reference` and `fast`.
"""
function variant_step!(st, p, k, v; times = nothing)
    (; numerics, grid, fields, dt) = st
    s, ws = only(st.species), only(st.ws)
    be = get_backend(s)
    n = Int32.(grid.n)
    g = Int32(PMC.guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    T = eltype(st.ρ)
    st.fresh[] = false
    if v.every > 0 && (k - 1) % v.every == 0
        phase!(() -> PMC.sort_by_cell!(s, grid, ws), times, :sort, be)
    end
    pf = PME.particle_fields(st)
    phase!(times, :push, be) do
        PMC.launcher(PME.push_kernel!, be, nothing)(s, pf, numerics.pusher, PME.read_shape(numerics), g, s.q / s.m * dt / 2, dt_dx, n;
                                                    ndrange = PMC.capacity(s))
    end
    PME.sorts(st, s) && phase!(() -> PMC.resort_by_cell!(s, grid, ws), times, :sort, be)
    phase!(times, :deposit, be) do
        fill!(fields.Jx, zero(T))
        fill!(fields.Jy, zero(T))
        fill!(fields.Jz, zero(T))
        if v.nz
            sx = map(h -> h / dt, PMC.cellsize(grid))
            flux_atomic_nz_kernel!(be)(fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, PME.rho_stagger(Val(3)), n, g, sx;
                                       ndrange = PMC.capacity(s))
        else
            deposit_flux!(PME.deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, PME.rho_stagger(Val(3)), grid, ws; dt)
        end
        foreach(J -> PMC.fold_guards!(J, grid), (fields.Jx, fields.Jy, fields.Jz))
    end
    phase!(() -> PME.solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k), times, :fields, be)
    return st
end

sync(be) = KernelAbstractions.synchronize(be)

function new_state(cells, ppc, steps, v, be, T)
    prob = uniform_problem(cells, ppc, steps; T)
    st = EMState(prob, numerics(v.mode); record = (), backend = be)
    p = params(prob)
    initialize!(st, p)
    sync(be)
    return st, p
end

function split_steps!(st, p, v, be, ks)
    times = Dict(ph => 0.0 for ph in PHASES)
    for k in ks
        variant_step!(st, p, k, v; times)
    end
    return Dict(String(ph) => 1e3 * times[ph] / length(ks) for ph in PHASES)    # ms per step
end

"One variant on one size: the early and late splits, the block trajectory, the zero fraction."
function run_one(cells, ppc, steps, block, split, v, be, T)
    t0 = time()
    st, p = new_state(cells, ppc, steps + 2split + 1, v, be, T)
    np = alive_count(only(st.species))
    t_ready = time()
    variant_step!(st, p, 1, v)
    sync(be)
    t_first = time() - t_ready
    early = split_steps!(st, p, v, be, 2:(split + 1))
    k0 = split + 2
    marks = Tuple{Int,Float64}[]
    sync(be)
    t1 = time_ns()
    push!(marks, (k0 - 1, 0.0))
    for k in k0:(k0 + steps - 1)
        variant_step!(st, p, k, v)
        if (k - k0 + 1) % block == 0
            sync(be)
            push!(marks, (k, (time_ns() - t1) / 1e9))
        end
    end
    kl = k0 + steps
    late = split_steps!(st, p, v, be, kl:(kl + split - 1))
    nz = nonzero_per_particle(st)
    per_block = [1e3 * (marks[j][2] - marks[j - 1][2]) / (marks[j][1] - marks[j - 1][1]) for j in 2:length(marks)]
    out = Dict{String,Any}("variant" => v.name, "cells" => collect(cells), "ppc" => collect(ppc), "particles" => np,
        "setup_seconds" => t_ready - t0, "first_step_seconds" => t_first, "split_early_ms" => early, "split_late_ms" => late,
        "marks" => [[m[1], m[2]] for m in marks], "ms_per_step_blocks" => per_block,
        "ms_per_step_first_block" => per_block[1], "ms_per_step_last_block" => per_block[end],
        "ns_per_particle_step_last_block" => 1e6 * per_block[end] / np,
        "flux_nonzero_per_particle" => nz, "flux_adds_per_particle" => 144)
    @info "$(join(cells, '×')) ppc $(join(ppc, '×')) $(v.name): first block $(round(per_block[1]; digits = 3)) ms/step, last $(round(per_block[end]; digits = 3)) ms/step, " *
          "nonzero flux values $(round(nz; digits = 1)) of 144" early late
    st = nothing
    return out
end

function release(be)
    GC.gc(true)
    be isa CPU || CUDACore.reclaim()
    return nothing
end

parse_kv(args) = Dict(String(a[1]) => String(a[2]) for a in (split(x, '='; limit = 2) for x in args))

function run_sweep(backend_name, tag, kv)
    be = backend_name == "cuda" ? CUDABackend() : CPU()
    sizes = parse.(Int, split(get(kv, "cells", "64,96,128"), ','))
    ppc = Tuple(parse.(Int, split(get(kv, "ppc", "2x2x2"), 'x')))
    steps = parse(Int, get(kv, "steps", "400"))
    block = parse(Int, get(kv, "block", "20"))
    split_n = parse(Int, get(kv, "split", "5"))
    T = get(kv, "float", "32") == "64" ? Float64 : Float32
    variants = parse_variant.(split(get(kv, "variants", "reference,fast,fast-nz,fast-sort10,fast-nz-sort10,fast-sort50"), ','))
    steps % block == 0 || error("steps must be a multiple of block")
    dir = joinpath(REPO, "scripts", "runs", "fast3d-$tag")
    mkpath(dir)
    gpu = be isa CPU ? "none" : string(CUDACore.name(CUDACore.device()))
    results = Any[]
    meta = Dict{String,Any}("tag" => tag, "backend" => backend_name, "gpu" => gpu, "cpu" => Sys.cpu_info()[1].model,
        "threads" => Threads.nthreads(), "julia" => string(VERSION), "float" => string(T), "steps" => steps, "block" => block,
        "split" => split_n, "ppc" => collect(ppc), "cells" => sizes, "variants" => [v.name for v in variants],
        "started" => string(now(UTC)))
    for c in sizes, v in variants
        push!(results, run_one((c, c, c), ppc, steps, block, split_n, v, be, T))
        release(be)
        open(io -> JSON.print(io, Dict("meta" => meta, "results" => results), 2), joinpath(dir, "results.json"), "w")
    end
    write(joinpath(dir, "table.md"), table(results))
    write_manifest(dir; root = REPO, extra = Dict{String,Any}("run_id" => "fast3d-$tag", "work_package" => "#550",
        "generator" => "scripts/fast3d/fast3d.jl", "args" => join(ARGS, ' '), "backend" => backend_name, "gpu" => gpu, "float" => string(T),
        "outputs" => Dict(f => "sha256:" * RunManifest.file_hash(joinpath(dir, f)) for f in ("results.json", "table.md"))))
    print(table(results))
    return dir
end

fmt(x) = string(round(x; sigdigits = 3))

function table(results)
    io = IOBuffer()
    println(io, "| Cells | Particles | Variant | ms/step, steps $(results[1]["marks"][1][1] + 1)+ (first block) | ms/step, last block | ns per particle-step, last block | ",
            "late split: sort / push / deposit / fields (ms) | nonzero flux values of 144 |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for r in results
        l = r["split_late_ms"]
        println(io, "| $(join(r["cells"], '×')) | $(r["particles"]) | $(r["variant"]) | $(fmt(r["ms_per_step_first_block"])) | $(fmt(r["ms_per_step_last_block"])) | ",
                "$(fmt(r["ns_per_particle_step_last_block"])) | $(fmt(l["sort"])) / $(fmt(l["push"])) / $(fmt(l["deposit"])) / $(fmt(l["fields"])) | ",
                "$(fmt(r["flux_nonzero_per_particle"])) |")
    end
    return String(take!(io))
end

# --- check: the variants against the module's step! on the CPU ----------------------------------

function state_arrays(st)
    s = only(st.species)
    f = st.fields
    return (Array.(s.cell)..., Array.(s.offset)..., Array.(s.u)..., Array(s.w), Array(f.Ex), Array(f.Ey), Array(f.Ez), Array(f.Bx), Array(f.By), Array(f.Bz),
            Array(f.Jx), Array(f.Jy), Array(f.Jz))
end

function check()
    be = CPU()
    cells, ppc, nsteps = (12, 10, 8), (2, 2, 2), 6
    worst = 0.0
    for (vname, against, bitwise) in (("reference", "reference", true), ("fast", "fast", Threads.nthreads() == 1),
                                      ("fast-nz", "fast", Threads.nthreads() == 1), ("fast-sort2", "fast", false),
                                      ("fast-nz-sort3", "fast", false))
        v, a = parse_variant(vname), parse_variant(against)
        st, p = new_state(cells, ppc, nsteps, v, be, Float64)
        ref, pr = new_state(cells, ppc, nsteps, a, be, Float64)
        for k in 1:nsteps
            variant_step!(st, p, k, v; times = isodd(k) ? Dict(ph => 0.0 for ph in PHASES) : nothing)
            step!(ref, pr, k; observer = nothing)
        end
        x, y = state_arrays(st), state_arrays(ref)
        if bitwise
            all(map(==, x, y)) || error("$vname is not bitwise the module's $against step")
            println("check: $vname is bitwise the module's $against step over $nsteps steps")
        else
            # a sort permutes the particles: compare the fields, which do not depend on the order but through roundoff
            e = maximum(j -> maximum(abs, x[j] .- y[j]) / max(maximum(abs, y[j]), eps()), 12:length(x))
            sort(Array(only(st.species).w)) == sort(Array(only(ref.species).w)) || error("$vname: the weights differ as sets")
            e < 1e-10 || error("$vname: fields differ from the module's $against step by $e (relative)")
            worst = max(worst, e)
            println("check: $vname's fields match the module's $against step to $(fmt(e)) relative over $nsteps steps")
        end
        nz = nonzero_per_particle(st)
        0 < nz <= 144 || error("nonzero count $nz out of range")
        println("check: $vname: $(fmt(nz)) nonzero flux values of 144 per particle on the last move")
    end
    return worst
end

# --- plot -----------------------------------------------------------------------------------------

function plot_run(dir)
    @eval using CairoMakie
    R = JSON.parsefile(joinpath(dir, "results.json"))
    res = R["results"]
    sizes = unique(r["cells"][1] for r in res)
    variants = unique(r["variant"] for r in res)
    fig = Base.invokelatest(Figure; size = (420 * length(sizes), 360))
    side = Dict{String,Any}("meta" => R["meta"], "series" => Any[])
    for (j, c) in enumerate(sizes)
        ax = Base.invokelatest(Axis, fig[1, j]; title = "$(c)³, $(join(R["meta"]["ppc"], '×')) per cell", xlabel = "step",
                               ylabel = j == 1 ? "ns per particle-step" : "")
        for v in variants
            r = only(filter(r -> r["cells"][1] == c && r["variant"] == v, res))
            ks = [m[1] for m in r["marks"][2:end]]
            y = 1e6 .* r["ms_per_step_blocks"] ./ r["particles"]
            Base.invokelatest(lines!, ax, ks, y; label = v)
            push!(side["series"], Dict("cells" => c, "variant" => v, "step" => ks, "ns_per_particle_step" => y))
        end
        j == length(sizes) && Base.invokelatest(axislegend, ax; position = :lt, labelsize = 10)
    end
    Base.invokelatest(save, joinpath(dir, "trajectory.png"), fig)
    open(io -> JSON.print(io, side, 2), joinpath(dir, "trajectory.json"), "w")
    println("wrote $(joinpath(dir, "trajectory.png"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if CMD == "check"
        check()
    elseif CMD == "run"
        run_sweep(ARGS[2], ARGS[3], parse_kv(ARGS[4:end]))
    elseif CMD == "plot"
        plot_run(ARGS[2])
    else
        error("fast3d.jl: run, check or plot")
    end
end
