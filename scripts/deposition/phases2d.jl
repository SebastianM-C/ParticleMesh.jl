# Where a step of the WP-045 case spends its time at the sizes of bench2d.jl, per element type
# and mode: the per-phase timer of scripts/optim/step_timing.jl on other sizes of the case.
#
#   julia +1.13 -t 4 --project=scripts scripts/deposition/phases2d.jl [cpu|cuda] [S,M,L] [f32,f64,d32,d64] [fast]
#
# `timed_step!` is the copy of the module's `step!` that step_timing.jl times (a timer around
# each phase, the device waited for after each). Every case runs twice in one process and the
# second run is timed. `fast` adds the Fast mode for the plain element types. Writes
# `phases2d.json`, `tables.md` and a manifest into scripts/runs/deposition-phases2d-<backend>-<time>/.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "..", "optim", "wake2d_case.jl"))
using ParticleMeshElectromagnetic
using ForwardDiff: Dual

const PME = ParticleMeshElectromagnetic
const PMC = ParticleMesh.ParticleMeshCore
const PHASES = (:push, :sort, :deposit, :window, :fields, :other)

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
elseif BACKEND_NAME == "cpu"
    const BE = CPU()
else
    error("phases2d.jl: the first argument is cpu or cuda, got $BACKEND_NAME")
end

"The sizes of bench2d.jl."
const SIZES = Dict("S" => (ppl = 16, dy_cells = 4, ppc = (2, 2), nsteps = 512),
                   "M" => (ppl = 32, dy_cells = 4, ppc = (4, 4), nsteps = 1024),
                   "L" => (ppl = 32, dy_cells = 1, ppc = (4, 4), nsteps = 1400))

d2(::Type{T}) where {T} = (Dual{Nothing}(T(7), one(T), zero(T)), Dual{Nothing}(T(100), zero(T), one(T)))
const ELEMENTS = Dict("f32" => ("Float32", (7.0f0, 100.0f0)), "f64" => ("Float64", (7.0, 100.0)),
                      "d32" => ("Dual{Float32,2}", d2(Float32)), "d64" => ("Dual{Float64,2}", d2(Float64)))

function timed!(f, times, phase)
    t0 = time_ns()
    f()
    KernelAbstractions.synchronize(BE)
    times[phase] += (time_ns() - t0) / 1e9
    return nothing
end

"The module's `step!` (lib/ParticleMeshElectromagnetic/src/step.jl) with a timer around each phase, as in step_timing.jl."
function timed_step!(st, p, k, times)
    PME.check_params(st, p)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = PME.window_frame(st)
    T = eltype(st.ρ)
    timed!(times, :other) do
        PME.observe!(st.recorder, st, k)
        PME.inject_witness!(PME.witness_frame(st), p.witness, grid, k, wgs)
    end
    timed!(times, :push) do
        for s in (st.species..., PME.witness_species(st)...)
            PME.launcher(PME.push_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2,
                                                                dt_dx, n; ndrange = capacity(s))
        end
    end
    timed!(() -> PME.retire_x!(window, st), times, :other)
    timed!(times, :sort) do
        for (s, ws) in zip(st.species, st.ws)
            PME.sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
        end
    end
    timed!(times, :other) do
        fill!(fields.Jx, zero(T)); fill!(fields.Jy, zero(T)); fill!(fields.Jz, zero(T))
    end
    timed!(times, :deposit) do
        for (s, ws) in zip(st.species, st.ws)
            deposit_flux!(PME.deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, PME.rho_stagger(Val(2)), grid, ws;
                          dt = st.dt, PME.transverse_currents(Val(2))..., workgroupsize = wgs)
        end
    end
    timed!(times, :other) do
        for J in (fields.Jx, fields.Jy, fields.Jz)
            PME.zero_x_edges!(window, J; workgroupsize = wgs)
            fold_guards!(J, grid; workgroupsize = wgs)
        end
        PME.filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
        PME.retire!(PME.open_absorber(st), st)
    end
    timed!(() -> PME.solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs), times, :fields)
    timed!(() -> PME.advance_window!(window, st, p, k), times, :window)
    timed!(() -> PME.reads_kinetic(st.recorder, k + 1) && PME.kinetic!(st), times, :other)
    return st
end

"One size, element type and mode: the phases of the second of two runs, in seconds."
function measure(size, label, x, mode)
    prob = wake_problem(x...; SIZES[size]...)
    numerics = EMNumerics(; mode)
    p = params(prob)
    times = Dict{Symbol,Float64}()
    local st, total
    for _ in 1:2
        st = EMState(prob, numerics; backend = BE, record = ())
        initialize!(st, prob)
        foreach(ph -> times[ph] = 0.0, PHASES)
        KernelAbstractions.synchronize(BE)
        t0 = time_ns()
        for k in 1:prob.nsteps
            timed_step!(st, p, k, times)
        end
        total = (time_ns() - t0) / 1e9
    end
    return Dict{String,Any}("size" => size, "T" => label, "mode" => string(nameof(typeof(mode))), "steps" => prob.nsteps, "total" => total,
                            "cells" => collect(prob.cells), "alive" => alive_count(st.species[1]), "capacity" => capacity(st.species[1]),
                            "shifts" => PME.window_shifts(st, prob.nsteps),
                            "phases" => Dict{String,Any}(string(ph) => times[ph] for ph in PHASES))
end

fmt(x) = x >= 100 ? string(round(Int, x)) : x >= 10 ? string(round(x; digits = 1)) : x >= 1 ? string(round(x; digits = 2)) : string(round(x; sigdigits = 2))

"The Markdown table of the rows: milliseconds per step, in all and per phase."
function table(rows)
    io = IOBuffer()
    println(io, "Milliseconds per step (the run's seconds over its steps).\n")
    println(io, "| size | cells | particles at the end | element type | mode | step | ", join(PHASES, " | "), " |")
    println(io, "|---|---|---|---|---|---|", join(("---" for _ in PHASES), "|"), "|")
    for r in rows
        per(x) = fmt(1e3 * x / r["steps"])
        println(io, "| ", r["size"], " | ", join(r["cells"], " × "), " | ", r["alive"], " | `", r["T"], "` | ", r["mode"], " | ", per(r["total"]), " | ",
                join((per(r["phases"][string(ph)]) for ph in PHASES), " | "), " |")
    end
    return String(take!(io))
end

function main()
    sizes = length(ARGS) >= 2 ? split(ARGS[2], ",") : ["S"]
    elements = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["f32", "f64", "d32", "d64"]
    fast = "fast" in ARGS
    gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
    extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
    dir = run_dir("phases2d-$BACKEND_NAME", @__FILE__; prefix = "deposition", work_package = "deposition study",
                  backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                  params = Dict{String,Any}("case" => "wake_problem(7, 100)", "sizes" => collect(String, sizes),
                                            "elements" => collect(String, elements), "fast" => fast),
                  extra_inputs = [joinpath(@__DIR__, "..", "optim", "wake2d_case.jl")], extra)
    rows = Dict{String,Any}[]
    for size in sizes, e in elements
        label, x = ELEMENTS[e]
        for mode in (fast && !startswith(label, "Dual") ? (Reference(), Fast()) : (Reference(),))
            push!(rows, measure(size, label, x, mode))
            println(table(rows[end:end]))
            open(io -> JSON.print(io, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => cpu_model(), "gpu" => gpu, "backend" => BACKEND_NAME,
                                           "phases" => collect(string.(PHASES))), 2),
                 joinpath(dir, "phases2d.json"), "w")
        end
    end
    write(joinpath(dir, "tables.md"), table(rows))
    println("wrote ", relpath(dir, REPO))
    return dir
end

Base.invokelatest(main)
