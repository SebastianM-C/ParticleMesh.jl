# Throughput of the module's step: ns per particle-step, Fast and Reference, per case. No
# threshold (A7: measure, then propose; the factor of Smilei on the same card waits for the Smilei
# GPU reference, cross-code stage 2).
#
# CPU (WP-014 budget), at the thread count of the process, Float64 at the TIMING sizes:
#   julia +1.13 --project=scripts -t 1  scripts/em1d/timing.jl
#   julia +1.13 --project=scripts -t 16 scripts/em1d/timing.jl
# CUDA (WP-013, A7's first throughput numbers on the RTX 4080 SUPER): Float32 and Float64 at
# 2^16, 2^20 and 2^24 particles, the GPU synchronized around every timed region, with the
# sort, the flux deposition, the field solve and the scan timed separately on the state's own
# arrays, and the device bytes of the state:
#   julia +1.13 --project=scripts scripts/em1d/timing.jl cuda
include(joinpath(@__DIR__, "common.jl"))
using CUDACore: CUDACore, CUDABackend

const TIMING = (("landau", 4096), ("two-stream", 1000), ("weibel", 512))
const TIMING_STEPS = 50
const CUDA_PARTICLES = (2^16, 2^20, 2^24)
const CUDA_TYPES = (Float32, Float64)

sync(be) = KernelAbstractions.synchronize(be)

"Seconds per call of `f` on `be`, the median of `reps` synchronized calls after one warm-up call."
function timed(f, be; reps = 5)
    f(); sync(be)
    ts = map(1:reps) do _
        sync(be)
        @elapsed (f(); sync(be))
    end
    return sort(ts)[cld(reps, 2)]
end

"Device (or host, on the CPU) bytes allocated while `f` runs."
allocated_bytes(::CPU, f) = @allocated f()
allocated_bytes(::CUDABackend, f) = CUDACore.@allocated f()

function state_for(name, ::Type{T}, mode, be; ppc = nothing, nsteps) where {T}
    case = load_case(name)
    prob = problem_from_case(case, T; ppc, t_end = (nsteps + 10) * case["time"]["dt"])
    num = numerics_from_case(case; mode)
    local st
    bytes = allocated_bytes(be, () -> (st = EMState(prob, num; backend = be); nothing))
    initialize!(st, prob)
    return st, ParticleMeshElectromagnetic.params(prob), prob, bytes
end

function time_case(name, ppc, mode)
    st, p, _, _ = state_for(name, Float64, mode, CPU(); ppc, nsteps = TIMING_STEPS)
    run!(st, p, 5)                                   # warm-up (compilation)
    t = @elapsed run!(st, p, TIMING_STEPS; first = 6)
    np = sum(alive_count, st.species)
    return (; ns = t / (np * TIMING_STEPS) * 1e9, particles = np, seconds = t)
end

"The CUDA row of one case, element type, mode and particle count: the step and its phases."
function time_case_cuda(name, ::Type{T}, mode, nparticles, be) where {T}
    case = load_case(name)
    nspecies = count(s -> s["mobile"], case["species"])
    ppc = max(1, nparticles ÷ (nspecies * case["grid"]["cells"]))
    nsteps = nparticles >= 2^24 ? 10 : 30
    st, p, prob, bytes = state_for(name, T, mode, be; ppc, nsteps = nsteps + 10)
    np = sum(alive_count, st.species)
    k = Ref(0)
    step() = step!(st, p, (k[] = mod1(k[] + 1, length(st.hist.K)); k[]))
    run!(st, p, 3); sync(be); k[] = 3
    t_step = timed(() -> foreach(_ -> step(), 1:nsteps), be; reps = 3) / nsteps
    (; numerics, grid, fields) = st
    t_sort = timed(be) do
        for (s, ws) in zip(st.species, st.ws)
            ParticleMeshElectromagnetic.sorts(st, s) && sort_by_cell!(s, grid, ws)
        end
    end
    t_flux = timed(be) do
        for (s, ws) in zip(st.species, st.ws)
            deposit_flux!(numerics.mode, fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, RHO_STAGGER, grid, ws; dt = p.dt,
                          quantity_y = ParticleMeshElectromagnetic.CurrentY(), quantity_z = ParticleMeshElectromagnetic.CurrentZ())
        end
    end
    t_solve = timed(() -> solve_fields!(numerics.solver, fields, grid, p.dt), be)
    scratch = similar(st.ρ)
    t_scan = timed(() -> cumsum_periodic!(scratch, st.ρ, grid), be)
    ns(t) = t / np * 1e9
    return Dict{String,Any}("case" => name, "type" => string(T), "mode" => string(mode), "particles" => np, "ppc" => ppc,
                            "cells" => ncells(grid), "steps" => nsteps,
                            "ns_per_particle_step" => ns(t_step), "sort_ns" => ns(t_sort), "flux_ns" => ns(t_flux),
                            "solve_ns" => ns(t_solve), "scan_ns" => ns(t_scan), "scan_us" => t_scan * 1e6,
                            "state_device_bytes" => bytes, "state_bytes_per_particle" => bytes / np)
end

function main_cpu()
    params = Dict{String,Any}("cases" => [Dict("case" => c, "ppc" => n) for (c, n) in TIMING], "steps" => TIMING_STEPS)
    dir = run_dir("timing-t$(Threads.nthreads())", @__FILE__, "landau"; params)
    rows = Dict{String,Any}[]
    for (name, ppc) in TIMING, mode in (Reference(), Fast())
        r = time_case(name, ppc, mode)
        push!(rows, Dict("case" => name, "mode" => string(mode), "threads" => Threads.nthreads(), "ns_per_particle_step" => r.ns,
                         "particles" => r.particles, "seconds" => r.seconds))
        println(rpad(name, 12), rpad(string(mode), 12), " ", round(r.ns, digits = 1), " ns/particle-step (", r.particles, " particles)")
    end
    fig = Figure(size = (700, 400))
    ax = Axis(fig[1, 1]; ylabel = "ns per particle-step", title = "EM step on the CPU, $(Threads.nthreads()) threads",
              xticks = (1:length(TIMING), [c for (c, _) in TIMING]))
    for (j, m) in enumerate(("Reference()", "Fast()"))
        v = [only(r["ns_per_particle_step"] for r in rows if r["case"] == c && r["mode"] == m) for (c, _) in TIMING]
        barplot!(ax, (1:length(TIMING)) .+ (j - 1.5) * 0.35, v; width = 0.35, label = m)
    end
    axislegend(ax)
    save_plot(dir, "timing", fig, Dict("rows" => rows, "cpu" => Sys.cpu_info()[1].model))
    println("outputs in ", relpath(dir, REPO))
end

function main_cuda()
    CUDACore.functional() || error("timing.jl cuda: CUDA is not functional on this machine")
    be = CUDABackend()
    gpu = CUDACore.name(CUDACore.device())
    total = Int(CUDACore.totalmem(CUDACore.device()))
    params = Dict{String,Any}("cases" => [c for (c, _) in TIMING], "particles" => collect(CUDA_PARTICLES),
                              "types" => string.(CUDA_TYPES))
    dir = run_dir("timing-cuda", @__FILE__, "landau"; params, work_package = "WP-013", backend = "KernelAbstractions CUDA (CUDACore)",
                  extra = Dict{String,Any}("gpu" => gpu, "gpu_total_memory_bytes" => total,
                                           "cudacore" => string(pkgversion(CUDACore))))
    rows = Dict{String,Any}[]
    for (name, _) in TIMING, T in CUDA_TYPES, mode in (Reference(), Fast()), np in CUDA_PARTICLES
        r = time_case_cuda(name, T, mode, np, be)
        push!(rows, r)
        println(rpad(name, 12), rpad(string(T), 9), rpad(string(mode), 12), lpad(r["particles"], 9), "  ",
                round(r["ns_per_particle_step"], sigdigits = 3), " ns/particle-step (sort ", round(r["sort_ns"], sigdigits = 2),
                ", flux ", round(r["flux_ns"], sigdigits = 2), ", solve ", round(r["solve_ns"], sigdigits = 2), ", scan ",
                round(r["scan_us"], sigdigits = 3), " µs), ", round(r["state_bytes_per_particle"], sigdigits = 3), " B/particle")
        GC.gc(); CUDACore.reclaim()
    end
    # The largest Float32 particle count of each case that fits on the card: the state's bytes are
    # affine in the particle count; the slope and intercept from the two largest sizes, Reference.
    fits = Dict{String,Any}()
    for (name, _) in TIMING
        rs = sort([r for r in rows if r["case"] == name && r["type"] == "Float32" && r["mode"] == "Reference()"]; by = r -> r["particles"])
        a, b = rs[end - 1], rs[end]
        slope = (b["state_device_bytes"] - a["state_device_bytes"]) / (b["particles"] - a["particles"])
        fixed = b["state_device_bytes"] - slope * b["particles"]
        fits[name] = Dict("bytes_per_particle" => slope, "fixed_bytes" => fixed, "max_particles" => floor(Int, (total - fixed) / slope))
        println("Float32 ", name, ": ", round(slope, sigdigits = 3), " B/particle, at most ~", fits[name]["max_particles"],
                " particles in ", round(total / 2^30, digits = 1), " GiB (state only, no harness)")
    end
    fig = Figure(size = (900, 700))
    for (i, T) in enumerate(CUDA_TYPES), (j, (name, _)) in enumerate(TIMING)
        ax = Axis(fig[i, j]; xscale = log2, yscale = log10, xlabel = "particles", ylabel = "ns per particle-step",
                  title = "$name, $T, $gpu")
        for mode in ("Reference()", "Fast()")
            rs = sort([r for r in rows if r["case"] == name && r["type"] == string(T) && r["mode"] == mode]; by = r -> r["particles"])
            scatterlines!(ax, [r["particles"] for r in rs], [r["ns_per_particle_step"] for r in rs]; label = mode)
        end
        i == 1 && j == 1 && axislegend(ax)
    end
    save_plot(dir, "timing-cuda", fig, Dict("rows" => rows, "fits" => fits, "gpu" => gpu))
    println("outputs in ", relpath(dir, REPO))
end

isempty(ARGS) || ARGS[1] == "cpu" ? main_cpu() : ARGS[1] == "cuda" ? main_cuda() : error("usage: timing.jl [cpu|cuda]")
