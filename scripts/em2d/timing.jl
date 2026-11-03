# Throughput of the module's periodic 2D3V step on the 2D cross-code cases at their full size
# (WP-030: ParticleMesh's column next to Smilei's GPU build and PIConGPU on the RTX 4080 SUPER):
# ns per mobile particle-step, Reference and Fast, Float64 (the precision of the reference runs)
# and Float32, the GPU synchronized around every timed region. A step includes the module's
# per-step diagnostics (the field energies and mode coefficients the histories record).
#   julia +1.13 --project=scripts scripts/em2d/timing.jl cuda [steps]
#   julia +1.13 --project=scripts -t 16 scripts/em2d/timing.jl cpu [steps]
# A measurement, no threshold (A7).
include(joinpath(@__DIR__, "..", "em1d", "common.jl"))
using CUDACore: CUDACore, CUDABackend

const CASES_2D = ("landau-oblique-2d", "weibel-2d")
const TYPES = (Float64, Float32)

sync(be) = KernelAbstractions.synchronize(be)

"""
    time_case(name, T, mode, be; nsteps) -> Dict

One case at its full size on backend `be`: the state built and initialized for the case's
default seed, 5 warm-up steps (compilation), then the median over 3 repetitions of `nsteps`
synchronized steps, per mobile particle-step.
"""
function time_case(name, ::Type{T}, mode, be; nsteps) where {T}
    case = load_case(name)
    prob = problem_from_case(case, T; t_end = (5 + 3nsteps + 2) * case["time"]["dt"])
    st = EMState(prob, numerics_from_case(case; mode); backend = be)
    initialize!(st, prob)
    p = ParticleMeshElectromagnetic.params(prob)
    run!(st, p, 5); sync(be)
    k = Ref(6)
    ts = map(1:3) do _
        sync(be)
        t = @elapsed (run!(st, p, nsteps; first = k[]); sync(be))
        k[] += nsteps
        t
    end
    mobile = sum(alive_count, st.species)
    t = sort(ts)[2]
    return Dict{String,Any}("case" => name, "type" => string(T), "mode" => string(mode), "mobile_particles" => mobile,
                            "cells" => collect(ncells(st.grid)), "steps" => nsteps, "seconds" => ts,
                            "ns_per_mobile_particle_step" => t / (mobile * nsteps) * 1e9)
end

function main(which, nsteps)
    be = which == "cuda" ? CUDABackend() : CPU()
    which == "cuda" && (CUDACore.functional() || error("timing.jl cuda: CUDA is not functional on this machine"))
    extra = which == "cuda" ? Dict{String,Any}("gpu" => CUDACore.name(CUDACore.device()),
                                               "cudacore" => string(pkgversion(CUDACore))) : Dict{String,Any}()
    dir = run_dir("timing-$which", @__FILE__, nothing; prefix = "em2d", work_package = "WP-030",
                  params = Dict{String,Any}("cases" => collect(CASES_2D), "types" => collect(string.(TYPES)), "steps" => nsteps),
                  extra_inputs = [joinpath(CASES, "$c.toml") for c in CASES_2D],
                  backend = which == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU", extra)
    rows = Dict{String,Any}[]
    for name in CASES_2D, T in TYPES, mode in (Reference(), Fast())
        r = time_case(name, T, mode, be; nsteps)
        push!(rows, r)
        println(rpad(name, 20), rpad(string(T), 9), rpad(string(mode), 12), lpad(r["mobile_particles"], 10), "  ",
                round(r["ns_per_mobile_particle_step"], sigdigits = 3), " ns per mobile particle-step")
        GC.gc(); which == "cuda" && CUDACore.reclaim()
    end
    fig = Figure(size = (800, 450))
    labels = ["$c $(string(T))" for c in CASES_2D for T in TYPES]
    ax = Axis(fig[1, 1]; ylabel = "ns per mobile particle-step", xticks = (1:length(labels), labels),
              title = "ParticleMesh 2D step, $(get(extra, "gpu", Sys.cpu_info()[1].model))")
    for (j, m) in enumerate(("Reference()", "Fast()"))
        v = [only(r["ns_per_mobile_particle_step"] for r in rows if r["case"] == c && r["type"] == string(T) && r["mode"] == m)
             for c in CASES_2D for T in TYPES]
        barplot!(ax, (1:length(labels)) .+ (j - 1.5) * 0.35, v; width = 0.35, label = m)
    end
    axislegend(ax; position = :lt)
    save_plot(dir, "timing-2d", fig, merge(Dict{String,Any}("rows" => rows), extra))
    println("outputs in ", relpath(dir, REPO))
end

main(get(ARGS, 1, "cuda"), parse(Int, get(ARGS, 2, "50")))
