# CPU throughput of the module's step (WP-014 budget, no threshold: the throughput budget is on
# the GPU): ns per particle-step, Fast and Reference, for each case, at the thread count of the
# process. Run once per thread count:
#   julia +1.13 --project=scripts -t 1  scripts/em1d/timing.jl
#   julia +1.13 --project=scripts -t 16 scripts/em1d/timing.jl
include(joinpath(@__DIR__, "common.jl"))

const TIMING = (("landau", 4096), ("two-stream", 1000), ("weibel", 512))
const TIMING_STEPS = 50

function time_case(name, ppc, mode)
    case = load_case(name)
    prob = problem_from_case(case, Float64; ppc, t_end = 1)
    st = EMState(prob, numerics_from_case(case; mode))
    initialize!(st, prob)
    p = ParticleMeshElectromagnetic.params(prob)
    run!(st, p, 5)                                   # warm-up (compilation)
    t = @elapsed run!(st, p, TIMING_STEPS; first = 6)
    np = sum(alive_count, st.species)
    return (; ns = t / (np * TIMING_STEPS) * 1e9, particles = np, seconds = t)
end

function main()
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

main()
