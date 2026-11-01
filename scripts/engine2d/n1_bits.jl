# The N = 1 bits and timings before and after the 2D port (WP-018 deliverable 0, T-260): the
# electromagnetic module's T-218 state digests of each case and element type (its own
# subprocess worker, run in the module's test environment at 1 and 4 threads), and 1D timings
# of the module's step on the Landau case (64 cells, four particle counts, repeated for the
# spread) at the thread count of this process. Run once per thread count, on the commit before
# a change of the engine and on the commit after it:
#   julia +1.13 --project=scripts -t 1 scripts/engine2d/n1_bits.jl
#   julia +1.13 --project=scripts -t 4 scripts/engine2d/n1_bits.jl
include(joinpath(@__DIR__, "common.jl"))
using ParticleMeshElectromagnetic
const LANDAU_CASE = joinpath(REPO, "scripts", "crosscode", "cases", "landau.toml")

"T-218's digests: `(threads, case, type) => digest` from the module's determinism worker."
function em_digests(nt)
    worker = joinpath(EM_TEST, "determinism_worker.jl")
    out = read(`$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$EM_TEST $worker`, String)
    lines = filter(startswith("DIGEST"), split(out, '\n'))
    return [Dict("threads" => parse(Int, f[2]), "case" => f[3], "type" => f[4], "digest" => f[5]) for f in split.(lines)]
end

"The module's step on the Landau case: ns per particle-step, `reps` repeats of `steps` steps."
function n1_timings(; ppcs = (64, 256, 1024, 4096), n = 64, steps = 50, reps = 3)
    rows = Dict{String,Any}[]
    case = TOML.parsefile(LANDAU_CASE)
    for T in (Float32, Float64), mode in (Fast(), Reference()), ppc in ppcs
        prob = problem_from_case(case, T; ppc, cells = n, t_end = (5 + reps * steps + 1) * case["time"]["dt"])
        st = EMState(prob, numerics_from_case(case; mode))
        initialize!(st, prob)
        p = ParticleMeshElectromagnetic.params(prob)
        run!(st, p, 5)                                   # warm-up (compilation)
        np = sum(alive_count, st.species)
        k = 6
        ns = map(1:reps) do _
            t = @elapsed run!(st, p, steps; first = k)
            k += steps
            1e9 * t / (np * steps)
        end
        push!(rows, Dict{String,Any}("T" => string(T), "mode" => string(nameof(typeof(mode))), "ppc" => ppc, "particles" => np,
                                     "ns_per_particle_step" => ns, "median" => median(ns)))
        println(rows[end])
    end
    return rows
end

function main()
    nt = Threads.nthreads()
    dir = run_dir("n1-bits-t$nt", @__FILE__;
                  params = Dict{String,Any}("timing" => "the module's step, landau.toml at 64 cells, 3 repeats of 50 steps", "threads" => nt),
                  extra_inputs = [LANDAU_CASE, joinpath(EM_TEST, "determinism_worker.jl"), joinpath(EM_TEST, "determinism_cases.jl"),
                                  joinpath(EM_TEST, "em_helpers.jl"), joinpath(EM_TEST, "setup.jl")])
    digests = vcat(em_digests(1), em_digests(4))
    foreach(println, digests)
    rows = n1_timings()
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "particles", ylabel = "ns per particle-step",
              title = "1D module step (Landau), $nt thread(s), $(cpu_model()), 3 repeats")
    for T in ("Float32", "Float64"), mode in ("Fast", "Reference")
        r = filter(x -> x["T"] == T && x["mode"] == mode, rows)
        x = [q["particles"] for q in r]
        scatterlines!(ax, x, [q["median"] for q in r]; label = "$T, $mode")
        rangebars!(ax, x, [minimum(q["ns_per_particle_step"]) for q in r], [maximum(q["ns_per_particle_step"]) for q in r])
    end
    axislegend(ax)
    save_plot(dir, "n1_timing_t$nt", fig, Dict("rows" => rows, "threads" => nt, "cpu" => cpu_model()))
    open(io -> JSON.print(io, Dict("t218_digests" => digests, "commit" => RunManifest.git_commit(REPO).commit), 2),
         joinpath(dir, "n1_digests.json"), "w")
    println("n1_bits: outputs in ", relpath(dir, REPO))
    return dir
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
