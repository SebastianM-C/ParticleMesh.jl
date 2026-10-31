# CPU time per particle-step and scratch per particle for both deposition modes (WP-010
# deliverable 6; WP-011 adds the scratch with a two-array extras tuple). Recorded without
# thresholds (A7's throughput budget is on the GPU).
#   julia +1.13 --project=scripts -t 1 scripts/es1d/timing.jl
#   julia +1.13 --project=scripts -t 16 scripts/es1d/timing.jl
include(joinpath(@__DIR__, "common.jl"))
using ParticleMesh.ParticleMeshCore: Workspace, scratch_bytes

"Bytes of the particle arrays of a species, per particle."
species_bytes(s) = (sizeof(s.cell[1]) + sizeof(s.offset[1]) + sum(sizeof, s.u) + sizeof(s.w) + sizeof(s.status)) / capacity(s)

function timing(; ppcs = (64, 256, 1024, 4096), n = 64, steps = 50)
    dir = run_dir("timing-t$(Threads.nthreads())", @__FILE__; params = Dict("ppcs" => collect(ppcs), "n" => n, "steps" => steps))
    rows = []
    for T in (Float32, Float64), mode in (Fast(), Reference()), ppc in ppcs
        st, p, _ = landau_case(T; n, ppc, mode, tmax_ωp = 10)
        initialize!(st, p)
        run!(st, p, 5)                                   # warm-up (compilation)
        t = @elapsed run!(st, p, steps; first = 6)
        np = n * ppc
        s = st.species[1]
        # WP-011: the scratch with a two-array extras tuple (old cell and offset, as a
        # charge-conserving deposition would declare them), from a species of the same capacity.
        sx = Species(KernelAbstractions.get_backend(s), T, capacity(s); q = -1, m = 1, extras = (old_cell = Int32, old_offset = T))
        push!(rows, Dict("T" => string(T), "mode" => string(typeof(mode)), "ppc" => ppc, "particles" => np,
                         "ns_per_particle_step" => 1e9 * t / (np * steps),
                         "scratch_bytes_per_particle" => scratch_bytes(st.ws[1]) / capacity(s),
                         "scratch_bytes_per_particle_extras2" => scratch_bytes(Workspace(sx, st.grid)) / capacity(s),
                         "particle_bytes_per_particle" => species_bytes(s)))
        println(rows[end])
    end
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "particles", ylabel = "ns per particle-step",
              title = "1D electrostatic step, KA CPU backend, $(Threads.nthreads()) thread(s), $(Sys.cpu_info()[1].model)")
    for T in ("Float32", "Float64"), mode in ("Fast", "Reference")
        r = filter(x -> x["T"] == T && x["mode"] == mode, rows)
        scatterlines!(ax, [x["particles"] for x in r], [x["ns_per_particle_step"] for x in r]; label = "$T, $mode")
    end
    axislegend(ax)
    save_plot(dir, "timing", fig, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
    println("timing: outputs in $dir")
    return dir
end

abspath(PROGRAM_FILE) == @__FILE__() && timing()
