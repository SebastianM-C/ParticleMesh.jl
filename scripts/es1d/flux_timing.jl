# CPU time per particle of deposit_flux! against deposit! in both modes, and scratch per
# particle with the two old-position extras (WP-009 budget). Recorded without thresholds.
#   julia +1.13 --project=scripts -t 1 scripts/es1d/flux_timing.jl
#   julia +1.13 --project=scripts -t 16 scripts/es1d/flux_timing.jl
include(joinpath(@__DIR__, "common.jl"))
using ParticleMesh.ParticleMeshCore: Workspace, scratch_bytes, advance, ALIVE

struct QUy end
@inline (::QUy)(s, i) = s.q * s.w[i] * s.u[2][i]
struct QUz end
@inline (::QUz)(s, i) = s.q * s.w[i] * s.u[3][i]

"A loaded species with the old position stored and every particle moved by up to 0.9 cells."
function moved(::Type{T}, n, ppc) where {T}
    grid = PeriodicGrid(n, T(1 // 2), flux_guard_width(Quadratic, Center()))
    s = Species(CPU(), T, n * ppc; q = -1, m = 1, extras = (old_cell = Int32, old_offset = T))
    load_quiet!(s, grid, (x, p) -> p, one(T), ppc; vth = (T(1 // 2), T(1 // 2), T(1 // 2)), seed = 1, stream = 1)
    copyto!(s.extras.old_cell, s.cell[1])
    copyto!(s.extras.old_offset, s.offset[1])
    for i in 1:capacity(s)
        Δ = clamp(s.u[1][i], -T(9 // 10), T(9 // 10))
        s.cell[1][i], s.offset[1][i] = advance(s.cell[1][i], s.offset[1][i], Δ, Int32(n))
    end
    return s, grid
end

"Median wall time of `f()` over `k` calls after one warm-up call."
function timed(f, k)
    f()
    return sort([@elapsed(f()) for _ in 1:k])[cld(k, 2)]
end

function flux_timing(; ppcs = (256, 4096), n = 64, k = 21)
    dir = run_dir("flux-timing-t$(Threads.nthreads())", @__FILE__; params = Dict("ppcs" => collect(ppcs), "n" => n, "repetitions" => k))
    rows = []
    for T in (Float32, Float64), mode in (Fast(), Reference()), ppc in ppcs
        s, grid = moved(T, n, ppc)
        ws = Workspace(s, grid)
        sort_by_cell!(s, grid, ws)
        ρ = field(CPU(), T, grid)
        J = ntuple(_ -> field(CPU(), T, grid), 3)
        np = capacity(s)
        t_dep = timed(() -> deposit!(mode, ρ, s, Quadratic, Center(), grid, ws), k)
        t_flux = timed(() -> deposit_flux!(mode, J..., s, Quadratic, Center(), grid, ws; dt = T(1 // 4), quantity_y = QUy(),
                                          quantity_z = QUz()), k)
        bare = Species(CPU(), T, np; q = -1, m = 1)
        push!(rows, Dict("T" => string(T), "mode" => string(typeof(mode)), "ppc" => ppc, "particles" => np,
                         "ns_per_particle_deposit" => 1e9 * t_dep / np, "ns_per_particle_flux" => 1e9 * t_flux / np,
                         "scratch_bytes_per_particle" => scratch_bytes(ws) / np,
                         "scratch_bytes_per_particle_no_extras" => scratch_bytes(Workspace(bare, grid)) / np,
                         "extras_bytes_per_particle" => (sizeof(s.extras.old_cell) + sizeof(s.extras.old_offset)) / np))
        println(rows[end])
    end
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "particles", ylabel = "ns per particle",
              title = "deposit_flux! (solid) and deposit! (dashed), quadratic, KA CPU, $(Threads.nthreads()) thread(s)")
    for T in ("Float32", "Float64"), mode in ("Fast", "Reference")
        r = filter(x -> x["T"] == T && x["mode"] == mode, rows)
        l = scatterlines!(ax, [x["particles"] for x in r], [x["ns_per_particle_flux"] for x in r]; label = "$T, $mode")
        lines!(ax, [x["particles"] for x in r], [x["ns_per_particle_deposit"] for x in r]; linestyle = :dash, color = l.color)
    end
    axislegend(ax)
    save_plot(dir, "flux_timing", fig, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
    println("flux timing: outputs in $dir")
    return dir
end

abspath(PROGRAM_FILE) == @__FILE__() && flux_timing()
