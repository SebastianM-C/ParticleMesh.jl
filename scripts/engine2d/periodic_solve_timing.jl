# CPU time of one periodic solve, `inverse_laplacian_periodic!` followed by `gradient_periodic!`,
# against the grid size at N = 1 and N = 2, for Float32, Float64 and Dual{Float32,3} (WP-021),
# with `cumsum_periodic!` at N = 1 for comparison. Recorded without thresholds.
#   julia +1.13 --project=scripts -t 1 scripts/engine2d/periodic_solve_timing.jl
#   julia +1.13 --project=scripts -t 4 scripts/engine2d/periodic_solve_timing.jl
include(joinpath(@__DIR__, "common.jl"))
using ForwardDiff: Dual

const D3 = Dual{Nothing,Float32,3}
label(T) = T === D3 ? "Dual{Float32,3}" : string(T)
basetype(::Type{T}) where {T<:AbstractFloat} = T
basetype(::Type{<:Dual{<:Any,V}}) where {V} = V

const SIZES_2D = ((16, 12), (32, 24), (64, 48), (128, 96), (256, 192), (512, 384))
const SIZES_1D = (64, 256, 1024, 4096, 16384)

"A source with a nonzero mean on the interior of a field on `grid` (values do not affect the time)."
function source(::Type{T}, grid) where {T}
    s = field(CPU(), T, grid)
    g = guards(grid)
    for J in CartesianIndices(grid.n)
        j = Tuple(J)
        s[(j .+ g)...] = T(sin(0.37j[1]) + cos(1.1j[end]) / 5 + 3 / 10)
    end
    return s
end

"Seconds per call: one warm-up, then 5 samples, each the mean over enough calls to last about 0.1 s."
function adaptive(f)
    f()
    t0 = @elapsed f()
    inner = clamp(ceil(Int, 0.1 / t0), 1, 50)
    return (; timed(f; reps = 5, inner)..., inner)
end

function periodic_solve_timing()
    nt = Threads.nthreads()
    dir = run_dir("periodic-solve-t$nt", @__FILE__; work_package = "WP-021",
                  params = Dict{String,Any}("sizes_2d" => [collect(n) for n in SIZES_2D], "sizes_1d" => collect(SIZES_1D),
                                            "types" => [label(T) for T in (Float32, Float64, D3)], "samples" => 5))
    rows = Dict{String,Any}[]
    for T in (Float32, Float64, D3)
        B = basetype(T)
        for n in (SIZES_1D..., SIZES_2D...)
            grid = n isa Tuple ? PeriodicGrid(n, (B(1 // 4), B(3 // 8)), 2) : PeriodicGrid(n, B(3 // 8), 2)
            N = length(grid.n)
            s = source(T, grid)
            ws = SpectralWorkspace(CPU(), T, grid)
            φ = field(CPU(), T, grid)
            G = ntuple(_ -> field(CPU(), T, grid), N)
            t = adaptive(() -> (inverse_laplacian_periodic!(φ, s, grid, ws); gradient_periodic!(G, φ, grid)))
            row = Dict{String,Any}("type" => label(T), "N" => N, "n" => collect(grid.n), "cells" => prod(grid.n), "op" => "solve+gradient",
                                   "median_s" => t.median, "min_s" => t.min, "max_s" => t.max, "inner" => t.inner)
            push!(rows, row)
            println(label(T), " ", grid.n, " solve+gradient ", round(t.median * 1e3, sigdigits = 3), " ms [",
                    round(t.min * 1e3, sigdigits = 3), ", ", round(t.max * 1e3, sigdigits = 3), "]")
            if N == 1
                out = field(CPU(), T, grid)
                c = adaptive(() -> cumsum_periodic!(out, s, grid))
                push!(rows, Dict{String,Any}("type" => label(T), "N" => 1, "n" => [n], "cells" => n, "op" => "cumsum_periodic!",
                                             "median_s" => c.median, "min_s" => c.min, "max_s" => c.max, "inner" => c.inner))
            end
        end
    end
    fig = Figure(size = (900, 420))
    for (k, N) in enumerate((1, 2))
        ax = Axis(fig[1, k]; xscale = log10, yscale = log10, xlabel = "cells", ylabel = "seconds per call",
                  title = "N = $N, $nt thread$(nt == 1 ? "" : "s"), CPU backend")
        for T in (Float32, Float64, D3), op in ("solve+gradient", "cumsum_periodic!")
            sel = filter(r -> r["N"] == N && r["type"] == label(T) && r["op"] == op, rows)
            isempty(sel) && continue
            scatterlines!(ax, [r["cells"] for r in sel], [r["median_s"] for r in sel]; label = "$(label(T)) $op",
                          linestyle = op == "solve+gradient" ? :solid : :dash)
        end
        axislegend(ax; position = :lt, labelsize = 10)
    end
    path = save_plot(dir, "periodic_solve_t$nt", fig, Dict("rows" => rows, "cpu" => cpu_model(), "threads" => nt))
    println("wrote ", path)
    return rows
end

periodic_solve_timing()
