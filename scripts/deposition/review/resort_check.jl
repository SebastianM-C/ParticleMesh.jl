# `resort_by_cell!` on a backend against `sort_by_cell!` on the CPU, on the study's synthetic states:
#
#   julia +1.13 -t 4 --project=<repo>/scripts resort_check.jl <repo> [cpu|cuda] [cells|nodes|centres|mixed]
#
# The third argument is the bins of every workspace, where the engine has them (WP-041's
# nearest-node bins): the cells (the default), `NodeBins` of order 2 on nodes on every axis, of
# order 1 at centres on every axis, or of order 2 with the cell on x and nodes on the other axes (as
# the module bins a moving window's species).
# Two copies of one species, one sorted in full on the CPU with a plain workspace, one sorted from
# its last order on the backend with a workspace of `append`. Per round every alive particle moves
# by up to 0.99 cells per axis and 5% die; then, by round, nothing is added, the whole tail is
# filled (every third slot left free), half of it, or one particle. After each sort the two must
# hold the same bits in every array, the permutation, the segments and the count. One species
# starts full, so that the tail runs past the capacity. 3x5 is the smallest grid the incremental sort
# takes, 2x7 one it hands to the full sort. Also: launch shapes 64 and 256 give the
# same bits, and a second workspace that takes the full sort gives them too. Prints one line per
# case (with the number of rounds in which the sort started from the last order: all, or none on a
# grid below 3 cells per axis) and throws on a failure.
const REPO = ARGS[1]
include(joinpath(REPO, "scripts", "engine2d", "common.jl"))
include(joinpath(REPO, "scripts", "deposition", "common.jl"))
const NAME = length(ARGS) >= 2 ? ARGS[2] : "cpu"
if NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
else
    const BE = CPU()
end
const BINS = length(ARGS) >= 3 ? ARGS[3] : "cells"
"The workspaces' `bins` keyword for `N` axes."
function bins_kw(N)
    BINS == "cells" && return (;)
    BINS == "nodes" && return (; bins = PMC.NodeBins(PMC.Quadratic, ntuple(_ -> PMC.Node(), N)))
    BINS == "centres" && return (; bins = PMC.NodeBins(PMC.Linear, ntuple(_ -> PMC.Center(), N)))
    BINS == "mixed" && return (; bins = PMC.NodeBins(PMC.Quadratic, ntuple(d -> d == 1 ? PMC.Center() : PMC.Node(), N)))
    error("the third argument is cells, nodes, centres or mixed, got $BINS")
end
const FAILED = String[]

"Add up to `k` particles behind the count on the host (every third slot stays free), and advance the count by `k`."
function append!(s::PMC.Species{N,T}, grid, k; seed) where {N,T}
    rng = Xoshiro(seed)
    V = base_type(T)
    cap = PMC.capacity(s)
    cell, offset, status, w = map(Array, s.cell), map(Array, s.offset), Array(s.status), Array(s.w)
    c0 = Int(Array(s.count)[1])
    added = 0
    for j in 1:k
        i = c0 + j
        i <= cap || break
        j % 3 == 0 && continue
        for d in 1:N
            cell[d][i] = rand(rng, 1:grid.n[d])
            offset[d][i] = lift(T, rand(rng, V), rng)
        end
        w[i] = lift(T, 1 + rand(rng, V), rng)
        status[i] = PMC.ALIVE
        added += 1
    end
    foreach(copyto!, s.cell, cell)
    foreach(copyto!, s.offset, offset)
    copyto!(s.w, w)
    copyto!(s.status, status)
    copyto!(s.count, Int32[min(c0 + k, cap)])
    return added
end

alive(s) = count(==(PMC.ALIVE), Array(s.status))
"Whether the next `resort_by_cell!` of `s` with `w` starts from the last order, and is not the full sort (host values)."
taken(w, s, grid) = w.resort !== nothing && all(>=(3), grid.n) && (isdefined(PMC, :knows_order) ? PMC.knows_order(w.resort, s) : w.resort.ordered[])
function agree(a, wa, b, wb)
    return all(map(same, species_arrays(a), species_arrays(b))) && Array(wa.perm) == Array(wb.perm) &&
           Array(wa.seg_start) == Array(wb.seg_start) && Array(wa.seg_stop) == Array(wb.seg_stop) && Array(a.count) == Array(b.count) &&
           Int(Array(b.count)[1]) == alive(b)
end

function check(::Type{T}, n, ppc, fill; rounds = 8, kill = 0.05) where {T}
    N = length(n)
    V = base_type(T)
    label = "$(T <: Dual ? "Dual{$(V),3}" : string(T)) $(join(n, "x")) ppc $ppc fill $fill kill $kill"
    grid = PMC.PeriodicGrid(n, ntuple(d -> T(V(0.3) * d), N), min(3, minimum(n) ÷ 2))
    k = max(4, prod(n) * ppc ÷ 50)
    a = synthetic_species(CPU(), T, grid; ppc, fill)
    b = synthetic_species(BE, T, grid; ppc, fill)
    c, d = copy_species(b), copy_species(b)
    kw = bins_kw(N)
    wa, wb, wc, wd = PMC.Workspace(a, grid; kw...), PMC.Workspace(b, grid; append = k, kw...), PMC.Workspace(c, grid; append = k, kw...),
                     PMC.Workspace(d, grid; append = k, kw...)
    PMC.sort_by_cell!(a, grid, wa)
    foreach((s, w) -> PMC.resort_by_cell!(s, grid, w), (b, c, d), (wb, wc, wd))                  # the first sort: in full
    ok = agree(a, wa, b, wb)
    shapes = true
    added = 0
    inc = 0                                           # the rounds in which the sort started from the last order
    for r in 1:rounds
        for s in (a, b, c, d)
            move!(s, grid; dmax = 0.99, kill, seed = 10r)
            m = r % 4 == 1 ? 0 : r % 4 == 2 ? k : r % 4 == 3 ? k ÷ 2 : 1
            x = m == 0 ? 0 : append!(s, grid, m; seed = 20r)
            s === a && (added += x)
        end
        PMC.sort_by_cell!(a, grid, wa)
        inc += taken(wb, b, grid) && taken(wc, c, grid)
        PMC.resort_by_cell!(b, grid, wb)
        PMC.resort_by_cell!(c, grid, wc; workgroupsize = isodd(r) ? 64 : 256)
        PMC.sort_by_cell!(d, grid, wd)                                                            # the full sort in a workspace with append
        ok &= agree(a, wa, b, wb)
        shapes &= agree(a, wa, c, wc) && agree(a, wa, d, wd)
    end
    path = inc == (all(>=(3), n) ? rounds : 0)
    path || push!(FAILED, "$label: the sort started from the last order in $inc of $rounds rounds")
    ok || push!(FAILED, "$label: resort_by_cell! against the CPU's sort_by_cell!")
    shapes || push!(FAILED, "$label: launch shapes, or the full sort with append")
    println(rpad(label, 62), ok && shapes && path ? "ok    " : "FAILED", "  ", alive(a), " alive of ", PMC.capacity(a), ", ", added, " added, tail of ", k,
            ", from the last order in ", inc, " of ", rounds, " rounds")
    flush(stdout)
    return nothing
end

function main()
    for T in (Float32, Float64, Dual{Nothing,Float32,3}), (n, ppc) in (((37, 23), 5), ((96, 64), 8), ((3, 5), 6), ((2, 7), 6), ((13, 11, 9), 3), ((24, 20, 16), 4)),
        fill in (0.8, 1.0)

        check(T, n, ppc, fill)
    end
    # 1% die per round and the tail is 2% of the particles: from a full species, the tail holds particles and runs past the capacity
    for T in (Float32, Float64, Dual{Nothing,Float32,3}), (n, ppc) in (((37, 23), 5), ((96, 64), 8), ((13, 11, 9), 3))
        check(T, n, ppc, 1.0; kill = 0.01)
    end
    isempty(FAILED) || error("$(length(FAILED)) checks failed:\n" * join(FAILED, "\n"))
    println("backend ", NAME, ", bins ", BINS, ": all checks passed")
end
Base.invokelatest(main)
