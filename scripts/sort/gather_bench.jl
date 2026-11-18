# How much of the sort's data movement can go (WP-041): the gather that applies the sort's
# permutation to a 2D species with the flux extras, timed alone in five forms on one realistic
# permutation, and the whole sorts next to it.
#
#   julia +1.13 -t 4 --project=scripts scripts/sort/gather_bench.jl [cpu|cuda] [S,M,L] [Float32,Float64,Dual]
#
# The forms of the gather (every one gives the same arrays):
#   per_array      the engine's default: per array, a gather into a shared buffer of its element type and a copy back
#   fused_copy     one launch gathers every array into a second set of arrays, one launch copies them back
#                  (the engine's path for a workspace built with `append`)
#   per_array_swap per array, a gather into a second array and no copy back (a swap of buffers)
#   fused_swap64   one launch gathers every array into a second set, no copy back, `Int64` permutation
#   fused_swap32   the same with an `Int32` permutation
# and the sorts: `sort_by_cell!` (key sort, gather, segments) and `resort_by_cell!` (from the
# order of the last sort), both with the engine's default gather of the tree the script runs in
# (`sort_by_cell_noswap` and `resort_by_cell_noswap`: with `Workspace(; swap = false)` where the
# tree has that keyword).
#
# The state: a species sorted by nearest-node bins (order 2, `Node()` on both axes, the module's
# default), 16 particles per cell, the last 3 % of the slots free, then 30 % of the alive
# particles moved to a neighbouring cell on one axis: the permutation of one step's sort. The
# times are the median of `reps` calls after two warm-up calls, the device synchronized around
# each. Prints a table and writes `scripts/runs/sort-gather-<backend>-<stamp>/` with the
# numbers (`gather.json`) and the run's manifest (F3). One run on one machine; not reviewed.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ForwardDiff: Dual
using Random

const PMC = ParticleMesh.ParticleMeshCore
using .PMC: Workspace, sort_by_cell!, resort_by_cell!, Species, NodeBins, Quadratic, Node, PeriodicGrid, flux_extras, capacity,
            ALIVE, FREE, launcher, species_arrays, gather_all_kernel!, copy_all_kernel!, gather_perm_kernel!

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
    gpu_name() = CUDACore.name(CUDACore.device())
elseif BACKEND_NAME == "cpu"
    const BE = CPU()
    gpu_name() = "none"
else
    error("gather_bench.jl: the first argument is cpu or cuda, got $BACKEND_NAME")
end

const SIZES = Dict("S" => (64, 56), "M" => (256, 224), "L" => (896, 256))
const PPC = 16
const FREE_FRACTION = 0.03
const MOVED = 0.3
const SIZE_LIST = length(ARGS) >= 2 ? split(ARGS[2], ",") : ["S", "M", "L"]
const TYPE_LIST = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["Float32", "Float64", "Dual"]
element_type(name) = name == "Float32" ? Float32 : name == "Float64" ? Float64 : Dual{Nothing,Float32,3}

host_value(::Type{T}, x) where {T<:AbstractFloat} = T(x)
host_value(::Type{Dual{Nothing,F,K}}, x) where {F,K} = Dual{Nothing}(F(x), ntuple(k -> F(x / (k + 1)), Val(K))...)

# The sorted state with one step's moves: host arrays, written into `s`.
function fill_state!(s, n, rng)
    T = eltype(s)
    cap = capacity(s)
    alive = round(Int, cap * (1 - FREE_FRACTION))
    cx = zeros(Int32, cap); cy = zeros(Int32, cap)
    ox = zeros(T, cap); oy = zeros(T, cap)
    for i in 1:alive
        c = (i - 1) ÷ PPC        # column-major cell, `PPC` per cell
        cx[i] = Int32(c % n[1] + 1); cy[i] = Int32(c ÷ n[1] + 1)
        ox[i] = host_value(T, rand(rng)); oy[i] = host_value(T, rand(rng))
    end
    for i in 1:alive
        rand(rng) < MOVED || continue
        step = rand(rng, (-1, 1))
        if rand(rng, Bool)
            cx[i] = Int32(mod1(cx[i] + step, n[1]))
        else
            cy[i] = Int32(mod1(cy[i] + step, n[2]))
        end
    end
    status = [i <= alive ? ALIVE : FREE for i in 1:cap]
    copyto!(s.cell[1], cx); copyto!(s.cell[2], cy)
    copyto!(s.offset[1], ox); copyto!(s.offset[2], oy)
    for d in 1:3
        copyto!(s.u[d], [host_value(T, randn(rng)) for _ in 1:cap])
    end
    copyto!(s.w, fill(host_value(T, 0.5), cap))
    copyto!(s.status, status)
    copyto!(s.extras.old_cell_1, cx); copyto!(s.extras.old_cell_2, cy)
    copyto!(s.extras.old_offset_1, ox); copyto!(s.extras.old_offset_2, oy)
    return s
end

gather_arrays(s) = species_arrays(s)

# One step's moves of a sorted state: 30 % of the alive particles to a neighbouring cell on one axis.
function move!(s, n, seed)
    cx, cy = Array(s.cell[1]), Array(s.cell[2])
    rng = MersenneTwister(seed)
    alive = Array(s.status) .== ALIVE
    for i in eachindex(cx)
        (alive[i] && rand(rng) < MOVED) || continue
        st = rand(rng, (-1, 1))
        rand(rng, Bool) ? (cx[i] = Int32(mod1(cx[i] + st, n[1]))) : (cy[i] = Int32(mod1(cy[i] + st, n[2])))
    end
    copyto!(s.cell[1], cx); copyto!(s.cell[2], cy)
    return s
end

@kernel function gather32_kernel!(dst, @Const(src), @Const(perm))
    i = @index(Global, Linear)
    @inbounds PMC.copy_each!(dst, src, i, perm[i])
end

function median_time(f; reps)
    f(); f()
    KernelAbstractions.synchronize(BE)
    ts = Float64[]
    for _ in 1:reps
        KernelAbstractions.synchronize(BE)
        t0 = time_ns()
        f()
        KernelAbstractions.synchronize(BE)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return median(ts)
end

function bench(sizename, tname; reps)
    T = element_type(tname)
    n = SIZES[sizename]
    cap = round(Int, prod(n) * PPC / (1 - FREE_FRACTION))
    grid = PeriodicGrid(n, (T(0.1), T(0.1)), 3)
    bins = NodeBins(Quadratic, (Node(), Node()))
    s = Species(BE, T, cap; q = -1, m = 1, N = 2, extras = flux_extras(T, 2))
    fill_state!(s, n, MersenneTwister(1))
    saved = map(copy, gather_arrays(s))
    ws = Workspace(s, grid; bins, append = 0)
    sort_by_cell!(s, grid, ws)                     # ws.perm: this step's permutation
    perm32 = similar(ws.perm, Int32); copyto!(perm32, Int32.(Array(ws.perm)))
    perm64 = similar(ws.perm, Int64); copyto!(perm64, Int64.(Array(ws.perm)))
    a = gather_arrays(s)
    spare = map(similar, a)
    tmp = Dict(E => similar(s.w, E, cap) for E in unique(map(eltype, a)))
    wg = nothing
    out = Dict{String,Any}()
    out["per_array"] = median_time(; reps) do
        for x in a
            launcher(gather_perm_kernel!, BE, wg)(tmp[eltype(x)], x, perm64; ndrange = cap)
            copyto!(x, tmp[eltype(x)])
        end
    end
    out["fused_copy"] = median_time(; reps) do
        launcher(gather_all_kernel!, BE, wg)(spare, a, perm64; ndrange = cap)
        launcher(copy_all_kernel!, BE, wg)(a, spare; ndrange = cap)
    end
    out["per_array_swap"] = median_time(; reps) do
        for (x, y) in zip(a, spare)
            launcher(gather_perm_kernel!, BE, wg)(y, x, perm64; ndrange = cap)
        end
    end
    out["fused_swap64"] = median_time(() -> launcher(gather_all_kernel!, BE, wg)(spare, a, perm64; ndrange = cap); reps)
    out["fused_swap32"] = median_time(() -> launcher(gather32_kernel!, BE, wg)(spare, a, perm32; ndrange = cap); reps)
    # the whole sorts, each from the moved state: the full sort, and the sort from the last order
    restore!() = foreach(copyto!, gather_arrays(s), saved)
    sorted = (restore!(); sort_by_cell!(s, grid, ws); map(copy, gather_arrays(s)))
    function time_sort(f)
        ts = Float64[]
        for r in 1:(reps + 2)
            foreach(copyto!, gather_arrays(s), sorted)
            sort_by_cell!(s, grid, ws)                                 # the order of the last sort
            move!(s, n, r)
            KernelAbstractions.synchronize(BE)
            t0 = time_ns()
            f()
            KernelAbstractions.synchronize(BE)
            r > 2 && push!(ts, (time_ns() - t0) / 1e6)
        end
        return median(ts)
    end
    out["sort_by_cell"] = time_sort(() -> sort_by_cell!(s, grid, ws))
    out["resort_by_cell"] = time_sort(() -> resort_by_cell!(s, grid, ws))
    if :swap in Base.kwarg_decl(only(methods(Workspace, (typeof(s), typeof(grid)))))
        ws = Workspace(s, grid; bins, append = 0, swap = false)
        out["sort_by_cell_noswap"] = time_sort(() -> sort_by_cell!(s, grid, ws))
        out["resort_by_cell_noswap"] = time_sort(() -> resort_by_cell!(s, grid, ws))
    end
    # the bits: the default sort and resort against the copy-back sort, from the same moved state
    if haskey(out, "sort_by_cell_noswap")
        bits(x) = reinterpret(UInt8, Array(x))
        same = true
        for f in (sort_by_cell!, resort_by_cell!)
            states = map((true, false)) do sw
                foreach(copyto!, gather_arrays(s), saved)
                w = Workspace(s, grid; bins, append = 64, swap = sw)
                sort_by_cell!(s, grid, w)
                move!(s, n, 99)
                f(s, grid, w)
                (map(a -> bits(a), gather_arrays(s)), Array(w.perm), Array(w.seg_start), Array(w.seg_stop), Array(s.count))
            end
            same &= states[1] == states[2]
        end
        out["same_bits_swap_noswap"] = same
    end
    out["particles"] = cap
    out["bytes_per_slot"] = sum(x -> sizeof(eltype(x)), a)
    return out
end

function main()
    reps = BACKEND_NAME == "cuda" ? 30 : 7
    forms = ("per_array", "fused_copy", "per_array_swap", "fused_swap64", "fused_swap32", "sort_by_cell", "resort_by_cell",
             "sort_by_cell_noswap", "resort_by_cell_noswap")
    results = Dict{String,Any}()
    println("| size | type | slots | B/slot | ", join(forms, " | "), " | same bits |")
    println("|", repeat("---|", 5 + length(forms)))
    for sz in SIZE_LIST, tn in TYPE_LIST
        r = bench(sz, tn; reps)
        results["$sz/$tn"] = r
        println("| $sz | $tn | $(r["particles"]) | $(r["bytes_per_slot"]) | ", join((haskey(r, f) ? string(round(r[f]; digits = 3)) : "-" for f in forms), " | "),
                " | ", get(r, "same_bits_swap_noswap", "-"), " |")
    end
    dir = run_dir("gather-$BACKEND_NAME", @__FILE__; prefix = "sort", work_package = "WP-041",
                  params = Dict{String,Any}("sizes" => SIZE_LIST, "types" => TYPE_LIST, "ppc" => PPC, "moved" => MOVED,
                                            "free_fraction" => FREE_FRACTION, "reps" => reps),
                  backend = string(BE), extra = Dict{String,Any}("gpu" => gpu_name()))
    open(io -> JSON.print(io, Dict("unit" => "ms", "results" => results), 2), joinpath(dir, "gather.json"), "w")
    println("wrote ", dir)
end

main()
