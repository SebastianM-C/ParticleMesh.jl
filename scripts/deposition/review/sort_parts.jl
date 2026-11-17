# Where a sort by nearest-node bins spends more than a sort by cell: the key kernel, the sort of
# the keys, the gather and the segment kernel, each timed alone on a backend, on a synthetic state
# (scripts/deposition/common.jl) that was sorted by the workspace's bins and then moved by up to
# half a cell per axis:
#
#   julia +1.13 -t 4 --project=<repo>/scripts sort_parts.jl <repo> [cpu|cuda] [cells per axis, 224x64] [ppc, 4]
#
# Prints, per element type, milliseconds per call (the least of 15) by cell and by `NodeBins` of
# order 2 on nodes.
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
const CELLS = length(ARGS) >= 3 ? Tuple(parse.(Int, split(ARGS[3], "x"))) : (224, 64)
const PPC = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4

function parts(::Type{T}, bins) where {T}
    V = base_type(T)
    grid = PMC.PeriodicGrid(CELLS, ntuple(d -> T(V(0.3) * d), 2), 3)
    s = synthetic_species(BE, T, grid; ppc = PPC, fill = 0.9)
    ws = PMC.Workspace(s, grid; flux = (PMC.Quadratic, (PMC.Node(), PMC.Node())), bins)
    PMC.sort_by_cell!(s, grid, ws)
    move!(s, grid; dmax = 0.5)
    U = copy_species(s)
    cap = Int64(PMC.capacity(s))
    n = Int32.(grid.n)
    launch(k) = PMC.launcher(k, BE, nothing)
    reset() = restore!(s, U)
    keys!() = launch(PMC.sort_key_nd_kernel!)(ws.key, s, ws.bins, Int64.(grid.n), Int64(prod(grid.n)), cap; ndrange = cap)
    t = Dict{String,Float64}()
    t["whole sort"] = least(() -> PMC.sort_by_cell!(s, grid, ws), BE; reps = 15, setup = reset)
    t["keys"] = least(keys!, BE; reps = 15, setup = reset)
    t["sort of the keys"] = least(() -> PMC.sort_keys!(ws, cap, BE, nothing), BE; reps = 15, setup = () -> (reset(); keys!()))
    t["gather"] = least(() -> PMC.permute_species!(s, ws, nothing), BE; reps = 15, setup = () -> (reset(); keys!(); PMC.sort_keys!(ws, cap, BE, nothing)))
    PMC.sort_by_cell!(reset(), grid, ws)
    t["segments"] = least(BE; reps = 15) do
        launch(PMC.reset_segments_kernel!)(ws.seg_start, ws.seg_stop, s.count; ndrange = prod(grid.n))
        if isdefined(PMC, :segments_from_keys_kernel!)       # #480: the bounds from the sorted keys, which the sort above left in `ws.key`
            launch(PMC.segments_from_keys_kernel!)(ws.seg_start, ws.seg_stop, s.count, ws.key, Int64(prod(grid.n)), cap; ndrange = cap)
        else
            launch(PMC.segments_nd_kernel!)(ws.seg_start, ws.seg_stop, s.count, s, ws.bins, n, Int32(cap); ndrange = cap)
        end
    end
    return t, alive_count(s)
end

function main()
    println("backend ", NAME, ", ", join(CELLS, " × "), " cells, ", PPC, " slots per cell; ms per call")
    for T in (Float32, Float64)
        c, alive = parts(T, PMC.CellBins())
        b, _ = parts(T, PMC.NodeBins(PMC.Quadratic, (PMC.Node(), PMC.Node())))
        println(T, ", ", alive, " alive")
        for k in ("whole sort", "keys", "sort of the keys", "gather", "segments")
            println("  ", rpad(k, 18), "by cell ", rpad(round(1e3 * c[k]; sigdigits = 3), 8), "by node bins ", round(1e3 * b[k]; sigdigits = 3))
        end
    end
end
Base.invokelatest(main)
