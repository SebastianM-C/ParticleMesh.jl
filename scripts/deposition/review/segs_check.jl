# The full sort (`sort_by_cell!`) of the engine against a reference computed on the host, and digests
# of what it leaves, to compare two trees bit for bit (main against the head of #480, which takes the
# segment bounds from the sorted keys):
#
#   julia +1.13 -t 4 --project=<repo>/scripts segs_check.jl <repo> [cpu|cuda] > <tree>.txt   # then diff two trees' files
#
# Synthetic states (scripts/deposition/common.jl) of Float32, Float64 and Dual{Float32,3} on grids of
# 23, 37 × 23, 8 × 9 and 9 × 8 × 8 cells, sorted by the cells, by `NodeBins` of order 2 on nodes, of
# order 1 at centres, and of order 2 with the cell on x (as the module bins a window's species), in
# the patterns:
#   mixed        four slots per cell, a fifth free, moved by up to 0.99 cells, 5% killed
#   sparse       a twentieth of the slots alive, moved: most bins are empty
#   full         every slot alive, moved: the last alive slot is the last slot
#   thin         one slot per cell, every slot alive, moved
#   empty        no slot alive
#   last slot    the last slot alone alive
#   first slot   the first slot alone alive
#   last cell    the slots of the last cell alone alive, not moved: by node bins their upper halves
#                wrap to the first bin of an axis
# Per case, on the host: the alive slots are the first `count`; the bin of every alive slot (the
# engine's `particle_bin`, evaluated here slot by slot) does not decrease; each bin's segment is the
# first and the last slot of that bin, and (1, 0) for an empty bin; within a bin the slots keep the
# order they had; every array is the unsorted one gathered by the permutation; the launch shape 64
# gives the same bits. On a device, the arrays, the permutation and the segments are also compared
# with the CPU's. Prints a digest per case and throws on a failure.
const REPO = ARGS[1]
include(joinpath(REPO, "scripts", "engine2d", "common.jl"))
include(joinpath(REPO, "scripts", "deposition", "common.jl"))
const NAME = length(ARGS) >= 2 ? ARGS[2] : "cpu"
if NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
    CUDACore.allowscalar(false)
else
    const BE = CPU()
end
const FAILED = String[]
bits(a::AbstractArray) = reinterpret(UInt8, vec(Array(a)))
digest(arrays) = string(foldl((h, a) -> hash(collect(bits(a)), h), arrays; init = UInt(0)); base = 16, pad = 16)
type_label(::Type{T}) where {T} = T <: Dual ? "Dual{$(base_type(T)),3}" : string(T)
const PATTERNS = ("mixed", "sparse", "full", "thin", "empty", "last slot", "first slot", "last cell")

function state(be, ::Type{T}, grid::PMC.PeriodicGrid{N}, pattern) where {T,N}
    fill = pattern in ("full", "thin") ? 1.0 : pattern == "sparse" ? 0.05 : 0.8
    s = synthetic_species(be, T, grid; ppc = pattern == "thin" ? 1 : 4, fill)
    pattern in ("mixed", "sparse", "full", "thin") && move!(s, grid; dmax = 0.99, kill = pattern == "mixed" ? 0.05 : 0.0)
    status = Array(s.status)
    if pattern == "empty"
        status .= PMC.FREE
    elseif pattern == "last slot"
        status .= PMC.FREE
        status[end] = PMC.ALIVE
    elseif pattern == "first slot"
        status .= PMC.FREE
        status[1] = PMC.ALIVE
    elseif pattern == "last cell"
        cell = map(Array, s.cell)
        status .= [all(d -> cell[d][i] == grid.n[d], 1:N) ? PMC.ALIVE : PMC.FREE for i in eachindex(status)]
    end
    copyto!(s.status, status)
    return s
end

"The sorted species, permutation, segments and count on the host: whether they are the sort of `u` by the workspace's bins."
function is_the_sort(s::PMC.Species{N}, u, grid, bins, perm, a, b, count) where {N}
    n = Tuple(Int32.(grid.n))
    ncell = prod(grid.n)
    cap = PMC.capacity(s)
    sort(perm) == 1:cap || return "the permutation is not one"
    all(((x, y),) -> bits(x) == bits(y[perm]), zip(PMC.species_arrays(s), PMC.species_arrays(u))) || return "an array is not the gather by the permutation"
    status = s.status
    ra, rb = ones(Int32, ncell), zeros(Int32, ncell)
    seen, last_bin = 0, 0
    for i in 1:cap
        status[i] == PMC.ALIVE || continue
        seen == i - 1 || return "an alive slot behind a free one"
        L = Int(PMC.linear_cell(map(Int64, PMC.particle_bin(bins, s, i, n)), map(Int64, n)))
        L >= last_bin || return "the bins decrease"
        L == last_bin ? (perm[i] > perm[i - 1] || return "a bin's slots changed their order") : (ra[L] = i)
        rb[L] = i
        seen, last_bin = i, L
    end
    issorted(@view perm[(seen + 1):cap]) || return "the free slots changed their order"
    count == seen || return "the count is $count, $seen slots are alive"
    a == ra || return "seg_start differs from the host's in $(Base.count(a .!= ra)) bins"
    b == rb || return "seg_stop differs from the host's in $(Base.count(b .!= rb)) bins"
    return ""
end

host(s::PMC.Species{N,T}) where {N,T} = to_host!(PMC.Species(CPU(), T, PMC.capacity(s); q = s.q, m = s.m, N, extras = map(eltype, s.extras)), s)
function to_host!(dst, src)
    foreach((x, y) -> copyto!(x, Array(y)), PMC.species_arrays(dst), PMC.species_arrays(src))
    copyto!(dst.count, Array(src.count))
    return dst
end

function sorted(be, ::Type{T}, grid, bins, pattern; workgroupsize = nothing) where {T}
    s = state(be, T, grid, pattern)
    u = host(s)
    ws = PMC.Workspace(s, grid; bins)
    PMC.sort_by_cell!(s, grid, ws; workgroupsize)
    return (; s = host(s), u, perm = Array(ws.perm), a = Array(ws.seg_start), b = Array(ws.seg_stop), count = Int(Array(s.count)[1]),
            key = Array(ws.key))
end
all_arrays(r) = (PMC.species_arrays(r.s)..., r.perm, r.a, r.b, [r.count], r.key)

function case(::Type{T}, grid::PMC.PeriodicGrid{N}, bins_label, bins, pattern) where {T,N}
    label = rpad("$(type_label(T)) $(join(grid.n, "x")) $bins_label, $pattern", 78)
    r = sorted(BE, T, grid, bins, pattern)
    why = is_the_sort(r.s, r.u, grid, bins, r.perm, r.a, r.b, r.count)
    r64 = sorted(BE, T, grid, bins, pattern; workgroupsize = 64)
    why == "" && digest(all_arrays(r64)) != digest(all_arrays(r)) && (why = "the launch shape 64 gives other bits")
    if why == "" && NAME == "cuda"
        c = sorted(CPU(), T, grid, bins, pattern)
        digest(all_arrays(c)) == digest(all_arrays(r)) || (why = "the device's bits differ from the CPU's")
    end
    why == "" || push!(FAILED, "$label: $why")
    println(label, digest(all_arrays(r)), "  alive ", rpad(r.count, 6), why == "" ? "" : "FAILED: $why")
    flush(stdout)
    return nothing
end

function main()
    N_, C_ = PMC.Node(), PMC.Center()
    D3 = Dual{Nothing,Float32,3}
    for T in (Float32, Float64, D3)
        V = base_type(T)
        grids = (PMC.PeriodicGrid(23, T(V(0.4)), 3), PMC.PeriodicGrid((37, 23), ntuple(d -> T(V(0.3) * d), 2), 3),
                 PMC.PeriodicGrid((8, 9), ntuple(d -> T(V(0.3) * d), 2), 3), PMC.PeriodicGrid((9, 8, 8), ntuple(d -> T(V(0.3) * d), 3), 3))
        for grid in grids
            N = length(grid.n)
            binnings = (("cells", PMC.CellBins()), ("order 2 on nodes", PMC.NodeBins(PMC.Quadratic, ntuple(_ -> N_, N))),
                        ("order 1 at centres", PMC.NodeBins(PMC.Linear, ntuple(_ -> C_, N))),
                        ("order 2, the cell on x", PMC.NodeBins(PMC.Quadratic, ntuple(d -> d == 1 ? C_ : N_, N))))
            for (bl, bins) in binnings, pattern in PATTERNS
                case(T, grid, bl, bins, pattern)
            end
        end
    end
    if isempty(FAILED)
        println(stderr, "backend ", NAME, ": the full sort is the host's in every case")
    else
        println(stderr, "backend ", NAME, ": FAILED in ", length(FAILED), " cases")
        foreach(f -> println(stderr, "  ", f), FAILED)
        exit(1)
    end
end
Base.invokelatest(main)
