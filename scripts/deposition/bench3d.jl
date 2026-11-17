# A 3D flux deposition and the sorts on a synthetic 3D state (the engine has no 3D flux
# deposition; `sort_by_cell!` is generic in N):
#
#   julia +1.13 -t 4 --project=scripts scripts/deposition/bench3d.jl [cpu|cuda] [cells, e.g. 64x64x64] [ppc] [f32,f64,d32]
#
# The state is `ppc` particles per cell at random offsets, every one moved by a random
# displacement below half a cell per axis (common.jl). The sorts are timed on the moved state
# with the segments of the sort before the move, the depositions on the sorted state; each time
# is the least of 7 calls with the device waited for (`PM_BENCH_REPS` sets another count). The
# variants are compared with one another before they are timed (check.jl holds each to continuity
# on a small grid). A tile size whose buffers would take more than 3 GiB is skipped
# (`PM_BENCH_MAX_GIB` sets another limit), and so are the atomics in the 6-wide frame above
# 40 million particles, where one call takes tens of seconds. Writes `bench3d.json`, `tables.md`
# and a manifest into scripts/runs/deposition-bench3d-<backend>-<time>/. `PM_BENCH_ONLY` (substrings
# of the row names, separated by commas) limits a run to the rows that match.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "common.jl"))

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
elseif BACKEND_NAME == "cpu"
    const BE = CPU()
else
    error("bench3d.jl: the first argument is cpu or cuda, got $BACKEND_NAME")
end

const REPS = parse(Int, get(ENV, "PM_BENCH_REPS", "7"))
const MAX_GIB = parse(Float64, get(ENV, "PM_BENCH_MAX_GIB", "3"))
const ONLY = filter(!isempty, split(get(ENV, "PM_BENCH_ONLY", ""), ","))
wanted(name) = isempty(ONLY) || any(o -> occursin(o, name), ONLY)
const ELEMENTS = Dict("f32" => Float32, "f64" => Float64, "d32" => Dual{Nothing,Float32,3})
type_label(::Type{T}) where {T} = T <: Dual ? "Dual{$(base_type(T)),3}" : string(T)

function shuffled(s)
    c = copy_species(s)
    perm = randperm(Xoshiro(3), PMC.capacity(s))
    foreach((a, b) -> copyto!(a, Array(b)[perm]), species_arrays(c), species_arrays(s))
    return c
end

ms(t) = 1e3 * t

function measure(::Type{T}, n, ppc) where {T}
    V = base_type(T)
    shape, stg = PMC.Quadratic, (PMC.Node(), PMC.Node(), PMC.Node())
    grid = PMC.PeriodicGrid(n, ntuple(_ -> T(V(0.2)), 3), 3)
    dt = T(V(0.1))
    sx = map(h -> h / dt, grid.dx)
    s = synthetic_species(BE, T, grid; ppc)
    ws = PMC.Workspace(s, grid)
    PMC.sort_by_cell!(s, grid, ws)
    seg0 = (copy(ws.seg_start), copy(ws.seg_stop))
    move!(s, grid; dmax = 0.5)
    U = copy_species(s)
    time_ = Dict{String,Any}()
    diff = Dict{String,Any}()

    # the sorts, on the moved state
    timed!(f, name; kw...) = wanted(name) && (time_[name] = ms(least(f, BE; reps = REPS, kw...)); nothing)
    timed!(() -> PMC.sort_by_cell!(s, grid, ws), "engine sort_by_cell!"; setup = () -> restore!(s, U))
    timed!("engine sort: keys and sortperm! alone"; setup = () -> restore!(s, U)) do
        PMC.launcher(PMC.sort_key_nd_kernel!, BE, nothing)(ws.key, s.cell, s.status, Int64.(grid.n), Int64(prod(grid.n)), Int64(PMC.capacity(s));
                                                           ndrange = PMC.capacity(s))
        PMC.AK.sortperm!(ws.perm, ws.key; workspace = ws.ak)
    end
    if wanted("engine sort: keys, sort!")                         # the permutation from a plain sort of the keys (they end in the slot)
        restore!(s, U)
        PMC.launcher(PMC.sort_key_nd_kernel!, BE, nothing)(ws.key, s.cell, s.status, Int64.(grid.n), Int64(prod(grid.n)), Int64(PMC.capacity(s)); ndrange = PMC.capacity(s))
        PMC.AK.sortperm!(ws.perm, ws.key; workspace = ws.ak)
        perm0 = Array(ws.perm)
        for (alg, tag) in (BE isa CPU ? ((PMC.AK.Auto(), "the default sort"),) : ((PMC.AK.MergeSort(), "merge sort"), (PMC.AK.RadixSort(), "radix sort")))
            name = "engine sort: keys, sort! (" * tag * ") and decode alone"
            wk = PMC.AK.workspace(PMC.AK.sort!, ws.key; alg)
            perm_by_key_sort!(ws, s, grid; alg, workspace = wk)
            diff[name] = Array(ws.perm) != perm0
            timed!(() -> perm_by_key_sort!(ws, s, grid; alg, workspace = wk), name; setup = () -> restore!(s, U))
        end
    end
    restore!(s, U)
    PMC.sort_by_cell!(s, grid, ws)
    n1 = PMC.alive_count(s)
    same_as_engine(b) = all(((p, q),) -> Array(p)[1:n1] == Array(q)[1:n1], zip(species_arrays(b.dst), species_arrays(s))) &&
                        Array(b.seg_start) == Array(ws.seg_start) && Array(b.seg_stop) == Array(ws.seg_stop) && PMC.alive_count(b.dst) == n1
    for (name, make, sort!) in (("incremental sort", IncrementalSort, incremental_sort!), ("incremental sort, scatter form", ScatterSort, scatter_sort!),
                               ("incremental sort, permutation form", PermSort, perm_sort!))
        wanted(name) || continue
        b = make(s, grid)
        sort!(b, U, grid, seg0...)
        diff[name] = !same_as_engine(b)
        b isa PermSort && timed!(() -> incremental_perm!(b, U, grid, seg0...), "$name, the permutation alone")
        timed!(() -> sort!(b, U, grid, seg0...), "$name, two buffers")
        work = copy_species(U)
        segW = map(copy, seg0)
        timed!(() -> sort!(b, work, grid, segW...; copy_back = true), "$name, copied back";
               setup = () -> (restore!(work, U); foreach(copyto!, segW, seg0)))
    end

    # the depositions, on the sorted state (`s`, with its segments in `ws`)
    seg = (ws.seg_start, ws.seg_stop)
    J = zero_fields(BE, T, grid)
    ref = zero_fields(BE, T, grid)
    tb4 = TileBuffers(BE, T, grid, shape, stg, 4)
    deposit_tiles!(ref, tb4, s, MinFrame(), shape, stg, grid, seg..., sx, nothing)
    function variant!(f, name; fold = false)                     # `fold`: the variant bins its particles in other cells at the seam
        wanted(name) || return nothing
        a = f(zero_fields(BE, T, grid))
        diff[name] = fold ? field_difference(folded_host(a, grid), folded_host(ref, grid)) : field_difference(a, ref)
        time_[name] = ms(least(() -> f(J), BE; reps = REPS))
        return nothing
    end
    for (stp, name) in ((stg, "patch in registers, 5 × 6 × 6 per component"),
                        (ntuple(_ -> PMC.Center(), 3), "patch in registers, nearest-node binning, 4 × 5 × 5 per component"))
        wanted(name) || continue
        near = stp !== stg
        bytes = (near ? 100 : 180) * prod(n) * sizeof(T)
        if bytes > MAX_GIB * 2^30
            println("skipping $name: its patches would take $(round(bytes / 2^30; digits = 1)) GiB")
            continue
        end
        sp = near ? half_cell_shifted(s, grid) : s
        wsp = near ? PMC.Workspace(sp, grid) : ws
        near && PMC.sort_by_cell!(sp, grid, wsp)
        patch = patch3_buffer(BE, T, grid, shape, stp)
        variant!(a -> deposit_patch3!(a, patch, sp, shape, stp, grid, wsp.seg_start, wsp.seg_stop, sx), name; fold = near)
        if near                                                  # the straight-line stencil, and fewer accumulators per launch for wide element types
            full = deposit_patch3!(zero_fields(BE, T, grid), patch, sp, shape, stp, grid, wsp.seg_start, wsp.seg_stop, sx)
            for layers in (5, 3, 2, 1)
                tag = "patch in registers, nearest-node binning, straight-line stencil, $(layers == 5 ? "all" : layers) planes per launch"
                wanted(tag) || continue
                b = deposit_patch3_slabs!(zero_fields(BE, T, grid), patch, sp, shape, stp, grid, wsp.seg_start, wsp.seg_stop, sx, layers)
                diff[tag * " == the patch above (bits)"] = !all(map((x, y) -> Array(x) == Array(y), full, b))
                variant!(a -> deposit_patch3_slabs!(a, patch, sp, shape, stp, grid, wsp.seg_start, wsp.seg_stop, sx, layers), tag; fold = true)
            end
        end
    end
    variant!(a -> deposit_tiles!(a, tb4, s, MinFrame(), shape, stg, grid, seg..., sx, nothing), "tiles S = 4")
    BE isa CPU && variant!(a -> deposit_tiles!(a, tb4, s, MinFrame(), shape, stg, grid, seg..., sx, nothing; launch = PMC.item_launcher),
                           "tiles S = 4, work-groups of one")
    for S in (1, 2, 8)
        wanted("tiles S = $S") || continue
        bytes = 3 * (S + 5)^3 * prod(cld.(n, S)) * sizeof(T)
        if bytes > MAX_GIB * 2^30
            println("skipping tiles S = $S: its buffers would take $(round(bytes / 2^30; digits = 1)) GiB")
            continue
        end
        tb = TileBuffers(BE, T, grid, shape, stg, S)
        variant!(a -> deposit_tiles!(a, tb, s, MinFrame(), shape, stg, grid, seg..., sx, nothing), "tiles S = $S")
    end
    for S in (5, 8)
        variant!(a -> deposit_coloured!(a, s, MinFrame(), shape, stg, grid, seg..., sx, nothing, S), "coloured tiles S = $S")
        BE isa CPU && S == 8 &&
            variant!(a -> deposit_coloured!(a, s, MinFrame(), shape, stg, grid, seg..., sx, nothing, S; launch = PMC.item_launcher),
                     "coloured tiles S = $S, work-groups of one")
    end
    if T <: AbstractFloat && wanted("atomics")
        wide = n1 <= 40_000_000
        wide || println("skipping the atomics in the 6-wide frame: $n1 particles")
        variant!(a -> deposit_atomic!(a, s, MinFrame(), shape, stg, grid, sx, nothing), "atomics 4 × 4 × 4, sorted particles")
        wide && variant!(a -> deposit_atomic!(a, s, CellFrame(), shape, stg, grid, sx, nothing), "atomics 6 × 6 × 6, sorted particles")
        let r = shuffled(s)
            variant!(a -> deposit_atomic!(a, r, MinFrame(), shape, stg, grid, sx, nothing), "atomics 4 × 4 × 4, shuffled particles")
            wide && variant!(a -> deposit_atomic!(a, r, CellFrame(), shape, stg, grid, sx, nothing), "atomics 6 × 6 × 6, shuffled particles")
        end
    end
    return Dict{String,Any}("T" => type_label(T), "cells" => collect(n), "ppc" => ppc, "alive" => n1, "capacity" => PMC.capacity(s),
                            "ms" => time_, "difference" => diff)
end

fmt(x) = x >= 100 ? string(round(Int, x)) : x >= 10 ? string(round(x; digits = 1)) : x >= 1 ? string(round(x; digits = 2)) : string(round(x; sigdigits = 2))

function table(rows)
    io = IOBuffer()
    r1 = first(rows)
    println(io, "$(join(r1["cells"], " × ")) cells, $(r1["ppc"]) particles per cell, $(r1["alive"]) particles. Milliseconds per call (nanoseconds per particle).\n")
    println(io, "| | ", join(("`$(r["T"])`" for r in rows), " | "), " |")
    println(io, "|---|", join(("---" for _ in rows), "|"), "|")
    for name in sort(unique(vcat((collect(keys(r["ms"])) for r in rows)...)))
        println(io, "| ", name, " | ", join((haskey(r["ms"], name) ? "$(fmt(r["ms"][name])) ($(fmt(1e6 * r["ms"][name] / r["alive"])))" : "" for r in rows), " | "), " |")
    end
    return String(take!(io))
end

function main()
    n = length(ARGS) >= 2 ? Tuple(parse.(Int, split(ARGS[2], "x"))) : (32, 32, 32)
    ppc = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    elements = length(ARGS) >= 4 ? split(ARGS[4], ",") : ["f32", "f64", "d32"]
    gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
    extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
    dir = run_dir("bench3d-$BACKEND_NAME", @__FILE__; prefix = "deposition", work_package = "deposition study",
                  backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                  params = Dict{String,Any}("cells" => collect(n), "ppc" => ppc, "elements" => collect(String, elements), "reps" => REPS,
                                            "max_buffer_gib" => MAX_GIB, "only" => collect(String, ONLY)),
                  extra_inputs = [joinpath(@__DIR__, "proto.jl"), joinpath(@__DIR__, "common.jl")], extra)
    rows = Dict{String,Any}[]
    for e in elements
        push!(rows, measure(ELEMENTS[e], n, ppc))
        println(table(rows[end:end]))
        bad = filter(((k, v),) -> v isa Bool ? v : v > (occursin("32", rows[end]["T"]) ? 1e-3 : 1e-10), rows[end]["difference"])
        isempty(bad) || println("DIFFERENCES ABOVE ROUNDOFF: ", bad)
        open(io -> JSON.print(io, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => cpu_model(), "gpu" => gpu, "backend" => BACKEND_NAME), 2),
             joinpath(dir, "bench3d.json"), "w")
    end
    write(joinpath(dir, "tables.md"), table(rows))
    println("wrote ", relpath(dir, REPO))
    return dir
end

Base.invokelatest(main)
