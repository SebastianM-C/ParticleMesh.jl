# The engine's sort and 2D flux deposition against the prototypes of proto.jl, on states of the
# WP-045 case (scripts/optim/wake2d_case.jl):
#
#   julia +1.13 -t 4 --project=scripts scripts/deposition/bench2d.jl [cpu|cuda] [S,M,L] [f32,f64,d32,d64]
#
# Per size and element type the case runs to its end in Reference mode; one more push and a sort
# give the sorted state every deposition variant is timed on, and one further push the unsorted
# state (with the segments of the sort before it) that the sorts are timed on. Each time is the
# least of 9 calls, with the device waited for. Every variant is compared with the engine on the
# same state before it is timed (the incremental sort bitwise, the depositions to roundoff), and
# the differences are in the output. Writes `bench2d.json`, `tables.md` and a manifest into
# scripts/runs/deposition-bench2d-<backend>-<time>/. `PM_BENCH_ONLY` (substrings of the row names,
# separated by commas) limits a run to the rows that match.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "..", "optim", "wake2d_case.jl"))
include(joinpath(@__DIR__, "common.jl"))

const PME = ParticleMeshElectromagnetic

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
elseif BACKEND_NAME == "cpu"
    const BE = CPU()
else
    error("bench2d.jl: the first argument is cpu or cuda, got $BACKEND_NAME")
end

const ONLY = filter(!isempty, split(get(ENV, "PM_BENCH_ONLY", ""), ","))
wanted(name) = isempty(ONLY) || any(o -> occursin(o, name), ONLY)

"The sizes: the case's box at `ppl` cells per wavelength along x, Δy = `dy_cells`·Δx, run to t ≈ 185."
const SIZES = Dict("S" => (ppl = 16, dy_cells = 4, ppc = (2, 2), nsteps = 512),
                   "M" => (ppl = 32, dy_cells = 4, ppc = (4, 4), nsteps = 1024),
                   "L" => (ppl = 32, dy_cells = 1, ppc = (4, 4), nsteps = 1400))

d2(::Type{T}) where {T} = (Dual{Nothing}(T(7), one(T), zero(T)), Dual{Nothing}(T(100), zero(T), one(T)))
const ELEMENTS = Dict("f32" => ("Float32", (7.0f0, 100.0f0)), "f64" => ("Float64", (7.0, 100.0)),
                      "d32" => ("Dual{Float32,2}", d2(Float32)), "d64" => ("Dual{Float64,2}", d2(Float64)))

"The module's push of one step, and the window's retirement after it (step.jl)."
function push_once!(st)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    for s in st.species
        PME.launcher(PME.push_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, Int32(guards(grid)), s.q / s.m * dt / 2,
                                                            map(h -> dt / h, grid.dx), Int32.(grid.n); ndrange = capacity(s))
    end
    PME.retire_x!(PME.window_frame(st), st)
    return st
end

function final_state(x, size, mode)
    prob = wake_problem(x...; SIZES[size]...)
    st = EMState(prob, EMNumerics(; mode); backend = BE, record = ())
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    return st
end

"A copy of the species with its slots in a random order."
function shuffled(s)
    c = copy_species(s)
    perm = randperm(Xoshiro(3), PMC.capacity(s))
    foreach((a, b) -> copyto!(a, Array(b)[perm]), species_arrays(c), species_arrays(s))
    return c
end

ms(t) = 1e3 * t

function measure(size, label, x)
    st = final_state(x, size, Reference())
    s, ws, grid = st.species[1], st.ws[1], st.grid
    T = eltype(s)
    shape, stg = st.numerics.shape, PME.rho_stagger(Val(2))
    qz = PME.transverse_currents(Val(2)).quantity_z
    sx = map(h -> h / st.dt, grid.dx)
    n = Int32.(grid.n)
    J = zero_fields(BE, T, grid)
    fresh() = zero_fields(BE, T, grid)
    seg = (ws.seg_start, ws.seg_stop)
    time_ = Dict{String,Any}()
    diff = Dict{String,Any}()

    # the sorted state: every deposition variant
    push_once!(st)
    sort_by_cell!(s, grid, ws)
    V = copy_species(s)
    segV = map(copy, seg)
    alive = alive_count(s)
    ref = fresh()
    PMC._deposit_flux_2d!(Reference(), ref..., s, shape, stg, grid, ws, sx, qz, nothing)
    folded(a) = (foreach(x -> fold_guards!(x, grid), a); a)
    ref_folded = folded(map(copy, ref))
    timed!(f, name; kw...) = wanted(name) && (time_[name] = ms(least(f, BE; kw...)); nothing)
    function variant!(f, name; fold = false)                     # `fold`: the variant bins its particles in other cells at the seam
        wanted(name) || return nothing
        diff[name] = fold ? field_difference(folded(f(fresh())), ref_folded) : field_difference(f(fresh()), ref)
        time_[name] = ms(least(() -> f(J), BE))
        return nothing
    end
    timed!(() -> push_once!(st), "push"; setup = () -> restore!(s, V))
    restore!(s, V)
    variant!(a -> PMC._deposit_flux_2d!(Reference(), a..., s, shape, stg, grid, ws, sx, qz, nothing), "engine: patch + reduce")
    M = PMC.flux_widths(shape, stg)
    timed!("engine: patch kernel alone") do
        PMC.item_launcher(PMC.flux_patch_2d_kernel!, BE, nothing)(ws.flux_patch, s, seg..., shape, stg, n, sx, qz, ws.flux_layout, Val(prod(M));
                                                                  ndrange = prod(n))
    end
    variant!(a -> deposit_patch!(a, s, factored_stencil, shape, stg, grid, ws, sx, qz), "patch, factored stencil")
    variant!(a -> deposit_patch!(a, s, :split, shape, stg, grid, ws, sx, qz), "patch, one component per launch")
    if wanted("nearest-node")
        stc = (Center(), Center())
        sh = half_cell_shifted(s, grid)
        wsh = Workspace(sh, grid; flux = (shape, stc))
        sort_by_cell!(sh, grid, wsh)
        variant!(a -> PMC._deposit_flux_2d!(Reference(), a..., sh, shape, stc, grid, wsh, sx, qz, nothing), "patch, nearest-node binning (5 × 5)";
                 fold = true)
        variant!(a -> deposit_patch!(a, sh, factored_stencil, shape, stc, grid, wsh, sx, qz), "patch, nearest-node binning, factored stencil"; fold = true)
        variant!(a -> deposit_patch!(a, sh, :split, shape, stc, grid, wsh, sx, qz), "patch, nearest-node binning, one component per launch";
                 fold = true)
    end
    for (S, frame) in ((1, CellFrame()), (1, MinFrame()), (2, MinFrame()), (4, MinFrame()), (8, MinFrame()), (16, MinFrame()))
        name = "tiles S = $S" * (frame isa CellFrame ? ", cell frame" : "")
        wanted(name) || continue
        tb = TileBuffers(BE, T, grid, shape, stg, S)
        variant!(a -> deposit_tiles!(a, tb, s, frame, shape, stg, grid, seg..., sx, qz), name)
        BE isa CPU && S == 4 &&
            variant!(a -> deposit_tiles!(a, tb, s, frame, shape, stg, grid, seg..., sx, qz; launch = PMC.item_launcher), name * ", work-groups of one")
    end
    for S in (5, 8, 16)
        variant!(a -> deposit_coloured!(a, s, MinFrame(), shape, stg, grid, seg..., sx, qz, S), "coloured tiles S = $S")
        BE isa CPU && S == 8 &&
            variant!(a -> deposit_coloured!(a, s, MinFrame(), shape, stg, grid, seg..., sx, qz, S; launch = PMC.item_launcher),
                     "coloured tiles S = $S, work-groups of one")
    end
    if T <: AbstractFloat && wanted("atomics")
        variant!(a -> PMC._deposit_flux_2d!(Fast(), a..., s, shape, stg, grid, ws, sx, qz, nothing), "engine atomics, sorted particles")
        variant!(a -> deposit_atomic!(a, s, MinFrame(), shape, stg, grid, sx, qz), "atomics 4 × 4, sorted particles")
        let r = shuffled(s)
            variant!(a -> PMC._deposit_flux_2d!(Fast(), a..., r, shape, stg, grid, ws, sx, qz, nothing), "engine atomics, shuffled particles")
            variant!(a -> deposit_atomic!(a, r, MinFrame(), shape, stg, grid, sx, qz), "atomics 4 × 4, shuffled particles")
        end
    end

    # the unsorted state: the sorts
    push_once!(st)
    U = copy_species(s)
    timed!(() -> sort_by_cell!(s, grid, ws), "engine sort_by_cell!"; setup = () -> restore!(s, U))
    timed!("engine sort: keys and sortperm! alone"; setup = () -> restore!(s, U)) do
        PMC.launcher(PMC.sort_key_nd_kernel!, BE, nothing)(ws.key, s.cell, s.status, Int64.(grid.n), Int64(prod(grid.n)), Int64(capacity(s));
                                                           ndrange = capacity(s))
        PMC.AK.sortperm!(ws.perm, ws.key; workspace = ws.ak)
    end
    restore!(s, U)
    sort_by_cell!(s, grid, ws)                                   # the engine's result, to compare with
    n1 = alive_count(s)
    same_as_engine(b) = all(((p, q),) -> Array(p)[1:n1] == Array(q)[1:n1], zip(species_arrays(b.dst), species_arrays(s))) &&
                        Array(b.dst.status) == Array(s.status) && Array(b.seg_start) == Array(ws.seg_start) &&
                        Array(b.seg_stop) == Array(ws.seg_stop) && alive_count(b.dst) == n1
    for (name, make, sort!) in (("incremental sort", IncrementalSort, incremental_sort!), ("incremental sort, scatter form", ScatterSort, scatter_sort!))
        wanted(name) || continue
        b = make(s, grid)
        sort!(b, U, grid, segV...)
        diff[name] = !same_as_engine(b)
        timed!(() -> sort!(b, U, grid, segV...), "$name, two buffers")
        work = copy_species(U)
        segW = map(copy, segV)
        timed!(() -> sort!(b, work, grid, segW...; copy_back = true), "$name, copied back";
               setup = () -> (restore!(work, U); foreach(copyto!, segW, segV)))
    end

    row = Dict{String,Any}("size" => size, "T" => label, "cells" => collect(grid.n), "ppc" => collect(SIZES[size].ppc), "steps" => SIZES[size].nsteps,
                           "alive" => alive, "capacity" => capacity(s), "ms" => time_, "difference" => diff)
    if T <: AbstractFloat && wanted("state of a Fast run")       # the state a Fast run leaves: its particles were never sorted
        sf = final_state(x, size, Fast())
        push_once!(sf)
        f = sf.species[1]
        row["alive_fast"] = count(==(PMC.ALIVE), Array(f.status))
        time_["engine atomics, state of a Fast run"] = ms(least(() -> PMC._deposit_flux_2d!(Fast(), J..., f, shape, stg, grid, ws, sx, qz, nothing), BE))
        time_["atomics 4 × 4, state of a Fast run"] = ms(least(() -> deposit_atomic!(J, f, MinFrame(), shape, stg, grid, sx, qz), BE))
    end
    return row
end

fmt(x) = x >= 100 ? string(round(Int, x)) : x >= 10 ? string(round(x; digits = 1)) : x >= 1 ? string(round(x; digits = 2)) : string(round(x; sigdigits = 2))

function table(rows)
    io = IOBuffer()
    for size in unique(r["size"] for r in rows)
        rs = filter(r -> r["size"] == size, rows)
        r1 = first(rs)
        println(io, "Size $size: $(r1["cells"][1]) × $(r1["cells"][2]) cells, $(prod(r1["ppc"])) particles per cell, $(r1["alive"]) alive. Milliseconds per call.\n")
        println(io, "| | ", join(("`$(r["T"])`" for r in rs), " | "), " |")
        println(io, "|---|", join(("---" for _ in rs), "|"), "|")
        names = unique(vcat((collect(keys(r["ms"])) for r in rs)...))
        for name in sort(names)
            println(io, "| ", name, " | ", join((haskey(r["ms"], name) ? fmt(r["ms"][name]) : "" for r in rs), " | "), " |")
        end
        println(io)
    end
    return String(take!(io))
end

function main()
    sizes = length(ARGS) >= 2 ? split(ARGS[2], ",") : ["S"]
    elements = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["f32", "f64", "d32", "d64"]
    gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
    extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
    dir = run_dir("bench2d-$BACKEND_NAME", @__FILE__; prefix = "deposition", work_package = "deposition study",
                  backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                  params = Dict{String,Any}("case" => "wake_problem(7, 100)", "sizes" => collect(String, sizes), "elements" => collect(String, elements),
                                            "only" => collect(String, ONLY)),
                  extra_inputs = [joinpath(@__DIR__, "..", "optim", "wake2d_case.jl"), joinpath(@__DIR__, "proto.jl"), joinpath(@__DIR__, "common.jl")],
                  extra)
    rows = Dict{String,Any}[]
    for size in sizes, e in elements
        label, x = ELEMENTS[e]
        push!(rows, measure(String(size), label, x))
        println(table(rows[end:end]))
        bad = filter(((k, v),) -> v isa Bool ? v : v > (occursin("32", label) ? 1e-3 : 1e-10), rows[end]["difference"])
        isempty(bad) || println("DIFFERENCES ABOVE ROUNDOFF: ", bad)
        open(io -> JSON.print(io, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => cpu_model(), "gpu" => gpu, "backend" => BACKEND_NAME), 2),
             joinpath(dir, "bench2d.json"), "w")
    end
    write(joinpath(dir, "tables.md"), table(rows))
    println("wrote ", relpath(dir, REPO))
    return dir
end

Base.invokelatest(main)
