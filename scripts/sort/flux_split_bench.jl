# How many components of the 2D Reference flux one launch of its patch kernel should sum (WP-041):
# `deposit_flux!(Reference(), ...; components = c)` for c = 3 (all of them, one launch), 2 (Jx and
# Jy, then Jz) and 1 (one launch per component), every choice the same bits, on a sorted state,
# for the bins the electromagnetic module uses at order 2: the periodic box's nearest-node bins
# (`NodeBins` at `(Node(), Node())`, a 5 × 5 frame), the moving window's with its x axis open
# (`NodeBins(...; open = (true, false))`, the `n + 1` bins on x of the deposition study, 5 × 5), and
# the window's before them, whose x axis binned by cell (`NodeBins` at `(Center(), Node())`, a 6 × 5
# frame).
#
#   julia +1.13 -t 4 --project=scripts scripts/sort/flux_split_bench.jl [cpu|cuda] [S,M,L] [Float32,Float64,Dual32x2,Dual32x3,Dual64x2]
#
# The sizes are those of the deposition study: S 224 × 64 cells at 4 particles per cell, M 448 × 128
# at 16, L 448 × 512 at 16. The state: particles at random offsets, sorted by the bins, each moved
# by up to half a cell per axis from its stored old position. A time is the median of `reps` calls
# of the deposit (patch kernels and reduction) after two warm-up calls, the device synchronized
# around each; `default` is the call without `components` (the engine's choice for the backend and
# type). Prints a table and writes `scripts/runs/sort-flux-split-<backend>-<stamp>/` with the
# numbers (`split.json`) and the run's manifest (F3). One run on one machine; not reviewed.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ForwardDiff: Dual
using Random

const PMC = ParticleMesh.ParticleMeshCore
using .PMC: Workspace, sort_by_cell!, deposit_flux!, Species, NodeBins, Quadratic, Node, Center, PeriodicGrid, flux_extras, capacity,
            ALIVE, Reference, field

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
    gpu_name() = CUDACore.name(CUDACore.device())
elseif BACKEND_NAME == "cpu"
    const BE = CPU()
    gpu_name() = "none"
else
    error("flux_split_bench.jl: the first argument is cpu or cuda, got $BACKEND_NAME")
end

const SIZES = Dict("S" => ((224, 64), 4), "M" => ((448, 128), 16), "L" => ((448, 512), 16))
const TYPES = Dict("Float32" => Float32, "Float64" => Float64, "Dual32x2" => Dual{Nothing,Float32,2},
                   "Dual32x3" => Dual{Nothing,Float32,3}, "Dual64x2" => Dual{Nothing,Float64,2})
const SIZE_LIST = length(ARGS) >= 2 ? split(ARGS[2], ",") : ["S", "M", "L"]
const TYPE_LIST = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["Float32", "Float64", "Dual32x2", "Dual32x3", "Dual64x2"]
const BINS = (("periodic 5x5", NodeBins(Quadratic, (Node(), Node()))), ("window open 5x5", NodeBins(Quadratic, (Node(), Node()); open = (true, false))),
               ("window 6x5", NodeBins(Quadratic, (Center(), Node()))))
const RHO = (Node(), Node())

host_value(::Type{T}, x) where {T<:AbstractFloat} = T(x)
host_value(::Type{Dual{Nothing,F,K}}, x) where {F,K} = Dual{Nothing}(F(x), ntuple(k -> F(x / (k + 1)), Val(K))...)

"q·w·u₃, the module's quantity for Jz."
struct QZ end
@inline (::QZ)(s, i) = s.q * s.w[i] * s.u[3][i]

# `np` particles at random cells and offsets, each with an old position up to half a cell away per axis.
function fill_state!(s, n, rng)
    T = eltype(s)
    cap = capacity(s)
    c = [zeros(Int32, cap) for _ in 1:2]; δ = [zeros(T, cap) for _ in 1:2]
    c0 = [zeros(Int32, cap) for _ in 1:2]; δ0 = [zeros(T, cap) for _ in 1:2]
    for i in 1:cap, d in 1:2
        x = rand(rng) * n[d]
        x0 = mod(x + (rand(rng) - 1 / 2), n[d])
        c[d][i] = Int32(min(floor(Int, x), n[d] - 1) + 1); δ[d][i] = host_value(T, x - (c[d][i] - 1))
        c0[d][i] = Int32(min(floor(Int, x0), n[d] - 1) + 1); δ0[d][i] = host_value(T, x0 - (c0[d][i] - 1))
    end
    foreach(d -> (copyto!(s.cell[d], c[d]); copyto!(s.offset[d], δ[d])), 1:2)
    copyto!(s.extras.old_cell_1, c0[1]); copyto!(s.extras.old_cell_2, c0[2])
    copyto!(s.extras.old_offset_1, δ0[1]); copyto!(s.extras.old_offset_2, δ0[2])
    for d in 1:3
        copyto!(s.u[d], [host_value(T, rand(rng) - 1 / 2) for _ in 1:cap])
    end
    copyto!(s.w, fill(host_value(T, 0.5), cap))
    copyto!(s.status, fill(ALIVE, cap))
    copyto!(s.count, Int32[cap])
    return s
end

function median_time(f; reps)
    f(); f()
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

bits(J) = reinterpret(UInt8, vec(Array(J)))

function bench(sizename, tname; reps)
    T = TYPES[tname]
    n, ppc = SIZES[sizename]
    cap = prod(n) * ppc
    grid = PeriodicGrid(n, (T(0.1), T(0.1)), 3)
    s = Species(BE, T, cap; q = -1, m = 1, N = 2, extras = flux_extras(T, 2))
    out = Dict{String,Any}("particles" => cap)
    for (label, bins) in BINS
        fill_state!(s, n, MersenneTwister(1))
        ws = Workspace(s, grid; flux = (Quadratic, RHO), bins)
        sort_by_cell!(s, grid, ws)
        J = (field(BE, T, grid), field(BE, T, grid), field(BE, T, grid))
        dep(c) = (foreach(x -> fill!(x, zero(T)), J);
                  deposit_flux!(Reference(), J..., s, Quadratic, RHO, grid, ws; dt = T(1 // 4), quantity_z = QZ(), components = c))
        ref = (dep(3); map(bits, J))
        r = Dict{String,Any}()
        for c in (nothing, 3, 2, 1)
            r[c === nothing ? "default" : string(c)] = median_time(() -> dep(c); reps)
            r["same_bits_" * (c === nothing ? "default" : string(c))] = map(bits, J) == ref
        end
        r["default_launches"] = length(PMC.flux_components_2d(nothing))
        out[label] = r
    end
    return out
end

function main()
    reps = BACKEND_NAME == "cuda" ? 30 : 5
    results = Dict{String,Any}()
    println("| size | type | bins | 3 per launch | 2 | 1 | default (launches) | same bits |")
    println("|---|---|---|---|---|---|---|---|")
    for sz in SIZE_LIST, tn in TYPE_LIST
        r = bench(sz, tn; reps)
        results["$sz/$tn"] = r
        for (label, _) in BINS
            x = r[label]
            same = all(x["same_bits_$k"] for k in ("default", "3", "2", "1"))
            println("| $sz | $tn | $label | ", join((round(x[k]; sigdigits = 3) for k in ("3", "2", "1")), " | "),
                    " | $(round(x["default"]; sigdigits = 3)) ($(x["default_launches"])) | $same |")
        end
        flush(stdout)
    end
    dir = run_dir("flux-split-$BACKEND_NAME", @__FILE__; prefix = "sort", work_package = "WP-041",
                  params = Dict{String,Any}("sizes" => SIZE_LIST, "types" => TYPE_LIST, "reps" => reps),
                  backend = string(BE), extra = Dict{String,Any}("gpu" => gpu_name()))
    open(io -> JSON.print(io, Dict("unit" => "ms", "results" => results), 2), joinpath(dir, "split.json"), "w")
    println("wrote ", dir)
end

main()
