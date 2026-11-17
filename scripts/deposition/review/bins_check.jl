# The engine's Reference depositions and its sort by nearest-node bins (`NodeBins`, WP-041) against
# the same by cell, on a backend, on states the suite does not use:
#
#   julia +1.13 -t 4 --project=<repo>/scripts bins_check.jl <repo> [cpu|cuda]
#
# Synthetic states (scripts/deposition/common.jl) of Float32, Float64 and Dual{Float32,3}, both
# shapes: a fraction of the slots free, every alive particle moved by up to 0.99 cells per axis from
# an old position, a twelfth of the new offsets set to 0, the float before 1/2, 1/2 or the float
# before 1 (the edges of a bin), 5% of the particles killed before the sort (their slots, and the
# slots free from the start, keep their arrays' values).
#
# N = 2, the four stagger pairs of the deposit, grids 37 × 23 and 8 × 9 (2g cells on x) with 4
# guard cells, and the bins: of the deposit's shape and staggers; of the other stagger on both axes;
# of the other shape; of the other stagger on x alone (with order 2 on nodes that is how the module
# bins the species of a moving window). Per case:
#   in bins     every alive slot lies in the segment of its bin, the bin computed here on the host
#               from the stencil's definition (not with the engine's `particle_bin`), and the
#               segments hold the alive slots and no other
#   flux        `deposit_flux!` by bins against by cells, guards folded; discrete continuity; the
#               same bits on a second call and at launch shapes 64 and 256
#   ρ           `deposit!` by bins against a sum over the particles on the host, guards folded
# N = 3 (9 × 8 × 8 cells): `in bins` and `ρ` for bins of the deposit's staggers and of the other
# stagger on x alone.
# On a device, the fields by bins are also compared with the CPU's on the same state.
# Prints the largest value of each check over its bound and throws on a failure.
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
const WORST = Dict{String,Float64}()
const COUNTS = Dict{String,Int}()

function report(label, name, value, bound)
    ok = value <= bound
    ok || push!(FAILED, "$label: $name")
    COUNTS[name] = get(COUNTS, name, 0) + 1
    value isa Bool || (WORST[name] = max(get(WORST, name, 0.0), Float64(value) / Float64(bound)))
    ok || (println(rpad("$label: $name", 110), "FAILED  ", value); flush(stdout))
    return ok
end

with_value(x::AbstractFloat, v) = typeof(x)(v)
with_value(x::Dual{G}, v) where {G} = Dual{G}(oftype(ForwardDiff.value(x), v), ForwardDiff.partials(x))
val(x) = ForwardDiff.value(x)

"""
The state of a case: synthetic particles in cell order, a part of the new offsets on the edges of a
bin, the old position up to `dmax` cells away per axis (in the extras), a fraction dead.
"""
function new_state(be, ::Type{T}, grid::PMC.PeriodicGrid{N}; ppc, dmax = 0.99, kill = 0.05, plant = 1 / 12, seed = 7) where {T,N}
    V = base_type(T)
    s = synthetic_species(be, T, grid; ppc, fill = 0.8)
    rng = Xoshiro(seed)
    n = Int32.(grid.n)
    cell, offset, status = map(Array, s.cell), map(Array, s.offset), Array(s.status)
    old_cell, old_offset = map(copy, cell), map(copy, offset)
    special = (zero(V), prevfloat(V(1 // 2)), V(1 // 2), prevfloat(one(V)))
    for i in eachindex(status)
        status[i] == PMC.ALIVE || continue
        for d in 1:N
            rand(rng) < plant && (offset[d][i] = with_value(offset[d][i], special[rand(rng, 1:4)]))
            Δ = T(V(dmax) * (2 * rand(rng, V) - 1))
            old_cell[d][i], old_offset[d][i] = PMC.advance(cell[d][i], offset[d][i], Δ, n[d])
        end
        rand(rng) < kill && (status[i] = PMC.FREE)
    end
    kc, ko = old_keys(N)
    foreach((k, b) -> copyto!(s.extras[k], b), kc, old_cell)
    foreach((k, b) -> copyto!(s.extras[k], b), ko, old_offset)
    foreach(copyto!, s.offset, offset)
    copyto!(s.status, status)
    return s
end

# The bin of a particle from the stencil's definition: with order 2 on nodes or order 1 at centres
# the stencil starts one node later in the upper half of the cell, with the two other pairs it does
# not depend on the offset. Within one unit of rounding of 1/2 either half may claim the particle:
# the engine's order 2 on nodes takes `floor(δ + 1/2)`, and the float before 1/2 plus 1/2 rounds to 1.
moves(shape, st) = (PMC.order(shape) == 2 && st isa PMC.Node) || (PMC.order(shape) == 1 && st isa PMC.Center)
function host_bins(bins, c, δ, n, d)
    (bins isa PMC.NodeBins && moves(bins.shape, bins.stagger[d])) || return (c,)
    x = val(δ)
    up = mod1(c + Int32(1), n)
    return abs(x - 0.5) <= eps(typeof(x)) ? (c, up) : x > 0.5 ? (up,) : (c,)
end

"Whether every alive slot lies in the segment of its bin, the segments hold alive slots only, and all of them."
function in_bins(s::PMC.Species{N}, grid, ws) where {N}
    n = Int32.(grid.n)
    cell, offset, status = map(Array, s.cell), map(Array, s.offset), Array(s.status)
    a, b = Array(ws.seg_start), Array(ws.seg_stop)
    count = Int(Array(s.count)[1])
    alive = findall(==(PMC.ALIVE), status)
    alive == collect(1:count) || return false
    sum(max.(b .- a .+ 1, 0)) == count || return false
    for i in alive
        cands = ntuple(d -> host_bins(ws.bins, cell[d][i], offset[d][i], n[d], d), N)
        any(Iterators.product(cands...)) do bin
            L = PMC.linear_cell(map(Int32, bin), n)
            a[L] <= i <= b[L]
        end || return false
    end
    return true
end

other(st::PMC.Node) = PMC.Center()
other(st::PMC.Center) = PMC.Node()
other(shape::PMC.Shape) = shape === PMC.Linear ? PMC.Quadratic : PMC.Linear
binned(shape, st::Tuple) = (("bins of the deposit", PMC.NodeBins(shape, st)), ("bins of the other stagger", PMC.NodeBins(shape, map(other, st))),
                           ("bins of the other shape", PMC.NodeBins(other(shape), st)),
                           ("bins of the other stagger on x", PMC.NodeBins(shape, (other(st[1]), Base.tail(st)...))))

"The state (the same particles for every bins) sorted by the workspace's bins."
function sorted_state(be, ::Type{T}, grid, shape, st, bins; ppc, flux = true) where {T}
    s = new_state(be, T, grid; ppc)
    ws = flux ? PMC.Workspace(s, grid; flux = (shape, st), bins) : PMC.Workspace(s, grid; bins)
    PMC.sort_by_cell!(s, grid, ws)
    return s, ws
end

function flux(be, ::Type{T}, grid, s, ws, shape, st, dt; workgroupsize = nothing) where {T}
    J = zero_fields(be, T, grid)
    PMC.deposit_flux!(PMC.Reference(), J..., s, shape, st, grid, ws; dt, quantity_z = MomentumZ(), workgroupsize)
    return map(Array, J)
end
function rho(be, ::Type{T}, grid, s, ws, shape, st; workgroupsize = nothing) where {T}
    ρ = KernelAbstractions.zeros(be, T, (grid.n .+ 2grid.g)...)
    PMC.deposit!(PMC.Reference(), ρ, s, shape, st, grid, ws; workgroupsize)
    return Array(ρ)
end
type_label(::Type{T}) where {T} = T <: Dual ? "Dual{$(base_type(T)),3}" : string(T)
stagger_label(st) = join(map(x -> nameof(typeof(x)), st), "/")

function check2d(::Type{T}, shape, st, n, ppc, tol) where {T}
    V = base_type(T)
    dt = T(V(0.1))
    grid = PMC.PeriodicGrid(n, ntuple(d -> T(V(0.3) * d), 2), 4)
    cells = nothing
    for (kind, bins) in (("cells", PMC.CellBins()), binned(shape, st)...)
        label = "$(type_label(T)) order $(PMC.order(shape)) $(stagger_label(st)) $(join(n, "x")), $kind"
        s, ws = sorted_state(BE, T, grid, shape, st, bins; ppc)
        report(label, "a particle outside its bin's segment", !in_bins(s, grid, ws), false)
        J = flux(BE, T, grid, s, ws, shape, st, dt)
        ρ = rho(BE, T, grid, s, ws, shape, st)
        Jf = map(a -> fold_host!(copy(a), grid), J)
        report(label, "flux: continuity", continuity_residual(J, s, shape, st, grid, dt), tol)
        ρh = fold_host!(rho_host(s, shape, st, grid), grid)
        report(label, "ρ against the sum on the host", maxmag(fold_host!(copy(ρ), grid) .- ρh) / maxmag(ρh), tol)
        if kind == "cells"
            cells = Jf
            continue
        end
        report(label, "flux against the cells'", field_difference(Jf, cells), tol)
        report(label, "flux: a second call gives other bits", !all(map(same, J, flux(BE, T, grid, s, ws, shape, st, dt))), false)
        for wg in (64, 256)
            report(label, "flux: launch shape $wg gives other bits", !all(map(same, J, flux(BE, T, grid, s, ws, shape, st, dt; workgroupsize = wg))), false)
            report(label, "ρ: launch shape $wg gives other bits", !same(ρ, rho(BE, T, grid, s, ws, shape, st; workgroupsize = wg)), false)
        end
        if !(BE isa CPU)
            sc, wsc = sorted_state(CPU(), T, grid, shape, st, bins; ppc)
            report(label, "the state differs from the CPU's", !all(map(same, species_arrays(s), species_arrays(sc))), false)
            report(label, "flux against the CPU's", field_difference(J, flux(CPU(), T, grid, sc, wsc, shape, st, dt)), tol)
            report(label, "ρ against the CPU's", field_difference((ρ,), (rho(CPU(), T, grid, sc, wsc, shape, st),)), tol)
        end
    end
    return nothing
end

function check3d(::Type{T}, shape, st, n, ppc, tol) where {T}
    V = base_type(T)
    grid = PMC.PeriodicGrid(n, ntuple(d -> T(V(0.3) * d), 3), 4)
    kinds = binned(shape, st)
    for (kind, bins) in (("cells", PMC.CellBins()), kinds[1], kinds[4])
        label = "$(type_label(T)) order $(PMC.order(shape)) $(stagger_label(st)) $(join(n, "x")), $kind"
        s, ws = sorted_state(BE, T, grid, shape, st, bins; ppc, flux = false)
        report(label, "a particle outside its bin's segment", !in_bins(s, grid, ws), false)
        ρ = rho(BE, T, grid, s, ws, shape, st)
        ρh = fold_host!(rho_host(s, shape, st, grid), grid)
        report(label, "ρ against the sum on the host", maxmag(fold_host!(copy(ρ), grid) .- ρh) / maxmag(ρh), tol)
        report(label, "ρ: launch shape 64 gives other bits", !same(ρ, rho(BE, T, grid, s, ws, shape, st; workgroupsize = 64)), false)
        if !(BE isa CPU)
            sc, wsc = sorted_state(CPU(), T, grid, shape, st, bins; ppc, flux = false)
            report(label, "ρ against the CPU's", field_difference((ρ,), (rho(CPU(), T, grid, sc, wsc, shape, st),)), tol)
        end
    end
    return nothing
end

function main()
    cases = 0
    N, C = PMC.Node(), PMC.Center()
    for (T, tol) in ((Float32, 2f-4), (Float64, 1e-12), (Dual{Nothing,Float32,3}, 2f-4)), shape in (PMC.Linear, PMC.Quadratic)
        for st in ((N, N), (C, C), (C, N), (N, C)), (n, ppc) in (((37, 23), 5), ((8, 9), 7))
            check2d(T, shape, st, n, ppc, tol)
            cases += 5
        end
        for st in ((N, N, N), (C, N, C))
            check3d(T, shape, st, (9, 8, 8), 6, tol)
            cases += 3
        end
        println("done: ", type_label(T), " order ", PMC.order(shape))
        flush(stdout)
    end
    println("backend ", NAME, ", ", cases, " cases. Per check: how often it ran, and its largest value over its bound.")
    for k in sort(collect(keys(COUNTS)))
        println("  ", rpad(k, 46), lpad(COUNTS[k], 5), "   ", haskey(WORST, k) ? round(WORST[k]; sigdigits = 3) : "never")
    end
    isempty(FAILED) || error("$(length(FAILED)) checks failed:\n" * join(FAILED, "\n"))
    println("all checks passed")
end
Base.invokelatest(main)
