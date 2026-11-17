# The prototypes against the engine and against discrete continuity, on the CPU backend:
#
#   julia +1.13 -t 4 --project=scripts scripts/deposition/check.jl
#
# N = 2: the incremental sort gives the arrays, segments and count of `sort_by_cell!`; every
# deposition variant agrees with the engine's Reference `deposit_flux!` to roundoff and holds
# continuity. N = 3 (no engine deposition to compare with): the incremental sort against
# `sort_by_cell!`, continuity of every variant, and their mutual agreement. The scatter form of
# the incremental sort is held to the same result at both N. The tile variants
# give the same bits for every work-group size. Prints one line per check and throws on a failure.
include(joinpath(@__DIR__, "common.jl"))

const BE = CPU()
const FAILED = String[]

function report(name, value, bound)
    ok = value <= bound
    println(rpad(name, 74), ok ? "ok    " : "FAILED", "  ", value isa Bool ? "" : string(Float32(value)))
    ok || push!(FAILED, name)
    return ok
end

same(a, b) = Array(a) == Array(b)

function check_sort(s, grid, ws, label)
    b = IncrementalSort(s, grid)
    e = copy_species(s)
    we = PMC.Workspace(e, grid)
    PMC.sort_by_cell!(e, grid, we)
    incremental_sort!(b, s, grid, ws.seg_start, ws.seg_stop)
    count = PMC.alive_count(e)
    alive = 1:count
    arrays = all(((x, y),) -> Array(x)[alive] == Array(y)[alive], zip(species_arrays(b.dst), species_arrays(e)))
    ok = arrays && same(b.dst.status, e.status) && same(b.seg_start, we.seg_start) && same(b.seg_stop, we.seg_stop) &&
         PMC.alive_count(b.dst) == count
    report("$label incremental sort == sort_by_cell! ($count alive)", !ok, false)
    b2 = ScatterSort(s, grid)
    scatter_sort!(b2, s, grid, ws.seg_start, ws.seg_stop)
    arrays = all(((x, y),) -> Array(x)[alive] == Array(y)[alive], zip(species_arrays(b2.dst), species_arrays(e)))
    ok = arrays && same(b2.dst.status, e.status) && same(b2.seg_start, we.seg_start) && same(b2.seg_stop, we.seg_stop) &&
         PMC.alive_count(b2.dst) == count
    report("$label incremental sort, scatter form == sort_by_cell!", !ok, false)
    incremental_sort!(b, s, grid, ws.seg_start, ws.seg_stop; copy_back = true)      # leaves `s` sorted, its segments in `ws`
    return nothing
end

function check(::Type{T}, N; n, ppc, tol) where {T}
    label = "N = $N $(T <: Dual ? "Dual{$(base_type(T)),3}" : string(T))"
    shape = PMC.Quadratic
    st = ntuple(_ -> PMC.Node(), N)
    V = base_type(T)
    grid = PMC.PeriodicGrid(n, ntuple(d -> T(V(0.3) * d), N), 3)
    dt = T(V(0.1))
    sx = map(h -> h / dt, grid.dx)
    qz = N == 2 ? MomentumZ() : nothing
    s = synthetic_species(BE, T, grid; ppc, fill = 0.8)
    ws = N == 2 ? PMC.Workspace(s, grid; flux = (shape, st)) : PMC.Workspace(s, grid)
    PMC.sort_by_cell!(s, grid, ws)
    move!(s, grid; dmax = 0.99, kill = 0.05)
    check_sort(s, grid, ws, label)
    J() = zero_fields(BE, T, grid)
    seg = (ws.seg_start, ws.seg_stop)
    variants = Pair{String,Any}[]
    if T <: AbstractFloat
        push!(variants, "atomics, cell frame" => deposit_atomic!(J(), s, CellFrame(), shape, st, grid, sx, qz))
        push!(variants, "atomics, minimal frame" => deposit_atomic!(J(), s, MinFrame(), shape, st, grid, sx, qz))
    end
    for (S, frame) in ((1, CellFrame()), (1, MinFrame()), (2, MinFrame()), (4, MinFrame()), (8, MinFrame()))
        tb = TileBuffers(BE, T, grid, shape, st, S)
        a = deposit_tiles!(J(), tb, s, frame, shape, st, grid, seg..., sx, qz)
        push!(variants, "tiles S = $S, $(nameof(typeof(frame)))" => a)
        bits = all((1, 7)) do wg
            b = deposit_tiles!(J(), tb, s, frame, shape, st, grid, seg..., sx, qz; workgroupsize = wg)
            all(map(same, a, b))
        end
        report("$label tiles S = $S $(nameof(typeof(frame))): same bits for work-groups 1, 7, default", !bits, false)
    end
    for S in (5, 8)
        a = deposit_coloured!(J(), s, MinFrame(), shape, st, grid, seg..., sx, qz, S)
        b = deposit_coloured!(J(), s, MinFrame(), shape, st, grid, seg..., sx, qz, S; workgroupsize = 3)
        push!(variants, "coloured tiles S = $S" => a)
        report("$label coloured tiles S = $S: same bits for work-groups 3, default", !all(map(same, a, b)), false)
    end
    if N == 2
        ref = J()
        PMC.deposit_flux!(PMC.Reference(), ref..., s, shape, st, grid, ws; dt, quantity_z = qz)
        report("$label engine Reference: continuity", continuity_residual(ref, s, shape, st, grid, dt), tol)
        push!(variants, "patch, engine stencil" => deposit_patch!(J(), s, engine_stencil, shape, st, grid, ws, sx, qz))
        push!(variants, "patch, factored stencil" => deposit_patch!(J(), s, factored_stencil, shape, st, grid, ws, sx, qz))
        push!(variants, "patch, one component per launch" => deposit_patch!(J(), s, :split, shape, st, grid, ws, sx, qz))
        report("$label patch with the engine's stencil == engine (bits)", !all(map(same, variants[end - 2][2], ref)), false)
    else
        ref = variants[1][2]
        push!(variants, "patch in registers, cell binning" => deposit_patch3!(J(), patch3_buffer(BE, T, grid, shape, st), s, shape, st, grid, seg..., sx))
        stc = ntuple(_ -> PMC.Center(), 3)                       # nearest-node binning: the shifted species at `Center()`
        sh = half_cell_shifted(s, grid)
        wsh = PMC.Workspace(sh, grid)
        PMC.sort_by_cell!(sh, grid, wsh)
        a = deposit_patch3!(J(), patch3_buffer(BE, T, grid, shape, stc), sh, shape, stc, grid, wsh.seg_start, wsh.seg_stop, sx)
        report("$label patch in registers, nearest-node binning: continuity", continuity_residual(a, s, shape, st, grid, dt), tol)
        report("$label patch in registers, nearest-node binning: against $(variants[1][1]), folded",
               field_difference(folded_host(a, grid), folded_host(ref, grid)), tol)
    end
    for (name, a) in variants
        report("$label $name: continuity", continuity_residual(a, s, shape, st, grid, dt), tol)
        report("$label $name: against $(N == 2 ? "the engine" : variants[1][1])", field_difference(a, ref), tol)
    end
    return nothing
end

for (T, tol) in ((Float32, 2f-4), (Float64, 1e-12), (Dual{Nothing,Float32,3}, 2f-4))
    check(T, 2; n = (37, 23), ppc = 5, tol)
    check(T, 3; n = (13, 11, 9), ppc = 3, tol)
end
isempty(FAILED) || error("$(length(FAILED)) checks failed:\n" * join(FAILED, "\n"))
println("all checks passed")
