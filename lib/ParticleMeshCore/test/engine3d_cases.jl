# The computations of the 3D engine tests (WP-042: T-870 to T-876), shared by engine3d.jl and by
# threads_worker.jl, which repeats the Reference-mode ones with 1 and 4 threads and three launch
# shapes. Needs setup.jl, helpers.jl, digest.jl, engine_cases.jl, engine2d_cases.jl and
# spectral_cases.jl.
#
# Test sizes (CPU budget): grids of 8 × 6 × 5 cells with a different cell size per axis, 1/4, 3/8
# and 1/2 in c/ω₀, and 2 × 1 × 3 particles per cell when loading, so that an exchange of two axes
# fails a test.

const N3 = (8, 6, 5)
"The cell sizes of the 3D test grids, in the base type of `T` (dyadic, so positions are exact)."
dx3(::Type{T}) where {T} = (basetype(T)(1 // 4), basetype(T)(3 // 8), basetype(T)(1 // 2))
grid3(::Type{T}, g; n = N3) where {T} = PeriodicGrid(n, dx3(T), g)
const PPC3 = (2, 1, 3)

"""
The stagger triples of the 3D tests: ρ's nodes, the cell centres, and the three components of a
Yee electric field (Center along its own axis) and of a Yee magnetic field (Node along its own).
"""
const STAGGER_TRIPLES = ((Node(), Node(), Node()), (Center(), Center(), Center()),
                         (Center(), Node(), Node()), (Node(), Center(), Node()), (Node(), Node(), Center()),
                         (Node(), Center(), Center()), (Center(), Node(), Center()), (Center(), Center(), Node()))

"A 3D field on `grid` with interior values `f(j1, j2, j3)` and filled guards."
function test_field_3d(::Type{T}, grid, f) where {T}
    a = field(backend, T, grid)
    g = guards(grid)
    for j3 in 1:ncells(grid)[3], j2 in 1:ncells(grid)[2], j1 in 1:ncells(grid)[1]
        a[g + j1, g + j2, g + j3] = f(j1, j2, j3)
    end
    return fill_guards!(a, grid)
end

"A field on `grid` (any N) whose every entry, guards included, is `seeded(T, f(i))` of its linear index `i`."
function indexed_field(::Type{T}, grid, f) where {T}
    a = field(backend, T, grid)
    foreach(i -> a[i] = seeded(T, f(i)), eachindex(a))
    return a
end

"""
    test_species_3d(T, grid, np; seed, extras)

`np` alive particles over every cell (the linear cell `7i mod n₁n₂n₃` of particle `i`, so the
slots are not in cell order), the first eight in the eight corner cells, with offsets from a
fixed LCG that include 0 and values just below 1, weights in [1/2, 3/2), `u₁` in [-1, 1), `u₂` a
cosine and the particle id in `u₃`. Five free slots at the end.
"""
function test_species_3d(::Type{T}, grid, np::Integer; seed::Integer = 1, q = -1, extras = NamedTuple()) where {T}
    n = ncells(grid)
    s = Species(backend, T, np + 5; q, m = 1, N = 3, extras)
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    corners = [(a, b, c) for c in (1, n[3]), b in (1, n[2]), a in (1, n[1])]
    for i in 1:np
        c = i <= 8 ? corners[i] : ParticleMeshCore.cartesian_cell(mod1(7i, prod(n)), n)
        for d in 1:3
            s.cell[d][i] = Int32(c[d])
            δ = i == 9 ? 0.0 : i == 10 ? Float64(prevfloat(one(basetype(T)))) : nextu()
            s.offset[d][i] = seeded(T, δ)
        end
        s.w[i] = seeded(T, 0.5 + nextu())
        s.u[1][i] = seeded(T, 2nextu() - 1)
        s.u[2][i] = seeded(T, cos(1.7i + seed))
        s.u[3][i] = T(i)                                # particle id
        s.status[i] = ParticleMeshCore.ALIVE
    end
    fill!(s.count, Int32(np))
    return s
end

"""
    moved_species_3d(T, grid, np; seed, extras, dead)

`test_species_3d` moved once through the engine's per-axis `advance` by displacements in
(-9/10, 9/10) cells per axis, so particles cross every seam; particle 8 starts in the last corner
cell at offsets 15/16 and moves by 1/8 on every axis, diagonally through the corner into cell
(1, 1, 1), and particle 1 makes the opposite move. `dead` slots are freed.
"""
function moved_species_3d(::Type{T}, grid, np; seed = 3, extras = NamedTuple(), dead = ()) where {T}
    s = test_species_3d(T, grid, np; seed, extras)
    n = Int32.(ncells(grid))
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    for d in 1:3
        s.offset[d][8] = seeded(T, 15 / 16)
        s.offset[d][1] = seeded(T, 1 / 16)
    end
    for i in 1:np
        Δ = i == 8 ? ntuple(_ -> seeded(T, 1 / 8), 3) : i == 1 ? ntuple(_ -> seeded(T, -1 / 8), 3) :
            ntuple(_ -> seeded(T, 0.9 * (2nextu() - 1)), 3)
        c, δ = advance(ntuple(d -> s.cell[d][i], 3), ntuple(d -> s.offset[d][i], 3), Δ, n)
        for d in 1:3
            s.cell[d][i] = c[d]
            s.offset[d][i] = δ[d]
        end
    end
    foreach(i -> s.status[i] = ParticleMeshCore.FREE, dead)
    return s
end

"Deposit `quantity` of `s` into a fresh 3D field in `mode` on `be`, sorting first when the mode needs it; guards not folded."
function deposited_3d(mode, s, shape, st, grid; quantity = ChargeWeight(), workgroupsize = nothing, be = CPU())
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    return on(be, deposit!, mode, field(backend, T, grid), s, shape, st, grid, ws; quantity, workgroupsize)
end

"The oblique cosine density of the 3D load: `p[1]·(1 + p[2]·cos(p[3]·x₁ + p[4]·x₂ + p[5]·x₃))`."
struct ObliqueDensity3 end
@inline (::ObliqueDensity3)(x, p) = p[1] * (one(p[1]) + p[2] * cos(p[3] * x[1] + p[4] * x[2] + p[5] * x[3]))

"The mode of the box that the 3D load's density has, per axis."
const LOAD_MODE_3D = (1, 2, -1)
"Parameters of the oblique density on `grid`: the mode `LOAD_MODE_3D` of the box, relative amplitude 1/10."
oblique_params_3d(::Type{T}, grid) where {T} =
    (seeded(T, 1), T(1 // 10), ntuple(d -> T(2π * LOAD_MODE_3D[d] / (ncells(grid)[d] * cellsize(grid)[d])), 3)...)

const LOAD_VDRIFT_3D = (1 // 10, 0 // 1, -1 // 10)
const LOAD_VTH_3D = (1 // 20, 1 // 30, 1 // 40)

"The 3D quiet load of the tests into a fresh species, written from slot `first`, on `be`; with `dist`, the distribution form."
function load_3d(::Type{T}, grid, ppc; seed = 4, stream = 2, p = oblique_params_3d(T, grid), workgroupsize = nothing, first = 1,
                 density_at = :particle, dist = nothing, extra_slots = 3, be = CPU()) where {T}
    s = Species(backend, T, first - 1 + prod(ncells(grid)) * prod(ppc) + extra_slots; q = -1, m = 1, N = 3)
    dist === nothing ||
        return on(be, load_quiet!, s, grid, ObliqueDensity3(), p, ppc, dist; seed, stream, first, density_at, workgroupsize)
    return on(be, load_quiet!, s, grid, ObliqueDensity3(), p, ppc; vdrift = T.(LOAD_VDRIFT_3D), vth = T.(LOAD_VTH_3D), seed, stream, first,
              density_at, workgroupsize)
end

const DEAD3 = (9, 40, 41, 150)

"""
    sorted_3d(T; workgroupsize) -> (s, ws, grid)

300 particles of `moved_species_3d` with four slots freed, the extras `old_slot::Int32` (the slot
before the sort) and `old_offset::T` (a copy of the axis-3 offset), sorted once.
"""
function sorted_3d(::Type{T}; workgroupsize = nothing, be = CPU()) where {T}
    grid = grid3(T, 2)
    s = moved_species_3d(T, grid, 300; extras = (old_slot = Int32, old_offset = T), dead = DEAD3)
    for i in 1:capacity(s)
        s.extras.old_slot[i] = Int32(i)
        s.extras.old_offset[i] = s.offset[3][i]
    end
    ws = Workspace(s, grid)
    on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    return s, ws, grid
end

"The interior source of the 3D solve at node `j`: a smooth part, a constant (a nonzero mean) and a high-frequency part."
solve_value(j1, j2, j3) = sin(0.37j1) + cos(1.1j2) / 5 + sin(0.8j3) / 3 + 3 / 10 + sin(0.9j1 * j2 + 0.4j3) / 2

function solve_input(::Type{T}, grid::PeriodicGrid{3}) where {T}
    s = fill!(field(backend, T, grid), T(NaN))
    g = guards(grid)
    for j3 in 1:ncells(grid)[3], j2 in 1:ncells(grid)[2], j1 in 1:ncells(grid)[1]
        s[g + j1, g + j2, g + j3] = varied(T, solve_value(j1, j2, j3), j1 + 7j2 + 31j3)
    end
    return s
end

"Grids of the 3D solve's digests: even and odd sizes per axis (the odd ones have no Nyquist mode)."
const SOLVE_GRIDS_3D = ((8, 6, 5), (5, 7, 4))

"The stagger triples of the subprocess digests: ρ's nodes and one Yee triple, which takes the other weight path on one axis."
const DIGEST_TRIPLES = ((Node(), Node(), Node()), (Center(), Node(), Node()))

"""
    engine3d_digests(be = CPU(); wgs = WG_SIZES) -> Vector{String}

`"kind|type|case|workgroupsize|digest"` of every 3D result whose bits must not depend on the
launch shape or the thread count: the sort with extras, the Reference deposition of a moved
species (every seam and the corner crossed) for both orders and two stagger triples, the gather,
fill and fold, the load in its keyword form and with a Maxwell–Jüttner distribution, and the
periodic solve with its gradient. On the backend `be` (the solve on the CPU), for the launch
shapes `wgs`.
"""
function engine3d_digests(be = CPU(); wgs = WG_SIZES)
    lines = String[]
    for T in TYPES, wg in wgs
        L = type_label(T)
        s, ws, grid = sorted_3d(T; workgroupsize = wg, be)
        push!(lines, "sort3d|$L|extras|$wg|$(fnv_digest(species_arrays(s)..., ws.perm, ws.seg_start, ws.seg_stop))")
        for shape in SHAPES, st in DIGEST_TRIPLES
            grid = grid3(T, guard_width(shape))
            m = moved_species_3d(T, grid, 300; dead = DEAD3)
            ρ = deposited_3d(Reference(), m, shape, st, grid; workgroupsize = wg, be)
            push!(lines, "deposit3d|$L|$(order(shape)),$(stagger_label(st))|$wg|$(fnv_digest(ρ))")
            f = test_field_3d(T, grid, (a, b, c) -> seeded(T, sin(0.7a) + cos(0.3b) / 2 + sin(0.5c) / 3))
            out = fill!(similar(m.w), zero(T))
            on(be, gather!, out, f, m, shape, st, grid; workgroupsize = wg)
            push!(lines, "gather3d|$L|$(order(shape)),$(stagger_label(st))|$wg|$(fnv_digest(out))")
        end
        grid = grid3(T, 2; n = (9, 4, 5))
        x = indexed_field(T, grid, i -> cos(1.3i))
        push!(lines, "fill3d|$L|9x4x5|$wg|$(fnv_digest(on(be, fill_guards!, copy(x), grid; workgroupsize = wg)))")
        push!(lines, "fold3d|$L|9x4x5|$wg|$(fnv_digest(on(be, fold_guards!, copy(x), grid; workgroupsize = wg)))")
        s = load_3d(T, grid3(T, 2), PPC3; workgroupsize = wg, be)
        push!(lines, "load3d|$L|ppc=2x1x3|$wg|$(fnv_digest(species_arrays(s)...))")
        B = basetype(T)
        s = load_3d(T, grid3(T, 2), PPC3; workgroupsize = wg, dist = MaxwellJuttner(B(1 // 20), B.((1 // 10, 0, -1 // 5))), be)
        push!(lines, "load3d|$L|juttner|$wg|$(fnv_digest(species_arrays(s)...))")
        for n in SOLVE_GRIDS_3D
            grid = grid3(T, 2; n)
            φ, G = solved(solve_input(T, grid), grid; workgroupsize = wg)
            push!(lines, "psolve3d|$L|n=$(join(n, 'x'))|$wg|$(fnv_digest(φ, G...))")
        end
    end
    return lines
end
