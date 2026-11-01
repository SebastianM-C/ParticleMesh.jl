# The computations of the 2D engine tests (WP-018: T-261 to T-269), shared by engine2d.jl and by
# threads_worker.jl, which repeats the Reference-mode ones with 1 and 4 threads and three launch
# shapes. Needs setup.jl, helpers.jl, digest.jl and engine_cases.jl.
#
# Test sizes (CPU budget, WP-018): non-square grids of 16 × 12 cells (and 32 × 24 for T-268) with
# different cell sizes per axis, 1/4 and 3/8 in c/ω₀, and 2 × 3 particles per cell when loading
# (4 × 6 for T-268), so that a transposition of the axes fails a test.

const N2 = (16, 12)
"The cell sizes of the 2D test grids, in the base type of `T` (dyadic, so positions are exact)."
dx2(::Type{T}) where {T} = (basetype(T)(1 // 4), basetype(T)(3 // 8))
grid2(::Type{T}, g; n = N2) where {T} = PeriodicGrid(n, dx2(T), g)
const PPC2 = (2, 3)

"Every stagger pair, mixed pairs included."
const STAGGER_PAIRS = ((Node(), Node()), (Center(), Center()), (Center(), Node()), (Node(), Center()))
stagger_label(st) = join((nameof(typeof(x)) for x in st), ",")

"A 2D field on `grid` with interior values `f(j1, j2)` and filled guards."
function test_field_2d(::Type{T}, grid, f) where {T}
    a = field(backend, T, grid)
    g = guards(grid)
    for j2 in 1:ncells(grid)[2], j1 in 1:ncells(grid)[1]
        a[g + j1, g + j2] = f(j1, j2)
    end
    return fill_guards!(a, grid)
end

"""
    test_species_2d(T, grid, np; seed, extras, corner = true)

`np` alive particles over every cell (the linear cell `7i mod n₁n₂` of particle `i`, so the slots
are not in cell order), with offsets from a fixed LCG that include 0 and values just below 1,
weights in [1/2, 3/2), `u₁` and `u₂` in [-1, 1) and the particle id in `u₃`. With `corner`, the
first four sit in the four corner cells. Five free slots at the end.
"""
function test_species_2d(::Type{T}, grid, np::Integer; seed::Integer = 1, q = -1, extras = NamedTuple(), corner = true) where {T}
    n = ncells(grid)
    s = Species(backend, T, np + 5; q, m = 1, N = 2, extras)
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    corners = ((n[1], n[2]), (1, 1), (n[1], 1), (1, n[2]))
    for i in 1:np
        c = corner && i <= 4 ? corners[i] : ParticleMeshCore.cartesian_cell(mod1(7i, prod(n)), n)
        s.cell[1][i], s.cell[2][i] = Int32(c[1]), Int32(c[2])
        for d in 1:2
            δ = i == 5 ? 0.0 : i == 6 ? Float64(prevfloat(one(basetype(T)))) : nextu()
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
    moved_species_2d(T, grid, np; seed, extras, dead)

`test_species_2d` moved once through the engine's per-axis `advance` by displacements in
(-9/10, 9/10) cells per axis, so particles cross both seams; particle 1 starts in the last
corner cell at offsets (15/16, 15/16) and moves by (1/8, 1/8), diagonally through the corner into
cell (1, 1), and particle 2 makes the opposite move. `dead` slots are freed.
"""
function moved_species_2d(::Type{T}, grid, np; seed = 3, extras = NamedTuple(), dead = ()) where {T}
    s = test_species_2d(T, grid, np; seed, extras)
    n = Int32.(ncells(grid))
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    s.offset[1][1], s.offset[2][1] = seeded(T, 15 / 16), seeded(T, 15 / 16)
    s.offset[1][2], s.offset[2][2] = seeded(T, 1 / 16), seeded(T, 1 / 16)
    for i in 1:np
        Δ = i == 1 ? (seeded(T, 1 / 8), seeded(T, 1 / 8)) : i == 2 ? (seeded(T, -1 / 8), seeded(T, -1 / 8)) :
            (seeded(T, 0.9 * (2nextu() - 1)), seeded(T, 0.9 * (2nextu() - 1)))
        c, δ = advance((s.cell[1][i], s.cell[2][i]), (s.offset[1][i], s.offset[2][i]), Δ, n)
        s.cell[1][i], s.cell[2][i] = c
        s.offset[1][i], s.offset[2][i] = δ
    end
    foreach(i -> s.status[i] = ParticleMeshCore.FREE, dead)
    return s
end

"Deposit `quantity` of `s` into a fresh field in `mode` on `be`, sorting first when the mode needs it; guards not folded."
function deposited_2d(mode, s, shape, st, grid; quantity = ChargeWeight(), workgroupsize = nothing, be = CPU())
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    return on(be, deposit!, mode, field(backend, T, grid), s, shape, st, grid, ws; quantity, workgroupsize)
end

"q·w·u_y, the module-supplied quantity of T-265 (2D species keep the id in u_z)."
struct ChargeUy2 end
@inline (::ChargeUy2)(s, i) = s.q * s.w[i] * s.u[2][i]
struct AbsChargeUy2 end
@inline (::AbsChargeUy2)(s, i) = abs(s.q * s.w[i] * s.u[2][i])

"The oblique cosine density of T-267: `p[1]·(1 + p[2]·cos(p[3]·x₁ + p[4]·x₂))`."
struct ObliqueDensity end
@inline (::ObliqueDensity)(x, p) = p[1] * (one(p[1]) + p[2] * cos(p[3] * x[1] + p[4] * x[2]))
"A density constant along axis 2 (T-269): `p[1]·(1 + p[2]·cos(p[3]·x₁))`, of an `NTuple{2}` or a scalar position."
struct XDensity end
@inline (::XDensity)(x::Tuple, p) = p[1] * (one(p[1]) + p[2] * cos(p[3] * x[1]))
@inline (::XDensity)(x, p) = p[1] * (one(p[1]) + p[2] * cos(p[3] * x))

"Parameters of the oblique density on `grid`: mode (1, 2) of the box, relative amplitude 1/10."
oblique_params(::Type{T}, grid) where {T} =
    (seeded(T, 1), T(1 // 10), T(2π / (ncells(grid)[1] * cellsize(grid)[1])), T(2 * 2π / (ncells(grid)[2] * cellsize(grid)[2])))

load_2d(::Type{T}, grid, ppc; seed = 4, stream = 2, p = oblique_params(T, grid), workgroupsize = nothing, extras = NamedTuple(),
        be = CPU()) where {T} =
    on(be, load_quiet!, Species(backend, T, prod(ncells(grid)) * prod(ppc) + 3; q = -1, m = 1, N = 2, extras), grid, ObliqueDensity(), p, ppc;
                vdrift = (T(1 // 10), zero(T), T(-1 // 10)), vth = (T(1 // 20), T(1 // 30), T(1 // 40)), seed, stream, workgroupsize)

const DEAD2 = (9, 40, 41, 150)

"""
    sorted_2d(T; workgroupsize) -> (s, ws, grid)

T-266's species: 300 particles of `moved_species_2d` with four slots freed, the extras
`old_slot::Int32` (the slot before the sort) and `old_offset::T` (a copy of the axis-1 offset),
sorted once.
"""
function sorted_2d(::Type{T}; workgroupsize = nothing, be = CPU()) where {T}
    grid = grid2(T, 2)
    s = moved_species_2d(T, grid, 300; extras = (old_slot = Int32, old_offset = T), dead = DEAD2)
    for i in 1:capacity(s)
        s.extras.old_slot[i] = Int32(i)
        s.extras.old_offset[i] = s.offset[1][i]
    end
    ws = Workspace(s, grid)
    on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    return s, ws, grid
end

"""
    engine2d_digests(be = CPU(); wgs = WG_SIZES) -> Vector{String}

`"kind|type|case|workgroupsize|digest"` of every 2D result whose bits must not depend on the
launch shape or the thread count: the sort with extras, the Reference deposition of a moved
species (both seams and the corner crossed) for both orders, two stagger pairs per order (the
natural pair and the mixed pair (Center, Node), which takes the `floor` path on one axis) and
two quantities, the gather at the natural pair, fill and fold, and the load. Every stagger pair
is covered in process by T-262 to T-265; this set is kept small because each subprocess
compiles it anew. On the backend `be`, for the launch shapes `wgs`.
"""

"The stagger pairs of the subprocess digests: the natural pair of the order and the mixed pair."
digest_pairs(::Shape{1}) = ((Node(), Node()), (Center(), Node()))
digest_pairs(::Shape{2}) = ((Center(), Center()), (Center(), Node()))
function engine2d_digests(be = CPU(); wgs = WG_SIZES)
    lines = String[]
    for T in TYPES, wg in wgs
        L = type_label(T)
        s, ws, grid = sorted_2d(T; workgroupsize = wg, be)
        push!(lines, "sort2d|$L|extras|$wg|$(fnv_digest(species_arrays(s)..., ws.perm, ws.seg_start, ws.seg_stop))")
        for shape in SHAPES, st in digest_pairs(shape)
            grid = grid2(T, guard_width(shape))
            for (ql, quantity) in (("qw", ChargeWeight()), ("qwuy", ChargeUy2()))
                m = moved_species_2d(T, grid, 300; dead = DEAD2)
                ρ = deposited_2d(Reference(), m, shape, st, grid; quantity, workgroupsize = wg, be)
                push!(lines, "deposit2d|$L|$(order(shape)),$(stagger_label(st)),$ql|$wg|$(fnv_digest(ρ))")
            end
            st === first(digest_pairs(shape)) || continue
            m = moved_species_2d(T, grid, 300; dead = DEAD2)
            f = test_field_2d(T, grid, (a, b) -> seeded(T, sin(0.7a) + cos(0.3b) / 2))
            out = fill!(similar(m.w), zero(T))
            on(be, gather!, out, f, m, shape, st, grid; workgroupsize = wg)
            push!(lines, "gather2d|$L|$(order(shape)),$(stagger_label(st))|$wg|$(fnv_digest(out))")
        end
        grid = grid2(T, 2; n = (9, 4))
        x = field(backend, T, grid)
        foreach(i -> x[i] = seeded(T, cos(1.3i)), eachindex(x))
        push!(lines, "fill2d|$L|9x4|$wg|$(fnv_digest(on(be, fill_guards!, copy(x), grid; workgroupsize = wg)))")
        push!(lines, "fold2d|$L|9x4|$wg|$(fnv_digest(on(be, fold_guards!, copy(x), grid; workgroupsize = wg)))")
        s = load_2d(T, grid2(T, 2), PPC2; workgroupsize = wg, be)
        push!(lines, "load2d|$L|ppc=2x3|$wg|$(fnv_digest(species_arrays(s)...))")
    end
    return lines
end
