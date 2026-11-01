# The computations of the WP-011 contract tests (T-172 to T-177), shared by engine.jl and by
# threads_worker.jl, which repeats them with 1 and 4 threads. Needs setup.jl, helpers.jl and
# digest.jl.

using KernelAbstractions: @kernel, @index
const Atomix = ParticleMeshCore.Atomix      # the engine's own; no test dependency of its own

"Launch shapes T-177 compares: the backend default and two explicit sizes."
const WG_SIZES = (nothing, 7, 64)
"Grids of T-172: one tile, and three tiles with a partial last one."
const SCAN_GRIDS = (64, 2 * ParticleMeshCore.SCAN_TILE + 37)

"An input value of `T` whose partials, for duals, vary with `j` (so the scan's partials are exercised)."
varied(::Type{T}, x, j) where {T<:AbstractFloat} = T(x)
varied(::Type{D3}, x, j) = mkdual(x, sin(2.3j), cos(0.7j) / 2, x^2)

function scan_input(::Type{T}, n) where {T}
    grid = PeriodicGrid(n, basetype(T)(3 // 8), 2)
    a = test_field(T, grid, j -> varied(T, sin(0.37j) + cos(1.1j) / 5 + 3 / 10, j))
    return grid, a
end

"`cumsum_periodic!` of the T-172 input on `n` cells, on the backend `be`: `(out, a, grid)` as host arrays."
function scan_output(::Type{T}, n; workgroupsize = nothing, be = CPU()) where {T}
    grid, a = scan_input(T, n)
    out = field(backend, T, grid)
    on(be, cumsum_periodic!, out, a, grid; workgroupsize)
    return out, a, grid
end

"""
    serial_tiled_cumsum(a, dx, tile)

A plain host implementation of the association order documented for `cumsum_periodic!`, on
the interior values `a`: tiles of `tile` elements; a serial left-to-right sweep inside each
tile starting from zero; tile totals combined by a serial scan in tile order with no addition
to the first tile; the mean of `a` and the mean of the output summed over the same tiles in the
same order.
"""
function serial_tiled_cumsum(a::Vector{T}, dx, tile) where {T}
    n = length(a)
    tiles = [((k - 1) * tile + 1):min(k * tile, n) for k in 1:cld(n, tile)]
    function tilesum(v, r)
        s = zero(T)
        for j in r
            s += v[j]
        end
        return s
    end
    function combine(ts)
        t = ts[1]
        for k in 2:length(ts)
            t += ts[k]
        end
        return t
    end
    ā = combine([tilesum(a, r) for r in tiles]) / T(n)
    out = similar(a)
    for r in tiles
        acc = zero(T)
        for j in r
            acc += (a[j] - ā) * dx
            out[j] = acc
        end
    end
    totals = [out[last(r)] for r in tiles]
    local off
    for k in 2:length(tiles)
        off = k == 2 ? totals[1] : off + totals[k - 1]
        for j in tiles[k]
            out[j] += off
        end
    end
    Ē = combine([tilesum(out, r) for r in tiles]) / T(n)
    for j in 1:n
        out[j] -= Ē
    end
    return out
end

# The spike's q·w deposition kernels (WP-010, `src/transfers.jl` at the commit before WP-011),
# copied verbatim as the oracle of T-174's bitwise check of the default quantity.
@kernel function spike_deposit_atomic_kernel!(ρ, s, shape::Shape{P}, stagger, g::Int32) where {P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ParticleMeshCore.ALIVE
        qw = s.q * s.w[i]
        f, w = shape_weights(shape, stagger, s.offset[1][i])
        base = s.cell[1][i] + f + g
        for k in 1:(P + 1)
            Atomix.@atomic ρ[base + Int32(k - 1)] += qw * w[k]
        end
    end
end

@kernel function spike_deposit_sorted_kernel!(ρ, s, @Const(seg_start), @Const(seg_stop), shape, stagger,
                                              n::Int32, g::Int32, lo::Int32, hi::Int32)
    a = Int32(@index(Global, Linear))
    T = eltype(ρ)
    j = a - g
    acc = zero(T)
    @inbounds for c in max(Int32(1), j - hi):min(n, j - lo)
        for i in seg_start[c]:seg_stop[c]
            f, w = shape_weights(shape, stagger, s.offset[1][i])
            acc += s.q * s.w[i] * ParticleMeshCore.pick(w, j - c - f)
        end
    end
    @inbounds ρ[a] += acc
end

function spike_deposit!(mode, ρ, s, shape, stagger, grid, ws)
    if effective_mode(mode, eltype(ρ)) isa Fast
        spike_deposit_atomic_kernel!(get_backend(ρ))(ρ, s, shape, stagger, Int32(guards(grid)); ndrange = capacity(s))
    else
        lo, hi = ParticleMeshCore.stencil_reach(shape, stagger)
        spike_deposit_sorted_kernel!(get_backend(ρ))(ρ, s, ws.seg_start, ws.seg_stop, shape, stagger, Int32(ncells(grid)),
                                              Int32(guards(grid)), Int32(lo), Int32(hi); ndrange = length(ρ))
    end
    return ρ
end

"q·w·u_y, the module-supplied quantity of T-174 and T-175."
struct ChargeUy end
@inline (::ChargeUy)(s, i) = s.q * s.w[i] * s.u[2][i]
"|q·w·u_y|: deposited, the sum of the absolute contributions to each node."
struct AbsChargeUy end
@inline (::AbsChargeUy)(s, i) = abs(s.q * s.w[i] * s.u[2][i])

"The T-174 species: `test_species` with `u_y` in [-1, 1) instead of the particle id."
function quantity_species(::Type{T}, grid, np; seed) where {T}
    s = test_species(T, grid, np; seed)
    for i in 1:np
        s.u[2][i] = seeded(T, cos(1.7i + seed))
    end
    return s
end

"Deposit into a fresh field (guards not folded) on `be`, sorting first when the mode needs it; `spike = true` runs the oracle."
function deposit_case(::Type{T}, mode, shape, stagger; np = 300, seed = 5, quantity = ChargeWeight(),
                      workgroupsize = nothing, spike = false, uy = false, be = CPU()) where {T}
    grid = PeriodicGrid(12, basetype(T)(1 // 2), guard_width(shape))
    s = uy ? quantity_species(T, grid, np; seed) : test_species(T, grid, np; seed)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    ρ = field(backend, T, grid)
    if spike
        on(be, spike_deposit!, mode, ρ, s, shape, stagger, grid, ws)
    else
        on(be, deposit!, mode, ρ, s, shape, stagger, grid, ws; quantity, workgroupsize)
    end
    return ρ
end

const DEAD_SLOTS = (17, 101, 250)

"""
    sorted_extras(T; workgroupsize) -> (species, workspace)

T-173's species: 500 alive particles from `test_species` with three of them freed in the
middle and three free slots at the end, carrying the extras `old_slot::Int32` (the slot before
the sort) and `old_offset::T` (a copy of the offset), sorted once.
"""
function sorted_extras(::Type{T}; workgroupsize = nothing, be = CPU()) where {T}
    grid = PeriodicGrid(9, one(basetype(T)), 2)
    s = test_species(T, grid, 500; seed = 13, extras = (old_slot = Int32, old_offset = T))
    for i in DEAD_SLOTS
        s.status[i] = ParticleMeshCore.FREE
    end
    for i in 1:capacity(s)
        s.extras.old_slot[i] = Int32(i)
        s.extras.old_offset[i] = s.offset[1][i]
    end
    ws = Workspace(s, grid)
    on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    return s, ws, grid
end

"Every slot's extras still describe the particle in that slot: the old slot is the permutation, the offset copy is bitwise the offset, the id matches."
function extras_match(s, ws)
    np = 500
    slot_ok = all(i -> s.extras.old_slot[i] == ws.perm[i], 1:capacity(s))
    offset_ok = same_bits(s.extras.old_offset, s.offset[1])
    id_ok = all(i -> s.extras.old_slot[i] > np || Int(value_of(s.u[2][i])) == s.extras.old_slot[i], 1:capacity(s))
    return slot_ok && offset_ok && id_ok
end

species_arrays(s) = (s.cell..., s.offset..., s.u..., s.w, s.status, s.count, values(s.extras)...)

"""
    thread_digests(be = CPU(); wgs = WG_SIZES) -> Vector{String}

`"kind|type|case|workgroupsize|digest"` for every Reference-mode result whose bits must not
depend on the launch shape or the thread count: `cumsum_periodic!` (T-172, T-177),
`sort_by_cell!` with extras (T-173, T-177), `deposit!` with the default and a supplied quantity
(T-177), and T-173's match flag; on the backend `be`, for the launch shapes `wgs`.
"""
function thread_digests(be = CPU(); wgs = WG_SIZES)
    lines = String[]
    for T in TYPES, wg in wgs
        for n in SCAN_GRIDS
            out, _, _ = scan_output(T, n; workgroupsize = wg, be)
            push!(lines, "scan|$(type_label(T))|n=$n|$wg|$(fnv_digest(out))")
        end
        s, ws, _ = sorted_extras(T; workgroupsize = wg, be)
        push!(lines, "sort|$(type_label(T))|extras|$wg|$(fnv_digest(species_arrays(s)..., ws.perm, ws.seg_start, ws.seg_stop))")
        push!(lines, "sortmatch|$(type_label(T))|extras|$wg|$(extras_match(s, ws))")
        for shape in SHAPES, stagger in STAGGERS, (qlabel, quantity, uy) in (("qw", ChargeWeight(), false), ("qwuy", ChargeUy(), true))
            ρ = deposit_case(T, Reference(), shape, stagger; quantity, uy, workgroupsize = wg, be)
            push!(lines, "deposit|$(type_label(T))|$(order(shape)),$(nameof(typeof(stagger))),$qlabel|$wg|$(fnv_digest(ρ))")
        end
    end
    return lines
end

"`FASTSPIKE|type|case|flag`: the Fast default deposit bitwise against the spike's kernel (meaningful on one thread only)."
function fast_spike_lines()
    lines = String[]
    for T in (Float32, Float64), shape in SHAPES, stagger in STAGGERS
        ρ = deposit_case(T, Fast(), shape, stagger)
        ρs = deposit_case(T, Fast(), shape, stagger; spike = true)
        push!(lines, "FASTSPIKE|$(type_label(T))|$(order(shape)),$(nameof(typeof(stagger)))|$(same_bits(ρ, ρs))")
    end
    return lines
end
