# Mesh operators: physics-free operations on fields. A module names what they solve.

"""
Tile size of the association order of `cumsum_periodic!`: a package constant, so the order, and
with it the bits, never depends on the launch configuration or the backend.
"""
const SCAN_TILE = 1024

@inline function tile_sum(v, g::Int32, lo::Int32, hi::Int32, ::Type{T}) where {T}
    s = zero(T)
    @inbounds for j in lo:hi
        s += v[g + j]
    end
    return s
end

@inline tile_range(k::Int32, tile::Int32, n::Int32) = ((k - Int32(1)) * tile + Int32(1)):min(k * tile, n)

# The documented order, in one work item: a later backend may run the tiles in parallel, as long
# as it keeps this association.
@kernel function cumsum_periodic_kernel!(out, @Const(a), dx, n::Int32, g::Int32, tile::Int32)
    T = eltype(out)
    ntiles = cld(n, tile)
    @inbounds begin
        # The mean of a: tile sums, combined in tile order (no addition to the first tile).
        r = tile_range(Int32(1), tile, n)
        asum = tile_sum(a, g, first(r), last(r), T)
        for k in Int32(2):ntiles
            r = tile_range(k, tile, n)
            asum += tile_sum(a, g, first(r), last(r), T)
        end
        ā = asum / T(n)
        # A serial sweep from zero inside each tile.
        for k in Int32(1):ntiles
            acc = zero(T)
            for j in tile_range(k, tile, n)
                acc += (a[g + j] - ā) * dx
                out[g + j] = acc
            end
        end
        # The tile totals by a serial exclusive scan in tile order; each tile's offset is added
        # once to its elements, none to the first tile.
        if ntiles > Int32(1)
            off = out[g + last(tile_range(Int32(1), tile, n))]
            for k in Int32(2):ntiles
                r = tile_range(k, tile, n)
                total = out[g + last(r)]
                for j in r
                    out[g + j] += off
                end
                off += total
            end
        end
        # The mean of the output over the same tiles in the same order, subtracted.
        r = tile_range(Int32(1), tile, n)
        osum = tile_sum(out, g, first(r), last(r), T)
        for k in Int32(2):ntiles
            r = tile_range(k, tile, n)
            osum += tile_sum(out, g, first(r), last(r), T)
        end
        ō = osum / T(n)
        for j in Int32(1):n
            out[g + j] -= ō
        end
    end
end

"""
    cumsum_periodic!(out, a, grid; workgroupsize = nothing)

The zero-mean periodic prefix sum of the field `a` on the interior of `grid`, written into the
field `out` with its guards filled (both arrays with guards; the guards of `a` are not read).
With `ā` the mean of the interior of `a` and `dx = cellsize(grid)`, `out` is the periodic
antiderivative of `a - ā`:

    out[j] = Σ_{k ≤ j} (a[k] - ā)·dx - Ō,   Ō = mean over j of the sum before it,

so that, up to roundoff, `(out[j] - out[j-1]) / dx = a[j] - ā` for every interior `j` (with
`out[0] ≡ out[n]`, the periodic image) and `Σ_j out[j] = 0` (T-172). A module uses it as its
1D periodic Gauss or Poisson solve; the engine does not name the equation.

Association order, part of the contract: the interior is cut into tiles of `SCAN_TILE`
elements (the last one partial); inside each tile a serial left-to-right sweep from zero
accumulates `(a[k] - ā)·dx`; the tile totals are combined by a serial exclusive scan in tile
order, and each tile's offset is added once to its elements, none to the first tile. `ā` and
`Ō` are each the sum over the same tiles in the same order (a serial sum inside each tile,
the tile sums added in tile order to the first one), divided by `n`. So a grid of at most
`SCAN_TILE` cells is one serial sweep. The order is fixed by the operator, not by the launch
configuration or the backend; every backend implementing it gives the same association, and
results differ across backends only by their own arithmetic.

It runs as one work item on every backend for now (O(n), serial), which ignores
`workgroupsize`; the guard fill takes it. A parallel scan of the same order is a later
backend's (WP-013).
"""
function cumsum_periodic!(out::AbstractVector, a::AbstractVector, grid::PeriodicGrid{1}; workgroupsize = nothing)
    backend = get_backend(out)
    cumsum_periodic_kernel!(backend)(out, a, cellsize(grid), Int32(ncells(grid)), Int32(guards(grid)), Int32(SCAN_TILE);
                                     ndrange = 1)
    fill_guards!(out, grid; workgroupsize)
    return out
end
