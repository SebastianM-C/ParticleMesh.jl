# The bins that `sort_by_cell!` groups particles by and the `Reference()` deposition walks:
# cells, or the cells of the nearest stencil (WP-041).

"""
    CellBins(), NodeBins(shape, stagger)

What a [`Workspace`](@ref)'s sort groups particles by (its keyword `bins`). `CellBins()`, the
default: the particle's cell. `NodeBins(shape, stagger)`, `stagger` one `Stagger` per axis: the
first node of the particle's stencil with that shape and stagger, as the cell `c + h` with
`h = f(δ) - lo` per axis (`shape_weights`' `f`, `stencil_reach`'s `lo`), wrapped periodically.
`h` is 0 or 1, and 0 for every offset where the stencil does not depend on it (order 1 on
`Node()`, order 2 on `Center()`); where it does (order 2 on `Node()`, order 1 on `Center()`),
the particles of a bin share their stencil's nodes, so a bin's stencil is `P + 1` nodes per axis
where a cell's is `P + 2`, and its flux stencil `P + 3` where a cell's is `P + 4`. A bin moves
by at most one per axis when its particle moves by less than one cell, as a cell does.

The `Reference()` deposition walks the bins in the same mechanism as the cells, with the reach of
`deposit_reach`; its bits depend on the bins (they are those of the cells for `CellBins()`), and
are deterministic for either.
"""
struct CellBins end
struct NodeBins{S<:Shape,G<:Tuple}
    shape::S
    stagger::G
end
NodeBins(shape::Shape, stagger::Stagger) = NodeBins(shape, (stagger,))

# The shift `h` of one axis, 0 or 1: the bin is `c + h`.
@inline bin_shift(::CellBins, ::Int, δ) = Int32(0)
@inline function bin_shift(b::NodeBins, d::Int, δ)
    f, _ = shape_weights(b.shape, b.stagger[d], δ)
    return f - Int32(stencil_reach(b.shape, b.stagger[d])[1])
end

"""
    particle_bin(bins, species, i, n) -> NTuple{N,Int32}

The bin of particle `i` per axis on a periodic grid of `n` cells per axis. Device-callable.
"""
@inline particle_bin(::CellBins, s, i, n::NTuple{N,Int32}) where {N} = @inbounds ntuple(d -> s.cell[d][i], Val(N))
@inline function particle_bin(b::NodeBins, s, i, n::NTuple{N,Int32}) where {N}
    return @inbounds ntuple(Val(N)) do d
        a = s.cell[d][i] + bin_shift(b, d, s.offset[d][i])
        ifelse(a > n[d], a - n[d], a)
    end
end

"""
    deposit_reach(bins, shape, stagger, d) -> (lo, hi)

The nodes a particle in bin `b` touches with `shape` at `stagger` on axis `d`: `b + lo:b + hi`
(`stencil_reach` for `CellBins()`). For `NodeBins` of this shape and stagger on axis `d`, the
`P + 1` nodes of the stencil; for another shape or stagger, the reach of the cells one before the
bin and the bin itself.
"""
@inline deposit_reach(::CellBins, shape::Shape, stagger::Stagger, d::Int) = map(Int32, stencil_reach(shape, stagger))
@inline function deposit_reach(b::NodeBins, shape::Shape{P}, stagger::Stagger, d::Int) where {P}
    lo, hi = map(Int32, stencil_reach(shape, stagger))
    lb, _ = stencil_reach(b.shape, b.stagger[d])
    hmax = Int32(_width(bin_hmax(b, d)))
    matched = b.shape === shape && b.stagger[d] === stagger
    return matched ? (Int32(lb), Int32(lb + P)) : (lo - hmax, hi)
end

# The largest shift of axis `d`, 0 or 1, as a static value: the cell's stencil width less the bin's.
@inline bin_hmax(::CellBins, d::Int) = Val(0)
@inline bin_hmax(b::NodeBins, d::Int) = Val(_width(stencil_width(b.shape, b.stagger[d])) - order(b.shape) - 1)

@inline stencil_width(::Shape{1}, ::Node) = Val(2)
@inline stencil_width(::Shape{1}, ::Center) = Val(3)
@inline stencil_width(::Shape{2}, ::Center) = Val(3)
@inline stencil_width(::Shape{2}, ::Node) = Val(4)
_width(::Val{M}) where {M} = M
