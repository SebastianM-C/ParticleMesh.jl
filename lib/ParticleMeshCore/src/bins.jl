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

`NodeBins(shape, stagger; open)`, `open` one `Bool` per axis (all `false` by default): on an open
axis the bin is not wrapped, so that a particle near the last node of the axis is in the bin
`n + 1` past the last cell, and the axis has `n + 1` bins (`nbins`). A caller whose axis is
not periodic (the moving window's x axis, whose x-guards are dropped instead of folded) takes it
open, since a wrapped bin there would deposit the particle's stencil at the other edge. The bins
of an open axis reach one node further past the last cell than its periodic bins where the stencil
shifts within a cell, as far as its cells reach (checked against the guards). On a periodic grid an open axis gives the same field as
the periodic one, to roundoff (another order of summation where a particle sits in bin `n + 1`).
"""
struct CellBins end
struct NodeBins{S<:Shape,G<:Tuple,O<:Tuple}
    shape::S
    stagger::G
    open::O
    function NodeBins(shape::S, stagger::G, open::O) where {S<:Shape,N,G<:NTuple{N,Stagger},O<:NTuple{N,Bool}}
        return new{S,G,O}(shape, stagger, open)
    end
end
NodeBins(shape::Shape, stagger::NTuple{N,Stagger}; open::NTuple{N,Bool} = ntuple(_ -> false, Val(N))) where {N} =
    NodeBins(shape, stagger, open)
NodeBins(shape::Shape, stagger::Stagger; open::Bool = false) = NodeBins(shape, (stagger,), (open,))

"""
    nbins(bins, n) -> bins per axis

The number of bins per axis on a grid of `n` cells per axis (an integer at N = 1, a tuple
otherwise), of the same integer type: the cells for `CellBins()` and the periodic axes of
`NodeBins`, `n + 1` on an open axis of `NodeBins`. The segments of a workspace and its flux
patches are per linear bin, `prod(nbins(bins, n))` of them.
"""
nbins(::CellBins, n) = n
nbins(b::NodeBins, n::Integer) = n + oftype(n, b.open[1])
nbins(b::NodeBins, n::Tuple) = map((m, o) -> m + oftype(m, o), n, b.open)

# The reach of the bins against the guards: on an open axis the last bin is one past the last
# cell, and holds particles where the bin's shift can be 1 (`bin_hmax`), so there its nodes reach
# one further on the high side, as far as the last cell's.
guard_reach(::CellBins, reach) = reach
guard_reach(b::NodeBins, reach) =
    ntuple(d -> (reach[d][1], reach[d][2] + oftype(reach[d][2], b.open[d] * _width(bin_hmax(b, d)))), length(reach))

# The shift `h` of one axis, 0 or 1: the bin is `c + h`.
@inline bin_shift(::CellBins, ::Int, δ) = Int32(0)
@inline function bin_shift(b::NodeBins, d::Int, δ)
    f, _ = shape_weights(b.shape, b.stagger[d], δ)
    return f - Int32(stencil_reach(b.shape, b.stagger[d])[1])
end

"""
    particle_bin(bins, species, i, nb) -> NTuple{N,Int32}

The bin of particle `i` per axis, `nb` the bins per axis (`nbins`): `c + h`, wrapped to the first
bin past `nb`, which on a periodic axis (`nb` its cells) is the periodic image and on an open axis
(`nb` one more) never happens. Device-callable.
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
