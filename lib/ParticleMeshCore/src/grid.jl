"""
    PeriodicGrid{N,T}(n, dx, g)
    PeriodicGrid(n::Integer, dx, g)              # N = 1
    PeriodicGrid(n::NTuple{N}, dx::NTuple{N}, g)  # N = 1 or 2

A periodic structured grid of `n` interior cells per axis with cell size `dx` (normalized units,
c/ω₀) and `g` guard cells on each side of every axis. Cell `i ∈ 1:n_d` of axis `d` covers
`[(i-1)dx_d, i·dx_d)`. A field on the grid is a column-major array of size `n .+ 2g`; interior
index `i` of axis `d` is array index `i + g`. The tuple constructor checks `n_d ≥ 2g` per axis
(the condition under which fill and fold touch disjoint cells).

Cells linearize column-major, axis 1 fastest: [`linear_cell`](@ref) and its inverse
[`cartesian_cell`](@ref). N = 1 and N = 2 are implemented and tested.
"""
struct PeriodicGrid{N,T}
    n::NTuple{N,Int}
    dx::NTuple{N,T}
    g::Int
    PeriodicGrid{N,T}(n, dx, g) where {N,T} = new{N,T}(n, dx, g)
end

PeriodicGrid(n::Integer, dx::T, g::Integer) where {T} = PeriodicGrid((n,), (dx,), g)

function PeriodicGrid(n::NTuple{N,Integer}, dx::NTuple{N,T}, g::Integer) where {N,T}
    g >= 0 || throw(ArgumentError("guard width $g < 0"))
    all(m -> m >= max(1, 2g), n) || throw(ArgumentError("every axis needs n_d ≥ max(1, 2g); n = $n, g = $g"))
    return PeriodicGrid{N,T}(Int.(n), dx, Int(g))
end

ncells(grid::PeriodicGrid) = grid.n
ncells(grid::PeriodicGrid{1}) = grid.n[1]
cellsize(grid::PeriodicGrid) = grid.dx
cellsize(grid::PeriodicGrid{1}) = grid.dx[1]
guards(grid::PeriodicGrid) = grid.g
Base.length(grid::PeriodicGrid{1}) = cellsize(grid) * ncells(grid)

"""
    field(backend, T, grid)

A zeroed field with guards on `grid`, allocated on `backend` with element type `T`.
"""
field(backend, ::Type{T}, grid::PeriodicGrid) where {T} =
    KernelAbstractions.zeros(backend, T, (grid.n .+ 2grid.g)...)

"Array index range of the interior cells along the one axis (N = 1); one range per axis otherwise."
interior(grid::PeriodicGrid{1}) = (grid.g + 1):(grid.g + ncells(grid))
interior(grid::PeriodicGrid{N}) where {N} = ntuple(d -> (grid.g + 1):(grid.g + grid.n[d]), Val(N))
interior_view(a::AbstractVector, grid::PeriodicGrid{1}) = view(a, interior(grid))
interior_view(a::AbstractArray{<:Any,N}, grid::PeriodicGrid{N}) where {N} = view(a, interior(grid)...)

"""
    linear_cell(c::NTuple{N,I}, n::NTuple{N,I}) -> I

The column-major linear index of cell `c` (axis 1 fastest) on `n` cells per axis, `1:prod(n)`.
Device-callable; the sort key and the segment bounds of `Workspace` use it.
"""
@inline function linear_cell(c::NTuple{N,I}, n::NTuple{N,I}) where {N,I<:Integer}
    L = c[N] - one(I)
    for d in (N - 1):-1:1
        L = L * n[d] + (c[d] - one(I))
    end
    return L + one(I)
end

"""
    cartesian_cell(L::I, n::NTuple{N,I}) -> NTuple{N,I}

The inverse of [`linear_cell`](@ref): the per-axis cell of linear index `L`.
"""
@inline cartesian_cell(L::I, n::NTuple{N,I}) where {N,I<:Integer} = _cartesian(L - one(I), n)
@inline _cartesian(r::I, ::Tuple{}) where {I} = ()
@inline _cartesian(r::I, n::Tuple) where {I} = (r % first(n) + one(I), _cartesian(r ÷ first(n), Base.tail(n))...)
