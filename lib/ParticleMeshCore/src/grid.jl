"""
    PeriodicGrid{N,T}(n, dx, g)

A periodic structured grid of `n` interior cells per axis with cell size `dx` (normalized units,
c/ω₀) and `g` guard cells on each side. Cell `i ∈ 1:n` covers `[(i-1)dx, i·dx)`. A field on the
grid is an array of size `n .+ 2g`; interior index `i` is array index `i + g`.

Only `N = 1` is implemented.
"""
struct PeriodicGrid{N,T}
    n::NTuple{N,Int}
    dx::NTuple{N,T}
    g::Int
end

PeriodicGrid(n::Integer, dx::T, g::Integer) where {T} = PeriodicGrid{1,T}((Int(n),), (dx,), Int(g))

ncells(grid::PeriodicGrid) = grid.n
ncells(grid::PeriodicGrid{1}) = grid.n[1]
cellsize(grid::PeriodicGrid{1}) = grid.dx[1]
guards(grid::PeriodicGrid) = grid.g
Base.length(grid::PeriodicGrid{1}) = cellsize(grid) * ncells(grid)

"""
    field(backend, T, grid)

A zeroed field with guards on `grid`, allocated on `backend` with element type `T`.
"""
field(backend, ::Type{T}, grid::PeriodicGrid) where {T} =
    KernelAbstractions.zeros(backend, T, (grid.n .+ 2grid.g)...)

"Array index range of the interior cells along the one axis."
interior(grid::PeriodicGrid{1}) = (grid.g + 1):(grid.g + ncells(grid))
interior_view(a::AbstractVector, grid::PeriodicGrid{1}) = view(a, interior(grid))
