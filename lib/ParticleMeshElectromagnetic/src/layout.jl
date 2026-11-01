# The module-declared field layout: which components exist and where each one sits.

"""
Stagger of the charge density `ρ`, the primal nodes of the Yee grid: the cell edges along every
resolved axis, for every shape order. In 1D Ey, Ez, Bx, Jy and Jz share them; in N dimensions
Ez and Jz (the components of the collapsed axis z at N = 2).
"""
const RHO_STAGGER = Node()

# The staggers of component c by the rule of `field_layout`: a bare `Stagger` at N = 1 (the
# engine's 1D convention), an `NTuple{N,Stagger}` above.
_stagger(::Val{N}, c::Int, on, off) where {N} = ntuple(d -> d == c ? on : off, Val(N))
e_stagger(::Val{1}, c::Int) = c == 1 ? Center() : Node()
e_stagger(v::Val{N}, c::Int) where {N} = _stagger(v, c, Center(), Node())
b_stagger(::Val{1}, c::Int) = c == 1 ? Node() : Center()
b_stagger(v::Val{N}, c::Int) where {N} = _stagger(v, c, Node(), Center())

"""
    field_layout(Val(N))

The module's field layout on an `N`-dimensional Yee grid, as a `NamedTuple` of engine staggers
with the keys Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz: with ρ on the primal nodes ([`RHO_STAGGER`](@ref)),
component c of E (and of J) sits at `Center()` along axis c and at `Node()` along every other
resolved axis, component c of B at `Node()` along axis c and at `Center()` along the others; a
component of a collapsed axis (c > N) has no axis of its own, so E_c and J_c sit at `Node()` and
B_c at `Center()` on every axis. A stagger is a bare `Stagger` at N = 1 and an `NTuple{N}` above.

| component | N = 1 | N = 2 |
|---|---|---|
| Ex, Jx | `Center()` | `(Center(), Node())` |
| Ey, Jy | `Node()` | `(Node(), Center())` |
| Ez, Jz | `Node()` | `(Node(), Node())` |
| Bx | `Node()` | `(Node(), Center())` |
| By | `Center()` | `(Center(), Node())` |
| Bz | `Center()` | `(Center(), Center())` |

At N = 3 it is the standard Yee cell. A stagger's index `j` along an axis of cell size Δ sits at
(j − 1)·Δ for `Node()` and at (j − 1/2)·Δ for `Center()`, so the forward difference maps `Node()`
to `Center()` values and the backward difference `Center()` to `Node()` ([`solve_fields!`](@ref)).
"""
field_layout(v::Val{N}) where {N} = (Ex = e_stagger(v, 1), Ey = e_stagger(v, 2), Ez = e_stagger(v, 3),
                                     Bx = b_stagger(v, 1), By = b_stagger(v, 2), Bz = b_stagger(v, 3),
                                     Jx = e_stagger(v, 1), Jy = e_stagger(v, 2), Jz = e_stagger(v, 3))

"""
    FIELD_LAYOUT

The module's 1D field layout, [`field_layout`](@ref)`(Val(1))`: the 1D Yee grid with the primal
nodes at the cell edges (`Node()`, where ρ sits) and the dual nodes at the cell centres
(`Center()`), half a cell to the right of the primal node with the same index.

| component | stagger | position of index `j` |
|---|---|---|
| Ey, Ez, Bx, Jy, Jz | `Node()` | x = (j − 1)·Δx |
| Ex, By, Bz, Jx | `Center()` | x = (j − 1/2)·Δx |

This is the layout of `deposit_flux!` with ρ at `Node()`: `Jx[j]` is the face between the ρ
nodes `j` and `j + 1`, so the discrete continuity equation is
(ρⁿ⁺¹[j] − ρⁿ[j])/Δt + (Jx[j] − Jx[j−1])/Δx = 0 and Gauss's law takes the backward difference of
Ex, (Ex[j] − Ex[j−1])/Δx = ρ[j] − mean(ρ). Every field is an engine field (with guards) on one
`PeriodicGrid`; the transverse components exist and vary only along x.
"""
const FIELD_LAYOUT = field_layout(Val(1))

"""
    em_fields(backend, T, grid::PeriodicGrid{N})

The zeroed fields of [`field_layout`](@ref)`(Val(N))`, one engine field (with guards) per
component, as a `NamedTuple` with the same keys; N = 1 and N = 2.
"""
em_fields(backend, ::Type{T}, grid::PeriodicGrid{N}) where {T,N} = map(_ -> field(backend, T, grid), field_layout(Val(N)))

"""
    em_guard_width(shape)

Guard cells the module's step needs on each side: those of `deposit_flux!` with ρ at
[`RHO_STAGGER`](@ref) (its old and new stencils together span one node more than one stencil),
which also cover the gathers and the Yee stencils (one cell).
"""
em_guard_width(shape::Shape) = max(ParticleMeshCore.flux_guard_width(shape, RHO_STAGGER), 1)
