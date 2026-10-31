# The module-declared field layout: which components exist and where each one sits.

"""
Stagger of the charge density `ρ`, the primal nodes of the 1D Yee grid: the cell edges, for
every shape order. Ey, Ez, Bx, Jy and Jz share them.
"""
const RHO_STAGGER = Node()

"""
    FIELD_LAYOUT

The module's field layout as a `NamedTuple` of engine staggers: the 1D Yee grid with the
primal nodes at the cell edges (`Node()`, where ρ sits) and the dual nodes at the cell centres
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
const FIELD_LAYOUT = (Ex = Center(), Ey = Node(), Ez = Node(),
                      Bx = Node(), By = Center(), Bz = Center(),
                      Jx = Center(), Jy = Node(), Jz = Node())

"""
    em_fields(backend, T, grid)

The zeroed fields of [`FIELD_LAYOUT`](@ref), one engine field per component, as a
`NamedTuple` with the same keys.
"""
em_fields(backend, ::Type{T}, grid::PeriodicGrid{1}) where {T} = map(_ -> field(backend, T, grid), FIELD_LAYOUT)

"""
    em_guard_width(shape)

Guard cells the module's step needs on each side: those of `deposit_flux!` with ρ at
[`RHO_STAGGER`](@ref) (its old and new stencils together span one node more than one stencil),
which also cover the gathers and the Yee stencils (one cell).
"""
em_guard_width(shape::Shape) = max(ParticleMeshCore.flux_guard_width(shape, RHO_STAGGER), 1)
