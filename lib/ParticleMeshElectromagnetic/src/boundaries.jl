# Open transverse boundaries at N = 2 (WP-027, A2): the problem says that axis 2 is open
# (`EMProblem(...; boundaries = (:periodic, :open))`), the numerics say how: a simple absorbing
# layer for the fields (`AbsorbingLayer`, A2: PML later) and particles retired through the alive
# mask when they leave the box across the open axis (§0.4 rule 1: no buffer changes size).
#
# The grid underneath stays the engine's `PeriodicGrid{2}`: the layer occupies the outer `cells`
# rows on each side of the box, so the two layers meet at the periodic seam y = 0 ≡ Ly and form
# one slab of 2·cells rows. A wave that leaves the interior crosses that slab before it can come
# back from the other side, and the absorbing slab damps it on the way. In the layer the
# conductivity is σ(y) = strength·(depth/thickness)^order, with depth the distance into the layer
# from its inner edge (thickness = cells·Δy, so σ = strength at the box edge and 0 at the inner
# edge), the same σ for E and for B (the matched lossy medium, whose impedance is that of vacuum,
# so a continuum wave at normal incidence enters it without reflection). It is applied by
# operator splitting: after each Yee sub-step of length τ (Δt/2 for each B half step, Δt for the
# E step), every component is multiplied by exp(−σ·τ) at its own stagger along y. The interior
# update is untouched, so Gauss's law and ∇·B hold there as without the layer; in the layer
# D⁻·E changes by D⁻·((e^{−σΔt} − 1)E), so Gauss's law is broken there, which is what absorbing
# the field of charges that left means.

"""
    AbsorbingLayer(; cells, strength, order = 2)

The simple absorbing layer of the open transverse boundaries (A2), a numerics choice of
[`EMNumerics`](@ref) (`absorber`): `cells` rows on each side of the box along axis 2 (inside the
box), the conductivity at the box edge `strength` (a rate, in ω₀, A1) and the polynomial `order`
of its grading, σ(y) = strength·(depth/(cells·Δy))^order with depth the distance into the layer
from its inner edge. The same σ damps E and B (a matched layer). The layer must be at least as
thick as the grid's guards and leave interior rows between the two layers (checked by the state).
"""
struct AbsorbingLayer
    cells::Int
    strength::Float64
    order::Int
    function AbsorbingLayer(cells::Integer, strength::Real, order::Integer)
        cells >= 1 || throw(ArgumentError("an absorbing layer needs at least one cell, got $cells"))
        strength >= 0 && isfinite(strength) || throw(ArgumentError("the layer strength must be finite and ≥ 0, got $strength"))
        order >= 0 || throw(ArgumentError("the grading order must be ≥ 0, got $order"))
        return new(Int(cells), Float64(strength), Int(order))
    end
end
AbsorbingLayer(; cells, strength, order = 2) = AbsorbingLayer(cells, strength, order)

"""
    layer_sigma(layer, e, thickness) -> σ

The conductivity of `layer` at the distance `e` from the nearest box edge along the open axis, for
a layer of `thickness` (= cells·Δy): `strength·((thickness − e)/thickness)^order` inside the layer,
zero outside, in the element type of `e`. A device function.
"""
@inline function layer_sigma(strength::T, order::Int, e::T, thickness::T) where {T}
    depth = (thickness - e) / thickness
    return ifelse(depth > zero(T), strength * depth^order, zero(T))
end

# The damping factor exp(−σ·τ) per array row j (guards included) for a field whose rows sit at
# (j₀ + s)·Δy, j₀ the interior row j aliases; written once by the state.
@kernel function layer_factor_kernel!(fac, strength, order::Int, s, τ, dy, n2::Int32, g::Int32, cells::Int32)
    j = Int32(@index(Global, Linear))
    T = eltype(fac)
    j0 = mod(j - g - Int32(1), n2)
    y = (T(j0) + s) * dy
    ly = T(n2) * dy
    e = min(y, ly - y)
    σ = layer_sigma(strength, order, e, T(cells) * dy)
    @inbounds fac[j] = exp(-σ * τ)
end

"""
    LayerFactors

The absorbing layer placed on a grid by the state: the damping factors exp(−σ·τ) per array row
along axis 2 (guards included), for the rows at `Node()` and at `Center()` along y and for the
sub-step lengths τ = Δt/2 (`half`) and Δt (`full`); the number of layer rows `cells`, the guard
width and the interior rows, which fix the rows the damping visits. `dt` is the time step the
factors were made for.
"""
struct LayerFactors{V,T}
    node_half::V
    center_half::V
    node_full::V
    center_full::V
    cells::Int
    g::Int
    n::NTuple{2,Int}
    dt::T
end

"""
    LayerFactors(layer::AbsorbingLayer, grid::PeriodicGrid{2}, dt; backend, workgroupsize = nothing)

Write the damping factors of `layer` on `grid` for the step `dt` (whose type is the element
type). Throws an `ArgumentError` unless `guards(grid) ≤ layer.cells` (so that every guard row is
a layer row and the damping visits it with its interior image's factor) and `2·cells < n₂`
(interior rows remain between the two layers).
"""
function LayerFactors(l::AbsorbingLayer, grid::PeriodicGrid{2}, dt::T; backend, workgroupsize = nothing) where {T}
    n, h, g = ncells(grid), cellsize(grid), guards(grid)
    g <= l.cells || throw(ArgumentError("the absorbing layer of $(l.cells) cells must be at least as thick as the $g guard cells"))
    2 * l.cells < n[2] ||
        throw(ArgumentError("two absorbing layers of $(l.cells) cells leave no interior rows of the $(n[2]) cells along axis 2"))
    fac = ntuple(_ -> KernelAbstractions.allocate(backend, T, n[2] + 2g), 4)
    for (f, s, τ) in zip(fac, (zero(T), one(T) / 2, zero(T), one(T) / 2), (dt / 2, dt / 2, dt, dt))
        launcher(layer_factor_kernel!, backend, workgroupsize)(f, T(l.strength), l.order, s, τ, T(h[2]), Int32(n[2]),
                                                             Int32(g), Int32(l.cells); ndrange = length(f))
    end
    return LayerFactors(fac..., l.cells, g, n, dt)
end

# Multiply the rows of the two layers (guards included) by their factors: ndrange
# (n₁ + 2g, 2(g + cells)), the first g + cells array rows and the last g + cells. Every entry is
# written by one work item, so the result does not depend on the launch shape.
@kernel function absorb_kernel!(a, @Const(fac), lo::Int32, shift::Int32)
    i, k = @index(Global, NTuple)
    j = ifelse(Int32(k) <= lo, Int32(k), Int32(k) + shift)
    @inbounds a[i, j] *= fac[j]
end

function absorb_rows!(a, fac, L::LayerFactors, wgs)
    lo = L.g + L.cells
    shift = L.n[2] + 2L.g - 2lo
    launcher(absorb_kernel!, get_backend(a), wgs)(a, fac, Int32(lo), Int32(shift); ndrange = (size(a, 1), 2lo))
    return a
end

# The layer's step after each Yee sub-step: B after a half step, E after the full step, each
# component with the factor of its own stagger along y (field_layout(Val(2))): Ex, Ez, By on Node()
# rows, Ey, Bx, Bz on Center() rows. With no layer nothing is done.
absorb_b!(::Nothing, f, wgs) = nothing
absorb_e!(::Nothing, f, wgs) = nothing
function absorb_b!(L::LayerFactors, f, wgs)
    absorb_rows!(f.Bx, L.center_half, L, wgs)
    absorb_rows!(f.By, L.node_half, L, wgs)
    absorb_rows!(f.Bz, L.center_half, L, wgs)
    return nothing
end
function absorb_e!(L::LayerFactors, f, wgs)
    absorb_rows!(f.Ex, L.node_full, L, wgs)
    absorb_rows!(f.Ey, L.center_full, L, wgs)
    absorb_rows!(f.Ez, L.node_full, L, wgs)
    return nothing
end

"""
    solve_fields!(::Yee, f, grid::PeriodicGrid{2}, dt, antennas::Tuple, t, layer; workgroupsize = nothing)

The Yee step with the antennas' corrections ([`solve_fields!`](@ref) with antennas) and the
absorbing layer (a [`LayerFactors`](@ref), or `nothing`): after each sub-step and its antenna
correction, the components it changed are multiplied by exp(−σ·τ) in the layer rows, τ the
sub-step's length. With `layer = nothing` it is the step with antennas, bit for bit. The damping
writes the guard rows with their interior images' factors, so every guard stays filled.
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid{2}, dt, antennas::Tuple, t, layer; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    yee_b_half!(f, grid, h; workgroupsize)
    antenna_b!(antennas, f, h, t, workgroupsize)
    absorb_b!(layer, f, workgroupsize)
    yee_e!(f, grid, dt; workgroupsize)
    antenna_e!(antennas, f, dt, t + h, workgroupsize)
    absorb_e!(layer, f, workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    antenna_b!(antennas, f, h, t + T(dt), workgroupsize)
    absorb_b!(layer, f, workgroupsize)
    return f
end

# Particles that left the box across the open axis 2 this step are retired: below the Courant
# limit a particle moves less than one cell per axis per step, so its cell along axis 2 changes by
# at most one, except across the periodic seam, where it jumps by n₂ − 1 ≥ 2 (deposit_flux! needs
# n₂ ≥ 3). After the current of that last move is deposited, the slot is marked free; it keeps
# its data, every kernel skips it, and the next sort moves it behind the alive particles.
@kernel function retire_kernel!(s)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE && abs(s.cell[2][i] - s.extras.old_cell_2[i]) > Int32(1)
        s.status[i] = ParticleMeshCore.FREE
    end
end

retire!(::Nothing, st) = nothing
function retire!(::LayerFactors, st)
    wgs = st.numerics.workgroupsize
    for s in st.species
        launcher(retire_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
    end
    return nothing
end
