# Open transverse boundaries (WP-027 at N = 2, WP-072 at N = 3; A2): the problem says which
# transverse axes are open (`EMProblem(...; boundaries = (:periodic, :open))`, at N = 3 y, z or
# both), the numerics say how: a simple absorbing layer for the fields (`AbsorbingLayer`, A2: PML
# later), placed on each open axis as its own modifier (`LayerFactors{d}`), and particles retired
# through the alive mask when they leave the box across an open axis (§0.4 rule 1: no buffer
# changes size). The text below speaks of y; an open z is the same with the axes exchanged, and
# where two layers overlap (the edges of a 3D box) each damps by its own factor.
#
# The grid underneath stays the engine's `PeriodicGrid{N}`: the layer occupies the outer `cells`
# rows on each side of the box, so the two layers meet at the periodic seam y = 0 ≡ Ly and form
# one slab of 2·cells rows. A wave that leaves the interior crosses that slab before it can come
# back from the other side, and the absorbing slab damps it on the way. In the layer the
# conductivity is σ(y) = strength·(depth/thickness)^order, with depth the distance into the layer
# from its inner edge (thickness = cells·Δy, so σ = strength at the box edge and 0 at the inner
# edge), the same σ for E and for B (the matched lossy medium, whose impedance is that of vacuum,
# so a continuum wave at normal incidence enters it without reflection). The depth is computed
# in integer half rows, so a row exactly at the inner edge has depth zero in every element type.
# It is applied by symmetric (Strang) operator splitting: each Yee sub-step of length τ (Δt/2 for
# each B half step, Δt for the E step) is preceded and followed by a multiplication of the
# components it updates by exp(−σ·τ/2) at their own stagger along y, so the layer is second order
# in Δt (a damping only after the sub-step is first order: its reflection falls as Δ, not Δ²).
# The interior update is untouched, so Gauss's law and ∇·B hold there as without the layer; in the
# layer D⁻·E changes by D⁻·((e^{−σΔt/2} − 1)·…), so Gauss's law is broken there, which is what
# absorbing the field of charges that left means.

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
    layer_sigma(strength, order, p, n2, cells) -> σ

The conductivity of a layer of `cells` rows at the half row `p` (the height p·Δy/2 above y = 0,
`p` even on Node() rows and odd on Center() rows) of a box of `n2` rows:
`strength·(d/(2·cells))^order` with `d = 2·cells − min(p, 2·n2 − p)` the depth into the layer in
half rows, zero where `d ≤ 0` (outside the layer, and exactly at its inner edge for every `order`
and element type, since `d` is an integer), in the element type of `strength`. A device function.
"""
@inline function layer_sigma(strength::T, order::Int, p::Integer, n2::Integer, cells::Integer) where {T}
    d = 2 * cells - min(p, 2 * n2 - p)
    return ifelse(d > 0, strength * (T(d) / T(2 * cells))^order, zero(T))
end

# The damping factor exp(−σ·τ) per array row j along the layer's axis (guards included) for a field
# whose rows sit at the half rows 2j₀ + sh (sh = 0 for Node(), 1 for Center()), j₀ the interior row
# j aliases, on an axis of nd rows; written once by the state.
@kernel function layer_factor_kernel!(fac, strength, order::Int, sh::Int32, τ, nd::Int32, g::Int32, cells::Int32)
    j = Int32(@index(Global, Linear))
    j0 = mod(j - g - Int32(1), nd)
    σ = layer_sigma(strength, order, Int32(2) * j0 + sh, nd, cells)
    @inbounds fac[j] = exp(-σ * τ)
end

"""
    LayerFactors{D}

The absorbing layer of the open axis `D` placed on a grid by the state (axis 2 at N = 2; axes 2
and 3 at N = 3, one `LayerFactors` each): the damping factors exp(−σ·τ/2) per array row along
the axis (guards included), applied before and after a sub-step of length τ, for the rows at
`Node()` and at `Center()` along it and for τ = Δt/2 (`half`, the B half steps: factors
exp(−σΔt/4)) and τ = Δt (`full`, the E step: exp(−σΔt/2)); the number of layer rows `cells`,
the guard width and the interior cells `n`, which fix the rows the damping visits. `dt` is the
time step the factors were made for. A [`FieldModifier`](@ref): it damps the components a
sub-step updates before the sub-step and after it, and has no parameter half. Where two open
axes' layers cross (a corner at N = 3) both damp, σ_y + σ_z.
"""
struct LayerFactors{D,V,N,T} <: FieldModifier
    node_half::V
    center_half::V
    node_full::V
    center_full::V
    cells::Int
    g::Int
    n::NTuple{N,Int}
    dt::T
end
LayerFactors{D}(nh::V, ch::V, nf::V, cf::V, cells, g, n::NTuple{N,Int}, dt::T) where {D,V,N,T} = LayerFactors{D,V,N,T}(nh, ch, nf, cf, cells, g, n, dt)

"""
    LayerFactors(layer::AbsorbingLayer, grid::PeriodicGrid, dt; axis = 2, backend, workgroupsize = nothing)

Write the damping factors of `layer` on the open `axis` of `grid` (N = 2 or 3) for the step `dt`
(whose type is the element type). Throws an `ArgumentError` unless `guards(grid) ≤ layer.cells`
(so that every guard row is a layer row and the damping visits it with its interior image's
factor) and `2·cells < n_axis` (interior rows remain between the two layers).
"""
place_absorber(l::AbsorbingLayer, grid, dt; axis::Integer = 2, backend, workgroupsize = nothing) =
    LayerFactors(l, grid, dt; axis, backend, workgroupsize)

function LayerFactors(l::AbsorbingLayer, grid::PeriodicGrid{N}, dt::T; axis::Integer = 2, backend, workgroupsize = nothing) where {N,T}
    n, g = ncells(grid), guards(grid)
    2 <= axis <= N || throw(ArgumentError("an absorbing layer lies on axis 2 or 3 of the grid, got axis $axis at N = $N"))
    g <= l.cells || throw(ArgumentError("the absorbing layer of $(l.cells) cells must be at least as thick as the $g guard cells"))
    2 * l.cells < n[axis] ||
        throw(ArgumentError("two absorbing layers of $(l.cells) cells leave no interior rows of the $(n[axis]) cells along axis $axis"))
    fac = ntuple(_ -> KernelAbstractions.allocate(backend, T, n[axis] + 2g), 4)
    for (f, sh, τ) in zip(fac, Int32.((0, 1, 0, 1)), (dt / 4, dt / 4, dt / 2, dt / 2))
        launcher(layer_factor_kernel!, backend, workgroupsize)(f, T(l.strength), l.order, sh, τ, Int32(n[axis]), Int32(g),
                                                             Int32(l.cells); ndrange = length(f))
    end
    return LayerFactors{axis}(fac..., l.cells, g, n, dt)
end

# Multiply the rows of the two layers along the axis d (guards included) by their factors: ndrange
# the array's size with axis d cut to 2(g + cells), the first g + cells array rows and the last
# g + cells. Every entry is written by one work item, so the result does not depend on the launch
# shape.
@kernel function absorb_kernel!(a, @Const(fac), lo::Int32, shift::Int32, ::Val{d}) where {d}
    K = @index(Global, Cartesian)
    k = Int32(K[d])
    j = ifelse(k <= lo, k, k + shift)
    I = CartesianIndex(ntuple(a -> a == d ? Int(j) : K[a], Val(length(K))))
    @inbounds a[I] *= fac[j]
end

function absorb_rows!(a, fac, L::LayerFactors{D}, wgs) where {D}
    lo = L.g + L.cells
    shift = L.n[D] + 2L.g - 2lo
    dims = ntuple(k -> k == D ? 2lo : size(a, k), ndims(a))
    launcher(absorb_kernel!, get_backend(a), wgs)(a, fac, Int32(lo), Int32(shift), Val(D); ndrange = dims)
    return a
end

# The layer's half damping before and after each Yee sub-step: B around a half step, E around the
# full step, each component with the factor of its own stagger along the layer's axis d
# (field_layout): E_c on Center() rows for c = d and on Node() rows otherwise, B_c the other way
# round (at N = 2 along y: Ex, Ez, By on Node() rows, Ey, Bx, Bz on Center() rows). Symmetric
# splitting, second order in Δt. The damping writes the guard rows with their interior images'
# factors, so every guard stays filled.
function absorb_b!(L::LayerFactors{D}, f, wgs) where {D}
    absorb_rows!(f.Bx, D == 1 ? L.node_half : L.center_half, L, wgs)
    absorb_rows!(f.By, D == 2 ? L.node_half : L.center_half, L, wgs)
    absorb_rows!(f.Bz, D == 3 ? L.node_half : L.center_half, L, wgs)
    return nothing
end
function absorb_e!(L::LayerFactors{D}, f, wgs) where {D}
    absorb_rows!(f.Ex, D == 1 ? L.center_full : L.node_full, L, wgs)
    absorb_rows!(f.Ey, D == 2 ? L.center_full : L.node_full, L, wgs)
    absorb_rows!(f.Ez, D == 3 ? L.center_full : L.node_full, L, wgs)
    return nothing
end

modifier_late(L::LayerFactors) = (L,)
before_substep!(L::LayerFactors, mp, f, grid, ::BHalf, wgs) = absorb_b!(L, f, wgs)
after_substep!(L::LayerFactors, mp, f, grid, ::BHalf, wgs) = absorb_b!(L, f, wgs)
before_substep!(L::LayerFactors, mp, f, grid, ::EStep, wgs) = absorb_e!(L, f, wgs)
after_substep!(L::LayerFactors, mp, f, grid, ::EStep, wgs) = absorb_e!(L, f, wgs)

# Particles that left the box across an open axis this step are retired: below the Courant limit a
# particle moves less than one cell per axis per step, so its cell along an open axis d changes by
# at most one, except across the periodic seam, where it jumps by n_d − 1 ≥ 2 (deposit_flux! needs
# n_d ≥ 3). After the current of that last move is deposited, the slot is marked free; it keeps its
# data, every kernel skips it, and the next sort moves it behind the alive particles. `axes` are
# the open axes, as a tuple of `Val`s.
@inline old_cell(x, ::Val{1}) = x.old_cell_1
@inline old_cell(x, ::Val{2}) = x.old_cell_2
@inline old_cell(x, ::Val{3}) = x.old_cell_3
@inline crossed(s, i, ::Tuple{}) = false
@inline crossed(s, i, axes::Tuple) =
    (vd = first(axes); d = axis_of(vd); @inbounds abs(s.cell[d][i] - old_cell(s.extras, vd)[i]) > Int32(1)) || crossed(s, i, Base.tail(axes))

@kernel function retire_kernel!(s, axes)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE && crossed(s, i, axes)
        s.status[i] = ParticleMeshCore.FREE
    end
end

"""
    retire_open!(absorbers::Tuple, state)

Retire the particles (the species', a dopant's electrons and a witness bunch's) that crossed an
open axis' edge this step, the axes being those of the state's placed `absorbers`
([`open_absorbers`](@ref)); nothing without one.
"""
retire_open!(::Tuple{}, st) = nothing
function retire_open!(absorbers::Tuple, st)
    wgs = st.numerics.workgroupsize
    axes = map(L -> Val(absorber_axis(L)), absorbers)
    for s in (deposited_species(st)..., witness_species(st)...)
        launcher(retire_kernel!, get_backend(s), wgs)(s, axes; ndrange = capacity(s))
    end
    return nothing
end
