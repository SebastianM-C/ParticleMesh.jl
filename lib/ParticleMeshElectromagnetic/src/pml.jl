# The convolutional perfectly matched layer of the open transverse axes (WP-034 at N = 2, WP-072 at N = 3; A2: "PML
# later"): the complex-frequency-shifted stretched-coordinate PML in the recursive-convolution form
# of Roden and Gedney (2000), with PIConGPU 0.8.0's coefficients and time rule
# (`fields/absorber/pml/Pml.kernel`).
# Along the open axis y the derivative ∂_y becomes (1/s_y)∂_y with s_y = κ + σ/(α + iω), which in
# the time domain is (1/κ)∂_y F + ψ, ψ the convolution of ∂_y F with the kernel
# −σ/κ²·exp(−(σ/κ + α)t). Over a sub-step of length τ with ∂_y F held, ψ advances exactly by
#     ψ ← b·ψ + c·D_y F,   b = exp(−(σ/κ + α)τ),   c = σ(b − 1)/(κ(σ + κα))
# (c = 0 where σ = 0). Four terms of the 2D curls carry a y-difference: Bx from D⁺_y Ez, Bz from
# D⁺_y Ex (on B's Center() rows) and Ex from D⁻_y Bz, Ez from D⁻_y Bx (on E's Node() rows). Each
# has its convolution field ψ on the layer rows only, and each ψ advances once per step with
# τ = Δt: the E fields' in the E step, the B fields' in the second B half step, from E^{n+1}. The
# first B half step of the next step reads the same E^{n+1}, so it reuses that ψ without advancing
# it. This is PIConGPU's rule (its `updatePsiB` is set in one of the two B half steps only); it
# keeps ψ_B a function of E at integer times, and the half steps' B sum to one Δt-centred update.
#
# The update is a correction after the plain Yee sub-step, which already added the vacuum term
# ±τ·D_y F: in the layer rows X ← X ± τ·((1/κ − 1)·D_y F + ψ), with ψ advanced first where it
# advances (τ = Δt/2 for the B half steps, Δt for the E step), the sign that
# of the curl term (Bx −, Bz +, Ex +, Ez −). The Yee kernels, the antennas and the interior are
# untouched; only the roundoff order differs from a fused kernel. Off the layer κ = 1, σ = α = 0, so
# the correction is exactly zero.
#
# Where the layer sits is WP-027's: the outer `cells` rows on each side of the box along y, which
# meet at the periodic seam as one slab of 2·cells rows, a wave crossing it is damped twice before
# it comes back from the other side. The profiles are functions of the relative depth
# r = d/(2·cells), d = 2·cells − min(p, 2n₂ − p) the depth in half rows of the half row p (WP-027's
# integer convention, so the inner edge has r = 0 exactly in every element type):
#     σ = σ_max·r^m,   κ = 1 + (κ_max − 1)·r^{m_κ},   α = α_max·(1 − r)^{m_α} where r > 0, else 0,
# PIConGPU's grading with an order of its own for κ, so that Smilei's default profiles
# (σ = 20u², κ = 1 + 79u⁴, α = 0) are expressible too. The current enters the E step as anywhere
# else; it is not damped in the layer.

"""
    PML(; cells, sigma_max = nothing, order = 4, kappa_max = 1, kappa_order = order, alpha_max = 0, alpha_order = 1)

The convolutional perfectly matched layer of the open transverse boundary (WP-034, A2), a
numerics choice of [`EMNumerics`](@ref) (`absorber`) next to [`AbsorbingLayer`](@ref): `cells`
rows on each side of the box along axis 2 (inside the box, meeting at the periodic seam), with the
conductivity σ = sigma_max·r^order, the stretching κ = 1 + (kappa_max − 1)·r^kappa_order and the
frequency shift α = alpha_max·(1 − r)^alpha_order (zero off the layer), r the relative depth into
the layer (0 at its inner edge, 1 at the box edge). σ and α are rates in ω₀ (σ/ε₀, A1).

`sigma_max = nothing` takes PIConGPU's optimum 0.8·(order + 1)/Δy (Taflove and Hagness, eq.
7.66, in normalized units), fixed when the state places the layer on its grid. The defaults are
PIConGPU 0.8.0's (order 4, κ_max = 1), except α: PIConGPU's 0.2 S/m is about 1·10⁻⁵ ω₀ at
λ₀ = 0.8 µm, so the default here is zero. Smilei's default profiles over its layer are
`PML(; cells, sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4)` up to its depth
convention. The current deposited in the layer is not damped: with plasma inside it, α = 0 lets
slow currents build up a field there, and a positive `alpha_max` (0.2 to 1) keeps it about two
orders of magnitude smaller (questions/wp-034-pml.md §5). These are numerics, not differentiable
parameters (A3). The layer must be at least as thick as the grid's guards and leave interior rows
between the two layers (checked by the state). The same `PML` on a moving window's back edge,
along x, is `EMNumerics(; back_absorber)` ([`BackPMLFields`](@ref)).
"""
struct PML
    cells::Int
    sigma_max::Union{Nothing,Float64}
    order::Int
    kappa_max::Float64
    kappa_order::Int
    alpha_max::Float64
    alpha_order::Int
    function PML(cells::Integer, sigma_max, order::Integer, kappa_max::Real, kappa_order::Integer, alpha_max::Real,
                 alpha_order::Integer)
        cells >= 1 || throw(ArgumentError("a PML needs at least one cell, got $cells"))
        sigma_max === nothing || (sigma_max isa Real && sigma_max >= 0 && isfinite(sigma_max)) ||
            throw(ArgumentError("sigma_max must be nothing or finite and ≥ 0, got $sigma_max"))
        kappa_max >= 1 && isfinite(kappa_max) || throw(ArgumentError("kappa_max must be finite and ≥ 1, got $kappa_max"))
        alpha_max >= 0 && isfinite(alpha_max) || throw(ArgumentError("alpha_max must be finite and ≥ 0, got $alpha_max"))
        order >= 0 && kappa_order >= 0 && alpha_order >= 0 ||
            throw(ArgumentError("the grading orders must be ≥ 0, got $order, $kappa_order, $alpha_order"))
        sm = sigma_max === nothing ? nothing : Float64(sigma_max)
        return new(Int(cells), sm, Int(order), Float64(kappa_max), Int(kappa_order), Float64(alpha_max), Int(alpha_order))
    end
end
PML(; cells, sigma_max = nothing, order = 4, kappa_max = 1, kappa_order = order, alpha_max = 0, alpha_order = 1) =
    PML(cells, sigma_max, order, kappa_max, kappa_order, alpha_max, alpha_order)

"""
    pml_sigma_max(pml, dy) -> σ_max

The layer's σ_max on a grid of row spacing `dy`: `pml.sigma_max`, or PIConGPU's optimum
`0.8·(order + 1)/dy` when it is `nothing`, in the type of `dy`.
"""
pml_sigma_max(l::PML, dy::T) where {T} = l.sigma_max === nothing ? 4 * T(l.order + 1) / (5 * dy) : T(l.sigma_max)

"""
    pml_profile(sm, km, am, orders, p, n2, cells) -> (σ, κ, α)

The PML's σ, κ and α at the half row `p` (height p·Δy/2) of a box of `n2` rows with layers of
`cells` rows, from the maxima `sm`, `km`, `am` (element type `T`) and the grading `orders`
(σ, κ, α): the relative depth r = d/(2·cells), d = 2·cells − min(p, 2·n2 − p) in integer half
rows; off the layer (d ≤ 0) σ = α = 0 and κ = 1 exactly. A device function.
"""
@inline function pml_profile(sm::T, km::T, am::T, orders::NTuple{3,Int}, p::Integer, n2::Integer, cells::Integer) where {T}
    d = 2 * cells - min(p, 2 * n2 - p)
    r = T(max(d, 0)) / T(2 * cells)
    inl = d > 0
    σ = ifelse(inl, sm * r^orders[1], zero(T))
    κ = ifelse(inl, one(T) + (km - one(T)) * r^orders[2], one(T))
    α = ifelse(inl, am * (one(T) - r)^orders[3], zero(T))
    return (σ, κ, α)
end

"""
    pml_coefficients(σ, κ, α, τ) -> (b, c, q)

The recursive-convolution coefficients of a sub-step of length `τ` at a point of the layer:
b = exp(−(σ/κ + α)τ), c = σ(b − 1)/(κ(σ + κα)) (zero where σ = 0) and q = 1/κ − 1, the
stretching's correction of the vacuum difference. A device function.
"""
@inline function pml_coefficients(σ::T, κ::T, α::T, τ) where {T}
    b = exp(-(σ / κ + α) * τ)
    den = κ * (σ + κ * α)
    c = ifelse(σ > zero(T), σ * (b - one(T)) / ifelse(den > zero(T), den, one(T)), zero(T))
    return (b, c, inv(κ) - one(T))
end

# --- one layer along one axis, at N = 2 and 3 -------------------------------------------------------
#
# A layer lies along one axis d: on an open transverse axis (d = 2, and d = 3 at N = 3) the outer
# `cells` rows on each side, meeting at the periodic seam (`TwoSided`); on a moving window's back
# edge (d = 1) its first `cells` columns (`BackSide`). Two components of each field have a curl term
# with a difference along d, the components c ≠ d: of B_c the term −h·D⁺_d E_o where d is c's first
# cyclic successor and +h·D⁺_d E_o where it is the second, o the third axis (o ≠ c, d), and of E_c
# the same with the opposite signs and D⁻_d B_o (`yee.jl`'s curls). Along y at N = 2 that is
# Bx −D⁺_y Ez, Bz +D⁺_y Ex, Ex +D⁻_y Bz, Ez −D⁻_y Bx (the header's four), along x on the back edge
# By +D⁺ₓEz, Bz −D⁺ₓEy, Ey −D⁻ₓBz, Ez +D⁻ₓBy, and along z at N = 3 Bx +D⁺_z Ey, By −D⁺_z Ex,
# Ex −D⁻_z By, Ey +D⁻_z Bx. Each term has its convolution field, of the interior's shape with
# axis d cut to the layer's rows, and the correction of the header, entry by entry: the same
# expressions at every N and on every axis, so a layer on y at N = 2 keeps its bits. Across a
# corner (both transverse axes open at N = 3, or the back edge beside an open axis) each axis
# corrects its own differences, which is the stretched-coordinate PML's corner: s_y stretches ∂_y
# and s_z stretches ∂_z, independently.

"How a layer sits along its axis: on both sides of an open axis' periodic seam."
struct TwoSided end
"How a layer sits along its axis: on the first columns only, a moving window's back edge."
struct BackSide end

# The interior row j₀ (0-based) of layer row k: the first `cells` rows and the last `cells` on an open
# axis of n rows, k − 1 on the back edge. The number of layer rows is `layer_rows`.
@inline pml_row(k::Int32, n2::Int32, cells::Int32) = ifelse(k <= cells, k - Int32(1), n2 - Int32(2) * cells + k - Int32(1))
@inline layer_row(::TwoSided, k::Int32, nd::Int32, cells::Int32) = pml_row(k, nd, cells)
@inline layer_row(::BackSide, k::Int32, nd::Int32, cells::Int32) = k - Int32(1)
layer_rows(::TwoSided, cells) = 2 * cells
layer_rows(::BackSide, cells) = cells

# The coefficients per layer row k of a stagger sh along the axis (0 Node(), 1 Center()) and a
# sub-step τ, the profile at the half row 2j₀ + sh of an axis of nd rows.
@kernel function pml_coef_kernel!(b, c, q, sm, km, am, orders::NTuple{3,Int}, sh::Int32, τ, nd::Int32, cells::Int32, side)
    k = Int32(@index(Global, Linear))
    j0 = layer_row(side, k, nd, cells)
    σ, κ, α = pml_profile(sm, km, am, orders, Int32(2) * j0 + sh, nd, cells)
    bb, cc, qq = pml_coefficients(σ, κ, α, τ)
    @inbounds b[k] = bb
    @inbounds c[k] = cc
    @inbounds q[k] = qq
end

"The components (c₁, c₂), ascending, whose curl terms carry a difference along axis d."
@inline pml_components(::Val{1}) = (Val(2), Val(3))
@inline pml_components(::Val{2}) = (Val(1), Val(3))
@inline pml_components(::Val{3}) = (Val(1), Val(2))
# The third axis o of component c and axis d, and whether B_c's term along d has the minus sign
# (d is c's first cyclic successor; E_c's term then has the plus sign).
@inline other_axis(::Val{c}, ::Val{d}) where {c,d} = 6 - c - d
@inline b_minus(::Val{c}, ::Val{d}) where {c,d} = mod1(c + 1, 3) == d
const AXIS_LETTERS = (:x, :y, :z)
# The names of the four convolution fields of a layer along d: B of c₁ and c₂, then E of c₁ and c₂.
psi_names(d::Integer) = (cs = filter(!=(d), (1, 2, 3)); Tuple(Symbol(F, AXIS_LETTERS[c]) for F in (:B, :E) for c in cs))

# Place the profile of `l` along axis d of `grid` for the step `dt` on the side's rows: the
# coefficient tables per stagger (node, center), σ_max, and the four convolution fields of the
# interior's shape with axis d cut to the layer's rows, zero, with a scratch of their size.
function place_layer(l::PML, grid::PeriodicGrid{N}, dt::T, d::Integer, side; backend, workgroupsize = nothing) where {N,T}
    n, h = ncells(grid), cellsize(grid)
    sm = pml_sigma_max(l, T(h[d]))
    orders = (l.order, l.kappa_order, l.alpha_order)
    rows = layer_rows(side, l.cells)
    sets = map((0, 1)) do sh
        v = ntuple(_ -> KernelAbstractions.allocate(backend, T, rows), 3)
        launcher(pml_coef_kernel!, backend, workgroupsize)(v..., sm, T(l.kappa_max), T(l.alpha_max), orders, Int32(sh), dt, Int32(n[d]),
                                                           Int32(l.cells), side; ndrange = rows)
        v
    end
    dims = ntuple(a -> a == d ? rows : n[a], Val(N))
    ψ = NamedTuple{psi_names(d)}(ntuple(_ -> KernelAbstractions.zeros(backend, T, dims), 4))
    return sets, ψ, KernelAbstractions.zeros(backend, T, dims), sm
end

"""
    PMLFields{D}

The PML of an open transverse axis `D` placed on a grid by the state (axis 2 at N = 2; axes 2 and
3 at N = 3, one `PMLFields` each): per stagger along the axis (`node` for E's differences,
`center` for B's) the coefficients b, c (for one step, τ = Δt) and q = 1/κ − 1 per layer row, and
`ψ`, the convolution fields of the four terms with a difference along `D`, named by their
component (along y: `ψ.Bx`, `ψ.Bz` on B's Center() rows, `ψ.Ex`, `ψ.Ez` on E's Node() rows; along
z: `ψ.Bx`, `ψ.By`, `ψ.Ex`, `ψ.Ey`), each of the interior's shape with axis `D` cut to its 2·cells
layer rows (the first `cells` interior rows and the last `cells`), zero at t = 0, and a scratch
array of their size for the moving window's shift. `n` are the interior cells, `dt` the time step
the coefficients were made for, `sigma_max` the σ_max they use. A [`FieldModifier`](@ref): a
correction after each sub-step, with no parameter half.
"""
struct PMLFields{D,V,P<:NamedTuple,M,N,T} <: FieldModifier
    node::NTuple{3,V}
    center::NTuple{3,V}
    ψ::P
    scratch::M
    cells::Int
    g::Int
    n::NTuple{N,Int}
    dt::T
    sigma_max::T
end
PMLFields{D}(node::NTuple{3,V}, center, ψ::P, scratch::M, cells, g, n::NTuple{N,Int}, dt::T, sm) where {D,V,P,M,N,T} =
    PMLFields{D,V,P,M,N,T}(node, center, ψ, scratch, cells, g, n, dt, sm)

"""
    PMLFields(pml::PML, grid::PeriodicGrid, dt; axis = 2, backend, workgroupsize = nothing)

Place `pml` on the open `axis` of `grid` (N = 2 or 3) for the step `dt` (whose type is the element
type): write the coefficients once and allocate the convolution fields, zero. Throws an
`ArgumentError` unless `guards(grid) ≤ pml.cells` and `2·cells < n_axis` (interior rows remain
between the two layers).
"""
place_absorber(l::PML, grid, dt; axis::Integer = 2, backend, workgroupsize = nothing) = PMLFields(l, grid, dt; axis, backend, workgroupsize)

function PMLFields(l::PML, grid::PeriodicGrid{N}, dt::T; axis::Integer = 2, backend, workgroupsize = nothing) where {N,T}
    n, g = ncells(grid), guards(grid)
    2 <= axis <= N || throw(ArgumentError("an open axis' PML lies on axis 2 or 3 of the grid, got axis $axis at N = $N"))
    g <= l.cells || throw(ArgumentError("the PML of $(l.cells) cells must be at least as thick as the $g guard cells"))
    2 * l.cells < n[axis] ||
        throw(ArgumentError("two PMLs of $(l.cells) cells leave no interior rows of the $(n[axis]) cells along axis $axis"))
    sets, ψ, scratch, sm = place_layer(l, grid, dt, axis, TwoSided(); backend, workgroupsize)
    return PMLFields{axis}(sets..., ψ, scratch, l.cells, g, n, dt, sm)
end

"""
    BackPMLFields

The PML of the moving window's back edge placed on a grid (a state's `back_absorber`), along axis
1 on its first `cells` columns: per stagger along x (`node` for E's x-differences, `center` for
B's) the coefficients b, c (τ = Δt) and q = 1/κ − 1 per layer column, the convolution fields
`ψ.By`, `ψ.Bz` (Center() columns) and `ψ.Ey`, `ψ.Ez` (Node() columns), each of the interior's
shape with axis 1 cut to the `cells` layer columns, zero at t = 0, and a scratch of their size for
the shift. `n`, `dt` and `sigma_max` as in [`PMLFields`](@ref) (σ_max from Δx₁). A
[`FieldModifier`](@ref) of a moving-window state, after the open axes' PMLs in its tuple.
"""
struct BackPMLFields{V,P<:NamedTuple,M,N,T} <: FieldModifier
    node::NTuple{3,V}
    center::NTuple{3,V}
    ψ::P
    scratch::M
    cells::Int
    g::Int
    n::NTuple{N,Int}
    dt::T
    sigma_max::T
end

"""
    BackPMLFields(pml::PML, grid::PeriodicGrid, dt; backend, workgroupsize = nothing)

Place `pml` on the back `pml.cells` columns of `grid` (N = 2 or 3) for the step `dt`. Throws an
`ArgumentError` unless `guards(grid) ≤ pml.cells` and `2·cells < n₁` (the profile evaluated at the
back reads no row of the front).
"""
function BackPMLFields(l::PML, grid::PeriodicGrid{N}, dt::T; backend, workgroupsize = nothing) where {N,T}
    n, g = ncells(grid), guards(grid)
    g <= l.cells || throw(ArgumentError("the back-edge PML of $(l.cells) cells must be at least as thick as the $g guard cells"))
    2 * l.cells < n[1] || throw(ArgumentError("a back-edge PML of $(l.cells) cells needs more than $(2 * l.cells) window columns, got $(n[1])"))
    sets, ψ, scratch, sm = place_layer(l, grid, dt, 1, BackSide(); backend, workgroupsize)
    return BackPMLFields(sets..., ψ, scratch, l.cells, g, n, dt, sm)
end

"Any placed PML: of an open axis or of the window's back edge."
const PlacedPML = Union{PMLFields,BackPMLFields}
pml_axis(::PMLFields{D}) where {D} = Val(D)
pml_axis(::BackPMLFields) = Val(1)
pml_side(::PMLFields) = TwoSided()
pml_side(::BackPMLFields) = BackSide()

# One correction: one work item per layer entry K of X (ψ's index). Along the axis d the layer row
# k = K[d] is the interior row j₀ = layer_row(k) (array index g + 1 + j₀); along the others the
# array index is g + K[a]. D the difference of F along d at X's stagger (forward from Node() for B,
# `fwd = true`; backward from Center() for E), then ψ ← bψ + c·D where `adv` (else ψ is read as it
# is) and X ← X + s·τ·(q·D + ψ), s the curl term's sign.
@kernel function pml_correct_kernel!(X, ψ, @Const(F), @Const(b), @Const(c), @Const(q), sτ, rd, fwd::Bool, adv::Bool, g::Int32, nd::Int32,
                                     cells::Int32, vd::Val{d}, side) where {d}
    K = @index(Global, Cartesian)
    vN = Val(length(K))
    k = Int32(K[d])
    jd = g + Int32(1) + layer_row(side, k, nd, cells)
    I = CartesianIndex(ntuple(a -> a == d ? jd : Int32(K[a]) + g, vN))
    e = unit_step(vd, vN)
    @inbounds begin
        D = ifelse(fwd, F[I + e] - F[I], F[I] - F[I - e]) * rd
        p = ifelse(adv, b[k] * ψ[K] + c[k] * D, ψ[K])
        ψ[K] = p
        X[I] += sτ * (q[k] * D + p)
    end
end

function pml_correct!(X, ψ, F, coef, sτ, rd, fwd, adv, L::PlacedPML, grid, wgs)
    vd = pml_axis(L)
    launcher(pml_correct_kernel!, get_backend(X), wgs)(X, ψ, F, coef..., sτ, rd, fwd, adv, Int32(L.g), Int32(L.n[axis_of(vd)]),
                                                         Int32(L.cells), vd, pml_side(L); ndrange = size(ψ))
    fill_guards!(X, grid; workgroupsize = wgs)
    return X
end
axis_of(::Val{d}) where {d} = d

# After a B half step of length h: B_c ± h(q·D⁺_d E_o + ψ) for the two components c of the axis
# (in ascending order, on their Center() rows along d), the ψ advanced (`adv`) in the second half
# step only; after the E step of length Δt: E_c ∓ Δt(q·D⁻_d B_o + ψ) on their Node() rows. The
# guards of E are filled before the B corrections read them and vice versa.
function pml_b!(L::PlacedPML, f, grid, h, adv, wgs)
    T = eltype(f.Bx)
    vd = pml_axis(L)
    rd = inv(cellsize(grid)[axis_of(vd)])
    B, E = (f.Bx, f.By, f.Bz), (f.Ex, f.Ey, f.Ez)
    v1, v2 = pml_components(vd)
    sτ(vc) = b_minus(vc, vd) ? -T(h) : T(h)
    pml_correct!(B[axis_of(v1)], L.ψ[1], E[other_axis(v1, vd)], L.center, sτ(v1), rd, true, adv, L, grid, wgs)
    pml_correct!(B[axis_of(v2)], L.ψ[2], E[other_axis(v2, vd)], L.center, sτ(v2), rd, true, adv, L, grid, wgs)
    return nothing
end
function pml_e!(L::PlacedPML, f, grid, dt, wgs)
    T = eltype(f.Ex)
    vd = pml_axis(L)
    rd = inv(cellsize(grid)[axis_of(vd)])
    B, E = (f.Bx, f.By, f.Bz), (f.Ex, f.Ey, f.Ez)
    v1, v2 = pml_components(vd)
    sτ(vc) = b_minus(vc, vd) ? T(dt) : -T(dt)
    pml_correct!(E[axis_of(v1)], L.ψ[3], B[other_axis(v1, vd)], L.node, sτ(v1), rd, false, true, L, grid, wgs)
    pml_correct!(E[axis_of(v2)], L.ψ[4], B[other_axis(v2, vd)], L.node, sτ(v2), rd, false, true, L, grid, wgs)
    return nothing
end
back_b!(L::BackPMLFields, f, grid, h, adv, wgs) = pml_b!(L, f, grid, h, adv, wgs)
back_e!(L::BackPMLFields, f, grid, dt, wgs) = pml_e!(L, f, grid, dt, wgs)

# The PML as a field modifier: after each Yee sub-step, its correction of the four terms with a
# difference along its axis in the layer rows (their convolution fields advanced first, B's in the
# second half step only). With a zero profile (σ = α = 0, κ = 1) the corrections add zero, so the
# solve is the one without it bit for bit up to the sign of zero (T-544).
modifier_first(L::PMLFields) = (L,)
after_substep!(L::PlacedPML, mp, f, grid, s::BHalf, wgs) = pml_b!(L, f, grid, s.τ, s.second, wgs)
after_substep!(L::PlacedPML, mp, f, grid, s::EStep, wgs) = pml_e!(L, f, grid, s.τ, wgs)

# The moving window's shift (WP-028) moves E and B one column back; the convolution fields belong
# to the same points and move with them: ψ[i, …] ← ψ[i + 1, …] on ψ's columns 1 … m₁ − 1, zero on
# the last (an open axis' layer spans the window's columns, so its last is the front, with zero
# fields ahead of the window; the back edge's last layer column takes a field point from outside the
# layer, where nothing was convolved). One work item per entry.
@kernel function pml_shift_kernel!(ψ, @Const(src), m1::Int32)
    I = @index(Global, Cartesian)
    i32 = Int32(I[1])
    rest = Base.tail(Tuple(I))
    @inbounds ψ[I] = ifelse(i32 < m1, src[CartesianIndex(min(i32 + Int32(1), m1), rest...)], zero(eltype(ψ)))
end
function shift_absorber!(L::PlacedPML, wgs)
    for ψ in values(L.ψ)
        copyto!(L.scratch, ψ)
        launcher(pml_shift_kernel!, get_backend(ψ), wgs)(ψ, L.scratch, Int32(size(ψ, 1)); ndrange = size(ψ))
    end
    return nothing
end

# The convolution fields are field state, evolved by every step and part of the state's evolved
# arrays (`evolved`, the harness's partition). A run starts them at zero, with the fields:
# `initialize!` zeroes them (so step 1's first B half step reads ψ_B = 0, not c·D⁺ E⁰).
evolved_arrays(L::PlacedPML) = values(L.ψ)
on_arrays(L::PMLFields{D}, ψ::NTuple{4,Any}) where {D} =
    PMLFields{D}(L.node, L.center, NamedTuple{keys(L.ψ)}(ψ), L.scratch, L.cells, L.g, L.n, L.dt, L.sigma_max)
on_arrays(L::BackPMLFields, ψ::NTuple{4,Any}) =
    BackPMLFields(L.node, L.center, NamedTuple{keys(L.ψ)}(ψ), L.scratch, L.cells, L.g, L.n, L.dt, L.sigma_max)

"The placed absorbers an open axis can have."
const OpenAbsorber = Union{LayerFactors,PMLFields}
"The placed absorber of the open axis `d`."
const OpenAbsorberOf{d} = Union{LayerFactors{d},PMLFields{d}}
"The open axis of a placed absorber."
absorber_axis(::Union{LayerFactors{d},PMLFields{d}}) where {d} = d

"""
    reset_convolution!(state) -> state

Zero the evolved arrays of the state's field modifiers: the convolution fields of its PMLs (the
open axes' and the window's back edge's), and those of any other modifier with state
(`reset_absorber!`); nothing without one, and nothing at N = 1. `initialize!` calls it, with the fields it zeroes; no
step does (the fields are part of what a run evolves, [`evolved`](@ref)), so whoever sets the
fields of a state that has run by hand, instead of `initialize!` or [`restore!`](@ref), calls it
too.
"""
reset_convolution!(st::EMState) = (reset_absorbers!(st.modifiers); st)
@generated function reset_absorbers!(ms::NTuple{M,FieldModifier}) where {M}
    return quote
        Base.Cartesian.@nexprs $M i -> reset_absorber!(ms[i])
        return nothing
    end
end
"The columns the back-edge PML takes (0 without one)."
back_cells(::Nothing) = 0
back_cells(L::BackPMLFields) = L.cells
