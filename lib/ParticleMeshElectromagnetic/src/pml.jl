# The convolutional perfectly matched layer of the open transverse axis at N = 2 (WP-034, A2: "PML
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

# The coefficients per layer row k (1 … 2·cells: interior rows j₀ = k − 1 for k ≤ cells, else
# n₂ − 2·cells + k − 1) of a stagger sh along y (0 Node(), 1 Center()) and a sub-step τ.
@inline pml_row(k::Int32, n2::Int32, cells::Int32) = ifelse(k <= cells, k - Int32(1), n2 - Int32(2) * cells + k - Int32(1))

@kernel function pml_coef_kernel!(b, c, q, sm, km, am, orders::NTuple{3,Int}, sh::Int32, τ, n2::Int32, cells::Int32)
    k = Int32(@index(Global, Linear))
    j0 = pml_row(k, n2, cells)
    σ, κ, α = pml_profile(sm, km, am, orders, Int32(2) * j0 + sh, n2, cells)
    bb, cc, qq = pml_coefficients(σ, κ, α, τ)
    @inbounds b[k] = bb
    @inbounds c[k] = cc
    @inbounds q[k] = qq
end

"""
    PMLFields

The PML placed on a grid by the state: per stagger along y (`node` for E's y-differences,
`center` for B's) the coefficients b, c (for one step, τ = Δt) and q = 1/κ − 1 per layer row,
and the convolution fields ψ of the four y-difference terms (`ψBx`,
`ψBz` on Center() rows, `ψEx`, `ψEz` on Node() rows), each of n₁ columns by 2·cells layer rows
(the first `cells` interior rows and the last `cells`), zero at t = 0, and a scratch array of
their size for the moving window's shift. `dt` is the time step the coefficients were made for,
`sigma_max` the σ_max they use. A [`FieldModifier`](@ref): a correction after each sub-step, with
no parameter half.
"""
struct PMLFields{V,M,T} <: FieldModifier
    node::NTuple{3,V}
    center::NTuple{3,V}
    ψBx::M
    ψBz::M
    ψEx::M
    ψEz::M
    scratch::M
    cells::Int
    g::Int
    n::NTuple{2,Int}
    dt::T
    sigma_max::T
end

"""
    PMLFields(pml::PML, grid::PeriodicGrid{2}, dt; backend, workgroupsize = nothing)

Place `pml` on `grid` for the step `dt` (whose type is the element type): write the coefficients
once and allocate the convolution fields, zero. Throws an `ArgumentError` unless
`guards(grid) ≤ pml.cells` and `2·cells < n₂` (interior rows remain between the two layers).
"""
place_absorber(l::PML, grid, dt; backend, workgroupsize = nothing) = PMLFields(l, grid, dt; backend, workgroupsize)

function PMLFields(l::PML, grid::PeriodicGrid{2}, dt::T; backend, workgroupsize = nothing) where {T}
    n, h, g = ncells(grid), cellsize(grid), guards(grid)
    g <= l.cells || throw(ArgumentError("the PML of $(l.cells) cells must be at least as thick as the $g guard cells"))
    2 * l.cells < n[2] || throw(ArgumentError("two PMLs of $(l.cells) cells leave no interior rows of the $(n[2]) cells along axis 2"))
    sm = pml_sigma_max(l, T(h[2]))
    orders = (l.order, l.kappa_order, l.alpha_order)
    rows = 2 * l.cells
    sets = map(((sh, τ),) -> begin
                   v = ntuple(_ -> KernelAbstractions.allocate(backend, T, rows), 3)
                   launcher(pml_coef_kernel!, backend, workgroupsize)(v..., sm, T(l.kappa_max), T(l.alpha_max), orders, Int32(sh), τ,
                                                                      Int32(n[2]), Int32(l.cells); ndrange = rows)
                   v
               end, ((0, dt), (1, dt)))
    ψ = ntuple(_ -> KernelAbstractions.zeros(backend, T, n[1], rows), 5)
    return PMLFields(sets..., ψ..., l.cells, g, n, dt, sm)
end

# One correction: ndrange (n₁, 2·cells), one work item per layer entry of X. With the layer row k
# at the interior row j₀ (array row j = g + 1 + j₀) and column i (array column g + i), the
# y-difference D of F at X's stagger (forward from Node() for B, `fwd = true`; backward from
# Center() for E), then ψ ← bψ + c·D where `adv` (else ψ is read as it is) and
# X ← X + s·τ·(q·D + ψ), s the curl term's sign.
@kernel function pml_correct_kernel!(X, ψ, @Const(F), @Const(b), @Const(c), @Const(q), sτ, rdy, fwd::Bool, adv::Bool, g::Int32,
                                     n2::Int32, cells::Int32)
    i, k = @index(Global, NTuple)
    k32 = Int32(k)
    ii = Int32(i) + g
    j = g + Int32(1) + pml_row(k32, n2, cells)
    @inbounds begin
        D = ifelse(fwd, F[ii, j + Int32(1)] - F[ii, j], F[ii, j] - F[ii, j - Int32(1)]) * rdy
        p = ifelse(adv, b[k] * ψ[i, k] + c[k] * D, ψ[i, k])
        ψ[i, k] = p
        X[ii, j] += sτ * (q[k] * D + p)
    end
end

function pml_correct!(X, ψ, F, coef, sτ, rdy, fwd, adv, L::PMLFields, grid, wgs)
    launcher(pml_correct_kernel!, get_backend(X), wgs)(X, ψ, F, coef..., sτ, rdy, fwd, adv, Int32(L.g), Int32(L.n[2]),
                                                         Int32(L.cells); ndrange = (L.n[1], 2 * L.cells))
    fill_guards!(X, grid; workgroupsize = wgs)
    return X
end

# After a B half step of length h: Bx −= h(q·D⁺_y Ez + ψBx), Bz += h(q·D⁺_y Ex + ψBz), each on its
# Center() rows, the ψ advanced (`adv`) in the second half step only; after the E step of length Δt:
# Ex += Δt(q·D⁻_y Bz + ψEx), Ez −= Δt(q·D⁻_y Bx + ψEz) on Node() rows. The guards of E are filled
# before the B corrections read them and vice versa.
function pml_b!(L::PMLFields, f, grid, h, adv, wgs)
    T = eltype(f.Bx)
    rdy = inv(cellsize(grid)[2])
    pml_correct!(f.Bx, L.ψBx, f.Ez, L.center, -T(h), rdy, true, adv, L, grid, wgs)
    pml_correct!(f.Bz, L.ψBz, f.Ex, L.center, T(h), rdy, true, adv, L, grid, wgs)
    return nothing
end
function pml_e!(L::PMLFields, f, grid, dt, wgs)
    T = eltype(f.Ex)
    rdy = inv(cellsize(grid)[2])
    pml_correct!(f.Ex, L.ψEx, f.Bz, L.node, T(dt), rdy, false, true, L, grid, wgs)
    pml_correct!(f.Ez, L.ψEz, f.Bx, L.node, -T(dt), rdy, false, true, L, grid, wgs)
    return nothing
end

# The PML as a field modifier: after each Yee sub-step, its correction of the four y-difference
# terms in the layer rows (their convolution fields advanced first, B's in the second half step
# only). With a zero profile (σ = α = 0, κ = 1) the corrections add zero, so the solve is the one
# without it bit for bit up to the sign of zero (T-544).
modifier_first(L::PMLFields) = (L,)
after_substep!(L::PMLFields, mp, f, grid, s::BHalf, wgs) = pml_b!(L, f, grid, s.τ, s.second, wgs)
after_substep!(L::PMLFields, mp, f, grid, s::EStep, wgs) = pml_e!(L, f, grid, s.τ, wgs)

# The moving window's shift (WP-028) moves E and B one column back; the convolution fields belong
# to the same points and move with them: ψ[i, k] ← ψ[i + 1, k] on the columns 1 … n₁ − 1, zero on
# the front column (zero fields ahead of the window). One work item per entry.
@kernel function pml_shift_kernel!(ψ, @Const(src), n1::Int32)
    i, k = @index(Global, NTuple)
    i32 = Int32(i)
    @inbounds ψ[i, k] = ifelse(i32 < n1, src[min(i32 + Int32(1), n1), k], zero(eltype(ψ)))
end
function shift_absorber!(L::PMLFields, wgs)
    for ψ in (L.ψBx, L.ψBz, L.ψEx, L.ψEz)
        copyto!(L.scratch, ψ)
        launcher(pml_shift_kernel!, get_backend(ψ), wgs)(ψ, L.scratch, Int32(L.n[1]); ndrange = size(ψ))
    end
    return nothing
end

retire!(::PMLFields, st) = retire_open!(st)

# The convolution fields are field state, evolved by every step and part of the state's evolved
# arrays (`evolved`, the harness's partition). A run starts them at zero, with the fields:
# `initialize!` zeroes them (so step 1's first B half step reads ψ_B = 0, not c·D⁺_y E⁰).
evolved_arrays(L::PMLFields) = (L.ψBx, L.ψBz, L.ψEx, L.ψEz)
on_arrays(L::PMLFields, ψ::NTuple{4,Any}) = PMLFields(L.node, L.center, ψ..., L.scratch, L.cells, L.g, L.n, L.dt, L.sigma_max)

# --- the moving window's back edge (WP-034 with WP-028) -------------------------------------------
#
# The same PML along x on the window's back edge: its first `cells` columns, the depth counted in
# integer half columns from the back face (half column p = 2i₀ + sh of interior column i₀, depth
# d = 2·cells − p, r = d/(2·cells)), one layer only (the front is where the undisturbed outside
# comes in). Four terms of the 2D curls carry an x-difference: By from D⁺ₓEz (+), Bz from D⁺ₓEy (−)
# on B's Center() columns, and Ey from D⁻ₓBz (−), Ez from D⁻ₓBy (+) on E's Node() columns. Each has
# its convolution field on the layer columns, advanced once per step as on the y axis. Behind the
# layer the window's x-guards stay zero (its field-free outside), so what the layer has not
# absorbed reflects there and crosses it again. The profile is fixed in the window, which moves at
# c: a shift moves E, B and the convolution fields one column back together (each ψ belongs to its
# field's point), so a point deepens by one column in the layer per shift and the last layer column
# starts from ψ = 0. The x and y corrections of Bz are independent, so the corners take both.

"""
    BackPMLFields

The PML of the moving window's back edge placed on a grid (a state's `back_absorber`): per
stagger along x (`node` for E's x-differences, `center` for B's) the coefficients b, c (τ = Δt)
and q = 1/κ − 1 per layer column, the convolution fields `ψBy`, `ψBz` (Center() columns) and
`ψEy`, `ψEz` (Node() columns), each `cells` columns by n₂ rows, zero at t = 0, and a scratch of
their size for the shift. `dt` and `sigma_max` as in [`PMLFields`](@ref) (σ_max from Δx₁). A
[`FieldModifier`](@ref) of a moving-window state, after the open axis' PML in its tuple.
"""
struct BackPMLFields{V,M,T} <: FieldModifier
    node::NTuple{3,V}
    center::NTuple{3,V}
    ψBy::M
    ψBz::M
    ψEy::M
    ψEz::M
    scratch::M
    cells::Int
    g::Int
    n::NTuple{2,Int}
    dt::T
    sigma_max::T
end

@kernel function back_coef_kernel!(b, c, q, sm, km, am, orders::NTuple{3,Int}, sh::Int32, τ, n1::Int32, cells::Int32)
    k = Int32(@index(Global, Linear))
    σ, κ, α = pml_profile(sm, km, am, orders, Int32(2) * (k - Int32(1)) + sh, n1, cells)
    bb, cc, qq = pml_coefficients(σ, κ, α, τ)
    @inbounds b[k] = bb
    @inbounds c[k] = cc
    @inbounds q[k] = qq
end

"""
    BackPMLFields(pml::PML, grid::PeriodicGrid{2}, dt; backend, workgroupsize = nothing)

Place `pml` on the back `pml.cells` columns of `grid` for the step `dt`. Throws an `ArgumentError`
unless `guards(grid) ≤ pml.cells` and `2·cells < n₁` (the profile evaluated at the back reads no
row of the front).
"""
function BackPMLFields(l::PML, grid::PeriodicGrid{2}, dt::T; backend, workgroupsize = nothing) where {T}
    n, h, g = ncells(grid), cellsize(grid), guards(grid)
    g <= l.cells || throw(ArgumentError("the back-edge PML of $(l.cells) cells must be at least as thick as the $g guard cells"))
    2 * l.cells < n[1] || throw(ArgumentError("a back-edge PML of $(l.cells) cells needs more than $(2 * l.cells) window columns, got $(n[1])"))
    sm = pml_sigma_max(l, T(h[1]))
    orders = (l.order, l.kappa_order, l.alpha_order)
    sets = map((0, 1)) do sh
        v = ntuple(_ -> KernelAbstractions.allocate(backend, T, l.cells), 3)
        launcher(back_coef_kernel!, backend, workgroupsize)(v..., sm, T(l.kappa_max), T(l.alpha_max), orders, Int32(sh), dt,
                                                            Int32(n[1]), Int32(l.cells); ndrange = l.cells)
        v
    end
    ψ = ntuple(_ -> KernelAbstractions.zeros(backend, T, l.cells, n[2]), 5)
    return BackPMLFields(sets..., ψ..., l.cells, g, n, dt, sm)
end

# One correction along x: ndrange (cells, n₂). Layer column k is the interior column i₀ = k − 1
# (array column g + k) and row j the array row g + j; D the x-difference of F at X's stagger
# (forward from Node() for B, backward from Center() for E), then ψ ← bψ + c·D where `adv` and
# X ← X + s·τ·(q·D + ψ).
@kernel function back_correct_kernel!(X, ψ, @Const(F), @Const(b), @Const(c), @Const(q), sτ, rdx, fwd::Bool, adv::Bool, g::Int32)
    k, j = @index(Global, NTuple)
    ii = g + Int32(k)
    jj = g + Int32(j)
    @inbounds begin
        D = ifelse(fwd, F[ii + Int32(1), jj] - F[ii, jj], F[ii, jj] - F[ii - Int32(1), jj]) * rdx
        p = ifelse(adv, b[k] * ψ[k, j] + c[k] * D, ψ[k, j])
        ψ[k, j] = p
        X[ii, jj] += sτ * (q[k] * D + p)
    end
end

function back_correct!(X, ψ, F, coef, sτ, rdx, fwd, adv, L::BackPMLFields, grid, wgs)
    launcher(back_correct_kernel!, get_backend(X), wgs)(X, ψ, F, coef..., sτ, rdx, fwd, adv, Int32(L.g); ndrange = (L.cells, L.n[2]))
    fill_guards!(X, grid; workgroupsize = wgs)
    return X
end

# After a B half step of length h: By += h(q·D⁺ₓEz + ψBy), Bz −= h(q·D⁺ₓEy + ψBz) on their Center()
# columns, ψ advanced in the second half step only; after the E step: Ey −= Δt(q·D⁻ₓBz + ψEy),
# Ez += Δt(q·D⁻ₓBy + ψEz) on Node() columns.
function back_b!(L::BackPMLFields, f, grid, h, adv, wgs)
    T = eltype(f.By)
    rdx = inv(cellsize(grid)[1])
    back_correct!(f.By, L.ψBy, f.Ez, L.center, T(h), rdx, true, adv, L, grid, wgs)
    back_correct!(f.Bz, L.ψBz, f.Ey, L.center, -T(h), rdx, true, adv, L, grid, wgs)
    return nothing
end
function back_e!(L::BackPMLFields, f, grid, dt, wgs)
    T = eltype(f.Ey)
    rdx = inv(cellsize(grid)[1])
    back_correct!(f.Ey, L.ψEy, f.Bz, L.node, -T(dt), rdx, false, true, L, grid, wgs)
    back_correct!(f.Ez, L.ψEz, f.By, L.node, T(dt), rdx, false, true, L, grid, wgs)
    return nothing
end

after_substep!(L::BackPMLFields, mp, f, grid, s::BHalf, wgs) = back_b!(L, f, grid, s.τ, s.second, wgs)
after_substep!(L::BackPMLFields, mp, f, grid, s::EStep, wgs) = back_e!(L, f, grid, s.τ, wgs)

# The window's shift: ψ[k, j] ← ψ[k + 1, j] on the layer columns 1 … cells − 1, zero on the last
# (its field point comes from outside the layer, where nothing was convolved).
@kernel function back_shift_kernel!(ψ, @Const(src), cells::Int32)
    k, j = @index(Global, NTuple)
    k32 = Int32(k)
    @inbounds ψ[k, j] = ifelse(k32 < cells, src[min(k32 + Int32(1), cells), j], zero(eltype(ψ)))
end
function shift_absorber!(L::BackPMLFields, wgs)
    for ψ in (L.ψBy, L.ψBz, L.ψEy, L.ψEz)
        copyto!(L.scratch, ψ)
        launcher(back_shift_kernel!, get_backend(ψ), wgs)(ψ, L.scratch, Int32(L.cells); ndrange = size(ψ))
    end
    return nothing
end
evolved_arrays(L::BackPMLFields) = (L.ψBy, L.ψBz, L.ψEy, L.ψEz)
on_arrays(L::BackPMLFields, ψ::NTuple{4,Any}) = BackPMLFields(L.node, L.center, ψ..., L.scratch, L.cells, L.g, L.n, L.dt, L.sigma_max)

"The placed absorbers an open axis can have."
const OpenAbsorber = Union{LayerFactors,PMLFields}

"""
    reset_convolution!(state) -> state

Zero the evolved arrays of the state's field modifiers: the convolution fields of its PMLs (the
open axis' and the window's back edge's), and those of any other modifier with state
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
