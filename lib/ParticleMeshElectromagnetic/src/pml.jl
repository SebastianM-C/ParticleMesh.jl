# The convolutional perfectly matched layer of the open transverse axis at N = 2 (WP-034, A2: "PML
# later"): the complex-frequency-shifted stretched-coordinate PML in the recursive-convolution form
# of Roden and Gedney (2000), as PIConGPU 0.8.0 implements it (`fields/absorber/pml/Pml.kernel`).
# Along the open axis y the derivative ∂_y becomes (1/s_y)∂_y with s_y = κ + σ/(α + iω), which in
# the time domain is (1/κ)∂_y F + ψ, ψ the convolution of ∂_y F with the kernel
# −σ/κ²·exp(−(σ/κ + α)t). Over a sub-step of length τ with ∂_y F held, ψ advances exactly by
#     ψ ← b·ψ + c·D_y F,   b = exp(−(σ/κ + α)τ),   c = σ(b − 1)/(κ(σ + κα))
# (c = 0 where σ = 0). Four terms of the 2D curls carry a y-difference: Bx from D⁺_y Ez, Bz from
# D⁺_y Ex (each B half step, τ = Δt/2) and Ex from D⁻_y Bz, Ez from D⁻_y Bx (the E step, τ = Δt).
# Each has its convolution field ψ on the layer rows only.
#
# The update is a correction after the plain Yee sub-step, which already added the vacuum term
# ±τ·D_y F: in the layer rows X ← X ± τ·((1/κ − 1)·D_y F + ψ), with ψ advanced first, the sign that
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
convention. These are numerics, not differentiable parameters (A3). The layer must be at least as
thick as the grid's guards and leave interior rows between the two layers (checked by the state).
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

The PML placed on a grid by the state: per stagger along y (`node`, `center`) and sub-step
(`half` for the B half steps, τ = Δt/2; `full` for the E step, τ = Δt) the coefficients b, c and
q = 1/κ − 1 per layer row, and the convolution fields ψ of the four y-difference terms (`ψBx`,
`ψBz` on Center() rows, `ψEx`, `ψEz` on Node() rows), each of n₁ columns by 2·cells layer rows
(the first `cells` interior rows and the last `cells`), zero at t = 0. `dt` is the time step the
coefficients were made for, `sigma_max` the σ_max they use.
"""
struct PMLFields{V,M,T}
    node_half::NTuple{3,V}
    center_half::NTuple{3,V}
    node_full::NTuple{3,V}
    center_full::NTuple{3,V}
    ψBx::M
    ψBz::M
    ψEx::M
    ψEz::M
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
               end, ((0, dt / 2), (1, dt / 2), (0, dt), (1, dt)))
    ψ = ntuple(_ -> KernelAbstractions.zeros(backend, T, n[1], rows), 4)
    return PMLFields(sets..., ψ..., l.cells, g, n, dt, sm)
end

# One correction: ndrange (n₁, 2·cells), one work item per layer entry of X. With the layer row k
# at the interior row j₀ (array row j = g + 1 + j₀) and column i (array column g + i), the
# y-difference D of F at X's stagger (forward from Node() for B, `fwd = true`; backward from
# Center() for E), then ψ ← bψ + c·D and X ← X + s·τ·(q·D + ψ), s the curl term's sign.
@kernel function pml_correct_kernel!(X, ψ, @Const(F), @Const(b), @Const(c), @Const(q), sτ, rdy, fwd::Bool, g::Int32, n2::Int32,
                                     cells::Int32)
    i, k = @index(Global, NTuple)
    k32 = Int32(k)
    ii = Int32(i) + g
    j = g + Int32(1) + pml_row(k32, n2, cells)
    @inbounds begin
        D = ifelse(fwd, F[ii, j + Int32(1)] - F[ii, j], F[ii, j] - F[ii, j - Int32(1)]) * rdy
        p = b[k] * ψ[i, k] + c[k] * D
        ψ[i, k] = p
        X[ii, j] += sτ * (q[k] * D + p)
    end
end

function pml_correct!(X, ψ, F, coef, sτ, rdy, fwd, L::PMLFields, grid, wgs)
    launcher(pml_correct_kernel!, get_backend(X), wgs)(X, ψ, F, coef..., sτ, rdy, fwd, Int32(L.g), Int32(L.n[2]), Int32(L.cells);
                                                         ndrange = (L.n[1], 2 * L.cells))
    fill_guards!(X, grid; workgroupsize = wgs)
    return X
end

# After a B half step of length h: Bx −= h(q·D⁺_y Ez + ψBx), Bz += h(q·D⁺_y Ex + ψBz), each on its
# Center() rows; after the E step of length Δt: Ex += Δt(q·D⁻_y Bz + ψEx), Ez −= Δt(q·D⁻_y Bx + ψEz)
# on Node() rows. The guards of E are filled before the B corrections read them and vice versa.
function pml_b!(L::PMLFields, f, grid, h, wgs)
    T = eltype(f.Bx)
    rdy = inv(cellsize(grid)[2])
    pml_correct!(f.Bx, L.ψBx, f.Ez, L.center_half, -T(h), rdy, true, L, grid, wgs)
    pml_correct!(f.Bz, L.ψBz, f.Ex, L.center_half, T(h), rdy, true, L, grid, wgs)
    return nothing
end
function pml_e!(L::PMLFields, f, grid, dt, wgs)
    T = eltype(f.Ex)
    rdy = inv(cellsize(grid)[2])
    pml_correct!(f.Ex, L.ψEx, f.Bz, L.node_full, T(dt), rdy, false, L, grid, wgs)
    pml_correct!(f.Ez, L.ψEz, f.Bx, L.node_full, -T(dt), rdy, false, L, grid, wgs)
    return nothing
end

"""
    solve_fields!(::Yee, f, grid::PeriodicGrid{2}, dt, antennas::Tuple, t, pml::PMLFields; workgroupsize = nothing)

The Yee step with the antennas' corrections and the convolutional PML: after each Yee sub-step,
the PML's correction of the four y-difference terms in the layer rows (their convolution fields
advanced first), then the antennas' correction. With a zero profile (σ = α = 0, κ = 1) the
corrections add zero, so the step is the antennas' solve bit for bit up to the sign of zero.
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid{2}, dt, antennas::Tuple, t, L::PMLFields; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    yee_b_half!(f, grid, h; workgroupsize)
    pml_b!(L, f, grid, h, workgroupsize)
    antenna_b!(antennas, f, h, t, workgroupsize)
    yee_e!(f, grid, dt; workgroupsize)
    pml_e!(L, f, grid, dt, workgroupsize)
    antenna_e!(antennas, f, dt, t + h, workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    pml_b!(L, f, grid, h, workgroupsize)
    antenna_b!(antennas, f, h, t + T(dt), workgroupsize)
    return f
end

retire!(::PMLFields, st) = retire_open!(st)

# The convolution fields are part of the field state but not of the state's arrays: step 1 starts
# them at zero, so every run from step 1 (`run!`, or a solve of the harness, which takes the steps
# in order on one integrator) is the same run.
reset_absorber!(::Any) = nothing
reset_absorber!(L::PMLFields) = (foreach(ψ -> fill!(ψ, zero(eltype(ψ))), (L.ψBx, L.ψBz, L.ψEx, L.ψEz)); nothing)
