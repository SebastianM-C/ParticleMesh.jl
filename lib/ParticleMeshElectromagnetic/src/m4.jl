# The M4 solver (WP-079): Yee's leapfrog and Ampère's law, with Faraday's law taking a wider
# difference that cancels the scheme's dispersion to fourth order (Lu, Kilian, Guo, Li, Liang,
# J. Comput. Phys. 413, 109388 (2020), section 4 and appendix C, of the family of Blinne et al.,
# Comput. Phys. Commun. 224, 273 (2018); the coefficients are Smilei's MF_Solver2D_M4 and
# MF_Solver3D_M4). Along a resolved axis d the B update's forward difference D⁺_d F becomes
#
#     D̃_d F = D⁺_d F + δ_d·Δ³_d F/Δ_d + Σ_{j ≠ d} β_dj·Δ²_j D⁺_d F,
#     δ_d = ((Δt/Δ_d)² − 1)/12,   β_dj = (Δt/Δ_j)²/12,
#
# Δ³_d F = F[i+2] − 3F[i+1] + 3F[i] − F[i−1] the third difference along d (the reach of two cells
# ahead and one behind) and Δ²_j the second difference along the resolved axis j ≠ d (the
# transverse average of D⁺_d over the rows j ± 1). In the usual notation D̃_d is
# α_d(F[i+1] − F[i]) + δ_d(F[i+2] − F[i−1]) + Σ_j β_dj(the same difference on the rows j ± 1), with
# α_d = 1 − 2Σ_j β_dj − 3δ_d; it is written as D⁺ plus small corrections here so that the
# corrections, of order (kΔ)², carry their own roundoff. On a mode e^{ik·x} D̃_d is D⁺_d times
#
#     A_d = 1 − 2δ_d(1 − cos k_dΔ_d) − Σ_{j ≠ d} 2β_dj(1 − cos k_jΔ_j),
#
# so the scheme's exact discrete relation is sin²(ωΔt/2)/Δt² = Σ_d A_d·sin²(k_dΔ_d/2)/Δ_d², which
# is ω = |k| to fourth order in kΔ in every direction (δ_d cancels the k_d⁴ terms, and β_dj with
# β_jd the cross terms k_d²k_j², since β_djΔ_j² + β_jdΔ_d² = Δt²/6; Yee's is second order) at every Δt.
#
# Two consequences, both tested (T-1082, T-1083):
# - ∇·B. The differences D̃_d along different axes commute (shift-invariant operators), so the
#   divergence taken with them, Σ_c D̃_c B_c on B's cell centres, is kept by the B update to
#   roundoff: Σ_c D̃_c(∇̃×E)_c = 0 term by term. Yee's ∇·B, Σ_c D⁺_c B_c, is not: on a mode its
#   change has the factor A_{c₂} − A_{c₁} = ((1 − cos k_{c₂}Δ_{c₂}) − (1 − cos k_{c₁}Δ_{c₁}))/6 per
#   pair of axes, of order (kΔ)². Gauss's law is Ampère's and does not change: ∇·E − ρ is kept as
#   with Yee.
# - The Courant limit. With λ_d = Δt²/Δ_d², x_d = sin²(k_dΔ_d/2) ∈ [0, 1] and P = Σ λ_d x_d, the
#   relation's right side times Δt²/4 is P − P²/3 + Σ λ_d x_d²/3 ≤ (4P − P²)/3, which is below 1
#   for P < 1 and reaches 1 only at the corner mode x_d = 1 with Σ λ_d = 1. So M4 is stable exactly
#   where Yee is, Δt < 1/√(Σ 1/Δ_d²), at every N and aspect ratio, and A_d ≥ 2/3 there (D̃ never
#   changes sign). Above the limit the corner mode grows by (μ + √(μ² − 1))² per step,
#   μ² = S²(4 − S²)/3, S = Δt/Δt_Yee.
#
# The corrections reach two cells ahead and one behind along every resolved axis, so the grid
# needs two guards ([`em_guard_width`](@ref)). Everything else of the step is Yee's: the E
# update, the current, the staggers, the stored B as the mean of the half steps. The coefficients
# depend on the full Δt, not on the length of the half step, so the solve makes them once per step.

"""
    M4()

The M4 low-dispersion solver in the mesh-solve slot (WP-079, A10), for N = 1, 2 and 3: the
leapfrog and the E update of [`Yee`](@ref), with Faraday's law taking the difference of Lu et al.
(2020) along each resolved axis, whose coefficients follow the time step (no parameter):
δ_d = ((Δt/Δ_d)² − 1)/12 on the third difference along d and β_dj = (Δt/Δ_j)²/12 on the
transverse average over the rows of the other axes. Its discrete dispersion is fourth order in
every direction at every Δt, its Courant limit is Yee's ([`courant_limit`](@ref)), it keeps Gauss's law
as Yee does and its own ∇·B (the divergence with its difference) to roundoff, and it needs two
guard cells. A choice of
[`EMNumerics`](@ref) (`solver = M4()`); the absorbers, the antennas and the moving window take it
into account (`faraday_stencil`).
"""
struct M4 end

"""
    YeeStencil()

The difference Yee's B update takes along an axis, `F[i+1] − F[i]`: what the field modifiers that
correct the B update (a PML's, an antenna's) need to know of a [`Yee`](@ref) solver.
"""
struct YeeStencil end

"""
    M4Stencil{N,T}

The coefficients of [`M4`](@ref)'s B difference for one time step on a grid of N axes: `δ[d]`
on the third difference along d and `β[d][j]` (zero for j = d) on the second difference along j of
the difference along d. Isbits, for the device. `faraday_stencil(M4(), grid, dt)` makes it.
"""
struct M4Stencil{N,T}
    δ::NTuple{N,T}
    β::NTuple{N,NTuple{N,T}}
    # only the parametric constructor: the default `M4Stencil(δ, β)` leaves T unbound at N = 0
    M4Stencil{N,T}(δ::NTuple{N,T}, β::NTuple{N,NTuple{N,T}}) where {N,T} = new{N,T}(δ, β)
end

"""
    faraday_stencil(solver, grid, dt) -> stencil

What the field modifiers that correct the B update need of the solver for the time step `dt` (in
its element type): [`YeeStencil`](@ref)`()` for [`Yee`](@ref), the [`M4Stencil`](@ref) of the
step for [`M4`](@ref).
"""
faraday_stencil(::Yee, grid, dt) = YeeStencil()
function faraday_stencil(::M4, grid::PeriodicGrid{N}, dt::T) where {N,T}
    r2 = map(d -> (dt / T(d))^2, grid.dx)
    δ = map(r -> (r - one(T)) / 12, r2)
    β = ntuple(d -> ntuple(j -> j == d ? zero(T) : r2[j] / 12, Val(N)), Val(N))
    return M4Stencil{N,T}(δ, β)
end

"Guard cells the solver's stencils need on each side: one for Yee, two for M4 (its third difference reaches two cells ahead)."
solver_reach(::Yee) = 1
solver_reach(::M4) = 2
solver_reach(::YeeStencil) = 1
solver_reach(::M4Stencil) = 2

# The transverse term β_dj·Δ²_j D⁺_d F at I for the axis j ≠ d (none for j = d: the axis itself).
@inline m4_transverse(::Val{j}, ::Val{d}, F, I, d1, st, e, vN) where {j,d} =
    j == d ? zero(d1) : m4_transverse_term(Val(j), F, I, d1, st.β[d][j], e, vN)
@inline function m4_transverse_term(vj, F, I, d1, β, e, vN)
    ej = unit_step(vj, vN)
    @inbounds s = (F[I + e + ej] - F[I + ej]) + (F[I + e - ej] - F[I - ej])
    return β * (s - 2 * d1)
end

"""
    forward_difference(stencil, F, I, Val(d), Val(N))

The B update's difference of `F` along the resolved axis `d` at the index `I` (not divided by the
cell size): `F[I + e_d] − F[I]` for a [`YeeStencil`](@ref), and for an [`M4Stencil`](@ref) that
plus δ_d times the third difference and the β terms of the other resolved axes. `F` is anything
indexed by a `CartesianIndex`. A device function.
"""
@inline function forward_difference(::YeeStencil, F, I, vd, vN)
    e = unit_step(vd, vN)
    @inbounds return F[I + e] - F[I]
end
@inline function forward_difference(st::M4Stencil{N}, F, I, vd::Val{d}, vN) where {N,d}
    e = unit_step(vd, vN)
    @inbounds d1 = F[I + e] - F[I]
    @inbounds d3 = (F[I + 2 * e] - F[I - e]) - 3 * d1
    r = d1 + st.δ[d] * d3
    return r + m4_transverse_sum(st, vd, F, I, d1, e, vN)
end
# The β terms of every resolved axis j ≠ d, written out per N (a closure over Val(j) would not infer).
@inline m4_transverse_sum(st::M4Stencil{1}, vd, F, I, d1, e, vN) = zero(d1)
@inline m4_transverse_sum(st::M4Stencil{2}, vd, F, I, d1, e, vN) =
    m4_transverse(Val(1), vd, F, I, d1, st, e, vN) + m4_transverse(Val(2), vd, F, I, d1, st, e, vN)
@inline m4_transverse_sum(st::M4Stencil{3}, vd, F, I, d1, e, vN) =
    (m4_transverse(Val(1), vd, F, I, d1, st, e, vN) + m4_transverse(Val(2), vd, F, I, d1, st, e, vN)) +
    m4_transverse(Val(3), vd, F, I, d1, st, e, vN)

# op(x, coef_d·D̃_d a) at index I for a resolved axis d; x for a collapsed one.
@inline m4_term(::Val{false}, op, x, a, I, coef, st, vd, vN) = x
@inline m4_term(::Val{true}, op, x, a, I, coef, st, vd::Val{d}, vN) where {d} = op(x, coef[d] * forward_difference(st, a, I, vd, vN))

"""
    faraday_update(b, E, I, h, stencil, Val(c), Val(N))

`b − h_{c₁}·D̃_{c₁}E_{c₂} + h_{c₂}·D̃_{c₂}E_{c₁}` at the index `I`, the B update of component `c` from
the value `b` with the stencil's difference ([`forward_difference`](@ref)), `h_d = h/Δ_d` and `E`
three things indexed by a `CartesianIndex`. A device function: the solver's kernel and the
antenna's corrections take the same expression.
"""
@inline function faraday_update(b, E, I, h, st, vc, vN)
    v1, v2 = cyclic(vc)
    b = m4_term(resolved(v1, vN), -, b, component(E, v2), I, h, st, v1, vN)
    b = m4_term(resolved(v2, vN), +, b, component(E, v1), I, h, st, v2, vN)
    return b
end

@inline m4_b_component!(::Val{false}, B, E, I, h, st, vc, vN) = nothing
@inline function m4_b_component!(::Val{true}, B, E, I, h, st, vc, vN)
    Bc = component(B, vc)
    @inbounds Bc[I] = faraday_update(Bc[I], E, I, h, st, vc, vN)
    return nothing
end

@kernel function m4_b_kernel!(B, E, h, st, g::Int)
    I0 = @index(Global, Cartesian)
    vN = Val(length(I0))
    I = I0 + CartesianIndex(ntuple(_ -> g, vN))
    m4_b_component!(b_dynamic(Val(1), vN), B, E, I, h, st, Val(1), vN)
    m4_b_component!(b_dynamic(Val(2), vN), B, E, I, h, st, Val(2), vN)
    m4_b_component!(b_dynamic(Val(3), vN), B, E, I, h, st, Val(3), vN)
end

"""
    m4_b_half!(f, grid, h, stencil::M4Stencil; workgroupsize = nothing)

Advance the B components of the fields `f` (guards of E filled) by the time `h` with M4's
difference of the `stencil` (made for the full step, [`faraday_stencil`](@ref)),
`B ← B − h·∇̃×E`, then fill the guards of every component it changed, as [`yee_b_half!`](@ref)
does. Throws an `ArgumentError` on a grid with fewer than two guards.
"""
function m4_b_half!(f::NamedTuple, grid::PeriodicGrid{N}, h, st::M4Stencil{N}; workgroupsize = nothing) where {N}
    guards(grid) >= 2 || throw(ArgumentError("the M4 solver needs at least 2 guard cells, the grid has $(guards(grid))"))
    T = eltype(f.Bz)
    hd = map(d -> T(h) / d, grid.dx)
    launcher(m4_b_kernel!, get_backend(f.Bz), workgroupsize)((f.Bx, f.By, f.Bz), (f.Ex, f.Ey, f.Ez), hd, st, guards(grid);
                                                             ndrange = grid.n)
    N >= 2 && fill_guards!(f.Bx, grid; workgroupsize)
    fill_guards!(f.By, grid; workgroupsize)
    fill_guards!(f.Bz, grid; workgroupsize)
    return f
end

# The B half step of the solve as `substep!` calls it, with the stencil of the step.
struct M4BHalf{S}
    st::S
end
(u::M4BHalf)(f, grid, τ; workgroupsize = nothing) = m4_b_half!(f, grid, τ, u.st; workgroupsize)

"""
    courant_limit(::M4, grid) -> Δt_max

The stability limit of the M4 solver, Yee's: `1/√(Σ_d 1/Δ_d²)` (the header of `m4.jl` shows that
M4's discrete relation reaches |sin(ωΔt/2)| = 1 first at the corner mode, at Yee's limit, for
every N and aspect ratio; T-1083).
"""
courant_limit(::M4, grid::PeriodicGrid) = courant_limit(Yee(), grid)

"""
    m4_wavenumber(dt, dx) -> k
    m4_wavenumber(dt, dx, δ) -> k

The wavenumber of the frequency ω₀ = 1 travelling along an axis of cell size `dx` in the M4
scheme with the time step `dt` (δ its coefficient along the axis, `((dt/dx)² − 1)/12` by
default): with u = sin²(kΔx/2) the relation along the axis, sin²(Δt/2)/Δt² = u(1 − 4δu)/Δx², is
the quadratic 4δu² − u + q = 0, q = Δx²·sin²(Δt/2)/Δt², whose small root is
u = 2q/(1 + √(1 − 16δq)); δ = 0 gives Yee's u = q.
"""
function m4_wavenumber(dt::T, dx, δ = ((dt / T(dx))^2 - one(T)) / 12) where {T}
    q = (T(dx) * sin(dt / 2) / dt)^2
    u = 2 * q / (one(T) + sqrt(one(T) - 16 * δ * q))
    return 2 * asin(sqrt(u)) / T(dx)
end
