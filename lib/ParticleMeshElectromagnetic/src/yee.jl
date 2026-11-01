# The Yee solver in N dimensions: staggered curl updates on the layout of `field_layout(Val(N))`,
# periodic through the engine's fill_guards!. With c = 1 (A1), Maxwell's equations read
#     ∂ₜE = ∇×B − J,   ∂ₜB = −∇×E,   (∇×F)_c = ∂_{c₁}F_{c₂} − ∂_{c₂}F_{c₁}
# with (c, c₁, c₂) a cyclic permutation of (1, 2, 3) and ∂_d = 0 along a collapsed axis d > N.
# Along a resolved axis d, the forward difference D⁺_d a[j] = (a[j + e_d] − a[j])/Δ_d maps
# Node() to Center() values (the derivatives of E in the B update), the backward difference
# D⁻_d b[j] = (b[j] − b[j − e_d])/Δ_d maps Center() to Node() values (those of B in the E update);
# on a periodic grid (D⁺_d)ᵀ = −D⁻_d, so the two discrete curls are each other's transpose, which
# is what makes the leapfrog conserve the staggered energy (T-210, T-323) and keep ∇·B (T-322).
# A derivative along a collapsed axis is absent: dropped by dispatch on `Val(d <= N)`, not
# computed as a difference of equal values. At N = 1 the expressions are those of the 1D solver
# before WP-022 term by term, so its bits are unchanged (the digest in WP-022's pull request).

"""
    Yee()

The Yee FDTD solver in the mesh-solve slot (A2, A10), for any number N of resolved axes (N = 1
and N = 2 claimed and tested): B in two half steps around a full E step, so that E and B are both
known at integer times for the push. Its stability limit is [`courant_limit`](@ref).
"""
struct Yee end

# The unit step along axis d in N dimensions, a compile-time constant.
@inline unit_step(::Val{d}, ::Val{N}) where {d,N} = CartesianIndex(ntuple(k -> k == d ? 1 : 0, Val(N)))
# The cyclic successors (c₁, c₂) of component c.
@inline cyclic(::Val{1}) = (Val(2), Val(3))
@inline cyclic(::Val{2}) = (Val(3), Val(1))
@inline cyclic(::Val{3}) = (Val(1), Val(2))
# Whether axis d is resolved at N (a Val{Bool}, so the absent terms are removed by dispatch).
@inline resolved(::Val{d}, ::Val{N}) where {d,N} = Val(d <= N)
@inline component(F::Tuple, ::Val{c}) where {c} = @inbounds F[c]

struct Forward end
struct Backward end

# op(x, coef_d·D_d a) at index I for a resolved axis d (D = D⁺ or D⁻ times Δ_d); x for a collapsed one.
@inline curl_term(::Val{false}, op, x, a, I, coef, vd, vN, dir) = x
@inline function curl_term(::Val{true}, op, x, a, I, coef, vd::Val{d}, vN, ::Forward) where {d}
    @inbounds return op(x, coef[d] * (a[I + unit_step(vd, vN)] - a[I]))
end
@inline function curl_term(::Val{true}, op, x, a, I, coef, vd::Val{d}, vN, ::Backward) where {d}
    @inbounds return op(x, coef[d] * (a[I] - a[I - unit_step(vd, vN)]))
end

# Whether B_c has a curl term at N: some axis other than c is resolved.
@inline b_dynamic(::Val{c}, ::Val{N}) where {c,N} = Val(N >= 2 || c != 1)

# B_c ← B_c − h·(∇×E)_c = B_c − h_{c₁}·D⁺_{c₁}E_{c₂} + h_{c₂}·D⁺_{c₂}E_{c₁}, with h_d = h/Δ_d.
@inline b_component!(::Val{false}, B, E, I, h, vc, vN) = nothing
@inline function b_component!(::Val{true}, B, E, I, h, vc, vN)
    v1, v2 = cyclic(vc)
    Bc = component(B, vc)
    @inbounds b = Bc[I]
    b = curl_term(resolved(v1, vN), -, b, component(E, v2), I, h, v1, vN, Forward())
    b = curl_term(resolved(v2, vN), +, b, component(E, v1), I, h, v2, vN, Forward())
    @inbounds Bc[I] = b
    return nothing
end

# E_c ← E_c + dt·(∇×B)_c − dt·J_c = E_c + k_{c₁}·D⁻_{c₁}B_{c₂} − k_{c₂}·D⁻_{c₂}B_{c₁} − dt·J_c, with
# k_d = dt/Δ_d. Written as E − (…), so zero fields and currents stay +0.0 bit for bit (T-213).
@inline function e_component!(E, B, J, I, k, dt, vc, vN)
    v1, v2 = cyclic(vc)
    Ec = component(E, vc)
    @inbounds e = Ec[I]
    e = curl_term(resolved(v1, vN), +, e, component(B, v2), I, k, v1, vN, Backward())
    e = curl_term(resolved(v2, vN), -, e, component(B, v1), I, k, v2, vN, Backward())
    @inbounds Ec[I] = e - dt * component(J, vc)[I]
    return nothing
end

# One work item per interior node, updating the components it owns at that index.
@kernel function yee_b_kernel!(B, E, h, g::Int)
    I0 = @index(Global, Cartesian)
    vN = Val(length(I0))
    I = I0 + CartesianIndex(ntuple(_ -> g, vN))
    b_component!(b_dynamic(Val(1), vN), B, E, I, h, Val(1), vN)
    b_component!(b_dynamic(Val(2), vN), B, E, I, h, Val(2), vN)
    b_component!(b_dynamic(Val(3), vN), B, E, I, h, Val(3), vN)
end

@kernel function yee_e_kernel!(E, B, J, k, dt, g::Int)
    I0 = @index(Global, Cartesian)
    vN = Val(length(I0))
    I = I0 + CartesianIndex(ntuple(_ -> g, vN))
    e_component!(E, B, J, I, k, dt, Val(1), vN)
    e_component!(E, B, J, I, k, dt, Val(2), vN)
    e_component!(E, B, J, I, k, dt, Val(3), vN)
end

"""
    yee_b_half!(f, grid, h; workgroupsize = nothing)

Advance the B components of the fields `f` (a [`field_layout`](@ref) `NamedTuple` on `grid`,
guards of E filled) by the time `h` (normally Δt/2), `B ← B − h·∇×E` with the forward differences
of E along the resolved axes; then fill the guards of every component it changed. At N = 1:
`By[j] += h·(Ez[j+1] − Ez[j])/Δx`, `Bz[j] −= h·(Ey[j+1] − Ey[j])/Δx`, and Bx (constant in 1D) is
not touched; at N ≥ 2 all three components are dynamic.
"""
function yee_b_half!(f::NamedTuple, grid::PeriodicGrid{N}, h; workgroupsize = nothing) where {N}
    T = eltype(f.Bz)
    hd = map(d -> T(h) / d, grid.dx)
    launcher(yee_b_kernel!, get_backend(f.Bz), workgroupsize)((f.Bx, f.By, f.Bz), (f.Ex, f.Ey, f.Ez), hd, guards(grid);
                                                              ndrange = grid.n)
    N >= 2 && fill_guards!(f.Bx, grid; workgroupsize)
    fill_guards!(f.By, grid; workgroupsize)
    fill_guards!(f.Bz, grid; workgroupsize)
    return f
end

"""
    yee_e!(f, grid, dt; workgroupsize = nothing)

Advance the E components of the fields `f` (guards of B filled, the currents folded) by `dt`,
`E ← E + dt·(∇×B − J)` with the backward differences of B along the resolved axes; then fill
their guards. At N = 1: `Ex[j] −= dt·Jx[j]`, `Ey[j] −= dt·((Bz[j] − Bz[j−1])/Δx + Jy[j])`,
`Ez[j] += dt·((By[j] − By[j−1])/Δx − Jz[j])`.
"""
function yee_e!(f::NamedTuple, grid::PeriodicGrid{N}, dt; workgroupsize = nothing) where {N}
    T = eltype(f.Ex)
    kd = map(d -> T(dt) / d, grid.dx)
    launcher(yee_e_kernel!, get_backend(f.Ex), workgroupsize)((f.Ex, f.Ey, f.Ez), (f.Bx, f.By, f.Bz), (f.Jx, f.Jy, f.Jz),
                                                              kd, T(dt), guards(grid); ndrange = grid.n)
    fill_guards!(f.Ex, grid; workgroupsize)
    fill_guards!(f.Ey, grid; workgroupsize)
    fill_guards!(f.Ez, grid; workgroupsize)
    return f
end

"""
    solve_fields!(::Yee, f, grid, dt; workgroupsize = nothing)

The mesh-solve slot: from Eⁿ, Bⁿ and the current Jⁿ⁺¹ᐟ² to Eⁿ⁺¹, Bⁿ⁺¹ by
`Bⁿ⁺¹ᐟ² = Bⁿ − (Δt/2)∇×Eⁿ`, `Eⁿ⁺¹ = Eⁿ + Δt(∇×Bⁿ⁺¹ᐟ² − Jⁿ⁺¹ᐟ²)`, `Bⁿ⁺¹ = Bⁿ⁺¹ᐟ² − (Δt/2)∇×Eⁿ⁺¹`
([`yee_b_half!`](@ref), [`yee_e!`](@ref)), on the fields `f` of [`field_layout`](@ref)`(Val(N))`
on a `PeriodicGrid{N}` (N = 1 and N = 2; guards of E filled on entry, the currents folded). Every
component it changes has its guards filled on return.

In vacuum a mode of wave vector k advances by the ω of the exact discrete relation
sin(ωΔt/2)/Δt = √(Σ_d sin²(k_dΔ_d/2)/Δ_d²) (T-210, T-320), and Σ(Eⁿ)² + Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² is
conserved (T-323); it is stable for Δt below [`courant_limit`](@ref) and grows beyond it (T-325).
It does not check Δt: the state constructor does ([`check_courant`](@ref)).
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid, dt; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    yee_b_half!(f, grid, h; workgroupsize)
    yee_e!(f, grid, dt; workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    return f
end

"""
    courant_limit(::Yee, grid) -> Δt_max

The stability limit of the Yee solver on `grid`: `1/√(Σ_d 1/Δ_d²)` (c = 1, A1), in the grid's
element type; at N = 1 exactly the cell size Δx. At the highest mode of the grid the scheme's
one-step map has the amplification factor (S + √(S² − 1))² per step for S = Δt/Δt_max > 1 and
modulus one below (T-325). The limit is at most min_d Δ_d (rounded down to it where rounding
would put it above), so with |v| < c a time step below it also keeps every particle within one
cell per axis per step, which the engine's `deposit_flux!` needs. A later solver brings its own
method.
"""
courant_limit(::Yee, grid::PeriodicGrid{1}) = cellsize(grid)
courant_limit(::Yee, grid::PeriodicGrid) = min(minimum(grid.dx), inv(sqrt(sum(d -> inv(d * d), grid.dx))))

"""
    check_courant(solver, grid, dt) -> dt

Throw an `ArgumentError` naming the limit unless `dt < courant_limit(solver, grid)`. The state
constructor calls it for the numerics' solver, so a state exists only below the limit: the
solver is stable and, since the limit is at most min_d Δ_d and |v| < c, every particle moves
less than one cell per axis per step, as `deposit_flux!` requires.
"""
function check_courant(solver, grid::PeriodicGrid, dt)
    limit = courant_limit(solver, grid)
    dt < limit ||
        throw(ArgumentError("dt = $dt must be below the Courant limit $limit of $(nameof(typeof(solver))) on the cell sizes $(grid.dx)"))
    return dt
end
