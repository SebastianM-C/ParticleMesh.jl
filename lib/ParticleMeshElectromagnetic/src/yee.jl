# The 1D Yee solver: staggered curl updates on the layout of FIELD_LAYOUT, periodic through the
# engine's fill_guards!. With ∂_y = ∂_z = 0 and c = 1 (A1), Maxwell's equations reduce to
#     ∂ₜEx = −Jx,   ∂ₜEy = −∂ₓBz − Jy,   ∂ₜEz = ∂ₓBy − Jz,
#     ∂ₜBx = 0,     ∂ₜBy = ∂ₓEz,         ∂ₜBz = −∂ₓEy.
# The forward difference D⁺a[j] = (a[j+1] − a[j])/Δx maps primal (Node) to dual (Center) values,
# the backward difference D⁻b[j] = (b[j] − b[j−1])/Δx dual to primal; on a periodic grid
# (D⁺)ᵀ = −D⁻, which is what makes the leapfrog conserve the staggered energy of T-210.

"""
    Yee()

The 1D Yee FDTD solver in the mesh-solve slot (A2, A10): B in two half steps around a full E
step, so that E and B are both known at integer times for the push.
"""
struct Yee end

@kernel function yee_b_kernel!(By, Bz, @Const(Ey), @Const(Ez), h_dx, g::Int32)
    j = Int32(@index(Global, Linear))
    a = g + j
    @inbounds begin
        By[a] = By[a] + h_dx * (Ez[a + Int32(1)] - Ez[a])
        Bz[a] = Bz[a] - h_dx * (Ey[a + Int32(1)] - Ey[a])
    end
end

# Written as E − (…) so that zero transverse fields and currents stay +0.0 bit for bit (T-213).
@kernel function yee_e_kernel!(Ex, Ey, Ez, @Const(By), @Const(Bz), @Const(Jx), @Const(Jy), @Const(Jz), dt, dt_dx, g::Int32)
    j = Int32(@index(Global, Linear))
    a = g + j
    @inbounds begin
        Ex[a] = Ex[a] - dt * Jx[a]
        Ey[a] = Ey[a] - dt_dx * (Bz[a] - Bz[a - Int32(1)]) - dt * Jy[a]
        Ez[a] = Ez[a] + dt_dx * (By[a] - By[a - Int32(1)]) - dt * Jz[a]
    end
end

"""
    yee_b_half!(f, grid, h; workgroupsize = nothing)

Advance By and Bz of the fields `f` (a [`FIELD_LAYOUT`](@ref) `NamedTuple`, guards of Ey and
Ez filled) by the time `h` (normally Δt/2), `B ← B − h·∇×E`:
`By[j] += h·(Ez[j+1] − Ez[j])/Δx`, `Bz[j] −= h·(Ey[j+1] − Ey[j])/Δx`; then fill their guards.
Bx is constant in 1D and not touched.
"""
function yee_b_half!(f::NamedTuple, grid::PeriodicGrid{1}, h; workgroupsize = nothing)
    T = eltype(f.Bz)
    launcher(yee_b_kernel!, get_backend(f.Bz), workgroupsize)(f.By, f.Bz, f.Ey, f.Ez, T(h) / cellsize(grid),
                                                              Int32(guards(grid)); ndrange = ncells(grid))
    fill_guards!(f.By, grid; workgroupsize)
    fill_guards!(f.Bz, grid; workgroupsize)
    return f
end

"""
    yee_e!(f, grid, dt; workgroupsize = nothing)

Advance Ex, Ey, Ez of the fields `f` (guards of By and Bz filled, the currents folded) by `dt`,
`E ← E + dt·(∇×B − J)`: `Ex[j] −= dt·Jx[j]`, `Ey[j] −= dt·((Bz[j] − Bz[j−1])/Δx + Jy[j])`,
`Ez[j] += dt·((By[j] − By[j−1])/Δx − Jz[j])`; then fill their guards.
"""
function yee_e!(f::NamedTuple, grid::PeriodicGrid{1}, dt; workgroupsize = nothing)
    T = eltype(f.Ex)
    launcher(yee_e_kernel!, get_backend(f.Ex), workgroupsize)(f.Ex, f.Ey, f.Ez, f.By, f.Bz, f.Jx, f.Jy, f.Jz, T(dt),
                                                              T(dt) / cellsize(grid), Int32(guards(grid));
                                                              ndrange = ncells(grid))
    fill_guards!(f.Ex, grid; workgroupsize)
    fill_guards!(f.Ey, grid; workgroupsize)
    fill_guards!(f.Ez, grid; workgroupsize)
    return f
end

"""
    solve_fields!(::Yee, f, grid, dt; workgroupsize = nothing)

The mesh-solve slot: from Eⁿ, Bⁿ and the current Jⁿ⁺¹ᐟ² to Eⁿ⁺¹, Bⁿ⁺¹ by
`Bⁿ⁺¹ᐟ² = Bⁿ − (Δt/2)∇×Eⁿ`, `Eⁿ⁺¹ = Eⁿ + Δt(∇×Bⁿ⁺¹ᐟ² − Jⁿ⁺¹ᐟ²)`, `Bⁿ⁺¹ = Bⁿ⁺¹ᐟ² − (Δt/2)∇×Eⁿ⁺¹`.
In vacuum a mode of wavenumber k advances by the ω of the exact discrete relation
sin(ωΔt/2)/Δt = sin(kΔx/2)/Δx, and Σ(Eⁿ)² + Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² is conserved (T-210); stable for
Δt < Δx.
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid{1}, dt; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    yee_b_half!(f, grid, h; workgroupsize)
    yee_e!(f, grid, dt; workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    return f
end
