# Charge-conserving flux deposition (Esirkepov's scheme) for N = 2: the per-particle stencil and
# the atomic Fast() path. The Reference() path is in flux2d_reference.jl.

"""
    flux_guard_width(shape, stagger::NTuple{N,Stagger}) -> NTuple{N,Int}

The guard cells `deposit_flux!` needs on each side, per axis: `flux_guard_width(shape, stagger[d])`.
The grid's one guard width must be at least the largest of them.
"""
flux_guard_width(shape::Shape, st::NTuple{N,Stagger}) where {N} = map(s -> flux_guard_width(shape, s), st)

"""
    flux_extras(T, N)

The species extras `deposit_flux!` reads the position before the step from, as a module
declares them (`Species(...; extras = merge(flux_extras(T, N), more))`): for `N = 1`
`(old_cell = Int32, old_offset = T)`, for `N = 2` one cell and one offset per axis,
`(old_cell_1 = Int32, old_cell_2 = Int32, old_offset_1 = T, old_offset_2 = T)`. The `Int32` and
`T` extras reuse the sort's core permutation buffers, so they add no scratch.
"""
function flux_extras(::Type{T}, N::Integer) where {T}
    N == 1 && return (old_cell = Int32, old_offset = T)
    N == 2 && return (old_cell_1 = Int32, old_cell_2 = Int32, old_offset_1 = T, old_offset_2 = T)
    throw(ArgumentError("deposit_flux! is implemented for N = 1 and N = 2, not N = $N"))
end

has_flux_extras(s::Species{1}) = haskey(s.extras, :old_cell) && haskey(s.extras, :old_offset)
has_flux_extras(s::Species{2}) = all(k -> haskey(s.extras, k), (:old_cell_1, :old_cell_2, :old_offset_1, :old_offset_2))

@inline old_cell_2d(s, i) = @inbounds (s.extras.old_cell_1[i], s.extras.old_cell_2[i])
@inline old_offset_2d(s, i) = @inbounds (s.extras.old_offset_1[i], s.extras.old_offset_2[i])

"The column-major outer product `f(a[k1], b[k2])` of two tuples, axis 1 fastest, with its length static."
@inline outer(f, a::NTuple{M1,Any}, b::NTuple{M2,Any}) where {M1,M2} =
    ntuple(k -> f(a[(k - 1) % M1 + 1], b[(k - 1) ÷ M1 + 1]), Val(M1 * M2))

"`outer` over two pairs of tuples of equal length per axis: `f((a[1][k1], a[2][k1]), (b[1][k2], b[2][k2]))`."
@inline outer2(f, a::NTuple{2,NTuple{M1,Any}}, b::NTuple{2,NTuple{M2,Any}}) where {M1,M2} =
    ntuple(k -> f((a[1][(k - 1) % M1 + 1], a[2][(k - 1) % M1 + 1]), (b[1][(k - 1) ÷ M1 + 1], b[2][(k - 1) ÷ M1 + 1])), Val(M1 * M2))

"""
    flux_stencil(s, i, shape, stagger::NTuple{2}, n::NTuple{2,Int32}, sx::NTuple{2}, quantity_z)
        -> (lo, M, (jx, jy, jz))

The device-level 2D flux of particle `i`: its contributions to the nodes
`cell .+ lo .+ (k1, k2) .- 1`, `k_d = 1:M[d]`, of Jx, Jy and Jz, each an `NTuple{M[1]·M[2]}`
listed column-major. The frame is the particle's current cell; its old cell is at most one cell
away per axis, periodically on `n`. With `S0_d`, `S1_d` the old and new 1D shape weights of axis
`d` on its node range (`flux_axis_weights`), `ΔS_d = S1_d - S0_d`, `A_d = (S0_d + S1_d)/2` and
`P_d` the prefix sum of `ΔS_d` in increasing node order, Esirkepov's decomposition is

    jx[k1, k2] = -(q·w·sx₁) · (P_1[k1] · A_2[k2])
    jy[k1, k2] = -(q·w·sx₂) · (A_1[k1] · P_2[k2])
    jz[k1, k2] = quantity_z(s, i) · (A_1[k1]·A_2[k2] + ΔS_1[k1]·ΔS_2[k2]/12),

with `sx_d = Δx_d/Δt`. Expanded, `jx` is `-(q·w·sx₁) Σ_{m ≤ k1} ΔS_1[m]·(S0_2[k2] + ΔS_2[k2]/2)`, the
form of Smilei's `Projector2D2Order::currents` and PIConGPU's `Esirkepov2D`, and the weight of
`jz` is `S0_1·S0_2 + (ΔS_1·S0_2 + S0_1·ΔS_2)/2 + ΔS_1·ΔS_2/3`, Esirkepov's out-of-plane weight,
which is not the average of the old and new products. At rest (`ΔS = 0`) `jx`, `jy` are zero and
`jz` is bitwise `quantity_z` times the tensor-product weights `deposit!` uses. The contributions
of a particle are the same bits in every mode and mechanism.
"""
@inline function flux_stencil(s, i, shape::Shape, st::NTuple{2,Stagger}, n::NTuple{2,Int32}, sx::NTuple{2,Any}, quantity_z)
    c = particle_cell(s, i, Val(2))
    δ = particle_offset(s, i, Val(2))
    c0 = old_cell_2d(s, i)
    δ0 = old_offset_2d(s, i)
    lo1, Wo1, Wn1 = flux_axis_weights(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
    lo2, Wo2, Wn2 = flux_axis_weights(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
    D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
    A1, A2 = map((a, b) -> (a + b) / 2, Wo1, Wn1), map((a, b) -> (a + b) / 2, Wo2, Wn2)
    P1, P2 = prefix_sums(D1), prefix_sums(D2)
    @inbounds qw = s.q * s.w[i]
    qws1, qws2 = qw * sx[1], qw * sx[2]
    qz = quantity_z(s, i)
    T = typeof(qw)
    jx = outer((p, a) -> -qws1 * (p * a), P1, A2)
    jy = outer((a, p) -> -qws2 * (a * p), A1, P2)
    jz = outer2(((a1, d1), (a2, d2)) -> qz * (a1 * a2 + d1 * d2 / T(12)), (A1, D1), (A2, D2))
    return ((lo1, lo2), (length(Wo1), length(Wo2)), (jx, jy, jz))
end

@kernel function flux_atomic_2d_kernel!(Jx, Jy, Jz, s, shape, st, n::NTuple{2,Int32}, g::Int32, sx, quantity_z)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        lo, M, (jx, jy, jz) = flux_stencil(s, i, shape, st, n, sx, quantity_z)
        c = particle_cell(s, i, Val(2))
        b1, b2 = Int(c[1] + lo[1] + g), Int(c[2] + lo[2] + g)
        for k in 1:length(jx)
            idx = CartesianIndex(b1 + (k - 1) % M[1], b2 + (k - 1) ÷ M[1])
            Atomix.@atomic Jx[idx] += jx[k]
            Atomix.@atomic Jy[idx] += jy[k]
            Atomix.@atomic Jz[idx] += jz[k]
        end
    end
end

"An unset keyword: `deposit_flux!` at N = 2 takes no `quantity_y`."
struct NoQuantity end

"""
    deposit_flux!(mode, Jx, Jy, Jz, species::Species{2}, shape, stagger::NTuple{2,Stagger}, grid, ws;
                  dt, quantity_z, workgroupsize = nothing)

The charge-conserving current of one step at N = 2 (Esirkepov's scheme, `flux_stencil`), added
into the three fields `Jx`, `Jy`, `Jz` (matrices with ρ's layout and guards). `stagger` is ρ's
stagger, and every shape weight of all three components is evaluated at it. `Jx[j]` is the
face half a cell to the right of ρ's node `j` along axis 1 and `Jy[j]` the face half a cell up
along axis 2, both at ρ's index `j`; `Jz` sits on ρ's nodes. For ρ at `(Node(), Node())`, the
module's layout, they are at `(Center(), Node())`, `(Node(), Center())` and `(Node(), Node())`. The
in-plane components carry `q·w·Δx_d/dt`, with each axis' cell size; `Jz` carries the module's
`quantity_z(species, i)` (q·w·u_z/γ for a current). `Jy` is in-plane at N = 2, so a
`quantity_y` is an `ArgumentError`. The old position comes from the extras of
`flux_extras(T, 2)`, which the module stores before its push and the sort carries.

With ρ from `deposit!` with the same shape and stagger before and after the step, discrete
continuity `(ρ_new - ρ_old)/dt + (Jx[j] - Jx[j - e₁])/Δx₁ + (Jy[j] - Jy[j - e₂])/Δx₂ = 0` holds at
every node to roundoff (T-300); `Σ Jx = Σ q·w·Δ₁/dt`, `Σ Jy = Σ q·w·Δ₂/dt` with Δ_d the
displacement, `Σ Jz = Σ quantity_z` (T-301, T-309); a particle at rest deposits exactly zero Jx
and Jy (T-308).

Preconditions: every particle moved less than one cell per axis since the old position was
stored, the seams and the corner included (not detected); at least 3 cells per axis; at least
`maximum(flux_guard_width(shape, stagger))` guard cells (checked). The guard convention is
`deposit!`'s (fold each field). `Fast()` adds with atomics, one work item per particle, and falls
back to `Reference()` for element types without them (`effective_mode`). `Reference()` needs
`sort_by_cell!` after the move and a workspace built with `Workspace(species, grid; flux = (shape,
stagger))`; it is bitwise across reruns, thread counts and launch shapes (T-303).
"""
function deposit_flux!(mode, Jx::AbstractMatrix{T}, Jy::AbstractMatrix{T}, Jz::AbstractMatrix{T}, s::Species{2},
                       shape::Shape, stagger::NTuple{2,Stagger}, grid::PeriodicGrid{2}, ws::Workspace;
                       dt, quantity_z, quantity_y = NoQuantity(), workgroupsize = nothing) where {T}
    quantity_y isa NoQuantity ||
        throw(ArgumentError("deposit_flux! at N = 2 takes no quantity_y: Jy is in-plane, from Esirkepov's decomposition"))
    has_flux_extras(s) ||
        throw(ArgumentError("deposit_flux! at N = 2 reads the old position from the extras of flux_extras(T, 2): $(keys(flux_extras(T, 2)))"))
    gw = maximum(flux_guard_width(shape, stagger))
    guards(grid) >= gw ||
        throw(ArgumentError("deposit_flux! needs $gw guard cells for this shape and stagger, the grid has $(guards(grid))"))
    all(>=(3), ncells(grid)) || throw(ArgumentError("deposit_flux! needs at least 3 cells per axis"))
    sx = map(h -> h / dt, cellsize(grid))
    _deposit_flux_2d!(effective_mode(mode, T), Jx, Jy, Jz, s, shape, stagger, grid, ws, sx, quantity_z, workgroupsize)
end

function _deposit_flux_2d!(::Fast, Jx, Jy, Jz, s, shape, st, grid, ws, sx, qz, workgroupsize)
    launcher(flux_atomic_2d_kernel!, get_backend(s), workgroupsize)(Jx, Jy, Jz, s, shape, st, Int32.(ncells(grid)),
                                                                   Int32(guards(grid)), sx, qz; ndrange = capacity(s))
    return Jx, Jy, Jz
end

# The Reference() path: per-cell patches, then a per-node reduction (the mechanism measured in
# reports/wp-020-flux-2d.md against per-node recomputation and per-particle scratch).

"The per-axis stencil width of the 2D flux as plain integers."
@inline flux_widths(shape::Shape, st::NTuple{2,Stagger}) = map(s -> _valof(flux_width(shape, s)), st)
_valof(::Val{V}) where {V} = V

"""
    flux_patch_values(shape, stagger::NTuple{2,Stagger})

Values of `T` per linear cell in the flux patch of `Workspace(species, grid; flux = (shape,
stagger))`: `3·M₁·M₂`, three components on the `M₁ × M₂` flux stencil.
"""
flux_patch_values(shape::Shape, st::NTuple{2,Stagger}) = 3 * prod(flux_widths(shape, st))
flux_patch_values(shape::Shape, st) = throw(ArgumentError("flux scratch takes ρ's stagger pair at N = 2, got $st"))

"Elementwise sum of two tuples of one length."
@inline tuple_add(a::NTuple{K,Any}, b::NTuple{K,Any}) where {K} = ntuple(k -> a[k] + b[k], Val(K))

# One work item per linear cell: its particles in slot order, each stencil entry summed from zero
# in a register tuple and written once into the cell's patch (`patch[L, k]`, cell-major so that
# neighbouring work items write neighbouring entries; the three components one after the other,
# Jx first). An empty cell writes zeros.
@kernel function flux_patch_2d_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), shape, st, n::NTuple{2,Int32}, sx,
                                       quantity_z, ::Val{K}) where {K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    ax = ntuple(_ -> zero(T), Val(K))
    ay, az = ax, ax
    @inbounds for i in seg_start[L]:seg_stop[L]
        _, _, (jx, jy, jz) = flux_stencil(s, i, shape, st, n, sx, quantity_z)
        ax = tuple_add(ax, jx)
        ay = tuple_add(ay, jy)
        az = tuple_add(az, jz)
    end
    @inbounds for k in 1:K
        patch[L, k] = ax[k]
        patch[L, K + k] = ay[k]
        patch[L, 2K + k] = az[k]
    end
end

# One work item per node (guards included): the patches of the cells whose flux stencil reaches
# it, in increasing column-major linear index, summed from zero.
@kernel function flux_reduce_2d_kernel!(Jx, Jy, Jz, @Const(patch), n::NTuple{2,Int32}, g::Int32, lo::NTuple{2,Int32},
                                        hi::NTuple{2,Int32}, M::NTuple{2,Int32})
    I = @index(Global, Cartesian)
    T = eltype(Jx)
    K = M[1] * M[2]
    j1, j2 = Int32(I[1]) - g, Int32(I[2]) - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for c2 in max(Int32(1), j2 - hi[2]):min(n[2], j2 - lo[2]), c1 in max(Int32(1), j1 - hi[1]):min(n[1], j1 - lo[1])
        L = c1 + (c2 - Int32(1)) * n[1]
        k = (j1 - c1 - lo[1]) + (j2 - c2 - lo[2]) * M[1] + Int32(1)
        ax += patch[L, k]
        ay += patch[L, K + k]
        az += patch[L, 2K + k]
    end
    @inbounds begin
        Jx[I] += ax
        Jy[I] += ay
        Jz[I] += az
    end
end

function _deposit_flux_2d!(::Reference, Jx, Jy, Jz, s, shape, st, grid, ws, sx, qz, workgroupsize)
    check_workspace(ws, s, grid)
    need = flux_patch_values(shape, st)
    size(ws.flux_patch) == (prod(ncells(grid)), need) ||
        throw(ArgumentError("the Reference() deposit_flux! at N = 2 needs Workspace(species, grid; flux = (shape, stagger)) for this " *
                            "shape and stagger: $need values per cell, the workspace has $(size(ws.flux_patch, 2))"))
    backend = get_backend(s)
    n = Int32.(ncells(grid))
    M = flux_widths(shape, st)
    r = map(x -> flux_reach(shape, x), st)
    lo, hi = (r[1][1], r[2][1]), (r[1][2], r[2][2])
    launcher(flux_patch_2d_kernel!, backend, workgroupsize)(ws.flux_patch, s, ws.seg_start, ws.seg_stop, shape, st, n, sx, qz,
                                                            Val(prod(M)); ndrange = prod(n))
    launcher(flux_reduce_2d_kernel!, backend, workgroupsize)(Jx, Jy, Jz, ws.flux_patch, n, Int32(guards(grid)), lo, hi, Int32.(M);
                                                             ndrange = size(Jx))
    return Jx, Jy, Jz
end
