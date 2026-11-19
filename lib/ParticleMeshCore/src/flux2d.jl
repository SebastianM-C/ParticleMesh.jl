# Charge-conserving flux deposition (Esirkepov's scheme) for N = 2: the per-particle stencil and
# its two paths, atomic Fast() and per-cell-patch Reference().

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
`(old_cell_1 = Int32, old_cell_2 = Int32, old_offset_1 = T, old_offset_2 = T)`, and for `N = 3`
likewise with a third axis. The `Int32` and `T` extras reuse the sort's core permutation buffers,
so they add no scratch.
"""
function flux_extras(::Type{T}, N::Integer) where {T}
    N == 1 && return (old_cell = Int32, old_offset = T)
    N == 2 && return (old_cell_1 = Int32, old_cell_2 = Int32, old_offset_1 = T, old_offset_2 = T)
    N == 3 && return (old_cell_1 = Int32, old_cell_2 = Int32, old_cell_3 = Int32, old_offset_1 = T, old_offset_2 = T, old_offset_3 = T)
    throw(ArgumentError("deposit_flux! is implemented for N = 1, 2 and 3, not N = $N"))
end

has_flux_extras(s::Species{1}) = haskey(s.extras, :old_cell) && haskey(s.extras, :old_offset)
has_flux_extras(s::Species{2}) = all(k -> haskey(s.extras, k), (:old_cell_1, :old_cell_2, :old_offset_1, :old_offset_2))

@inline old_cell_2d(s, i) = @inbounds (s.extras.old_cell_1[i], s.extras.old_cell_2[i])
@inline old_offset_2d(s, i) = @inbounds (s.extras.old_offset_1[i], s.extras.old_offset_2[i])

# `outer` and `outer2` build the `M1·M2` entries of a stencil component with `ntuple`, one call of a
# closure per entry. Both closures are forced inline, the entry's and `f`: left to the compiler's
# cost model they stay out of line for the larger bodies (`outer2`'s for every element type, all
# three components' for a `Dual`), and each entry then costs a call and, since `k` is no longer
# a constant there, an integer division by `M1` (T-800). The arithmetic of an entry is the same
# either way.

"The column-major outer product `f(a[k1], b[k2])` of two tuples, axis 1 fastest, with its length static."
@inline outer(f::F, a::NTuple{M1,Any}, b::NTuple{M2,Any}) where {F,M1,M2} =
    ntuple(k -> (@inline; @inline f(a[(k - 1) % M1 + 1], b[(k - 1) ÷ M1 + 1])), Val(M1 * M2))

"`outer` over two pairs of tuples of equal length per axis: `f((a[1][k1], a[2][k1]), (b[1][k2], b[2][k2]))`."
@inline outer2(f::F, a::NTuple{2,NTuple{M1,Any}}, b::NTuple{2,NTuple{M2,Any}}) where {F,M1,M2} =
    ntuple(k -> (@inline; @inline f((a[1][(k - 1) % M1 + 1], a[2][(k - 1) % M1 + 1]),
                                    (b[1][(k - 1) ÷ M1 + 1], b[2][(k - 1) ÷ M1 + 1]))), Val(M1 * M2))

"""
    CellFrame(), ParticleFrame()

The node range of a particle's 2D flux stencil per axis. `CellFrame()`: the `M` nodes of
`flux_reach` around the particle's current cell (6 at order 2 on nodes), the same for every
particle of a cell, which the `Reference()` patch needs. `ParticleFrame()`: the `P + 2` nodes of
the union of the particle's old and new stencil (`flux_axis_frame`, 4 at order 2), along the
component's own axis one fewer (`P + 1`): the prefix sum of the weight changes on the last node is
the sum of all of them, zero up to roundoff, so that column is left out, as PIConGPU, WarpX and
Smilei do. `Fast()` adds in the particle's frame.
"""
struct CellFrame end
struct ParticleFrame end

"""
    BinFrame(bins)

The node range of a particle's 2D flux stencil per axis in the frame of its bin (`NodeBins` or
`CellBins`): the `M` nodes of `flux_bin_reach` around the bin, the same for every particle of a
bin, which the `Reference()` patch needs. With `CellBins()` it is `CellFrame()`, the same bits; with
`NodeBins` of the flux's shape and ρ's stagger, 5 nodes per axis at order 2 on nodes where the cell's
frame has 6.
"""
struct BinFrame{B}
    bins::B
end

@inline axis_frame(::CellFrame, shape, stagger, c, δ, c0, δ0, n, d) = flux_axis_weights(shape, stagger, c, δ, c0, δ0, n)
@inline axis_frame(::ParticleFrame, shape, stagger, c, δ, c0, δ0, n, d) = flux_axis_frame(shape, stagger, c, δ, c0, δ0, n)
@inline function axis_frame(fr::BinFrame, shape, stagger, c, δ, c0, δ0, n, d)
    lo, _ = flux_bin_reach(fr.bins, shape, stagger, d)
    h = bin_shift(fr.bins, d, δ)                                    # the bin is the cell c + h
    return (lo, flux_axis_weights_at(shape, stagger, c, δ, c0, δ0, n, h + lo, flux_bin_width(fr.bins, shape, stagger, d))...)
end

# The prefix sums of a component along its own axis on the frame's nodes.
@inline frame_prefix(::CellFrame, D) = prefix_sums(D)
@inline frame_prefix(::BinFrame, D) = prefix_sums(D)
@inline frame_prefix(::ParticleFrame, D) = Base.front(prefix_sums(D))

"""
    flux_stencil(s, i, shape, stagger::NTuple{2}, n::NTuple{2,Int32}, sx::NTuple{2}, quantity_z,
                 frame = CellFrame()) -> (lo, M, (jx, jy, jz))

The device-level 2D flux of particle `i`: its contributions to the nodes
`cell .+ lo .+ (k1, k2) .- 1` of Jx, Jy and Jz, each a tuple listed column-major. In the
`CellFrame()` every component covers `k_d = 1:M[d]` (`flux_reach`) and `M` is returned; in the
`ParticleFrame()` (`flux_axis_frame`) `lo` is the particle's first node, Jx covers
`(P + 1) × (P + 2)` nodes, Jy `(P + 2) × (P + 1)` and Jz `(P + 2) × (P + 2)`, and `M` is
`(P + 2, P + 2)`. The particle's old cell is at most one cell away per axis, periodically on `n`.
With `S0_d`, `S1_d` the old and new 1D shape weights of axis `d` on the frame's nodes,
`ΔS_d = S1_d - S0_d`, `A_d = (S0_d + S1_d)/2`, `P_d` the prefix sum of `ΔS_d` in increasing node
order and `k_d = -(q·w·sx_d)`, Esirkepov's decomposition is

    jx[k1, k2] = (k_1·P_1[k1]) · A_2[k2]
    jy[k1, k2] = A_1[k1] · (k_2·P_2[k2])
    jz[k1, k2] = (qz·A_1[k1]) · A_2[k2] + ((qz·ΔS_1[k1]) / 12) · ΔS_2[k2],     qz = quantity_z(s, i),

each axis' factor scaled once before the outer product (#454), with `sx_d = Δx_d/Δt`. Expanded,
`jx` is `-(q·w·sx₁) Σ_{m ≤ k1} ΔS_1[m]·(S0_2[k2] + ΔS_2[k2]/2)`, the form of Smilei's
`Projector2D2Order::currents` and PIConGPU's `Esirkepov2D`, and the weight of `jz` is
`S0_1·S0_2 + (ΔS_1·S0_2 + S0_1·ΔS_2)/2 + ΔS_1·ΔS_2/3`, Esirkepov's out-of-plane weight, which is
not the average of the old and new products. At rest (`ΔS = 0`) `jx`, `jy` are zero. On the nodes
both frames hold, a particle's contributions are the same bits in both (the cell frame's extra
nodes have `ΔS = 0` before the particle's first node, so its prefix sums there are exact zeros);
the cell frame's other nodes hold zeros, and along a component's own axis the prefix sum of all
`ΔS` (T-840).
"""
@inline function flux_stencil(s, i, shape::Shape, st::NTuple{2,Stagger}, n::NTuple{2,Int32}, sx::NTuple{2,Any}, quantity_z,
                              frame = CellFrame())
    c = particle_cell(s, i, Val(2))
    δ = particle_offset(s, i, Val(2))
    c0 = old_cell_2d(s, i)
    δ0 = old_offset_2d(s, i)
    lo1, Wo1, Wn1 = axis_frame(frame, shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1], 1)
    lo2, Wo2, Wn2 = axis_frame(frame, shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2], 2)
    D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
    A1, A2 = map((a, b) -> (a + b) / 2, Wo1, Wn1), map((a, b) -> (a + b) / 2, Wo2, Wn2)
    @inbounds qw = s.q * s.w[i]
    k1, k2 = -(qw * sx[1]), -(qw * sx[2])
    p1 = map(x -> k1 * x, frame_prefix(frame, D1))
    p2 = map(x -> k2 * x, frame_prefix(frame, D2))
    qz = quantity_z(s, i)
    a1 = map(x -> qz * x, A1)
    e1 = map(x -> qz * x / 12, D1)
    jx = outer((p, a) -> p * a, p1, A2)
    jy = outer((a, p) -> a * p, A1, p2)
    jz = outer2(((a, e), (b, d)) -> a * b + e * d, (a1, e1), (A2, D2))
    return ((lo1, lo2), (length(Wo1), length(Wo2)), (jx, jy, jz))
end

# Atomic adds of the column-major block `v` of `M1` rows at the field index `(b1, b2)`. The adds
# are `:monotonic` (relaxed): nothing reads `J` before the kernel ends, so they need atomicity and
# no ordering, and Atomix's default sequentially consistent adds stalled the kernel on memory
# barriers (Nsight Compute on an RTX 5090: 28 barrier stall cycles per issued instruction, the top reason).
# The `K` adds are written out with `ntuple`, not left to a loop, so that a reverse-mode AD of the
# kernel (Enzyme) keeps their addresses in its fixed-size tape instead of allocating device memory
# per work item and step (WP-050); the same adds in the same order.
@inline function atomic_add_block!(J, v::NTuple{K,Any}, b1::Int, b2::Int, ::Val{M1}) where {K,M1}
    ntuple(Val(K)) do k
        @inline
        @inbounds Atomix.@atomic :monotonic J[CartesianIndex(b1 + (k - 1) % M1, b2 + (k - 1) ÷ M1)] += v[k]
        nothing
    end
    return nothing
end

# One work item per particle, in the particle's frame: 2(P + 1)(P + 2) + (P + 2)² atomic adds
# (40 at order 2, against 108 in the cell's frame).
@kernel function flux_atomic_2d_kernel!(Jx, Jy, Jz, s, shape::Shape{P}, st, n::NTuple{2,Int32}, g::Int32, sx, quantity_z) where {P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        f, _, (jx, jy, jz) = flux_stencil(s, i, shape, st, n, sx, quantity_z, ParticleFrame())
        c = particle_cell(s, i, Val(2))
        b1, b2 = Int(c[1] + f[1] + g), Int(c[2] + f[2] + g)
        atomic_add_block!(Jx, jx, b1, b2, Val(P + 1))
        atomic_add_block!(Jy, jy, b1, b2, Val(P + 2))
        atomic_add_block!(Jz, jz, b1, b2, Val(P + 2))
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
`deposit!`'s (fold each field). `Fast()` adds with atomics, one work item per particle, in the
particle's frame (`ParticleFrame()`: 40 adds at order 2 where the cell's frame has 108), so it
differs from `Reference()` per node by roundoff, the prefix sums of all weight changes it leaves
out included (T-302), and falls back to `Reference()` for element types without them (`effective_mode`), sorting first (T-345). `Reference()` needs
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
    fallback_sort!(mode, T, s, grid, ws, workgroupsize)
    _deposit_flux_2d!(effective_mode(mode, T), Jx, Jy, Jz, s, shape, stagger, grid, ws, sx, quantity_z, workgroupsize)
end

function _deposit_flux_2d!(::Fast, Jx, Jy, Jz, s, shape, st, grid, ws, sx, qz, workgroupsize)
    launcher(flux_atomic_2d_kernel!, get_backend(s), workgroupsize)(Jx, Jy, Jz, s, shape, st, Int32.(ncells(grid)),
                                                                   Int32(guards(grid)), sx, qz; ndrange = capacity(s))
    return Jx, Jy, Jz
end

# The Reference() path: per-cell patches, then a per-node reduction (the mechanism measured in
# reports/wp-020-flux-2d.md against per-node recomputation and per-particle scratch).

"The per-axis stencil width of the 2D flux in the frame of a bin (`flux_bin_width`) as plain integers."
@inline flux_widths(shape::Shape, st::NTuple{2,Stagger}, bins = CellBins()) =
    (_valof(flux_bin_width(bins, shape, st[1], 1)), _valof(flux_bin_width(bins, shape, st[2], 2)))
_valof(::Val{V}) where {V} = V

"""
    flux_patch_values(shape, stagger::NTuple{2,Stagger}, bins = CellBins())

Values of `T` per linear bin in the flux patch of `Workspace(species, grid; flux = (shape,
stagger), bins)`: `3·M₁·M₂`, three components on the `M₁ × M₂` flux stencil of a bin
(`flux_bin_reach`): 108 at order 2 on nodes by cell, 75 by `NodeBins` of that shape and stagger.
"""
flux_patch_values(shape::Shape, st::NTuple{2,Stagger}, bins = CellBins()) = 3 * prod(flux_widths(shape, st, bins))
flux_patch_values(shape::Shape, st, bins = CellBins()) = throw(ArgumentError("flux scratch takes ρ's stagger, one per axis (N = 2 or 3), got $st"))

"""
    StencilMajor(), CellMajor()

The two storage orders of the flux patches of a workspace, with `K = 3·M₁·M₂` values per cell and
`n` cells. `StencilMajor()` is a `K × n` array, a cell's values one contiguous column
(`patch[k, cell]`); `CellMajor()` is `n × K`, one stencil entry of neighbouring cells at
neighbouring addresses (`patch[cell, k]`). The deposited current has the same bits with either
(T-803).
"""
struct StencilMajor end
struct CellMajor end

"""
    flux_patch_layout(backend)

The storage order of the flux patches that `Workspace(species, grid; flux)` takes on `backend`:
`StencilMajor()` on the CPU backend, where the patch kernel costs 1.4 to 2.7 times less with it
than with `CellMajor()` on a run of 14336 cells at 4 particles per cell
(`reports/wp-061-flux-deposition-cost.md`), and `CellMajor()` on every other backend, the order
WP-020 chose for a GPU; no GPU measurement of one against the other exists.
"""
flux_patch_layout(::CPU) = StencilMajor()
flux_patch_layout(::KernelAbstractions.Backend) = CellMajor()

"The size of the patch array of `values` values per cell and `cells` cells in a layout."
flux_patch_size(::StencilMajor, values, cells) = (values, cells)
flux_patch_size(::CellMajor, values, cells) = (cells, values)

"The index pair of value `k` of linear cell `L` in a layout."
@inline patch_index(::StencilMajor, k, L) = (k, L)
@inline patch_index(::CellMajor, k, L) = (L, k)

"Elementwise sum of two tuples of one length."
@inline tuple_add(a::NTuple{K,Any}, b::NTuple{K,Any}) where {K} = ntuple(k -> a[k] + b[k], Val(K))

# One work item per linear bin (cell, for `CellBins()`): its particles in slot order, each stencil entry summed from zero
# in a register tuple and written once into the cell's patch (value `k` of cell `L` at
# `patch_index(layout, k, L)`, the three components one after the other, Jx first). An empty cell
# writes zeros.
@kernel function flux_patch_2d_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), shape, st, frame, n::NTuple{2,Int32}, sx,
                                       quantity_z, layout, ::Val{K}) where {K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    ax = ntuple(_ -> zero(T), Val(K))
    ay, az = ax, ax
    @inbounds for i in seg_start[L]:seg_stop[L]
        _, _, (jx, jy, jz) = flux_stencil(s, i, shape, st, n, sx, quantity_z, frame)
        ax = tuple_add(ax, jx)
        ay = tuple_add(ay, jy)
        az = tuple_add(az, jz)
    end
    @inbounds for k in 1:K
        patch[patch_index(layout, k, L)...] = ax[k]
        patch[patch_index(layout, K + k, L)...] = ay[k]
        patch[patch_index(layout, 2K + k, L)...] = az[k]
    end
end

# One work item per node (guards included): the patches of the bins whose flux stencil reaches
# it, in increasing column-major linear index, summed from zero.
@kernel function flux_reduce_2d_kernel!(Jx, Jy, Jz, @Const(patch), layout, n::NTuple{2,Int32}, g::Int32, lo::NTuple{2,Int32},
                                        hi::NTuple{2,Int32}, M::NTuple{2,Int32})
    I = @index(Global, Cartesian)
    T = eltype(Jx)
    K = M[1] * M[2]
    j1, j2 = Int32(I[1]) - g, Int32(I[2]) - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for c2 in max(Int32(1), j2 - hi[2]):min(n[2], j2 - lo[2]), c1 in max(Int32(1), j1 - hi[1]):min(n[1], j1 - lo[1])
        L = c1 + (c2 - Int32(1)) * n[1]
        k = (j1 - c1 - lo[1]) + (j2 - c2 - lo[2]) * M[1] + Int32(1)
        ax += patch[patch_index(layout, k, L)...]
        ay += patch[patch_index(layout, K + k, L)...]
        az += patch[patch_index(layout, 2K + k, L)...]
    end
    @inbounds begin
        Jx[I] += ax
        Jy[I] += ay
        Jz[I] += az
    end
end

function _deposit_flux_2d!(::Reference, Jx, Jy, Jz, s, shape, st, grid, ws, sx, qz, workgroupsize)
    check_workspace(ws, s, grid)
    ws.flux == (shape, st) ||
        throw(ArgumentError("the Reference() deposit_flux! at N = 2 needs Workspace(species, grid; flux = (shape, stagger)) for this " *
                            "shape and stagger, $((shape, st)); the workspace was built with flux = $(ws.flux)"))
    need = flux_patch_values(shape, st, ws.bins)
    size(ws.flux_patch) == flux_patch_size(ws.flux_layout, need, prod(ncells(grid))) ||
        throw(ArgumentError("the Reference() deposit_flux! at N = 2 needs Workspace(species, grid; flux = (shape, stagger)) for this " *
                            "shape and stagger: $need values per cell in the layout $(ws.flux_layout), the workspace's patch array " *
                            "is $(size(ws.flux_patch))"))
    backend = get_backend(s)
    n = Int32.(ncells(grid))
    M = flux_widths(shape, st, ws.bins)
    r = (flux_bin_reach(ws.bins, shape, st[1], 1), flux_bin_reach(ws.bins, shape, st[2], 2))
    check_bin_guards(grid, r)
    lo, hi = (r[1][1], r[2][1]), (r[1][2], r[2][2])
    frame = ws.bins isa CellBins ? CellFrame() : BinFrame(ws.bins)
    item_launcher(flux_patch_2d_kernel!, backend, workgroupsize)(ws.flux_patch, s, ws.seg_start, ws.seg_stop, shape, st, frame, n, sx, qz,
                                                                 ws.flux_layout, Val(prod(M)); ndrange = prod(n))
    launcher(flux_reduce_2d_kernel!, backend, workgroupsize)(Jx, Jy, Jz, ws.flux_patch, ws.flux_layout, n, Int32(guards(grid)), lo, hi,
                                                             Int32.(M); ndrange = size(Jx))
    return Jx, Jy, Jz
end
