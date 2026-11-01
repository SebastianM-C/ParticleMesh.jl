# Charge-conserving flux deposition (Esirkepov's scheme) for N = 1.

"""
    flux_reach(shape, stagger) -> (lo, hi)

The nodes a particle's flux stencil can touch, relative to its current cell: the union of the
stencils at the old and the new position, for a move of less than one cell. One node wider than
`stencil_reach` on each side.
"""
@inline function flux_reach(shape::Shape, stagger::Stagger)
    lo, hi = stencil_reach(shape, stagger)
    return (Int32(lo - 1), Int32(hi + 1))
end

# The number of nodes of the flux stencil, hi - lo + 1 of `flux_reach`, as a static value.
@inline flux_width(::Shape{1}, ::Node) = Val(4)
@inline flux_width(::Shape{1}, ::Center) = Val(5)
@inline flux_width(::Shape{2}, ::Center) = Val(5)
@inline flux_width(::Shape{2}, ::Node) = Val(6)

"""
    flux_guard_width(shape, stagger)

Guard cells `deposit_flux!` needs on each side: `max(-lo, hi)` of `flux_reach`. One more than
`guard_width(shape)` for every pair but order 2 at `Center()`, where it is 2.
"""
function flux_guard_width(shape::Shape, stagger::Stagger)
    lo, hi = flux_reach(shape, stagger)
    return Int(max(-lo, hi))
end

# Running sums of a tuple, each one the serial left-to-right sum of its prefix (so every entry
# is the same bits as a sequential prefix sum), with the length static.
@inline function prefix_sums(t::NTuple{M,T}) where {M,T}
    return ntuple(Val(M)) do k
        acc = t[1]
        for m in 2:k
            acc += t[m]
        end
        acc
    end
end

"""
    flux_stencil(s, i, shape, stagger, n, sx, quantity_y, quantity_z) -> (lo, jx, jy, jz)

The device-level flux of particle `i`: its contributions to the nodes `cell + lo + k - 1`,
`k = 1:M`, of Jx, Jy and Jz, with `M` fixed by the shape and stagger (`flux_reach`). The frame
is the particle's current cell; its old cell, from `s.extras.old_cell`, is at most one cell
away, periodically on `n` cells. With `W_old`, `W_new` the shape weights at the old and new
position on that node range and `sx = Δx/Δt`:

    jx[k] = -(q·w·sx) · Σ_{m ≤ k} (W_new[m] - W_old[m])
    jy[k] = quantity_y(s, i) · (W_old[k] + W_new[k]) / 2,   jz likewise.

The prefix sum runs in increasing `k`, so the contributions of a particle are the same bits in
both modes.
"""
@inline function flux_stencil(s, i, shape::Shape, stagger::Stagger, n::Int32, sx, quantity_y, quantity_z)
    @inbounds begin
        c1 = s.cell[1][i]
        c0 = s.extras.old_cell[i]
        f1, w1 = shape_weights(shape, stagger, s.offset[1][i])
        f0, w0 = shape_weights(shape, stagger, s.extras.old_offset[i])
    end
    e = mod(c0 - c1 + Int32(1), n) - Int32(1)          # the old cell relative to the new one: -1, 0 or 1
    lo, _ = flux_reach(shape, stagger)
    M = flux_width(shape, stagger)
    Wn = ntuple(m -> pick(w1, lo + Int32(m - 1) - f1), M)
    Wo = ntuple(m -> pick(w0, lo + Int32(m - 1) - e - f0), M)
    S = prefix_sums(map(-, Wn, Wo))
    qws = s.q * s.w[i] * sx
    jx = map(x -> -qws * x, S)
    avg = map((a, b) -> (a + b) / 2, Wo, Wn)
    qy = quantity_y(s, i)
    qz = quantity_z(s, i)
    return (lo, jx, map(a -> qy * a, avg), map(a -> qz * a, avg))
end

@kernel function flux_atomic_kernel!(Jx, Jy, Jz, s, shape, stagger, n::Int32, g::Int32, sx, quantity_y, quantity_z)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        lo, jx, jy, jz = flux_stencil(s, i, shape, stagger, n, sx, quantity_y, quantity_z)
        base = s.cell[1][i] + lo + g
        for k in 1:length(jx)
            Atomix.@atomic Jx[base + Int32(k - 1)] += jx[k]
            Atomix.@atomic Jy[base + Int32(k - 1)] += jy[k]
            Atomix.@atomic Jz[base + Int32(k - 1)] += jz[k]
        end
    end
end

# One work item per node (guards included), as the Reference deposition: the cells whose flux
# stencil reaches the node in increasing order, within a cell its particles in slot order.
@kernel function flux_sorted_kernel!(Jx, Jy, Jz, s, @Const(seg_start), @Const(seg_stop), shape, stagger,
                                     n::Int32, g::Int32, lo::Int32, hi::Int32, sx, quantity_y, quantity_z)
    a = Int32(@index(Global, Linear))
    T = eltype(Jx)
    j = a - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for c in max(Int32(1), j - hi):min(n, j - lo)
        for i in seg_start[c]:seg_stop[c]
            l, jx, jy, jz = flux_stencil(s, i, shape, stagger, n, sx, quantity_y, quantity_z)
            k = j - c - l
            ax += pick(jx, k)
            ay += pick(jy, k)
            az += pick(jz, k)
        end
    end
    @inbounds begin
        Jx[a] += ax
        Jy[a] += ay
        Jz[a] += az
    end
end

"""
    deposit_flux!(mode, Jx, Jy, Jz, species, shape, stagger, grid, ws;
                  dt, quantity_y, quantity_z, workgroupsize = nothing)

Add the charge-conserving current of one step of every alive particle into `Jx`, `Jy`, `Jz`
(fields with the layout and guards of ρ), from the position before the step, read from the
species extras `old_cell::Int32` and `old_offset::T`, to the current one. With ρ deposited by
`deposit!(mode, ρ, species, shape, stagger, grid, ws)` (the default quantity q·w) before and
after the step, discrete continuity holds to roundoff on every node `j` (T-110):

    (ρ_new[j] - ρ_old[j]) / dt + (Jx[j] - Jx[j-1]) / Δx = 0.

Layout: `Jx[j]` is the face half a cell to the right of ρ's node `j`, stored at ρ's index `j`:
for ρ at `Node()` (x = jΔx) it sits at (j + 1/2)Δx, which is `Center()` index `j`; for ρ at
`Center()` (x = (j + 1/2)Δx) it sits at (j + 1)Δx, which is `Node()` index `j + 1`. The
continuity equation above, which T-110 checks for every shape and stagger, pins this offset.
`Jy[j]` and `Jz[j]` sit on ρ's nodes and are `quantity_y(species, i)` and
`quantity_z(species, i)` (isbits device callables, as `deposit!`'s quantity; q·w·u_y/γ for a
current) times the shape weights averaged over the old and the new position. Per particle, Jx
is `-(q·w·Δx/dt)` times the prefix sum of the change of its shape weights over the nodes in
increasing order (Esirkepov's formula, which in 1D reduces to that sum), so `Σ_j Jx[j] = Σ
q·w·(displacement)/dt` and `Σ_j Jy[j] = Σ quantity_y` (densities per node, as `deposit!`).
The flux is linear in the weights (A3) and a particle at rest deposits exactly zero Jx.

Preconditions: every particle moved by less than one cell since the old position was stored
(`|Δ| < Δx`), across a cell face and the periodic seam included; displacements of a cell or
more are outside the contract and not detected. `n ≥ 3` cells, and at least
`flux_guard_width(shape, stagger)` guard cells (checked). The guard convention is `deposit!`'s:
the contributions that cross the seam land in the guards, and `fold_guards!` on each of the
three fields moves them into the interior.

`Reference()` needs `sort_by_cell!(species, grid, ws)` after the move (the segments are by the
current cell; the extras follow the sort) and is bitwise deterministic: one work item per node
adds the contributions of the cells whose stencil reaches it, in increasing cell order and
within a cell in slot order, across reruns, thread counts and `workgroupsize` (T-113). `Fast()`
needs no sort: it adds with atomics, and for element types without them sorts the species by
cell itself and runs `Reference()` (`effective_mode`, #196); a caller that sorts anyway passes
`effective_mode(mode, T)`.
"""
function deposit_flux!(mode, Jx::AbstractVector{T}, Jy::AbstractVector{T}, Jz::AbstractVector{T}, s::Species{1},
                       shape::Shape, stagger::Stagger, grid::PeriodicGrid{1}, ws::Workspace;
                       dt, quantity_y, quantity_z, workgroupsize = nothing) where {T}
    haskey(s.extras, :old_cell) && haskey(s.extras, :old_offset) ||
        throw(ArgumentError("deposit_flux! reads the old position from the extras old_cell and old_offset"))
    guards(grid) >= flux_guard_width(shape, stagger) ||
        throw(ArgumentError("deposit_flux! needs $(flux_guard_width(shape, stagger)) guard cells for this shape and stagger, the grid has $(guards(grid))"))
    ncells(grid) >= 3 || throw(ArgumentError("deposit_flux! needs at least 3 cells"))
    sx = cellsize(grid) / dt
    fallback_sort!(mode, T, s, grid, ws, workgroupsize)
    _deposit_flux!(effective_mode(mode, T), Jx, Jy, Jz, s, shape, stagger, grid, ws, sx, quantity_y, quantity_z, workgroupsize)
end

function _deposit_flux!(::Fast, Jx, Jy, Jz, s, shape, stagger, grid, ws, sx, qy, qz, workgroupsize)
    launcher(flux_atomic_kernel!, get_backend(s), workgroupsize)(Jx, Jy, Jz, s, shape, stagger, Int32(ncells(grid)),
                                                                Int32(guards(grid)), sx, qy, qz; ndrange = capacity(s))
    return Jx, Jy, Jz
end

function _deposit_flux!(::Reference, Jx, Jy, Jz, s, shape, stagger, grid, ws, sx, qy, qz, workgroupsize)
    check_workspace(ws, s, grid)
    lo, hi = flux_reach(shape, stagger)
    launcher(flux_sorted_kernel!, get_backend(s), workgroupsize)(Jx, Jy, Jz, s, ws.seg_start, ws.seg_stop, shape, stagger,
                                                                Int32(ncells(grid)), Int32(guards(grid)), lo, hi, sx, qy, qz;
                                                                ndrange = length(Jx))
    return Jx, Jy, Jz
end
