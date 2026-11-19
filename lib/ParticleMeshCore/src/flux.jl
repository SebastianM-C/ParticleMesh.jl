# Charge-conserving flux deposition (Esirkepov's scheme) for N = 1; N = 2 is in flux2d.jl.

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

"""
    flux_axis_weights(shape, stagger, c, δ, c0, δ0, n) -> (lo, Wo, Wn)

One axis of a particle's flux stencil: the old and new shape weights on the `M` nodes
`c + lo + m - 1`, `m = 1:M`, of `flux_reach` (zero where a stencil does not reach), with the
current cell `c` as the frame; the old cell `c0` is at most one cell away, periodically on `n`.
"""
@inline function flux_axis_weights(shape::Shape, stagger::Stagger, c::Int32, δ, c0::Int32, δ0, n::Int32)
    lo, _ = flux_reach(shape, stagger)
    return (lo, flux_axis_weights_at(shape, stagger, c, δ, c0, δ0, n, lo, flux_width(shape, stagger))...)
end

# The old and new weights `(Wo, Wn)` of one axis on the `M` nodes `c + first + m - 1`, `first`
# relative to the particle's current cell `c`.
@inline function flux_axis_weights_at(shape::Shape, stagger::Stagger, c::Int32, δ, c0::Int32, δ0, n::Int32, first::Int32,
                                      M::Val)
    f1, w1 = shape_weights(shape, stagger, δ)
    f0, w0 = shape_weights(shape, stagger, δ0)
    e = mod(c0 - c + Int32(1), n) - Int32(1)           # the old cell relative to the new one: -1, 0 or 1
    Wn = ntuple(m -> pick(w1, first + Int32(m - 1) - f1), M)
    Wo = ntuple(m -> pick(w0, first + Int32(m - 1) - e - f0), M)
    return (Wo, Wn)
end

# The bins' flux reach needs the guards of `flux_guard_width` for bins of the flux's shape and
# stagger, and one more where a bin's shift does not fit the flux's stagger (`deposit_reach`).
function check_bin_guards(grid, reach)
    gw = maximum(r -> max(-r[1], r[2]), reach)
    guards(grid) >= gw ||
        throw(ArgumentError("the Reference() deposit_flux! with these bins needs $gw guard cells, the grid has $(guards(grid)); " *
                            "NodeBins of the flux's own shape and stagger need no more than the cells"))
    return nothing
end

"""
    flux_bin_reach(bins, shape, stagger, d) -> (lo, hi)

The nodes a particle's flux stencil can touch on axis `d`, relative to its bin (`NodeBins` or
`CellBins`, whose bin is the cell): `deposit_reach` one node wider on each side, for a move of
less than one cell. `flux_reach` for `CellBins()`; at order 2 on `Node()` with the bins of that
shape and stagger, 5 nodes where a cell's are 6.
"""
@inline function flux_bin_reach(bins, shape::Shape, stagger::Stagger, d::Int)
    lo, hi = deposit_reach(bins, shape, stagger, d)
    return (lo - Int32(1), hi + Int32(1))
end

# The number of nodes of `flux_bin_reach` as a static value.
@inline flux_bin_width(::CellBins, shape::Shape, stagger::Stagger, d::Int) = flux_width(shape, stagger)
@inline flux_bin_width(b::NodeBins, shape::Shape{P}, stagger::Stagger, d::Int) where {P} =
    b.shape === shape && b.stagger[d] === stagger ? Val(P + 3) : Val(_width(flux_width(shape, stagger)) + _width(bin_hmax(b, d)))

"""
    flux_axis_frame(shape::Shape{P}, stagger, c, δ, c0, δ0, n) -> (f, Wo, Wn)

One axis of a particle's flux stencil in the particle's own frame: the old and new shape weights
on the `P + 2` nodes `c + f + m - 1`, `m = 1:P+2`, the union of the old and the new stencil
(zero where one does not reach), with `f` the first node of either relative to the current cell
`c`. The old cell `c0` is at most one cell away, periodically on `n`, so the two stencils' first
nodes differ by at most one. On the nodes the two frames share, the weights are those of
`flux_axis_weights`, which are zero on every other node of the cell's frame.
"""
@inline function flux_axis_frame(shape::Shape{P}, stagger::Stagger, c::Int32, δ, c0::Int32, δ0, n::Int32) where {P}
    f1, w1 = shape_weights(shape, stagger, δ)
    fo, w0 = shape_weights(shape, stagger, δ0)
    f0 = fo + mod(c0 - c + Int32(1), n) - Int32(1)     # the old stencil's first node relative to the current cell
    f = min(f1, f0)
    Wn = ntuple(m -> pick(w1, f + Int32(m - 1) - f1), Val(P + 2))
    Wo = ntuple(m -> pick(w0, f + Int32(m - 1) - f0), Val(P + 2))
    return (f, Wo, Wn)
end

"""
    flux_axis_slopes(shape::Shape{P}, stagger, c, δ, c0, δ0, n) -> (dWo, dWn)

The derivatives of `flux_axis_frame`'s weights on the same `P + 2` nodes: `dWo[m] = ∂Wo[m]/∂δ0`
and `dWn[m] = ∂Wn[m]/∂δ` (`shape_slopes` placed as the weights are), each zero where its
stencil does not reach. The reverse rules of the flux deposition read them (WP-077).
"""
@inline function flux_axis_slopes(shape::Shape{P}, stagger::Stagger, c::Int32, δ, c0::Int32, δ0, n::Int32) where {P}
    f1, d1 = shape_slopes(shape, stagger, δ)
    fo, d0 = shape_slopes(shape, stagger, δ0)
    f0 = fo + mod(c0 - c + Int32(1), n) - Int32(1)
    f = min(f1, f0)
    dWn = ntuple(m -> pick(d1, f + Int32(m - 1) - f1), Val(P + 2))
    dWo = ntuple(m -> pick(d0, f + Int32(m - 1) - f0), Val(P + 2))
    return (dWo, dWn)
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
    @inbounds lo, Wo, Wn = flux_axis_weights(shape, stagger, s.cell[1][i], s.offset[1][i], s.extras.old_cell[i],
                                             s.extras.old_offset[i], n)
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

# One work item per node (guards included), as the Reference deposition: the bins (the cells, for
# `CellBins()`) whose flux stencil reaches the node in increasing order, within a bin its particles
# in slot order.
@kernel function flux_sorted_kernel!(Jx, Jy, Jz, s, @Const(seg_start), @Const(seg_stop), shape, stagger, bins,
                                     n::Int32, g::Int32, lo::Int32, hi::Int32, sx, quantity_y, quantity_z)
    a = Int32(@index(Global, Linear))
    T = eltype(Jx)
    j = a - g
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for b in max(Int32(1), j - hi):min(n, j - lo)
        for i in seg_start[b]:seg_stop[b]
            l, jx, jy, jz = flux_stencil(s, i, shape, stagger, n, sx, quantity_y, quantity_z)
            k = j - (b - bin_shift(bins, 1, s.offset[1][i])) - l       # the particle's cell, unwrapped from its bin
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
current bin, `ws.bins`: the cell for `CellBins()`, or `NodeBins`, which may take one more guard
cell, checked; the extras follow the sort) and is bitwise deterministic: one work item per node
adds the contributions of the bins whose stencil reaches it, in increasing bin order and
within a bin in slot order, across reruns, thread counts and `workgroupsize` (T-113). `Fast()`
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
    lo, hi = flux_bin_reach(ws.bins, shape, stagger, 1)
    check_bin_guards(grid, ((lo, hi),))
    launcher(flux_sorted_kernel!, get_backend(s), workgroupsize)(Jx, Jy, Jz, s, ws.seg_start, ws.seg_stop, shape, stagger, ws.bins,
                                                                Int32(ncells(grid)), Int32(guards(grid)), lo, hi, sx, qy, qz;
                                                                ndrange = length(Jx))
    return Jx, Jy, Jz
end
