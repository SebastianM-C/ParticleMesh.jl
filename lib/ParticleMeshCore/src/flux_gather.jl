# The transpose of the charge-conserving flux deposition: a gather with Esirkepov's weights along
# each particle's move, for N = 1, 2 and 3 (WP-071).

"""
    gather_flux!(ox, oy, oz, Ex, Ey, Ez, species, shape, stagger, grid; dt, workgroupsize = nothing)

The transpose of `deposit_flux!` with the same `shape`, ρ's `stagger` (one per axis for `N ≥ 2`)
and `dt`: for every alive particle `i`, `ox[i]`, `oy[i]`, `oz[i]` are the fields `Ex`, `Ey`, `Ez`
(ρ's layout and the current's offsets, guards filled) read with the weights by which particle `i`
deposits `Jx`, `Jy`, `Jz`, per unit of the quantity it deposits there. The quantity of a component
in space (Jx at N = 1, Jx and Jy at N = 2, all three at N = 3) is `q·w`, so those outputs are
`(Δx_d/dt)` times Esirkepov's weights summed against the field; the quantity of a component
out of space is the module's `quantity_y` or `quantity_z`, and those outputs are the field read
with the transverse weights alone (at N = 1 the average of the old and the new shape weights, at
N = 2 Esirkepov's out-of-plane weight). So, with `J` from `deposit_flux!` in either mode, its
guards folded, and `E` with its guards filled,

    Σ_j (Jx[j]·Ex[j] + Jy[j]·Ey[j] + Jz[j]·Ez[j]) = Σ_i (q·w_i)·(Σ_{d in space} o_d[i]) + Σ_i quantity_y(i)·oy[i] + Σ_i quantity_z(i)·oz[i]

(the last two sums at N = 1, the last at N = 2) to roundoff (T-1000). An output in space is the
power per unit charge: velocity times the field averaged along the move with Esirkepov's weights.
Divided by the move's velocity `Δ_d/dt` it is that weighted average. At N = 1 it is exactly the
field read with the shape one order lower at the faces and integrated along the move (T-1001); at
N ≥ 2 it is that straight-line average only for a move along one axis, and differs from it for an
oblique move. At order 2 its limit for a vanishing move, at least linear in the move, is the field
read with the shape one order lower along the component's own axis and the shape itself across it
(T-1002), the energy-conserving gather; at order 1 the shape one order lower is the discontinuous
top hat, and the limit holds only away from its jumps. A particle at rest gives
exactly zero in space (T-1003).

The weights are those of `deposit_flux!`'s `Fast()` path, in the particle's own frame
(`flux_axis_frame`, the frame `flux_axis_weights` gives for the cell): one work item per particle,
reading `(P + 1)(P + 2)^(N-1)` nodes for a component in space and `(P + 2)^N` for one out of space,
with nothing written but the particle's own outputs, so it needs no sort, no workspace and no
mode, and its bits do not depend on the launch shape. It covers the dependence on the deposited
quantity (where parameters enter, A3), not the one on the positions.

Preconditions are `deposit_flux!`'s: the old position in the extras of `flux_extras(T, N)`, a
move of less than one cell per axis (not detected) and at least
`maximum(flux_guard_width(shape, stagger))` guard cells (checked; a grid needs at least two guard
widths of cells per axis, so at least 4); the guards of `Ex`, `Ey`,
`Ez` filled (`fill_guards!`, the transpose of the fold). Dead slots are left as they are.
"""
function gather_flux!(ox::AbstractVector, oy::AbstractVector, oz::AbstractVector, Ex::AbstractArray{<:Any,N},
                      Ey::AbstractArray{<:Any,N}, Ez::AbstractArray{<:Any,N}, s::Species{N}, shape::Shape, stagger,
                      grid::PeriodicGrid{N}; dt, workgroupsize = nothing) where {N}
    stagger isa (N == 1 ? Stagger : NTuple{N,Stagger}) ||
        throw(ArgumentError("gather_flux! at N = $N takes ρ's stagger, $(N == 1 ? "one Stagger" : "one per axis"), got $stagger"))
    has_flux_extras(s) ||
        throw(ArgumentError("gather_flux! reads the old position from the extras of flux_extras(T, $N): $(keys(flux_extras(eltype(s), N)))"))
    gw = maximum(flux_guard_width(shape, stagger))
    guards(grid) >= gw ||
        throw(ArgumentError("gather_flux! needs $gw guard cells for this shape and stagger, the grid has $(guards(grid))"))
    sx = map(h -> h / dt, Tuple(cellsize(grid)))              # as deposit_flux! forms it
    launcher(flux_gather_kernel!, get_backend(s), workgroupsize)(ox, oy, oz, Ex, Ey, Ez, s, shape, stagger, Int32.(Tuple(ncells(grid))),
                                                                Int32(guards(grid)), sx; ndrange = capacity(s))
    return ox, oy, oz
end

@kernel function flux_gather_kernel!(ox, oy, oz, @Const(Ex), @Const(Ey), @Const(Ez), s, shape, stagger, n, g::Int32, sx)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        x, y, z = flux_gather_particle(s, i, Ex, Ey, Ez, shape, stagger, n, g, sx)
        ox[i] = x
        oy[i] = y
        oz[i] = z
    end
end

# Σ_k w[k]·a[base + k - 1] over the length of `w`, in increasing k.
@inline function dot_row(w::NTuple{M,Any}, a, base::Int32, ::Type{T}) where {M,T}
    acc = zero(T)
    @inbounds for k in 1:M
        acc += w[k] * a[base + Int32(k - 1)]
    end
    return acc
end

# N = 1: Jx = -(q·w·sx)·P on the P + 1 nodes of the prefix sums, Jy, Jz = quantity·A on P + 2.
@inline function flux_gather_particle(s, i, Ex, Ey, Ez, shape::Shape, st::Stagger, n::NTuple{1,Int32}, g::Int32, sx)
    T = eltype(s.offset[1])
    @inbounds f, Wo, Wn = flux_axis_frame(shape, st, s.cell[1][i], s.offset[1][i], s.extras.old_cell[i], s.extras.old_offset[i], n[1])
    p = Base.front(prefix_sums(map(-, Wn, Wo)))
    A = map((a, b) -> (a + b) / 2, Wo, Wn)
    @inbounds base = s.cell[1][i] + f + g
    return (-(sx[1] * dot_row(p, Ex, base, T)), dot_row(A, Ey, base, T), dot_row(A, Ez, base, T))
end

# Σ_{k1, k2} u[k1]·v[k2]·a[b1 + k1 - 1, b2 + k2 - 1], each row of axis 1 summed first.
@inline function dot_outer(u::NTuple{M1,Any}, v::NTuple{M2,Any}, a, b1::Int32, b2::Int32, ::Type{T}) where {M1,M2,T}
    acc = zero(T)
    @inbounds for k2 in 1:M2
        row = zero(T)
        for k1 in 1:M1
            row += u[k1] * a[b1 + Int32(k1 - 1), b2 + Int32(k2 - 1)]
        end
        acc += v[k2] * row
    end
    return acc
end

# N = 2: Jx = -(q·w·sx₁)·P₁ ⊗ A₂, Jy = -(q·w·sx₂)·A₁ ⊗ P₂, Jz = quantity·(A₁ ⊗ A₂ + ΔS₁ ⊗ ΔS₂/12).
@inline function flux_gather_particle(s, i, Ex, Ey, Ez, shape::Shape, st::NTuple{2,Stagger}, n::NTuple{2,Int32}, g::Int32, sx)
    T = eltype(s.offset[1])
    c = particle_cell(s, i, Val(2))
    δ = particle_offset(s, i, Val(2))
    c0 = old_cell_2d(s, i)
    δ0 = old_offset_2d(s, i)
    f1, Wo1, Wn1 = flux_axis_frame(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
    f2, Wo2, Wn2 = flux_axis_frame(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
    D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
    A1, A2 = map((a, b) -> (a + b) / 2, Wo1, Wn1), map((a, b) -> (a + b) / 2, Wo2, Wn2)
    p1, p2 = Base.front(prefix_sums(D1)), Base.front(prefix_sums(D2))
    b1, b2 = c[1] + f1 + g, c[2] + f2 + g
    x = -(sx[1] * dot_outer(p1, A2, Ex, b1, b2, T))
    y = -(sx[2] * dot_outer(A1, p2, Ey, b1, b2, T))
    z = dot_outer(A1, A2, Ez, b1, b2, T) + dot_outer(D1, D2, Ez, b1, b2, T) / 12
    return (x, y, z)
end

# Component C at N = 3: Σ over its (P + 1)(P + 2)² nodes of P_C ⊗ (A_a ⊗ A_b + ΔS_a ⊗ ΔS_b/12) times
# the field, a < b the other two axes; the caller scales by -sx_C.
@inline function dot_component_3d(::Val{C}, a3, fr, b::NTuple{3,Int32}, ::Type{T}) where {C,T}
    a, bb = transverse_axes(Val(C))
    _, Woc, Wnc = fr[C]
    _, Woa, Wna = fr[a]
    _, Wob, Wnb = fr[bb]
    p = Base.front(prefix_sums(map(-, Wnc, Woc)))
    Aa, Ab = map((x, y) -> (x + y) / 2, Woa, Wna), map((x, y) -> (x + y) / 2, Wob, Wnb)
    Da = map(-, Wna, Woa)
    Eb = map((x, y) -> (x - y) / 12, Wnb, Wob)
    D = ntuple(d -> d == C ? length(p) : length(Aa), Val(3))
    acc = zero(T)
    @inbounds for K in CartesianIndices(map(Base.OneTo, D))
        m = Tuple(K)
        w = p[m[C]] * (Aa[m[a]] * Ab[m[bb]] + Da[m[a]] * Eb[m[bb]])
        acc += w * a3[b[1] + Int32(m[1] - 1), b[2] + Int32(m[2] - 1), b[3] + Int32(m[3] - 1)]
    end
    return acc
end

# N = 3: every component in space, J_C = -(q·w·sx_C)·P_C ⊗ (A_a ⊗ A_b + ΔS_a ⊗ ΔS_b/12).
@inline function flux_gather_particle(s, i, Ex, Ey, Ez, shape::Shape, st::NTuple{3,Stagger}, n::NTuple{3,Int32}, g::Int32, sx)
    T = eltype(s.offset[1])
    c = particle_cell(s, i, Val(3))
    δ = particle_offset(s, i, Val(3))
    c0 = old_cell_3d(s, i)
    δ0 = old_offset_3d(s, i)
    # Each axis written out: `st` may mix staggers, and a closure over the axis would index it dynamically.
    fr = (flux_axis_frame(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1]), flux_axis_frame(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2]),
          flux_axis_frame(shape, st[3], c[3], δ[3], c0[3], δ0[3], n[3]))
    b = (c[1] + fr[1][1] + g, c[2] + fr[2][1] + g, c[3] + fr[3][1] + g)
    x = -(sx[1] * dot_component_3d(Val(1), Ex, fr, b, T))
    y = -(sx[2] * dot_component_3d(Val(2), Ey, fr, b, T))
    z = -(sx[3] * dot_component_3d(Val(3), Ez, fr, b, T))
    return (x, y, z)
end
