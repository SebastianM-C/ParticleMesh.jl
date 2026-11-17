# Prototype kernels of the deposition study (scripts/deposition/README.md): alternatives to the
# engine's sort and to its 2D flux deposition, and a 3D flux deposition, written as
# KernelAbstractions kernels generic in the element type so that one source runs on the CPU and on
# a GPU. Nothing here is engine code or tested as such; check.jl compares every variant with the
# engine (2D) and with discrete continuity (2D and 3D), and bench2d.jl / bench3d.jl time them.
using ParticleMesh, KernelAbstractions

const PMC = ParticleMesh.ParticleMeshCore
const AK = PMC.AK
const Atomix = PMC.Atomix

# --- the per-axis frame of a particle's flux stencil ---------------------------------------------

"The engine's frame: the nodes `flux_reach` of the particle's current cell, 6 per axis at order 2 on nodes."
struct CellFrame end
"The particle's own frame: the union of its old and new stencil, `order + 2` nodes per axis (4 at order 2)."
struct MinFrame end

"`(f, Wo, Wn)`: the old and new shape weights of one axis on the nodes `c + f + m - 1`, `m = 1:M`."
@inline axis_frame(::CellFrame, shape, stg, c::Int32, δ, c0::Int32, δ0, n::Int32) = PMC.flux_axis_weights(shape, stg, c, δ, c0, δ0, n)
@inline function axis_frame(::MinFrame, shape::PMC.Shape{P}, stg, c::Int32, δ, c0::Int32, δ0, n::Int32) where {P}
    f1, w1 = PMC.shape_weights(shape, stg, δ)
    fo, w0 = PMC.shape_weights(shape, stg, δ0)
    f0 = fo + mod(c0 - c + Int32(1), n) - Int32(1)      # the old stencil's first node in the new cell's frame
    f = min(f1, f0)                                     # a move of less than a cell: |f1 - f0| ≤ 1
    Wn = ntuple(m -> PMC.pick(w1, f + Int32(m - 1) - f1), Val(P + 2))
    Wo = ntuple(m -> PMC.pick(w0, f + Int32(m - 1) - f0), Val(P + 2))
    return (f, Wo, Wn)
end

# Columns of a flux component along its own axis: the prefix sum of the last node is the sum of
# all weight differences, zero to roundoff, and the minimal frame leaves it out (as PIConGPU,
# WarpX and Smilei do); the cell frame keeps the engine's.
@inline flux_cols(::CellFrame, M) = M
@inline flux_cols(::MinFrame, M) = M - 1

@inline avg(a, b) = (a + b) / 2

# --- where a particle's contributions go ---------------------------------------------------------

"Plain adds into the three fields (one writer per node at a time: the caller's obligation)."
struct GridSink{A}
    Jx::A
    Jy::A
    Jz::A
    g::Int32
end
"Atomic adds into the three fields."
struct AtomicSink{A}
    Jx::A
    Jy::A
    Jz::A
    g::Int32
end
"Plain adds into the buffer of one tile, `buf[l..., component, tile]` with `l = node - o`."
struct TileSink{B,N}
    buf::B
    tile::Int32
    o::NTuple{N,Int32}
end

@inline component(k, ::Val{1}) = k.Jx
@inline component(k, ::Val{2}) = k.Jy
@inline component(k, ::Val{3}) = k.Jz

@inline function add!(k::GridSink, C::Val, j::NTuple{N,Int32}, v) where {N}
    A = component(k, C)
    I = CartesianIndex(map(x -> Int(x + k.g), j))
    @inbounds A[I] += v
    return nothing
end
@inline function add!(k::AtomicSink, C::Val, j::NTuple{N,Int32}, v) where {N}
    A = component(k, C)
    I = CartesianIndex(map(x -> Int(x + k.g), j))
    @inbounds Atomix.@atomic A[I] += v
    return nothing
end
@inline function add!(k::TileSink{B,N}, ::Val{C}, j::NTuple{N,Int32}, v) where {B,N,C}
    I = CartesianIndex(map((x, o) -> Int(x - o), j, k.o)..., C, Int(k.tile))
    @inbounds k.buf[I] += v
    return nothing
end

# --- one particle's flux, scattered into a sink --------------------------------------------------

@inline old_cell_3d(s, i) = @inbounds (s.extras.old_cell_1[i], s.extras.old_cell_2[i], s.extras.old_cell_3[i])
@inline old_offset_3d(s, i) = @inbounds (s.extras.old_offset_1[i], s.extras.old_offset_2[i], s.extras.old_offset_3[i])

"The extras a 3D flux deposition reads the old position from (the engine's `flux_extras` stops at N = 2)."
flux_extras_3d(::Type{T}) where {T} = (old_cell_1 = Int32, old_cell_2 = Int32, old_cell_3 = Int32,
                                       old_offset_1 = T, old_offset_2 = T, old_offset_3 = T)

# N = 2: the engine's stencil (flux2d.jl) with each axis scaled once: Jx = p₁ ⊗ A₂,
# Jy = A₁ ⊗ p₂, Jz = (qz·A₁) ⊗ A₂ + (qz·ΔS₁/12) ⊗ ΔS₂, with p_d = -(q·w·sx_d)·prefix(ΔS_d).
@inline function scatter!(sink, frame, s, i, shape, st::NTuple{2,PMC.Stagger}, n::NTuple{2,Int32}, sx, quantity_z)
    c = PMC.particle_cell(s, i, Val(2))
    δ = PMC.particle_offset(s, i, Val(2))
    c0 = PMC.old_cell_2d(s, i)
    δ0 = PMC.old_offset_2d(s, i)
    f1, Wo1, Wn1 = axis_frame(frame, shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
    f2, Wo2, Wn2 = axis_frame(frame, shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
    D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
    A1, A2 = map(avg, Wo1, Wn1), map(avg, Wo2, Wn2)
    @inbounds qw = s.q * s.w[i]
    k1, k2 = -(qw * sx[1]), -(qw * sx[2])
    p1 = map(x -> k1 * x, PMC.prefix_sums(D1))
    p2 = map(x -> k2 * x, PMC.prefix_sums(D2))
    qz = quantity_z(s, i)
    a1 = map(x -> qz * x, A1)
    d1 = map(x -> qz * x / 12, D1)
    M1, M2 = length(Wo1), length(Wo2)
    b1, b2 = c[1] + f1 - Int32(1), c[2] + f2 - Int32(1)
    @inbounds for m2 in 1:M2
        j2 = b2 + Int32(m2)
        for m1 in 1:flux_cols(frame, M1)
            add!(sink, Val(1), (b1 + Int32(m1), j2), p1[m1] * A2[m2])
        end
        if m2 <= flux_cols(frame, M2)
            for m1 in 1:M1
                add!(sink, Val(2), (b1 + Int32(m1), j2), A1[m1] * p2[m2])
            end
        end
        for m1 in 1:M1
            add!(sink, Val(3), (b1 + Int32(m1), j2), a1[m1] * A2[m2] + d1[m1] * D2[m2])
        end
    end
    return nothing
end

# N = 3: Esirkepov's weights W_x = ΔS₁·(S0₂S0₃ + ΔS₂S0₃/2 + S0₂ΔS₃/2 + ΔS₂ΔS₃/3), which is
# ΔS₁·(A₂A₃ + ΔS₂ΔS₃/12) with A = S0 + ΔS/2, summed along the flux axis:
# Jx[m₁,m₂,m₃] = p₁[m₁]·(A₂[m₂]A₃[m₃] + ΔS₂[m₂]ΔS₃[m₃]/12), and cyclically.
@inline function scatter!(sink, frame, s, i, shape, st::NTuple{3,PMC.Stagger}, n::NTuple{3,Int32}, sx, ::Nothing)
    c = PMC.particle_cell(s, i, Val(3))
    δ = PMC.particle_offset(s, i, Val(3))
    c0 = old_cell_3d(s, i)
    δ0 = old_offset_3d(s, i)
    f1, Wo1, Wn1 = axis_frame(frame, shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
    f2, Wo2, Wn2 = axis_frame(frame, shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
    f3, Wo3, Wn3 = axis_frame(frame, shape, st[3], c[3], δ[3], c0[3], δ0[3], n[3])
    D1, D2, D3 = map(-, Wn1, Wo1), map(-, Wn2, Wo2), map(-, Wn3, Wo3)
    A1, A2, A3 = map(avg, Wo1, Wn1), map(avg, Wo2, Wn2), map(avg, Wo3, Wn3)
    @inbounds qw = s.q * s.w[i]
    k1, k2, k3 = -(qw * sx[1]), -(qw * sx[2]), -(qw * sx[3])
    p1 = map(x -> k1 * x, PMC.prefix_sums(D1))
    p2 = map(x -> k2 * x, PMC.prefix_sums(D2))
    p3 = map(x -> k3 * x, PMC.prefix_sums(D3))
    E2, E3 = map(x -> x / 12, D2), map(x -> x / 12, D3)
    M1, M2, M3 = length(Wo1), length(Wo2), length(Wo3)
    b1, b2, b3 = c[1] + f1 - Int32(1), c[2] + f2 - Int32(1), c[3] + f3 - Int32(1)
    @inbounds for m3 in 1:M3, m2 in 1:M2
        t = A2[m2] * A3[m3] + D2[m2] * E3[m3]
        for m1 in 1:flux_cols(frame, M1)
            add!(sink, Val(1), (b1 + Int32(m1), b2 + Int32(m2), b3 + Int32(m3)), p1[m1] * t)
        end
    end
    @inbounds for m3 in 1:M3, m1 in 1:M1
        t = A1[m1] * A3[m3] + D1[m1] * E3[m3]
        for m2 in 1:flux_cols(frame, M2)
            add!(sink, Val(2), (b1 + Int32(m1), b2 + Int32(m2), b3 + Int32(m3)), p2[m2] * t)
        end
    end
    @inbounds for m2 in 1:M2, m1 in 1:M1
        t = A1[m1] * A2[m2] + D1[m1] * E2[m2]
        for m3 in 1:flux_cols(frame, M3)
            add!(sink, Val(3), (b1 + Int32(m1), b2 + Int32(m2), b3 + Int32(m3)), p3[m3] * t)
        end
    end
    return nothing
end

# --- atomics, one work item per particle ---------------------------------------------------------

@kernel function atomic_flux_kernel!(Jx, Jy, Jz, s, frame, shape, st, n, g::Int32, sx, quantity_z)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == PMC.ALIVE
        scatter!(AtomicSink(Jx, Jy, Jz, g), frame, s, i, shape, st, n, sx, quantity_z)
    end
end

"The flux of every alive particle by atomic adds (no sort needed), in the frame `frame`."
function deposit_atomic!(J, s, frame, shape, st, grid, sx, quantity_z; workgroupsize = nothing)
    PMC.launcher(atomic_flux_kernel!, get_backend(s), workgroupsize)(J..., s, frame, shape, st, Int32.(grid.n), Int32(grid.g), sx, quantity_z;
                                                                    ndrange = PMC.capacity(s))
    return J
end

# --- tiles: S cells per axis, one work item per tile ---------------------------------------------

# The particles of the tile whose first cell is `lo`, cell by cell in increasing linear index,
# within a cell in slot order, scattered into `sink`.
@inline function tile_scatter!(sink, lo::NTuple{2,Int32}, ::Val{S}, n::NTuple{2,Int32}, seg_start, seg_stop, frame, s, shape, st, sx, qz) where {S}
    @inbounds for c2 in lo[2]:min(lo[2] + Int32(S - 1), n[2]), c1 in lo[1]:min(lo[1] + Int32(S - 1), n[1])
        L = c1 + (c2 - Int32(1)) * n[1]
        for i in seg_start[L]:seg_stop[L]
            scatter!(sink, frame, s, i, shape, st, n, sx, qz)
        end
    end
    return nothing
end
@inline function tile_scatter!(sink, lo::NTuple{3,Int32}, ::Val{S}, n::NTuple{3,Int32}, seg_start, seg_stop, frame, s, shape, st, sx, qz) where {S}
    @inbounds for c3 in lo[3]:min(lo[3] + Int32(S - 1), n[3]), c2 in lo[2]:min(lo[2] + Int32(S - 1), n[2]),
                  c1 in lo[1]:min(lo[1] + Int32(S - 1), n[1])
        L = c1 + ((c2 - Int32(1)) + (c3 - Int32(1)) * n[2]) * n[1]
        for i in seg_start[L]:seg_stop[L]
            scatter!(sink, frame, s, i, shape, st, n, sx, qz)
        end
    end
    return nothing
end

# One work item per tile: zero the tile's buffer, then add its particles serially. The buffer
# holds the nodes `lo + rlo : lo + S - 1 + rhi` per axis (`rlo`, `rhi` of `flux_reach`), E = S + 5
# of them at order 2 on nodes, three components.
@kernel function tile_flux_kernel!(buf, s, @Const(seg_start), @Const(seg_stop), frame, shape, st, n::NTuple{N,Int32},
                                   nt::NTuple{N,Int32}, rlo::NTuple{N,Int32}, len::Int, sx, quantity_z, ::Val{S}) where {N,S}
    I = @index(Global, NTuple)
    t = map(Int32, I)
    τ = PMC.linear_cell(t, nt)
    T = eltype(buf)
    base = (Int(τ) - 1) * len
    @inbounds for k in 1:len
        buf[base + k] = zero(T)
    end
    lo = map(x -> (x - Int32(1)) * Int32(S) + Int32(1), t)
    sink = TileSink(buf, τ, map((l, r) -> l + r - Int32(1), lo, rlo))
    tile_scatter!(sink, lo, Val(S), n, seg_start, seg_stop, frame, s, shape, st, sx, quantity_z)
end

@inline function reduce_tiles(buf, j::NTuple{2,Int32}, ta, tb, nt, rlo, ::Val{S}) where {S}
    T = eltype(buf)
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for t2 in ta[2]:tb[2], t1 in ta[1]:tb[1]
        τ = t1 + (t2 - Int32(1)) * nt[1]
        l1 = j[1] - (t1 - Int32(1)) * Int32(S) - rlo[1]
        l2 = j[2] - (t2 - Int32(1)) * Int32(S) - rlo[2]
        ax += buf[l1, l2, 1, τ]
        ay += buf[l1, l2, 2, τ]
        az += buf[l1, l2, 3, τ]
    end
    return (ax, ay, az)
end
@inline function reduce_tiles(buf, j::NTuple{3,Int32}, ta, tb, nt, rlo, ::Val{S}) where {S}
    T = eltype(buf)
    ax, ay, az = zero(T), zero(T), zero(T)
    @inbounds for t3 in ta[3]:tb[3], t2 in ta[2]:tb[2], t1 in ta[1]:tb[1]
        τ = t1 + ((t2 - Int32(1)) + (t3 - Int32(1)) * nt[2]) * nt[1]
        l1 = j[1] - (t1 - Int32(1)) * Int32(S) - rlo[1]
        l2 = j[2] - (t2 - Int32(1)) * Int32(S) - rlo[2]
        l3 = j[3] - (t3 - Int32(1)) * Int32(S) - rlo[3]
        ax += buf[l1, l2, l3, 1, τ]
        ay += buf[l1, l2, l3, 2, τ]
        az += buf[l1, l2, l3, 3, τ]
    end
    return (ax, ay, az)
end

# One work item per node (guards included): the buffers of the tiles that hold it, in increasing
# linear tile index, summed from zero. Tile `t` of an axis holds the nodes
# `(t-1)S + 1 + rlo : tS + rhi`.
@kernel function tile_reduce_kernel!(Jx, Jy, Jz, @Const(buf), nt::NTuple{N,Int32}, g::Int32, rlo::NTuple{N,Int32}, rhi::NTuple{N,Int32},
                                     ::Val{S}) where {N,S}
    I = @index(Global, Cartesian)
    j = map(x -> Int32(x) - g, Tuple(I))
    ta = ntuple(d -> max(Int32(1), cld(j[d] - rhi[d], Int32(S))), Val(N))
    tb = ntuple(d -> min(nt[d], fld(j[d] - rlo[d] - Int32(1), Int32(S)) + Int32(1)), Val(N))
    ax, ay, az = reduce_tiles(buf, j, ta, tb, nt, rlo, Val(S))
    @inbounds begin
        Jx[I] += ax
        Jy[I] += ay
        Jz[I] += az
    end
end

"The scratch of the tile deposition: `S` cells per axis per tile, one buffer of `(S + reach)^N × 3` values per tile."
struct TileBuffers{B,N,S}
    buf::B
    nt::NTuple{N,Int32}
    rlo::NTuple{N,Int32}
    rhi::NTuple{N,Int32}
end

function TileBuffers(backend, ::Type{T}, grid::PMC.PeriodicGrid{N}, shape, st::NTuple{N,PMC.Stagger}, S::Integer) where {T,N}
    r = map(x -> PMC.flux_reach(shape, x), st)
    rlo, rhi = map(first, r), map(last, r)
    grid.g >= maximum(map((a, b) -> max(-a, b), rlo, rhi)) || throw(ArgumentError("the grid has too few guards for the flux reach"))
    nt = map(m -> Int32(cld(m, S)), grid.n)
    E = ntuple(d -> S + Int(rhi[d] - rlo[d]), Val(N))
    buf = KernelAbstractions.zeros(backend, T, E..., 3, prod(nt))
    return TileBuffers{typeof(buf),N,S}(buf, nt, rlo, rhi)
end

"""
The flux of a species sorted by cell, by tiles: every tile sums its particles serially into its
own buffer, then every node sums the buffers that hold it in a fixed order. No atomics, and the
bits do not depend on the launch shape (they do depend on `S`).
"""
function deposit_tiles!(J, tb::TileBuffers{B,N,S}, s, frame, shape, st, grid, seg_start, seg_stop, sx, quantity_z;
                        workgroupsize = nothing, launch = PMC.launcher) where {B,N,S}
    backend = get_backend(s)
    len = div(length(tb.buf), prod(tb.nt))
    launch(tile_flux_kernel!, backend, workgroupsize)(tb.buf, s, seg_start, seg_stop, frame, shape, st, Int32.(grid.n), tb.nt, tb.rlo, len, sx,
                                                      quantity_z, Val(S); ndrange = Int.(tb.nt))
    PMC.launcher(tile_reduce_kernel!, backend, workgroupsize)(J..., tb.buf, tb.nt, Int32(grid.g), tb.rlo, tb.rhi, Val(S); ndrange = size(J[1]))
    return J
end

# --- coloured tiles: no buffer, 2^N passes -------------------------------------------------------

# One work item per tile of one colour (every second tile per axis, starting at 1 + colour): its
# particles added straight into the fields. Tiles of one colour are 2S cells apart and write
# S + reach nodes, so with S ≥ reach they share no node.
@kernel function colour_flux_kernel!(Jx, Jy, Jz, s, @Const(seg_start), @Const(seg_stop), frame, shape, st, n::NTuple{N,Int32}, g::Int32,
                                     colour::NTuple{N,Int32}, sx, quantity_z, ::Val{S}) where {N,S}
    I = @index(Global, NTuple)
    t = ntuple(d -> Int32(2) * (Int32(I[d]) - Int32(1)) + Int32(1) + colour[d], Val(N))
    lo = map(x -> (x - Int32(1)) * Int32(S) + Int32(1), t)
    tile_scatter!(GridSink(Jx, Jy, Jz, g), lo, Val(S), n, seg_start, seg_stop, frame, s, shape, st, sx, quantity_z)
end

"""
The flux of a species sorted by cell, by coloured tiles: 2^N passes, each over the tiles of one
colour, which write disjoint nodes without atomics. No scratch; the bits depend on `S` and on
the order of the colours, not on the launch shape.
"""
function deposit_coloured!(J, s, frame, shape, st::NTuple{N,PMC.Stagger}, grid, seg_start, seg_stop, sx, quantity_z, S::Integer;
                           workgroupsize = nothing, launch = PMC.launcher) where {N}
    r = map(x -> PMC.flux_reach(shape, x), st)
    S >= maximum(map(x -> Int(x[2] - x[1]), r)) || throw(ArgumentError("coloured tiles need S ≥ the flux reach (5 at order 2 on nodes), got $S"))
    backend = get_backend(s)
    nt = map(m -> cld(m, S), grid.n)
    for colour in Iterators.product(ntuple(_ -> 0:1, Val(N))...)
        range = ntuple(d -> length((1 + colour[d]):2:nt[d]), Val(N))
        any(iszero, range) && continue
        launch(colour_flux_kernel!, backend, workgroupsize)(J..., s, seg_start, seg_stop, frame, shape, st, Int32.(grid.n), Int32(grid.g),
                                                            Int32.(colour), sx, quantity_z, Val(S); ndrange = range)
    end
    return J
end

# --- variants of the engine's per-cell patch (N = 2) ---------------------------------------------

"The engine's stencil with each axis scaled once and no division by `T(12)` (issue 454)."
@inline function factored_stencil(s, i, shape, st::NTuple{2,PMC.Stagger}, n::NTuple{2,Int32}, sx, quantity_z)
    c = PMC.particle_cell(s, i, Val(2))
    δ = PMC.particle_offset(s, i, Val(2))
    c0 = PMC.old_cell_2d(s, i)
    δ0 = PMC.old_offset_2d(s, i)
    _, Wo1, Wn1 = PMC.flux_axis_weights(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
    _, Wo2, Wn2 = PMC.flux_axis_weights(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
    D1, D2 = map(-, Wn1, Wo1), map(-, Wn2, Wo2)
    A1, A2 = map(avg, Wo1, Wn1), map(avg, Wo2, Wn2)
    @inbounds qw = s.q * s.w[i]
    k1, k2 = -(qw * sx[1]), -(qw * sx[2])
    p1 = map(x -> k1 * x, PMC.prefix_sums(D1))
    p2 = map(x -> k2 * x, PMC.prefix_sums(D2))
    qz = quantity_z(s, i)
    a1 = map(x -> qz * x, A1)
    d1 = map(x -> qz * x / 12, D1)
    jx = PMC.outer((p, a) -> p * a, p1, A2)
    jy = PMC.outer((a, p) -> a * p, A1, p2)
    jz = PMC.outer2(((a, d), (b, e)) -> a * b + d * e, (a1, d1), (A2, D2))
    return (jx, jy, jz)
end

@inline engine_stencil(s, i, shape, st, n, sx, quantity_z) = PMC.flux_stencil(s, i, shape, st, n, sx, quantity_z)[3]

# The engine's patch kernel (flux2d.jl) with the stencil as an argument.
@kernel function patch_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), stencil, shape, st, n::NTuple{2,Int32}, sx, quantity_z,
                               layout, ::Val{K}) where {K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    ax = ntuple(_ -> zero(T), Val(K))
    ay, az = ax, ax
    @inbounds for i in seg_start[L]:seg_stop[L]
        jx, jy, jz = stencil(s, i, shape, st, n, sx, quantity_z)
        ax = PMC.tuple_add(ax, jx)
        ay = PMC.tuple_add(ay, jy)
        az = PMC.tuple_add(az, jz)
    end
    @inbounds for k in 1:K
        patch[PMC.patch_index(layout, k, L)...] = ax[k]
        patch[PMC.patch_index(layout, K + k, L)...] = ay[k]
        patch[PMC.patch_index(layout, 2K + k, L)...] = az[k]
    end
end

# One component per launch: a third of the accumulators (36 instead of 108 at order 2 on nodes)
# for three passes over the particles.
@kernel function patch_component_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), ::Val{C}, shape, st, n::NTuple{2,Int32}, sx,
                                         quantity_z, layout, ::Val{K}) where {C,K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    acc = ntuple(_ -> zero(T), Val(K))
    @inbounds for i in seg_start[L]:seg_stop[L]
        acc = PMC.tuple_add(acc, factored_stencil(s, i, shape, st, n, sx, quantity_z)[C])
    end
    @inbounds for k in 1:K
        patch[PMC.patch_index(layout, (C - 1) * K + k, L)...] = acc[k]
    end
end

"The engine's Reference deposition with another patch kernel: `stencil` a function, or `:split` for one launch per component."
function deposit_patch!(J, s, stencil, shape, st, grid, ws, sx, quantity_z; workgroupsize = nothing)
    backend = get_backend(s)
    n = Int32.(grid.n)
    M = PMC.flux_widths(shape, st)
    r = map(x -> PMC.flux_reach(shape, x), st)
    lo, hi = (r[1][1], r[2][1]), (r[1][2], r[2][2])
    if stencil === :split
        for C in 1:3
            PMC.item_launcher(patch_component_kernel!, backend, workgroupsize)(ws.flux_patch, s, ws.seg_start, ws.seg_stop, Val(C), shape, st, n,
                                                                              sx, quantity_z, ws.flux_layout, Val(prod(M)); ndrange = prod(n))
        end
    else
        PMC.item_launcher(patch_kernel!, backend, workgroupsize)(ws.flux_patch, s, ws.seg_start, ws.seg_stop, stencil, shape, st, n, sx,
                                                                quantity_z, ws.flux_layout, Val(prod(M)); ndrange = prod(n))
    end
    PMC.launcher(PMC.flux_reduce_2d_kernel!, backend, workgroupsize)(J..., ws.flux_patch, ws.flux_layout, n, Int32(grid.g), lo, hi, Int32.(M);
                                                                    ndrange = size(J[1]))
    return J
end

# --- the incremental sort ------------------------------------------------------------------------
#
# Precondition: the species was sorted by cell (`seg_start`, `seg_stop` hold the segments of that
# sort), no particle was added since, and every alive particle is now in its segment's cell or
# in one of that cell's neighbours (a move of less than a cell per axis, periodically). Then a
# cell's new particles all sit in the old segments of its 3^N neighbours, and it collects them
# itself: one work item per cell, the neighbours in increasing linear index and each segment in
# slot order, which is the order of the engine's (cell, slot) key. Two passes (count, then move
# into a second buffer) around one prefix sum; no keys, no comparison sort, no permutation array.

"The neighbours of `c` on a periodic axis of `n ≥ 3` cells, in increasing order."
@inline function triple(c::Int32, n::Int32)
    lo, hi = c == Int32(1), c == n
    return (ifelse(lo | hi, Int32(1), c - Int32(1)), ifelse(lo, Int32(2), ifelse(hi, n - Int32(1), c)), ifelse(lo | hi, n, c + Int32(1)))
end

"The linear indices of the 3^N neighbours of cell `c` (itself included), in increasing order."
@inline function neighbor_cells(c::NTuple{2,Int32}, n::NTuple{2,Int32})
    t1, t2 = triple(c[1], n[1]), triple(c[2], n[2])
    return ntuple(k -> t1[(k - 1) % 3 + 1] + (t2[(k - 1) ÷ 3 + 1] - Int32(1)) * n[1], Val(9))
end
@inline function neighbor_cells(c::NTuple{3,Int32}, n::NTuple{3,Int32})
    t1, t2, t3 = triple(c[1], n[1]), triple(c[2], n[2]), triple(c[3], n[3])
    return ntuple(k -> t1[(k - 1) % 3 + 1] + ((t2[((k - 1) ÷ 3) % 3 + 1] - Int32(1)) + (t3[(k - 1) ÷ 9 + 1] - Int32(1)) * n[2]) * n[1], Val(27))
end

@kernel function new_linear_kernel!(newL, cell, @Const(status), n)
    i = @index(Global, Linear)
    @inbounds newL[i] = ifelse(status[i] == PMC.ALIVE, PMC.linear_cell(map(c -> c[i], cell), n), Int32(0))
end

@kernel function gather_count_kernel!(cnt, @Const(newL), @Const(seg_start), @Const(seg_stop), n::NTuple{N,Int32}) where {N}
    I = @index(Global, NTuple)
    c = map(Int32, I)
    Ld = PMC.linear_cell(c, n)
    nb = neighbor_cells(c, n)
    k = Int32(0)
    @inbounds for Ls in nb
        for i in seg_start[Ls]:seg_stop[Ls]
            k += Int32(newL[i] == Ld)
        end
    end
    @inbounds cnt[Ld] = k
end

@kernel function new_segments_kernel!(seg_start, seg_stop, count, @Const(cum), @Const(cnt), ncell::Int32)
    L = Int32(@index(Global, Linear))
    @inbounds begin
        k, e = cnt[L], cum[L]
        seg_start[L] = ifelse(k == Int32(0), Int32(1), e - k + Int32(1))
        seg_stop[L] = ifelse(k == Int32(0), Int32(0), e)
        L == ncell && (count[1] = e)
    end
end

@inline copy_each!(::Tuple{}, ::Tuple{}, j, i) = nothing
@inline function copy_each!(d::Tuple, s::Tuple, j, i)
    @inbounds first(d)[j] = first(s)[i]
    return copy_each!(Base.tail(d), Base.tail(s), j, i)
end

@inline function copy_slot!(d, s, j, i)
    copy_each!(d.cell, s.cell, j, i)
    copy_each!(d.offset, s.offset, j, i)
    copy_each!(d.u, s.u, j, i)
    copy_each!((d.w, d.status), (s.w, s.status), j, i)
    copy_each!(Tuple(d.extras), Tuple(s.extras), j, i)
    return nothing
end

@kernel function gather_move_kernel!(dst, src, @Const(newL), @Const(old_start), @Const(old_stop), @Const(new_start),
                                     n::NTuple{N,Int32}) where {N}
    I = @index(Global, NTuple)
    c = map(Int32, I)
    Ld = PMC.linear_cell(c, n)
    nb = neighbor_cells(c, n)
    @inbounds j = new_start[Ld]
    @inbounds for Ls in nb
        for i in old_start[Ls]:old_stop[Ls]
            if newL[i] == Ld
                copy_slot!(dst, src, j, i)
                j += Int32(1)
            end
        end
    end
end

@kernel function free_tail_kernel!(status, @Const(cum), ncell::Int32)
    i = Int32(@index(Global, Linear))
    @inbounds if i > cum[ncell]
        status[i] = PMC.FREE
    end
end

"The second buffer and the scratch of the incremental sort."
struct IncrementalSort{S,VI}
    dst::S
    newL::VI
    cnt::VI
    cum::VI
    seg_start::VI
    seg_stop::VI
end

function IncrementalSort(s::PMC.Species{N,T}, grid::PMC.PeriodicGrid{N}) where {N,T}
    backend = get_backend(s)
    cap = PMC.capacity(s)
    dst = PMC.Species(backend, T, cap; q = s.q, m = s.m, N, extras = map(eltype, s.extras))
    z(k) = KernelAbstractions.zeros(backend, Int32, k)
    ncell = prod(grid.n)
    return IncrementalSort(dst, z(cap), z(ncell), z(ncell), z(ncell), z(ncell))
end

species_arrays(s) = (s.cell..., s.offset..., s.u..., s.w, s.status, Tuple(s.extras)...)

"""
    incremental_sort!(b, species, grid, seg_start, seg_stop; copy_back = false)

The species sorted by (cell, slot) into `b.dst`, with the new segments in `b.seg_start`,
`b.seg_stop` and the alive count in `b.dst.count`, from the segments of the last sort
(precondition above). `copy_back = true` copies the arrays and the segments back, which a step
with two buffers that swap roles would not need.
"""
function incremental_sort!(b::IncrementalSort, s::PMC.Species{N}, grid, seg_start, seg_stop; workgroupsize = nothing, copy_back = false) where {N}
    backend = get_backend(s)
    cap, n, ncell = PMC.capacity(s), Int32.(grid.n), prod(grid.n)
    all(>=(3), grid.n) || throw(ArgumentError("the incremental sort needs at least 3 cells per axis"))
    PMC.launcher(new_linear_kernel!, backend, workgroupsize)(b.newL, s.cell, s.status, n; ndrange = cap)
    PMC.launcher(gather_count_kernel!, backend, workgroupsize)(b.cnt, b.newL, seg_start, seg_stop, n; ndrange = grid.n)
    AK.accumulate!(+, b.cum, b.cnt)
    PMC.launcher(new_segments_kernel!, backend, workgroupsize)(b.seg_start, b.seg_stop, b.dst.count, b.cum, b.cnt, Int32(ncell); ndrange = ncell)
    PMC.launcher(gather_move_kernel!, backend, workgroupsize)(b.dst, s, b.newL, seg_start, seg_stop, b.seg_start, n; ndrange = grid.n)
    PMC.launcher(free_tail_kernel!, backend, workgroupsize)(b.dst.status, b.cum, Int32(ncell); ndrange = cap)
    if copy_back
        foreach(copyto!, species_arrays(s), species_arrays(b.dst))
        copyto!(s.count, b.dst.count)
        copyto!(seg_start, b.seg_start)
        copyto!(seg_stop, b.seg_stop)
    end
    return b
end

# --- the incremental sort, scatter form ----------------------------------------------------------
#
# The same precondition and the same result as the gather form above, with every particle read
# once per pass instead of 3^N times. A source cell counts its particles per direction of their
# move (3^N bins); a destination cell walks its 3^N sources in increasing linear index, sums
# the bins that point at it and leaves in each the rank its particles start at in the
# destination's new segment; the move then walks each source cell once, in slot order. A bin
# belongs to one source and one destination, so no two work items write the same value.

"The move `-1, 0, 1` from cell `c` to cell `cn` on a periodic axis of `n ≥ 3` cells, as `0, 1, 2`."
@inline axis_step(c::Int32, cn::Int32, n::Int32) = mod(cn - c + Int32(1), n)

"The bin `1:3^N` of the move from cell `c` to its neighbour `cn`."
@inline direction(c::NTuple{2,Int32}, cn::NTuple{2,Int32}, n::NTuple{2,Int32}) =
    Int32(1) + axis_step(c[1], cn[1], n[1]) + Int32(3) * axis_step(c[2], cn[2], n[2])
@inline direction(c::NTuple{3,Int32}, cn::NTuple{3,Int32}, n::NTuple{3,Int32}) =
    Int32(1) + axis_step(c[1], cn[1], n[1]) + Int32(3) * axis_step(c[2], cn[2], n[2]) + Int32(9) * axis_step(c[3], cn[3], n[3])

"The 3^N neighbours of cell `c` (itself included), in increasing linear index."
@inline function neighbor_tuples(c::NTuple{2,Int32}, n::NTuple{2,Int32})
    t1, t2 = triple(c[1], n[1]), triple(c[2], n[2])
    return ntuple(k -> (t1[(k - 1) % 3 + 1], t2[(k - 1) ÷ 3 + 1]), Val(9))
end
@inline function neighbor_tuples(c::NTuple{3,Int32}, n::NTuple{3,Int32})
    t1, t2, t3 = triple(c[1], n[1]), triple(c[2], n[2]), triple(c[3], n[3])
    return ntuple(k -> (t1[(k - 1) % 3 + 1], t2[((k - 1) ÷ 3) % 3 + 1], t3[(k - 1) ÷ 9 + 1]), Val(27))
end

# One work item per source cell: its alive particles per direction.
@kernel function scatter_count_kernel!(hist, cell, @Const(status), @Const(seg_start), @Const(seg_stop), n::NTuple{N,Int32}) where {N}
    I = @index(Global, NTuple)
    c = map(Int32, I)
    Ls = PMC.linear_cell(c, n)
    @inbounds for k in axes(hist, 1)
        hist[k, Ls] = Int32(0)
    end
    @inbounds for i in seg_start[Ls]:seg_stop[Ls]
        if status[i] == PMC.ALIVE
            hist[direction(c, map(a -> a[i], cell), n), Ls] += Int32(1)
        end
    end
end

# One work item per destination cell: its count, and in every bin that points at it the number
# of its particles that come from sources before that one.
@kernel function scatter_rank_kernel!(cnt, hist, n::NTuple{N,Int32}) where {N}
    I = @index(Global, NTuple)
    c = map(Int32, I)
    r = Int32(0)
    @inbounds for cs in neighbor_tuples(c, n)
        Ls = PMC.linear_cell(cs, n)
        k = direction(cs, c, n)
        m = hist[k, Ls]
        hist[k, Ls] = r
        r += m
    end
    @inbounds cnt[PMC.linear_cell(c, n)] = r
end

# One work item per source cell: its alive particles, in slot order, each to the next rank of
# its bin.
@kernel function scatter_move_kernel!(dst, src, rank, @Const(old_start), @Const(old_stop), @Const(new_start), n::NTuple{N,Int32}) where {N}
    I = @index(Global, NTuple)
    c = map(Int32, I)
    Ls = PMC.linear_cell(c, n)
    @inbounds for i in old_start[Ls]:old_stop[Ls]
        if src.status[i] == PMC.ALIVE
            cn = map(a -> a[i], src.cell)
            k = direction(c, cn, n)
            r = rank[k, Ls]
            rank[k, Ls] = r + Int32(1)
            copy_slot!(dst, src, new_start[PMC.linear_cell(cn, n)] + r, i)
        end
    end
end

"The second buffer and the scratch of the scatter form: 3^N bins per cell."
struct ScatterSort{S,VI,MI}
    dst::S
    hist::MI
    cnt::VI
    cum::VI
    seg_start::VI
    seg_stop::VI
end

function ScatterSort(s::PMC.Species{N,T}, grid::PMC.PeriodicGrid{N}) where {N,T}
    backend = get_backend(s)
    dst = PMC.Species(backend, T, PMC.capacity(s); q = s.q, m = s.m, N, extras = map(eltype, s.extras))
    z(k...) = KernelAbstractions.zeros(backend, Int32, k...)
    ncell = prod(grid.n)
    return ScatterSort(dst, z(3^N, ncell), z(ncell), z(ncell), z(ncell), z(ncell))
end

"`incremental_sort!` in its scatter form: the same arguments, precondition and result."
function scatter_sort!(b::ScatterSort, s::PMC.Species{N}, grid, seg_start, seg_stop; workgroupsize = nothing, copy_back = false) where {N}
    backend = get_backend(s)
    cap, n, ncell = PMC.capacity(s), Int32.(grid.n), prod(grid.n)
    all(>=(3), grid.n) || throw(ArgumentError("the incremental sort needs at least 3 cells per axis"))
    PMC.launcher(scatter_count_kernel!, backend, workgroupsize)(b.hist, s.cell, s.status, seg_start, seg_stop, n; ndrange = grid.n)
    PMC.launcher(scatter_rank_kernel!, backend, workgroupsize)(b.cnt, b.hist, n; ndrange = grid.n)
    AK.accumulate!(+, b.cum, b.cnt)
    PMC.launcher(new_segments_kernel!, backend, workgroupsize)(b.seg_start, b.seg_stop, b.dst.count, b.cum, b.cnt, Int32(ncell); ndrange = ncell)
    PMC.launcher(scatter_move_kernel!, backend, workgroupsize)(b.dst, s, b.hist, seg_start, seg_stop, b.seg_start, n; ndrange = grid.n)
    PMC.launcher(free_tail_kernel!, backend, workgroupsize)(b.dst.status, b.cum, Int32(ncell); ndrange = cap)
    if copy_back
        foreach(copyto!, species_arrays(s), species_arrays(b.dst))
        copyto!(s.count, b.dst.count)
        copyto!(seg_start, b.seg_start)
        copyto!(seg_stop, b.seg_stop)
    end
    return b
end

# --- the incremental sort, permutation form ------------------------------------------------------
#
# The scatter form writes every array of a particle from the work item of its source cell, to
# addresses that follow no order. Here the per-cell passes build only the permutation the
# engine's sort would find (`perm[j]` is the slot that slot `j` held before), and the arrays
# move through the engine's gather, one work item per slot: the counting and the ranking
# replace the keys and `sortperm!`, the rest is the engine's.

# One work item per source cell: its alive particles, in slot order, each to the next rank of
# its bin, as an entry of the permutation.
@kernel function scatter_perm_kernel!(perm, @Const(cell), @Const(status), rank, @Const(old_start), @Const(old_stop), @Const(new_start),
                                      n::NTuple{N,Int32}) where {N}
    I = @index(Global, NTuple)
    c = map(Int32, I)
    Ls = PMC.linear_cell(c, n)
    @inbounds for i in old_start[Ls]:old_stop[Ls]
        if status[i] == PMC.ALIVE
            cn = map(a -> a[i], cell)
            k = direction(c, cn, n)
            r = rank[k, Ls]
            rank[k, Ls] = r + Int32(1)
            perm[new_start[PMC.linear_cell(cn, n)] + r] = i
        end
    end
end

# The slots past the alive ones keep their own content (their status is set free afterwards).
@kernel function perm_tail_kernel!(perm, @Const(cum), ncell::Int32)
    i = Int32(@index(Global, Linear))
    @inbounds if i > cum[ncell]
        perm[i] = i
    end
end

"The second buffer and the scratch of the permutation form: 3^N bins per cell and the permutation."
struct PermSort{S,VI,MI}
    dst::S
    hist::MI
    cnt::VI
    cum::VI
    seg_start::VI
    seg_stop::VI
    perm::VI
end

function PermSort(s::PMC.Species, grid::PMC.PeriodicGrid)
    b = ScatterSort(s, grid)
    return PermSort(b.dst, b.hist, b.cnt, b.cum, b.seg_start, b.seg_stop, KernelAbstractions.zeros(get_backend(s), Int32, PMC.capacity(s)))
end

"The permutation of the sort by (cell, slot) in `b.perm` and the new segments in `b`, from the segments of the last sort."
function incremental_perm!(b::PermSort, s::PMC.Species{N}, grid, seg_start, seg_stop; workgroupsize = nothing) where {N}
    backend = get_backend(s)
    cap, n, ncell = PMC.capacity(s), Int32.(grid.n), prod(grid.n)
    all(>=(3), grid.n) || throw(ArgumentError("the incremental sort needs at least 3 cells per axis"))
    PMC.launcher(scatter_count_kernel!, backend, workgroupsize)(b.hist, s.cell, s.status, seg_start, seg_stop, n; ndrange = grid.n)
    PMC.launcher(scatter_rank_kernel!, backend, workgroupsize)(b.cnt, b.hist, n; ndrange = grid.n)
    AK.accumulate!(+, b.cum, b.cnt)
    PMC.launcher(new_segments_kernel!, backend, workgroupsize)(b.seg_start, b.seg_stop, b.dst.count, b.cum, b.cnt, Int32(ncell); ndrange = ncell)
    PMC.launcher(perm_tail_kernel!, backend, workgroupsize)(b.perm, b.cum, Int32(ncell); ndrange = cap)
    PMC.launcher(scatter_perm_kernel!, backend, workgroupsize)(b.perm, s.cell, s.status, b.hist, seg_start, seg_stop, b.seg_start, n; ndrange = grid.n)
    return b
end

"`incremental_sort!` in its permutation form: the same arguments, precondition and result."
function perm_sort!(b::PermSort, s::PMC.Species{N}, grid, seg_start, seg_stop; workgroupsize = nothing, copy_back = false) where {N}
    backend = get_backend(s)
    cap, ncell = PMC.capacity(s), prod(grid.n)
    incremental_perm!(b, s, grid, seg_start, seg_stop; workgroupsize)
    gather! = PMC.launcher(PMC.gather_perm_kernel!, backend, workgroupsize)
    foreach((d, a) -> gather!(d, a, b.perm; ndrange = cap), species_arrays(b.dst), species_arrays(s))
    PMC.launcher(free_tail_kernel!, backend, workgroupsize)(b.dst.status, b.cum, Int32(ncell); ndrange = cap)
    if copy_back
        foreach(copyto!, species_arrays(s), species_arrays(b.dst))
        copyto!(s.count, b.dst.count)
        copyto!(seg_start, b.seg_start)
        copyto!(seg_stop, b.seg_stop)
    end
    return b
end

# --- a patch in registers at N = 3, one component per launch -------------------------------------
#
# The engine's 2D mechanism at N = 3: one work item per cell sums its particles' stencils in a
# tuple (registers on a GPU) and writes it once, then one work item per node sums the cells that
# reach it, in increasing linear index. All three components at once would be 3·6³ = 648 values
# per cell at order 2 on nodes. One component per launch, without the last (zero) column of its
# prefix sum, is 5·6·6 = 180, and 4·5·5 = 100 when the particles are binned by nearest node (the
# `Center()` frame). The patch array holds one component at a time.

"The old and new weights of the three axes in the cell's frame, as differences and averages: `((D1, D2, D3), (A1, A2, A3))`."
@inline function axis_parts(s, i, shape, st::NTuple{3,PMC.Stagger}, n::NTuple{3,Int32})
    c = PMC.particle_cell(s, i, Val(3))
    δ = PMC.particle_offset(s, i, Val(3))
    c0 = old_cell_3d(s, i)
    δ0 = old_offset_3d(s, i)
    _, Wo1, Wn1 = PMC.flux_axis_weights(shape, st[1], c[1], δ[1], c0[1], δ0[1], n[1])
    _, Wo2, Wn2 = PMC.flux_axis_weights(shape, st[2], c[2], δ[2], c0[2], δ0[2], n[2])
    _, Wo3, Wn3 = PMC.flux_axis_weights(shape, st[3], c[3], δ[3], c0[3], δ0[3], n[3])
    return ((map(-, Wn1, Wo1), map(-, Wn2, Wo2), map(-, Wn3, Wo3)), (map(avg, Wo1, Wn1), map(avg, Wo2, Wn2), map(avg, Wo3, Wn3)))
end

"`Aa ⊗ Ab + Da ⊗ Db / 12`, the first index fastest."
@inline function transverse(Aa::NTuple{Ma,Any}, Ab::NTuple{Mb,Any}, Da::NTuple{Ma,Any}, Db::NTuple{Mb,Any}) where {Ma,Mb}
    Eb = map(x -> x / 12, Db)
    return ntuple(j -> Aa[(j - 1) % Ma + 1] * Ab[(j - 1) ÷ Ma + 1] + Da[(j - 1) % Ma + 1] * Eb[(j - 1) ÷ Ma + 1], Val(Ma * Mb))
end

# One particle's values of one component, the first axis fastest, on the `M` nodes of the cell's
# frame per axis and one fewer along the component's own axis.
@inline function component_stencil(::Val{1}, s, i, shape, st, n, sx, ::Val{K}) where {K}
    (D1, D2, D3), (_, A2, A3) = axis_parts(s, i, shape, st, n)
    @inbounds k = -(s.q * s.w[i] * sx[1])
    p = map(x -> k * x, PMC.prefix_sums(D1))
    t = transverse(A2, A3, D2, D3)
    W = length(D1) - 1
    return ntuple(l -> p[(l - 1) % W + 1] * t[(l - 1) ÷ W + 1], Val(K))
end
@inline function component_stencil(::Val{2}, s, i, shape, st, n, sx, ::Val{K}) where {K}
    (D1, D2, D3), (A1, _, A3) = axis_parts(s, i, shape, st, n)
    @inbounds k = -(s.q * s.w[i] * sx[2])
    p = map(x -> k * x, PMC.prefix_sums(D2))
    t = transverse(A1, A3, D1, D3)
    M1, W = length(D1), length(D2) - 1
    return ntuple(l -> p[((l - 1) ÷ M1) % W + 1] * t[(l - 1) % M1 + 1 + ((l - 1) ÷ (M1 * W)) * M1], Val(K))
end
@inline function component_stencil(::Val{3}, s, i, shape, st, n, sx, ::Val{K}) where {K}
    (D1, D2, D3), (A1, A2, _) = axis_parts(s, i, shape, st, n)
    @inbounds k = -(s.q * s.w[i] * sx[3])
    p = map(x -> k * x, PMC.prefix_sums(D3))
    t = transverse(A1, A2, D1, D2)
    M12 = length(D1) * length(D2)
    return ntuple(l -> p[(l - 1) ÷ M12 + 1] * t[(l - 1) % M12 + 1], Val(K))
end

@kernel function patch3_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), comp, shape, st, n::NTuple{3,Int32}, sx, layout, ::Val{K}) where {K}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    acc = ntuple(_ -> zero(T), Val(K))
    @inbounds for i in seg_start[L]:seg_stop[L]
        acc = PMC.tuple_add(acc, component_stencil(comp, s, i, shape, st, n, sx, Val(K)))
    end
    @inbounds for k in 1:K
        patch[PMC.patch_index(layout, k, L)...] = acc[k]
    end
end

# One work item per node (guards included): the cells whose frame holds it, in increasing linear
# index, summed from zero. `W` is the frame's width per axis for this component.
@kernel function reduce3_kernel!(J, @Const(patch), layout, n::NTuple{3,Int32}, g::Int32, lo::NTuple{3,Int32}, W::NTuple{3,Int32})
    I = @index(Global, Cartesian)
    T = eltype(J)
    j1, j2, j3 = Int32(I[1]) - g - lo[1], Int32(I[2]) - g - lo[2], Int32(I[3]) - g - lo[3]
    a = zero(T)
    @inbounds for c3 in max(Int32(1), j3 - W[3] + Int32(1)):min(n[3], j3), c2 in max(Int32(1), j2 - W[2] + Int32(1)):min(n[2], j2),
                  c1 in max(Int32(1), j1 - W[1] + Int32(1)):min(n[1], j1)
        L = c1 + ((c2 - Int32(1)) + (c3 - Int32(1)) * n[2]) * n[1]
        k = (j1 - c1) + ((j2 - c2) + (j3 - c3) * W[2]) * W[1] + Int32(1)
        a += patch[PMC.patch_index(layout, k, L)...]
    end
    @inbounds J[I] += a
end

"The patch array of the 3D patch deposition: the values of one component per cell."
function patch3_buffer(backend, ::Type{T}, grid::PMC.PeriodicGrid{3}, shape, st::NTuple{3,PMC.Stagger}) where {T}
    M = map(x -> PMC._valof(PMC.flux_width(shape, x)), st)
    K = maximum(C -> prod(ntuple(d -> M[d] - (d == C), 3)), 1:3)
    layout = PMC.flux_patch_layout(backend)
    return (buf = KernelAbstractions.zeros(backend, T, PMC.flux_patch_size(layout, K, prod(grid.n))...), layout = layout)
end

"The flux of a species sorted by cell (the cells of the stagger's binning), by per-cell patches, one component per launch."
function deposit_patch3!(J, patch, s, shape, st::NTuple{3,PMC.Stagger}, grid, seg_start, seg_stop, sx; workgroupsize = nothing)
    backend = get_backend(s)
    n = Int32.(grid.n)
    M = map(x -> PMC._valof(PMC.flux_width(shape, x)), st)
    lo = map(x -> PMC.flux_reach(shape, x)[1], st)
    for C in 1:3
        W = ntuple(d -> M[d] - (d == C), 3)
        PMC.item_launcher(patch3_kernel!, backend, workgroupsize)(patch.buf, s, seg_start, seg_stop, Val(C), shape, st, n, sx, patch.layout,
                                                                 Val(prod(W)); ndrange = prod(n))
        PMC.launcher(reduce3_kernel!, backend, workgroupsize)(J[C], patch.buf, patch.layout, n, Int32(grid.g), lo, Int32.(W); ndrange = size(J[C]))
    end
    return J
end

# --- the 3D register patch in slabs, with a straight-line stencil ---------------------------------
#
# With a wide element type the 100 or 180 accumulators of a component no longer stay in registers
# (400 numbers for `Dual{Float32,3}` with nearest-node binning). A launch can take `layers`
# planes of the cell's frame along the third axis instead: the entries `K0 + 1 : K0 + Ks` of the
# component's tuple, 20 or 25 per plane with nearest-node binning. A cell sums the same particles
# in the same order whatever the split, and an entry is the same expression as in
# `component_stencil`, so the patch is that of `deposit_patch3!`; the price is one evaluation of a
# particle's weights per launch.
#
# The entries are written out by a generated function. `component_stencil` leaves its closures to
# the compiler's cost model, which keeps them out of line for component 2 and, with a `Dual`, for
# all three: every entry is then a call with an integer division for its indices (as flux2d.jl
# notes for `outer`), and a launch that needs a fifth of the entries would still pay for all.

"`(p, Aa, Ab, Da, Eb)` of component `C`: its scaled prefix sums along its own axis and, on the two other axes `a < b`, the averages, `ΔS_a` and `ΔS_b / 12`."
@inline function component_parts(::Val{1}, s, i, shape, st, n, sx)
    (D1, D2, D3), (_, A2, A3) = axis_parts(s, i, shape, st, n)
    @inbounds k = -(s.q * s.w[i] * sx[1])
    return (map(x -> k * x, PMC.prefix_sums(D1)), A2, A3, D2, map(x -> x / 12, D3))
end
@inline function component_parts(::Val{2}, s, i, shape, st, n, sx)
    (D1, D2, D3), (A1, _, A3) = axis_parts(s, i, shape, st, n)
    @inbounds k = -(s.q * s.w[i] * sx[2])
    return (map(x -> k * x, PMC.prefix_sums(D2)), A1, A3, D1, map(x -> x / 12, D3))
end
@inline function component_parts(::Val{3}, s, i, shape, st, n, sx)
    (D1, D2, D3), (A1, A2, _) = axis_parts(s, i, shape, st, n)
    @inbounds k = -(s.q * s.w[i] * sx[3])
    return (map(x -> k * x, PMC.prefix_sums(D3)), A1, A2, D1, map(x -> x / 12, D2))
end

# Where entry `l` of component `C` reads its factors, `(own, a, b)`: the first axis fastest, `W`
# nodes along the component's own axis, `Ma` and `Mb` on the two others (as `component_stencil`).
entry_index(::Val{1}, l, W, Ma, Mb) = ((l - 1) % W + 1, ((l - 1) ÷ W) % Ma + 1, (l - 1) ÷ (W * Ma) + 1)
entry_index(::Val{2}, l, W, Ma, Mb) = (((l - 1) ÷ Ma) % W + 1, (l - 1) % Ma + 1, (l - 1) ÷ (Ma * W) + 1)
entry_index(::Val{3}, l, W, Ma, Mb) = ((l - 1) ÷ (Ma * Mb) + 1, (l - 1) % Ma + 1, ((l - 1) ÷ Ma) % Mb + 1)

"The entries `K0 + 1 : K0 + Ks` of component `C` from its `component_parts`, each `p[own]·(Aa[a]·Ab[b] + Da[a]·Eb[b])`, as straight-line code."
@generated function slab_values(vC::Val{C}, parts::Tuple{NTuple{P,Any},NTuple{Ma,Any},NTuple{Mb,Any},NTuple{Ma,Any},NTuple{Mb,Any}}, ::Val{K0},
                                ::Val{Ks}) where {C,P,Ma,Mb,K0,Ks}
    W = P - 1                                                    # the last prefix sum is zero to roundoff and has no node of its own
    transverse = Expr[]
    seen = Tuple{Int,Int}[]
    entries = Expr[]
    for k in 1:Ks
        o, a, b = entry_index(Val(C), K0 + k, W, Ma, Mb)
        t = Symbol(:t_, a, :_, b)
        if !((a, b) in seen)
            push!(seen, (a, b))
            push!(transverse, :($t = Aa[$a] * Ab[$b] + Da[$a] * Eb[$b]))
        end
        push!(entries, :(p[$o] * $t))
    end
    return quote
        $(Expr(:meta, :inline))
        p, Aa, Ab, Da, Eb = parts
        @inbounds begin
            $(transverse...)
            ($(entries...),)
        end
    end
end

@kernel function patch3_slab_kernel!(patch, s, @Const(seg_start), @Const(seg_stop), comp, shape, st, n::NTuple{3,Int32}, sx, layout,
                                     vK0::Val{K0}, vKs::Val{Ks}) where {K0,Ks}
    L = Int32(@index(Global, Linear))
    T = eltype(patch)
    acc = ntuple(_ -> zero(T), Val(Ks))
    @inbounds for i in seg_start[L]:seg_stop[L]
        acc = PMC.tuple_add(acc, slab_values(comp, component_parts(comp, s, i, shape, st, n, sx), vK0, vKs))
    end
    @inbounds for k in 1:Ks
        patch[PMC.patch_index(layout, K0 + k, L)...] = acc[k]
    end
end

"`deposit_patch3!` with the straight-line stencil and `layers` planes of the frame's third axis per launch (all of them from `layers = 6`)."
function deposit_patch3_slabs!(J, patch, s, shape, st::NTuple{3,PMC.Stagger}, grid, seg_start, seg_stop, sx, layers::Integer; workgroupsize = nothing)
    backend = get_backend(s)
    n = Int32.(grid.n)
    M = map(x -> PMC._valof(PMC.flux_width(shape, x)), st)
    lo = map(x -> PMC.flux_reach(shape, x)[1], st)
    for C in 1:3
        W = ntuple(d -> M[d] - (d == C), 3)
        plane = W[1] * W[2]
        for a in 1:layers:W[3]
            count = min(layers, W[3] - a + 1)
            PMC.item_launcher(patch3_slab_kernel!, backend, workgroupsize)(patch.buf, s, seg_start, seg_stop, Val(C), shape, st, n, sx, patch.layout,
                                                                          Val((a - 1) * plane), Val(count * plane); ndrange = prod(n))
        end
        PMC.launcher(reduce3_kernel!, backend, workgroupsize)(J[C], patch.buf, patch.layout, n, Int32(grid.g), lo, Int32.(W); ndrange = size(J[C]))
    end
    return J
end

# --- the engine's permutation from a plain sort of its keys --------------------------------------
#
# The engine's keys are unique and end in the slot (`c·capacity + slot - 1`, sort.jl), so the
# sorted keys are the permutation: `AK.sortperm!`, a merge sort that moves keys and indices, can
# be an `AK.sort!` of the keys alone and one decode. For `sort!`, AcceleratedKernels also has a
# radix sort on GPUs.

@kernel function perm_from_keys_kernel!(perm, @Const(key), cap::Int64)
    j = @index(Global, Linear)
    @inbounds perm[j] = key[j] % cap + 1
end

"""
The engine's sort keys of `s` into `ws.key` (N ≥ 2). The key kernel takes the species and the
workspace's bins where the engine has bins (WP-041's nearest-node bins), the cell and status arrays before.
"""
function engine_keys!(ws, s::PMC.Species, grid; workgroupsize = nothing)
    cap = Int64(PMC.capacity(s))
    keys! = PMC.launcher(PMC.sort_key_nd_kernel!, get_backend(s), workgroupsize)
    if isdefined(PMC, :CellBins)
        keys!(ws.key, s, ws.bins, Int64.(grid.n), Int64(prod(grid.n)), cap; ndrange = cap)
    else
        keys!(ws.key, s.cell, s.status, Int64.(grid.n), Int64(prod(grid.n)), cap; ndrange = cap)
    end
    return ws.key
end

"`ws.perm` as `sort_by_cell!` finds it (N ≥ 2), from `AK.sort!(ws.key; alg, workspace)` and a decode; the keys are left sorted."
function perm_by_key_sort!(ws, s::PMC.Species, grid; alg = AK.Auto(), workspace = nothing, workgroupsize = nothing)
    backend = get_backend(s)
    cap = Int64(PMC.capacity(s))
    engine_keys!(ws, s, grid; workgroupsize)
    AK.sort!(ws.key; alg, workspace)
    PMC.launcher(perm_from_keys_kernel!, backend, workgroupsize)(ws.perm, ws.key, cap; ndrange = cap)
    return ws.perm
end
