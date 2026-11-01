# Particle ↔ mesh transfers: gather and scalar deposition in two modes.

"Deterministic deposition: sort by cell, then each node reduces the segments of the cells in its stencil, in a fixed order (A4)."
struct Reference end
"Deposition by atomic adds; order-dependent in floating point."
struct Fast end

"Accumulation path of an element type: atomics for primitive floats, the sorted reduction for every other type (A8)."
struct Atomic end
struct Sorted end
accumulation(::Type{<:Union{Float32,Float64}}) = Atomic()
accumulation(::Type) = Sorted()

"""
    effective_mode(mode, T)

The mode deposition actually runs in: `Fast()` for element types without atomics (`Dual`,
compensated types) falls back to `Reference()`. A step that sorts only when needed asks this.
"""
effective_mode(::Reference, ::Type) = Reference()
effective_mode(::Fast, ::Type{T}) where {T} = _effective(accumulation(T))
_effective(::Atomic) = Fast()
_effective(::Sorted) = Reference()

"""
    gather_at(a, c, δ, shape, stagger, g)

The device-level gather: the value of field `a` (guards filled) at offset `δ` of cell `c`.
"""
@inline function gather_at(a, c::Int32, δ::T, shape::Shape{P}, stagger, g::Int32) where {T,P}
    f, w = shape_weights(shape, stagger, δ)
    base = c + f + g
    acc = zero(T)
    @inbounds for k in 1:(P + 1)
        acc += w[k] * a[base + Int32(k - 1)]
    end
    return acc
end

@kernel function gather_kernel!(out, @Const(a), s, shape, stagger, g::Int32)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        out[i] = gather_at(a, s.cell[1][i], s.offset[1][i], shape, stagger, g)
    end
end

"""
    gather!(out, a, species, shape, stagger, grid; workgroupsize = nothing)

`out[i]` = the field `a` (guards filled) at particle `i`, for every alive particle. For N = 2,
`a` is a matrix and `stagger` an `NTuple{2,Stagger}` (T-264).
"""
function gather!(out::AbstractVector, a::AbstractVector, s::Species{1}, shape::Shape, stagger::Stagger,
                 grid::PeriodicGrid{1}; workgroupsize = nothing)
    launcher(gather_kernel!, get_backend(s), workgroupsize)(out, a, s, shape, stagger, Int32(guards(grid));
                                                           ndrange = capacity(s))
    return out
end

"""
    ChargeWeight()

The default deposited quantity, `q·w` of particle `i`: `ChargeWeight()(s, i) = s.q * s.w[i]`.
A module supplies its own as any isbits callable `(species, i) -> value of T`, evaluated inside
the deposition kernels for alive particles only (for example the electromagnetic module's
`q·w·v_y` for a current component).
"""
struct ChargeWeight end
@inline (::ChargeWeight)(s, i) = s.q * s.w[i]

@kernel function deposit_atomic_kernel!(ρ, s, quantity, shape::Shape{P}, stagger, g::Int32) where {P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        qw = quantity(s, i)
        f, w = shape_weights(shape, stagger, s.offset[1][i])
        base = s.cell[1][i] + f + g
        for k in 1:(P + 1)
            Atomix.@atomic ρ[base + Int32(k - 1)] += qw * w[k]
        end
    end
end

@inline function pick(w::NTuple{M,T}, d::Int32) where {M,T}
    inrange = (d >= Int32(0)) & (d < Int32(M))
    return ifelse(inrange, w[clamp(d, Int32(0), Int32(M - 1)) + Int32(1)], zero(T))
end

# One work item per node (guards included): the cells whose stencil reaches the node, in
# increasing order, and within a cell its particles in sorted (slot) order. No scratch.
@kernel function deposit_sorted_kernel!(ρ, s, quantity, @Const(seg_start), @Const(seg_stop), shape, stagger,
                                        n::Int32, g::Int32, lo::Int32, hi::Int32)
    a = Int32(@index(Global, Linear))
    T = eltype(ρ)
    j = a - g                                   # node index, 1-g:n+g
    acc = zero(T)
    @inbounds for c in max(Int32(1), j - hi):min(n, j - lo)
        for i in seg_start[c]:seg_stop[c]
            f, w = shape_weights(shape, stagger, s.offset[1][i])
            acc += quantity(s, i) * pick(w, j - c - f)
        end
    end
    @inbounds ρ[a] += acc
end

"""
    deposit!(mode, ρ, species, shape, stagger, grid, ws; quantity = ChargeWeight(), workgroupsize = nothing)

Add `quantity(species, i)` (by default `q·w`) of every alive particle into the field `ρ` (with
guards) with the given shape and stagger. Guards receive the contributions that cross the
periodic seam; `fold_guards!` moves them into the interior ("deposit into guards, then fold").
`Reference()` needs `sort_by_cell!(species, grid, ws)` since the last particle move and is
bitwise deterministic: one work item per node adds, in increasing cell order and within a cell
in slot order, the contributions of the cells in its stencil. `Fast()` uses atomics where
`accumulation` allows them and falls back to `Reference()` otherwise (`effective_mode`). In
both modes `deposit!` is the adjoint of `gather!` with the same weighting, including fill and
fold (T-132, T-175).

For N = 2, `ρ` is a matrix with guards, `stagger` an `NTuple{2,Stagger}` and the weights the
tensor product of the per-axis weights; `Reference()` needs the 2D `sort_by_cell!`, and a node
adds the contributions of its stencil's cells in increasing column-major linear index, within a
cell in slot order (the A4 bitwise contract for N = 2; T-265, T-266).
"""
function deposit!(mode, ρ::AbstractVector{T}, s::Species{1}, shape::Shape, stagger::Stagger,
                  grid::PeriodicGrid{1}, ws::Workspace; quantity = ChargeWeight(), workgroupsize = nothing) where {T}
    _deposit!(effective_mode(mode, T), ρ, s, quantity, shape, stagger, grid, ws, workgroupsize)
end

function _deposit!(::Fast, ρ, s, quantity, shape, stagger, grid, ws, workgroupsize)
    launcher(deposit_atomic_kernel!, get_backend(s), workgroupsize)(ρ, s, quantity, shape, stagger, Int32(guards(grid));
                                                                   ndrange = capacity(s))
    return ρ
end

function _deposit!(::Reference, ρ, s, quantity, shape, stagger, grid, ws, workgroupsize)
    lo, hi = stencil_reach(shape, stagger)
    launcher(deposit_sorted_kernel!, get_backend(s), workgroupsize)(ρ, s, quantity, ws.seg_start, ws.seg_stop, shape,
                                                                   stagger, Int32(ncells(grid)), Int32(guards(grid)),
                                                                   Int32(lo), Int32(hi); ndrange = length(ρ))
    return ρ
end

# N axes (tested for N = 2). The kernel mechanism is a `CartesianIndices` loop over per-axis
# ranges, chosen from the prototype measurement of WP-018 (`reports/wp-018-engine-2d.md`):
# it ties explicit nested loops within the run-to-run spread, beats a recursive tuple loop nest,
# and is generic in N. Every loop below visits its box column-major, axis 1 fastest.

"The tensor-product weight `w_1[k_1]·w_2[k_2]·…`, multiplied in axis order (`tensor_weights`' products)."
@inline tensor_weight(fw::NTuple{N,Any}, k::NTuple{N,Int}) where {N} = prod(ntuple(d -> fw[d][2][k[d]], Val(N)))

"As `pick` per axis: the tensor-product weight of node `j` for a particle in cell `c`, zero outside its stencil."
@inline tensor_pick(fw::NTuple{N,Any}, j::NTuple{N,Int32}, c::NTuple{N,Int32}) where {N} =
    prod(ntuple(d -> pick(fw[d][2], j[d] - c[d] - fw[d][1]), Val(N)))

@inline particle_offset(s, i, ::Val{N}) where {N} = @inbounds ntuple(d -> s.offset[d][i], Val(N))
@inline particle_cell(s, i, ::Val{N}) where {N} = @inbounds ntuple(d -> s.cell[d][i], Val(N))

"""
    gather_at(a, c::NTuple{N,Int32}, δ::NTuple{N,T}, shape, stagger::NTuple{N}, g)

The device-level gather on N axes: Σ over the `(P+1)^N` stencil, axis 1 fastest, of the
tensor-product weight times the field (guards filled).
"""
@inline function gather_at(a, c::NTuple{N,Int32}, δ::NTuple{N,T}, shape::Shape{P}, stagger::NTuple{N,Stagger},
                           g::Int32) where {N,T,P}
    fw = shape_weights(shape, stagger, δ)
    acc = zero(T)
    @inbounds for K in CartesianIndices(ntuple(_ -> Base.OneTo(P + 1), Val(N)))
        k = Tuple(K)
        idx = ntuple(d -> c[d] + fw[d][1] + g + Int32(k[d] - 1), Val(N))
        acc += tensor_weight(fw, k) * a[idx...]
    end
    return acc
end

@kernel function gather_nd_kernel!(out, @Const(a), s, shape, stagger::NTuple{N,Stagger}, g::Int32) where {N}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        out[i] = gather_at(a, particle_cell(s, i, Val(N)), particle_offset(s, i, Val(N)), shape, stagger, g)
    end
end

function gather!(out::AbstractVector, a::AbstractArray{<:Any,N}, s::Species{N}, shape::Shape, stagger::NTuple{N,Stagger},
                 grid::PeriodicGrid{N}; workgroupsize = nothing) where {N}
    launcher(gather_nd_kernel!, get_backend(s), workgroupsize)(out, a, s, shape, stagger, Int32(guards(grid));
                                                              ndrange = capacity(s))
    return out
end

@kernel function deposit_atomic_nd_kernel!(ρ, s, quantity, shape::Shape{P}, stagger::NTuple{N,Stagger}, g::Int32) where {N,P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        qw = quantity(s, i)
        c = particle_cell(s, i, Val(N))
        fw = shape_weights(shape, stagger, particle_offset(s, i, Val(N)))
        for K in CartesianIndices(ntuple(_ -> Base.OneTo(P + 1), Val(N)))
            k = Tuple(K)
            idx = CartesianIndex(ntuple(d -> Int(c[d] + fw[d][1] + g) + k[d] - 1, Val(N)))
            Atomix.@atomic ρ[idx] += qw * tensor_weight(fw, k)
        end
    end
end

# One work item per node (guards included): the cells of the node's stencil in increasing linear
# index (the CartesianIndices of their per-axis ranges, axis 1 fastest), within a cell its
# particles in sorted (slot) order. No scratch.
@kernel function deposit_sorted_nd_kernel!(ρ, s, quantity, @Const(seg_start), @Const(seg_stop), shape,
                                           stagger::NTuple{N,Stagger}, n::NTuple{N,Int32}, g::Int32,
                                           lo::NTuple{N,Int32}, hi::NTuple{N,Int32}) where {N}
    I = @index(Global, Cartesian)
    T = eltype(ρ)
    j = ntuple(d -> Int32(I[d]) - g, Val(N))                   # node index per axis, 1-g:n+g
    acc = zero(T)
    @inbounds for C in CartesianIndices(ntuple(d -> max(Int32(1), j[d] - hi[d]):min(n[d], j[d] - lo[d]), Val(N)))
        c = ntuple(d -> Int32(C[d]), Val(N))
        L = linear_cell(c, n)
        for i in seg_start[L]:seg_stop[L]
            fw = shape_weights(shape, stagger, particle_offset(s, i, Val(N)))
            acc += quantity(s, i) * tensor_pick(fw, j, c)
        end
    end
    @inbounds ρ[I] += acc
end

function deposit!(mode, ρ::AbstractArray{T,N}, s::Species{N}, shape::Shape, stagger::NTuple{N,Stagger},
                  grid::PeriodicGrid{N}, ws::Workspace; quantity = ChargeWeight(), workgroupsize = nothing) where {T,N}
    _deposit_nd!(effective_mode(mode, T), ρ, s, quantity, shape, stagger, grid, ws, workgroupsize)
end

function _deposit_nd!(::Fast, ρ, s, quantity, shape, stagger, grid, ws, workgroupsize)
    launcher(deposit_atomic_nd_kernel!, get_backend(s), workgroupsize)(ρ, s, quantity, shape, stagger, Int32(guards(grid));
                                                                      ndrange = capacity(s))
    return ρ
end

function _deposit_nd!(::Reference, ρ, s, quantity, shape, stagger, grid::PeriodicGrid{N}, ws, workgroupsize) where {N}
    r = stencil_reach(shape, stagger)
    lo = ntuple(d -> Int32(r[d][1]), Val(N))
    hi = ntuple(d -> Int32(r[d][2]), Val(N))
    launcher(deposit_sorted_nd_kernel!, get_backend(s), workgroupsize)(ρ, s, quantity, ws.seg_start, ws.seg_stop, shape, stagger,
                                                                      Int32.(grid.n), Int32(guards(grid)), lo, hi; ndrange = size(ρ))
    return ρ
end
