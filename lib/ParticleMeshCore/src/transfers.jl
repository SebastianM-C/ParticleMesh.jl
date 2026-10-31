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
    gather!(out, a, species, shape, stagger, grid)

`out[i]` = the field `a` (guards filled) at particle `i`, for every alive particle.
"""
function gather!(out::AbstractVector, a::AbstractVector, s::Species{1}, shape::Shape, stagger::Stagger,
                 grid::PeriodicGrid{1})
    gather_kernel!(get_backend(s))(out, a, s, shape, stagger, Int32(guards(grid)); ndrange = capacity(s))
    return out
end

@kernel function deposit_atomic_kernel!(ρ, s, shape::Shape{P}, stagger, g::Int32) where {P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        qw = s.q * s.w[i]
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
@kernel function deposit_sorted_kernel!(ρ, s, @Const(seg_start), @Const(seg_stop), shape, stagger,
                                        n::Int32, g::Int32, lo::Int32, hi::Int32)
    a = Int32(@index(Global, Linear))
    T = eltype(ρ)
    j = a - g                                   # node index, 1-g:n+g
    acc = zero(T)
    @inbounds for c in max(Int32(1), j - hi):min(n, j - lo)
        for i in seg_start[c]:seg_stop[c]
            f, w = shape_weights(shape, stagger, s.offset[1][i])
            acc += s.q * s.w[i] * pick(w, j - c - f)
        end
    end
    @inbounds ρ[a] += acc
end

"""
    deposit!(mode, ρ, species, shape, stagger, grid, ws)

Add `q·w` of every alive particle into the field `ρ` (with guards) with the given shape and
stagger. Guards receive the contributions that cross the periodic seam; `fold_guards!` moves
them into the interior. `Reference()` needs `sort_by_cell!(species, grid, ws)` since the last
particle move and is bitwise deterministic; `Fast()` uses atomics where `accumulation` allows
them and falls back to `Reference()` otherwise (`effective_mode`).
"""
function deposit!(mode, ρ::AbstractVector{T}, s::Species{1}, shape::Shape, stagger::Stagger,
                  grid::PeriodicGrid{1}, ws::Workspace) where {T}
    _deposit!(effective_mode(mode, T), ρ, s, shape, stagger, grid, ws)
end

function _deposit!(::Fast, ρ, s, shape, stagger, grid, ws)
    deposit_atomic_kernel!(get_backend(s))(ρ, s, shape, stagger, Int32(guards(grid)); ndrange = capacity(s))
    return ρ
end

function _deposit!(::Reference, ρ, s, shape, stagger, grid, ws)
    lo, hi = stencil_reach(shape, stagger)
    deposit_sorted_kernel!(get_backend(s))(ρ, s, ws.seg_start, ws.seg_stop, shape, stagger,
                                           Int32(ncells(grid)), Int32(guards(grid)), Int32(lo), Int32(hi);
                                           ndrange = length(ρ))
    return ρ
end
