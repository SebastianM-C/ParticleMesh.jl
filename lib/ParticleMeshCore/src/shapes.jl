"""
    Shape{P}()

B-spline shape function of order `P` (1: cloud in cell, 2: quadratic, the A2 default). The order
is type-level configuration, so kernels specialize on it.
"""
struct Shape{P} end
const Linear = Shape{1}()
const Quadratic = Shape{2}()
order(::Shape{P}) where {P} = P

"Guard cells needed on each side for any stagger: the stencil reach of the shape."
guard_width(::Shape{P}) where {P} = P

"""
Where the nodes of a field sit within a cell: `Node()` at the left cell edge (offset 0),
`Center()` at the cell centre (offset 1/2).
"""
abstract type Stagger end
struct Node <: Stagger end
struct Center <: Stagger end

@inline stagger_offset(::Node, ::Type{T}) where {T} = zero(T)
@inline stagger_offset(::Center, ::Type{T}) where {T} = one(T) / 2

"""
    shape_weights(shape, stagger, δ) -> (f::Int32, w::NTuple{P+1})

Weights of a particle at in-cell offset `δ ∈ [0, 1)` of cell `c` on the nodes `c + f + k`,
`k = 0:P`, of a field with the given stagger. The weights sum to one (partition of unity).

The pairs (order 1, `Node`) and (order 2, `Center`) have a stencil independent of `δ`; the
other two need `floor` of the offset, which is the only place a `Dual` offset meets an integer.
"""
@inline function shape_weights(::Shape{1}, ::Node, δ::T) where {T}
    return (Int32(0), (one(T) - δ, δ))
end

@inline function shape_weights(::Shape{1}, s::Stagger, δ::T) where {T}
    u = δ - stagger_offset(s, T)
    f = floor(Int32, u)
    ξ = u - T(f)
    return (f, (one(T) - ξ, ξ))
end

@inline function quadratic_weights(ξ::T) where {T}
    h = one(T) / 2
    a = h - ξ
    b = h + ξ
    return (a * a / 2, T(3) / 4 - ξ * ξ, b * b / 2)
end

@inline function shape_weights(::Shape{2}, ::Center, δ::T) where {T}
    return (Int32(-1), quadratic_weights(δ - one(T) / 2))
end

@inline function shape_weights(::Shape{2}, s::Stagger, δ::T) where {T}
    u = δ - stagger_offset(s, T)
    b = floor(Int32, u + one(T) / 2)
    return (b - Int32(1), quadratic_weights(u - T(b)))
end

"""
    stencil_reach(shape, stagger) -> (lo, hi)

For any offset in `[0, 1)`, the nodes touched by a particle in cell `c` lie in `c+lo:c+hi`.
"""
stencil_reach(::Shape{1}, ::Node) = (0, 1)
stencil_reach(::Shape{1}, ::Center) = (-1, 1)
stencil_reach(::Shape{2}, ::Center) = (-1, 1)
stencil_reach(::Shape{2}, ::Node) = (-1, 2)

"""
    advance(cell, δ, Δ, n) -> (cell′, δ′)

Move a particle by `Δ` cells on a periodic axis of `n` cells: the offset stays in `[0, 1)`,
including the case where `δ + Δ` is a tiny negative number and `δ + Δ - floor(δ + Δ)` rounds up
to one; the cell wraps periodically. Exact when `Δ` is a dyadic fraction representable in `T`.
"""
@inline function advance(cell::Int32, δ::T, Δ::T, n::Int32) where {T}
    y = δ + Δ
    k = floor(Int32, y)
    δ′ = y - T(k)
    up = δ′ >= one(T)
    δ′ = ifelse(up, δ′ - one(T), δ′)
    k += Int32(up)
    return (mod1(cell + k, n), δ′)
end

# N axes: one shape for every axis, one stagger per axis.

"""
    shape_weights(shape, stagger::NTuple{N,Stagger}, δ::NTuple{N,T}) -> NTuple{N,(f, w)}

The 1D weights of each axis: `(f_d, w_d) = shape_weights(shape, stagger[d], δ[d])`. The weight of
node `c + f + k - 1` (per axis) is the tensor product `w_1[k_1] · w_2[k_2]`, multiplied in that
order; [`tensor_weights`](@ref) lists them.
"""
@inline shape_weights(shape::Shape, st::NTuple{N,Stagger}, δ::NTuple{N,T}) where {N,T} =
    map((s, x) -> shape_weights(shape, s, x), st, δ)      # `map`: a mixed stagger tuple is heterogeneous

"""
    tensor_weights(shape, stagger::NTuple{2,Stagger}, δ::NTuple{2,T}) -> ((f1, f2), W)

The first node per axis and the `(P+1)²` tensor-product weights, column-major
(`W[k1 + (P+1)(k2-1)] = w1[k1] · w2[k2]`), bitwise the products the 2D kernels form. They sum to
one (T-262).
"""
@inline function tensor_weights(shape::Shape{P}, st::NTuple{2,Stagger}, δ::NTuple{2,T}) where {P,T}
    (f1, w1), (f2, w2) = shape_weights(shape, st, δ)
    W = ntuple(k -> w1[(k - 1) % (P + 1) + 1] * w2[(k - 1) ÷ (P + 1) + 1], Val((P + 1) * (P + 1)))
    return ((f1, f2), W)
end

"Per axis, the `(lo, hi)` of [`stencil_reach`](@ref) for that axis' stagger."
stencil_reach(shape::Shape, st::NTuple{N,Stagger}) where {N} = map(s -> stencil_reach(shape, s), st)

"""
    advance(cell::NTuple{N,Int32}, δ::NTuple{N,T}, Δ::NTuple{N,T}, n::NTuple{N,Int32}) -> (cell′, δ′)

[`advance`](@ref) applied per axis: each offset stays in `[0, 1)` and each cell wraps on its own
periodic axis, so a diagonal move through a corner wraps both. On periodic axes this wrap is the
single-device particle exchange.
"""
@inline function advance(cell::NTuple{N,Int32}, δ::NTuple{N,T}, Δ::NTuple{N,T}, n::NTuple{N,Int32}) where {N,T}
    r = ntuple(d -> advance(cell[d], δ[d], Δ[d], n[d]), Val(N))
    return (ntuple(d -> r[d][1], Val(N)), ntuple(d -> r[d][2], Val(N)))
end
