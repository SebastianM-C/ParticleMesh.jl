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
