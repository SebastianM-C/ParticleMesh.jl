# Pushers: the momentum update of one particle in given fields, as a device function. The step
# dispatches on the pusher type, so another pusher (WP-015) is a new subtype and a new method of
# `push_momentum`, with no change to the step.

"""
    Pusher

Supertype of the momentum integrators. A subtype `P` implements
`push_momentum(::P, u, E, B, qm_h)`: the momentum `uⁿ⁺¹ᐟ²` from `uⁿ⁻¹ᐟ²`, the fields `E`, `B` at
the particle at time n and `qm_h = (q/m)·Δt/2`, all as `NTuple{3,T}` (A1 units: `u = γv` per
unit mass, in c).
"""
abstract type Pusher end

"""
    Boris()

The relativistic Boris pusher with half-step momentum centring (A2): half an electric kick,
a rotation about B by the angle 2·atan(|q|BΔt/(2mγ̄)) with γ̄ the Lorentz factor after the
half kick, and the second half kick. The rotation conserves |u| exactly in exact arithmetic
(T-211).
"""
struct Boris <: Pusher end

"Lorentz factor `γ = √(1 + |u|²)` of the momentum `u` (per unit mass, in c)."
@inline lorentz_factor(u1::T, u2::T, u3::T) where {T} = sqrt(one(T) + u1 * u1 + u2 * u2 + u3 * u3)
@inline lorentz_factor(u::NTuple{3,T}) where {T} = lorentz_factor(u[1], u[2], u[3])

@inline cross3(a::NTuple{3,T}, b::NTuple{3,T}) where {T} =
    (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

"""
    push_momentum(::Boris, u, E, B, qm_h) -> u′

`u⁻ = u + qm_h·E`; `t = qm_h·B/γ(u⁻)`; `s = 2t/(1 + |t|²)`; `u′ = u⁻ + u⁻×t`;
`u⁺ = u⁻ + u′×s`; returns `u⁺ + qm_h·E`. Generic in `T`; a negative `qm_h` integrates backwards
(the half step back of `initialize!`).
"""
@inline function push_momentum(::Boris, u::NTuple{3,T}, E::NTuple{3,T}, B::NTuple{3,T}, qm_h::T) where {T}
    um = (u[1] + qm_h * E[1], u[2] + qm_h * E[2], u[3] + qm_h * E[3])
    f = qm_h / lorentz_factor(um)
    t = (f * B[1], f * B[2], f * B[3])
    c = 2 / (one(T) + t[1] * t[1] + t[2] * t[2] + t[3] * t[3])
    s = (c * t[1], c * t[2], c * t[3])
    a = cross3(um, t)
    up = (um[1] + a[1], um[2] + a[2], um[3] + a[3])
    b = cross3(up, s)
    return (um[1] + b[1] + qm_h * E[1], um[2] + b[2] + qm_h * E[2], um[3] + b[3] + qm_h * E[3])
end
