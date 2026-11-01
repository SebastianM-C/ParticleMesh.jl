# Pushers: the momentum update of one particle in given fields, as a device function. The step
# dispatches on the pusher type, so each pusher is a subtype and a method of
# `push_momentum`, with no change to the step (WP-015 adds Vay, HigueraCary, BorisExactRotation).

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

# --- WP-015: Vay, Higuera–Cary and the exact-rotation Boris --------------------------------------
#
# The three are derived from their defining equations in the variables of `push_momentum`:
# `τ = qm_h·B` (so the Boris `t` is τ/γ), `u⁻ = u + qm_h·E` the momentum after the first electric
# half kick. Boris and Higuera–Cary solve the same implicit midpoint rotation
# `u⁺ − u⁻ = (u⁺ + u⁻) × τ/γ`, Boris with γ = γ(u⁻) and Higuera–Cary with the γ of their Eq. 20;
# Vay solves `u⁺ = u′ + u⁺ × τ/γ(u⁺)` for u⁺ (see `push_momentum(::Vay, ...)`).

"""
    Vay()

The pusher of Vay (Phys. Plasmas 15, 056701 (2008)): the velocity in the magnetic force is the
average of the old and the new velocity, `uⁿ⁺¹ᐟ² = uⁿ⁻¹ᐟ² + 2qm_h·(E + (vⁿ⁻¹ᐟ² + vⁿ⁺¹ᐟ²)/2 × B)`,
solved in closed form for `uⁿ⁺¹ᐟ²`.

- Phase-space volume: not preserved; the one-step Jacobian determinant is J(uⁿ⁻¹ᐟ²)/J(uⁿ⁺¹ᐟ²)
  with J(u) = 1 + (τ² + (u·τ)²)/γ(u)⁴ (Higuera and Cary, Phys. Plasmas 24, 052104 (2017),
  Eqs. 40–41; T-234).
- E×B velocity: a particle moving at E×B/B² (E < B) stays there exactly, relativistic or not.
- Energy for E = 0: |u| exact.
- Gyrophase in uniform B: 2·atan(ΩΔt/(2γ)) per step, the same as Boris (Vay, App. B).

One square root more than Boris.
"""
struct Vay <: Pusher end

"""
    HigueraCary()

The pusher of Higuera and Cary (Phys. Plasmas 24, 052104 (2017)): the Boris kick–rotation–kick
with the rotation's γ taken from their Eq. 20,
`γ_new² = (σ + √(σ² + 4(τ² + (u⁻·τ)²)))/2`, `σ = γ(u⁻)² − τ²`, instead of γ(u⁻).

- Phase-space volume: preserved (their Sec. IV): the rotation angle depends on |u⁻| and u⁻·τ
  only, both invariant under the rotation about τ.
- E×B velocity: a particle moving at E×B/B² (E < B) stays there exactly (their Eqs. 28–29).
- Energy for E = 0: |u| exact.
- Gyrophase in uniform B: 2·atan(|τ|/γ_new) per step, with γ_new ≤ γ (their Eq. 25).

One square root more than Boris.
"""
struct HigueraCary <: Pusher end

"""
    BorisExactRotation()

Boris with the magnetic rotation done exactly, "Boris-C" of Zenitani and Umeda (Phys. Plasmas
25, 112110 (2018)): half electric kick, rotation of u⁻ about B by θ = ΩΔt/γ(u⁻) (Rodrigues'
formula), half electric kick.

- Phase-space volume: preserved (Zenitani and Umeda, App. A).
- E×B velocity: wrong at O((ΩΔt)²), even nonrelativistically: the fixed point is off by about
  (θ/2)/tan(θ/2).
- Energy for E = 0: |u| exact.
- Gyrophase in uniform B: exact, θ = ΩΔt/γ per step.

Exact-rotation schemes become unstable at ΩΔt/γ = kπ (He, Sun, Liu and Qin, J. Comput. Phys.
281, 135 (2015); Knapp et al., Phys. Rev. E 92, 063310 (2015)), where the Cayley form of Boris
is not.

The rotation needs cos θ, sin θ/θ and (1 − cos θ)/θ², written as functions of θ² = |w|², with
w = 2τ/γ: no |B| = √(B·B), whose derivative is infinite at B = 0, and no branch on |B|. Below
θ² = √eps(T) the three come from their Taylor series (truncation below eps(T)²); above, from
sin and cos of θ = √θ². The two are chosen with `ifelse`, a select of two computed values,
not a branch, and the square root sees max(θ², √eps(T)), so duals stay finite through B = 0.
"""
struct BorisExactRotation <: Pusher end

@inline dot3(a::NTuple{3,T}, b::NTuple{3,T}) where {T} = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"The γ of Higuera and Cary's Eq. 20 for momentum `u` and `τ = qm_h·B`: the root of γ⁴ − σγ² − (τ² + (u·τ)²) = 0, σ = 1 + u² − τ²."
@inline function eq20_gamma(u::NTuple{3,T}, τ::NTuple{3,T}) where {T}
    τ² = dot3(τ, τ)
    uτ = dot3(u, τ)
    σ = one(T) + dot3(u, u) - τ²
    return sqrt((σ + sqrt(σ * σ + 4 * (τ² + uτ * uτ))) / 2)
end

# u⁺ solving u⁺ = v + u⁺ × t: u⁺ = (v + (v·t)t + v×t)/(1 + t²) (the inverse of I − [t]ₓ).
@inline function solve_rotation(v::NTuple{3,T}, t::NTuple{3,T}) where {T}
    s = one(T) / (one(T) + dot3(t, t))
    vt = dot3(v, t)
    c = cross3(v, t)
    return (s * (v[1] + vt * t[1] + c[1]), s * (v[2] + vt * t[2] + c[2]), s * (v[3] + vt * t[3] + c[3]))
end

"""
    push_momentum(::Vay, u, E, B, qm_h) -> u′

`u′ = u + qm_h·(2E + u×B/γ(u))`, `τ = qm_h·B`, `γ⁺` from Eq. 20 of Higuera and Cary applied to
u′ (Vay's Eq. 11), `t = τ/γ⁺`, and `u⁺ = (u′ + (u′·t)t + u′×t)/(1 + t²)`, the solution of
`u⁺ = u′ + u⁺×t`.
"""
@inline function push_momentum(::Vay, u::NTuple{3,T}, E::NTuple{3,T}, B::NTuple{3,T}, qm_h::T) where {T}
    τ = (qm_h * B[1], qm_h * B[2], qm_h * B[3])
    ig = one(T) / lorentz_factor(u)
    a = cross3(u, τ)
    up = (u[1] + 2 * qm_h * E[1] + ig * a[1], u[2] + 2 * qm_h * E[2] + ig * a[2], u[3] + 2 * qm_h * E[3] + ig * a[3])
    f = one(T) / eq20_gamma(up, τ)
    return solve_rotation(up, (f * τ[1], f * τ[2], f * τ[3]))
end

"""
    push_momentum(::HigueraCary, u, E, B, qm_h) -> u′

`u⁻ = u + qm_h·E`; `τ = qm_h·B`; `γ_new` from Eq. 20 applied to u⁻; `t = τ/γ_new`; then the
rotation `u⁺ − u⁻ = (u⁺ + u⁻)×t` in the two-cross-product form of `Boris` (`s = 2t/(1 + t²)`,
`u′ = u⁻ + u⁻×t`, `u⁺ = u⁻ + u′×s`); returns `u⁺ + qm_h·E`. The midpoint form
`u⁺ = 2ū − u⁻`, `ū = (u⁻ + (u⁻·t)t + u⁻×t)/(1 + t²)`, is the same map with a larger rounding
drift of |u| (T-231).
"""
@inline function push_momentum(::HigueraCary, u::NTuple{3,T}, E::NTuple{3,T}, B::NTuple{3,T}, qm_h::T) where {T}
    um = (u[1] + qm_h * E[1], u[2] + qm_h * E[2], u[3] + qm_h * E[3])
    τ = (qm_h * B[1], qm_h * B[2], qm_h * B[3])
    f = one(T) / eq20_gamma(um, τ)
    t = (f * τ[1], f * τ[2], f * τ[3])
    c = 2 / (one(T) + dot3(t, t))
    s = (c * t[1], c * t[2], c * t[3])
    a = cross3(um, t)
    up = (um[1] + a[1], um[2] + a[2], um[3] + a[3])
    b = cross3(up, s)
    return (um[1] + b[1] + qm_h * E[1], um[2] + b[2] + qm_h * E[2], um[3] + b[3] + qm_h * E[3])
end

"""
    rotation_coefficients(θ²) -> (cos θ, sin θ/θ, (1 − cos θ)/θ²)

Smooth in θ² ≥ 0 including 0, branch-free (see [`BorisExactRotation`](@ref)).
"""
@inline function rotation_coefficients(x::T) where {T}
    xc = sqrt(eps(T))
    # Taylor series in x = θ², truncated after x² (the next terms are below x³/720 < eps(T)²)
    cs = one(T) - x / 2 + x * x / 24
    ss = one(T) - x / 6 + x * x / 120
    ks = one(T) / 2 - x / 24 + x * x / 720
    θ = sqrt(max(x, xc))
    sh = sin(θ / 2)
    cl = cos(θ)
    sl = sin(θ) / θ
    kl = 2 * (sh / θ) * (sh / θ)                       # (1 − cos θ)/θ², no cancellation
    small = x < xc
    return ifelse(small, cs, cl), ifelse(small, ss, sl), ifelse(small, ks, kl)
end

"""
    push_momentum(::BorisExactRotation, u, E, B, qm_h) -> u′

`u⁻ = u + qm_h·E`; `w = 2qm_h·B/γ(u⁻)`, θ² = |w|²; `u⁺ = cos θ·u⁻ + (sin θ/θ)·u⁻×w +
((1 − cos θ)/θ²)·(u⁻·w)·w`, the rotation whose first order is the Boris `u⁻ + u⁻×w`; returns
`u⁺ + qm_h·E`.
"""
@inline function push_momentum(::BorisExactRotation, u::NTuple{3,T}, E::NTuple{3,T}, B::NTuple{3,T}, qm_h::T) where {T}
    um = (u[1] + qm_h * E[1], u[2] + qm_h * E[2], u[3] + qm_h * E[3])
    f = 2 * qm_h / lorentz_factor(um)
    w = (f * B[1], f * B[2], f * B[3])
    c, s, k = rotation_coefficients(dot3(w, w))
    a = cross3(um, w)
    kw = k * dot3(um, w)
    return (c * um[1] + s * a[1] + kw * w[1] + qm_h * E[1],
            c * um[2] + s * a[2] + kw * w[2] + qm_h * E[2],
            c * um[3] + s * a[3] + kw * w[3] + qm_h * E[3])
end

"""
    push_particle(pusher, x, u, t, Δt, qm, E, B) -> (x′, u′)

One step of one particle in prescribed fields, for test-particle runs (WP-015): drift half a
step, push the momentum with the fields there, drift half a step,

    x̄ = x + (Δt/2)·u/γ(u),   u′ = push_momentum(pusher, u, E(x̄, t + Δt/2), B(x̄, t + Δt/2), qm·Δt/2),
    x′ = x̄ + (Δt/2)·u′/γ(u′).

`E` and `B` are callables `(x, t) -> NTuple{3}`; `x`, `u` are `NTuple{3,T}` with u = γv; `qm` is
q/m. This is the PIC leapfrog with u at integer and x at half steps (x̄ = xⁿ⁺¹ᐟ²), written so
that the returned state is synchronized: x′ and u′ both at t + Δt, and xⁿ = (xⁿ⁻¹ᐟ² + xⁿ⁺¹ᐟ²)/2.
In exact arithmetic two consecutive half drifts are the leapfrog's full drift; in floating point
they round differently. The momentum update is the same device function the PIC step calls.
"""
@inline function push_particle(pusher::Pusher, x::NTuple{3,T}, u::NTuple{3,T}, t::T, Δt::T, qm::T, E, B) where {T}
    h = Δt / 2
    g = h / lorentz_factor(u)
    x̄ = (x[1] + g * u[1], x[2] + g * u[2], x[3] + g * u[3])
    th = t + h
    u′ = push_momentum(pusher, u, E(x̄, th), B(x̄, th), qm * h)
    g′ = h / lorentz_factor(u′)
    return (x̄[1] + g′ * u′[1], x̄[2] + g′ * u′[2], x̄[3] + g′ * u′[3]), u′
end
