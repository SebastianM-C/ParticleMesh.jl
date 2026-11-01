# Pushers: the momentum update of one particle in given fields, as a device function. The step
# dispatches on the pusher type, so each pusher is a subtype and a method of
# `push_momentum`, with no change to the step (WP-015 adds Vay, HigueraCary, BorisExactRotation;
# WP-016 adds QuadraticUnitary).

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
θ² = √eps(T) the three come from their Taylor series (truncation below eps(T)^(3/2)/720); above, from
sin and cos of θ = √θ². The two are chosen with `ifelse`, a select of two computed values,
not a branch, and the square root sees max(θ², √eps(T)), so duals stay finite through B = 0.

For θ² < 1 the rotation is written in the two-cross-product form of `Boris`,
`u⁺ = u⁻ + (u⁻ + u⁻×t)×s` with t = tan(θ/2)·ŵ and s = sin θ·ŵ, the Cayley form at the exact
half angle. Rodrigues' formula written directly multiplies u⁻ by the rounded cos θ ≈ 1 and drifts
linearly in |u| where θ² ≲ eps(T) (#203, T-245); the Boris form rounds only the corrections.
For θ² ≥ 1, where tan(θ/2) is unbounded at θ = π, Rodrigues' formula is used; the two are
chosen with `ifelse`, and the unused value stays finite.
"""
struct BorisExactRotation <: Pusher end

@inline dot3(a::NTuple{3,T}, b::NTuple{3,T}) where {T} = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"""
    eq20_gamma(u, τ)

The γ of Higuera and Cary's Eq. 20 for momentum `u` and `τ = qm_h·B`: the positive root of
γ⁴ − σγ² − q = 0, σ = 1 + u² − τ², q = τ² + (u·τ)². For σ ≥ 0 it is γ² = (σ + √(σ² + 4q))/2;
for σ < 0 (|τ| > γ(u), ΩΔt/γ > 2) that sum cancels, and the equal form 2q/(√(σ² + 4q) − σ) is
used instead (q > 0 there). The two are chosen with `ifelse`, a select, not a branch; the
second's denominator is itself selected (1 where σ ≥ 0), so the unused value is finite for
every input.
"""
@inline function eq20_gamma(u::NTuple{3,T}, τ::NTuple{3,T}) where {T}
    τ² = dot3(τ, τ)
    uτ = dot3(u, τ)
    σ = one(T) + dot3(u, u) - τ²
    q = τ² + uτ * uτ
    r = sqrt(σ * σ + 4 * q)
    # the unused branch must stay finite too (a select evaluates both, and a reverse-mode adjoint
    # multiplies the unused one's partials by zero): its denominator is selected on the input,
    # 1 where σ ≥ 0, so 2q/d is finite for every input; where σ < 0, r − σ > −σ > 0
    d = ifelse(σ >= zero(T), one(T), r - σ)
    return sqrt(ifelse(σ >= zero(T), (σ + r) / 2, 2 * q / d))
end

# u⁺ solving u⁺ = v + u⁺ × t: u⁺ = (v + (v·t)t + v×t)/(1 + t²) (the inverse of I − [t]ₓ),
# written as the increment v + ((v·t)t + v×t − t²v)/(1 + t²). Scaling the whole vector by the
# rounded 1/(1 + t²) changes |u| by the same relative amount every step, a linear drift where
# t² ≲ eps(T) (#203, T-245); the increment form rounds only the small correction.
@inline function solve_rotation(v::NTuple{3,T}, t::NTuple{3,T}) where {T}
    t² = dot3(t, t)
    s = one(T) / (one(T) + t²)
    vt = dot3(v, t)
    c = cross3(v, t)
    return (v[1] + s * (vt * t[1] + c[1] - t² * v[1]),
            v[2] + s * (vt * t[2] + c[2] - t² * v[2]),
            v[3] + s * (vt * t[3] + c[3] - t² * v[3]))
end

"""
    push_momentum(::Vay, u, E, B, qm_h) -> u′

`u′ = u + qm_h·(2E + u×B/γ(u))`, `τ = qm_h·B`, `γ⁺` from Eq. 20 of Higuera and Cary applied to
u′ (Vay's Eq. 11), `t = τ/γ⁺`, and `u⁺ = u′ + ((u′·t)t + u′×t − t²u′)/(1 + t²)`, the
solution of `u⁺ = u′ + u⁺×t` written as an increment on u′ (T-245).
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
    # Taylor series in x = θ², truncated after x²: the largest omitted term, x³/720 in cos θ, is
    # below eps(T)·√eps(T)/720 for x < √eps(T), well under eps(T)
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

`u⁻ = u + qm_h·E`; `w = 2qm_h·B/γ(u⁻)`, θ² = |w|²; the rotation of u⁻ about w by θ, for
θ² < 1 as `u⁻ + (u⁻ + u⁻×t)×s` with `t = (tan(θ/2)/θ)·w`, `s = (sin θ/θ)·w`, and for θ² ≥ 1 as
`cos θ·u⁻ + (sin θ/θ)·u⁻×w + ((1 − cos θ)/θ²)·(u⁻·w)·w` (Rodrigues); returns `u⁺ + qm_h·E`.
tan(θ/2)/θ is ((1 − cos θ)/θ²)/(sin θ/θ).
"""
@inline function push_momentum(::BorisExactRotation, u::NTuple{3,T}, E::NTuple{3,T}, B::NTuple{3,T}, qm_h::T) where {T}
    um = (u[1] + qm_h * E[1], u[2] + qm_h * E[2], u[3] + qm_h * E[3])
    f = 2 * qm_h / lorentz_factor(um)
    w = (f * B[1], f * B[2], f * B[3])
    x = dot3(w, w)
    c, s, k = rotation_coefficients(x)
    small = x < one(T)
    # the Boris form: tan(θ/2)/θ = k/s, its denominator selected (1 where θ² ≥ 1, so the unused
    # value is finite at θ = π, where s = 0)
    q = k / ifelse(small, s, one(T))
    t = (q * w[1], q * w[2], q * w[3])
    sv = (s * w[1], s * w[2], s * w[3])
    a = cross3(um, t)
    b = cross3((um[1] + a[1], um[2] + a[2], um[3] + a[3]), sv)
    # Rodrigues
    r = cross3(um, w)
    kw = k * dot3(um, w)
    return (ifelse(small, um[1] + b[1], c * um[1] + s * r[1] + kw * w[1]) + qm_h * E[1],
            ifelse(small, um[2] + b[2], c * um[2] + s * r[2] + kw * w[2]) + qm_h * E[2],
            ifelse(small, um[3] + b[3], c * um[3] + s * r[3] + kw * w[3]) + qm_h * E[3])
end

# --- WP-016: the quadratic-unitary pusher --------------------------------------------------------

"""
    QuadraticUnitary()

The quadratic-unitary pusher of Gordon and Hafizi ("Special unitary particle pusher for extreme
fields", arXiv:2006.10146 (2020), Eq. 18): the four-velocity (γ, u) as the spinor
ζ = γ + σ·u (their Eq. 3), advanced over a proper-time step Δs by ζ′ = ΛζΛ† (Eq. 4) with the
second-order invariant truncation of the exact operator,

    Λ = (1 + σ·Ψ + Ψ·Ψ/4)/(1 − Ψ·Ψ/4),   Ψ = (q/m)(E + iB)Δs/2,

and the proper-time step from the lab step by their second-order Eq. 22,
`Δs = (Δt/γ)(1 − (q/m)(E·u)Δt/(2γ²))`, evaluated at the incoming momentum uⁿ⁻¹ᐟ².

Properties (their Secs. 3 and 4), in exact arithmetic for the fields held at the particle:

- Minkowski norm: preserved exactly (det Λ = 1): the γ the update returns equals √(1 + |u′|²)
  (T-241, through [`quadratic_unitary_step`](@ref)).
- Energy for E = 0: exact; the rotation angle per step is 4·atan(ΩΔt/(4γ)), against
  2·atan(ΩΔt/(2γ)) for Boris (T-241).
- Plane-wave (null) fields, E·B = 0 and |E| = |B|: Ψ·Ψ = 0 and Λ is the exact operator, so
  k·u, here γ − u_z, is preserved by every step; the lab-time advance is still second order,
  through Eq. 22 and the field varying across the step.
- Crossed fields with E·B = 0: Λ is the exact operator at a rescaled step, so a particle moving
  at E×B/B² (E < B) stays there (T-242).
- Uniform E, B = 0: u is not exact, unlike Boris, Vay and Higuera–Cary (T-242).

The update is written in real arithmetic, with the common real denominator |1 − Ψ·Ψ/4|²
factored out as the paper suggests, and as an increment on (γ, u) so that a step with a small
rotation rounds like Boris (T-245). It has no branch in the fields. The module's particle
state is u alone, so γ = √(1 + |u|²) is recomputed on entry: one square root, as in Boris
(the paper carries γ and steps in proper time, which needs none). The denominator vanishes
where E·B = 0 and (q/m)²(E² − B²)Δs² = 16, a pure electric kick of |q|EΔs/m = 4; Eq. 22's
Δs vanishes where (q/m)(E·u)Δt = 2γ². Both lie outside any resolved step.
"""
struct QuadraticUnitary <: Pusher end

"""
    quadratic_unitary_step(γ, u, e, h) -> (γ′, u′)

The four-velocity update of [`QuadraticUnitary`](@ref) for e = κE, h = κB, κ = (q/m)Δs/2
(Ψ = e + ih): with α = 1 + (e² − h²)/4, β = (e·h)/2 and D² = |1 − Ψ·Ψ/4|² = (2 − α)² + β²,

    γ′ = γ + (2/D²)·(e²γ + (αe + βh)·u − u·(e×h)),
    u′ = u + (2/D²)·(−h²u + γ(αe + βh + e×h) + (e·u)e + (h·u)h + u×(αh − βe)),

the real expansion of ζ′ = ΛζΛ†. γ is taken as given, not recomputed from u, so the
Minkowski norm of the result can be checked (T-241).
"""
@inline function quadratic_unitary_step(γ::T, u::NTuple{3,T}, e::NTuple{3,T}, h::NTuple{3,T}) where {T}
    e² = dot3(e, e)
    h² = dot3(h, h)
    α = one(T) + (e² - h²) / 4
    β = dot3(e, h) / 2
    δ = 2 - α
    g = 2 / (δ * δ + β * β)
    eu = dot3(e, u)
    hu = dot3(h, u)
    s = cross3(e, h)
    r = cross3(u, (α * h[1] - β * e[1], α * h[2] - β * e[2], α * h[3] - β * e[3]))
    p = (α * e[1] + β * h[1], α * e[2] + β * h[2], α * e[3] + β * h[3])
    γ′ = γ + g * (e² * γ + dot3(p, u) - dot3(u, s))
    u′ = (u[1] + g * (-h² * u[1] + γ * (p[1] + s[1]) + eu * e[1] + hu * h[1] + r[1]),
          u[2] + g * (-h² * u[2] + γ * (p[2] + s[2]) + eu * e[2] + hu * h[2] + r[2]),
          u[3] + g * (-h² * u[3] + γ * (p[3] + s[3]) + eu * e[3] + hu * h[3] + r[3]))
    return γ′, u′
end

"""
    quadratic_unitary_kappa(u, E, qm_h) -> (γ, κ)

γ = √(1 + |u|²) and κ = (q/m)Δs/2 of [`QuadraticUnitary`](@ref) from qm_h = (q/m)Δt/2 and
Gordon and Hafizi's Eq. 22: κ = (qm_h/γ)(1 − qm_h·(E·u)/γ²).
"""
@inline function quadratic_unitary_kappa(u::NTuple{3,T}, E::NTuple{3,T}, qm_h::T) where {T}
    γ = lorentz_factor(u)
    ig = one(T) / γ
    return γ, qm_h * ig * (one(T) - qm_h * dot3(E, u) * ig * ig)
end

"""
    push_momentum(::QuadraticUnitary, u, E, B, qm_h) -> u′

κ from Eq. 22 ([`quadratic_unitary_kappa`](@ref)), then [`quadratic_unitary_step`](@ref) with
e = κE, h = κB; returns u′ and drops γ′, which equals √(1 + |u′|²) to roundoff.
"""
@inline function push_momentum(::QuadraticUnitary, u::NTuple{3,T}, E::NTuple{3,T}, B::NTuple{3,T}, qm_h::T) where {T}
    γ, κ = quadratic_unitary_kappa(u, E, qm_h)
    _, u′ = quadratic_unitary_step(γ, u, (κ * E[1], κ * E[2], κ * E[3]), (κ * B[1], κ * B[2], κ * B[3]))
    return u′
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
