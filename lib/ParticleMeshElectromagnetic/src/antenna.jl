# The laser antenna (WP-026, A2, A3): a Gaussian beam launched through a plane of the 2D grid
# as a total-field/scattered-field (Huygens) source. With c = 1 (A1) and the Yee layout of
# `field_layout(Val(2))`, the plane is the primal-node column `ia` along axis 1: the E nodes of
# columns ≥ ia and the B nodes of columns ≥ ia (at x ≥ x_a + Δx/2) hold the total field, the
# columns behind it the scattered field. Only the x-differences of the curl cross the plane,
# so the incident field enters four updates, each through a known term of the curl:
#
#     Bz[ia−1] += h/Δx · Ey_inc(x_a, t)          By[ia−1] −= h/Δx · Ez_inc(x_a, t)
#     Ey[ia]   += Δt/Δx · Bz_inc(x_a − Δx/2, t + Δt/2)   Ez[ia] −= Δt/Δx · By_inc(x_a − Δx/2, t + Δt/2)
#
# (h = Δt/2, after each B half step at its time, and after the E step), the equivalent of an
# electric current sheet −n×B_inc/Δx at the E nodes and a magnetic one n×E_inc/Δx at the B nodes
# of the column behind. For an incident wave travelling along +x, Bz_inc = Ey_inc and
# By_inc = −Ez_inc. The sheets carry surface charges: the discrete divergence of E at the
# column ia and that of B at the cell column ia − 1 are those of the incident field's missing
# normal components (Ex_inc, Bx_inc), transient, back to zero when the pulse has passed up to
# the pulse's zero-frequency content; everywhere else Gauss's law and ∇·B = 0 hold as without
# the antenna (T-406). The scattered region receives what the incident field fails to satisfy
# of the discrete Maxwell equations, the leakage measured in T-401.

"""
    AxisTime()

The time of a [`GaussianLaser`](@ref)'s envelope and chirp, its default: the time after the
envelope peak on the axis, `t − delay`, the same at every y of the antenna plane. The wavefront's
curvature is in the carrier only.
"""
struct AxisTime end

"""
    WavefrontTime()

The time of a [`GaussianLaser`](@ref)'s envelope and chirp along the curved wavefront:
`σ = t − delay − (y − center)²/(2R)`, R = ξ·(1 + z_R²/ξ²) the wavefront's radius on the plane, the
time the carrier's phase has in both conventions. Every frequency of the pulse then has the beam
radius and the wavefront radius of ω₀ on the plane, and a plane ahead of the focus (R < 0) is
reached off the axis earlier than on it. On the axis the two conventions are the same field on the
plane, and differ past it: a distance L on, the [`AxisTime`](@ref) pulse arrives later by
`(w²/(4|R|))/(1 + w⁴·(1/L − 1/|R|)²/4)`, w the beam radius on the plane (T-696). This is the
convention of Smilei's `LaserGaussian2D` and of PIConGPU's `GaussianPulse`
(`scripts/lwfa/lwfa.jl`), for runs that are compared with them.
"""
struct WavefrontTime end

"""
    GaussianLaser(; a0, waist, duration, delay, focus, position, center, chirp = 0, polarization = 0, phase = 0,
                  envelope = AxisTime())

A Gaussian laser pulse launched along +x (axis 1) through the antenna plane x = `position`, in
A1 units (lengths in c/ω₀, times in 1/ω₀, fields in mₑcω₀/e). At the plane the field is

    E(y, t) = a0 · |u| · exp(−(t − delay)²/duration²) · cos(φ(t) − arg u) · (ŷ cos θ + ẑ sin θ)
    φ(t) = (t − delay) + chirp·(t − delay)² − phase,   θ = polarization

with u = (1 + iξ/z_R)^(−1/2) · exp(−(y − center)²/(waist²·(1 + iξ/z_R))), ξ = position − focus and
z_R = waist²/2: the paraxial Gaussian beam in two dimensions at ω₀ = 1, of 1/e field radius
`waist` at the focal plane x = `focus`, with the 2D Gouy phase −½ atan(ξ/z_R). So `a0` is the
peak field at the focus (a₀ = eE/(mₑcω₀)), `duration` the 1/e half width of the field envelope
(the intensity FWHM is duration·√(2 ln 2)), `delay` the time the envelope peak crosses the
plane, `chirp` the linear chirp (the instantaneous frequency at the plane is
1 + 2·chirp·(t − delay)), `phase` the carrier-envelope phase and `polarization` the angle of a
linear polarization from y towards z. The magnetic field of the incident wave is B = x̂ × E.
With `envelope = WavefrontTime()` the envelope and the chirp take the time along the curved
wavefront, `t − delay − (y − center)²/(2R)`, in place of `t − delay` ([`WavefrontTime`](@ref)).

Parameters enter through the antenna only (A3): `a0`, `waist`, `duration`, `delay`, `chirp`,
`phase` and `polarization` through the source amplitude and time dependence, `focus` (and the
`waist`) through the antenna's transverse amplitude and phase; any of them may be a `Dual`.
`position` and `center` place the plane and the axis; the state puts the plane on the nearest
primal node (`position` is not differentiable). The energy the pulse carries through the plane,
∫∫ (E × B)_x dy dt, is about π·a0²·waist·duration/4 per unit length in z ([`laser_energy`](@ref)).

At N = 3 (WP-073) `center = (y_c, z_c)` gives the axis, and the beam is the round paraxial Gaussian
beam in three dimensions, u = (1 + iξ/z_R)^(−1)·exp(−((y − y_c)² + (z − z_c)²)/(waist²·(1 + iξ/z_R))),
of Gouy phase −atan(ξ/z_R); the field is E = a0·Re[u·G]·(ŷ cos θ + ẑ sin θ) with G the time factor
above, so `a0` is again the peak of the paraxial incident field at the focus (the launched beam is
that field's propagation by the Yee scheme, which departs from the paraxial beam at order k_⊥⁴ and
by the scheme's dispersion, T-1022). Its energy through the plane is
π^{3/2}·a0²·waist²·duration/(4√2). A number as `center` is a beam of the 2D plane, a pair one of
the 3D box, and the problem checks that the laser fits its dimension.

The pulse should start in vacuum: the envelope at t = 0 is exp(−delay²/duration²), so a delay of
about six durations starts it below Float64 roundoff.

The antenna is exact at the carrier frequency only: its B (the k_y² term and the half-cell delay)
is that of ω₀ = 1, so a plane wave at ω₀ is launched exactly and the rest of the band, about
1/duration wide and wider with chirp, leaks behind the plane and deviates from its prescribed
profile in proportion. It is meant for narrow-band pulses (`questions/wp-026-antenna.md` §5).
"""
struct GaussianLaser{T,P,E,C}
    a0::T
    waist::T
    duration::T
    delay::T
    focus::T
    chirp::T
    polarization::T
    phase::T
    position::P
    center::C
    envelope::E
end

function GaussianLaser(; a0, waist, duration, delay, focus, position, center::Union{Real,NTuple{2,Real}}, chirp = 0, polarization = 0,
                       phase = 0, envelope::Union{AxisTime,WavefrontTime} = AxisTime())
    T = promote_type(typeof(a0), typeof(waist), typeof(duration), typeof(delay), typeof(focus), typeof(chirp),
                     typeof(polarization), typeof(phase))
    P = promote_type(typeof(position), map(typeof, Tuple(center))...)
    c = center_as(P, center)
    return GaussianLaser{T,P,typeof(envelope),typeof(c)}(T(a0), T(waist), T(duration), T(delay), T(focus), T(chirp), T(polarization),
                                                         T(phase), P(position), c, envelope)
end

# The axis of a laser in the type `P`: a number (y, at N = 2) or a pair (y, z, at N = 3).
center_as(::Type{P}, c::Real) where {P} = P(c)
center_as(::Type{P}, c::NTuple{2,Real}) where {P} = map(P, c)

# The number of transverse axes a laser's beam has: 1 (a beam of the 2D plane) or 2 (a round 3D beam).
transverse_axes(l) = length(Tuple(l.center))

"""
    laser_energy(laser::GaussianLaser) -> W

The energy per unit length in z that the antenna's incident field carries through its plane,
∫∫ (E × B)_x dy dt = (π/4)·a0²·waist·duration·(1 + cos(2θ)/(2·waist²)), θ the polarization, up
to terms of order exp(−duration²/(2(1 + chirp²·duration⁴))), the overlap of the envelope's
spectrum with twice the carrier frequency. The transverse integral of |u|² is waist·√(π/2) at every
ξ, the envelope's ∫ exp(−2s²/duration²)·cos²φ ds is duration·√(π/2)/2, and the magnetic field of
[`laser_field`](@ref) adds ±∫|∂u/∂y|²/2 dy, a fraction ⟨k_y²⟩/2 = 1/(2·waist²) of it, with the
sign of the polarization: a focused beam polarized in the plane carries more energy than its
transverse E alone, one polarized along z less. With [`WavefrontTime`](@ref) the pulse's time
depends on y through the wavefront, which adds its bandwidth on the curved part:
⟨k_y²⟩ = (1 + ξ²/(z_R² + ξ²)·(1/duration² + chirp²·duration²))/waist², ξ = position − focus.
"""
laser_energy(l::GaussianLaser) = π * l.a0^2 * l.waist * l.duration / 4 * (1 + cos(2 * l.polarization) / (2 * l.waist^2))
function laser_energy(l::GaussianLaser{<:Any,<:Any,WavefrontTime,<:Real})
    zR = l.waist^2 / 2
    ξ = l.position - l.focus
    ky2 = (1 + ξ^2 / (zR^2 + ξ^2) * (1 / l.duration^2 + l.chirp^2 * l.duration^2)) / l.waist^2
    return π * l.a0^2 * l.waist * l.duration / 4 * (1 + cos(2 * l.polarization) * ky2 / 2)
end

# When the pulse is over at its plane, for the checks that it starts in vacuum or that a moving
# window may pass the plane: six durations after the delay, the envelope below Float64 roundoff.
pulse_duration(l::GaussianLaser) = l.duration
pulse_end(l::GaussianLaser) = l.delay + 6 * l.duration

"""
    laser_energy(laser::GaussianLaser{T,P,E,<:NTuple{2}}) -> W

The energy that the incident field of a round 3D beam (a `center` of two coordinates) carries
through its plane, ∫∫∫ (E × B)_x dy dz dt = π^{3/2}·a0²·waist²·duration/(4√2), up to the overlap of
the envelope's spectrum with twice the carrier frequency as at N = 2. The transverse integral of
|u|² is π·waist²/2 at every ξ, the envelope's as at N = 2; the magnetic field of
[`laser_field`](@ref) adds ±(⟨k_y²⟩ − ⟨k_z²⟩)/2 and a term in ∂²u/∂y∂z, which vanish over the round
beam's plane (by its symmetry under y ↔ z and y → −y), for [`AxisTime`](@ref) and
[`WavefrontTime`](@ref) alike.
"""
laser_energy(l::GaussianLaser{<:Any,<:Any,<:Any,<:NTuple{2}}) = π * sqrt(π / 2) * l.a0^2 * l.waist^2 * l.duration / 4

"""
    laser_field(laser, ξ, dy, s) -> (e, bp, bs)

The incident field of `laser` at the distance `ξ` from the focal plane along x, the transverse
distance `dy` from the axis and the time `s` after the envelope peak at that x (the retarded
time minus the delay), before the polarization: the field `e = a0·Re[u·exp(−s²/duration²)·exp(−iφ)]`,
`φ = s + chirp·s² − phase`, and the magnetic amplitudes of a wave that travels along +x with
this transverse field, `bp = Re[(u − ∂²u/∂y²/2)·…]` for the in-plane polarization (Bz = bp·cos θ
with Ey = e·cos θ) and `bs = Re[(u + ∂²u/∂y²/2)·…]` for the polarization along z (By = −bs·sin θ
with Ez = e·sin θ). A plane-wave component at angle α to the axis has Bz/Ey = 1/cos α and
By/Ez = −cos α; the second derivative is their expansion to order α² ≈ k_y², so the incident E and
B agree with each other to order (k_y/k)⁴ and the antenna leaks less into the region behind it.
An isbits device function, generic in the element type.

With `laser.envelope = WavefrontTime()` the field is `e = Re F`,
`F = a0·|u|·exp(−σ²/duration²)·exp(−i(σ + chirp·σ² − phase + ½ atan(ξ/z_R)))`, `σ = s − dy²/(2R)`,
and `bp`, `bs` are `Re[F ∓ ∂²F/∂y²/2]` with the derivative of the whole F, its dependence on y
through σ included.
"""
@inline laser_field(l::GaussianLaser, ξ, dy, s) = laser_field(l.envelope, l, ξ, dy, s)

# The laser's functions and the antennas' kernels call sin and cos, never sincos: Enzyme's reverse
# pass of a KernelAbstractions kernel on a GPU cannot allocate a tape that holds a sincos of an
# active value (#547, T-1140). Inside a kernel the two differ by up to 2 ulp (T-1141).

@inline function laser_field(::AxisTime, l::GaussianLaser, ξ, dy, s)
    w0 = l.waist
    w2 = w0 * w0
    zR = w2 / 2
    a = ξ / zR
    d = 1 + a * a
    q = dy * dy / w2
    # u = (1 + ia)^(−1/2)·exp(−q·(1 − ia)/d): modulus d^(−1/4)·exp(−q/d), argument q·a/d − atan(a)/2.
    amp = exp(-q / d) / sqrt(sqrt(d))
    ψ = q * a / d - atan(a) / 2
    # ∂²u/∂y² = u·c₂ with c₂ = (4q/(1 + ia)² − 2/(1 + ia))/w0².
    d2 = d * d
    c2r = (4 * q * (1 - a * a) / d2 - 2 / d) / w2
    c2i = (2 * a / d - 8 * q * a / d2) / w2
    r = s / l.duration
    env = l.a0 * amp * exp(-r * r)
    φ = s + l.chirp * s * s - l.phase - ψ
    sφ, cφ = sin(φ), cos(φ)
    # Re[(1 + β·c₂)·e^{−i(φ − ψ)}] = (1 + β·c2r)·cos + β·c2i·sin.
    e = env * cφ
    bp = env * ((1 - c2r / 2) * cφ - c2i / 2 * sφ)
    bs = env * ((1 + c2r / 2) * cφ + c2i / 2 * sφ)
    return e, bp, bs
end

@inline function laser_field(::WavefrontTime, l::GaussianLaser, ξ, dy, s)
    w0 = l.waist
    w2 = w0 * w0
    zR = w2 / 2
    a = ξ / zR
    d = 1 + a * a
    q = dy * dy / w2
    # |u| = A = d^(−1/4)·exp(−q/d), and q·a/d = dy²/(2R) is the wavefront's delay
    amp = exp(-q / d) / sqrt(sqrt(d))
    σ = s - q * a / d
    τ2 = l.duration * l.duration
    env = l.a0 * amp * exp(-σ * σ / τ2)
    φ = σ + l.chirp * σ * σ - l.phase + atan(a) / 2
    sφ, cφ = sin(φ), cos(φ)
    # F = a0·A(y)·H(σ) up to a constant phase, H = exp(−σ²/τ² − i(σ + chirp·σ² − phase)), so with
    # p = A′/A, c = ∂(dy²/(2R))/∂y and h = H′/H = hr + i·hi:  ∂²F/∂y² = F·c₂,
    # c₂ = (p − c·h)² + (A″/A − p²) − c′·h + c²·h′,  h′ = −2/τ² − 2i·chirp.
    # A pulse without end has h = −i and c₂ = u″/u, the c₂ of AxisTime.
    p = -2 * dy / (w2 * d)
    c = 2 * dy * a / (w2 * d)
    c1 = 2 * a / (w2 * d)
    hr = -2 * σ / τ2
    hi = -(1 + 2 * l.chirp * σ)
    g = p - c * hr
    c2r = g * g - c * c * hi * hi - 2 / (w2 * d) - c1 * hr - 2 * c * c / τ2
    c2i = -2 * g * c * hi - c1 * hi - 2 * l.chirp * c * c
    # Re[(1 + β·c₂)·e^{−iφ}] = (1 + β·c2r)·cos φ + β·c2i·sin φ.
    e = env * cφ
    bp = env * ((1 - c2r / 2) * cφ - c2i / 2 * sφ)
    bs = env * ((1 + c2r / 2) * cφ + c2i / 2 * sφ)
    return e, bp, bs
end

"""
    laser_field(laser, ξ, dy, dz, s) -> (e, bp, bs, byz)

The incident field of the round 3D beam of `laser` (WP-073) at the distance `ξ` from the focal
plane, the transverse distances `dy`, `dz` from the axis and the time `s` after the envelope peak:
`e = Re[u·G]`, G = a0·exp(−s²/duration²)·exp(−iφ) as at N = 2 and u the 3D beam of
[`GaussianLaser`](@ref), and the magnetic amplitudes of a wave along +x with this transverse field,
to order k_⊥² as at N = 2: `bp = Re[(u − ∂²_y u/2 + ∂²_z u/2)·G]`, `bs = Re[(u + ∂²_y u/2 − ∂²_z u/2)·G]`
and `byz = Re[∂_y∂_z u·G]`, so that Ey = e·cos θ, Ez = e·sin θ come with Bz = bp·cos θ − byz·sin θ
and By = byz·cos θ − bs·sin θ (a plane-wave component (k_x, k_y, k_z) of Ey alone has
Bz/Ey = (k_x² + k_y²)/k_x ≈ 1 + k_y²/2 − k_z²/2 and By/Ey = −k_y·k_z/k_x). With
[`WavefrontTime`](@ref) the derivatives are those of the whole field, its dependence on y and z
through σ = s − r²/(2R) included. An isbits device function, generic in the element type.
"""
@inline laser_field(l::GaussianLaser, ξ, dy, dz, s) = laser_field(l.envelope, l, ξ, dy, dz, s)

# u″/u of the beam along one transverse axis at the distance x from its axis (q = x²/w²), the
# 2D formula of AxisTime: (4q/(1 + ia)² − 2/(1 + ia))/w², and u′/u = −2x/(w²(1 + ia)).
@inline function beam_terms(x, w2, a, d)
    q = x * x / w2
    d2 = d * d
    c2r = (4 * q * (1 - a * a) / d2 - 2 / d) / w2
    c2i = (2 * a / d - 8 * q * a / d2) / w2
    pr = -2 * x / (w2 * d)
    pi_ = 2 * x * a / (w2 * d)
    return q, c2r, c2i, pr, pi_
end

# The field and the magnetic amplitudes from the complex coefficients F_yy/F, F_zz/F, F_yz/F.
@inline function field_3d(env, cφ, sφ, yr, yi, zr, zi, xr, xi)
    e = env * cφ
    bp = env * ((1 - yr / 2 + zr / 2) * cφ + (zi - yi) / 2 * sφ)
    bs = env * ((1 + yr / 2 - zr / 2) * cφ + (yi - zi) / 2 * sφ)
    byz = env * (xr * cφ + xi * sφ)
    return e, bp, bs, byz
end

@inline function laser_field(::AxisTime, l::GaussianLaser, ξ, dy, dz, s)
    w2 = l.waist * l.waist
    zR = w2 / 2
    a = ξ / zR
    d = 1 + a * a
    qy, yr, yi, py, piy = beam_terms(dy, w2, a, d)
    qz, zr, zi, pz, piz = beam_terms(dz, w2, a, d)
    q = qy + qz
    # u = (1 + ia)^(−1)·exp(−q·(1 − ia)/d): modulus d^(−1/2)·exp(−q/d), argument q·a/d − atan(a).
    amp = exp(-q / d) / sqrt(d)
    ψ = q * a / d - atan(a)
    # ∂_y∂_z u/u = (u_y/u)(u_z/u), each −2x(1 − ia)/(w²d) = px + i·pix
    xr = py * pz - piy * piz
    xi = py * piz + piy * pz
    r = s / l.duration
    env = l.a0 * amp * exp(-r * r)
    φ = s + l.chirp * s * s - l.phase - ψ
    sφ, cφ = sin(φ), cos(φ)
    return field_3d(env, cφ, sφ, yr, yi, zr, zi, xr, xi)
end

@inline function laser_field(::WavefrontTime, l::GaussianLaser, ξ, dy, dz, s)
    w2 = l.waist * l.waist
    zR = w2 / 2
    a = ξ / zR
    d = 1 + a * a
    q = (dy * dy + dz * dz) / w2
    amp = exp(-q / d) / sqrt(d)
    σ = s - q * a / d
    τ2 = l.duration * l.duration
    env = l.a0 * amp * exp(-σ * σ / τ2)
    φ = σ + l.chirp * σ * σ - l.phase + atan(a)
    sφ, cφ = sin(φ), cos(φ)
    # As at N = 2 per axis: p = A′/A = −2x/(w²d), c = ∂(r²/(2R))/∂x = 2xa/(w²d), h = H′/H,
    # F_xx/F = (p − c·h)² − 2/(w²d) − c′·h + c²·h′ and F_yz/F = (p_y − c_y·h)(p_z − c_z·h) + c_y·c_z·h′.
    hr = -2 * σ / τ2
    hi = -(1 + 2 * l.chirp * σ)
    c1 = 2 * a / (w2 * d)
    py, pz = -2 * dy / (w2 * d), -2 * dz / (w2 * d)
    cy, cz = 2 * dy * a / (w2 * d), 2 * dz * a / (w2 * d)
    gy, gz = py - cy * hr, pz - cz * hr
    yr = gy * gy - cy * cy * hi * hi - 2 / (w2 * d) - c1 * hr - 2 * cy * cy / τ2
    yi = -2 * gy * cy * hi - c1 * hi - 2 * l.chirp * cy * cy
    zr = gz * gz - cz * cz * hi * hi - 2 / (w2 * d) - c1 * hr - 2 * cz * cz / τ2
    zi = -2 * gz * cz * hi - c1 * hi - 2 * l.chirp * cz * cz
    xr = gy * gz - cy * cz * hi * hi - 2 * cy * cz / τ2
    xi = -hi * (gy * cz + gz * cy) - 2 * l.chirp * cy * cz
    return field_3d(env, cφ, sφ, yr, yi, zr, zi, xr, xi)
end

"""
    AntennaPlane{T,N}

Where a laser's antenna lies on a 2D or 3D grid, the part of an [`Antenna`](@ref) that a state holds
(configuration; the laser itself is a parameter, [`EMParams`](@ref)): the primal column `ia`
(interior index along axis 1) of the plane, its position `xa = (ia − 1)·Δx`, the cell sizes `dx`, the
guard width, the cells `n` and the box lengths `len` per axis (the transverse ones for the nearest
periodic image of the axis), the half cell `hx` from the E plane to the B plane behind it and the time `ht` by which the
incident wave reaches the B plane earlier: `ht = k·Δx/2` with `k` the Yee wavenumber of ω₀ = 1
along x, `sin(kΔx/2)/Δx = sin(Δt/2)/Δt`, the phase of the discrete plane wave between the two
planes (with c·Δx/2 instead, the antenna would leak (k − 1)·Δx/4 of the amplitude). The column is
the nearest primal node to the laser's `position`, an integer, so the position is not a
differentiable parameter (A3) and a laser placed elsewhere needs a new state
([`check_params`](@ref) refuses it). `stencil` is the solver's B difference
([`faraday_stencil`](@ref)): [`YeeStencil`](@ref)`()` for the four corrections above, an
[`M4Stencil`](@ref) for the M4 solver's (`m4_antenna.jl`), whose wider B difference crosses the
plane on more columns and rows; then `k` is M4's wavenumber of ω₀ ([`m4_wavenumber`](@ref)) and
`ρ = sin(Δt/2)·Δx/(Δt·sin(kΔx/2))` the ratio of B to E of its discrete plane wave at ω₀, which
scales the incident B of the E correction (one for Yee, exactly).

    AntennaPlane(position, grid::PeriodicGrid{N}, dt, solver = Yee())

The plane on the primal column nearest to `position` for the time step `dt` (whose type is the
element type) and the solver, which must leave at least the grid's guard width of interior
columns on either side of the corrected columns (`ia − 1` and `ia` for Yee, `ia − 2` to `ia` for
M4), so the corrections never touch an x-guard and the plane is not split by the periodic seam
(an `ArgumentError` otherwise).
"""
struct AntennaPlane{T,N,S}
    ia::Int32
    xa::T
    dx::NTuple{N,T}
    g::Int32
    n::NTuple{N,Int32}
    len::NTuple{N,T}
    hx::T
    ht::T
    stencil::S
    ρ::T
end
AntennaPlane(ia, xa::T, dx, g, n, len, hx, ht) where {T} = AntennaPlane(ia, xa, dx, g, n, len, hx, ht, YeeStencil(), one(T))

function AntennaPlane(position, grid::PeriodicGrid{N}, dt::T, solver = Yee()) where {N,T}
    n, h, g = ncells(grid), cellsize(grid), guards(grid)
    st = faraday_stencil(solver, grid, dt)
    r = solver_reach(st)
    ia = round(Int, position / h[1]) + 1
    g + 1 + r <= ia <= n[1] - g ||
        throw(ArgumentError("antenna position $(position) is column $ia; it must lie in columns $(g + 1 + r) to $(n[1] - g) of the $(n[1]) cells, away from the periodic seam"))
    hx = T(h[1]) / 2
    k, ρ = plane_wavenumber(st, dt, T(h[1]))
    return AntennaPlane(Int32(ia), T((ia - 1) * h[1]), T.(h), Int32(g), Int32.(n), ntuple(d -> T(n[d] * h[d]), Val(N)), hx, k * hx, st, ρ)
end

# The wavenumber of ω₀ = 1 along x in the solver's scheme and the B/E ratio of its plane wave.
plane_wavenumber(::YeeStencil, dt::T, dx::T) where {T} = (asin(dx / dt * sin(dt / 2)) / (dx / 2), one(T))
function plane_wavenumber(st::M4Stencil, dt::T, dx::T) where {T}
    k = m4_wavenumber(dt, dx, st.δ[1])
    return (k, sin(dt / 2) * dx / (dt * sin(k * dx / 2)))
end

# A laser of the parameters belongs on the plane the state built for it: within half a cell of the
# plane's column, up to the rounding of the position's conversion to the element type.
function check_plane(a::AntennaPlane{T}, l) where {T}
    c = l.position / a.dx[1]
    abs(c - (a.ia - 1)) <= one(T) / 2 + 8 * eps(T) * max(one(T), abs(c)) ||
        throw(ArgumentError("laser position $(l.position) is not on the state's antenna plane (column $(a.ia), x = $(a.xa)): the plane's column is fixed when the state is built, so a laser placed elsewhere needs a new state (A3: the position is not a differentiable parameter)"))
    check_profile(a, l)
    return nothing
end
check_profile(::AntennaPlane, ::GaussianLaser) = nothing

"""
    Antenna{L,T,N}

A laser ([`GaussianLaser`](@ref) or [`ShapedLaser`](@ref)) on an [`AntennaPlane`](@ref), isbits for the device: the laser and the
plane's fields (`ia`, `xa`, `dx`, `g`, `n`, `len`, `hx`, `ht`). The step makes it from the
parameters' laser and the state's plane, `Antenna(laser, plane)`.

    Antenna(laser, grid::PeriodicGrid, dt)

places `laser`, converted to the type of `dt`, on its own plane `AntennaPlane(laser.position, grid, dt)`.
"""
struct Antenna{L,T,N,S}
    laser::L
    ia::Int32
    xa::T
    dx::NTuple{N,T}
    g::Int32
    n::NTuple{N,Int32}
    len::NTuple{N,T}
    hx::T
    ht::T
    stencil::S
    ρ::T
end
Antenna(l, ia, xa::T, dx, g, n, len, hx, ht) where {T} = Antenna(l, ia, xa, dx, g, n, len, hx, ht, YeeStencil(), one(T))

Antenna(l, a::AntennaPlane) = Antenna(l, a.ia, a.xa, a.dx, a.g, a.n, a.len, a.hx, a.ht, a.stencil, a.ρ)
Antenna(l, grid::PeriodicGrid, dt::T, solver = Yee()) where {T} = Antenna(convert_laser(T, l), AntennaPlane(l.position, grid, dt, solver))
# The antenna moved back by `s` columns, a moving window's (`window_antennas`).
shift_antenna(a::Antenna, s) = Antenna(a.laser, a.ia - Int32(s), a.xa, a.dx, a.g, a.n, a.len, a.hx, a.ht, a.stencil, a.ρ)

function convert_laser(::Type{T}, l::GaussianLaser) where {T}
    c = center_as(T, l.center)
    return GaussianLaser{T,T,typeof(l.envelope),typeof(c)}(T(l.a0), T(l.waist), T(l.duration), T(l.delay), T(l.focus), T(l.chirp),
                                                            T(l.polarization), T(l.phase), T(l.position), c, l.envelope)
end

# The transverse distance from the axis (its coordinate `c`) along `axis` of the row j (any index,
# guards included) of a field whose rows sit at (j₀ − 1 + s)·Δ, with j₀ the interior row j
# aliases: the guards get the bits of their interior images, and the nearest periodic image of
# the axis keeps the profile continuous across the seam.
@inline function antenna_d(a::Union{Antenna,AntennaPlane}, axis, c, j::Int32, s)
    T = typeof(a.xa)
    j0 = mod(j - a.g - Int32(1), a.n[axis])
    dy = (T(j0) + s) * a.dx[axis] - c
    return dy - a.len[axis] * round(dy / a.len[axis])
end
@inline antenna_dy(a::Antenna, j::Int32, s) = antenna_d(a, 2, a.laser.center, j, s)

# The incident field across a periodic axis is that of the nearest image of the beam's axis. Its even
# terms are continuous where the images meet, at the antipode of the axis, but the cross term ∂_y∂_z u
# of the 3D B is odd in each offset and changes sign there; a row within Δ/1000 of the antipode takes
# the mean of the two images, zero, so that moving the axis by whole cells moves the field with it.
@inline function antipode_weight(a::Union{Antenna,AntennaPlane}, axis, d)
    T = typeof(a.xa)
    return ifelse(abs(2 * abs(d) - a.len[axis]) * 1000 < a.dx[axis], zero(T), one(T))
end

# After the B half step to time t: Bz[ia−1] += c·Ey_inc(x_a, t), By[ia−1] −= c·Ez_inc(x_a, t),
# c = h/Δx; one work item per row, guards included. Bz rows sit at Center(), By rows at Node().
@kernel function antenna_b_kernel!(By, Bz, a, c, t)
    j = Int32(@index(Global, Linear))
    i = a.g + a.ia - Int32(1)
    bz, by = antenna_b_rows(a, j, t, typeof(c))
    @inbounds begin
        Bz[i, j] += c * bz
        By[i, j] -= c * by
    end
end

# What the kernels add per row j (and layer k at N = 3), before the factor c: one function per
# kernel, so that a test differentiates exactly what a kernel adds (T-1140). `T` is the type of c.
@inline function antenna_b_rows(a::Antenna{<:GaussianLaser,<:Any,2}, j::Int32, t, ::Type{T}) where {T}
    l = a.laser
    s = t - l.delay
    ξ = a.xa - l.focus
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    ec, _, _ = laser_field(l, ξ, antenna_dy(a, j, one(T) / 2), s)
    en, _, _ = laser_field(l, ξ, antenna_dy(a, j, zero(T)), s)
    return ec * cθ, en * sθ
end

# After the E step to time t + Δt (the B field of the plane behind at t + Δt/2):
# Ey[ia] += c·Bz_inc and Ez[ia] −= c·By_inc at x_a − Δx/2, c = Δt/Δx, with Bz_inc = bp·cos θ and
# By_inc = −bs·sin θ. The incident wave at x_a − Δx/2 is that of x_a a time ht later.
@kernel function antenna_e_kernel!(Ey, Ez, a, c, t)
    j = Int32(@index(Global, Linear))
    i = a.g + a.ia
    ey, ez = antenna_e_rows(a, j, t, typeof(c))
    @inbounds begin
        Ey[i, j] += c * ey
        Ez[i, j] += c * ez
    end
end

@inline function antenna_e_rows(a::Antenna{<:GaussianLaser,<:Any,2}, j::Int32, t, ::Type{T}) where {T}
    l = a.laser
    s = t + a.ht - l.delay
    ξ = a.xa - a.hx - l.focus
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    _, bp, _ = laser_field(l, ξ, antenna_dy(a, j, one(T) / 2), s)
    _, _, bs = laser_field(l, ξ, antenna_dy(a, j, zero(T)), s)
    return bp * cθ, bs * sθ
end

# The kernels at N = 3 (WP-073), one work item per (row, layer) of the plane, guards included. The
# same four corrections as at N = 2; Ey and Bz sit at (Center(), Node()) across the plane (y, z),
# Ez and By at (Node(), Center()), and each component of the incident wave is evaluated at its own
# place: after a B half step to time t, Bz[ia−1] += c·Ey_inc and By[ia−1] −= c·Ez_inc at x_a; after
# the E step, Ey[ia] += c·Bz_inc and Ez[ia] −= c·By_inc at x_a − Δx/2 a time ht later, with
# Bz_inc = bp·cos θ − byz·sin θ and By_inc = byz·cos θ − bs·sin θ ([`laser_field`](@ref)).
@kernel function antenna_b_kernel_3d!(By, Bz, a, c, t)
    J = @index(Global, Cartesian)
    j, k = Int32(J[1]), Int32(J[2])
    i = a.g + a.ia - Int32(1)
    bz, by = antenna_b_rows(a, j, k, t, typeof(c))
    @inbounds begin
        Bz[i, j, k] += c * bz
        By[i, j, k] -= c * by
    end
end

@inline function antenna_b_rows(a::Antenna{<:GaussianLaser,<:Any,3}, j::Int32, k::Int32, t, ::Type{T}) where {T}
    l = a.laser
    s = t - l.delay
    ξ = a.xa - l.focus
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    h = one(T) / 2
    ey, _, _, _ = laser_field(l, ξ, antenna_d(a, 2, l.center[1], j, h), antenna_d(a, 3, l.center[2], k, zero(T)), s)
    ez, _, _, _ = laser_field(l, ξ, antenna_d(a, 2, l.center[1], j, zero(T)), antenna_d(a, 3, l.center[2], k, h), s)
    return ey * cθ, ez * sθ
end

@kernel function antenna_e_kernel_3d!(Ey, Ez, a, c, t)
    J = @index(Global, Cartesian)
    j, k = Int32(J[1]), Int32(J[2])
    i = a.g + a.ia
    ey, ez = antenna_e_rows(a, j, k, t, typeof(c))
    @inbounds begin
        Ey[i, j, k] += c * ey
        Ez[i, j, k] += c * ez
    end
end

@inline function antenna_e_rows(a::Antenna{<:GaussianLaser,<:Any,3}, j::Int32, k::Int32, t, ::Type{T}) where {T}
    l = a.laser
    s = t + a.ht - l.delay
    ξ = a.xa - a.hx - l.focus
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    h = one(T) / 2
    dyc, dzn = antenna_d(a, 2, l.center[1], j, h), antenna_d(a, 3, l.center[2], k, zero(T))
    dyn, dzc = antenna_d(a, 2, l.center[1], j, zero(T)), antenna_d(a, 3, l.center[2], k, h)
    _, bp, _, xy = laser_field(l, ξ, dyc, dzn, s)
    _, _, bs, xz = laser_field(l, ξ, dyn, dzc, s)
    xy *= antipode_weight(a, 2, dyc) * antipode_weight(a, 3, dzn)
    xz *= antipode_weight(a, 2, dyn) * antipode_weight(a, 3, dzc)
    return bp * cθ - xy * sθ, bs * sθ - xz * cθ
end

# An antenna acts while its plane lies in the box: always on a static grid (the constructor checks
# it), until the moving window has passed it otherwise (`window_antennas`, WP-028).
active(a::Antenna) = a.ia >= a.g + Int32(1 + solver_reach(a.stencil))

antenna_b!(::Tuple{}, f, h, t, wgs) = nothing
function antenna_b!(as::Tuple, f, h, t, wgs)
    for a in as
        active(a) || continue
        T = eltype(f.Bz)
        b_source!(a, f, T(h) / a.dx[1], T(t), wgs)
    end
    return nothing
end
b_source!(a::Antenna{<:GaussianLaser,<:Any,2}, f, c, t, wgs) =
    launcher(antenna_b_kernel!, get_backend(f.Bz), wgs)(f.By, f.Bz, a, c, t; ndrange = size(f.Bz, 2))
b_source!(a::Antenna{<:GaussianLaser,<:Any,3}, f, c, t, wgs) =
    launcher(antenna_b_kernel_3d!, get_backend(f.Bz), wgs)(f.By, f.Bz, a, c, t; ndrange = (size(f.Bz, 2), size(f.Bz, 3)))

antenna_e!(::Tuple{}, f, dt, t, wgs) = nothing
function antenna_e!(as::Tuple, f, dt, t, wgs)
    for a in as
        active(a) || continue
        T = eltype(f.Ey)
        e_source!(a, f, T(dt) / a.dx[1] * a.ρ, T(t), wgs)
    end
    return nothing
end
e_source!(a::Antenna{<:GaussianLaser,<:Any,2}, f, c, t, wgs) =
    launcher(antenna_e_kernel!, get_backend(f.Ey), wgs)(f.Ey, f.Ez, a, c, t; ndrange = size(f.Ey, 2))
e_source!(a::Antenna{<:GaussianLaser,<:Any,3}, f, c, t, wgs) =
    launcher(antenna_e_kernel_3d!, get_backend(f.Ey), wgs)(f.Ey, f.Ez, a, c, t; ndrange = (size(f.Ey, 2), size(f.Ey, 3)))

"""
    AntennaPlanes(planes, clock)

The laser antennas of a state as one [`FieldModifier`](@ref). Its state half is the `planes`, one
[`AntennaPlane`](@ref) per laser of the problem, in the lasers' order, and the `clock` of a moving
window (a [`WindowClock`](@ref), `nothing` on a static grid): the planes stay in the lab frame, so
their window columns move back with the window's shifts. Its parameter half is the lasers of the
run's parameters (`params.lasers`, A3): at step `k` the [`Antenna`](@ref)s of those lasers on the
planes, where the window has them at the start of the step. After each Yee sub-step it adds the
total-field/scattered-field corrections: after the first B half step the incident E at the start
of the step, after the E step the incident B half a step later, after the second B half step the
incident E at the end. The corrected columns lie away from the x-guards, and the corrections
write the y-guards with the values of their interior images, so every guard stays filled.
"""
struct AntennaPlanes{P<:Tuple,C} <: FieldModifier
    planes::P
    clock::C
    AntennaPlanes(planes::Tuple, clock) = new{typeof(planes),typeof(clock)}(planes, clock)
end

antenna_planes(m::AntennaPlanes) = m.planes
modifier_params(m::AntennaPlanes, p, k) = placed_antennas(m.clock, map(Antenna, p.lasers, m.planes), k)
placed_antennas(::Nothing, antennas::Tuple, k) = antennas

after_substep!(::AntennaPlanes, antennas::Tuple, f, grid, s::BHalf, wgs) = antenna_b!(antennas, f, s.τ, s.t, wgs)
after_substep!(::AntennaPlanes, antennas::Tuple, f, grid, s::EStep, wgs) = antenna_e!(antennas, f, s.τ, s.t, wgs)
