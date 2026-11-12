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

The pulse should start in vacuum: the envelope at t = 0 is exp(−delay²/duration²), so a delay of
about six durations starts it below Float64 roundoff.

The antenna is exact at the carrier frequency only: its B (the k_y² term and the half-cell delay)
is that of ω₀ = 1, so a plane wave at ω₀ is launched exactly and the rest of the band, about
1/duration wide and wider with chirp, leaks behind the plane and deviates from its prescribed
profile in proportion. It is meant for narrow-band pulses (`questions/wp-026-antenna.md` §5).
"""
struct GaussianLaser{T,P,E}
    a0::T
    waist::T
    duration::T
    delay::T
    focus::T
    chirp::T
    polarization::T
    phase::T
    position::P
    center::P
    envelope::E
end

function GaussianLaser(; a0, waist, duration, delay, focus, position, center, chirp = 0, polarization = 0, phase = 0,
                       envelope::Union{AxisTime,WavefrontTime} = AxisTime())
    T = promote_type(typeof(a0), typeof(waist), typeof(duration), typeof(delay), typeof(focus), typeof(chirp),
                     typeof(polarization), typeof(phase))
    P = promote_type(typeof(position), typeof(center))
    return GaussianLaser{T,P,typeof(envelope)}(T(a0), T(waist), T(duration), T(delay), T(focus), T(chirp), T(polarization),
                                               T(phase), P(position), P(center), envelope)
end

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
function laser_energy(l::GaussianLaser{<:Any,<:Any,WavefrontTime})
    zR = l.waist^2 / 2
    ξ = l.position - l.focus
    ky2 = (1 + ξ^2 / (zR^2 + ξ^2) * (1 / l.duration^2 + l.chirp^2 * l.duration^2)) / l.waist^2
    return π * l.a0^2 * l.waist * l.duration / 4 * (1 + cos(2 * l.polarization) * ky2 / 2)
end

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
    sφ, cφ = sincos(s + l.chirp * s * s - l.phase - ψ)
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
    sφ, cφ = sincos(σ + l.chirp * σ * σ - l.phase + atan(a) / 2)
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
    AntennaPlane{T}

Where a laser's antenna lies on a 2D grid, the part of an [`Antenna`](@ref) that a state holds
(configuration; the laser itself is a parameter, [`EMParams`](@ref)): the primal column `ia`
(interior index along axis 1) of the plane, its position `xa = (ia − 1)·Δx`, the cell sizes, the
guard width, the transverse cells and the box height (for the nearest periodic image of the
axis), the half cell `hx` from the E plane to the B plane behind it and the time `ht` by which the
incident wave reaches the B plane earlier: `ht = k·Δx/2` with `k` the Yee wavenumber of ω₀ = 1
along x, `sin(kΔx/2)/Δx = sin(Δt/2)/Δt`, the phase of the discrete plane wave between the two
planes (with c·Δx/2 instead, the antenna would leak (k − 1)·Δx/4 of the amplitude). The column is
the nearest primal node to the laser's `position`, an integer, so the position is not a
differentiable parameter (A3) and a laser placed elsewhere needs a new state
([`check_params`](@ref) refuses it).

    AntennaPlane(position, grid::PeriodicGrid{2}, dt)

The plane on the primal column nearest to `position` for the time step `dt` (whose type is the
element type), which must leave at least the grid's guard width of interior columns on either
side of the corrected columns `ia − 1` and `ia` (so the corrections never touch an x-guard and
the plane is not split by the periodic seam; an `ArgumentError` otherwise).
"""
struct AntennaPlane{T}
    ia::Int32
    xa::T
    dx::NTuple{2,T}
    g::Int32
    n2::Int32
    ly::T
    hx::T
    ht::T
end

function AntennaPlane(position, grid::PeriodicGrid{2}, dt::T) where {T}
    n, h, g = ncells(grid), cellsize(grid), guards(grid)
    ia = round(Int, position / h[1]) + 1
    g + 2 <= ia <= n[1] - g ||
        throw(ArgumentError("antenna position $(position) is column $ia; it must lie in columns $(g + 2) to $(n[1] - g) of the $(n[1]) cells, away from the periodic seam"))
    hx = T(h[1]) / 2
    k = asin(T(h[1]) / dt * sin(dt / 2)) / hx
    return AntennaPlane(Int32(ia), T((ia - 1) * h[1]), T.(h), Int32(g), Int32(n[2]), T(n[2] * h[2]), hx, k * hx)
end

# A laser of the parameters belongs on the plane the state built for it: within half a cell of the
# plane's column, up to the rounding of the position's conversion to the element type.
function check_plane(a::AntennaPlane{T}, l::GaussianLaser) where {T}
    c = l.position / a.dx[1]
    abs(c - (a.ia - 1)) <= one(T) / 2 + 8 * eps(T) * max(one(T), abs(c)) ||
        throw(ArgumentError("laser position $(l.position) is not on the state's antenna plane (column $(a.ia), x = $(a.xa)): the plane's column is fixed when the state is built, so a laser placed elsewhere needs a new state (A3: the position is not a differentiable parameter)"))
    return nothing
end

"""
    Antenna{L,T}

A [`GaussianLaser`](@ref) on an [`AntennaPlane`](@ref), isbits for the device: the laser and the
plane's fields (`ia`, `xa`, `dx`, `g`, `n2`, `ly`, `hx`, `ht`). The step makes it from the
parameters' laser and the state's plane, `Antenna(laser, plane)`.

    Antenna(laser::GaussianLaser, grid::PeriodicGrid{2}, dt)

places `laser`, converted to the type of `dt`, on its own plane `AntennaPlane(laser.position, grid, dt)`.
"""
struct Antenna{L,T}
    laser::L
    ia::Int32
    xa::T
    dx::NTuple{2,T}
    g::Int32
    n2::Int32
    ly::T
    hx::T
    ht::T
end

Antenna(l::GaussianLaser, a::AntennaPlane) = Antenna(l, a.ia, a.xa, a.dx, a.g, a.n2, a.ly, a.hx, a.ht)
Antenna(l::GaussianLaser, grid::PeriodicGrid{2}, dt::T) where {T} = Antenna(convert_laser(T, l), AntennaPlane(l.position, grid, dt))

convert_laser(::Type{T}, l::GaussianLaser) where {T} =
    GaussianLaser{T,T,typeof(l.envelope)}(T(l.a0), T(l.waist), T(l.duration), T(l.delay), T(l.focus), T(l.chirp), T(l.polarization),
                                          T(l.phase), T(l.position), T(l.center), l.envelope)

# The transverse distance from the axis of the row j (any index, guards included) of a field
# whose rows sit at (j₀ − 1 + s)·Δy, with j₀ the interior row j aliases: the guards get the bits
# of their interior images, and the nearest periodic image of the axis keeps the profile
# continuous across the seam.
@inline function antenna_dy(a::Antenna, j::Int32, s)
    T = typeof(a.xa)
    j0 = mod(j - a.g - Int32(1), a.n2)
    dy = (T(j0) + s) * a.dx[2] - a.laser.center
    return dy - a.ly * round(dy / a.ly)
end

# After the B half step to time t: Bz[ia−1] += c·Ey_inc(x_a, t), By[ia−1] −= c·Ez_inc(x_a, t),
# c = h/Δx; one work item per row, guards included. Bz rows sit at Center(), By rows at Node().
@kernel function antenna_b_kernel!(By, Bz, a, c, t)
    j = Int32(@index(Global, Linear))
    T = typeof(c)
    i = a.g + a.ia - Int32(1)
    l = a.laser
    s = t - l.delay
    ξ = a.xa - l.focus
    sθ, cθ = sincos(l.polarization)
    @inbounds begin
        ec, _, _ = laser_field(l, ξ, antenna_dy(a, j, one(T) / 2), s)
        en, _, _ = laser_field(l, ξ, antenna_dy(a, j, zero(T)), s)
        Bz[i, j] += c * (ec * cθ)
        By[i, j] -= c * (en * sθ)
    end
end

# After the E step to time t + Δt (the B field of the plane behind at t + Δt/2):
# Ey[ia] += c·Bz_inc and Ez[ia] −= c·By_inc at x_a − Δx/2, c = Δt/Δx, with Bz_inc = bp·cos θ and
# By_inc = −bs·sin θ. The incident wave at x_a − Δx/2 is that of x_a a time ht later.
@kernel function antenna_e_kernel!(Ey, Ez, a, c, t)
    j = Int32(@index(Global, Linear))
    T = typeof(c)
    i = a.g + a.ia
    l = a.laser
    s = t + a.ht - l.delay
    ξ = a.xa - a.hx - l.focus
    sθ, cθ = sincos(l.polarization)
    @inbounds begin
        _, bp, _ = laser_field(l, ξ, antenna_dy(a, j, one(T) / 2), s)
        _, _, bs = laser_field(l, ξ, antenna_dy(a, j, zero(T)), s)
        Ey[i, j] += c * (bp * cθ)
        Ez[i, j] += c * (bs * sθ)
    end
end

# An antenna acts while its plane lies in the box: always on a static grid (the constructor checks
# it), until the moving window has passed it otherwise (`window_antennas`, WP-028).
active(a::Antenna) = a.ia >= a.g + Int32(2)

antenna_b!(::Tuple{}, f, h, t, wgs) = nothing
function antenna_b!(as::Tuple, f, h, t, wgs)
    for a in as
        active(a) || continue
        T = eltype(f.Bz)
        launcher(antenna_b_kernel!, get_backend(f.Bz), wgs)(f.By, f.Bz, a, T(h) / a.dx[1], T(t); ndrange = size(f.Bz, 2))
    end
    return nothing
end

antenna_e!(::Tuple{}, f, dt, t, wgs) = nothing
function antenna_e!(as::Tuple, f, dt, t, wgs)
    for a in as
        active(a) || continue
        T = eltype(f.Ey)
        launcher(antenna_e_kernel!, get_backend(f.Ey), wgs)(f.Ey, f.Ez, a, T(dt) / a.dx[1], T(t); ndrange = size(f.Ey, 2))
    end
    return nothing
end

"""
    solve_fields!(::Yee, f, grid::PeriodicGrid{2}, dt, antennas::Tuple, t; workgroupsize = nothing)

The Yee step of [`solve_fields!`](@ref) from time `t` to `t + dt` with the antennas'
total-field/scattered-field corrections after each sub-step: after the first B half step the
incident E at time `t`, after the E step the incident B at `t + dt/2`, after the second B half
step the incident E at `t + dt` (see [`Antenna`](@ref)). With no antennas it is the plain step,
bit for bit. The corrected columns lie away from the x-guards, and the corrections write the
y-guards with the values of their interior images, so every guard stays filled.
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid{2}, dt, antennas::Tuple, t; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    yee_b_half!(f, grid, h; workgroupsize)
    antenna_b!(antennas, f, h, t, workgroupsize)
    yee_e!(f, grid, dt; workgroupsize)
    antenna_e!(antennas, f, dt, t + h, workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    antenna_b!(antennas, f, h, t + T(dt), workgroupsize)
    return f
end
