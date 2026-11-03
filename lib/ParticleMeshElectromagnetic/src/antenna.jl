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
# the antenna (T-384). The scattered region receives what the incident field fails to satisfy
# of the discrete Maxwell equations, the leakage measured in T-380.

"""
    GaussianLaser(; a0, waist, duration, delay, focus, position, center, chirp = 0, polarization = 0, phase = 0)

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

Parameters enter through the antenna only (A3): `a0`, `waist`, `duration`, `delay`, `chirp`,
`phase` and `polarization` through the source amplitude and time dependence, `focus` (and the
`waist`) through the antenna's transverse amplitude and phase; any of them may be a `Dual`.
`position` and `center` place the plane and the axis; the state puts the plane on the nearest
primal node (`position` is not differentiable). The energy the pulse carries through the plane,
∫∫ (E × B)_x dy dt, is about π·a0²·waist·duration/4 per unit length in z ([`laser_energy`](@ref)).

The pulse should start in vacuum: the envelope at t = 0 is exp(−delay²/duration²), so a delay of
about six durations starts it below Float64 roundoff.
"""
struct GaussianLaser{T,P}
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
end

function GaussianLaser(; a0, waist, duration, delay, focus, position, center, chirp = 0, polarization = 0, phase = 0)
    T = promote_type(typeof(a0), typeof(waist), typeof(duration), typeof(delay), typeof(focus), typeof(chirp),
                     typeof(polarization), typeof(phase))
    P = promote_type(typeof(position), typeof(center))
    return GaussianLaser{T,P}(T(a0), T(waist), T(duration), T(delay), T(focus), T(chirp), T(polarization), T(phase),
                              P(position), P(center))
end

"""
    laser_energy(laser::GaussianLaser) -> W

The energy per unit length in z that the antenna's incident field carries through its plane,
∫∫ (E × B)_x dy dt = (π/4)·a0²·waist·duration·(1 + cos(2θ)/(2·waist²)), θ the polarization, up
to terms of order exp(−duration²/2). The transverse integral of |u|² is waist·√(π/2) at every
ξ, the envelope's ∫ exp(−2s²/duration²)·cos²φ ds is duration·√(π/2)/2, and the magnetic field of
[`laser_field`](@ref) adds ±∫|∂u/∂y|²/2 dy, a fraction ⟨k_y²⟩/2 = 1/(2·waist²) of it, with the
sign of the polarization: a focused beam polarized in the plane carries more energy than its
transverse E alone, one polarized along z less.
"""
laser_energy(l::GaussianLaser) = π * l.a0^2 * l.waist * l.duration / 4 * (1 + cos(2 * l.polarization) / (2 * l.waist^2))

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
"""
@inline function laser_field(l::GaussianLaser, ξ, dy, s)
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

"""
    Antenna{L,T}

A [`GaussianLaser`](@ref) placed on a 2D grid by the state, isbits for the device: the primal
column `ia` (interior index along axis 1) of the plane, its position `xa = (ia − 1)·Δx`, the cell
sizes, the guard width, the transverse cells and the box height (for the nearest periodic image
of the axis), the half cell `hx` from the E plane to the B plane behind it and the time `ht` by
which the incident wave reaches the B plane earlier: `ht = k·Δx/2` with `k` the Yee wavenumber
of ω₀ = 1 along x, `sin(kΔx/2)/Δx = sin(Δt/2)/Δt`, the phase of the discrete plane wave between
the two planes (with c·Δx/2 instead, the antenna would leak (k − 1)·Δx/4 of the amplitude).
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

"""
    Antenna(laser::GaussianLaser, grid::PeriodicGrid{2}, dt) -> Antenna

Place `laser` on `grid` for the time step `dt` (whose type is the element type): its plane on the primal column nearest to `laser.position`, which must
leave at least the grid's guard width of interior columns on either side of the corrected
columns `ia − 1` and `ia` (so the corrections never touch an x-guard and the plane is not split by
the periodic seam; an `ArgumentError` otherwise).
"""
function Antenna(l::GaussianLaser, grid::PeriodicGrid{2}, dt::T) where {T}
    n, h, g = ncells(grid), cellsize(grid), guards(grid)
    ia = round(Int, l.position / h[1]) + 1
    g + 2 <= ia <= n[1] - g ||
        throw(ArgumentError("antenna position $(l.position) is column $ia; it must lie in columns $(g + 2) to $(n[1] - g) of the $(n[1]) cells, away from the periodic seam"))
    hx = T(h[1]) / 2
    k = asin(T(h[1]) / dt * sin(dt / 2)) / hx
    return Antenna(convert_laser(T, l), Int32(ia), T((ia - 1) * h[1]), T.(h), Int32(g), Int32(n[2]),
                   T(n[2] * h[2]), hx, k * hx)
end

convert_laser(::Type{T}, l::GaussianLaser) where {T} =
    GaussianLaser{T,T}(T(l.a0), T(l.waist), T(l.duration), T(l.delay), T(l.focus), T(l.chirp), T(l.polarization),
                       T(l.phase), T(l.position), T(l.center))

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

antenna_b!(::Tuple{}, f, h, t, wgs) = nothing
function antenna_b!(as::Tuple, f, h, t, wgs)
    for a in as
        T = eltype(f.Bz)
        launcher(antenna_b_kernel!, get_backend(f.Bz), wgs)(f.By, f.Bz, a, T(h) / a.dx[1], T(t); ndrange = size(f.Bz, 2))
    end
    return nothing
end

antenna_e!(::Tuple{}, f, dt, t, wgs) = nothing
function antenna_e!(as::Tuple, f, dt, t, wgs)
    for a in as
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
