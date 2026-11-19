# A laser of any shape on the antenna (WP-066, A2, A3): the pulse is a transverse profile times a
# temporal envelope from the standalone module `PulseShapes`, which this file is the only user of.
# The antenna is the total-field/scattered-field plane of antenna.jl; only the incident field
# changes. With the field Re[F], F = a0·U(ξ, y)·A(s)·e^{−i(s − phase)}, the time factor
# G = a0·A(s)·e^{−i(s − phase)} is the same on every row of the plane, so the host computes it once
# per sub-step (a SpectralEnvelope's quadrature runs there, not on every row) and the kernels take
# it as an argument; they evaluate the profile U and its U″ per row.

"""
    ShapedLaser(; a0, delay, focus, position, center, profile, envelope, polarization = 0, phase = 0)

A laser pulse launched along +x (axis 1) through the antenna plane x = `position`, like a
[`GaussianLaser`](@ref), with its transverse profile and its temporal envelope given by shapes of
[`PulseShapes`](@ref): at the plane the field is

    E(y, t) = Re[a0 · U(ξ, y − center) · A(t − delay) · e^{−i(t − delay − phase)}] · (ŷ cos θ + ẑ sin θ)

with ξ = position − focus, `U = profile(profile, ξ, y)` (a [`GaussianBeam`](@ref) or a
[`PupilBeam`](@ref): a near field and a wavefront of aberration modes through a lens) and
`A = envelope(envelope, s)` (a [`GaussianEnvelope`](@ref) or a [`SpectralEnvelope`](@ref): a
spectral phase), θ = `polarization`. The magnetic field is that of a wave travelling along +x
with this transverse field to order k_y², `Re[(U ∓ U″/2)·…]` for the in-plane and the out-of-plane
polarization, as for a `GaussianLaser`. The envelope's time is the same at every y of the plane
(the [`AxisTime`](@ref) convention): the wavefront's curvature is in U.

`a0` is the peak field of the unaberrated beam at its focus with the transform-limited envelope;
an aberration or a spectral phase changes only phases, so the energy ([`laser_energy`](@ref)) is
kept and the peak drops. With `GaussianBeam(; waist)` and `GaussianEnvelope(; duration, chirp)`
the incident field is that of `GaussianLaser(; a0, waist, duration, chirp, …)` (T-904).

Parameters enter through the antenna only (A3): `a0`, `delay`, `focus`, `polarization`, `phase` and
the numbers of the profile and the envelope ([`PulseShapes.parameter_names`](@ref)), any of them
a `Dual`. `position` and `center` are configuration, as for a `GaussianLaser`. The antenna is
exact at ω₀ only (`questions/wp-026-antenna.md` §5): a spectral phase does not widen the band, so a
stretched pulse is launched as cleanly as its transform-limited one. A [`PupilBeam`](@ref) must
have an [`PulseShapes.aliasing_period`](@ref) of at least twice the box height (refused
otherwise, [`check_params`](@ref)). As for any beam, the field must be negligible at the box's y
boundary, or the driven field jumps there: the side lobes of a flattened or aberrated pupil reach
much further than its waist (a few percent of the peak 50 c/ω₀ from the axis at `na = 0.3`).
"""
struct ShapedLaser{T,P,B<:TransverseProfile,E<:TemporalEnvelope}
    a0::T
    delay::T
    focus::T
    polarization::T
    phase::T
    profile::B
    envelope::E
    position::P
    center::P
end

function ShapedLaser(; a0, delay, focus, position, center, profile::TransverseProfile, envelope::TemporalEnvelope,
                     polarization = 0, phase = 0)
    T = promote_type(typeof(a0), typeof(delay), typeof(focus), typeof(polarization), typeof(phase),
                     map(typeof, parameter_values(profile))..., map(typeof, parameter_values(envelope))...)
    T = T <: Union{Integer,Rational} ? Float64 : T
    P = promote_type(typeof(position), typeof(center))
    return shaped_laser(T, P, (a0, delay, focus, polarization, phase), profile, envelope, position, center)
end

function shaped_laser(::Type{T}, ::Type{P}, x::Tuple, profile, envelope, position, center) where {T,P}
    b, e = convert_eltype(T, profile), convert_eltype(T, envelope)
    return ShapedLaser{T,P,typeof(b),typeof(e)}(map(T, x)..., b, e, P(position), P(center))
end

convert_laser(::Type{T}, l::ShapedLaser) where {T} =
    shaped_laser(T, T, (l.a0, l.delay, l.focus, l.polarization, l.phase), l.profile, l.envelope, l.position, l.center)

is_laser(l) = l isa GaussianLaser || l isa ShapedLaser

"""
    laser_energy(laser::ShapedLaser) -> W

The energy per unit length in z that the incident field carries through the plane,
∫∫ (E × B)_x dy dt = (a0²/2)·∫|A|² ds·∫|U|² dy·(1 + cos(2θ)·⟨k_y²⟩/2), from the envelope's
[`PulseShapes.envelope_energy`](@ref) and the profile's [`PulseShapes.profile_moments`](@ref), up to
the envelope's overlap with twice the carrier frequency as for a [`GaussianLaser`](@ref).
"""
function laser_energy(l::ShapedLaser)
    power, ky2 = profile_moments(l.profile)
    return l.a0^2 / 2 * envelope_energy(l.envelope) * power * (1 + cos(2 * l.polarization) * ky2 / 2)
end

pulse_duration(l::ShapedLaser) = envelope_extent(l.envelope)[2]
function pulse_end(l::ShapedLaser)
    c, h = envelope_extent(l.envelope)
    return l.delay + c + 6 * h
end

# The time factor a0·A(s)·e^{−i(s − phase)} of the plane at the time s after the delay.
@inline time_factor(l::ShapedLaser, s) = l.a0 * envelope(l.envelope, s) * cis(l.phase - s)

# e = Re[U·G], bp = Re[(U − U″/2)·G], bs = Re[(U + U″/2)·G] at (ξ, dy) for the time factor G.
@inline function shaped_field(l::ShapedLaser, ξ, dy, g::Complex)
    u, u2 = profile(l.profile, ξ, dy)
    ug = u * g
    hg = u2 * g / 2
    return real(ug), real(ug) - real(hg), real(ug) + real(hg)
end

@inline laser_field(l::ShapedLaser, ξ, dy, s) = shaped_field(l, ξ, dy, time_factor(l, s))

function check_profile(a::AntennaPlane, l::ShapedLaser)
    aliasing_period(l.profile) >= 2 * a.len[2] ||
        throw(ArgumentError("the laser's profile repeats every $(aliasing_period(l.profile)) along y, less than twice the box height $(a.len[2]): raise its samples"))
    return nothing
end

# The kernels of antenna.jl with the time factor `g` of the sub-step's time in place of the time.
@kernel function shaped_b_kernel!(By, Bz, a, c, g)
    j = Int32(@index(Global, Linear))
    T = typeof(c)
    i = a.g + a.ia - Int32(1)
    l = a.laser
    ξ = a.xa - l.focus
    sθ, cθ = sincos(l.polarization)
    @inbounds begin
        ec, _, _ = shaped_field(l, ξ, antenna_dy(a, j, one(T) / 2), g)
        en, _, _ = shaped_field(l, ξ, antenna_dy(a, j, zero(T)), g)
        Bz[i, j] += c * (ec * cθ)
        By[i, j] -= c * (en * sθ)
    end
end

@kernel function shaped_e_kernel!(Ey, Ez, a, c, g)
    j = Int32(@index(Global, Linear))
    T = typeof(c)
    i = a.g + a.ia
    l = a.laser
    ξ = a.xa - a.hx - l.focus
    sθ, cθ = sincos(l.polarization)
    @inbounds begin
        _, bp, _ = shaped_field(l, ξ, antenna_dy(a, j, one(T) / 2), g)
        _, _, bs = shaped_field(l, ξ, antenna_dy(a, j, zero(T)), g)
        Ey[i, j] += c * (bp * cθ)
        Ez[i, j] += c * (bs * sθ)
    end
end

b_source!(a::Antenna{<:ShapedLaser}, f, c, t, wgs) =
    launcher(shaped_b_kernel!, get_backend(f.Bz), wgs)(f.By, f.Bz, a, c, time_factor(a.laser, t - a.laser.delay);
                                                       ndrange = size(f.Bz, 2))
e_source!(a::Antenna{<:ShapedLaser}, f, c, t, wgs) =
    launcher(shaped_e_kernel!, get_backend(f.Ey), wgs)(f.Ey, f.Ez, a, c, time_factor(a.laser, t + a.ht - a.laser.delay);
                                                       ndrange = size(f.Ey, 2))
