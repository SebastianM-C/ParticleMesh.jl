# The transverse profiles: U(ξ, y), a solution of the paraxial equation 2i·∂U/∂ξ + ∂²U/∂y² = 0 in
# two dimensions (a slab beam: one transverse axis), normalized to 1 on the axis at the focus of
# the unaberrated beam.

"""
A transverse profile: [`profile`](@ref)`(b, ξ, y)` is `(U, U″)`, the complex profile at the distance
`ξ` from the focal plane and `y` from the axis and its second derivative along y.
"""
abstract type TransverseProfile end

"""
    GaussianBeam(; waist)

The paraxial Gaussian beam in two dimensions, `U = (1 + iξ/z_R)^(−1/2)·exp(−y²/(waist²·(1 + iξ/z_R)))`,
z_R = waist²/2: `waist` the 1/e field radius at the focus, the Gouy phase −½·atan(ξ/z_R). The
profile of ParticleMesh's `GaussianLaser`.
"""
struct GaussianBeam{T} <: TransverseProfile
    waist::T
end

GaussianBeam(; waist) = (T = shape_eltype(waist); GaussianBeam{T}(T(waist)))

@inline function profile(b::GaussianBeam, ξ, y)
    w2 = b.waist * b.waist
    a = 2 * ξ / w2
    d = 1 + a * a
    q = y * y / w2
    # U = (1 + ia)^(−1/2)·exp(−q·(1 − ia)/d): modulus d^(−1/4)·exp(−q/d), argument q·a/d − atan(a)/2;
    # U″ = U·c₂ with c₂ = (4q/(1 + ia)² − 2/(1 + ia))/w².
    m = exp(-q / d) / sqrt(sqrt(d))
    sψ, cψ = sincos(q * a / d - atan(a) / 2)
    d2 = d * d
    c2r = (4 * q * (1 - a * a) / d2 - 2 / d) / w2
    c2i = (2 * a / d - 8 * q * a / d2) / w2
    ur, ui = m * cψ, m * sψ
    return Complex(ur, ui), Complex(ur * c2r - ui * c2i, ur * c2i + ui * c2r)
end

"The near-field amplitude across the lens: [`pupil_amplitude`](@ref)`(p, ρ)`, ρ ∈ [−1, 1] the pupil coordinate."
abstract type PupilAmplitude end

"""
    GaussianPupil(; fill)

A Gaussian near field `exp(−ρ²/fill²)`, `fill` its 1/e field radius as a fraction of the aperture's
half width. With a fill well inside the aperture it focuses to the [`GaussianBeam`](@ref) of
waist `2/(na·fill)`.
"""
struct GaussianPupil{T} <: PupilAmplitude
    fill::T
end
GaussianPupil(; fill) = (T = shape_eltype(fill); GaussianPupil{T}(T(fill)))

"""
    SuperGaussianPupil(; radius, order)

A flattened near field `exp(−(ρ²/radius²)^order)`, the flat-top of a large amplifier: `radius` its
half width as a fraction of the aperture's, `order` an integer (1 is a Gaussian, higher is
flatter). `order` is configuration, not a parameter.
"""
struct SuperGaussianPupil{T} <: PupilAmplitude
    radius::T
    order::Int32
end
SuperGaussianPupil(; radius, order::Integer) = (T = shape_eltype(radius); SuperGaussianPupil{T}(T(radius), Int32(order)))

"""
    UniformPupil()

A uniformly filled aperture with a hard edge: amplitude 1 on |ρ| ≤ 1. It focuses to
`sin(na·y)/(na·y)` in the focal plane.
"""
struct UniformPupil <: PupilAmplitude end

@inline pupil_amplitude(p::GaussianPupil, ρ) = exp(-(ρ / p.fill)^2)
@inline function pupil_amplitude(p::SuperGaussianPupil, ρ)
    r = (ρ / p.radius)^2
    return exp(-r^p.order)
end
@inline pupil_amplitude(::UniformPupil, ρ) = one(ρ)

"""
    legendre(n, ρ)

The Legendre polynomial Pₙ(ρ) by its recurrence (n + 1)·Pₙ₊₁ = (2n + 1)·ρ·Pₙ − n·Pₙ₋₁.
"""
function legendre(n::Integer, ρ)
    p0, p1 = one(ρ), ρ
    n == 0 && return p0
    for m in 1:(n - 1)
        p0, p1 = p1, ((2m + 1) * ρ * p1 - m * p0) / (m + 1)
    end
    return p1
end

"""
    PupilBeam(; na, amplitude = UniformPupil(), aberrations = (), samples = 128)

The focus of an ideal lens of numerical aperture `na` (the sine of the marginal ray's angle,
1/(2·f-number)) fed by the near field `amplitude` ([`GaussianPupil`](@ref),
[`SuperGaussianPupil`](@ref), [`UniformPupil`](@ref)) with the wavefront error

    W(ρ) = Σₙ aberrations[n] · Pₙ(ρ)      (in waves, n = 1, 2, …)

over the pupil coordinate ρ ∈ [−1, 1]: the Legendre polynomials are the aberration modes of a
slab beam, orthogonal over the pupil like Zernike's over a disk: P₁ tilt (the focus moves by
`2π·aberrations[1]/na` along y), P₂ defocus (it moves by `6π·aberrations[2]/na²` along x), P₃ a
coma-like and P₄ a spherical-like term. The y-cut of a Zernike wavefront can be fitted onto them.
The ray through ρ reaches the focus with the transverse wavenumber κ = −na·ρ, and in the paraxial
(Debye) approximation

    U(ξ, y) = Σ_k a(ρ_k) · exp(i(2π·W(ρ_k) + κ_k·y − κ_k²·ξ/2)) / Σ_k a(ρ_k)

over the `samples` midpoints ρ_k of the pupil, each term a plane wave of the paraxial equation, so
U solves it exactly. The normalization puts the unaberrated beam's focus at 1 on the axis; an
aberration only changes phases, so the beam keeps its energy and its focus loses intensity (the
Strehl ratio |U(0, y_peak)|²). The sum is periodic in y with the period
[`aliasing_period`](@ref)` = π·samples/na`, which must exceed the domain the profile is used on.
"""
struct PupilBeam{T,A<:PupilAmplitude,M} <: TransverseProfile
    na::T
    amplitude::A
    aberrations::NTuple{M,T}
    samples::Int32
end

function PupilBeam(; na, amplitude::PupilAmplitude = UniformPupil(), aberrations = (), samples::Integer = 128)
    T = shape_eltype(na, parameter_values(amplitude)..., aberrations...)
    samples >= 2 || throw(ArgumentError("PupilBeam: $samples samples; at least 2"))
    return PupilBeam{T,typeof(convert_eltype(T, amplitude)),length(aberrations)}(T(na), convert_eltype(T, amplitude),
                                                                                    map(T, Tuple(aberrations)), Int32(samples))
end

"""
    wavefront(b::PupilBeam, ρ) -> W

The wavefront error in waves at the pupil coordinate ρ, Σₙ aberrations[n]·Pₙ(ρ).
"""
@inline function wavefront(b::PupilBeam, ρ)
    p0, p1 = one(ρ), ρ
    w = zero(ρ * b.na)
    for n in 1:length(b.aberrations)
        @inbounds w += b.aberrations[n] * p1
        p0, p1 = p1, ((2n + 1) * ρ * p1 - n * p0) / (n + 1)
    end
    return w
end

@inline function profile(b::PupilBeam, ξ, y)
    T = typeof(b.na * ξ * y)
    dρ = 2 / T(b.samples)
    twoπ = 2 * convert(T, π)
    ur = ui = dr = di = s = zero(T)
    for k in Int32(1):b.samples
        ρ = (k - one(T) / 2) * dρ - 1
        a = pupil_amplitude(b.amplitude, ρ)
        κ = -b.na * ρ
        sn, cs = sincos(twoπ * wavefront(b, ρ) + κ * y - κ * κ * ξ / 2)
        ur += a * cs
        ui += a * sn
        dr -= a * κ * κ * cs
        di -= a * κ * κ * sn
        s += a
    end
    return Complex(ur / s, ui / s), Complex(dr / s, di / s)
end

"""
    aliasing_period(b::TransverseProfile) -> P

The period in y of a profile's quadrature, `π·samples/na` for a [`PupilBeam`](@ref), infinite for
a closed form.
"""
aliasing_period(b::PupilBeam) = b.samples * convert(typeof(b.na), π) / b.na
aliasing_period(b::GaussianBeam) = convert(typeof(b.waist), Inf)

"""
    profile_moments(b) -> (power, ky2)

The transverse integral of |U|² at any ξ (the paraxial beam keeps it) and the mean squared
transverse wavenumber ⟨k_y²⟩ = ∫|∂U/∂y|² dy / ∫|U|² dy. For a [`GaussianBeam`](@ref)
`(waist·√(π/2), 1/waist²)`; for a [`PupilBeam`](@ref), by Parseval over one period of its sum,
`power = 2π·Σa_k²/(δκ·(Σa_k)²)` with δκ = 2·na/samples and `ky2 = Σa_k²κ_k²/Σa_k²`.
"""
profile_moments(b::GaussianBeam) = (b.waist * sqrt(convert(typeof(b.waist), π) / 2), inv(b.waist^2))
function profile_moments(b::PupilBeam)
    T = typeof(b.na)
    dρ = 2 / T(b.samples)
    s = s2 = k2 = zero(T)
    for k in Int32(1):b.samples
        ρ = (k - one(T) / 2) * dρ - 1
        a = pupil_amplitude(b.amplitude, ρ)
        s += a
        s2 += a * a
        k2 += a * a * (b.na * ρ)^2
    end
    return (2 * convert(T, π) * s2 / (b.na * dρ * s^2), k2 / s2)
end
