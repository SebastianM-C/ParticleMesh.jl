# The round beam of three dimensions (WP-085): U(ξ, y, z), a solution of the paraxial equation
# 2i·∂U/∂ξ + ∂²U/∂y² + ∂²U/∂z² = 0, the focus of a round pupil with a wavefront of Zernike modes.
# Its quadrature over the pupil is a sum of plane waves of the paraxial equation, one per sample;
# `pupil_sample` gives a sample and `pupil_terms` its plane wave at a point, the two halves that a
# kernel tabulating a plane splits between a pass over the pupil and a pass over the plane.

"""
    zernike_index(j) -> (n, m)

The radial order `n` and the azimuthal frequency `m` of the Zernike polynomial of OSA/ANSI index
`j` (j = (n(n + 2) + m)/2, n ≥ 0, |m| ≤ n, n − m even): 0 piston (0, 0), 1 tilt (1, −1), 2 tilt
(1, 1), 3 oblique astigmatism (2, −2), 4 defocus (2, 0), 5 astigmatism (2, 2), 6 to 9 trefoil and
coma (3, −3), (3, −1), (3, 1), (3, 3), 12 primary spherical (4, 0).
"""
@inline function zernike_index(j::Integer)
    n = zero(j)
    while (n + 1) * (n + 2) ÷ 2 <= j
        n += one(j)
    end
    return n, 2j - n * (n + 2)
end

"""
    zernike_radial(n, m, ρ)

The Zernike radial polynomial R_n^m(ρ), m ≥ 0, n − m even:
Σ_s (−1)^s (n − s)!/(s!·((n + m)/2 − s)!·((n − m)/2 − s)!)·ρ^(n − 2s), by Horner's rule in ρ².
"""
@inline function zernike_radial(n::Integer, m::Integer, ρ)
    a, b = (n + m) ÷ 2, (n - m) ÷ 2
    c = one(ρ)                                   # the leading coefficient, the binomial (n choose b)
    for i in 1:b
        c = c * (n - b + i) / i
    end
    ρ2 = ρ * ρ
    r = zero(ρ)
    for s in 0:b
        r = r * ρ2 + c
        s < b && (c = -c * (a - s) * (b - s) / ((s + 1) * (n - s)))
    end
    for _ in 1:m
        r *= ρ
    end
    return r
end

"""
    zernike(j, ρ, θ)

The Zernike polynomial of OSA/ANSI index `j` at the pupil point of radius ρ ∈ [0, 1] and angle θ
(from the pupil's first axis towards its second), normalized to unit rms over the unit disk
(ANSI Z80.28): `N·R_n^|m|(ρ)·cos(mθ)` for m ≥ 0 and `N·R_n^|m|(ρ)·sin(|m|θ)` for m < 0, with
N = √(2(n + 1)) for m ≠ 0 and √(n + 1) for m = 0 ([`zernike_index`](@ref), [`zernike_radial`](@ref)).
"""
@inline function zernike(j::Integer, ρ, θ)
    n, m = zernike_index(j)
    am = abs(m)
    r = zernike_radial(n, am, ρ)
    norm = sqrt(convert(typeof(r), ifelse(m == 0, n + 1, 2 * (n + 1))))
    ang = ifelse(m < 0, sin(am * θ), cos(am * θ))
    return norm * r * ang
end

"""
    ZernikeBeam(; na, amplitude = UniformPupil(), aberrations = (), radial = 32, angular = 64)

The focus of an ideal lens of numerical aperture `na` fed through a round pupil by the near field
`amplitude` (a [`PupilAmplitude`](@ref) of the pupil radius ρ ∈ [0, 1]: [`GaussianPupil`](@ref),
[`SuperGaussianPupil`](@ref), [`UniformPupil`](@ref)) with the wavefront error

    W(ρ, θ) = Σⱼ aberrations[j] · Zⱼ(ρ, θ)      (in waves, OSA/ANSI j = 1, 2, …)

of Zernike polynomials ([`zernike`](@ref)), each of unit rms over the pupil, so that the rms of the
wavefront is √(Σⱼ aberrations[j]²) once tilt and piston are left out. The pupil's first axis is y
and its second z, so θ runs from y towards z: Z₂ = 2ρ·cos θ tilts the beam along y, Z₅ = √6ρ²·cos 2θ
focuses its y and z at two planes (astigmatism), Z₄ = √3(2ρ² − 1) moves the focus along x by
`8√3π·aberrations[4]/na²` and Z₁₂ is the primary spherical aberration. The ray through the pupil
point ρ⃗ = ρ(cos θ, sin θ) reaches the focus with the transverse wavenumber κ⃗ = −na·ρ⃗, and in the
paraxial (Debye) approximation

    U(ξ, y, z) = Σ_q w_q·a(ρ_q)·exp(i(2π·W(ρ_q, θ_q) + κ⃗_q·(y, z) − |κ⃗_q|²·ξ/2)) / Σ_q w_q·a(ρ_q)

over the samples of the disk: Gauss–Legendre in ρ on `radial` nodes (exact for the polynomials of
degree up to 2·radial − 1, so the hard edge of a uniform pupil costs nothing) times the trapezoid
rule in θ on `angular` points θ_k = 2π(k − ½)/angular (exact for the harmonics below `angular`),
with the weight w = ρ·w_ρ. Each term is a plane wave of the paraxial equation, so U solves it
exactly, and the normalization puts the unaberrated beam's focus at 1 on the axis. An aberration
changes phases only, so the beam keeps its energy ([`profile_moments`](@ref)) and its focus loses
intensity: Maréchal's Strehl ratio exp(−(2πσ)²) for a small rms σ.

The quadrature resolves the field out to the radius [`quadrature_radius`](@ref) from the axis:
`angular` must exceed the harmonics of the pupil's plane waves at that radius, about na·r, and
`radial` their oscillations along ρ. `angular` is a multiple of four, so the samples have the
symmetries of the square (y ↔ z and y → −y) and an unaberrated beam is round to roundoff.
`radial` and `angular` are configuration, not parameters.
"""
struct ZernikeBeam{T,A<:PupilAmplitude,M} <: TransverseProfile
    na::T
    amplitude::A
    aberrations::NTuple{M,T}
    radial::Int32
    angular::Int32
end

function ZernikeBeam(; na, amplitude::PupilAmplitude = UniformPupil(), aberrations = (), radial::Integer = 32, angular::Integer = 64)
    T = shape_eltype(na, parameter_values(amplitude)..., aberrations...)
    radial >= 1 || throw(ArgumentError("ZernikeBeam: $radial radial nodes; at least 1"))
    angular >= 4 && angular % 4 == 0 || throw(ArgumentError("ZernikeBeam: $angular angular points; a positive multiple of 4"))
    return ZernikeBeam{T,typeof(convert_eltype(T, amplitude)),length(aberrations)}(T(na), convert_eltype(T, amplitude),
                                                                                    map(T, Tuple(aberrations)), Int32(radial), Int32(angular))
end

"""
    transverse_dimensions(b::TransverseProfile) -> 1 or 2

The number of transverse axes of a profile: 1 for the beams of the plane ([`GaussianBeam`](@ref),
[`PupilBeam`](@ref), `profile(b, ξ, y)`), 2 for the round beam of three dimensions
([`ZernikeBeam`](@ref), `profile(b, ξ, y, z)`).
"""
transverse_dimensions(::TransverseProfile) = 1
transverse_dimensions(::ZernikeBeam) = 2

"""
    wavefront(b::ZernikeBeam, ρ, θ) -> W

The wavefront error in waves at the pupil point (ρ, θ), Σⱼ aberrations[j]·Zⱼ(ρ, θ).
"""
@inline function wavefront(b::ZernikeBeam, ρ, θ)
    w = zero(ρ * b.na)
    for j in 1:length(b.aberrations)
        @inbounds w += b.aberrations[j] * zernike(j, ρ, θ)
    end
    return w
end

"""
    radial_node(N, i, ::Type{T}) -> (ρ, w)

The `i`-th Gauss–Legendre node of `N ≥ 1` on [0, 1] and its weight, by Newton's method on the
Legendre polynomial P_N from the asymptotic guess, eight iterations in `T` (no branch on the data).
"""
@inline function radial_node(N::Integer, i::Integer, ::Type{T}) where {T}
    x = cos(convert(T, π) * (i - one(T) / 4) / (N + one(T) / 2))
    dp = one(T)
    for _ in 1:8
        p0, p1 = one(T), x
        for m in 1:(N - 1)
            p0, p1 = p1, ((2m + 1) * x * p1 - m * p0) / (m + 1)
        end
        dp = N * (x * p1 - p0) / (x * x - 1)
        x -= p1 / dp
    end
    p0, p1 = one(T), x
    for m in 1:(N - 1)
        p0, p1 = p1, ((2m + 1) * x * p1 - m * p0) / (m + 1)
    end
    dp = N * (x * p1 - p0) / (x * x - 1)
    return (1 + x) / 2, 1 / ((1 - x * x) * dp * dp)
end

"""
    pupil_sample(b::ZernikeBeam, ρ, wρ, k) -> (pr, pi, κy, κz, wa)

The sample of the pupil on the radial node ρ of weight wρ ([`radial_node`](@ref)) at the angle
θ_k = 2π(k − ½)/angular: its weighted amplitude wa = ρ·wρ·a(ρ), its phasor
(pr, pi) = wa·exp(2πi·W(ρ, θ_k)) and its transverse wavenumber κ⃗ = −na·ρ(cos θ_k, sin θ_k).
"""
@inline function pupil_sample(b::ZernikeBeam, ρ, wρ, k::Integer)
    T = typeof(b.na * ρ)
    twoπ = 2 * convert(T, π)
    θ = twoπ * (k - one(T) / 2) / b.angular
    sθ, cθ = sin(θ), cos(θ)
    wa = ρ * wρ * pupil_amplitude(b.amplitude, ρ)
    ϕ = twoπ * wavefront(b, ρ, θ)
    return wa * cos(ϕ), wa * sin(ϕ), -b.na * ρ * cθ, -b.na * ρ * sθ, wa
end

"""
    pupil_terms(pr, pi, κy, κz, ξ, y, z) -> (u, uyy, uzz, uyz)

The plane wave of one pupil sample ([`pupil_sample`](@ref)) at (ξ, y, z): its complex value
u = (pr + i·pi)·exp(i(κy·y + κz·z − (κy² + κz²)·ξ/2)) and its second derivatives across the beam,
−κy²·u, −κz²·u and −κy·κz·u.
"""
@inline function pupil_terms(pr, pi, κy, κz, ξ, y, z)
    ϕ = κy * y + κz * z - (κy * κy + κz * κz) * ξ / 2
    sn, cs = sin(ϕ), cos(ϕ)
    u = Complex(pr * cs - pi * sn, pr * sn + pi * cs)
    return u, -κy * κy * u, -κz * κz * u, -κy * κz * u
end

"""
    profile(b::ZernikeBeam, ξ, y, z) -> (U, Uyy, Uzz, Uyz)

The complex profile of the round beam at the distance `ξ` from the focal plane and (`y`, `z`) from
the axis, and its second derivatives ∂²U/∂y², ∂²U/∂z², ∂²U/∂y∂z, by the pupil sum of
[`ZernikeBeam`](@ref). An isbits device function of the element type.
"""
@inline function profile(b::ZernikeBeam, ξ, y, z)
    T = typeof(b.na * ξ * y * z)
    u = uyy = uzz = uyz = zero(Complex{T})
    s = zero(T)
    for i in Int32(1):b.radial
        ρ, wρ = radial_node(b.radial, i, T)
        for k in Int32(1):b.angular
            pr, pi, κy, κz, wa = pupil_sample(b, ρ, wρ, k)
            t = pupil_terms(pr, pi, κy, κz, ξ, y, z)
            u += t[1]
            uyy += t[2]
            uzz += t[3]
            uyz += t[4]
            s += wa
        end
    end
    return u / s, uyy / s, uzz / s, uyz / s
end

"""
    quadrature_radius(b::ZernikeBeam) -> r

The distance from the axis up to which the pupil sum resolves the beam, the smaller of two
estimates from the error bounds of the rules (not measured beyond T-1161): `angular/(4·na)`, where
the trapezoid rule's first alias, a Bessel function of order `angular` at na·r, is below
(e/8)^angular/√(2π·angular) of the beam (10⁻¹⁵ at 32 points), and `radial/na`, where the
oscillation of the plane waves along ρ, na·r radians, stays below the degree 2·radial − 1 that
Gauss–Legendre integrates exactly. A box needs it at least at its half diagonal.
"""
quadrature_radius(b::ZernikeBeam) = min(b.angular / (4 * b.na), b.radial / b.na)

"""
    profile_moments(b::ZernikeBeam) -> (power, ky2)

The transverse integral of |U|² at any ξ (the paraxial beam keeps it) and ⟨k_y²⟩, the mean of
κ_y² over |Ũ|²: by Parseval, `power = (2π/na)²·Σ w_q a_q²/(δθ·(Σ w_q a_q)²)` with δθ = 2π/angular,
and `ky2 = Σ w_q a_q² κ_y,q²/Σ w_q a_q²`. Neither depends on the wavefront, and ⟨k_z²⟩ = ⟨k_y²⟩,
⟨k_y k_z⟩ = 0 for every pupil amplitude (a function of ρ alone).
"""
function profile_moments(b::ZernikeBeam)
    T = typeof(b.na)
    s = s2 = k2 = zero(T)
    for i in Int32(1):b.radial
        ρ, wρ = radial_node(b.radial, i, T)
        w = ρ * wρ
        a = pupil_amplitude(b.amplitude, ρ)
        s += b.angular * w * a
        s2 += b.angular * w * a * a
        k2 += b.angular * w * a * a * (b.na * ρ)^2 / 2      # the mean of cos²θ over the angles is ½
    end
    δθ = 2 * convert(T, π) / b.angular
    return ((2 * convert(T, π) / b.na)^2 * s2 / (δθ * s^2), k2 / s2)
end
