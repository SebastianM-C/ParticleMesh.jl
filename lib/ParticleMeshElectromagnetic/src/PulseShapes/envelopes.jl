# The temporal envelopes: A(s) such that the field on the axis is Re[A(s)·e^{−is}], s the time
# after the envelope's reference time, the carrier at ω₀ = 1.

"A temporal envelope: [`envelope`](@ref)`(e, s)` is its complex value at the time `s`."
abstract type TemporalEnvelope end

# The element type of a shape made from numbers: their promotion, Float64 for integers and rationals.
shape_eltype(xs...) = (T = promote_type(map(typeof, xs)...); T <: Union{Integer,Rational} ? Float64 : T)

"""
    GaussianEnvelope(; duration, chirp = 0)

The envelope `A(s) = exp(−s²/duration²)·exp(−i·chirp·s²)`: `duration` the 1/e half width of the
field envelope (the intensity FWHM is duration·√(2 ln 2)), `chirp` a temporal chirp, the
instantaneous frequency being 1 + 2·chirp·s. The envelope of ParticleMesh's `GaussianLaser`.
"""
struct GaussianEnvelope{T} <: TemporalEnvelope
    duration::T
    chirp::T
end

function GaussianEnvelope(; duration, chirp = 0)
    T = shape_eltype(duration, chirp)
    return GaussianEnvelope{T}(T(duration), T(chirp))
end

@inline function envelope(e::GaussianEnvelope, s)
    r = s / e.duration
    m = exp(-r * r)
    sn, cs = sincos(e.chirp * s * s)
    return Complex(m * cs, -m * sn)
end

"""
    SpectralEnvelope(; duration, gdd = 0, tod = 0, fod = 0, samples = 512, span = 12)

The envelope of a Gaussian spectrum with a polynomial spectral phase,

    A(s) = (duration/(2√π)) ∫ exp(−Δω²·duration²/4) · exp(iφ(Δω)) · exp(−iΔω·s) dΔω,
    φ(Δω) = gdd·Δω²/2 + tod·Δω³/6 + fod·Δω⁴/24,

Δω the offset from the carrier. Without phase it is the transform-limited `exp(−s²/duration²)`
of [`GaussianEnvelope`](@ref), of peak 1; a phase keeps the spectrum, hence the energy
`∫|A|² ds = duration·√(π/2)`, and lowers the peak. A positive `gdd` (in 1/ω₀², fs² in a lab's
units) gives an up-chirp, the frequency rising in time as 1 + s/gdd far out; a positive `tod`
delays the centroid by tod/(2·duration²) and puts the trailing satellites after the peak.
[`envelope_moments`](@ref) has the centroid and the variance in closed form.

The integral is the midpoint rule on `samples` points over |Δω| ≤ span/duration (the spectrum
there is below exp(−span²/4)), which makes the envelope periodic in s with the period
`π·samples·duration/span` ([`aliasing_period`](@ref)). The envelope is set to zero beyond half a
period from its centroid, so the replicas never reach an antenna, and the constructor refuses
a pulse whose centroid plus twelve of its durations ([`envelope_extent`](@ref), at least 2·rms)
does not fit in half a period: raise `samples` for a strongly stretched pulse. `span` is at
least 8 (the spectrum cut at exp(−16)).
"""
struct SpectralEnvelope{T} <: TemporalEnvelope
    duration::T
    gdd::T
    tod::T
    fod::T
    span::T
    samples::Int32
end

function SpectralEnvelope(; duration, gdd = 0, tod = 0, fod = 0, samples::Integer = 512, span = 12)
    T = shape_eltype(duration, gdd, tod, fod)
    return check_envelope(SpectralEnvelope{T}(T(duration), T(gdd), T(tod), T(fod), T(span), Int32(samples)))
end

function check_envelope(e::SpectralEnvelope)
    e.samples >= 2 || throw(ArgumentError("SpectralEnvelope: $(e.samples) samples; at least 2"))
    e.span >= 8 || throw(ArgumentError("SpectralEnvelope: span $(e.span) cuts the spectrum above exp(−16); at least 8"))
    c, h = envelope_extent(e)
    abs(c) + 12 * h <= aliasing_period(e) / 2 ||
        throw(ArgumentError("SpectralEnvelope: the pulse (centroid $c, duration $h) does not fit in half the period $(aliasing_period(e)) of its $(e.samples) spectral samples; raise `samples`"))
    return e
end

"""
    aliasing_period(e::SpectralEnvelope) -> P

The period in s of the envelope's spectral quadrature, `π·samples·duration/span`.
"""
aliasing_period(e::SpectralEnvelope) = e.samples * convert(typeof(e.duration), π) * e.duration / e.span

@inline function envelope(e::SpectralEnvelope, s)
    τ = e.duration
    Ω = e.span / τ
    dω = 2 * Ω / e.samples
    h = τ / 2
    p2, p3, p4 = e.gdd / 2, e.tod / 6, e.fod / 24
    ar = zero(dω * s)
    ai = zero(ar)
    for k in Int32(1):e.samples
        ω = (k - one(dω) / 2) * dω - Ω
        a = exp(-(ω * h)^2)
        sn, cs = sincos(ω * ω * (p2 + ω * (p3 + ω * p4)) - ω * s)
        ar += a * cs
        ai += a * sn
    end
    scale = τ * dω / (2 * sqrt(convert(typeof(τ), π)))
    c = first(envelope_moments(e))
    inside = abs(s - c) <= aliasing_period(e) / 2
    return Complex(ifelse(inside, scale * ar, zero(ar)), ifelse(inside, scale * ai, zero(ai)))
end

"""
    envelope_moments(e) -> (mean, variance)

The centroid and the variance in time of the intensity |A(s)|², in closed form: for a Gaussian
spectrum of variance σ² = 1/duration² (of |Ã|²) and the phase φ, `mean = ⟨φ′⟩ = tod·σ²/2` and
`variance = duration²/4 + ⟨φ′²⟩ − ⟨φ′⟩² = duration²/4 + gdd²σ² + tod²σ⁴/2 + 5·fod²σ⁶/12 + gdd·fod·σ⁴`,
the moments of the Gaussian over Δω. For a [`GaussianEnvelope`](@ref), `(0, duration²/4)` (its
temporal chirp does not stretch it).
"""
envelope_moments(e::GaussianEnvelope) = (zero(e.duration), e.duration^2 / 4)
function envelope_moments(e::SpectralEnvelope)
    σ2 = inv(e.duration^2)
    m = e.tod * σ2 / 2
    v = e.duration^2 / 4 + e.gdd^2 * σ2 + e.tod^2 * σ2^2 / 2 + 5 * e.fod^2 * σ2^3 / 12 + e.gdd * e.fod * σ2^2
    return (m, v)
end

"""
    envelope_extent(e) -> (center, h)

Where the pulse is in time, for the checks of a code that must start it in vacuum or let it pass:
the field is negligible more than `6h` from `center`. `center` is the centroid and `h` at least
the duration `2·√variance` ([`envelope_moments`](@ref)), which is `duration` for a
transform-limited Gaussian; a [`GaussianEnvelope`](@ref) gives `(0, duration)` exactly. The rms
does not bound a pulse whose group delay φ′(Δω) grows faster than Δω: the slow tails of a TOD or
an FOD reach much further (half a percent of the peak at 6 rms durations for `fod = 3000`,
`duration = 5`). So for a [`SpectralEnvelope`](@ref) `6h` is also at least the largest
|φ′(Δω) − center| over the band |Δω| ≤ 7/duration (where the spectrum falls to exp(−12.25)) plus
three durations; what arrives outside is then the spectrum beyond the band, about erfc(3.5) ≈ 1e-6
of the peak, and the non-stationary edges of the band, of order exp(−12.25)/(6√π) ≈ 5e-7. For a
GDD alone and for a moderate TOD the rms is the larger.
"""
envelope_extent(e::GaussianEnvelope) = (zero(e.duration), e.duration)
function envelope_extent(e::SpectralEnvelope)
    m, v = envelope_moments(e)
    return (m, max(2 * sqrt(v), (group_delay_range(e, m) + 3 * e.duration) / 6))
end

# max |φ′(Δω) − c| over |Δω| ≤ 7/duration: φ′ = gdd·Δω + tod·Δω²/2 + fod·Δω³/6 at the band's edges
# and at its turning points inside, the roots of φ″ = gdd + tod·Δω + fod·Δω²/2.
function group_delay_range(e::SpectralEnvelope, c)
    Ω = 7 / e.duration
    φ1(w) = e.gdd * w + e.tod * w^2 / 2 + e.fod * w^3 / 6 - c
    r = max(abs(φ1(Ω)), abs(φ1(-Ω)))
    if !iszero(e.fod)
        d = e.tod^2 - 2 * e.fod * e.gdd
        if d >= 0
            for w in ((-e.tod + sqrt(d)) / e.fod, (-e.tod - sqrt(d)) / e.fod)
                abs(w) <= Ω && (r = max(r, abs(φ1(w))))
            end
        end
    elseif !iszero(e.tod)
        w = -e.gdd / e.tod
        abs(w) <= Ω && (r = max(r, abs(φ1(w))))
    end
    return r
end

"""
    envelope_energy(e) -> ∫|A(s)|² ds

`duration·√(π/2)` for a [`GaussianEnvelope`](@ref) and a [`SpectralEnvelope`](@ref) alike: a
temporal chirp keeps the modulus, a spectral phase keeps the spectrum (Parseval).
"""
envelope_energy(e::Union{GaussianEnvelope,SpectralEnvelope}) = e.duration * sqrt(convert(typeof(e.duration), π) / 2)
