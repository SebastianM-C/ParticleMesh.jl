# WP-066: shaped laser pulses, T-900 to T-908. The standalone module PulseShapes (a spectral phase,
# a near field and Legendre wavefront modes through a lens) against closed forms, and the
# ShapedLaser that the antenna launches from it, against GaussianLaser, its flux and the Yee
# angular-spectrum propagation of its incident field. Reuses antenna_cases.jl's runs and analysis.

using ForwardDiff: ForwardDiff
using SpecialFunctions: airyai
using ParticleMeshElectromagnetic: PulseShapes, laser_field, check_params
using .PulseShapes: envelope, envelope_moments, envelope_extent, envelope_energy, profile, profile_moments, aliasing_period,
                    pupil_amplitude, parameter_names, parameter_values, with_parameter_values, convert_eltype

const PULSE_SHAPES_FILE = joinpath(pkgdir(ParticleMeshElectromagnetic), "src", "PulseShapes", "PulseShapes.jl")

"The pupil of the tests: a flattened near field with a wavefront of four Legendre modes (waves)."
test_pupil(; na = 3 // 10, aberrations = (0.02, 0.05, 0.08, -0.04), samples = 96) =
    PupilBeam(; na, amplitude = SuperGaussianPupil(; radius = 3 // 4, order = 3), aberrations, samples)

"The shapes of T-900 in the element type `T`."
test_shapes(T) = (GaussianEnvelope(; duration = T(6), chirp = T(1 // 100)), SpectralEnvelope(; duration = T(5), gdd = T(6), tod = T(20)),
                  GaussianBeam(; waist = T(5)), convert_eltype(T, test_pupil()),
                  PupilBeam(; na = T(1 // 5), amplitude = GaussianPupil(; fill = T(1 // 2))), PupilBeam(; na = T(1 // 4)))

# A shape of the standalone copy `P` with the configuration of `x` (its own type, so that its
# functions dispatch on it).
function standalone_shape(P, x)
    n = nameof(typeof(x))
    x isa PulseShapes.GaussianEnvelope && return getfield(P, n)(; duration = x.duration, chirp = x.chirp)
    x isa PulseShapes.SpectralEnvelope && return getfield(P, n)(; duration = x.duration, gdd = x.gdd, tod = x.tod, fod = x.fod,
                                                                samples = x.samples, span = x.span)
    x isa PulseShapes.GaussianBeam && return getfield(P, n)(; waist = x.waist)
    a = x.amplitude
    pa = a isa UniformPupil ? getfield(P, :UniformPupil)() :
         a isa GaussianPupil ? getfield(P, :GaussianPupil)(; fill = a.fill) : getfield(P, :SuperGaussianPupil)(; radius = a.radius, order = a.order)
    return getfield(P, n)(; na = x.na, amplitude = pa, aberrations = x.aberrations, samples = x.samples)
end

# --- T-900 ----------------------------------------------------------------------------------

# The module is self-contained: its source has no `using` or `import`, and included into an empty
# module (which has Base and nothing else) it evaluates every shape to the bits of the package's
# copy, so it can leave for a package of its own. Its shapes are isbits in the A8 matrix (they go
# into the antenna's kernels by value).
@testset "T-900 PulseShapes stands alone: no imports, and included into an empty module it gives the package's bits; its shapes are isbits [A8, WP-066]" begin
    @test !any(l -> occursin(r"^\s*(using|import)\b", l), vcat(readlines.(readdir(dirname(PULSE_SHAPES_FILE); join = true))...))
    m = Module(:Standalone)
    Base.include(m, PULSE_SHAPES_FILE)
    P = getfield(m, :PulseShapes)
    @test P !== PulseShapes
    for T in TYPES, x in test_shapes(T)
        @test isbitstype(typeof(x))
        x2 = standalone_shape(P, x)
        @test parentmodule(typeof(x2)) === P
        if x isa PulseShapes.TemporalEnvelope
            for s in (-7.0, 0.0, 3.3)
                @test same_bits([envelope(x, T(s))], [getfield(P, :envelope)(x2, T(s))])
            end
        else
            for (ξ, y) in ((-30.0, 0.0), (0.0, 4.5), (12.0, -9.0))
                @test same_bits(collect(profile(x, T(ξ), T(y))), collect(getfield(P, :profile)(x2, T(ξ), T(y))))
            end
        end
    end
end

# --- T-901 ----------------------------------------------------------------------------------

# Closed forms of the Gaussian spectrum with a polynomial phase. Without phase the envelope is
# exp(−s²/τ²); with a GDD φ₂ it is (1 − 2iφ₂/τ²)^(−1/2)·exp(−s²/(τ² − 2iφ₂)) (a Gaussian integral);
# with a TOD φ₃ alone, completing the cube (α = τ²/4, β = φ₃/6, c = (3β)^(1/3)),
#   A(s) = (τ/(2√π))·(2π/c)·Ai((α²/(3β) − s)/c)·exp(2α³/(27β²) − α·s/(3β)),
# a real envelope with its satellites after the peak. The quadrature is the midpoint rule on a
# Gaussian spectrum, spectrally accurate, so each holds to roundoff: eps(T) times the largest
# phase of the sum, span·max|s|/τ, which sincos rounds. The duals' partials in the GDD are the
# derivative of the closed form, by ForwardDiff on both.
@testset "T-901 SpectralEnvelope is the closed-form transform-limited, GDD and TOD envelope to roundoff, and its GDD partial the closed form's [A3, A8, WP-066]" begin
    C = emC("T-901")
    for T in (Float32, Float64)
        τ, φ2 = T(6), T(20)
        ss = T.(-40:0.37:40)
        scale = base_eps(T) * (1 + 12 * 40 / 6)
        e0 = SpectralEnvelope(; duration = τ)
        err0 = maximum(s -> abs(envelope(e0, s) - exp(-(s / τ)^2)), ss) / scale
        e2 = SpectralEnvelope(; duration = τ, gdd = φ2)
        cf2(s) = 1 / sqrt(1 - 2im * φ2 / τ^2) * exp(-s^2 / (τ^2 - 2im * φ2))
        err2 = maximum(s -> abs(envelope(e2, s) - cf2(s)), ss) / scale
        τ3, φ3 = 4.0, 30.0
        e3 = SpectralEnvelope(; duration = T(τ3), tod = T(φ3))
        α, β = τ3^2 / 4, φ3 / 6
        c = cbrt(3β)
        air(s) = τ3 / (2 * sqrt(π)) * 2π / c * airyai((α^2 / (3β) - s) / c) * exp(2α^3 / (27β^2) - α * s / (3β))
        err3 = maximum(s -> abs(envelope(e3, T(s)) - air(s)), -30:0.37:60) / (base_eps(T) * (1 + 12 * 60 / τ3))
        for (k, err) in (("TL", err0), ("GDD", err2), ("TOD", err3))
            record!("T-901 $k", T, err)
            @test err <= C
        end
    end
    # the dual path: ∂A/∂φ₂ against the closed form's derivative, Float64 duals
    τ = 6.0
    for s in (-11.0, 0.0, 4.0, 15.0), φ2 in (-15.0, 8.0)
        d = ForwardDiff.derivative(p -> real(envelope(SpectralEnvelope(; duration = τ, gdd = p), s)), φ2) +
            im * ForwardDiff.derivative(p -> imag(envelope(SpectralEnvelope(; duration = τ, gdd = p), s)), φ2)
        dc = ForwardDiff.derivative(p -> real(1 / sqrt(1 - 2im * p / τ^2) * exp(-s^2 / (τ^2 - 2im * p))), φ2) +
             im * ForwardDiff.derivative(p -> imag(1 / sqrt(1 - 2im * p / τ^2) * exp(-s^2 / (τ^2 - 2im * p))), φ2)
        @test abs(d - dc) <= 1e-13
    end
end

# --- T-902 ----------------------------------------------------------------------------------

# Exact identities of a spectral phase: it keeps the spectrum, hence ∫|A|² ds = τ·√(π/2)
# (Parseval), and moves the moments of |A|² by those of φ′ over the Gaussian spectrum
# (envelope_moments: centroid tod/(2τ²), variance τ²/4 + gdd²/τ² + tod²/(2τ⁴) + 5fod²/(12τ⁶) +
# gdd·fod/τ⁴), measured by the trapezoid rule over the envelope's period (spectrally accurate).
# A positive GDD is an up-chirp: the instantaneous frequency 1 − d(arg A)/ds rises through the
# pulse, as 1 + s/gdd far out. Past half its period from the centroid the envelope is zero (no
# replica reaches an antenna), and a pulse that does not fit is refused.
@testset "T-902 a spectral phase keeps the energy and moves the centroid and the variance by the closed forms; positive GDD is an up-chirp; replicas are cut [WP-066]" begin
    C = emC("T-902")
    for e in (SpectralEnvelope(; duration = 6.0, gdd = 20.0), SpectralEnvelope(; duration = 4.0, tod = 30.0),
              SpectralEnvelope(; duration = 5.0, gdd = -12.0, tod = 40.0, fod = 300.0, samples = 2048))
        m, v = envelope_moments(e)
        P = aliasing_period(e)
        ds = 0.05
        ss = (m - P / 2 + ds):ds:(m + P / 2 - ds)
        I = [abs2(envelope(e, s)) for s in ss]
        W = sum(I) * ds
        m1 = sum(ss .* I) * ds / W
        v1 = sum((ss .- m1) .^ 2 .* I) * ds / W
        errs = (abs(W / envelope_energy(e) - 1), abs(m1 - m) / sqrt(v), abs(v1 / v - 1))
        record!("T-902", Float64, maximum(errs) / 1e-12)
        @test maximum(errs) <= C * 1e-12
        @test envelope(e, m + P / 2 + 1) == 0 && envelope(e, m - P / 2 - 1) == 0
    end
    e = SpectralEnvelope(; duration = 6.0, gdd = 60.0)
    ω(s) = 1 - ForwardDiff.derivative(x -> angle(envelope(e, x)), s)
    @test ω(-10.0) < ω(0.0) < ω(10.0)
    @test ω(60.0) - 1 ≈ 4 * 60 * 60 / (6.0^4 + 4 * 60^2) rtol = 1e-10   # arg A = −2φ₂s²/(τ⁴ + 4φ₂²) of the closed form
    @test_throws ArgumentError SpectralEnvelope(; duration = 5.0, gdd = 400.0, samples = 64)
end

# --- T-903 ----------------------------------------------------------------------------------

# A PupilBeam is a sum of plane waves of the paraxial equation, so 2i·∂U/∂ξ + ∂²U/∂y² = 0 holds to
# roundoff for any pupil (ForwardDiff on ξ and y), and its U″ is ∂²U/∂y². Closed forms: a Gaussian
# pupil of fill f well inside the aperture focuses to GaussianBeam(waist = 2/(na·f)) (truncated at
# exp(−1/f²)); a uniform pupil's discrete sum is the Dirichlet kernel sin(na·y)/(N·sin(na·y/N)), so
# U vanishes exactly at y = mπ/na in the focal plane; tilt P₁ moves the focus to y = 2π·c₁/na and
# defocus P₂ to ξ = 6π·c₂/na² (with the piston e^{−iπc₂}), exactly, for any other aberration and
# near field. The power ∫|U|² over a period does not depend on ξ or on the aberrations, and is
# profile_moments' (Parseval).
@testset "T-903 PupilBeam solves the paraxial equation, focuses a Gaussian pupil to GaussianBeam and a uniform one to the sinc, and tilt and defocus move the focus by the closed forms [A2, A7, WP-066]" begin
    C = emC("T-903")
    b = test_pupil()
    U(bb, ξ, y) = profile(bb, ξ, y)[1]
    dξ(bb, ξ, y) = ForwardDiff.derivative(x -> real(U(bb, x, y)), ξ) + im * ForwardDiff.derivative(x -> imag(U(bb, x, y)), ξ)
    d2(bb, ξ, y) = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> real(U(bb, ξ, w)), z), y) +
                   im * ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> imag(U(bb, ξ, w)), z), y)
    worst = 0.0
    for ξ in (-60.0, 0.0, 25.0), y in (0.0, 3.1, -11.0)
        worst = max(worst, abs(2im * dξ(b, ξ, y) + d2(b, ξ, y)) / b.na^2, abs(profile(b, ξ, y)[2] - d2(b, ξ, y)) / b.na^2)
    end
    record!("T-903 pde", Float64, worst / (eps() * b.samples))
    @test worst <= C * eps() * b.samples
    # Gaussian pupil → GaussianBeam, U and U″
    na, fill = 0.4, 0.15
    bg = PupilBeam(; na, amplitude = GaussianPupil(; fill), samples = 256)
    g = GaussianBeam(; waist = 2 / (na * fill))
    w = g.waist
    errg = maximum(abs(profile(bg, ξ, y)[1] - profile(g, ξ, y)[1]) + w^2 * abs(profile(bg, ξ, y)[2] - profile(g, ξ, y)[2])
                   for ξ in (-400.0, -60.0, 0.0, 150.0), y in (0.0, 4.0, -20.0, 55.0))
    record!("T-903 gaussian", Float64, errg / (exp(-1 / fill^2) + 1e3 * eps()))
    @test errg <= C * (exp(-1 / fill^2) + 1e3 * eps())
    # uniform pupil: zeros of the focal plane and the Dirichlet kernel
    bu = PupilBeam(; na = 0.1, samples = 64)
    @test maximum(m -> abs(U(bu, 0.0, m * π / 0.1)), 1:5) <= 1e3 * eps()
    @test maximum(y -> abs(U(bu, 0.0, y) - sin(0.1 * y) / (64 * sin(0.1 * y / 64))), (0.7, 13.0, 40.0)) <= 1e2 * eps()
    # tilt and defocus
    mk(ab) = test_pupil(; na = 0.1, aberrations = ab, samples = 128)
    Δy, Δξ = 2π * 0.3 / 0.1, 6π * 0.2 / 0.01
    errt = maximum(abs(U(mk((0.3, 0.0, 0.1)), 10.0, y) - U(mk((0.0, 0.0, 0.1)), 10.0, y - Δy)) for y in -50:1.3:50)
    errd = maximum(abs(U(mk((0.0, 0.2, 0.1)), 10.0, y) - cis(-π * 0.2) * U(mk((0.0, 0.0, 0.1)), 10.0 - Δξ, y)) for y in -50:1.3:50)
    record!("T-903 tilt", Float64, errt / (1e3 * eps()))
    record!("T-903 defocus", Float64, errd / (1e3 * eps()))
    @test errt <= C * 1e3 * eps() && errd <= C * 1e3 * eps()
    # power over one period: independent of ξ and of the aberrations, and profile_moments'
    P = aliasing_period(b)
    ny = 4 * b.samples
    yy = range(-P / 2; step = P / ny, length = ny)
    pw(bb, ξ) = sum(y -> abs2(U(bb, ξ, y)), yy) * P / ny
    p0 = profile_moments(b)[1]
    errp = maximum(abs(pw(bb, ξ) / p0 - 1) for bb in (b, test_pupil(; aberrations = ())), ξ in (0.0, 80.0))
    record!("T-903 power", Float64, errp / (1e3 * eps()))
    @test errp <= C * 1e3 * eps()
end

# --- T-904 ----------------------------------------------------------------------------------

# Maréchal: for a uniform pupil the field at the paraxial focus is the pupil mean of e^{2πiW}, so
# the Strehl ratio |U(0, 0)|² = 1 − (2πσ)² + O((2πσ)⁴), σ² = Σₙ cₙ²/(2n + 1) the variance of W over
# the pupil (⟨Pₙ⟩ = 0, ⟨Pₙ²⟩ = 1/(2n + 1)). Held at two small rms values, the residual against
# (2πσ)⁴; the energy (profile_moments) does not move.
@testset "T-904 Maréchal: a small wavefront error of rms σ lowers the focus of a uniform pupil to 1 − (2πσ)² [A7, WP-066]" begin
    C = emC("T-904")
    for c in ((0.0, 0.0, 0.01, 0.015, -0.01), (0.0, 0.0, 0.02, 0.03, -0.02))
        b = PupilBeam(; na = 0.2, aberrations = c, samples = 512)
        σ2 = sum(c[n]^2 / (2n + 1) for n in eachindex(c))
        x = (2π)^2 * σ2
        S = abs2(profile(b, 0.0, 0.0)[1])
        record!("T-904", Float64, abs(S - (1 - x)) / x^2)
        @test abs(S - (1 - x)) <= C * x^2
        @test profile_moments(b)[1] == profile_moments(PupilBeam(; na = 0.2, samples = 512))[1]
    end
end

# --- T-905 ----------------------------------------------------------------------------------

# A ShapedLaser of a GaussianBeam and a GaussianEnvelope is the GaussianLaser (AxisTime) of the
# same numbers: e, bp and bs to roundoff, Float64 and duals. For a pupil beam with a spectral
# phase, laser_energy is the flux ∫∫ (E × B)_x dy dt = ∫∫ (e·bp·cos²θ + e·bs·sin²θ) of a host
# quadrature (trapezoid over one period in y and over the pulse in t, spectrally accurate), up to
# the envelope's overlap with 2ω₀, exp(−τ²/2) = exp(−18) at τ = 6, the scale of the test.
@testset "T-905 a ShapedLaser of a GaussianBeam and a GaussianEnvelope is the GaussianLaser's incident field, and laser_energy is the flux of a pupil beam with a spectral phase [A2, A3, WP-066]" begin
    C = emC("T-905")
    for T in (Float64, D3)
        sd(x, i) = T === D3 ? mkdual(x, ntuple(k -> k == i ? 1 : 0, 3)...) : T(x)
        kw = (; a0 = sd(0.7, 1), delay = T(30), focus = sd(3, 2), position = 0.0, center = 0.0, polarization = T(0.3), phase = T(0.2))
        g = GaussianLaser(; kw..., waist = T(5), duration = T(6), chirp = sd(0.004, 3))
        s = ShapedLaser(; kw..., profile = GaussianBeam(; waist = T(5)), envelope = GaussianEnvelope(; duration = T(6), chirp = sd(0.004, 3)))
        worst = 0.0
        for ξ in (-20.0, 0.0, 7.0), y in (0.0, 2.0, -6.0), t in (-9.0, 0.0, 4.3)
            a, b = laser_field(g, T(ξ), T(y), T(t)), laser_field(s, T(ξ), T(y), T(t))
            for (x, z) in zip(a, b)
                worst = max(worst, abs(value_of(x) - value_of(z)))
                T === D3 && (worst = max(worst, maximum(abs.(Tuple(ForwardDiff.partials(x)) .- Tuple(ForwardDiff.partials(z))))))
            end
        end
        record!("T-905 gaussian", T, worst / (64 * base_eps(T)))
        @test worst <= C * 64 * base_eps(T)
        @test value_of(laser_energy(s)) ≈ value_of(laser_energy(g)) rtol = 8 * base_eps(T)
    end
    for θ in (0.0, π / 2, 0.6)
        l = ShapedLaser(; a0 = 0.5, delay = 0.0, focus = 40.0, position = 0.0, center = 0.0, polarization = θ, phase = 0.3,
                        profile = test_pupil(), envelope = SpectralEnvelope(; duration = 6.0, gdd = 15.0, tod = 40.0, samples = 256))
        P = aliasing_period(l.profile)
        ny, dt = 4 * l.profile.samples, 0.05
        m, v = envelope_moments(l.envelope)
        sθ, cθ = sincos(θ)
        Φ = 0.0
        for t in (m - 14 * sqrt(v)):dt:(m + 14 * sqrt(v)), j in 0:(ny - 1)
            y = -P / 2 + j * P / ny
            e, bp, bs = laser_field(l, -40.0, y, t)
            Φ += e * (bp * cθ^2 + bs * sθ^2)
        end
        Φ *= dt * P / ny
        rel = abs(Φ / laser_energy(l) - 1)
        record!("T-905 flux", Float64, rel / exp(-18))
        @test rel <= C * exp(-18)
    end
end

# --- T-906 ----------------------------------------------------------------------------------

"""
    shaped_problem(T; length = (200, 200), na = 3//10, position = 10, focus = 40, tend = 150, polarization = 0.6,
                   gdd = 6, tod = 20, aberrations = (0.02, 0.05, 0.08, -0.04)) -> EMProblem{2}

antenna_problem's box (16 × 8 cells per wavelength, Courant 0.95) with a ShapedLaser: the test
pupil focused at `focus` and a spectral phase on a transform-limited `duration`, its centroid at
the plane at `delay`, by default six durations of the stretched pulse (below roundoff at t = 0).
The box is 200 high because the flattened, aberrated pupil's side lobes are still a few percent of
the peak 50 from the axis: in a box 100 high the field the antenna drives does not repeat across
the y boundary, and that jump, not the antenna's order, set the error of T-906 (about 15 times
its scale, against 1.3 at 200).
"""
function shaped_problem(::Type{T}; length = (200, 200), na = 3 // 10, position = 10, focus = 40, tend = 150, polarization = 0.6,
                        gdd = 6, tod = 20, aberrations = (0.02, 0.05, 0.08, -0.04), a0 = 1, delay = nothing, duration = 8) where {T}
    h = (2π / 16, 2π / 8)
    n = (round(Int, length[1] / h[1]), round(Int, length[2] / h[2]))
    L = (T(n[1] * h[1]), T(n[2] * h[2]))
    dt = T(95 // 100) / sqrt(1 / (L[1] / n[1])^2 + 1 / (L[2] / n[2])^2)
    e = SpectralEnvelope(; duration = T(duration), gdd = T(gdd), tod = T(tod), samples = 256)
    c, w = envelope_extent(SpectralEnvelope(; duration = Float64(duration), gdd = Float64(value_of(gdd)), tod = Float64(value_of(tod)), samples = 256))
    laser = ShapedLaser(; a0 = T(a0), delay = T(delay === nothing ? 6 * w - c : delay), focus = T(focus), position = Float64(position),
                        center = Float64(value_of(L[2])) / 2, polarization = T(polarization), envelope = e,
                        profile = PupilBeam(; na = T(na), amplitude = SuperGaussianPupil(; radius = T(3 // 4), order = 3),
                                            aberrations = map(T, aberrations), samples = 96))
    return EMProblem("shaped", L, n, dt, round(Int, tend / value_of(dt)), (), 1, (1, 0); lasers = (laser,))
end

"The scale of T-906: the antenna's B error (3/16)·√⟨k_y⁸⟩ over the pupil's angular spectrum (T-402's for a Gaussian)."
function pupil_angular_scale(b)
    ρ = ((1:b.samples) .- 0.5) .* (2 / b.samples) .- 1
    a2 = [Float64(value_of(pupil_amplitude(b.amplitude, x)))^2 for x in ρ]
    return 3 / 16 * sqrt(sum(a2 .* (Float64(value_of(b.na)) .* ρ) .^ 8) / sum(a2))
end

# The launched beam of a shaped laser (an aberrated flattened pupil, a pulse stretched by GDD and
# TOD) is the free Yee beam of its incident field: on every probe column its ω₀ component is the
# Yee angular-spectrum propagation of the incident ω₀ component (T-402's analysis), to the
# antenna's order (k_y/k)⁴ of the pupil's spectrum, in the A8 matrix; and the antenna launches one
# way, the energy behind it held to T-401's constant for the transform-limited beam (a spectral
# phase keeps the band the antenna is exact on). Shared with the GPU rerun (T-909).
function body_T906(be)
    C, Cl = emC("T-906"), emC("T-906", "C_leak")
    for T in TYPES
        prob = shaped_problem(T)
        r = antenna_run(prob; probes = (10, 25, 40, 60, 80), be)
        sc = pupil_angular_scale(prob.lasers[1].profile)
        for comp in (:Ey, :Ez)
            err = maximum(x -> x.err, angular_error(r; component = comp))
            record!("T-906 $comp", T, err / sc, be)
            @test err <= C * sc
        end
        # one way: the pulse lies within x ≤ 10 + 150 − 6·w at the end, the wrapped leak beyond
        _, w = envelope_extent(prob.lasers[1].envelope)
        fwd, back = forward_window(r, value_of(150 - 2 * w))
        f = map(Array, r.st.fields)
        frac = Float64(value_of(plain_energy(f, r.st.grid; cols = back) / conserved_energy(f, r.st.grid, prob.dt; cols = fwd)))
        record!("T-906 leak", T, frac, be)
        @test frac <= Cl
    end
end

@testset "T-906 the launched shaped beam is the Yee angular-spectrum propagation of its incident field to the antenna's order (k_y/k)⁴, and launches one way, in the A8 matrix [A2, A7, A8, WP-066]" begin
    body_T906(CPU())
end

# --- T-907 ----------------------------------------------------------------------------------

# A3: the numbers of the shapes enter through the antenna's source. The ω₀ intensity on the axis of
# a probe past the plane, by ForwardDiff through the whole run against an aberration coefficient
# (coma-like P₃) and against the GDD, agrees with the finite-difference floor (fd_scan, A7).
function shaped_observable(x, mode, seed; which)
    T = typeof(x)
    kw = which === :coma ? (; aberrations = (0.0, 0.0, x), gdd = 3) : (; aberrations = (0.0, 0.0, 0.05), gdd = x)
    # the delay of the reference pulse (GDD 3), the same for every x
    prob = shaped_problem(T; length = (60, 40), na = 1 // 2, position = 5, focus = 15, tend = 55, polarization = 0, tod = 0, delay = 25, duration = 3, kw...)
    r = antenna_run(prob; mode, probes = (20,))
    A = r.A[1].Ey
    return abs2(A[length(A) ÷ 2])
end

@testset "T-907 the shapes' numbers enter through the antenna: gradients in an aberration and in the GDD agree with the finite-difference floor [A3, A7, WP-066]" begin
    C = emC("T-907")
    hs = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    for (which, x0, sc) in ((:coma, 0.05, 0.1), (:gdd, 3.0, 10.0))
        f = (x, mode, seed) -> shaped_observable(x, mode, seed; which)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 1), x0)
        s = fd_scan(f, x0, hs .* sc; seed = 1, ad)
        record!("T-907 $which", Float64, s.rel[s.istar] / s.floor_rel)
        @test abs(ad) > 0
        @test s.rel[s.istar] <= C * max(s.floor_rel, 1e-10)
    end
end

"When the pulse of `l` is over at its plane, six stretched durations after its centroid."
pulse_end_of(l) = ParticleMeshElectromagnetic.pulse_end(l)

# --- T-908 ----------------------------------------------------------------------------------

# The shaped laser through the parameter layer: its tunables are its five numbers and its shapes'
# in their order, and with_tunables gives the same laser back in the problem's and the duals'
# element types; check_params refuses a profile that repeats within twice the box height; a moving
# window refuses to pass the plane before six stretched durations after the pulse's centroid; and
# the step with a shaped antenna has no runtime dispatch (JET, released Julia only).
@testset "T-908 a ShapedLaser's tunables round-trip, a repeating profile and an early window are refused, and its step has no runtime dispatch [A3, A8, WP-066]" begin
    prob = shaped_problem(Float64; tend = 2)
    l = prob.lasers[1]
    names = tunable_names(prob)
    @test names == [Symbol(:laser1_, n) for n in (:a0, :delay, :focus, :polarization, :phase, :na, :radius, :aberration1, :aberration2,
                                                  :aberration3, :aberration4, :duration, :gdd, :tod, :fod)]
    v = tunables(prob)
    @test v == [l.a0, l.delay, l.focus, l.polarization, l.phase, l.profile.na, l.profile.amplitude.radius, l.profile.aberrations...,
                l.envelope.duration, l.envelope.gdd, l.envelope.tod, l.envelope.fod]
    p2 = with_tunables(params(prob), v)
    @test typeof(p2.lasers[1]) == typeof(params(prob).lasers[1]) && tunables(p2) == v
    vd = ForwardDiff.Dual{Nothing}.(v, 1.0)
    pd = with_tunables(params(prob), vd)
    @test eltype(pd.lasers[1].profile.aberrations) == eltype(vd) && pd.lasers[1].envelope.samples == l.envelope.samples
    # a profile of period π·8/0.3 ≈ 84 < 2·200
    bad = ShapedLaser(; a0 = 1.0, delay = l.delay, focus = l.focus, position = l.position, center = l.center, envelope = l.envelope,
                      profile = test_pupil(; samples = 8))
    st = EMState(prob, EMNumerics())
    @test_throws ArgumentError check_params(st, EMParams(params(prob); lasers = (bad,)))
    check_params(st, params(prob))
    # a window that starts while the stretched pulse still crosses its plane
    p = shaped_problem(Float64; tend = 2)
    lw = p.lasers[1]
    # the plane stops acting s_off = ia − g − 1 shifts after the start (window.jl), a time below its
    # position: a start 5 before pulse_end − position is refused, one 5 after pulse_end is not
    early = pulse_end_of(lw) - lw.position - 5
    mkwin(start) = EMProblem("shaped-window", p.length, p.cells, p.dt, 2, (), 1, (1, 0); lasers = (lw,), boundaries = (:window, :periodic),
                             window = MovingWindow(; start), profile = DensityRamp(; vacuum = 50.0, ramp = 8.0))
    @test_throws ArgumentError EMState(mkwin(early), EMNumerics())
    EMState(mkwin(pulse_end_of(lw) + 5), EMNumerics())
    if isempty(VERSION.prerelease)
        @eval using JET
        targets = (ParticleMeshCore, ParticleMeshElectromagnetic, PulseShapes)
        for T in (Float32, D3)
            pr = shaped_problem(T; tend = 2)
            s = EMState(pr, EMNumerics())
            initialize!(s, pr)
            pp = params(pr)
            step!(s, pp, 1)
            rep = JET.get_reports(JET.report_opt(step!, (typeof(s), typeof(pp), Int); target_modules = targets))
            isempty(rep) || foreach(x -> show(stdout, MIME"text/plain"(), x), rep)
            @test isempty(rep)
        end
    end
end

