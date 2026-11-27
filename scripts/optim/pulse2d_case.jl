# The WP-087 case: WP-065's trapped-electron case (trapped2d_case.jl) with the plasma and the laser
# energy fixed and the pulse's shape free. The pulse is a ShapedLaser (WP-066): a fixed
# super-Gaussian near field focused by a lens of fixed numerical aperture, with a wavefront of
# Legendre aberration modes, and a Gaussian spectrum with a spectral phase. Its five parameters are
# the focus, the coma-like and the spherical-like modes P₃ and P₄ (in waves), the GDD and the TOD.
# Everything the optimizer moves is a phase (the focus moves the lens), so the pulse carries the
# same energy at every point and a₀ is a constant of the case.
isdefined(@__MODULE__, :TCASE) || include(joinpath(@__DIR__, "trapped2d_case.jl"))

"""
Fixed parameters of the WP-087 case beyond [`TCASE`](@ref) and [`BCASE`](@ref) (A1 units).

- The plasma is WP-065's at its base density `density` (n_c), with the dopant at
  `TCASE.dopant_fraction` of it; the delay is the base point's `delay`; the energy is the budget
  of a₀ = `a0_energy` at WP-065's base pulse (waist 10, duration 6), [`budget`](@ref).
- The near field is `SuperGaussianPupil(; radius, order)` across a lens of numerical aperture
  `na`, summed over `samples` rays ([`PulseShapes.PupilBeam`](@ref)). Its amplitude at the
  aperture's edge is exp(−(1/radius)^(2·order)) ≈ 5·10⁻⁵, and `na` puts the unaberrated focus's
  intensity FWHM at the base Gaussian's, 10·√(2 ln 2) ≈ 11.8 c/ω₀ (T-1180). At the same
  energy its a₀ is 1.898, the Gaussian's 1.9 to 0.1 %. Of the unaberrated pulse's power, 0.4 %
  (focus at 64) to 1.4 % (at 110) falls outside the rows inside the y absorbers at the antenna
  plane, and up to 3.5 % at the box's corners ([`launched_fraction`](@ref)): the flat-top's edge
  diffracts, and the aberrations steepen the rays.
- The spectrum is Gaussian with the transform-limited `duration` of the base pulse, through
  [`PulseShapes.SpectralEnvelope`](@ref) with `spectral_samples` points.
- `modes` are the Legendre modes the optimizer moves. P₁ (tilt) and P₂ (defocus) stay at zero:
  tilt steers the beam off the axis and defocus moves the focus, which `focus` already does.
  P₃ is odd in y, P₄ even.
- `x0` is the unaberrated, transform-limited pulse focused at the base point's 80; `names` and
  `scale` name each parameter and give the scale of z = x ./ scale in problem.jl.
"""
const PCASE = (a0_energy = 1.9, density = 0.072, delay = 54.0, duration = 6.0, na = 0.325556, radius = 0.75, order = 4, samples = 128,
               spectral_samples = 512, modes = (3, 4),
               x0 = (80.0, 0.0, 0.0, 0.0, 0.0), names = ("focus", "aberration3", "aberration4", "gdd", "tod"),
               scale = (80.0, 0.25, 0.25, 10.0, 25.0))

"""
The optimizer's box (WP-087): the focus over WP-068's 64 to 110, each mode within ±0.3 waves (an
rms wavefront error of 0.3/√(2n + 1), 0.11 waves for P₃ and 0.10 for P₄), the GDD within ±12 and the TOD within ±30 (1/ω₀² and 1/ω₀³). At the corners
of the spectral phase the pulse still starts after t = 0 and ends before the window passes the
antenna (`check_pulse_time`, T-1180). The modes' bound keeps the power the antenna's interior rows
launch above 96 % at every corner (T-1181): at ±0.4 the worst corner, focused at 110, loses 6 %.
"""
const PULSE_BOX = (lo = [64.0, -0.3, -0.3, -12.0, -30.0], hi = [110.0, 0.3, 0.3, 12.0, 30.0], scale = collect(PCASE.scale))

"""
    pulse_shape(x) -> (; profile, envelope)

The pulse's transverse profile and temporal envelope at the five parameters `x` (focus, P₃, P₄,
GDD, TOD), in their promoted element type: a [`PulseShapes.PupilBeam`](@ref) with the aberration
of mode n at `x[1 + k]` for the k-th of `PCASE.modes` and zero for the others, and a
[`PulseShapes.SpectralEnvelope`](@ref). The focus is the laser's, not the shape's.
"""
function pulse_shape(x)
    length(x) == length(PCASE.x0) || throw(ArgumentError("pulse_shape takes the $(length(PCASE.x0)) parameters, got $(length(x))"))
    T = promote_type(map(typeof, Tuple(x))...)
    n = maximum(PCASE.modes)
    aberrations = ntuple(m -> (k = findfirst(==(m), PCASE.modes); k === nothing ? zero(T) : T(x[1 + k])), n)
    profile = PupilBeam(; na = T(pulse_na()), amplitude = SuperGaussianPupil(; radius = T(PCASE.radius), order = PCASE.order),
                        aberrations, samples = PCASE.samples)
    k = length(PCASE.modes)
    envelope = SpectralEnvelope(; duration = T(PCASE.duration), gdd = T(x[2 + k]), tod = T(x[3 + k]), samples = PCASE.spectral_samples)
    return (; profile, envelope)
end

"""
    focal_fwhm(na; radius, order, samples) -> FWHM

The intensity FWHM in y of the unaberrated focus of the case's near field through a lens of
numerical aperture `na`, by bisection on |U(0, y)|² = 1/2 (the focus is 1 on the axis and falls
monotonically to its first zero).
"""
function focal_fwhm(na; radius = PCASE.radius, order = PCASE.order, samples = PCASE.samples)
    b = PupilBeam(; na, amplitude = SuperGaussianPupil(; radius, order), samples)
    f(y) = abs2(first(PulseShapes.profile(b, 0.0, y))) - 0.5
    lo, hi = 0.0, 1 / na
    while f(hi) > 0                               # doubled until past the half-maximum, short of the first zero at about π/(na·radius)
        lo, hi = hi, 2 * hi
    end
    for _ in 1:80
        m = (lo + hi) / 2
        f(m) > 0 ? (lo = m) : (hi = m)
    end
    return 2 * (lo + hi) / 2
end

"The case's numerical aperture, `PCASE.na`: the one whose unaberrated focus has the base Gaussian's intensity FWHM, to 2·10⁻⁶ (T-1180)."
pulse_na() = PCASE.na

"""
    pulse_a0(shape; energy = budget()) -> a₀

The a₀ at which a [`ShapedLaser`](@ref) of `shape` carries `energy` per unit length in z
(`laser_energy`, polarized in the plane), in the shape's element type. The energy of the phase-only
shape does not depend on the free parameters, so neither does a₀, exactly: its partials are zero.
"""
function pulse_a0(shape; energy = budget(PCASE.a0_energy))
    unit = ShapedLaser(; a0 = 1, delay = PCASE.delay, focus = PCASE.x0[1], position = 0.0, center = 0.0, shape.profile, shape.envelope)
    return sqrt(energy / laser_energy(unit))
end

"""
    check_pulse_time(shape; delay = PCASE.delay, start = BCASE.start)

Refuses a pulse that has not started at t = 0 or has not ended when the window passes the antenna
plane (the window starts at `start`): its centroid ± six of its durations
([`PulseShapes.envelope_extent`](@ref)) must lie in [−delay, start − delay].
"""
function check_pulse_time(shape; delay = PCASE.delay, start = BCASE.start)
    c, h = map(v -> Float64(value_of(v)), PulseShapes.envelope_extent(shape.envelope))
    c - 6h >= -delay && c + 6h <= start - delay ||
        throw(ArgumentError("the pulse (centroid $c, duration $h) does not start after t = 0 and end before the window passes the antenna at delay $delay"))
    return nothing
end

"""
    pulse_problem(x; energy, ppl, ny, capacity, kw...) -> EMProblem{2}

The case at the five parameters `x` (focus, P₃, P₄, GDD, TOD): WP-065's problem at the base
density and delay with the [`ShapedLaser`](@ref) of [`pulse_shape`](@ref) at [`pulse_a0`](@ref)
in place of the Gaussian, and [`trapped_capacity`](@ref) slots by default; the other keywords as
[`born_problem`](@ref).
"""
function pulse_problem(x; energy = budget(PCASE.a0_energy), ppl::Integer = BCASE.ppl, ny::Integer = 1,
                       capacity::Integer = trapped_capacity(ppl, ny), kw...)
    shape = pulse_shape(x)
    check_pulse_time(shape)
    a0 = pulse_a0(shape; energy)
    F = typeof(float(value_of(a0)))
    density = F(PCASE.density)
    return born_problem(a0, F(TCASE.dopant_fraction) * density; shape, focus = x[1], delay = F(PCASE.delay), duration = F(PCASE.duration),
                        density, ppl, ny, capacity, kw...)
end

"""
The case's state after its run at the five parameters `x`: the run of [`pulse_problem`](@ref)'s
problem (and its keywords) on `backend` with `gather` ([`born_numerics`](@ref)).
"""
pulse_state(x; ppl::Integer = BCASE.ppl, ny::Integer = 1, backend = CPU(), gather = nothing, kw...) =
    run_to_end(start_of(pulse_problem(x; ppl, ny, kw...); ppl, ny, backend, gather))

"The objective at the five parameters: WP-065's [`trapped_energy`](@ref) of the run (keywords of [`pulse_problem`](@ref), `fade` and `rise`)."
pulse_objective(x; fade = TCASE.fade, rise = TCASE.rise, kw...) = trapped_energy(pulse_state(x; kw...); fade, rise)

"""
    launched_fraction(prob) -> (; box, interior)

The share of the profile's power ([`PulseShapes.profile_moments`](@ref), the integral of |U|² over
all y) that falls on the antenna's rows at the antenna plane: over the whole box height (`box`)
and over the rows inside the y absorbers (`interior`). An aberrated pupil's wings that leave the
box are energy the budget counts and the box never launches (#487), so a point well below 1 is
not at the fixed energy.
"""
function launched_fraction(prob; absorber_cells::Integer = BCASE.pml_y * (prob.cells[2] ÷ BCASE.rows))
    l = prob.lasers[1]
    b = PulseShapes.convert_eltype(Float64, l.profile)
    ξ = Float64(value_of(l.position)) - Float64(value_of(l.focus))
    hy = Float64(value_of(prob.length[2])) / prob.cells[2]
    yc = Float64(value_of(l.center))
    power = first(PulseShapes.profile_moments(b))
    rows = [abs2(first(PulseShapes.profile(b, ξ, (j - 0.5) * hy - yc))) * hy for j in 1:prob.cells[2]]
    inner = (absorber_cells + 1):(prob.cells[2] - absorber_cells)
    return (; box = sum(rows) / power, interior = sum(rows[inner]) / power)
end

"""
    strehl(shape) -> S

The Strehl ratio of the shape's focus in vacuum: the peak of |U|² over the focal plane's y,
relative to the unaberrated beam's 1, on a grid of 2001 points over ±3 FWHM refined by golden
section around its largest sample.
"""
function strehl(shape)
    b = PulseShapes.convert_eltype(Float64, shape.profile)
    I(y) = abs2(first(PulseShapes.profile(b, 0.0, y)))
    w = 3 * TCASE.x0[2] * sqrt(2 * log(2))
    ys = range(-w, w; length = 2001)
    k = argmax(I.(ys))
    lo, hi = ys[max(k - 1, 1)], ys[min(k + 1, end)]
    g = (sqrt(5) - 1) / 2
    for _ in 1:60
        a, c = hi - g * (hi - lo), lo + g * (hi - lo)
        I(a) > I(c) ? (hi = c) : (lo = a)
    end
    return I((lo + hi) / 2)
end
