# The 3D half of WP-087: the 3D trapped-beam case (trapped3d_case.jl, WP-084) with the plasma and
# the laser energy fixed and the pulse's shape free. The pulse is a 3D ShapedLaser (WP-085): a
# fixed round super-Gaussian near field focused by a lens of fixed numerical aperture, with a
# wavefront of Zernike modes up to radial order 4, and a Gaussian spectrum with a spectral phase.
# Its fourteen parameters are the focus, eleven Zernike modes (OSA j = 3 and 5 to 14, in waves of
# unit rms: no piston, no tilts, no defocus, and the spherical mode balanced against defocus), the
# GDD and the TOD. Everything the optimizer moves is a phase (the focus moves the lens), so the
# pulse carries the same energy at every point and a₀ is a constant of the case, as in the 2D pilot
# (pulse2d_case.jl).
isdefined(@__MODULE__, :budget3) || include(joinpath(@__DIR__, "trapped3d_case.jl"))
using ParticleMeshElectromagnetic: tabulate, launched_energy
using ParticleMeshElectromagnetic.PulseShapes: radial_node, zernike

"""
Fixed parameters of the 3D case beyond [`TCASE`](@ref), [`BCASE`](@ref) and the 3D case of
[`trapped3d_problem`](@ref) (A1 units).

- The plasma is the 3D case's at its base density `density` (n_c), with the dopant at
  `TCASE.dopant_fraction` of it; the delay is the base point's `delay`; the energy is the round
  Gaussian's budget of a₀ = `a0_energy` at the base pulse (waist 10, duration 6), [`budget3`](@ref).
- The near field is `SuperGaussianPupil(; radius, order)` across a lens of numerical aperture
  `na`, the 2D pilot's near field made round: its amplitude at the rim is ≈ 5·10⁻⁵, and `na` puts
  the unaberrated focus's intensity FWHM at the round base Gaussian's, 10·√(2 ln 2) ≈ 11.8 c/ω₀
  (T-1230). The pupil sum has `radial` × `angular` samples ([`PulseShapes.ZernikeBeam`](@ref)),
  enough to resolve the beam over the box's whole cross-section at every point of the box
  ([`PulseShapes.quadrature_radius`](@ref), T-1230).
- The spectrum is Gaussian with the transform-limited `duration` of the base pulse, through
  [`PulseShapes.SpectralEnvelope`](@ref) with `spectral_samples` points.
- `modes` are the Zernike modes (OSA/ANSI j) the optimizer moves; j = 1, 2 (tilts) and 4 (defocus)
  are not free, since tilts steer the beam off the axis and defocus moves the focus, which `focus`
  already does; j = 12 is the spherical mode balanced against defocus ([`spherical_balance`](@ref)),
  so j = 4 carries its balance and is zero otherwise. With E along y (polarization 0), the modes with m < 0 (sin mθ: 3, 6, 7, 10, 11) are
  odd under the mirror z → −z, which keeps the polarization, so their gradient vanishes at an
  unaberrated pulse (T-1233); a mode odd under y → −y is paired with the opposite carrier phase.
- `x0` is the unaberrated, transform-limited pulse focused at the base point's 80; `names` and
  `scale` name each parameter and give the scale of z = x ./ scale in problem.jl.
"""
const PCASE3 = (a0_energy = 1.9, density = 0.072, delay = 54.0, duration = 6.0, na = 0.371536, radius = 0.75, order = 4, radial = 64, angular = 256,
                spectral_samples = 512, modes = (3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                x0 = (80.0, ntuple(_ -> 0.0, 11)..., 0.0, 0.0),
                names = ("focus", ntuple(k -> "z" * string((3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)[k]), 11)..., "gdd", "tod"),
                scale = (80.0, ntuple(_ -> 0.1, 11)..., 10.0, 25.0))

"""
The optimizer's box (the 3D half of WP-087): the focus over WP-068's 64 to 110, each Zernike mode
within ±`PULSE3_MODE_BOUND` waves rms, the GDD within ±12 and the TOD within ±30 (1/ω₀² and 1/ω₀³),
the 2D pilot's spectral box. Eleven modes at their bounds together are a wavefront of about a third
of a wave rms, whose wings leave the box's cross-section: the share of the energy driven inside the
transverse absorbers is 99.2 % at the unaberrated pulse (98 % at focus 110) and falls to 88.5 % at
the worst of the box's corners (`questions/wp-087-pulse-shape-study.md` §8, T-1231), so every
point records its share ([`launched_share3`](@ref)).
"""
const PULSE3_MODE_BOUND = 0.1
const PULSE3_BOX = (lo = [64.0; fill(-PULSE3_MODE_BOUND, 11); -12.0; -30.0], hi = [110.0; fill(PULSE3_MODE_BOUND, 11); 12.0; 30.0],
                    scale = collect(PCASE3.scale))

"The number of the case's parameters, 14."
pulse3_nparams() = length(PCASE3.x0)

"""
    pulse3_shape(x; na = pulse3_na(), quadrature = (PCASE3.radial, PCASE3.angular)) -> (; profile, envelope)

The pulse's transverse profile and temporal envelope at the fourteen parameters `x` (focus, the
modes of `PCASE3.modes`, GDD, TOD), in their promoted element type: a
[`PulseShapes.ZernikeBeam`](@ref) with the aberration of mode j at `x[1 + k]` for the k-th of
`PCASE3.modes`, −α·x for j = 4 with x the spherical mode's ([`spherical_balance`](@ref)), and zero
for j = 1, 2, and a [`PulseShapes.SpectralEnvelope`](@ref). The focus is
the laser's, not the shape's. `quadrature` is the pupil sum's radial nodes and angular points
(the CPU tests take fewer on their smaller box).
"""
function pulse3_shape(x; na = pulse3_na(), quadrature = (PCASE3.radial, PCASE3.angular))
    length(x) == pulse3_nparams() || throw(ArgumentError("pulse3_shape takes the $(pulse3_nparams()) parameters, got $(length(x))"))
    T = promote_type(map(typeof, Tuple(x))...)
    n = maximum(PCASE3.modes)
    c = ntuple(m -> (k = findfirst(==(m), PCASE3.modes); k === nothing ? zero(T) : T(x[1 + k])), n)
    aberrations = ntuple(m -> m == 4 ? -T(PULSE3_BALANCE) * c[12] : c[m], n)     # the spherical mode balanced against defocus
    profile = ZernikeBeam(; na = T(na), amplitude = SuperGaussianPupil(; radius = T(PCASE3.radius), order = PCASE3.order), aberrations,
                          radial = quadrature[1], angular = quadrature[2])
    k = length(PCASE3.modes)
    envelope = SpectralEnvelope(; duration = T(PCASE3.duration), gdd = T(x[2 + k]), tod = T(x[3 + k]), samples = PCASE3.spectral_samples)
    return (; profile, envelope)
end

"""
    spherical_balance(; radius, order, nodes = 256) -> α

The defocus in the case's near field's Z₁₂ (primary spherical aberration): the coefficient of Z₄
in the least-squares fit of Z₁₂ by piston and Z₄ over the pupil, weighted by the field's
amplitude, ρ·a(ρ) dρ, on `nodes` Gauss–Legendre nodes. Zernike modes are orthogonal over the
uniform disk, not over this flat-top, whose amplitude falls off before the rim: there 0.1 waves
of Z₁₂ alone move the beam's on-axis peak by about 45 c/ω₀, nearly the focus box's width (the
physics breaker of WP-087's 3D half). So the case's mode j = 12 is Z₁₂ − α·Z₄, whose
amplitude-weighted defocus is zero: to first order in the wavefront it keeps the peak at `focus`
and the focus parameter and the mode do not trade for each other (T-1230).
"""
function spherical_balance(; radius = PCASE3.radius, order = PCASE3.order, nodes::Integer = 256)
    a = SuperGaussianPupil(; radius, order)
    s = sz4 = sz12 = s44 = s412 = 0.0
    for i in 1:nodes
        ρ, wρ = radial_node(nodes, i, Float64)
        w = ρ * wρ * PulseShapes.pupil_amplitude(a, ρ)
        z4, z12 = zernike(4, ρ, 0.0), zernike(12, ρ, 0.0)
        s += w; sz4 += w * z4; sz12 += w * z12; s44 += w * z4^2; s412 += w * z4 * z12
    end
    return (s412 / s - sz4 * sz12 / s^2) / (s44 / s - (sz4 / s)^2)
end

"The case's balance of Z₁₂ against defocus, [`spherical_balance`](@ref)."
const PULSE3_BALANCE = spherical_balance()

"""
    focal_fwhm3(na; radius, order, radial, angular) -> FWHM

The intensity FWHM along y of the unaberrated round focus of the case's near field through a lens
of numerical aperture `na`, by bisection on |U(0, y, 0)|² = 1/2 (1 on the axis, falling
monotonically to its first zero).
"""
function focal_fwhm3(na; radius = PCASE3.radius, order = PCASE3.order, radial = 32, angular = 64)
    b = ZernikeBeam(; na, amplitude = SuperGaussianPupil(; radius, order), radial, angular)
    f(y) = abs2(first(PulseShapes.profile(b, 0.0, y, 0.0))) - 0.5
    lo, hi = 0.0, 1 / na
    while f(hi) > 0
        lo, hi = hi, 2 * hi
    end
    for _ in 1:80
        m = (lo + hi) / 2
        f(m) > 0 ? (lo = m) : (hi = m)
    end
    return 2 * (lo + hi) / 2
end

"The case's numerical aperture, `PCASE3.na`: the one whose unaberrated round focus has the base Gaussian's intensity FWHM, to 10⁻⁶ (T-1230)."
pulse3_na() = PCASE3.na

"The 3D antenna's laser of `shape` at a₀ = 1 with the case's delay and `focus`, in the element type `T` (the shape's), its center the 3D case's axis (only the energy and the shape are read from it)."
pulse3_unit(shape; T = typeof(shape.profile.na), focus = PCASE3.x0[1], rows::Integer = BCASE.rows) =
    ShapedLaser(; a0 = one(T), delay = T(PCASE3.delay), focus = T(focus), polarization = zero(T), phase = zero(T), position = 0.0,
                center = (rows * π / 8, rows * π / 8), shape.profile, shape.envelope)

"""
    pulse3_a0(shape; energy = budget3()) -> a₀

The a₀ at which a 3D [`ShapedLaser`](@ref) of `shape` carries `energy` (`laser_energy`), in the
shape's element type (a Float32 dual stays one: the energy and the unit laser are taken in it). A
phase-only shape's energy does not depend on the free parameters, so neither does a₀, exactly: its
partials are zero.
"""
function pulse3_a0(shape; energy = budget3(PCASE3.a0_energy))
    T = typeof(shape.profile.na)
    return sqrt(convert(T, energy) / laser_energy(pulse3_unit(shape; T)))
end

"""
    check_pulse3_time(shape; delay, start)

Refuses a pulse that has not started at t = 0 or has not ended when the window passes the antenna
plane: [`check_pulse_time`](@ref)'s condition, centroid ± six durations within [−delay, start − delay].
"""
function check_pulse3_time(shape; delay = PCASE3.delay, start = BCASE.start)
    c, h = map(v -> Float64(value_of(v)), PulseShapes.envelope_extent(shape.envelope))
    c - 6h >= -delay && c + 6h <= start - delay ||
        throw(ArgumentError("the pulse (centroid $c, duration $h) does not start after t = 0 and end before the window passes the antenna at delay $delay"))
    return nothing
end

"""
    pulse3_problem(x; energy, phase, quadrature, ppl, nt, rows, float, kw...) -> EMProblem{3}

The 3D case at the fourteen parameters `x`: [`trapped3d_problem`](@ref) at the base density and
delay with the [`ShapedLaser`](@ref) of [`pulse3_shape`](@ref) at [`pulse3_a0`](@ref) and the
carrier `phase` in place of the Gaussian, its pupil sum on `quadrature` samples; the other
keywords as `trapped3d_problem`'s (Float32 by default, as there).
"""
function pulse3_problem(x; energy = budget3(PCASE3.a0_energy), phase = 0.0, quadrature = (PCASE3.radial, PCASE3.angular), ppl::Integer = 16,
                        nt::Integer = 1, rows::Integer = BCASE.rows, float::Type{<:AbstractFloat} = Float32, kw...)
    shape = pulse3_shape(x; quadrature)
    check_pulse3_time(shape)
    a0 = pulse3_a0(shape; energy)
    return trapped3d_problem(PCASE3.duration, TCASE.x0[2], x[1], PCASE3.delay, PCASE3.density; float, ppl, nt, rows,
                             shape = (; shape..., phase), a0, kw...)
end

"""
    pulse3_start(x; backend, ppl, nt, gather, kw...) -> (st, p, prob)

The case's state at its start at the fourteen parameters `x` on `backend`, its parameters with the
laser's table built on the state ([`tabulate`](@ref)), and its problem ([`pulse3_problem`](@ref)'s
keywords).
"""
function pulse3_start(x; backend = CPU(), ppl::Integer = 16, nt::Integer = 1, gather = CompensatedGather(; k = 1), kw...)
    prob = pulse3_problem(x; ppl, nt, kw...)
    return trapped3d_start(prob; backend, ppl, nt, gather)
end

"The state at the end of the case's run at `x` ([`pulse3_start`](@ref)'s keywords)."
pulse3_state(x; kw...) = run_to_end(pulse3_start(x; kw...))

"The objective at the fourteen parameters: the 3D case's [`trapped_energy3`](@ref) of the run (keywords of [`pulse3_start`](@ref), `fade` and `rise`)."
pulse3_objective(x; fade = TCASE.fade, rise = TCASE.rise, kw...) = trapped_energy3(pulse3_state(x; kw...); fade, rise)

"""
    launched_share3(laser; absorber_cells = BCASE.pml_y) -> (; box, interior)

The share of a tabulated 3D laser's energy ([`laser_energy`](@ref)) that the antenna drives
through the box's cross-section ([`launched_energy`](@ref), `box`) and through the rows and layers
inside the transverse absorbers (`interior`), from its table. An aberrated pupil's wings that
leave the cross-section are energy the budget counts and the box never launches, so a point well
below 1 is not at the fixed energy.
"""
function launched_share3(l; absorber_cells::Integer = BCASE.pml_y)
    V = Float64.(Array(value_of.(l.profile.values)))
    a0 = Float64(value_of(l.a0))
    E = Float64(value_of(laser_energy(l)))
    t = l.profile
    k = absorber_cells
    ny, nz = size(V, 1), size(V, 2)
    w = a0^2 / 2 * Float64(value_of(PulseShapes.envelope_energy(l.envelope))) * Float64(value_of(t.plane.dx[2])) * Float64(value_of(t.plane.dx[3]))
    sθ, cθ = sincos(Float64(value_of(l.polarization)))
    s(J, K) = w * (cθ^2 * sum(abs2, view(V, J, K, 1:2)) + sθ^2 * sum(abs2, view(V, J, K, 7:8)))
    return (; box = s(1:ny, 1:nz) / E, interior = s((k + 1):(ny - k), (k + 1):(nz - k)) / E)
end

"""
    strehl3(shape) -> S

The Strehl ratio of the shape's focus in vacuum: the peak of |U|² over the focal plane, relative to
the unaberrated beam's 1, from a grid of 25 × 25 points over ±1.5 FWHM refined by a compass search
around its largest sample.
"""
function strehl3(shape)
    b = PulseShapes.convert_eltype(Float64, shape.profile)
    I(y, z) = abs2(first(PulseShapes.profile(b, 0.0, y, z)))
    w = 1.5 * TCASE.x0[2] * sqrt(2 * log(2))
    ys = range(-w, w; length = 25)
    best = argmax(((y, z),) -> I(y, z), ((y, z) for y in ys, z in ys))
    y, z = best
    f = I(y, z)
    s = step(ys) / 2
    while s > 1e-6
        moved = false
        for (dy, dz) in ((s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s))
            g = I(y + dy, z + dz)
            g > f && ((y, z, f) = (y + dy, z + dz, g); moved = true; break)
        end
        moved || (s /= 2)
    end
    return f
end
