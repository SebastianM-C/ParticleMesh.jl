# The WP-064 case: a small 2D LWFA run whose pulse ionizes the K shell of a nitrogen dopant in a
# short section (WP-059), so that electrons are born inside the wake and some are trapped, with the
# energy those born electrons carry in the first bucket as the objective. The plasma's own
# electrons are another species and are never read by the objective: which electrons count is a
# label, not an energy cut.

"""
Fixed parameters of the case (A1 units).

- The box is 14 λ₀ long (`ppl` cells per λ₀ along x) and 128 cells of Δy = π/4 wide, a moving
  window with PMLs at the open y edges and at the back, as the run of the ionization explainer
  page: the laser (waist 10, duration 6, focus at x = 80) enters a plasma of density 0.072 n_c
  behind a ramp from x = 64 to 104. The ramp is that long on purpose: behind one of 8 c/ω₀ the wake
  traps the plasma's own electrons at every a₀ from 1.5 up, behind this one at λ₀/16 only from
  a₀ = 2.25, and then behind window x = 12 (#465; `born2d.jl` reports the guard at every point).
- The dopant is nitrogen at 1 % of the plasma density on the 36 rows around the axis, in the
  section x = 108 to 132 (ramps of 4), which starts 4 c/ω₀ behind the ramp's end, with the two
  K-shell levels N⁵⁺ → N⁶⁺ → N⁷⁺ at 0.8 µm. The
  L shell is left out: the pulse's foot ionizes it completely, far ahead of the wake, and its
  electrons are part of the plasma's density. ADK rates with no barrier-suppression correction
  and no ionization current (WP-059), so the charge injected is not a physical prediction.
- `fade = (lo, hi)` is the objective's position weight in window x (c/ω₀): 0 behind `lo`, 1 ahead
  of `hi`. At the read-out of the λ₀/16 run at a₀ = 1.9 the pulse's peak is at window x ≈ 49 and the
  first bucket lies between the zero crossings of E_x at x ≈ 27 and 40 (`born2d.jl` measures both),
  so the weight is 1 over the bucket; the back band ends at x ≈ 3.5 at λ₀/16.
- `u_fast` is the momentum above which a born electron is listed as trapped in the run's tables;
  the objective does not use it.
"""
const BCASE = (waist = 10.0, delay = 54.0, focus = 80.0, vacuum = 64.0, ramp = 40.0, start = 108.0, a0 = 1.9,
               duration = 6.0, density = 0.072, ppl = 32, length_wavelengths = 14, rows = 128, dy = π / 4, ppc = (2, 2),
               t_end = 300.0, pml_y = 12, pml_x_wavelengths = 0.25,
               dopant = 0.00072, section = (start = 108.0, stop = 132.0, ramp = 4.0), half_rows = 18, cutoff = 24.0,
               every = 1, capacity = 400_000, wavelength = 0.8e-6, fade = (12.0, 24.0), u_fast = 5.0)

"""
    born_problem(a0, dopant; t_end, ppl, ny, every, capacity, section, duration, density, ramp, waist, delay, focus) -> EMProblem{2}

The case at a normalized vector potential `a0` and a dopant density `dopant` (n_c); the element
type follows the arguments. `ppl` cells per λ₀ along x at the same box (it must be a multiple of
8), and `ny` times as many cells along y as the case's 128 of Δy = π/4 (Δy = π/(4·ny), with the
PMLs and the dopant's rows the same width); `t_end` the read-out time, `ramp` the length of the
plasma's density ramp (the dopant's `section` does not move with it: pass both); `waist`,
`delay` and `focus` the pulse's ([`GaussianLaser`](@ref)), or with `shape = (; profile, envelope)`
of [`PulseShapes`](@ref) (and optionally a carrier `phase`) a [`ShapedLaser`](@ref) of that profile
and envelope at `a0`, `delay` and `focus` in place of the Gaussian (`waist` and `duration` then unused, WP-087); `rows` the box's
height in cells of Δy = π/4 (the case's 128; the absorbers' and the dopant's widths stay, WP-087
widens it to check what its flat-top pulse loses at the y edges); `extend` whole wavelengths added to
the window behind the case's 14, with the window's start delayed by as much, so that the box at
every time holds the case's box and that much more of the wake behind it (the front, the plasma
and the pulse are where they are in the case, in the lab frame). The element type is `float`
(Float64 by default, Float32 for the GPU), with every plain float argument taken in it, or the
argument's own type when it is a dual (a dual of `float` keeps the run in `float`).

Float32 serves the objective and most of its ForwardDiff gradient: Float32 duals on an RTX 5090
against Float64 on the CPU (WP-065's `trapped_energy`) give J within 8·10⁻⁵ at λ₀/16 and λ₀/32
at five points, and d ln J/d ln x of the free parameters and the budget within 0.05, except at
0.085 n_c and λ₀/32, where they differ by 2 (open in WP-065's report); at λ₀/8 to t = 190, J
within 1.6·10⁻⁵ (T-914). Float64 stays the default for the central
differences that check the gradient: their steps reach 2⁻¹⁸ of a parameter, below Float32's
resolution of it (2⁻²⁴), where the difference of two Float32 runs is roundoff.
"""
function born_problem(a0 = BCASE.a0, dopant = BCASE.dopant; t_end = BCASE.t_end, ppl::Integer = BCASE.ppl, ny::Integer = 1, every::Integer = BCASE.every,
                      capacity::Integer = BCASE.capacity, section = BCASE.section, duration = BCASE.duration, density = BCASE.density, ramp = BCASE.ramp,
                      waist = BCASE.waist, delay = BCASE.delay, focus = BCASE.focus, extend::Integer = 0,
                      float::Type{<:AbstractFloat} = Float64, shape = nothing, rows::Integer = BCASE.rows)
    ppl % 8 == 0 || throw(ArgumentError("ppl must be a multiple of 8 to keep Δy = π/4, got $ppl"))
    ny >= 1 || throw(ArgumentError("ny must be at least 1, got $ny"))
    extend >= 0 || throw(ArgumentError("extend must be at least 0, got $extend"))
    base(x) = x isa AbstractFloat ? float : typeof(x)
    shaped = shape === nothing ? () : (PulseShapes.parameter_values(shape.profile)..., PulseShapes.parameter_values(shape.envelope)...)
    T = promote_type(base(a0), base(dopant), base(duration), base(density), base(waist), base(delay), base(focus), map(base, shaped)..., float)
    z3 = (zero(T), zero(T), zero(T))
    h = 2 * T(π) / ppl
    hy = (ppl ÷ 8) * h / ny
    cells = ((BCASE.length_wavelengths + extend) * ppl, rows * ny)
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    nsteps = ceil(Int, t_end / Float64(ForwardDiff.value(dt)))
    hy64 = Float64(ForwardDiff.value(hy))
    yc = cells[2] * hy64 / 2
    position, center = max(12, 192 ÷ ppl) * (2 * T(π) / 32), cells[2] * hy / 2
    pulse = shape === nothing ?
            GaussianLaser(; a0 = T(a0), waist = T(waist), duration = T(duration), delay = T(delay), focus = T(focus), position, center) :
            ShapedLaser(; a0 = T(a0), delay = T(delay), focus = T(focus), position, center, profile = PulseShapes.convert_eltype(T, shape.profile),
                        envelope = PulseShapes.convert_eltype(T, shape.envelope), phase = T(get(shape, :phase, 0)))
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(density), zero(T), (1, 0), BCASE.ppc, z3, z3, 1)
    levels = (adk_level(T, :N, 6; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff),
              adk_level(T, :N, 7; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff))
    ion = FieldIonization(; density = T(dopant), levels, capacity, every, ppc = (1, 1),
                          rows = (yc - BCASE.half_rows * ny * hy64, yc + BCASE.half_rows * ny * hy64),
                          section = DopantSection(; start = T(section.start), stop = T(section.stop), ramp = T(section.ramp)))
    return EMProblem("born2d", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :open), window = MovingWindow(; start = T(BCASE.start) + extend * 2 * T(π)),
                     profile = DensityRamp(; vacuum = T(BCASE.vacuum), ramp = T(ramp)), ionization = ion)
end

"""
The case's numerics: Reference mode, PMLs at the open y edges (`ny` times 12 cells) and at the
window's back (a quarter wavelength, at least 4 cells; the antenna is 12 cells of λ₀/32 from the
back edge, at least 6 cells), and the particles' `gather`: the grid's fields (`nothing`, the
default, WP-064's and WP-065's runs) or a `CompensatedGather` (WP-067).
"""
born_numerics(ppl::Integer = BCASE.ppl, ny::Integer = 1; gather = nothing) =
    EMNumerics(; mode = Reference(), absorber = PML(; cells = BCASE.pml_y * ny), back_absorber = PML(; cells = max(4, round(Int, BCASE.pml_x_wavelengths * ppl))), gather)

"The case's state at its start on `backend` and its parameters; `gather` as [`born_numerics`](@ref), `ppl`, `ny` and the rest as [`born_problem`](@ref)."
born_start(a0 = BCASE.a0, dopant = BCASE.dopant; ppl::Integer = BCASE.ppl, ny::Integer = 1, backend = CPU(), gather = nothing, kw...) =
    start_of(born_problem(a0, dopant; ppl, ny, kw...); ppl, ny, backend, gather)

"The state of a problem of the case at its start on `backend`, with the case's numerics (and `gather`) at the `ppl` and `ny` the problem was made with, its parameters, and the problem."
function start_of(prob; ppl::Integer, ny::Integer, backend = CPU(), gather = nothing)
    st = EMState(prob, born_numerics(ppl, ny; gather); record = (), backend)
    p = params(prob)
    initialize!(st, p)
    return st, p, prob
end

"The state at the end of the run of a start (a state, its parameters and its problem). A birth that found no free slot is an error: the capacity was too small."
function run_to_end((st, p, prob))
    run!(st, p, prob.nsteps)
    ionization_lost(st) == 0 || error("$(ionization_lost(st)) births found no free slot: raise `capacity`")
    return st
end

"The case's state after its run (the arguments of [`born_start`](@ref))."
born_state(a0 = BCASE.a0, dopant = BCASE.dopant; kw...) = run_to_end(born_start(a0, dopant; kw...))

"""
    smooth_rise(x, lo, hi)

The C² step from 0 at `x ≤ lo` to 1 at `x ≥ hi`: `s³(10 − 15s + 6s²)` of `s = (x − lo)/(hi − lo)`.
Its first and second derivatives vanish at both ends, and `smooth_rise(x) + smooth_rise(lo + hi − x) = 1`.
"""
@inline function smooth_rise(x, lo, hi)
    s = clamp((x - lo) / (hi - lo), zero(x), one(x))
    return s * s * s * (10 - 15 * s + 6 * s * s)
end

"""
    check_fade(st, fade)

Refuses a position weight that is not 0 on the window's back band: `fade[1]` must lie at or ahead
of the band's end (`window_back_band` columns from the back edge), and `fade[2] > fade[1]`.
"""
function check_fade(st, fade)
    band = back_band_end(st)
    fade[2] > fade[1] || throw(ArgumentError("the position weight needs fade[2] > fade[1], got $fade"))
    fade[1] >= band || throw(ArgumentError("the position weight must vanish on the window's back band, which ends at x = $band; got fade[1] = $(fade[1])"))
    return nothing
end

# The summand of an electron slot: weight × kinetic energy × position weight, zero for a free slot.
@inline function slot_energy(status, c1, o1, u1, u2, u3, w, h1, lo, hi)
    x = (c1 - 1 + o1) * h1
    return ifelse(status == ALIVE, w * (sqrt(1 + u1 * u1 + u2 * u2 + u3 * u3) - 1) * smooth_rise(x, lo, hi), zero(w))
end

"""
    born_energy(st; fade = BCASE.fade) -> J

The objective: the kinetic energy the ionization-born electrons carry ahead of the window's back,

    J = Δx·Δy · Σ_i w_i·(γ_i − 1)·S(x_i),

over the electrons the state's `FieldIonization` has released and that are alive (the species
[`ionization_electrons`](@ref); no electron of the plasma's species enters), in n_c·mₑc²·(c/ω₀)²,
an energy per unit length along z. `S` is [`smooth_rise`](@ref) of the window coordinate
x between `fade[1]` and `fade[2]`: an electron that falls behind the wake leaves the sum smoothly
before it reaches the back band, where the window's edge would take it out at a step that depends
on the parameters (#370, `questions/wp-059-field-ionization.md` §3). The summands are taken over
the device arrays, so it runs on any backend, and added on the host one after the other in slot
order: Base's `sum` lets the compiler reassociate for SIMD, differently for `Float64` and for a
dual on some CPUs, and a run with duals must give the plain run's `J` bit for bit (T-833).
"""
function born_energy(st; fade = BCASE.fade)
    check_fade(st, fade)
    e = ionization_electrons(st)
    h = cellsize(st.grid)
    T = eltype(e.w)
    lo, hi = T(fade[1]), T(fade[2])
    return h[1] * h[2] * foldl(+, Array(slot_energy.(e.status, e.cell[1], e.offset[1], e.u[1], e.u[2], e.u[3], e.w, h[1], lo, hi)))
end

"The objective of the case at `a0` and `dopant` (the keywords of [`born_problem`](@ref) and `fade`)."
born_objective(a0 = BCASE.a0, dopant = BCASE.dopant; fade = BCASE.fade, kw...) = born_energy(born_state(a0, dopant; kw...); fade)

"The window x at which the back band ends: a read-out starts there or ahead of it (`window_back_band`)."
back_band_end(st) = window_back_band(st) * Float64(ForwardDiff.value(cellsize(st.grid)[1]))

function check_from(st, from)
    from >= back_band_end(st) || throw(ArgumentError("the guard reads ahead of the window's back band, which ends at x = $(back_band_end(st)); got from = $from"))
    return nothing
end

# The alive particles of non-zero weight of a plasma species ahead of `from`, as plain numbers:
# window x, kinetic energy, weight. The particles loaded in the vacuum and at the ramp's foot have
# the weight zero (the profile enters through weights, A3) and move as test particles; they carry
# nothing.
function plasma_arrays(s, h1, from)
    f(a) = Float64.(ForwardDiff.value.(Array(a)))
    x = (Array(s.cell[1]) .- 1 .+ f(s.offset[1])) .* h1
    w = f(s.w)
    keep = (Array(s.status) .== ALIVE) .& (w .> 0) .& (x .>= from)
    u = map(a -> f(a)[keep], s.u)
    return x[keep], sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1, w[keep]
end

"""
    plasma_peak_energy(st; from = BCASE.fade[1]) -> γ − 1

The guard that goes with the objective: the highest kinetic energy (mₑc²) of an alive electron of
non-zero weight of the plasma's species at window x ≥ `from`, a plain number. In the laser a
plasma electron quivers up to about a₀²/2; a value far above that says plasma electrons were
injected into the wake themselves, energy the objective does not see. The default reads where the
objective's position weight is not zero; `from = back_band_end(st)` reads the whole window ahead
of the back band, the buckets behind the first one included.
"""
function plasma_peak_energy(st; from = BCASE.fade[1])
    check_from(st, from)
    h1 = Float64(ForwardDiff.value(cellsize(st.grid)[1]))
    return maximum(st.species; init = 0.0) do s
        maximum(plasma_arrays(s, h1, from)[2]; init = 0.0)
    end
end

"""
    plasma_fast_weight(st; from = BCASE.fade[1], energy = BCASE.u_fast) -> weight

The weight (times Δx·Δy) of the plasma's electrons at window x ≥ `from` whose kinetic energy
exceeds `energy` (mₑc²): the charge the wake has injected from the plasma itself. A count with a
hard cut: it changes in steps of a particle's weight.
"""
function plasma_fast_weight(st; from = BCASE.fade[1], energy = BCASE.u_fast)
    check_from(st, from)
    h = Float64.(ForwardDiff.value.(cellsize(st.grid)))
    return h[1] * h[2] * sum(st.species; init = 0.0) do s
        _, g, w = plasma_arrays(s, h[1], from)
        sum(w[g .> energy]; init = 0.0)
    end
end

"""
    plasma_fast_energy(st; from = BCASE.fade[1], energy = BCASE.u_fast) -> energy

The kinetic energy Δx·Δy·Σ w(γ − 1) of the electrons [`plasma_fast_weight`](@ref) counts, in the
objective's units: what the plasma's own injected electrons carry next to J.
"""
function plasma_fast_energy(st; from = BCASE.fade[1], energy = BCASE.u_fast)
    check_from(st, from)
    h = Float64.(ForwardDiff.value.(cellsize(st.grid)))
    return h[1] * h[2] * sum(st.species; init = 0.0) do s
        _, g, w = plasma_arrays(s, h[1], from)
        sum((w .* g)[g .> energy]; init = 0.0)
    end
end

"""
    born_summary(st; fade = BCASE.fade, u_fast = BCASE.u_fast) -> NamedTuple

Plain numbers about the born electrons for a run's tables: `born`, the weight the atoms have
released as electrons, summed over the atoms still in the window (Σ w·emitted); `alive`, the weight still in the
window; `counted`, Σ w·S(x); `fast`, the weight with u_x > `u_fast`; `fast_counted`, the same times
S(x); `u_max`; `energy`, the objective; `mean_energy`, energy over counted weight (mₑc²); `plasma_peak`, `plasma_fast` and `plasma_fast_energy`, the guards ([`plasma_peak_energy`](@ref), [`plasma_fast_weight`](@ref) and [`plasma_fast_energy`](@ref) above `u_fast`) ahead of `fade[1]`; and the same three with `_window`, ahead of the back band. Weights are times Δx·Δy (n_c·(c/ω₀)²).
"""
function born_summary(st; fade = BCASE.fade, u_fast = BCASE.u_fast)
    check_fade(st, fade)
    band = back_band_end(st)
    h = Float64.(ForwardDiff.value.(cellsize(st.grid)))
    area = h[1] * h[2]
    e = ionization_electrons(st)
    alive = Array(e.status) .== ALIVE
    f(a) = Float64.(ForwardDiff.value.(Array(a)))[alive]
    x = (Array(e.cell[1])[alive] .- 1 .+ f(e.offset[1])) .* h[1]
    w, ux = f(e.w), f(e.u[1])
    S = smooth_rise.(x, fade[1], fade[2])
    a = ionization_state(st)
    born = sum(Float64.(ForwardDiff.value.(a.w)) .* Float64.(ForwardDiff.value.(a.emitted)))
    energy = Float64(ForwardDiff.value(born_energy(st; fade)))
    counted = sum(w .* S)
    return (; born = area * born, alive = area * sum(w), counted = area * counted, fast = area * sum(w[ux .> u_fast]),
            fast_counted = area * sum((w .* S)[ux .> u_fast]), u_max = isempty(ux) ? 0.0 : maximum(ux), energy,
            mean_energy = counted > 0 ? energy / (area * counted) : 0.0, plasma_peak = plasma_peak_energy(st; from = fade[1]),
            plasma_fast = plasma_fast_weight(st; from = fade[1], energy = u_fast),
            plasma_fast_energy = plasma_fast_energy(st; from = fade[1], energy = u_fast),
            plasma_peak_window = plasma_peak_energy(st; from = band), plasma_fast_window = plasma_fast_weight(st; from = band, energy = u_fast),
            plasma_fast_energy_window = plasma_fast_energy(st; from = band, energy = u_fast), particles = count(alive))
end

# The WP-065 case: WP-064's ionization case above at a fixed laser energy, with an
# objective over every trapped electron, born by ionization or injected from the plasma itself.
# Its five parameters are the pulse's duration, waist, focus and delay and the plasma's density;
# a₀ follows from the duration and the waist so that the pulse carries the same energy at every
# point (WP-052's coupling, with the waist free as well).

"""
Fixed parameters of the WP-065 case beyond [`BCASE`](@ref) (A1 units).

- `energy` is the laser energy per unit length in z that the pulse carries through the antenna
  plane, [`laser_energy`](@ref) of the case's pulse at `a0_energy` (π·a₀²·w·τ/4·(1 + 1/(2w²)) at
  the base waist 10 and duration 6). The study's three budgets are those of a₀ = 1.9, 2.175 and
  2.4 at the base pulse (`budgets`), around the onset of the plasma's own injection in WP-064.
- `dopant_fraction`: the dopant's density is this fraction of the plasma's at every density, so
  the density moves both.
- `rise = (lo, hi)` is the trapping weight in the forward momentum u_x: 0 below `lo`, 1 above
  `hi` ([`smooth_rise`](@ref)). The wake's phase velocity at 0.072 n_c has γ_φ ≈ (n_c/n)^½ ≈ 3.7,
  so an electron with u_x above about 3.6 outruns the wake and stays in the bucket; the plasma's
  own electrons quiver below u_x ≈ 1 in the clean runs of WP-064 (their highest γ − 1 is 0.3 to
  1.0), and the born electrons are born near rest.
- `fade = (lo, hi)` is the position weight in window x (c/ω₀), WP-064's: 1 over the first bucket
  behind the pulse, 0 behind x = 12, so J counts every trapped electron of the first bucket,
  born or the plasma's own. The plasma injects itself in this case only into the buckets behind
  the first, wave-breaking charge that grows with the length of wake the window keeps (4.7 times
  the born energy at a₀ = 2.4 at λ₀/16 with the window 4 λ₀ longer, 23 times with it 6 λ₀ longer):
  it is the dark current, reported beside J ([`trapped_summary`](@ref)) and not counted in it.
- `x0` is the base point (duration, waist, focus, delay, density); `names` and `scale` give each
  parameter's name and the scale of z = x ./ scale in problem.jl.
- `free` are the indices of the parameters an optimizer moves: all but the delay, which stays at
  `x0[4]`. ForwardDiff's derivative in the delay does not give J's change over a finite step:
  at λ₀/32 it is 11 to 16 in d ln J/d ln delay where central differences and runs 2⁻¹⁰ apart
  give −0.3, and it changes sign from one grid to the next (WP-065's report).
"""
const TCASE = (a0_energy = 1.9, budgets = (1.9, 2.175, 2.4), dopant_fraction = 0.01, rise = (3.0, 5.0), fade = (12.0, 24.0),
               x0 = (6.0, 10.0, 80.0, 54.0, 0.072), names = ("duration", "waist", "focus", "delay", "density"),
               scale = (6.0, 10.0, 80.0, 54.0, 0.072), free = (1, 2, 3, 5))

"""
    pulse_energy(a0, duration, waist) -> W

The energy per unit length in z of the case's pulse ([`laser_energy`](@ref) of a `GaussianLaser`
polarized in the plane, the case's pulse): π·a₀²·w·τ/4·(1 + 1/(2w²)).
"""
pulse_energy(a0, duration, waist) = π * a0^2 * waist * duration / 4 * (1 + 1 / (2 * waist^2))

"""
The optimizer's box over the free parameters `TCASE.free` (duration, waist, focus, density), in
physical units, with their scales (WP-068): the duration from 4 to 9, the waist from 6 to 14, the
focus from 64 to 110 and the density from 0.05 to 0.10 n_c, so z = x ./ scale stays in [0.6, 1.5].
At the budget of a₀ = 1.9 at the base pulse, a₀ runs across it from about 1.3 (τ = 9, w = 14) to
about 3.0 (τ = 4, w = 6), past the plasma's own injection into the first bucket from a₀ ≈ 2.45 to
2.55 (WP-065).
"""
const TRAPPED_BOX = (lo = [4.0, 6.0, 64.0, 0.05], hi = [9.0, 14.0, 110.0, 0.10], scale = [TCASE.scale[i] for i in TCASE.free])

"""
    trapped_point(free) -> (duration, waist, focus, delay, density)

The case's five parameters from the four free ones (in `TCASE.free`'s order), the delay at the
base point's `TCASE.x0[4]` (every parameter outside `TCASE.free` at its base value), in the free parameters' type.
"""
function trapped_point(free)
    length(free) == length(TCASE.free) || throw(ArgumentError("trapped_point takes the $(length(TCASE.free)) free parameters, got $(length(free))"))
    T = promote_type(map(typeof, Tuple(free))...)
    x = Vector{T}(undef, length(TCASE.x0))
    for i in eachindex(x)
        k = findfirst(==(i), TCASE.free)
        x[i] = k === nothing ? T(TCASE.x0[i]) : free[k]
    end
    return Tuple(x)
end

"The energy budget of the pulse with `a0` at the base duration and waist."
budget(a0 = TCASE.a0_energy) = pulse_energy(a0, TCASE.x0[1], TCASE.x0[2])

"""
    a0_for_energy(duration, waist; energy) -> a₀

The a₀ at which a pulse of `duration` and `waist` carries `energy` per unit length in z, the
inverse of [`pulse_energy`](@ref): the fixed-energy coupling, through which ∂W/∂τ = ∂W/∂w = 0.
"""
a0_for_energy(duration, waist; energy = budget()) = sqrt(energy / (π * waist * duration / 4 * (1 + 1 / (2 * waist^2))))

"""
    trapped_capacity(ppl, ny) -> slots

The born electrons' slots for a run at `ppl` and `ny`: the dopant's macro-particles grow as
ppl²·ny (cells in x and y times the window's travel in cells), and the case at a₀ = 2.4 fills
24,880 slots at λ₀/16 and 204,790 at λ₀/32 with ny = 2 at t = 200, once the pulse has left
the dopant's section (x ≤ 132), so 40,000·ppl²·ny/256 keeps
a margin of about 1.5, and never fewer than `BCASE.capacity` (the runs of WP-064 keep their bits).
"""
trapped_capacity(ppl, ny) = max(BCASE.capacity, cld(40_000 * ppl^2 * ny, 256))

"""
    trapped_problem(duration, waist, focus, delay, density; energy, ppl, ny, capacity, kw...) -> EMProblem{2}

The case at the five parameters, with a₀ from [`a0_for_energy`](@ref) and the dopant at
`TCASE.dopant_fraction` of `density`, and [`trapped_capacity`](@ref) slots by default; the other
keywords as [`born_problem`](@ref).
"""
function trapped_problem(duration, waist, focus, delay, density; energy = budget(), ppl::Integer = BCASE.ppl, ny::Integer = 1,
                         capacity::Integer = trapped_capacity(ppl, ny), kw...)
    a0 = a0_for_energy(duration, waist; energy)
    # the fraction in density's own precision, so that a dual of Float32 stays one
    fraction = convert(typeof(float(ForwardDiff.value(density))), TCASE.dopant_fraction)
    return born_problem(a0, fraction * density; duration, waist, focus, delay, density, ppl, ny, capacity, kw...)
end

"""
The case's state after its run at the five parameters: the run of [`trapped_problem`](@ref)'s
problem (and its keywords) on `backend`, the particles reading the fields through `gather`
([`born_numerics`](@ref): `nothing`, or WP-068's `CompensatedGather()`); a birth without a free
slot is an error.
"""
trapped_state(duration, waist, focus, delay, density; ppl::Integer = BCASE.ppl, ny::Integer = 1, backend = CPU(), gather = nothing, kw...) =
    run_to_end(start_of(trapped_problem(duration, waist, focus, delay, density; ppl, ny, kw...); ppl, ny, backend, gather))

# The summand of a slot: weight × kinetic energy × position weight × trapping weight, zero for a free slot.
@inline function slot_trapped(status, c1, o1, u1, u2, u3, w, h1, lo, hi, ulo, uhi)
    x = (c1 - 1 + o1) * h1
    return ifelse(status == ALIVE, w * (sqrt(1 + u1 * u1 + u2 * u2 + u3 * u3) - 1) * smooth_rise(x, lo, hi) * smooth_rise(u1, ulo, uhi), zero(w))
end

# One species' sum, in slot order on the host (as `born_energy`, so that a run with duals gives
# the plain run's bits).
function species_trapped(s, h1, lo, hi, ulo, uhi)
    T = eltype(s.w)
    return foldl(+, Array(slot_trapped.(s.status, s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, h1, T(lo), T(hi), T(ulo), T(uhi))); init = zero(T))
end

function check_rise(rise)
    rise[2] > rise[1] || throw(ArgumentError("the trapping weight needs rise[2] > rise[1], got $rise"))
    return nothing
end

"""
    trapped_parts(st; fade = TCASE.fade, rise = TCASE.rise) -> (; born, plasma)

The two parts of [`trapped_energy`](@ref): the born electrons' (the species of
[`ionization_electrons`](@ref)) and the plasma's (every species of the state; in this case the
electrons, the ions being a fixed background).
"""
function trapped_parts(st; fade = TCASE.fade, rise = TCASE.rise)
    check_fade(st, fade)
    check_rise(rise)
    h = cellsize(st.grid)
    a = h[1] * h[2]
    born = a * species_trapped(ionization_electrons(st), h[1], fade[1], fade[2], rise[1], rise[2])
    plasma = a * foldl(+, map(s -> species_trapped(s, h[1], fade[1], fade[2], rise[1], rise[2]), st.species))
    return (; born, plasma)
end

"""
    trapped_energy(st; fade = TCASE.fade, rise = TCASE.rise) -> J

The objective: the kinetic energy of every trapped electron ahead of the window's back,

    J = Δx·Δy · Σ_i w_i·(γ_i − 1)·S(x_i)·R(u_x,i),

over the alive electrons of the plasma's species and of the ionization's, in
n_c·mₑc²·(c/ω₀)² (per unit length along z). `S` is the position weight
([`smooth_rise`](@ref) of the window x from `fade[1]` to `fade[2]`, zero on the back band), and
`R` the trapping weight, the same C² step in the forward momentum u_x from `rise[1]` to
`rise[2]`: an electron counts once it moves faster than the wake. Unlike WP-064's
[`born_energy`](@ref), which electrons count is decided by their momentum, not their label.
"""
function trapped_energy(st; kw...)
    p = trapped_parts(st; kw...)
    return p.born + p.plasma
end

"The objective at the five parameters (keywords of [`trapped_problem`](@ref), `fade` and `rise`)."
trapped_objective(duration, waist, focus, delay, density; fade = TCASE.fade, rise = TCASE.rise, kw...) =
    trapped_energy(trapped_state(duration, waist, focus, delay, density; kw...); fade, rise)

# The alive particles of a species with their window x, u_x, kinetic energy and weight, as plain numbers.
function species_arrays(s, h1)
    f(a) = Float64.(ForwardDiff.value.(Array(a)))
    alive = Array(s.status) .== ALIVE
    x = (Array(s.cell[1])[alive] .- 1 .+ f(s.offset[1])[alive]) .* h1
    u = map(a -> f(a)[alive], s.u)
    return (; x, ux = u[1], energy = sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1, w = f(s.w)[alive])
end

"""
    trapped_summary(st; fade = TCASE.fade, rise = TCASE.rise) -> NamedTuple

Plain numbers for a run's tables: `J` and its parts `J_born` and `J_plasma`; the counted weights
Σ w·S·R of each (`weight_born`, `weight_plasma`, times Δx·Δy) and their mean energies (mₑc²); the
weight of each with u_x above `rise[2]` over the whole window ahead of the back band
(`fast_born_window`, `fast_plasma_window`), which shows trapped charge the position weight leaves
out; the highest u_x of each; the dark current, the plasma's electrons with u_x above `rise[2]`
between the back band's end and `fade[1]` (`dark_weight`, and `dark_energy` in J's units); and the
number of alive particles. The dark region starts at the back band's end, which a longer window
(`extend`, as under a power cap) pushes back, so `dark_weight` and `dark_energy` of runs with
different `extend` count different regions and are not comparable.
"""
function trapped_summary(st; fade = TCASE.fade, rise = TCASE.rise)
    parts = trapped_parts(st; fade, rise)
    h = Float64.(ForwardDiff.value.(cellsize(st.grid)))
    area = h[1] * h[2]
    band = back_band_end(st)
    counted(q) = area * sum(q.w .* smooth_rise.(q.x, fade[1], fade[2]) .* smooth_rise.(q.ux, rise[1], rise[2]); init = 0.0)
    window(q) = area * sum(q.w[(q.ux .> rise[2]) .& (q.x .>= band)]; init = 0.0)
    dark(q, v) = area * sum((q.w .* v)[(q.ux .> rise[2]) .& (q.x .>= band) .& (q.x .< fade[1])]; init = 0.0)
    b = species_arrays(ionization_electrons(st), h[1])
    ps = [species_arrays(s, h[1]) for s in st.species]
    Jb, Jp = Float64(ForwardDiff.value(parts.born)), Float64(ForwardDiff.value(parts.plasma))
    wb, wp = counted(b), sum(counted, ps)
    return (; J = Jb + Jp, J_born = Jb, J_plasma = Jp, weight_born = wb, weight_plasma = wp,
            mean_energy_born = wb > 0 ? Jb / wb : 0.0, mean_energy_plasma = wp > 0 ? Jp / wp : 0.0,
            fast_born_window = window(b), fast_plasma_window = sum(window, ps),
            dark_weight = sum(q -> dark(q, 1.0), ps), dark_energy = sum(q -> dark(q, q.energy), ps),
            ux_max_born = maximum(b.ux; init = 0.0), ux_max_plasma = maximum(q -> maximum(q.ux; init = 0.0), ps),
            particles = length(b.w) + sum(q -> length(q.w), ps))
end
