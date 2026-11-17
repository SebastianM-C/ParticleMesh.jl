# The WP-064 case: a small 2D LWFA run whose pulse ionizes the K shell of a nitrogen dopant in a
# short section (WP-059), so that electrons are born inside the wake and some are trapped, with the
# energy those born electrons carry in the first bucket as the objective. The plasma's own
# electrons are another species and are never read by the objective: which electrons count is a
# label, not an energy cut.
using ParticleMesh, KernelAbstractions
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_back_band
using ParticleMeshElectromagnetic.ParticleMeshCore: ALIVE
using ForwardDiff: ForwardDiff
include(joinpath(@__DIR__, "..", "ionization", "atomic.jl"))

value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)

"""
Fixed parameters of the case (A1 units).

- The box is 14 λ₀ long (`ppl` cells per λ₀ along x) and 128 cells of Δy = π/4 wide, a moving
  window with PMLs at the open y edges and at the back, as the run of the ionization explainer
  page: the laser (waist 10, duration 6, focus at x = 80) enters a plasma of density 0.072 n_c
  behind a ramp at x = 64 to 72.
- The dopant is nitrogen at 1 % of the plasma density on the 36 rows around the axis, in the
  section x = 76 to 100 (ramps of 4), with the two K-shell levels N⁵⁺ → N⁶⁺ → N⁷⁺ at 0.8 µm. The
  L shell is left out: the pulse's foot ionizes it completely, far ahead of the wake, and its
  electrons are part of the plasma's density. ADK rates with no barrier-suppression correction
  and no ionization current (WP-059), so the charge injected is not a physical prediction.
- `fade = (lo, hi)` is the objective's position weight in window x (c/ω₀): 0 behind `lo`, 1 ahead
  of `hi`. At the read-out of the λ₀/16 run at a₀ = 1.65 the pulse's peak is at window x ≈ 47 and the
  first bucket lies between the zero crossings of E_x at x ≈ 27 and 40 (`born2d.jl` measures both),
  so the weight is 1 over the bucket; the back band ends at x ≈ 3.5 at λ₀/16.
- `u_fast` is the momentum above which a born electron is listed as trapped in the run's tables;
  the objective does not use it.
"""
const BCASE = (waist = 10.0, delay = 54.0, focus = 80.0, vacuum = 64.0, ramp = 8.0, start = 108.0, a0 = 2.2,
               duration = 6.0, density = 0.072, ppl = 32, length_wavelengths = 14, rows = 128, dy = π / 4, ppc = (2, 2),
               t_end = 300.0, pml_y = 12, pml_x_wavelengths = 0.25,
               dopant = 0.00072, section = (start = 76.0, stop = 100.0, ramp = 4.0), half_rows = 18, cutoff = 24.0,
               every = 1, capacity = 400_000, wavelength = 0.8e-6, fade = (12.0, 24.0), u_fast = 5.0)

"""
    born_problem(a0, dopant; t_end, ppl, every, capacity, section) -> EMProblem{2}

The case at a normalized vector potential `a0` and a dopant density `dopant` (n_c); the element
type follows the arguments. `ppl` cells per λ₀ along x at the same box and the same Δy (it must be
a multiple of 8), `t_end` the read-out time.
"""
function born_problem(a0 = BCASE.a0, dopant = BCASE.dopant; t_end = BCASE.t_end, ppl::Integer = BCASE.ppl, every::Integer = BCASE.every,
                      capacity::Integer = BCASE.capacity, section = BCASE.section, duration = BCASE.duration, density = BCASE.density)
    ppl % 8 == 0 || throw(ArgumentError("ppl must be a multiple of 8 to keep Δy = π/4, got $ppl"))
    T = promote_type(typeof(a0), typeof(dopant), typeof(duration), typeof(density), Float64)
    z3 = (zero(T), zero(T), zero(T))
    h = 2 * T(π) / ppl
    hy = (ppl ÷ 8) * h
    cells = (BCASE.length_wavelengths * ppl, BCASE.rows)
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    nsteps = ceil(Int, t_end / Float64(value_of(dt)))
    hy64 = Float64(value_of(hy))
    yc = cells[2] * hy64 / 2
    pulse = GaussianLaser(; a0 = T(a0), waist = T(BCASE.waist), duration = T(duration), delay = T(BCASE.delay), focus = T(BCASE.focus),
                          position = max(12, 192 ÷ ppl) * (2 * T(π) / 32), center = cells[2] * hy / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(density), zero(T), (1, 0), BCASE.ppc, z3, z3, 1)
    levels = (adk_level(T, :N, 6; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff),
              adk_level(T, :N, 7; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff))
    ion = FieldIonization(; density = T(dopant), levels, capacity, every, ppc = (1, 1),
                          rows = (yc - BCASE.half_rows * hy64, yc + BCASE.half_rows * hy64),
                          section = DopantSection(; start = T(section.start), stop = T(section.stop), ramp = T(section.ramp)))
    return EMProblem("born2d", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :open), window = MovingWindow(; start = T(BCASE.start)),
                     profile = DensityRamp(; vacuum = T(BCASE.vacuum), ramp = T(BCASE.ramp)), ionization = ion)
end

"The case's numerics: Reference mode, PMLs at the open y edges and at the window's back (a quarter wavelength, at least 4 cells; the antenna is 12 cells of λ₀/32 from the back edge, at least 6 cells)."
born_numerics(ppl::Integer = BCASE.ppl) =
    EMNumerics(; mode = Reference(), absorber = PML(; cells = BCASE.pml_y), back_absorber = PML(; cells = max(4, round(Int, BCASE.pml_x_wavelengths * ppl))))

"The case's state at its start and its parameters; `ppl` and the rest as [`born_problem`](@ref)."
function born_start(a0 = BCASE.a0, dopant = BCASE.dopant; ppl::Integer = BCASE.ppl, kw...)
    prob = born_problem(a0, dopant; ppl, kw...)
    st = EMState(prob, born_numerics(ppl); record = ())
    p = params(prob)
    initialize!(st, p)
    return st, p, prob
end

"The case's state after its run. A birth that found no free slot is an error: the capacity was too small."
function born_state(a0 = BCASE.a0, dopant = BCASE.dopant; kw...)
    st, p, prob = born_start(a0, dopant; kw...)
    run!(st, p, prob.nsteps)
    ionization_lost(st) == 0 || error("$(ionization_lost(st)) births found no free slot: raise `capacity`")
    return st
end

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
    band = window_back_band(st) * Float64(value_of(cellsize(st.grid)[1]))
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
on the parameters (#370, `questions/wp-059-field-ionization.md` §3). The sum is taken in slot
order over the device arrays, so it runs on any backend.
"""
function born_energy(st; fade = BCASE.fade)
    check_fade(st, fade)
    e = ionization_electrons(st)
    h = cellsize(st.grid)
    T = eltype(e.w)
    lo, hi = T(fade[1]), T(fade[2])
    return h[1] * h[2] * sum(slot_energy.(e.status, e.cell[1], e.offset[1], e.u[1], e.u[2], e.u[3], e.w, h[1], lo, hi))
end

"The objective of the case at `a0` and `dopant` (the keywords of [`born_problem`](@ref) and `fade`)."
born_objective(a0 = BCASE.a0, dopant = BCASE.dopant; fade = BCASE.fade, kw...) = born_energy(born_state(a0, dopant; kw...); fade)

"""
    plasma_peak_energy(st; fade = BCASE.fade) -> γ − 1

The guard that goes with the objective: the highest kinetic energy (mₑc²) of an alive electron of
non-zero weight of the plasma's species ahead of `fade[1]`, a plain number. In the laser a plasma electron quivers up
to about a₀²/2; a value far above that says plasma electrons were injected into the wake
themselves, energy the objective does not see.
"""
function plasma_peak_energy(st; fade = BCASE.fade)
    check_fade(st, fade)
    h1 = Float64(value_of(cellsize(st.grid)[1]))
    peak = 0.0
    for s in st.species
        x, g, _ = plasma_arrays(s, h1)
        keep = x .>= fade[1]
        any(keep) && (peak = max(peak, maximum(g[keep])))
    end
    return peak
end

# The alive particles of non-zero weight of a plasma species as plain numbers: window x, kinetic
# energy, weight. The particles loaded in the vacuum and at the ramp's foot have the weight zero
# (the profile enters through weights, A3) and move as test particles; they carry nothing.
function plasma_arrays(s, h1)
    alive = (Array(s.status) .== ALIVE) .& (value_of.(Array(s.w)) .> 0)
    f(a) = Float64.(value_of.(Array(a)))[alive]
    u = map(f, s.u)
    return (Array(s.cell[1])[alive] .- 1 .+ f(s.offset[1])) .* h1, sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1, f(s.w)
end

"""
    plasma_fast_weight(st; fade = BCASE.fade, energy = BCASE.u_fast) -> weight

The weight (times Δx·Δy) of the plasma's electrons ahead of `fade[1]` whose kinetic energy exceeds
`energy` (mₑc²): the charge the wake has injected from the plasma itself.
"""
function plasma_fast_weight(st; fade = BCASE.fade, energy = BCASE.u_fast)
    check_fade(st, fade)
    h = Float64.(value_of.(cellsize(st.grid)))
    return h[1] * h[2] * sum(st.species; init = 0.0) do s
        x, g, w = plasma_arrays(s, h[1])
        sum(w[(x .>= fade[1]) .& (g .> energy)]; init = 0.0)
    end
end

"""
    born_summary(st; fade = BCASE.fade, u_fast = BCASE.u_fast) -> NamedTuple

Plain numbers about the born electrons for a run's tables: `born`, the weight the atoms have
released as electrons, summed over the atoms still in the window (Σ w·emitted); `alive`, the weight still in the
window; `counted`, Σ w·S(x); `fast`, the weight with u_x > `u_fast`; `fast_counted`, the same times
S(x); `u_max`; `energy`, the objective; `mean_energy`, energy over counted weight (mₑc²); `plasma_peak`, [`plasma_peak_energy`](@ref); and `plasma_fast`, [`plasma_fast_weight`](@ref) above `u_fast`. Weights are times Δx·Δy (n_c·(c/ω₀)²).
"""
function born_summary(st; fade = BCASE.fade, u_fast = BCASE.u_fast)
    h = Float64.(value_of.(cellsize(st.grid)))
    area = h[1] * h[2]
    e = ionization_electrons(st)
    alive = Array(e.status) .== ALIVE
    f(a) = Float64.(value_of.(Array(a)))[alive]
    x = (Array(e.cell[1])[alive] .- 1 .+ f(e.offset[1])) .* h[1]
    w, ux = f(e.w), f(e.u[1])
    S = smooth_rise.(x, fade[1], fade[2])
    a = ionization_state(st)
    born = sum(Float64.(value_of.(a.w)) .* Float64.(value_of.(a.emitted)))
    energy = Float64(value_of(born_energy(st; fade)))
    counted = sum(w .* S)
    return (; born = area * born, alive = area * sum(w), counted = area * counted, fast = area * sum(w[ux .> u_fast]),
            fast_counted = area * sum((w .* S)[ux .> u_fast]), u_max = isempty(ux) ? 0.0 : maximum(ux), energy,
            mean_energy = counted > 0 ? energy / (area * counted) : 0.0, plasma_peak = plasma_peak_energy(st; fade),
            plasma_fast = plasma_fast_weight(st; fade, energy = u_fast), particles = count(alive))
end
