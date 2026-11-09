# The WP-045 optimization case: WP-043's small 2D LWFA run (lib/ParticleMeshElectromagnetic/test/lwfa_gradient.jl)
# with a weaker, wider pulse in a wider box, the duration and the focus free.
using ParticleMesh, KernelAbstractions
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_back_band
using ForwardDiff: ForwardDiff

value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)

"""
Fixed parameters of the case (A1 units). The box is 224 × 64 cells of Δx = λ₀/16 and Δy = 4Δx
(88 × 101 c/ω₀, periodic in y: 5 waists either side of the axis; at 32 rows the beam's radius
at the antenna reached the half-box and set the focus optimum).
"""
const CASE = (a0 = 1.0, waist = 10.0, density = 0.1, vacuum = 64.0, ramp = 8.0, delay = 54.0, start = 108.0,
              cells = (224, 64), nsteps = 512, ppc = (2, 2), dy_cells = 4)

"""
    wake_problem(duration, focus; nsteps, ppl, dy_cells, cells, ppc, a0, waist) -> EMProblem{2}

The case at a pulse duration (1/e field half width) and focus; the element type follows the
arguments. The pulse's delay is 54 and the window starts at 108 = delay + 6·9, after the antenna
has finished, for durations up to 9. `ppl` (cells per λ₀ along x, Δx = λ₀/ppl), `dy_cells` (Δy in
units of Δx), `cells` (by default the 14 λ₀ × 16 λ₀ box at that resolution) and `ppc` are there for
resolution checks; the antenna stays at x = 3λ₀/8 at every `ppl` divisible by 8. `a0` and `waist`
override the case's pulse (WP-051: low-a₀ and plane-wave comparisons).
"""
function wake_problem(duration, focus; nsteps = CASE.nsteps, ppl = 16, dy_cells = CASE.dy_cells,
                      cells = (14ppl, 16ppl ÷ dy_cells), ppc = CASE.ppc, a0 = CASE.a0, waist = CASE.waist)
    T = promote_type(typeof(duration), typeof(focus))
    z3 = (zero(T), zero(T), zero(T))
    h = 2 * T(π) / ppl
    hy = dy_cells * h
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    pulse = GaussianLaser(; a0 = T(a0), waist = T(waist), duration = T(duration), delay = T(CASE.delay), focus = T(focus),
                          position = 6 * (2 * T(π) / 16), center = cells[2] * hy / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(CASE.density), zero(T), (1, 0), ppc, z3, z3, 1)
    return EMProblem("wake2d", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :periodic), window = MovingWindow(; start = T(CASE.start)),
                     profile = DensityRamp(; vacuum = T(CASE.vacuum), ramp = T(CASE.ramp)))
end

function wake_state(duration, focus; kw...)
    prob = wake_problem(duration, focus; kw...)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    return st
end

"E_x on the axis's node row, interior columns past the window's back band (WP-044)."
function axis_Ex(st)
    g, n = guards(st.grid), ncells(st.grid)
    return Array(st.fields.Ex)[(g + 1 + window_back_band(st)):(g + n[1]), g + 1 + n[2] ÷ 2]
end

"""
The objective: the RMS wake amplitude √(2·mean E_x²) of the on-axis E_x past the back band at the
last step, in units of E₀ = √n₀ (for a sinusoid of amplitude A over whole periods it is A). It
replaces a soft maximum of E_x, whose optimum sat where two wake crests tie (the soft maximum's
bonus of log 2/β there outweighed the crests' own change) and where the hard maximum had a
local minimum.
"""
wake_amplitude(duration, focus; kw...) = (v = axis_Ex(wake_state(duration, focus; kw...)); sqrt(2 * sum(abs2, v) / length(v)) / sqrt(CASE.density))
