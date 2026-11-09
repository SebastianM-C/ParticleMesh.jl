# The WP-045 optimization case: WP-043's small 2D LWFA run (lib/ParticleMeshElectromagnetic/test/lwfa_gradient.jl)
# at a denser plasma and a weaker pulse, with the duration and the focus free.
using ParticleMesh, KernelAbstractions
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_back_band
using ForwardDiff: ForwardDiff

value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)

"Fixed parameters of the case (A1 units)."
const CASE = (a0 = 1.0, waist = 10.0, density = 0.1, vacuum = 64.0, ramp = 8.0, delay = 54.0, start = 108.0,
              cells = (224, 32), nsteps = 512, beta = 200.0)

"""
    wake_problem(duration, focus; nsteps) -> EMProblem{2}

The case at a pulse duration (1/e field half width) and focus; the element type follows the arguments.
The window starts at 72 = delay + 6·6, so durations up to 6.
"""
function wake_problem(duration, focus; nsteps = CASE.nsteps)
    T = promote_type(typeof(duration), typeof(focus))
    z3 = (zero(T), zero(T), zero(T))
    cells = CASE.cells
    h = 2 * T(π) / 16
    hy = 4h
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    pulse = GaussianLaser(; a0 = T(CASE.a0), waist = T(CASE.waist), duration = T(duration), delay = T(CASE.delay), focus = T(focus),
                          position = 6h, center = cells[2] * hy / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(CASE.density), zero(T), (1, 0), (2, 2), z3, z3, 1)
    return EMProblem("wake2d", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :periodic), window = MovingWindow(; start = T(CASE.start)),
                     profile = DensityRamp(; vacuum = T(CASE.vacuum), ramp = T(CASE.ramp)))
end

function wake_state(duration, focus; nsteps = CASE.nsteps)
    prob = wake_problem(duration, focus; nsteps)
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

function soft_maximum(v, β)
    m = maximum(value_of, v)
    return m + log(sum(e -> exp(β * (e - m)), v)) / β
end

"The objective: the wake amplitude (soft maximum of the on-axis E_x past the back band) at the last step, in units of E₀ = √n₀."
wake_amplitude(duration, focus; kw...) = soft_maximum(axis_Ex(wake_state(duration, focus; kw...)), CASE.beta) / sqrt(CASE.density)
