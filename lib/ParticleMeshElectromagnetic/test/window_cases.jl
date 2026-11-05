# WP-028: the moving window's problems, used by window.jl.

using KernelAbstractions: get_backend
using ParticleMeshElectromagnetic: injection_column, load_window_columns!, velocity_momentum, window_density_params

const ZERO3(T) = (zero(T), zero(T), zero(T))

"""
    window_plasma(T; cells, ppc, nsteps, start, vacuum, ramp, drift, thermal, density, seed) -> EMProblem{2}

A plasma entering a moving window through its front: one electron species of the
[`DensityRamp`](@ref) profile (vacuum, ramp, plateau) against its immobile background, cold and at
rest by default, Δx = 1/2, Δt at 0.9 of the Courant limit, so the window moves about 0.64 cells
per step. `ramp` may be a dual (the A3 parameter).
"""
function window_plasma(::Type{T}; cells = (24, 6), ppc = (2, 2), nsteps = 40, start = 0, vacuum = 4, ramp = 4,
                       drift = ZERO3(T), thermal = ZERO3(T), density = one(T), seed = 5, open_y = false) where {T}
    L = (T(cells[1]) / 2, T(cells[2]) / 2)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(density), zero(T), (1, 0), ppc, T.(drift), T.(thermal), 1)
    return EMProblem("window-plasma", L, cells, T(dt), nsteps, (e,), seed, (1, 0); boundaries = (:window, open_y ? :open : :periodic),
                     window = MovingWindow(; start = T(start)), profile = DensityRamp(; vacuum = T(vacuum), ramp = ramp isa T ? ramp : T(ramp)))
end

"""
    window_lwfa(T; nsteps, a0, ramp, cells) -> EMProblem{2}

A small laser–plasma run in the window: 8 cells per laser wavelength (Δx = π/4), 64 × 16
cells, Δt at 0.95 of the Courant limit; a Gaussian pulse (a₀ = 1, waist 6, duration 3,
delay 18) launched by an antenna at x = 6, which stays in the lab frame; a cold electron plasma
of 0.02 n_c behind a vacuum of 24 c/ω₀ and a ramp of 8; the window starts at t = 33, when the
antenna has finished (six durations after the delay), and the box (50 c/ω₀) holds the pulse to
six durations ahead of its peak, so the field at the front stays below Float64 roundoff and the
plasma there at rest. `plasma = false` leaves the vacuum pulse alone.
"""
function window_lwfa(::Type{T}; nsteps = 200, a0 = 1, ramp = 8, cells = (64, 16), seed = 1, plasma = true) where {T}
    h = 2 * T(π) / 8
    dt = T(95) / 100 * h / sqrt(T(2))
    laser = GaussianLaser(; a0 = T(a0), waist = T(6), duration = T(3), delay = T(18), focus = T(6), position = T(6),
                          center = cells[2] * h / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(2) / 100, zero(T), (1, 0), (2, 2), ZERO3(T), ZERO3(T), 1)
    return EMProblem("window-lwfa", (cells[1] * h, cells[2] * h), cells, dt, nsteps, plasma ? (e,) : (), seed, (1, 0); lasers = (laser,),
                     boundaries = (:window, :periodic), window = MovingWindow(; start = T(33)),
                     profile = DensityRamp(; vacuum = T(24), ramp = ramp isa T ? ramp : T(ramp)))
end

"The initialized state of a window problem."
function window_state(prob; mode = Reference(), workgroupsize = nothing, headroom = 1 / 4, absorber = nothing, be = CPU())
    st = EMState(prob, EMNumerics(; mode, workgroupsize, headroom, absorber); backend = be)
    initialize!(st, prob)
    return st
end

"The alive particles of `s` as a sorted vector of (cell₁, cell₂, bits of δ₁, δ₂, u₁, u₂, u₃, w)."
function particle_set(s)
    alive = Array(s.status) .== ParticleMeshCore.ALIVE
    c1, c2 = Array(s.cell[1])[alive], Array(s.cell[2])[alive]
    rest = map(a -> Array(a)[alive], (s.offset[1], s.offset[2], s.u[1], s.u[2], s.u[3], s.w))
    return sort([(c1[i], c2[i], map(a -> components(a[i]), rest)...) for i in eachindex(c1)])
end

"The fresh cold load of the window's plasma columns of species 1 at position `s` (a scratch species)."
function fresh_window_load(st, s_win; species = 1, thermal = false)
    T = eltype(st.ρ)
    load = st.aux.window.loads[species]
    thermal || (load = (; load..., vdrift = ZERO3(T), vth = ZERO3(T)))
    sp = st.species[species]
    n = ncells(st.grid)
    be = get_backend(st.ρ)
    sc = Species(be, T, prod(n) * prod(load.ppc); q = sp.q, m = sp.m, N = 2)
    load_window_columns!(sc, st.grid, load, KernelAbstractions.zeros(be, Int32, 1), 1, injection_column(st.aux.window), s_win)
    return sc
end

"""
    window_gauss(st, s) -> (; res, scale)

Gauss's law of the window, `D⁻ₓEx + D⁻_yEy − (ρ − ρ₀(s))` with ρ from [`window_charge!`](@ref)
and the background's ρ₀ from [`window_background!`](@ref), per interior node; `scale` is T-360's
`(n₁ + n₂)·max|ρ|`. Returns the matrix of residuals (absmax over the partials of a dual).
"""
function window_gauss(st, s_win)
    T = eltype(st.ρ)
    grid = st.grid
    be = get_backend(st.ρ)
    ρ, ρ0 = field(be, T, grid), field(be, T, grid)
    window_charge!(ρ, st)
    window_background!(ρ0, st, s_win)
    ρ, ρ0 = Array(ρ), Array(ρ0)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    Ex, Ey = Array(st.fields.Ex), Array(st.fields.Ey)
    res = [absmax((Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2] - (ρ[i, j] - ρ0[i, j]))
           for i in (g + 1):(g + n[1]), j in (g + 1):(g + n[2])]
    return (; res, scale = sum(n) * maximum(absmax, ρ[(g + 1):(g + n[1]), (g + 1):(g + n[2])]))
end

"A window problem's static twin: the same box, `cells₁` columns, with a window that never moves."
function static_twin(prob::EMProblem{2,T}, n1) where {T}
    h = prob.length[1] / prob.cells[1]
    return EMProblem("static-twin", (n1 * h, prob.length[2]), (n1, prob.cells[2]), prob.dt, prob.nsteps, prob.species,
                     prob.seed, prob.kmode; lasers = prob.lasers, boundaries = prob.boundaries,
                     window = MovingWindow(; start = T(Inf)), profile = prob.profile)
end

"T-474: a Reference run of `window_lwfa` (120 steps, about 50 shifts) per element type, one digest each."
function window_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        prob = window_lwfa(T; nsteps = 120)
        st = window_state(prob; workgroupsize, be)
        run!(st, params(prob), prob.nsteps)
        push!(out, (T, state_digest_2d(st)))
    end
    return out
end
