# WP-059: the field ionization's problems, used by ionization.jl and by the determinism worker
# (T-775 at 1 and 4 threads). After em2d_cases.jl (`zero3`, `bits`).

using KernelAbstractions: get_backend
using ParticleMeshElectromagnetic: ionization_frame, RHO_STAGGER_2D

"""
    ion_in_E(T; E0, B0, levels, density, ppc, every, rows, section, capacity, nsteps, courant, plasma, box, cells, boundaries) -> EMProblem{2}

A dopant in a uniform background field on a periodic box of 16 × 8 cells of 1/2 (`cells` on the
lengths `box`; `boundaries = (:periodic, :open)` opens the y edges), Δt at `courant` (0.9) of the
Courant limit: `levels` (one level of prefactor 1/2, field 3 and power 1 by default,
whose rate in |E| = 1/2 is 3·e⁻² ≈ 0.406) at the density `density` on `ppc` points per cell of the
rows `rows`, born every `every` steps. Without `plasma` the only species has zero density, and with
the prescribed fields of `NoCurrent()` the field at every atom is `E0` for the whole run; with
`plasma` it is a thermal electron plasma (density 1, v_th = 0.1, a drift and an oblique
perturbation, so its particles cross cells and both seams) for the runs with the charge-conserving
current. `capacity` defaults to one slot per atom and birth step, and one birth step more.
"""
function ion_in_E(::Type{T}; E0 = (one(T) / 2, zero(T), zero(T)), B0 = zero3(T), levels = (ADKLevel(one(T) / 2, T(3), one(T)),),
                  density = one(T) / 100, ppc = (1, 1), every = 1, rows = nothing, section = nothing, capacity = nothing, nsteps = 20,
                  courant = 9 // 10, plasma = false, box = (8, 4), cells = (16, 8), boundaries = (:periodic, :periodic)) where {T}
    L = (T(box[1]), T(box[2]))
    dt = T(courant) * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2)))
    tenth = one(T) / 10
    e = plasma ? SpeciesDescription{T}(:electron, -one(T), one(T), one(T), tenth, (2, 1), (2, 2), (3tenth, -2tenth, tenth), (tenth, tenth, tenth), 1) :
        SpeciesDescription{T}(:electron, -one(T), one(T), zero(T), zero(T), (1, 0), (1, 1), zero3(T), zero3(T), 1)
    cap = capacity === nothing ? prod(cells) * prod(ppc) * (nsteps ÷ every + 2) : capacity
    ion = FieldIonization(; density, levels, capacity = cap, section, ppc, every, rows)
    return EMProblem("ion-in-E", L, cells, dt, nsteps, (e,), 3, (2, 1); E0, B0, ionization = ion, boundaries)
end

"""
    ion_window(T; nsteps, a0, density, prefactor, start, stop, ramp, every, ppc, rows, capacity, plasma, cutoff, levels) -> EMProblem{2}

A small ionization-injection run: `window_lwfa`'s box and pulse (window_cases.jl: 8 cells per laser
wavelength, 64 × 16 cells, a Gaussian pulse of a₀ = 1, waist 6 and duration 3 from an antenna at
x = 6, a cold plasma of 0.02 n_c behind a vacuum of 24 c/ω₀ and a ramp of 8, the window moving
from t = 33) with a dopant of the density `density` (0.005 n_c) in the section that rises over
`ramp` (3) from `start` (28) and falls from `stop` (34), on the 8 rows about the axis, with one
level of prefactor `prefactor` (1/5), field 3 and power 1: its rate at |E| = 1 is 0.6·e⁻¹ ≈ 0.22,
and it is exactly zero below |E| = 1/`cutoff` (1/64 by default, the level's cutoff); `levels`
replaces it. At the default
110 steps (t ≈ 58) the pulse has passed the section and the section is still inside the window; by
150 steps it has left through the back edge. `capacity` (4096) holds every birth of the run; 2048
hold the 110 steps' (they need 1912) and not those of 130 steps. `a0`, `density`, `prefactor` and
the section's three numbers may be duals. The box is too coarse and the run too short for a
physical result (`every = 2` at 8 cells per wavelength is a sixth of a laser period between
births): its born charge and energy are quantities for tests.
"""
function ion_window(::Type{T}; nsteps = 110, a0 = 1, density = T(1) / 200, prefactor = T(1) / 5, start = 28, stop = 34, ramp = 3,
                    every = 2, ppc = (1, 1), rows = (3.2, 9.4), capacity = 4096, plasma = true, cutoff = 64, cells = (64, 16), seed = 1,
                    levels = (ADKLevel(T(prefactor), T(3), one(T); cutoff = T(cutoff)),)) where {T}
    h = 2 * T(π) / 8
    dt = T(95) / 100 * h / sqrt(T(2))
    pulse = GaussianLaser(; a0 = T(a0), waist = T(6), duration = T(3), delay = T(18), focus = T(6), position = T(6), center = cells[2] * h / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(2) / 100, zero(T), (1, 0), (2, 2), zero3(T), zero3(T), 1)
    ion = FieldIonization(; density = T(density), levels, capacity, ppc, every, rows,
                          section = DopantSection(; start = T(start), stop = T(stop), ramp = T(ramp)))
    return EMProblem("ion-window", (cells[1] * h, cells[2] * h), cells, dt, nsteps, plasma ? (e,) : (), seed, (1, 0); lasers = (pulse,),
                     boundaries = (:window, :periodic), window = MovingWindow(; start = T(33)),
                     profile = DensityRamp(; vacuum = T(24), ramp = T(8)), ionization = ion)
end

"The initialized state of an ionization problem with the charge-conserving current (`current = NoCurrent()` for prescribed fields; `absorber` for an open edge)."
function ion_state(prob; mode = Reference(), workgroupsize = nothing, current = Esirkepov(), absorber = nothing, be = CPU())
    st = EMState(prob, EMNumerics(; mode, workgroupsize, current, absorber); backend = be)
    initialize!(st, prob)
    return st
end

"A hash of the bits of everything a state evolves (`evolved`: the species, the fields, the window's counters, the atoms and their electrons), of `K` and of the histories."
function ion_digest(st::EMState{2})
    h = UInt64(0)
    for a in (evolved(st)..., st.K, values(st.recorder.hist)...)
        h = hash(bits(a), h)
    end
    return h
end

"T-775: a Reference run of `ion_window` (150 steps: the births, about 55 shifts, atoms and electrons leaving through the back edge) per element type, one digest each."
function ion_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        prob = ion_window(T; nsteps = 150)
        st = ion_state(prob; workgroupsize, be)
        run!(st, params(prob), prob.nsteps)
        push!(out, (T, ion_digest(st)))
    end
    return out
end
