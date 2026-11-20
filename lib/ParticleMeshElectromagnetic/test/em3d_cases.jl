# WP-042: the 3D problems of em3d_step.jl and the bitwise digests that its tests and the
# subprocess worker (em3d_worker.jl, T-952 at 1 and 4 threads) share. After em2d_cases.jl
# (`zero3`).

"""
    plasma_step_3d(T; amplitude = 1/10, cells = (8, 7, 6), ppc = (2, 2, 2), nsteps = 20, seed = 7, vth = 1/10) -> EMProblem{3}

Two mobile species that cross cells and the periodic seams of every axis within a few steps,
`plasma_step_2d`'s plasma in 3D: thermal electrons (v_th = `vth` in every component, a
Maxwell–Jüttner load) with the drift (0.3, −0.2, 0.25) and an oblique perturbation of mode
(2, 1, −1), and positive particles of mass 4 (v_th = 0.05) drifting along (−0.2, 0.25, −0.15) with
the mode (1, −1, 1), on a box of 2π × 3π/2 × 5π/4 (three cell sizes), Δt at 0.9 of the Courant
limit. The electrons' `amplitude` may be a dual (T-955).
"""
function plasma_step_3d(::Type{T}; amplitude = one(T) / 10, cells = (8, 7, 6), ppc = (2, 2, 2), nsteps = 20, seed = 7,
                        vth = one(T) / 10) where {T}
    L = (2 * T(π), 3 * T(π) / 2, 5 * T(π) / 4)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(amplitude), (2, 1, -1), ppc, (3tenth, -2tenth, 5tenth / 2),
                              (T(vth), T(vth), T(vth)), 1)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1, 1), ppc, (-2tenth, 5tenth / 2, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2)
    return EMProblem("plasma-step-3d", L, cells, T(dt), nsteps, (e, p), seed, (2, 1, -1))
end

"""
    neutral_plasma_3d(T; nsteps = 20) -> EMProblem{3}

A quasi-neutral plasma whose two species' charges cancel to 10⁻³: drifting thermal electrons
(v_th = 0.1, drift (0.1, −0.05, 0.02), perturbation 10⁻³ of mode (1, 1, 1)) and ions of mass 100
(v_th = 0.01) at the same density, on 10 × 8 × 6 cells of a box of 2π × 3π/2 × 6π/5, Δt at 0.9
of the Courant limit. Its net charge is a thousandth of each species', so T-950 checks it on the
species' scale (the physics breaker's case).
"""
function neutral_plasma_3d(::Type{T}; nsteps = 20) where {T}
    L = (2 * T(π), 3 * T(π) / 2, 6 * T(π) / 5)
    cells = (10, 8, 6)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), one(T) / 1000, (1, 1, 1), (2, 2, 2), (tenth, -tenth / 2, tenth / 5),
                              (tenth, tenth, tenth), 1)
    i = SpeciesDescription{T}(:ion, one(T), 100 * one(T), one(T), zero(T), (1, 1, 1), (2, 2, 2), zero3(T),
                              (tenth / 10, tenth / 10, tenth / 10), 2)
    return EMProblem("neutral-plasma-3d", L, cells, T(dt), nsteps, (e, i), 11, (1, 1, 1))
end

"The 3D state of `prob` with the `current` (`Esirkepov()` by default), initialized."
function state_3d(prob; mode = Reference(), workgroupsize = nothing, shape = Quadratic, be = CPU(), current = Esirkepov(),
                  node_bins = true)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, shape, current, node_bins); backend = be)
    initialize!(st, prob)
    return st
end

"A hash of the bits of every particle array (every axis, extras included), field, `K` and history of a 3D state."
function state_digest_3d(st::EMState{3})
    h = UInt64(0)
    for s in st.species, a in (s.cell..., s.offset..., s.u..., s.w, s.status, s.count, values(s.extras)...)
        h = hash(bits(a), h)
    end
    for a in (values(st.fields)..., st.K, values(st.recorder.hist)...)
        h = hash(bits(a), h)
    end
    return h
end

"""
    em3d_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-952: a Reference run of [`plasma_step_3d`](@ref) with the `Esirkepov()` current, 12 steps, per
element type and shape order, and for order 2 with cell bins too (`node_bins = false`), one
digest each.
"""
function em3d_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES, (label, shape, node_bins) in (("order2", Quadratic, true), ("order2-cells", Quadratic, false), ("order1", Linear, true))
        prob = plasma_step_3d(T; nsteps = 12)
        st = state_3d(prob; workgroupsize, be, shape, node_bins)
        run!(st, params(prob), prob.nsteps)
        push!(out, (label, T, state_digest_3d(st)))
    end
    return out
end

# The cold beams of T-953 (em3d_step.jl): a 2D problem and the same problem in 3D, invariant along
# one axis. Both vary along both axes of the plane (mode (1, 1)) and drift in all three components,
# so every in-plane Esirkepov component and the third one carry a current.
const INV3_DRIFT = (1 // 5, 3 // 20, -1 // 8)
const INV3_CELLS = (12, 10)
const INV3_LENGTH = (2, 3 // 2)                    # in π

function beams_plane_2d(::Type{T}; ppc = (3, 3), nsteps = 20, amplitude = one(T) / 20, cells = INV3_CELLS,
                        len = INV3_LENGTH, boundaries = (:periodic, :periodic)) where {T}
    L = T.(len) .* T(π)
    h = L ./ cells
    v = T.(INV3_DRIFT)
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, T(amplitude), (1, 1), ppc, v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), (1, 1), ppc, .-v, zero3(T), 2))
    return EMProblem("beams-plane-2d", L, cells, inv3_dt(h), nsteps, sp, 1, (1, 1); boundaries)
end

# The step of both runs: 0.6 of the 3D Courant limit with the invariant axis' cells, which is
# below the 2D limit too.
inv3_dt(h) = 3 * inv(sqrt(sum(abs2, inv.(h)) + inv(2h[1])^2)) / 5

"""
    inv3_axes(axis) -> σ

The cyclic map of the 2D axes (and vector components) 1, 2, 3 onto the 3D ones for a 3D problem
invariant along `axis`: `σ[a]` is the 3D axis of the 2D axis `a`, `σ[3] = axis`. A cyclic
permutation is a rotation, so E, B and J map as vectors with no sign.
"""
inv3_axes(axis) = ntuple(a -> mod1(a + axis, 3), 3)

"""
    beams_3d(T, axis; ppc, nsteps, amplitude, cells, len, boundaries) -> EMProblem{3}

[`beams_plane_2d`](@ref) in 3D, invariant along `axis`: the plane's axes on the 3D axes
`inv3_axes(axis)`, the drift's components likewise, and the invariant axis with 6 cells (the
fewest the guards allow) of twice the plane's first cell size and 2 particles per cell; the
plane's `cells`, `len` and `boundaries` as the 2D problem's, the invariant axis periodic.
"""
function beams_3d(::Type{T}, axis; ppc = (3, 3), nsteps = 20, amplitude = one(T) / 20, cells = INV3_CELLS, len = INV3_LENGTH,
                  boundaries = (:periodic, :periodic)) where {T}
    σ = inv3_axes(axis)
    L2 = T.(len) .* T(π)
    h = L2 ./ cells
    put(x2, x3) = (y = Vector{Any}(undef, 3); y[σ[1]] = x2[1]; y[σ[2]] = x2[2]; y[σ[3]] = x3; Tuple(y))
    box = put(L2, 12h[1])
    n3 = put(cells, 6)
    p3 = put(ppc, 2)
    m = put((1, 1), 0)
    v2 = T.(INV3_DRIFT)
    v = put((v2[1], v2[2]), v2[3])
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, T(amplitude), m, p3, v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), m, p3, .-v, zero3(T), 2))
    return EMProblem("beams-3d", box, n3, inv3_dt(h), nsteps, sp, 1, m; boundaries = put(boundaries, :periodic))
end

# WP-072: the 3D plasma of em3d_open.jl, open along y, z or both, and the bitwise digests that its
# tests and the worker (em3d_worker.jl, T-1014 at 1 and 4 threads) share.

"""
    open_plasma_3d(T; boundaries = (:periodic, :open, :open), cells = (6, 14, 12), ppc = (2, 2, 2), nsteps = 30,
                   amplitude = 1/10, seed = 7) -> EMProblem{3}

`open_plasma_2d`'s two species in 3D: thermal electrons (v_th = 0.1) with the drift
(0.3, −0.25, 0.2) and the perturbation of mode (1, 1, −1), positive particles of mass 4 drifting
along (−0.2, 0.3, −0.15) with the mode (1, −1, 1), on a box of 2π × 3π × 3π (three cell sizes
on the default cells), Δt at 0.9 of the Courant limit, open on the axes `boundaries` says:
within the 30 steps about a third of the particles leave through the open faces, others cross
the periodic seams.
"""
function open_plasma_3d(::Type{T}; boundaries = (:periodic, :open, :open), cells = (6, 14, 12), ppc = (2, 2, 2), nsteps = 30,
                        amplitude = one(T) / 10, seed = 7) where {T}
    L = (2 * T(π), 3 * T(π), 3 * T(π))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(amplitude), (1, 1, -1), ppc, (3tenth, -5tenth / 2, 2tenth),
                              (tenth, tenth, tenth), 1)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1, 1), ppc, (-2tenth, 3tenth, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2)
    return EMProblem("open-plasma-3d", L, cells, T(dt), nsteps, (e, p), seed, (1, 1, -1); boundaries)
end

"The initialized 3D state of `prob` with the `absorber` on its open axes, in `mode`, with the `Esirkepov()` current."
function open_state_3d(prob, absorber; mode = Reference(), workgroupsize = nothing, be = CPU(), shape = Quadratic, solver = Yee())
    st = EMState(prob, EMNumerics(; mode, workgroupsize, shape, current = Esirkepov(), absorber, solver); backend = be)
    initialize!(st, prob)
    return st
end

const OPEN3_ABSORBERS = (("layer", AbsorbingLayer(; cells = 4, strength = 1)), ("pml", PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)))

"""
    open3d_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-1014: a Reference run of [`open_plasma_3d`](@ref) open on y and z (12 steps, particles
retired at all four open faces) per element type and absorber (the layer, the PML), one digest
of every particle array, field, convolution field and history each.
"""
function open3d_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES, (label, ab) in OPEN3_ABSORBERS
        prob = open_plasma_3d(T; nsteps = 12)
        st = open_state_3d(prob, ab; workgroupsize, be)
        run!(st, params(prob), prob.nsteps)
        h = state_digest_3d(st)
        for a in evolved(st)
            h = hash(bits(a), h)
        end
        push!(out, (label, T, h))
    end
    return out
end

# WP-073: the laser antenna at N = 3, the runs of em3d_antenna.jl and the digests its worker shares.

"""
    antenna3d_problem(T; ncx = 12, nct = 6, length = (60, 24, 24), waist = 4, duration = 5, position = 6, focus = 14,
                      polarization = 0, chirp = 0, a0 = 1, tend = 60, species = (), courant = 95//100, phase = 0,
                      envelope = AxisTime(), boundaries = periodic, center = middle) -> EMProblem{3}

`antenna_problem` in 3D: a box of `length` (c/ω₀) with `ncx` cells per laser wavelength 2π along x
and `nct` along y and z (a number, or one per axis), Δt at `courant` of the Yee limit, `round(tend/Δt)` steps, and one round
[`GaussianLaser`](@ref) whose envelope peaks at the antenna plane at t = 6·duration, its axis at
the middle of the box's cross-section unless `center` gives it. Every laser parameter may be a dual (A3).
"""
function antenna3d_problem(::Type{T}; ncx = 12, nct = 6, length = (60, 24, 24), waist = 4, duration = 5, position = 6, focus = 14,
                           polarization = 0, chirp = 0, a0 = 1, tend = 60, species = (), courant = 95 // 100, phase = 0,
                           envelope = AxisTime(), boundaries = (:periodic, :periodic, :periodic), center = nothing) where {T}
    ny, nz = nct isa Tuple ? nct : (nct, nct)
    h = (2π / ncx, 2π / ny, 2π / nz)
    n = ntuple(d -> round(Int, length[d] / h[d]), 3)
    L = ntuple(d -> T(n[d] * h[d]), 3)
    dt = T(courant) / sqrt(sum(d -> 1 / (L[d] / n[d])^2, 1:3))
    c = center === nothing ? (n[2] * h[2] / 2, n[3] * h[3] / 2) : center
    laser = GaussianLaser(; a0 = T(a0), waist = T(waist), duration = T(duration), delay = 6 * T(duration), focus = T(focus),
                          position = Float64(position), center = c, chirp = T(chirp), polarization = T(polarization), phase = T(phase),
                          envelope)
    return EMProblem("antenna-3d", L, n, dt, round(Int, tend / value_of(dt)), species, 1, (1, 0, 0); lasers = (laser,), boundaries)
end

"""
    antenna3d_run(prob; mode = Reference(), probes = (), snaps = (), be = CPU(), absorber = nothing, solver = Yee(),
                  antenna_solver = solver) -> (; st, prob, A, ip, shots)

[`antenna_run`](@ref) in 3D: `A[q]` the discrete-time Fourier transforms at ω = 1 of Ey and Ez on
the probe plane nearest to `probes[q]` (interior rows and layers), `shots` the fields after the
steps in `snaps`.
"""
function antenna3d_run(prob; mode = Reference(), probes = (), snaps = (), be = CPU(), absorber = nothing, workgroupsize = nothing,
                       solver = Yee(), antenna_solver = solver)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, current = Esirkepov(), absorber, solver); backend = be)
    st = replant_antennas(st, prob, antenna_solver)
    initialize!(st, prob)
    p = params(prob)
    (; grid, fields) = st
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    T = eltype(st.ρ)
    dt = prob.dt
    ip = [round(Int, x / value_of(h[1])) + 1 for x in probes]
    J, K = (g + 1):(g + n[2]), (g + 1):(g + n[3])
    A = [(Ey = zeros(Complex{T}, n[2], n[3]), Ez = zeros(Complex{T}, n[2], n[3])) for _ in ip]
    function accumulate!(k)
        w = cis(k * dt) * dt
        Ey, Ez = Array(fields.Ey), Array(fields.Ez)
        for (q, i) in enumerate(ip)
            A[q].Ey .+= w .* view(Ey, g + i, J, K)
            A[q].Ez .+= w .* view(Ez, g + i, J, K)
        end
    end
    accumulate!(0)
    shots = []
    for k in 1:prob.nsteps
        step!(st, p, k)
        isempty(ip) || accumulate!(k)
        k in snaps && push!(shots, (t = k * dt, f = map(Array, fields)))
    end
    return (; st, prob, A, ip, shots)
end

"""
    antenna3d_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-1027: Reference runs of a laser launched into a drifting plasma with y and z open (the PML),
50 steps, and of the same laser polarized at 0.6 rad in a periodic box, per element type, one
digest of every particle array, field, convolution field and history each.
"""
function antenna3d_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES, (label, bd, θ) in (("open", (:periodic, :open, :open), 0), ("periodic", (:periodic, :periodic, :periodic), 3 // 5))
        tenth = one(T) / 10
        sp = (SpeciesDescription{T}(:electron, -one(T), one(T), tenth, zero(T), (1, 0, 0), (1, 2, 2), (3tenth, tenth, -tenth),
                                    (tenth / 2, tenth / 2, tenth / 2), 1),)
        prob = antenna3d_problem(T; ncx = 8, nct = 6, length = (16, 12, 12), waist = 3, duration = 2, position = 4, focus = 8,
                                 polarization = θ, a0 = 1 // 2, tend = 15, species = sp, boundaries = bd)
        absorber = label == "open" ? PML(; cells = 4, kappa_max = 2, alpha_max = 0.1) : nothing
        st = EMState(prob, EMNumerics(; workgroupsize, current = Esirkepov(), absorber); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        h = state_digest_3d(st)
        for a in evolved(st)
            h = hash(bits(a), h)
        end
        push!(out, (label, T, h))
    end
    return out
end

# WP-074: the moving window at N = 3, the runs of em3d_window.jl and the digests its worker shares.
# After window_cases.jl (`ZERO3`) and the 3D cases above.

"""
    window_plasma_3d(T; cells = (24, 6, 6), ppc = (2, 2, 2), nsteps = 40, start = 0, vacuum = 4, ramp = 4, drift, thermal,
                     density = 1, seed = 5, bd = (:periodic, :periodic)) -> EMProblem{3}

[`window_plasma`](@ref) in 3D: one electron species of the [`DensityRamp`](@ref) profile entering a
moving window through its front, cold and at rest by default, Δx = 1/2 on every axis, Δt at 0.9
of the Courant limit (the window moves about 0.52 cells per step), y and z periodic or open as
`bd` says. `ramp` may be a dual (the A3 parameter).
"""
function window_plasma_3d(::Type{T}; cells = (24, 6, 6), ppc = (2, 2, 2), nsteps = 40, start = 0, vacuum = 4, ramp = 4,
                          drift = ZERO3(T), thermal = ZERO3(T), density = one(T), seed = 5, bd = (:periodic, :periodic)) where {T}
    L = ntuple(d -> T(cells[d]) / 2, 3)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(density), zero(T), (1, 0, 0), ppc, T.(drift), T.(thermal), 1;
                              momentum = :gaussian_velocity)
    return EMProblem("window-plasma-3d", L, cells, T(dt), nsteps, (e,), seed, (1, 0, 0); boundaries = (:window, bd...),
                     window = MovingWindow(; start = T(start)), profile = DensityRamp(; vacuum = T(vacuum), ramp = ramp isa T ? ramp : T(ramp)))
end

"""
    window_lwfa_3d(T; nsteps = 200, a0 = 1, ramp = 8, cells = (72, 12, 12), position = 10, start = 33, plasma = true, laser = true,
                   bd = (:periodic, :periodic), ppc = (2, 1, 1), seed = 1, ionization = nothing) -> EMProblem{3}

[`window_lwfa`](@ref) in 3D: 8 cells per laser wavelength along x (Δx = π/4) and 4 across
(Δy = Δz = π/2), Δt at 0.95 of the Courant limit; a round [`GaussianLaser`](@ref) (a₀ = 1, waist
6, duration 3, delay 18, focused on its plane) launched by an antenna at x = `position` (column
14), which stays in the lab frame, its axis on the box's axis; a cold electron plasma of
0.02 n_c behind a vacuum of 24 c/ω₀ and a ramp of 8; the window starts at t = 33, before which the
antenna has finished (six durations after the delay) and after which it is passed only once the
back edge's PML of up to 6 cells has left it, and the box (56.5 c/ω₀) holds the pulse to ten
durations ahead of its peak when the window starts (with 60 columns, seven, the front's cut of the
pulse's tail reaches 3.7·10⁻¹² of the field, T-1036). `ionization` is a [`FieldIonization`](@ref) of the problem (WP-075).
"""
function window_lwfa_3d(::Type{T}; nsteps = 200, a0 = 1, ramp = 8, cells = (72, 12, 12), position = 10, start = 33, plasma = true,
                        laser = true, bd = (:periodic, :periodic), ppc = (2, 1, 1), seed = 1, ionization = nothing) where {T}
    h = (2 * T(π) / 8, 2 * T(π) / 4, 2 * T(π) / 4)
    dt = T(95) / 100 / sqrt(sum(d -> inv(h[d])^2, 1:3))
    pulse = GaussianLaser(; a0 = T(a0), waist = T(6), duration = T(3), delay = T(18), focus = T(position), position = T(position),
                          center = (cells[2] * h[2] / 2, cells[3] * h[3] / 2))
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(2) / 100, zero(T), (1, 0, 0), ppc, ZERO3(T), ZERO3(T), 1)
    return EMProblem("window-lwfa-3d", ntuple(d -> cells[d] * h[d], 3), cells, dt, nsteps, plasma ? (e,) : (), seed, (1, 0, 0);
                     lasers = laser ? (pulse,) : (), boundaries = (:window, bd...), window = MovingWindow(; start = T(start)),
                     profile = DensityRamp(; vacuum = T(24), ramp = ramp isa T ? ramp : T(ramp)), ionization)
end

"A 3D window problem's static twin: the same box with `n1` columns, with a window that never moves."
function static_twin_3d(prob::EMProblem{3,T}, n1) where {T}
    h = prob.length[1] / prob.cells[1]
    return EMProblem("static-twin-3d", (n1 * h, prob.length[2], prob.length[3]), (n1, prob.cells[2], prob.cells[3]), prob.dt, prob.nsteps,
                     prob.species, prob.seed, prob.kmode; lasers = prob.lasers, boundaries = prob.boundaries,
                     window = MovingWindow(; start = T(Inf)), profile = prob.profile)
end

"The initialized state of a 3D window problem, with `back` on the back edge, `absorber` on its open axes and `solver` the field solver."
function window_state_3d(prob; mode = Reference(), workgroupsize = nothing, headroom = 1 / 4, absorber = nothing, back = nothing, be = CPU(),
                         solver = Yee())
    st = EMState(prob, EMNumerics(; mode, workgroupsize, headroom, absorber, back_absorber = back, solver); backend = be)
    initialize!(st, prob)
    return st
end

"The alive particles of a 3D species `s` as a sorted vector of (cell₁, cell₂, cell₃, bits of δ₁, δ₂, δ₃, u₁, u₂, u₃, w)."
function particle_set_3d(s)
    alive = Array(s.status) .== ParticleMeshCore.ALIVE
    c = map(a -> Array(a)[alive], s.cell)
    rest = map(a -> Array(a)[alive], (s.offset..., s.u..., s.w))
    return sort([(c[1][i], c[2][i], c[3][i], map(a -> components(a[i]), rest)...) for i in eachindex(c[1])])
end

"The fresh cold load of the 3D window's plasma columns of species 1 at position `s_win`, of the parameters `p` (a scratch species)."
function fresh_window_load_3d(st, p, s_win)
    T = eltype(st.ρ)
    load = (; window_load(st, p, 1)..., vdrift = ZERO3(T), vth = ZERO3(T))
    sp = st.species[1]
    be = get_backend(st.ρ)
    sc = Species(be, T, prod(ncells(st.grid)) * prod(load.ppc); q = sp.q, m = sp.m, N = 3)
    load_window_columns!(sc, st.grid, load, KernelAbstractions.zeros(be, Int32, 1), 1, injection_column(st.aux.window), s_win)
    return sc
end

"""
    window_gauss_3d(st, p, s) -> (; res, scale)

Gauss's law of the 3D window, `Σ_d D⁻_d E_d − (ρ − ρ₀(s))` with ρ from [`window_charge!`](@ref) and
the background's ρ₀ from [`window_background!`](@ref), per interior node (an array over the
n₁ × n₂ × n₃ nodes, absmax over the partials of a dual); `scale = (n₁ + n₂ + n₃)·max|ρ|`.
"""
function window_gauss_3d(st, p, s_win)
    T = eltype(st.ρ)
    grid = st.grid
    be = get_backend(st.ρ)
    ρ, ρ0 = field(be, T, grid), field(be, T, grid)
    window_charge!(ρ, st)
    window_background!(ρ0, st, p, s_win)
    ρ, ρ0 = Array(ρ), Array(ρ0)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    E = map(Array, (st.fields.Ex, st.fields.Ey, st.fields.Ez))
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), 3)), 3)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    res = [absmax(sum(d -> (E[d][j] - E[d][j - e[d]]) / h[d], 1:3) - (ρ[j] - ρ0[j])) for j in I]
    return (; res, scale = sum(n) * maximum(absmax, ρ[I]))
end

# The window runs of T-1030: a 2D window problem on a plane and the same problem in 3D, invariant
# along y or z. A cold plasma drifting backwards and across the plane through a ramp, so that its
# charge against the background drives fields that vary along both axes of the plane (its
# particles leave through the back edge and, with `open`, through the plane's open edge).
const WINDOW_INV_DRIFT = (-1 // 5, 3 // 20, -1 // 10)
const WINDOW_INV_CELLS = (24, 12)

function window_plane_2d(::Type{T}; open = false, nsteps = 64, ramp = 4) where {T}
    h = (one(T) / 2, one(T) / 2)
    L = T.(WINDOW_INV_CELLS) .* h
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), zero(T), (1, 0), (2, 2), T.(WINDOW_INV_DRIFT), ZERO3(T), 1;
                              momentum = :gaussian_velocity)
    return EMProblem("window-plane-2d", L, WINDOW_INV_CELLS, inv3_dt(h), nsteps, (e,), 5, (1, 0); boundaries = (:window, open ? :open : :periodic),
                     window = MovingWindow(; start = zero(T)), profile = DensityRamp(; vacuum = T(4), ramp = ramp isa T ? ramp : T(ramp)))
end

"""
    window_map_3d(axis) -> NamedTuple

The 3D component and sign of each 2D one for a 3D window problem invariant along `axis` (3 or 2),
the window's axis staying axis 1: the identity for `axis = 3`; for `axis = 2` the rotation by a
quarter turn about x that takes the plane's y to z, so a 2D vector (v_x, v_y, v_z) is
(v_x, −v_z, v_y) in 3D (E, B and J alike, a proper rotation).
"""
function window_map_3d(axis)
    axis == 3 && return map(c -> (c, 1), (Ex = :Ex, Ey = :Ey, Ez = :Ez, Bx = :Bx, By = :By, Bz = :Bz, Jx = :Jx, Jy = :Jy, Jz = :Jz))
    m(F) = ((Symbol(F, :x), 1), (Symbol(F, :z), 1), (Symbol(F, :y), -1))
    E, B, J = m(:E), m(:B), m(:J)
    return (Ex = E[1], Ey = E[2], Ez = E[3], Bx = B[1], By = B[2], Bz = B[3], Jx = J[1], Jy = J[2], Jz = J[3])
end

"[`window_plane_2d`](@ref) in 3D, invariant along `axis` (3 or 2): 6 cells of Δ = 1 and 2 particles per cell along it, the plane on the others."
function window_inv_3d(::Type{T}, axis; open = false, nsteps = 64, ramp = 4) where {T}
    h2 = (one(T) / 2, one(T) / 2)
    v = T.(WINDOW_INV_DRIFT)
    put(a, b, c) = axis == 3 ? (a, b, c) : (a, c, b)
    cells = put(WINDOW_INV_CELLS..., 6)
    L = put(T.(WINDOW_INV_CELLS) .* h2..., T(6))
    drift = axis == 3 ? v : (v[1], -v[3], v[2])
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), zero(T), (1, 0, 0), put(2, 2, 2), drift, ZERO3(T), 1;
                              momentum = :gaussian_velocity)
    return EMProblem("window-inv-3d", L, cells, inv3_dt(h2), nsteps, (e,), 5, (1, 0, 0);
                     boundaries = put(:window, open ? :open : :periodic, :periodic),
                     window = MovingWindow(; start = zero(T)), profile = DensityRamp(; vacuum = T(4), ramp = ramp isa T ? ramp : T(ramp)))
end

"""
Largest |3D − 2D| of the window runs over every component and every line along the invariant
axis, each relative to the largest 2D value of its kind (the fields E, B together; the currents J
together), with the map and signs of [`window_map_3d`](@ref).
"""
function window_invariant_error(s2, s3, axis)
    g2, g3, n = guards(s2.grid), guards(s3.grid), ncells(s2.grid)
    m = window_map_3d(axis)
    plane(c) = Array(getfield(s2.fields, c))[(g2 + 1):(g2 + n[1]), (g2 + 1):(g2 + n[2])]
    kind(c) = c in (:Jx, :Jy, :Jz) ? :J : :F
    scale = Dict(k => maximum(c -> Float64(maximum(absmax, plane(c))), filter(c -> kind(c) == k, collect(keys(m)))) for k in (:F, :J))
    err = 0.0
    for (c, (c3, sg)) in pairs(m)
        a2 = plane(c)
        A3 = Array(getfield(s3.fields, c3))
        for r in 1:6
            sl = axis == 3 ? A3[(g3 + 1):(g3 + n[1]), (g3 + 1):(g3 + n[2]), g3 + r] : A3[(g3 + 1):(g3 + n[1]), g3 + r, (g3 + 1):(g3 + n[2])]
            d = Float64(maximum(absmax, sg .* sl .- a2))
            err = max(err, iszero(d) ? 0.0 : d / scale[kind(c)])
        end
    end
    return err
end

"""
    window3d_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-1034: Reference runs per element type, one digest of every particle array, field, convolution
field, counter and history each: the laser of [`window_lwfa_3d`](@ref) into its plasma with y and
z open (the PML) and a PML on the back edge, 110 steps (35 shifts, the antenna passed), and a warm
plasma drifting through a ramp (thermal 0.1 c, every refill drawing), 40 steps.
"""
function window3d_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    pml = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)
    for T in TYPES
        tenth = one(T) / 10
        runs = (("lwfa", window_lwfa_3d(T; nsteps = 110, bd = (:open, :open)), pml, PML(; cells = 6, alpha_max = 0.2)),
                ("warm", window_plasma_3d(T; drift = (tenth, tenth / 2, -tenth / 2), thermal = (tenth, tenth, tenth), vacuum = 2), nothing, nothing))
        for (label, prob, ab, back) in runs
            st = window_state_3d(prob; workgroupsize, be, absorber = ab, back)
            run!(st, params(prob), prob.nsteps)
            h = state_digest_3d(st)
            for a in evolved(st)
                h = hash(bits(a), h)
            end
            push!(out, (label, T, h))
        end
    end
    return out
end

# WP-075: the 3D problems with a dopant (em3d_ionization.jl) and the bitwise digests its tests and
# the worker (T-1045 at 1 and 4 threads) share. After ionization_cases.jl.

"""
    ion_box_3d(T; E0, B0, levels, density, ppc, every, rows, section, capacity, nsteps, courant, plasma, box, cells, boundaries) -> EMProblem{3}

`ion_in_E` in 3D: a dopant in a uniform background field on a periodic box of 16 × 6 × 6 cells of
1/2 (`cells` on the lengths `box`; `boundaries` opens transverse faces), Δt at `courant` (0.9) of
the Courant limit, `levels` (one ADK level of prefactor 1/2, field 3 and power 1 by default) at
`density` on `ppc` points per cell of the transverse `rows`, born every `every` steps. Without
`plasma` the only species has zero density (the prescribed fields of `NoCurrent()` stay `E0`,
`B0`); with it a thermal electron plasma (density 1, v_th = 0.1, a drift and an oblique
perturbation of mode (2, 1, 1), 2 × 2 × 2 per cell). `capacity` defaults to one slot per atom and
birth step, and one birth step more.
"""
function ion_box_3d(::Type{T}; E0 = (one(T) / 2, zero(T), zero(T)), B0 = zero3(T), levels = (ADKLevel(one(T) / 2, T(3), one(T)),),
                    density = one(T) / 100, ppc = (1, 1, 1), every = 1, rows = nothing, section = nothing, capacity = nothing, nsteps = 20,
                    courant = 9 // 10, plasma = false, box = (8, 3, 3), cells = (16, 6, 6), boundaries = (:periodic, :periodic, :periodic)) where {T}
    L = T.(box)
    dt = T(courant) * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3)))
    tenth = one(T) / 10
    e = plasma ? SpeciesDescription{T}(:electron, -one(T), one(T), one(T), tenth, (2, 1, 1), (2, 2, 2), (3tenth, -2tenth, tenth), (tenth, tenth, tenth), 1) :
        SpeciesDescription{T}(:electron, -one(T), one(T), zero(T), zero(T), (1, 0, 0), (1, 1, 1), zero3(T), zero3(T), 1)
    cap = capacity === nothing ? prod(cells) * prod(ppc) * (nsteps ÷ every + 2) : capacity
    ion = FieldIonization(; density, levels, capacity = cap, section, ppc, every, rows)
    return EMProblem("ion-box-3d", L, cells, dt, nsteps, (e,), 3, (2, 1, 1); E0, B0, ionization = ion, boundaries)
end

"""
    ion_window_3d(T; nsteps = 150, a0 = 1, density = 1/200, prefactor = 1/5, start = 28, stop = 34, ramp = 3, every = 2, ppc = (1, 1, 1),
                  rows, capacity = 16384, plasma = true, cutoff = 64, cells = (72, 12, 12), levels) -> EMProblem{3}

`ion_window` in 3D: [`window_lwfa_3d`](@ref)'s box, round pulse (a₀ = 1, waist 6, duration 3 from
an antenna at x = 10) and plasma (0.02 n_c behind a vacuum of 24 and a ramp of 8, the window
moving from t = 33), with a dopant of `density` in the section that rises over `ramp` from `start`
and falls from `stop`, on the 4 × 4 transverse cells about the axis (`rows`), one level of
`prefactor`, field 3 and power 1 (`levels` replaces it), born every `every` steps. `a0`, `density`,
`prefactor` and the section's numbers may be duals; a test problem, too coarse for physics.
"""
function ion_window_3d(::Type{T}; nsteps = 150, a0 = 1, density = T(1) / 200, prefactor = T(1) / 5, start = 28, stop = 34, ramp = 3, every = 2,
                       ppc = (1, 1, 1), rows = ((6.3, 12.5), (6.3, 12.5)), capacity = 16384, plasma = true, cutoff = 64, cells = (72, 12, 12),
                       levels = (ADKLevel(T(prefactor), T(3), one(T); cutoff = T(cutoff)),)) where {T}
    ion = FieldIonization(; density = T(density), levels, capacity, ppc, every, rows,
                          section = DopantSection(; start = T(start), stop = T(stop), ramp = T(ramp)))
    return window_lwfa_3d(T; nsteps, a0, cells, plasma, ionization = ion)
end

"`ion_window_3d` on 72 × 8 × 8 cells, the dopant on the 4 × 4 transverse cells about the axis: the cheaper run of the gradient, the restart and the allocation tests."
ion_window_3d_small(::Type{T}; kw...) where {T} = ion_window_3d(T; cells = (72, 8, 8), rows = ((3.2, 9.4), (3.2, 9.4)), kw...)

"The two levels of the invariant runs, an ADK and a Tong–Lin level, whose rates are of order one at the plasma's fields of about 0.2."
inv_levels(::Type{T}) where {T} = (ADKLevel(one(T) / 2, T(3) / 5, one(T)), TongLinLevel(one(T) / 2, one(T), T(3) / 2, one(T) / 2))

"""
    ion_plane_2d(T; window = false, open = false, nsteps) -> EMProblem{2}

The 2D runs of T-1040 with a dopant of [`inv_levels`](@ref) at 0.05 n_c, 2 × 1 per cell on the rows
whose centres lie in (1.1, 4.9), born every 2 steps. `window = false`: two cold beams drifting along
±(0.2, 0.15, −0.125) with a perturbation of mode (1, 1) on 16 × 12 cells of 1/2 in the uniform field
E₀ = (0.2, 0.1, 0), y periodic or open; `window = true`: `window_plane_2d`'s plasma, whose
charge at the ramp drives the fields, with the dopant in the section from 6 to 14 (ramp 2).
"""
function ion_plane_2d(::Type{T}; window = false, open = false, nsteps = window ? 64 : 20) where {T}
    sec = window ? DopantSection(; start = T(6), stop = T(14), ramp = T(2)) : nothing
    natoms = 16 * 8 * 2 + 8 * 8 * 2
    ion = FieldIonization(; density = T(1) / 20, levels = inv_levels(T), capacity = natoms * (nsteps ÷ 2 + 2), ppc = (2, 1), every = 2,
                          rows = (1.1, 4.9), section = sec)
    if window
        p = window_plane_2d(T; open, nsteps)
        return EMProblem("ion-plane-2d", p.length, p.cells, p.dt, nsteps, p.species, p.seed, p.kmode; boundaries = p.boundaries, window = p.window,
                         profile = p.profile, ionization = ion)
    end
    h = (one(T) / 2, one(T) / 2)
    cells = (16, 12)
    L = T.(cells) .* h
    v = T.(INV3_DRIFT)
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, one(T) / 20, (1, 1), (2, 2), v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), (1, 1), (2, 2), .-v, zero3(T), 2))
    return EMProblem("ion-plane-2d", L, cells, inv3_dt(h), nsteps, sp, 1, (1, 1); E0 = (T(1) / 5, T(1) / 10, zero(T)),
                     boundaries = (:periodic, open ? :open : :periodic), ionization = ion)
end

"""
    ion_inv_3d(T, axis; window = false, open = false, nsteps) -> EMProblem{3}

[`ion_plane_2d`](@ref) in 3D, invariant along `axis` (3 or 2; for 2 the plane's y on z, a quarter turn
about x, as [`window_map_3d`](@ref)): 6 cells of Δ = 1 along it, 2 particles per cell and one atom,
the dopant's rows on the plane's transverse axis, the drift and E₀ turned with the plane.
"""
function ion_inv_3d(::Type{T}, axis; window = false, open = false, nsteps = window ? 64 : 20) where {T}
    put(a, b, c) = axis == 3 ? (a, b, c) : (a, c, b)
    turn(v) = axis == 3 ? v : (v[1], -v[3], v[2])
    sec = window ? DopantSection(; start = T(6), stop = T(14), ramp = T(2)) : nothing
    natoms = 6 * (16 * 8 * 2 + 8 * 8 * 2)
    ion = FieldIonization(; density = T(1) / 20, levels = inv_levels(T), capacity = natoms * (nsteps ÷ 2 + 2), ppc = put(2, 1, 1), every = 2,
                          rows = put(nothing, (1.1, 4.9), nothing)[2:3], section = sec)
    if window
        p = window_inv_3d(T, axis; open, nsteps)
        return EMProblem("ion-inv-3d", p.length, p.cells, p.dt, nsteps, p.species, p.seed, p.kmode; boundaries = p.boundaries, window = p.window,
                         profile = p.profile, ionization = ion)
    end
    h2 = (one(T) / 2, one(T) / 2)
    cells = put(16, 12, 6)
    L = put(T(8), T(6), T(6))
    v = turn(T.(INV3_DRIFT))
    m = put(1, 1, 0)
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, one(T) / 20, m, put(2, 2, 2), v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), m, put(2, 2, 2), .-v, zero3(T), 2))
    return EMProblem("ion-inv-3d", L, cells, inv3_dt(h2), nsteps, sp, 1, m; E0 = turn((T(1) / 5, T(1) / 10, zero(T))),
                     boundaries = put(:periodic, open ? :open : :periodic, :periodic), ionization = ion)
end

"The initialized 3D state of an ionization problem (`current = NoCurrent()` for prescribed fields; `absorber` on its open axes; `gather` the particles' gather)."
function ion_state_3d(prob; mode = Reference(), workgroupsize = nothing, current = Esirkepov(), absorber = nothing, be = CPU(), gather = nothing,
                      headroom = 1 / 4)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, current, absorber, gather, headroom); backend = be)
    initialize!(st, prob)
    return st
end

"A hash of the bits of everything a 3D state evolves, of `K` and of the histories."
function ion_digest_3d(st::EMState{3})
    h = UInt64(0)
    for a in (evolved(st)..., st.K, values(st.recorder.hist)...)
        h = hash(bits(a), h)
    end
    return h
end

"""
    ion3d_digests(; workgroupsize = nothing, be = CPU(), gather = nothing) -> Vector{(label, T, digest)}

T-1045 (and T-1054 with `gather`): Reference runs per element type, one digest each: the dopant in
the 3D window (`ion_window_3d`, 110 steps: births, shifts, atoms and electrons leaving through the
back edge) and a thermal plasma with a dopant of two levels in the box open along y and z
(the PML), 20 steps.
"""
function ion3d_digests(; workgroupsize = nothing, be = CPU(), gather = nothing)
    out = Tuple{String,DataType,UInt64}[]
    pml = PML(; cells = 3, kappa_max = 2, alpha_max = 0.1)
    for T in TYPES
        w = ion_window_3d(T; nsteps = 110)
        st = ion_state_3d(w; workgroupsize, be, gather)
        run!(st, params(w), w.nsteps)
        push!(out, ("window", T, ion_digest_3d(st)))
        b = ion_box_3d(T; levels = (ADKLevel(one(T) / 2, T(3), one(T)), TongLinLevel(one(T) / 2, T(4), T(3) / 2, one(T))), density = T(1) / 20,
                       ppc = (2, 1, 1), every = 2, plasma = true, cells = (16, 10, 10), box = (8, 5, 5), boundaries = (:periodic, :open, :open))
        sb = ion_state_3d(b; workgroupsize, be, gather, absorber = pml)
        run!(sb, params(b), b.nsteps)
        push!(out, ("open", T, ion_digest_3d(sb)))
    end
    return out
end

# --- WP-075's helpers, which em3d_ionization.jl and em3d_gather.jl share -------------------


"Start em3d_ion_worker.jl at `nt` threads, its output going to a file; once per suite."
function start_ion3d_worker(nt)
    get!(ION3D_WORKERS, nt) do
        worker = joinpath(@__DIR__, "em3d_ion_worker.jl")
        out = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$(dirname(Base.active_project())) $worker`
        (run(pipeline(cmd; stdout = out, stderr = stderr); wait = false), out)
    end
end
const ION3D_WORKERS = Dict{Int,Tuple{Base.Process,String}}()

"The lines of the ionization worker at `nt` threads that start with `prefix`, after it finished."
function ion3d_worker_lines(nt, prefix)
    proc, out = start_ion3d_worker(nt)
    wait(proc)
    success(proc) || error("em3d_ion_worker.jl with $nt threads failed: $proc")
    return filter(startswith(prefix), split(read(out, String), '\n'))
end

"The alive electrons of a 3D state's ionization, on the host."
function born_3d(st)
    e = ionization_electrons(st)
    alive = Array(e.status) .== ParticleMeshCore.ALIVE
    pick(a) = Array(a)[alive]
    return (; n = count(alive), cell = map(pick, e.cell), offset = map(pick, e.offset), u = map(pick, e.u), w = pick(e.w))
end

"""
    ion_gauss_3d(st, p, k) -> (; res, bare, scale, escale, ions)

Gauss's law of a 3D state with ionization after step `k`, per interior node: `res`, the residual
`Σ_d D⁻_d E_d − ρ` with ρ the charge of every particle ([`window_charge!`](@ref): the species and the
born electrons) plus the dopant ions' ([`ionization_charge!`](@ref)), less the neutralizing
background (the mean of a periodic box; a window's `window_background!`; none in a box with an
open axis, whose runs here start without charge); `bare`, the same without the ions; `scale`,
`(n₁ + n₂ + n₃)·max|ρ|` of the particles; `escale`, the divergence's own roundoff scale
`(Σ_d 2/Δ_d)·max|E|`; `ions`, the largest ion charge density.
"""
function ion_gauss_3d(st, p, k)
    T = eltype(st.ρ)
    grid = st.grid
    be = get_backend(st.ρ)
    ρ, ρi, ρ0 = field(be, T, grid), field(be, T, grid), field(be, T, grid)
    window_charge!(ρ, st)
    ionization_charge!(ρi, st)
    window = st.aux.window
    window === nothing || window_background!(ρ0, st, p, window_shifts(st, k))
    ρ, ρi, ρ0 = Array(ρ), Array(ρi), Array(ρ0)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    periodic = window === nothing && st.numerics.absorber === nothing
    mean(a) = periodic ? sum(a[I]) / prod(n) : zero(T)
    m, mi = mean(ρ), mean(ρi)
    E = map(Array, (st.fields.Ex, st.fields.Ey, st.fields.Ez))
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), 3)), 3)
    div(j) = sum(d -> (E[d][j] - E[d][j - e[d]]) / h[d], 1:3)
    res = [absmax(div(j) - (ρ[j] - m + ρi[j] - mi - ρ0[j])) for j in I]
    bare = [absmax(div(j) - (ρ[j] - m - ρ0[j])) for j in I]
    hv = map(value_of, h)
    escale = sum(d -> 2 / hv[d], 1:3) * maximum(a -> maximum(absmax, a[I]), E)
    return (; res, bare, scale = sum(n) * maximum(absmax, ρ[I]), escale, ions = maximum(absmax, ρi[I]))
end

"The parameters of T-1046: the laser's a₀, the dopant's density, the level's prefactor and the section's start (c/ω₀)."
const ION3_X0 = (1.0, 0.005, 0.2, 28.0)
const ION3_NAMES = ("a0", "density", "prefactor", "start")

"""
The born charge (the electrons' summed weight) and their kinetic energy Σ w·(γ − 1) after 90 steps
of `ion_window_3d_small` at `x = (a0, density, prefactor, start)`, when no electron has left the window.
"""
function ion3d_objectives(x, mode = Reference(); be = CPU(), gather = nothing)
    T = promote_type(map(typeof, x)...)
    prob = ion_window_3d_small(T; nsteps = 90, a0 = x[1], density = x[2], prefactor = x[3], start = x[4])
    st = ion_state_3d(prob; mode, be, gather)
    run!(st, params(prob), prob.nsteps)
    el = born_3d(st)
    ionization_lost(st) == 0 || error("a birth found no slot")
    γ = sqrt.(1 .+ el.u[1] .^ 2 .+ el.u[2] .^ 2 .+ el.u[3] .^ 2)
    return [sum(el.w), sum(el.w .* (γ .- 1))]
end
