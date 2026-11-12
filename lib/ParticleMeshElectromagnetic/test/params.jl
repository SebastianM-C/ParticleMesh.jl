# WP-055: the step as a function of the state and the parameters, T-720 to T-725. The parameters
# (`EMParams`: the lasers, the densities and the profile, the witness bunch, the seed) reach
# `initialize!` and `step!` as an argument, and the state keeps no copy of them: one state gives
# the run of whichever parameters it is handed, and the diagnostics belong to an observer. After
# window_cases.jl (`ZERO3`, `window_plasma`), boundaries_cases.jl (`open_plasma_2d`) and harness.jl (its helpers).

using SciMLBase: SciMLBase, DiscreteProblem, remake, solve
using OrdinaryDiffEqFunctionMap: FunctionMap
using ParticleMeshElectromagnetic: witness_frame, witness_species, check_setup, AntennaPlane

"""
    params_case(x; nsteps = 120, cells = (64, 16), position = 6, start = 33, plasma = true, witness = nothing, open_y = false) -> EMProblem{2}

`window_lwfa`'s small laser–plasma run with `x = (a0, duration, density, ramp)` of one element
type: 8 cells per laser wavelength (Δx = π/4), Δt at 0.95 of the Courant limit; a Gaussian pulse
(waist 6, delay 18) launched by an antenna at x = `position`; a cold electron plasma of `density`
behind a vacuum of 24 c/ω₀ and a ramp of `ramp`, so the first window (50 c/ω₀) already holds
plasma and the refill brings more; the window starts at t = `start`, after the antenna has
finished for durations up to 3.15. Every entry of `x` may be a dual: a₀ and the duration enter
through the antenna's source, the density and the ramp through the weights of the initial load
and of the refill (A3).
"""
function params_case(x; nsteps = 120, cells = (64, 16), position = 6, start = 33, plasma = true, witness = nothing, open_y = false)
    T = promote_type(map(typeof, x)...)
    a0, duration, density, ramp = map(T, x)
    h = 2 * T(π) / 8
    dt = T(95) / 100 * h / sqrt(T(2))
    pulse = GaussianLaser(; a0, waist = T(6), duration, delay = T(18), focus = T(position), position = T(position), center = cells[2] * h / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), density, zero(T), (1, 0), (2, 2), ZERO3(T), ZERO3(T), 1)
    return EMProblem("params-case", (cells[1] * h, cells[2] * h), cells, dt, nsteps, plasma ? (e,) : (), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, open_y ? :open : :periodic), window = MovingWindow(; start = T(start)),
                     profile = DensityRamp(; vacuum = T(24), ramp), witness)
end

"The alive particles' arrays of a species in slot order, with the status and the count: what a run evolves of it (a free slot keeps the data of whatever held it last, and the extras are the push's scratch)."
function particle_arrays(s)
    alive = Array(s.status) .== ParticleMeshCore.ALIVE
    return (Array(s.status), Array(s.count), map(a -> Array(a)[alive], (s.cell..., s.offset..., s.u..., s.w))...)
end

"""
    run_arrays(st) -> Vector

Everything a run evolves or records in a state, on the host, in a fixed order: the species and the
witness bunch (`particle_arrays`), the fields, `K`, the recorder's histories and, at N = 2, a
PML's convolution fields (the open axis' and the back edge's) and the window's counters.
"""
function run_arrays(st::EMState)
    out = Any[]
    for s in (st.species..., witness_species(st)...)
        append!(out, particle_arrays(s))
    end
    append!(out, map(Array, values(st.fields)))
    push!(out, Array(st.K))
    append!(out, map(Array, values(st.recorder.hist)))
    ndims(st) == 2 || return out
    L = st.aux.layer
    L isa PMLFields && append!(out, map(Array, (L.ψBx, L.ψBz, L.ψEx, L.ψEz)))
    w = st.aux.window
    w === nothing && return out
    append!(out, map(Array, (w.lost..., w.fast...)))
    w.back === nothing || append!(out, map(Array, (w.back.ψBy, w.back.ψBz, w.back.ψEy, w.back.ψEz)))
    return out
end

"Whether two states hold the same run, bit for bit (`run_arrays`)."
same_run(a, b) = (x = run_arrays(a); y = run_arrays(b); length(x) == length(y) && all(map(same_bits, x, y)))

"A `Dual{Float64,4}` with the given value and partials."
mkdual4(v, ps...) = Dual{Nothing}(Float64(v), Float64.(ps)...)

# --- T-720 ----------------------------------------------------------------------------------

# The derivative through the parameters is the derivative of the state rebuilt from the dual
# problem, bit for bit (value and partials, A4: one element type, one backend): path A builds the
# state from the dual problem, as every gradient test does (T-581); path B takes a state built
# from another problem of the same configuration (other values, no partials), which has already
# run that problem, and hands it the dual parameters, in `initialize!` and in every step. a₀ and
# the duration reach the fields through the antenna's source in the step, the density and the
# ramp through the weights of the initial load and of the refill (38 shifts in 120 steps). The
# comparison is not empty: every seeded parameter has reached E, and the plasma's two the weights
# (the loaded positions carry none, A3).
function body_T720(be)
    x0 = (1.0, 3.0, 0.02, 8.0)
    base0 = (0.5, 2.5, 0.05, 6.0)
    seeds3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))                 # a₀, the duration, the density
    seeds4 = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))     # and the ramp
    for (xd, xb, np) in ((map((v, s) -> mkdual(v, s...), x0, seeds3), map(v -> mkdual(v, 0, 0, 0), base0), 3),
                         (map((v, s) -> mkdual4(v, s...), x0, seeds4), map(v -> mkdual4(v, 0, 0, 0, 0), base0), 4)),
        mode in (Reference(), Fast())
        probd = params_case(xd)
        pd = params(probd)
        num = EMNumerics(; mode)
        a = EMState(probd, num; backend = be)
        initialize!(a, probd)
        run!(a, pd, probd.nsteps)
        base = params_case(xb)
        b = EMState(base, num; backend = be)
        initialize!(b, base)
        run!(b, params(base), base.nsteps)
        @test !same_run(a, b)                         # the base problem's own run is another run
        initialize!(b, pd)
        @test all(e -> all(iszero, partials(e)), Array(b.species[1].offset[1]))     # loaded positions carry no partials (A3)
        run!(b, pd, probd.nsteps)
        @test same_run(a, b)
        @test window_shifts(a, probd.nsteps) > 30 && all(iszero, window_lost(a))
        Ex = Array(a.fields.Ex)
        w = particle_arrays(a.species[1])[end]
        for i in 1:np
            @test any(e -> partials(e)[i] != 0, Ex)
            @test any(e -> partials(e)[i] != 0, w) == (i >= 3)       # the weights carry the density's and the ramp's only
        end
    end
end

@testset "T-720 dual laser and density parameters handed to a state built from other values give the bits of the state rebuilt from the dual problem [A3, A4, A8, WP-055]" begin
    body_T720(CPU())
end

# --- T-721 ----------------------------------------------------------------------------------

# One state, two parameter sets, two runs, each an exact discrete identity. In vacuum the scheme
# is linear in the antenna's source and a factor 2 is exact in binary arithmetic, so the fields of
# a₀ → 2a₀ are twice the fields of a₀ bit for bit, partials included: the step reads the laser
# from its parameters and nothing of the first run is left in the state. A cold plasma at rest
# only receives weights (no current, T-471): in the window's initial load, and after 40 steps,
# when every particle comes from a refill (29 shifts through 24 columns), the weights of
# density → 2·density are twice the weights, the same particles alive: the initial load and the
# refill read the density from the parameters. Every parameter of a static load is read from them
# too: a state built for one plasma and handed another's seed, background fields, density and
# perturbation or thermal speed (at N = 2), amplitude and seed (at N = 1), is the state built for
# that other plasma, bit for bit over the first steps. The witness bunch's centre enters the
# weights at the injection step and nothing else: two centres on one state give the same gains
# and other weights.
function body_T721(be)
    for T in TYPES
        a0 = seeded(T, 1)
        prob1 = params_case((a0, T(3), T(2) / 100, T(8)); plasma = false, nsteps = 80)
        prob2 = params_case((2 * a0, T(3), T(2) / 100, T(8)); plasma = false, nsteps = 80)
        st = EMState(prob1, EMNumerics(); backend = be)
        initialize!(st, prob1)
        run!(st, params(prob1), prob1.nsteps)
        F1 = map(Array, st.fields)
        initialize!(st, params(prob2))
        run!(st, params(prob2), prob2.nsteps)
        F2 = map(Array, st.fields)
        @test maximum(absmax, F1.Ey) > 1 // 10                  # the pulse is in the box
        @test all(c -> same_bits(F2[c], 2 .* F1[c]), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
        @test !same_bits(F1.Ey, F2.Ey)

        n0 = seeded(T, 1)
        cold1, cold2 = window_plasma(T; density = n0), window_plasma(T; density = 2 * n0)
        st = EMState(cold1, EMNumerics(); backend = be)
        initialize!(st, cold1)
        l1 = particle_arrays(st.species[1])                    # the initial load
        run!(st, params(cold1), cold1.nsteps)
        @test window_shifts(st, cold1.nsteps) > ncells(st.grid)[1]
        s1 = particle_arrays(st.species[1])                    # refills only
        initialize!(st, params(cold2))
        l2 = particle_arrays(st.species[1])
        run!(st, params(cold2), cold2.nsteps)
        s2 = particle_arrays(st.species[1])
        for (a, b) in ((l1, l2), (s1, s2))
            @test !isempty(a[end]) && any(!iszero, a[end])
            @test same_bits(b[end], 2 .* a[end])                        # the weights
            @test all(map(same_bits, a[1:(end - 1)], b[1:(end - 1)]))   # the same particles at the same places
        end

        base = plasma_2d(T)
        st = state_2d(base; be, current = Esirkepov())
        run!(st, params(base), 3)
        others = ((; seed = 12), (; E0 = (T(1) / 10, zero(T), T(2) / 10), B0 = (zero(T), T(3) / 10, T(1) / 2)),
                  (; density = seeded(T, 2), amplitude = seeded(T, 1) / 10), (; vth = T(1) / 10))
        for kw in others
            other = plasma_2d(T; kw...)
            ref = state_2d(other; be, current = Esirkepov())
            run!(ref, params(other), 3)
            @test !same_run(st, ref)
            initialize!(st, params(other))
            run!(st, params(other), 3)
            @test same_run(st, ref)
        end
        st1, _, _, _ = case_state("landau", T; ppc = 16, t_end = 1, be)
        ref1, q1, _, _ = case_state("landau", T; ppc = 16, t_end = 1, be, amplitude = seeded(T, 1) / 50, seed = 7)
        run!(ref1, q1, 5)
        initialize!(st1, q1)
        run!(st1, q1, 5)
        @test same_run(st1, ref1)
    end
    bunch(c) = WitnessBunch(; inject = 40.0, region = ((20.0, 30.0), (5.0, 7.0)), momentum = 3.0, center = (c, 6.0), length = 2.0)
    prob1, prob2 = map(c -> params_case((1.0, 3.0, 0.02, 8.0); witness = bunch(c), nsteps = 100), (24.0, 27.0))
    st = EMState(prob1, EMNumerics(); backend = be)
    g = map((prob1, prob2)) do prob
        initialize!(st, params(prob))
        run!(st, params(prob), prob.nsteps)
        witness_gains(st, params(prob))
    end
    @test all(g[1].alive) && any(!iszero, g[1].gain)
    @test same_bits(g[1].gain, g[2].gain)
    @test !same_bits(g[1].w, g[2].w)
end

@testset "T-721 one state run with two parameter sets gives each one's run: fields exactly linear in a₀, refilled weights in the density [A3, A4, A8, WP-055]" begin
    body_T721(CPU())
end

# --- T-722 ----------------------------------------------------------------------------------

# The diagnostics are the observer's. A step with `observer = nothing` writes no history, takes
# any `k`, and leaves the state the default observer leaves (observing changes nothing); a second
# recorder for another mode, observed next to the state's own, holds bit for bit the histories of
# a state built for a problem with that diagnostic mode, and the state's own recorder those of a
# plain run; a function is called with the state and the step at the start of every step, and what
# the step's observer records at step k is what a recorder shown the state by hand before that
# step records (the observation is of the start of the step); a recorder refuses a step past its
# length, a state of another dimension or of another grid.
@testset "T-722 the histories and the diagnostic mode belong to the observer: none, another mode next to the default, a function [A4, A10, WP-055]" begin
    T = Float64
    # N = 2
    prob = plasma_step_2d(T; nsteps = 6)
    other = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species, prob.seed, (1, 2))
    num = EMNumerics(; current = Esirkepov())
    mk(q) = (s = EMState(q, num); initialize!(s, q); s)
    plain, silent, both, ref = mk(prob), mk(prob), mk(prob), mk(other)
    p = params(prob)
    run!(plain, p, prob.nsteps)
    run!(ref, params(other), other.nsteps)
    run!(silent, p, prob.nsteps; observer = nothing)
    @test all(h -> all(iszero, h), histories(silent))
    @test all(c -> same_bits(Array(plain.fields[c]), Array(silent.fields[c])), keys(plain.fields))
    step!(silent, p, prob.nsteps + 1; observer = nothing)        # no history, no limit on k
    @test_throws ArgumentError step!(plain, p, prob.nsteps + 1)
    r2 = HistoryRecorder(other, both)
    seen = Int[]
    run!(both, p, prob.nsteps; observer = (both.recorder, r2, (s, k) -> (push!(seen, k); @test s === both)))
    @test seen == 1:prob.nsteps
    @test r2.kmode == (1, 2) && both.recorder.kmode == prob.kmode
    @test all(map(same_bits, values(histories(both)), values(histories(plain))))
    @test all(map(same_bits, values(histories(r2)), values(histories(ref))))
    @test !same_bits(histories(r2).Ex_re, histories(both).Ex_re)
    byhand = mk(prob)
    rh = HistoryRecorder(prob, byhand)
    for k in 1:prob.nsteps
        observe!(rh, byhand, k)
        step!(byhand, p, k)
    end
    @test all(map(same_bits, values(histories(rh)), values(histories(plain)))) && any(!iszero, histories(rh).WEx)
    @test histories(rh).WEx[1] != histories(rh).WEx[2]                 # the entries are of different times
    short = HistoryRecorder(prob, plain; nsteps = 2)
    @test length(short) == 2
    @test_throws ArgumentError observe!(short, plain, 3)
    @test_throws ArgumentError observe!(short, plain, 0)
    big = EMState(plasma_step_2d(T; nsteps = 6, cells = (12, 8)), num)
    @test_throws ArgumentError observe!(big.recorder, plain, 1)   # tables of another grid
    # N = 1
    st1, p1, prob1, case = case_state("landau", T; ppc = 16, t_end = 1)
    alt = problem_from_case(case, T; ppc = 16, t_end = 1)
    alt = EMProblem(alt.name, alt.length, alt.cells, alt.dt, alt.nsteps, alt.species, alt.seed, 3 * alt.kmode)
    ref1 = EMState(alt, st1.numerics)
    initialize!(ref1, alt)
    run!(ref1, params(alt), alt.nsteps)
    r1 = HistoryRecorder(alt, st1)
    rh1 = HistoryRecorder(prob1, st1)
    for k in 1:prob1.nsteps
        observe!(rh1, st1, k)
        step!(st1, p1, k; observer = (st1.recorder, r1))
    end
    @test all(map(same_bits, values(histories(rh1)), values(histories(st1)))) && histories(rh1).WEx[1] != histories(rh1).WEx[2]
    @test all(map(same_bits, values(histories(r1)), values(histories(ref1))))
    @test same_bits(histories(st1).WEx, histories(ref1).WEx) && !same_bits(histories(st1).Ex_re, histories(r1).Ex_re)
    @test_throws ArgumentError observe!(r1, plain, 1)             # a 1D recorder, a 2D state
    @test_throws ArgumentError observe!(plain.recorder, st1, 1)
end

# --- T-723 ----------------------------------------------------------------------------------

# `params(problem)` holds the problem's numbers and nothing else, and what fits a state is checked
# where the parameters are read (`initialize!`, `step!`, and so `run!`): parameters of another element type,
# species count or witness, a laser off the state's antenna plane, a pulse that the window would
# pass while it emits; a problem of another configuration (cells, cell sizes, time step, a species'
# particles per cell, stream, momentum distribution, charge or mass, the boundaries, the window's
# start, the witness's placement, charge or mass) is not this state's, while one that differs in
# parameters only is. A laser moved within its plane's half cell
# is the same run: the position only chooses the column (A3).
@testset "T-723 the parameters are the problem's numbers, and a state refuses the ones that do not fit it [A3, A10, WP-055]" begin
    T = Float64
    x = (1.0, 3.0, 0.02, 8.0)
    wb = WitnessBunch(; inject = 40.0, region = ((20.0, 30.0), (5.0, 7.0)), momentum = 3.0, center = (25.0, 6.0), length = 2.0)
    prob = params_case(x; witness = wb)
    p = params(prob)
    @test p isa EMParams{T} && eltype(p) === T
    @test p.lasers[1] === ParticleMeshElectromagnetic.convert_laser(T, prob.lasers[1])
    e = prob.species[1]
    @test p.species[1] === SpeciesParams{T,2}(e.density, e.amplitude, (2π * e.mode[1] / prob.length[1], 2π * e.mode[2] / prob.length[2]), e.drift, e.thermal)
    @test p.profile === prob.profile && p.seed == prob.seed && p.E0 === prob.E0 && p.B0 === prob.B0
    @test p.witness === WitnessParams{T}(prob.witness.momentum, prob.witness.center, prob.witness.length, prob.witness.width)
    @test EMParams(p) === p && EMParams(p; seed = 9).seed == 9 && EMParams(p; seed = 9).lasers === p.lasers
    magnetized = plasma_2d(T; E0 = (T(1) / 10, T(2) / 10, T(3) / 10), B0 = (T(4) / 10, T(5) / 10, T(6) / 10), seed = 11)
    pm = params(magnetized)
    @test pm.E0 === magnetized.E0 && pm.B0 === magnetized.B0 && pm.seed == 11 && pm.profile === nothing && pm.witness === nothing
    em = magnetized.species[1]
    @test (pm.species[1].density, pm.species[1].amplitude, pm.species[1].drift, pm.species[1].thermal) === (em.density, em.amplitude, em.drift, em.thermal)
    landau = problem_from_case(load_case("landau"), T)
    p1 = params(landau)
    @test p1.species[1].wavevector == (2π * landau.species[1].mode / landau.length,) && p1.lasers == () && p1.witness === nothing
    st = EMState(prob, EMNumerics())
    @test check_params(st, p) === p
    @test st.aux.antennas[1] isa AntennaPlane{T} && isbits(st.aux.antennas[1])
    refused(q) = (@test_throws ArgumentError check_params(st, q); @test_throws ArgumentError initialize!(st, q);
                  @test_throws ArgumentError step!(st, q, 1); @test_throws ArgumentError run!(st, q, 1))
    refused(params(params_case(Float32.(x); witness = wb)))                          # another element type
    refused(EMParams(p; species = ()))                                               # no species
    refused(EMParams(p; witness = nothing))                                          # the state has a bunch
    refused(EMParams(p; lasers = ()))                                                # the state has a plane
    refused(params(params_case(x; witness = wb, position = 10)))                     # the laser on another column
    refused(EMParams(p; lasers = (GaussianLaser(; a0 = 1.0, waist = 6.0, duration = 4.0, delay = 18.0, focus = 6.0, position = 6.0, center = 6.0),)))   # still emitting when the window passes
    @test_throws ArgumentError check_params(EMState(params_case(x), EMNumerics()), p)   # a bunch for a state without one
    static = plasma_step_2d(T; nsteps = 2)
    @test_throws ArgumentError check_params(EMState(static, EMNumerics()), EMParams(params(static); profile = prob.profile))
    st1, q1, _, _ = case_state("landau", T; ppc = 4, t_end = 1)
    @test_throws ArgumentError check_params(st1, EMParams(q1; lasers = prob.lasers))
    # a problem of another configuration is not this state's
    e2(; ppc = (2, 2), stream = 1, charge = -one(T), mass = one(T)) =
        (SpeciesDescription{T}(:electron, charge, mass, e.density, zero(T), (1, 0), ppc, ZERO3(T), ZERO3(T), stream),)
    like(; dt = prob.dt, species = prob.species, cells = prob.cells, box = prob.length, window = prob.window, witness = prob.witness) =
        EMProblem(prob.name, box, cells, dt, prob.nsteps, species, prob.seed, prob.kmode; lasers = prob.lasers,
                  boundaries = prob.boundaries, window, profile = prob.profile, witness)
    bunch(; kw...) = WitnessBunch(; inject = 40.0, region = ((20.0, 30.0), (5.0, 7.0)), momentum = 3.0, center = (25.0, 6.0), length = 2.0, kw...)
    @test check_setup(st, like()) isa EMProblem
    @test check_setup(st, like(; witness = bunch(; momentum = 5.0, center = (22.0, 6.0), length = 1.0))) isa EMProblem   # parameters only
    for q in (like(; dt = prob.dt / 2), like(; species = e2(; ppc = (1, 2))), like(; species = e2(; stream = 2)), like(; cells = (64, 12)),
              like(; box = (prob.length[1], 2 * prob.length[2])), like(; species = e2(; charge = -2 * one(T))), like(; species = e2(; mass = 2 * one(T))),
              like(; window = MovingWindow(; start = 34.0)), like(; witness = nothing), like(; witness = bunch(; inject = 41.0)),
              like(; witness = bunch(; region = ((20.0, 31.0), (5.0, 7.0)))), like(; witness = bunch(; ppc = (2, 1))), like(; witness = bunch(; mass = 2.0)),
              like(; witness = bunch(; charge = 1.0)))
        @test_throws ArgumentError check_setup(st, q)
        @test_throws ArgumentError initialize!(st, q)
    end
    @test_throws ArgumentError check_setup(EMState(static, EMNumerics()), plasma_step_2d(T; nsteps = 2, momentum = :gaussian_velocity))
    @test_throws ArgumentError check_setup(st1, static)                                           # N = 2 against N = 1
    open2, ab = open_plasma_2d(T; nsteps = 2)
    closed = EMProblem(open2.name, open2.length, open2.cells, open2.dt, open2.nsteps, open2.species, open2.seed, open2.kmode)
    @test_throws ArgumentError check_setup(EMState(open2, EMNumerics(; absorber = ab)), closed)   # an open axis against a periodic one
    @test_throws ArgumentError check_setup(EMState(closed, EMNumerics()), open2)
    # within its plane's half cell the position changes nothing: it only chooses the column
    l = prob.lasers[1]
    q = EMParams(p; lasers = (GaussianLaser(; a0 = l.a0, waist = l.waist, duration = l.duration, delay = l.delay, focus = l.focus, position = 6.1, center = l.center),))
    @test AntennaPlane(6.1, st.grid, prob.dt) === st.aux.antennas[1]
    a, b = EMState(prob, EMNumerics()), EMState(prob, EMNumerics())
    initialize!(a, p)
    run!(a, p, 60)
    initialize!(b, q)
    run!(b, q, 60)
    @test same_run(a, b) && any(!iszero, Array(a.fields.Ey))
end

# --- T-724 ----------------------------------------------------------------------------------

# `initialize!` starts everything a run evolves, so a state that has run (other parameters, to the
# end) is a new state after it, bit for bit over the whole next run: with a PML on the open axis and
# on the window's back edge (convolution fields), a witness bunch (injected and partly retired)
# and a laser in a plasma; and with a buffer too small for the refill, whose `lost` counter counts
# one run, not two.
function body_T724(be)
    for T in TYPES
        yc = 16 * 2π / 8 / 2
        wb = WitnessBunch(; inject = 40.0, region = ((20.0, 30.0), (yc - 1, yc + 1)), ppc = (2, 1), momentum = T(3), center = (T(25), T(yc)), length = T(2))
        case(a0, n0) = params_case((seeded(T, a0), T(3), seeded(T, n0), T(8)); nsteps = 100, cells = (72, 16), position = 10, witness = wb, open_y = true)
        prob, first = case(1, 2 // 100), case(3 // 2, 1 // 100)
        num = EMNumerics(; back_absorber = PML(; cells = 6), absorber = PML(; cells = 4, alpha_max = 0.2))
        fresh = EMState(prob, num; backend = be)
        initialize!(fresh, prob)
        run!(fresh, params(prob), prob.nsteps)
        used = EMState(prob, num; backend = be)
        initialize!(used, first)
        run!(used, params(first), first.nsteps)
        @test !same_run(fresh, used)
        @test any(!iszero, Array(used.aux.layer.ψEx)) && any(!iszero, Array(used.aux.window.back.ψEy))
        @test witness_alive(used) > 0
        initialize!(used, prob)
        @test all(iszero, Array(used.aux.layer.ψEx)) && all(iszero, Array(used.aux.window.back.ψEy)) && witness_alive(used) == 0
        run!(used, params(prob), prob.nsteps)
        @test same_run(fresh, used)
    end
    fwd = window_plasma(Float64; drift = (0.5, 0.0, 0.0), nsteps = 40, vacuum = -10, density = 1e-4)
    tight = window_state(fwd; headroom = 0, be)
    run!(tight, params(fwd), fwd.nsteps)
    lost = window_lost(tight)
    @test lost[1] > 0
    initialize!(tight, fwd)
    @test window_lost(tight) == (0,)
    run!(tight, params(fwd), fwd.nsteps)
    @test window_lost(tight) == lost
end

@testset "T-724 initialize! starts everything a run evolves: a state that ran other parameters runs as a new one, bitwise [A4, A8, WP-055]" begin
    body_T724(CPU())
end

# --- T-725 ----------------------------------------------------------------------------------

# The harness follows. Its map takes the step's parameters from the problem's `p`: a plain
# `remake(prob; p = p′)` with another laser, as a problem description or as its `EMParams`, gives
# that laser's run (in vacuum, twice a₀ gives twice the fields bit for bit) with no new cache,
# where it used to need `remake_reload`; a `p′` that does not fit the cache is refused by the
# step. A solve starts from `u0` without `initialize!`, so the map zeroes at step 1 what the
# partition does not hold of the fields, a PML's convolution fields: a second solve of a problem
# with a PML is the first, and both are the loop.
@testset "T-725 the harness: remake(p) changes the step's parameters with no new cache, and a second solve with a PML is the first [A3, A9, WP-055, WP-008]" begin
    T = Float64
    prob1 = params_case((1.0, 3.0, 0.02, 8.0); plasma = false, nsteps = 80)
    prob2 = params_case((2.0, 3.0, 0.02, 8.0); plasma = false, nsteps = 80)
    dprob = DiscreteProblem(prob1, EMNumerics(); backend, save_everystep = false)
    sol1 = solve(dprob, FunctionMap())
    F1 = map(copy, sol1.u[end].x)
    for p′ in (prob2, params(prob2))
        again = remake(dprob; p = p′)
        @test harness_state(again) === harness_state(dprob)                      # the same cache
        sol2 = solve(again, FunctionMap())
        @test any(!iszero, F1[2]) && all(map((a, b) -> same_bits(b, 2 .* a), F1, sol2.u[end].x))
    end
    @test partition_bits(solve(dprob, FunctionMap()).u[end]) == map(bits, F1)    # and back
    @test_throws ArgumentError solve(remake(dprob; p = params_case((1.0, 3.0, 0.02, 8.0); plasma = false, nsteps = 80, position = 10)), FunctionMap())
    @test_throws ArgumentError solve(remake(dprob; p = params(params_case((1.0f0, 3.0f0, 0.02f0, 8.0f0); plasma = false, nsteps = 80))), FunctionMap())
    # a PML's convolution fields are outside the partition: a second solve starts them at zero as the first did
    for T in TYPES
        prob, _ = open_plasma_2d(T; nsteps = 20)
        num = EMNumerics(; current = Esirkepov(), absorber = PML(; cells = 4, sigma_max = 1.5, order = 3, kappa_max = 2, kappa_order = 2, alpha_max = 0.2))
        dp = DiscreteProblem(prob, num; backend, save_everystep = false)
        a = solve(dp, FunctionMap())
        ua, ha = partition_bits(a.u[end]), history_bits(harness_state(a))
        @test any(!iszero, Array(harness_state(a).aux.layer.ψEx))
        b = solve(dp, FunctionMap())
        ref = loop_state(prob, num)
        @test partition_bits(b.u[end]) == ua == state_bits(ref)
        @test history_bits(harness_state(b)) == ha == history_bits(ref)
    end
end
