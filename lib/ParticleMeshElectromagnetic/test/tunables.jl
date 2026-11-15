# WP-059: the differentiable parameters of a problem description as one vector, and as the
# `Tunable` portion of a SciML structure, T-770 to T-773. After lwfa_gradient.jl (the WP-043 case
# and T-581's gradient) and window_cases.jl (`ZERO3`).

using SciMLBase: SciMLBase, DiscreteProblem, remake, solve
using OrdinaryDiffEqFunctionMap: FunctionMap
using SciMLStructures: SciMLStructures, Tunable, Constants, Discrete, Caches, Initials

"A 2D description, periodic along x and open along y, with every tunable a different number and no part at its default: two lasers, two species, background fields, a witness bunch."
function tunables_periodic(::Type{T}) where {T}
    cells = (32, 8)
    h = 2 * T(π) / 8
    l1 = GaussianLaser(; a0 = T(2), waist = T(5), duration = T(3), delay = T(18), focus = T(7), chirp = T(1) / 8, polarization = T(1) / 4,
                       phase = T(1) / 2, position = 4h, center = 3h)
    l2 = GaussianLaser(; a0 = T(3) / 2, waist = T(6), duration = T(4), delay = T(20), focus = T(9), chirp = T(1) / 16, polarization = T(3) / 4,
                       phase = T(5) / 8, position = 8h, center = 5h, envelope = WavefrontTime())
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(1) / 10, T(1) / 32, (1, 0), (2, 2), (T(1) / 64, T(2) / 64, T(3) / 64),
                              (T(5) / 64, T(6) / 64, T(7) / 64), 1)
    i = SpeciesDescription{T}(:ion, one(T), T(1836), T(3) / 10, T(3) / 32, (2, 1), (1, 2), (T(9) / 64, T(10) / 64, T(11) / 64),
                              (T(13) / 64, T(13) / 64, T(13) / 64), 2)
    wb = WitnessBunch(; inject = 4.0, region = ((2.0, 6.0), (1.0, 3.0)), ppc = (2, 3), momentum = (T(3), T(1) / 2, T(1) / 3),
                      center = (T(4), T(2)), length = T(5) / 4, width = T(7) / 4, charge = T(-2), mass = T(3))
    return EMProblem("tunables-periodic", (cells[1] * h, cells[2] * h), cells, T(95) / 100 * h / sqrt(T(2)), 20, (e, i), 3, (2, 1);
                     E0 = (T(1) / 16, T(2) / 16, T(3) / 16), B0 = (T(5) / 16, T(6) / 16, T(7) / 16), lasers = (l1, l2),
                     boundaries = (:periodic, :open), witness = wb)
end

"""
A moving-window 2D description with a density ramp and a witness bunch (the two optional parts
together): one laser and one species, drifting and warm in a window that starts at 0, or cold and
at rest in one that starts at `start` > 0.
"""
function tunables_window(::Type{T}; start = 0) where {T}
    cells = (32, 8)
    h = 2 * T(π) / 8
    pulse = GaussianLaser(; a0 = T(2), waist = T(5), duration = T(3), delay = T(18), focus = T(7), position = 4h, center = 4h)
    drift, thermal = start > 0 ? (ZERO3(T), ZERO3(T)) : ((T(1) / 64, T(2) / 64, T(3) / 64), (T(5) / 64, T(6) / 64, T(7) / 64))
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(1) / 10, zero(T), (1, 0), (2, 2), drift, thermal, 1; momentum = :gaussian_velocity)
    wb = WitnessBunch(; inject = 4.0, region = ((2.0, 6.0), (1.0, 3.0)), momentum = T(3), center = (T(4), T(2)), length = T(5) / 4)
    return EMProblem("tunables-window", (cells[1] * h, cells[2] * h), cells, T(95) / 100 * h / sqrt(T(2)), 20, (e,), 3, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :periodic), window = MovingWindow(; start = T(start)),
                     profile = DensityRamp(; vacuum = T(6), ramp = T(4)), witness = wb)
end
tunables_window_late(::Type{T}) where {T} = tunables_window(T; start = 37)

"The three 2D descriptions of T-770 to T-772, each with the names of its tunables."
const TUNABLE_FIXTURES = ((tunables_periodic, () -> PERIODIC_NAMES), (tunables_window, () -> WINDOW_NAMES), (tunables_window_late, () -> WINDOW_NAMES))

# The tunables of a description read field by field, written out a second time, independently of
# `ParticleMeshElectromagnetic.part_values`: what the vector must hold, and in which order.
laser_fields(l) = [l.a0, l.waist, l.duration, l.delay, l.focus, l.chirp, l.polarization, l.phase]
species_fields(d) = [d.density, d.amplitude, d.drift[1], d.drift[2], d.drift[3], d.thermal[1], d.thermal[2], d.thermal[3]]
witness_fields(b) = [b.momentum[1], b.momentum[2], b.momentum[3], b.center[1], b.center[2], b.length, b.width]
function expected_tunables(p)
    v = eltype(p)[]
    foreach(l -> append!(v, laser_fields(l)), p.lasers)
    foreach(d -> append!(v, species_fields(d)), p.species)
    p.profile === nothing || append!(v, [p.profile.vacuum, p.profile.ramp])
    p.witness === nothing || append!(v, witness_fields(p.witness))
    return v
end

const PERIODIC_NAMES = [:laser1_a0, :laser1_waist, :laser1_duration, :laser1_delay, :laser1_focus, :laser1_chirp, :laser1_polarization,
                        :laser1_phase, :laser2_a0, :laser2_waist, :laser2_duration, :laser2_delay, :laser2_focus, :laser2_chirp,
                        :laser2_polarization, :laser2_phase, :species1_density, :species1_amplitude, :species1_drift_x, :species1_drift_y,
                        :species1_drift_z, :species1_thermal_x, :species1_thermal_y, :species1_thermal_z, :species2_density,
                        :species2_amplitude, :species2_drift_x, :species2_drift_y, :species2_drift_z, :species2_thermal_x,
                        :species2_thermal_y, :species2_thermal_z, :witness_momentum_x, :witness_momentum_y, :witness_momentum_z,
                        :witness_center_x, :witness_center_y, :witness_length, :witness_width]

const WINDOW_NAMES = [:laser1_a0, :laser1_waist, :laser1_duration, :laser1_delay, :laser1_focus, :laser1_chirp, :laser1_polarization,
                      :laser1_phase, :species1_density, :species1_amplitude, :species1_drift_x, :species1_drift_y, :species1_drift_z,
                      :species1_thermal_x, :species1_thermal_y, :species1_thermal_z, :profile_vacuum, :profile_ramp, :witness_momentum_x,
                      :witness_momentum_y, :witness_momentum_z, :witness_center_x, :witness_center_y, :witness_length, :witness_width]

# What `with_tunables` must leave as it was: everything of a description that is not a tunable.
laser_rest(l) = (l.position, l.center, l.envelope)
species_rest(d) = (d.name, d.charge, d.mass, d.mode, d.ppc, d.stream, d.momentum)
witness_rest(b) = b === nothing ? nothing : (b.inject, b.region, b.ppc, b.charge, b.mass)
problem_rest(p) = (p.name, p.length, p.cells, p.dt, p.nsteps, p.seed, p.kmode, p.E0, p.B0, p.boundaries, p.window, map(laser_rest, p.lasers),
                   map(species_rest, p.species), witness_rest(p.witness))

# New values for a description: every entry changed by its own factor, zeros kept (a window's
# amplitude must stay zero), the isotropic thermal speeds of the Maxwell–Jüttner ion kept equal.
function shifted(names, v::Vector{T}) where {T}
    w = [x * (1 + T(k) / 128) for (k, x) in enumerate(v)]
    iso = findall(n -> n in (:species2_thermal_x, :species2_thermal_y, :species2_thermal_z), names)
    isempty(iso) || (w[iso] .= w[first(iso)])
    return w
end

@testset "T-770 the tunables of a description: names, order, and the round trip through the vector [A3, A8, WP-059]" begin
    for T in TYPES, (build, getnames) in TUNABLE_FIXTURES
        p = build(T)
        names = getnames()
        v = tunables(p)
        @test v isa Vector{T}
        @test tunable_names(p) == names
        @test tunable_count(p) == length(names) == length(v)
        @test same_bits(v, expected_tunables(p))
        @test with_tunables(p, v) === p                         # egal: every field, with its type and its bits
        # the vector written back: each entry lands in its own field, nothing else moves
        w = shifted(names, v)
        @test !any(k -> !iszero(v[k]) && isfinite(v[k]) && w[k] == v[k], eachindex(v))
        q = with_tunables(p, w)
        @test typeof(q) === typeof(p)
        @test same_bits(expected_tunables(q), w)
        @test same_bits(tunables(q), w)
        @test problem_rest(q) === problem_rest(p)
        # the parameters of the description and the description's parameters: the two interfaces agree
        ps = params(p)
        @test tunable_names(ps) == names
        @test same_bits(tunables(ps), v)
        @test with_tunables(ps, v) === ps
        @test with_tunables(ps, w) === params(q)
        # one entry at a time
        for k in eachindex(v)
            u = copy(v)
            u[k] = w[k]
            names[k] in (:species2_thermal_x, :species2_thermal_y, :species2_thermal_z) && continue   # an anisotropic Maxwell–Jüttner load is refused, below
            @test findall(!, expected_tunables(with_tunables(p, u)) .=== v) == (w[k] === v[k] ? Int[] : [k])
        end
    end
    # a 1D description: species only
    for T in TYPES
        p = problem_from_case(load_case("two-stream"), T)
        @test ndims(p) == 1
        v = tunables(p)
        @test tunable_names(p) == [Symbol(:species, i, :_, n) for i in eachindex(p.species) for n in
                                   (:density, :amplitude, :drift_x, :drift_y, :drift_z, :thermal_x, :thermal_y, :thermal_z)]
        @test same_bits(v, expected_tunables(p))
        @test with_tunables(p, v) === p
        @test with_tunables(params(p), v) === params(p)
        w = [x * (1 + T(k) / 128) + T(k) / 1024 for (k, x) in enumerate(v)]      # zeros move too
        q = with_tunables(p, w)
        @test same_bits(expected_tunables(q), w) && same_bits(tunables(q), w)
        @test problem_rest(q) === problem_rest(p)
        @test with_tunables(params(p), w) === params(q)
    end
    let p = problem_from_case(load_case("two-stream"), Float64), v = tunables(p)
        q = with_tunables(p, Float32.(v))
        @test eltype(q) === Float32 && ndims(q) == 1
        @test same_bits(expected_tunables(q), Float32.(v))
        @test q.length === Float32(p.length) && q.kmode === Float32(p.kmode) && q.dt === Float32(p.dt)
        @test with_tunables(params(p), Float32.(v)) === params(q)
    end
    # another element type: the vector's, for every number of the description; what is
    # configuration in a type of its own (a laser's position and centre, the witness's injection
    # time and region) stays as it is
    for (build, _) in TUNABLE_FIXTURES, S in (Float32, D3)
        p = build(Float64)
        v = tunables(p)
        q = with_tunables(p, S.(v))
        @test eltype(q) === S
        @test tunables(q) isa Vector{S}
        @test same_bits(tunables(q), S.(v)) && same_bits(expected_tunables(q), S.(v))
        @test q.dt === S(p.dt) && q.length === S.(p.length) && q.E0 === S.(p.E0) && q.B0 === S.(p.B0)
        @test (q.name, q.cells, q.nsteps, q.seed, q.kmode, q.boundaries) === (p.name, p.cells, p.nsteps, p.seed, p.kmode, p.boundaries)
        @test (q.window === nothing) == (p.window === nothing) && (p.window === nothing || q.window.start === S(p.window.start))
        @test map(d -> (d.charge, d.mass), q.species) === map(d -> (S(d.charge), S(d.mass)), p.species)
        @test map(species_rest, q.species) == map(species_rest, p.species)
        @test (q.witness.charge, q.witness.mass) === (S(p.witness.charge), S(p.witness.mass))
        @test (q.witness.inject, q.witness.region, q.witness.ppc) === (p.witness.inject, p.witness.region, p.witness.ppc)
        @test map(laser_rest, q.lasers) === map(laser_rest, p.lasers)
        @test eltype(params(q)) === S
        @test with_tunables(params(p), S.(v)) === params(q)
    end
    @test tunables_periodic(Float64).witness.charge == -2 && tunables_periodic(Float64).species[2].mass == 1836
    # duals: the partials are the vector's
    dv = [D3(x, ForwardDiff.Partials((Float32(k), 0.0f0, 1.0f0))) for (k, x) in enumerate(tunables(tunables_periodic(Float32)))]
    iso = findall(n -> n in (:species2_thermal_x, :species2_thermal_y, :species2_thermal_z), PERIODIC_NAMES)
    @test_throws ArgumentError with_tunables(tunables_periodic(Float32), dv)   # one temperature with three different partials is not isotropic
    dv[iso] .= dv[first(iso)]
    q = with_tunables(tunables_periodic(Float32), dv)
    @test same_bits(tunables(q), dv)                                  # the partials are carried
    @test ForwardDiff.value.(tunables(q)) == tunables(tunables_periodic(Float32))
    @test with_tunables(q, tunables(q)) === q
    # a dual description is not written back in a type that cannot hold its numbers
    for x in (q, params(q))
        @test_throws ArgumentError with_tunables(x, ForwardDiff.value.(tunables(q)))
        @test_throws ArgumentError with_tunables(x, ForwardDiff.Dual{:other,Float32,3}.(ForwardDiff.value.(tunables(q))))
    end
    @test q.dt === D3(tunables_periodic(Float32).dt)                  # the configuration carries no partial
end

@testset "T-771 what the vector refuses [A3, WP-059]" begin
    p = tunables_periodic(Float64)
    v = tunables(p)
    for x in (p, params(p))
        @test_throws ArgumentError with_tunables(x, v[1:(end - 1)])
        @test_throws ArgumentError with_tunables(x, vcat(v, 1.0))
        @test_throws ArgumentError with_tunables(x, Real[v...])               # not a concrete element type
        @test_throws ArgumentError with_tunables(x, Any[v...])
        @test_throws ArgumentError with_tunables(x, complex.(v))
        @test_throws ArgumentError with_tunables(x, round.(Int, v))
    end
    # a part in another element type than the description's
    e32 = SpeciesDescription{Float32}(:electron, -1, 1, 0.1, 0, (1, 0), (2, 2), ZERO3(Float32), ZERO3(Float32), 1)
    mixed = EMProblem("mixed", p.length, p.cells, p.dt, 5, (e32,), 1, (1, 0))
    @test_throws ArgumentError tunables(mixed)
    @test_throws ArgumentError tunables(params(mixed))
    l32 = GaussianLaser(; a0 = 1.0f0, waist = 5.0f0, duration = 3.0f0, delay = 18.0f0, focus = 7.0f0, position = 3.0, center = 3.0)
    @test_throws ArgumentError tunables(EMProblem("mixed", p.length, p.cells, p.dt, 5, (), 1, (1, 0); lasers = (l32,)))
    # values the description's constructors refuse
    names = tunable_names(p)
    set(x, name, val) = (u = copy(v); u[findfirst(==(name), names)] = val; with_tunables(x, u))
    @test_throws ArgumentError set(p, :species2_thermal_y, 0.5)             # the Maxwell–Jüttner ion is isotropic
    w = tunables_window(Float64)
    wv, wn = tunables(w), tunable_names(w)
    wset(name, val) = (u = copy(wv); u[findfirst(==(name), wn)] = val; with_tunables(w, u))
    @test_throws ArgumentError wset(:profile_ramp, 0.0)
    @test_throws ArgumentError wset(:profile_ramp, -1.0)
    @test_throws ArgumentError wset(:species1_amplitude, 0.1)               # a window's plasma starts neutral
    # a description with no tunable
    empty = EMProblem("empty", p.length, p.cells, p.dt, 5, (), 1, (1, 0))
    @test tunables(empty) == Float64[] && tunable_names(empty) == Symbol[] && tunable_count(empty) == 0
    @test with_tunables(empty, Float64[]) === empty
    @test_throws ArgumentError with_tunables(empty, [1.0])
end

@testset "T-772 the Tunable portion of the SciML structure is the vector [A9, WP-059]" begin
    for T in TYPES, p in (tunables_periodic(T), tunables_window(T), tunables_window_late(T)), x in (p, params(p))
        @test SciMLStructures.isscimlstructure(x)
        @test !SciMLStructures.ismutablescimlstructure(x)
        @test SciMLStructures.hasportion(Tunable(), x)
        @test !any(q -> SciMLStructures.hasportion(q, x), (Constants(), Discrete(), Caches(), Initials()))
        v, repack, aliases = SciMLStructures.canonicalize(Tunable(), x)
        @test same_bits(v, tunables(x)) && v isa Vector{T}
        @test aliases === false
        @test repack(v) === x
        w = shifted(tunable_names(x), v)
        @test repack(w) === with_tunables(x, w) === SciMLStructures.replace(Tunable(), x, w)
        @test same_bits(SciMLStructures.canonicalize(Tunable(), repack(w))[1], w)
    end
    # a vector of another element type gives the structure in that type, by repack and by replace
    for p in (tunables_periodic(Float64), tunables_window_late(Float64)), x in (p, params(p)), S in (Float32, D3)
        v, repack, _ = SciMLStructures.canonicalize(Tunable(), x)
        @test eltype(repack(S.(v))) === S
        @test repack(S.(v)) === with_tunables(x, S.(v)) === SciMLStructures.replace(Tunable(), x, S.(v))
    end
    @test !SciMLStructures.isscimlstructure(EMNumerics())
end

# The gradient through a solve, with the parameters taken from and written to the problem by the
# SciML interface alone: the five parameters of WP-043 found by name in the Tunable portion,
# replaced by duals, the problem reloaded (`remake_reload`: the density and the ramp enter through
# the initial load) and solved. It is the same arithmetic as T-581's hand-built dual problem, so
# the gradient has T-581's bits.
@testset "T-773 the gradient through solve by the Tunable portion has the bits of the hand-built dual run [A3, A7, WP-059, WP-043]" begin
    x0 = collect(LWFA_GRADIENT_X0)
    base = lwfa_gradient_problem(LWFA_GRADIENT_X0)
    dprob = DiscreteProblem(base, EMNumerics(; mode = Reference()))
    v, _, _ = SciMLStructures.canonicalize(Tunable(), dprob.p)
    idx = map(n -> findfirst(==(n), tunable_names(dprob.p)), (:laser1_a0, :laser1_duration, :laser1_focus, :species1_density, :profile_ramp))
    @test v[collect(idx)] == x0
    function J(x)
        w = convert(Vector{eltype(x)}, v)
        w[collect(idx)] .= x
        sol = solve(remake_reload(dprob; p = SciMLStructures.replace(Tunable(), dprob.p, w)), FunctionMap(); save_everystep = false)
        return soft_maximum(axis_Ex(harness_state(sol)), LWFA_GRADIENT_BETA)
    end
    @test bits([J(x0)]) == bits([lwfa_wake_amplitude(LWFA_GRADIENT_X0)])
    ad = ForwardDiff.gradient(J, x0)
    ref = T581_AD[] === nothing ? ForwardDiff.gradient(x -> lwfa_wake_amplitude(x), x0) : T581_AD[]
    @info "T-773 gradient by the Tunable portion" ad ref
    @test all(!iszero, ad)
    @test bits(ad) == bits(ref)
    # a plain remake keeps u0 and the cache: with parameters of another element type it is refused, not run
    D = ForwardDiff.Dual{Nothing,Float64,1}
    w = D.(v)
    w[idx[1]] = D(v[idx[1]], ForwardDiff.Partials((1.0,)))
    for p in (with_tunables(dprob.p, w), with_tunables(params(dprob.p), w))
        @test_throws ArgumentError solve(remake(dprob; p), FunctionMap(); save_everystep = false)
    end
end
