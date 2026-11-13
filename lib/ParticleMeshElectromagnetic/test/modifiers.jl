# WP-057: the one field solve and its tuple of field modifiers, T-750 to T-754 and T-759. The solve walks
# whatever tuple it is handed, around the three Yee sub-steps, so the tests add modifiers of their
# own (defined here, not in the package) and hold the solve against the sub-steps written out. After
# params.jl (`params_case`), restart.jl (`restart_full`) and boundaries_cases.jl.

using ParticleMeshElectromagnetic: resolve_params, with_modifiers, pml_b!, pml_e!, back_b!, back_e!, antenna_b!, antenna_e!,
                                   absorb_b!, absorb_e!, zero_x_edges!, window_antennas, window_frame, LayerFactors, PMLFields, BackPMLFields
const PME = ParticleMeshElectromagnetic

# --- modifiers of the tests' own -------------------------------------------------------------------

"A modifier with no method of its own: every hook is the interface's default."
struct Idle <: FieldModifier end

"""
After a sub-step, `X ← a·X + c` on every entry (guards included) of the components it updated.
With `a = 2` it does not commute with a correction that adds to the field, and with `c ≠ 0` it
shows in a guard that a later modifier sets to zero; with `a = 1`, `c = 0` it changes nothing.
"""
struct Affine{T} <: FieldModifier
    a::T
    c::T
end
affine!(m::Affine, x) = (x .= m.a .* x .+ m.c; nothing)
PME.after_substep!(m::Affine, mp, f, grid, ::BHalf, wgs) = (affine!(m, f.Bx); affine!(m, f.By); affine!(m, f.Bz))
PME.after_substep!(m::Affine, mp, f, grid, ::EStep, wgs) = (affine!(m, f.Ex); affine!(m, f.Ey); affine!(m, f.Ez))

"A modifier with a parameter half: before each sub-step it scales the components the sub-step updates by `params.scale`."
struct Scaled <: FieldModifier end
PME.modifier_params(::Scaled, p, k) = (p.scale, k)
PME.before_substep!(::Scaled, mp, f, grid, ::BHalf, wgs) = (f.Bx .*= mp[1]; f.By .*= mp[1]; f.Bz .*= mp[1]; nothing)
PME.before_substep!(::Scaled, mp, f, grid, ::EStep, wgs) = (f.Ex .*= mp[1]; f.Ey .*= mp[1]; f.Ez .*= mp[1]; nothing)

"A modifier that changes nothing and writes down, on the host, every call the solve makes of it, with the fields' bits at that moment."
struct Spy <: FieldModifier
    id::Int
    log::Vector{Any}
end
field_bits(f) = map(bits, values(f))
PME.modifier_params(m::Spy, p, k) = (push!(m.log, (m.id, :params, p, k)); (m.id, p, k))
PME.before_substep!(m::Spy, mp, f, grid, s, wgs) = (push!(m.log, (m.id, :before, s, mp, field_bits(f))); nothing)
PME.after_substep!(m::Spy, mp, f, grid, s, wgs) = (push!(m.log, (m.id, :after, s, mp, field_bits(f))); nothing)

"Random fields of the layout on `grid`, guards filled."
function random_fields(be, ::Type{T}, grid, seed) where {T}
    rng = Xoshiro(seed)
    f = em_fields(CPU(), T, grid)
    for a in values(f)
        interior_view(a, grid) .= seeded.(T, rand(rng, size(interior_view(a, grid))...) .- 1 // 2)
        fill_guards!(a, grid)
    end
    return map(a -> copyto!(KernelAbstractions.allocate(be, T, size(a)...), a), f)
end
same_fields(a, b) = all(map(same_bits, values(a), values(b)))
test_grid(::Type{T}, N) where {T} = N == 1 ? PeriodicGrid(12, T(1 // 2), 3) : PeriodicGrid((12, 10), (T(1 // 2), T(2 // 3)), 3)
test_dt(::Type{T}) where {T} = T(3 // 10)

# --- T-750: the walk, and a test's modifier among the package's --------------------------------------

"""
The solve of a window state with a PML on the open axis (`L`), a PML on the back edge (`B`),
antennas and the window's edges, written out sub-step by sub-step: the order of the package's
solve for this configuration before WP-057 (the method of `pml.jl`), which the digest holds the
tuple walk to. `extra(slot, components)` is called at slot `j` of each sub-step's corrections,
`j = 0` right after the Yee update and `j = 4` after the edges.
"""
function written_out_pml!(f, grid, dt, L, B, antennas, edges, t, extra)
    T = eltype(f.Ex)
    h = T(dt) / 2
    bs, es = (f.Bx, f.By, f.Bz), (f.Ex, f.Ey, f.Ez)
    for (adv, tb) in ((false, t), (true, t + T(dt)))
        yee_b_half!(f, grid, h)
        extra(0, bs)
        pml_b!(L, f, grid, h, adv, nothing)
        extra(1, bs)
        back_b!(B, f, grid, h, adv, nothing)
        extra(2, bs)
        antenna_b!(antennas, f, h, tb, nothing)
        extra(3, bs)
        foreach(a -> zero_x_edges!(edges, a), bs)
        extra(4, bs)
        adv && break
        yee_e!(f, grid, dt)
        extra(0, es)
        pml_e!(L, f, grid, dt, nothing)
        extra(1, es)
        back_e!(B, f, grid, dt, nothing)
        extra(2, es)
        antenna_e!(antennas, f, dt, t + h, nothing)
        extra(3, es)
        foreach(a -> zero_x_edges!(edges, a), es)
        extra(4, es)
    end
    return f
end

"The same with an absorbing layer (`A`) on the open axis in place of the PML: the order of the method of `window.jl` before WP-057."
function written_out_layer!(f, grid, dt, A, B, antennas, edges, t, extra)
    T = eltype(f.Ex)
    h = T(dt) / 2
    bs, es = (f.Bx, f.By, f.Bz), (f.Ex, f.Ey, f.Ez)
    for (adv, tb) in ((false, t), (true, t + T(dt)))
        absorb_b!(A, f, nothing)
        yee_b_half!(f, grid, h)
        extra(0, bs)
        back_b!(B, f, grid, h, adv, nothing)
        extra(1, bs)
        antenna_b!(antennas, f, h, tb, nothing)
        extra(2, bs)
        absorb_b!(A, f, nothing)
        extra(3, bs)
        foreach(a -> zero_x_edges!(edges, a), bs)
        extra(4, bs)
        adv && break
        absorb_e!(A, f, nothing)
        yee_e!(f, grid, dt)
        extra(0, es)
        back_e!(B, f, grid, dt, nothing)
        extra(1, es)
        antenna_e!(antennas, f, dt, t + h, nothing)
        extra(2, es)
        absorb_e!(A, f, nothing)
        extra(3, es)
        foreach(a -> zero_x_edges!(edges, a), es)
        extra(4, es)
    end
    return f
end

"A window run with an open axis 2, a laser polarized between y and z and a PML on the back edge, `k0` steps in, while the antenna emits."
function modifier_state(::Type{T}, absorber, be; k0 = 30) where {T}
    prob = params_case((seeded(T, 1), T(3), seeded(T, 2 // 100), T(8)); nsteps = 100, cells = (72, 16), position = 10, open_y = true, polarization = 1)
    st = EMState(prob, EMNumerics(; absorber, back_absorber = PML(; cells = 6)); backend = be)
    initialize!(st, prob)
    run!(st, params(prob), k0; observer = nothing)
    return st, params(prob)
end

"Everything a solve of `st` writes: the fields and the convolution fields of its PMLs."
solve_arrays(st) = (values(st.fields)..., evolved_arrays(open_absorber(st))..., evolved_arrays(back_absorber(st))...)
same_solve(a, b) = all(map(same_bits, solve_arrays(a), solve_arrays(b)))

function body_T750(be)
    # (a) the walk: three spies and an idle modifier around the three sub-steps, at N = 1 and N = 2
    for N in (1, 2), T in TYPES
        grid, dt, k = test_grid(T, N), test_dt(T), 4
        f = random_fields(be, T, grid, 50 + N)
        g = map(copy, f)
        h, t = dt / 2, (k - 1) * dt
        steps = (BHalf(h, t, false), EStep(dt, t + h), BHalf(h, t + dt, true))
        seen = Any[field_bits(g)]                      # the fields before and after each sub-step, written out
        for (s, update!) in zip(steps, (yee_b_half!, yee_e!, yee_b_half!))
            update!(g, grid, s.τ)
            push!(seen, field_bits(g))
        end
        log = Any[]
        token = (; name = :token)
        solve_fields!(Yee(), f, grid, dt, (Spy(1, log), Idle(), Spy(2, log), Spy(3, log)), token, k)
        expected = Any[(i, :params, token, k) for i in 1:3]
        for (j, s) in enumerate(steps), (hook, when) in ((:before, j), (:after, j + 1)), i in 1:3
            push!(expected, (i, hook, s, (i, token, k), seen[when]))
        end
        @test length(log) == 3 + 18
        @test log == expected
        @test all(map((a, b) -> typeof(a) === typeof(b), log, expected))      # the sub-steps' lengths and times in the fields' type
        @test same_fields(f, g)                                               # spies and an idle modifier change nothing
    end
    # (b) the package's order, and a test's modifier at every place of it: with a PML on the open
    # axis, and with an absorbing layer
    for T in TYPES, (absorber, written_out!) in ((PML(; cells = 4, alpha_max = 0.2), written_out_pml!),
                                                 (AbsorbingLayer(; cells = 4, strength = 1), written_out_layer!))
        k = 31
        st, p = modifier_state(T, absorber, be)
        ms = st.modifiers
        @test length(ms) == 4 && ms[end] isa WindowEdges
        antennas = window_antennas(map(Antenna, p.lasers, antenna_planes(st)), window_shifts(st, k - 1))
        @test only(antennas).ia == only(antenna_planes(st)).ia - window_shifts(st, k - 1)
        t = (k - 1) * st.dt
        bump = Affine(T(2), seeded(T, 1 // 8))
        results = []
        for j in (nothing, 0, 1, 2, 3, 4)
            a, _ = modifier_state(T, absorber, be)
            b, _ = modifier_state(T, absorber, be)
            @test same_solve(a, b)
            ma = j === nothing ? a.modifiers : (a.modifiers[1:j]..., bump, a.modifiers[(j + 1):end]...)
            solve_fields!(Yee(), a.fields, a.grid, a.dt, ma, p, k)
            extra(slot, comps) = slot === j && foreach(x -> affine!(bump, x), comps)
            written_out!(b.fields, b.grid, b.dt, open_absorber(b), back_absorber(b), antennas, window_frame(b), t, extra)
            @test same_solve(a, b)
            push!(results, map(x -> bits(Array(x)), solve_arrays(a)))
        end
        @test allunique(results)                                      # the modifier acts, and every place of it is another result
        # through step!: the state's own tuple is walked. Last in the tuple, the modifier leaves its
        # constant in the x-guards the edges have just zeroed; before the edges, they stay zero
        gcols = (1:guards(st.grid)..., (size(st.fields.Bz, 1) - guards(st.grid) + 1):size(st.fields.Bz, 1)...)
        for (place, guard) in ((length(ms), bump.c), (length(ms) - 1, zero(T)))
            a, _ = modifier_state(T, absorber, be)
            a2 = with_modifiers(a, (a.modifiers[1:place]..., bump, a.modifiers[(place + 1):end]...))
            step!(a2, p, k; observer = nothing)
            @test all(c -> same_bits(Array(getfield(a.fields, c))[collect(gcols), :], fill(guard, length(gcols), size(a.fields.Bz, 2))),
                      (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
        end
        # idle modifiers anywhere in a state's tuple leave its step's bits
        a, _ = modifier_state(T, absorber, be)
        b, _ = modifier_state(T, absorber, be)
        step!(with_modifiers(a, (Idle(), a.modifiers[1:2]..., Idle(), a.modifiers[3:end]..., Idle())), p, k; observer = nothing)
        step!(b, p, k; observer = nothing)
        @test isempty(differences(a, b, k; cache = false)) && same_solve(a, b)
    end
end

@testset "T-750 the solve walks its tuple of modifiers around each sub-step, and a test's own modifier acts at its place among the package's [A2, A9, WP-057]" begin
    body_T750(CPU())
end

# --- T-751: no modifiers ---------------------------------------------------------------------------

function body_T751(be)
    for N in (1, 2), T in TYPES
        grid, dt = test_grid(T, N), test_dt(T)
        f = random_fields(be, T, grid, 60 + N)
        f4, f7, g = map(copy, f), map(copy, f), map(copy, f)
        for k in 1:5
            solve_fields!(Yee(), f4, grid, dt)
            solve_fields!(Yee(), f7, grid, dt, (), nothing, k)
            yee_b_half!(g, grid, dt / 2)                # the periodic solve, written out
            yee_e!(g, grid, dt)
            yee_b_half!(g, grid, dt / 2)
        end
        @test same_fields(f4, g) && same_fields(f7, g)
        @test !same_fields(f, g)
    end
end

@testset "T-751 the solve with no modifiers is the periodic solve bit for bit, and a periodic state holds none [A2, A4, WP-057]" begin
    body_T751(CPU())
    st1, _, _, _ = case_state("weibel", Float64; ppc = 4, t_end = 1)
    @test st1 isa EMState{1} && st1.modifiers === ()
    p2 = plasma_step_2d(Float64)
    @test state_2d(p2; current = Esirkepov()).modifiers === ()
    # a non-modifier is not taken into a state, nor walked
    @test_throws MethodError with_modifiers(st1, (1,))
    @test_throws MethodError solve_fields!(Yee(), st1.fields, st1.grid, st1.dt, (1,), nothing, 1)
end

# --- T-752: no dynamic dispatch and no allocation, at any length ---------------------------------------

function solve_bytes(f, grid, dt, ms, p)
    solve_fields!(Yee(), f, grid, dt, ms, p, 1)
    solve_fields!(Yee(), f, grid, dt, ms, p, 2)
    return @allocated solve_fields!(Yee(), f, grid, dt, ms, p, 3)
end
function state_step_bytes(st, p)
    step!(st, p, 31; observer = nothing)
    step!(st, p, 32; observer = nothing)
    return @allocated step!(st, p, 33; observer = nothing)
end

@testset "T-752 walking a tuple of modifiers costs the solve and the step no allocation, at 1, 5, 33 and 40 modifiers of mixed types [A9, WP-057]" begin
    one_ = Affine(1.0, 0.0)
    p = (; scale = 1.0)
    for N in (1, 2)
        grid, dt = test_grid(Float64, N), test_dt(Float64)
        f = random_fields(CPU(), Float64, grid, 70 + N)
        base = solve_bytes(f, grid, dt, (), p)
        sizes = Pair{Int,Int}[]
        for n in (1, 5, 33, 40)
            idle = ntuple(_ -> Idle(), n)
            mixed = ntuple(i -> i % 3 == 0 ? one_ : i % 3 == 1 ? Idle() : Scaled(), n)     # a tuple of n elements of three types
            @test !isconcretetype(eltype(mixed)) || n == 1
            @test_alloc solve_bytes(f, grid, dt, idle, p) == base
            push!(sizes, n => solve_bytes(f, grid, dt, mixed, p))
            @test_alloc last(sizes[end]) == base
            @test @inferred(resolve_params(mixed, p, 3)) === ntuple(i -> i % 3 == 2 ? (1.0, 3) : nothing, n)
        end
        @info "T-752 bytes per solve" N base sizes = join(sizes, ", ")
    end
    # a state's step: 36 modifiers of the test's around the package's four cost its step nothing
    if Threads.nthreads() == 1
        st, ps = modifier_state(Float64, PML(; cells = 4, alpha_max = 0.2), CPU())
        plain = state_step_bytes(st, ps)
        st2, _ = modifier_state(Float64, PML(; cells = 4, alpha_max = 0.2), CPU())
        many = with_modifiers(st2, (ntuple(_ -> Idle(), 18)..., st2.modifiers..., ntuple(i -> isodd(i) ? one_ : Idle(), 18)...))
        @test length(many.modifiers) == 40
        loaded = state_step_bytes(many, ps)
        @info "T-752 bytes per step" plain loaded
        @test_alloc loaded == plain
    end
    # the accessors are inferred, for the package's tuples and past 32 elements
    st, _ = modifier_state(Float64, PML(; cells = 4), CPU())
    @test @inferred(open_absorber(st)) isa PMLFields && @inferred(back_absorber(st)) isa BackPMLFields && @inferred(antenna_planes(st)) isa Tuple{AntennaPlane{Float64}}
    long = with_modifiers(st, (ntuple(_ -> Idle(), 35)..., st.modifiers...))
    @test @inferred(open_absorber(long)) === open_absorber(st) && @inferred(back_absorber(long)) === back_absorber(st) &&
          @inferred(antenna_planes(long)) === antenna_planes(st)
end

# --- T-753: the parameter half ---------------------------------------------------------------------------

function body_T753(be)
    for T in TYPES
        st, p = modifier_state(T, PML(; cells = 4), be; k0 = 0)
        plane = only(antenna_planes(st))
        kshift = findfirst(k -> window_shifts(st, k) > 0, 1:200)
        for k in (1, 31, kshift, kshift + 1, kshift + 7)
            mps = @inferred resolve_params(st.modifiers, p, k)
            @test map(m -> m isa AntennaPlanes, st.modifiers) == map(mp -> mp !== nothing, mps)       # the antennas alone read parameters
            a = only(mps[findfirst(m -> m isa AntennaPlanes, st.modifiers)])
            @test isbits(a) && a.laser === only(p.lasers)
            @test a === Antenna(only(p.lasers), plane.ia - Int32(window_shifts(st, k - 1)), plane.xa, plane.dx, plane.g, plane.n2, plane.ly, plane.hx, plane.ht)
        end
        @test window_shifts(st, kshift - 1) == 0 && window_shifts(st, kshift) == 1
        # other parameters, the same state: the modifier hands on what it is given and has kept nothing
        l = only(p.lasers)
        l2 = GaussianLaser(; a0 = 2 * l.a0, waist = l.waist, duration = l.duration, delay = l.delay, focus = l.focus, position = l.position,
                           center = l.center, chirp = l.chirp, polarization = l.polarization, phase = l.phase, envelope = l.envelope)
        p2 = EMParams(p; lasers = (l2,))
        i = findfirst(m -> m isa AntennaPlanes, st.modifiers)
        @test only(resolve_params(st.modifiers, p2, 1)[i]).laser.a0 === 2 * l.a0
        @test only(resolve_params(st.modifiers, p, 1)[i]).laser.a0 === l.a0
        @test isbits(st.modifiers[i])                                 # planes and a clock: no array, no laser
        @test !any(x -> x isa GaussianLaser, (st.modifiers[i].planes..., st.modifiers[i].clock))
    end
    # a modifier of the test's with a parameter half: scaled by the parameters' factor before each
    # sub-step, against the sub-steps written out
    for N in (1, 2), T in TYPES
        grid, dt = test_grid(T, N), test_dt(T)
        f = random_fields(be, T, grid, 80 + N)
        g = map(copy, f)
        s = seeded(T, 3 // 4)
        solve_fields!(Yee(), f, grid, dt, (Idle(), Scaled()), (; scale = s), 2)
        foreach(a -> a .*= s, (g.Bx, g.By, g.Bz))
        yee_b_half!(g, grid, dt / 2)
        foreach(a -> a .*= s, (g.Ex, g.Ey, g.Ez))
        yee_e!(g, grid, dt)
        foreach(a -> a .*= s, (g.Bx, g.By, g.Bz))
        yee_b_half!(g, grid, dt / 2)
        @test same_fields(f, g)
    end
end

@testset "T-753 a modifier's parameter half comes from the step's parameters at its step, and the state keeps none of it [A3, WP-057]" begin
    body_T753(CPU())
end

# --- T-754: what a state holds ---------------------------------------------------------------------------

"Every array reachable from `x` through tuples, named tuples and the structs of the engine and the module."
reach!(out, x::AbstractArray) = (push!(out, x); out)
reach!(out, x::Union{Tuple,NamedTuple}) = (foreach(y -> reach!(out, y), x); out)
function reach!(out, x)
    parentmodule(typeof(x)) in (ParticleMeshCore, ParticleMeshElectromagnetic) || return out
    foreach(f -> reach!(out, getfield(x, f)), fieldnames(typeof(x)))
    return out
end
reach(x) = reach!(AbstractArray[], x)

@testset "T-754 a state holds each modifier once, in the solve's order, and a state over other arrays holds the PMLs over those [A4, WP-057]" begin
    T = Float64
    kinds(st) = map(m -> nameof(typeof(m)), st.modifiers)
    pml, layer, back = PML(; cells = 4), AbsorbingLayer(; cells = 4, strength = 1), PML(; cells = 6)
    win(; open_y = false, kw...) = params_case((1.0, 3.0, 0.02, 8.0); nsteps = 100, cells = (72, 16), position = 10, open_y, kw...)
    static, sl = open_plasma_2d(T)
    lasers = laser_box(T, (GaussianLaser(; a0 = 1.0, waist = 4.0, duration = 2.0, delay = 12.0, focus = 12.0, position = 6.0, center = 6.0),))
    @test kinds(EMState(static, EMNumerics(; absorber = sl))) == (:LayerFactors,)
    @test kinds(EMState(static, EMNumerics(; absorber = pml))) == (:PMLFields,)
    @test kinds(EMState(lasers, EMNumerics())) == (:AntennaPlanes,)
    @test kinds(EMState(win(), EMNumerics())) == (:AntennaPlanes, :WindowEdges)
    @test kinds(EMState(win(), EMNumerics(; back_absorber = back))) == (:BackPMLFields, :AntennaPlanes, :WindowEdges)
    @test kinds(EMState(win(; open_y = true), EMNumerics(; absorber = pml, back_absorber = back))) == (:PMLFields, :BackPMLFields, :AntennaPlanes, :WindowEdges)
    @test kinds(EMState(win(; open_y = true), EMNumerics(; absorber = layer, back_absorber = back))) == (:BackPMLFields, :AntennaPlanes, :LayerFactors, :WindowEdges)
    @test kinds(EMState(win(; open_y = true), EMNumerics(; absorber = layer))) == (:AntennaPlanes, :LayerFactors, :WindowEdges)
    nolaser = EMProblem("no laser", win().length, win().cells, win().dt, 10, win().species, 1, (1, 0); boundaries = (:window, :open),
                        window = MovingWindow(; start = 0.0))
    @test kinds(EMState(nolaser, EMNumerics(; absorber = pml))) == (:PMLFields, :WindowEdges)
    # the constructor refuses an antenna plane inside the back edge's PML, and one the layer reaches
    # while it emits, before any parameters are checked (window_lwfa's plane is columns 8 and 9)
    @test_throws ArgumentError EMState(window_lwfa(T; nsteps = 4), EMNumerics(; back_absorber = PML(; cells = 8)))
    @test_throws ArgumentError EMState(window_lwfa(T; nsteps = 4), EMNumerics(; back_absorber = PML(; cells = 6)))
    @test EMState(window_lwfa(T; nsteps = 4), EMNumerics(; back_absorber = PML(; cells = 4))) isa EMState
    # the window's clock in the antennas' modifier is the frame's, and static antennas have none
    w = EMState(win(), EMNumerics())
    @test w.modifiers[1].clock === WindowClock(w.aux.window) === WindowClock(w.aux.window.window.start, w.aux.window.dt, w.aux.window.dx1)
    @test all(k -> window_shifts(w.modifiers[1].clock, k) == window_shifts(w, k), 0:100) && window_shifts(w, 100) > 0
    @test EMState(lasers, EMNumerics()).modifiers[1].clock === nothing
    @test w.modifiers[2] === WindowEdges(guards(w.grid))
    # one holder: no array of a modifier is reachable from anywhere else in the state, and the
    # convolution fields are in `evolved` once each
    prob, num = restart_full(T)
    st = EMState(prob, num)
    mine = reach(st.modifiers)
    @test length(mine) == 2 * (6 + 5) && allunique(objectid.(mine))
    rest = reach((st.species, st.ws, st.fields, st.ρ, st.K, st.recorder, st.aux))
    @test !any(a -> any(b -> a === b, rest), mine)
    ψ = (evolved_arrays(open_absorber(st))..., evolved_arrays(back_absorber(st))...)
    @test length(ψ) == 8 && all(a -> count(b -> a === b, evolved(st)) == 1, ψ)
    # a state over other arrays: its PMLs are over those arrays, its tuple has the order and the
    # other modifiers of the state's, and a step of it leaves the state's own arrays alone
    initialize!(st, prob)
    p = params(prob)
    run!(st, p, 30; observer = nothing)
    held = map(copy, evolved(st))
    v = PME.state_view(st, map(copy, evolved(st)))
    @test typeof(v.modifiers) === typeof(st.modifiers) && v.modifiers[3] === st.modifiers[3] && v.modifiers[4] === st.modifiers[4]
    ψv = (evolved_arrays(open_absorber(v))..., evolved_arrays(back_absorber(v))...)
    @test all(a -> count(b -> a === b, evolved(v)) == 1, ψv) && !any(a -> any(b -> a === b, ψ), ψv)
    step!(v, p, 31; observer = nothing)
    @test all(map(same_bits, evolved(st), held))
    @test !all(map(same_bits, ψv, ψ))
    step!(st, p, 31; observer = nothing)
    @test all(map(same_bits, evolved(v), evolved(st)))
    # a modifier of a test's survives the view, at its place
    tagged = with_modifiers(st, (Idle(), st.modifiers[1:2]..., Affine(1.0, 0.0), st.modifiers[3:4]...))
    tv = PME.on_evolved(tagged, map(copy, evolved(tagged)), similar(tagged.K), tagged.recorder)
    @test map(m -> nameof(typeof(m)), tv.modifiers) == (:Idle, :PMLFields, :BackPMLFields, :Affine, :AntennaPlanes, :WindowEdges)
    @test evolved(tagged) === evolved(st)
end

# --- T-759: a modifier with state of its own ----------------------------------------------------------

"""
A modifier with an evolved array: `n`, one entry, counts the E steps it has seen, and after each
it multiplies Ez by `1 + n/1024`. A run with it depends on the count, so a restart that loses the
array is another run.
"""
struct Mem{A} <: FieldModifier
    n::A
end
PME.evolved_arrays(m::Mem) = (m.n,)
PME.on_arrays(m::Mem, a::Tuple{Any}) = Mem(a[1])
PME.after_substep!(m::Mem, mp, f, grid, ::EStep, wgs) = (m.n .+= 1; f.Ez .*= 1 .+ m.n ./ 1024; nothing)

"`modifier_state`'s problem and numerics (a window, a PML on the open axis and on the back edge, a laser), with a `Mem` added to the state's tuple at `place`."
function mem_case(::Type{T}) where {T}
    prob = params_case((seeded(T, 1), T(3), seeded(T, 2 // 100), T(8)); nsteps = 100, cells = (72, 16), position = 10, open_y = true, polarization = 1)
    return prob, EMNumerics(; absorber = PML(; cells = 4, alpha_max = 0.2), back_absorber = PML(; cells = 6))
end
with_mem(st, place = length(st.modifiers)) =
    with_modifiers(st, (st.modifiers[1:place]..., Mem(KernelAbstractions.zeros(get_backend(st.ρ), eltype(st), 1)), st.modifiers[(place + 1):end]...))
mem_of(st) = only(filter(m -> m isa Mem, collect(Any, st.modifiers)))
mem_count(st) = value_of(only(Array(mem_of(st).n)))

function body_T759(be)
    for T in TYPES
        prob, num = mem_case(T)
        p = params(prob)
        mk(place...) = with_mem(EMState(prob, num; backend = be), place...)
        plain = EMState(prob, num; backend = be)
        # the evolved list: the state's own, in its order, and the modifier's array where the tuple
        # has the modifier among the others (after the back edge's PML when it is last, before it
        # when it is first), in front of the witness bunch; no existing array moves
        base, np = evolved(plain), length(evolved(plain))
        @test np == 10 + 9 + 4 + 2 + 4                 # one species, the fields, the open axis' ψ, the counters, the back edge's ψ
        for (place, at) in ((length(plain.modifiers), np), (0, np - 4), (2, np))
            st = with_mem(plain, place)
            ev = @inferred evolved(st)
            @test length(ev) == np + 1 && ev[at + 1] === mem_of(st).n
            @test (ev[1:at]..., ev[(at + 2):end]...) === base
        end
        # a run starts the count at zero, and counts
        a = mk()
        fill!(mem_of(a).n, 7)
        initialize!(a, prob)
        @test mem_count(a) == 0
        run!(a, p, 20)
        @test mem_count(a) == 20
        initialize!(plain, prob)
        run!(plain, p, 20)
        @test !same_bits(Array(a.fields.Ez), Array(plain.fields.Ez))      # the modifier acts
        # a restart in the middle, into a new state with the modifier: the uninterrupted run's bits,
        # the count included; through a file too
        b = mk()
        initialize!(b, prob)
        run!(b, p, 10)
        snap = snapshot(b, 10)
        @test length(snap.arrays) == np + 1
        c = restore!(mk(), snap)
        @test mem_count(c) == 10
        run!(c, p, 10; first = 11)
        @test isempty(differences(a, c, 10)) && mem_count(c) == 20 && same_bits(Array(mem_of(a).n), Array(mem_of(c).n))
        file = tempname()
        save_snapshot(file, snap)
        d = restore!(mk(), load_snapshot(file, mk()))
        run!(d, p, 10; first = 11)
        @test isempty(differences(a, d, 10)) && same_bits(Array(mem_of(a).n), Array(mem_of(d).n))
        # a state without the modifier is another layout, both ways
        @test_throws ArgumentError restore!(EMState(prob, num; backend = be), snap)
        @test_throws ArgumentError restore!(mk(), snapshot(plain, 20))
        # a state over other arrays has the modifier over those, and steps them alone
        held = map(copy, evolved(a))
        v = PME.state_view(a, map(copy, evolved(a)))
        @test mem_of(v).n !== mem_of(a).n && mem_of(v).n === evolved(v)[np + 1]
        step!(v, p, 21; observer = nothing)
        @test mem_count(v) == 21 && mem_count(a) == 20 && all(map(same_bits, evolved(a), held))
        # the harness: the modifier's array is in the partition, a solve is the loop's run, and a
        # second solve starts the count again
        dprob = DiscreteProblem(prob, num; backend = be, nsteps = 20, save_everystep = false, modifiers = ms -> (ms..., Mem(KernelAbstractions.zeros(be, T, 1))))
        @test length(dprob.u0.x) == np + 1
        for _ in 1:2
            sol = solve(dprob, FunctionMap())
            h = harness_state(sol)
            @test mem_count(h) == 20 && isempty(differences(a, h, 0; cache = false)) && same_bits(Array(mem_of(a).n), Array(mem_of(h).n))
            @test mem_count(harness_state(dprob)) == 0
        end
        # a problem rebuilt from the harness problem keeps the modifier, with an array of its own
        again = remake_reload(dprob; p = prob)
        @test length(again.u0.x) == np + 1 && mem_of(harness_state(again)).n !== mem_of(harness_state(dprob)).n
        @test mem_count(harness_state(solve(again, FunctionMap()))) == 20
    end
end

@testset "T-759 a modifier with an evolved array of its own is in the evolved list, in a snapshot, a restart, a state over other arrays and the harness's partition [A4, A9, WP-057]" begin
    body_T759(CPU())
end
