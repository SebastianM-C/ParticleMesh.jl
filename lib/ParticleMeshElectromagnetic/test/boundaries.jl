# WP-027: the open transverse boundary, T-450 to T-457. The fields leave through the absorbing
# layer (`AbsorbingLayer`, the outer rows of the box along y), the particles that cross the box
# edge are retired through the alive mask. The reflection runs are the vacuum slab packets of
# boundaries_cases.jl, compared with the discrete theory of the same scheme (`layer_rt`); the
# plasma runs are `open_plasma_2d`, whose particles leave through both edges.

using ForwardDiff: ForwardDiff, Partials
using ParticleMeshElectromagnetic: RHO_STAGGER_2D, LayerFactors, retire!, absorb_b!, absorb_e!

# --- T-450 ----------------------------------------------------------------------------------

# The reflection R and the transmission through the slab of both layers, Tr, of a slab packet at
# the carrier angle θ, against the discrete theory, over the thickness (8, 16, 32 cells at a fixed
# nominal attenuation, strength·cells = 32) and the angle (0, 0.4, 0.8 rad), in both
# polarizations. The theory is checked on its own first: with zero strength there is nothing to
# reflect and the slab passes the mode with the phase of its Yee k_y, which pins the root it
# calls upward. Then the runs: each energy fraction within C·`slab_scale` of the prediction, R
# falling with the thickness at every angle measured and rising with the angle at every
# thickness (the matched layer is reflectionless at normal incidence in the continuum only).
const T430_RUNS = Dict{Any,Any}()
t430_run(cells, θ, pol) = get!(T430_RUNS, (cells, θ, pol)) do
    geo = slab_geometry(; theta = θ, cells, strength = 32 / cells)
    (; run = slab_run(Float64, geo, pol), pred = slab_prediction(geo, pol), geo)
end

@testset "T-450 the absorbing layer's reflection and transmission against the scheme's discrete theory, over thickness and angle [A2, A7, WP-027]" begin
    for θ in (0.0, 0.8), pol in (:TE, :TM)
        geo = slab_geometry(; theta = θ, strength = 0.0)
        for (ky, _) in slab_modes(geo)[1:7:end]
            r, t = layer_rt(geo, pol, ky)
            @test abs(r) < 1e-13
            @test abs(abs(t) - 1) < 1e-12
            nrows = 2 * geo.cells + 6
            @test abs(cis(ky * nrows * geo.dy) - t / abs(t)) < 1e-9
        end
    end
    # TE and TM are the same 1D problem at normal incidence, for every grading order: the same |r|
    # (r itself differs by the phase of the field it is read on, Ez on Node() rows or Bz on Center()
    # rows); an inexact edge row would break it at order 0
    for order in (0, 1, 2)
        geo = slab_geometry(; theta = 0.0, cells = 8, strength = 4.0, order)
        for (ky, _) in slab_modes(geo)[1:7:end]
            rTE, rTM = abs(first(layer_rt(geo, :TE, ky))), abs(first(layer_rt(geo, :TM, ky)))
            @test abs(rTE - rTM) <= 1e-10 * rTE
        end
    end
    # the symmetric splitting is second order: at a fixed physical layer (one wavelength thick,
    # strength 2) the normal-incidence reflection |r|² falls by about 2⁴ per halving of Δ (a damping
    # after the sub-step only falls by 2²)
    rn(ncy) = (geo = slab_geometry(; theta = 0.0, cells = ncy, strength = 2.0, ncx = 2ncy, ncy); abs2(first(layer_rt(geo, :TE, geo.ky0))))
    r16, r32, r64 = rn(16), rn(32), rn(64)
    @info "T-450 normal-incidence |r|² against Δ" r16 r32 r64
    @test r16 / r32 > 10 && r32 / r64 > 10
    C = emC("T-450")
    worst = 0.0
    Rs = Dict{Any,Float64}()
    for (cells, θ) in ((8, 0.4), (16, 0.4), (32, 0.4), (16, 0.0), (16, 0.8), (8, 0.0), (8, 0.8)), pol in (:TE, :TM)
        (; run, pred, geo) = t430_run(cells, θ, pol)
        k = round(Int, geo.tmeas / geo.dt)
        eR = abs(run.R - pred.R) / slab_scale(Float64, k, pred.R)
        eT = abs(run.Tr - pred.Tr) / slab_scale(Float64, k, pred.Tr)
        @info "T-450" cells θ pol R = run.R R_theory = pred.R Tr = run.Tr Tr_theory = pred.Tr ratio_R = eR ratio_T = eT
        worst = max(worst, eR, eT)
        Rs[(cells, θ, pol)] = run.R
        @test pred.R > 0                         # the layer reflects something measurable
        @test eR <= C && eT <= C
    end
    record!("T-450", Float64, worst)
    for pol in (:TE, :TM)
        @test Rs[(8, 0.4, pol)] > Rs[(16, 0.4, pol)] > Rs[(32, 0.4, pol)]
        @test Rs[(16, 0.0, pol)] < 1e-7          # an unmatched layer (E or B alone, or at other rates) reflects far more
        for cells in (8, 16)
            @test Rs[(cells, 0.0, pol)] < Rs[(cells, 0.4, pol)] < Rs[(cells, 0.8, pol)]
        end
    end
end

# --- T-451 ----------------------------------------------------------------------------------

# Gauss's law with particles leaving: the layer leaves the curl updates untouched and damps E
# around them, so the E step changes D⁻·E only through the factors e^{−σΔt/2}, not at a node whose four
# E faces carry σ = 0: the node rows j₀ = cells + 1 … n₂ − cells − 1 (interior row indices from
# 0). A retired particle sits within a cell of the box edge, so its charge, which leaves ρ while
# its field stays, is inside the layer too (cells ≥ the guard width ≥ the shape's reach). On the
# free rows D⁻·E = ρ − ρ̄₀ to roundoff after every step, ρ̄₀ the mean charge at t = 0 (the
# immobile background), with the scale of T-360; ∇·B likewise at the cell centres whose Bx and
# By are undamped. Both are far from roundoff in the layer, which shows the check sees it.

@testset "T-451 Gauss's law and ∇·B to roundoff off the absorbing layer with particles leaving, all three types, both modes [A2, A7, A8, WP-027]" begin
    C, CB = emC("T-451"), emC("T-451", "C_divB")
    for T in TYPES, mode in (Reference(), Fast())
        prob, ab = open_plasma_2d(T; amplitude = seeded(T, 1 / 10))
        st = open_state(prob, ab; mode)
        r0 = open_residuals(st, zero(T))
        ρbar0 = r0.ρbar
        r = open_residuals(st, ρbar0)
        worst = r.free / (base_eps(T) * r.scale)
        worstB = 0.0
        worst_layer = 0.0
        n0 = alive_total(st)
        p = params(prob)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = open_residuals(st, ρbar0)
            worst = max(worst, r.free / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
            worst_layer = max(worst_layer, Float64(r.layer / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        left = n0 - alive_total(st)
        @info "T-451" T mode worst worstB worst_layer left of = n0
        @test left > 50                                   # particles left through the edges
        @test worst_layer > 1e4                           # the layer is not charge conserving
        record!("T-451 $(nameof(typeof(mode)))", T, worst)
        record!("T-451 divB $(nameof(typeof(mode)))", T, worstB)
        @test worst <= C
        @test worstB <= CB
    end
end

# --- T-452 ----------------------------------------------------------------------------------

# The retirement: exactly the particles whose cell along y jumped across the seam in this step
# are marked free, the particles that crossed the periodic x seam stay, the alive charge falls by
# the retired particles' charge, and the current of the last move is deposited before the
# retirement: the first step's J is the periodic box's bit for bit (the same fields, push and
# deposit), while the field update after it differs (the layer damps); the kinetic energy at
# n + ½ counts only the particles still in the box. Fast mode in Float64,
# which does not permute the slots, so a slot is the same particle before and after the step.
@testset "T-452 particles that cross the open edge are retired after their current is deposited, the x seam keeps them [A2, WP-027]" begin
    prob, ab = open_plasma_2d(Float64)
    st = open_state(prob, ab; mode = Fast())
    p = params(prob)
    retired = crossed_x = 0
    for k in 1:prob.nsteps
        before = map(s -> (status = Array(s.status), c1 = Array(s.cell[1]), c2 = Array(s.cell[2]), qw = s.q .* Array(s.w)), st.species)
        step!(st, p, k)
        for (s, b) in zip(st.species, before)
            status, c1, c2 = Array(s.status), Array(s.cell[1]), Array(s.cell[2])
            was = b.status .== ParticleMeshCore.ALIVE
            wrap_y = was .& (abs.(c2 .- b.c2) .> 1)
            wrap_x = was .& (abs.(c1 .- b.c1) .> 1)
            @test (was .& (status .!= ParticleMeshCore.ALIVE)) == wrap_y
            @test all(status[wrap_x .& .!wrap_y] .== ParticleMeshCore.ALIVE)
            @test sum(b.qw[was .& (status .== ParticleMeshCore.ALIVE)]) ≈ sum(b.qw[was]) - sum(b.qw[wrap_y])
            retired += count(wrap_y)
            crossed_x += count(wrap_x .& .!wrap_y)
        end
        # the kinetic energy at n + ½ is that of the particles still in the box (retired first)
        Kh = sum(st.species) do s
            alive, w, u = Array(s.status) .== ParticleMeshCore.ALIVE, Array(s.w), map(Array, s.u)
            sum(i -> (u² = u[1][i]^2 + u[2][i]^2 + u[3][i]^2; s.m * w[i] * u² / (sqrt(1 + u²) + 1)), findall(alive); init = 0.0)
        end
        @test sum(Array(st.K)) ≈ Kh rtol = 1e-12
    end
    @info "T-452" retired crossed_x
    @test retired > 50 && crossed_x > 50
    # the first step's current: the open box's against the periodic box's, Reference mode
    for T in TYPES
        po, ab = open_plasma_2d(T)
        pp = EMProblem(po.name, po.length, po.cells, po.dt, po.nsteps, po.species, po.seed, po.kmode)
        so, sp = open_state(po, ab), state_2d(pp; current = Esirkepov())
        step!(so, params(po), 1)
        step!(sp, params(pp), 1)
        @test all(c -> bits(Array(getfield(so.fields, c))) == bits(Array(getfield(sp.fields, c))), (:Jx, :Jy, :Jz))
        @test alive_total(so) < alive_total(sp)
        @test Array(so.fields.Ey) != Array(sp.fields.Ey)
    end
end

# --- T-453 ----------------------------------------------------------------------------------

# With zero strength the layer multiplies by exactly one: a vacuum slab packet in an open box is
# the periodic box's bit for bit, through the seam; and with no layer the 7-argument solve is the
# antennas' solve bit for bit; with a layer, every guard is its interior image after the solve. A
# nonzero layer damps only its own rows: its factors are exactly one
# on the rows outside it, exp(−σ·τ) of the closed-form profile on its rows, and the guard rows
# carry their interior images' factors.
@testset "T-453 a zero-strength layer is the periodic step bit for bit, and the factors sit on the layer rows [A2, A4, A8, WP-027]" begin
    for T in TYPES
        geo = slab_geometry(; theta = 0.0, cells = 8, strength = 0.0)
        prob, num = slab_problem(T, geo)
        pp = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, (), 1, prob.kmode)
        so = EMState(prob, num)
        sp = EMState(pp, EMNumerics(; current = NoCurrent()))
        for s in (so, sp)
            initialize!(s, s === so ? prob : pp)
            slab_init!(s, geo, :TE)
        end
        rows = slab_rows(so, geo)
        W0 = slab_energy(so, rows.interior)
        run!(so, params(prob), prob.nsteps)
        run!(sp, params(pp), pp.nsteps)
        @test all(c -> bits(Array(getfield(so.fields, c))) == bits(Array(getfield(sp.fields, c))), keys(so.fields))
        # the whole packet crossed the seam: it is back in the lower half of the interior
        @test slab_energy(so, rows.lower) / W0 > 0.99
        # with a layer every guard is its interior image after the solve (the damping visits the
        # guard rows and columns with their images' factors): filling the guards again changes no bit
        sl = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = AbsorbingLayer(; cells = 8, strength = 3.0)))
        initialize!(sl, prob)
        slab_init!(sl, geo, :TE)
        foreach(a -> a .= T.(randn(Xoshiro(7), size(a))), values(sl.fields))
        foreach(a -> fill_guards!(a, sl.grid), values(sl.fields))
        solve_fields!(Yee(), sl.fields, sl.grid, prob.dt, (), zero(T), sl.aux.layer)
        refilled = map(a -> fill_guards!(copy(a), sl.grid), sl.fields)
        @test all(c -> bits(Array(getfield(sl.fields, c))) == bits(Array(getfield(refilled, c))), keys(sl.fields))
        # no layer: the 7-argument solve is the 6-argument one
        f1 = map(copy, sp.fields)
        f2 = map(copy, sp.fields)
        solve_fields!(Yee(), f1, sp.grid, pp.dt, (), zero(T), nothing)
        solve_fields!(Yee(), f2, sp.grid, pp.dt, (), zero(T))
        @test all(c -> bits(Array(getfield(f1, c))) == bits(Array(getfield(f2, c))), keys(f1))
    end
    # the factors of a layer of 4 cells on 16 rows with 3 guards: Node() rows j₀ = 0…3 and 13…15
    # damped, j₀ = 4 … 12 exactly one; Center() rows j₀ = 0…3 and 12…15 damped; e^{−σ·τ/2} around a
    # sub-step of τ = Δt (E) or Δt/2 (B), with σ = strength·(depth/thickness)², and the guards their
    # interior images
    for order in (2, 0)
        prob, ab = open_plasma_2d(Float64; layer = 4, strength = 1, order)
        st = open_state(prob, ab)
        L = st.aux.layer
        g, n2 = guards(st.grid), ncells(st.grid)[2]
        @test g == 3
        for (fac, s, τ) in ((L.node_full, 0, prob.dt), (L.center_full, 1, prob.dt), (L.node_half, 0, prob.dt / 2),
                            (L.center_half, 1, prob.dt / 2))
            f = Array(fac)
            for j in eachindex(f)
                j0 = mod(j - g - 1, n2)
                e = min(2j0 + s, 2n2 - 2j0 - s)            # half rows to the nearest edge
                σ = e < 8 ? 1.0 * ((8 - e) / 8)^order : 0.0
                @test f[j] ≈ exp(-σ * τ / 2) rtol = 1e-14
                @test (f[j] == 1) == (σ == 0)
            end
        end
        @test count(<(1), Array(L.node_full)[(g + 1):(g + n2)]) == 7
        @test count(<(1), Array(L.center_full)[(g + 1):(g + n2)]) == 8
    end
end

# --- T-454 ----------------------------------------------------------------------------------

@testset "T-454 the open-boundary step is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-027]" begin
    ref = open_step_digests()
    @test ref == open_step_digests()
    for wgs in (32, 256)
        @test open_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        lines = filter(startswith("EM2DO "), split(worker_output(nt), '\n'))
        got = Dict(String(f[3]) => parse(UInt64, f[4]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(ref)
        for (T, d) in ref
            @test get(got, type_label(T), nothing) == d
        end
    end
end

# --- T-455 ----------------------------------------------------------------------------------

# The element-type matrix on one reflection run (8 cells, θ = 0.8, R ≈ 2 %, large enough for
# Float32's roundoff): each type within C·slab_scale of the discrete theory. The dual's initial
# fields carry a partial equal to the value in the first direction (the derivative with respect
# to the packet's amplitude) and zero in the others; the vacuum step is linear and applies the
# same operations to the value and each partial, so the partials stay the values bit for bit.
@testset "T-455 the reflection in Float32, Float64 and Dual, and the dual's partials linear [A7, A8, WP-027]" begin
    C = emC("T-455")
    geo = slab_geometry(; theta = 0.8, cells = 8, strength = 4.0)
    pred = slab_prediction(geo, :TE)
    k = round(Int, geo.tmeas / geo.dt)
    for T in TYPES
        prob, num = slab_problem(T, geo)
        st = EMState(prob, num)
        initialize!(st, prob)
        slab_init!(st, geo, :TE)
        if T === D3
            for c in keys(st.fields)
                a = getfield(st.fields, c)
                v = value.(a)
                a .= D3.(v, Partials.(tuple.(v, zero.(v), zero.(v))))
            end
        end
        rows = slab_rows(st, geo)
        W0 = slab_energy(st, rows.interior)
        run!(st, params(prob), prob.nsteps)
        R, Tr = slab_energy(st, rows.upper) / W0, slab_energy(st, rows.lower) / W0
        e = max(abs(R - pred.R) / slab_scale(T, k, pred.R), abs(Tr - pred.Tr) / slab_scale(T, k, pred.Tr))
        @info "T-455" T R R_theory = pred.R Tr Tr_theory = pred.Tr ratio = e
        record!("T-455", T, e)
        @test e <= C
        if T === D3
            for c in keys(st.fields)
                a = Array(getfield(st.fields, c))
                @test all(x -> partials(x)[1] == value(x) && iszero(partials(x)[2]) && iszero(partials(x)[3]), a)
            end
        end
    end
end

# --- T-456 ----------------------------------------------------------------------------------

@testset "T-456 input checks and the case file's open boundary and layer [A1, A10, WP-027]" begin
    @test_throws ArgumentError AbsorbingLayer(; cells = 0, strength = 1)
    @test_throws ArgumentError AbsorbingLayer(; cells = 4, strength = -1)
    @test_throws ArgumentError AbsorbingLayer(; cells = 4, strength = Inf)
    @test_throws ArgumentError AbsorbingLayer(; cells = 4, strength = 1, order = -1)
    prob, ab = open_plasma_2d(Float64)
    pp = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species, prob.seed, prob.kmode)
    @test pp.boundaries == (:periodic, :periodic) && prob.boundaries == (:periodic, :open)
    # an open axis needs the layer, a periodic one refuses it
    @test_throws ArgumentError EMState(prob, EMNumerics())
    @test_throws ArgumentError EMState(pp, EMNumerics(; absorber = ab))
    # the layer fits: at least the guard width, and interior rows between the layers
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = AbsorbingLayer(; cells = 2, strength = 1)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = AbsorbingLayer(; cells = 8, strength = 1)))
    @test EMState(prob, EMNumerics(; absorber = AbsorbingLayer(; cells = 7, strength = 1))) isa EMState{2}
    # the binomial filter is refused with the layer (untested together), zero passes are not
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = ab, filter = BinomialFilter(1)))
    @test EMState(prob, EMNumerics(; absorber = ab, filter = BinomialFilter(0))) isa EMState{2}
    @test EMState(pp, EMNumerics(; filter = BinomialFilter(1))) isa EMState{2}
    # boundaries: one per axis, :periodic or :open, axis 1 periodic; N = 1 periodic
    mk(b) = EMProblem("x", prob.length, prob.cells, prob.dt, 1, prob.species, 1, prob.kmode; boundaries = b)
    @test_throws ArgumentError mk((:open, :periodic))
    @test_throws ArgumentError mk((:periodic, :reflecting))
    @test_throws ArgumentError mk((:periodic,))
    @test mk(("periodic", "open")).boundaries == (:periodic, :open)
    @test_throws ArgumentError mk(:open)
    @test_throws ArgumentError mk("open")
    @test_throws ArgumentError EMProblem("x", 1.0, 8, 0.1, 1, (), 1, 1.0; boundaries = (:open,))
    @test_throws ArgumentError EMState(beams_1d(Float64), EMNumerics(; absorber = ab))
    # the time step is the one the layer was made for
    st = open_state(prob, ab)
    @test_throws ArgumentError step!(st, EMParams(prob.dt / 2, prob.kmode), 1)
    # the case file: grid.boundaries and [numerics.absorbing_layer]
    case = deepcopy(load_case("weibel-2d"))
    case["grid"]["boundaries"] = ["periodic", "open"]
    @test_throws ArgumentError numerics_from_case(case)          # open without a layer
    case["numerics"]["absorbing_layer"] = Dict("cells" => 6, "strength" => 1.5)
    num = numerics_from_case(case)
    @test num.absorber == AbsorbingLayer(; cells = 6, strength = 1.5, order = 2)
    case["numerics"]["absorbing_layer"]["order"] = 3
    @test numerics_from_case(case).absorber.order == 3
    pc = problem_from_case(case, Float64; cells = (16, 24), dt = 0.05)
    @test pc.boundaries == (:periodic, :open)
    @test EMState(pc, numerics_from_case(case)) isa EMState{2}
    periodic = deepcopy(load_case("weibel-2d"))
    periodic["numerics"]["absorbing_layer"] = Dict("cells" => 6, "strength" => 1.5)
    @test_throws ArgumentError numerics_from_case(periodic)      # a layer without an open boundary
    bad = deepcopy(case)
    bad["grid"]["boundaries"] = ["periodic", "reflecting"]
    @test_throws ArgumentError problem_from_case(bad, Float64)
    one_d = deepcopy(load_case("landau"))
    one_d["grid"]["boundaries"] = "open"
    @test_throws ArgumentError problem_from_case(one_d, Float64)
end

# --- T-457 ----------------------------------------------------------------------------------

function step_bytes_open(mode; ppc, cells)
    prob, ab = open_plasma_2d(Float64; ppc, cells, nsteps = 4)
    st = open_state(prob, ab; mode)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    return @allocated step!(st, p, 3)
end

@testset "T-457 allocation per open-boundary step does not grow with particle count or grid size [A7, WP-027]" begin
    for mode in (Reference(), Fast())
        small = step_bytes_open(mode; ppc = (2, 2), cells = (16, 16))
        more_particles = step_bytes_open(mode; ppc = (8, 8), cells = (16, 16))
        larger_grid = step_bytes_open(mode; ppc = (2, 2), cells = (64, 48))
        @info "T-457 bytes per open-boundary step" mode small more_particles larger_grid threads = Threads.nthreads()
        # at more than one thread the Reference sort allocates per particle (#82, as T-220); CI runs one thread
        if Threads.nthreads() == 1 || mode isa Fast
            @test more_particles <= small
            @test larger_grid <= small
        end
    end
end
