# WP-028: the moving window, T-470 to T-479. The box moves at c along axis 1 by whole cells,
# its x edges are the undisturbed outside (zero fields, no current), the plasma of a longitudinal
# profile enters through the front (loaded one column behind it) and leaves through the back. The
# problems are in window_cases.jl.

using ParticleMeshCore: quiet_particle_2d

# --- T-470 ----------------------------------------------------------------------------------

# The refill is the engine's quiet start at the global cell: the fresh column of the window at
# position s holds, bit for bit, the particles a static load of the whole domain gives that column
# (lattice offsets, Philox draws keyed by the global cell, weights of the profile at the global x),
# for a thermal plasma, so a seed means the same thing in the window and in a static box (A4). The
# profile's parameter enters through the weights only (A3): with the ramp length a dual, the
# positions carry no partials and the weights carry exactly the derivative of the profile, here
# checked against the closed form of ∂/∂L sin²(π(x − x_v)/(2L)).
function body_T470(be)
    for T in TYPES
        ramp = seeded(T, 4)
        prob = window_plasma(T; thermal = (T(1) / 10, T(1) / 20, T(1) / 30), drift = (T(1) / 50, zero(T), zero(T)), vacuum = 14, ramp)
        st = EMState(prob, EMNumerics(); backend = be)
        w = st.aux.window
        n = ncells(st.grid)
        s_win = 8                     # the fresh column, global column 31, lies on the ramp (x from 14 to 18)
        col = injection_column(w)
        # the window's column `col` at position s_win, through the refill kernel
        sc = Species(be, T, n[2] * 4; q = -one(T), m = one(T), N = 2)
        load_window_columns!(sc, st.grid, w.loads[1], KernelAbstractions.zeros(be, Int32, 1), col, 1, s_win)
        @test alive_count(sc) == n[2] * 4
        # the static domain of n₁ + s_win columns, loaded by the engine with the same profile
        big = PeriodicGrid((n[1] + s_win, n[2]), cellsize(st.grid), guards(st.grid))
        sb = Species(be, T, prod(ncells(big)) * 4; q = -one(T), m = one(T), N = 2)
        load_quiet!(sb, big, ParticleMeshElectromagnetic.RampedCosine(), w.loads[1].p, (2, 2); vdrift = w.loads[1].vdrift,
                    vth = w.loads[1].vth, seed = prob.seed, stream = 1)
        sel = findall(==(Int32(col + s_win)), Array(sb.cell[1]))
        @test length(sel) == n[2] * 4
        a = Array(sc.cell[2]); b = Array(sb.cell[2])[sel]
        @test a == b
        for (x, y) in ((sc.offset[1], sb.offset[1]), (sc.offset[2], sb.offset[2]), (sc.w, sb.w))
            @test same_bits(Array(x), Array(y)[sel])
        end
        # the momenta γv of the same draws (to a few ulps: the host and the kernel may fuse differently, A4)
        ub = map(d -> Array(sb.u[d])[sel], 1:3)
        ug = [velocity_momentum((ub[1][i], ub[2][i], ub[3][i])) for i in eachindex(sel)]
        for d in 1:3
            @test all(i -> absdiff(Array(sc.u[d])[i], ug[i][d]) <= 4 * base_eps(T) * absmax(ug[i][d]), eachindex(sel))
        end
        @test all(==(Int32(col)), Array(sc.cell[1]))
        if T === D3
            @test all(x -> all(iszero, partials(x)), Array(sc.offset[1]))
            # ∂w/∂L on the ramp, against the closed form; w = density·sin²(πt/2)/4, t = (x − x_v)/L
            hx = value(cellsize(st.grid)[1])
            for (δ, wt) in zip(Array(sc.offset[1]), Array(sc.w))
                x = (col + s_win - 1 + value(δ)) * hx
                t = (x - 14) / 4
                @test 0 < t < 1
                dwdL = -(π / 2) * sinpi(t) * t / 4 / 4
                @test collect(Float64.(Tuple(partials(wt)))) ≈ dwdL .* [1 / 8, 2 / 8, 3 / 8] atol = 1e-6
            end
        end
    end
end

@testset "T-470 the window's refill is the engine's quiet start at the global cell, the profile entering through the weights [A3, A4, A8, WP-028]" begin
    body_T470(CPU())
end

# --- T-471 ----------------------------------------------------------------------------------

# A cold plasma at rest stays what it is while the window moves through it: no current (a particle
# at rest deposits exactly zero, T-308), so the fields stay +0.0 bit for bit, and after every step
# the window holds, as a set, exactly the particles of a fresh cold load of its plasma columns at
# its position, whose charge is the background's: ρ = ρ₀(s), in Reference mode bit for bit (neutral
# to the last bit), with the atomics of Fast mode to the roundoff of their order (the ramp's weights
# are not dyadic).
# The ramp (vacuum 4, ramp 4) enters through the front; after 40 steps the window has moved 29
# cells, more than its 24, so every particle has been loaded by a refill. In the plateau every
# node carries the same bits.
function body_T471(be)
    for T in TYPES, mode in (Reference(), Fast())
        prob = window_plasma(T)
        st = window_state(prob; mode, be)
        p = params(prob)
        g, n = guards(st.grid), ncells(st.grid)
        ρ, ρ0 = field(be, T, st.grid), field(be, T, st.grid)
        ok_fields = ok_charge = ok_set = true
        for k in 1:prob.nsteps
            step!(st, p, k)
            s_win = window_shifts(st, k)
            ok_fields &= all(a -> same_bits(a, zero(a)), values(st.fields))
            window_charge!(ρ, st)
            window_background!(ρ0, st, s_win)
            ok_charge &= effective_mode(mode, T) isa Reference ? same_bits(ρ, ρ0) :
                         maximum(absdiff.(Array(ρ), Array(ρ0))) <= 8 * base_eps(T) * maximum(absmax, Array(ρ))
            ok_set &= particle_set(st.species[1]) == particle_set(fresh_window_load(st, s_win))
        end
        s_end = window_shifts(st, prob.nsteps)
        @test s_end > n[1]
        @test ok_fields
        @test ok_charge
        @test ok_set
        # the plateau is uniform bit for bit: the nodes 3 … injection column − 1 (away from the
        # back edge and from the plasma's front, whose stencils miss the outside or the empty front
        # columns) whose particles all lie beyond the ramp's end at x = 8
        hx = value_of(cellsize(st.grid)[1])
        plateau = [i for i in 3:(injection_column(st.aux.window) - 1) if (i - 3 + s_end) * hx >= 8]
        @test length(plateau) > 10
        vals = Array(ρ)[g .+ plateau, (g + 1):(g + n[2])]
        @test all(v -> same_bits([v], [vals[1]]), vals)
        @test window_lost(st) == (0,)
    end
end

@testset "T-471 a cold plasma at rest stays uniform and neutral to the last bit in the moving window, all three types, both modes [A2, A3, A4, A8, WP-028]" begin
    body_T471(CPU())
end

# --- T-472 ----------------------------------------------------------------------------------

# A vacuum pulse is the same with and without the window. The static twin is the same problem on
# a box long enough for the whole run, with a window that never moves (the same back edge). Before
# the window moves they differ only where the pulse's precursor and tail meet the window's front
# edge, to roundoff. Afterwards they differ
# only by what the window leaves out: at the back the 2D wake of the pulse (in two dimensions a
# pulse leaves a trailing field) meets the window's back edge, and that difference reaches forward
# through Yee's stencil, falling by orders of magnitude per few columns; at the front the window
# cuts the pulse's Gaussian tail, six durations ahead of its peak and below Float64 roundoff. The
# antenna stays in the lab frame, so it keeps launching while the window moves until it is passed.
# Float64 only: fields only, no deposition, the same kernels for every type.
function body_T472(be)
    C = emC("T-472")
    prob = window_lwfa(Float64; plasma = false, nsteps = 200)
    n = prob.cells
    win = window_state(prob; be)
    s_end = window_shifts(win, prob.nsteps)
    twin = static_twin(prob, n[1] + s_end + 8)
    ref = window_state(twin; be)
    p = params(prob)
    g = guards(win.grid)
    J = (g + 1):(g + n[2])
    worst_before = 0.0
    worst = 0.0
    back = 0.0
    for k in 1:prob.nsteps
        step!(win, p, k)
        step!(ref, p, k)
        s_win = window_shifts(win, k)
        for c in keys(win.fields)
            c in (:Jx, :Jy, :Jz) && continue
            a, b = Array(getfield(win.fields, c)), Array(getfield(ref.fields, c))
            m = maximum(abs, b[(g + 1 + s_win):(g + n[1] + s_win), J])
            m > 0 || continue
            if s_win == 0                 # the same back edge: every column
                worst_before = max(worst_before, maximum(abs, a[(g + 1):(g + n[1]), J] .- b[(g + 1):(g + n[1]), J]) / m)
            else
                d = maximum(abs, a[(g + 25):(g + n[1]), J] .- b[(g + 25 + s_win):(g + n[1] + s_win), J]) / m
                worst = max(worst, d)
                back = max(back, maximum(abs, a[(g + 1):(g + 8), J] .- b[(g + 1 + s_win):(g + 8 + s_win), J]) / m)
            end
        end
    end
    @info "T-472 window against its static twin" shifts = s_end worst_before worst back
    @test s_end > n[1]
    @test worst_before <= C * eps(Float64)
    @test back > 1e-3                     # the back edge does differ, so the comparison sees it
    record!("T-472", Float64, worst / eps(Float64), be)
    @test worst <= C * eps(Float64)
end

@testset "T-472 a vacuum pulse is the same with and without the moving window, off the back edge [A2, A7, WP-028]" begin
    body_T472(CPU())
end

# --- T-473 ----------------------------------------------------------------------------------

# Gauss's law across shifts. The plasma starts neutral against its immobile background, E⁰ = 0;
# each step keeps D⁻·E − ρ constant to roundoff (T-360); a shift moves E, the particles and the
# background one column back together, and the new front column is the undisturbed outside, zero
# field and a fresh neutral load. So D⁻·E = ρ − ρ₀(s) holds to roundoff at every node except:
# node 1, whose difference reads the zero x-guard in place of the field behind the box; node 2,
# the reach of a particle retired at the back edge with its charge; and the antenna's two columns,
# which carry the surface charges of its sheets (T-406), in the lab frame, so they move back with
# the shifts. The laser drives the plasma (|u| about 0.4), the window moves 70 cells, more than its
# 64, so the whole plasma has entered through the front; scale as T-360.
#
# A warm plasma (thermal speeds 0.1 c, no laser) is the case that needs the front treatment: its
# particles reach past their cells, and its fields reach the front column. With the fresh plasma
# loaded on the injection column and Ex on the front face carried along by the shift, Gauss's law
# holds at every node but 1 and 2 there as well; loaded on column n₁ − 1, or with the face's Ex
# zeroed, the residual reaches 10¹⁰ to 10¹² of the scale on every column within 60 steps (the
# measurements are in the PR that added this case).
function body_T473(be)
    C = emC("T-473")
    for T in TYPES, mode in (Reference(), Fast())
        prob = window_lwfa(T)
        st = window_state(prob; mode, be)
        p = params(prob)
        ia = st.aux.antennas[1].ia            # the plane's interior column on the static grid
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 3 == 0 || continue
            s_win = window_shifts(st, k)
            r = window_gauss(st, s_win)
            skip = (1, 2, ia - 1 - s_win, ia - s_win)
            cols = [i for i in axes(r.res, 1) if !(i in skip)]
            worst = max(worst, Float64(maximum(r.res[cols, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        umax = maximum(abs ∘ value_of, Array(st.species[1].u[1]))
        @info "T-473" T mode worst shifts = window_shifts(st, prob.nsteps) umax
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        @test umax > 1 / 10
        record!("T-473 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
    for T in TYPES, mode in (Reference(), Fast())
        prob = window_plasma(T; cells = (24, 6), vacuum = -100, ramp = 1, thermal = (T(1) / 10, T(1) / 10, T(1) / 10), nsteps = 60)
        st = window_state(prob; mode, be)
        p = params(prob)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = window_gauss(st, window_shifts(st, k))
            worst = max(worst, Float64(maximum(r.res[3:end, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        @info "T-473 warm" T mode worst shifts = window_shifts(st, prob.nsteps)
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-473 warm $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
    # a warm plasma streaming forward through a ramp: the front treatment with a drift and a profile
    for T in TYPES, mode in (Reference(), Fast())
        prob = window_plasma(T; cells = (24, 6), vacuum = 2, ramp = 4, drift = (T(3) / 10, zero(T), zero(T)),
                             thermal = (T(1) / 10, T(1) / 10, T(1) / 10), nsteps = 60)
        st = window_state(prob; mode, be)
        p = params(prob)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = window_gauss(st, window_shifts(st, k))
            worst = max(worst, Float64(maximum(r.res[3:end, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        @info "T-473 warm, drift and ramp" T mode worst shifts = window_shifts(st, prob.nsteps)
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-473 warm ramp $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end

@testset "T-473 Gauss's law to roundoff across window shifts with a laser driving the plasma, all three types, both modes [A2, A4, A7, A8, WP-028]" begin
    body_T473(CPU())
end

# --- T-474 ----------------------------------------------------------------------------------

# The window step in Reference mode is bitwise across reruns and launch shapes: the shift is one
# work item per entry, the refill writes each fresh particle once at a slot fixed by the alive
# count, and the sort fixes the order the deposition sees. The two modes agree to roundoff in the
# fields (Float64, the run of T-473 to 120 steps).
function body_T474(be)
    ref = window_digests(; be)
    @test ref == window_digests(; be)
    for wgs in (be isa CPU ? (32, 256) : (64, 256))
        @test window_digests(; workgroupsize = wgs, be) == ref
    end
    prob = window_lwfa(Float64; nsteps = 120)
    a, b = window_state(prob; mode = Reference(), be), window_state(prob; mode = Fast(), be)
    run!(a, params(prob), prob.nsteps)
    run!(b, params(prob), prob.nsteps)
    m = maximum(c -> maximum(abs, Array(getfield(a.fields, c))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    d = maximum(c -> maximum(abs, Array(getfield(a.fields, c)) .- Array(getfield(b.fields, c))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    @info "T-474 Reference against Fast" rel = d / m
    record!("T-474", Float64, d / (m * eps(Float64)), be)
    @test d <= emC("T-474") * eps(Float64) * m
end

@testset "T-474 the moving-window step is bitwise across reruns and launch shapes in Reference mode, and the modes agree [A4, WP-028]" begin
    body_T474(CPU())
end

# --- T-475 ----------------------------------------------------------------------------------

# The gradient through the refill (A3, A7): the field energy after 30 steps of a plasma drifting
# at 0.1 c into the window (the drift against the immobile background charges the ramp, so the
# fields depend on the ramp length L), by ForwardDiff with L a dual, against the finite-difference
# floor of `fd_scan` (Float64, Reference, one seed). Every particle in the window at the end was
# loaded at a refill or at t = 0 with the weights of the profile; the positions and the shift
# times do not depend on L, so the observable is smooth in it.
function body_T475(be)
    f(L, mode, seed) = begin
        T = typeof(L)
        prob = window_plasma(T; ramp = L, drift = (T(1) / 10, zero(T), zero(T)), nsteps = 30, seed, cells = (24, 6))
        st = window_state(prob; mode, be)
        run!(st, params(prob), prob.nsteps)
        sum(c -> sum(abs2, Array(getfield(st.fields, c))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    end
    x0 = 4.0
    ad = ForwardDiff.derivative(x -> f(x, Reference(), 1), x0)
    s = fd_scan(f, x0, [10.0^(-k) for k in 2:7]; seed = 1, ad)
    @info "T-475" ad fd = s.fd floor = s.floor_rel rel = s.rel[s.istar]
    @test abs(ad) > 0
    record!("T-475", Float64, s.rel[s.istar] / s.floor_rel, be)
    @test s.rel[s.istar] <= emC("T-475") * s.floor_rel
end

@testset "T-475 the field energy after window shifts is differentiable in the ramp length through the refill [A3, A7, WP-028]" begin
    body_T475(CPU())
end

# --- T-476 ----------------------------------------------------------------------------------

# The edges and the buffers. A plasma drifting backwards at 0.9 c (fast enough to cross the back
# edge before the next shift drops its column) crosses the back edge between shifts: exactly the particles whose cell along x jumped across the seam in a step are retired,
# before the current (Fast mode, Float64, no permutation, so a slot is a particle). Their current
# would land at the front (the engine places it at the new cell, across the seam), so in a step
# without a shift Gauss's law must not change on the nodes 3 … n₁ (it changes on nodes 1 and 2,
# which the retired particles' charge left). A dilute plasma (10⁻⁴ n_c, so its own field does
# not stop it) drifting forwards at 0.5 c stays 1/(1 − v) = 2 times as long in the window as one
# at rest, so its particle count grows: with no headroom the refills run out of slots and count
# the particles they could not place; with a headroom of 3/2 none is lost (the default 1/4 is for
# a plasma that does not stream with the window).
function body_T476(be)
    prob = window_plasma(Float64; drift = (-0.9, 0.0, 0.0), nsteps = 20, vacuum = -10)
    st = window_state(prob; mode = Fast(), be)
    p = params(prob)
    retired = 0
    n1 = prob.cells[1]
    g = guards(st.grid)
    for k in 1:prob.nsteps
        s = st.species[1]
        before = (status = Array(s.status), c1 = Array(s.cell[1]))
        shifts = window_shifts(st, k) - window_shifts(st, k - 1)
        r0 = shifts == 0 ? window_gauss(st, window_shifts(st, k)).res : nothing
        step!(st, p, k)
        shifts == 0 || continue       # a shift step re-sorts the slots
        r1 = window_gauss(st, window_shifts(st, k))
        @test maximum(abs.(r1.res[3:end, :] .- r0[3:end, :])) <= 64 * eps(Float64) * r1.scale
        was = before.status .== ParticleMeshCore.ALIVE
        now = Array(s.status) .== ParticleMeshCore.ALIVE
        jumped = was .& (abs.(Array(s.extras.old_cell_1) .- Array(s.cell[1])) .> 1)
        @test (was .& .!now) == jumped
        retired += count(jumped)
    end
    @info "T-476 retired at the back edge" retired
    @test retired > 10
    fwd = window_plasma(Float64; drift = (0.5, 0.0, 0.0), nsteps = 40, vacuum = -10, density = 1e-4)
    tight = window_state(fwd; headroom = 0, be)
    run!(tight, params(fwd), fwd.nsteps)
    roomy = window_state(fwd; headroom = 3 / 2, be)
    run!(roomy, params(fwd), fwd.nsteps)
    @info "T-476 lost particles" tight = window_lost(tight) roomy = window_lost(roomy)
    @test window_lost(tight)[1] > 0
    @test window_lost(roomy) == (0,)
end

@testset "T-476 particles crossing the back edge are retired before their current, and the buffers count what does not fit [A2, A4, WP-028]" begin
    body_T476(CPU())
end

# --- T-477 ----------------------------------------------------------------------------------

@testset "T-477 input checks: the window, its profile and its antennas [A2, A10, WP-028]" begin
    T = Float64
    e = SpeciesDescription{T}(:e, -1.0, 1.0, 1.0, 0.0, (1, 0), (1, 1), ZERO3(T), ZERO3(T), 1)
    e1 = SpeciesDescription{T}(:e, -1.0, 1.0, 1.0, 0.0, 1, 1, ZERO3(T), ZERO3(T), 1)
    ep = SpeciesDescription{T}(:e, -1.0, 1.0, 1.0, 0.1, (1, 0), (1, 1), ZERO3(T), ZERO3(T), 1)
    box, cells = (8.0, 4.0), (16, 8)
    w = MovingWindow(; start = 0)
    mk(; kw...) = EMProblem("x", box, cells, 0.3, 4, (e,), 1, (1, 0); kw...)
    @test mk(; boundaries = (:window, :periodic), window = w) isa EMProblem{2}
    @test mk(; boundaries = (:window, :open), window = w) isa EMProblem{2}
    @test_throws ArgumentError mk(; boundaries = (:window, :periodic))                    # no MovingWindow
    @test_throws ArgumentError mk(; window = w)                                           # axis 1 not :window
    @test_throws ArgumentError mk(; boundaries = (:periodic, :window))                    # the window is axis 1
    @test_throws ArgumentError mk(; profile = DensityRamp(; vacuum = 1, ramp = 2))         # a profile needs the window
    @test_throws ArgumentError mk(; boundaries = (:window, :periodic), window = w, E0 = (0.0, 0.1, 0.0))
    @test_throws ArgumentError EMProblem("x", box, cells, 0.3, 4, (ep,), 1, (1, 0); boundaries = (:window, :periodic), window = w)
    @test_throws ArgumentError EMProblem("x", 8.0, 16, 0.3, 4, (e1,), 1, 1.0; boundaries = (:window,), window = w)
    @test_throws ArgumentError DensityRamp(; vacuum = 1, ramp = 0)
    @test_throws ArgumentError EMNumerics(; headroom = -1)
    # an antenna the window passes before its pulse is over
    laser = GaussianLaser(; a0 = 0.1, waist = 2.0, duration = 3.0, delay = 18.0, focus = 2.0, position = 2.0, center = 2.0)
    early = EMProblem("x", box, cells, 0.3, 4, (e,), 1, (1, 0); lasers = (laser,), boundaries = (:window, :periodic), window = w)
    @test_throws ArgumentError EMState(early, EMNumerics())
    late = EMProblem("x", box, cells, 0.3, 4, (e,), 1, (1, 0); lasers = (laser,), boundaries = (:window, :periodic),
                     window = MovingWindow(; start = 40))
    @test EMState(late, EMNumerics()) isa EMState{2}
    # the window counts its shifts from its start
    st = window_state(window_plasma(Float64; start = 1.0))
    a = st.aux.window
    @test window_shifts(st, 0) == 0
    @test all(k -> window_shifts(st, k) == max(0, floor(Int, (k * a.dt - 1.0) / a.dx1)), 0:50)
    @test injection_column(a) == ncells(st.grid)[1] - 3           # quadratic shape: reach 2 past the cell
    lin = EMState(window_plasma(Float64), EMNumerics(; shape = Linear))
    @test injection_column(lin.aux.window) == ncells(lin.grid)[1] - 2
    filt = EMState(window_plasma(Float64), EMNumerics(; filter = BinomialFilter(2)))
    @test injection_column(filt.aux.window) == ncells(filt.grid)[1] - 5
    # a window too narrow to keep plasma behind its front margin (the grid itself needs n₁ ≥ 2g)
    @test_throws ArgumentError EMState(window_plasma(Float64; cells = (8, 6)), EMNumerics(; filter = BinomialFilter(4)))
    @test EMState(window_plasma(Float64; cells = (8, 6)), EMNumerics(; filter = BinomialFilter(3))) isa EMState{2}
    # a window that starts late takes cold plasma only (before it moves, its front is a fixed edge)
    @test_throws ArgumentError window_plasma(Float64; start = 1.0, thermal = (0.01, 0.01, 0.01))
    @test_throws ArgumentError window_plasma(Float64; start = 1.0, drift = (0.0, 0.1, 0.0))
    @test window_plasma(Float64; start = 0, thermal = (0.01, 0.01, 0.01)) isa EMProblem{2}
    # the window's loads evaluate the density at each particle only
    wp = window_plasma(Float64)
    @test_throws ArgumentError initialize!(EMState(wp, EMNumerics()), wp; density_at = :cell_centre)
end

# --- T-478 ----------------------------------------------------------------------------------

# The filter in the window does not wrap along x: a current confined to the back columns, filtered
# with the window's edges, stays out of the front columns, while the periodic filter carries it
# across the seam; along y the filter is the periodic one.
function step_bytes_window(; ppc, cells)
    prob = window_plasma(Float64; ppc, cells, drift = (0.1, 0.0, 0.0), nsteps = 8, vacuum = -10)
    st = window_state(prob)
    p = params(prob)
    run!(st, p, 3)
    return @allocated run!(st, p, 3; first = 4)      # steps 4 to 6, at least one shift
end

function device_step_bytes_window(be; ppc, cells)
    prob = window_plasma(Float64; ppc, cells, drift = (0.1, 0.0, 0.0), nsteps = 9, vacuum = -10)
    st = window_state(prob; be)
    p = params(prob)
    run!(st, p, 3)
    KernelAbstractions.synchronize(be)
    sync_run!(first) = (run!(st, p, 3; first); KernelAbstractions.synchronize(be))
    return (; host = (@allocated sync_run!(4)), dev = (CUDACore.@allocated sync_run!(7)))
end

function body_T478(be)
    st = window_state(window_plasma(Float64); be)
    grid = st.grid
    g, n = guards(grid), ncells(grid)
    h = field(CPU(), Float64, grid)
    h[g + 1, g + 3] = 1.0
    a, b = copyto!(field(be, Float64, grid), h), copyto!(field(be, Float64, grid), h)
    binomial!(a, field(be, Float64, grid), grid, 2; edges = st.aux.window)
    binomial!(b, field(be, Float64, grid), grid, 2)
    a, b = Array(a), Array(b)
    @test all(iszero, a[(g + n[1] - 2):(g + n[1]), :])
    @test maximum(abs, b[(g + n[1] - 1):(g + n[1]), :]) > 0
    @test sum(a[(g + 1):(g + n[1]), (g + 1):(g + n[2])]) < 1          # what left through the back edge is gone
end

@testset "T-478 the filter does not wrap along the window's axis, and allocation per window step does not grow [A2, A7, WP-028]" begin
    body_T478(CPU())
    small = step_bytes_window(; ppc = (2, 2), cells = (24, 6))
    more_particles = step_bytes_window(; ppc = (6, 6), cells = (24, 6))
    larger_grid = step_bytes_window(; ppc = (2, 2), cells = (72, 18))
    @info "T-478 bytes per three window steps" small more_particles larger_grid threads = Threads.nthreads()
    if Threads.nthreads() == 1
        @test_alloc more_particles <= small
        @test_alloc larger_grid <= small
    end
end

# --- T-479 ----------------------------------------------------------------------------------

# A hot plasma in the window (#257): the loader draws the thermal spread in velocity, so its
# Gaussian tail passes c. The initial load refuses such a draw with an ArgumentError (thermal 0.2
# and 0.3 c on T-473's warm box, where the draws at t = 0 reach c). A refill, which cannot raise
# inside a step, loads such a particle at the drift velocity and counts it: on a small box at
# thermal 0.3 c, the first seed whose initial load stays below c and whose refills reach c runs
# with finite momenta, a positive count, and Gauss's law as T-473 off nodes 1 and 2.
function body_T479(be)
    for th in (0.2, 0.3)
        @test_throws ArgumentError window_state(window_plasma(Float64; cells = (24, 6), vacuum = -100, ramp = 1, thermal = (th, th, th)); be)
    end
    C = emC("T-473")
    for T in TYPES
        th = T(3) / 10
        found = false
        for seed in 1:40
            prob = window_plasma(T; cells = (16, 6), ppc = (1, 1), vacuum = -100, ramp = 1, thermal = (th, th, th), nsteps = 150, seed)
            st = try
                window_state(prob; be)
            catch err
                err isa ArgumentError || rethrow()
                continue
            end
            @test window_superluminal(st) == (0,)
            p = params(prob)
            worst = 0.0
            for k in 1:prob.nsteps
                step!(st, p, k)
                k % 5 == 0 || continue
                r = window_gauss(st, window_shifts(st, k))
                worst = max(worst, Float64(maximum(r.res[3:end, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
            end
            window_superluminal(st)[1] > 0 || continue
            s = st.species[1]
            alive = Array(s.status) .== ParticleMeshCore.ALIVE
            @info "T-479 refills at or above c" T seed fast = window_superluminal(st) worst
            @test all(d -> all(x -> isfinite(value_of(x)), Array(s.u[d])[alive]), 1:3)
            record!("T-479", T, worst, be)
            @test worst <= C
            found = true
            break
        end
        @test found
    end
end

@testset "T-479 a drawn speed at or above c is refused in the window's initial load and replaced and counted in its refills [A2, A4, A8, WP-028]" begin
    body_T479(CPU())
end
