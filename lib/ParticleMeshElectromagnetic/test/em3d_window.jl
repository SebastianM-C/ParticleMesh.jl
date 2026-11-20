# WP-074: the moving window of the 3D step, T-1030 to T-1038 (T-1039 is their rerun on CUDA, in
# gpu.jl). The 2D window (WP-028) and its back-edge PML (WP-034) are one method over N: the shift
# moves every E and B column back with the transverse sheets of each column, the refill loads the
# engine's quiet start on the injection column's n₂·n₃ cells at their global cells, the antennas
# stay in the lab frame and the PML of the back edge corrects By, Bz, Ey, Ez across x as at N = 2.
# The 2D theory of the window carries over through T-1030: a 3D window run that does not vary
# along y or z is the 2D window run. The problems are in em3d_cases.jl (`window_plasma_3d`,
# `window_lwfa_3d`, the invariant pair); after em3d_step.jl, whose helpers and worker this file
# reuses, and window_cases.jl.

using ParticleMeshElectromagnetic: injection_column, load_window_columns!, velocity_momentum, BackPMLFields, AntennaPlanes, WindowEdges,
                                   back_absorber, open_absorbers, state_view

"The number of alive particles of species 1 of `st`."
alive_1(st) = count(==(ParticleMeshCore.ALIVE), Array(st.species[1].status))

# --- T-1030 ---------------------------------------------------------------------------------

# The 3D window is the 2D window where the problem does not vary along the third axis: a cold
# plasma drifting backwards and across the plane through a ramp (`window_plane_2d`), and the same
# problem in 3D invariant along z (the plane on x, y) or along y (the plane on x, z, a quarter
# turn about x, so the components map with the signs of `window_map_3d`), with the plane's
# transverse axis periodic, or open with the PML and a PML on the back edge. The window moves 30
# cells (more than its 24) in 40 steps, so the shift, the refill at the injection column, the
# retirement at the back edge and the back-edge PML act on every invariant line at once: the alive
# count stays 12 times the 2D one (6 cells and 2 particles per cell along the invariant axis).
# The 2D run's current along its collapsed axis is the quantity q·w·u_z/γ, the 3D one's
# Esirkepov's; compared as T-953 after initialize! and after every step, with the ramp length
# seeded (a dual's partials compared as well).
function body_T1030(be)
    C = emC("T-1030")
    pml = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)
    for T in TYPES, mode in (Reference(), Fast()), axis in (3, 2), open in (false, true)
        ramp = seeded(T, 4)
        p2, p3 = window_plane_2d(T; open, ramp), window_inv_3d(T, axis; open, ramp)
        nums = EMNumerics(; mode, absorber = open ? pml : nothing, back_absorber = open ? PML(; cells = 4, alpha_max = 0.2) : nothing)
        s2, s3 = EMState(p2, nums; backend = be), EMState(p3, nums; backend = be)
        @test open == (back_absorber(s3) isa BackPMLFields)
        initialize!(s2, p2)
        initialize!(s3, p3)
        n0 = alive_1(s2)
        @test alive_1(s3) == 12 * n0
        scale = sum(ncells(s2.grid)) * 2
        worst = invariant_error_window(s2, s3, axis, scale, 1)
        same_count = true
        for k in 1:p2.nsteps
            step!(s2, params(p2), k)
            step!(s3, params(p3), k)
            worst = max(worst, invariant_error_window(s2, s3, axis, scale, k))
            same_count &= alive_1(s3) == 12 * alive_1(s2)
        end
        @test window_shifts(s3, p3.nsteps) > p3.cells[1]
        @test same_count
        @test window_lost(s3) == (0,)
        # the run is not trivial: a field along every axis of the plane, and with the open edge every component
        comps = open ? keys(window_map_3d(axis)) : (:Ex, :Ey, :Ez, :Bz, :Jx, :Jy, :Jz)
        @test all(c -> maximum(absmax, Array(getfield(s2.fields, c))) > 0, comps)
        record!("T-1030 axis $axis $(open ? "open" : "periodic") $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end
invariant_error_window(s2, s3, axis, scale, k) = window_invariant_error(s2, s3, axis) / (base_eps(eltype(s2)) * k * scale)

@testset "T-1030 a 3D window run invariant along y or z is the 2D window run, periodic and open with the back-edge PML [A2, A3, A7, A8, WP-074]" begin
    body_T1030(CPU())
end

# --- T-1031 ---------------------------------------------------------------------------------

# T-471 in 3D: a cold plasma at rest stays what it is while the window moves through it and
# renews it. No current, so the fields stay +0.0 bit for bit; after every step the window holds,
# as a set, exactly the particles of a fresh cold load of its plasma columns at its position, and
# their charge is the background's, ρ = ρ₀(s), bit for bit in Reference mode (neutral to the last
# bit), to the roundoff of the atomics' order in Fast mode (at most C·eps·max|ρ|, the 27 nodes of
# each particle's stencil added in the order the threads reach them). The ramp (vacuum 4, ramp 4) enters
# through the front; after 48 steps the window has moved 24 cells, its whole length, so every
# particle has been loaded by a refill. In the plateau every node carries the same bits. Then the
# same with y and z open (the PML) and a PML on the back edge, whose corrections of a zero field
# add +0.0 (Float64, Reference).
function body_T1031(be)
    pml = PML(; cells = 3, kappa_max = 2, alpha_max = 0.1)
    cases = [(T, m, false) for T in TYPES for m in (Reference(), Fast())]
    push!(cases, (Float64, Reference(), true))
    for (T, mode, open) in cases
        prob = window_plasma_3d(T; nsteps = 48, cells = open ? (24, 8, 8) : (24, 6, 6), bd = open ? (:open, :open) : (:periodic, :periodic))
        st = window_state_3d(prob; mode, be, absorber = open ? pml : nothing, back = open ? PML(; cells = 4) : nothing)
        p = params(prob)
        g, n = guards(st.grid), ncells(st.grid)
        ρ, ρ0 = field(be, T, st.grid), field(be, T, st.grid)
        ok_fields = ok_charge = ok_set = true
        fast = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            s_win = window_shifts(st, k)
            ok_fields &= all(a -> same_bits(a, zero(a)), values(st.fields))
            window_charge!(ρ, st)
            window_background!(ρ0, st, p, s_win)
            if effective_mode(mode, T) isa Reference
                ok_charge &= same_bits(ρ, ρ0)
            else
                fast = max(fast, Float64(maximum(absdiff.(Array(ρ), Array(ρ0))) / (base_eps(T) * maximum(absmax, Array(ρ)))))
            end
            ok_set &= particle_set_3d(st.species[1]) == particle_set_3d(fresh_window_load_3d(st, p, s_win))
        end
        s_end = window_shifts(st, prob.nsteps)
        @test s_end >= n[1]
        @test ok_fields
        @test ok_charge
        @test ok_set
        if !(effective_mode(mode, T) isa Reference)
            record!("T-1031 Fast", T, fast, be)
            @test fast <= emC("T-1031")
        end
        hx = value_of(cellsize(st.grid)[1])
        plateau = [i for i in 3:(injection_column(st.aux.window) - 1) if (i - 3 + s_end) * hx >= 8]
        @test length(plateau) >= 8
        vals = Array(ρ)[g .+ plateau, (g + 1):(g + n[2]), (g + 1):(g + n[3])]
        @test all(v -> same_bits([v], [vals[1]]), vals)
        @test window_lost(st) == (0,)
        open && @test length(open_absorbers(st)) == 2
    end
end

@testset "T-1031 a cold plasma at rest stays uniform and neutral to the last bit while the 3D window renews it, all three types, both modes [A2, A3, A4, A8, WP-074]" begin
    body_T1031(CPU())
end

# --- T-1032 ---------------------------------------------------------------------------------

# T-473 in 3D: Gauss's law across shifts with a round laser driving the plasma. The plasma starts
# neutral against its immobile background, each step keeps D⁻·E − ρ to roundoff (T-950), and a
# shift moves E, the particles and the background one column back together, the new front column
# the undisturbed outside. So Σ_d D⁻_d E_d = ρ − ρ₀(s) at every node but node 1 (its difference
# reads the zero x-guard), node 2 (the reach of a particle retired at the back edge) and the
# antenna's two columns (its surface charges, T-1024), which stay in the lab frame and so move
# back with the shifts until the window has passed them. The window moves 113 cells, more than its
# 72, so the whole plasma has entered through the front, and the laser drives it to |u| about
# 0.4. Then the same with y and z open (PML) and a PML of 6 cells on the back edge, on the nodes
# off the layers (Float64, both modes), and T-473's warm plasmas (thermal 0.1 c), at rest and
# drifting through a ramp, which need the front treatment of the window (the injection column,
# Ex carried along on the front face). Scale as T-950 with the charge's own maximum.
function window_gauss_worst_3d(st, p, k, skip_cols; rows = nothing, T = eltype(st.ρ))
    s_win = window_shifts(st, k)
    r = window_gauss_3d(st, p, s_win)
    n = size(r.res)
    cols = [i for i in 1:n[1] if !(i in skip_cols(s_win))]
    J, K = rows === nothing ? (1:n[2], 1:n[3]) : rows
    return Float64(maximum(r.res[cols, J, K]) / (base_eps(T) * r.scale * sqrt(k + 1)))
end

function body_T1032(be)
    C = emC("T-1032")
    pml = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)
    cases = [(T, m, false) for T in TYPES for m in (Reference(), Fast())]
    append!(cases, [(Float64, m, true) for m in (Reference(), Fast())])
    for (T, mode, open) in cases
        prob = window_lwfa_3d(T; bd = open ? (:open, :open) : (:periodic, :periodic))
        st = window_state_3d(prob; mode, be, absorber = open ? pml : nothing, back = open ? PML(; cells = 6, alpha_max = 0.2) : nothing)
        p = params(prob)
        ia = antenna_planes(st)[1].ia
        cb = open ? 6 : 2
        n = ncells(st.grid)
        rows = open ? ((pml.cells + 2):(n[2] - pml.cells), (pml.cells + 2):(n[3] - pml.cells)) : nothing
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 4 == 0 || continue
            worst = max(worst, window_gauss_worst_3d(st, p, k, s -> (1:cb..., ia - 1 - s, ia - s); rows))
        end
        umax = maximum(abs ∘ value_of, Array(st.species[1].u[1]))
        @info "T-1032" T mode open worst shifts = window_shifts(st, prob.nsteps) umax
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        @test umax > (open ? 1 / 50 : 1 / 10)      # with y and z open the PML takes the beam's wings
        record!("T-1032 $(open ? "open, back PML" : "periodic") $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
    tenth(T) = one(T) / 10
    for T in TYPES, mode in (Reference(), Fast()), (label, kw) in (("warm", (vacuum = -100, ramp = 1)),
                                                                   ("warm ramp", (vacuum = 2, ramp = 4, drift = (3 // 10, 0, 0))))
        drift = haskey(kw, :drift) ? T.(kw.drift) : ZERO3(T)
        prob = window_plasma_3d(T; vacuum = kw.vacuum, ramp = kw.ramp, drift, thermal = (tenth(T), tenth(T), tenth(T)), nsteps = 60)
        st = window_state_3d(prob; mode, be)
        p = params(prob)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            worst = max(worst, window_gauss_worst_3d(st, p, k, s -> (1, 2)))
        end
        @info "T-1032 $label" T mode worst shifts = window_shifts(st, prob.nsteps)
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-1032 $label $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end

@testset "T-1032 Gauss's law to roundoff across 3D window shifts with a laser driving the plasma, all three types, both modes [A2, A4, A7, A8, WP-074]" begin
    body_T1032(CPU())
end

# --- T-1033 ---------------------------------------------------------------------------------

# T-470 in 3D: the refill is the engine's quiet start at the global cell. The fresh column of the
# window at position s holds, bit for bit and in the same order, the particles a static load of
# the whole domain of n₁ + s columns gives that column (the lattice offsets along all three axes,
# the Philox draws keyed by the global cell (c₁ + s, c₂, c₃), the weights of the profile at the
# global x), for a thermal, drifting plasma, so a seed means the same thing in the 3D window and in
# a static 3D box (A4); the momenta are γv of the same draws, to a few ulps of the draw's two
# terms (drift + thermal·z: the two kernels may contract that sum into one rounding or not, which
# shows where the terms cancel, measured 51 ulps of |u| in Float64). The ramp length enters through the
# weights only (A3): with it a dual, the positions carry no partials and the weights exactly the
# derivative of the profile, against the closed form of ∂/∂L sin²(π(x − x_v)/(2L)).
function body_T1033(be)
    for T in TYPES
        ramp = seeded(T, 4)
        tenth = one(T) / 10
        prob = window_plasma_3d(T; thermal = (tenth, tenth / 2, tenth / 3), drift = (tenth / 5, tenth / 7, zero(T)), vacuum = 14, ramp,
                                cells = (24, 6, 7), ppc = (2, 1, 3))
        st = EMState(prob, EMNumerics(); backend = be)
        w = st.aux.window
        load = window_load(st, params(prob), 1)
        n = ncells(st.grid)
        s_win = 8                     # the fresh column, global column 29, lies on the ramp (x from 14 to 18)
        col = injection_column(w)
        K = n[2] * n[3] * 6
        sc = Species(be, T, K; q = -one(T), m = one(T), N = 3)
        load_window_columns!(sc, st.grid, load, KernelAbstractions.zeros(be, Int32, 1), col, 1, s_win)
        @test alive_count(sc) == K
        big = PeriodicGrid((n[1] + s_win, n[2], n[3]), cellsize(st.grid), guards(st.grid))
        sb = Species(be, T, prod(ncells(big)) * 6; q = -one(T), m = one(T), N = 3)
        load_quiet!(sb, big, ParticleMeshElectromagnetic.RampedCosine(), load.p, (2, 1, 3); vdrift = load.vdrift, vth = load.vth,
                    seed = prob.seed, stream = 1)
        sel = findall(==(Int32(col + s_win)), Array(sb.cell[1]))
        @test length(sel) == K
        @test all(==(Int32(col)), Array(sc.cell[1]))
        for d in 2:3
            @test Array(sc.cell[d]) == Array(sb.cell[d])[sel]
        end
        for (x, y) in ((sc.offset[1], sb.offset[1]), (sc.offset[2], sb.offset[2]), (sc.offset[3], sb.offset[3]), (sc.w, sb.w))
            @test same_bits(Array(x), Array(y)[sel])
        end
        ub = map(d -> Array(sb.u[d])[sel], 1:3)
        ug = [velocity_momentum((ub[1][i], ub[2][i], ub[3][i])) for i in eachindex(sel)]
        for d in 1:3
            terms(i) = absmax(ug[i][d]) + absmax(load.vdrift[d]) + absmax(ug[i][d] - load.vdrift[d])
            @test all(i -> absdiff(Array(sc.u[d])[i], ug[i][d]) <= 4 * base_eps(T) * terms(i), eachindex(sel))
        end
        # every component of the momenta is drawn: the thermal spread is there along each axis
        @test all(d -> length(unique(value_of.(Array(sc.u[d])))) > K ÷ 2, 1:3)
        if T === D3
            @test all(x -> all(iszero, partials(x)), vcat(map(a -> Array(a), sc.offset)...))
            hx = value(cellsize(st.grid)[1])
            for (δ, wt) in zip(Array(sc.offset[1]), Array(sc.w))
                x = (col + s_win - 1 + value(δ)) * hx
                t = (x - 14) / 4
                @test 0 < t < 1
                dwdL = -(π / 2) * sinpi(t) * t / 4 / 6
                @test collect(Float64.(Tuple(partials(wt)))) ≈ dwdL .* [1 / 8, 2 / 8, 3 / 8] atol = 1e-6
            end
        end
    end
end

@testset "T-1033 the 3D window's refill is the engine's static quiet start at the global cell bit for bit, the profile entering through the weights [A3, A4, A8, WP-074]" begin
    body_T1033(CPU())
end

# --- T-1034 ---------------------------------------------------------------------------------

# The 3D window step in Reference mode is bitwise across reruns, launch shapes and 1 and 4
# threads (`window3d_step_digests`: the laser into its plasma with y, z and the back edge
# absorbing, and a warm plasma whose every refill draws), and the two modes agree to roundoff in
# the fields (Float64, the laser run to 120 steps).
@testset "T-1034 the 3D window step is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode, and the modes agree [A4, WP-074]" begin
    ref = window3d_step_digests()
    @test length(ref) == 2 * length(TYPES) && allunique(last.(ref))
    @test window3d_step_digests() == ref
    for wgs in (32, 256)
        @test window3d_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        fields = split.(em3d_worker_lines(nt, "EM3DW "))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
    prob = window_lwfa_3d(Float64; nsteps = 120)
    a, b = window_state_3d(prob; mode = Reference()), window_state_3d(prob; mode = Fast())
    run!(a, params(prob), prob.nsteps)
    run!(b, params(prob), prob.nsteps)
    cs = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
    m = maximum(c -> maximum(abs, Array(getfield(a.fields, c))), cs)
    d = maximum(c -> maximum(abs, Array(getfield(a.fields, c)) .- Array(getfield(b.fields, c))), cs)
    @info "T-1034 Reference against Fast" rel = d / m
    record!("T-1034", Float64, d / (m * eps(Float64)))
    @test d <= emC("T-1034") * eps(Float64) * m
end

# --- T-1035 ---------------------------------------------------------------------------------

# The gradient through the 3D refill (A3, A7), T-475 in 3D: the field energy after 30 steps of a
# plasma drifting at 0.1 c into the window (the drift against the immobile background charges the
# ramp, so the fields depend on the ramp length L), by ForwardDiff with L a dual, against the
# finite-difference floor of `fd_scan` (Float64, Reference, one seed). Then the same run with L a
# `Dual{Float32,3}`: its primal is the Float32 run's to roundoff and its partial ∂/∂L the central
# difference of Float64 runs at L ± 10⁻³ (A8), on Ex, the only component the drift along x drives.
function body_T1035(be)
    run(L; T = typeof(L), mode = Reference(), nsteps = 30) = begin
        prob = window_plasma_3d(T; ramp = L, drift = (T(1) / 10, zero(T), zero(T)), nsteps, cells = (24, 6, 6), ppc = (2, 1, 1))
        st = window_state_3d(prob; mode, be)
        run!(st, params(prob), prob.nsteps)
        st
    end
    energy(st) = sum(c -> sum(abs2, Array(getfield(st.fields, c))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    f(L, mode, seed) = energy(run(L; mode))
    x0 = 4.0
    ad = ForwardDiff.derivative(x -> f(x, Reference(), 1), x0)
    s = fd_scan(f, x0, [10.0^(-k) for k in 2:7]; seed = 1, ad)
    @info "T-1035" ad fd = s.fd floor = s.floor_rel rel = s.rel[s.istar]
    @test abs(ad) > 0
    record!("T-1035", Float64, s.rel[s.istar] / s.floor_rel, be)
    @test s.rel[s.istar] <= emC("T-1035") * s.floor_rel
    sd, sf = run(mkdual(4, 1, 0, 0)), run(4.0f0)
    δ = 1e-3
    sp, sm = run(4.0 + δ), run(4.0 - δ)
    # every component against the largest value and slope of all six (the drift along x drives Ex;
    # the others are the current's roundoff across a uniform plasma)
    cs = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
    A = map(c -> Array(getfield(sd.fields, c)), cs)
    B = map(c -> Array(getfield(sf.fields, c)), cs)
    slope = map(c -> (Array(getfield(sp.fields, c)) .- Array(getfield(sm.fields, c))) ./ (2δ), cs)
    @test all(a -> all(x -> all(isfinite, partials(x)) && iszero(partials(x)[2]) && iszero(partials(x)[3]), a), A)
    bmax, smax = maximum(b -> maximum(abs, b), B), maximum(a -> maximum(abs, a), slope)
    worst = maximum(i -> maximum(abs, value_of.(A[i]) .- B[i]), 1:6) / (eps(Float32) * bmax * 30)
    worst_partial = maximum(i -> maximum(abs, map(x -> Float64(partials(x)[1]), A[i]) .- slope[i]), 1:6) / (eps(Float32) * smax * 30)
    @test smax > 0 && any(x -> !iszero(partials(x)[1]), A[1])
    @info "T-1035 Dual{Float32,3}" worst worst_partial
    record!("T-1035 primal", D3, worst, be)
    record!("T-1035 partial", D3, worst_partial, be)
    @test worst <= emC("T-1035", "C_primal")
    @test worst_partial <= emC("T-1035", "C_partial")
end

@testset "T-1035 the field energy after 3D window shifts is differentiable in the ramp length through the refill, and Dual{Float32,3} carries it [A3, A7, A8, WP-074]" begin
    body_T1035(CPU())
end

# --- T-1036 ---------------------------------------------------------------------------------

# T-472 in 3D: a vacuum pulse is the same with and without the window. The antenna stays in the
# lab frame, so in the window its plane moves back with the shifts and keeps launching until the
# pulse is over and the window passes it; the static twin is the same problem on a box long enough
# for the whole run with a window that never moves. Before the window moves the two differ only
# where the pulse's precursor and tail meet the window's front, to roundoff, and after it only by
# what the window leaves out, at the back edge (whose difference reaches forward through Yee's
# stencil, falling by orders of magnitude per few columns) and ahead of the front (the pulse's
# tail, six durations ahead of its peak); compared from column 25 on. With a PML of 6 cells on the
# back edge of both, the same holds, and against free space (a static box with 100 more columns
# behind and a bare back too far away to be seen) the eight columns behind the layer differ far
# less than the bare back edge's (T-570 in 3D). Float64, fields only.
function window_vacuum_3d(back; be = CPU(), extra = 100)
    prob = window_lwfa_3d(Float64; plasma = false, nsteps = 200)
    n = prob.cells
    h = prob.length[1] / n[1]
    nums = EMNumerics(; back_absorber = back)
    win = EMState(prob, nums; backend = be)
    initialize!(win, prob)
    s_end = window_shifts(win, prob.nsteps)
    twin = static_twin_3d(prob, n[1] + s_end + 8)
    ref = EMState(twin, nums; backend = be)
    initialize!(ref, twin)
    far = window_lwfa_3d(Float64; plasma = false, nsteps = prob.nsteps, position = 10 + extra * h, cells = (n[1] + s_end + 8 + extra, n[2], n[3]),
                         start = Inf)
    free = EMState(far, EMNumerics(); backend = be)
    initialize!(free, far)
    p, pf = params(prob), params(far)
    g = guards(win.grid)
    J = (g + 1):(g + n[2])
    K = (g + 1):(g + n[3])
    cb = back === nothing ? 0 : back.cells
    before = worst = backdiff = near = 0.0
    for k in 1:prob.nsteps
        step!(win, p, k)
        step!(ref, params(twin), k)
        step!(free, pf, k)
        s = window_shifts(win, k)
        for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
            a, b, f = Array(getfield(win.fields, c)), Array(getfield(ref.fields, c)), Array(getfield(free.fields, c))
            m = maximum(abs, b[(g + 1 + s):(g + n[1] + s), J, K])
            m > 0 || continue
            d(r) = maximum(abs, a[g .+ r, J, K] .- b[g .+ s .+ r, J, K]) / m
            r = (cb + 1):(cb + 8)
            near = max(near, maximum(abs, a[g .+ r, J, K] .- f[g .+ extra .+ s .+ r, J, K]) / m)
            if s == 0
                before = max(before, d(1:n[1]))
            else
                worst = max(worst, d(25:n[1]))
                backdiff = max(backdiff, d(1:8))
            end
        end
    end
    return (; shifts = s_end, n1 = n[1], before, worst, backdiff, near)
end

function body_T1036(be)
    C = emC("T-1036")
    bare = window_vacuum_3d(nothing; be)
    layer = window_vacuum_3d(PML(; cells = 6); be)
    @info "T-1036 the 3D window against its static twin and free space" bare layer
    for r in (bare, layer)
        @test r.shifts > r.n1
        @test r.before <= C * eps(Float64)
        @test r.worst <= C * eps(Float64)
    end
    @test bare.backdiff > 1e-6                 # the back edge does differ from the twin's interior, so the comparison sees it
    record!("T-1036 bare", Float64, bare.worst / eps(Float64), be)
    record!("T-1036 back PML", Float64, layer.worst / eps(Float64), be)
    d = get!(MEASURED, "T-1036 near the back edge against free space", Dict{String,Float64}())
    d["bare"], d["back PML"] = bare.near, layer.near
    @test layer.near <= emC("T-1036", "C_near") * bare.near
end

@testset "T-1036 a vacuum pulse is the same with and without the 3D window off the back edge, the antenna left in the lab frame, and the back-edge PML absorbs [A2, A7, WP-074]" begin
    body_T1036(CPU())
end

# --- T-1037 ---------------------------------------------------------------------------------

# The checks and the parts of the 3D window state: a window on another axis, without its
# MovingWindow, with an amplitude, background fields or a warm plasma before a late start is
# refused, and so is a back edge's absorber without a window or other than a PML, an antenna
# inside the back edge's PML or passed while it emits, a window too short for the shape's front
# margin, and the filter; the state holds its modifiers in the order of `field_modifiers` (the open
# axes' PMLs, the back edge's, the antennas with the window's clock, the window's edges), its
# buffers with the headroom, and evolves the back edge's convolution fields and the counters once
# each; a zero-profile back-edge PML is the bare window's step bit for bit, and a run restored
# from a snapshot continues bit for bit.
@testset "T-1037 input checks and the parts of the 3D window state [A2, A3, A10, WP-074]" begin
    T = Float64
    e = SpeciesDescription{T}(:e, -1.0, 1.0, 1.0, 0.0, (1, 0, 0), (1, 1, 1), ZERO3(T), ZERO3(T), 1)
    ep = SpeciesDescription{T}(:e, -1.0, 1.0, 1.0, 0.1, (1, 0, 0), (1, 1, 1), ZERO3(T), ZERO3(T), 1)
    box, cells = (8.0, 3.0, 3.0), (16, 6, 6)
    w = MovingWindow(; start = 0)
    mk(sp = (e,); kw...) = EMProblem("x", box, cells, 0.25, 4, sp, 1, (1, 0, 0); kw...)
    @test mk(; boundaries = (:window, :periodic, :periodic), window = w) isa EMProblem{3}
    @test mk(; boundaries = (:window, :open, :periodic), window = w, profile = DensityRamp(; vacuum = 1, ramp = 2)) isa EMProblem{3}
    @test_throws ArgumentError mk(; boundaries = (:window, :periodic, :periodic))                 # no MovingWindow
    @test_throws ArgumentError mk(; window = w)                                                  # axis 1 not :window
    @test_throws ArgumentError mk(; boundaries = (:periodic, :periodic, :window), window = w)     # the window is axis 1
    @test_throws ArgumentError mk(; profile = DensityRamp(; vacuum = 1, ramp = 2))                # a profile needs the window
    @test_throws ArgumentError mk(; boundaries = (:window, :periodic, :periodic), window = w, B0 = (0.0, 0.0, 0.1))
    @test_throws ArgumentError mk((ep,); boundaries = (:window, :periodic, :periodic), window = w)
    @test_throws ArgumentError window_plasma_3d(T; start = 1.0, thermal = (0.01, 0.01, 0.01))
    @test window_plasma_3d(T; start = 1.0) isa EMProblem{3}
    @test_throws ArgumentError EMState(window_plasma_3d(T), EMNumerics(; filter = BinomialFilter(1)))
    @test_throws ArgumentError EMState(window_plasma_3d(T), EMNumerics(; back_absorber = AbsorbingLayer(; cells = 4, strength = 1)))
    @test_throws ArgumentError EMState(plasma_step_3d(T), EMNumerics(; back_absorber = PML(; cells = 4)))      # no window
    @test_throws ArgumentError EMState(window_plasma_3d(T; cells = (8, 6, 6)), EMNumerics(; back_absorber = PML(; cells = 4)))   # 2·4 ≥ 8
    # the fewest columns the guards allow leave plasma columns behind the front margin (the 2D check of a too short window needs filter passes)
    @test injection_column(EMState(window_plasma_3d(T; cells = (6, 6, 6)), EMNumerics()).aux.window) == 3
    wp = window_plasma_3d(T)
    @test_throws ArgumentError initialize!(EMState(wp, EMNumerics()), wp; density_at = :cell_centre)
    # the antennas: passed while emitting (the window starts too early), inside the back edge's PML
    @test_throws ArgumentError EMState(window_lwfa_3d(T; start = 20), EMNumerics())
    @test_throws ArgumentError EMState(window_lwfa_3d(T; position = 4), EMNumerics(; back_absorber = PML(; cells = 6)))
    @test EMState(window_lwfa_3d(T; position = 4, start = 40), EMNumerics()) isa EMState{3}
    @test_throws ArgumentError EMState(window_lwfa_3d(T; position = 4, start = 40), EMNumerics(; back_absorber = PML(; cells = 6)))
    # the parts of the state
    prob = window_lwfa_3d(T; nsteps = 60, bd = (:open, :open))
    pml = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)
    st = EMState(prob, EMNumerics(; absorber = pml, back_absorber = PML(; cells = 6), headroom = 1 / 2))
    @test map(m -> nameof(typeof(m)), st.modifiers) == (:PMLFields, :PMLFields, :BackPMLFields, :AntennaPlanes, :WindowEdges)
    @test st.modifiers[4].clock == ParticleMeshElectromagnetic.WindowClock(st.aux.window)
    @test back_absorber(st) === st.modifiers[3] && open_absorbers(st) === st.modifiers[1:2]
    @test capacity(st.species[1]) == ceil(Int, 3 / 2 * prod(prob.cells) * 2)
    @test injection_column(st.aux.window) == prob.cells[1] - 3
    ψ = values(back_absorber(st).ψ)
    @test keys(back_absorber(st).ψ) == (:By, :Bz, :Ey, :Ez) && all(a -> size(a) == (6, 12, 12), ψ)
    @test all(a -> count(b -> a === b, evolved(st)) == 1, (ψ..., st.aux.window.lost..., st.aux.window.fast...))
    # a zero-profile back-edge PML is the bare window's step bit for bit (its corrections add zeros)
    for T in TYPES
        pz = window_lwfa_3d(T; nsteps = 90)
        a = window_state_3d(pz; back = PML(; cells = 6, sigma_max = 0, kappa_max = 1, alpha_max = 0))
        b = window_state_3d(pz)
        run!(a, params(pz), pz.nsteps)
        run!(b, params(pz), pz.nsteps)
        @test window_shifts(a, pz.nsteps) > 0
        @test all(c -> bits(Array(getfield(a.fields, c))) == bits(Array(getfield(b.fields, c))), keys(a.fields))
        @test particle_set_3d(a.species[1]) == particle_set_3d(b.species[1])
    end
    # a snapshot after 30 steps, restored into a state that ran on, continues bit for bit
    p = params(prob)
    initialize!(st, p)
    run!(st, p, 30)
    snap = snapshot(st, 30)
    run!(st, p, 30; first = 31)
    want = map(a -> bits(Array(a)), evolved(st))
    restore!(st, snap)
    run!(st, p, 30; first = 31)
    @test map(a -> bits(Array(a)), evolved(st)) == want
    @test window_shifts(st, 60) > 0
end

# --- T-1038 ---------------------------------------------------------------------------------

# T-476 in 3D, the edges and the buffers. A plasma drifting backwards at 0.9 c (and across y and z)
# crosses the back edge between shifts: exactly the particles whose cell along x jumped across the
# seam in a step are retired, before the current (Fast mode, Float64, which keeps the slots), so in
# a step without a shift Gauss's law does not change on the nodes 3 … n₁. A dilute plasma (10⁻⁴ n_c)
# drifting forwards at 0.5 c stays twice as long in the window as one at rest: with no headroom
# the refills run out of slots and count what they could not place, with a headroom of 3/2 none
# is lost.
function body_T1038(be)
    prob = window_plasma_3d(Float64; drift = (-0.9, 0.2, -0.1), nsteps = 20, vacuum = -10)
    st = window_state_3d(prob; mode = Fast(), be)
    p = params(prob)
    retired = 0
    for k in 1:prob.nsteps
        s = st.species[1]
        before = (status = Array(s.status), c1 = Array(s.cell[1]))
        shifts = window_shifts(st, k) - window_shifts(st, k - 1)
        r0 = shifts == 0 ? window_gauss_3d(st, p, window_shifts(st, k)).res : nothing
        step!(st, p, k)
        shifts == 0 || continue
        r1 = window_gauss_3d(st, p, window_shifts(st, k))
        @test maximum(abs.(r1.res[3:end, :, :] .- r0[3:end, :, :])) <= 64 * eps(Float64) * r1.scale
        was = before.status .== ParticleMeshCore.ALIVE
        now = Array(s.status) .== ParticleMeshCore.ALIVE
        jumped = was .& (abs.(Array(s.extras.old_cell_1) .- Array(s.cell[1])) .> 1)
        @test (was .& .!now) == jumped
        retired += count(jumped)
    end
    @info "T-1038 retired at the back edge" retired
    @test retired > 10
    fwd = window_plasma_3d(Float64; drift = (0.5, 0.0, 0.0), nsteps = 46, vacuum = -10, density = 1e-4)
    tight = window_state_3d(fwd; headroom = 0, be)
    run!(tight, params(fwd), fwd.nsteps)
    roomy = window_state_3d(fwd; headroom = 3 / 2, be)
    run!(roomy, params(fwd), fwd.nsteps)
    @info "T-1038 lost particles" tight = window_lost(tight) roomy = window_lost(roomy)
    @test window_lost(tight)[1] > 0
    @test window_lost(roomy) == (0,)
end

@testset "T-1038 particles crossing the 3D window's back edge are retired before their current, and the buffers count what does not fit [A2, A4, WP-074]" begin
    body_T1038(CPU())
end
