# WP-034: the PML on the moving window's back edge (WP-028), T-570 to T-576. The window's x-guards
# are its field-free outside, which reflects what reaches the back edge: in 2D the trailing field a
# pulse leaves behind it (T-472). A `PML` as `back_absorber` absorbs it in the window's first
# `cells` columns; its convolution fields move back with E and B at each shift.

using ParticleMeshElectromagnetic: BackPMLFields, back_b!, back_e!, shift_absorber!

# --- T-570 ----------------------------------------------------------------------------------

# T-472's pulse: without a back absorber the window differs from free space by about 0.1 of the
# largest field on its back columns (the trailing field reflected by the field-free outside). With a
# PML of 8 cells (PIConGPU's profile), the difference on the eight columns behind the layer is more
# than 100 times smaller. Against the static twin with the same back edge, the window is the same to
# roundoff before it moves and keeps T-472's agreement from column 25 on.
@testset "T-570 a PML on the moving window's back edge removes the back-edge reflection of T-472's pulse [A2, A7, WP-034]" begin
    C = emC("T-472")
    bare = back_comparison(nothing)
    pml = back_comparison(PML(; cells = 8))
    @info "T-570 back edge" bare = bare.near pml = pml.near pml.before pml.far shifts = pml.shifts
    @test pml.shifts > 64
    @test bare.near > 0.05
    @test pml.near < emC("T-570") * bare.near
    @test pml.before <= C * eps(Float64)
    @test pml.far <= C * eps(Float64)
    record!("T-570", Float64, pml.near / bare.near)
end

# --- T-571 ----------------------------------------------------------------------------------

# A zero profile adds exactly zero: the window's laser–plasma run with a back-edge PML of σ = α = 0,
# κ = 1 is the run without one bit for bit, fields and particles (120 steps, past the window's
# start), and with an open y axis the y-PML's step and the matched layer's apply the same back-edge
# corrections. Each correction on its own
# changes only its two components on the layer columns, advancing its convolution fields only
# where asked; a shift moves ψ one column back with the last column zero; step 1 zeroes ψ.
@testset "T-571 a zero-profile back-edge PML is the window's step bit for bit; corrections, shift and reset [A2, A4, A8, WP-034]" begin
    for T in TYPES
        prob = back_lwfa(T; nsteps = 120)
        a = back_state(prob; back = PML(; cells = 6, sigma_max = 0, kappa_max = 1, alpha_max = 0))
        b = back_state(prob)
        run!(a, params(prob), prob.nsteps)
        run!(b, params(prob), prob.nsteps)
        @test window_shifts(a, prob.nsteps) > 0
        @test all(c -> bits(Array(getfield(a.fields, c))) == bits(Array(getfield(b.fields, c))), keys(a.fields))
        @test particle_set(a.species[1]) == particle_set(b.species[1])
    end
    # with an open y axis: the y-PML's step (zero profile) and the matched layer's (zero strength)
    # apply the same back-edge corrections, bit for bit
    warm = window_plasma(Float64; cells = (24, 16), thermal = (0.1, 0.1, 0.1), nsteps = 60, open_y = true)
    edge = PML(; cells = 6, kappa_max = 2, alpha_max = 0.1)
    a = back_state(warm; back = edge, absorber = PML(; cells = 4, sigma_max = 0, kappa_max = 1, alpha_max = 0))
    b = back_state(warm; back = edge, absorber = AbsorbingLayer(; cells = 4, strength = 0))
    run!(a, params(warm), warm.nsteps)
    run!(b, params(warm), warm.nsteps)
    @test window_shifts(a, warm.nsteps) > 0
    @test all(c -> bits(Array(getfield(a.fields, c))) == bits(Array(getfield(b.fields, c))), keys(a.fields))
    @test all(c -> getfield(a.aux.window.back, c) == getfield(b.aux.window.back, c), (:ψBy, :ψBz, :ψEy, :ψEz))
    @test particle_set(a.species[1]) == particle_set(b.species[1])
    prob = back_lwfa(Float64; nsteps = 4)
    st = back_state(prob; back = PML(; cells = 6, kappa_max = 2, alpha_max = 0.1))
    foreach(a -> (a .= randn(Xoshiro(21), size(a)); fill_guards!(a, st.grid)), values(st.fields))
    g, n = guards(st.grid), ncells(st.grid)
    rows, cols = (g + 1):(g + n[2]), (g + 1):(g + 6)
    pml = PML(; cells = 6, kappa_max = 2, alpha_max = 0.1)
    for (fn, changed, ψs, adv) in ((back_b!, (:By, :Bz), (:ψBy, :ψBz), false), (back_b!, (:By, :Bz), (:ψBy, :ψBz), true),
                                   (back_e!, (:Ey, :Ez), (:ψEy, :ψEz), true))
        L = BackPMLFields(pml, st.grid, prob.dt; backend = CPU())
        foreach(c -> (getfield(L, c) .= randn(Xoshiro(23), size(getfield(L, c)))), (:ψBy, :ψBz, :ψEy, :ψEz))
        ψ0 = map(c -> copy(getfield(L, c)), (ψBy = :ψBy, ψBz = :ψBz, ψEy = :ψEy, ψEz = :ψEz))
        f = map(copy, st.fields)
        fn === back_b! ? fn(L, f, st.grid, prob.dt / 2, adv, nothing) : fn(L, f, st.grid, prob.dt, nothing)
        for c in keys(f)
            if c in changed
                @test all(i -> Array(getfield(f, c))[i, rows] != Array(getfield(st.fields, c))[i, rows], cols)
                @test Array(getfield(f, c))[(g + 7):(g + n[1]), rows] == Array(getfield(st.fields, c))[(g + 7):(g + n[1]), rows]
            else
                @test bits(Array(getfield(f, c))) == bits(Array(getfield(st.fields, c)))
            end
        end
        for c in keys(ψ0)
            @test (getfield(L, c) == ψ0[c]) == !(adv && c in ψs)
        end
    end
    L = st.aux.window.back
    foreach(c -> (getfield(L, c) .= randn(Xoshiro(29), size(getfield(L, c)))), (:ψBy, :ψBz, :ψEy, :ψEz))
    old = map(c -> copy(getfield(L, c)), (:ψBy, :ψBz, :ψEy, :ψEz))
    shift_absorber!(L, nothing)
    for (c, o) in zip((:ψBy, :ψBz, :ψEy, :ψEz), old)
        @test getfield(L, c)[1:5, :] == o[2:6, :] && all(iszero, getfield(L, c)[6, :])
    end
    # step 1 zeroes the convolution fields: a state with stale ψ takes the fresh state's first step
    fresh = back_state(prob; back = pml)
    used = back_state(prob; back = pml)
    foreach(c -> (getfield(used.aux.window.back, c) .= 1), (:ψBy, :ψBz, :ψEy, :ψEz))
    step!(fresh, params(prob), 1)
    step!(used, params(prob), 1)
    @test all(c -> bits(Array(getfield(fresh.fields, c))) == bits(Array(getfield(used.fields, c))), keys(fresh.fields))
end

# --- T-572 ----------------------------------------------------------------------------------

# Gauss's law across window shifts with the back-edge PML: its corrections change Ey and Ez on the
# layer's Node() columns, so D⁻·E = ρ − ρ₀(s) breaks on nodes 1 … cells and holds to roundoff on
# every other node but the antenna's two columns (T-473's run with a laser driving the plasma,
# T-473's scale and constant), all three types, both modes.
@testset "T-572 Gauss's law to roundoff off the back-edge PML across window shifts, all three types, both modes [A2, A7, A8, WP-034]" begin
    C = emC("T-473")
    cb = 6
    for T in TYPES, mode in (Reference(), Fast())
        prob = back_lwfa(T)
        st = back_state(prob; back = PML(; cells = cb, alpha_max = 0.2), mode)
        p = params(prob)
        ia = st.aux.antennas[1].ia
        worst = inside = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 3 == 0 || continue
            s_win = window_shifts(st, k)
            r = window_gauss(st, s_win)
            cols = [i for i in axes(r.res, 1) if i > cb && !(i in (ia - 1 - s_win, ia - s_win))]
            worst = max(worst, Float64(maximum(r.res[cols, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
            inside = max(inside, Float64(maximum(r.res[1:cb, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        @info "T-572" T mode worst inside shifts = window_shifts(st, prob.nsteps)
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-572 $(nameof(typeof(mode)))", T, worst)
        @test worst <= C
    end
end

# --- T-573 ----------------------------------------------------------------------------------

@testset "T-573 input checks of the back-edge PML [A2, A10, WP-034]" begin
    prob = back_lwfa(Float64; nsteps = 4)
    @test_throws ArgumentError back_state(prob; back = AbsorbingLayer(; cells = 4, strength = 1))
    @test_throws ArgumentError back_state(prob; back = PML(; cells = 2))     # thinner than the guards
    @test back_state(prob; back = PML(; cells = 8)).aux.window.back isa BackPMLFields
    @test back_state(prob; back = PML(; cells = 4)).aux.window.back.sigma_max ≈ 0.8 * 5 / cellsize(back_state(prob).grid)[1]
    # 2·cells < n₁, on a window without a laser
    dark = window_lwfa(Float64; nsteps = 4, laser = false)
    @test_throws ArgumentError back_state(dark; back = PML(; cells = 32))
    @test back_state(dark; back = PML(; cells = 31)).aux.window.back isa BackPMLFields
    # the antenna (window_lwfa's, columns 8 and 9) stays out of the layer while it emits
    lwfa = window_lwfa(Float64; nsteps = 4)
    @test_throws ArgumentError back_state(lwfa; back = PML(; cells = 8))      # inside the layer from the start
    @test_throws ArgumentError back_state(lwfa; back = PML(; cells = 6))      # reached before its pulse is over
    @test back_state(lwfa; back = PML(; cells = 4)).aux.window.back isa BackPMLFields
    @test back_state(window_lwfa(Float64; nsteps = 4, start = 40); back = PML(; cells = 6)).aux.window.back isa BackPMLFields
    plain, _ = open_plasma_2d(Float64)
    @test_throws ArgumentError EMState(plain, EMNumerics(; absorber = PML(; cells = 4), back_absorber = PML(; cells = 4)))
    @test_throws ArgumentError EMState(beams_1d(Float64), EMNumerics(; back_absorber = PML(; cells = 4)))   # no window at N = 1
end

# --- T-574 ----------------------------------------------------------------------------------

# The window's step with the back-edge PML is bitwise across reruns and launch shapes in Reference
# mode, the convolution fields included.
function back_digest(; workgroupsize = nothing)
    prob = back_lwfa(Float64; nsteps = 120)
    st = back_state(prob; back = PML(; cells = 6, kappa_max = 2, alpha_max = 0.1), workgroupsize)
    run!(st, params(prob), prob.nsteps)
    h = state_digest_2d(st)
    L = st.aux.window.back
    for a in (L.ψBy, L.ψBz, L.ψEy, L.ψEz)
        h = hash(bits(a), h)
    end
    return h
end

@testset "T-574 the window's step with the back-edge PML is bitwise across reruns and launch shapes in Reference mode [A4, WP-034]" begin
    ref = back_digest()
    @test back_digest() == ref
    @test back_digest(; workgroupsize = 32) == ref
    @test back_digest(; workgroupsize = 256) == ref
end

# --- T-575 ----------------------------------------------------------------------------------

# A long run: random fields in the empty window with PIConGPU's 8-cell back-edge PML, 4000 steps.
# In a window that never moves the energy never exceeds that of the same run without the layer
# (whose field-free outside reflects everything, about 1.17 of the start at its largest, the
# staggered energy of random fields) and ends below half of it; in one that moves from t = 0 everything that does not travel with it leaves through
# the layer, and less than 5 % of the energy stays.
@testset "T-575 random fields with the back-edge PML stay bounded over 4000 steps, the window static and moving [A2, A7, WP-034]" begin
    bare = back_long_run(nothing; start = Inf, nsteps = 4000)
    still = back_long_run(PML(; cells = 8); start = Inf, nsteps = 4000)
    moving = back_long_run(PML(; cells = 8); start = 0, nsteps = 4000)
    @info "T-575" bare = (maximum(bare.w), last(bare.w)) still = (maximum(still.w[2:end]), last(still.w)) moving = (maximum(moving.w[2:end]), last(moving.w))
    @test maximum(still.w) <= maximum(bare.w)
    @test last(still.w) < last(bare.w) / 2
    @test maximum(moving.w[10:end]) < 0.05
end

# --- T-576 ----------------------------------------------------------------------------------

# The back edge against its discrete theory: WP-027's slab packet turned on its side travels along
# −x into the back-edge PML of a static window, whose zero x-guards reflect what the layer leaves,
# and the energy that comes back matches `back_rt` (pml_back_cases.jl) at T-540's scale.
# Cells are not square (8 per wavelength along x, 16 along y), the angles 0 and 0.4 rad, both
# polarizations, PIConGPU's profile, Smilei's grading with α and κ with α. The theory first: a zero
# profile reflects everything (|r| = 1), and at normal incidence TE and TM see the same |r|.
@testset "T-576 the back edge's reflection of a slab packet against its discrete theory, both polarizations, κ and α [A2, A7, WP-034]" begin
    zero_pml = PML(; cells = 8, sigma_max = 0, kappa_max = 1, alpha_max = 0)
    mixed = PML(; cells = 8, sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4, alpha_max = 0.1)
    for θ in (0.0, 0.4), pol in (:TE, :TM)
        geo = back_slab_geometry(; theta = θ)
        for (kx, _) in back_slab_modes(geo)[1:5:end]
            @test abs(abs(back_rt(geo, zero_pml, pol, kx)) - 1) < 1e-12
            θ == 0 && @test abs(abs(back_rt(geo, mixed, :TE, kx)) - abs(back_rt(geo, mixed, :TM, kx))) <= 1e-10 * abs(back_rt(geo, mixed, :TE, kx))
        end
    end
    C = emC("T-576")
    worst = 0.0
    cfgs = [(0.0, PML(; cells = 8)), (0.4, PML(; cells = 8)), (0.4, mixed), (0.4, PML(; cells = 8, kappa_max = 4, alpha_max = 0.2))]
    for (θ, pml) in cfgs, pol in (:TE, :TM)
        geo = back_slab_geometry(; theta = θ)
        r = back_slab_run(Float64, geo, pml, pol)
        pred = back_prediction(geo, pml, pol)
        e = abs(r.R - pred) / slab_scale(Float64, r.k, pred)
        @info "T-576" θ pml.kappa_max pml.alpha_max pol R = r.R R_theory = pred ratio = e
        worst = max(worst, e)
        @test 0 < pred < 1e-3
        @test e <= C
    end
    record!("T-576", Float64, worst)
end
