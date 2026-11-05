# WP-034: the PML on the moving window's back edge (WP-028), T-570 to T-574. The window's x-guards
# are its field-free outside, which reflects what reaches the back edge: in 2D the trailing field a
# pulse leaves behind it (T-472). A `PML` as `back_absorber` absorbs it in the window's first
# `cells` columns; its convolution fields move back with E and B at each shift.

using ParticleMeshElectromagnetic: BackPMLFields, back_b!, back_e!, shift_absorber!

"A window state with `back` as the back edge's absorber."
function back_state(prob; back = nothing, mode = Reference(), workgroupsize = nothing)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, headroom = 1 / 4, back_absorber = back))
    initialize!(st, prob)
    return st
end

"""
    back_comparison(back; nsteps = 200) -> (; shifts, before, near, far)

T-472's vacuum pulse in the window against its static twin, both with `back` on their back edge:
the largest difference relative to the twin's largest field, before the window moves (`before`,
every column), and after, on the eight columns behind the layer (`near`, columns cells + 1 …
cells + 8) and on columns 25 … n₁ (`far`, T-472's comparison).
"""
function back_comparison(back; nsteps = 200)
    prob = window_lwfa(Float64; plasma = false, nsteps)
    n = prob.cells
    win = back_state(prob; back)
    shifts = window_shifts(win, prob.nsteps)
    ref = back_state(static_twin(prob, n[1] + shifts + 8); back)
    p = params(prob)
    g = guards(win.grid)
    J = (g + 1):(g + n[2])
    cb = back === nothing ? 0 : back.cells
    before = near = far = 0.0
    for k in 1:prob.nsteps
        step!(win, p, k)
        step!(ref, p, k)
        s = window_shifts(win, k)
        for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
            a, b = Array(getfield(win.fields, c)), Array(getfield(ref.fields, c))
            m = maximum(abs, b[(g + 1 + s):(g + n[1] + s), J])
            m > 0 || continue
            d(r) = maximum(abs, a[g .+ r, J] .- b[g .+ s .+ r, J]) / m
            if s == 0
                before = max(before, d(1:n[1]))
            else
                near = max(near, d((cb + 1):(cb + 8)))
                far = max(far, d(25:n[1]))
            end
        end
    end
    return (; shifts, before, near, far)
end

# --- T-570 ----------------------------------------------------------------------------------

# T-472's pulse: without a back absorber the window differs from its static twin by 0.15 of the
# largest field on its back columns (the trailing field reflected by the field-free outside).
# With a PML of 8 cells (PIConGPU's profile) on both, the difference on the eight columns behind
# the layer is more than 100 times smaller; before the window moves the two are the same to
# roundoff (the same back edge), and from column 25 on the window keeps T-472's agreement.
@testset "T-570 a PML on the moving window's back edge removes the back-edge reflection of T-472's pulse [A2, A7, WP-034]" begin
    C = emC("T-472")
    bare = back_comparison(nothing)
    pml = back_comparison(PML(; cells = 8))
    @info "T-570 back edge" bare pml
    @test pml.shifts > 64
    @test bare.near > 0.1
    @test pml.near < emC("T-570") * bare.near
    @test pml.before <= C * eps(Float64)
    @test pml.far <= C * eps(Float64)
    record!("T-570", Float64, pml.near / bare.near)
end

# --- T-571 ----------------------------------------------------------------------------------

# A zero profile adds exactly zero: the window's laser–plasma run with a back-edge PML of σ = α = 0,
# κ = 1 is the run without one bit for bit, fields and particles (120 steps, past the window's
# start). Each correction on its own
# changes only its two components on the layer columns, advancing its convolution fields only
# where asked; a shift moves ψ one column back with the last column zero; step 1 zeroes ψ.
@testset "T-571 a zero-profile back-edge PML is the window's step bit for bit; corrections, shift and reset [A2, A4, A8, WP-034]" begin
    for T in TYPES
        prob = window_lwfa(T; nsteps = 120)
        a = back_state(prob; back = PML(; cells = 6, sigma_max = 0, kappa_max = 1, alpha_max = 0))
        b = back_state(prob)
        run!(a, params(prob), prob.nsteps)
        run!(b, params(prob), prob.nsteps)
        @test window_shifts(a, prob.nsteps) > 0
        @test all(c -> bits(Array(getfield(a.fields, c))) == bits(Array(getfield(b.fields, c))), keys(a.fields))
        @test particle_set(a.species[1]) == particle_set(b.species[1])
    end
    prob = window_lwfa(Float64; nsteps = 4)
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
        prob = window_lwfa(T)
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
    prob = window_lwfa(Float64; nsteps = 4)
    @test_throws ArgumentError back_state(prob; back = AbsorbingLayer(; cells = 4, strength = 1))
    @test_throws ArgumentError back_state(prob; back = PML(; cells = 2))     # thinner than the guards
    @test_throws ArgumentError back_state(prob; back = PML(; cells = 32))    # 2·cells ≥ n₁
    @test back_state(prob; back = PML(; cells = 31)).aux.window.back isa BackPMLFields
    plain, _ = open_plasma_2d(Float64)
    @test_throws ArgumentError EMState(plain, EMNumerics(; absorber = PML(; cells = 4), back_absorber = PML(; cells = 4)))
    @test back_state(prob; back = PML(; cells = 4)).aux.window.back.sigma_max ≈ 0.8 * 5 / cellsize(back_state(prob).grid)[1]
end

# --- T-574 ----------------------------------------------------------------------------------

# The window's step with the back-edge PML is bitwise across reruns and launch shapes in Reference
# mode, the convolution fields included.
function back_digest(; workgroupsize = nothing)
    prob = window_lwfa(Float64; nsteps = 120)
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
