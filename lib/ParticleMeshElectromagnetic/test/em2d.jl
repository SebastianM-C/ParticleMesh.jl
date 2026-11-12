# WP-024: the module's problem, state, push and diagnostics at N = 2, in prescribed fields
# (T-350 to T-358). The 2D step here deposits no current (`NoCurrent()`); test particles are a
# species of zero density, so their weights, charge and current are zero.

using ParticleMeshElectromagnetic: kick_2d_kernel!, fields_at, FIELD_LAYOUT_2D
using ParticleMeshCore: stagger_offset
using ForwardDiff: partials

const L2D = FIELD_LAYOUT_2D

"Phase of (u₁, u₂) in Float64."
uphase(u) = atan(Float64(value_of(u[2])), Float64(value_of(u[1])))
"The difference of two angles, wrapped to (−π, π]."
wrap(Δ) = mod(Δ + π, 2π) - π

# --- T-350 ----------------------------------------------------------------------------------

@testset "T-350 test particles in prescribed uniform fields at N = 2 [A1, A2, WP-024]" begin
    C, Cφ = emC("T-350"), emC("T-350", "C_phase")
    @testset "gyration in uniform Bz, $(type_label(T))" for T in TYPES
        # q/m = −1 and B along +z: each Boris step turns (u₁, u₂) by +2·atan(BΔt/2γ) (T-211); the
        # back kick of initialize! (the pusher with −Δt/2) by −2·atan(BΔt/4γ).
        B0, dt, nsteps = 2, 1 // 10, 200
        worstu = worstφ = 0.0
        transverse = true
        for u0 in (1 // 10, 1)
            prob = test_particles_2d(T; dt, nsteps, u0 = (T(u0), zero(T), zero(T)), B0 = (zero(T), zero(T), T(B0)))
            st = state_2d(prob)
            p = params(prob)
            γ = sqrt(1 + big(u0)^2)
            umag0 = absmax(T(u0))
            φ0 = -2 * atan(big(B0) * big(dt) / (4γ))
            θ = 2 * atan(big(B0) * big(dt) / (2γ))
            for k in 1:nsteps
                step!(st, p, k)
                s = first(st.species)
                for i in 1:alive_count(s)
                    u = (s.u[1][i], s.u[2][i], s.u[3][i])
                    worstu = max(worstu, absdiff(sqrt(u[1]^2 + u[2]^2 + u[3]^2), T(u0)) / (base_eps(T) * umag0 * k))
                    worstφ = max(worstφ, abs(wrap(uphase(u) - Float64(φ0 + k * θ))) / (base_eps(T) * k))
                    transverse &= u[3] === zero(T)
                end
            end
        end
        @test transverse
        record!("T-350 |u|", T, worstu)
        record!("T-350 phase", T, worstφ)
        @test worstu <= C
        @test worstφ <= Cφ
    end

    @testset "the E×B drift in uniform Ey and Bz, $(type_label(T))" for T in (Float32, Float64)
        # T-211's crossed fields from rest: E = E₀ŷ, B = B₀ẑ, v_d = E₀/B₀ along x.
        E0, B0, dt, nsteps = T(1) / 2, one(T), T(1) / 20, 40_000
        prob = test_particles_2d(T; cells = (6, 6), length = (T(3), T(3)), dt, nsteps, ppc = (1, 1),
                                 E0 = (zero(T), E0, zero(T)), B0 = (zero(T), zero(T), B0))
        st = state_2d(prob)
        p = params(prob)
        vmean = 0.0
        for k in 1:nsteps
            step!(st, p, k)
            u = first_u(st)
            vmean += Float64(u[1]) / sqrt(1 + Float64(u[1])^2 + Float64(u[2])^2 + Float64(u[3])^2)
        end
        rel = (vmean / nsteps - E0 / B0) / (E0 / B0)
        @info "T-350 E×B drift" T rel
        d = get!(MEASURED, "T-350 drift relative error", Dict{String,Float64}())
        d[type_label(T)] = abs(rel)
        @test abs(rel) <= emC("T-211", "drift_rel")
    end

    @testset "an oblique uniform E accelerates along E on both axes, $(type_label(T))" for T in TYPES
        # B = 0: Boris is exact, u^{k−1/2} = (q/m)·E·(k − 1/2)Δt, and the particle moves by
        # Σ_j Δt·u^{j−1/2}/γ along each axis. E₁ ≠ E₂, so exchanged components fail.
        E = (T(1) / 10, -T(1) / 4, zero(T))
        dt, nsteps = T(1) / 10, 20
        prob = test_particles_2d(T; cells = (8, 8), length = (T(4), T(4)), dt, nsteps, ppc = (1, 1), E0 = E)
        st = state_2d(prob)
        p = params(prob)
        s = first(st.species)
        x0 = [ntuple(d -> Float64(value_of((s.cell[d][i] - 1 + s.offset[d][i]) * cellsize(st.grid)[d])), 2) for i in 1:alive_count(s)]
        disp = zeros(2)
        umax = maximum(e -> abs(Float64(value_of(e))), E) * Float64(value_of(dt))   # |Δu| per step
        worstu = worstx = 0.0
        for k in 1:nsteps
            step!(st, p, k)
            uref = ntuple(d -> -Float64(value_of(E[d])) * (k - 1 / 2) * Float64(value_of(dt)), 3)
            γ = sqrt(1 + sum(abs2, uref))
            disp .+= Float64(value_of(dt)) .* (uref[1] / γ, uref[2] / γ)
            for i in 1:alive_count(s)
                for d in 1:3
                    worstu = max(worstu, abs(Float64(value_of(s.u[d][i])) - uref[d]) / (base_eps(T) * k * umax))
                end
                for d in 1:2
                    x = Float64(value_of((s.cell[d][i] - 1 + s.offset[d][i]) * cellsize(st.grid)[d]))
                    Δ = mod(x - x0[i][d] - disp[d] + 2, 4) - 2         # periodic box of side 4
                    worstx = max(worstx, abs(Δ) / (base_eps(T) * k * 4))
                end
            end
        end
        @test disp[1] < 0 && disp[2] > 0 && abs(disp[2] / disp[1] + 5 / 2) < 1e-6    # along (q/m)·E, q/m = −1
        record!("T-350 oblique u", T, worstu)
        record!("T-350 oblique x", T, worstx)
        @test worstu <= C
        @test worstx <= C
    end
end

# --- T-351 ----------------------------------------------------------------------------------

@testset "T-351 the 2D gather picks each component at its own stagger [A7, WP-024]" begin
    C = emC("T-351")
    for T in TYPES, shape in SHAPES
        ε = base_eps(T)
        B = basetype(T)
        grid = PeriodicGrid((7, 6), (B(1) / 2, B(2) / 3), em_guard_width(shape))
        f = em_fields(CPU(), T, grid)
        g, h = guards(grid), cellsize(grid)
        # component c holds a + b·x + c·y at the positions of its own stagger, guards included
        # (every particle is then away from the seam for the polynomial, as in T-264).
        coef = Dict(c => (B(i) / 3, B(2i - 7) / 5, B(5 - i) / 4) for (i, c) in enumerate(keys(L2D)))
        xnode(idx, d, st) = (idx - g - 1 + stagger_offset(st[d], B)) * h[d]
        for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
            a, b, cy = coef[c]
            st = L2D[c]
            A = getfield(f, c)
            for i2 in axes(A, 2), i1 in axes(A, 1)
                A[i1, i2] = a + b * xnode(i1, 1, st) + cy * xnode(i2, 2, st)
            end
        end
        rng = Xoshiro(7)
        worst = 0.0
        worst_transposed = Inf
        for _ in 1:300
            cell = (Int32(rand(rng, 1:7)), Int32(rand(rng, 1:6)))
            δ = (T(rand(rng, B)), T(rand(rng, B)))
            x = ((cell[1] - 1 + δ[1]) * h[1], (cell[2] - 1 + δ[2]) * h[2])
            E, Bv = fields_at(f, cell, δ, shape, Int32(g))
            for (v, c) in zip((E..., Bv...), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
                a, b, cy = coef[c]
                exact = a + b * x[1] + cy * x[2]
                scale = abs(a) + abs(b) * 7 * h[1] + abs(cy) * 6 * h[2]
                worst = max(worst, absdiff(v, exact) / (ε * scale))
                st = L2D[c]
                if st[1] != st[2]
                    vt = ParticleMeshCore.gather_at(getfield(f, c), cell, δ, shape, reverse(st), Int32(g))
                    worst_transposed = min(worst_transposed, absdiff(vt, exact) / (ε * scale))
                end
            end
        end
        record!("T-351", T, worst)
        @test worst <= C
        @test worst_transposed > 100C        # the transposed stagger is half a cell off on both axes
    end
end

# --- T-352 ----------------------------------------------------------------------------------

# The phase error of test particles in uniform Bz at t_end, from initialize! with Δt, against the
# exact relativistic gyration φ(t) = φ₀ + (B/γ)·t (q/m = −1) at the half step t = (k − 1/2)Δt of
# u^{k−1/2}. `undo = true` reverses the back kick of initialize! with a forward half kick first.
function centring_error(::Type{T}, N, dt; undo = false, tend = 4) where {T}
    B0, u0 = one(T), one(T)
    nsteps = round(Int, tend / dt)
    Bf = (zero(T), zero(T), B0)
    prob = N == 1 ? test_particles_1d(T; cells = 8, length = T(4), dt, nsteps, u0 = (u0, zero(T), zero(T)), B0 = Bf) :
                    test_particles_2d(T; cells = (8, 8), length = (T(4), T(4)), dt, nsteps, u0 = (u0, zero(T), zero(T)), B0 = Bf)
    num = N == 1 ? EMNumerics() : EMNumerics(; current = NoCurrent())
    st = EMState(prob, num)
    initialize!(st, prob)
    if undo
        s = first(st.species)
        kernel = N == 1 ? kick_kernel! : kick_2d_kernel!
        kernel(CPU())(s, st.fields, num.pusher, num.shape, Int32(guards(st.grid)), s.q / s.m * prob.dt / 4; ndrange = capacity(s))
    end
    run!(st, params(prob), nsteps)
    γ = sqrt(1 + big(Float64(value_of(u0)))^2)
    t = (nsteps - big(1) / 2) * big(Float64(value_of(prob.dt)))
    return abs(wrap(uphase(first_u(st)) - Float64(t / γ)))
end

@testset "T-352 half-step centring: second-order gyration phase from initialize! [A2, WP-024]" begin
    for T in TYPES, N in (1, 2)
        e1, e2 = centring_error(T, N, 1 / 10), centring_error(T, N, 1 / 20)
        u1, u2 = centring_error(T, N, 1 / 10; undo = true), centring_error(T, N, 1 / 20; undo = true)
        order, order_undone = log2(e1 / e2), log2(u1 / u2)
        @info "T-352 convergence order of the gyration phase" T N order order_undone
        get!(MEASURED, "T-352 order", Dict{String,Float64}())["$(type_label(T)) N=$N"] = order
        get!(MEASURED, "T-352 order, kick undone", Dict{String,Float64}())["$(type_label(T)) N=$N"] = order_undone
        @test order > 1.8
        @test order_undone < 1.2
    end
end

# --- T-353 ----------------------------------------------------------------------------------

@testset "T-353 the initial field at N = 2: Gauss's law from the solve, and E0, B0 added exactly [A2, A7, WP-024]" begin
    C = emC("T-353")
    for T in TYPES, mode in ((2, 1), (1, -3))
        prob = plasma_2d(T; mode, cells = (12, 9), amplitude = seeded(T, 1 / 10))
        st = state_2d(prob)
        grid = st.grid
        g, n, h = guards(grid), ncells(grid), cellsize(grid)
        ρ, φ, Ex, Ey = Array(st.ρ), Array(st.aux.φ), Array(st.fields.Ex), Array(st.fields.Ey)
        I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
        ρ̄ = sum(ρ[I1, I2]) / prod(n)
        res = maximum(absmax((Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2] - (ρ[i, j] - ρ̄)) for i in I1, j in I2)
        Lnorm = 4 / h[1]^2 + 4 / h[2]^2
        scale = sum(n) * (maximum(absmax, ρ[I1, I2]) + Lnorm * maximum(absmax, φ[I1, I2]))
        r = record!("T-353", T, res / (base_eps(T) * value_of(scale)))
        @test r <= C
        # the background fields: the same problem with E0, B0 gives the fields plus E0, B0, bit for bit
        E0 = (T(1) / 3, -T(1) / 7, T(2) / 9)
        B0 = (-T(1) / 5, T(3) / 11, T(1) / 2)
        st0 = state_2d(prob)
        st1 = state_2d(plasma_2d(T; mode, cells = (12, 9), amplitude = seeded(T, 1 / 10), E0, B0))
        for (c, v) in zip((:Ex, :Ey, :Ez, :Bx, :By, :Bz), (E0..., B0...))
            @test same_bits(Array(getfield(st1.fields, c)), Array(getfield(st0.fields, c)) .+ v)
        end
    end
end

# --- T-354 ----------------------------------------------------------------------------------

# Write A·cos(k·x + ψ) at the positions of `st` (an NTuple{2} stagger) into every entry of `a`.
function set_mode!(a, grid, st, m, L, A, ψ)
    g, h = guards(grid), cellsize(grid)
    B = typeof(value_of(h[1]))
    T = eltype(a)
    for i2 in axes(a, 2), i1 in axes(a, 1)
        x = ((i1 - g - 1) + stagger_offset(st[1], B)) * h[1]
        y = ((i2 - g - 1) + stagger_offset(st[2], B)) * h[2]
        a[i1, i2] = A * cos(2 * T(π) * (m[1] * x / L[1] + m[2] * y / L[2]) + ψ)
    end
    return a
end

@testset "T-354 the 2D diagnostics against closed forms [A7, WP-024]" begin
    C, CW = emC("T-354"), emC("T-354", "C_energy")
    for T in TYPES
        ε = base_eps(T)
        m = (2, -1)
        L = (T(3), T(2))
        prob = test_particles_2d(T; cells = (12, 10), length = L, dt = 1 // 10, nsteps = 2, kmode = m)
        st = EMState(prob, EMNumerics(; current = NoCurrent()))
        initialize!(st, prob)
        amps = (Ex = (T(3) / 4, T(1) / 3), Ey = (-T(1) / 2, T(2)), Ez = (T(1) / 5, zero(T)),
                Bx = (T(2) / 3, -T(1)), By = (T(1) / 7, T(1) / 2), Bz = (T(5) / 4, -T(2) / 3))
        for (c, (A, ψ)) in pairs(amps)
            set_mode!(getfield(st.fields, c), st.grid, L2D[c], m, L, A, ψ)
        end
        observe!(st.recorder, st, 1)
        h = histories(st)
        area = value_of(L[1] * L[2])
        worstW = worstA = 0.0
        for (c, w) in ((:Ex, :WEx), (:Ey, :WEy), (:Ez, :WEz), (:Bx, :WBx), (:By, :WBy), (:Bz, :WBz))
            A = value_of(amps[c][1])
            worstW = max(worstW, abs(value_of(getfield(h, w)[1]) - A^2 * area / 4) / (ε * A^2 * area))
        end
        for c in (:Ex, :Ey, :Bz)
            A, ψ = value_of.(amps[c])
            re, im = value_of(getfield(h, Symbol(c, :_re))[1]), value_of(getfield(h, Symbol(c, :_im))[1])
            worstA = max(worstA, abs(re - A * cos(ψ)) / (ε * abs(A)), abs(im - A * sin(ψ)) / (ε * abs(A)))
        end
        k = (2π * m[1] / value_of(L[1]), 2π * m[2] / value_of(L[2]))
        k̂ = k ./ hypot(k...)
        (Ax, ψx), (Ay, ψy) = value_of.(amps.Ex), value_of.(amps.Ey)
        Elre, Elim = k̂[1] * Ax * cos(ψx) + k̂[2] * Ay * cos(ψy), k̂[1] * Ax * sin(ψx) + k̂[2] * Ay * sin(ψy)
        worstA = max(worstA, abs(value_of(h.El_re[1]) - Elre) / ε, abs(value_of(h.El_im[1]) - Elim) / ε)
        record!("T-354 energy", T, worstW)
        record!("T-354 amplitude", T, worstA)
        @test worstW <= CW
        @test worstA <= C

        # a y-invariant 2D state gives the 1D amplitudes and the 1D energies times L_y
        n1, Lx, Ly = 12, T(3), T(2)
        p1 = test_particles_1d(T; cells = n1, length = Lx, dt = 1 // 10, nsteps = 1, kmode = 2 * T(π) * 3 / Lx)
        p2 = test_particles_2d(T; cells = (n1, 6), length = (Lx, Ly), dt = 1 // 10, nsteps = 1, kmode = (3, 0))
        s1, s2 = EMState(p1, EMNumerics()), EMState(p2, EMNumerics(; current = NoCurrent()))
        initialize!(s1, p1)
        initialize!(s2, p2)
        g1, h1 = guards(s1.grid), cellsize(s1.grid)
        for (c, (A, ψ)) in pairs(amps)
            set_mode!(getfield(s2.fields, c), s2.grid, L2D[c], (3, 0), (Lx, Ly), A, ψ)
            a = getfield(s1.fields, c)
            o = stagger_offset(FIELD_LAYOUT[c], basetype(T))
            for i in eachindex(a)
                a[i] = A * cos(2 * T(π) * 3 * ((i - g1 - 1) + o) * h1 / Lx + ψ)
            end
        end
        step!(s1, params(p1), 1)
        observe!(s2.recorder, s2, 1)
        q1, q2 = histories(s1), histories(s2)
        worst = 0.0
        for key in (:Ex_re, :Ex_im, :Ey_re, :Ey_im, :Bz_re, :Bz_im)
            worst = max(worst, absdiff(getfield(q2, key)[1], getfield(q1, key)[1]) / (ε * 2))
        end
        for key in (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz)
            W1 = getfield(q1, key)[1] * Ly
            worst = max(worst, absdiff(getfield(q2, key)[1], W1) / (ε * absmax(W1)))
        end
        record!("T-354 y-invariant", T, worst)
        @test worst <= C
    end
end

# --- T-355 ----------------------------------------------------------------------------------

@testset "T-355 the 2D diagnostics and the kinetic-energy reduction are bitwise and equal a serial sum [A4, A7, WP-024]" begin
    ref = em2d_digests()
    @test ref == em2d_digests()
    for wgs in (32, 256)
        @test em2d_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        lines = filter(startswith("EM2D "), split(worker_output(nt), '\n'))
        got = Dict(String(f[3]) => parse(UInt64, f[4]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        for (T, d) in ref
            @test get(got, type_label(T), nothing) == d
        end
    end
    # the reductions against serial sums on the host, in both deposition modes
    C = emC("T-355")
    for T in TYPES, mode in (Reference(), Fast())
        prob = plasma_2d(T)
        st = state_2d(prob; mode)
        s = first(st.species)
        dA = prod(cellsize(st.grid))
        alive = findall(==(ParticleMeshCore.ALIVE), Array(s.status))
        kin = sum(i -> KineticEnergy()(s, i), alive) * dA
        g, n = guards(st.grid), ncells(st.grid)
        Ex = Array(st.fields.Ex)[(g + 1):(g + n[1]), (g + 1):(g + n[2])]
        WEx = sum(abs2, Ex) * dA / 2
        observe!(st.recorder, st, 1)
        h = histories(st)
        rK = absdiff(h.K[1], kin) / (base_eps(T) * length(alive) * absmax(kin))
        rW = absdiff(h.WEx[1], WEx) / (base_eps(T) * prod(n) * absmax(WEx))
        record!("T-355 serial", T, max(rK, rW))
        @test rK <= C
        @test rW <= C
    end
end

# --- T-356 ----------------------------------------------------------------------------------

@testset "T-356 Dual through initialize! and one step at N = 2 [A3, A8, WP-024]" begin
    C, Cp = emC("T-356"), emC("T-356", "C_partials")
    n0 = mkdual(1, 1, 0, 0)                      # ∂/∂n₀ in the first partial
    pd = plasma_2d(D3; density = n0, nsteps = 1)
    pf = plasma_2d(Float32; nsteps = 1)
    sd, sf = state_2d(pd), state_2d(pf)
    # after initialize!: E is linear in the density, ∂E/∂n₀ = E/n₀ = E (the solve's roundoff of
    # the partials and the values differs, scaled by Σ n_d as T-314)
    worstp = 0.0
    nsum = sum(ncells(sd.grid))
    for c in (:Ex, :Ey)
        a = Array(getfield(sd.fields, c))
        scale = maximum(x -> abs(value_of(x)), a)
        worstp = max(worstp, maximum(x -> abs(partials(x)[1] - value_of(x)), a) / (eps(Float32) * nsum * scale))
    end
    record!("T-356 partials", D3, worstp)
    @test worstp <= Cp
    step!(sd, params(pd), 1)
    step!(sf, params(pf), 1)
    worst = 0.0
    bitwise = true
    Escale = max(maximum(abs, sf.fields.Ex), maximum(abs, sf.fields.Ey))   # Bz is roundoff: the curl of a gradient
    for c in (:Ex, :Ey, :Bz)
        a, b = Array(getfield(sd.fields, c)), Array(getfield(sf.fields, c))
        scale = Escale
        worst = max(worst, maximum(abs, value_of.(a) .- b) / (eps(Float32) * scale))
        bitwise &= value_of.(a) == b
    end
    s, t = first(sd.species), first(sf.species)
    for d in 1:3
        worst = max(worst, maximum(abs, value_of.(s.u[d]) .- t.u[d]) / (eps(Float32) * maximum(abs, t.u[d])))
        bitwise &= value_of.(s.u[d]) == t.u[d]
        @test all(x -> all(isfinite, partials(x)), s.u[d])
        d <= 2 && @test any(x -> !iszero(partials(x)[1]), s.u[d])     # u_z sees no in-plane field
    end
    @info "T-356 the Dual primal against the Float32 run" worst bitwise
    record!("T-356 primal", D3, worst)
    @test worst <= C
end

# --- T-357 ----------------------------------------------------------------------------------

function step_bytes_2d(mode; ppc, cells)
    prob = plasma_2d(Float64; ppc, cells, nsteps = 4)
    st = state_2d(prob; mode)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    return @allocated step!(st, p, 3)
end

@testset "T-357 allocation per 2D step does not grow with particle count or grid size [A7, WP-024]" begin
    for mode in (Reference(), Fast())
        small = step_bytes_2d(mode; ppc = (2, 2), cells = (16, 12))
        more_particles = step_bytes_2d(mode; ppc = (8, 8), cells = (16, 12))
        larger_grid = step_bytes_2d(mode; ppc = (2, 2), cells = (64, 48))
        @info "T-357 bytes per 2D step" mode small more_particles larger_grid threads = Threads.nthreads()
        @test_alloc more_particles <= small
        @test_alloc larger_grid <= small
    end
end

# --- T-358 ----------------------------------------------------------------------------------

@testset "T-358 input checks of the 2D problem and state [A7, A10, WP-024]" begin
    T = Float64
    prob = plasma_2d(T)
    @test_throws ArgumentError EMState(prob, EMNumerics(; current = nothing))                # no current choice at N = 2
    p1 = test_particles_1d(T; dt = 1 // 10, nsteps = 2)
    @test_throws ArgumentError EMState(p1, EMNumerics(; current = NoCurrent()))              # NoCurrent() at N = 1
    sd1 = SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), 1, 4, zero3(T), zero3(T), 1)
    sd2 = SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), (1, 0), (2, 2), zero3(T), zero3(T), 1)
    @test_throws ArgumentError EMProblem("x", (T(1), T(1)), (8, 8), T(1) / 20, 2, (sd1,), 1, (1, 0))   # a 1D species in 2D
    @test_throws ArgumentError EMProblem("x", T(1), 8, T(1) / 20, 2, (sd2,), 1, T(2π))               # a 2D species in 1D
    @test_throws ArgumentError EMProblem("x", (T(1), T(1)), (8, 8), T(1) / 20, 2, (sd2,), 1, (0, 0))   # the zero mode
    @test_throws ArgumentError EMProblem("x", (T(1), T(1)), (8, 8), T(1) / 20, 2, (sd2,), 1, (1.0, 0.0)) # not integers
    st = state_2d(prob)
    @test_throws ArgumentError step!(st, params(prob), prob.nsteps + 1)
    @test_throws ArgumentError HistoryRecorder(prob, st; kmode = (0, 0))                     # the zero mode
    step!(st, params(prob), 1)
end
