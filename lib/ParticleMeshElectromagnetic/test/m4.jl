# T-1080 to T-1084: the M4 solver (WP-079) held to the exact discrete theory of its own scheme,
# at N = 1, 2 and 3 on the grids of `nd_grid` (cell sizes that differ per axis), as yee_nd.jl holds
# Yee to its own: the dispersion relation of every mode, the fourth order of its phase error, the
# divergence it keeps, its Courant limit and the growth beyond it, and its stencil. The theory is
# m4_cases.jl's, written from the formulas of Lu et al. (2020), not from the kernels. After
# yee_nd.jl (its helpers).

@testset "T-1080 discrete dispersion: an oblique eigenmode in each polarization follows the exact M4 relation [A7, A8, WP-079]" begin
    C = emC("T-1080")
    nsteps = 200
    for N in (1, 2, 3), T in TYPES, pol in (1, 2), cfl in (1 // 2, 95 // 100)
        B = basetype(T)
        grid = nd_grid(Val(N), B)
        dt = B(cfl) * courant_limit(M4(), grid)
        m = (3, -2, 1)[1:N]
        mode = m4_mode(grid, m, pol, dt)
        layout = field_layout(Val(N))
        A = seeded(T, 1)
        f = mode_fields(T, grid, mode, A)
        bdt = big(Float64(dt))
        worst = 0.0
        for s in 1:nsteps
            solve_fields!(M4(), f, grid, dt)
            s % 50 == 0 || continue
            err = 0.0
            for k in EB_KEYS
                v = interior_view(getfield(f, k), grid)
                for J in CartesianIndices(v)
                    err = max(err, part_err(v[J], A, mode_value(mode, layout, k, J, grid.n, s * bdt)))
                end
            end
            worst = max(worst, err / (base_eps(T) * s))
        end
        record!("T-1080", T, worst)
        @test worst <= C
    end
    # The test sees the scheme: the same run against Yee's relation is off by far more than roundoff.
    grid = nd_grid(Val(2), Float64)
    dt = 0.95 * courant_limit(M4(), grid)
    my, mm = yee_mode(grid, (3, -2), 2, dt), m4_mode(grid, (3, -2), 2, dt)
    @test abs(my.ω - mm.ω) * nsteps * dt > 1e6 * eps()
    f = mode_fields(Float64, grid, mm, 1.0)
    for _ in 1:nsteps
        solve_fields!(M4(), f, grid, dt)
    end
    v = interior_view(f.Ez, grid)
    layout = field_layout(Val(2))
    @test maximum(J -> abs(v[J] - Float64(mode_value(my, layout, :Ez, J, grid.n, nsteps * big(dt)))), CartesianIndices(v)) > 1e6 * eps()
end

# The phase-velocity error 1 − ω/|k| of a run of `solver` on its own eigenmode (polarization 2),
# measured from the phase of the Fourier coefficient of Ez over one period, Float64.
function solver_phase_error(solver, N, scale, m; cfl = 95 // 100)
    grid = N == 1 ? PeriodicGrid(16scale, 2 * π / (16scale), 2) :
           PeriodicGrid((16scale, 12scale), (2 * π / (16scale), 2 * π / (12scale)), 2)
    dt = Float64(cfl) * courant_limit(solver, grid)
    mode = solver isa M4 ? m4_mode(grid, m, 2, dt) : yee_mode(grid, m, 2, dt)
    f = mode_fields(Float64, grid, mode, 1.0)
    phases = [cispi(-2 * sum((J[d] - 1) * m[d] / grid.n[d] for d in 1:N)) for J in CartesianIndices(grid.n)]
    proj() = sum(interior_view(f.Ez, grid) .* phases)
    nsteps = round(Int, 2π / dt)
    c0 = proj()
    φ = 0.0
    for _ in 1:nsteps
        solve_fields!(solver, f, grid, dt)
        c1 = proj()
        φ += angle(c0 / c1)
        c0 = c1
    end
    ω = φ / (nsteps * dt)
    return (; err = 1 - ω / sqrt(sum(abs2, m)), ω, ω_discrete = Float64(mode.ω))
end

@testset "T-1081 convergence to Maxwell: M4's phase-velocity error falls as Δ⁴ along the axis and the diagonal, Yee's as Δ² [A7, WP-079]" begin
    tol = emC("T-1081", "order_tol")
    worst = 0.0
    for (N, m, label) in ((1, (1,), "N = 1"), (2, (1, 0), "N = 2 axis"), (2, (1, 1), "N = 2 diagonal"))
        r = [solver_phase_error(M4(), N, s, m) for s in (1, 2, 4)]
        errs = [x.err for x in r]
        orders = [log2(abs(errs[i] / errs[i + 1])) for i in 1:2]
        y = [solver_phase_error(Yee(), N, s, m).err for s in (1, 2, 4)]
        @info "T-1081 phase-velocity error 1 − ω/|k|" label M4 = errs orders Yee = y ω_rel_to_discrete = [x.ω / x.ω_discrete - 1 for x in r]
        @test all(o -> abs(o - 4) <= tol, orders)
        @test all(i -> abs(errs[i]) < abs(y[i]) / 10, 1:3)          # and far below Yee's at every resolution
        @test all(x -> abs(x.ω / x.ω_discrete - 1) < 1e-10, r)       # the run is the relation's ω
        worst = max(worst, maximum(o -> abs(o - 4), orders))
    end
    MEASURED["T-1081 |order − 4|"] = Dict("Float64" => worst)
end

@testset "T-1082 divergence identities: M4's own ∇·B is kept, Yee's is not, and ∇·E changes by −Δt·∇·J [A7, A8, WP-079]" begin
    C, Cj = emC("T-1082"), emC("T-1082", "C_current")
    for N in (1, 2, 3), T in TYPES
        grid = nd_grid(Val(N), basetype(T))
        Δ = bigΔ(grid)
        L = sum(d -> 2 / d, Δ)
        dt = basetype(T)(95 // 100) * courant_limit(M4(), grid)
        δ, β = m4_coefficients(Δ, dt)
        f = random_fields(T, grid; seed = 10820 + N)
        b0 = [bigfields(f, grid, k) for k in 0:nparts(T)]
        div0 = [m4_div(Bvec(u), Δ, δ, β) for u in b0]
        dy0 = [div_plus(Bvec(u), Δ) for u in b0]
        worst, yee = 0.0, 0.0
        for s in 1:100
            solve_fields!(M4(), f, grid, dt)
            s % 25 == 0 || continue
            for k in 0:nparts(T)
                u = bigfields(f, grid, k)
                M = max(maxabs_big(Bvec(u)), maxabs_big(Bvec(b0[k + 1])))
                worst = max(worst, Float64(maximum(abs, m4_div(Bvec(u), Δ, δ, β) .- div0[k + 1]) / (M * L)) / (base_eps(T) * s))
                yee = max(yee, Float64(maximum(abs, div_plus(Bvec(u), Δ) .- dy0[k + 1]) / (M * L)) / (base_eps(T) * s))
            end
        end
        record!("T-1082 M4 div B", T, worst)
        @test worst <= C
        # Yee's ∇·B is not M4's invariant at N ≥ 2 (at N = 1 only Bx enters it, and Bx is constant)
        N >= 2 && @test yee > 1000 * C
        f = random_fields(T, grid; seed = 10825 + N)
        worstj = 0.0
        for s in 1:10
            before = [bigfields(f, grid, k) for k in 0:nparts(T)]
            solve_fields!(M4(), f, grid, dt)
            for k in 0:nparts(T)
                u0, u1 = before[k + 1], bigfields(f, grid, k)
                bdt = big(Float64(dt))
                res = div_minus(Evec(u1), Δ) .- div_minus(Evec(u0), Δ) .+ bdt .* div_minus(Jvec(u0), Δ)
                M = max(maxabs_big(Evec(u0)), maxabs_big(Evec(u1)), maxabs_big(Bvec(u0)), maxabs_big(Bvec(u1)), bdt * maxabs_big(Jvec(u0)))
                worstj = max(worstj, Float64(maximum(abs, res) / (M * L)) / base_eps(T))
            end
        end
        record!("T-1082 div E with J", T, worstj)
        @test worstj <= Cj
    end
end

"The plain field norm Σ(E² + B²) of the interiors of `f`, in Float64."
field_norm2(f, grid) = sum(k -> sum(x -> Float64(value_of(x))^2, interior_view(getfield(f, k), grid)), EB_KEYS)

@testset "T-1083 M4's Courant limit is Yee's: bounded at 0.99, growth (μ + √(μ² − 1))² with μ² = S²(4 − S²)/3 at 1.01 [A7, WP-079]" begin
    Cb, Cg = emC("T-1083"), emC("T-1083", "C_growth")
    for N in (1, 2, 3)
        grid = nd_grid(Val(N), Float64)
        @test courant_limit(M4(), grid) === courant_limit(Yee(), grid)
        # the theory's maximum of the relation's right side over a fine k grid at S = 1 is 1, at the corner
        Δ = Float64.(grid.dx)
        dt = courant_limit(Yee(), grid)
        λ = [dt^2 / d^2 for d in Δ]
        xs = range(0, 1; length = 41)
        f(x) = (P = sum(λ .* x); P - P^2 / 3 + sum(λ .* x .^ 2) / 3)
        vals = [f(collect(x)) for x in Iterators.product(ntuple(_ -> xs, N)...)]
        @test maximum(vals) ≈ 1 atol = 1e-12
        @test CartesianIndices(vals)[argmax(vals)] == last(CartesianIndices(vals))
        # 0.99: random fields stay bounded over 4000 steps, with no growth between the halves
        dt = 0.99 * courant_limit(M4(), grid)
        fl = random_fields(Float64, grid; seed = 10830 + N, current = false)
        n0 = field_norm2(fl, grid)
        peak, mid = 0.0, 0.0
        for s in 1:4000
            solve_fields!(M4(), fl, grid, dt)
            r = field_norm2(fl, grid) / n0
            peak = max(peak, r)
            s == 2000 && (mid = peak)
        end
        record!("T-1083 bounded", Float64, peak)
        @test peak <= Cb
        @test peak <= 1.05 * mid
        # 1.01: the fastest mode is the corner, and the run grows by its eigenvalue per step
        S = 1.01
        dt = S * courant_limit(M4(), grid)
        μ = sqrt(S^2 * (4 - S^2) / 3)
        λ4 = (μ + sqrt(μ^2 - 1))^2
        fl = random_fields(Float64, grid; seed = 10835 + N, current = false)
        norms = Float64[]
        for s in 1:400
            solve_fields!(M4(), fl, grid, dt)
            s % 100 == 0 && push!(norms, field_norm2(fl, grid))
        end
        g = sqrt(norms[end] / norms[end - 1])^(1 / 100)
        rel = abs(g / λ4 - 1)
        @info "T-1083 growth per step at 1.01" N g λ4 Yee = (S + sqrt(S^2 - 1))^2
        record!("T-1083 growth", Float64, rel)
        @test rel <= Cg
        @test abs(g / (S + sqrt(S^2 - 1))^2 - 1) > 0.01          # not Yee's growth (5 % larger)
    end
end

@testset "T-1084 the M4 stencil: Lu et al.'s coefficients, Yee's bits where they vanish, a y-invariant run is the 1D run, two guards [A7, A8, A10, WP-079]" begin
    # the coefficients of the 2D grid of the low-dispersion study: Smilei's α_x = 1 − 2β_xy − 3δ_x
    dx, dy = 2π / 32, π / 4 / 2
    grid = PeriodicGrid((64, 16), (dx, dy), 3)
    dt = 0.95 * courant_limit(M4(), grid)
    st = faraday_stencil(M4(), grid, dt)
    @test st.δ[1] ≈ ((dt / dx)^2 - 1) / 12 rtol = 4eps()
    @test st.δ[2] ≈ ((dt / dy)^2 - 1) / 12 rtol = 4eps()
    @test st.β[1][2] ≈ (dt / dy)^2 / 12 rtol = 4eps()
    @test st.β[2][1] ≈ (dt / dx)^2 / 12 rtol = 4eps()
    @test st.β[1][1] === 0.0 && st.β[2][2] === 0.0
    αx = 1 - 2st.β[1][2] - 3st.δ[1]
    @test αx > 1 && αx < 1.1                                    # δ_x < 0 below Δt = Δx
    @test faraday_stencil(Yee(), grid, dt) === YeeStencil()
    for T in TYPES
        B = basetype(T)
        # N = 1 at Δt = Δx: δ = 0 exactly and M4's B update is Yee's bit for bit
        g1 = PeriodicGrid(40, B(1 // 4), 2)
        s1 = faraday_stencil(M4(), g1, T(1 // 4))
        @test iszero(s1.δ[1])
        f1 = random_fields(T, g1; seed = 10840)
        f2 = map(copy, f1)
        yee_b_half!(f1, g1, T(1 // 8))
        m4_b_half!(f2, g1, T(1 // 8), s1)
        @test all(same_bits(getfield(f1, k), getfield(f2, k)) for k in keys(f1))
        # a 2D run that does not vary along y is the 1D run, bit for bit, in every component
        n1, h1 = 24, B(1 // 3)
        ga, gb = PeriodicGrid(n1, h1, 2), PeriodicGrid((n1, 6), (h1, B(1 // 2)), 2)
        dt2 = B(9 // 10) * courant_limit(M4(), gb)
        fa = random_fields(T, ga; seed = 10841, current = false)
        fb = em_fields(CPU(), T, gb)
        for k in EB_KEYS
            a = interior_view(getfield(fa, k), ga)
            interior_view(getfield(fb, k), gb) .= reshape(a, :, 1)
            fill_guards!(getfield(fb, k), gb)
        end
        # Bx and Ex: the 1D layout's Bx (constant) and Ex (current only) are the y-invariant 2D ones
        for _ in 1:20
            solve_fields!(M4(), fa, ga, dt2)
            solve_fields!(M4(), fb, gb, dt2)
        end
        @test all(k -> all(j -> interior_view(getfield(fb, k), gb)[:, j] == interior_view(getfield(fa, k), ga), 1:6), EB_KEYS)
    end
    # two guards: the state's grid has them for every shape, and the half step refuses fewer
    @test em_guard_width(ParticleMeshCore.Linear, M4()) >= 2
    @test em_guard_width(Quadratic, M4()) == em_guard_width(Quadratic)
    g1 = PeriodicGrid(12, 0.5, 1)
    @test_throws ArgumentError m4_b_half!(em_fields(CPU(), Float64, g1), g1, 0.1, faraday_stencil(M4(), g1, 0.2))
    @test_throws ArgumentError check_courant(M4(), grid, courant_limit(Yee(), grid))
    @test check_courant(M4(), grid, dt) == dt
end
