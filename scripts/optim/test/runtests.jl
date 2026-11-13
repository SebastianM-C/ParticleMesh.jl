# Tests of the optimization layer (WP-058) and the cases (WP-051, WP-052):
#   julia +1.13 --project=scripts scripts/optim/test/runtests.jl
using Test
using ForwardDiff: ForwardDiff
include(joinpath(@__DIR__, "..", "problem.jl"))

@testset "T-760 L-BFGS-B through Optimization.jl finds interior and bound maxima of analytic objectives, one fg call per point [WP-058]" begin
    # An anisotropic concave quadratic with its maximum inside the box. Every point the solver
    # evaluates costs one fg call: the value and the gradient of a point come from the same call.
    A = [3.0 1.0; 1.0 0.5]
    c = [0.3, 0.7]
    calls = Ref(0)
    fq(z) = (calls[] += 1; r = z .- c; (-r' * A * r / 2, -A * r))
    r = maximize_box(fq, [0.9, 0.1], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-10)
    @test r.z ≈ c atol = 1e-8
    @test r.reason == "gtol" && maximum(abs, r.g) <= 1e-10
    @test r.evals == calls[] == length(r.path) == length(unique(first.(r.path)))
    @test r.path[1][1] == [0.9, 0.1]                                               # the start is the first point
    @test r.f == maximum(p[2] for p in r.path)
    # The maximum outside the box: on the bound z₁ = 0.5, where the constrained optimum of z₂ is
    # c₂ − A₂₁(0.5 − c₁)/A₂₂, with the gradient pointing out of the box.
    r = maximize_box(fq, [0.9, 0.1], [0.5, 0.0], [1.0, 1.0]; gtol = 1e-9, maxiters = 200)
    @test r.z[1] == 0.5
    @test r.z[2] ≈ c[2] - A[2, 1] * (0.5 - c[1]) / A[2, 2] atol = 1e-8
    @test r.g[1] < 0 && r.reason == "gtol"
    # Held at the upper bound z₁ = 0.2 instead.
    r = maximize_box(fq, [0.05, 0.1], [0.0, 0.0], [0.2, 1.0]; gtol = 1e-9, maxiters = 200)
    @test r.z[1] == 0.2
    @test r.z[2] ≈ c[2] - A[2, 1] * (0.2 - c[1]) / A[2, 2] atol = 1e-8
    @test r.g[1] > 0 && r.reason == "gtol"
    # A start outside the box is projected into it first, and no point leaves the box.
    r = maximize_box(fq, [3.0, -2.0], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-10, maxiters = 200)
    @test r.path[1][1] == [1.0, 0.0]
    @test all(p -> all(0 .<= p[1] .<= 1), r.path)
    @test r.z ≈ c atol = 1e-8
    # A smooth non-quadratic objective (a Gaussian bump on a slope), maximum inside. L-BFGS-B stops
    # on the objective's decrease (about 2e-9, fixed by the wrapper) before a gradient of 1e-9; the
    # gradient there is near 2e-7.
    fb(z) = (e = exp(-sum(abs2, z .- [0.4, 0.6]) / 0.1); (e + 0.01z[1], e .* (-2 .* (z .- [0.4, 0.6]) ./ 0.1) .+ [0.01, 0.0]))
    r = maximize_box(fb, [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-9, maxiters = 100)
    @test maximum(abs, r.g) < 1e-6 && r.reason == "ftol"
    r = maximize_box(fb, [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-6, maxiters = 100)
    @test maximum(abs, r.g) <= 1e-6 && r.reason == "gtol"
    # The tolerance reaches the solver: a loose one stops earlier, with a larger gradient.
    loose = maximize_box(fb, [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-1, maxiters = 100)
    @test loose.reason == "gtol" && maximum(abs, loose.g) > 1e-6 && loose.evals < r.evals
    # Out of iterations: the reason says so.
    r = maximize_box(fb, [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-12, maxiters = 1)
    @test r.reason == "maxiter"
end

@testset "T-761 a scaled box maps physical parameters both ways, and an objective's gradient in z is ForwardDiff's [WP-058]" begin
    b = ScaledBox([4.0, 60.0], [9.0, 130.0], [6.0, 90.0])
    @test zlo(b) == [4.0 / 6.0, 60.0 / 90.0] && zhi(b) == [9.0 / 6.0, 130.0 / 90.0]
    @test all(isapprox.(phys(b, scaled(b, (7.0, 100.0))), (7.0, 100.0); rtol = 1e-15))
    @test_throws ArgumentError ScaledBox([1.0], [0.0], [1.0])
    @test_throws ArgumentError ScaledBox([0.0], [1.0], [0.0])
    @test_throws DimensionMismatch ScaledBox([0.0, 0.0], [1.0, 1.0], [1.0])
    @test_throws DimensionMismatch dual_point(b, [1.0, 1.0, 1.0])
    # The value and the gradient in z from one run with duals, against ForwardDiff on z directly
    # and against the chain rule written out (∂J/∂z_i = ∂J/∂x_i · scale_i).
    Jx(τ, f) = τ^2 * sin(f / 30) + τ * f / 100
    z = scaled(b, (6.5, 95.0))
    v, g = value_gradient(Jx, b, z)
    @test v == Jx(phys(b, z)...)
    @test g ≈ ForwardDiff.gradient(w -> Jx(phys(b, w)...), z) rtol = 1e-14
    τ, f = phys(b, z)
    @test g ≈ [(2τ * sin(f / 30) + f / 100) * 6.0, (τ^2 * cos(f / 30) / 30 + τ / 100) * 90.0] rtol = 1e-13
    # The objective records every run with its kind; maximize uses one run with duals per point.
    o = Objective(Jx, b; verbose = false)
    @test o(z) == v && value_gradient(o, z) == (v, g)
    @test [e["kind"] for e in o.log] == ["value", "gradient"] && o.log[2]["grad"] == g
    @test all(e -> e["z"] == z && e["phys"] == collect(phys(b, z)) && e["J"] == v, o.log)
    q = Objective((τ, f) -> -(τ - 7.7)^2 - ((f - 113) / 10)^2, b; verbose = false)
    r = maximize(q, scaled(b, (4.5, 70.0)))
    @test all(isapprox.(phys(q, r.z), (7.7, 113.0); atol = 1e-4))
    @test length(q.log) == r.evals && all(e -> e["kind"] == "gradient", q.log)
end

include(joinpath(@__DIR__, "..", "wake2d_case.jl"))
include(joinpath(@__DIR__, "..", "slab.jl"))

@testset "T-660 the 1D cold-fluid wake reproduces its linear limit and the optimum k_pτ = 2 [WP-051]" begin
    kp = sqrt(CASE.density)
    for β in (1.0, sqrt(1 - CASE.density))
        lin(τ) = sqrt(2π) / 8 * 0.02^2 * kp * τ * exp(-kp^2 * τ^2 / 8) / β^2
        for τ in (4.0, 6.0, 9.0)
            @test wake1d(0.02, τ; β).J ≈ lin(τ) rtol = 2e-3
        end
        τopt, _ = golden_max(τ -> wake1d(0.02, τ; β).J, 3.0, 12.0; tol = 1e-4)
        @test τopt ≈ 2 / kp rtol = 2e-3
    end
    # Nonlinear: at a0 = 1 the optimum is longer.
    @test golden_max(τ -> wake1d(1.0, τ).J, 3.0, 12.0; tol = 1e-3)[1] > 2 / kp + 0.05
    # golden_max on a known parabola.
    @test golden_max(x -> -(x - 1.234)^2, 0.0, 3.0; tol = 1e-8)[1] ≈ 1.234 atol = 1e-7
end

@testset "T-661 the slab window's Hann weight averages E² of an oscillation to half its square at any phase [WP-051]" begin
    # The weight over 1.5 or 1 periods has a spectral zero at 2ω, the ripple of E²; sampled at the
    # case's Δt the mean of sin² is 1/2 to the quadrature error, at every phase. A boxcar is not.
    ωp, dt = sqrt(CASE.density), 0.95 / sqrt(1 / (2π / 16)^2 + 1 / (4 * 2π / 16)^2)
    Tp = 2π / ωp
    for L in (1.5Tp, Tp), φ in range(0, π; length = 7)
        s = 0:dt:L
        w = hann.(s ./ L)
        @test sum(w .* sin.(ωp .* s .+ φ) .^ 2) / sum(w) ≈ 0.5 atol = 2e-4
    end
    s = 0:dt:(1.3Tp)
    @test maximum(φ -> abs(sum(sin.(ωp .* s .+ φ) .^ 2) / length(s) - 0.5), range(0, π; length = 7)) > 0.02
    @test hann(0.0) == 0 && hann(1.0) == 0 && hann(0.5) ≈ 1 && hann(-0.1) == 0 && hann(1.1) == 0
end

@testset "T-662 the case's resolution keywords keep WP-045's defaults and the antenna's place [WP-051]" begin
    p = wake_problem(7.0, 110.0; nsteps = 1)
    h = 2π / 16
    @test p.cells == CASE.cells
    @test p.dt == 0.95 / sqrt(1 / h^2 + 1 / (4h)^2)
    @test p.lasers[1].position == 6 * (2π / 16)
    @test p.lasers[1].a0 == CASE.a0 && p.lasers[1].waist == CASE.waist
    q = wake_problem(7.0, 110.0; nsteps = 1, CONVERGED...)
    @test q.cells == (14 * 32, 16 * 32 ÷ 2)
    @test q.lasers[1].position == p.lasers[1].position                      # 3λ₀/8 at every ppl
    @test q.length[1] ≈ p.length[1] && q.length[2] ≈ p.length[2]            # the same 14 λ₀ × 16 λ₀ box
    @test wake_problem(7.0, 110.0; nsteps = 1, a0 = 0.3, waist = 1e5).lasers[1].a0 == 0.3
    # The antenna is a cell edge only for ppl divisible by 8; a Δy that does not divide the box
    # would shrink it silently. Both refuse.
    @test_throws ArgumentError wake_problem(7.0, 110.0; nsteps = 1, ppl = 20)
    @test_throws ArgumentError wake_problem(7.0, 110.0; nsteps = 1, dy_cells = 3)
    @test wake_problem(7.0, 110.0; nsteps = 1, dy_cells = 3, cells = (224, 85)).cells == (224, 85)
end

@testset "T-663 the slab sums of a synthetic ringing plasma in a moving window give its amplitude [WP-051]" begin
    # E_x = A·cos(ω_p(t − x) + φ) at every lab column, read through a window of 224 columns that
    # starts moving at t = 108 (one column per Δx of time), past a back band of 9 columns.
    A, φ, ωp = 0.37, 0.4, sqrt(CASE.density)
    dx = 2π / 16
    dt = 0.95 / sqrt(1 / dx^2 + 1 / (4dx)^2)
    n1, band = 224, 9
    win = (main = WINDOWS.main, early = WINDOWS.early, short = WINDOWS.short)
    sums = SlabSums{Float64}(length(win))
    E(x, t) = A * cos(ωp * (t - x) + φ)
    prev, shprev = nothing, 0
    for k in 1:ceil(Int, last_sample(win) / dt) + 1
        t = k * dt
        sh = max(0, floor(Int, (t - 108) / dx))
        ex = [E((band + jj + sh - 0.5) * dx, t) for jj in 1:(n1 - band)]
        prev === nothing || accumulate_slab!(sums, win, ex, prev, sh, shprev, band, t, dt, dx, ωp)
        prev, shprev = ex, sh
    end
    v = slab_values(sums, win, dt, dx, 1.0)
    for i in 1:3
        @test all(c -> abs(c - 1) < 0.01, v.coverage[i])
        @test v.slab[i] ≈ A rtol = 2e-3
        @test v.osc[i] ≈ A rtol = 2e-3
    end
    # A field that grows with the delay behind the light front: the early window reads less.
    sums2 = SlabSums{Float64}(2)
    G(x, t) = (1 + (t - x - 72) / 100) * E(x, t)
    prev = nothing
    for k in 1:ceil(Int, last_sample(win) / dt) + 1
        t = k * dt
        sh = max(0, floor(Int, (t - 108) / dx))
        ex = [G((band + jj + sh - 0.5) * dx, t) for jj in 1:(n1 - band)]
        prev === nothing || accumulate_slab!(sums2, (win.main, win.early), ex, prev, sh, shprev, band, t, dt, dx, ωp)
        prev, shprev = ex, sh
    end
    w2 = slab_values(sums2, (win.main, win.early), dt, dx, 1.0)
    @test w2.slab[2] < w2.slab[1]
    @test w2.slab[1] ≈ A * 1.15 rtol = 0.01                                 # the growth at the window's middle, s = 87
    # Exact read-outs: a field that is the lab position (each column must be labelled by its own x)
    # and one that is the delay s = t − x (each sample must sit at t − Δt/2): J² = 2·mean(x²) over
    # the slab's columns, and 2·(87² + L²(1/12 − 1/(2π²))), the mean square of s under a Hann weight
    # of length L = 30 centred on 87.
    for (f, expect) in (((x, t) -> x, sqrt(2 * sum(c -> ((c - 0.5) * dx)^2, ceil(Int, 72 / dx + 0.5):floor(Int, 152 / dx + 0.5)) /
                                             length(ceil(Int, 72 / dx + 0.5):floor(Int, 152 / dx + 0.5)))),
                        ((x, t) -> t - x, sqrt(2 * (87^2 + 30^2 * (1 / 12 - 1 / (2π^2))))))
        s3 = SlabSums{Float64}(1)
        prev = nothing
        for k in 1:ceil(Int, last_sample(win) / dt) + 1
            t = k * dt
            sh = max(0, floor(Int, (t - 108) / dx))
            ex = [f((band + jj + sh - 0.5) * dx, t) for jj in 1:(n1 - band)]
            prev === nothing || accumulate_slab!(s3, (win.main,), ex, prev, sh, shprev, band, t, dt, dx, ωp)
            prev, shprev = ex, sh
        end
        v3 = slab_values(s3, (win.main,), dt, dx, 1.0)
        @test v3.slab[1] ≈ expect rtol = 2e-5
        @test all(c -> abs(c - 1) < 1e-4, v3.coverage[1])
    end
end

@testset "T-664 the box read-out interpolates between the steps around its time [WP-051]" begin
    dt = 0.3
    w = box_weights((5dt, 7.25dt), dt)
    @test w[5] == [(1, 1.0)]
    @test only(w[7])[1] == 2 && only(w[7])[2] ≈ 0.75
    @test only(w[8])[1] == 2 && only(w[8])[2] ≈ 0.25
    @test !haskey(w, 6)
end

@testset "T-665 the dispersive 1D wake reduces to the linear limit where the pulse enters [WP-051]" begin
    # At z = 0 the pulse has not spread yet: the linear limit with the 1/β² factor, to the
    # spectral width's correction (+1.3 % at τ = 9, more at shorter pulses).
    kp, β = sqrt(CASE.density), sqrt(1 - CASE.density)
    lin(a0, τ) = sqrt(2π) / 8 * a0^2 * kp * τ * exp(-kp^2 * τ^2 / 8) / β^2
    @test wake1d_dispersive(0.02, 9.0; x1 = 68.0, x2 = 68.0, nz = 1) ≈ lin(0.02, 9.0) rtol = 0.03
    @test wake1d_dispersive(0.02, 9.0; x1 = 68.0, x2 = 68.0, nz = 1) > lin(0.02, 9.0)
    # Deeper in, the spread pulse drives less, and the optimum over the slab is longer than 2/ω_p.
    @test wake1d_dispersive(0.02, 6.0; x1 = 150.0, x2 = 150.0, nz = 1) < wake1d_dispersive(0.02, 6.0; x1 = 68.0, x2 = 68.0, nz = 1)
    @test 6.6 < golden_max(τ -> wake1d_dispersive(0.02, τ), 5.0, 9.0; tol = 1e-3)[1] < 7.0
    # The nonlinear wake1d at a0 = 1 on its default window (a stored value of this code, WP-051).
    @test wake1d(1.0, 7.0).J ≈ 0.35761 rtol = 1e-4
end
# Tests of the WP-052 fixed-energy case (witness2d_case.jl): two full runs of the case, about a minute.
include(joinpath(@__DIR__, "..", "witness2d_case.jl"))

@testset "T-762 the cases' boxes are the reports' boxes [WP-045, WP-052, WP-058]" begin
    # reports/wp-045-wake-optimization.md: τ ∈ [4, 9], x_f ∈ [60, 130], z = (τ/6, x_f/90).
    @test (WAKE_BOX.lo, WAKE_BOX.hi, WAKE_BOX.scale) == ([4.0, 60.0], [9.0, 130.0], [6.0, 90.0])
    # reports/wp-052-fixed-energy-optimization.md: τ ∈ [4, 9], n₀ ∈ [0.04, 0.10], x_c ∈ [30, 50], z = (τ/6, n₀/0.07, x_c/40).
    @test (WITNESS_BOX.lo, WITNESS_BOX.hi, WITNESS_BOX.scale) == ([4.0, 0.04, 30.0], [9.0, 0.10, 50.0], [6.0, 0.07, 40.0])
end

@testset "T-680 a₀ from the fixed energy inverts laser_energy, with no energy derivative along τ [A3, WP-052]" begin
    for τ in (4.0, 5.5, 9.0), E in (WCASE.energy, WCASE.energy / 2.25)
        prob = witness_problem(τ, 0.07, 40.0; energy = E)
        @test prob.lasers[1].a0 == a0_at_energy(τ; energy = E)
        @test laser_energy(prob.lasers[1]) ≈ E rtol = 4eps()
    end
    # through a dual in τ the energy's derivative is zero and a₀'s is −a₀/(2τ)
    d = ForwardDiff.Dual{Nothing}(6.0, 1.0)
    l = witness_problem(d, 0.07, 40.0).lasers[1]
    @test abs(ForwardDiff.partials(laser_energy(l))[1]) <= 8eps() * WCASE.energy
    @test ForwardDiff.partials(l.a0)[1] ≈ -ForwardDiff.value(l.a0) / 12 rtol = 4eps()
    @test a0_at_energy(6.0) ≈ 1.5 rtol = 4eps()
    # the keywords reach the problem
    p = witness_problem(5.0, 0.07, 40.0; inject = 50.0, nsteps = 300, focus = 70.0)
    @test p.witness.inject == 50.0 && p.nsteps == 300 && p.lasers[1].focus == 70.0
    # the fixed-a₀ comparison of witness2d.jl: the energy that holds a₀ gives that a₀
    for τ in (4.0, 7.5)
        @test a0_at_energy(τ; energy = WCASE.energy * (1.65 / a0_at_energy(τ))^2) ≈ 1.65 rtol = 4eps()
    end
end

@testset "T-681 the witness centre enters through the weights only: one run gives the objective at every centre [A3, WP-047, WP-052]" begin
    st1, p1 = witness_state(5.0, 0.07, 38.0)
    st2, p2 = witness_state(5.0, 0.07, 44.0)
    # lattice_x is the kernel's lattice: the weights it set are the Gaussian of these positions
    for (st, p, c) in ((st1, p1, 38.0), (st2, p2, 44.0))
        x0 = lattice_x(st)
        g = witness_gains(st, p)
        @test length(x0) == length(g.w)
        @test g.w ≈ exp.(-((x0 .- c) ./ WCASE.length) .^ 2 ./ 2) rtol = 1e-14
    end
    @test witness_gains(st1, p1).gain == witness_gains(st2, p2).gain            # test particles: the centre does not move them
    @test gain_at_centers(st1, p1, [38.0, 44.0]) ≈ [witness_energy_gain(st1, p1), witness_energy_gain(st2, p2)] rtol = 1e-13
    @test !(witness_energy_gain(st1, p1) ≈ witness_energy_gain(st2, p2))
    # a lost witness is refused, as witness_energy_gain refuses it (last: this changes st1)
    s = ParticleMeshElectromagnetic.witness_frame(st1).species
    status = Array(s.status)
    status[1] = ParticleMeshElectromagnetic.ParticleMeshCore.FREE
    copyto!(s.status, status)
    @test_throws ArgumentError gain_at_centers(st1, p1, [40.0])
end

@testset "T-682 the interior-maximum test of the run's slices refuses one-sided and flat slices [WP-052]" begin
    xs = [0.9, 0.95, 1.0, 1.05, 1.1]
    @test is_interior_max(xs, [1.0, 2.0, 3.0, 2.5, 1.0], 1.0, 3.0)
    @test !is_interior_max(xs, [1.0, 2.0, 3.0, 3.5, 1.0], 1.0, 3.0)                 # higher on one side
    @test !is_interior_max(xs, [1.0, 2.0, 3.0, 3.0, 1.0], 1.0, 3.0)                 # a tie
    @test !is_interior_max([1.0, 1.05, 1.1], [3.0, 2.0, 1.0], 1.0, 3.0)             # one side only (cut by a bound)
    @test !is_interior_max([0.9, 1.1], [1.0, 1.0], 1.0, 3.0)                        # the optimum not in the slice
    @test !is_interior_max(xs, [1.0, 2.0, 2.9, 2.5, 1.0], 1.0, 3.0)                 # the slice's centre is not the optimum's value
end
include(joinpath(@__DIR__, "bayes.jl"))
