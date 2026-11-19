# Tests of the optimization layer (WP-058) and the cases (WP-051, WP-052):
#   julia +1.13 --project=scripts scripts/optim/test/runtests.jl
using Test
using ForwardDiff: ForwardDiff
using OptimizationMadNLP: MadNLPOptimizer
using MadNLP: MadNLP
include(joinpath(@__DIR__, "..", "problem.jl"))

@testset "T-760 a case problem is maximized by L-BFGS-B in scaled coordinates, inside the box and on its bounds [WP-058]" begin
    # A concave quadratic in physical units with its maximum at (7.7, 113), inside WP-045's box.
    box = (lo = [4.0, 60.0], hi = [9.0, 130.0], scale = [6.0, 90.0])
    Jq(τ, f) = -(τ - 7.7)^2 - ((f - 113) / 10)^2 - (τ - 7.7) * (f - 113) / 20
    prob = case_problem(Jq, box, (4.5, 70.0))
    @test prob.u0 == [4.5 / 6.0, 70.0 / 90.0] && prob.lb == box.lo ./ box.scale && prob.ub == box.hi ./ box.scale
    @test prob.sense === MaxSense
    @test prob.f(prob.u0, prob.p) == Jq(4.5, 70.0)                       # the problem's objective is J, not −J
    rec = Record(prob, LBFGSB())
    sol = solve(prob, LBFGSB(); callback = rec, maxiters = 50, reltol = 1e-9)
    @test sol.retcode == ReturnCode.Success
    @test sol.u .* box.scale ≈ [7.7, 113.0] atol = 1e-6
    @test sol.objective ≈ 0 atol = 1e-12
    @test sol.objective == maximum(rec.objective) == Jq((sol.u .* box.scale)...)
    # The record holds every point L-BFGS-B evaluated, the start first, with J there (not the −J the
    # solver minimizes), and times that do not decrease.
    @test rec.u[1] == prob.u0 && rec.u[end] == sol.u
    @test rec.objective == [Jq((u .* box.scale)...) for u in rec.u]
    @test issorted(rec.seconds) && length(rec.seconds) == length(rec.u) == sol.stats.fevals
    @test path(rec, box.scale)[end] == Dict("z" => sol.u, "phys" => sol.u .* box.scale, "J" => sol.objective)
    # A point costs L-BFGS-B a plain run for the value and a run with duals for the gradient. The 1
    # is prob.f above.
    @test runs(prob.p.timer, "plain run").n == length(rec.u) + 1
    @test runs(prob.p.timer, "run with duals").n == length(rec.u)
    @test runs(TimerOutput(), "plain run") == (; n = 0, seconds = nothing)
    # `runs` gives the mean seconds of a section, and the record's seconds count from its making.
    slept = TimerOutput()
    total = @elapsed for _ in 1:4
        @timeit slept "plain run" sleep(0.05)
    end
    @test runs(slept, "plain run").n == 4 && 0.045 <= runs(slept, "plain run").seconds <= total / 4
    t0 = time()
    timed = Record(prob, LBFGSB())
    solve(prob, LBFGSB(); callback = timed, maxiters = 50, reltol = 1e-9)
    @test 0 <= timed.seconds[1] <= timed.seconds[end] <= time() - t0
    # A verbose record prints each call, a quiet one nothing.
    printed = mktemp() do file, io
        redirect_stdout(io) do
            Record(prob, LBFGSB(); verbose = true)((; u = [1.0, 2.0], objective = -3.0))
            Record(prob, LBFGSB())((; u = [7.0, 8.0], objective = -9.0))
        end
        close(io)
        read(file, String)
    end
    @test occursin("point 1: u = [1.0, 2.0]  objective = 3.0", printed) && !occursin("7.0", printed)
    # The tolerance reaches the solver: a loose one stops earlier.
    loose = Record(prob, LBFGSB())
    solve(prob, LBFGSB(); callback = loose, maxiters = 50, reltol = 1e-1)
    @test length(loose.u) < length(rec.u)
    @test solve(prob, LBFGSB(); maxiters = 1, reltol = 1e-12).retcode == ReturnCode.MaxIters
    # The maximum outside the box: the solution sits on the bound τ = 6 (then 9 from the other side
    # for a maximum beyond it), with the other parameter at its constrained optimum
    # f = 113 − 2.5(τ − 7.7) and the gradient pointing out of the box.
    for (b, τb, sgn) in (((lo = [4.0, 60.0], hi = [6.0, 130.0], scale = box.scale), 6.0, 1), ((lo = [8.5, 60.0], hi = [9.0, 130.0], scale = box.scale), 8.5, -1))
        pb = case_problem(Jq, b, (b.lo[1] + 0.1, 70.0))
        sb = solve(pb, LBFGSB(); maxiters = 100, reltol = 1e-9)
        @test sb.u[1] * 6.0 == τb
        @test sb.u[2] * 90.0 ≈ 113 - 2.5 * (τb - 7.7) atol = 1e-6
        @test sign(value_gradient(pb, sb.u)[2][1]) == sgn
    end
    # A start outside the box is projected into it first, and no point leaves the box.
    out = case_problem(Jq, box, (12.0, 20.0))
    rec = Record(out, LBFGSB())
    sol = solve(out, LBFGSB(); callback = rec, maxiters = 100, reltol = 1e-9)
    @test rec.u[1] == [9.0 / 6.0, 60.0 / 90.0]
    @test all(u -> all(out.lb .<= u .<= out.ub), rec.u)
    @test sol.u .* box.scale ≈ [7.7, 113.0] atol = 1e-6
end

@testset "T-761 the gradient of a case problem is the chain rule in z, from one run with duals [WP-058]" begin
    box = (lo = [4.0, 60.0], hi = [9.0, 130.0], scale = [6.0, 90.0])
    Jx(τ, f) = τ^2 * sin(f / 30) + τ * f / 100
    prob = case_problem(Jx, box, (6.5, 95.0))
    z = prob.u0
    v, g = value_gradient(prob, z)
    τ, f = z .* box.scale
    @test v == Jx(τ, f)
    @test g ≈ [(2τ * sin(f / 30) + f / 100) * 6.0, (τ^2 * cos(f / 30) / 30 + τ / 100) * 90.0] rtol = 1e-13
    @test g ≈ ForwardDiff.gradient(w -> Jx((w .* box.scale)...), z) rtol = 1e-14
    @test runs(prob.p.timer, "run with duals").n == 1 && runs(prob.p.timer, "plain run").n == 0
    @test run_kind(z) == "plain run" && run_kind(ForwardDiff.Dual.(z, 1.0)) == "run with duals"
    # The scale is the problem's parameter: the objective reads it from `p`.
    @test prob.f(z, (; scale = 2 .* box.scale, timer = TimerOutput())) == Jx(13.0, 190.0)
    # A timer passed in is the one the runs go to.
    timer = TimerOutput()
    shared = case_problem(Jx, box, (6.5, 95.0); timer)
    shared.f(z, shared.p)
    @test runs(timer, "plain run").n == 1
end

@testset "T-762 the record turns the −J that L-BFGS-B reports back into J, and MadNLP solves the same problem [WP-058]" begin
    box = (lo = [4.0, 60.0], hi = [9.0, 130.0], scale = [6.0, 90.0])
    Jq(τ, f) = 3 - (τ - 7.7)^2 - ((f - 113) / 10)^2
    prob = case_problem(Jq, box, (4.5, 70.0))
    # L-BFGS-B has no sense of its own: Optimization.jl gives it −J, and its callback too (#441).
    raw = Float64[]
    solve(prob, LBFGSB(); callback = (state, args...) -> (push!(raw, state.objective); false), maxiters = 50)
    @test !OptimizationBase.supports_sense(LBFGSB()) && Record(prob, LBFGSB()).negated
    @test raw[1] == -Jq(4.5, 70.0)
    # A record of a minimization is left as it is.
    pmin = OptimizationProblem(OptimizationFunction((z, p) -> sum(abs2, z .- 0.3), AutoForwardDiff()), [0.9, 0.1]; lb = [0.0, 0.0], ub = [1.0, 1.0])
    rmin = Record(pmin, LBFGSB())
    smin = solve(pmin, LBFGSB(); callback = rmin, maxiters = 50)
    @test !rmin.negated && rmin.objective[1] == sum(abs2, [0.9, 0.1] .- 0.3) && rmin.objective[end] == smin.objective
    # MadNLP with a limited-memory Hessian (an interior-point method) reaches the same maximum.
    sol = solve(prob, MadNLPOptimizer(; hessian_approximation = MadNLP.CompactLBFGS); maxiters = 100, verbose = false)
    @test sol.retcode == ReturnCode.Success
    @test sol.u .* box.scale ≈ [7.7, 113.0] atol = 1e-5
    @test sol.objective ≈ 3 atol = 1e-9
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

@testset "T-763 the cases' boxes are the reports' boxes [WP-045, WP-052, WP-058]" begin
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

# Tests of the WP-064 case (born2d_case.jl): the objective on ionization-born electrons. T-833 and
# T-834 run the case at λ₀/8 to t = 190, about a minute in all.
include(joinpath(@__DIR__, "..", "born2d_case.jl"))

"The case's state at λ₀/8 before its first step, with `n` hand-made born electrons written into the first slots."
function born_test_state(xs, us, ws)
    st, p, prob = born_start(; ppl = 8, t_end = 1.0)
    e = ionization_electrons(st)
    h = cellsize(st.grid)[1]
    for (i, (x, u, w)) in enumerate(zip(xs, us, ws))
        c = floor(Int, x / h)
        e.cell[1][i] = c + 1; e.offset[1][i] = x / h - c
        e.cell[2][i] = 60 + i; e.offset[2][i] = 0.5
        e.u[1][i], e.u[2][i], e.u[3][i] = u
        e.w[i] = w; e.status[i] = ALIVE
    end
    return st
end

@testset "T-830 the position weight is a C² step: 0 and 1 outside, antisymmetric about the middle, flat to second order at both ends [A2, WP-064]" begin
    lo, hi = 12.0, 24.0
    @test smooth_rise(lo, lo, hi) == 0 && smooth_rise(-3.0, lo, hi) == 0 && smooth_rise(hi, lo, hi) == 1 && smooth_rise(99.0, lo, hi) == 1
    @test smooth_rise(18.0, lo, hi) == 0.5
    xs = range(lo, hi; length = 97)
    @test all(x -> isapprox(smooth_rise(x, lo, hi) + smooth_rise(lo + hi - x, lo, hi), 1; atol = 4eps()), xs)
    @test issorted(smooth_rise.(xs, lo, hi))
    d1(x) = ForwardDiff.derivative(y -> smooth_rise(y, lo, hi), x)
    d2(x) = ForwardDiff.derivative(d1, x)
    @test d1(lo) == 0 && d1(hi) == 0 && d2(lo) == 0 && d2(hi) == 0
    # just inside the ends the second derivative is O(distance): 60·s/(hi − lo)² to first order
    @test d2(lo + 1e-6) ≈ 60 * (1e-6 / 12) / 144 rtol = 1e-4
    @test d2(hi - 1e-6) ≈ -60 * (1e-6 / 12) / 144 rtol = 1e-4
    @test d1(18.0) ≈ 15 / 8 / 12 rtol = 4eps()                       # the steepest slope, 15/8 per width
    @test smooth_rise(15.0f0, 12.0f0, 24.0f0) isa Float32
    @test smooth_rise(15.0f0, 12.0f0, 24.0f0) ≈ 0.25^3 * (10 - 15 * 0.25 + 6 * 0.25^2) rtol = 4eps(Float32)
end

@testset "T-831 the objective is Δx·Δy·Σ w(γ − 1)S(x) over the born electrons alone [A3, WP-064]" begin
    xs = [5.0, 12.0, 15.0, 18.0, 30.0, 60.0]
    us = [(3.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, -2.0, 0.5), (0.0, 0.0, 0.0), (10.0, 0.3, -0.2), (0.4, 0.0, 0.0)]
    ws = [1.0e-3, 2.0e-3, 5.0e-4, 7.0e-4, 3.0e-4, 1.1e-3]
    st = born_test_state(xs, us, ws)
    h = cellsize(st.grid)
    @test h[1] ≈ 2π / 8 && h[2] ≈ π / 4
    S(x) = (s = clamp((x - 12) / 12, 0, 1); s^3 * (10 - 15s + 6s^2))
    expected = h[1] * h[2] * sum(w * (sqrt(1 + sum(abs2, u)) - 1) * S(x) for (x, u, w) in zip(xs, us, ws))
    J = born_energy(st)
    @test J ≈ expected rtol = 8eps()
    @test J > 0
    # the electrons behind fade[1] (x = 5 and 12) count for nothing: without them the same bits
    e = ionization_electrons(st)
    e.status[1] = 0x00; e.status[2] = 0x00
    @test born_energy(st) === J
    e.status[1] = ALIVE; e.status[2] = ALIVE
    # a free slot's leftover numbers do not enter
    e.u[1][7] = 1.0e6; e.w[7] = 1.0; e.cell[1][7] = 40
    @test born_energy(st) === J
    # the plasma's electrons do not enter, whatever their energy (a label, not an energy cut)
    s = st.species[1]
    s.u[1] .= 50.0
    @test born_energy(st) === J
    # another weight: with fade = (0.9·x, x) of the electron at 30 only it and the one at 60 count, in full
    @test born_energy(st; fade = (27.0, 30.0)) ≈ h[1] * h[2] * (ws[5] * (sqrt(1 + sum(abs2, us[5])) - 1) + ws[6] * (sqrt(1 + 0.16) - 1)) rtol = 8eps()
    # a weight that reaches into the window's back band, or runs backwards, is refused
    band = window_back_band(st) * h[1]
    @test band ≈ 9 * 2π / 8
    @test_throws ArgumentError born_energy(st; fade = (band - 0.01, 24.0))
    @test_throws ArgumentError born_energy(st; fade = (24.0, 12.0))
    @test_throws ArgumentError born_energy(st; fade = (24.0, 24.0))
    @test born_energy(st; fade = (band, 24.0)) > J
    # duals in the momenta and positions pass through: ∂J/∂u_x of one electron is Δx·Δy·w·S·u_x/γ
    g5 = sqrt(1 + sum(abs2, us[5]))
    dJ = ForwardDiff.derivative(0.0) do t
        slot_energy(ALIVE, Int32(1), t + 30.0, 10.0 + t, 0.3, -0.2, ws[5], 1.0, 12.0, 24.0)
    end
    @test dJ ≈ ws[5] * 10.0 / g5 rtol = 8eps()                                    # S = 1 there: only the momentum's part
    dS = ForwardDiff.derivative(t -> slot_energy(ALIVE, Int32(1), 18.0 + t, 10.0, 0.3, -0.2, ws[5], 1.0, 12.0, 24.0), 0.0)
    @test dS ≈ ws[5] * (g5 - 1) * 15 / 8 / 12 rtol = 8eps()                       # mid-fade: only the weight's slope
end

@testset "T-832 the guard reads the plasma's weighted electrons ahead of a chosen place, not the born ones [WP-064]" begin
    st = born_test_state([30.0], [(40.0, 0.0, 0.0)], [1.0e-3])
    h = cellsize(st.grid)
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st) == 0            # a cold plasma; the born electron at γ ≈ 40 is not read
    s = st.species[1]
    col(i) = (s.cell[1][i] - 1 + s.offset[1][i]) * h[1]
    weighted = findall(i -> s.status[i] == ALIVE && s.w[i] > 0 && col(i) >= 12.0, eachindex(s.w))
    empty = findall(i -> s.status[i] == ALIVE && s.w[i] == 0 && col(i) >= 12.0, eachindex(s.w))
    behind = findall(i -> s.status[i] == ALIVE && s.w[i] > 0 && col(i) < 12.0, eachindex(s.w))
    @test !isempty(empty)                                                       # the vacuum's particles have the weight zero
    if isempty(weighted) || isempty(behind)                                     # at the start the plasma is ahead of the window: place two
        i, j = empty[1], empty[2]
        s.w[i] = 2.0e-3; s.w[j] = 4.0e-3
        s.cell[1][j] = 3; s.offset[1][j] = 0.5                                  # x ≈ 1.96, behind the fade
        weighted, behind = [i], [j]
        empty = empty[3:end]
    end
    i, j, k = weighted[1], behind[1], empty[1]
    s.u[1][k] = 30.0                                                            # a fast particle of weight zero
    s.u[1][j] = 20.0                                                            # a fast one behind the fade
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st) == 0
    s.u[1][i] = 3.0; s.u[2][i] = 4.0; s.u[3][i] = 1.0                           # γ − 1 = √27 − 1
    @test plasma_peak_energy(st) ≈ sqrt(27) - 1 rtol = 4eps()
    @test plasma_fast_weight(st) == 0                                           # below the 5 mₑc² of the tables
    @test plasma_fast_weight(st; energy = 4.0) ≈ h[1] * h[2] * s.w[i] rtol = 4eps()
    @test plasma_fast_weight(st; energy = 4.19) > 0 && plasma_fast_weight(st; energy = 4.2) == 0   # √27 − 1 = 4.196: all three momenta count
    # the fade's edge to within a cell: the same particle 0.3 cells behind fade[1] is not read, 0.3 cells ahead it is
    c0, o0 = s.cell[1][i], s.offset[1][i]
    s.cell[1][i] = floor(Int, 12.0 / h[1]) + 1; s.offset[1][i] = 12.0 / h[1] - floor(12.0 / h[1]) - 0.3
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st; energy = 4.0) == 0
    s.offset[1][i] += 0.6
    @test plasma_peak_energy(st) ≈ sqrt(27) - 1 rtol = 4eps()
    @test plasma_fast_weight(st; energy = 4.0) ≈ h[1] * h[2] * s.w[i] rtol = 4eps()
    s.cell[1][i] = c0; s.offset[1][i] = o0
    # a retired particle is not read
    s.status[i] = 0x00
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st; energy = 4.0) == 0
    s.status[i] = ALIVE
    @test_throws ArgumentError plasma_fast_weight(st; from = 1.0)
    @test plasma_peak_energy(st; from = 8.0) == plasma_peak_energy(st)           # x ≈ 1.96 is still behind
    @test_throws ArgumentError plasma_peak_energy(st; from = 1.0)               # inside the back band
    @test_throws ArgumentError plasma_fast_energy(st; from = 1.0)
    # from the back band's end the whole window is read: the fast particle behind the fade too (u_x = 20)
    band = back_band_end(st)
    s.cell[1][j] = floor(Int, band / h[1]) + 2; s.offset[1][j] = 0.5
    @test plasma_peak_energy(st) ≈ sqrt(27) - 1 rtol = 4eps()
    @test plasma_peak_energy(st; from = band) ≈ sqrt(401) - 1 rtol = 4eps()
    @test plasma_fast_weight(st; from = band) ≈ h[1] * h[2] * s.w[j] rtol = 4eps()
    @test plasma_fast_energy(st; from = band) ≈ h[1] * h[2] * s.w[j] * (sqrt(401) - 1) rtol = 4eps()
    @test plasma_fast_energy(st; from = band, energy = 4.0) ≈ h[1] * h[2] * (s.w[j] * (sqrt(401) - 1) + s.w[i] * (sqrt(27) - 1)) rtol = 4eps()
    @test plasma_fast_energy(st) == 0
end

"The state of the case at λ₀/8 at t = 190."
born_small_state(a0, dopant = BCASE.dopant; kw...) = born_state(a0, dopant; ppl = 8, t_end = 190.0, capacity = 60_000, kw...)
"The objective of the case at λ₀/8 read at t = 190."
born_small(a0, dopant = BCASE.dopant; kw...) = born_objective(a0, dopant; ppl = 8, t_end = 190.0, capacity = 60_000, kw...)
# At t = 190 the born electrons of this run lie at window x = 27 to 53: the case's weight is 1 on
# all of them, and this one cuts through them.
const ACROSS = (30.0, 50.0)

@testset "T-833 ForwardDiff through the run gives the objective's slope in a₀ and in the dopant density, with the position weight across the electrons too [A3, A7, WP-064]" begin
    a0 = 2.2
    std = born_small_state(ForwardDiff.Dual{Nothing}(a0, 1.0, 0.0), ForwardDiff.Dual{Nothing}(BCASE.dopant, 0.0, 1.0))
    st = born_small_state(a0)
    ha = a0 * 2.0^-12
    sta = (born_small_state(a0 + ha), born_small_state(a0 - ha))
    for fade in (BCASE.fade, ACROSS)
        Jd, J = born_energy(std; fade), born_energy(st; fade)
        @test J > 1.0e-4                                                        # the K shell ionizes and the electrons gain energy
        @test ForwardDiff.value(Jd) == J                                        # the duals do not change the run
        fd_a = (born_energy(sta[1]; fade) - born_energy(sta[2]; fade)) / 2ha
        @test ForwardDiff.partials(Jd)[1] ≈ fd_a rtol = 1.0e-5
        @test ForwardDiff.partials(Jd)[1] > 0
    end
    @test 0.1 * born_energy(st) < born_energy(st; fade = ACROSS) < 0.5 * born_energy(st)   # the second weight does cut (0.22 measured)
    Jd, J = born_energy(std), born_energy(st)
    ad = ForwardDiff.partials(Jd)
    hd = BCASE.dopant * 2.0^-8
    fd_d = (born_small(a0, BCASE.dopant + hd) - born_small(a0, BCASE.dopant - hd)) / 2hd
    @test ad[2] ≈ fd_d rtol = 1.0e-5
    # the dopant is 1 % of the plasma: the born electrons barely load the wake, so J is close to
    # proportional to their density, and not exactly
    @test 0.9 < ad[2] * BCASE.dopant / J < 1.1
    @test ad[2] * BCASE.dopant != J
end

@testset "T-834 the run's objective repeats bit for bit, its keywords reach it, and a full buffer is an error [A4, WP-064]" begin
    st = born_small_state(2.2)
    s = born_summary(st)
    @test s.energy === born_energy(st) === born_small(2.2)                      # a second run: the same bits in J
    @test born_small(2.2; fade = ACROSS) === born_energy(st; fade = ACROSS) < s.energy
    @test s.born ≈ s.alive rtol = 1.0e-12                                       # at t = 190 no born electron has left the window
    @test s.particles == count(==(ALIVE), ionization_electrons(st).status) > 4000
    @test s.plasma_peak == plasma_peak_energy(st) > 0
    @test_throws ErrorException born_state(2.2; ppl = 8, t_end = 190.0, capacity = 1000)   # the run needs more than 4000 slots
end

@testset "T-835 the summary's weights on hand-placed electrons: alive, counted, fast, and the plasma's [WP-064]" begin
    # T-831's electrons, and one fast electron in the middle of the fade (S = 1/2)
    xs = [5.0, 12.0, 15.0, 18.0, 30.0, 60.0, 18.0]
    us = [(3.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, -2.0, 0.5), (0.0, 0.0, 0.0), (10.0, 0.3, -0.2), (0.4, 9.0, 0.0), (7.0, 0.0, 0.0)]
    ws = [1.0e-3, 2.0e-3, 5.0e-4, 7.0e-4, 3.0e-4, 1.1e-3, 4.0e-4]
    st = born_test_state(xs, us, ws)
    h = cellsize(st.grid)
    area = h[1] * h[2]
    S(x) = (q = clamp((x - 12) / 12, 0, 1); q^3 * (10 - 15q + 6q^2))
    s = born_summary(st)
    @test s.particles == 7
    @test s.born == 0                                                           # the atoms have released nothing: these were placed by hand
    @test s.alive ≈ area * sum(ws) rtol = 8eps()
    @test s.counted ≈ area * sum(ws .* S.(xs)) rtol = 8eps()
    @test s.fast ≈ area * (ws[5] + ws[7]) rtol = 8eps()                         # u_x > 5: not the one with u_y = 9
    @test s.fast_counted ≈ area * (ws[5] + ws[7] / 2) rtol = 8eps()
    @test s.alive > s.counted > s.fast > s.fast_counted > 0
    @test s.u_max == 10.0
    @test s.energy === born_energy(st)
    @test s.mean_energy ≈ s.energy / s.counted rtol = 4eps()
    @test born_summary(st; u_fast = 0.5).fast ≈ area * (sum(ws) - ws[4] - ws[6]) rtol = 8eps()
    @test born_summary(st; fade = (27.0, 30.0)).counted ≈ area * (ws[5] + ws[6]) rtol = 8eps()
    # the plasma's part: one weighted fast particle ahead of the fade
    p = st.species[1]
    i = findfirst(j -> p.status[j] == ALIVE && (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] >= 12.0, eachindex(p.w))
    p.w[i] = 2.0e-3; p.u[1][i] = 8.0
    s = born_summary(st)
    @test s.plasma_peak ≈ sqrt(65) - 1 rtol = 4eps()
    @test s.plasma_fast ≈ area * 2.0e-3 rtol = 4eps()
    @test s.plasma_fast_energy ≈ area * 2.0e-3 * (sqrt(65) - 1) rtol = 4eps()
    @test (s.plasma_peak_window, s.plasma_fast_window, s.plasma_fast_energy_window) == (s.plasma_peak, s.plasma_fast, s.plasma_fast_energy)
    @test born_summary(st; u_fast = 8.0).plasma_fast == 0
    # a second fast one between the back band and the fade is in the window's numbers only
    k = findfirst(j -> p.status[j] == ALIVE && back_band_end(st) + 1 <= (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] < 11.0, eachindex(p.w))
    p.w[k] = 5.0e-3; p.u[2][k] = 12.0
    s2 = born_summary(st)
    @test (s2.plasma_peak, s2.plasma_fast, s2.plasma_fast_energy) == (s.plasma_peak, s.plasma_fast, s.plasma_fast_energy)
    @test s2.plasma_peak_window ≈ sqrt(145) - 1 rtol = 4eps()
    @test s2.plasma_fast_window ≈ area * 7.0e-3 rtol = 4eps()
    @test s2.plasma_fast_energy_window ≈ area * (2.0e-3 * (sqrt(65) - 1) + 5.0e-3 * (sqrt(145) - 1)) rtol = 4eps()
end

# Tests of the WP-065 case (trapped2d_case.jl): every trapped electron at a fixed laser energy.
# T-912 runs the case at λ₀/8 to t = 190 eleven times, a few minutes in all.
include(joinpath(@__DIR__, "..", "trapped2d_case.jl"))

@testset "T-910 a₀ from the fixed energy inverts laser_energy in duration and waist, with no energy derivative along either [A3, WP-065]" begin
    E = budget(2.175)
    @test a0_for_energy(TCASE.x0[1], TCASE.x0[2]; energy = E) ≈ 2.175 rtol = 4eps()
    @test budget() == pulse_energy(1.9, 6.0, 10.0)
    for (τ, w) in ((6.0, 10.0), (4.5, 7.0), (8.0, 14.0))
        prob = trapped_problem(τ, w, 80.0, 54.0, 0.072; energy = E, ppl = 8, t_end = 1.0)
        l = prob.lasers[1]
        @test l.duration == τ && l.waist == w && l.focus == 80.0 && l.delay == 54.0
        @test laser_energy(l) ≈ E rtol = 8eps()                               # the engine's energy of the pulse it launches
        # the energy does not move with τ or w: its derivative through the problem's laser is zero to roundoff
        g = ForwardDiff.gradient(v -> laser_energy(trapped_problem(v[1], v[2], 80.0, 54.0, 0.072; energy = E, ppl = 8, t_end = 1.0).lasers[1]), [τ, w])
        @test maximum(abs, g) < 1.0e-12 * E
        # and a₀ does: −a₀/(2τ) along τ
        @test ForwardDiff.derivative(t -> a0_for_energy(t, w; energy = E), τ) ≈ -a0_for_energy(τ, w; energy = E) / (2τ) rtol = 8eps()
    end
    # the dopant follows the plasma's density, and the other parameters reach the problem
    prob = trapped_problem(6.0, 10.0, 70.0, 50.0, 0.05; ppl = 8, t_end = 1.0)
    @test prob.ionization.density ≈ TCASE.dopant_fraction * 0.05 rtol = 4eps()
    @test prob.species[1].density == 0.05 && prob.lasers[1].focus == 70.0 && prob.lasers[1].delay == 50.0
    @test prob.lasers[1].a0 ≈ 1.9 rtol = 4eps()
    # each laser parameter alone can carry a dual (the element type promotes over all of them)
    for (k, v) in ((3, 70.0), (4, 50.0), (2, 12.0))
        f(t) = (x = Any[6.0, 10.0, 70.0, 50.0, 0.05]; x[k] = t; trapped_problem(x...; energy = E, ppl = 8, t_end = 1.0))
        l = f(ForwardDiff.Dual{Nothing}(v, 1.0)).lasers[1]
        @test ForwardDiff.value(getfield(l, (:a0, :waist, :focus, :delay)[k])) == v
        @test ForwardDiff.partials(getfield(l, (:a0, :waist, :focus, :delay)[k]))[1] == 1
    end
    # the state's run is the problem's: at a point away from the base, its antenna and its dopant are trapped_problem's
    st, _, sp = start_of(trapped_problem(5.0, 12.0, 70.0, 50.0, 0.05; energy = E, ppl = 8, t_end = 1.0); ppl = 8, ny = 1)
    @test trapped_state(5.0, 12.0, 70.0, 50.0, 0.05; energy = E, ppl = 8, t_end = 1.0, capacity = 60_000) isa typeof(st)
    @test sp.lasers[1].a0 ≈ a0_for_energy(5.0, 12.0; energy = E) rtol = 4eps()
    @test sp.ionization.density ≈ 0.0005 rtol = 4eps()
end

@testset "T-911 the objective is Δx·Δy·Σ w(γ − 1)S(x)R(u_x) over the born and the plasma's electrons alike [A3, WP-065]" begin
    xs = [5.0, 15.0, 18.0, 30.0, 30.0, 60.0]
    us = [(8.0, 0.0, 0.0), (6.0, 1.0, 0.0), (4.0, -2.0, 0.5), (2.0, 3.0, 0.0), (10.0, 0.3, -0.2), (3.5, 0.0, 0.0)]
    ws = [1.0e-3, 2.0e-3, 5.0e-4, 7.0e-4, 3.0e-4, 1.1e-3]
    st = born_test_state(xs, us, ws)
    h = cellsize(st.grid)
    # the case's position weight is WP-064's first-bucket weight (12, 24)
    @test TCASE.fade == BCASE.fade
    tparts(st; kw...) = trapped_parts(st; fade = BCASE.fade, kw...)
    tenergy(st; kw...) = trapped_energy(st; fade = BCASE.fade, kw...)
    S(x) = (q = clamp((x - 12) / 12, 0, 1); q^3 * (10 - 15q + 6q^2))
    R(u) = (q = clamp((u - 3) / 2, 0, 1); q^3 * (10 - 15q + 6q^2))
    γm(u) = sqrt(1 + sum(abs2, u)) - 1
    born = h[1] * h[2] * sum(w * γm(u) * S(x) * R(u[1]) for (x, u, w) in zip(xs, us, ws))
    parts = tparts(st)
    @test parts.born ≈ born rtol = 8eps()
    @test parts.plasma == 0                                                    # a cold plasma: no electron above u_x = 3
    @test tenergy(st) == parts.born + parts.plasma
    # the electron at x = 30 with u_y = 3 and u_x = 2 has γ − 1 = 2.74 and counts for nothing: it is the forward momentum that decides
    us2 = copy(us); us2[4] = (2.0, 0.0, 0.0)
    @test tparts(born_test_state(xs, us2, ws)).born === parts.born
    # a fast plasma electron counts as a born one would: the label does not matter
    p = st.species[1]
    i = findfirst(j -> p.status[j] == ALIVE && (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] >= 30.0, eachindex(p.w))
    xi = (p.cell[1][i] - 1 + p.offset[1][i]) * h[1]
    p.w[i] = 2.0e-3; p.u[1][i] = 4.5; p.u[2][i] = 1.0
    # and its summary: the plasma's numbers are its own, the born ones unchanged
    sp = trapped_summary(st; fade = BCASE.fade)
    Rp = h[1] * h[2] * 2.0e-3 * S(xi) * R(4.5)
    @test sp.weight_plasma ≈ Rp rtol = 8eps()
    @test sp.mean_energy_plasma ≈ sqrt(1 + 4.5^2 + 1) - 1 rtol = 8eps()
    @test sp.ux_max_plasma == 4.5
    @test sp.fast_plasma_window == 0                                            # u_x = 4.5 is below the rise's top
    p.u[1][i] = 5.5
    @test trapped_summary(st; fade = BCASE.fade).fast_plasma_window ≈ h[1] * h[2] * 2.0e-3 rtol = 8eps()
    @test trapped_summary(st; fade = BCASE.fade).weight_born === sp.weight_born
    p.u[1][i] = 4.5
    @test tparts(st).plasma ≈ h[1] * h[2] * 2.0e-3 * (sqrt(1 + 4.5^2 + 1) - 1) * S(xi) * R(4.5) rtol = 8eps()
    @test tparts(st).born === parts.born
    @test tenergy(st) ≈ parts.born + h[1] * h[2] * 2.0e-3 * (sqrt(1 + 4.5^2 + 1) - 1) * S(xi) * R(4.5) rtol = 8eps()
    # a retired one is not read, nor a free slot's leftovers
    p.status[i] = 0x00
    @test tparts(st).plasma == 0
    e = ionization_electrons(st)
    e.u[1][7] = 1.0e6; e.w[7] = 1.0; e.cell[1][7] = 40
    @test tparts(st).born === parts.born
    # another trapping weight: (1, 2) counts the electron at u_x = 2 in full and every other in full
    @test tparts(st; rise = (1.0, 2.0)).born ≈ h[1] * h[2] * sum(w * γm(u) * S(x) for (x, u, w) in zip(xs, us, ws)) rtol = 8eps()
    @test_throws ArgumentError tenergy(st; rise = (5.0, 3.0))
    @test_throws ArgumentError tenergy(st; rise = (3.0, 3.0))
    @test_throws ArgumentError trapped_energy(st; fade = (1.0, 24.0))     # inside the back band
    # ∂J/∂u_x of one electron mid-rise is the energy's part and the rise's: w·S·(u_x/γ·R + (γ − 1)·R′)
    d = ForwardDiff.derivative(t -> slot_trapped(ALIVE, Int32(1), 30.0, 4.0 + t, 0.0, 0.0, 1.0, 1.0, 12.0, 24.0, 3.0, 5.0), 0.0)
    g = sqrt(17.0)
    @test d ≈ 4 / g * R(4.0) + (g - 1) * 15 / 8 / 2 rtol = 8eps()
    # the summary's numbers on the same electrons
    s = trapped_summary(st; fade = BCASE.fade)
    @test s.J === tenergy(st) && s.J_born === parts.born && s.J_plasma == 0
    @test s.weight_born ≈ h[1] * h[2] * sum(w * S(x) * R(u[1]) for (x, u, w) in zip(xs, us, ws)) rtol = 8eps()
    @test s.mean_energy_born ≈ s.J_born / s.weight_born rtol = 4eps()
    @test s.fast_born_window ≈ h[1] * h[2] * (ws[2] + ws[5]) rtol = 8eps()   # u_x > 5 ahead of the back band (x = 5 is inside it at λ₀/8)
    @test s.ux_max_born == 10.0 && s.particles == 6 + count(==(ALIVE), p.status)
    # a fast plasma electron between the back band's end (7.1) and the weight (12) is dark current, not J
    k = findfirst(j -> p.status[j] == ALIVE && back_band_end(st) + 0.5 <= (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] < 11.5, eachindex(p.w))
    p.w[k] = 5.0e-3; p.u[1][k] = 6.0
    sd = trapped_summary(st; fade = BCASE.fade)
    @test sd.dark_weight ≈ h[1] * h[2] * 5.0e-3 rtol = 8eps()
    @test sd.dark_energy ≈ h[1] * h[2] * 5.0e-3 * (sqrt(37.0) - 1) rtol = 8eps()
    @test sd.J === s.J
    @test sd.fast_plasma_window ≈ h[1] * h[2] * 5.0e-3 rtol = 8eps()
    @test s.dark_weight == 0
    # a fast born electron between the back band's end (7.1) and the weight (12) is in the window's number only
    st9 = born_test_state([xs; 9.0], [us; (6.0, 0.0, 0.0)], [ws; 4.0e-4])
    s9 = trapped_summary(st9; fade = BCASE.fade)
    @test s9.fast_born_window ≈ h[1] * h[2] * (ws[2] + ws[5] + 4.0e-4) rtol = 8eps()
    @test s9.J_born === trapped_summary(born_test_state(xs, us, ws); fade = BCASE.fade).J_born
end

"The WP-065 objective at λ₀/8 read at t = 190, with a trapping weight from u_x = 0.5 to 3 that cuts through the electrons of that early time (the born ones reach u_x = 2.7, the plasma's 1.0)."
trapped_small(x...; kw...) = trapped_objective(x...; ppl = 8, t_end = 190.0, capacity = 60_000, rise = (0.5, 3.0), fade = BCASE.fade, energy = budget(2.2), kw...)

@testset "T-912 ForwardDiff through the run gives the objective's gradient in all five parameters [A3, A7, WP-065]" begin
    x0 = collect(TCASE.x0)
    xd = [ForwardDiff.Dual{Nothing}(x0[i], ntuple(j -> Float64(i == j), 5)) for i in 1:5]
    Jd = trapped_small(xd...)
    J = trapped_small(x0...)
    @test ForwardDiff.value(Jd) === J                                           # the duals do not change the run
    @test J > 1.0e-4
    st = trapped_state(x0...; ppl = 8, t_end = 190.0, capacity = 60_000, energy = budget(2.2))
    parts = trapped_parts(st; rise = (0.5, 3.0), fade = BCASE.fade)
    @test parts.born > 0 && parts.plasma > 0                                    # both species are read
    ad = ForwardDiff.partials(Jd)
    # At this early time J is mostly the plasma's quiver in the pulse (the rise from 0.5 reaches
    # it): the test is of the derivative through both species, not of trapping. J is rough in the
    # delay below its scale (central differences there reach ForwardDiff only at steps below about
    # 10⁻³), so every direction is checked at a relative step of 2⁻¹⁸.
    for i in 1:5
        hi = x0[i] * 2.0^-18
        xp, xm = copy(x0), copy(x0)
        xp[i] += hi; xm[i] -= hi
        fd = (trapped_small(xp...) - trapped_small(xm...)) / 2hi
        @test ad[i] ≈ fd rtol = 1.0e-5
        @test abs(ad[i]) > 1.0e-3 * J / x0[i]                                  # every parameter moves J
    end
end

@testset "T-913 a finer y keeps the box, the absorbers' and the dopant's widths, and ny = 1 is WP-064's case [WP-065]" begin
    p1 = born_problem(; ppl = 16, t_end = 1.0)
    p2 = born_problem(; ppl = 16, t_end = 1.0, ny = 2)
    @test p2.cells == (p1.cells[1], 2 * p1.cells[2])
    @test p2.length[2] ≈ p1.length[2] rtol = 4eps()
    @test p2.length[1] == p1.length[1]
    @test p2.dt < p1.dt
    @test p2.lasers[1].center ≈ p1.lasers[1].center rtol = 4eps()
    r1, r2 = p1.ionization.rows, p2.ionization.rows
    @test r2[1] ≈ r1[1] rtol = 8eps()
    @test r2[2] ≈ r1[2] rtol = 8eps()
    @test born_numerics(16, 2).absorber.cells == 2 * born_numerics(16).absorber.cells
    @test born_numerics(16, 2).back_absorber.cells == born_numerics(16).back_absorber.cells
    @test born_start(; ppl = 16, ny = 2, t_end = 1.0, capacity = 10)[1].numerics.absorber.cells == 2 * BCASE.pml_y
    @test_throws ArgumentError born_problem(; ppl = 16, ny = 0)
    # a longer window keeps more wake behind: its front, and so the lab frame's pulse and plasma, stay where they were
    pe = born_problem(; ppl = 16, t_end = 1.0, extend = 4)
    @test pe.cells == (18 * 16, 128)
    @test pe.window.start ≈ BCASE.start + 8π rtol = 4eps()
    @test pe.length[1] - pe.window.start ≈ p1.length[1] - p1.window.start rtol = 8eps()   # the front's lab position at any time after both start
    @test pe.lasers[1].position == p1.lasers[1].position
    @test_throws ArgumentError born_problem(; ppl = 16, extend = -1)
    # the case's position weight starts ahead of the back band at λ₀/16 and finer
    for (ppl, ny) in ((16, 1), (32, 2), (64, 4))
        st, _, _ = born_start(; ppl, ny, t_end = 1.0, capacity = 10)
        @test back_band_end(st) < TCASE.fade[1]
    end
    # at ny = 1 the grid is WP-064's to the bit: its formulas before the keywords
    for ppl in (8, 16, 32)
        q = born_problem(; ppl, t_end = 1.0)
        h = 2 * π / ppl
        hy = (ppl ÷ 8) * h
        @test q.cells == (14 * ppl, 128)
        @test q.dt === 95 / 100 / sqrt(1 / h^2 + 1 / hy^2)
        @test q.length === (14 * ppl * h, 128 * hy)
        yc = 128 * hy / 2
        @test q.ionization.rows === (yc - 18 * hy, yc + 18 * hy)
        @test born_numerics(ppl).absorber.cells == 12
    end
end

@testset "T-914 the case runs in Float32, with J and its derivative within 10⁻⁴ of Float64 [A8, WP-065]" begin
    kw = (; ppl = 8, t_end = 190.0, capacity = 60_000, energy = budget(2.2), float = Float32)
    st32 = trapped_state(TCASE.x0...; kw...)
    @test eltype(st32.species[1].w) == Float32 && eltype(ionization_electrons(st32).w) == Float32
    J32 = trapped_energy(st32; fade = BCASE.fade, rise = (0.5, 3.0))
    @test J32 isa Float32
    J64 = trapped_small(TCASE.x0...)
    @test J32 ≈ J64 rtol = 1.0e-4                                               # 1.6·10⁻⁵ measured
    # a Float32 dual keeps the run in Float32, and its derivative in the focus is Float64's
    d32 = trapped_energy(trapped_state(6.0f0, 10.0f0, ForwardDiff.Dual{Nothing}(80.0f0, 1.0f0), 54.0f0, 0.072f0; kw...); fade = BCASE.fade, rise = (0.5, 3.0))
    d64 = trapped_small(6.0, 10.0, ForwardDiff.Dual{Nothing}(80.0, 1.0), 54.0, 0.072)
    @test d32 isa ForwardDiff.Dual{Nothing,Float32}
    @test ForwardDiff.partials(d32)[1] ≈ ForwardDiff.partials(d64)[1] rtol = 1.0e-4
    # a Float32 dual in the density stays Float32 through the dopant's fraction of it
    stn = trapped_state(6.0f0, 10.0f0, 80.0f0, 54.0f0, ForwardDiff.Dual{Nothing}(0.072f0, 1.0f0); kw..., t_end = 1.0)
    @test eltype(ionization_electrons(stn).w) == ForwardDiff.Dual{Nothing,Float32,1}
    @test trapped_capacity(16, 1) == BCASE.capacity && trapped_capacity(64, 4) == 2_560_000
end

# Tests of the WP-068 optimizer (trapped2d_opt.jl): the box, the gather keyword, the store of
# points and one L-BFGS-B step on the case at λ₀/8.
include(joinpath(@__DIR__, "..", "trapped2d_opt.jl"))

@testset "T-970 the optimizer's box is WP-068's and holds the base point and the starts with z in [0.6, 1.5], and the delay stays at the base point's [WP-068]" begin
    base = [TCASE.x0[i] for i in TCASE.free]
    @test TRAPPED_BOX.scale == [TCASE.scale[i] for i in TCASE.free] == [6.0, 10.0, 80.0, 0.072]
    @test STARTS.base == base
    for s in values(STARTS)
        @test length(s) == 4 && all(TRAPPED_BOX.lo .< s .< TRAPPED_BOX.hi)
    end
    @test TRAPPED_BOX.lo == [4.0, 6.0, 64.0, 0.05] && TRAPPED_BOX.hi == [9.0, 14.0, 110.0, 0.10]   # WP-068's box
    zlo, zhi = TRAPPED_BOX.lo ./ TRAPPED_BOX.scale, TRAPPED_BOX.hi ./ TRAPPED_BOX.scale
    @test all(0.6 .<= zlo .< 1) && all(1 .< zhi .<= 1.5)
    # the four free parameters and the fixed delay make the case's five, in any element type
    @test trapped_point(base) === TCASE.x0
    @test trapped_point([4.0, 6.0, 64.0, 0.05]) === (4.0, 6.0, 64.0, 54.0, 0.05)
    d = ForwardDiff.Dual{Nothing}(7.0f0, 1.0f0)
    p = trapped_point([d, ForwardDiff.Dual{Nothing}(8.0f0, 0.0f0), ForwardDiff.Dual{Nothing}(90.0f0, 0.0f0), ForwardDiff.Dual{Nothing}(0.08f0, 0.0f0)])
    @test p[4] isa ForwardDiff.Dual{Nothing,Float32,1} && ForwardDiff.value(p[4]) == 54.0f0 && ForwardDiff.partials(p[4])[1] == 0
    @test p[1] === d
    @test_throws ArgumentError trapped_point([6.0, 10.0])
    @test_throws ArgumentError trapped_point(collect(TCASE.x0))
    # a₀ at the 1.9 budget: about 1.3 and 3.0 at the box's corners, and the starts' as STARTS says
    @test a0_for_energy(9.0, 14.0) ≈ 1.31 atol = 0.01
    @test a0_for_energy(4.0, 6.0) ≈ 2.99 atol = 0.01
    @test [round(a0_for_energy(s[1], s[2]); digits = 2) for s in (STARTS.base, STARTS.long, STARTS.short, STARTS.narrow)] == [1.9, 1.5, 2.32, 2.1]
end

@testset "T-971 the gather keyword reaches the case's numerics and changes the run; gather = nothing is the default run [WP-067, WP-068]" begin
    @test born_numerics(32, 2).gather === nothing
    @test born_numerics(32, 2; gather = CompensatedGather()).gather isa CompensatedGather
    @test born_numerics(32, 2; gather = CompensatedGather()).absorber.cells == born_numerics(32, 2).absorber.cells
    st, _, _ = born_start(; ppl = 8, t_end = 1.0, capacity = 10, gather = CompensatedGather())
    @test st.numerics.gather isa CompensatedGather
    @test born_start(; ppl = 8, t_end = 1.0, capacity = 10)[1].numerics.gather === nothing
    @test opt_gather("compensated") isa CompensatedGather && opt_gather("plain") === nothing
    @test_throws ErrorException opt_gather("sharp")
    sc = trapped_state(TCASE.x0...; ppl = 8, t_end = 190.0, capacity = 60_000, energy = budget(2.2), gather = CompensatedGather())
    @test sc.numerics.gather isa CompensatedGather
    J0 = trapped_small(TCASE.x0...)
    @test trapped_small(TCASE.x0...; gather = nothing) === J0
    Jc = trapped_energy(sc; fade = BCASE.fade, rise = (0.5, 3.0))
    # the particles feel the laser field the quadratic shape removes at λ₀/8 (sinc(π/8)³ ≈ 0.92)
    @test Jc > 1.05 * J0
    @test run_point(STARTS.base; ppl = 8, ny = 1, t_end = 190.0, capacity = 60_000, energy = budget(2.2), gather = CompensatedGather(),
                    grad = false, read = (; fade = BCASE.fade, rise = (0.5, 3.0)))["J"] === Float64(Jc)
end

@testset "T-972 the store gives the value and the gradient of a point from one run, and a stopped optimization continues from its records on the same path [WP-068]" begin
    # a smooth objective in place of the case, with its exact gradient, counting its runs
    scale = TRAPPED_BOX.scale
    c = [7.3, 11.0, 120.0, 0.065]                                  # the focus's maximum outside the box
    Jq(x) = 2.0 - sum(((x .- c) ./ scale) .^ 2)
    gq(x) = -2 .* (x .- c) ./ scale .^ 2
    ran = Ref(0)
    fake(free) = (ran[] += 1; Dict{String,Any}("J" => Jq(free), "gradient" => gq(free), "a0" => a0_for_energy(free[1], free[2])))
    dir = mktempdir()
    store = PointStore(joinpath(dir, "points"); run = fake)
    calls = Dict{String,Any}[]
    prob = case_problem(stored_objective(store, calls), TRAPPED_BOX, Tuple(STARTS.base))
    J, g = value_gradient(prob, prob.u0)
    @test J == Jq(STARTS.base) && g == gq(STARTS.base) .* scale    # the chain rule in z through the record's gradient
    @test prob.f(prob.u0, prob.p) == J
    @test ran[] == 1 && store.made == 1                              # one run for the value and the gradient
    @test [c["asked"] for c in calls] == ["gradient", "value"] && [c["stored"] for c in calls] == [false, true]
    @test isfile(joinpath(dir, "points", point_key(STARTS.base) * ".json"))
    @test point_key([1.0, 2.0]) == "3ff0000000000000-4000000000000000"
    @test point_key([0.072, 80.0]) != point_key([nextfloat(0.072), 80.0]) != point_key([0.072, prevfloat(80.0)])
    # an optimization from a start, then again from the records only: the same path, no run
    r1 = optimize_start(store, STARTS.long, joinpath(dir, "long"); maxiters = 30, pgtol = 1e-8)
    @test r1["free"][3] == TRAPPED_BOX.hi[3]                          # the focus on its bound
    @test r1["free"][[1, 2, 4]] ≈ c[[1, 2, 4]] rtol = 1e-6
    @test r1["projected_gradient"] <= 1e-8
    n = ran[]
    again = PointStore(joinpath(dir, "points"); run = fake)
    @test length(again.records) == length(store.records)
    r2 = optimize_start(again, STARTS.long, joinpath(dir, "long2"); maxiters = 30, pgtol = 1e-8)
    @test ran[] == n && again.made == 0
    @test r2["path"] == [merge(p, Dict("seconds" => q["seconds"])) for (p, q) in zip(r1["path"], r2["path"])]
    @test r2["free"] == r1["free"] && r2["J"] == r1["J"]
    @test JSON.parsefile(joinpath(dir, "long", "result.json"))["free"] == r1["free"]
    @test all(p -> haskey(store.records, p["key"]), r1["path"])            # every point of the path has its record
    # stopped after one iteration, then continued from the records: the uninterrupted path, and
    # only the points past the stop are run
    sdir = mktempdir()
    stopped = PointStore(sdir; run = fake)
    rs = optimize_start(stopped, STARTS.long, joinpath(sdir, "a"); maxiters = 1, pgtol = 1e-8)
    @test rs["retcode"] == "MaxIters" && rs["iterations"] == 1 < r1["iterations"]
    n = ran[]
    resumed = PointStore(sdir; run = fake)
    rr = optimize_start(resumed, STARTS.long, joinpath(sdir, "b"); maxiters = 30, pgtol = 1e-8)
    @test [p["z"] for p in rr["path"]] == [p["z"] for p in r1["path"]] && rr["free"] == r1["free"]
    @test ran[] - n == resumed.made == r1["points_run"] - rs["points_run"] > 0
    # the stopped result's projected gradient is in z, the record's physical gradient times the scale
    gs = stopped.records[rs["key"]]["gradient"] .* scale
    @test rs["projected_gradient"] == projected_gradient(gs, Float64.(rs["u"]), prob.lb, prob.ub) > 1e-6
    # pgtol reaches the solver: a loose one stops before a tight one
    loose = optimize_start(PointStore(mktempdir(); run = fake), STARTS.long, joinpath(dir, "loose"); maxiters = 30, pgtol = 0.6)
    @test loose["iterations"] < r1["iterations"] && loose["retcode"] == "Success" && loose["projected_gradient"] <= 0.6
    # each distinct point ran once, and every point stayed in the box
    @test length(unique(c["key"] for c in r1["calls"])) == r1["points_run"]
    @test all(c -> all(TRAPPED_BOX.lo .<= c["free"] .<= TRAPPED_BOX.hi), r1["calls"])
    # the projected gradient drops the components that push out of the box at their bounds
    lb, ub = prob.lb, prob.ub
    @test projected_gradient([1.0, -2.0, 0.5, 0.1], ub, lb, ub) == 2.0
    @test projected_gradient([1.0, -2.0, 0.5, 0.1], [ub[1], lb[2], ub[3], 1.0], lb, ub) == 0.1
    @test projected_gradient([-1.0, 2.0, 0.0, 0.0], [ub[1], lb[2], 1.0, 1.0], lb, ub) == 2.0
    # the check's central differences step one free parameter each, by its relative 2^-power
    jobs = check_jobs(STARTS.base, Dict{String,Any}("float" => 32, "ppl" => 48, "ny" => 3, "fdppl" => 32, "fdny" => 2, "power" => 13))
    @test [j[1] for j in jobs[1:3]] == ["fine-plain", "fine-grad", "f64-grad"] && length(jobs) == 3 + 8
    @test jobs[1][2][:float] == Float32 && jobs[3][2][:float] == Float64 && jobs[1][2][:ppl] == 48 && jobs[3][2][:ppl] == 32
    @test [j[2][:grad] for j in jobs[1:3]] == [false, true, true] && jobs[1][2][:ny] == jobs[2][2][:ny] == 3 && jobs[2][2][:ppl] == 48
    @test jobs[3][2][:ny] == 2 && jobs[2][2][:float] == Float32
    @test all(j -> j[2][:ppl] == 32 && j[2][:ny] == 2 && j[2][:float] == Float64 && !j[2][:grad], jobs[4:end])
    @test all(j -> j[2][:h] == STARTS.base[j[2][:direction]] * 2.0^-13, jobs[4:end])
    for (id, j) in jobs[4:end]
        i = j[:direction]
        @test j[:free] == STARTS.base .+ [k == i ? j[:sign] * STARTS.base[i] * 2.0^-13 : 0.0 for k in 1:4]
        @test id == "fd-$(TCASE.names[TCASE.free[i]])-$(j[:sign] > 0 ? "p" : "m")"
    end
    @test parse_options(["ppl=48", "gather=plain", "pgtol=0.5", "out"], OPT_DEFAULTS) ==
          (merge(Dict{String,Any}(string(k) => v for (k, v) in pairs(OPT_DEFAULTS)), Dict{String,Any}("ppl" => 48, "gather" => "plain", "pgtol" => 0.5)), ["out"])
    @test_throws ErrorException parse_options(["speed=2"], OPT_DEFAULTS)
end

@testset "T-973 one L-BFGS-B step on the case at λ₀/8 moves J up, every point a run with duals in the free directions with the delay fixed [A3, WP-068]" begin
    read = (; fade = BCASE.fade, rise = (0.5, 3.0))
    dir = mktempdir()
    store = PointStore(dir; run = free -> run_point(free; ppl = 8, ny = 1, t_end = 190.0, capacity = 60_000, energy = budget(2.2), read))
    r = optimize_start(store, STARTS.base, joinpath(dir, "base"); maxiters = 1, pgtol = 0.0)
    start = store.records[point_key(STARTS.base)]
    @test start["J"] === trapped_small(TCASE.x0...)                    # the run with duals gives the plain run's J
    @test r["J"] > start["J"]
    @test r["iterations"] == 1 && r["points_run"] == length(store.records) >= 2
    # the record's gradient is ForwardDiff's in the free parameters, the delay's derivative left out
    x0 = collect(TCASE.x0)
    xd = [ForwardDiff.Dual{Nothing}(x0[i], ntuple(j -> Float64(i == TCASE.free[j]), 4)) for i in 1:5]
    @test start["gradient"] == collect(ForwardDiff.partials(trapped_small(xd...)))
    @test all(rec -> rec["x"][4] == TCASE.x0[4] && length(rec["gradient"]) == 4, values(store.records))
    @test all(rec -> haskey(rec, "dark_weight") && rec["a0"] == a0_for_energy(rec["free"][1], rec["free"][2]; energy = budget(2.2)), values(store.records))
    # the summary reads with the same weights as J, and the record's z is the point over the box's scale
    @test all(rec -> rec["J_born"] + rec["J_plasma"] == rec["J"] && rec["z"] == rec["free"] ./ TRAPPED_BOX.scale, values(store.records))
    @test r["path"][1]["J"] == start["J"] && r["path"][end]["J"] == r["J"]
end

@testset "T-974 the scan's grids: the plane that holds a path, and the pairs of free parameters over the box through the optimum [WP-068]" begin
    # a path in the plane of two known directions through the optimum
    zopt = [1.0, 1.1, 1.2, 0.9]
    a, b = [1.0, 1.0, 0.0, 0.0] ./ sqrt(2), [0.0, 0.0, 0.6, 0.8]
    zs = [zopt .+ s .* a .+ t .* b for (s, t) in ((0.3, -0.1), (0.2, 0.05), (0.1, 0.02), (0.0, 0.0), (-0.05, 0.1))]
    pl = path_plane(zs, zopt)
    @test pl.residual < 1e-12
    @test abs(vdot(pl.e1, pl.e2)) < 1e-12 && vnorm(pl.e1) ≈ 1 && vnorm(pl.e2) ≈ 1
    for (k, z) in enumerate(zs)
        @test zopt .+ pl.s[k] .* pl.e1 .+ pl.t[k] .* pl.e2 ≈ z atol = 1e-12
    end
    # a straight path still gets a second direction, orthogonal to the first
    line = path_plane([zopt .+ s .* a for s in (0.0, 0.1, 0.2)], zopt)
    @test abs(abs(vdot(line.e1, a)) - 1) < 1e-12 && abs(vdot(line.e1, line.e2)) < 1e-12 && line.residual < 1e-12
    # a path that never moves in the first coordinates still gets its own plane
    b2 = [0.0, 1.0, 1.0, 0.0] ./ sqrt(2)
    off = path_plane([zopt .+ s .* b2 .+ t .* b for (s, t) in ((0.3, -0.1), (0.2, 0.05), (0.0, 0.0), (-0.05, 0.1))], zopt)
    @test off.residual < 1e-12 && off.e1[1] == 0 && off.e2[1] == 0
    stay = path_plane([zopt .+ s .* [0.0, 0.0, 0.6, 0.8] for s in (0.0, 0.1, 0.3)], zopt)
    @test stay.e1 ≈ [0.0, 0.0, 0.6, 0.8] && stay.s ≈ [0.0, 0.1, 0.3] && all(iszero, stay.t)
    @test path_plane([zopt], zopt).residual == 0                         # a path of one point
    # the jobs of a run with two starts of an objective with two maxima, the higher one near `short`
    scale = TRAPPED_BOX.scale
    c1, c2 = [7.3, 11.0, 120.0, 0.065], [4.6, 7.0, 68.0, 0.098]
    bump(x, c) = exp(-sum(((x .- c) ./ scale) .^ 2) / 0.02)
    fake(free) = (J = bump(free, c1) + 2 * bump(free, c2);
                  g = -2 .* ((free .- c1) .* bump(free, c1) .+ 2 .* (free .- c2) .* bump(free, c2)) ./ scale .^ 2 ./ 0.02;
                  Dict{String,Any}("J" => J, "gradient" => g, "a0" => 1.9))
    dir = mktempdir()
    store = PointStore(joinpath(dir, "points"); run = fake)
    rs = Dict(nm => optimize_start(store, STARTS[nm], joinpath(dir, string(nm)); maxiters = 50, pgtol = 1e-10) for nm in (:long, :short))
    @test rs[:short]["J"] > rs[:long]["J"]
    @test rs[:short]["free"] ≈ c2 rtol = 1e-4
    jobs = scan_jobs(dir, Dict{String,Any}("n" => 5); coords = Coords())
    pairs = filter(j -> j[3]["kind"] == "pair", jobs)
    planes = filter(j -> j[3]["kind"] == "plane", jobs)
    @test length(pairs) == 6 * 25 && length(unique(j[1] for j in jobs)) == length(jobs)
    @test 0 < length(planes) <= 2 * 25
    @test all(j -> all(TRAPPED_BOX.lo .- 1e-12 .<= j[2] .<= TRAPPED_BOX.hi .+ 1e-12), jobs)
    @test all(j -> j[3]["start"] == "short", pairs)                       # the pairs go through the best optimum
    best = rs[:short]["free"]
    for (id, x, spec) in pairs
        others = setdiff(1:4, (spec["a"], spec["b"]))
        @test x[others] == best[others]
        @test x[spec["a"]] == range(TRAPPED_BOX.lo[spec["a"]] / scale[spec["a"]], TRAPPED_BOX.hi[spec["a"]] / scale[spec["a"]]; length = 5)[spec["i"]] * scale[spec["a"]]
        @test x[spec["b"]] == range(TRAPPED_BOX.lo[spec["b"]] / scale[spec["b"]], TRAPPED_BOX.hi[spec["b"]] / scale[spec["b"]]; length = 5)[spec["j"]] * scale[spec["b"]]
    end
    @test sort(unique((j[3]["a"], j[3]["b"]) for j in pairs)) == [(a, b) for a in 1:4 for b in (a + 1):4]
    # every plane point is on its start's plane, and the grid reaches past the path on both sides
    for (id, x, spec) in planes
        r = load_results(dir)[spec["start"]]
        p = path_plane([Float64.(q["z"]) for q in r["path"]], Float64.(r["u"]))
        @test x ./ scale ≈ p.origin .+ spec["s"] .* p.e1 .+ spec["t"] .* p.e2 atol = 1e-12
        @test spec["s_grid"][1] < minimum(p.s) && spec["s_grid"][end] > maximum(p.s)
        @test spec["t_grid"][1] < minimum(p.t) && spec["t_grid"][end] > maximum(p.t)
        @test spec["s"] == spec["s_grid"][spec["i"]] && spec["t"] == spec["t_grid"][spec["j"]]
        @test spec["e1"] == p.e1 && spec["e2"] == p.e2 && spec["origin"] == p.origin
    end
end

include(joinpath(@__DIR__, "..", "trapped2d_opt_report.jl"))

@testset "T-975 the report's central differences and the scan's planes from the records [WP-068]" begin
    # the check's central differences against the Float64 gradient, in d ln J/dz, from synthetic records
    dir = mktempdir()
    g = [0.5, -0.2, 0.01, 30.0]
    J0 = 2.0
    write_job(id, d) = open(io -> JSON.print(io, d), joinpath(dir, id * ".json"), "w")
    write_job("f64-grad", Dict("J" => J0, "gradient" => g))
    for i in 1:4, (sfx, sgn) in (("p", 1), ("m", -1))
        h = STARTS.base[i] * 2.0^-13
        write_job("fd-$(TCASE.names[TCASE.free[i]])-$sfx", Dict("J" => J0 + sgn * h * 1.1 * g[i], "h" => h))
    end
    rows = check_rows(dir)
    @test [r["direction"] for r in rows] == ["duration", "waist", "focus", "density"]
    for (i, r) in enumerate(rows)
        @test r["fd"] ≈ 1.1 * g[i] rtol = 1e-9
        @test r["ad_log"] == g[i] * TRAPPED_BOX.scale[i] / J0
        @test r["fd_log"] ≈ 1.1 * r["ad_log"] rtol = 1e-9
    end
    @test isempty(check_rows(joinpath(dir, "none")))
    # a scan's plane from its jobs' points, with and without the plane recorded in them
    o, e1, e2 = [1.0, 1.1, 1.2, 0.9], [0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 0.8, -0.6]
    jobs = [Dict{String,Any}("kind" => "plane", "start" => "x", "i" => i, "j" => j, "s" => s, "t" => t, "z" => o .+ s .* e1 .+ t .* e2)
            for (i, s) in enumerate((-0.1, 0.0, 0.2)) for (j, t) in enumerate((-0.05, 0.1))]
    pl = scan_plane(jobs)
    @test isapprox(pl.origin, o; atol = 1e-14) && isapprox(pl.e1, e1; atol = 1e-12) && isapprox(pl.e2, e2; atol = 1e-12)
    rec = [merge(j, Dict{String,Any}("origin" => o, "e1" => e1, "e2" => e2)) for j in jobs]
    @test scan_plane(rec) == (; origin = o, e1, e2)
end

@testset "T-976 a peak power cap as a floor on the duration, in the solver's coordinates [WP-068]" begin
    # CETAL's 25 fs at 4.2e18 cm^-3: T_p = 2π/ω_p = 54.4 fs, so τ_FWHM ≥ 0.460 λ_p, τ ≥ 2.454/√n
    cap = cetal_cap()
    Tp = 2π / (5.641e4 * sqrt(4.2e18)) * 1e15
    @test 54.0 < Tp < 54.8 && cap.p == 0.5
    @test cap.c ≈ 25 / Tp * 2π / sqrt(2 * log(2)) rtol = 1e-14
    @test round(cap.c; digits = 2) == 2.45 && round(duration_floor(cap, 0.10); digits = 1) == 7.8 && round(duration_floor(cap, 0.05); digits = 1) == 11.0
    @test cetal_cap(; fwhm_fs = 35.0).c ≈ 1.4 * cap.c rtol = 1e-14
    # the floor in the shortest duration is the power cap: P ∝ W/τ at the fixed energy, so at the
    # floor every waist has the same peak power a₀²·w·(1 + 1/(2w²)) ∝ W/τ_floor
    W = budget()
    power(τ, w) = a0_for_energy(τ, w; energy = W)^2 * w * (1 + 1 / (2 * w^2))
    @test power(duration_floor(cap, 0.08), 6.0) ≈ power(duration_floor(cap, 0.08), 14.0) rtol = 1e-12
    @test power(duration_floor(cap, 0.08), 6.0) ≈ 4W / (π * duration_floor(cap, 0.08)) rtol = 1e-12
    # the coordinates: the duration's place between the floor and 12, from 0 to 1, the others as in the box
    c = Coords(cap)
    b = Coords()
    @test c.duration_hi == 12.0 && c.lo == [0.0; b.lo[2:end]] && c.hi == [1.0; b.hi[2:end]]
    @test b.lo == TRAPPED_BOX.lo ./ TRAPPED_BOX.scale && free_of(b, [1.0, 1.0, 1.0, 1.0]) == TRAPPED_BOX.scale
    @test_throws ArgumentError Coords(cap; duration_hi = 10.5)          # below the floor at 0.05 n_c, 11.0
    x = [9.0, 8.0, 100.0, 0.09]
    f9 = cap.c / sqrt(0.09)
    @test isapprox(free_of(c, z_of(c, x)), x; rtol = 1e-14) && isapprox(z_of(c, x)[1], (9.0 - f9) / (12.0 - f9); rtol = 1e-14)
    @test free_of(c, [0.0, 1.0, 1.0, 1.25])[1] == duration_floor(cap, 0.09) && free_of(c, [1.0, 1.0, 1.0, 1.25])[1] ≈ 12.0
    @test isapprox(shown(c, x)[1], z_of(c, x)[1]; rtol = 1e-14) && shown(c, x)[2:end] == x[2:end] && shown(b, x) == x
    @test shown_name(c, 1) == "τ: floor→12" && shown_name(b, 1) == "duration" && shown_bounds(c).lo == [0.0; TRAPPED_BOX.lo[2:end]]
    @test coords_of(Dict{String,Any}()) isa Coords{Nothing} && coords_of(Dict{String,Any}("cap" => "none")) isa Coords{Nothing}
    @test coords_of(Dict{String,Any}("cap" => "cetal25", "duration_hi" => 13.0)).duration_hi == 13.0
    @test_throws ArgumentError coords_of(Dict{String,Any}("cap" => "cetal35"))   # its floor at 0.05 n_c is 15.4
    @test_throws ErrorException coords_of(Dict{String,Any}("cap" => "1PW"))
    @test OPT_DEFAULTS.cap == "none" && OPT_DEFAULTS.duration_hi == 12.0
    # the gradient in z is the chain rule through the floor, as ForwardDiff through the map gives it
    f(free) = free[4]^2 / free[1] + sin(free[2]) * free[3]
    gf(free) = [-free[4]^2 / free[1]^2, cos(free[2]) * free[3], sin(free[2]), 2 * free[4] / free[1]]
    z = z_of(c, x)
    @test grad_z(c, z, gf(free_of(c, z))) ≈ ForwardDiff.gradient(u -> f(free_of(c, u)), z) rtol = 1e-12
    @test grad_z(b, z_of(b, x), gf(x)) ≈ gf(x) .* TRAPPED_BOX.scale rtol = 1e-14
    # the starts under the cap keep their waist, focus and density, with the duration at the same
    # place between the floor and 12 as between 4 and 9 in the box
    st = starts_of(c)
    @test keys(st) == keys(STARTS) && starts_of(b) === STARTS
    for k in keys(STARTS)
        @test st[k][2:end] ≈ STARTS[k][2:end] rtol = 1e-15
        @test z_of(c, st[k])[1] ≈ (STARTS[k][1] - 4) / 5 rtol = 1e-12
    end
    # L-BFGS-B under the cap on J = n/τ, which wants the shortest pulse into the densest plasma: it
    # ends on the floor at the density's bound, and no point it asks for breaks the floor
    fake(free) = Dict{String,Any}("J" => free[4] / free[1], "gradient" => [-free[4] / free[1]^2, 0.0, 0.0, 1 / free[1]], "a0" => 1.9)
    dir = mktempdir()
    store = PointStore(joinpath(dir, "points"); run = fake)
    r = optimize_start(store, st.base, joinpath(dir, "base"); maxiters = 30, pgtol = 1e-10, coords = c)
    @test r["u"][1] == 0.0 && r["free"][4] == 0.10
    @test isapprox(r["free"][1], duration_floor(cap, 0.10); rtol = 1e-14) && isapprox(r["J"], 0.10 / duration_floor(cap, 0.10); rtol = 1e-14)
    @test r["free"][2:3] ≈ st.base[2:3] rtol = 1e-15                     # no gradient in waist and focus
    @test all(q -> q["free"][1] >= duration_floor(cap, q["free"][4]) * (1 - 1e-14), r["calls"])
    @test r["projected_gradient"] == 0.0 && on_bounds(c, r["u"]) == "τ: floor→12=lo, density=hi"
    @test all(p -> haskey(store.records, p["key"]), r["path"])          # the path's keys are the records' points
    # the scan over the cap's coordinates: the pair grids span the duration's place, every job keeps the floor and 12
    jobs = scan_jobs(dir, Dict{String,Any}("n" => 5); coords = c)
    @test !isempty(jobs) && all(j -> duration_floor(cap, j[2][4]) * (1 - 1e-12) <= j[2][1] <= 12 * (1 + 1e-12), jobs)
    p12 = filter(j -> j[3]["kind"] == "pair" && j[3]["a"] == 1, jobs)
    @test sort(unique(round(shown(c, j[2])[1]; digits = 10) for j in p12)) ≈ collect(range(0.0, 1.0; length = 5)) atol = 1e-12
    # the window for the longest pulse: delay + 6τ = 54 + 72 must stay within its start 108 + 2πk
    @test window_extend(b) == 0 && window_extend(c) == 3 && window_extend(Coords(cap; duration_hi = 11.5)) == 3
    @test 54 + 6 * 12 <= BCASE.start + 2π * 3 && 54 + 6 * 12 > BCASE.start + 2π * 2
    @test run_read(0) == (;) && run_read(3).fade == TCASE.fade .+ 6π
    @test run_geometry(Dict{String,Any}("cap" => "cetal25")) == (; extend = 3, read = run_read(3))
    long = trapped_problem(12.0, 10.0, 80.0, TCASE.x0[4], 0.05; ppl = 8, t_end = 20.0)
    @test_throws ArgumentError start_of(long; ppl = 8, ny = 1)            # the case's window refuses the longest pulse
    @test start_of(trapped_problem(12.0, 10.0, 80.0, TCASE.x0[4], 0.05; ppl = 8, t_end = 20.0, extend = 3); ppl = 8, ny = 1)[1] isa EMState
    # the longer window leaves the run as it is: the same J at the base point with the weight moved back
    kw = (; ppl = 8, ny = 1, t_end = 190.0, capacity = 60_000, energy = budget(2.2), gather = CompensatedGather(), grad = false)
    J0 = run_point(STARTS.base; kw..., read = (; fade = BCASE.fade, rise = (0.5, 3.0)))["J"]
    J3 = run_point(STARTS.base; kw..., extend = 3, read = (; fade = BCASE.fade .+ 6π, rise = (0.5, 3.0)))["J"]
    @test J0 > 0.2 && isapprox(J3, J0; rtol = 1e-7)                     # 2·10⁻⁹ apart here (the plasma's own injection carries roundoff), 10⁻¹² at λ₀/16 to t = 300
end

@testset "T-977 the objective's variants: born alone, and any origin with the plasma's share capped; one start from another run's optimum [WP-068]" begin
    p = (; born = 0.2, plasma = 0.3)
    @test objective_value(p, "any", 0.05) == 0.5 && objective_value(p, "born", 0.05) == 0.2
    @test objective_value(p, "plasmacap", 0.05) ≈ 0.2 + 0.05 * tanh(6.0) rtol = 1e-15
    @test objective_value((; born = 0.2, plasma = 10.0), "plasmacap", 0.05) < 0.2 + 0.05 + 1e-12   # the plasma adds at most κ
    @test objective_value((; born = 0.2, plasma = 0.0), "plasmacap", 0.05) == 0.2
    @test_throws ArgumentError objective_value(p, "plasma", 0.05)
    # small plasma counts in full, and the derivative in the plasma part is sech² ∈ (0, 1]
    d(x) = ForwardDiff.derivative(q -> objective_value((; born = 0.1, plasma = q), "plasmacap", 0.05), x)
    @test d(0.0) == 1.0 && 0 < d(0.1) < 1 && d(1.0) < 1e-6
    @test objective_value((; born = 0.2f0, plasma = 0.3f0), "plasmacap", 0.05) isa Float32
    @test OPT_DEFAULTS.objective == "any" && OPT_DEFAULTS.kappa == 0.05 && OPT_DEFAULTS.from == ""
    @test run_setup(Dict{String,Any}()) == (; extend = 0, read = (;), objective = "any", kappa = 0.05)
    @test run_setup(Dict{String,Any}("cap" => "cetal25", "objective" => "born", "kappa" => 0.1)) == (; extend = 3, read = run_read(3), objective = "born", kappa = 0.1)
    @test point_config(merge(parse_options(String[], OPT_DEFAULTS)[1], Dict("backend" => "cpu", "objective" => "plasmacap"))).objective == "plasmacap"
    # one run at λ₀/8: the born objective is the born part of the same run, the default is their sum
    kw = (; ppl = 8, ny = 1, t_end = 190.0, capacity = 60_000, energy = budget(2.2), gather = CompensatedGather(), grad = false,
          read = (; fade = BCASE.fade, rise = (0.5, 3.0)))
    ra = run_point(STARTS.base; kw...)
    rb = run_point(STARTS.base; kw..., objective = "born")
    rc = run_point(STARTS.base; kw..., objective = "plasmacap", kappa = 0.05)
    @test ra["J_plasma"] > 0.1 && rb["J"] == ra["J_born"] && rb["J_plasma"] == ra["J_plasma"] && rb["objective"] == "born"
    @test rc["J"] ≈ ra["J_born"] + 0.05 * tanh(ra["J_plasma"] / 0.05) rtol = 1e-12
    # `from`: one start at the best optimum of another run directory, which the manifest records
    dir = mktempdir()
    for (nm, J, free) in (("a", 0.3, [8.0, 7.0, 110.0, 0.1]), ("b", 0.5, [8.5, 6.5, 109.0, 0.1]))
        mkpath(joinpath(dir, nm))
        open(io -> JSON.print(io, Dict("J" => J, "free" => free)), joinpath(dir, nm, "result.json"), "w")
    end
    @test start_points(Dict{String,Any}("from" => dir)) == (; from = [8.5, 6.5, 109.0, 0.1])
    @test start_points(Dict{String,Any}("from" => "", "cap" => "cetal25")) == starts_of(coords_of(Dict{String,Any}("cap" => "cetal25")))
    @test_throws ErrorException start_points(Dict{String,Any}("from" => mktempdir()))
end
