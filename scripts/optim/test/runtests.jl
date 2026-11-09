# Tests of the WP-045 maximizer and the WP-052 case: julia +1.13 --project=scripts scripts/optim/test/runtests.jl
using Test
include(joinpath(@__DIR__, "..", "maximize.jl"))

@testset "T-600 the box-constrained maximizer finds interior and bound maxima of analytic objectives, with a capped step [WP-045, WP-052]" begin
    # An anisotropic concave quadratic with its maximum inside the box.
    A = [3.0 1.0; 1.0 0.5]
    c = [0.3, 0.7]
    fq(z) = (r = z .- c; (-r' * A * r / 2, -A * r))
    r = maximize_box(fq, [0.9, 0.1], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-10, xtol = 1e-12)
    @test r.z ≈ c atol = 1e-8
    @test r.reason in ("gtol", "xtol")
    @test all(i -> r.path[i + 1][2] >= r.path[i][2], 1:(length(r.path) - 1))   # monotone ascent
    # The same quadratic with its maximum outside the box: the maximizer is on the bound z₁ = 0.5,
    # where the constrained optimum of z₂ is c₂ − A₂₁(0.5 − c₁)/A₂₂.
    r = maximize_box(fq, [0.9, 0.1], [0.5, 0.0], [1.0, 1.0]; gtol = 1e-10, xtol = 1e-12)
    @test r.z[1] == 0.5
    @test r.z[2] ≈ c[2] - A[2, 1] * (0.5 - c[1]) / A[2, 2] atol = 1e-8
    @test r.g[1] < 0                                                              # pointing out of the box
    # The maximum outside the box's upper corner on z₁: held at the upper bound z₁ = 0.2, with the
    # same constrained optimum formula for z₂, and the stop on the projected gradient.
    c2 = [0.3, 0.7]
    r = maximize_box(fq, [0.05, 0.1], [0.0, 0.0], [0.2, 1.0]; gtol = 1e-9, xtol = 0.0, maxiter = 200)
    @test r.z[1] == 0.2
    @test r.z[2] ≈ c2[2] - A[2, 1] * (0.2 - c2[1]) / A[2, 2] atol = 1e-8
    @test r.g[1] > 0
    @test r.reason == "gtol"
    # The bound case above also stops on the projected gradient, not on a small step.
    r = maximize_box(fq, [0.9, 0.1], [0.5, 0.0], [1.0, 1.0]; gtol = 1e-9, xtol = 0.0, maxiter = 200)
    @test r.reason == "gtol" && r.z[1] == 0.5
    # A start outside the box is projected into it first.
    r = maximize_box(fq, [3.0, -2.0], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-10, xtol = 0.0, maxiter = 200)
    @test all(p -> all(0 .<= p[1] .<= 1), r.path)
    @test r.z ≈ c atol = 1e-8
    # A smooth non-quadratic objective (a Gaussian bump on a slope), maximum inside.
    fb(z) = (e = exp(-sum(abs2, z .- [0.4, 0.6]) / 0.1); (e + 0.01z[1], e .* (-2 .* (z .- [0.4, 0.6]) ./ 0.1) .+ [0.01, 0.0]))
    r = maximize_box(fb, [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-9, xtol = 1e-12, maxiter = 100)
    @test maximum(abs, r.g) < 1e-8
    # A capped step (WP-052): no accepted step is longer than maxstep in any coordinate, and the
    # interior maximum is still reached; uncapped, the first step from this start is 0.6 long.
    r = maximize_box(fq, [0.9, 0.1], [-5.0, -5.0], [5.0, 5.0]; gtol = 1e-10, xtol = 1e-12, maxiter = 200, maxstep = 0.05)
    @test all(i -> maximum(abs, r.path[i + 1][1] .- r.path[i][1]) <= 0.05 + 1e-15, 1:(length(r.path) - 1))
    @test r.z ≈ c atol = 1e-8
    r1 = maximize_box(fq, [0.9, 0.1], [-5.0, -5.0], [5.0, 5.0]; maxiter = 1)
    @test maximum(abs, r1.path[2][1] .- r1.path[1][1]) ≈ 0.6 rtol = 1e-12   # the default is no cap (WP-045 relies on it)
end

# Tests of the WP-052 fixed-energy case (witness2d_case.jl): two full runs of the case, about a minute.
include(joinpath(@__DIR__, "..", "witness2d_case.jl"))

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
    st1 = witness_state(5.0, 0.07, 38.0)
    st2 = witness_state(5.0, 0.07, 44.0)
    # lattice_x is the kernel's lattice: the weights it set are the Gaussian of these positions
    for (st, c) in ((st1, 38.0), (st2, 44.0))
        x0 = lattice_x(st)
        g = witness_gains(st)
        @test length(x0) == length(g.w)
        @test g.w ≈ exp.(-((x0 .- c) ./ WCASE.length) .^ 2 ./ 2) rtol = 1e-14
    end
    @test witness_gains(st1).gain == witness_gains(st2).gain            # test particles: the centre does not move them
    @test gain_at_centers(st1, [38.0, 44.0]) ≈ [witness_energy_gain(st1), witness_energy_gain(st2)] rtol = 1e-13
    @test !(witness_energy_gain(st1) ≈ witness_energy_gain(st2))
    # a lost witness is refused, as witness_energy_gain refuses it (last: this changes st1)
    s = ParticleMeshElectromagnetic.witness_frame(st1).species
    status = Array(s.status)
    status[1] = ParticleMeshElectromagnetic.ParticleMeshCore.FREE
    copyto!(s.status, status)
    @test_throws ArgumentError gain_at_centers(st1, [40.0])
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
