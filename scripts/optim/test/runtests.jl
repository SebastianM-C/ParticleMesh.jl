# Tests of the WP-045 maximizer on analytic objectives: julia +1.13 --project=scripts scripts/optim/test/runtests.jl
using Test
include(joinpath(@__DIR__, "..", "maximize.jl"))

@testset "T-600 the box-constrained maximizer finds interior and bound maxima of analytic objectives [WP-045]" begin
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
