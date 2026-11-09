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
    # A smooth non-quadratic objective (a Gaussian bump on a slope), maximum inside.
    fb(z) = (e = exp(-sum(abs2, z .- [0.4, 0.6]) / 0.1); (e + 0.01z[1], e .* (-2 .* (z .- [0.4, 0.6]) ./ 0.1) .+ [0.01, 0.0]))
    r = maximize_box(fb, [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]; gtol = 1e-9, xtol = 1e-12, maxiter = 100)
    @test maximum(abs, r.g) < 1e-8
end
