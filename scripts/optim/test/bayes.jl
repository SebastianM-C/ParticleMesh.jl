# Tests of the WP-048 Bayesian optimizer on analytic objectives:
#   julia +1.13 --project=scripts scripts/optim/test/bayes.jl
using Test
include(joinpath(@__DIR__, "..", "bayes.jl"))

@testset "T-620 the Latin hypercube puts one point in each stratum and repeats bit for bit [WP-048]" begin
    for (n, d, seed) in ((5, 2, 1), (17, 3, 42))
        P = latin_hypercube(n, d, seed)
        @test length(P) == n && all(p -> length(p) == d && all(0 .<= p .< 1), P)
        for j in 1:d
            @test sort([floor(Int, p[j] * n) for p in P]) == 0:(n - 1)
        end
        @test latin_hypercube(n, d, seed) == P
        @test latin_hypercube(n, d, seed + 1) != P
    end
end

@testset "T-621 the GP interpolates its data, fits a known length scale, and EI vanishes at the data [WP-048]" begin
    # A smooth 1D function sampled densely: the posterior mean reproduces the samples, the posterior
    # standard deviation is small there, and the fitted length scale is of the order of the
    # function's scale (sin(2πx/0.5) varies on 0.5/2π ≈ 0.08 … 0.5, so the bracket 0.03 to 1).
    X = [[x] for x in range(0, 1; length = 15)]
    y = [sin(4π * x[1]) for x in X]
    fit = fit_gp(X, y)
    C = cholesky(Symmetric(gram(X, fit.θ)))
    α = C \ y
    for (x, yi) in zip(X, y)
        μ, s = posterior(x, X, fit.θ, C, α)
        @test μ ≈ yi atol = 1e-2
        @test s < 0.05
    end
    @test 0.03 < exp(fit.θ[1]) < 1.0
    μ, s = posterior([0.53], X, fit.θ, C, α)
    @test μ ≈ sin(4π * 0.53) atol = 0.05
    # Expected improvement is non-negative, ≈ 0 where the mean is far below the best with small spread,
    # and equals the mean's excess when the spread vanishes above the best.
    @test expected_improvement(-3.0, 1e-3, 0.0, 0.0) ≈ 0 atol = 1e-12
    @test expected_improvement(0.5, 1e-9, 0.0, 0.0) ≈ 0.5
    @test expected_improvement(0.0, 1.0, 0.0, 0.0) ≈ 1 / sqrt(2π)
    # Its ForwardDiff gradient matches central differences.
    f(u) = expected_improvement(posterior(u, X, fit.θ, C, α)..., maximum(y) - 0.3, 0.01)
    g = ForwardDiff.gradient(f, [0.37])
    h = 1e-6
    @test g[1] ≈ (f([0.37 + h]) - f([0.37 - h])) / 2h rtol = 1e-5
end

# The Branin function on [−5, 10] × [0, 15]; its minimum 0.397887… is reached at three points.
branin(x) = (x[2] - 5.1 / (4π^2) * x[1]^2 + 5 / π * x[1] - 6)^2 + 10 * (1 - 1 / (8π)) * cos(x[1]) + 10
const BRANIN_MIN = 5 / (4π)   # = 0.3978873577…

@testset "T-622 Bayesian optimization reaches the Branin minimum within 40 evaluations and repeats bit for bit [WP-048]" begin
    gaps = Float64[]
    for seed in 1:3
        r = bayesopt(x -> -branin(x), [-5.0, 0.0], [10.0, 15.0]; budget = 40, seed)
        @test length(r.y) == 40
        push!(gaps, -r.best_y - BRANIN_MIN)
        @test -r.best_y - BRANIN_MIN < 2e-2          # measured 0.0064, 0.0069, 0.0105 for seeds 1 to 3; Branin spans about 300 on the box
        seed == 1 && @test bayesopt(x -> -branin(x), [-5.0, 0.0], [10.0, 15.0]; budget = 12, seed).y == r.y[1:12]
    end
    println("Branin gaps after 40 evaluations, seeds 1 to 3: ", gaps)
end
