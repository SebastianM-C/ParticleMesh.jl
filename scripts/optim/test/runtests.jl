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
