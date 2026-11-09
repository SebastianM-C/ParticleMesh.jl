# WP-048: a Bayesian-optimization baseline for the WP-045 problem. A Gaussian process with a
# Matérn-5/2 kernel (one length scale per parameter) on the unit box, hyperparameters by maximum
# marginal likelihood, and expected improvement as the acquisition. Written out here rather than
# taken from a package: it is short, it adds no dependency to scripts/, and every choice is visible.

import LinearAlgebra
using LinearAlgebra: Symmetric, cholesky, diag, dot, norm, I
using SpecialFunctions: erfc
using ForwardDiff: ForwardDiff
isdefined(@__MODULE__, :maximize_box) || include(joinpath(@__DIR__, "maximize.jl"))

# Counter-based uniforms in [0, 1) (splitmix64 of seed and counter), so a run is reproducible
# bit for bit without a random-number state.
function splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end
uniform01(seed, stream, i) = (splitmix64(splitmix64(UInt64(seed)) ⊻ splitmix64(UInt64(stream) << 32 + UInt64(i))) >> 11) * 2.0^-53

"""
    latin_hypercube(n, d, seed) -> Vector{Vector{Float64}}

`n` points in the unit box with exactly one point in each of the `n` strata of every coordinate.
"""
function latin_hypercube(n, d, seed)
    cols = map(1:d) do j
        perm = sortperm([uniform01(seed, j, i) for i in 1:n])
        [(perm[i] - 1 + uniform01(seed, d + j, i)) / n for i in 1:n]
    end
    return [[cols[j][i] for j in 1:d] for i in 1:n]
end

# The kernel. θ = (log ℓ₁, …, log ℓ_d, log σ_f², log σ_n²). The tiny shift under the square root keeps
# the derivative finite at r = 0 without changing a value at Float64 precision.
function matern52(x, y, θ)
    d = length(x)
    r2 = sum(((x[i] - y[i]) / exp(θ[i]))^2 for i in 1:d)
    r = sqrt(r2 + 1e-30)
    s5 = sqrt(5.0) * r
    return exp(θ[d + 1]) * (1 + s5 + 5r2 / 3) * exp(-s5)
end

gram(X, θ) = [matern52(X[i], X[j], θ) for i in eachindex(X), j in eachindex(X)] + (exp(θ[end]) + 1e-10) * I

"The negative log marginal likelihood of standardized data `y` at points `X` (zero prior mean)."
function gp_nll(θ, X, y)
    C = cholesky(Symmetric(gram(X, θ)))
    α = C \ y
    return dot(y, α) / 2 + sum(log, diag(C.U)) + length(y) * log(2π) / 2
end

"Bounds of θ: length scales 0.02 to 5 box widths, signal variance 0.01 to 100, noise variance 1e-8 to 1e-1 (standardized units)."
θbounds(d) = ([fill(log(0.02), d); log(0.01); log(1e-8)], [fill(log(5.0), d); log(100.0); log(1e-1)])

"""
    fit_gp(X, y; starts) -> (; θ, nll)

Maximum marginal likelihood over θ in `θbounds`, by the WP-045 maximizer with ForwardDiff gradients
from a few fixed starts (and `starts`, such as the previous fit).
"""
function fit_gp(X, y; starts = Vector{Float64}[])
    d = length(X[1])
    lo, hi = θbounds(d)
    fixed = [[fill(log(ℓ), d); 0.0; log(1e-4)] for ℓ in (0.1, 0.3, 1.0)]
    best = (; θ = fixed[1], nll = Inf)
    for θ0 in [fixed; starts]
        function fg(θ)
            try
                v = -gp_nll(θ, X, y)
                return v, ForwardDiff.gradient(t -> -gp_nll(t, X, y), θ)
            catch err
                err isa LinearAlgebra.PosDefException || rethrow()
                return -Inf, zero(θ)
            end
        end
        r = maximize_box(fg, clamp.(θ0, lo, hi), lo, hi; maxiter = 60, gtol = 1e-5, xtol = 1e-6)
        -r.f < best.nll && (best = (; θ = r.z, nll = -r.f))
    end
    return best
end

"The GP posterior at `x`: mean and standard deviation, from the factor `C` and weights `α = C \\ y`."
function posterior(x, X, θ, C, α)
    k = [matern52(x, xi, θ) for xi in X]
    μ = dot(k, α)
    v = C.L \ k
    s2 = exp(θ[length(x) + 1]) - dot(v, v)
    return μ, sqrt(max(s2, 1e-18))
end

normcdf(z) = erfc(-z / sqrt(2.0)) / 2
normpdf(z) = exp(-z^2 / 2) / sqrt(2π)

"Expected improvement over `ybest` (maximization) with exploration margin `ξ`."
function expected_improvement(μ, s, ybest, ξ)
    δ = μ - ybest - ξ
    z = δ / s
    return δ * normcdf(z) + s * normpdf(z)
end

"""
    propose(X, y, θ, seed, iter; ξ, ncand, nrefine) -> u

The next point in the unit box: expected improvement on `ncand` counter-based candidates, then the
WP-045 maximizer with ForwardDiff gradients from the `nrefine` best of them. Points closer than
1e-6 to an evaluated one are skipped.
"""
function propose(X, y, θ, seed, iter; ξ = 0.01, ncand = 2048, nrefine = 5)
    d = length(X[1])
    C = cholesky(Symmetric(gram(X, θ)))
    α = C \ y
    ybest = maximum(y)
    ei(u) = expected_improvement(posterior(u, X, θ, C, α)..., ybest, ξ)
    cands = [[uniform01(seed, 1000 + iter, (i - 1) * d + j) for j in 1:d] for i in 1:ncand]
    vals = ei.(cands)
    best_u, best_v = cands[argmax(vals)], maximum(vals)
    for i in partialsortperm(vals, 1:min(nrefine, ncand); rev = true)
        r = maximize_box(u -> (ei(u), ForwardDiff.gradient(ei, u)), cands[i], zeros(d), ones(d); maxiter = 40, gtol = 1e-10, xtol = 1e-8)
        far = minimum(xi -> norm(r.z .- xi), X) > 1e-6
        far && r.f > best_v && ((best_u, best_v) = (r.z, r.f))
    end
    return best_u, best_v
end

"""
    bayesopt(f, lo, hi; budget, n_init, seed, ξ, verbose) -> (; x, y, best_x, best_y, θ, ei)

Maximize `f` over the box `lo ≤ x ≤ hi` with `budget` evaluations: a Latin hypercube of `n_init`
points, then one expected-improvement point at a time, refitting the GP's hyperparameters after
every evaluation. Values are standardized (mean 0, standard deviation 1) before each fit. `x` and
`y` list every evaluation in order; `ei` holds the acquisition's maximum at each proposal.
"""
function bayesopt(f, lo, hi; budget = 30, n_init = 2length(lo) + 1, seed = 1, ξ = 0.01, verbose = false)
    d = length(lo)
    to_box(u) = lo .+ u .* (hi .- lo)
    U = latin_hypercube(n_init, d, seed)
    y = Float64[]
    for u in U
        push!(y, f(to_box(u)))
    end
    θs, eis = Vector{Float64}[], Float64[]
    while length(y) < budget
        m = sum(y) / length(y)
        s = sqrt(sum(abs2, y .- m) / max(length(y) - 1, 1))
        s = s > 0 ? s : 1.0
        ys = (y .- m) ./ s
        fit = fit_gp(U, ys; starts = isempty(θs) ? Vector{Float64}[] : [θs[end]])
        push!(θs, fit.θ)
        u, v = propose(U, ys, fit.θ, seed, length(y); ξ)
        push!(eis, v)
        push!(U, u)
        push!(y, f(to_box(u)))
        verbose && println("  BO $(length(y)): x = ", round.(to_box(u); digits = 4), "  f = ", y[end], "  EI = ", v, "  ℓ = ", round.(exp.(fit.θ[1:d]); digits = 3))
    end
    ib = argmax(y)
    return (; x = to_box.(U), y, best_x = to_box(U[ib]), best_y = y[ib], θ = θs, ei = eis)
end

"""
    reach(best_so_far, cum_seconds, Jstar, ε) -> (; calls, seconds) or nothing

The first call whose best value so far is within ε·|J*| of `Jstar`, and the summed seconds up to and
including it; `nothing` if no call gets there.
"""
function reach(best_so_far, cum_seconds, Jstar, ε)
    i = findfirst(b -> b >= Jstar - ε * abs(Jstar), best_so_far)
    return i === nothing ? nothing : (; calls = i, seconds = cum_seconds[i])
end
