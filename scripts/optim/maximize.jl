# A small box-constrained quasi-Newton maximizer for the WP-045 problem: few parameters, an
# objective that costs seconds, gradients from ForwardDiff. No dependency beyond LinearAlgebra.

using LinearAlgebra: I, dot, norm

"""
    maximize_box(fg, z0, lo, hi; maxiter = 30, gtol = 1e-4, xtol = 1e-4, c1 = 1e-4, maxhalvings = 8) -> (; z, f, g, path, evals, reason)

Maximize `f` over the box `lo ≤ z ≤ hi`, with `fg(z) -> (f, ∇f)`. Projected BFGS: the step is the
ascent direction `H·g` restricted to the free coordinates (a coordinate on a bound whose gradient
points out of the box is held), projected onto the box, with Armijo backtracking (halving) on the
projected step. `H` is the inverse-Hessian estimate of the negated objective, updated only when the
curvature condition holds, and reset when the set of held coordinates changes. Stops when the projected gradient's largest component is below `gtol`,
when a step changes `z` by less than `xtol` in every coordinate, or after `maxiter` iterations.
`path` holds `(z, f, g)` of every accepted iterate, `evals` the number of `fg` calls.
"""
function maximize_box(fg, z0, lo, hi; maxiter = 30, gtol = 1e-4, xtol = 1e-4, c1 = 1e-4, maxhalvings = 8)
    n = length(z0)
    proj(z) = clamp.(z, lo, hi)
    held(z, g) = [(z[i] <= lo[i] && g[i] < 0) || (z[i] >= hi[i] && g[i] > 0) for i in 1:n]
    pgrad(z, g) = [held(z, g)[i] ? 0.0 : g[i] for i in 1:n]
    z = proj(collect(float.(z0)))
    f, g = fg(z)
    evals = 1
    H = Matrix{Float64}(I, n, n)
    path = [(copy(z), f, copy(g))]
    reason = "maxiter"
    lastfree = trues(n)
    for _ in 1:maxiter
        if maximum(abs, pgrad(z, g)) < gtol
            reason = "gtol"
            break
        end
        free = .!held(z, g)
        free == lastfree || (H = Matrix{Float64}(I, n, n))     # the active set changed: restart the curvature estimate
        lastfree = free
        d = zeros(n)
        Hf = H[free, free]
        d[free] = Hf * g[free]
        dot(d, g) <= 0 && (H = Matrix{Float64}(I, n, n); d = pgrad(z, g))   # not an ascent direction: reset
        α = 1.0
        accepted = false
        local zn, fn, gn
        for _ in 0:maxhalvings
            zn = proj(z .+ α .* d)
            fn, gn = fg(zn)
            evals += 1
            if fn >= f + c1 * dot(g, zn .- z)
                accepted = true
                break
            end
            α /= 2
        end
        if !accepted
            reason = "line search"
            break
        end
        s, y = zn .- z, -(gn .- g)          # BFGS on −f
        sy = dot(s, y)
        if sy > 1e-12 * norm(s) * norm(y)
            ρ = 1 / sy
            H = (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'
        end
        step = maximum(abs, s)
        z, f, g = zn, fn, gn
        push!(path, (copy(z), f, copy(g)))
        if step < xtol
            reason = "xtol"
            break
        end
    end
    return (; z, f, g, path, evals, reason)
end
