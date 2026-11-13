# WP-058: the layer between the cases of scripts/optim and Optimization.jl.
#
# A case is an objective J(x₁, …, x_n) of physical parameters in a box. The optimizers work in scaled
# coordinates z = x ./ scale, so that every parameter is O(1) (A1). `Objective` runs the case and
# records every run; `maximize` hands −J to a bound-constrained solver of Optimization.jl (L-BFGS-B by
# default), with the gradient from one ForwardDiff run per point. `bayes.jl` takes the same objective.

using Optimization: OptimizationFunction, OptimizationProblem, ReturnCode, solve
using OptimizationLBFGSB: LBFGSB
using ForwardDiff: ForwardDiff, Dual, value, partials

"""
    ScaledBox(lo, hi, scale)

The box `lo ≤ x ≤ hi` of the physical parameters, and the scale of each: the optimizers see
`z = x ./ scale` in the box `zlo(b) ≤ z ≤ zhi(b)`.
"""
struct ScaledBox
    lo::Vector{Float64}
    hi::Vector{Float64}
    scale::Vector{Float64}
    function ScaledBox(lo, hi, scale)
        length(lo) == length(hi) == length(scale) || throw(DimensionMismatch("lo, hi and scale need one entry per parameter"))
        all(lo .< hi) || throw(ArgumentError("every lower bound must lie below its upper bound: lo = $lo, hi = $hi"))
        all(>(0), scale) || throw(ArgumentError("every scale must be positive: $scale"))
        return new(collect(Float64, lo), collect(Float64, hi), collect(Float64, scale))
    end
end

zlo(b::ScaledBox) = b.lo ./ b.scale
zhi(b::ScaledBox) = b.hi ./ b.scale
"The physical parameters at `z`, as a tuple; the element type follows `z` (duals stay duals)."
phys(b::ScaledBox, z) = ntuple(i -> z[i] * b.scale[i], length(b.scale))
"The scaled coordinates of physical parameters `x`."
scaled(b::ScaledBox, x) = collect(Float64, x) ./ b.scale

"""
    value_gradient(J, b::ScaledBox, z) -> (J, ∇J)

`J` at `z` and its gradient in `z`, from one call of `J` with ForwardDiff duals, one partial per
parameter.
"""
function value_gradient(J, b::ScaledBox, z)
    v = J(dual_point(b, z)...)
    return value(v), collect(Float64, partials(v))
end

"""
    dual_point(b::ScaledBox, z) -> tuple of duals

The physical parameters at `z` as ForwardDiff duals whose partials are the derivatives in `z`, one
per parameter: a function of them carries its gradient in `z`.
"""
dual_point(b::ScaledBox, z) = _dual_point(b, z, Val(length(b.scale)))
function _dual_point(b, z, ::Val{N}) where {N}
    length(z) == N || throw(DimensionMismatch("z has $(length(z)) entries, the box $N parameters"))
    return phys(b, ntuple(i -> Dual{Nothing}(Float64(z[i]), ntuple(j -> Float64(i == j), Val(N))...), Val(N)))
end

"""
    Objective(J, box; label = "J", verbose = true)

The objective `J(x...)` of a case on `box`, as the optimizers call it, with a record of every run in
`log`: `o(z)` runs `J` once and returns its value; `value_gradient(o, z)` runs it once with duals
and returns the value and the gradient in `z`. Each entry of `log` holds the kind ("value" or
"gradient"), `z`, the physical parameters, J, the gradient (for "gradient") and the run's seconds.
"""
struct Objective{F}
    J::F
    box::ScaledBox
    log::Vector{Dict{String,Any}}
    label::String
    verbose::Bool
end
Objective(J, box::ScaledBox; label = "J", verbose = true) = Objective(J, box, Dict{String,Any}[], label, verbose)

phys(o::Objective, z) = phys(o.box, z)

function (o::Objective)(z::AbstractVector{<:Real})
    t = @elapsed v = o.J(phys(o.box, z)...)
    f = Float64(v)
    push!(o.log, Dict{String,Any}("kind" => "value", "z" => collect(Float64, z), "phys" => collect(phys(o.box, z)), "J" => f, "seconds" => t))
    o.verbose && (println("  value  x = ", round.(phys(o.box, z); digits = 4), "  ", o.label, " = ", f, "  ", round(t; digits = 1), " s"); flush(stdout))
    return f
end

function value_gradient(o::Objective, z::AbstractVector{<:Real})
    t = @elapsed (f, g) = value_gradient(o.J, o.box, z)
    push!(o.log, Dict{String,Any}("kind" => "gradient", "z" => collect(Float64, z), "phys" => collect(phys(o.box, z)), "J" => f, "grad" => g, "seconds" => t))
    o.verbose && (println("  grad   x = ", round.(phys(o.box, z); digits = 4), "  ", o.label, " = ", f, "  ∇ = ", g, "  ", round(t; digits = 1), " s"); flush(stdout))
    return f, g
end

"The gradient with the components held at a bound (pointing out of the box) set to zero."
projected_gradient(z, g, lo, hi) = [(z[i] <= lo[i] && g[i] < 0) || (z[i] >= hi[i] && g[i] > 0) ? 0.0 : g[i] for i in eachindex(z)]

"""
    maximize_box(fg, z0, lo, hi; maxiters = 30, gtol = 1e-4, solver = LBFGSB()) -> (; z, f, g, path, evals, reason, retcode)

Maximize `f` over the box `lo ≤ z ≤ hi` with `fg(z) -> (f, ∇f)`, as the minimization of `−f` by
`solver` through Optimization.jl. The solver asks for the value and then the gradient at each
point; both come from the one `fg` call of that point, so each point costs one `fg` call. `gtol`
is L-BFGS-B's tolerance on the largest component of the projected gradient. `reason` is "gtol"
when the solution's projected gradient is below it, "ftol" for L-BFGS-B's other successful stop
(an iteration lowered `−f` by less than about 2e-9 relative), "maxiter", or the return code.
`z`, `f` and `g` are the solution, its value and its gradient; `path` holds `(z, f, g)` of every
point evaluated, in order (line-search trials included), and `evals` their number.
"""
function maximize_box(fg, z0, lo, hi; maxiters = 30, gtol = 1e-4, solver = LBFGSB())
    path = Tuple{Vector{Float64},Float64,Vector{Float64}}[]
    function at(z)
        for p in Iterators.reverse(path)
            p[1] == z && return p
        end
        f, g = fg(collect(Float64, z))
        p = (collect(Float64, z), Float64(f), collect(Float64, g))
        push!(path, p)
        return p
    end
    negf(z, _) = -at(z)[2]
    neggrad!(G, z, _) = (G .= .-at(z)[3]; G)
    lo, hi = collect(Float64, lo), collect(Float64, hi)
    prob = OptimizationProblem(OptimizationFunction(negf; grad = neggrad!), clamp.(collect(Float64, z0), lo, hi); lb = lo, ub = hi)
    sol = solve(prob, solver; maxiters, reltol = gtol)
    _, f, g = at(sol.u)
    # The wrapper turns L-BFGS-B's stop message into the return code and does not keep it, so the
    # reason is read from the result: the projected gradient below `gtol` is "gtol", another
    # successful stop is L-BFGS-B's stop on the decrease of the objective ("ftol").
    reason = if maximum(abs, projected_gradient(sol.u, g, lo, hi)) <= gtol
        "gtol"
    elseif sol.retcode == ReturnCode.Success
        "ftol"
    elseif sol.retcode == ReturnCode.MaxIters
        "maxiter"
    else
        string(sol.retcode)
    end
    return (; z = collect(Float64, sol.u), f, g, path, evals = length(path), reason, retcode = sol.retcode)
end

"""
    maximize(o::Objective, z0; maxiters = 30, gtol = 1e-4, solver = LBFGSB())

`maximize_box` of the objective `o` over its box from `z0` (scaled coordinates), with one run with
duals per point; every run lands in `o.log`.
"""
maximize(o::Objective, z0; kw...) = maximize_box(z -> value_gradient(o, z), z0, zlo(o.box), zhi(o.box); kw...)
