# WP-058: a case as an Optimization.jl problem.
#
# A case is an objective J(x₁, …, x_n) of physical parameters in a box. `case_problem` makes it an
# `OptimizationProblem` in the scaled coordinates z = x ./ scale, so that every parameter is O(1)
# (A1), to be maximized (`MaxSense`) with ForwardDiff gradients (`AutoForwardDiff`). Any solver of
# Optimization.jl that takes bounds solves it:
#
#     prob = case_problem(wake_amplitude, WAKE_BOX, (4.5, 70.0))
#     rec = Record(prob, LBFGSB())
#     sol = solve(prob, LBFGSB(); callback = rec, maxiters = 25, reltol = 1e-4)
#     sol.u .* WAKE_BOX.scale        # the optimum in physical units; sol.objective is J there
#
# `bayes.jl` adds a Bayesian optimizer as one more solver of the same problem.

using Optimization: Optimization, OptimizationFunction, OptimizationProblem, AutoForwardDiff, MaxSense, ReturnCode, solve
using OptimizationBase: OptimizationBase
using OptimizationLBFGSB: LBFGSB
using DifferentiationInterface: DifferentiationInterface
using ForwardDiff: ForwardDiff
using TimerOutputs: TimerOutputs, TimerOutput, @timeit

"The section of a problem's timer that a run at `z` goes to: a run with duals carries the gradient."
run_kind(z) = eltype(z) <: ForwardDiff.Dual ? "run with duals" : "plain run"

"""
    case_problem(J, box, x0; timer = TimerOutput()) -> OptimizationProblem

The maximization of the case objective `J(x...)` over `box`, a named tuple `(; lo, hi, scale)` in
physical units, from the start `x0` (physical units). The problem's unknown is z = x ./ box.scale.
Its parameters are `p = (; scale, timer)`: `prob.f(z, prob.p)` is J at `z`, and `p.timer` times
every run of `J` in the sections "plain run" and "run with duals".
"""
function case_problem(J, box, x0; timer = TimerOutput())
    f(z, p) = @timeit p.timer run_kind(z) J((z .* p.scale)...)
    return OptimizationProblem(OptimizationFunction(f, AutoForwardDiff()), x0 ./ box.scale, (; box.scale, timer);
                               lb = box.lo ./ box.scale, ub = box.hi ./ box.scale, sense = MaxSense)
end

"""
    value_gradient(prob, z) -> (J, ∇J)

The objective of `prob` at `z` and its gradient in z, from one run with duals.
"""
value_gradient(prob, z) = DifferentiationInterface.value_and_gradient(w -> prob.f(w, prob.p), AutoForwardDiff(), z)

"The number of runs of `kind` (\"plain run\" or \"run with duals\") in `timer`, and their mean seconds (`nothing` without any)."
function runs(timer::TimerOutput, kind)
    haskey(timer, kind) || return (; n = 0, seconds = nothing)
    n = TimerOutputs.ncalls(timer[kind])
    return (; n, seconds = TimerOutputs.time(timer[kind]) / 1e9 / n)
end

"""
    Record(prob, alg)

A callback for `solve(prob, alg; callback)` that keeps what the solver reports at each call: the
point `u`, the objective there and the wall-clock `seconds` since the record was made. L-BFGS-B and
`BayesianEI` call it once per point they evaluate. With `verbose` it prints each call.

A solver without a sense of its own minimizes −J for a `MaxSense` problem, and Optimization.jl
hands that −J to the callback (#440); the record turns it back, so `objective` is always J.
"""
struct Record
    u::Vector{Vector{Float64}}
    objective::Vector{Float64}
    seconds::Vector{Float64}
    start::Float64
    negated::Bool
    verbose::Bool
end
Record(prob, alg; verbose = false) =
    Record(Vector{Float64}[], Float64[], Float64[], time(), prob.sense === MaxSense && !OptimizationBase.supports_sense(alg), verbose)

function (r::Record)(state, args...)
    push!(r.u, copy(state.u))
    push!(r.objective, r.negated ? -state.objective : state.objective)
    push!(r.seconds, time() - r.start)
    r.verbose && (println("  point ", length(r.u), ": u = ", round.(r.u[end]; digits = 5), "  objective = ", r.objective[end], "  ", round(r.seconds[end]; digits = 1), " s"); flush(stdout))
    return false
end

"""
    path(rec, scale) -> Vector of Dict

The record's points for a results file: the scaled point, the physical one, and the objective.
"""
path(rec::Record, scale) = [Dict("z" => u, "phys" => u .* scale, "J" => J) for (u, J) in zip(rec.u, rec.objective)]
