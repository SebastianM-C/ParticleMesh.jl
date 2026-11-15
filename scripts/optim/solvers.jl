# WP-058: the WP-045 problem (wake2d_case.jl through problem.jl) given to two solvers of
# Optimization.jl from WP-045's two starts: the bound-constrained L-BFGS-B, and MadNLP's
# interior-point method with a limited-memory Hessian. What each costs is the number of runs of the
# case, counted by the problem's timer: plain runs for values, runs with duals for gradients.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/solvers.jl
#
# Writes scripts/runs/optim-solvers-<stamp>/: manifest.toml, results.json. The run asserts that
# every solve succeeds and that all of them end at the same value.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "problem.jl"))
using OptimizationMadNLP: MadNLPOptimizer
using MadNLP: MadNLP

# The tolerances are each solver's own: L-BFGS-B's `reltol` is on the largest component of the
# projected gradient, MadNLP's `abstol` on its scaled optimality error.
const SOLVERS = ("L-BFGS-B" => (LBFGSB(), (; maxiters = 25, reltol = 1e-4)),
                 "MadNLP, limited memory" => (MadNLPOptimizer(; hessian_approximation = MadNLP.CompactLBFGS), (; maxiters = 50, abstol = 1e-4)))
const STARTS = ("start A" => (4.5, 70.0), "start B" => (8.5, 125.0))

function main()
    dir = run_dir("solvers", @__FILE__; prefix = "optim", work_package = "WP-058",
                  extra_inputs = [joinpath(@__DIR__, "wake2d_case.jl"), joinpath(@__DIR__, "problem.jl")],
                  params = Dict{String,Any}("case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE)),
                                            "solvers" => Dict(name => Dict(string(k) => v for (k, v) in pairs(kw)) for (name, (_, kw)) in SOLVERS)))
    rows = Dict{String,Any}[]
    for (start, x0) in STARTS, (name, (alg, kw)) in SOLVERS
        println("$name from $start")
        prob = case_problem(wake_amplitude, WAKE_BOX, x0)
        sol = solve(prob, alg; kw...)
        plain, duals = runs(prob.p.timer, "plain run"), runs(prob.p.timer, "run with duals")
        push!(rows, Dict("solver" => name, "start" => start, "x0" => collect(x0), "phys" => sol.u .* WAKE_BOX.scale, "J" => sol.objective,
                         "retcode" => string(sol.retcode), "iterations" => sol.stats.iterations, "plain_runs" => plain.n,
                         "plain_seconds" => plain.seconds, "dual_runs" => duals.n, "dual_seconds" => duals.seconds, "seconds" => sol.stats.time))
        println("$name from $start: $(sol.retcode), (τ, focus) = $(sol.u .* WAKE_BOX.scale), J = $(sol.objective), ",
                "$(plain.n) plain runs and $(duals.n) runs with duals, $(round(sol.stats.time; digits = 1)) s")
        flush(stdout)
    end
    Js = [r["J"] for r in rows]
    checks = [Dict("check" => "every solve succeeds", "pass" => all(r -> r["retcode"] == "Success", rows), "detail" => join((r["retcode"] for r in rows), ", ")),
              Dict("check" => "every solve ends at the same value", "pass" => maximum(Js) - minimum(Js) <= 1e-6 * maximum(Js), "detail" => "J = $Js")]
    foreach(c -> println(c["pass"] ? "PASS " : "FAIL ", c["check"], ": ", c["detail"]), checks)
    open(io -> JSON.print(io, Dict("runs" => rows, "checks" => checks), 2), joinpath(dir, "results.json"), "w")
    println("done: $dir")
    failed = [c["check"] for c in checks if !c["pass"]]
    isempty(failed) || error("WP-058 checks failed: $(join(failed, "; ")) (results in $dir)")
    return dir
end

main()
