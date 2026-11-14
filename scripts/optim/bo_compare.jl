# WP-048: how many points, and how much CPU time, Bayesian optimization (`BayesianEI`, bayes.jl) and
# the gradient-based L-BFGS-B need to reach the optimum of the WP-045 problem: the wake amplitude of
# the small 2D LWFA case (wake2d_case.jl) over the pulse duration τ and the focus, at fixed a₀ and
# density, in the WP-045 box. Both are solvers of the same Optimization.jl problem (problem.jl).
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/bo_compare.jl [nstarts] [budget]
#
# Every run starts from the same Latin-hypercube seeds: BO from the 5-point design of seed s, L-BFGS-B
# from that design's first point. A point costs BO one plain run, and L-BFGS-B a plain run for the
# value and a run with duals for the gradient. The solvers' callback records every point with its
# value and the wall time so far (`Record`), and the problem's timer the runs of each kind. The
# reference optimum J* is the largest value any run evaluated; a run "reaches" a tolerance ε at the
# first point whose best value so far is within ε·J* of J*.
#
# Writes scripts/runs/optim-bo_compare-<stamp>/: manifest.toml, results.json, bo_compare.png + .json.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "problem.jl"))
include(joinpath(@__DIR__, "bayes.jl"))

# BO's exploration margins (standardized units): 0, the default, and 0.01, a common default elsewhere.
const XIS = (0.0, 0.01)
# The WP-045 box and scaling (`WAKE_BOX`): z = (τ/6, focus/90), τ ∈ [4, 9], focus ∈ [60, 130].
const SCALE = WAKE_BOX.scale

"""
The WP-045 problem from `x0` solved by `alg`: one entry of the results, with every point the solver
reported, its value and the seconds up to it, and the number and mean seconds of the runs of each kind.
"""
function compare_run(method, alg, x0; seed, kw...)
    prob = case_problem(wake_amplitude, WAKE_BOX, x0)
    rec = Record(prob, alg; verbose = true)
    sol = solve(prob, alg; callback = rec, kw...)
    plain, duals = runs(prob.p.timer, "plain run"), runs(prob.p.timer, "run with duals")
    return Dict{String,Any}("method" => method, "seed" => seed, "z" => rec.u, "J" => rec.objective, "cum_seconds" => rec.seconds,
                            "best_z" => sol.u, "best_J" => sol.objective, "retcode" => string(sol.retcode),
                            "plain_runs" => plain.n, "plain_seconds" => plain.seconds, "dual_runs" => duals.n, "dual_seconds" => duals.seconds), sol
end

function main(; nstarts = 3, budget = 30)
    dir = run_dir("bo_compare", @__FILE__; prefix = "optim", work_package = "WP-048",
                  extra_inputs = [joinpath(@__DIR__, f) for f in ("wake2d_case.jl", "problem.jl", "bayes.jl")],
                  params = Dict{String,Any}("nstarts" => nstarts, "budget" => budget, "case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE))))
    # Compile both kinds of run before timing any (the first of each costs 15 to 20 s more).
    warm = case_problem(wake_amplitude, WAKE_BOX, (WAKE_BOX.lo .+ WAKE_BOX.hi) ./ 2)
    solve(warm, LBFGSB(); maxiters = 1)
    t0 = time()
    results = Dict{String,Any}[]
    n = length(SCALE)
    for seed in 1:nstarts
        for ξ in XIS
            println("BO ξ = $ξ, seed $seed")
            r, sol = compare_run("BO ξ=$ξ", BayesianEI(; seed, ξ), WAKE_BOX.lo; seed, maxiters = budget)   # BO takes no start
            r["xi"] = ξ
            r["lengthscales"] = [exp.(θ[1:n]) for θ in sol.original.θ]
            r["ei"] = sol.original.ei
            push!(results, r)
        end
        println("L-BFGS-B, seed $seed")
        x0 = WAKE_BOX.lo .+ latin_hypercube(2n + 1, n, seed)[1] .* (WAKE_BOX.hi .- WAKE_BOX.lo)
        push!(results, compare_run("L-BFGS-B", LBFGSB(), x0; seed, maxiters = 25, reltol = 1e-4)[1])
    end
    Jstar = maximum(maximum(r["J"]) for r in results)
    tols = (1e-2, 1e-3, 1e-4)
    for r in results
        r["best_so_far"] = accumulate(max, r["J"])
        r["reach"] = Dict(string(ε) => (q = reach(r["best_so_far"], r["cum_seconds"], Jstar, ε);
                                        q === nothing ? nothing : Dict("calls" => q.calls, "seconds" => q.seconds)) for ε in tols)
        r["best_phys"] = r["best_z"] .* SCALE
        println(r["method"], " seed ", r["seed"], ": ", length(r["J"]), " points, best J = ", r["best_J"], " at ", r["best_phys"], ", reach = ", r["reach"])
    end
    weighted(n, s) = sum(r[n] * r[s] for r in results if r[n] > 0) / sum(r[n] for r in results)
    out = Dict{String,Any}("Jstar" => Jstar, "tolerances" => collect(tols), "runs" => results, "box_tau" => [WAKE_BOX.lo[1], WAKE_BOX.hi[1]],
                           "box_focus" => [WAKE_BOX.lo[2], WAKE_BOX.hi[2]], "mean_seconds_plain" => weighted("plain_runs", "plain_seconds"),
                           "mean_seconds_duals" => weighted("dual_runs", "dual_seconds"), "seconds" => time() - t0)
    out["xis"] = collect(XIS)
    open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    plot_results(dir, out)
    println("J* = $Jstar; mean run: plain $(out["mean_seconds_plain"]) s, with duals $(out["mean_seconds_duals"]) s; done in $(round(out["seconds"]; digits = 1)) s: $dir")
    return dir
end

function plot_results(dir, out)
    Jstar = Float64(out["Jstar"])
    fig = Figure(size = (1300, 430))
    ax1 = Axis(fig[1, 1]; xlabel = "points evaluated", ylabel = "1 − best J / J* (floored at 1e-6)", yscale = log10, title = "Gap to the best value found, per point")
    ax2 = Axis(fig[1, 2]; xlabel = "CPU wall time (s)", ylabel = "1 − best J / J*", yscale = log10, title = "… per second of simulation")
    ax3 = Axis(fig[1, 3]; xlabel = "pulse duration τ (1/ω₀)", ylabel = "focus x_f (c/ω₀)", title = "Evaluated points",
               limits = (out["box_tau"]..., out["box_focus"]...))
    colors = Dict("BO ξ=0.0" => :royalblue, "BO ξ=0.01" => :seagreen, "L-BFGS-B" => :orangered)
    floor = 1e-6
    for r in out["runs"]
        c = colors[r["method"]]
        gap = max.(1 .- Float64.(r["best_so_far"]) ./ Jstar, floor)
        stairs!(ax1, 1:length(gap), gap; color = c, step = :post, label = r["method"])
        stairs!(ax2, Float64.(r["cum_seconds"]), gap; color = c, step = :post)
        P = reduce(hcat, Float64.(z) .* SCALE for z in r["z"])
        scatter!(ax3, P[1, :], P[2, :]; color = (c, 0.6), marker = r["method"] == "L-BFGS-B" ? :utf8 : :circle, markersize = 7)
        r["method"] == "L-BFGS-B" && lines!(ax3, P[1, :], P[2, :]; color = (c, 0.4))
    end
    for ε in Float64.(out["tolerances"])
        hlines!(ax1, [ε]; color = :gray, linestyle = :dot)
        hlines!(ax2, [ε]; color = :gray, linestyle = :dot)
    end
    axislegend(ax1; position = :rt, merge = true)
    save_plot(dir, "bo_compare", fig, Dict("Jstar" => Jstar, "gap_floor" => floor,
                                           "runs" => [Dict(k => r[k] for k in ("method", "seed", "best_so_far", "cum_seconds", "reach")) for r in out["runs"]]))
end

if length(ARGS) == 2 && ARGS[1] == "replot"
    plot_results(ARGS[2], JSON.parsefile(joinpath(ARGS[2], "results.json")))
else
    main(; nstarts = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3, budget = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30)
end
