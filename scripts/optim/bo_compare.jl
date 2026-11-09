# WP-048: how many evaluations, and how much CPU time, Bayesian optimization (bayes.jl) and the
# gradient-based projected BFGS of WP-045 (maximize.jl, ForwardDiff gradients) need to reach the
# optimum of the WP-045 problem: the wake amplitude of the small 2D LWFA case (wake2d_case.jl)
# over the pulse duration τ and the focus, at fixed a₀ and density, in the WP-045 box.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/bo_compare.jl [nstarts] [budget]
#
# Every run starts from the same Latin-hypercube seeds: BO from the 5-point design of seed s, BFGS
# from that design's first point. Each objective call is logged with its kind (a plain run, or a
# run with dual numbers that also gives the gradient) and its wall time. The reference optimum J*
# is the largest value any run evaluated; a run "reaches" a tolerance ε at the first call whose
# best value so far is within ε·J* of J*.
#
# Writes scripts/runs/optim-bo_compare-<stamp>/: manifest.toml, results.json, bo_compare.png + .json.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "maximize.jl"))
include(joinpath(@__DIR__, "bayes.jl"))
using ForwardDiff: Dual, value, partials

# The WP-045 box and scaling: z = (τ/6, focus/90), τ ∈ [4, 9], focus ∈ [60, 130].
const SCALE = (6.0, 90.0)
const LO = [4.0 / SCALE[1], 60.0 / SCALE[2]]
const HI = [9.0 / SCALE[1], 130.0 / SCALE[2]]
phys(z) = (z[1] * SCALE[1], z[2] * SCALE[2])

"An objective that logs each call: (kind, z, J, seconds)."
struct Logged
    log::Vector{Dict{String,Any}}
end
Logged() = Logged(Dict{String,Any}[])

function (L::Logged)(z::AbstractVector{<:Real})                 # a plain run: the value only
    t = @elapsed J = wake_amplitude(phys(z)...)
    push!(L.log, Dict("kind" => "value", "z" => collect(z), "J" => J, "seconds" => t))
    println("  value  (τ, focus) = ", round.(phys(z); digits = 3), "  J = ", J, "  ", round(t; digits = 1), " s")
    flush(stdout)
    return J
end

"Value and gradient in one run with a dual (three partials, the third unseeded, as in WP-045's wake2d.jl: two partials segfault on the CPU backend)."
function value_gradient(L::Logged, z)
    d1 = Dual{Nothing}(z[1], 1.0, 0.0, 0.0)
    d2 = Dual{Nothing}(z[2], 0.0, 1.0, 0.0)
    t = @elapsed v = wake_amplitude(phys((d1, d2))...)
    J, g = value(v), collect(Tuple(partials(v))[1:2])
    push!(L.log, Dict("kind" => "gradient", "z" => collect(z), "J" => J, "grad" => g, "seconds" => t))
    println("  grad   (τ, focus) = ", round.(phys(z); digits = 3), "  J = ", J, "  ∇J = ", g, "  ", round(t; digits = 1), " s")
    flush(stdout)
    return J, g
end

function main(; nstarts = 3, budget = 30)
    dir = run_dir("bo_compare", @__FILE__; prefix = "optim", work_package = "WP-048",
                  extra_inputs = [joinpath(@__DIR__, f) for f in ("wake2d_case.jl", "maximize.jl", "bayes.jl")],
                  params = Dict{String,Any}("nstarts" => nstarts, "budget" => budget, "case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE))))
    # Compile both kinds of call before timing any (the first call of each costs 15 to 20 s more).
    warm = Logged()
    zc = (LO .+ HI) ./ 2
    warm(zc); value_gradient(warm, zc)
    t0 = time()
    runs = Dict{String,Any}[]
    for seed in 1:nstarts
        println("BO, seed $seed")
        L = Logged()
        r = bayesopt(L, LO, HI; budget, seed, verbose = true)
        push!(runs, Dict("method" => "BO", "seed" => seed, "calls" => L.log, "best_z" => r.best_x, "best_J" => r.best_y,
                         "lengthscales" => [exp.(θ[1:2]) for θ in r.θ], "ei" => r.ei))
        println("BFGS, seed $seed")
        L = Logged()
        z0 = LO .+ latin_hypercube(2length(LO) + 1, length(LO), seed)[1] .* (HI .- LO)
        r = maximize_box(z -> value_gradient(L, z), z0, LO, HI; maxiter = 25, gtol = 1e-4, xtol = 1e-4)
        push!(runs, Dict("method" => "BFGS", "seed" => seed, "calls" => L.log, "best_z" => r.z, "best_J" => r.f, "reason" => r.reason))
    end
    Jstar = maximum(c["J"] for r in runs for c in r["calls"])
    tols = (1e-2, 1e-3, 1e-4)
    for r in runs
        best, sec = -Inf, 0.0
        r["best_so_far"] = Float64[]
        r["cum_seconds"] = Float64[]
        for c in r["calls"]
            best = max(best, c["J"]); sec += c["seconds"]
            push!(r["best_so_far"], best); push!(r["cum_seconds"], sec)
        end
        r["reach"] = Dict(string(ε) => (q = reach(r["best_so_far"], r["cum_seconds"], Jstar, ε);
                                        q === nothing ? nothing : Dict("calls" => q.calls, "seconds" => q.seconds)) for ε in tols)
        r["best_phys"] = collect(phys(r["best_z"]))
        println(r["method"], " seed ", r["seed"], ": ", length(r["calls"]), " calls, best J = ", r["best_J"], " at ", r["best_phys"], ", reach = ", r["reach"])
    end
    tv = [c["seconds"] for r in runs for c in r["calls"] if c["kind"] == "value"]
    tg = [c["seconds"] for r in runs for c in r["calls"] if c["kind"] == "gradient"]
    out = Dict{String,Any}("Jstar" => Jstar, "tolerances" => collect(tols), "runs" => runs, "box_tau" => [4.0, 9.0], "box_focus" => [60.0, 130.0],
                           "median_seconds_value" => median(tv), "median_seconds_gradient" => median(tg), "seconds" => time() - t0)
    open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    plot_results(dir, out)
    println("J* = $Jstar; median call: value $(median(tv)) s, gradient $(median(tg)) s; done in $(round(out["seconds"]; digits = 1)) s: $dir")
    return dir
end

function plot_results(dir, out)
    Jstar = Float64(out["Jstar"])
    fig = Figure(size = (1300, 430))
    ax1 = Axis(fig[1, 1]; xlabel = "objective calls", ylabel = "1 − best J / J* (floored at 1e-6)", yscale = log10, title = "Gap to the best value found, per call")
    ax2 = Axis(fig[1, 2]; xlabel = "CPU wall time (s)", ylabel = "1 − best J / J*", yscale = log10, title = "… per second of simulation")
    ax3 = Axis(fig[1, 3]; xlabel = "pulse duration τ (1/ω₀)", ylabel = "focus x_f (c/ω₀)", title = "Evaluated points",
               limits = (out["box_tau"]..., out["box_focus"]...))
    colors = Dict("BO" => :royalblue, "BFGS" => :orangered)
    floor = 1e-6
    for r in out["runs"]
        c = colors[r["method"]]
        gap = max.(1 .- Float64.(r["best_so_far"]) ./ Jstar, floor)
        stairs!(ax1, 1:length(gap), gap; color = c, step = :post, label = r["method"])
        stairs!(ax2, Float64.(r["cum_seconds"]), gap; color = c, step = :post)
        P = reduce(hcat, [Float64(z[1]) * SCALE[1], Float64(z[2]) * SCALE[2]] for z in (cl["z"] for cl in r["calls"]))
        scatter!(ax3, P[1, :], P[2, :]; color = (c, 0.6), marker = r["method"] == "BO" ? :circle : :utf8, markersize = 7)
        r["method"] == "BFGS" && lines!(ax3, P[1, :], P[2, :]; color = (c, 0.4))
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
