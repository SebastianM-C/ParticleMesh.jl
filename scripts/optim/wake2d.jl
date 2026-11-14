# WP-045: the first 2D optimization. Maximize the wake amplitude of a small 2D LWFA run
# (wake2d_case.jl) over the pulse duration τ and the focus, at fixed a₀ and density, with
# ForwardDiff gradients (Float64, Reference) and the bound-constrained L-BFGS-B of Optimization.jl
# (problem.jl, WP-058); check the optimum against a brute-force grid scan, a finite-difference check of
# the gradient, slices through the optimum, and the 1D linear-theory duration 2/ω_p.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/wake2d.jl
#
# Writes scripts/runs/optim-wake2d-<stamp>/: manifest.toml, results.json, wake2d.png + wake2d.json.
# The run asserts its checks (`check`): a failed one stops it with an error after writing
# results.json, so a wrong gradient or a disagreeing optimizer cannot finish quietly.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))

# The parameters, scaled to O(1) (A1): z = (τ/6, focus/90) in WP-045's box (`WAKE_BOX`).
const SCALE = WAKE_BOX.scale
phys(z) = phys(WAKE_BOX, z)
const TAU, FOCUS = (WAKE_BOX.lo[1], WAKE_BOX.hi[1]), (WAKE_BOX.lo[2], WAKE_BOX.hi[2])   # the box in physical units
const OBJECTIVE = Objective(wake_amplitude, WAKE_BOX)

"The objective at `z`, from a plain run."
J(z) = OBJECTIVE(z)
"The objective and its gradient in z, in one run with duals."
fg(z) = value_gradient(OBJECTIVE, z)

function main()
    dir = run_dir("wake2d", @__FILE__; prefix = "optim", work_package = "WP-045",
                  extra_inputs = [joinpath(@__DIR__, "wake2d_case.jl"), joinpath(@__DIR__, "problem.jl")],
                  params = Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE)))
    out = Dict{String,Any}("case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE)),
                           "scale" => collect(SCALE), "box_tau" => collect(TAU), "box_focus" => collect(FOCUS))
    t0 = time()

    # 1. Gradient check at the first start: central differences in z at three steps.
    z0 = [4.5 / SCALE[1], 70.0 / SCALE[2]]
    f0, g0 = fg(z0)
    checks = Dict{String,Any}[]
    check(name, ok, detail) = (push!(checks, Dict("check" => name, "pass" => ok, "detail" => detail)); println(ok ? "PASS " : "FAIL ", name, ": ", detail))
    J0 = J(z0)
    check("fg's value is J", isapprox(f0, J0; rtol = 1e-12), "fg = $f0, J = $J0")   # the dual's value path may differ by ulps (A4)
    fdcheck = Dict{String,Any}[]
    for h in (1e-2, 1e-3, 1e-4)
        fd = [(J(z0 .+ h .* (1:2 .== i)) - J(z0 .- h .* (1:2 .== i))) / 2h for i in 1:2]
        push!(fdcheck, Dict("h" => h, "fd" => fd, "rel" => abs.(fd .- g0) ./ abs.(g0)))
        println("FD check h = $h: fd = $fd, ad = $g0")
    end
    out["gradient_check"] = Dict("z" => z0, "ad" => g0, "fd" => fdcheck)
    relmin = minimum(maximum(Float64.(c["rel"])) for c in fdcheck)
    check("AD against central FD at the first start", relmin < 1e-5, "best max relative difference over h = $relmin")

    # 2. The optimizer from two starts.
    runs = Dict{String,Any}[]
    for (name, zs) in (("start A", z0), ("start B", [8.5 / SCALE[1], 125.0 / SCALE[2]]))
        println("optimizing from $name")
        t = @elapsed r = maximize(OBJECTIVE, zs; maxiters = 25, gtol = 1e-4)
        push!(runs, Dict("name" => name, "start" => zs, "z" => r.z, "phys" => collect(phys(r.z)), "J" => r.f, "grad" => r.g,
                         "reason" => r.reason, "evals" => r.evals, "seconds" => t,
                         "path" => [Dict("z" => p[1], "phys" => collect(phys(p[1])), "J" => p[2], "grad" => p[3]) for p in r.path]))
        println("$name: $(r.reason) after $(r.evals) evaluations, (τ, focus) = $(phys(r.z)), J = $(r.f), ∇J = $(r.g)")
    end
    out["optimizer"] = runs
    Ja, Jb = runs[1]["J"], runs[2]["J"]
    check("both starts reach the same value", abs(Ja - Jb) <= 1e-4 * max(Ja, Jb), "J = $Ja and $Jb")
    # Each start stops on one of L-BFGS-B's convergence tests: the projected gradient below gtol, or an
    # iteration that lowers −J by less than about 2e-9 relative ("ftol"; OptimizationLBFGSB does not pass
    # that tolerance through, and at a sharp maximum it can fire with the gradient still above gtol).
    # Either way, the two starts must end at the same point.
    check("both starts stop on a convergence test", all(r -> r["reason"] in ("gtol", "ftol"), runs),
          join(("$(r["reason"]), |∇J|∞ = $(maximum(abs, r["grad"]))" for r in runs), "; "))
    dz = maximum(abs, Float64.(runs[1]["z"]) .- Float64.(runs[2]["z"]))
    check("both starts end at the same point", dz < 1e-4, "largest difference in z: $dz")

    # 3. The brute-force grid scan over the box (7 × 7).
    τs, fs = collect(range(TAU...; length = 7)), collect(range(FOCUS...; length = 7))
    grid = [wake_amplitude(τ, f) for τ in τs, f in fs]
    ib = argmax(grid)
    out["grid"] = Dict("tau" => τs, "focus" => fs, "J" => [grid[i, :] for i in axes(grid, 1)], "best" => [τs[ib[1]], fs[ib[2]], grid[ib]])
    println("grid best: τ = $(τs[ib[1]]), focus = $(fs[ib[2]]), J = $(grid[ib])")
    check("the optimizer is at least the grid's best", max(Ja, Jb) >= grid[ib], "optimizer $(max(Ja, Jb)), grid $(grid[ib])")

    # 4. Slices through the optimizer's best point, ±10 % in each parameter.
    best = runs[argmax([r["J"] for r in runs])]
    τbest, fbest = best["phys"]
    sτ = [τbest * (1 + d) for d in (-0.1, -0.05, 0.0, 0.05, 0.1) if TAU[1] <= τbest * (1 + d) <= TAU[2]]
    sf = [fbest * (1 + d) for d in (-0.1, -0.05, 0.0, 0.05, 0.1) if FOCUS[1] <= fbest * (1 + d) <= FOCUS[2]]
    out["slices"] = Dict("tau" => sτ, "J_tau" => [wake_amplitude(τ, fbest) for τ in sτ],
                         "focus" => sf, "J_focus" => [wake_amplitude(τbest, f) for f in sf])
    jc = best["J"]
    below(v) = all(x -> x < jc * (1 - 1e-9) || isapprox(x, jc; rtol = 1e-9), v) && count(x -> isapprox(x, jc; rtol = 1e-9), v) == 1
    check("the slices fall on both sides", below(out["slices"]["J_tau"]) && below(out["slices"]["J_focus"]),
          "centre $jc, τ slice $(out["slices"]["J_tau"]), focus slice $(out["slices"]["J_focus"])")

    # 4b. Resolution and box width: J at the optimum and at τ ± 5 % with 4 × 4 particles per cell,
    #     with Δy = 2Δx (128 rows, the same box), and with a box twice as wide (128 rows of Δy = 4Δx;
    #     the periodic images of the beam). Recorded, not asserted: the case is not converged in Δy.
    res = Dict{String,Any}()
    for (name, kw) in (("ppc 4x4", (; ppc = (4, 4))), ("dy = 2dx", (; dy_cells = 2, cells = (CASE.cells[1], 2CASE.cells[2]))),
                     ("box width x2", (; cells = (CASE.cells[1], 2CASE.cells[2]))))
        vals = [wake_amplitude(τbest * (1 + d), fbest; kw...) for d in (-0.05, 0.0, 0.05)]
        res[name] = Dict("tau" => [τbest * (1 + d) for d in (-0.05, 0.0, 0.05)], "J" => vals)
        println("resolution $name: J at τ·(0.95, 1, 1.05) = $vals")
    end
    out["resolution"] = res

    # 5. The 1D linear-theory duration: the ponderomotive drive a² ∝ exp(−2t²/τ²) excites a wake
    #    ∝ τ·exp(−ω_p²τ²/8), largest at ω_p·τ = 2 (derived here; 1D, linear, no diffraction).
    out["theory_tau_1d"] = 2 / sqrt(CASE.density)
    out["seconds"] = time() - t0

    out["checks"] = checks
    open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    plot_results(dir, out)
    println("done in $(round(out["seconds"]; digits = 1)) s: $dir")
    failed = [c["check"] for c in checks if !c["pass"]]
    isempty(failed) || error("WP-045 checks failed: $(join(failed, "; ")) (results in $dir)")
    return dir
end

function plot_results(dir, out)
    g = out["grid"]
    τs, fs = Float64.(g["tau"]), Float64.(g["focus"])
    Jg = permutedims(reduce(hcat, Float64.(r) for r in g["J"]))       # Jg[i, j] at (τs[i], fs[j])
    fig = Figure(size = (1100, 420))
    ax = Axis(fig[1, 1]; xlabel = "pulse duration τ (1/ω₀)", ylabel = "focus x_f (c/ω₀)",
              title = "RMS wake amplitude / E₀ (grid) and optimizer paths")
    hm = heatmap!(ax, τs, fs, Jg; colormap = :viridis)
    Colorbar(fig[1, 2], hm)
    colors = (:red, :orange)
    for (k, r) in enumerate(out["optimizer"])
        P = reduce(hcat, Float64.(p["phys"]) for p in r["path"])
        lines!(ax, P[1, :], P[2, :]; color = colors[k], linewidth = 2, label = r["name"])
        scatter!(ax, P[1, :], P[2, :]; color = colors[k], markersize = 7)
    end
    vlines!(ax, [out["theory_tau_1d"]]; color = :magenta, linestyle = :dash, linewidth = 2, label = "1D linear 2/ω_p")
    axislegend(ax; position = :rb)
    s = out["slices"]
    ax2 = Axis(fig[1, 3]; xlabel = "relative change of the parameter", ylabel = "RMS wake amplitude / E₀", title = "Slices through the optimum")
    best = out["optimizer"][argmax([r["J"] for r in out["optimizer"]])]
    τbest, fbest = best["phys"]
    scatterlines!(ax2, Float64.(s["tau"]) ./ τbest .- 1, Float64.(s["J_tau"]); label = "τ")
    scatterlines!(ax2, Float64.(s["focus"]) ./ fbest .- 1, Float64.(s["J_focus"]); label = "focus")
    axislegend(ax2; position = :cb)
    save_plot(dir, "wake2d", fig, out)
end

# `julia ... wake2d.jl replot <run dir>` redraws the figure from the run's results.json.
if length(ARGS) == 2 && ARGS[1] == "replot"
    plot_results(ARGS[2], JSON.parsefile(joinpath(ARGS[2], "results.json")))
else
    main()
end
