# WP-045: the first 2D optimization. Maximize the wake amplitude of a small 2D LWFA run
# (wake2d_case.jl) over the pulse duration τ and the focus, at fixed a₀ and density, with
# ForwardDiff gradients (Float64, Reference) and the box-constrained quasi-Newton maximizer of
# maximize.jl; check the optimum against a brute-force grid scan, a finite-difference check of
# the gradient, slices through the optimum, and the 1D linear-theory duration 2/ω_p.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/wake2d.jl
#
# Writes scripts/runs/optim-wake2d-<stamp>/: manifest.toml, results.json, wake2d.png + wake2d.json.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "maximize.jl"))
using ForwardDiff: Dual, value, partials

# The parameters, scaled to O(1) (A1): z = (τ/6, focus/90), in the box τ ∈ [4, 9], focus ∈ [60, 130].
const SCALE = (6.0, 90.0)
const LO = [4.0 / SCALE[1], 60.0 / SCALE[2]]
const HI = [9.0 / SCALE[1], 130.0 / SCALE[2]]
phys(z) = (z[1] * SCALE[1], z[2] * SCALE[2])

J(z) = wake_amplitude(phys(z)...)

"""
The objective and its gradient in z, in one run with a dual. The dual carries three partials,
the third unseeded: the step segfaults on the KernelAbstractions CPU (PoCL) backend with
`Dual{Float64,2}` while 1, 3 and 5 partials run (WP-045; the upstream issue is linked in the PR).
"""
function fg(z)
    d1 = Dual{Nothing}(z[1], 1.0, 0.0, 0.0)
    d2 = Dual{Nothing}(z[2], 0.0, 1.0, 0.0)
    v = wake_amplitude(phys((d1, d2))...)
    f, g = value(v), collect(Tuple(partials(v))[1:2])
    println("  z = ", round.(z; digits = 5), "  (τ, focus) = ", round.(phys(z); digits = 3), "  J = ", f, "  ∇J = ", g)
    flush(stdout)
    return f, g
end

function main()
    dir = run_dir("wake2d", @__FILE__; prefix = "optim", work_package = "WP-045",
                  extra_inputs = [joinpath(@__DIR__, "wake2d_case.jl"), joinpath(@__DIR__, "maximize.jl")],
                  params = Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE)))
    out = Dict{String,Any}("case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE)),
                           "scale" => collect(SCALE), "box_tau" => [4.0, 9.0], "box_focus" => [60.0, 130.0])
    t0 = time()

    # 1. Gradient check at the first start: central differences in z at three steps.
    z0 = [4.5 / SCALE[1], 70.0 / SCALE[2]]
    f0, g0 = fg(z0)
    fdcheck = Dict{String,Any}[]
    for h in (1e-2, 1e-3, 1e-4)
        fd = [(J(z0 .+ h .* (1:2 .== i)) - J(z0 .- h .* (1:2 .== i))) / 2h for i in 1:2]
        push!(fdcheck, Dict("h" => h, "fd" => fd, "rel" => abs.(fd .- g0) ./ abs.(g0)))
        println("FD check h = $h: fd = $fd, ad = $g0")
    end
    out["gradient_check"] = Dict("z" => z0, "ad" => g0, "fd" => fdcheck)

    # 2. The optimizer from two starts.
    runs = Dict{String,Any}[]
    for (name, zs) in (("start A", z0), ("start B", [8.5 / SCALE[1], 125.0 / SCALE[2]]))
        println("optimizing from $name")
        t = @elapsed r = maximize_box(fg, zs, LO, HI; maxiter = 25, gtol = 1e-4, xtol = 1e-4)
        push!(runs, Dict("name" => name, "start" => zs, "z" => r.z, "phys" => collect(phys(r.z)), "J" => r.f, "grad" => r.g,
                         "reason" => r.reason, "evals" => r.evals, "seconds" => t,
                         "path" => [Dict("z" => p[1], "phys" => collect(phys(p[1])), "J" => p[2], "grad" => p[3]) for p in r.path]))
        println("$name: $(r.reason) after $(r.evals) evaluations, (τ, focus) = $(phys(r.z)), J = $(r.f), ∇J = $(r.g)")
    end
    out["optimizer"] = runs

    # 3. The brute-force grid scan over the box (7 × 7).
    τs, fs = collect(range(4.0, 9.0; length = 7)), collect(range(60.0, 130.0; length = 7))
    grid = [wake_amplitude(τ, f) for τ in τs, f in fs]
    ib = argmax(grid)
    out["grid"] = Dict("tau" => τs, "focus" => fs, "J" => [grid[i, :] for i in axes(grid, 1)], "best" => [τs[ib[1]], fs[ib[2]], grid[ib]])
    println("grid best: τ = $(τs[ib[1]]), focus = $(fs[ib[2]]), J = $(grid[ib])")

    # 4. Slices through the optimizer's best point, ±10 % in each parameter.
    best = runs[argmax([r["J"] for r in runs])]
    τbest, fbest = best["phys"]
    sτ = [τbest * (1 + d) for d in (-0.1, -0.05, 0.0, 0.05, 0.1) if 4.0 <= τbest * (1 + d) <= 9.0]
    sf = [fbest * (1 + d) for d in (-0.1, -0.05, 0.0, 0.05, 0.1) if 60.0 <= fbest * (1 + d) <= 130.0]
    out["slices"] = Dict("tau" => sτ, "J_tau" => [wake_amplitude(τ, fbest) for τ in sτ],
                         "focus" => sf, "J_focus" => [wake_amplitude(τbest, f) for f in sf])

    # 5. The 1D linear-theory duration: the ponderomotive drive a² ∝ exp(−2t²/τ²) excites a wake
    #    ∝ τ·exp(−ω_p²τ²/8), largest at ω_p·τ = 2 (derived here; 1D, linear, no diffraction).
    out["theory_tau_1d"] = 2 / sqrt(CASE.density)
    out["seconds"] = time() - t0

    open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    plot_results(dir, out)
    println("done in $(round(out["seconds"]; digits = 1)) s: $dir")
    return dir
end

function plot_results(dir, out)
    g = out["grid"]
    τs, fs = Float64.(g["tau"]), Float64.(g["focus"])
    Jg = permutedims(reduce(hcat, Float64.(r) for r in g["J"]))       # Jg[i, j] at (τs[i], fs[j])
    fig = Figure(size = (1100, 420))
    ax = Axis(fig[1, 1]; xlabel = "pulse duration τ (1/ω₀)", ylabel = "focus x_f (c/ω₀)",
              title = "Wake amplitude E_x/E₀ (grid) and optimizer paths")
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
    ax2 = Axis(fig[1, 3]; xlabel = "relative change of the parameter", ylabel = "E_x/E₀", title = "Slices through the optimum")
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
