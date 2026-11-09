# WP-052: the 2D optimization at a fixed laser energy. Maximize the witness bunch's mean energy
# gain (WP-047) of a small 2D LWFA run (witness2d_case.jl) over the pulse duration τ, the plasma
# density n₀ and the witness centre x_c, with a₀ coupled to τ so that the laser energy stays fixed;
# ForwardDiff gradients (Float64, Reference) and the box-constrained quasi-Newton maximizer of
# maximize.jl (WP-045). Checks: the launched energy against τ, AD against finite differences, two
# starts, a brute-force grid scan, slices through the optimum, the optimum inside the box, and the
# same problem at fixed a₀ and at a lower energy, where the duration runs to a bound.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/witness2d.jl
#
# Writes scripts/runs/optim-witness2d-<stamp>/: manifest.toml, results.json, witness2d.png + .json.
# The run asserts its checks (`check`): a failed one stops it with an error after writing
# results.json.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "witness2d_case.jl"))
include(joinpath(@__DIR__, "maximize.jl"))
using ForwardDiff: Dual, value, partials

# The parameters, scaled to O(1) (A1): z = (τ/6, n₀/0.07, x_c/40) in the box τ ∈ [4, 9],
# n₀ ∈ [0.04, 0.10], x_c ∈ [30, 50] (3 lengths inside the lattice, which starts at x = 24).
# Steps are capped at 0.1 in z.
const SCALE = (6.0, 0.07, 40.0)
const BOX = ((4.0, 9.0), (0.04, 0.10), (30.0, 50.0))
const LO = [BOX[i][1] / SCALE[i] for i in 1:3]
const HI = [BOX[i][2] / SCALE[i] for i in 1:3]
phys(z) = (z[1] * SCALE[1], z[2] * SCALE[2], z[3] * SCALE[3])

J(z) = witness_gain(phys(z)...)

"""
The objective and its gradient in z, in one run with a three-partial dual. With `a0` given, the
pulse keeps that a₀ instead of the case's energy (its energy then grows with τ: section 5).
"""
function fg(z; a0 = nothing)
    d = ntuple(i -> Dual{Nothing}(z[i], ntuple(j -> Float64(i == j), 3)...), 3)
    τ, n, c = phys(d)
    kw = a0 === nothing ? (;) : (; energy = energy_at_a0(a0, τ))
    v = witness_gain(τ, n, c; kw...)
    f, g = value(v), collect(Tuple(partials(v)))
    println("  (τ, n₀, x_c) = ", round.(phys(z); digits = 4), a0 === nothing ? "" : "  a₀ = $a0", "  J = ", f, "  ∇J = ", g)
    flush(stdout)
    return f, g
end

"The energy of a pulse of amplitude `a0` and duration `τ`: the energy keyword that holds a₀ fixed."
energy_at_a0(a0, τ) = WCASE.energy * (a0 / a0_at_energy(τ))^2

"The best of the objective over the centre box at (τ, n₀) from one run (the centre is free: weights only), on a 0.25 grid."
function best_over_center(τ, n; kw...)
    st = witness_state(τ, n, 40.0; kw...)
    cs = collect(BOX[3][1]:0.25:BOX[3][2])
    Js = gain_at_centers(st, cs)
    i = argmax(Js)
    return (; J = Js[i], center = cs[i], Js, centers = cs)
end

"""
The 1D linear-theory reference for the density (derived here, not from a source): at a fixed
energy and duration the wake amplitude is ∝ n₀·exp(−n₀τ²/8) (E₀ = √n₀ times k_pτ·a₀²·exp(−k_p²τ²/8)
with a₀²τ fixed); a witness at c slips against the wake's phase velocity √(1 − n₀) at the rate
s = 1 − √(1 − n₀), so over a distance L its best gain is the amplitude times 2·sin(k_p·s·L/2)/(k_p·s)
while k_p·s·L ≤ π, and 2/(k_p·s) past it (dephased). Returns the density that maximizes it.
"""
function linear_density_optimum(τ, L)
    G(n) = (kp = sqrt(n); s = 1 - sqrt(1 - n); n * exp(-n * τ^2 / 8) * 2 * sin(min(kp * s * L / 2, π / 2)) / (kp * s))
    ns = range(0.01, 0.3; length = 29001)
    return ns[argmax(G.(ns))]
end

function main()
    cfg = Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(WCASE))
    dir = run_dir("witness2d", @__FILE__; prefix = "optim", work_package = "WP-052",
                  extra_inputs = [joinpath(@__DIR__, "witness2d_case.jl"), joinpath(@__DIR__, "maximize.jl")], params = cfg)
    out = Dict{String,Any}("case" => cfg, "scale" => collect(SCALE), "box" => [collect(b) for b in BOX])
    checks = Dict{String,Any}[]
    check(name, ok, detail) = (push!(checks, Dict("check" => name, "pass" => ok, "detail" => detail)); println(ok ? "PASS " : "FAIL ", name, ": ", detail); flush(stdout))
    t0 = time()

    # 0. The fixed energy in the simulation: the field energy the antenna has launched into vacuum
    #    (no plasma, 300 steps, t ≈ 108, before the window moves) at three durations.
    vac = Dict{String,Any}[]
    for τ in (4.0, 6.5, 9.0)
        prob = witness_problem(τ, 0.0, 40.0; nsteps = 300, inject = 50.0)
        st = EMState(prob, EMNumerics(; mode = Reference()))
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        W = field_energy(histories(st))
        push!(vac, Dict("tau" => τ, "a0" => a0_at_energy(τ), "W" => W[end], "W_20_steps_before" => W[end - 20],
                        "laser_energy" => laser_energy(prob.lasers[1])))
        println("vacuum τ = $τ: launched W = $(W[end]), laser_energy = $(laser_energy(prob.lasers[1]))")
    end
    out["launched_energy"] = vac
    Ws = [v["W"] for v in vac]
    check("laser_energy is the case's energy at every τ", all(v -> isapprox(v["laser_energy"], WCASE.energy; rtol = 1e-14), vac),
          "laser_energy = $([v["laser_energy"] for v in vac]), energy = $(WCASE.energy)")
    spread = (maximum(Ws) - minimum(Ws)) / minimum(Ws)
    check("the launched energy is the same at every τ", spread < 2e-3, "relative spread $spread over τ = 4, 6.5, 9; W/energy = $(Ws ./ WCASE.energy)")

    # 1. The gradient at the first start against central differences in z at three steps.
    z0 = [4.5 / SCALE[1], 0.05 / SCALE[2], 36.0 / SCALE[3]]
    f0, g0 = fg(z0)
    J0 = J(z0)
    check("fg's value is J", isapprox(f0, J0; rtol = 1e-12), "fg = $f0, J = $J0")
    fdcheck = Dict{String,Any}[]
    for h in (1e-2, 1e-3, 1e-4)
        fd = [(J(z0 .+ h .* (1:3 .== i)) - J(z0 .- h .* (1:3 .== i))) / 2h for i in 1:3]
        push!(fdcheck, Dict("h" => h, "fd" => fd, "rel" => abs.(fd .- g0) ./ abs.(g0)))
        println("FD check h = $h: fd = $fd, ad = $g0")
    end
    out["gradient_check"] = Dict("z" => z0, "ad" => g0, "fd" => fdcheck)
    relmin = minimum(maximum(Float64.(c["rel"])) for c in fdcheck)
    check("AD against central FD at the first start", relmin < 1e-5, "best max relative difference over h = $relmin")

    # 2. The optimizer from two starts on opposite sides of the box.
    runs = Dict{String,Any}[]
    for (name, zs) in (("start A", z0), ("start B", [8.5 / SCALE[1], 0.095 / SCALE[2], 47.0 / SCALE[3]]))
        println("optimizing from $name")
        t = @elapsed r = maximize_box(fg, zs, LO, HI; maxiter = 40, gtol = 1e-4, xtol = 1e-5, maxstep = 0.1)
        push!(runs, Dict("name" => name, "start" => zs, "z" => r.z, "phys" => collect(phys(r.z)), "J" => r.f, "grad" => r.g,
                         "a0" => a0_at_energy(phys(r.z)[1]), "reason" => r.reason, "evals" => r.evals, "seconds" => t,
                         "path" => [Dict("z" => p[1], "phys" => collect(phys(p[1])), "J" => p[2], "grad" => p[3]) for p in r.path]))
        println("$name: $(r.reason) after $(r.evals) evaluations, (τ, n₀, x_c) = $(phys(r.z)), J = $(r.f), ∇J = $(r.g)")
    end
    out["optimizer"] = runs
    Ja, Jb = runs[1]["J"], runs[2]["J"]
    check("both starts reach the same value", abs(Ja - Jb) <= 1e-4 * max(Ja, Jb), "J = $Ja and $Jb")
    # a stop on a small step (xtol) is checked before the gradient of the new point is, so the gradient is checked here
    check("both starts end with the gradient below gtol", all(r -> maximum(abs, r["grad"]) < 1e-4, runs),
          join(("$(r["reason"]), |∇J|∞ = $(maximum(abs, r["grad"]))" for r in runs), "; "))
    best = runs[argmax([r["J"] for r in runs])]
    zb = Float64.(best["z"])
    slack = [min(zb[i] - LO[i], HI[i] - zb[i]) / (HI[i] - LO[i]) for i in 1:3]
    check("the optimum is inside the box", all(>(0.02), slack), "distance to the nearest bound as a fraction of the box: $slack")

    # 3. The brute-force grid scan over (τ, n₀), 6 × 7 runs, the centre maximized on a 0.25 grid from each run.
    τs, ns = collect(range(4.0, 9.0; length = 6)), collect(range(0.04, 0.10; length = 7))
    grid = [best_over_center(τ, n) for τ in τs, n in ns]
    Jg = [g.J for g in grid]
    ib = argmax(Jg)
    out["grid"] = Dict("tau" => τs, "density" => ns, "J" => [Jg[i, :] for i in axes(Jg, 1)],
                       "center" => [[g.center for g in grid[i, :]] for i in axes(grid, 1)],
                       "best" => [τs[ib[1]], ns[ib[2]], grid[ib].center, Jg[ib]])
    println("grid best: τ = $(τs[ib[1]]), n₀ = $(ns[ib[2]]), x_c = $(grid[ib].center), J = $(Jg[ib])")
    check("the optimizer is at least the grid's best", best["J"] >= Jg[ib], "optimizer $(best["J"]), grid $(Jg[ib])")

    # 4. Slices through the optimum, ±5 % and ±10 % in τ and n₀ (inside the box), and in x_c from one run.
    τb, nb, cb = Float64.(best["phys"])
    rel = (-0.1, -0.05, 0.0, 0.05, 0.1)
    inbox(v, i) = BOX[i][1] <= v <= BOX[i][2]
    sτ = [τb * (1 + d) for d in rel if inbox(τb * (1 + d), 1)]
    sn = [nb * (1 + d) for d in rel if inbox(nb * (1 + d), 2)]
    sc = [cb * (1 + d) for d in rel if inbox(cb * (1 + d), 3)]
    stb = witness_state(τb, nb, cb)
    out["slices"] = Dict("tau" => sτ, "J_tau" => [witness_gain(τ, nb, cb) for τ in sτ],
                         "density" => sn, "J_density" => [witness_gain(τb, n, cb) for n in sn],
                         "center" => sc, "J_center" => gain_at_centers(stb, sc))
    jc = best["J"]
    s = out["slices"]
    check("gain_at_centers at the optimum's centre is witness_energy_gain", isapprox(gain_at_centers(stb, [cb])[1], jc; rtol = 1e-12),
          "$(gain_at_centers(stb, [cb])[1]) and $jc")
    check("the slices fall below the optimum on both sides",
          is_interior_max(sτ, s["J_tau"], τb, jc) && is_interior_max(sn, s["J_density"], nb, jc) && is_interior_max(sc, s["J_center"], cb, jc),
          "centre $jc, τ slice $(s["J_tau"]), n₀ slice $(s["J_density"]), x_c slice $(s["J_center"])")

    # 5. Why the energy is fixed: the same problem at fixed a₀ (the optimum's), where the energy grows
    #    with τ. A τ slice (the centre maximized) and the optimizer from start A, which should hold τ
    #    at its upper bound; and the τ scan at a fixed energy a factor 2.25 lower (a₀ = 1 at τ = 6).
    τf = [τb * (1 + d) for d in (-0.2, 0.0, 0.2)]
    a0b = a0_at_energy(τb)
    a0s = [witness_problem(τ, nb, cb; energy = energy_at_a0(a0b, τ)).lasers[1].a0 for τ in τf]
    check("the fixed-a₀ problems have the optimum's a₀", all(a -> isapprox(a, a0b; rtol = 1e-14), a0s), "a₀ = $a0s, optimum's $a0b")
    fixed_a0 = [best_over_center(τ, nb; energy = energy_at_a0(a0b, τ)) for τ in τf]
    println("optimizing at fixed a₀ = $a0b from start A")
    t = @elapsed r = maximize_box(z -> fg(z; a0 = a0b), z0, LO, HI; maxiter = 40, gtol = 1e-4, xtol = 1e-5, maxstep = 0.1)
    println("fixed a₀: $(r.reason) after $(r.evals) evaluations, (τ, n₀, x_c) = $(phys(r.z)), J = $(r.f), ∇J = $(r.g)")
    out["fixed_a0"] = Dict("a0" => a0b, "tau" => τf, "J" => [r.J for r in fixed_a0], "center" => [r.center for r in fixed_a0],
                           "optimizer" => Dict("start" => z0, "z" => r.z, "phys" => collect(phys(r.z)), "J" => r.f, "grad" => r.g,
                                               "reason" => r.reason, "evals" => r.evals, "seconds" => t,
                                               "path" => [Dict("phys" => collect(phys(p[1])), "J" => p[2]) for p in r.path]))
    check("at fixed a₀ the gain grows with τ", issorted([q.J for q in fixed_a0]), "J at τ = $τf: $([q.J for q in fixed_a0])")
    check("at fixed a₀ the optimizer ends with τ on its upper bound", r.z[1] == HI[1] && r.g[1] > 0, "τ = $(phys(r.z)[1]), ∂J/∂z₁ = $(r.g[1])")
    τl = [4.0, 5.0, 6.0, 7.0]
    low = [best_over_center(τ, nb; energy = WCASE.energy / 2.25) for τ in τl]
    out["low_energy"] = Dict("energy" => WCASE.energy / 2.25, "tau" => τl, "J" => [r.J for r in low], "center" => [r.center for r in low])
    println("lower energy: J at τ = $τl: $([r.J for r in low])")

    # 6. The 1D linear-theory density for the duration found and the witness's acceleration distance
    #    (from its injection to the last step), recorded, not asserted.
    prob = witness_problem(τb, nb, cb)
    dt = Float64(prob.dt)
    L = (prob.nsteps - ParticleMeshElectromagnetic.witness_step(WCASE.inject, dt) + 1) * dt
    out["theory"] = Dict("distance" => L, "density_1d_linear" => linear_density_optimum(τb, L),
                         "note" => "1D linear, a witness at c, derived in the docstring of linear_density_optimum; at fixed energy 1D linear theory puts the duration optimum at τ → 0")
    out["seconds"] = time() - t0
    out["checks"] = checks
    open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    plot_results(dir, out)
    println("done in $(round(out["seconds"]; digits = 1)) s: $dir")
    failed = [c["check"] for c in checks if !c["pass"]]
    isempty(failed) || error("WP-052 checks failed: $(join(failed, "; ")) (results in $dir)")
    return dir
end

function plot_results(dir, out)
    g = out["grid"]
    τs, ns = Float64.(g["tau"]), Float64.(g["density"])
    Jg = permutedims(reduce(hcat, Float64.(r) for r in g["J"]))       # Jg[i, j] at (τs[i], ns[j])
    fig = Figure(size = (1500, 440))
    ax = Axis(fig[1, 1]; xlabel = "pulse duration τ (1/ω₀)", ylabel = "density n₀ (n_c)",
              title = "Witness gain (mₑc², best centre) at fixed energy")
    hm = heatmap!(ax, τs, ns, Jg; colormap = :viridis)
    Colorbar(fig[1, 2], hm)
    for (k, r) in enumerate(out["optimizer"])
        P = reduce(hcat, Float64.(p["phys"]) for p in r["path"])
        scatterlines!(ax, P[1, :], P[2, :]; color = (:red, :orange)[k], linewidth = 2, markersize = 7, label = r["name"])
    end
    Pf = reduce(hcat, Float64.(p["phys"]) for p in out["fixed_a0"]["optimizer"]["path"])
    scatterlines!(ax, Pf[1, :], Pf[2, :]; color = :white, linestyle = :dot, markersize = 5, label = "fixed a₀ (not this map)")
    hlines!(ax, [out["theory"]["density_1d_linear"]]; color = :magenta, linestyle = :dash, label = "1D linear dephasing n₀")
    axislegend(ax; position = :rt)
    s = out["slices"]
    best = out["optimizer"][argmax([r["J"] for r in out["optimizer"]])]
    τb, nb, cb = Float64.(best["phys"])
    ax2 = Axis(fig[1, 3]; xlabel = "relative change of the parameter", ylabel = "witness gain (mₑc²)", title = "Slices through the optimum")
    scatterlines!(ax2, Float64.(s["tau"]) ./ τb .- 1, Float64.(s["J_tau"]); label = "τ (a₀ coupled)")
    scatterlines!(ax2, Float64.(s["density"]) ./ nb .- 1, Float64.(s["J_density"]); label = "n₀")
    scatterlines!(ax2, Float64.(s["center"]) ./ cb .- 1, Float64.(s["J_center"]); label = "x_c")
    axislegend(ax2; position = :cb)
    ax3 = Axis(fig[1, 4]; xlabel = "pulse duration τ (1/ω₀)", ylabel = "witness gain (mₑc², best centre)",
               title = "τ at n₀ = $(round(nb; digits = 4)): fixed energy or fixed a₀")
    gi = argmin(abs.(ns .- nb))
    scatterlines!(ax3, τs, Jg[:, gi]; label = "fixed energy, grid n₀ = $(round(ns[gi]; digits = 3))")
    scatterlines!(ax3, Float64.(out["fixed_a0"]["tau"]), Float64.(out["fixed_a0"]["J"]); label = "fixed a₀ = $(round(out["fixed_a0"]["a0"]; digits = 3))")
    scatterlines!(ax3, Float64.(out["low_energy"]["tau"]), Float64.(out["low_energy"]["J"]); label = "energy / 2.25")
    scatter!(ax3, [τb], [best["J"]]; color = :red, markersize = 12, label = "optimum")
    axislegend(ax3; position = :lb)
    save_plot(dir, "witness2d", fig, out)
end

# `julia ... witness2d.jl replot <run dir>` redraws the figure from the run's results.json.
if length(ARGS) == 2 && ARGS[1] == "replot"
    plot_results(ARGS[2], JSON.parsefile(joinpath(ARGS[2], "results.json")))
else
    main()
end
