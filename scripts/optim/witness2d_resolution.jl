# WP-052: the fixed-energy optimum of witness2d.jl at a finer grid. The best-centre witness gain
# on a τ slice (n₀ fixed) and an n₀ slice (τ fixed) through the optimum, at the base resolution
# (Δx = λ₀/16, Δy = λ₀/4) and at Δx = λ₀/32, Δy = λ₀/16 (the resolution WP-051 found converged
# in Δy), the same box and end time, with the vertex of the parabola through each slice.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/witness2d_resolution.jl [τ* n₀*]
#
# Writes scripts/runs/optim-witness2d-resolution-<stamp>/: manifest.toml, results.json. Recorded,
# not asserted.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "witness2d_case.jl"))

"The best of the objective over centres 30 to 50 (0.25 apart) from one run at (τ, n₀)."
function best_center(τ, n; kw...)
    st, p = witness_state(τ, n, 40.0; kw...)
    cs = collect(30.0:0.25:50.0)
    Js = gain_at_centers(st, p, cs)
    i = argmax(Js)
    return (; J = Js[i], center = cs[i])
end

"The vertex of the parabola through three points."
vertex(x, y) = (c = hcat(x .^ 2, x, ones(3)) \ y; -c[2] / (2c[1]))

function main(τb = 4.9356, nb = 0.0720)
    dir = run_dir("witness2d-resolution", @__FILE__; prefix = "optim", work_package = "WP-052",
                  extra_inputs = [joinpath(@__DIR__, "witness2d_case.jl")],
                  params = Dict{String,Any}("tau" => τb, "density" => nb))
    tend = WCASE.nsteps * Float64(witness_problem(τb, nb, 40.0).dt)
    variants = (("dx = lambda/16, dy = lambda/4 (base)", (;)),
                ("dx = lambda/32, dy = lambda/16", (; ppl = 32, dy_cells = 2, cells = (2WCASE.cells[1], 4WCASE.cells[2]), t_end = tend)))
    τs, ns = [4.0, τb, 6.0], [0.064, nb, 0.080]
    out = Dict{String,Any}("tau_star" => τb, "density_star" => nb, "t_end" => tend)
    t0 = time()
    for (name, kw) in variants
        rτ = [best_center(τ, nb; kw...) for τ in τs]
        rn = [best_center(τb, n; kw...) for n in ns]
        Jτ, Jn = [r.J for r in rτ], [r.J for r in rn]
        out[name] = Dict("tau" => τs, "J_tau" => Jτ, "center_tau" => [r.center for r in rτ], "tau_vertex" => vertex(τs, Jτ),
                         "density" => ns, "J_density" => Jn, "center_density" => [r.center for r in rn], "density_vertex" => vertex(ns, Jn))
        println("$name: J(τ) = $Jτ, vertex $(vertex(τs, Jτ)); J(n₀) = $Jn, vertex $(vertex(ns, Jn))")
        flush(stdout)
    end
    out["seconds"] = time() - t0
    open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    println("done in $(round(out["seconds"]; digits = 1)) s: $dir")
    return dir
end

length(ARGS) == 2 ? main(parse.(Float64, ARGS)...) : main()
