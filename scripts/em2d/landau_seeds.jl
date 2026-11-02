# The seed spread of T-364's test-size oblique Landau run (WP-023): the configuration of the test
# (`landau-oblique-2d.toml` at 32 × 32 cells, Δt four times the case's, 16 × 16 electrons per cell,
# 5 % perturbation, t = 18, Float64, Reference), fitted by the case's `damping_rate_mode` rule on
# |k̂·Ê|² at every step, for the case's four seeds and seeds 1 to 16. It measures the spread the
# test's rate bound rests on and whether the seed mean agrees with linear theory.
#   julia +1.13 --project=scripts -t 16 scripts/em2d/landau_seeds.jl
include(joinpath(@__DIR__, "..", "em1d", "common.jl"))
using Statistics: mean, std

const SEEDS = [20261031, 20262031, 20263031, 20264031, 1:16...]

function one_seed(case, seed)
    n0 = Tuple(case["grid"]["cells"])
    cells = (32, 32)
    prob = problem_from_case(case, Float64; seed, ppc = (16, 16), cells, dt = case["time"]["dt"] * minimum(n0 ./ cells),
                             amplitude = 1 / 20, t_end = 18)
    st = EMState(prob, numerics_from_case(case))
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    h = histories(st)
    t = (0:(prob.nsteps - 1)) .* Float64(prob.dt)
    o = case["observables"]["damping_rate_mode"]
    f = damping_from_peaks(t, abs2.(complex.(h.El_re, h.El_im)); fit_start = o["fit_start"], floor_factor = o["floor_factor"])
    return (; seed, gamma = isfinite(f.gamma) ? f.gamma : nothing, omega = isfinite(f.omega) ? f.omega : nothing,
              maxima = length(f.t_fit))
end

function main()
    case = load_case("landau-oblique-2d")
    dir = run_dir("landau-seeds", @__FILE__, "landau-oblique-2d"; work_package = "WP-023", prefix = "em2d",
                  params = Dict{String,Any}("seeds" => SEEDS, "cells" => [32, 32], "ppc" => [16, 16], "amplitude" => 0.05,
                                            "t_end" => 18, "element_type" => "Float64", "mode" => "Reference"))
    rows = [one_seed(case, s) for s in SEEDS]
    θ, ω_r = case["observables"]["damping_rate_mode"]["theory"], case["observables"]["damping_rate_mode"]["theory_frequency"]
    g = Float64[r.gamma for r in rows if r.gamma !== nothing && r.maxima >= 3]
    w = Float64[r.omega for r in rows if r.omega !== nothing && r.maxima >= 3]
    summary = Dict{String,Any}("theory" => θ, "theory_frequency" => ω_r, "seeds" => SEEDS,
        "per_seed" => [Dict("seed" => r.seed, "gamma" => r.gamma, "omega" => r.omega, "maxima" => r.maxima) for r in rows],
        "fitted" => length(g), "without_fit" => length(rows) - length(g),
        "gamma_mean" => mean(g), "gamma_std" => std(g), "gamma_sem" => std(g) / sqrt(length(g)),
        "gamma_mean_minus_theory_in_sem" => (mean(g) - θ) / (std(g) / sqrt(length(g))),
        "gamma_std_relative" => std(g) / θ,
        "omega_mean" => mean(w), "omega_max_abs_relative" => maximum(abs, (w .- ω_r) ./ ω_r),
        "within_10_percent" => count(x -> abs(x - θ) / θ <= 0.1, g))
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xlabel = "seed index", ylabel = "γ [ω_p]", title = "landau-oblique-2d at test size: γ per seed")
    scatter!(ax, 1:length(g), g; color = :black, label = "per seed ($(length(g)) fits)")
    hlines!(ax, [mean(g)]; color = :black, label = "mean $(round(mean(g); sigdigits = 4)) ± $(round(std(g) / sqrt(length(g)); sigdigits = 2)) (sem)")
    band!(ax, [0.5, length(g) + 0.5], mean(g) - std(g), mean(g) + std(g); color = (:gray, 0.2))
    hlines!(ax, [θ]; color = :blue, linestyle = :dash, label = "theory $(round(θ; sigdigits = 5))")
    axislegend(ax; position = :rb)
    save_plot(dir, "landau-seeds", fig, summary)
    JSON.print(stdout, summary, 2)
    println("\noutputs in $dir")
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
