# The gradient check of WP-010 (T-143) as a plot: ForwardDiff against a central
# finite-difference scan (`fd_scan` of the core test helpers: Float64, Reference mode, common
# random numbers).
#   julia +1.13 --project=scripts scripts/es1d/gradient.jl
include(joinpath(@__DIR__, "common.jl"))
using ForwardDiff
using ForwardDiff: Dual, value
const backend = CPU()                                  # helpers.jl builds its species on it
include(joinpath(CORE_TEST, "helpers.jl"))             # fd_scan, the one definition of A7's floor

const GRAD_KW = (; n = 32, ppc = 64, tmax_ωp = 25, shape = Quadratic)
const GRAD_SEED = 2

function energy_at_end(α, mode = Reference(), seed = GRAD_SEED)
    st, p, info = landau_case(typeof(α); α, mode, seed, GRAD_KW...)
    initialize!(st, p)
    run!(st, p, info.nsteps)
    return st.W[end]
end

function gradient_scan(; α0 = 0.05)
    dir = run_dir("gradient", @__FILE__; params = Dict(string(k) => string(v) for (k, v) in pairs(merge(GRAD_KW, (; mode = Reference(), seed = GRAD_SEED, α0)))))
    ad = ForwardDiff.derivative(energy_at_end, α0)
    f0 = energy_at_end(α0)
    f0dual = value(energy_at_end(Dual{Nothing}(α0, 0.0)))
    hs = 10.0 .^ (-1:-0.25:-9)
    r = fd_scan(energy_at_end, α0, hs; seed = GRAD_SEED, ad)
    (; rel, istar, floor_rel) = r

    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "finite-difference step h",
              ylabel = "|FD(h) − AD| / |AD|", title = "dW/dα at t = 25/ω_p: ForwardDiff against central differences")
    scatterlines!(ax, hs, max.(rel, 1e-16); label = "scan")
    hlines!(ax, [floor_rel]; linestyle = :dash, color = :gray, label = "scan floor (spread of 3 neighbours)")
    scatter!(ax, [hs[istar]], [rel[istar]]; color = :red, markersize = 12, label = "h at the floor")
    axislegend(ax; position = :lt)
    save_plot(dir, "gradient_scan", fig, Dict(
        "ad" => ad, "f0" => f0, "f0_dual_value" => f0dual, "primal_vs_dual_value_relative" => abs(f0dual - f0) / abs(f0),
        "best_h" => hs[istar], "relative_difference_at_best_h" => rel[istar], "scan_floor_relative" => floor_rel,
        "min_relative_difference" => minimum(rel), "h" => hs, "relative_difference" => rel))
    println("gradient: AD = $ad, best h = $(hs[istar]), rel = $(rel[istar]), floor = $floor_rel; outputs in $dir")
    return dir
end

abspath(PROGRAM_FILE) == @__FILE__() && gradient_scan()
