# The gradient check of T-219 as a plot: ForwardDiff through the module's step against a central
# finite-difference scan (`fd_scan` of the engine's test helpers: Float64, Reference mode, common
# random numbers). The functional is the Ex field energy at the last step of the Landau case,
# as a function of the perturbation amplitude, which enters through the weights only (A3).
#   julia +1.13 --project=scripts scripts/em1d/gradient.jl
include(joinpath(@__DIR__, "common.jl"))
using ForwardDiff
using ForwardDiff: Dual, value
const CORE_TEST = joinpath(REPO, "lib", "ParticleMeshCore", "test")
const backend = CPU()                                  # helpers.jl builds its species on it
include(joinpath(CORE_TEST, "helpers.jl"))             # fd_scan, the one definition of A7's floor

const GRAD_KW = (; ppc = 64, t_end = 3)               # the configuration of T-219: 322 steps
const GRAD_SEED = 20261031

"Ex field energy of the Landau case at its last step, as a function of the perturbation amplitude."
function energy_at_end(α, mode = Reference(), seed = GRAD_SEED)
    case = load_case("landau")
    prob = problem_from_case(case, typeof(α); amplitude = α, seed, GRAD_KW...)
    st = EMState(prob, numerics_from_case(case; mode))
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    return Array(st.hist.WEx)[end]
end

function gradient_scan(; α0 = 0.01)
    dir = run_dir("gradient", @__FILE__, "landau";
                  params = Dict{String,Any}(string(k) => string(v) for (k, v) in pairs(merge(GRAD_KW, (; mode = Reference(), seed = GRAD_SEED, α0)))))
    ad = ForwardDiff.derivative(energy_at_end, α0)
    f0 = energy_at_end(α0)
    f0dual = value(energy_at_end(Dual{Nothing}(α0, 0.0)))
    hs = α0 .* 2.0 .^ (-1:-1:-24)
    r = fd_scan(energy_at_end, α0, hs; seed = GRAD_SEED, ad)
    (; rel, istar, floor_rel) = r

    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "finite-difference step h",
              ylabel = "|FD(h) − AD| / |AD|", title = "dW(Ex)/dα at the last step: ForwardDiff against central differences")
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
