# The finite-difference-checked gradient: T-143 (A3, A4, A7).

# Field energy at a fixed time of a small Landau run, as a function of the amplitude α of the
# initial density perturbation, which enters only through the weights (A3). Reference mode and
# the same seed on every evaluation: common random numbers (A4, A7).
const GRAD_KW = (; n = 32, ppc = 64, tmax_ωp = 25, shape = Quadratic, mode = Reference(), seed = 2)

function energy_at_end(α)
    T = typeof(α)
    st, p, info = landau_case(T; α, GRAD_KW...)
    initialize!(st, p)
    run!(st, p, info.nsteps)
    return st.W[end], st, info
end

const GRADIENT = Dict{String,Any}()

@testset "T-143 ForwardDiff gradient against a finite-difference scan [A3, A4, A7, WP-010]" begin
    α0 = 0.05
    f(α) = energy_at_end(α)[1]
    ad = ForwardDiff.derivative(f, α0)
    f0, st, info = energy_at_end(α0)
    # The run is long enough that particles cross cells and the periodic seam many times:
    # thermal speed × duration in units of the box length.
    crossings = info.vth * info.nsteps * info.dt / info.L
    @test crossings > 1
    hs = 10.0 .^ (-1:-0.5:-9)
    fd = [(f(α0 + h) - f(α0 - h)) / 2h for h in hs]
    rel = abs.(fd .- ad) ./ abs(ad)
    # The scan's floor, from the scan alone: the spread (max - min) of the difference quotients
    # over three consecutive steps, smallest where truncation (large h) and roundoff (small h)
    # balance. The roundoff part is not eps·|f|/h: rounding differences are amplified by the
    # particle dynamics (Float64 and Dual value parts of f differ by ~25 eps here), which the
    # spread measures and a formula would not.
    spread = [maximum(fd[(i - 1):(i + 1)]) - minimum(fd[(i - 1):(i + 1)]) for i in 2:(length(hs) - 1)]
    istar = argmin(spread) + 1
    floor_rel = spread[istar - 1] / abs(ad)
    @info "T-143 gradient scan" ad f0 crossings best_h = hs[istar] floor_rel rel_at_best = rel[istar] min_rel = minimum(rel) scan = collect(zip(hs, fd, rel))
    GRADIENT["scan"] = (; ad, f0, hs, fd, rel, istar, floor_rel)
    @test rel[istar] <= floor_rel          # agreement within the scan's error floor
end
