# The finite-difference-checked gradient: T-143 (A3, A4, A7), on top of `fd_scan` (helpers.jl).

# Field energy at a fixed time of a small Landau run, as a function of the amplitude α of the
# initial density perturbation, which enters only through the weights (A3). `fd_scan` calls it
# in Reference mode with one seed on every evaluation: common random numbers (A4, A7).
const GRAD_KW = (; n = 32, ppc = 64, tmax_ωp = 25, shape = Quadratic)
const GRAD_SEED = 2

function energy_at_end(α, mode, seed)
    T = typeof(α)
    st, p, info = landau_case(T; α, mode, seed, GRAD_KW...)
    initialize!(st, p)
    run!(st, p, info.nsteps)
    return st.W[end], st, info
end

const GRADIENT = Dict{String,Any}()

@testset "T-143 ForwardDiff gradient against a finite-difference scan [A3, A4, A7, WP-010]" begin
    α0 = 0.05
    f(α, mode, seed) = energy_at_end(α, mode, seed)[1]
    # The caller's obligation of fd_scan: the AD value from the same f, mode and seed.
    ad = ForwardDiff.derivative(α -> f(α, Reference(), GRAD_SEED), α0)
    f0, st, info = energy_at_end(α0, Reference(), GRAD_SEED)
    # The run is long enough that particles cross cells and the periodic seam many times:
    # thermal speed × duration in units of the box length.
    crossings = info.vth * info.nsteps * info.dt / info.L
    @test crossings > 1
    hs = 10.0 .^ (-1:-0.5:-9)
    r = fd_scan(f, α0, hs; seed = GRAD_SEED, ad)
    # The floor is measured, not computed from eps·|f|/h: rounding differences are amplified by
    # the particle dynamics (Float64 and Dual value parts of f differ by ~25 eps here).
    C = tolC("T-143")
    @info "T-143 gradient scan" ad f0 crossings best_h = hs[r.istar] r.floor_rel rel_at_best = r.rel[r.istar] ratio = r.rel[r.istar] / r.floor_rel min_rel = minimum(r.rel) scan = collect(zip(hs, r.fd, r.rel))
    GRADIENT["scan"] = (; ad, f0, hs, r...)
    MEASURED["T-143"] = Dict("rel/floor" => r.rel[r.istar] / r.floor_rel)
    @test r.rel[r.istar] <= C * r.floor_rel          # agreement within C times the scan's floor
end
