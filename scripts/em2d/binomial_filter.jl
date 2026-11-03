# The binomial current filter (WP-029): the measurements behind T-440 and T-441, and the filtered
# 2D cases against the filtered linear theory.
#   julia +1.13 --project=scripts -t 8 scripts/em2d/binomial_filter.jl
#
# 1. transfer: a random field through `binomial!` (1D, 64 cells; 2D, 32 × 24), every DFT
#    coefficient's ratio output/input against H_p = Π_d cos^{2p}(π m_d/n_d), p = 1, 2, 4, 8.
# 2. gauss: the oblique Landau case at T-364's test size (32 × 32 cells, 16 × 16 electrons per
#    cell, 5 % perturbation, t = 18) with p = 0, 1, 2 passes: the plain Gauss residual
#    r = D⁻·E − (ρ − mean ρ) relative to max|ρ − mean ρ| against time, the filtered invariant
#    D⁻·E − Fρ (constant in exact arithmetic), and at the end, mode by mode, |r̂ − r̂⁰| over
#    |(ρ − ρ⁰)^| against the closed form 1 − H_p.
# 3. rates: with the current filtered, ∂ₜE = ∇×B − FJ, so the plasma response of the mode m is
#    multiplied by H_p(m): a linear theory with ω_p² replaced by H_p·ω_p². Weibel (cold, no seed
#    noise) at T-365's test size (64 × 64 cells, 4 × 4 per beam per cell) with p = 0, 4, 8, 16:
#    γ against weibel_cold_growth_2d(k, v₀; ωp2 = H_p); Landau (the size of 2., p = 0, 2, 4): γ and
#    ω of the seed mean of the complex mode amplitude over 8 seeds against √H·landau_root(kλ_D/√H).
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: RHO_STAGGER_2D
include(joinpath(REPO, "scripts", "crosscode", "analysis.jl"))
using .Observables: damping_from_peaks, growth_from_window
include(joinpath(EM_TEST, "theory.jl"))
using .Theory: landau_root, weibel_cold_growth_2d

const CASES = joinpath(REPO, "scripts", "crosscode", "cases")
load_case(name) = TOML.parsefile(joinpath(CASES, "$name.toml"))

dft(A::AbstractVector) = [sum(A[j] * cispi(-2 * (k - 1) * (j - 1) / length(A)) for j in eachindex(A)) for k in eachindex(A)] ./ length(A)
function dft(A::AbstractMatrix)
    W(n) = [cispi(-2 * (k - 1) * (j - 1) / n) for k in 1:n, j in 1:n]
    return W(size(A, 1)) * A * transpose(W(size(A, 2))) ./ length(A)
end
transfer(m, n, p) = prod(d -> cospi((m[d] - 1) / n[d])^(2p), 1:length(n))
"The host reference filter (periodic `circshift`), as in the tests."
function binomial_host(a::AbstractArray{T,N}, passes) where {T,N}
    b = copy(a)
    for _ in 1:passes, d in 1:N
        b = (circshift(b, ntuple(k -> k == d ? 1 : 0, N)) .+ circshift(b, ntuple(k -> k == d ? -1 : 0, N)) .+ 2 .* b) ./ 4
    end
    return b
end

function measure_transfer()
    out = Dict{String,Any}()
    for (label, n, dx) in (("1D", (64,), (0.1,)), ("2D", (32, 24), (0.1, 0.15)))
        grid = PeriodicGrid(n, dx, 2)
        rows = Dict{String,Any}()
        for p in (1, 2, 4, 8)
            A = 2 .* rand(n...) .- 1
            a = field(CPU(), Float64, grid)
            interior_view(a, grid) .= A
            binomial!(a, similar(a), grid, p)
            Âi, Âo = dft(A), dft(collect(interior_view(a, grid)))
            ms = CartesianIndices(Âi)
            err = maximum(m -> abs(Âo[m] - transfer(m, n, p) * Âi[m]), ms) / (eps() * maximum(abs, A) * p * length(n))
            r = Dict{String,Any}("max_error_over_eps_p_N_max_input" => err)
            if label == "1D"
                half = 1:(n[1] ÷ 2 + 1)
                r["k_dx_over_pi"] = [(m - 1) * 2 / n[1] for m in half]
                r["measured"] = [real(Âo[m] / Âi[m]) for m in half]
                r["measured_imag"] = [imag(Âo[m] / Âi[m]) for m in half]
                r["closed"] = [transfer((m,), n, p) for m in half]
            end
            rows["p=$p"] = r
            println("transfer $label p = $p: max |Ĥ − H_p|·|Â| / (eps·p·N·max|A|) = ", round(err, sigdigits = 3))
        end
        out[label] = rows
    end
    fig = Figure(size = (640, 440))
    ax = Axis(fig[1, 1]; xlabel = "kΔx/π", ylabel = "output/input of the DFT coefficient",
              title = "Binomial filter, 64 cells: measured transfer (markers) and cos^(2p)(kΔx/2) (lines)")
    for (i, p) in enumerate((1, 2, 4, 8))
        r = out["1D"]["p=$p"]
        lines!(ax, r["k_dx_over_pi"], r["closed"]; color = Cycled(i), label = "p = $p")
        scatter!(ax, r["k_dx_over_pi"], r["measured"]; color = Cycled(i), markersize = 6)
    end
    axislegend(ax; position = :rt)
    return out, fig
end

"D⁻·E and ρ deposited now, interior arrays (as `gauss_parts` in the tests)."
function gauss_parts(st)
    (; grid, numerics) = st
    ρd = field(CPU(), Float64, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, RHO_STAGGER_2D, grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), grid.n, grid.dx
    Ex, Ey = st.fields.Ex, st.fields.Ey
    divE = [(Ex[g + i, g + j] - Ex[g + i - 1, g + j]) / h[1] + (Ey[g + i, g + j] - Ey[g + i, g + j - 1]) / h[2] for i in 1:n[1], j in 1:n[2]]
    return divE, collect(interior_view(ρd, grid))
end

"A 2D case at test size with `passes` filter passes (the case file's `filter_passes`), Float64, Reference."
function case_run(name, passes; ppc, cells, amplitude = nothing, t_end = nothing, gauss = false, seed = nothing)
    case = load_case(name)
    case["numerics"]["filter_passes"] = passes
    n0 = Tuple(case["grid"]["cells"])
    prob = problem_from_case(case, Float64; ppc, cells, dt = case["time"]["dt"] * minimum(n0 ./ cells), amplitude, t_end, seed)
    st = EMState(prob, numerics_from_case(case))
    initialize!(st, prob)
    p = params(prob)
    series = Dict{String,Vector{Float64}}("t" => Float64[], "plain" => Float64[], "invariant" => Float64[])
    divE0, ρ0 = gauss_parts(st)
    Q0, r0 = divE0 .- binomial_host(ρ0, passes), divE0 .- (ρ0 .- sum(ρ0) / length(ρ0))
    every = 10
    for k in 1:prob.nsteps
        step!(st, p, k)
        if gauss && (k % every == 0 || k == prob.nsteps)
            divE, ρ = gauss_parts(st)
            ρ̄ = sum(ρ) / length(ρ)
            push!(series["t"], k * prob.dt)
            push!(series["plain"], maximum(abs, divE .- (ρ .- ρ̄)) / maximum(abs, ρ .- ρ̄))
            Q = divE .- binomial_host(ρ, passes)
            push!(series["invariant"], maximum(abs, Q .- Q0) / (eps() * sum(prob.cells) * maximum(abs, ρ) * sqrt(k + 1)))
        end
    end
    spectrum = nothing
    if gauss
        divE, ρ = gauss_parts(st)
        r = divE .- (ρ .- sum(ρ) / length(ρ))
        R̂, D̂ = dft(r .- r0), dft(ρ .- ρ0)
        ms = [m for m in CartesianIndices(R̂) if Tuple(m) != (1, 1)]
        spectrum = Dict("one_minus_H" => [1 - transfer(m, prob.cells, passes) for m in ms],
                        "ratio" => [abs(R̂[m]) / abs(D̂[m]) for m in ms],
                        "max_abs_dev" => maximum(m -> abs(R̂[m] - (transfer(m, prob.cells, passes) - 1) * D̂[m]), ms) / maximum(abs, D̂))
    end
    return (; st, prob, case, h = histories(st), ts = (0:(prob.nsteps - 1)) .* prob.dt, series, spectrum)
end

"Seeds of the Landau ensemble (the first of the case's list)."
const LANDAU_SEEDS = 8

"NaN (a fit with fewer than three maxima) as `nothing`, for the JSON sidecars."
clean(x::AbstractFloat) = isnan(x) ? nothing : x
clean(x::AbstractDict) = Dict(k => clean(v) for (k, v) in x)
clean(x::AbstractVector) = map(clean, x)
clean(x) = x

"H_p of the case's diagnostic mode on the test grid."
mode_transfer(prob, passes) = transfer(CartesianIndex(ntuple(d -> mod(prob.kmode[d], prob.cells[d]) + 1, 2)), prob.cells, passes)

function binomial_filter_run()
    dir = run_dir("binomial-filter", @__FILE__; work_package = "WP-029", prefix = "em2d",
                  extra_inputs = [joinpath(CASES, "landau-oblique-2d.toml"), joinpath(CASES, "weibel-2d.toml"),
                                  joinpath(EM_TEST, "theory.jl"), joinpath(REPO, "scripts", "crosscode", "theory.jl"),
                                  joinpath(REPO, "scripts", "crosscode", "analysis.jl")],
                  params = Dict{String,Any}("type" => "Float64", "mode" => "Reference",
                                            "landau" => "32 × 32 cells, 16 × 16 per cell, amplitude 0.05, t_end 18; Gauss: passes 0, 1, 2, the default seed; rates: passes 0, 2, 4, the first $LANDAU_SEEDS seeds",
                                            "weibel" => "64 × 64 cells, 4 × 4 per beam per cell, passes 0, 4, 8, 16"))
    sidecar = Dict{String,Any}("cpu" => Sys.cpu_info()[1].model, "threads" => Threads.nthreads())
    tr, fig = measure_transfer()
    sidecar["transfer"] = tr
    println("wrote ", save_plot(dir, "transfer", fig, Dict("transfer" => tr)))

    # Landau: Gauss's law and the rate
    landau = Dict{String,Any}()
    fig = Figure(size = (1100, 440))
    ax1 = Axis(fig[1, 1]; xlabel = "t (1/ω₀)", ylabel = "max|D⁻·E − (ρ − ρ̄)| / max|ρ − ρ̄|",
               title = "Landau 2D, 32 × 32: plain Gauss residual")
    ax2 = Axis(fig[1, 2]; xlabel = "1 − H_p(m), closed form", ylabel = "|(r − r⁰)^| / |(ρ − ρ⁰)^|",
               title = "per Fourier mode m at t = 18")
    for (i, p) in enumerate((0, 1, 2))
        r = case_run("landau-oblique-2d", p; ppc = (16, 16), cells = (32, 32), amplitude = 1 / 20, t_end = 18, gauss = true)
        landau["gauss p=$p"] = Dict("series" => r.series, "invariant_max" => maximum(r.series["invariant"]),
                                    "plain_max" => maximum(r.series["plain"]), "spectrum_max_abs_dev" => r.spectrum["max_abs_dev"])
        println("landau gauss p = $p: plain residual ≤ ", round(maximum(r.series["plain"]), sigdigits = 3), ", invariant ≤ ",
                round(maximum(r.series["invariant"]), sigdigits = 3), " eps·scale, spectrum deviation ",
                round(r.spectrum["max_abs_dev"], sigdigits = 3))
        p == 0 && continue
        lines!(ax1, r.series["t"], r.series["plain"]; color = Cycled(i), label = "p = $p")
        scatter!(ax2, r.spectrum["one_minus_H"], r.spectrum["ratio"]; color = Cycled(i), markersize = 4, label = "p = $p")
    end
    # the rate: the seed mean of the complex mode amplitude over the case's first LANDAU_SEEDS seeds,
    # fitted by the case's rule (as its damping_rate_mode_ensemble observable)
    case = load_case("landau-oblique-2d")
    seeds = Int.(case["numerics"]["seeds"][1:LANDAU_SEEDS])
    for p in (0, 2, 4)
        local r
        A = nothing
        for seed in seeds
            r = case_run("landau-oblique-2d", p; ppc = (16, 16), cells = (32, 32), amplitude = 1 / 20, t_end = 18, seed)
            a = complex.(r.h.El_re, r.h.El_im)
            A = A === nothing ? a : A .+ a
        end
        o = case["observables"]["damping_rate_mode"]
        e = case["species"][1]
        f = damping_from_peaks(r.ts, abs2.(A ./ length(seeds)); fit_start = o["fit_start"], floor_factor = o["floor_factor"])
        H = mode_transfer(r.prob, p)
        kλ = case["wavevector"]["magnitude"] * e["thermal_velocity"] / sqrt(e["density"])
        ω = sqrt(H) * landau_root(kλ / sqrt(H))
        landau["rate p=$p"] = Dict("H_mode" => H, "gamma" => f.gamma, "omega" => f.omega, "gamma_theory" => -imag(ω),
                                   "omega_theory" => real(ω), "peaks" => length(f.t_fit), "seeds" => seeds)
        println("landau rate p = $p ($(length(seeds)) seeds): H = ", round(H, digits = 4), " γ = ", round(f.gamma, digits = 4),
                " (theory ", round(-imag(ω), digits = 4), ") ω = ", round(f.omega, digits = 4), " (theory ", round(real(ω), digits = 4),
                "), maxima ", length(f.t_fit))
    end
    lines!(ax2, [0, 1], [0, 1]; color = :gray, linestyle = :dash, label = "1 − H_p")
    axislegend(ax1; position = :rb)
    axislegend(ax2; position = :lt)
    sidecar["landau"] = landau
    println("wrote ", save_plot(dir, "gauss", fig, clean(Dict("landau" => landau))))

    # Weibel: the rate against the filtered cold theory
    weibel = Dict{String,Any}()
    for p in (0, 4, 8, 16)
        r = case_run("weibel-2d", p; ppc = (4, 4), cells = (64, 64))
        o = r.case["observables"]["growth_rate_mode"]
        v0 = abs(r.case["species"][1]["drift"][3])
        k = r.case["wavevector"]["magnitude"]
        Bt = abs2.(complex.(r.h.Bt_re, r.h.Bt_im))
        f = growth_from_window(r.ts, Bt; window = o["window"])
        H = mode_transfer(r.prob, p)
        γth = weibel_cold_growth_2d(k, v0; ωp2 = H)
        weibel["p=$p"] = Dict("H_mode" => H, "gamma" => f.gamma, "gamma_theory" => γth, "relative" => f.gamma / γth - 1)
        println("weibel p = $p: H = ", round(H, digits = 4), " γ = ", round(f.gamma, digits = 5), " theory ", round(γth, digits = 5),
                " relative ", round(f.gamma / γth - 1, sigdigits = 3))
    end
    g0, t0 = weibel["p=0"]["gamma"], weibel["p=0"]["gamma_theory"]
    for v in values(weibel)
        v["ratio_to_p0"], v["ratio_to_p0_theory"] = v["gamma"] / g0, v["gamma_theory"] / t0
    end
    sidecar["weibel"] = weibel
    fig = Figure(size = (640, 440))
    ax = Axis(fig[1, 1]; xlabel = "H_p of mode (2, 1)", ylabel = "γ / ω₀", title = "Weibel 2D, 64 × 64: γ with p filter passes")
    Hs = range(0.8, 1.0; length = 50)
    k = load_case("weibel-2d")["wavevector"]["magnitude"]
    lines!(ax, collect(Hs), [weibel_cold_growth_2d(k, 0.3; ωp2 = h) for h in Hs]; color = :gray, label = "cold theory, ω_p² → H·ω_p²")
    ps = sort(collect(keys(weibel)); by = s -> parse(Int, s[3:end]))
    scatter!(ax, [weibel[s]["H_mode"] for s in ps], [weibel[s]["gamma"] for s in ps]; label = "measured, p = 0, 4, 8, 16")
    axislegend(ax; position = :lt)
    println("wrote ", save_plot(dir, "weibel_rate", fig, clean(Dict("weibel" => weibel))))
    open(io -> JSON.print(io, clean(sidecar), 2), joinpath(dir, "summary.json"), "w")
    return dir
end

binomial_filter_run()
