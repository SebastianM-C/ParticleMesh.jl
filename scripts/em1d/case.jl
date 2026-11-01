# A 1D case through the electromagnetic module (WP-014), with the standard plot set, sidecars and
# a run manifest (F3). The case file is read unchanged; keywords only override particle counts,
# the perturbation amplitude or the end time.
#   julia +1.13 --project=scripts -t auto scripts/em1d/case.jl landau [ppc=4096] [amplitude=0.05] [t_end=25] [mode=Fast]
#   julia +1.13 --project=scripts -t auto scripts/em1d/case.jl two-stream [ppc=256]
#   julia +1.13 --project=scripts -t auto scripts/em1d/case.jl weibel
# `momenta_at=t_minus_half` takes the loaded momenta at −Δt/2 (Smilei's and PIConGPU's convention)
# instead of at t = 0 with the half-step kick back (`initialize!`); the manifest records which.
include(joinpath(@__DIR__, "common.jl"))

function parse_overrides(args)
    kw = Dict{Symbol,Any}()
    for a in args
        k, v = split(a, '='; limit = 2)
        kw[Symbol(k)] = k == "mode" ? (v == "Fast" ? Fast() : Reference()) : k == "momenta_at" ? Symbol(v) :
                        k == "ppc" ? parse(Int, v) : parse(Float64, v)
    end
    return kw
end

"The case's observable: the signal, the fit and the theory value (field-amplitude rate, in ω₀)."
function observable(name, case, ts, h)
    if name == "landau"
        o = case["observables"]["damping_rate"]
        e = case["species"][1]
        kλ = 2π / case["grid"]["length"] * e["thermal_velocity"] / sqrt(e["density"])
        ω = landau_root(kλ)
        fit = U -> damping_from_peaks(ts, U; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
        return (; signal = :WEx, label = "U(Ex)", fit, theory = -imag(ω), theory_omega = real(ω), sign = -1)
    elseif name == "two-stream"
        o = case["observables"]["growth_rate"]
        v0 = abs(case["species"][2]["drift"])
        kv0 = 2π / case["grid"]["length"] * v0 / sqrt(2case["species"][2]["density"])
        fit = U -> growth_from_window(ts, U; window = o["window"])
        return (; signal = :WEx, label = "U(Ex)", fit, theory = two_stream_cold_growth(kv0; gamma0 = 1 / sqrt(1 - v0^2)),
                theory_omega = 0.0, sign = 1)
    else
        o = case["observables"]["growth_rate"]
        v0 = abs(case["species"][1]["transverse_drift"][1])
        n0 = sum(s -> s["mobile"] ? s["density"] : 0.0, case["species"])
        fit = U -> growth_from_window(ts, U; window = o["window"])
        return (; signal = :WBz, label = "U(Bz)", fit, theory = weibel_cold_growth(2π / case["grid"]["length"], v0; ωp2 = n0),
                theory_omega = 0.0, sign = 1)
    end
end

function main(args)
    name = args[1]
    kw = parse_overrides(args[2:end])
    mode = pop!(kw, :mode, Reference())
    momenta_at = pop!(kw, :momenta_at, :t0)
    case = load_case(name)
    prob = problem_from_case(case, Float64; kw...)
    num = numerics_from_case(case; mode)
    params = Dict{String,Any}("case" => name, "overrides" => Dict(string(k) => v for (k, v) in kw), "mode" => string(mode),
                              "momenta_at" => string(momenta_at),
                              "steps" => prob.nsteps, "dt" => prob.dt, "cells" => prob.cells,
                              "ppc" => [s.ppc for s in prob.species], "amplitude" => [s.amplitude for s in prob.species])
    dir = run_dir(name, @__FILE__, name; params)
    st = EMState(prob, num)
    initialize!(st, prob; momenta_at)
    p = ParticleMeshElectromagnetic.params(prob)
    comp = name == "weibel" ? 2 : 1
    chunk = 50
    snapshot_steps = [0, chunk * ((prob.nsteps ÷ 2) ÷ chunk), prob.nsteps]   # chunk boundaries (the last chunk ends at nsteps)
    snaps = Dict(0 => phase_space(st, comp; stride = 8))
    tg, gauss = Float64[], Float64[]
    r, sc = gauss_check(st)
    push!(tg, 0.0); push!(gauss, r / (eps(Float64) * sc))
    wall = 0.0
    k = 1
    while k <= prob.nsteps
        m = min(chunk, prob.nsteps - k + 1)
        wall += @elapsed run!(st, p, m; first = k)
        k += m
        r, sc = gauss_check(st)
        push!(tg, (k - 1) * prob.dt); push!(gauss, r / (eps(Float64) * sc))
        (k - 1) in snapshot_steps && (snaps[k - 1] = phase_space(st, comp; stride = 8))
    end
    np = sum(alive_count, st.species)
    h = histories(st)
    ts = (0:(prob.nsteps - 1)) .* prob.dt
    obs = observable(name, case, ts, h)
    U = getfield(h, obs.signal)
    f = obs.fit(U)
    rel = (f.gamma - obs.theory) / obs.theory

    # 1. the observable's signal with the fit and the theory slope
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "$(obs.label) (mₑc²n_c c/ω₀)", title = "$name through ParticleMeshElectromagnetic")
    lines!(ax, ts, max.(U, 1e-30); label = "module")
    if !isempty(f.t_fit)
        scatter!(ax, f.t_fit, f.U_fit; color = :black, markersize = 4, label = "fit points, γ = $(round(f.gamma, digits = 4))")
        U0 = f.U_fit[1] * exp(-2 * obs.sign * obs.theory * f.t_fit[1])
        lines!(ax, ts, U0 .* exp.(2 * obs.sign * obs.theory .* ts); linestyle = :dot, color = :red, label = "theory γ = $(round(obs.theory, digits = 4))")
    end
    ylims!(ax, max(minimum(U[U .> 0]) / 10, maximum(U) * 1e-14), 10 * maximum(U))
    axislegend(ax; position = :rb)
    save_plot(dir, "$(name)_signal", fig, Dict(
        "signal" => string(obs.signal), "gamma_measured" => f.gamma, "gamma_theory" => obs.theory, "relative_error" => rel,
        "omega_measured" => get(f, :omega, nothing), "omega_theory" => obs.theory_omega, "fit_points" => length(f.t_fit),
        "fit_window_t" => isempty(f.t_fit) ? nothing : [first(f.t_fit), last(f.t_fit)]))

    # 2. every field energy and the kinetic energy change
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "energy (mₑc²n_c c/ω₀)", title = "$name: field energies")
    for c in (:WEx, :WEy, :WEz, :WBy, :WBz)
        v = getfield(h, c)
        any(>(0), v) && lines!(ax, ts, max.(v, 1e-30); label = string(c))
    end
    ΔK = abs.(h.K .- h.K[1])
    lines!(ax, ts, max.(ΔK, 1e-30); color = :black, linestyle = :dash, label = "|K − K₀|")
    axislegend(ax; position = :rb)
    save_plot(dir, "$(name)_energies", fig, Dict(string(c) * "_max" => maximum(getfield(h, c)) for c in (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz, :K)))

    # 3. conservation: total energy and Gauss's law
    tot = total_energy_history(h)
    drift = (tot .- tot[1]) ./ tot[1]
    fig = Figure(size = (800, 600))
    ax1 = Axis(fig[1, 1]; ylabel = "relative change of W + K", title = "$name: energy (fields + kinetic, at integer times)")
    lines!(ax1, ts[1:length(drift)], drift)
    ax2 = Axis(fig[2, 1]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "Gauss residual / (eps·n·max|ρ|)")
    lines!(ax2, tg, max.(gauss, 1e-3))
    save_plot(dir, "$(name)_conservation", fig, Dict(
        "energy_drift_max" => maximum(abs, drift), "energy_drift_final" => drift[end], "total_energy_initial" => tot[1],
        "gauss_residual_max_over_eps_n_maxrho" => maximum(gauss), "gauss_checks" => length(gauss)))

    # 4. phase space at three times
    fig = Figure(size = (1200, 400))
    for (j, s) in enumerate(snapshot_steps)
        xs, us = snaps[s]
        ax = Axis(fig[1, j]; xlabel = "x (c/ω₀)", ylabel = j == 1 ? (comp == 1 ? "uₓ (c)" : "u_y (c)") : "",
                  title = "t = $(round(s * prob.dt, digits = 2))/ω₀")
        hexbin!(ax, xs, us; bins = 64, colorscale = log10)
    end
    save_plot(dir, "$(name)_phase_space", fig, Dict("times" => [s * prob.dt for s in snapshot_steps], "component" => comp,
                                                    "samples_per_panel" => length(snaps[0][1])))

    summary = Dict("case" => name, "mode" => string(mode), "particles" => np, "steps" => prob.nsteps, "wall_seconds" => wall,
                   "ns_per_particle_step" => wall / (np * prob.nsteps) * 1e9, "threads" => Threads.nthreads(),
                   "gamma" => f.gamma, "gamma_theory" => obs.theory, "relative_error" => rel,
                   "energy_drift_max" => maximum(abs, drift), "gauss_max" => maximum(gauss))
    open(io -> JSON.print(io, summary, 2), joinpath(dir, "summary.json"), "w")
    println(JSON.json(summary, 2))
    println("outputs in ", relpath(dir, REPO))
end

main(ARGS)
