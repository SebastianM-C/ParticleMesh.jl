# Landau damping (A5, WP-010 T-141): the case of the test suite, with the standard plots.
#   julia +1.13 --project=scripts -t auto scripts/es1d/landau.jl
include(joinpath(@__DIR__, "common.jl"))

function landau(; kw = LANDAU_KW)
    st, p, info = landau_case(Float64; kw...)
    params = Dict(string(k) => string(v) for (k, v) in pairs(merge(kw, (; fit_tmax = LANDAU_FIT_TMAX))))
    dir = run_dir("landau", @__FILE__; params)
    ωth = landau_root(Float64(info.kλD))
    snaps = Dict{Int,Any}()
    snapshot_steps = [10, info.nsteps ÷ 2, info.nsteps]          # multiples of the diagnostic chunk
    initialize!(st, p)
    snaps[0] = phase_space(st; stride = 64)
    t_wall = @elapsed ts, diags = run_with_diagnostics!(st, p, info.nsteps; every = 10, snapshots = snapshot_steps,
                                                       on_snapshot = (k, s) -> (snaps[k] = (phase_space(s; stride = 64), Array(s.Ehalf))))
    A = mode_amplitude_history(st)
    W = field_energy_history(st)
    th = (0:(info.nsteps - 1)) .* Float64(info.dt)
    fit = fit_damping(th, A; ω_guess = real(ωth), tmax = LANDAU_FIT_TMAX)

    # 1. mode amplitude with the fit and the theory slope
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t ω_p", ylabel = "|E₁| (mₑcω₀/e)", title = "Landau damping, kλ_D = $(Float64(info.kλD))")
    lines!(ax, th, max.(A, 1e-12); label = "ParticleMesh |E₁|")
    scatter!(ax, fit.tpeaks, fit.Apeaks; color = :black, markersize = 6, label = "peaks used in the fit (t ≤ $(LANDAU_FIT_TMAX))")
    a0 = fit.Apeaks[1] * exp(-fit.γ * fit.tpeaks[1])
    lines!(ax, th, a0 .* exp.(fit.γ .* th); linestyle = :dash, label = "fit γ = $(round(fit.γ, digits = 4))")
    lines!(ax, th, a0 .* exp.(imag(ωth) .* th); linestyle = :dot, color = :red, label = "theory γ = $(round(imag(ωth), digits = 4))")
    ylims!(ax, 1e-5, 2 * maximum(A))
    axislegend(ax; position = :rt)
    rel_γ = (fit.γ - imag(ωth)) / imag(ωth)
    rel_ω = (fit.ω - real(ωth)) / real(ωth)
    save_plot(dir, "landau_mode_amplitude", fig, Dict(
        "gamma_measured" => fit.γ, "omega_measured" => fit.ω, "gamma_theory" => imag(ωth), "omega_theory" => real(ωth),
        "relative_error_gamma" => rel_γ, "relative_error_omega" => rel_ω, "peaks" => length(fit.tpeaks),
        "fit_tmax" => LANDAU_FIT_TMAX, "A_initial" => A[1], "A_final" => A[end]))

    # 2. energy and Gauss residual versus time
    K = [d.kinetic for d in diags]
    F = [d.field for d in diags]
    tot = K .+ F
    drift = (tot .- tot[1]) ./ tot[1]
    gauss = [d.gauss / (eps(Float64) * d.gauss_scale) for d in diags]
    fig = Figure(size = (800, 600))
    ax1 = Axis(fig[1, 1]; ylabel = "relative change of K + W", title = "Energy (K at t - dt/2, W at t)")
    lines!(ax1, ts, drift)
    ax2 = Axis(fig[2, 1]; yscale = log10, xlabel = "t ω_p", ylabel = "Gauss residual / (eps·n·max|ρ|)")
    lines!(ax2, ts, max.(gauss, 1e-3))
    save_plot(dir, "landau_conservation", fig, Dict(
        "energy_drift_max" => maximum(abs, drift), "energy_drift_final" => drift[end],
        "gauss_residual_max_over_eps_scale" => maximum(gauss), "kinetic_initial" => K[1], "field_initial" => F[1]))

    # 3. phase space at three times
    fig = Figure(size = (1200, 400))
    for (j, k) in enumerate(snapshot_steps)
        (xs, vs), _ = snaps[k]
        ax = Axis(fig[1, j]; xlabel = "x (c/ω₀)", ylabel = j == 1 ? "v_x (c)" : "", title = "t ω_p = $(round(k * Float64(info.dt), digits = 2))")
        hexbin!(ax, xs, vs; bins = 64, colorscale = log10)
    end
    save_plot(dir, "landau_phase_space", fig, Dict("times" => [k * Float64(info.dt) for k in snapshot_steps], "samples_per_panel" => length(snaps[0][1])))

    # 4. field lineouts
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1]; xlabel = "x (c/ω₀)", ylabel = "E (mₑcω₀/e)", title = "E(x) at the half nodes")
    xh = ((1:info.n) .- 0.5 .+ 0.5) .* Float64(cellsize(st.grid))
    for k in snapshot_steps
        lines!(ax, xh, snaps[k][2]; label = "t ω_p = $(round(k * Float64(info.dt), digits = 2))")
    end
    axislegend(ax)
    save_plot(dir, "landau_field", fig, Dict("max_abs_E" => Dict(string(k) => maximum(abs, snaps[k][2]) for k in snapshot_steps)))

    open(io -> JSON.print(io, Dict("wall_seconds" => t_wall, "particles" => info.n * info.ppc, "steps" => info.nsteps,
                                    "field_energy_initial" => W[1], "field_energy_final" => W[end]), 2),
         joinpath(dir, "summary.json"), "w")
    println("landau: γ = $(fit.γ) (theory $(imag(ωth)), rel $(rel_γ)), ω = $(fit.ω) (theory $(real(ωth)), rel $(rel_ω)); outputs in $dir")
    return dir
end

abspath(PROGRAM_FILE) == @__FILE__() && landau()
