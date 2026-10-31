# Cold two-stream instability (A5, WP-010 T-142): the case of the test suite, with the standard plots.
#   julia +1.13 --project=scripts -t auto scripts/es1d/two_stream.jl
include(joinpath(@__DIR__, "common.jl"))

function two_stream(; kw = TWO_STREAM_KW)
    st, p, info = two_stream_case(Float64; kw...)
    params = Dict(string(k) => string(v) for (k, v) in pairs(kw))
    dir = run_dir("two_stream", @__FILE__; params)
    γth = two_stream_growth(Float64(info.kv0_ωp))
    snapshot_steps = [round(Int, 15 / Float64(info.dt)), round(Int, 25 / Float64(info.dt)), info.nsteps]
    snaps = Dict{Int,Any}()
    initialize!(st, p)
    t_wall = @elapsed ts, diags = run_with_diagnostics!(st, p, info.nsteps; every = 10, snapshots = snapshot_steps,
                                                       on_snapshot = (k, s) -> (snaps[k] = (phase_space(s; stride = 4), Array(s.Ehalf))))
    A = mode_amplitude_history(st)
    th = (0:(info.nsteps - 1)) .* Float64(info.dt)
    fit = fit_growth(th, A)
    rel = (fit.γ - γth) / γth

    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t ω_p", ylabel = "|E₁| (mₑcω₀/e)", title = "Cold two-stream, k v₀/ω_p = $(Float64(info.kv0_ωp))")
    lines!(ax, th, max.(A, 1e-12); label = "ParticleMesh |E₁|")
    sel = findall(t -> fit.window[1] <= t <= fit.window[2], th)
    k0 = sel[1]
    vspan!(ax, fit.window...; color = (:gray, 0.2), label = "fit window")
    lines!(ax, th, A[k0] .* exp.(fit.γ .* (th .- th[k0])); linestyle = :dash, label = "fit γ = $(round(fit.γ, digits = 4))")
    lines!(ax, th, A[k0] .* exp.(γth .* (th .- th[k0])); linestyle = :dot, color = :red, label = "theory γ = $(round(γth, digits = 4))")
    ylims!(ax, 1e-6, 2 * maximum(A))
    axislegend(ax; position = :rb)
    save_plot(dir, "two_stream_growth", fig, Dict(
        "gamma_measured" => fit.γ, "gamma_theory" => γth, "relative_error" => rel, "fit_window" => collect(fit.window),
        "saturation_time" => fit.tsat, "saturation_amplitude" => fit.Asat, "A_initial" => A[1]))

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
    save_plot(dir, "two_stream_conservation", fig, Dict(
        "energy_drift_max" => maximum(abs, drift), "energy_drift_final" => drift[end],
        "gauss_residual_max_over_eps_scale" => maximum(gauss)))

    fig = Figure(size = (1200, 400))
    for (j, k) in enumerate(snapshot_steps)
        (xs, vs), _ = snaps[k]
        ax = Axis(fig[1, j]; xlabel = "x (c/ω₀)", ylabel = j == 1 ? "v_x (c)" : "", title = "t ω_p = $(round(k * Float64(info.dt), digits = 2))")
        scatter!(ax, xs, vs; markersize = 1.5, color = (:black, 0.4))
    end
    save_plot(dir, "two_stream_phase_space", fig, Dict("times" => [k * Float64(info.dt) for k in snapshot_steps]))

    open(io -> JSON.print(io, Dict("wall_seconds" => t_wall, "particles" => 2 * info.n * info.ppc, "steps" => info.nsteps), 2),
         joinpath(dir, "summary.json"), "w")
    println("two-stream: γ = $(fit.γ) (theory $γth, rel $rel); outputs in $dir")
    return dir
end

abspath(PROGRAM_FILE) == @__FILE__() && two_stream()
