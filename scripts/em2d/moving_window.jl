# The moving window (WP-028, the measurements behind T-472 and T-473).
#
# 1. A vacuum pulse with and without the window (window_vacuum.png): the pulse of `window_lwfa`
#    (lib/ParticleMeshElectromagnetic/test/window_cases.jl, no plasma) in the 64-column window and
#    in its static twin, a box long enough for the whole run with a window that never moves.
#    Per window column, the largest |ΔE|, |ΔB| over the components against the twin's columns at
#    the same lab position, relative to the twin's largest field in the window, at the end of the
#    run and as the maximum over the run; and the largest difference on the columns 25 … 64 and on
#    the back 8 columns against time.
# 2. Gauss's law across shifts (window_gauss.png): the laser–plasma run of T-473 (Float64,
#    Reference, 200 steps, about 90 shifts), per window column, the largest residual of
#    D⁻·E − (ρ − ρ₀(s)) over the run in units of (n₁ + n₂)·max|ρ| (T-360's scale), and the
#    residual against time on the columns away from the back nodes and the antenna, with the
#    alive particles and the shifts.
# 3. The cost of a shift (in the sidecar of window_gauss): the median time of a step that shifts
#    and of one that does not, and of `shift_window!` alone, on the same state.
#   julia +1.13 --project=scripts scripts/em2d/moving_window.jl
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
const ParticleMeshCore = ParticleMesh.ParticleMeshCore   # window_cases.jl names it; the scripts reach the engine through ParticleMesh
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: shift_window!
using ForwardDiff: ForwardDiff
components(x::Real) = (x,)
components(d::ForwardDiff.Dual) = (ForwardDiff.value(d), Tuple(ForwardDiff.partials(d))...)
absmax(a) = maximum(abs, components(a))
value_of(x::Real) = x
include(joinpath(EM_TEST, "window_cases.jl"))

const FIELDS = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)

function vacuum_rows(; nsteps = 200)
    prob = window_lwfa(Float64; plasma = false, nsteps)
    n = prob.cells
    win = window_state(prob)
    s_end = window_shifts(win, nsteps)
    ref = window_state(static_twin(prob, n[1] + s_end + 8))
    p = params(prob)
    g = guards(win.grid)
    J = (g + 1):(g + n[2])
    over_run = zeros(n[1])
    hist = Dict{String,Any}[]
    final = zeros(n[1])
    for k in 1:nsteps
        step!(win, p, k)
        step!(ref, p, k)
        s = window_shifts(win, k)
        m = maximum(c -> maximum(abs, getfield(ref.fields, c)[(g + 1 + s):(g + n[1] + s), J]), FIELDS)
        m > 0 || continue
        col = [maximum(c -> maximum(abs, getfield(win.fields, c)[g + i, J] .- getfield(ref.fields, c)[g + i + s, J]), FIELDS) / m
               for i in 1:n[1]]
        over_run .= max.(over_run, col)
        final = col
        push!(hist, Dict{String,Any}("step" => k, "t" => k * prob.dt, "shifts" => s, "max_field" => m,
                                     "front_columns" => maximum(col[25:end]), "back_columns" => maximum(col[1:8])))
    end
    return Dict{String,Any}("cells" => collect(n), "steps" => nsteps, "shifts" => s_end, "dt" => prob.dt,
                            "start" => prob.window.start, "per_column_final" => final, "per_column_over_run" => over_run,
                            "history" => hist)
end

function gauss_rows(; nsteps = 200)
    prob = window_lwfa(Float64; nsteps)
    st = window_state(prob)
    p = params(prob)
    ia = st.aux.antennas[1].ia
    n = prob.cells
    worst = zeros(n[1])
    hist = Dict{String,Any}[]
    for k in 1:nsteps
        step!(st, p, k)
        s = window_shifts(st, k)
        r = window_gauss(st, s)
        per = [maximum(r.res[i, :]) / r.scale for i in 1:n[1]]
        worst .= max.(worst, per)
        skip = (1, 2, ia - 1 - s, ia - s)
        push!(hist, Dict{String,Any}("step" => k, "t" => k * prob.dt, "shifts" => s,
                                     "off_edges" => maximum(per[i] for i in 1:n[1] if !(i in skip)),
                                     "back_nodes" => maximum(per[1:2]), "alive" => alive_count(st.species[1]),
                                     "lost" => window_lost(st)[1]))
    end
    return Dict{String,Any}("cells" => collect(n), "steps" => nsteps, "antenna_column" => ia, "per_column_worst" => worst,
                            "history" => hist, "eps" => eps(Float64))
end

"Median seconds of a step that shifts, of one that does not, and of `shift_window!`, on the T-473 state after 150 steps."
function shift_cost()
    prob = window_lwfa(Float64; nsteps = 400)
    st = window_state(prob)
    p = params(prob)
    run!(st, p, 150)
    shifting = Float64[]
    plain = Float64[]
    for k in 151:330
        t = @elapsed step!(st, p, k)
        push!(window_shifts(st, k) > window_shifts(st, k - 1) ? shifting : plain, t)
    end
    s = window_shifts(st, 330)
    shift_alone = [(@elapsed shift_window!(st, s)) for _ in 1:20]
    return Dict{String,Any}("step_with_shift_s" => median(shifting), "step_without_shift_s" => median(plain),
                            "shift_window_s" => median(shift_alone), "steps_with_shift" => length(shifting),
                            "steps_without_shift" => length(plain), "particles" => alive_count(st.species[1]),
                            "cells" => collect(prob.cells))
end

function window_report()
    dir = run_dir("moving-window", @__FILE__; work_package = "WP-028", prefix = "em2d",
                  extra_inputs = [joinpath(EM_TEST, "window_cases.jl")],
                  params = Dict{String,Any}("case" => "window_lwfa", "cells" => [64, 16], "cells_per_wavelength" => 8,
                                            "courant" => 0.95, "steps" => 200, "type" => "Float64", "mode" => "Reference"))
    vac = vacuum_rows()
    fig = Figure(size = (1000, 430))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "window column", ylabel = "max |ΔE, ΔB| / max |E, B|",
              title = "window against its static twin, vacuum pulse ($(vac["shifts"]) shifts)")
    lines!(ax, 1:length(vac["per_column_over_run"]), max.(vac["per_column_over_run"], 1e-300); label = "maximum over the run")
    lines!(ax, 1:length(vac["per_column_final"]), max.(vac["per_column_final"], 1e-300); label = "at the end")
    hlines!(ax, [eps(Float64)]; color = :gray, linestyle = :dot, label = "eps(Float64)")
    vlines!(ax, [25]; color = :gray, linestyle = :dash, label = "T-472 checks columns ≥ 25")
    axislegend(ax; position = :rt, labelsize = 10)
    ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "max |ΔE, ΔB| / max |E, B|",
               title = "against time (window starts at t = $(vac["start"]))")
    t = [h["t"] for h in vac["history"]]
    lines!(ax2, t, [max(h["back_columns"], 1e-300) for h in vac["history"]]; label = "back 8 columns")
    lines!(ax2, t, [max(h["front_columns"], 1e-300) for h in vac["history"]]; label = "columns 25 … 64")
    axislegend(ax2; position = :rb, labelsize = 10)
    p1 = save_plot(dir, "window_vacuum", fig, merge(vac, Dict("cpu" => cpu_model(), "threads" => Threads.nthreads())))
    println("wrote ", p1)
    gau = gauss_rows()
    cost = shift_cost()
    fig2 = Figure(size = (1000, 430))
    ax3 = Axis(fig2[1, 1]; yscale = log10, xlabel = "window column", ylabel = "max |D⁻·E − (ρ − ρ₀)| / ((n₁ + n₂) max|ρ|)",
               title = "Gauss's law per column, maximum over the run")
    lines!(ax3, 1:length(gau["per_column_worst"]), max.(gau["per_column_worst"], 1e-300))
    hlines!(ax3, [eps(Float64)]; color = :gray, linestyle = :dot, label = "eps(Float64)")
    axislegend(ax3; position = :rt, labelsize = 10)
    ax4 = Axis(fig2[1, 2]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "residual / ((n₁ + n₂) max|ρ|)",
               title = "against time")
    t2 = [h["t"] for h in gau["history"]]
    lines!(ax4, t2, [max(h["off_edges"], 1e-300) for h in gau["history"]]; label = "off nodes 1, 2 and the antenna")
    lines!(ax4, t2, [max(h["back_nodes"], 1e-300) for h in gau["history"]]; label = "nodes 1 and 2")
    axislegend(ax4; position = :rb, labelsize = 10)
    p2 = save_plot(dir, "window_gauss", fig2, merge(gau, Dict("cost" => cost, "cpu" => cpu_model(), "threads" => Threads.nthreads())))
    println("wrote ", p2)
    println(cost)
    return vac, gau, cost
end

window_report()
