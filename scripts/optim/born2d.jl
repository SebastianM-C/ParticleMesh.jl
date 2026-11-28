# WP-064: is the energy of ionization-born electrons an objective one can optimize on? The case of
# born2d_case.jl (a 2D LWFA run whose pulse ionizes a nitrogen dopant's K shell inside the wake),
# its objective J (the kinetic energy the born electrons carry ahead of the window's back), and
# a scan of J in one parameter, the pulse's a₀. No optimizer runs here.
#
# 1. The base run in time: J, the born, counted and trapped weight, and the guard on the plasma's
#    own electrons (where J reads, and over the whole window ahead of the back band), every `series_every` c/ω₀, with the pulse's place and the wake's zero crossings
#    at the read-out (where the position weight has to be 1).
# 2. The scan: J at `scan` values of a₀ from plain runs, and at every second one the ForwardDiff
#    derivative from a run with duals, through the Optimization.jl problem of problem.jl; the
#    seconds of both kinds of run.
# 3. Steepness and smoothness at the base a₀: the derivative against central differences over
#    steps a₀·2⁻ᵏ, and `fine` runs spaced a₀·2⁻¹² apart, whose departure from a cubic through them
#    is the roughness a finite difference or a line search would see.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/born2d.jl [ppl]
#   julia +1.13 --project=scripts scripts/optim/born2d.jl replot <run dir>
#   julia +1.13 --project=scripts scripts/optim/born2d.jl table <run dir>
#
# Writes scripts/runs/optim-born2d-<stamp>/: manifest.toml, results.json, born2d.png + .json.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "born2d_case.jl"))
include(joinpath(@__DIR__, "problem.jl"))

"The scan's range in a₀ as a box of problem.jl: one parameter, z = a₀/2."
const BORN_BOX = (lo = [1.5], hi = [2.4], scale = [2.0])
const RUN = (a0 = 1.9, scan = 13, fine = 33, fine_step = 2.0^-12, fd_powers = 3:2:17, series_every = 5.0)

plain(s::NamedTuple) = Dict{String,Any}(string(k) => v for (k, v) in pairs(s))

"The base run stepped in pieces of `every` c/ω₀, with the summary after each, and the final state."
function time_series(a0; ppl, every = RUN.series_every)
    st, p, prob = born_start(a0; ppl)
    dt = Float64(prob.dt)
    chunk = max(1, round(Int, every / dt))
    rows = Dict{String,Any}[]
    k = 0
    while k < prob.nsteps
        n = min(chunk, prob.nsteps - k)
        run!(st, p, n; first = k + 1)
        k += n
        push!(rows, merge(plain(born_summary(st)), Dict{String,Any}("t" => k * dt, "step" => k)))
    end
    ionization_lost(st) == 0 || error("a birth found no slot")
    return rows, st
end

"The on-axis fields of the read-out: the pulse's place (the peak of |E_y|) and the zero crossings of E_x smoothed over a wavelength."
function axis_landmarks(st; ppl)
    g, n = guards(st.grid), ncells(st.grid)
    h = Float64(cellsize(st.grid)[1])
    j = g + 1 + n[2] ÷ 2
    Ey = abs.(Array(st.fields.Ey)[(g + 1):(g + n[1]), j])
    Ex = Array(st.fields.Ex)[(g + 1):(g + n[1]), j]
    sm = [mean(@view Ex[max(1, i - ppl ÷ 2):min(n[1], i + ppl ÷ 2)]) for i in 1:n[1]]
    zeros_x = [i * h for i in 2:n[1] if sm[i - 1] * sm[i] < 0]
    return Dict{String,Any}("pulse_x" => argmax(Ey) * h, "pulse_peak" => maximum(Ey), "Ex_zero_crossings" => zeros_x,
                            "x" => collect((1:n[1]) .* h), "Ex" => Ex)
end

"The weight of the born electrons at the read-out by window x (bins of 2 c/ω₀): all, and those above `u_fast`."
function born_profile(st; width = 2.0)
    h = Float64.(cellsize(st.grid))
    e = ionization_electrons(st)
    alive = Array(e.status) .== ALIVE
    x = (Array(e.cell[1])[alive] .- 1 .+ Array(e.offset[1])[alive]) .* h[1]
    w, ux = Array(e.w)[alive] .* (h[1] * h[2]), Array(e.u[1])[alive]
    edges = 0.0:width:(ncells(st.grid)[1] * h[1])
    bin(v) = [sum(v[(x .>= a) .& (x .< a + width)]; init = 0.0) for a in edges]
    return Dict{String,Any}("x" => collect(edges) .+ width / 2, "weight" => bin(w), "fast" => bin(w .* (ux .> BCASE.u_fast)),
                            "energy" => bin(w .* (sqrt.(1 .+ ux .^ 2 .+ Array(e.u[2])[alive] .^ 2 .+ Array(e.u[3])[alive] .^ 2) .- 1)))
end

"The residual of `J` from the least-squares cubic through it, over the points `x`."
function cubic_residual(x, J)
    t = (x .- mean(x)) ./ (maximum(x) - minimum(x))
    A = hcat(ones(length(t)), t, t .^ 2, t .^ 3)
    return J .- A * (A \ J)
end

function main(ppl)
    cfg = Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v isa NamedTuple ? plain(v) : v) for (k, v) in pairs(BCASE))
    cfg["ppl"] = ppl
    cfg["run"] = Dict{String,Any}("a0" => RUN.a0, "scan" => RUN.scan, "fine" => RUN.fine, "fine_step" => RUN.fine_step,
                                  "fd_powers" => collect(RUN.fd_powers), "box" => [BORN_BOX.lo[1], BORN_BOX.hi[1]])
    dir = run_dir("born2d", @__FILE__; prefix = "optim", work_package = "WP-064",
                  extra_inputs = [joinpath(@__DIR__, "born2d_case.jl"), joinpath(@__DIR__, "problem.jl"),
                                  joinpath(REPO, "lib", "Wakefields", "src", "atomic.jl"), joinpath(REPO, "lib", "Wakefields", "src", "cases2d.jl")],
                  params = cfg)
    out = Dict{String,Any}("case" => cfg, "threads" => Threads.nthreads())
    checks = Dict{String,Any}[]
    check(name, ok, detail) = (push!(checks, Dict("check" => name, "pass" => ok, "detail" => detail)); println(ok ? "PASS " : "FAIL ", name, ": ", detail); flush(stdout))
    save() = open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w")
    t0 = time()
    J(a0) = born_objective(a0; ppl)

    # 1. The base run in time.
    rows, st = time_series(RUN.a0; ppl)
    last_row = rows[end]
    born_total = maximum(r["born"] for r in rows)
    out["series"] = rows
    out["readout"] = merge(axis_landmarks(st; ppl), Dict{String,Any}("profile" => born_profile(st), "born_total" => born_total,
                           "back_band" => window_back_band(st) * Float64(cellsize(st.grid)[1]), "steps" => rows[end]["step"]))
    println("base run a₀ = $(RUN.a0): J = $(last_row["energy"]), born $(born_total), trapped $(last_row["fast_counted"]), plasma above $(BCASE.u_fast): $(last_row["plasma_fast"])")
    check("the base run traps born electrons", last_row["fast_counted"] > 0.2 * born_total, "trapped weight $(last_row["fast_counted"]) of $(born_total) born")
    check("the trapped electrons are where the position weight is 1", last_row["fast_counted"] >= (1 - 1e-3) * last_row["fast"],
          "trapped and counted $(last_row["fast_counted"]), trapped $(last_row["fast"])")
    check("the plasma does not inject itself in the base run", last_row["plasma_fast_window"] == 0,
          "plasma weight above $(BCASE.u_fast) mₑc² in the whole window $(last_row["plasma_fast_window"]), highest plasma energy $(last_row["plasma_peak_window"])")
    save()

    # 2. The scan in a₀, through the Optimization.jl problem (z = a₀/2).
    prob = case_problem(J, BORN_BOX, (RUN.a0,))
    a0s = collect(range(BORN_BOX.lo[1], BORN_BOX.hi[1]; length = RUN.scan))
    scan = Dict{String,Any}[]
    for (i, a) in enumerate(a0s)
        stp = born_state(a; ppl)
        row = merge(plain(born_summary(stp)), Dict{String,Any}("a0" => a))
        if isodd(i)
            v, g = value_gradient(prob, [a / BORN_BOX.scale[1]])
            row["dJ_da0"] = g[1] / BORN_BOX.scale[1]
            row["J_dual"] = v
        end
        push!(scan, row)
        println("a₀ = $(round(a; digits = 4)): J = $(row["energy"])  dJ/da₀ = $(get(row, "dJ_da0", "-"))  trapped $(row["fast_counted"])  plasma fast $(row["plasma_fast"]) peak $(row["plasma_peak"])"); flush(stdout)
        out["scan"] = scan
        save()
    end
    check("at every a₀ the trapped born electrons are where the position weight is 1", all(r -> r["fast_counted"] >= (1 - 1e-3) * r["fast"], scan),
          "smallest counted share of the trapped weight: $(minimum(r["fast_counted"] / r["fast"] for r in scan))")
    check("the run with duals returns the plain run's J", all(r -> !haskey(r, "J_dual") || r["J_dual"] == r["energy"], scan), "at $(count(r -> haskey(r, "J_dual"), scan)) points")

    # 3. At the base a₀: the derivative against central differences, and the fine scan.
    z0 = [RUN.a0 / BORN_BOX.scale[1]]
    J0, g0 = value_gradient(prob, z0)
    ad = g0[1] / BORN_BOX.scale[1]
    fd = map(RUN.fd_powers) do k
        h = RUN.a0 * 2.0^-k
        d = (prob.f([(RUN.a0 + h) / BORN_BOX.scale[1]], prob.p) - prob.f([(RUN.a0 - h) / BORN_BOX.scale[1]], prob.p)) / 2h
        println("  h = a₀·2^-$k: FD $d, AD $ad, relative difference $(abs(d - ad) / abs(ad))"); flush(stdout)
        Dict{String,Any}("power" => k, "h" => h, "fd" => d, "relative_difference" => abs(d - ad) / abs(ad))
    end
    best = minimum(r["relative_difference"] for r in fd)
    # A7's floor: the spread of the three neighbouring difference quotients around the best step.
    ib = clamp(argmin([r["relative_difference"] for r in fd]), 2, length(fd) - 1)
    near = [fd[i]["fd"] for i in (ib - 1):(ib + 1)]
    out["derivative"] = Dict{String,Any}("a0" => RUN.a0, "J" => J0, "forwarddiff" => ad, "central" => fd, "best_relative_difference" => best,
                                         "floor" => (maximum(near) - minimum(near)) / abs(ad), "log_slope" => ad * RUN.a0 / J0)
    check("ForwardDiff agrees with a central difference", best < 1e-4, "best |AD − FD|/|AD| = $best over h = a₀·2^-$(RUN.fd_powers)")
    save()
    δ = RUN.a0 * RUN.fine_step
    xf = RUN.a0 .+ δ .* ((1:RUN.fine) .- (RUN.fine + 1) ÷ 2)
    Jf = [prob.f([x / BORN_BOX.scale[1]], prob.p) for x in xf]
    res = cubic_residual(xf, Jf)
    step_change = abs(ad) * δ
    out["fine"] = Dict{String,Any}("a0" => xf, "J" => Jf, "spacing" => δ, "residual" => res, "step_change" => step_change,
                                   "roughness" => maximum(abs, res) / step_change, "roughness_rms" => sqrt(mean(abs2, res)) / step_change,
                                   "tangent_mismatch" => abs((Jf[end] - Jf[1]) / (xf[end] - xf[1]) - ad) / abs(ad))
    println("fine scan: largest departure from a cubic $(maximum(abs, res)), $(out["fine"]["roughness"]) of the change over one spacing")

    out["cost"] = Dict{String,Any}("plain" => plain(runs(prob.p.timer, "plain run")), "dual" => plain(runs(prob.p.timer, "run with duals")))
    out["seconds"] = time() - t0
    out["checks"] = checks
    save()
    plot_results(dir, out)
    println(report_table(dir))
    println("done in $(round(out["seconds"]; digits = 1)) s: $dir")
    failed = [c["check"] for c in checks if !c["pass"]]
    isempty(failed) || error("WP-064 checks failed: $(join(failed, "; ")) (results in $dir)")
    return dir
end

function plot_results(dir, out)
    fig = Figure(size = (1500, 860))
    s = out["series"]
    t = [r["t"] for r in s]
    ax = Axis(fig[1, 1]; xlabel = "time (1/ω₀)", ylabel = "weight (n_c·(c/ω₀)²)", title = "Base run, a₀ = $(out["derivative"]["a0"]): born electrons in time")
    lines!(ax, t, [r["born"] for r in s]; label = "released by the atoms in the window", color = :gray, linestyle = :dash)
    lines!(ax, t, [r["alive"] for r in s]; label = "born, in the window")
    lines!(ax, t, [r["counted"] for r in s]; label = "counted by the position weight")
    lines!(ax, t, [r["fast_counted"] for r in s]; label = "counted, u_x > $(out["case"]["u_fast"])")
    lines!(ax, t, [r["plasma_fast"] for r in s]; label = "plasma above $(out["case"]["u_fast"]) mₑc², where J reads", color = :crimson)
    axislegend(ax; position = :rc, labelsize = 11)
    axJ = Axis(fig[1, 2]; xlabel = "time (1/ω₀)", ylabel = "J (n_c·mₑc²·(c/ω₀)²)", title = "The objective and the guard in time")
    lines!(axJ, t, [r["energy"] for r in s]; label = "J")
    axG = Axis(fig[1, 2]; yaxisposition = :right, ylabel = "γ − 1 (mₑc²)", ygridvisible = false, xgridvisible = false)
    hidexdecorations!(axG)
    lines!(axG, t, [r["plasma_peak"] for r in s]; color = :crimson, label = "plasma's highest energy")
    lines!(axG, t, [r["mean_energy"] for r in s]; color = :seagreen, label = "counted born electrons' mean energy")
    axislegend(axG; position = :lt)
    r = out["readout"]
    axP = Axis(fig[1, 3]; xlabel = "window x (c/ω₀)", ylabel = "born weight per 2 c/ω₀", title = "Read-out: where the born electrons are")
    barplot!(axP, Float64.(r["profile"]["x"]), Float64.(r["profile"]["weight"]); color = (:steelblue, 0.5), label = "all")
    barplot!(axP, Float64.(r["profile"]["x"]), Float64.(r["profile"]["fast"]); color = :steelblue, label = "u_x > $(out["case"]["u_fast"])")
    fade = Float64.(out["case"]["fade"])
    xs = range(0, maximum(Float64.(r["x"])); length = 400)
    axW = Axis(fig[1, 3]; yaxisposition = :right, ylabel = "position weight; E_x on the axis", ygridvisible = false, xgridvisible = false)
    hidexdecorations!(axW)
    lines!(axW, xs, smooth_rise.(xs, fade[1], fade[2]); color = :black, label = "position weight S(x)")
    lines!(axW, Float64.(r["x"]), Float64.(r["Ex"]); color = (:orange, 0.8), label = "E_x")
    vlines!(axW, [r["back_band"]]; color = :gray, linestyle = :dot, label = "back band's end")
    vlines!(axW, [r["pulse_x"]]; color = :crimson, linestyle = :dot, label = "pulse")
    axislegend(axW; position = :rb)
    axislegend(axP; position = :lt)

    sc = out["scan"]
    a = [q["a0"] for q in sc]
    Js = [q["energy"] for q in sc]
    ax2 = Axis(fig[2, 1]; xlabel = "a₀", ylabel = "J (n_c·mₑc²·(c/ω₀)²)", yscale = log10, title = "J against a₀: plain runs and ForwardDiff tangents")
    scatterlines!(ax2, a, Js; color = :black, markersize = 7)
    da = (a[2] - a[1]) / 2
    for q in sc
        haskey(q, "dJ_da0") || continue
        xt = [q["a0"] - da, q["a0"] + da]
        yt = q["energy"] .+ q["dJ_da0"] .* (xt .- q["a0"])
        all(>(0), yt) && lines!(ax2, xt, yt; color = :crimson, linewidth = 2.5)
    end
    ax3 = Axis(fig[2, 2]; xlabel = "a₀", ylabel = "weight (n_c·(c/ω₀)²)", yscale = log10, title = "Trapped charge: born electrons and the plasma's own")
    scatterlines!(ax3, a, [q["fast_counted"] for q in sc]; label = "born, counted, u_x > $(out["case"]["u_fast"])")
    pf = [q["plasma_fast"] for q in sc]
    any(>(0), pf) && scatterlines!(ax3, a[pf .> 0], pf[pf .> 0]; color = :crimson, label = "plasma above $(out["case"]["u_fast"]) mₑc², where J reads")
    pw = [q["plasma_fast_window"] for q in sc]
    any(>(0), pw) && scatterlines!(ax3, a[pw .> 0], pw[pw .> 0]; color = :crimson, linestyle = :dash, label = "plasma above $(out["case"]["u_fast"]) mₑc², whole window")
    axislegend(ax3; position = :rb)
    d = out["derivative"]
    ax4 = Axis(fig[2, 3]; xlabel = "finite-difference step h", ylabel = "|FD − AD| / |AD|", xscale = log2, yscale = log10,
               title = "dJ/da₀ at a₀ = $(d["a0"]): central differences against ForwardDiff")
    scatterlines!(ax4, [q["h"] for q in d["central"]], [max(q["relative_difference"], 1e-16) for q in d["central"]])
    save_plot(dir, "born2d", fig, out)
end

sig(x) = string(round(x; sigdigits = 4))

"""
    report_table(dir) -> String

The tables of reports/wp-064-born-electron-objective.md from the figure's sidecar `born2d.json` in the run directory `dir`.
"""
function report_table(dir)
    out = JSON.parsefile(joinpath(dir, "born2d.json"))
    io = IOBuffer()
    println(io, "| a₀ | J | dJ/da₀ (ForwardDiff) | born electrons trapped | born mean energy | plasma above $(out["case"]["u_fast"]) mₑc², where J reads: weight | its energy | its highest γ − 1 | plasma above $(out["case"]["u_fast"]) mₑc², whole window: weight | its energy |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for q in out["scan"]
        println(io, "| ", sig(q["a0"]), " | ", sig(q["energy"]), " | ", haskey(q, "dJ_da0") ? sig(q["dJ_da0"]) : "", " | ", sig(q["fast_counted"]), " | ",
                sig(q["mean_energy"]), " | ", sig(q["plasma_fast"]), " | ", sig(q["plasma_fast_energy"]), " | ", sig(q["plasma_peak"]), " | ",
                sig(q["plasma_fast_window"]), " | ", sig(q["plasma_fast_energy_window"]), " |")
    end
    d, f, c, r = out["derivative"], out["fine"], out["cost"], out["readout"]
    last_row = out["series"][end]
    println(io)
    println(io, "| Quantity | Value |")
    println(io, "|---|---|")
    println(io, "| Grid, steps, threads | λ₀/", out["case"]["ppl"], ", ", r["steps"], " steps, ", out["threads"], " threads |")
    println(io, "| Base run (a₀ = ", sig(d["a0"]), "): born weight, trapped and counted | ", sig(r["born_total"]), ", ", sig(last_row["fast_counted"]), " (",
            sig(100 * last_row["fast_counted"] / r["born_total"]), " %) |")
    println(io, "| Base run: plasma electrons above ", out["case"]["u_fast"], " mₑc² where J reads (window x ≥ ", out["case"]["fade"][1], "), against the trapped born weight | ", sig(last_row["plasma_fast"]), " (",
            sig(100 * last_row["plasma_fast"] / last_row["fast_counted"]), " %) |")
    println(io, "| Base run: the same over the whole window ahead of the back band | ", sig(last_row["plasma_fast_window"]), " (",
            sig(last_row["plasma_fast_window"] / last_row["fast_counted"]), " times) |")
    println(io, "| J and dJ/da₀ at the base | ", sig(d["J"]), ", ", sig(d["forwarddiff"]), " |")
    println(io, "| Steepness d ln J / d ln a₀ | ", sig(d["log_slope"]), " |")
    println(io, "| Best abs(FD − AD)/abs(AD), at h | ", sig(d["best_relative_difference"]), ", ", sig(d["central"][argmin([q["relative_difference"] for q in d["central"]])]["h"]), " |")
    println(io, "| The floor of A7: the spread of the three difference quotients around that step, over abs(AD) | ", sig(d["floor"]), " |")
    println(io, "| Fine scan (", length(f["a0"]), " runs, spacing ", sig(f["spacing"]), "): largest and RMS departure from a cubic, in changes of J per spacing | ",
            sig(f["roughness"]), ", ", sig(f["roughness_rms"]), " |")
    println(io, "| Seconds per plain run (", c["plain"]["n"], " runs) | ", sig(c["plain"]["seconds"]), " |")
    println(io, "| Seconds per run with one dual (", c["dual"]["n"], " runs) | ", sig(c["dual"]["seconds"]), " |")
    return String(take!(io))
end

if length(ARGS) == 2 && ARGS[1] == "replot"
    plot_results(ARGS[2], JSON.parsefile(joinpath(ARGS[2], "born2d.json")))
elseif length(ARGS) == 2 && ARGS[1] == "table"
    print(report_table(ARGS[2]))
else
    main(isempty(ARGS) ? 16 : parse(Int, ARGS[1]))
end
