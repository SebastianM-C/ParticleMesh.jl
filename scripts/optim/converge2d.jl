# WP-051: the convergence study of WP-045's 2D optimization, with the read-out-free slab objective
# of slab.jl. One subcommand per run, each writing scripts/runs/optim-converge2d-<cmd>-<stamp>/
# (manifest.toml, results.json):
#
#   julia +1.13 --project=scripts -t 16 scripts/optim/converge2d.jl theory
#       the 1D cold-fluid wake: the optimal τ against a₀, and J(τ) at a₀ = 1 and 0.3
#   ... converge2d.jl probe <τ> <focus> <ppl>x<dy_cells>[,<ppl>x<dy_cells>...] [fd]
#       per resolution, one run with duals at (τ, focus): J of every window with its gradient, and
#       WP-045's box objective at several read-out times; `fd` adds a central-difference check of
#       the main window's gradient at the first resolution
#   ... converge2d.jl optimize <ppl>x<dy_cells> <τ0> <focus0>
#       maximize J over the WP-045 box from (τ0, focus0) with L-BFGS-B (problem.jl)
#   ... converge2d.jl tauscan <ppl>x<dy_cells> <a0> <waist> <focus> <τ>[,<τ>...] [rows]
#       J of every window against τ at one focus (a plane wave: a large waist and a few rows)
#   ... converge2d.jl plots <run dir> [<run dir> ...]
#       the figures from earlier runs' results.json
#
# A first argument --group=<name> puts the run directories into scripts/runs/<name>/.
# The resolution is ppl cells per λ₀ along x and Δy = dy_cells·Δx; the box stays 14 λ₀ × 16 λ₀.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "slab.jl"))
using ForwardDiff: value, partials

const SCALE = WAKE_BOX.scale                   # z = (τ/6, focus/90), as in WP-045
phys(z) = phys(WAKE_BOX, z)
const DT45 = 0.95 / sqrt(1 / (2π / 16)^2 + 1 / (4 * 2π / 16)^2)      # WP-045's step
const BOX_TIMES = (512 * DT45, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0)   # first: WP-045's read-out

parse_res(s) = (r = parse.(Int, split(s, "x")); (ppl = r[1], dy_cells = r[2]))
reskw(r) = (; ppl = r.ppl, dy_cells = r.dy_cells)
windows_dict(w) = Dict(string(k) => [v.x1, v.x2, v.s1, v.s2] for (k, v) in pairs(w))

const GROUP = Ref("")      # `--group=<name>` as the first argument: run directories go to scripts/runs/<name>/ (one CPU-box output)

function newdir(cmd, params)
    return run_dir("converge2d-$cmd", @__FILE__; prefix = isempty(GROUP[]) ? "optim" : "$(GROUP[])/optim", work_package = "WP-051",
                   extra_inputs = [joinpath(@__DIR__, f) for f in ("wake2d_case.jl", "slab.jl", "problem.jl")],
                   params = merge(Dict{String,Any}("command" => cmd, "args" => copy(ARGS)), params))
end

save_results(dir, out) = (open(io -> JSON.print(io, out, 2), joinpath(dir, "results.json"), "w"); println("results: ", dir))

function theory()
    dir = newdir("theory", Dict{String,Any}())
    a0s = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
    kp, β = sqrt(CASE.density), sqrt(1 - CASE.density)
    out = Dict{String,Any}("tau_linear" => 2 / kp, "beta" => β)
    τs = collect(3.0:0.1:12.0)
    out["tau"] = τs
    for (name, b) in (("group_velocity", β), ("quasistatic_c", 1.0))
        opt = map(a0s) do a0
            τJ, J = golden_max(τ -> wake1d(a0, τ; β = b).J, 3.0, 14.0; tol = 1e-4)
            τE, E = golden_max(τ -> wake1d(a0, τ; β = b).Emax, 3.0, 14.0; tol = 1e-4)
            println("$name, a0 = $a0: τ_opt(J) = $τJ (J = $J), τ_opt(Emax) = $τE (Emax = $E)")
            Dict("a0" => a0, "tau_J" => τJ, "J" => J, "tau_Emax" => τE, "Emax" => E)
        end
        lin = [sqrt(2π) / 8 * 0.05^2 * kp * τ * exp(-kp^2 * τ^2 / 8) / b^2 for τ in τs]
        ode = [wake1d(0.05, τ; β = b).J for τ in τs]
        out[name] = Dict{String,Any}("optimum" => opt, "J" => Dict(string(a0) => [wake1d(a0, τ; β = b).J for τ in τs] for a0 in (0.3, 1.0)),
                                     "linear_max_rel_diff_a0_0.05" => maximum(abs.(ode .- lin) ./ lin))
        println("$name: linear check at a0 = 0.05, largest relative difference of the ODE from the formula ", out[name]["linear_max_rel_diff_a0_0.05"])
    end
    save_results(dir, out)
end

function observe_dual(τ, f, r)
    t = @elapsed o = wake_observe(dual_point(WAKE_BOX, scaled(WAKE_BOX, (τ, f)))...; box_times = BOX_TIMES, reskw(r)...)
    slab = Dict(string(k) => Dict("J" => value(v), "grad" => collect(Tuple(partials(v)))) for (k, v) in pairs(o.slab))
    osc = Dict(string(k) => Dict("J" => value(v), "grad" => collect(Tuple(partials(v)))) for (k, v) in pairs(o.osc))
    box = [Dict("t" => tb, "J" => value(v), "grad" => collect(Tuple(partials(v)))) for (tb, v) in zip(BOX_TIMES, o.box)]
    println("$(r.ppl)x$(r.dy_cells): main J = $(slab["main"]["J"]), ∇J = $(slab["main"]["grad"]); box J(t = $(BOX_TIMES[1])) = $(box[1]["J"]); $(round(t; digits = 1)) s, $(o.nsteps) steps")
    flush(stdout)
    return Dict{String,Any}("ppl" => r.ppl, "dy_cells" => r.dy_cells, "dx" => o.dx, "dy" => o.dx * r.dy_cells, "dt" => o.dt, "nsteps" => o.nsteps,
                            "seconds" => t, "slab" => slab, "osc" => osc, "box" => box, "coverage" => [collect(c) for c in o.coverage])
end

function probe(τ, f, ress, fd)
    dir = newdir("probe", Dict{String,Any}("tau" => τ, "focus" => f, "resolutions" => [collect(r) for r in ress]))
    out = Dict{String,Any}("tau" => τ, "focus" => f, "windows" => windows_dict(WINDOWS), "box_times" => collect(BOX_TIMES), "runs" => Any[])
    for (i, r) in enumerate(ress)
        push!(out["runs"], observe_dual(τ, f, r))
        if fd && i == 1
            g = out["runs"][1]["slab"]["main"]["grad"]
            z0 = [τ / SCALE[1], f / SCALE[2]]
            Jz(z) = slab_amplitude(phys(z)...; reskw(r)...)
            out["fd"] = map((1e-2, 1e-3)) do h
                fdg = [(Jz(z0 .+ h .* (1:2 .== k)) - Jz(z0 .- h .* (1:2 .== k))) / 2h for k in 1:2]
                println("FD h = $h: $fdg against AD $g")
                Dict("h" => h, "fd" => fdg, "rel" => abs.(fdg .- g) ./ abs.(g))
            end
        end
        save_results(dir, out)
    end
    return dir
end

function optimize(r, τ0, f0)
    dir = newdir("optimize", Dict{String,Any}("resolution" => collect(r), "start" => [τ0, f0]))
    obj = Objective((τ, f) -> slab_amplitude(τ, f; reskw(r)...), WAKE_BOX)
    t = @elapsed res = maximize(obj, scaled(WAKE_BOX, (τ0, f0)); maxiters = 25, gtol = 1e-4)
    out = Dict{String,Any}("resolution" => collect(r), "start" => [τ0, f0], "phys" => collect(phys(res.z)), "J" => res.f, "grad" => res.g,
                           "reason" => res.reason, "evals" => res.evals, "seconds" => t,
                           "path" => [Dict("phys" => collect(phys(p[1])), "J" => p[2], "grad" => p[3]) for p in res.path])
    println("$(r.ppl)x$(r.dy_cells): $(res.reason) after $(res.evals) evaluations: (τ, focus) = $(phys(res.z)), J = $(res.f)")
    save_results(dir, out)
end

function tauscan(r, a0, waist, f, τs, nrows)
    dir = newdir("tauscan", Dict{String,Any}("resolution" => collect(r), "a0" => a0, "waist" => waist, "focus" => f, "taus" => τs, "rows" => something(nrows, 0)))
    cells = nrows === nothing ? (14r.ppl, 16r.ppl ÷ r.dy_cells) : (14r.ppl, nrows)
    rows = map(τs) do τ
        t = @elapsed o = wake_observe(τ, f; reskw(r)..., a0, waist, cells)
        println("a0 = $a0, waist = $waist, τ = $τ: ", o.slab, " ($(round(t; digits = 1)) s)")
        flush(stdout)
        Dict("tau" => τ, "J" => Dict(string(k) => v for (k, v) in pairs(o.slab)), "osc" => Dict(string(k) => v for (k, v) in pairs(o.osc)), "seconds" => t)
    end
    # The optimum of each window's J(τ): a parabola through the best sample and its neighbours.
    opt = Dict{String,Any}()
    for k in keys(WINDOWS)
        J = [row["J"][string(k)] for row in rows]
        i = argmax(J)
        if 1 < i < length(J)
            x, y = τs[(i - 1):(i + 1)], J[(i - 1):(i + 1)]
            A = [x .^ 2 x ones(3)]
            c = A \ y
            opt[string(k)] = Dict("tau" => -c[2] / (2c[1]), "J" => c[3] - c[2]^2 / (4c[1]))
        else
            opt[string(k)] = Dict("tau" => τs[i], "J" => J[i], "at_edge" => true)
        end
    end
    println("optimum: ", opt)
    save_results(dir, Dict{String,Any}("resolution" => collect(r), "a0" => a0, "waist" => waist, "focus" => f, "rows" => nrows, "scan" => rows,
                                       "optimum" => opt, "windows" => windows_dict(WINDOWS)))
end

# --- figures -----------------------------------------------------------------------------------

kind_of(r) = haskey(r, "runs") ? "probe" : haskey(r, "scan") ? "tauscan" : haskey(r, "path") ? "optimize" : haskey(r, "tau_linear") ? "theory" : "other"

"""
    plots(dirs) -> dir

The study's figures from the results of earlier runs (the directories that hold results.json):
`readout` (WP-045's objective against its read-out time), `convergence` (J and its gradient at
one point against Δx and Δy) and `tau_offset` (the optimal τ of the plane-wave and 2D runs
against Δx, with the 1D theory).
"""
function plots(dirs)
    res = [merge(JSON.parsefile(joinpath(d, "results.json")), Dict("dir" => relpath(abspath(d), REPO))) for d in dirs if isfile(joinpath(d, "results.json"))]
    dir = newdir("plots", Dict{String,Any}("sources" => [r["dir"] for r in res]))
    probes = [r for r in res if kind_of(r) == "probe"]
    runs = reduce(vcat, [p["runs"] for p in probes]; init = Any[])

    # 1. WP-045's objective against the read-out time, at the coarsest probed resolution.
    base = runs[argmin([r["ppl"] * 100 + 8 ÷ r["dy_cells"] for r in runs])]
    tb = Float64[b["t"] for b in base["box"]]
    o = sortperm(tb)
    fig = Figure(size = (1000, 400))
    ax = Axis(fig[1, 1]; xlabel = "read-out time t (1/ω₀)", ylabel = "J / E₀",
              title = "WP-045's objective against its read-out time ($(base["ppl"]) cells/λ₀, Δy = $(base["dy_cells"])Δx)")
    scatterlines!(ax, tb[o], Float64[b["J"] for b in base["box"]][o]; label = "box RMS at t (WP-045)")
    hlines!(ax, [base["slab"]["main"]["J"]]; color = :red, linestyle = :dash, label = "slab objective (any t)")
    vlines!(ax, [tb[1]]; color = :gray, linestyle = :dot, label = "WP-045's read-out")
    axislegend(ax; position = :rb)
    ax2 = Axis(fig[1, 2]; xlabel = "read-out time t (1/ω₀)", ylabel = "∂J/∂z (z = τ/6, focus/90)", title = "Its gradient at (7.74, 113)")
    for (i, name) in enumerate(("τ", "focus"))
        scatterlines!(ax2, tb[o], Float64[b["grad"][i] for b in base["box"]][o]; label = "box, ∂/∂$name")
        hlines!(ax2, [base["slab"]["main"]["grad"][i]]; linestyle = :dash, color = Cycled(i), label = "slab, ∂/∂$name")
    end
    hlines!(ax2, [0.0]; color = :black, linewidth = 0.5)
    axislegend(ax2; position = :lt)
    save_plot(dir, "readout", fig, Dict("resolution" => [base["ppl"], base["dy_cells"]], "t" => tb[o],
                                        "box_J" => Float64[b["J"] for b in base["box"]][o], "box_grad" => [base["box"][i]["grad"] for i in o],
                                        "slab" => base["slab"]["main"]))

    # 2. Convergence at each probed point: J and its gradient against Δx (one line per Δy) and
    #    against Δy (one line per Δx).
    dyl(r) = r["ppl"] ÷ r["dy_cells"]                        # cells per λ₀ along y
    for p in probes
        rs = p["runs"]
        length(rs) >= 3 || continue
        tag = "convergence-tau$(round(p["tau"]; digits = 2))-focus$(round(p["focus"]; digits = 1))"
        fig = Figure(size = (1100, 640))
        sidecar = Dict{String,Any}("tau" => p["tau"], "focus" => p["focus"], "runs" => rs)
        for (col, (xl, key, xv, other, olab)) in enumerate((("Δx (λ₀)", dyl, r -> 1 / r["ppl"], dyl, "Δy = λ₀/"),
                                                             ("Δy (λ₀)", r -> r["ppl"], r -> 1 / dyl(r), r -> r["ppl"], "Δx = λ₀/")))
            a = Axis(fig[1, col]; xlabel = xl, ylabel = "J / E₀", xscale = log2, xreversed = true,
                     title = "at (τ, focus) = ($(round(p["tau"]; digits = 2)), $(round(p["focus"]; digits = 1)))")
            b = Axis(fig[2, col]; xlabel = xl, ylabel = "∂J/∂z (z = τ/6, focus/90)", xscale = log2, xreversed = true)
            for g in sort(unique(key.(rs)))
                series = sort([r for r in rs if key(r) == g]; by = xv, rev = true)
                length(series) >= 2 || continue
                x = Float64[xv(r) for r in series]
                scatterlines!(a, x, Float64[r["slab"]["main"]["J"] for r in series]; label = "$olab$g")
                scatterlines!(b, x, Float64[r["slab"]["main"]["grad"][1] for r in series]; label = "∂/∂(τ/6), $olab$g")
                scatterlines!(b, x, Float64[r["slab"]["main"]["grad"][2] for r in series]; linestyle = :dash, label = "∂/∂(focus/90), $olab$g")
            end
            hlines!(b, [0.0]; color = :black, linewidth = 0.5)
            axislegend(a; position = :rb)
            axislegend(b; position = :lb, labelsize = 10)
        end
        save_plot(dir, tag, fig, sidecar)
    end

    # 3. The optimal τ: plane-wave and 2D scans against Δx, with the 1D theory.
    scans = [r for r in res if kind_of(r) == "tauscan"]
    th = [r for r in res if kind_of(r) == "theory"]
    opts = [r for r in res if kind_of(r) == "optimize"]
    fig = Figure(size = (800, 480))
    a = Axis(fig[1, 1]; xlabel = "Δx (λ₀)", ylabel = "optimal τ (1/ω₀)", xscale = log2, xreversed = true,
             title = "Optimal duration of the slab objective")
    side = Dict{String,Any}()
    groups = Dict{Tuple{Float64,Bool},Vector{Any}}()
    for r in scans
        push!(get!(groups, (Float64(r["a0"]), r["waist"] > 1000), Any[]), r)
    end
    for ((a0, pw), g) in sort(collect(groups); by = first)
        sort!(g; by = r -> r["resolution"][1])
        x = Float64[1 / r["resolution"][1] for r in g]
        y = Float64[r["optimum"]["main"]["tau"] for r in g]
        lab = (pw ? "plane wave" : "2D, waist 10") * ", a₀ = $a0"
        scatterlines!(a, x, y; label = lab, linestyle = pw ? :solid : :dash)
        side[lab] = Dict("dx" => x, "tau" => y, "resolution" => [r["resolution"] for r in g])
    end
    for r in opts
        scatter!(a, [1 / r["resolution"][1]], [r["phys"][1]]; marker = :star5, markersize = 16, label = "2D optimum, a₀ = 1 ($(r["resolution"][1])x$(r["resolution"][2]))")
        side["2D optimum $(r["resolution"])"] = r["phys"]
    end
    kp = sqrt(CASE.density)
    lines_th = Dict("1D linear 2/ω_p" => 2 / kp, "1D linear with dispersion" => golden_max(τ -> wake1d_dispersive(0.3, τ), 5.0, 9.0; tol = 1e-3)[1])
    if !isempty(th)
        lines_th["1D nonlinear (fixed driver), a₀ = 1"] = only(o["tau_J"] for o in th[1]["group_velocity"]["optimum"] if o["a0"] == 1.0)
    end
    for (i, (k, v)) in enumerate(sort(collect(lines_th); by = last))
        hlines!(a, [v]; color = Cycled(i + 3), linestyle = :dot, label = "$k: $(round(v; digits = 2))")
    end
    side["theory"] = lines_th
    Legend(fig[1, 2], a; framevisible = false)
    save_plot(dir, "tau_offset", fig, side)
    println("figures in ", dir)
    return dir
end

if !isempty(ARGS) && abspath(PROGRAM_FILE) == @__FILE__
    if startswith(ARGS[1], "--group=")
        GROUP[] = ARGS[1][9:end]
        popfirst!(ARGS)
    end
    cmd = ARGS[1]
    if cmd == "theory"
        theory()
    elseif cmd == "probe"
        probe(parse(Float64, ARGS[2]), parse(Float64, ARGS[3]), parse_res.(split(ARGS[4], ",")), length(ARGS) >= 5 && ARGS[5] == "fd")
    elseif cmd == "optimize"
        optimize(parse_res(ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4]))
    elseif cmd == "tauscan"
        tauscan(parse_res(ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4]), parse(Float64, ARGS[5]), parse.(Float64, split(ARGS[6], ",")),
                length(ARGS) >= 7 ? parse(Int, ARGS[7]) : nothing)
    elseif cmd == "plots"
        plots(ARGS[2:end])
    else
        error("unknown command $cmd")
    end
end
