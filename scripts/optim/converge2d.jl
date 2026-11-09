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
#       maximize J over the WP-045 box from (τ0, focus0) with maximize.jl
#   ... converge2d.jl tauscan <ppl>x<dy_cells> <a0> <waist> <focus> <τ>[,<τ>...] [rows]
#       J of every window against τ at one focus (a plane wave: a large waist and a few rows)
#
# A first argument --group=<name> puts the run directories into scripts/runs/<name>/.
# The resolution is ppl cells per λ₀ along x and Δy = dy_cells·Δx; the box stays 14 λ₀ × 16 λ₀.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "slab.jl"))
include(joinpath(@__DIR__, "maximize.jl"))
using ForwardDiff: Dual, value, partials

const SCALE = (6.0, 90.0)                      # z = (τ/6, focus/90), as in WP-045
const LO = [4.0 / SCALE[1], 60.0 / SCALE[2]]
const HI = [9.0 / SCALE[1], 130.0 / SCALE[2]]
phys(z) = (z[1] * SCALE[1], z[2] * SCALE[2])
const DT45 = 0.95 / sqrt(1 / (2π / 16)^2 + 1 / (4 * 2π / 16)^2)      # WP-045's step
const BOX_TIMES = (512 * DT45, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0)   # first: WP-045's read-out

parse_res(s) = (r = parse.(Int, split(s, "x")); (ppl = r[1], dy_cells = r[2]))
reskw(r) = (; ppl = r.ppl, dy_cells = r.dy_cells)
windows_dict(w) = Dict(string(k) => [v.x1, v.x2, v.s1, v.s2] for (k, v) in pairs(w))

const GROUP = Ref("")      # `--group=<name>` as the first argument: run directories go to scripts/runs/<name>/ (one CPU-box output)

function newdir(cmd, params)
    return run_dir("converge2d-$cmd", @__FILE__; prefix = isempty(GROUP[]) ? "optim" : "$(GROUP[])/optim", work_package = "WP-051",
                   extra_inputs = [joinpath(@__DIR__, f) for f in ("wake2d_case.jl", "slab.jl", "maximize.jl")],
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
    d1 = Dual{Nothing}(τ / SCALE[1], 1.0, 0.0)
    d2 = Dual{Nothing}(f / SCALE[2], 0.0, 1.0)
    t = @elapsed o = wake_observe(phys((d1, d2))...; box_times = BOX_TIMES, reskw(r)...)
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
    function fg(z)
        d1 = Dual{Nothing}(z[1], 1.0, 0.0)
        d2 = Dual{Nothing}(z[2], 0.0, 1.0)
        v = slab_amplitude(phys((d1, d2))...; reskw(r)...)
        fv, gv = value(v), collect(Tuple(partials(v)))
        println("  (τ, focus) = ", round.(phys(z); digits = 4), "  J = ", fv, "  ∇J = ", gv)
        flush(stdout)
        return fv, gv
    end
    t = @elapsed res = maximize_box(fg, [τ0 / SCALE[1], f0 / SCALE[2]], LO, HI; maxiter = 25, gtol = 1e-4, xtol = 1e-4)
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
    else
        error("unknown command $cmd")
    end
end
