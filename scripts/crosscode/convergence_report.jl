# The report step of scripts/crosscode/convergence.jl (WP-031): from a scan's summary directory
# (summary/<point>.json, summary/cases/<point>.toml), the pre-registered fit of every point, the
# same fit on the exact linear solution, the residual between them against Δx and Δt, the local
# rate, the ratio to the linear solution and the window sensitivity; results.json, the figures
# (PNG with a JSON sidecar each) and a manifest in the output directory.

module ConvergenceReport

using TOML
using JSON
using CairoMakie
using Statistics: mean, median
using ..Convergence
using ..Convergence: Observables, Theory, SmileiDeck, write_manifest, file_hash, git_commit, ROOT
using .Observables: growth_from_window, linear_fit

"A point's record: its derived case, its summary and its axes (Δx of the first axis, Δt)."
function load_point(sumdir, p)
    case = TOML.parsefile(joinpath(sumdir, "cases", p * ".toml"))
    rec = JSON.parsefile(joinpath(sumdir, p * ".json"))
    L = case["grid"]["length"]
    N = case["grid"]["cells"]
    dx = L isa AbstractVector ? Float64(L[1]) / N[1] : Float64(L) / N
    return (; name = p, case, rec, dx, dt = Float64(case["time"]["dt"]), derived = case["derived"])
end

"The window variants of the sensitivity check: the case's (lo, hi) and four neighbours (#168)."
window_variants(lo, hi) = [(lo, hi), (lo, hi / 3), (3lo, hi), (hi / 3, hi), (lo / 3, hi)]

series_of(rec, key) = (s = rec["series"][key]; (Float64.(s["t"]), Float64.(s["U"])))

"The window an observable `key` of a case is fitted with (the 1D `mode` estimate uses the primary one's)."
function window_of(case, key)
    obs = case["observables"]
    key == "mode" && return Tuple(Float64.(only(o for (_, o) in obs if o["method"] == "exponential_window")["window"]))
    return Tuple(Float64.(obs[key]["window"]))
end

theory_of(case, key) = key == "mode" ? only(o for (_, o) in case["observables"] if o["method"] == "exponential_window")["theory"] :
    case["observables"][key]["theory"]

"""
    evaluate(pt, key) -> Dict

The pre-registered fit of series `key` of one point and the fit of the linear solution over
the same samples; offsets relative to the case's theory value θ: `offset` = γ/θ − 1,
`offset_ivp` = γ_ivp/θ − 1, `residual` = (γ − γ_ivp)/θ; the ratio U/U_ivp at the window's ends;
the same three numbers for each window variant.
"""
function evaluate(pt, key)
    t, U = series_of(pt.rec, key)
    θ = theory_of(pt.case, key)
    w = window_of(pt.case, key)
    f = point_fits(pt.case, key, t, U; window = w)
    out = Dict{String,Any}("theory" => θ, "window" => collect(w))
    if !isfinite(f.gamma)
        out["flag"] = "no fit"
        return out
    end
    merge!(out, Dict{String,Any}("gamma" => f.gamma, "gamma_stderr" => f.r.gamma_stderr, "gamma_ivp" => f.gamma_ivp,
        "offset" => f.gamma / θ - 1, "offset_ivp" => f.gamma_ivp / θ - 1, "residual" => (f.gamma - f.gamma_ivp) / θ,
        "fit_points" => length(f.idx), "fit_range" => [t[first(f.idx)], t[last(f.idx)]],
        "saturation" => Dict("t" => f.r.t_max, "U" => f.r.U_max),
        "ratio_to_ivp_window_start" => f.ratio[1], "ratio_to_ivp_window_end" => f.ratio[2]))
    out["window_sensitivity_post_hoc"] = map(window_variants(w...)) do wv
        g = point_fits(pt.case, key, t, U; window = wv)
        Dict{String,Any}("window" => collect(wv), "gamma" => isfinite(g.gamma) ? g.gamma : nothing,
            "offset" => isfinite(g.gamma) ? g.gamma / θ - 1 : nothing,
            "offset_ivp" => isfinite(g.gamma) ? g.gamma_ivp / θ - 1 : nothing,
            "residual" => isfinite(g.gamma) ? (g.gamma - g.gamma_ivp) / θ : nothing,
            "fit_points" => length(g.idx))
    end
    return out
end

plain(p) = p.derived["amplitude_factor"] == 1 && p.derived["ppc_factor"] == "1//1" && p.derived["seed"] == "case"

"""
    families(pts) -> Vector{Pair{String,Vector}}

The convergence families of a scan's points, among those with the case's amplitude, particles
per cell and seed: `"dx"`, the points at the case's Courant number (`dt_divisor` 1), coarse to
fine; and for every `refine` that has points with `dt_divisor` ≠ 1, `"dt[refine=<r>]"`, the
points of that refine, Δt large to small (the Δt series at fixed Δx).
"""
function families(pts)
    pl = [p for p in pts if plain(p)]
    out = Pair{String,Vector}["dx" => sort([p for p in pl if p.derived["dt_divisor"] == "1//1"]; by = p -> -p.dx)]
    for r in sort(unique(p.derived["refine"] for p in pl if p.derived["dt_divisor"] != "1//1"))
        push!(out, "dt[refine=$r]" => sort([p for p in pl if p.derived["refine"] == r]; by = p -> -p.dt))
    end
    return out
end

"The family of `families(pts)` named `name`."
family(pts, name) = last(only(f for f in families(pts) if first(f) == name))

"""
    observed_order(h, e) -> Vector

The order log(|e_i − e_{i+1}| / |e_{i+1} − e_{i+2}|)/log(h_i/h_{i+1}) of three consecutive points
of a series e(h) with a constant refinement ratio: the exponent p of e = e∞ + a hᵖ.
"""
observed_order(h, e) = [log(abs(e[i] - e[i+1]) / abs(e[i+1] - e[i+2])) / log(h[i] / h[i+1]) for i in 1:length(e)-2]

"Richardson extrapolation of the last two points of e(h) for order p."
richardson(h, e, p) = e[end] + (e[end] - e[end-1]) / ((h[end-1] / h[end])^p - 1)

function save_fig(out, name, fig, sidecar)
    save(joinpath(out, name * ".png"), fig)
    open(io -> JSON.print(io, sidecar, 2), joinpath(out, name * ".json"), "w")
end

"""
    report(sumdir, out; box_status = nothing) -> Dict

Evaluate every point of a summary directory and write results.json, the figures and
manifest.toml into `out`.
"""
function report(sumdir, out; box_status = nothing)
    mkpath(out)
    scan = TOML.parsefile(joinpath(sumdir, "scan.toml"))
    names = sort([splitext(f)[1] for f in readdir(sumdir) if endswith(f, ".json") && f != "results.json"])
    pts = [load_point(sumdir, p) for p in names]
    keys_ = sort(collect(keys(first(pts).rec["series"])))
    res = Dict{String,Any}("scan" => scan["name"], "case" => scan["case"], "points" => Dict{String,Any}())
    for pt in pts
        Utot = Float64.(pt.rec["energy"]["Utot"])
        res["points"][pt.name] = Dict{String,Any}("dx" => pt.dx, "dt" => pt.dt, "derived" => pt.derived,
            "wall_seconds" => pt.rec["wall_seconds"], "threads" => pt.rec["threads"], "cpu" => pt.rec["cpu"],
            "steps" => pt.rec["steps"], "energy_max_relative_change" => maximum(abs, (Utot .- Utot[1]) ./ Utot[1]),
            "observables" => Dict{String,Any}(k => evaluate(pt, k) for k in keys_))
    end
    # convergence of the residual and of the offset along the Δx and Δt families
    conv = Dict{String,Any}()
    for (which, fam) in families(pts), k in keys_
        length(fam) >= 2 || continue
        ev = [res["points"][p.name]["observables"][k] for p in fam]
        any(e -> !haskey(e, "residual"), ev) && continue
        h = which == "dx" ? [p.dx for p in fam] : [p.dt for p in fam]
        r = [e["residual"] for e in ev]
        o = [e["offset"] for e in ev]
        d = Dict{String,Any}("points" => [p.name for p in fam], "h" => h, "residual" => r, "offset" => o,
            "offset_ivp" => [e["offset_ivp"] for e in ev])
        if length(fam) >= 3
            p = observed_order(h, r)
            d["observed_order_residual"] = p
            # the limit at the median observed order, rounded to an integer of at least 1 (the Δt
            # families converge at first order, the time of the loaded momenta; Δx at second; the
            # finest triple can read high where the differences approach the fit's resolution)
            q = max(1, round(Int, median(p)))
            d["richardson_order"] = q
            d["richardson_residual"] = richardson(h, r, q)
        end
        conv["$(which)_$(k)"] = d
    end
    res["convergence"] = conv
    open(io -> JSON.print(io, res, 2), joinpath(out, "results.json"), "w")
    plots(out, pts, res, keys_)
    inputs = [joinpath(sumdir, f) for f in readdir(sumdir) if endswith(f, ".json") || endswith(f, ".toml")]
    append!(inputs, [joinpath(sumdir, "cases", f) for f in readdir(joinpath(sumdir, "cases"))])
    extra = Dict{String,Any}("report" => "scripts/crosscode/convergence.jl report",
        "report_inputs" => Dict("scripts/crosscode/" * f => "sha256:" * file_hash(joinpath(ROOT, "scripts", "crosscode", f))
                                for f in ("convergence.jl", "convergence_report.jl", "theory.jl", "analysis.jl")))
    if box_status !== nothing
        b = TOML.parsefile(box_status)
        extra["box"] = Dict{String,Any}(k => b[k] for k in ("branch", "request_commit", "started", "finished", "ok", "hardware", "image") if haskey(b, k))
    end
    # recorded relative to `out` (no absolute path of the machine enters the manifest)
    write_manifest(out; inputs = [relpath(abspath(f), abspath(out)) for f in inputs], root = ROOT, extra)
    return res
end

function plots(out, pts, res, keys_)
    primary = first(k for k in keys_ if k != "mode")
    case = first(pts).case
    θ = theory_of(case, primary)
    # 1. offset and residual against Δx and Δt
    fig = Figure(size = (1100, 700))
    ax1 = Axis(fig[1, 1]; xscale = log10, xlabel = "Δx [c/ω₀] (Courant number kept) or 20 Δt [1/ω₀] (Δx kept)",
        ylabel = "γ/θ − 1", title = "$(case["derived"]["from"]): $primary, offset from theory")
    ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "Δx or 20 Δt", ylabel = "|γ − γ_ivp| / θ",
        title = "measured − linear solution, same window")
    side = Dict{String,Any}("plot" => "offset of the pre-registered fit from theory (left) and its residual from the same fit on the exact linear solution (right), against Δx at the case's Courant number and against 20 Δt at fixed cells (one series per refine)", "theory" => θ)
    colors = [:black, :red, :blue, :green]
    for (i, (which, _)) in enumerate(families(pts)), k in keys_
        key = "$(which)_$(k)"
        haskey(res["convergence"], key) || continue
        d = res["convergence"][key]
        col, sc = colors[mod1(i, 4)], which == "dx" ? 1.0 : 20.0
        h = sc .* d["h"]
        ls = k == primary ? :solid : :dash
        lab = "$(k), $(which == "dx" ? "Δx" : "Δt, " * which[4:end-1])"
        scatterlines!(ax1, h, d["offset"]; color = col, linestyle = ls, label = lab * " measured")
        lines!(ax1, h, d["offset_ivp"]; color = col, linestyle = :dot, label = lab * " linear solution")
        r = abs.(d["residual"])
        scatterlines!(ax2, h, max.(r, 1e-7); color = col, linestyle = ls, label = lab)
        side[key] = d
    end
    if length(side) > 2      # at least one family plotted; the legends below the axes, off the data
        Legend(fig[2, 1], ax1; labelsize = 9, nbanks = 2, tellwidth = false)
        Legend(fig[2, 2], ax2; labelsize = 9, nbanks = 2, tellwidth = false)
    end
    save_fig(out, "offset-vs-resolution", fig, side)

    # 2. local rate against time: the coarsest and finest Δx points and the linear solution
    fam = family(pts, "dx")
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "local rate / θ − 1", title = "$primary: local growth rate (log U fitted over ±1/ω₀)")
    side = Dict{String,Any}("plot" => "local growth rate of the amplitude relative to theory against time, runs and the exact linear solution; the pre-registered window shaded")
    ev = res["points"][fam[end].name]["observables"][primary]
    haskey(ev, "fit_range") && vspan!(ax, ev["fit_range"]...; color = (:gray, 0.2))
    for (pt, col) in zip(unique([fam[1], fam[end]]), (:blue, :black))
        t, U = series_of(pt.rec, primary)
        tsat = res["points"][pt.name]["observables"][primary]["saturation"]["t"]
        keep = t .<= tsat
        tc, g = local_rate(t[keep], U[keep])
        lines!(ax, tc, g ./ θ .- 1; color = col, label = pt.name)
        side[pt.name] = Dict("t" => tc, "relative" => g ./ θ .- 1)
    end
    pt = fam[end]
    t, _ = series_of(pt.rec, primary)
    tsat = res["points"][pt.name]["observables"][primary]["saturation"]["t"]
    tl = t[t .<= tsat]
    tc, g = local_rate(tl, ivp_series(pt.case, primary, tl))
    lines!(ax, tc, g ./ θ .- 1; color = :red, linestyle = :dash, label = "linear solution")
    side["linear_solution"] = Dict("t" => tc, "relative" => g ./ θ .- 1)
    ylims!(ax, -0.3, 0.3)
    hlines!(ax, [0.0]; color = :gray)
    axislegend(ax; position = :rt)
    save_fig(out, "local-rate", fig, side)

    # 3. U / U_ivp against time for the Δx family, to saturation
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "U / U_linear", title = "$primary: run over the exact linear solution")
    side = Dict{String,Any}("plot" => "the fitted series over the exact linear solution at the same times, to saturation; the pre-registered window shaded (finest Δx)")
    haskey(ev, "fit_range") && vspan!(ax, ev["fit_range"]...; color = (:gray, 0.2))
    for (i, pt) in enumerate(fam)
        t, U = series_of(pt.rec, primary)
        tsat = res["points"][pt.name]["observables"][primary]["saturation"]["t"]
        Ul = ivp_series(pt.case, primary, t)
        e = res["points"][pt.name]["observables"][primary]
        Uref = haskey(e, "fit_range") ? Ul[findfirst(==(e["fit_range"][1]), t)] : maximum(Ul)
        # where the linear solution passes near zero (its early beats) the ratio means nothing
        keep = (t .<= tsat) .& (U .> 0) .& (Ul .> 1e-3 * Uref)
        rr = U[keep] ./ Ul[keep]
        lines!(ax, t[keep], rr; label = pt.name, color = i, colormap = :viridis, colorrange = (1, max(2, length(fam))))
        side[pt.name] = Dict("t" => t[keep][1:4:end], "ratio" => rr[1:4:end])
    end
    ylims!(ax, 0.8, 1.1)
    axislegend(ax; position = :lb)
    save_fig(out, "ratio-to-linear", fig, side)
end

"""
    compare_codes(ref_sumdir, other_sumdir, out; key = "growth_rate") -> Dict

Two codes on the same scan points (#226: ParticleMesh against Smilei at each Δt): for every point
in both summaries, the other code's fit over the reference run's window samples (the same
times), its difference from the reference γ, and U_other/U_ref at the window's ends and at
every sample to saturation; `comparison.json` and `code-ratio.png` (with its sidecar) in `out`.
"""
function compare_codes(ref_sumdir, other_sumdir, out; key = "growth_rate")
    mkpath(out)
    names = sort([splitext(f)[1] for f in readdir(other_sumdir) if endswith(f, ".json") && isfile(joinpath(ref_sumdir, f))])
    res = Dict{String,Any}("reference" => ref_sumdir, "other" => other_sumdir, "key" => key, "points" => Dict{String,Any}())
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; xlabel = "t [1/ω₀]", ylabel = "U_other / U_reference", title = "$key: the two codes at the same Δt")
    side = Dict{String,Any}("plot" => "ratio of the fitted series of the two codes at the same output times, to the reference run's saturation")
    for (i, p) in enumerate(names)
        a, b = load_point(ref_sumdir, p), load_point(other_sumdir, p)
        ta, Ua = series_of(a.rec, key)
        tb, Ub = series_of(b.rec, key)
        n = min(length(ta), length(tb))
        maximum(abs.(ta[1:n] .- tb[1:n])) <= 1e-9 * ta[n] || error("compare_codes: $p: the output times differ")
        fa = growth_from_window(ta, Ua; window = window_of(a.case, key))
        i1, i2 = findfirst(==(first(fa.t_fit)), ta), findfirst(==(last(fa.t_fit)), ta)
        fb = linear_fit(tb[i1:i2], log.(Ub[i1:i2]))
        own = evaluate(b, key)
        isat = findfirst(==(fa.t_max), ta)
        length(Ub) >= isat || error("compare_codes: $p: the other run ends before the reference's saturation")
        r = Ub[1:isat] ./ Ua[1:isat]
        res["points"][p] = Dict{String,Any}("dt" => a.dt, "dx" => a.dx, "gamma_reference" => fa.gamma,
            "gamma_other_same_samples" => fb.slope / 2,
            "relative_difference_same_samples" => (fb.slope / 2 - fa.gamma) / fa.gamma,
            "ratio_window_start" => r[i1], "ratio_window_end" => r[i2], "ratio_t0" => r[1],
            "other_own_rule" => own)
        keep = Ua[1:isat] .> 1e-3 * Ua[i1]
        lines!(ax, ta[1:isat][keep], r[keep]; label = "$p (Δt = $(round(a.dt; sigdigits = 4)))",
            color = i, colormap = :viridis, colorrange = (1, max(2, length(names))))
        side[p] = Dict("t" => ta[1:isat][keep][1:4:end], "ratio" => r[keep][1:4:end])
    end
    axislegend(ax; position = :lb)
    save_fig(out, "code-ratio", fig, side)
    open(io -> JSON.print(io, res, 2), joinpath(out, "comparison.json"), "w")
    return res
end

end
