# The seed ensembles of the oblique Landau case against linear theory with each code's
# thermal load and the relativistic push (WP-025; post hoc, not the case's pre-registered
# theory value). For each run: the rate and frequency of `landau_rate_relativistic` for its
# load (Smilei: `:classical_energy`; PIConGPU: `:momentum_maxwellian`), the non-relativistic
# root, the run's ensemble γ against both, and the ratio of the run's |⟨E·k̂⟩|² to the theory
# curve |⟨E·k̂⟩(0)|² e(t)² at the maxima the run's rule took. Also the rate for ParticleMesh's
# 3V load (`:velocity_maxwellian`), for the threshold proposal.
#
#   julia +1.13 --project=scripts scripts/crosscode/landau_relativistic.jl <smilei run> <picongpu run> <out dir>
#
# Writes <out dir>/landau-relativistic.png, its sidecar landau-relativistic.json, and manifest.toml.

include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "ensemble.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "theory.jl"))
using .Observables, .Ensemble, .RunManifest, .Theory
using CairoMakie, JSON, TOML

const LOADS = Dict("smilei" => :classical_energy, "picongpu" => :momentum_maxwellian)

"Linear interpolation of a `landau_rate_relativistic` solution e(t) at time x (step 0.0025, far below the dump cadence)."
function interp(sol, x)
    h = sol.t[2] - sol.t[1]
    i = clamp(floor(Int, x / h) + 1, 1, length(sol.t) - 1)
    w = (x - sol.t[i]) / h
    return (1 - w) * sol.e[i] + w * sol.e[i+1]
end

function compare_theory(runs, out)
    case = TOML.parsefile(joinpath(first(runs), "case.toml"))
    T = only(unique(s["thermal_velocity"]^2 for s in case["species"] if s["mobile"]))
    k = case["wavevector"]["magnitude"]
    o = case["observables"]["damping_rate_mode"]
    nr = landau_rate_relativistic(k, T; relativistic = false)
    pm = landau_rate_relativistic(k, T; load = :velocity_maxwellian)
    res = Dict{String,Any}("T" => T, "k" => k, "theory_preregistered" => o["theory"],
        "theory_frequency_preregistered" => o["theory_frequency"],
        "nonrelativistic_time_domain" => Dict("gamma" => nr.gamma, "omega" => nr.omega),
        "particlemesh_velocity_maxwellian" => Dict("gamma" => pm.gamma, "omega" => pm.omega),
        "note" => "post hoc: linear theory with the relativistic push and each code's load (theory.jl, landau_rate_relativistic); the grid is not included")
    fig = Figure(size = (900, 550))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t [1/ω₀]", ylabel = "|⟨E·k̂⟩|² of mode $(case["wavevector"]["mode"])",
        title = "$(case["name"]): seed ensembles against linear theory")
    for (j, dir) in enumerate(runs)
        m = TOML.parsefile(joinpath(dir, "manifest.toml"))
        code = get(m, "code", "smilei")
        ob = JSON.parsefile(joinpath(dir, "observables.json"))
        s = JSON.parsefile(joinpath(dir, "seed-coefficients.json"))
        t = Float64.(s["t"])
        c = [complex.(Float64.(re), Float64.(im)) for (re, im) in zip(s["re"], s["im"])]
        U = ensemble_energy(c)
        th = landau_rate_relativistic(k, T; load = LOADS[code])
        e = interp.(Ref(th), t)
        Uth = U[1] .* e .^ 2
        enr = interp.(Ref(nr), t)
        idx = [argmin(abs.(t .- x)) for x in ob["fit_times"]]
        γ, σ = ob["measured"], ob["measured_stderr"]
        res[code] = Dict{String,Any}("run" => basename(normpath(dir)), "load" => String(LOADS[code]),
            "theory_gamma" => th.gamma, "theory_omega" => th.omega,
            "measured" => γ, "measured_stderr" => σ, "measured_frequency" => ob["measured_frequency"],
            "minus_theory" => γ - th.gamma, "minus_theory_in_stderr" => (γ - th.gamma) / σ,
            "minus_preregistered" => γ - o["theory"], "minus_preregistered_in_stderr" => (γ - o["theory"]) / σ,
            "ratio_to_theory_at_maxima_used" => U[idx] ./ Uth[idx],
            "ratio_to_nonrelativistic_at_maxima_used" => U[idx] ./ (U[1] .* enr[idx] .^ 2))
        col = Makie.wong_colors()[j]
        p = U .> 0
        lines!(ax, t[p], U[p]; color = col, label = "$code, 32 seeds, γ = $(round(γ; sigdigits = 4))")
        lines!(ax, t, Uth; color = col, linestyle = :dash, label = "linear theory, $(LOADS[code]), relativistic push: γ = $(round(th.gamma; sigdigits = 4))")
    end
    s1 = JSON.parsefile(joinpath(first(runs), "seed-coefficients.json"))
    U0 = abs2(sum(complex(x[1], y[1]) for (x, y) in zip(s1["re"], s1["im"])) / length(s1["re"]))
    lines!(ax, nr.t, U0 .* nr.e .^ 2; color = :black, linestyle = :dot, label = "non-relativistic: γ = $(round(nr.gamma; sigdigits = 4))")
    ylims!(ax, 1e-10, 2e-6)
    axislegend(ax; position = :rt)
    sm, pc = res["smilei"], res["picongpu"]
    d = pc["measured"] - sm["measured"]
    res["difference_minus_theory_difference"] = d - (pc["theory_gamma"] - sm["theory_gamma"])
    res["difference_minus_theory_difference_in_stderr"] = res["difference_minus_theory_difference"] / hypot(sm["measured_stderr"], pc["measured_stderr"])
    mkpath(out)
    save(joinpath(out, "landau-relativistic.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, merge(Dict("plot" => "|⟨E·k̂⟩|² of both 32-seed ensembles with the linear-theory curves of their loads (relativistic push) and the non-relativistic one, all from |⟨E·k̂⟩(0)|²"), res), 2),
        joinpath(out, "landau-relativistic.json"), "w")
    root = dirname(dirname(@__DIR__))
    write_manifest(out; root, extra = Dict{String,Any}("inputs" => merge(
        Dict{String,Any}(relpath(joinpath(@__DIR__, f), root) => "sha256:" * file_hash(joinpath(@__DIR__, f))
                         for f in ("landau_relativistic.jl", "theory.jl", "ensemble.jl")),
        Dict{String,Any}(basename(normpath(r)) * "/" * f => "sha256:" * file_hash(joinpath(r, f))
                         for r in runs for f in ("seed-coefficients.json", "observables.json"))),
        "runs" => basename.(normpath.(runs)), "work_package" => "WP-025", "kind" => "post-hoc theory comparison", "julia" => string(VERSION)))
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || (println(stderr, "usage: landau_relativistic.jl <smilei run> <picongpu run> <out dir>"); exit(2))
    r = compare_theory(ARGS[1:2], ARGS[3])
    for c in ("smilei", "picongpu")
        x = r[c]
        println(c, ": theory ", round(x["theory_gamma"]; digits = 5), " ω ", round(x["theory_omega"]; digits = 4), "; measured ",
            round(x["measured"]; digits = 5), " ± ", round(x["measured_stderr"]; digits = 5), " → ", round(x["minus_theory_in_stderr"]; digits = 2),
            " errors (pre-registered: ", round(x["minus_preregistered_in_stderr"]; digits = 2), "); U/U_th at maxima ", round.(x["ratio_to_theory_at_maxima_used"]; digits = 3))
    end
    println("difference minus theory difference: ", round(r["difference_minus_theory_difference"]; digits = 5), " (", round(r["difference_minus_theory_difference_in_stderr"]; digits = 2), " errors)")
    println("ParticleMesh 3V velocity Maxwellian: ", r["particlemesh_velocity_maxwellian"])
end
