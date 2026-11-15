# The measurements of WP-059, the deterministic field ionization (reports/wp-059-field-ionization.md).
#
# 1. The rates (adk_rates.png): every level of nitrogen at 0.8 µm from `atomic.jl`, in 1/ω₀ against
#    a₀, with the rates of FBPIC 0.27.1's coefficient lines (`fbpic_reference.toml`, both variants)
#    on top, and the largest relative difference to each variant over every level of H, He, N, Ar.
# 2. A pulse over test atoms in vacuum (vacuum_fraction.png): a Gaussian pulse (a₀ = 2, waist 10,
#    duration 20, focused on the atoms' column) passes atoms of zero weight with nitrogen's sixth
#    level (N⁵⁺ → N⁶⁺, the first K-shell electron), at 16, 32 and 64 cells per wavelength along x;
#    the fraction each atom of the focal column has released after the pulse, against
#    1 − exp(−∫W(|E(t)|)dt) with E the paraxial field the antenna is given (`laser_field`),
#    integrated on a fine grid in time. The run differs from it by the Yee scheme's dispersion and
#    by the step's sampling of the rate (one value of |E| per Δt), both of which shrink with Δx.
# 3. The objectives against a₀ (objective_scan.png): the born charge and the born electrons'
#    kinetic energy of the small ionization-injection run of the tests (`ion_window`,
#    lib/ParticleMeshElectromagnetic/test/ionization_cases.jl) on 33 values of a₀ from 0.6 to 1.4,
#    and at 9 of them the ForwardDiff derivative from one dual run each, drawn as tangents; and at
#    a₀ = 1 the derivative in each of four parameters against a central difference.
#   julia +1.13 --project=scripts -t 4 scripts/ionization/ionization.jl                 the run, its figures and tables
#   julia +1.13 --project=scripts scripts/ionization/ionization.jl table <run directory>   the report's table from a run's sidecars
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
const ParticleMeshCore = ParticleMesh.ParticleMeshCore
using ParticleMeshElectromagnetic
using ForwardDiff: ForwardDiff
include(joinpath(@__DIR__, "atomic.jl"))
zero3(::Type{T}) where {T} = (zero(T), zero(T), zero(T))
bits(a::AbstractArray) = reinterpret(UInt8, vec(collect(a)))
const TYPES = (Float64,)
include(joinpath(EM_TEST, "ionization_cases.jl"))

const WAVELENGTH = 0.8e-6

# --- 1 ------------------------------------------------------------------------------------------

function rate_rows()
    ref = TOML.parsefile(joinpath(@__DIR__, "fbpic_reference.toml"))
    rows = Dict{String,Any}[]
    for lv in ref["levels"]
        el, Z = Symbol(lv["element"]), lv["charge"]
        level = adk_level(Float64, el, Z; wavelength = ref["wavelength_m"], cutoff = 1.0e6)
        rates = [adk_rate(level, a) for a in ref["a0"]]
        push!(rows, Dict{String,Any}("element" => lv["element"], "charge" => Z, "energy_eV" => lv["energy_eV"], "a0" => ref["a0"], "rate" => rates,
                                     "rate_fbpic" => lv["fbpic"]["rate"], "rate_rydberg" => lv["rydberg"]["rate"], "prefactor" => level.prefactor,
                                     "field" => level.field, "power" => level.power))
    end
    return rows
end

"The largest relative difference of the rates above `floor` (1/ω₀) to a variant of the reference."
function rate_difference(rows, variant; floor)
    worst = 0.0
    for r in rows, (a, b) in zip(r["rate"], r[variant])
        b > floor && (worst = max(worst, abs(a / b - 1)))
    end
    return worst
end

"N, N⁺, N²⁺, … for the charge `Z`."
ion_label(Z) = Z == 0 ? "N" : Z == 1 ? "N⁺" : "N" * ("²", "³", "⁴", "⁵", "⁶", "⁷")[Z - 1] * "⁺"

function rates_figure(dir)
    rows = rate_rows()
    fig = Figure(size = (760, 480))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "a₀ (field in mₑcω₀/e), λ₀ = 0.8 µm", ylabel = "ADK rate (1/ω₀)",
              title = "nitrogen, the seven levels: this code (lines), FBPIC 0.27.1's coefficients (crosses)")
    a = 10 .^ range(log10(0.02), log10(8); length = 200)
    for Z in 1:7
        level = adk_level(Float64, :N, Z; wavelength = WAVELENGTH, cutoff = 1.0e6)
        w = [adk_rate(level, x) for x in a]
        keep = w .> 1.0e-12
        lines!(ax, a[keep], w[keep]; label = "$(ion_label(Z - 1)) → $(ion_label(Z))")
        r = only(filter(r -> r["element"] == "N" && r["charge"] == Z, rows))
        pts = [(x, y) for (x, y) in zip(r["a0"], r["rate_fbpic"]) if y > 1.0e-12]
        scatter!(ax, first.(pts), last.(pts); marker = :xcross, color = :black, markersize = 9)
    end
    ylims!(ax, 1.0e-12, 1.0e6)
    axislegend(ax; position = :rb, labelsize = 10, nbanks = 2)
    summary = Dict{String,Any}("rydberg, every rate above 1e-300" => rate_difference(rows, "rate_rydberg"; floor = 1.0e-300),
                               "fbpic, rates above 1e-12" => rate_difference(rows, "rate_fbpic"; floor = 1.0e-12),
                               "fbpic, rates above 1e-3" => rate_difference(rows, "rate_fbpic"; floor = 1.0e-3))
    println("wrote ", save_plot(dir, "adk_rates", fig, Dict("levels" => rows, "largest_relative_difference" => summary)))
    return summary
end

# --- 2 ------------------------------------------------------------------------------------------

const VAC = (; a0 = 2.0, waist = 10.0, duration = 20.0, delay = 70.0, position = 4.0, focus = 60.0, length = (240.0, 60.0), rows_per_wavelength = 8)

"The pulse of the vacuum runs, with the antenna at `position` and the focus on the column the atoms are read on."
vacuum_pulse() = GaussianLaser(; a0 = VAC.a0, waist = VAC.waist, duration = VAC.duration, delay = VAC.delay, focus = VAC.focus,
                               position = VAC.position, center = VAC.length[2] / 2)

"The fraction released at the transverse distance `dy` from the axis in the focal plane, from the paraxial field times `scale`: 1 − exp(−∫W(scale·|e(s)|)ds), trapezoid rule at `ds`."
function analytic_fraction(level, dy; ds = 1.0e-3, scale = 1.0)
    pulse = vacuum_pulse()
    s = (-6 * VAC.duration):ds:(6 * VAC.duration)
    w = [adk_rate(level, scale * abs(first(laser_field(pulse, 0.0, dy, x)))) for x in s]
    return -expm1(-ds * (sum(w) - (first(w) + last(w)) / 2))
end

"""
What the quadratic-spline gather leaves of the amplitude of a plane wave of `ncx` cells per
wavelength along x at a particle's place: sinc³(k·Δx/2) with k·Δx = 2π/ncx (the transform of the
quadratic B-spline at the wave's wavenumber). The atoms take their field through that gather.
"""
gather_factor(ncx) = (sin(π / ncx) / (π / ncx))^3

"The largest |e| of the paraxial field on the axis in the focal plane."
peak_paraxial() = maximum(s -> abs(first(laser_field(vacuum_pulse(), 0.0, 0.0, s))), -4:1.0e-3:4)

"""
The vacuum run at `ncx` cells per wavelength along x: test atoms of zero weight on every cell, the
pulse launched at t = 0 and run until it has passed the focal column by six durations; the
released fraction (`pending`: nothing is born at zero weight) of the atoms of the focal column
against their distance from the axis, and the largest |Ey| the run had on the Ey node next to the focus
(interpolated in time between the steps).
"""
function vacuum_run(level, ncx)
    hx, hy = 2π / ncx, 2π / VAC.rows_per_wavelength
    cells = (round(Int, VAC.length[1] / hx), round(Int, VAC.length[2] / hy))
    L = (cells[1] * hx, cells[2] * hy)
    dt = 0.95 / sqrt(1 / hx^2 + 1 / hy^2)
    pulse = GaussianLaser(; a0 = VAC.a0, waist = VAC.waist, duration = VAC.duration, delay = VAC.delay, focus = VAC.focus,
                          position = VAC.position, center = L[2] / 2)
    nsteps = ceil(Int, (VAC.delay + VAC.focus - VAC.position + 6 * VAC.duration) / dt)
    ion = FieldIonization(; density = 0.0, levels = (level,), capacity = 1)
    prob = EMProblem("ion-vacuum", L, cells, dt, nsteps, (), 1, (1, 0); lasers = (pulse,), ionization = ion)
    st = EMState(prob, EMNumerics())
    initialize!(st, prob)
    # the largest |Ey| of the run on the Ey node next to the focus on the axis (a host read per step: the CPU backend)
    g = ParticleMeshCore.guards(st.grid)
    i0, j0 = g + round(Int, VAC.focus / hx), g + cells[2] ÷ 2
    p = params(prob)
    peak, y1, y2 = 0.0, 0.0, 0.0
    seconds = @elapsed for k in 1:nsteps
        step!(st, p, k)
        y3 = abs(st.fields.Ey[i0, j0])
        # the vertex of the parabola through three samples about a local maximum: the peak between the steps
        y2 >= y1 && y2 > y3 && (peak = max(peak, y2 - (y1 - y3)^2 / (8 * (y1 - 2y2 + y3))))
        y1, y2 = y2, y3
    end
    s = ionization_state(st)
    col = argmin(abs.(s.x .- VAC.focus))
    sel = findall(==(s.x[col]), s.x)
    return (; dy = s.y[sel] .- L[2] / 2, fraction = s.pending[sel], x = s.x[col], dt, nsteps, cells, seconds, peak)
end

function vacuum_figure(dir)
    level = adk_level(Float64, :N, 6; wavelength = WAVELENGTH)
    rows = Dict{String,Any}[]
    fig = Figure(size = (760, 460))
    ax = Axis(fig[1, 1]; xlabel = "distance from the axis (c/ω₀)", ylabel = "fraction of N⁵⁺ ionized to N⁶⁺ after the pulse",
              title = "a₀ = 2, waist 10, duration 20 c/ω₀, vacuum: the run against the integral of the rate")
    ys = range(-16, 16; length = 161)
    exact = [analytic_fraction(level, y) for y in ys]
    lines!(ax, ys, exact; color = :black, label = "1 − exp(−∫W(|E|)dt), paraxial field")
    for (ncx, marker) in ((16, :circle), (32, :utriangle), (64, :xcross))
        r = vacuum_run(level, ncx)
        ref = [analytic_fraction(level, y) for y in r.dy]
        i0 = argmin(abs.(r.dy))
        row = Dict{String,Any}("cells_per_wavelength" => ncx, "cells" => collect(r.cells), "steps" => r.nsteps, "dt" => r.dt, "column_x" => r.x,
                               "dy" => r.dy, "fraction" => r.fraction, "analytic" => ref, "nearest_axis_dy" => r.dy[i0],
                               "nearest_axis_fraction" => r.fraction[i0], "nearest_axis_analytic" => ref[i0],
                               "largest_abs_difference" => maximum(abs, r.fraction .- ref), "seconds" => r.seconds,
                               "peak_Ey_at_focus" => r.peak, "peak_Ey_paraxial" => peak_paraxial())
        println(Dict(k => v for (k, v) in row if !(k in ("dy", "fraction", "analytic"))))
        push!(rows, row)
        keep = abs.(r.dy) .<= 16
        scatter!(ax, r.dy[keep], r.fraction[keep]; marker, markersize = 9, label = "λ₀/$ncx along x, Δt = $(round(r.dt; digits = 3))")
    end
    axislegend(ax; position = :rt, labelsize = 10)
    println("wrote ", save_plot(dir, "vacuum_fraction", fig, Dict("level" => Dict("prefactor" => level.prefactor, "field" => level.field,
                                                                                    "power" => level.power, "cutoff" => level.cutoff),
                                                                    "pulse" => Dict(string(k) => v for (k, v) in pairs(VAC) if !(v isa Tuple)),
                                                                    "analytic_dy" => collect(ys), "analytic_fraction" => exact, "runs" => rows)))
    return rows
end

# --- 3 ------------------------------------------------------------------------------------------

const X0 = (1.0, 0.005, 0.2, 28.0)
const NAMES = ("a0", "density", "prefactor", "start")

"The born charge and the born electrons' kinetic energy of `ion_window` at `x = (a0, density, prefactor, start)` (the tests' `ion_objectives`)."
function objectives(x)
    T = promote_type(map(typeof, x)...)
    prob = ion_window(T; a0 = x[1], density = x[2], prefactor = x[3], start = x[4])
    st = ion_state(prob)
    run!(st, params(prob), prob.nsteps)
    e = ionization_electrons(st)
    alive = Array(e.status) .== 0x01
    ionization_lost(st) == 0 || error("a birth found no slot")
    w, u = Array(e.w)[alive], map(a -> Array(a)[alive], e.u)
    return [sum(w), sum(w .* (sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1))]
end

function scan_figure(dir)
    a = collect(range(0.6, 1.4; length = 33))
    vals = [objectives((x, X0[2:4]...)) for x in a]
    at = a[1:4:end]
    tangents = map(at) do x
        J = objectives((ForwardDiff.Dual(x, 1.0), X0[2:4]...))
        (ForwardDiff.value.(J), map(j -> ForwardDiff.partials(j)[1], J))
    end
    fig = Figure(size = (1000, 430))
    for (o, name) in enumerate(("born charge Σw (n_c·cells)", "born electrons' kinetic energy Σw(γ − 1)"))
        ax = Axis(fig[1, o]; xlabel = "a₀", ylabel = name, title = o == 1 ? "33 runs (dots), ForwardDiff tangents from 9 dual runs" : "110 steps, 64 × 16 cells")
        scatter!(ax, a, [v[o] for v in vals]; markersize = 6, color = :black)
        for (x, (v, d)) in zip(at, tangents)
            lines!(ax, [x - 0.04, x + 0.04], [v[o] - 0.04 * d[o], v[o] + 0.04 * d[o]]; color = :crimson, linewidth = 2)
        end
    end
    ad = ForwardDiff.jacobian(x -> objectives(x), collect(X0))
    table = Dict{String,Any}[]
    for (o, name) in enumerate(("charge", "energy")), i in 1:4
        h = X0[i] * 2.0^-10
        fd = (objectives(ntuple(j -> j == i ? X0[j] + h : X0[j], 4))[o] - objectives(ntuple(j -> j == i ? X0[j] - h : X0[j], 4))[o]) / 2h
        push!(table, Dict{String,Any}("objective" => name, "parameter" => NAMES[i], "forwarddiff" => ad[o, i], "central_difference" => fd,
                                      "h" => h, "relative_difference" => abs(fd - ad[o, i]) / abs(ad[o, i])))
    end
    println("wrote ", save_plot(dir, "objective_scan", fig, Dict("a0" => a, "objectives" => vals, "tangent_a0" => collect(at),
                                                                   "tangent_values" => first.(tangents), "tangent_slopes" => last.(tangents),
                                                                   "x0" => collect(X0), "parameters" => collect(NAMES), "derivatives" => table)))
    return table
end

sig(x) = string(round(x; sigdigits = 4))

"""
    report_table(dir) -> String

The table of reports/wp-059-field-ionization.md from the sidecars of the run directory `dir`
(`julia +1.13 --project=scripts scripts/ionization/ionization.jl table <dir>`): the rates against
FBPIC's coefficient lines; per vacuum run the fraction next to the axis, the integral of the rate
in the paraxial field there, the same integral with the field multiplied by `gather_factor` and by
the run's peak |Ey| over the paraxial field's at that distance from the axis (computed here, from
the sidecar's level), and the largest difference over the column; and the derivatives of the two
objectives with their relative difference to a central difference.
"""
function report_table(dir)
    rates = JSON.parsefile(joinpath(dir, "adk_rates.json"))["largest_relative_difference"]
    vac = JSON.parsefile(joinpath(dir, "vacuum_fraction.json"))
    der = JSON.parsefile(joinpath(dir, "objective_scan.json"))["derivatives"]
    lv = vac["level"]
    level = ADKLevel(lv["prefactor"], lv["field"], lv["power"]; cutoff = lv["cutoff"])
    io = IOBuffer()
    println(io, "| measure | value |")
    println(io, "|---|---|")
    println(io, "| rates against FBPIC's coefficient lines with the formula's unit of energy, every rate: largest relative difference | ",
            sig(rates["rydberg, every rate above 1e-300"]), " |")
    println(io, "| the same with FBPIC's unit of energy, rates above 1e-12 ω₀; above 1e-3 ω₀ | ", sig(rates["fbpic, rates above 1e-12"]), "; ",
            sig(rates["fbpic, rates above 1e-3"]), " |")
    for r in vac["runs"]
        n, dy = r["cells_per_wavelength"], r["nearest_axis_dy"]
        off_axis = exp(-(dy / VAC.waist)^2)                      # the paraxial amplitude at dy over the axis' (focal plane)
        seen = gather_factor(n) * r["peak_Ey_at_focus"] / (r["peak_Ey_paraxial"] * off_axis)
        println(io, "| vacuum pulse at λ₀/", n, " (Δt = ", sig(r["dt"]), ", ", r["steps"], " steps), fraction released next to the axis: the run; the integral; ",
                "the integral with the field times ", sig(seen), "; largest difference to the integral over the column | ", sig(r["nearest_axis_fraction"]), "; ",
                sig(r["nearest_axis_analytic"]), "; ", sig(analytic_fraction(level, dy; scale = seen)), "; ", sig(r["largest_abs_difference"]), " |")
    end
    for name in ("charge", "energy")
        rows = filter(r -> r["objective"] == name, der)
        println(io, "| born ", name, ", ForwardDiff derivative in ", join((r["parameter"] for r in rows), ", "), " | ",
                join((sig(r["forwarddiff"]) for r in rows), "; "), " |")
        println(io, "| born ", name, ", relative difference to a central difference | ", join((sig(r["relative_difference"]) for r in rows), "; "), " |")
    end
    return String(take!(io))
end

function ionization_report()
    dir = run_dir("ionization", @__FILE__; work_package = "WP-059", prefix = "ionization",
                  extra_inputs = [joinpath(@__DIR__, "atomic.jl"), joinpath(@__DIR__, "ionization_energies.toml"), joinpath(@__DIR__, "fbpic_reference.toml"),
                                  joinpath(EM_TEST, "ionization_cases.jl")],
                  params = Dict{String,Any}("wavelength_m" => WAVELENGTH, "type" => "Float64", "mode" => "Reference",
                                            "vacuum" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(VAC)),
                                            "vacuum_cells_per_wavelength" => [16, 32, 64], "scan_x0" => collect(X0)))
    summary = rates_figure(dir)
    println("\nLargest relative difference of the rates to FBPIC's coefficient lines:")
    foreach(k -> println("  ", k, ": ", sig(summary[k])), sort(collect(keys(summary))))
    rows = vacuum_figure(dir)
    println("\n| cells per λ₀ along x | Δt (1/ω₀) | steps | fraction next to the axis | the integral there | largest difference over the column | peak \\|Ey\\| at the focus | the paraxial field's |")
    println("|---|---|---|---|---|---|---|---|")
    for r in rows
        println("| ", r["cells_per_wavelength"], " | ", sig(r["dt"]), " | ", r["steps"], " | ", sig(r["nearest_axis_fraction"]), " | ",
                sig(r["nearest_axis_analytic"]), " | ", sig(r["largest_abs_difference"]), " | ", sig(r["peak_Ey_at_focus"]), " | ", sig(r["peak_Ey_paraxial"]), " |")
    end
    table = scan_figure(dir)
    println("\n| objective | parameter | ForwardDiff | central difference | relative difference |")
    println("|---|---|---|---|---|")
    for r in table
        println("| ", r["objective"], " | ", r["parameter"], " | ", sig(r["forwarddiff"]), " | ", sig(r["central_difference"]), " | ", sig(r["relative_difference"]), " |")
    end
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 2 && ARGS[1] == "table"
        print(report_table(ARGS[2]))
    else
        println()
        print(report_table(ionization_report()))
    end
end
