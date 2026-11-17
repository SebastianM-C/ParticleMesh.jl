# The measurements of WP-063, the level with the barrier-suppression factor of Tong and Lin
# (reports/wp-063-barrier-suppression.md).
#
# 1. The rates (bsi_rates.png): the helium-like ions of `ionization_energies.toml` at 0.8 µm, each
#    as `tong_lin_level` makes it and as the same level without the factor (`adk_level` with the
#    fit's conventions), in atomic units against the field over the field at which the barrier is
#    gone, with the MCTDHF rates of Lötstedt, Ciappina and Yamanouchi (`tong_lin.toml`) on top, and
#    the ratio of both to those rates.
# 2. A pulse over test atoms in vacuum (bsi_fraction.png): the fraction of N⁵⁺ ionized to N⁶⁺ on
#    the axis after the Gaussian pulse of `ionization.jl`'s vacuum runs (waist 10, duration 20),
#    1 − exp(−∫W(|E(t)|)dt) in the paraxial field, against a₀ for three levels: WP-059's default
#    (the ADK level with FBPIC's conventions), the ADK level with the fit's conventions, and the
#    Tong–Lin level; and the run itself at λ₀/32 with the Tong–Lin level at three amplitudes,
#    against the integral with the field the atoms see (the gather's factor and the run's peak).
# 3. An ionization-injection run (bsi_injection.png): a coarse 2D wakefield run (448 × 128 cells,
#    λ₀/32 along x, a₀ = 2.2, a moving window, a plasma of 0.072 n_c, a section doped with 1 % of
#    nitrogen whose K shell the field ionizes) to t = 300/ω₀ with each of the three pairs of
#    levels: the born electrons' charge, the part of it with uₓ > 5, and their spectrum.
#   julia +1.13 --project=scripts -t 4 scripts/ionization/barrier_suppression.jl                 the run, its figures and tables
#   julia +1.13 --project=scripts scripts/ionization/barrier_suppression.jl table <run directory>   the report's table from a run's sidecars
include(joinpath(@__DIR__, "ionization.jl"))        # the user layer, the vacuum pulse's run and integral

const HARTREE_EV = FINE_STRUCTURE^2 * ELECTRON_REST_ENERGY_EV
"mₑcω₀/e and ω₀ in atomic units at the report's wavelength."
const AU_FIELD, AU_RATE = field_unit_au(WAVELENGTH), photon_energy(WAVELENGTH) / FINE_STRUCTURE^2

"He, Li⁺, N⁵⁺, … for the helium-like ion of `element`."
helium_like_label(element, Z) = Z == 2 ? "He" : element * ("⁺", "²⁺", "³⁺", "⁴⁺", "⁵⁺", "⁶⁺", "⁷⁺", "⁸⁺", "⁹⁺", "¹⁰⁺", "¹¹⁺", "¹²⁺", "¹³⁺", "¹⁴⁺", "¹⁵⁺", "¹⁶⁺")[Z - 2]

"The two levels of the comparison for the transition that leaves `el` with the charge `charge`: with the factor, and the same without."
fit_levels(el, charge; cutoff = 1.0e6) =
    (tong_lin_level(Float64, el, charge; wavelength = WAVELENGTH, cutoff),
     adk_level(Float64, el, charge; wavelength = WAVELENGTH, lstar = 0, electrons = charge == nuclear_charge(el) - 1 ? 2 : 1, cutoff))

"The band of T-821 a field falls in, by its ratio to the barrier's field."
band(x) = x < 0.7 ? "below 0.7 E_bs" : x <= 1.75 ? "0.7 to 1.75 E_bs" : "above 1.75 E_bs"

# --- 1 ------------------------------------------------------------------------------------------

function reference_rows()
    rows = Dict{String,Any}[]
    for ion in TONG_LIN["mctdhf"]["ions"]
        el, Z = Symbol(ion["element"]), ion["Z"]
        tl, adk = fit_levels(el, Z - 1)
        num = tong_lin_numbers(ionization_energy(el, Z - 1), Z - 1; alpha = tong_lin_alpha(el, Z - 1), wavelength = WAVELENGTH, lstar = 0)
        E = [Fs * Z^3 / 8 / AU_FIELD for Fs in ion["scaled_field"]]
        push!(rows, Dict{String,Any}("element" => ion["element"], "Z" => Z, "label" => helium_like_label(ion["element"], Z),
                                     "alpha" => tong_lin_alpha(el, Z - 1), "energy_eV" => ionization_energy(el, Z - 1), "nstar" => num.nstar,
                                     "suppression" => num.suppression, "barrier_field_a0" => num.barrier_field,
                                     "barrier_field_au" => num.barrier_field * AU_FIELD, "scaled_field" => ion["scaled_field"],
                                     "field_over_barrier" => E ./ num.barrier_field, "rate_reference_au" => ion["rate"],
                                     "rate_tong_lin_au" => [ionization_rate(tl, e) * AU_RATE for e in E],
                                     "rate_adk_au" => [ionization_rate(adk, e) * AU_RATE for e in E]))
    end
    return rows
end

"Per band of fields, the largest |W/W_ref − 1| of the Tong–Lin level over the rows, and the least and largest ADK rate over the reference from 0.95 E_bs on."
function reference_summary(rows)
    worst = Dict{String,Any}("0.7 to 1.75 E_bs" => 0.0, "below 0.7 E_bs" => 0.0, "above 1.75 E_bs" => 0.0)
    lo, hi = Inf, 0.0
    for r in rows, (x, ref, tl, adk) in zip(r["field_over_barrier"], r["rate_reference_au"], r["rate_tong_lin_au"], r["rate_adk_au"])
        worst[band(x)] = max(worst[band(x)], abs(tl / ref - 1))
        x >= 0.95 && ((lo, hi) = (min(lo, adk / ref), max(hi, adk / ref)))
    end
    worst["ADK over reference from 0.95 E_bs, least"] = lo
    worst["ADK over reference from 0.95 E_bs, largest"] = hi
    return worst
end

function rates_figure_bsi(dir)
    rows = reference_rows()
    fig = Figure(size = (1100, 720))
    x = range(0.35, 3.1; length = 240)
    for (k, name) in enumerate(("He", "N", "Ar"))
        r = only(filter(r -> r["element"] == name, rows))
        tl, adk = fit_levels(Symbol(name), r["Z"] - 1)
        ax = Axis(fig[1, k]; yscale = log10, xlabel = "E / E_bs", ylabel = k == 1 ? "rate (atomic units)" : "",
                  title = "$(r["label"]): E_bs = $(round(r["barrier_field_au"]; sigdigits = 3)) a.u. (a₀ = $(round(r["barrier_field_a0"]; sigdigits = 3))), α = $(r["alpha"])")
        e = x .* r["barrier_field_a0"]
        lines!(ax, x, [ionization_rate(adk, v) * AU_RATE for v in e]; color = :gray40, linestyle = :dash, label = "tunnelling (ADK), no factor")
        lines!(ax, x, [ionization_rate(tl, v) * AU_RATE for v in e]; color = :crimson, label = "with Tong and Lin's factor")
        scatter!(ax, r["field_over_barrier"], r["rate_reference_au"]; color = :black, markersize = 8, label = "MCTDHF (Lötstedt et al.)")
        vlines!(ax, [1.0]; color = :gray70, linewidth = 1)
        xlims!(ax, 0.3, k == 1 ? 3.1 : 1.8)
        k == 1 && axislegend(ax; position = :rb, labelsize = 10)
    end
    axt = Axis(fig[2, 1:2]; xlabel = "E / E_bs", ylabel = "rate with the factor / MCTDHF rate", title = "the Tong–Lin level against the reference, six helium-like ions")
    axa = Axis(fig[2, 3]; xlabel = "E / E_bs", ylabel = "ADK rate / MCTDHF rate", title = "the same level without the factor")
    vspan!(axt, 0.7, 1.75; color = (:gray, 0.12))
    hlines!(axt, [1.0]; color = :gray50, linewidth = 1)
    hlines!(axa, [1.0]; color = :gray50, linewidth = 1)
    for (r, marker) in zip(rows, (:circle, :rect, :utriangle, :dtriangle, :diamond, :xcross))
        tl = r["rate_tong_lin_au"] ./ r["rate_reference_au"]
        adk = r["rate_adk_au"] ./ r["rate_reference_au"]
        scatterlines!(axt, r["field_over_barrier"], tl; marker, markersize = 9, label = r["label"])
        scatterlines!(axa, r["field_over_barrier"], adk; marker, markersize = 9, label = r["label"])
    end
    axislegend(axt; position = :lb, labelsize = 10, nbanks = 3)
    summary = reference_summary(rows)
    println("wrote ", save_plot(dir, "bsi_rates", fig, Dict("ions" => rows, "summary" => summary, "wavelength_m" => WAVELENGTH,
                                                            "field_unit_au" => AU_FIELD, "rate_unit_au" => AU_RATE)))
    return summary
end

# --- 2 ------------------------------------------------------------------------------------------

"The three choices of the levels N⁵⁺ → N⁶⁺ and N⁶⁺ → N⁷⁺ that sections 2 and 3 compare, each a name and a function of the cutoff."
const VARIANTS = (("ADK, WP-059's default (FBPIC's conventions)",
                   c -> (adk_level(Float64, :N, 6; wavelength = WAVELENGTH, cutoff = c), adk_level(Float64, :N, 7; wavelength = WAVELENGTH, cutoff = c))),
                  ("ADK, the fit's conventions (l* = 0, both 1s electrons)",
                   c -> (fit_levels(:N, 6; cutoff = c)[2], fit_levels(:N, 7; cutoff = c)[2])),
                  ("with Tong and Lin's factor",
                   c -> (fit_levels(:N, 6; cutoff = c)[1], fit_levels(:N, 7; cutoff = c)[1])))
const VARIANT_COLORS = (:gray40, :dodgerblue3, :crimson)
const VARIANT_STYLES = (:dash, :dashdot, :solid)

"The a₀ at which `f(a₀)` crosses 1/2, by linear interpolation on the grid `a` (NaN when it does not)."
function half_point(a, f)
    i = findfirst(>=(0.5), f)
    (i === nothing || i == 1) && return NaN
    return a[i - 1] + (0.5 - f[i - 1]) / (f[i] - f[i - 1]) * (a[i] - a[i - 1])
end

function fraction_figure(dir)
    a = collect(range(1.0, 4.0; length = 61))
    levels = map(v -> v[2](64.0)[1], VARIANTS)
    curves = [[analytic_fraction(l, 0.0; a0 = x, ds = 2.0e-3) for x in a] for l in levels]
    fig = Figure(size = (1100, 440))
    ax = Axis(fig[1, 1]; xlabel = "a₀ of the pulse (waist 10 c/ω₀, duration 20/ω₀, λ₀ = 0.8 µm)", ylabel = "fraction of N⁵⁺ ionized to N⁶⁺ on the axis",
              title = "after the pulse, in vacuum: 1 − exp(−∫W dt) (lines), the run at λ₀/32 (crosses)")
    for (k, (name, _)) in enumerate(VARIANTS)
        lines!(ax, a, curves[k]; color = VARIANT_COLORS[k], linestyle = VARIANT_STYLES[k], label = name)
    end
    Ebs = tong_lin_numbers(ionization_energy(:N, 6), 6; alpha = 0, wavelength = WAVELENGTH, lstar = 0).barrier_field
    vlines!(ax, [Ebs]; color = :gray70, linewidth = 1)
    text!(ax, Ebs, 0.03; text = " E_bs of N⁵⁺", fontsize = 10, color = :gray40)
    runs = Dict{String,Any}[]
    tl = levels[3]
    for a0 in (1.8, 2.2, 2.6)
        r = vacuum_run(tl, 32; a0)
        i0 = argmin(abs.(r.dy))
        seen = gather_factor(32) * r.peak / (peak_paraxial(; a0) * exp(-(r.dy[i0] / VAC.waist)^2))
        row = Dict{String,Any}("a0" => a0, "cells" => collect(r.cells), "steps" => r.nsteps, "dt" => r.dt, "nearest_axis_dy" => r.dy[i0],
                               "fraction" => r.fraction[i0], "integral" => analytic_fraction(tl, r.dy[i0]; a0),
                               "field_factor" => seen, "integral_with_field_factor" => analytic_fraction(tl, r.dy[i0]; a0, scale = seen),
                               "seconds" => r.seconds)
        println(row)
        push!(runs, row)
        scatter!(ax, [a0], [r.fraction[i0]]; marker = :xcross, color = :black, markersize = 12)
    end
    axislegend(ax; position = :rb, labelsize = 10)
    axr = Axis(fig[1, 2]; yscale = log10, xlabel = "field (mₑcω₀/e)", ylabel = "rate (1/ω₀)", title = "the rates of nitrogen's K shell: N⁵⁺ → N⁶⁺ (thick), N⁶⁺ → N⁷⁺ (thin)")
    e = range(0.8, 8.0; length = 240)
    for (k, (name, make)) in enumerate(VARIANTS), (j, level) in enumerate(make(1.0e6))
        lines!(axr, e, [ionization_rate(level, x) for x in e]; color = VARIANT_COLORS[k], linestyle = VARIANT_STYLES[k], linewidth = j == 1 ? 2.5 : 1)
    end
    ylims!(axr, 1.0e-6, 1.0e2)
    half = [half_point(a, c) for c in curves]
    println("wrote ", save_plot(dir, "bsi_fraction", fig, Dict("a0" => a, "variants" => [v[1] for v in VARIANTS], "fraction" => curves,
                                                               "a0_of_half" => half, "barrier_field_a0" => Ebs, "runs" => runs,
                                                               "pulse" => Dict(string(k) => v for (k, v) in pairs(VAC) if !(v isa Tuple)))))
    return (; half, runs)
end

# --- 3 ------------------------------------------------------------------------------------------

const INJ = (; a0 = 2.2, waist = 10.0, duration = 6.0, delay = 54.0, focus = 80.0, vacuum = 64.0, ramp = 8.0, window_start = 108.0, density = 0.072,
             cells_per_wavelength = 32, cells = (448, 128), ppc = (2, 2), dy_cells = 4, t_end = 300.0, pml_y = 12, pml_x = 8, dopant = 0.00072,
             section = (76.0, 100.0, 4.0), half_rows = 18, cutoff = 24.0, capacity = 400_000, fast = 5.0)

"The injection run's problem with the levels `levels`."
function injection_problem(levels)
    h = 2π / INJ.cells_per_wavelength
    hy = INJ.dy_cells * h
    dt = 0.95 / sqrt(1 / h^2 + 1 / hy^2)
    yc = INJ.cells[2] * hy / 2
    pulse = GaussianLaser(; a0 = INJ.a0, waist = INJ.waist, duration = INJ.duration, delay = INJ.delay, focus = INJ.focus, position = 12h, center = yc)
    e = SpeciesDescription{Float64}(:electron, -1.0, 1.0, INJ.density, 0.0, (1, 0), INJ.ppc, zero3(Float64), zero3(Float64), 1)
    ion = FieldIonization(; density = INJ.dopant, levels, capacity = INJ.capacity, every = 1, ppc = (1, 1),
                          rows = (yc - INJ.half_rows * hy, yc + INJ.half_rows * hy),
                          section = DopantSection(; start = INJ.section[1], stop = INJ.section[2], ramp = INJ.section[3]))
    return EMProblem("ionization-injection", (INJ.cells[1] * h, INJ.cells[2] * hy), INJ.cells, dt, ceil(Int, INJ.t_end / dt), (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :open), window = MovingWindow(; start = INJ.window_start),
                     profile = DensityRamp(; vacuum = INJ.vacuum, ramp = INJ.ramp), ionization = ion)
end

"The born electrons alive in the window: their weights and uₓ."
function born_electrons(st)
    e = ionization_electrons(st)
    alive = Array(e.status) .== 0x01
    return Array(e.w)[alive], Array(e.u[1])[alive]
end

"""
The injection run with `levels`: every 50 steps the born electrons' charge in the window and the
part with uₓ > `INJ.fast`, both over the dopant's density (cells of dopant, n_c·cells over n_c),
and at the end their spectrum in uₓ (weight per bin of width 1 over the dopant's density).
"""
function injection_run(levels)
    prob = injection_problem(levels)
    st = EMState(prob, EMNumerics(; mode = Reference(), absorber = PML(; cells = INJ.pml_y), back_absorber = PML(; cells = INJ.pml_x)))
    p = params(prob)
    initialize!(st, prob)
    t, charge, fast = Float64[], Float64[], Float64[]
    seconds = @elapsed for k in 1:prob.nsteps
        step!(st, p, k)
        if k % 50 == 0 || k == prob.nsteps
            w, ux = born_electrons(st)
            push!(t, k * prob.dt)
            push!(charge, sum(w) / INJ.dopant)
            push!(fast, sum(w[ux .> INJ.fast]; init = 0.0) / INJ.dopant)
        end
    end
    w, ux = born_electrons(st)
    edges = 0:1:45
    spectrum = [sum(w[(ux .>= lo) .& (ux .< lo + 1)]; init = 0.0) / INJ.dopant for lo in edges[1:(end - 1)]]
    return (; t, charge, fast, spectrum, edges = collect(edges), ux_max = maximum(ux), alive = length(w), lost = ionization_lost(st), seconds,
            steps = prob.nsteps, dt = prob.dt)
end

function injection_figure(dir)
    fig = Figure(size = (1100, 440))
    axc = Axis(fig[1, 1]; xlabel = "t (1/ω₀)", ylabel = "born electrons in the window (cells of dopant)",
               title = "all born electrons (thin) and those with uₓ > $(INJ.fast) (thick)")
    axs = Axis(fig[1, 2]; xlabel = "uₓ = γvₓ/c", ylabel = "weight per unit uₓ (cells of dopant)", title = "the born electrons' spectrum at t = $(INJ.t_end)/ω₀, a₀ = $(INJ.a0)")
    rows = Dict{String,Any}[]
    for (k, (name, make)) in enumerate(VARIANTS)
        r = injection_run(make(INJ.cutoff))
        row = Dict{String,Any}("variant" => name, "t" => r.t, "charge" => r.charge, "fast" => r.fast, "spectrum" => r.spectrum, "spectrum_edges" => r.edges,
                               "charge_end" => r.charge[end], "fast_end" => r.fast[end], "ux_max" => r.ux_max, "alive" => r.alive,
                               "lost" => r.lost, "seconds" => r.seconds, "steps" => r.steps, "dt" => r.dt)
        println(Dict(k => v for (k, v) in row if !(v isa Vector)))
        push!(rows, row)
        lines!(axc, r.t, r.charge; color = VARIANT_COLORS[k], linestyle = VARIANT_STYLES[k], linewidth = 1)
        lines!(axc, r.t, r.fast; color = VARIANT_COLORS[k], linestyle = VARIANT_STYLES[k], linewidth = 2.5, label = name)
        stairs!(axs, r.edges[1:(end - 1)] .+ 0.5, r.spectrum; step = :center, color = VARIANT_COLORS[k], linestyle = VARIANT_STYLES[k], label = name)
    end
    axislegend(axc; position = :lt, labelsize = 10)
    println("wrote ", save_plot(dir, "bsi_injection", fig, Dict("case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(INJ)),
                                                                "runs" => rows)))
    return rows
end

"""
    bsi_table(dir) -> String

The table of reports/wp-063-barrier-suppression.md from the sidecars of the run directory `dir`
(`julia +1.13 --project=scripts scripts/ionization/barrier_suppression.jl table <dir>`).
"""
function bsi_table(dir)
    rates = JSON.parsefile(joinpath(dir, "bsi_rates.json"))
    frac = JSON.parsefile(joinpath(dir, "bsi_fraction.json"))
    inj = JSON.parsefile(joinpath(dir, "bsi_injection.json"))["runs"]
    s = rates["summary"]
    io = IOBuffer()
    println(io, "| measure | value |")
    println(io, "|---|---|")
    println(io, "| the Tong–Lin level against the MCTDHF rates of six helium-like ions, largest \\|W/W_ref − 1\\|: fields from 0.7 to 1.75 E_bs; below 0.7 E_bs; above 1.75 E_bs | ",
            sig(s["0.7 to 1.75 E_bs"]), "; ", sig(s["below 0.7 E_bs"]), "; ", sig(s["above 1.75 E_bs"]), " |")
    println(io, "| the same level without the factor (ADK) over the MCTDHF rate, fields from 0.95 E_bs on: least; largest | ",
            sig(s["ADK over reference from 0.95 E_bs, least"]), "; ", sig(s["ADK over reference from 0.95 E_bs, largest"]), " |")
    n = only(filter(r -> r["element"] == "N", rates["ions"]))
    println(io, "| N⁵⁺ → N⁶⁺ at 0.8 µm: E_bs in atomic units; as a laser amplitude a₀; the level's suppression 4α·n*² | ", sig(n["barrier_field_au"]), "; ",
            sig(n["barrier_field_a0"]), "; ", sig(n["suppression"]), " |")
    println(io, "| N⁵⁺ at 0.5, 0.75, 1, 1.25 and 1.5 E_bs, rate with the factor over the MCTDHF rate | ",
            join((sig(a / b) for (a, b) in zip(n["rate_tong_lin_au"], n["rate_reference_au"])), "; "), " |")
    println(io, "| the same without the factor | ", join((sig(a / b) for (a, b) in zip(n["rate_adk_au"], n["rate_reference_au"])), "; "), " |")
    println(io, "| vacuum pulse, the a₀ at which half of N⁵⁺ is ionized on the axis: ", join(frac["variants"], "; "), " | ",
            join((sig(x) for x in frac["a0_of_half"]), "; "), " |")
    for r in frac["runs"]
        println(io, "| vacuum pulse of a₀ = ", r["a0"], " at λ₀/32 with the factor (", r["steps"], " steps), fraction next to the axis: the run; the integral; the integral with the field times ",
                sig(r["field_factor"]), " | ", sig(r["fraction"]), "; ", sig(r["integral"]), "; ", sig(r["integral_with_field_factor"]), " |")
    end
    for r in inj
        println(io, "| injection run, ", r["variant"], ": born electrons in the window at the end (cells of dopant); those with uₓ > 5; largest uₓ; births that found no slot | ",
                sig(r["charge_end"]), "; ", sig(r["fast_end"]), "; ", sig(r["ux_max"]), "; ", r["lost"], " |")
    end
    return String(take!(io))
end

function bsi_report()
    dir = run_dir("barrier-suppression", @__FILE__; work_package = "WP-063", prefix = "ionization",
                  extra_inputs = [joinpath(@__DIR__, "atomic.jl"), joinpath(@__DIR__, "ionization_energies.toml"), joinpath(@__DIR__, "tong_lin.toml"),
                                  joinpath(@__DIR__, "ionization.jl"), joinpath(EM_TEST, "ionization_cases.jl")],
                  params = Dict{String,Any}("wavelength_m" => WAVELENGTH, "type" => "Float64", "mode" => "Reference",
                                            "vacuum" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(VAC)),
                                            "vacuum_a0" => [1.8, 2.2, 2.6], "vacuum_cells_per_wavelength" => 32,
                                            "injection" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(INJ))))
    rates_figure_bsi(dir)
    fraction_figure(dir)
    injection_figure(dir)
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 2 && ARGS[1] == "table"
        print(bsi_table(ARGS[2]))
    else
        println()
        print(bsi_table(bsi_report()))
    end
end
