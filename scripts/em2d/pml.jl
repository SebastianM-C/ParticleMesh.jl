# The convolutional PML of the open transverse axis (WP-034, the measurements behind T-540 to T-543).
#
# 1. Reflection against angle and thickness (pml_reflection.png): the energy fraction R that WP-027's
#    vacuum slab packet brings back into the interior after hitting the PML with PIConGPU's profile
#    (order 4, σ_max = 0.8·5/Δy, κ = 1, α = 0), at the carrier angles 0 to 1.2 rad, for layers of 8,
#    16 and 32 cells, in both polarizations; measured by the module's step and predicted by the PML's
#    discrete theory (`pml_rt`, lib/ParticleMeshElectromagnetic/test/pml_cases.jl), next to the
#    theory of WP-027's matched layer at the same thickness and its nominal attenuation
#    strength·cells = 32. The envelope is 4/cos θ wavelengths wide (WP-027's script takes 3/cos θ;
#    the PML's far smaller R at large angles needs the slow modes of the incident packet further
#    away at the measurement time).
# 2. Profiles (pml_profiles.png): at 8 cells, R against angle (theory, carrier) for PIConGPU's
#    profile, Smilei's default grading (σ = 20u², κ = 1 + 79u⁴), κ_max = 4 and a frequency shift.
# 3. A long run (pml_long_run.png): random fields in an open vacuum box with 8-cell PMLs, the energy
#    over the box against time for three profiles (T-542).
# 4. Gauss's law (pml_gauss.png): the plasma of T-543 over 200 steps with a 4-cell PML, Float64,
#    Reference: the residual on the rows off the layer and on the layer rows.
#   julia +1.13 --project=scripts -t 4 scripts/em2d/pml.jl
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: RHO_STAGGER_2D
using ForwardDiff: ForwardDiff
using Random: Xoshiro
value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)
absmax(x::Real) = abs(value_of(x))
include(joinpath(EM_TEST, "boundaries_cases.jl"))
include(joinpath(EM_TEST, "pml_cases.jl"))

const ANGLES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2)
const THICKNESS = (8, 16, 32)

geometry(θ, cells) = slab_geometry(; theta = θ, cells, strength = 32 / cells, width = 4 / cos(θ))

const PROFILES = (("PIConGPU (m = 4, κ = 1, α = 0)", cells -> PML(; cells)),
                  ("Smilei grading (20u², 1 + 79u⁴)", cells -> PML(; cells, sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4)),
                  ("κ_max = 4", cells -> PML(; cells, kappa_max = 4)),
                  ("α_max = 0.2", cells -> PML(; cells, alpha_max = 0.2)))

function reflection_rows()
    rows = Dict{String,Any}[]
    for cells in THICKNESS, θ in ANGLES, pol in (:TE, :TM)
        geo = geometry(θ, cells)
        pml = slab_pml(geo)
        pred = pml_prediction(geo, pml, pol)
        t = @elapsed r = slab_run(Float64, geo, pol; absorber = pml)
        row = Dict{String,Any}("cells" => cells, "theta" => θ, "pol" => String(pol), "sigma_max" => pml_sigma_max(pml, geo.dy),
                               "R" => r.R, "R_theory" => pred.R, "Tr" => r.Tr, "Tr_theory" => pred.Tr,
                               "R_layer_theory" => slab_prediction(geo, pol).R, "nx" => geo.nx, "ny" => geo.ny,
                               "steps" => round(Int, geo.tmeas / geo.dt), "seconds" => t)
        println(row)
        push!(rows, row)
    end
    return rows
end

"The theories on a fine grid of angles (the plotted lines), at the carrier's k_y only."
function reflection_lines(profiles; thickness = THICKNESS)
    out = Dict{String,Any}[]
    θs = collect(range(0, 1.3; length = 131))
    for cells in thickness, pol in (:TE, :TM), (name, mk) in profiles
        Rs = map(θ -> (geo = geometry(θ, cells); abs2(first(pml_rt(geo, mk(cells), pol, geo.ky0)))), θs)
        push!(out, Dict{String,Any}("cells" => cells, "pol" => String(pol), "profile" => name, "theta" => θs, "R_carrier" => Rs))
    end
    for cells in thickness, pol in (:TE, :TM)
        Rs = map(θ -> (geo = geometry(θ, cells); abs2(first(layer_rt(geo, pol, geo.ky0)))), θs)
        push!(out, Dict{String,Any}("cells" => cells, "pol" => String(pol), "profile" => "matched layer (WP-027)", "theta" => θs,
                                    "R_carrier" => Rs))
    end
    return out
end

function long_rows(; nsteps = 8000, n = (16, 40), every = 20)
    out = Dict{String,Any}[]
    L = (2π, 2π * n[2] / n[1])
    dt = 0.95 / sqrt(sum(d -> (n[d] / L[d])^2, 1:2))
    for (name, mk) in (PROFILES[1], PROFILES[2], PROFILES[4])
        prob = EMProblem("long", L, n, dt, nsteps, (), 1, (1, 1); boundaries = (:periodic, :open))
        st = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = mk(8)))
        initialize!(st, prob)
        rng = Xoshiro(3)
        for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
            a = getfield(st.fields, c)
            a .= randn(rng, size(a))
            fill_guards!(a, st.grid)
        end
        g = guards(st.grid)
        W() = sum(c -> sum(abs2, view(getfield(st.fields, c), (g + 1):(g + n[1]), (g + 1):(g + n[2]))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz)) / 2
        W0 = W()
        p = params(prob)
        t, w = Float64[0.0], Float64[1.0]
        for k in 1:nsteps
            step!(st, p, k)
            k % every == 0 && (push!(t, k * dt); push!(w, W() / W0))
        end
        push!(out, Dict{String,Any}("profile" => name, "t" => t, "W_over_W0" => w, "cells" => collect(n), "layer" => 8, "dt" => dt))
    end
    return out
end

function gauss_history(; nsteps = 200)
    prob, _ = open_plasma_2d(Float64; nsteps)
    pml = PML(; cells = 4)
    st = open_state(prob, pml)
    ρbar0 = open_residuals(st, 0.0).ρbar
    rows = Dict{String,Any}[]
    p = params(prob)
    for k in 0:prob.nsteps
        k > 0 && step!(st, p, k)
        r = open_residuals(st, ρbar0)
        push!(rows, Dict{String,Any}("step" => k, "t" => k * prob.dt, "free" => r.free, "layer" => r.layer, "scale" => r.scale,
                                     "divB" => r.divB, "bscale" => r.bscale, "alive" => alive_total(st)))
    end
    return Dict{String,Any}("cells" => collect(ncells(st.grid)), "layer" => 4, "dt" => prob.dt, "rows" => rows)
end

function pml_report()
    dir = run_dir("pml", @__FILE__; work_package = "WP-034", prefix = "em2d",
                  extra_inputs = [joinpath(EM_TEST, "boundaries_cases.jl"), joinpath(EM_TEST, "pml_cases.jl")],
                  params = Dict{String,Any}("angles" => collect(ANGLES), "thickness" => collect(THICKNESS), "profile" => "PIConGPU",
                                            "type" => "Float64", "ncx" => 16, "ncy" => 8, "courant" => 0.95,
                                            "width_wavelengths" => "4/cos θ", "margin" => 8, "layer_strength_times_cells" => 32))
    rows = reflection_rows()
    lines = reflection_lines(PROFILES[1:1])
    colors = Dict(8 => :royalblue, 16 => :darkorange, 32 => :seagreen)
    fig = Figure(size = (1000, 480))
    for (col, pol) in enumerate(("TE", "TM"))
        ax = Axis(fig[1, col]; yscale = log10, xlabel = "carrier angle θ from the normal (rad)", ylabel = "reflected energy fraction R",
                  title = "$pol: PML (PIConGPU's profile) against the matched layer")
        for l in lines
            l["pol"] == pol || continue
            layer = startswith(l["profile"], "matched")
            lines!(ax, l["theta"], max.(l["R_carrier"], 1e-30); color = colors[l["cells"]], linestyle = layer ? :dash : :solid,
                   label = layer ? "matched layer theory, $(l["cells"]) cells" : "PML theory, carrier, $(l["cells"]) cells")
        end
        for cells in THICKNESS
            sel = filter(r -> r["pol"] == pol && r["cells"] == cells, rows)
            scatter!(ax, [r["theta"] for r in sel], [r["R_theory"] for r in sel]; color = colors[cells], markersize = 12,
                     strokewidth = 1, label = "PML theory, packet band, $cells cells")
            scatter!(ax, [r["theta"] for r in sel], [r["R"] for r in sel]; color = :black, marker = :xcross, markersize = 9,
                     label = "PML measured (module's step)")
        end
        col == 1 && axislegend(ax; position = :lt, labelsize = 9, merge = true, unique = true)
    end
    println("wrote ", save_plot(dir, "pml_reflection", fig, Dict("runs" => rows, "theory_lines" => lines, "cpu" => cpu_model(),
                                                                 "threads" => Threads.nthreads())))
    plines = reflection_lines(PROFILES; thickness = (8,))
    fig2 = Figure(size = (1000, 460))
    pc = Dict(zip(first.(PROFILES), (:royalblue, :darkorange, :seagreen, :purple)))
    for (col, pol) in enumerate(("TE", "TM"))
        ax = Axis(fig2[1, col]; yscale = log10, xlabel = "carrier angle θ (rad)", ylabel = "R at the carrier (theory)",
                  title = "$pol, 8 cells, four profiles and the matched layer")
        for l in plines
            l["pol"] == pol || continue
            lines!(ax, l["theta"], max.(l["R_carrier"], 1e-30); color = get(pc, l["profile"], :gray),
                   linestyle = startswith(l["profile"], "matched") ? :dash : :solid, label = l["profile"])
        end
        col == 1 && axislegend(ax; position = :lt, labelsize = 10)
    end
    println("wrote ", save_plot(dir, "pml_profiles", fig2, Dict("theory_lines" => plines)))
    lr = long_rows()
    fig3 = Figure(size = (620, 440))
    ax = Axis(fig3[1, 1]; xlabel = "t (1/ω₀)", ylabel = "field energy over its start", title = "random fields, 16 × 40 cells, 8-cell PMLs")
    for (l, c) in zip(lr, (:royalblue, :darkorange, :purple))
        lines!(ax, l["t"], l["W_over_W0"]; color = c, label = l["profile"])
    end
    axislegend(ax; position = :rt)
    println("wrote ", save_plot(dir, "pml_long_run", fig3, Dict("runs" => lr)))
    gh = gauss_history()
    fig4 = Figure(size = (640, 420))
    t = [r["t"] for r in gh["rows"]]
    ax1 = Axis(fig4[1, 1]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "|D⁻·E − (ρ − ρ̄₀)| / ((n₁ + n₂) max|ρ|)",
               title = "Gauss's law, PML of $(gh["layer"]) cells on $(gh["cells"][2]) rows")
    lines!(ax1, t, [max(r["free"], 1e-300) / r["scale"] for r in gh["rows"]]; label = "rows off the layer")
    lines!(ax1, t, [max(r["layer"], 1e-300) / r["scale"] for r in gh["rows"]]; label = "layer rows")
    hlines!(ax1, [eps(Float64)]; color = :gray, linestyle = :dot, label = "eps(Float64)")
    axislegend(ax1; position = :rc)
    println("wrote ", save_plot(dir, "pml_gauss", fig4, merge(gh, Dict("cpu" => cpu_model(), "threads" => Threads.nthreads()))))
    return rows
end

pml_report()
