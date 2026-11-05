# The PML on the moving window's back edge (WP-034, the measurements behind T-570 and T-575).
#
# 1. The back-edge reflection (pml_back_reflection.png): T-472's vacuum pulse in the window of
#    `back_lwfa` (lib/ParticleMeshElectromagnetic/test/pml_back_cases.jl: `window_lwfa` with the
#    antenna at x = 10 and 72 × 16 cells, 200 steps), against a free-space reference (a static box
#    with 100 more columns behind, the antenna moved by as many, its bare back too far away to
#    matter within the run), for several back edges; the largest difference on the eight columns
#    behind the layer relative to the reference's largest field, against time and over the run, and
#    against the static twin with the same back edge (T-472's comparison) before the window moves
#    and on columns 25 … n₁ after. The 16-cell layer needs the antenna further forward (x = 16,
#    80 columns), so it is measured there, next to 8 cells in the same geometry.
# 2. A long run (pml_back_long_run.png): random fields in the empty window with an 8-cell back
#    edge, the field energy over the window against its start, in a window that never moves and in
#    one that moves from t = 0, for α_max = 0, 0.2 and 1 and the bare back edge (T-575).
# 3. Against the discrete theory (pml_back_theory.png): WP-027's slab packet turned on its side,
#    along −x into the back edge of a static window (`back_slab_run`), the reflected energy R
#    measured by the module's step and predicted by `back_rt` (pml_back_cases.jl), at 0 to 0.8 rad,
#    8 and 16 cells, both polarizations, PIConGPU's profile (T-576).
#   julia +1.13 --project=scripts -t 4 scripts/em2d/pml_back.jl
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
const ParticleMeshCore = ParticleMesh.ParticleMeshCore   # window_cases.jl names it; the scripts reach the engine through ParticleMesh
using ParticleMeshElectromagnetic
using ForwardDiff: ForwardDiff
using Random: Xoshiro
components(x::Real) = (x,)
components(d::ForwardDiff.Dual) = (ForwardDiff.value(d), Tuple(ForwardDiff.partials(d))...)
absmax(a) = maximum(abs, components(a))
value_of(x::Real) = x
include(joinpath(EM_TEST, "window_cases.jl"))
include(joinpath(EM_TEST, "pml_back_cases.jl"))

const SMILEI = (; sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4)
const EDGES = (("field-free outside (WP-028)", nothing, BACK_POSITION, BACK_CELLS),
               ("PML, 4 cells", PML(; cells = 4), BACK_POSITION, BACK_CELLS),
               ("PML, 8 cells", PML(; cells = 8), BACK_POSITION, BACK_CELLS),
               ("PML, 8 cells, α_max = 0.2", PML(; cells = 8, alpha_max = 0.2), BACK_POSITION, BACK_CELLS),
               ("PML, 8 cells, κ_max = 5", PML(; cells = 8, kappa_max = 5), BACK_POSITION, BACK_CELLS),
               ("PML, 8 cells, Smilei's grading", PML(; cells = 8, SMILEI...), BACK_POSITION, BACK_CELLS),
               ("field-free outside, x_a = 16", nothing, 16, (80, 16)),
               ("PML, 8 cells, x_a = 16", PML(; cells = 8), 16, (80, 16)),
               ("PML, 16 cells, x_a = 16", PML(; cells = 16), 16, (80, 16)))

function reflection_rows()
    rows = Dict{String,Any}[]
    for (name, back, position, cells) in EDGES
        t = @elapsed r = back_comparison(back; position, cells)
        row = Dict{String,Any}("edge" => name, "position" => position, "cells" => collect(cells),
                               "layer" => back === nothing ? 0 : back.cells, "shifts" => r.shifts, "before" => r.before,
                               "far" => r.far, "near" => r.near, "history" => r.history, "seconds" => t)
        println(Dict(k => v for (k, v) in row if k != "history"))
        push!(rows, row)
    end
    return rows
end

const LONG = (("field-free outside", nothing), ("α_max = 0", PML(; cells = 8)), ("α_max = 0.2", PML(; cells = 8, alpha_max = 0.2)),
              ("α_max = 1", PML(; cells = 8, alpha_max = 1)))

function long_rows(; nsteps = 12000, every = 50)
    rows = Dict{String,Any}[]
    for (frame, start) in (("static", Inf), ("moving", 0.0)), (name, back) in LONG
        t = @elapsed r = back_long_run(back; start, nsteps, every)
        println(frame, " ", name, ": max ", maximum(r.w), ", end ", last(r.w), " (", round(t; digits = 1), " s)")
        push!(rows, Dict{String,Any}("frame" => frame, "edge" => name, "start" => string(start), "t" => r.t, "W_over_W0" => r.w))
    end
    return rows
end

const ANGLES = (0.0, 0.2, 0.4, 0.6, 0.8)

function theory_rows()
    rows = Dict{String,Any}[]
    for cells in (8, 16), θ in ANGLES, pol in (:TE, :TM)
        geo = back_slab_geometry(; theta = θ, cells, width = θ >= 0.8 ? 4 : 3)
        pml = PML(; cells)
        t = @elapsed r = back_slab_run(Float64, geo, pml, pol)
        row = Dict{String,Any}("cells" => cells, "theta" => θ, "pol" => String(pol), "R" => r.R, "R_theory" => back_prediction(geo, pml, pol),
                               "nx" => geo.nx, "ny" => geo.ny, "steps" => r.k, "seconds" => t)
        println(row)
        push!(rows, row)
    end
    return rows
end

function back_report()
    dir = run_dir("pml-back", @__FILE__; work_package = "WP-034", prefix = "em2d",
                  extra_inputs = [joinpath(EM_TEST, "window_cases.jl"), joinpath(EM_TEST, "pml_back_cases.jl")],
                  params = Dict{String,Any}("case" => "back_lwfa", "position" => BACK_POSITION, "cells" => collect(BACK_CELLS),
                                            "steps" => 200, "free_extra_columns" => 100, "type" => "Float64", "mode" => "Reference",
                                            "long_run_steps" => 12000, "long_run_layer" => 8,
                                            "theory_angles" => collect(ANGLES), "theory_cells" => [8, 16], "theory_ncx" => 8, "theory_ncy" => 16))
    rows = reflection_rows()
    fig = Figure(size = (720, 460))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "step", ylabel = "max |ΔE, ΔB| / max |E, B| behind the layer",
              title = "window against free space, T-472's vacuum pulse")
    for r in rows
        lines!(ax, 1:length(r["history"]), max.(r["history"], 1e-20); label = r["edge"],
               linestyle = r["position"] == BACK_POSITION ? :solid : :dash)
    end
    axislegend(ax; position = :lb, labelsize = 9)
    println("wrote ", save_plot(dir, "pml_back_reflection", fig, Dict("runs" => rows, "cpu" => cpu_model(), "threads" => Threads.nthreads())))
    lr = long_rows()
    fig2 = Figure(size = (1000, 430))
    for (col, frame) in enumerate(("static", "moving"))
        ax2 = Axis(fig2[1, col]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "field energy over its start",
                   title = "random fields, $frame window, 8-cell back edge")
        for r in lr
            r["frame"] == frame || continue
            lines!(ax2, r["t"], r["W_over_W0"]; label = r["edge"])
        end
        axislegend(ax2; position = frame == "static" ? :lt : :rt, labelsize = 10)
    end
    println("wrote ", save_plot(dir, "pml_back_long_run", fig2, Dict("runs" => lr)))
    tr = theory_rows()
    fig3 = Figure(size = (1000, 430))
    for (col, pol) in enumerate(("TE", "TM"))
        ax3 = Axis(fig3[1, col]; yscale = log10, xlabel = "carrier angle θ from the normal (rad)", ylabel = "reflected energy fraction R",
                   title = "$pol: back edge of a static window, PIConGPU's profile")
        for (cells, color) in ((8, :royalblue), (16, :darkorange))
            sel = filter(r -> r["pol"] == pol && r["cells"] == cells, tr)
            scatter!(ax3, [r["theta"] for r in sel], [r["R_theory"] for r in sel]; color, markersize = 12, label = "theory, $cells cells")
            scatter!(ax3, [r["theta"] for r in sel], [r["R"] for r in sel]; color = :black, marker = :xcross, markersize = 9,
                     label = "measured (module's step)")
        end
        col == 1 && axislegend(ax3; position = :rc, labelsize = 10, merge = true, unique = true)
    end
    println("wrote ", save_plot(dir, "pml_back_theory", fig3, Dict("runs" => tr, "cpu" => cpu_model(), "threads" => Threads.nthreads())))
    return rows, lr, tr
end

back_report()
