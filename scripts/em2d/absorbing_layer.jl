# The open transverse boundary (WP-027, the measurements behind T-430 and T-431).
#
# 1. Reflection against angle and thickness (layer_reflection.png): the energy fraction R that a
#    vacuum slab packet (one k_x, a Gaussian envelope along y, built from the scheme's upward
#    eigenmodes) brings back into the interior after hitting the absorbing layer, at the carrier
#    angles 0 to 1.2 rad from the normal, for layers of 8, 16 and 32 cells at the fixed nominal
#    attenuation strength·cells = 32 (order 2), in both polarizations; measured by the module's
#    step and predicted by the discrete frequency-domain theory of the layer (`layer_rt`,
#    lib/ParticleMeshElectromagnetic/test/boundaries_cases.jl). The packet's envelope is
#    3/cos θ wavelengths wide, so its spread in angle is the same at every θ.
# 2. Strength (layer_strength.png): at 16 cells and θ = 0.4, R and the transmission Tr through
#    both layers and the seam against the strength, measured and predicted.
# 3. Gauss's law (layer_gauss.png): the plasma of T-431 (two species leaving through both edges)
#    over 200 steps, Float64, Reference: the Gauss residual on the rows off the layer and on the
#    layer rows, and the number of particles alive.
#   julia +1.13 --project=scripts scripts/em2d/absorbing_layer.jl
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: RHO_STAGGER_2D
using ForwardDiff: ForwardDiff
value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)
include(joinpath(EM_TEST, "boundaries_cases.jl"))

const ANGLES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2)
const THICKNESS = (8, 16, 32)
const NOMINAL = 32.0                   # strength · cells
const STRENGTHS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

geometry(θ, cells, strength) = slab_geometry(; theta = θ, cells, strength, width = 3 / cos(θ))

function reflection_rows()
    rows = Dict{String,Any}[]
    for cells in THICKNESS, θ in ANGLES, pol in (:TE, :TM)
        geo = geometry(θ, cells, NOMINAL / cells)
        pred = slab_prediction(geo, pol)
        t = @elapsed r = slab_run(Float64, geo, pol)
        row = Dict{String,Any}("cells" => cells, "theta" => θ, "pol" => String(pol), "strength" => geo.strength,
                               "R" => r.R, "R_theory" => pred.R, "Tr" => r.Tr, "Tr_theory" => pred.Tr,
                               "nx" => geo.nx, "ny" => geo.ny, "steps" => round(Int, geo.tmeas / geo.dt), "seconds" => t)
        println(row)
        push!(rows, row)
    end
    return rows
end

"The theory on a fine grid of angles (the plotted lines), at the carrier's k_y only."
function reflection_lines()
    out = Dict{String,Any}[]
    for cells in THICKNESS, pol in (:TE, :TM)
        θs = collect(range(0, 1.3; length = 131))
        Rs = map(θs) do θ
            geo = geometry(θ, cells, NOMINAL / cells)
            abs2(first(layer_rt(geo, pol, geo.ky0)))
        end
        push!(out, Dict{String,Any}("cells" => cells, "pol" => String(pol), "theta" => θs, "R_carrier" => Rs))
    end
    return out
end

function strength_rows()
    rows = Dict{String,Any}[]
    for s in STRENGTHS, pol in (:TE, :TM)
        geo = geometry(0.4, 16, s)
        pred = slab_prediction(geo, pol)
        r = slab_run(Float64, geo, pol)
        row = Dict{String,Any}("strength" => s, "pol" => String(pol), "R" => r.R, "R_theory" => pred.R, "Tr" => r.Tr,
                               "Tr_theory" => pred.Tr)
        println(row)
        push!(rows, row)
    end
    return rows
end

function gauss_history(; nsteps = 200)
    prob, ab = open_plasma_2d(Float64; nsteps)
    st = open_state(prob, ab)
    (; grid) = st
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    L = ab.cells
    I1 = (g + 1):(g + n[1])
    free = (g + 1 + L + 1):(g + 1 + n[2] - L - 1)
    lay = setdiff((g + 1):(g + n[2]), free)
    function residuals(ρbar0)
        ρd = field(CPU(), Float64, grid)
        for (s, ws) in zip(st.species, st.ws)
            sort_by_cell!(s, grid, ws)
            deposit!(Reference(), ρd, s, st.numerics.shape, RHO_STAGGER_2D, grid, ws)
        end
        fold_guards!(ρd, grid)
        ρ, Ex, Ey = Array(ρd), Array(st.fields.Ex), Array(st.fields.Ey)
        res(i, j) = abs((Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2] - (ρ[i, j] - ρbar0))
        ρint = ρ[I1, (g + 1):(g + n[2])]
        return (maximum(res(i, j) for i in I1, j in free), maximum(res(i, j) for i in I1, j in lay), sum(ρint) / prod(n),
                sum(n) * maximum(abs, ρint))
    end
    ρbar0 = residuals(0.0)[3]
    alive() = sum(s -> count(==(0x01), Array(s.status)), st.species)
    rows = Dict{String,Any}[]
    p = params(prob)
    for k in 0:prob.nsteps
        k > 0 && step!(st, p, k)
        rf, rl, _, sc = residuals(ρbar0)
        push!(rows, Dict{String,Any}("step" => k, "t" => k * prob.dt, "free" => rf, "layer" => rl, "scale" => sc,
                                     "alive" => alive()))
    end
    return Dict{String,Any}("cells" => collect(n), "layer" => L, "strength" => ab.strength, "dt" => prob.dt, "rows" => rows,
                            "capacity" => sum(capacity, st.species))
end

function layer_report()
    dir = run_dir("absorbing-layer", @__FILE__; work_package = "WP-027", prefix = "em2d",
                  extra_inputs = [joinpath(EM_TEST, "boundaries_cases.jl")],
                  params = Dict{String,Any}("angles" => collect(ANGLES), "thickness" => collect(THICKNESS),
                                            "strength_times_cells" => NOMINAL, "order" => 2, "strengths" => collect(STRENGTHS),
                                            "type" => "Float64", "ncx" => 16, "ncy" => 8, "courant" => 0.95,
                                            "width_wavelengths" => "3/cos θ", "margin" => 8))
    rows = reflection_rows()
    lines = reflection_lines()
    colors = Dict(8 => :royalblue, 16 => :darkorange, 32 => :seagreen)
    fig = Figure(size = (1000, 460))
    for (col, pol) in enumerate(("TE", "TM"))
        ax = Axis(fig[1, col]; yscale = log10, xlabel = "carrier angle θ from the normal (rad)", ylabel = "reflected energy fraction R",
                  title = "$pol ($(pol == "TE" ? "Ez, Bx, By" : "Bz, Ex, Ey")), strength·cells = $(Int(NOMINAL))")
        for l in lines
            l["pol"] == pol || continue
            lines!(ax, l["theta"], l["R_carrier"]; color = colors[l["cells"]], label = "theory, carrier, $(l["cells"]) cells")
        end
        for cells in THICKNESS
            sel = filter(r -> r["pol"] == pol && r["cells"] == cells, rows)
            scatter!(ax, [r["theta"] for r in sel], [r["R_theory"] for r in sel]; color = colors[cells], marker = :circle,
                     markersize = 12, strokewidth = 1, label = "theory, packet band, $cells cells")
            scatter!(ax, [r["theta"] for r in sel], [r["R"] for r in sel]; color = :black, marker = :xcross, markersize = 9,
                     label = "measured (module's step), $cells cells")
        end
        col == 1 && axislegend(ax; position = :rb, nbanks = 1, labelsize = 10, merge = true, unique = true)
    end
    p1 = save_plot(dir, "layer_reflection", fig, Dict("runs" => rows, "theory_lines" => lines, "cpu" => cpu_model(),
                                                      "threads" => Threads.nthreads()))
    println("wrote ", p1)
    srows = strength_rows()
    fig2 = Figure(size = (620, 460))
    ax = Axis(fig2[1, 1]; xscale = log10, yscale = log10, xlabel = "strength σ_max (ω₀)", ylabel = "energy fraction",
              title = "16 cells, θ = 0.4 rad")
    for (pol, c) in (("TE", :royalblue), ("TM", :darkorange))
        sel = filter(r -> r["pol"] == pol, srows)
        s = [r["strength"] for r in sel]
        lines!(ax, s, [r["R_theory"] for r in sel]; color = c, label = "R theory, $pol")
        scatter!(ax, s, [r["R"] for r in sel]; color = c, marker = :xcross, label = "R measured, $pol")
        lines!(ax, s, [r["Tr_theory"] for r in sel]; color = c, linestyle = :dash, label = "Tr theory, $pol")
        scatter!(ax, s, [r["Tr"] for r in sel]; color = c, marker = :circle, label = "Tr measured, $pol")
    end
    axislegend(ax; position = :lb, labelsize = 10)
    p2 = save_plot(dir, "layer_strength", fig2, Dict("runs" => srows, "cpu" => cpu_model(), "threads" => Threads.nthreads()))
    println("wrote ", p2)
    gh = gauss_history()
    fig3 = Figure(size = (900, 420))
    t = [r["t"] for r in gh["rows"]]
    ax1 = Axis(fig3[1, 1]; yscale = log10, xlabel = "t (1/ω₀)", ylabel = "|D⁻·E − (ρ − ρ̄₀)| / ((n₁ + n₂) max|ρ|)",
               title = "Gauss's law, layer of $(gh["layer"]) cells on $(gh["cells"][2]) rows")
    lines!(ax1, t, [max(r["free"], 1e-300) / r["scale"] for r in gh["rows"]]; label = "rows off the layer")
    lines!(ax1, t, [max(r["layer"], 1e-300) / r["scale"] for r in gh["rows"]]; label = "layer rows")
    hlines!(ax1, [eps(Float64)]; color = :gray, linestyle = :dot, label = "eps(Float64)")
    axislegend(ax1; position = :rc)
    ax2 = Axis(fig3[1, 2]; xlabel = "t (1/ω₀)", ylabel = "alive particles", title = "particles retired at the edges")
    lines!(ax2, t, [r["alive"] for r in gh["rows"]])
    p3 = save_plot(dir, "layer_gauss", fig3, merge(gh, Dict("cpu" => cpu_model(), "threads" => Threads.nthreads())))
    println("wrote ", p3)
    return rows, srows, gh
end

layer_report()
