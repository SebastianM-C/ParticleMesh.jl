# The laser antenna in vacuum (WP-026, the measurements behind T-391 to T-395): a Gaussian beam
# launched through the module's 2D step, read on probe columns from the antenna plane to two
# Rayleigh lengths past the focus.
#
# 1. The beam (antenna_beam.png): the 1/e radius and the Gouy phase of the ω₀ component on 28
#    probe columns against the Yee angular-spectrum propagation of the incident field (exact for
#    the scheme) and the paraxial Gaussian beam with the scheme's keff and with the continuum's
#    k = 1, for the waists 6 and 12 c/ω₀ (the paraxial error falls as 1/w0²).
# 2. Resolution (antenna_resolution.png): at 12, 16, 24 and 32 cells per wavelength along x (half
#    as many along y), the energy past the plane against the discrete flux of the incident field
#    and against laser_energy, the energy behind the plane, and the centroid velocity against the
#    spectral group velocity and the on-axis value.
#   julia +1.13 --project=scripts scripts/em2d/antenna.jl
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic
using ForwardDiff: ForwardDiff
value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)
include(joinpath(EM_TEST, "antenna_cases.jl"))

const PROBES = Tuple(10:2:64)
const WAISTS = (6.0, 12.0)
const RESOLUTIONS = (12, 16, 24, 32)

function beam_rows(w0)
    Ly = w0 == 6 ? 120 : 200
    prob = antenna_problem(Float64; waist = w0, length = (180, Ly))
    r = antenna_run(prob; probes = PROBES)
    a = run_antenna(r)
    ang = angular_error(r)
    bp = beam_profile(r)
    # the angular-spectrum prediction's own radius and phase, from the incident transform
    h = a.dx
    S = dft(incident_dtft(r).Ey)
    ky = ky_modes(length(S), h[2])
    y = ((1:Int(a.n2)) .- 0.5) .* h[2] .- a.laser.center
    yk = yee_axis(h[1], prob.dt)
    j0 = argmin(abs.(y))
    rows = []
    for (q, b) in enumerate(bp)
        pred = idft(S .* [cis(yee_kx(1.0, k, h[1], h[2], prob.dt) * (b.x - a.xa)) for k in ky])
        I = abs2.(pred)
        ȳ = sum(y .* I) / sum(I)
        wp = 2 * sqrt(sum((y .- ȳ) .^ 2 .* I) / sum(I))
        par = paraxial_beam(r, b.x)
        # the continuum's paraxial beam: q(x) = q_a + (x − x_a), focus and Rayleigh length as prescribed
        zR = w0^2 / 2
        qa = complex(a.xa - a.laser.focus, -zR)
        qc = qa + (b.x - a.xa)
        push!(rows, Dict("x" => b.x, "w" => b.w, "phase" => b.phase, "w_angular" => wp,
                         "phase_angular" => angle(pred[j0] * cis(-yk.k * (b.x - a.xa))), "w_paraxial" => par.w,
                         "gouy_paraxial" => par.gouy, "w_continuum" => w0 * abs(qc) / zR, "gouy_continuum" => angle(sqrt(qa / qc)),
                         "angular_error" => ang[q].err))
    end
    φ0, φ0a = rows[1]["phase"], rows[1]["phase_angular"]
    for r in rows
        r["gouy"] = rem2pi(r["phase"] - φ0, RoundNearest)
        r["gouy_angular"] = rem2pi(r["phase_angular"] - φ0a, RoundNearest)
    end
    return Dict("waist" => w0, "focus" => Float64(a.laser.focus), "antenna_x" => Float64(a.xa), "keff" => yee_axis(h[1], prob.dt).keff,
                "cells" => collect(ncells(r.st.grid)), "dt" => prob.dt, "rows" => rows)
end

function resolution_row(ncx)
    prob = antenna_problem(Float64; ncx, ncy = ncx ÷ 2)
    dt = prob.dt
    r = antenna_run(prob; snaps = round(Int, 100 / dt):round(Int, 4 / dt):round(Int, 155 / dt))
    a = run_antenna(r)
    grid = r.st.grid
    n = ncells(grid)
    W = conserved_energy(r.st, dt; cols = a.ia:n[1])
    Φ = incident_flux(r)
    yk = yee_axis(a.dx[1], dt)
    back = plain_energy(map(Array, r.st.fields), grid; cols = 1:(a.ia - 2))
    ts = [s.t for s in r.shots]
    X = [energy_centroid(s.f, grid) for s in r.shots]
    M = [ones(length(ts)) ts]
    c = M \ X
    vs = spectral_group_velocity(r.shots[end].f, grid, dt)
    return Dict("ncx" => ncx, "dx" => a.dx[1], "dy" => a.dx[2], "dt" => dt, "cells" => collect(n), "W_over_flux" => W / Φ - 1,
                "flux_over_laser_energy" => Φ / laser_energy(a.laser) - 1,
                "flux_over_laser_energy_staggered" => Φ / (laser_energy(a.laser) * cos(dt / 2) * cos(yk.k * a.dx[1] / 2)) - 1,
                "back_fraction" => back / W, "v_centroid" => c[2], "v_spectral" => vs, "v_axis" => yk.vg,
                "centroid_fit_residual" => maximum(abs.(M * c - X)))
end

function antenna_report()
    dir = run_dir("antenna", @__FILE__; work_package = "WP-026", prefix = "em2d",
                  extra_inputs = [joinpath(EM_TEST, "antenna_cases.jl")],
                  params = Dict{String,Any}("probes" => collect(PROBES), "waists" => collect(WAISTS),
                                            "resolutions" => collect(RESOLUTIONS), "type" => "Float64", "duration" => 8,
                                            "antenna_x" => 10, "focus" => 28, "courant" => 0.95))
    beams = [beam_rows(w0) for w0 in WAISTS]
    fig = Figure(size = (980, 640))
    for (col, b) in enumerate(beams)
        rows = b["rows"]
        x = [r["x"] for r in rows]
        ax1 = Axis(fig[1, col]; xlabel = "x (c/ω₀)", ylabel = "1/e field radius (c/ω₀)",
                   title = "waist $(b["waist"]): radius of the ω₀ component")
        lines!(ax1, x, [r["w_paraxial"] for r in rows]; label = "paraxial, Yee keff")
        lines!(ax1, x, [r["w_continuum"] for r in rows]; linestyle = :dash, label = "paraxial, k = 1")
        lines!(ax1, x, [r["w_angular"] for r in rows]; color = :black, linestyle = :dot, label = "Yee angular spectrum")
        scatter!(ax1, x, [r["w"] for r in rows]; color = :red, markersize = 7, label = "measured")
        axislegend(ax1; position = :ct)
        ax2 = Axis(fig[2, col]; xlabel = "x (c/ω₀)", ylabel = "Gouy phase since the plane (rad)")
        lines!(ax2, x, [r["gouy_paraxial"] for r in rows]; label = "paraxial, Yee keff")
        lines!(ax2, x, [r["gouy_continuum"] for r in rows]; linestyle = :dash, label = "paraxial, k = 1")
        lines!(ax2, x, [r["gouy_angular"] for r in rows]; color = :black, linestyle = :dot, label = "Yee angular spectrum")
        scatter!(ax2, x, [r["gouy"] for r in rows]; color = :red, markersize = 7, label = "measured")
        axislegend(ax2; position = :rt)
        println("waist ", b["waist"], ": max |w/w_par − 1| = ", maximum(r -> abs(r["w"] / r["w_paraxial"] - 1), rows),
                ", max |w/w_angular − 1| = ", maximum(r -> abs(r["w"] / r["w_angular"] - 1), rows),
                ", max |Gouy − paraxial| = ", maximum(r -> abs(r["gouy"] - r["gouy_paraxial"]), rows),
                ", max angular error = ", maximum(r -> r["angular_error"], rows))
    end
    p1 = save_plot(dir, "antenna_beam", fig, Dict("beams" => beams, "cpu" => cpu_model(), "threads" => Threads.nthreads()))
    println("wrote ", p1)
    res = [resolution_row(ncx) for ncx in RESOLUTIONS]
    for r in res
        println(r)
    end
    fig2 = Figure(size = (980, 560))
    dx = [r["dx"] for r in res]
    ax = Axis(fig2[1, 1]; xscale = log10, yscale = log10, xlabel = "Δx (c/ω₀), Δy = 2Δx",
              ylabel = "relative difference", title = "energy (waist 6)")
    scatterlines!(ax, dx, [abs(r["flux_over_laser_energy"]) for r in res]; label = "|Φ/laser_energy − 1|")
    scatterlines!(ax, dx, [abs(r["flux_over_laser_energy_staggered"]) for r in res]; label = "|Φ/(laser_energy·cos·cos) − 1|")
    scatterlines!(ax, dx, [abs(r["W_over_flux"]) for r in res]; label = "|W/Φ − 1|")
    scatterlines!(ax, dx, [r["back_fraction"] for r in res]; label = "energy behind / W")
    Legend(fig2[2, 1], ax; framevisible = false)
    ax2 = Axis(fig2[1, 2]; xscale = log10, yscale = log10, xlabel = "Δx (c/ω₀), Δy = 2Δx",
               ylabel = "1 − v (c)", title = "group velocity (waist 6)")
    scatterlines!(ax2, dx, [1 - r["v_centroid"] for r in res]; label = "energy centroid")
    scatterlines!(ax2, dx, [1 - r["v_spectral"] for r in res]; label = "Yee, energy-weighted over the modes")
    scatterlines!(ax2, dx, [1 - r["v_axis"] for r in res]; label = "Yee, on the axis (plane wave)")
    Legend(fig2[2, 2], ax2; framevisible = false)
    p2 = save_plot(dir, "antenna_resolution", fig2, Dict("rows" => res, "cpu" => cpu_model(), "threads" => Threads.nthreads()))
    println("wrote ", p2)
    return beams, res
end

antenna_report()
