# Convergence of the Yee solver to Maxwell's equations (WP-022, the measurement behind T-321): the
# phase-velocity error 1 − ω/|k| of a discrete eigenmode advanced through `solve_fields!` for one
# period, at Courant number 0.95 of `courant_limit`, against the cell size, at N = 1 and at N = 2
# along the x axis, the y axis and the diagonal of a grid with Δy = 4Δx/3 (box 2π × 2π, so
# |k| = |m|). Float64, polarization 2 (Ez, Bx, By). Each measured ω is also compared with the
# exact discrete relation sin(ωΔt/2)/Δt = √(Σ_d sin²(k_dΔ_d/2)/Δ_d²).
#   julia +1.13 --project=scripts scripts/em2d/yee_convergence.jl
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using ParticleMeshElectromagnetic

const SCALES = (1 // 2, 1, 2, 4, 8, 16)            # 8 to 256 cells per wavelength along x
const CFL = 95 // 100
const CASES = (("N = 1", 1, (1,)), ("N = 2, x axis", 2, (1, 0)), ("N = 2, y axis", 2, (0, 1)), ("N = 2, diagonal", 2, (1, 1)))

grid_of(N, s) = N == 1 ? PeriodicGrid(Int(16s), 2π / Int(16s), 2) :
                PeriodicGrid((Int(16s), Int(12s)), (2π / Int(16s), 2π / Int(12s)), 2)

function discrete_omega(grid, m, dt)
    κ² = sum(d -> (2 * sin(big(π) * m[d] / grid.n[d]) / big(grid.dx[d]))^2, eachindex(m))
    return Float64(2 / big(dt) * asin(big(dt) * sqrt(κ²) / 2))
end

"Phase-velocity error of one run: Ez = cos(k·x), (Bx, By) of the discrete eigenmode at t = 0."
function measure(N, s, m)
    grid = grid_of(N, s)
    dt = Float64(CFL) * courant_limit(Yee(), grid)
    ω_d = discrete_omega(grid, m, dt)
    κ = ntuple(d -> 2 * sin(π * m[d] / grid.n[d]) / grid.dx[d], N)
    nκ = sqrt(sum(abs2, κ))
    c½ = cos(ω_d * dt / 2)
    f = em_fields(CPU(), Float64, grid)
    layout = field_layout(Val(N))
    shifts(st) = st isa Tuple ? map(x -> x isa Center ? 0.5 : 0.0, st) : (st isa Center ? 0.5 : 0.0,)
    θ(key, J) = 2π * sum((J[d] - 1 + shifts(getfield(layout, key))[d]) * m[d] / grid.n[d] for d in 1:N)
    amp = (Ez = 1.0, Bx = N >= 2 ? c½ * κ[2] / nκ : 0.0, By = -c½ * κ[1] / nκ)
    for key in (:Ez, :Bx, :By)
        v = interior_view(getfield(f, key), grid)
        for J in CartesianIndices(v)
            v[J] = getfield(amp, key) * cos(θ(key, J))
        end
        fill_guards!(getfield(f, key), grid)
    end
    phases = [cis(-θ(:Ez, J)) for J in CartesianIndices(grid.n)]
    proj() = sum(interior_view(f.Ez, grid) .* phases)
    nsteps = round(Int, 2π / dt)
    c0, φ = proj(), 0.0
    for _ in 1:nsteps
        solve_fields!(Yee(), f, grid, dt)
        c1 = proj()
        φ += angle(c0 / c1)
        c0 = c1
    end
    ω = φ / (nsteps * dt)
    kabs = sqrt(sum(abs2, m))
    return Dict{String,Any}("n" => collect(grid.n), "dx" => grid.dx[1], "dt" => dt, "steps" => nsteps, "omega" => ω,
                            "omega_discrete" => ω_d, "error" => 1 - ω / kabs, "error_discrete" => 1 - ω_d / kabs,
                            "omega_vs_discrete" => ω / ω_d - 1)
end

function yee_convergence()
    dir = run_dir("yee-convergence", @__FILE__; work_package = "WP-022", prefix = "em2d",
                  params = Dict{String,Any}("scales" => [Float64(s) for s in SCALES], "cfl" => Float64(CFL),
                                            "cases" => [c[1] for c in CASES], "type" => "Float64"))
    results = Dict{String,Any}()
    for (label, N, m) in CASES
        rows = [measure(N, s, m) for s in SCALES]
        errs = [r["error"] for r in rows]
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        results[label] = Dict("mode" => collect(m), "rows" => rows, "orders" => orders)
        println(label, ": errors ", round.(errs, sigdigits = 4), " orders ", round.(orders, digits = 4),
                " |ω/ω_discrete − 1| ≤ ", maximum(r -> abs(r["omega_vs_discrete"]), rows))
    end
    fig = Figure(size = (640, 460))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "Δx (c/ω₀), Δy = 4Δx/3 at N = 2",
              ylabel = "1 − ω/(c|k|)", title = "Yee phase-velocity error at 0.95 of the Courant limit, |k| = |m|")
    for (label, _, _) in CASES
        r = results[label]["rows"]
        scatterlines!(ax, [x["dx"] for x in r], [x["error"] for x in r]; label)
    end
    r = results["N = 2, x axis"]["rows"]
    dx = [x["dx"] for x in r]
    lines!(ax, dx, r[end]["error"] .* (dx ./ dx[end]) .^ 2; color = :gray, linestyle = :dash, label = "∝ Δx²")
    axislegend(ax; position = :lt)
    path = save_plot(dir, "yee_convergence", fig, Dict("results" => results, "cpu" => cpu_model(), "threads" => Threads.nthreads()))
    println("wrote ", path)
    return results
end

yee_convergence()
