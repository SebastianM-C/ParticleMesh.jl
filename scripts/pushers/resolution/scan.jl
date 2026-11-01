# WP-016: the figure-8 scan of the five pushers over a₀ and ω₀Δt, the resolution thresholds, and
# Gordon and Hafizi's constant-phase-step diagnostic (arXiv:2006.10146, Sec. 6: a₀ = 10, 63
# steps of ω₀Δs = 0.1).
#
#     julia +1.13 --project=scripts scripts/pushers/resolution/scan.jl
#
# writes scripts/runs/pushers-resolution-<stamp>/ with manifest.toml (F3), one PNG and JSON
# sidecar per error (`resolution_<error>.png`), the diagnostic (`gordon_hafizi_a0_10.png`), and
# summary.json with every error, the thresholds and their application to representative LWFA
# time steps. In the diagnostic the lab-time pushers take lab steps from the inverse of Eq. 22, as
# Gordon and Hafizi's do, and `QuadraticUnitary` runs both that way and, as theirs does, in
# proper time (`gordon_hafizi_proper`).
#
# The orbit and the errors are those of T-243 (lib/ParticleMeshElectromagnetic/test/
# pushers_resolution.jl, whose header states them): an electron (q/m = −1) at rest at η = 0 in
# A_x = a₀ sin(t − z), one period T_p = 2π(1 + a₀²/4), Float64, stepped with the module's
# `push_particle`. The relative errors:
#
# - gamma: max over the steps of |γ − u_z − 1|;
# - canonical: max over the steps of |u_x − a₀ sin η|/a₀ (u_x + qA_x, relative to a₀);
# - drift: |z(T_p)/T_p − v_d|/v_d, v_d = a₀²/(4 + a₀²);
# - position: |x(T_p) − (0, 0, πa₀²/2)|/(πa₀²/2).
#
# Thresholds: for each pusher, error and tolerance (10⁻³, 10⁻²), the largest scanned a₀·ω₀Δt
# such that every scanned point at or below it, over all four a₀, has the error below the
# tolerance; and per a₀ the largest scanned ω₀Δt such that it and every smaller step pass. The
# grid is √2-spaced, so a threshold is conservative by up to a factor √2.

using ParticleMesh
using ParticleMeshElectromagnetic
using CairoMakie
using JSON
using Dates

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "scripts", "crosscode", "manifest.jl"))
using .RunManifest: write_manifest, file_hash

const PUSHERS = (Boris(), Vay(), HigueraCary(), BorisExactRotation(), QuadraticUnitary())
const A0 = (1.0, 2.0, 5.0, 10.0)
const DT = Tuple(0.4 * 2.0^(-k / 2) for k in 0:10)          # 0.4 … 0.0125, √2-spaced
const ERRORS = (:gamma, :canonical, :drift, :position)
const LABELS = Dict(:gamma => "|γ − u_z − 1|", :canonical => "|u_x + qA_x|/a₀",
                    :drift => "drift velocity, relative", :position => "position after one period, relative")
const TOLS = (1e-3, 1e-2)
pname(p) = string(nameof(typeof(p)))
norm3(a) = sqrt(a[1]^2 + a[2]^2 + a[3]^2)

planewave(a0) = ((x, t) -> (-a0 * cos(t - x[3]), 0.0, 0.0), (x, t) -> (0.0, -a0 * cos(t - x[3]), 0.0))

"The four relative errors of one period at a₀ and ω₀Δt (the definitions of T-243)."
function figure8_errors(P, a0, dt)
    E, B = planewave(a0)
    Tp = 2π * (1 + a0^2 / 4)
    N = floor(Int, Tp / dt)
    x = (0.0, 0.0, 0.0)
    u = x
    eh = 0.0
    ep = 0.0
    for n in 1:N
        x, u = push_particle(P, x, u, (n - 1) * dt, dt, -1.0, E, B)
        eh = max(eh, abs(lorentz_factor(u) - u[3] - 1))
        ep = max(ep, abs(u[1] - a0 * sin(n * dt - x[3])))
    end
    x, u = push_particle(P, x, u, N * dt, Tp - N * dt, -1.0, E, B)
    zp = π * a0^2 / 2
    vd = a0^2 / (4 + a0^2)
    return (gamma = eh, canonical = ep / a0, drift = abs(x[3] / Tp - vd) / vd,
            position = norm3((x[1], x[2], x[3] - zp)) / zp, steps = N + 1)
end

"""
    exact_state(a0, t) -> (x, u)

The closed-form orbit at lab time t: η from t = η + (a₀²/4)(η − sin(2η)/2) by Newton's method
(dt/dη = γ ≥ 1, so the iteration from η = t/(1 + a₀²/4) converges), then x = a₀(1 − cos η),
z = (a₀²/4)(η − sin(2η)/2), u_x = a₀ sin η, u_z = u_x²/2.
"""
function exact_state(a0, t)
    η = t / (1 + a0^2 / 4)
    for _ in 1:100
        f = η + a0^2 / 4 * (η - sin(2η) / 2) - t
        η -= f / (1 + a0^2 / 4 * (1 - cos(2η)))
        abs(f) < 4eps(max(1.0, t)) && break
    end
    ux = a0 * sin(η)
    return (a0 * (1 - cos(η)), 0.0, a0^2 / 4 * (η - sin(2η) / 2)), (ux, 0.0, ux^2 / 2)
end

"""
    gordon_hafizi(P, a0, Δs, nsteps)

Their diagnostic: constant proper-time steps Δs, each turned into a lab step by the inverse of
their Eq. 22 at the start of the step, Δt = γΔs + (q/m)(E·u)Δs²/2, then `push_particle`. Returns
the orbit, the reference at the same lab times, and the errors.
"""
function gordon_hafizi(P, a0, Δs, nsteps)
    E, B = planewave(a0)
    x = (0.0, 0.0, 0.0)
    u = x
    t = 0.0
    xs, ts, ref = [x], [t], [x]
    eh = 0.0
    ep = 0.0
    dist = 0.0
    for _ in 1:nsteps
        γ = lorentz_factor(u)
        e = E(x, t)
        dt = γ * Δs - (e[1] * u[1] + e[2] * u[2] + e[3] * u[3]) * Δs^2 / 2   # q/m = −1
        x, u = push_particle(P, x, u, t, dt, -1.0, E, B)
        t += dt
        xe, _ = exact_state(a0, t)
        push!(xs, x); push!(ts, t); push!(ref, xe)
        eh = max(eh, abs(lorentz_factor(u) - u[3] - 1))
        ep = max(ep, abs(u[1] - a0 * sin(t - x[3])) / a0)
        dist = max(dist, norm3(x .- xe))
    end
    scale = π * a0^2 / 2
    return (; xs, ts, ref, gamma = eh, canonical = ep, max_distance = dist / scale,
            end_distance = norm3(xs[end] .- ref[end]) / scale, t_end = t)
end

"""
    gordon_hafizi_proper(a0, Δs, nsteps)

`QuadraticUnitary` as Gordon and Hafizi run it: in proper time, with no lab step. The world
line is leapfrogged as in their Eq. 20, x̄ = x + (Δs/2)u and t̄ = t + (Δs/2)γ; then (γ, u) is
advanced by `quadratic_unitary_step` with κ = (q/m)Δs/2 and the fields at (x̄, t̄); then
x′ = x̄ + (Δs/2)u′, t′ = t̄ + (Δs/2)γ′. γ is carried, not recomputed.
"""
function gordon_hafizi_proper(a0, Δs, nsteps)
    E, B = planewave(a0)
    x = (0.0, 0.0, 0.0)
    u = x
    γ = 1.0
    t = 0.0
    xs, ts, ref = [x], [t], [x]
    eh = 0.0
    ep = 0.0
    dist = 0.0
    h = Δs / 2
    κ = -h                                                           # (q/m)Δs/2 with q/m = −1
    for _ in 1:nsteps
        x̄ = x .+ h .* u
        t̄ = t + h * γ
        γ, u = ParticleMeshElectromagnetic.quadratic_unitary_step(γ, u, κ .* E(x̄, t̄), κ .* B(x̄, t̄))
        x = x̄ .+ h .* u
        t = t̄ + h * γ
        xe, _ = exact_state(a0, t)
        push!(xs, x); push!(ts, t); push!(ref, xe)
        eh = max(eh, abs(γ - u[3] - 1))
        ep = max(ep, abs(u[1] - a0 * sin(t - x[3])) / a0)
        dist = max(dist, norm3(x .- xe))
    end
    scale = π * a0^2 / 2
    return (; xs, ts, ref, gamma = eh, canonical = ep, max_distance = dist / scale,
            end_distance = norm3(xs[end] .- ref[end]) / scale, t_end = t)
end

"Largest scanned a₀·ω₀Δt with every point at or below it (all a₀) under `tol`; 0 if none."
function threshold_product(errs, tol)
    pts = sort([(a0 * dt, e) for ((a0, dt), e) in errs])
    best = 0.0
    for (p, _) in pts
        all(e < tol for (q, e) in pts if q <= p + 1e-12) && (best = p)
    end
    return best
end

"Largest scanned ω₀Δt at `a0` with it and every smaller step under `tol`; 0 if none."
function threshold_dt(errs, a0, tol)
    best = 0.0
    for dt in sort(collect(DT))
        errs[(a0, dt)] < tol || break
        best = dt
    end
    return best
end

"ω₀Δt at 0.95 of the 2D Courant limit for N_λ cells per laser wavelength along x and Δy = 4Δx."
lwfa_dt(nλ) = (dx = 2π / nλ; dy = 4dx; 0.95 / sqrt(1 / dx^2 + 1 / dy^2))

function main()
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    dir = joinpath(REPO, "scripts", "runs", "pushers-resolution-$stamp")
    mkpath(dir)
    inputs = String[@__FILE__, joinpath(REPO, "scripts", "crosscode", "manifest.jl"),
                    joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")]
    for (root, _, fs) in walkdir(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "src")), f in fs
        endswith(f, ".jl") && push!(inputs, joinpath(root, f))
    end
    hashes = Dict{String,Any}(relpath(f, REPO) => "sha256:" * file_hash(f) for f in sort(inputs))
    params = Dict{String,Any}("a0" => collect(A0), "omega0_dt" => collect(DT), "qm" => -1, "tolerances" => collect(TOLS),
                              "gordon_hafizi" => Dict("a0" => 10, "omega0_ds" => 0.1, "steps" => 63),
                              "lwfa_steps" => "0.95 of the 2D Courant limit, Δy = 4Δx, N_λ ∈ {16, 24, 32, 48, 64} cells per wavelength along x")
    write_manifest(dir; root = REPO,
                   extra = Dict{String,Any}("inputs" => hashes, "run_id" => basename(dir), "work_package" => "WP-016",
                                            "test" => "T-243", "parameters" => params, "julia" => string(VERSION),
                                            "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))

    # the scan
    res = Dict((pname(P), a0, dt) => figure8_errors(P, a0, dt) for P in PUSHERS, a0 in A0, dt in DT)
    summary = Dict{String,Any}()
    summary["errors"] = Dict(pname(P) => Dict(string(k) => Dict("a0=$a0" => [getfield(res[(pname(P), a0, dt)], k) for dt in DT] for a0 in A0)
                                              for k in ERRORS) for P in PUSHERS)
    summary["omega0_dt"] = collect(DT)
    th = Dict{String,Any}()
    for P in PUSHERS, k in ERRORS, tol in TOLS
        errs = Dict((a0, dt) => getfield(res[(pname(P), a0, dt)], k) for a0 in A0, dt in DT)
        d = get!(th, pname(P), Dict{String,Any}())
        d["$k < $tol"] = Dict("a0_dt" => threshold_product(errs, tol),
                              "dt_per_a0" => Dict("a0=$a0" => threshold_dt(errs, a0, tol) for a0 in A0))
    end
    summary["thresholds"] = th

    colors = Makie.wong_colors()
    for k in ERRORS
        fig = Figure(size = (1700, 700))
        side = Dict{String,Any}("error" => LABELS[k], "omega0_dt" => collect(DT), "a0" => collect(A0))
        for (j, P) in enumerate(PUSHERS)
            for (row, (xl, xf)) in enumerate((("a₀·ω₀Δt", (a0, dt) -> a0 * dt), ("ω₀Δt", (a0, dt) -> dt)))
                ax = Axis(fig[row, j]; xscale = log10, yscale = log10, xlabel = xl,
                          ylabel = row == 1 && j == 1 ? LABELS[k] : (row == 2 && j == 1 ? LABELS[k] : ""),
                          title = row == 1 ? pname(P) : "", limits = (nothing, (1e-16, 1)))
                hlines!(ax, collect(TOLS); color = :gray, linestyle = :dash)
                for (i, a0) in enumerate(A0)
                    e = [max(getfield(res[(pname(P), a0, dt)], k), 1e-16) for dt in DT]
                    scatterlines!(ax, [xf(a0, dt) for dt in DT], e; color = colors[i], label = "a₀ = $(Int(a0))")
                end
                row == 1 && j == length(PUSHERS) && axislegend(ax; position = :rb)
            end
            side[pname(P)] = Dict("a0=$a0" => [getfield(res[(pname(P), a0, dt)], k) for dt in DT] for a0 in A0)
        end
        Label(fig[0, :], "$(LABELS[k]) over one figure-8 period (floor 1e-16 in the plot; dashed: 10⁻³, 10⁻²)")
        save(joinpath(dir, "resolution_$k.png"), fig; px_per_unit = 2)
        open(io -> JSON.print(io, side), joinpath(dir, "resolution_$k.json"), "w")
    end

    # Gordon and Hafizi's diagnostic
    gh = Dict(pname(P) => gordon_hafizi(P, 10.0, 0.1, 63) for P in PUSHERS)
    gh["QuadraticUnitary in proper time"] = gordon_hafizi_proper(10.0, 0.1, 63)
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1]; xlabel = "z − v_d t (c/ω₀)", ylabel = "x (c/ω₀)",
              title = "a₀ = 10, 63 steps of ω₀Δs = 0.1, lab steps from the inverse of Eq. 22")
    vd = 100 / 104
    r = gh["Boris"]
    tt = range(0, r.t_end; length = 2000)
    ex = [exact_state(10.0, t)[1] for t in tt]
    lines!(ax, [p[3] - vd * t for (p, t) in zip(ex, tt)], [p[1] for p in ex]; color = :black, label = "exact")
    side = Dict{String,Any}()
    for (i, name) in enumerate((map(pname, PUSHERS)..., "QuadraticUnitary in proper time"))
        g = gh[name]
        scatterlines!(ax, [p[3] - vd * t for (p, t) in zip(g.xs, g.ts)], [p[1] for p in g.xs]; color = colors[i],
                      markersize = 5, label = name)
        side[name] = Dict("t" => g.ts, "x" => [p[1] for p in g.xs], "z" => [p[3] for p in g.xs],
                              "x_exact" => [p[1] for p in g.ref], "z_exact" => [p[3] for p in g.ref],
                              "gamma" => g.gamma, "canonical" => g.canonical, "max_distance" => g.max_distance,
                              "end_distance" => g.end_distance, "t_end" => g.t_end)
    end
    axislegend(ax; position = :rt)
    save(joinpath(dir, "gordon_hafizi_a0_10.png"), fig; px_per_unit = 2)
    open(io -> JSON.print(io, side), joinpath(dir, "gordon_hafizi_a0_10.json"), "w")
    summary["gordon_hafizi"] = Dict(k => Dict(f => v[f] for f in ("gamma", "canonical", "max_distance", "end_distance", "t_end"))
                                    for (k, v) in side)

    # representative LWFA steps: a₀·ω₀Δt against each threshold (no A5 2D LWFA case is specified yet)
    lw = Dict{String,Any}()
    for nλ in (16, 24, 32, 48, 64)
        dt = lwfa_dt(nλ)
        lw["N_lambda=$nλ"] = Dict("omega0_dt" => dt, "a0_dt" => Dict("a0=$a0" => a0 * dt for a0 in (1, 2, 3, 5, 10)))
    end
    summary["lwfa_steps"] = lw
    open(io -> JSON.print(io, summary, 2), joinpath(dir, "summary.json"), "w")

    for P in PUSHERS, k in ERRORS
        t3 = th[pname(P)]["$k < 0.001"]
        t2 = th[pname(P)]["$k < 0.01"]
        println(rpad("$(pname(P)) $k", 32), " a₀·ω₀Δt for 1e-3: ", rpad(round(t3["a0_dt"]; sigdigits = 3), 8),
                " 1e-2: ", rpad(round(t2["a0_dt"]; sigdigits = 3), 8),
                " ω₀Δt per a₀ (1e-3): ", join((round(t3["dt_per_a0"]["a0=$a0"]; sigdigits = 3) for a0 in A0), ", "))
    end
    for (name, g) in sort(collect(gh); by = first)
        println(rpad("GH a₀ = 10 $name", 44), " γ − u_z: ", round(g.gamma; sigdigits = 3), "  canonical: ",
                round(g.canonical; sigdigits = 3), "  max distance: ", round(g.max_distance; sigdigits = 3),
                "  end distance: ", round(g.end_distance; sigdigits = 3))
    end
    println("wrote ", dir)
    return dir
end

main()
