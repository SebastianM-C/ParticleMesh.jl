# The pushers on the electrons that field ionization releases in the pulse of the WP-065 case
# (trapped2d_case.jl), against the exact orbit: a plane wave of the case's envelope, E_y = B_z =
# a₀·exp(−φ²/τ²)·cos φ with φ = t − x and τ = 6 (the case's duration), and an electron born at rest
# at the place of its atom (x = 0) at a step time tⁿ, as the module's ionization does it (the
# pusher's half step back from rest in Eⁿ, Bⁿ, then the leapfrog with u at half steps), with the
# fields taken exactly at the particle: what is measured is the pusher's error alone, not the
# grid's. The exact orbit: with A(φ) = −∫E dφ, u_y = A(φ) − A(φ_b), γ − u_x = 1, u_x = u_y²/2 and
# x(φ) = ∫ u_x dφ, t = φ + x.
#
# Each birth is weighted by the charge the case's two nitrogen K-shell levels (ADK, as the case)
# release in that step at that a₀, so the read-outs are averages over the born charge. For every
# pusher, a₀ and time step: the weighted rms error of the residual u_y after the pulse, of the
# residual energy γ − 1, and of the position at the end, next to the physical rms of the residual
# u_y (the spread the trapping sees). The time steps are those of the resolution study's grids
# (gridppc.jl: 0.95 of the Courant limit with Δx = λ₀/ppl, Δy = λ₀/(8·ny)) and a √2 ladder.
#
#   julia +1.13 --project=scripts scripts/pushers/born_orbits.jl [out dir]
#
# Single runs on the CPU, not independently reviewed.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "..", "ionization", "atomic.jl"))
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: ionization_rate_sq

const TAU = 6.0
const PUSHERS = [("boris", Boris()), ("hc", HigueraCary()), ("vay", Vay()), ("exact", BorisExactRotation()), ("qu", QuadraticUnitary())]
const GRIDS = [(16, 1), (24, 1), (32, 2), (48, 3), (64, 4), (96, 6), (128, 8)]
grid_dt(ppl, ny) = 0.95 / sqrt((ppl / 2π)^2 + (4ny / π)^2)

field(a0, φ) = a0 * exp(-(φ / TAU)^2) * cos(φ)

"The exact orbit's tables on a fine φ grid: A(φ), and the cumulative ∫ (A − A_b)²/2 pieces as moments of A."
function exact_tables(a0; dφ = 2.0^-10, span = 8TAU)
    φ = collect(-span:dφ:span)
    E = field.(a0, φ)
    A = zeros(length(φ))
    for i in 2:length(φ)
        A[i] = A[i - 1] - (E[i - 1] + E[i]) * dφ / 2          # A = −∫E dφ (trapezoid)
    end
    cum(v) = (c = zeros(length(v)); for i in 2:length(v); c[i] = c[i - 1] + (v[i - 1] + v[i]) * dφ / 2; end; c)
    return (; φ, A, I0 = cum(ones(length(φ))), I1 = cum(A), I2 = cum(A .^ 2), dφ)
end

lin(tab, v, φ) = (s = (φ - tab.φ[1]) / tab.dφ; i = clamp(floor(Int, s) + 1, 1, length(v) - 1); f = s - (i - 1); v[i] * (1 - f) + v[i + 1] * f)

"The exact state at lab time t of the electron born at rest at x = 0 at t = φ_b: (x, u_x, u_y)."
function exact_state(tab, φb, t)
    Ab = lin(tab, tab.A, φb)
    xof(φ) = ((lin(tab, tab.I2, φ) - lin(tab, tab.I2, φb)) - 2Ab * (lin(tab, tab.I1, φ) - lin(tab, tab.I1, φb)) +
              Ab^2 * (lin(tab, tab.I0, φ) - lin(tab, tab.I0, φb))) / 2
    lo, hi = φb, t                                             # t = φ + x(φ), x ≥ 0, so φ ∈ [φ_b, t]
    for _ in 1:80
        m = (lo + hi) / 2
        m + xof(m) < t ? (lo = m) : (hi = m)
    end
    φ = (lo + hi) / 2
    uy = lin(tab, tab.A, φ) - Ab
    return (; x = xof(φ), ux = uy^2 / 2, uy)
end

"The released charge per step at the step times `ts` for an atom at x = 0 (the two K-shell levels, populations from the start of each step, as the module)."
function release(levels, a0, ts, dt)
    f = [1.0, 0.0]
    out = zeros(length(ts))
    for (n, t) in enumerate(ts)
        E2 = field(a0, t)^2
        d = [f[j] * -expm1(-ionization_rate_sq(levels[j], E2) * dt) for j in 1:2]
        out[n] = d[1] + d[2]
        f = [f[1] - d[1], f[2] + d[1] - d[2]]
    end
    return out
end

"The leapfrog of the module for one born electron: births at step `nb` (t = nb·dt), run to the step `nend`; returns x at nend and u at nend + 1/2."
function leapfrog(pusher, a0, dt, nb, nend)
    T = Float64
    E(x, t) = (zero(T), field(a0, t - x), zero(T))
    B(x, t) = (zero(T), zero(T), field(a0, t - x))
    x = 0.0
    u = push_momentum(pusher, (0.0, 0.0, 0.0), E(x, nb * dt), B(x, nb * dt), dt / 4)   # rest at tⁿ, half a step back (q = −1)
    for n in nb:(nend - 1)
        u = push_momentum(pusher, u, E(x, n * dt), B(x, n * dt), -dt / 2)
        x += dt * u[1] / lorentz_factor(u)
    end
    return x, u
end

"The leapfrog's states (t, x at integer steps, u_y at the half steps after them) of one born electron, every step from `nb` to `nend`."
function trace(pusher, a0, dt, nb, nend)
    E(x, t) = (0.0, field(a0, t - x), 0.0)
    B(x, t) = (0.0, 0.0, field(a0, t - x))
    x = 0.0
    u = push_momentum(pusher, (0.0, 0.0, 0.0), E(x, nb * dt), B(x, nb * dt), dt / 4)
    ts, xs, uys, uxs = Float64[], Float64[], Float64[], Float64[]
    for n in nb:(nend - 1)
        u = push_momentum(pusher, u, E(x, n * dt), B(x, n * dt), -dt / 2)
        push!(ts, n * dt); push!(xs, x); push!(uys, u[2]); push!(uxs, u[1])
        x += dt * u[1] / lorentz_factor(u)
    end
    return Dict("t" => ts, "x" => xs, "uy" => uys, "ux" => uxs)
end

function study(a0, dt, levels, tab)
    nlo, nhi = floor(Int, -3TAU / dt), ceil(Int, 3TAU / dt)
    ns = collect(nlo:nhi)
    w = release(levels, a0, ns .* dt, dt)
    keep = findall(>(1e-6 * maximum(w)), w)
    ns, w = ns[keep], w[keep] ./ sum(w[keep])
    nend = ceil(Int, 8TAU / dt)                                # the pulse has passed every electron by then (they drift at < c)
    tend = nend * dt
    ex = [exact_state(tab, n * dt, tend) for n in ns]
    uy_ex = [e.uy for e in ex]
    g_ex = [sqrt(1 + e.ux^2 + e.uy^2) - 1 for e in ex]
    phys = (; uy_rms = sqrt(sum(w .* uy_ex .^ 2)), g_mean = sum(w .* g_ex), x_mean = sum(w .* [e.x for e in ex]), births = length(ns))
    res = Dict{String,Any}()
    for (name, P) in PUSHERS
        duy, dg, dx, dux = Float64[], Float64[], Float64[], Float64[]
        for (k, n) in enumerate(ns)
            x, u = leapfrog(P, a0, dt, n, nend)
            push!(duy, u[2] - uy_ex[k])
            push!(dux, u[1] - ex[k].ux)
            push!(dg, sqrt(1 + u[1]^2 + u[2]^2 + u[3]^2) - 1 - g_ex[k])
            push!(dx, x - ex[k].x)
        end
        wr(v) = sqrt(sum(w .* v .^ 2))
        res[name] = Dict("uy_rms_err" => wr(duy), "uy_mean_err" => sum(w .* duy), "ux_rms_err" => wr(dux), "g_rms_err" => wr(dg),
                         "g_mean_err" => sum(w .* dg), "x_rms_err" => wr(dx), "x_mean_err" => sum(w .* dx), "uy_max_err" => maximum(abs, duy))
    end
    return Dict{String,Any}("a0" => a0, "dt" => dt, "a0dt" => a0 * dt, "physical" => Dict(pairs(phys)), "pushers" => res)
end

function main(dir)
    levels = (adk_level(Float64, :N, 6; wavelength = 0.8e-6, cutoff = 24.0), adk_level(Float64, :N, 7; wavelength = 0.8e-6, cutoff = 24.0))
    dts = vcat([grid_dt(g...) for g in GRIDS], [0.4 / sqrt(2)^k for k in 0:10])
    out = Dict{String,Any}("grids" => [Dict("ppl" => g[1], "ny" => g[2], "dt" => grid_dt(g...)) for g in GRIDS], "tau" => TAU, "runs" => Any[])
    for a0 in (1.9, 2.2, 2.6)
        tab = exact_tables(a0)
        # the physical picture at this a₀: release and residual u_y per birth phase on a fine step
        dtf = 2.0^-6
        ts = collect(-3TAU:dtf:3TAU)
        r = release(levels, a0, ts, dtf)
        Ainf = tab.A[end]
        out["phase_$a0"] = Dict("t" => ts[1:4:end], "release" => r[1:4:end] ./ (dtf * 4), "E" => field.(a0, ts[1:4:end]),
                                "residual_uy" => [Ainf - lin(tab, tab.A, t) for t in ts[1:4:end]], "released_total" => sum(r))
        for dt in dts
            s = study(a0, dt, levels, tab)
            push!(out["runs"], s)
            println("a0 = $a0, dt = $(round(dt; digits = 4)): ", join(["$k $(round(v["uy_rms_err"]; sigdigits = 2))" for (k, v) in s["pushers"]], ", "),
                    "  (physical uy rms $(round(s["physical"][:uy_rms]; sigdigits = 3)))")
            flush(stdout)
        end
    end
    # orbits to animate: a₀ = 1.9, births at three step times of λ₀/16's step near the central field peaks, every pusher, and the exact orbit
    a0, dt = 1.9, grid_dt(16, 1)
    tab = exact_tables(a0)
    out["orbits"] = map((-3, 0, 2)) do k
        nb = round(Int, (k * π + 0.15) / dt)
        nend = ceil(Int, 4TAU / dt)
        te = collect(range(nb * dt, nend * dt; length = 400))
        ex = [exact_state(tab, nb * dt, t) for t in te]
        prs = Dict{String,Any}()
        for (name, P) in PUSHERS
            tr = trace(P, a0, dt, nb, nend)
            # errors against the exact orbit at the leapfrog's own times: x at tⁿ, u at tⁿ + Δt/2
            tr["dx"] = [tr["x"][i] - exact_state(tab, nb * dt, tr["t"][i]).x for i in eachindex(tr["t"])]
            tr["duy"] = [tr["uy"][i] - exact_state(tab, nb * dt, tr["t"][i] + dt / 2).uy for i in eachindex(tr["t"])]
            prs[name] = tr
        end
        Dict{String,Any}("birth" => nb * dt, "dt" => dt, "exact" => Dict("t" => te, "x" => [e.x for e in ex], "uy" => [e.uy for e in ex], "ux" => [e.ux for e in ex]),
                         "pushers" => prs)
    end
    open(io -> JSON.print(io, out), joinpath(dir, "born_orbits.json"), "w")
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    dir = length(ARGS) >= 1 ? mkpath(abspath(ARGS[1])) :
          run_dir("born-orbits", @__FILE__; prefix = "pushers", work_package = "none (study for a report)",
                  extra_inputs = [joinpath(@__DIR__, "..", "ionization", "atomic.jl")], params = Dict{String,Any}("tau" => TAU))
    main(dir)
    println("wrote ", dir)
end
