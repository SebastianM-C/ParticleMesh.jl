# What the plasma does at and away from the optima of the 2D optimizations (WP-045, WP-051, WP-052):
# field and density snapshots of the wake case over the pulse duration and the focus, the witness
# bunch's gain per lattice point over the duration and the density, the sensitivity of the witness
# gain to parameters the optimizations hold fixed, and the wake objective with its gradient on
# another backend. Data for figures and pages; it asserts nothing about the optimizations.
#
#   julia +1.13 --project=scripts -t 8 scripts/optim/snapshots.jl wake <ppl> <dy_cells> <tag> τ1,τ2,… [focus1,focus2,…]
#   julia +1.13 --project=scripts -t 8 scripts/optim/snapshots.jl witness <tag>
#   julia +1.13 --project=scripts -t 8 scripts/optim/snapshots.jl sensitivity <tag>
#   julia +1.13 --project=scripts -t 8 scripts/optim/snapshots.jl backend <cpu|cuda> <tag>
#
# Each writes scripts/runs/optim-snapshots-<tag>/: manifest.toml and <mode>.json. No figure is drawn
# here (CairoMakie is not loaded, so a rented machine need not precompile it).

using ParticleMesh, KernelAbstractions
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_shifts, window_charge!, window_back_band, witness_frame
using ForwardDiff: ForwardDiff
using JSON, Dates, Base64

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "scripts", "crosscode", "manifest.jl"))
using .RunManifest: write_manifest
include(joinpath(@__DIR__, "wake2d_case.jl"))
include(joinpath(@__DIR__, "slab.jl"))
include(joinpath(@__DIR__, "witness2d_case.jl"))

const INPUTS = [@__FILE__, joinpath(@__DIR__, "wake2d_case.jl"), joinpath(@__DIR__, "slab.jl"), joinpath(@__DIR__, "witness2d_case.jl"),
                joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")]

function out_dir(tag, mode, params)
    dir = joinpath(REPO, "scripts", "runs", "optim-snapshots-$tag")
    mkpath(dir)
    write_manifest(dir; root = REPO, inputs = INPUTS,
                   extra = Dict{String,Any}("run_id" => basename(dir), "mode" => mode, "parameters" => params, "julia" => string(VERSION),
                                            "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
    return dir
end

sig(x; n = 4) = x isa AbstractArray ? map(v -> sig(v; n), x) : (isfinite(x) ? round(Float64(x); sigdigits = n) : nothing)

"A matrix as bytes: 0 … 254 linear between lo and hi, 255 for a missing value; base64, column-major."
function bytes(M, lo, hi)
    b = map(M) do v
        isfinite(v) ? UInt8(clamp(round(Int, 254 * (v - lo) / (hi - lo)), 0, 254)) : 0xff
    end
    return Dict{String,Any}("n" => collect(size(M)), "lo" => lo, "hi" => hi, "b64" => base64encode(vec(b)))
end

# A running max of |v| over ±h points: the envelope of a carrier with h about half a wavelength.
running_envelope(v, h) = [maximum(abs, @view v[max(1, i - h):min(end, i + h)]) for i in eachindex(v)]

"The axis lineouts of a state: E_x, the envelope of the laser's E_y, and (with `ρbuf`) the electron density over n₀."
function lineouts(st, k, n0; ρbuf = nothing, stride = 1, ppl = 16)
    g, n = guards(st.grid), ncells(st.grid)
    I = (g + 1):(g + n[1])
    row = g + 1 + n[2] ÷ 2
    Ex = Float64.(value_of.(Array(st.fields.Ex)[I, row]))
    Ey = Float64.(value_of.(Array(st.fields.Ey)[I, (row - 1):row]))
    env = running_envelope((Ey[:, 1] .+ Ey[:, 2]) ./ 2, ppl ÷ 2)
    o = Dict{String,Any}("shifts" => window_shifts(st, k), "Ex" => sig(Ex[1:stride:end] ./ sqrt(n0)), "a" => sig(env[1:stride:end]; n = 3))
    if ρbuf !== nothing
        window_charge!(ρbuf, st)
        o["ne"] = sig(-Float64.(Array(ρbuf)[I, row])[1:stride:end] ./ n0; n = 3)
    end
    return o
end

"The 2D maps of a state, on every `sx`-th column and `sy`-th row: electron density over n₀, E_x over E₀, E_y."
function maps(st, ρbuf, n0, sx, sy)
    g, n = guards(st.grid), ncells(st.grid)
    I, J = (g + 1):sx:(g + n[1]), (g + 1):sy:(g + n[2])
    window_charge!(ρbuf, st)
    ne = -Float64.(Array(ρbuf)[I, J]) ./ n0
    Ex = Float64.(Array(st.fields.Ex)[I, J]) ./ sqrt(n0)
    Ey = Float64.(Array(st.fields.Ey)[I, J])
    return Dict{String,Any}("ne" => bytes(ne, 0.0, 3.0), "Ex" => bytes(Ex, -0.6, 0.6), "Ey" => bytes(Ey, -1.6, 1.6))
end

# --- the wake case (WP-045, WP-051) ---------------------------------------------------------------

const BOX_TIME = 185.30578712462201      # WP-045's read-out: 512 steps of the base grid

"""
One point of the wake case: the two objectives from `wake_observe` (the slab's J and WP-045's
window J at its read-out time), then the same run again with snapshots: axis lineouts at `times`,
2D maps at `map_times`, and the on-axis E_x in the lab frame (columns x ∈ [60, 165] at the base Δx,
one row every `2π/16·1.84` in t from 100 on; the columns the window does not cover are missing).
"""
function wake_point(τ, focus; ppl, dy_cells, times = 120.0:15.0:255.0, map_times = (BOX_TIME, 240.0), with_maps = true, lab = true)
    n0 = CASE.density
    t0 = time()
    obs = wake_observe(τ, focus; box_times = (BOX_TIME,), ppl, dy_cells)
    prob = wake_problem(τ, focus; nsteps = obs.nsteps, ppl, dy_cells)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    initialize!(st, prob)
    p = params(prob)
    ρbuf = field(CPU(), Float64, st.grid)
    dt, h = obs.dt, obs.dx
    stride = ppl ÷ 16
    g, n = guards(st.grid), ncells(st.grid)
    row = g + 1 + n[2] ÷ 2
    band = window_back_band(st)
    at(t) = round(Int, t / dt)
    line_steps = Dict(at(t) => t for t in times)
    map_steps = with_maps ? Dict(at(t) => t for t in map_times) : Dict{Int,Float64}()
    lab_dt = 2π / 16 * 1.84
    lab_steps = lab ? Dict(at(t) => t for t in 100.0:lab_dt:(obs.nsteps * dt)) : Dict{Int,Float64}()
    labcols = round(Int, 60 / (h * stride)):round(Int, 165 / (h * stride))        # base columns c: x = (c − ½)·Δx_base
    labrows = Vector{Vector{Float64}}()
    lines, mapsout = Any[], Any[]
    for k in 1:prob.nsteps
        step!(st, p, k)
        if haskey(lab_steps, k)
            ex = Array(st.fields.Ex)[(g + 1):(g + n[1]), row]
            s = window_shifts(st, k)
            push!(labrows, [(i = c * stride - s; band < i <= n[1] ? ex[i] / sqrt(n0) : NaN) for c in labcols])
        end
        if haskey(line_steps, k)
            o = lineouts(st, k, n0; ρbuf, stride, ppl)
            o["t"] = k * dt
            push!(lines, o)
        end
        if haskey(map_steps, k)
            o = maps(st, ρbuf, n0, stride, max(1, 4 ÷ dy_cells))
            o["t"], o["shifts"] = k * dt, window_shifts(st, k)
            push!(mapsout, o)
        end
    end
    out = Dict{String,Any}("tau" => τ, "focus" => focus, "J_slab" => obs.slab.main, "J_window" => obs.box[1], "nsteps" => obs.nsteps,
                           "dt" => dt, "dx" => h, "stride" => stride, "band" => band, "lines" => lines, "maps" => mapsout,
                           "seconds" => time() - t0)
    lab && (out["lab"] = merge(bytes(reduce(hcat, labrows), -0.6, 0.6),
                               Dict("x0" => (first(labcols) - 0.5) * h * stride, "dx" => h * stride, "t0" => at(100.0) * dt, "dt" => lab_dt)))
    return out
end

function wake_mode(ppl, dy_cells, tag, taus, focuses)
    dir = out_dir(tag, "wake", Dict{String,Any}("ppl" => ppl, "dy_cells" => dy_cells, "taus" => taus, "focuses" => focuses))
    focus0 = (ppl, dy_cells) == (32, 2) ? 118.03939032920552 : 115.82783875101673      # WP-051's optima at the two grids
    tau0 = (ppl, dy_cells) == (32, 2) ? 8.219217151454469 : 7.894091418703013
    out = Dict{String,Any}("ppl" => ppl, "dy_cells" => dy_cells, "focus0" => focus0, "tau0" => tau0, "case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(CASE)),
                           "tau_scan" => Any[], "focus_scan" => Any[])
    save() = open(io -> JSON.print(io, out), joinpath(dir, "wake.json"), "w")
    for τ in taus
        push!(out["tau_scan"], wake_point(τ, focus0; ppl, dy_cells))
        println("wake τ = $τ: J_slab = ", out["tau_scan"][end]["J_slab"], ", ", round(out["tau_scan"][end]["seconds"]; digits = 1), " s"); flush(stdout)
        save()
    end
    for f in focuses
        push!(out["focus_scan"], wake_point(tau0, f; ppl, dy_cells, lab = false, map_times = (BOX_TIME,)))
        println("wake focus = $f: J_slab = ", out["focus_scan"][end]["J_slab"]); flush(stdout)
        save()
    end
    println("DONE wake")
end

# --- the witness case (WP-047, WP-052) ------------------------------------------------------------

energy_of(a0, τ) = π * a0^2 * WCASE.waist * τ / 4 * (1 + 1 / (2 * WCASE.waist^2))

"""
One point of the witness case, stepped as `witness_state` steps it, with read-only snapshots (so
the gains are the driver's): the gain of every lattice point at the end, the row-averaged gain
profile and the witnesses' window positions at `times`, the axis lineouts there, and the objective
over the centres 30:0.25:50 from this one run.
"""
function witness_point(τ, n0; energy = WCASE.energy, times = (150.5, 175.0, 200.0, 225.0, 250.0, 275.0))
    t0 = time()
    prob = witness_problem(τ, n0, 36.5; energy)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    p = params(prob)
    initialize!(st, p)
    dt = Float64(prob.dt)
    steps = Dict(round(Int, t / dt) => t for t in times)
    steps[prob.nsteps] = prob.nsteps * dt
    snaps = Any[]
    nrow(v) = (m = length(v) ÷ 2; (v[1:m] .+ v[(m + 1):end]) ./ 2)
    for k in 1:prob.nsteps
        step!(st, p, k)
        haskey(steps, k) || continue
        o = lineouts(st, k, n0)
        w = witness_gains(st, p)
        o["t"], o["alive"] = k * dt, count(w.alive)
        o["gain"], o["xw"] = sig(nrow(Float64.(w.gain))), sig(nrow(Float64.(w.x)))
        push!(snaps, o)
    end
    w = witness_gains(st, p)
    centers = collect(30.0:0.25:50.0)
    Jc = all(w.alive) ? gain_at_centers(st, p, centers) : fill(NaN, length(centers))
    return Dict{String,Any}("tau" => τ, "density" => n0, "a0" => a0_at_energy(τ; energy), "energy" => energy, "dt" => dt, "nsteps" => prob.nsteps,
                            "x0" => sig(lattice_x(st); n = 6), "gain" => sig(Float64.(w.gain); n = 6), "alive" => count(w.alive),
                            "centers" => centers, "J" => sig(Jc; n = 7), "J_at_36.5" => all(w.alive) ? witness_energy_gain(st, p) : NaN,
                            "snaps" => snaps, "seconds" => time() - t0)
end

function witness_mode(tag)
    taus = [4.0, 4.5, 4.9356, 5.5, 6.0, 7.0, 8.0, 9.0]
    dens = [0.05, 0.06, 0.072, 0.085, 0.095]
    dir = out_dir(tag, "witness", Dict{String,Any}("taus" => taus, "densities" => dens))
    out = Dict{String,Any}("case" => Dict(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(WCASE)), "fixed_energy" => Any[], "fixed_a0" => Any[])
    save() = open(io -> JSON.print(io, out), joinpath(dir, "witness.json"), "w")
    for n0 in dens, τ in taus
        push!(out["fixed_energy"], witness_point(τ, n0))
        println("witness τ = $τ, n₀ = $n0: best J = ", maximum(x -> x === nothing ? -Inf : x, out["fixed_energy"][end]["J"])); flush(stdout)
        save()
    end
    a0 = a0_at_energy(4.9356)
    for τ in taus
        push!(out["fixed_a0"], witness_point(τ, 0.072; energy = energy_of(a0, τ)))
        println("witness fixed a₀, τ = $τ: best J = ", maximum(x -> x === nothing ? -Inf : x, out["fixed_a0"][end]["J"])); flush(stdout)
        save()
    end
    println("DONE witness")
end

# --- sensitivities of the witness gain (which other parameters matter) ----------------------------

const SENS_NAMES = ("a0", "duration", "waist", "focus", "delay", "chirp", "density", "ramp", "vacuum", "u0", "center", "length")
"WP-052's optimum, with a₀ as its own parameter; `chirp` is 0 there, so its entry is the scale of z − 1 instead."
const SENS_P0 = (a0_at_energy(4.935566813273559), 4.935566813273559, WCASE.waist, WCASE.focus, WCASE.delay, 0.01, 0.07200235514790519,
                 WCASE.ramp, WCASE.vacuum, WCASE.u0, 36.55596361907769, WCASE.length)

"""
The witness gain as a function of z, every parameter of `SENS_NAMES` at `SENS_P0·z` (the chirp at
`0.01·(z − 1)`), a₀ independent of the duration: the case of `witness_problem` with every one of
them an argument. At z = 1 it is WP-052's optimum.
"""
function gain_of(z)
    T = eltype(z)
    q = NamedTuple{SENS_NAMES}(ntuple(i -> SENS_NAMES[i] == "chirp" ? SENS_P0[i] * (z[i] - 1) : SENS_P0[i] * z[i], length(z)))
    z3 = (zero(T), zero(T), zero(T))
    cells, ppl = WCASE.cells, 16
    h = 2 * T(π) / ppl
    hy = WCASE.dy_cells * h
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    pulse = GaussianLaser(; a0 = T(q.a0), waist = T(q.waist), duration = T(q.duration), delay = T(q.delay), focus = T(q.focus), chirp = T(q.chirp),
                          position = 6h, center = cells[2] * hy / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(q.density), zero(T), (1, 0), WCASE.ppc, z3, z3, 1)
    hy64 = Float64(value_of(hy))
    yc = cells[2] * hy64 / 2
    wb = WitnessBunch(; inject = WCASE.inject, region = (WCASE.region, (yc - hy64, yc + hy64)), ppc = (2, 1), momentum = T(q.u0),
                      center = (T(q.center), T(yc)), length = T(q.length))
    prob = EMProblem("witness2d", (cells[1] * h, cells[2] * hy), cells, dt, WCASE.nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :periodic), window = MovingWindow(; start = T(WCASE.start)),
                     profile = DensityRamp(; vacuum = T(q.vacuum), ramp = T(q.ramp)), witness = wb)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    p = params(prob)
    initialize!(st, p)
    run!(st, p, prob.nsteps)
    return witness_energy_gain(st, p)
end

function sensitivity_mode(tag)
    dir = out_dir(tag, "sensitivity", Dict{String,Any}("names" => collect(SENS_NAMES), "p0" => collect(SENS_P0)))
    n = length(SENS_NAMES)
    z0 = ones(n)
    t0 = time()
    J0 = gain_of(z0)
    Jcase = witness_gain(SENS_P0[2], SENS_P0[7], SENS_P0[11])
    println("J0 = $J0, the case's J = $Jcase, ", round(time() - t0; digits = 1), " s"); flush(stdout)
    t1 = time()
    g = ForwardDiff.gradient(gain_of, z0, ForwardDiff.GradientConfig(gain_of, z0, ForwardDiff.Chunk{n}()))
    tg = time() - t1
    println("∇J = $g, $(round(tg; digits = 1)) s"); flush(stdout)
    out = Dict{String,Any}("names" => collect(SENS_NAMES), "p0" => collect(SENS_P0), "J0" => J0, "J_case" => Jcase, "grad_z" => g,
                           "seconds_value" => t1 - t0, "seconds_gradient" => tg, "fd" => Any[])
    save() = open(io -> JSON.print(io, out), joinpath(dir, "sensitivity.json"), "w")
    save()
    for i in 1:n, hstep in (1e-2, 1e-3)
        e = [j == i ? hstep : 0.0 for j in 1:n]
        fd = (gain_of(z0 .+ e) - gain_of(z0 .- e)) / 2hstep
        push!(out["fd"], Dict("name" => SENS_NAMES[i], "h" => hstep, "fd" => fd, "ad" => g[i]))
        println("FD ", SENS_NAMES[i], " h = $hstep: fd = $fd, ad = ", g[i]); flush(stdout)
        save()
    end
    println("DONE sensitivity")
end

# --- the wake objective on a backend --------------------------------------------------------------

"WP-045's objective built on `be`; the element type follows `τ` and `focus`."
function wake_amplitude_on(be, τ, focus; mode = Reference(), kw...)
    prob = wake_problem(τ, focus; kw...)
    st = EMState(prob, EMNumerics(; mode); backend = be)
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    v = axis_Ex(st)
    return sqrt(2 * sum(abs2, v) / length(v)) / sqrt(CASE.density)
end

function backend_mode(name, tag)
    be = if name == "cuda"
        @eval using CUDACore: CUDACore
        @eval CUDACore.CUDABackend()
    else
        CPU()
    end
    device = name == "cuda" ? string(@eval CUDACore.name(CUDACore.device())) : Sys.cpu_info()[1].model
    dir = out_dir(tag, "backend-$name", Dict{String,Any}("backend" => name, "device" => device))
    out = Dict{String,Any}("backend" => name, "device" => device, "threads" => Threads.nthreads(), "runs" => Any[])
    save() = open(io -> JSON.print(io, out), joinpath(dir, "backend-$name.json"), "w")
    τ, focus = 7.742271854441553, 112.96457420993261         # WP-045's optimum
    grids = name == "cuda" ? ((16, 4), (32, 2), (48, 3), (64, 4)) : ((16, 4), (32, 2))
    for (ppl, dy_cells) in grids, T in (Float64, Float32), kind in ("value", "gradient"), mode in (Reference(), Fast())
        nsteps = CASE.nsteps * ppl ÷ 16
        f = if kind == "value"
            () -> wake_amplitude_on(be, T(τ), T(focus); mode, ppl, dy_cells, nsteps)
        else
            () -> wake_amplitude_on(be, ForwardDiff.Dual{Nothing}(T(τ), one(T), zero(T)), ForwardDiff.Dual{Nothing}(T(focus), zero(T), one(T)); mode, ppl, dy_cells, nsteps)
        end
        r = Dict{String,Any}("ppl" => ppl, "dy_cells" => dy_cells, "type" => string(T), "kind" => kind, "mode" => string(nameof(typeof(mode))), "nsteps" => nsteps,
                             "cells" => [14ppl, 16ppl ÷ dy_cells])
        try
            (ppl, dy_cells) == grids[1] && f()             # compile on the small grid
            t0 = time()
            v = f()
            r["seconds"] = time() - t0
            r["J"] = Float64(ForwardDiff.value(v))
            kind == "gradient" && (r["grad"] = Float64.(collect(ForwardDiff.partials(v))))
        catch err
            r["error"] = first(sprint(showerror, err), 400)
        end
        push!(out["runs"], r)
        println(r); flush(stdout)
        save()
    end
    println("DONE backend")
end

function main(args)
    mode = args[1]
    if mode == "wake"
        taus = parse.(Float64, split(args[5], ","))
        focuses = length(args) >= 6 ? parse.(Float64, split(args[6], ",")) : Float64[]
        wake_mode(parse(Int, args[2]), parse(Int, args[3]), args[4], taus, focuses)
    elseif mode == "witness"
        witness_mode(args[2])
    elseif mode == "sensitivity"
        sensitivity_mode(args[2])
    elseif mode == "backend"
        backend_mode(args[2], args[3])
    else
        error("mode $mode: wake, witness, sensitivity or backend")
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main(ARGS)
