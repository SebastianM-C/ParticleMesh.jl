# T-236 beyond test length (WP-015): the energy error of the four pushers and of classic RK4 (the
# baseline that does not preserve phase-space volume) over long runs in static fields, in Float32
# and Float64, for the two systems of T-236 (lib/ParticleMeshElectromagnetic/test/pushers.jl,
# whose header states the sources and the assumed initial conditions):
#
# - Higuera and Cary's integrable system (Phys. Plasmas 24, 052104 (2017), Eqs. 43–51 of
#   arXiv:1701.05605v1): E = −x x̂, B = 2y x̂, H = γ + x²/2 = 4, Δt = 1/10, from x = y = 0 with
#   p_y,0 ∈ {1.625, 1.775};
# - B = r ẑ, φ = 10⁻²/r (the field of Qin et al. 2013), H = γ + φ, Δt = π/10, x₀ = (0, −1, 0),
#   u₀ = (0.1, 0.01, 0) and (1, 0.1, 0).
#
# q/m = 1. The pushers step with `push_particle` (the module's device function); RK4 steps
# `lorentz_rhs!` with the same Δt, in the same element type.
#
#     julia +1.13 --project=scripts scripts/pushers/energy.jl [periods]
#
# runs `periods` periods of 2π (default 10⁵) and writes scripts/runs/pushers-energy-<stamp>/ with
# manifest.toml (F3), one PNG and JSON sidecar per case, and summary.json with the maximum
# |ΔH/H| and the drift rate, a least-squares slope of ΔH/H against t (per unit time, 1/ω₀) over
# 2000 samples. The plots show the maximum of |ΔH/H| over each of the 2000 blocks of steps, so the
# envelope does not alias.

using ParticleMesh
using ParticleMeshElectromagnetic
using CairoMakie
using JSON
using Dates

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "scripts", "crosscode", "manifest.jl"))
using .RunManifest: write_manifest, file_hash

struct RK4 end

const QIN_E = (x, t) -> (r = sqrt(x[1]^2 + x[2]^2); c = (one(r) / 100) / r^3; (c * x[1], c * x[2], zero(r)))
const QIN_B = (x, t) -> (zero(x[1]), zero(x[1]), sqrt(x[1]^2 + x[2]^2))
qin_H(x, u) = lorentz_factor(u) + (one(x[1]) / 100) / sqrt(x[1]^2 + x[2]^2)

const HC_E = (x, t) -> (-x[1], zero(x[1]), zero(x[1]))
const HC_B = (x, t) -> (2 * x[2], zero(x[1]), zero(x[1]))
hc_H(x, u) = lorentz_factor(u) + x[1]^2 / 2

"The cases: name, fields, invariant, Δt (as a function of T), start (x₀, u₀ in Float64)."
const CASES = [
    (name = "hc_py1.625", E = HC_E, B = HC_B, H = hc_H, dt = T -> T(1) / 10, x0 = (0.0, 0.0, 0.0), u0 = (sqrt(15 - 1.625^2), 1.625, 0.0)),
    (name = "hc_py1.775", E = HC_E, B = HC_B, H = hc_H, dt = T -> T(1) / 10, x0 = (0.0, 0.0, 0.0), u0 = (sqrt(15 - 1.775^2), 1.775, 0.0)),
    (name = "qin_nonrelativistic", E = QIN_E, B = QIN_B, H = qin_H, dt = T -> T(π) / 10, x0 = (0.0, -1.0, 0.0), u0 = (0.1, 0.01, 0.0)),
    (name = "qin_relativistic", E = QIN_E, B = QIN_B, H = qin_H, dt = T -> T(π) / 10, x0 = (0.0, -1.0, 0.0), u0 = (1.0, 0.1, 0.0)),
]

function step(::RK4, c, x::NTuple{3,T}, u::NTuple{3,T}, t::T, dt::T) where {T}
    p = (; qm = one(T), E = c.E, B = c.B)
    s = T[x..., u...]
    k1, k2, k3, k4 = (similar(s) for _ in 1:4)
    lorentz_rhs!(k1, s, p, t)
    lorentz_rhs!(k2, s .+ dt / 2 .* k1, p, t + dt / 2)
    lorentz_rhs!(k3, s .+ dt / 2 .* k2, p, t + dt / 2)
    lorentz_rhs!(k4, s .+ dt .* k3, p, t + dt)
    s .+= dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    return (s[1], s[2], s[3]), (s[4], s[5], s[6])
end
step(pusher::Pusher, c, x::NTuple{3,T}, u, t, dt) where {T} = push_particle(pusher, x, u, t, dt, one(T), c.E, c.B)

function run_energy(scheme, c, ::Type{T}, periods; samples = 2000) where {T}
    dt = c.dt(T)
    nsteps = round(Int, periods * 2π / Float64(dt))
    x = T.(c.x0)
    u = T.(c.u0)
    H0 = Float64(c.H(x, u))
    every = max(1, nsteps ÷ samples)
    ts, dH, env = Float64[], Float64[], Float64[]
    m = 0.0
    for n in 1:nsteps
        x, u = step(scheme, c, x, u, (n - 1) * dt, dt)
        r = (Float64(c.H(x, u)) - H0) / H0
        m = max(m, abs(r))
        if n % every == 0
            push!(ts, n * Float64(dt))
            push!(dH, r)
            push!(env, m)                                # max |ΔH/H| over the block of `every` steps
            m = 0.0
        end
    end
    tm = sum(ts) / length(ts)
    slope = sum((ts .- tm) .* dH) / sum((ts .- tm) .^ 2)
    return (; ts, dH, env, maxrel = maximum(env), slope, nsteps)
end

function main(periods)
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    dir = joinpath(REPO, "scripts", "runs", "pushers-energy-$stamp")
    mkpath(dir)
    inputs = String[@__FILE__, joinpath(REPO, "scripts", "crosscode", "manifest.jl"),
                    joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")]
    for (root, _, fs) in walkdir(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "src")), f in fs
        endswith(f, ".jl") && push!(inputs, joinpath(root, f))
    end
    hashes = Dict{String,Any}(relpath(f, REPO) => "sha256:" * file_hash(f) for f in sort(inputs))
    params = Dict{String,Any}("periods_of_2pi" => periods, "qm" => 1,
                              "cases" => [Dict("name" => c.name, "dt" => Float64(c.dt(Float64)), "x0" => collect(c.x0), "u0" => collect(c.u0))
                                          for c in CASES])
    write_manifest(dir; root = REPO,
                   extra = Dict{String,Any}("inputs" => hashes, "run_id" => basename(dir), "work_package" => "WP-015",
                                            "test" => "T-236", "parameters" => params, "julia" => string(VERSION),
                                            "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
    schemes = (Boris(), Vay(), HigueraCary(), BorisExactRotation(), RK4())
    summary = Dict{String,Any}()
    for c in CASES
        fig = Figure(size = (1100, 450))
        side = Dict{String,Any}("x" => "t (1/ω₀)", "y" => "max |ΔH/H| over each of 2000 blocks of steps (floor 1e-10 in the plot)",
                                "x0" => collect(c.x0), "u0" => collect(c.u0), "dt" => Float64(c.dt(Float64)))
        for (j, T) in enumerate((Float32, Float64))
            ax = Axis(fig[1, j]; xscale = log10, yscale = log10, xlabel = "t", ylabel = "max |ΔH/H| per block",
                      limits = (nothing, (1e-10, 1)), title = "$(c.name), $T")
            for sch in schemes
                name = string(nameof(typeof(sch)))
                r = run_energy(sch, c, T, periods)
                lines!(ax, r.ts, max.(r.env, 1e-10); label = name)
                summary["$(c.name) $name $T"] = Dict("max_rel" => r.maxrel, "drift_rate" => r.slope, "steps" => r.nsteps)
                side["$name $T"] = Dict("t" => r.ts, "block_max_abs_dH" => r.env, "dH_at_block_end" => r.dH,
                                         "max_rel" => r.maxrel, "drift_rate" => r.slope, "steps" => r.nsteps)
                println(rpad("$(c.name) $name $T", 44), " max|ΔH/H| = ", round(r.maxrel; sigdigits = 3),
                        "   drift rate = ", round(r.slope; sigdigits = 3))
            end
            j == 2 && axislegend(ax; position = :lt)
        end
        save(joinpath(dir, "energy_$(c.name).png"), fig; px_per_unit = 2)
        open(io -> JSON.print(io, side), joinpath(dir, "energy_$(c.name).json"), "w")
    end
    open(io -> JSON.print(io, summary, 2), joinpath(dir, "summary.json"), "w")
    println("wrote ", dir)
    return dir
end

main(isempty(ARGS) ? 100_000 : parse(Int, ARGS[1]))
