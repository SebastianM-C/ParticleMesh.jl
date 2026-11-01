# T-236 beyond test length (WP-015): the energy error of the four pushers and of classic RK4 (the
# baseline that does not preserve phase-space volume) over long runs in a static non-uniform B with
# a static non-quadratic potential, in Float32 and Float64.
#
# The system is that of T-236 (lib/ParticleMeshElectromagnetic/test/pushers.jl): B = r ẑ,
# φ = 10⁻²/r, q/m = 1, H = γ + φ conserved exactly; x₀ = (0, −1, 0), u₀ = (0.1, 0.01, 0) and
# (1, 0.1, 0), Δt = π/10. The pushers step with `push_particle` (the module's device function);
# RK4 steps `lorentz_rhs!` with the same Δt, in the same element type.
#
#     julia +1.13 --project=scripts scripts/pushers/energy.jl [periods]
#
# writes scripts/runs/pushers-energy-<stamp>/ with manifest.toml (F3), one PNG and JSON sidecar
# per initial condition, and summary.json with the maximum |ΔH/H| and the drift rate, a
# least-squares slope of ΔH/H against t (per unit time, 1/ω₀) over 2000 samples. The plots show
# the maximum of |ΔH/H| over each of the 2000 blocks of steps, so the envelope does not alias.

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

function step(::RK4, x::NTuple{3,T}, u::NTuple{3,T}, t::T, dt::T) where {T}
    p = (; qm = one(T), E = QIN_E, B = QIN_B)
    s = T[x..., u...]
    k1, k2, k3, k4 = (similar(s) for _ in 1:4)
    lorentz_rhs!(k1, s, p, t)
    lorentz_rhs!(k2, s .+ dt / 2 .* k1, p, t + dt / 2)
    lorentz_rhs!(k3, s .+ dt / 2 .* k2, p, t + dt / 2)
    lorentz_rhs!(k4, s .+ dt .* k3, p, t + dt)
    s .+= dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    return (s[1], s[2], s[3]), (s[4], s[5], s[6])
end
step(pusher::Pusher, x::NTuple{3,T}, u, t, dt) where {T} = push_particle(pusher, x, u, t, dt, one(T), QIN_E, QIN_B)

function run_energy(scheme, ::Type{T}, u0, nsteps; samples = 2000) where {T}
    dt = T(π) / 10
    x = (zero(T), -one(T), zero(T))
    u = T.(u0)
    H0 = Float64(qin_H(x, u))
    every = max(1, nsteps ÷ samples)
    ts, dH, env = Float64[], Float64[], Float64[]
    m = 0.0
    for n in 1:nsteps
        x, u = step(scheme, x, u, (n - 1) * dt, dt)
        r = (Float64(qin_H(x, u)) - H0) / H0
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
    return (; ts, dH, env, maxrel = maximum(env), slope)
end

function main(periods)
    nsteps = 20 * periods                                # Δt = π/10: 20 steps per gyroperiod 2π at r = 1
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    dir = joinpath(REPO, "scripts", "runs", "pushers-energy-$stamp")
    mkpath(dir)
    inputs = String[@__FILE__, joinpath(REPO, "scripts", "crosscode", "manifest.jl"),
                    joinpath(REPO, "Manifest.toml"), joinpath(REPO, "scripts", "Manifest.toml")]
    for (root, _, fs) in walkdir(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "src")), f in fs
        endswith(f, ".jl") && push!(inputs, joinpath(root, f))
    end
    hashes = Dict{String,Any}(relpath(f, REPO) => "sha256:" * file_hash(f) for f in sort(inputs))
    params = Dict{String,Any}("periods" => periods, "steps" => nsteps, "dt" => "π/10", "x0" => [0, -1, 0],
                              "u0" => [[0.1, 0.01, 0.0], [1.0, 0.1, 0.0]], "field" => "B = r ẑ, φ = 0.01/r, q/m = 1")
    write_manifest(dir; root = REPO,
                   extra = Dict{String,Any}("inputs" => hashes, "run_id" => basename(dir), "work_package" => "WP-015",
                                            "test" => "T-236", "parameters" => params, "julia" => string(VERSION),
                                            "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model))
    schemes = (Boris(), Vay(), HigueraCary(), BorisExactRotation(), RK4())
    summary = Dict{String,Any}()
    for (label, u0) in (("nonrelativistic", (0.1, 0.01, 0.0)), ("relativistic", (1.0, 0.1, 0.0)))
        fig = Figure(size = (1100, 450))
        side = Dict{String,Any}("x" => "t (1/ω₀)", "y" => "max |ΔH/H| over each block of steps/2000 steps (floor 1e-10 in the plot)", "u0" => collect(u0), "steps" => nsteps)
        for (j, T) in enumerate((Float32, Float64))
            ax = Axis(fig[1, j]; xscale = log10, yscale = log10, xlabel = "t", ylabel = "max |ΔH/H| per block", limits = (nothing, (1e-10, 1)),
                      title = "$label, u₀ = $(u0[1:2]), $T")
            for sch in schemes
                name = string(nameof(typeof(sch)))
                r = run_energy(sch, T, u0, nsteps)
                lines!(ax, r.ts, max.(r.env, 1e-10); label = name)
                summary["$label $name $T"] = Dict("max_rel" => r.maxrel, "drift_rate" => r.slope)
                side["$name $T"] = Dict("t" => r.ts, "block_max_abs_dH" => r.env, "dH_at_block_end" => r.dH,
                                         "max_rel" => r.maxrel, "drift_rate" => r.slope)
                println(rpad("$label $name $T", 44), " max|ΔH/H| = ", round(r.maxrel; sigdigits = 3),
                        "   drift rate = ", round(r.slope; sigdigits = 3))
            end
            j == 2 && axislegend(ax; position = :lt)
        end
        save(joinpath(dir, "energy_$label.png"), fig; px_per_unit = 2)
        open(io -> JSON.print(io, side), joinpath(dir, "energy_$label.json"), "w")
    end
    open(io -> JSON.print(io, summary, 2), joinpath(dir, "summary.json"), "w")
    println("wrote ", dir)
    return dir
end

main(isempty(ARGS) ? 100_000 : parse(Int, ARGS[1]))
