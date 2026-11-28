# WP-068: frame sequences of single runs of the trapped-electron case (trapped2d_case.jl) at
# optima of the optimizations, for pages and figures: maps of the electron density and the
# laser's envelope in the moving window, the born and the plasma's fast electrons as points, and
# J's two parts over time. Data only; it asserts nothing and draws nothing.
#
#   julia +1.13 --project=scripts -t 4 scripts/optim/trapped2d_frames.jl <out dir> <label> <run dir> [best | d,w,f,n] [key=value ...]
#       the run at the best optimum of <run dir>'s starts (or at the free parameters d,w,f,n) with
#       that run's geometry and objective (window extension, read-out); keys: ppl=32 ny=2 float=32
#       backend=cpu gather=compensated frames=50 (frames evenly in t), sample=2500 (points per kind
#       and frame); writes <out dir>/manifest.toml and <out dir>/<label>.json

const FRAME_ARGS = copy(ARGS)
empty!(ARGS)  # trapped2d_opt.jl reads ARGS as its own command
include(joinpath(@__DIR__, "trapped2d_opt.jl"))
using Random: Random
using Base64: base64encode

const FRAME_DEFAULTS = (ppl = 32, ny = 2, float = 32, backend = "cpu", gather = "compensated", frames = 50, sample = 2500)

q3(x) = isfinite(x) ? round(Float64(x); sigdigits = 3) : nothing

"`M` as bytes 0 … 254 linear between lo and hi, column-major, base64."
function frame_bytes(M, lo, hi)
    b = map(v -> UInt8(clamp(round(Int, 254 * (v - lo) / (hi - lo)), 0, 254)), M)
    return Dict{String,Any}("n" => collect(size(M)), "lo" => lo, "hi" => hi, "b64" => base64encode(vec(b)))
end

# Window x, y, u_x, γ − 1 and weight of a species' alive particles, as plain numbers.
function species_xy(s, h)
    f(a) = Float64.(value_of.(Array(a)))
    alive = Array(s.status) .== ALIVE
    x = (Array(s.cell[1])[alive] .- 1 .+ f(s.offset[1])[alive]) .* h[1]
    y = (Array(s.cell[2])[alive] .- 1 .+ f(s.offset[2])[alive]) .* h[2]
    u = map(a -> f(a)[alive], s.u)
    return (; x, y, ux = u[1], energy = sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1, w = f(s.w)[alive])
end

"Weighted 2D histogram of `p` on `nb` bins over `L`, divided by the bin area: a density."
function density_map(ps, L, nb)
    M = zeros(nb)
    a = prod(L ./ nb)
    for p in ps, i in eachindex(p.x)
        bx = clamp(floor(Int, p.x[i] / L[1] * nb[1]) + 1, 1, nb[1])
        by = clamp(floor(Int, p.y[i] / L[2] * nb[2]) + 1, 1, nb[2])
        M[bx, by] += p.w[i]
    end
    return M ./ a  # a weight is a density times the particle's share of its cell
end

"The laser's envelope: |E_y| maximized over ±half a wavelength along x, then over the bins."
function envelope_map(st, ppl, nb)
    g, n = guards(st.grid), ncells(st.grid)
    E = abs.(Float64.(value_of.(Array(st.fields.Ey)[(g + 1):(g + n[1]), (g + 1):(g + n[2])])))
    h = ppl ÷ 2
    R = similar(E)
    for j in axes(E, 2), i in axes(E, 1)
        R[i, j] = maximum(@view E[max(1, i - h):min(end, i + h), j])
    end
    sx, sy = n[1] ÷ nb[1], n[2] ÷ nb[2]
    return [maximum(@view R[((i - 1) * sx + 1):(i * sx), ((j - 1) * sy + 1):(j * sy)]) for i in 1:nb[1], j in 1:nb[2]]
end

"Up to `m` of the particles of `p` where `keep`, picked with a fixed seed: x, y, u_x rounded."
function pick(p, keep, m, seed)
    idx = findall(keep)
    length(idx) > m && (idx = sort(Random.shuffle(Random.Xoshiro(seed), idx)[1:m]))
    return Dict{String,Any}("x" => q3.(p.x[idx]), "y" => q3.(p.y[idx]), "ux" => q3.(p.ux[idx]), "count" => count(keep))
end

function main_frames(out, label, rundir, at, o)
    ro = TOML.parsefile(joinpath(rundir, "manifest.toml"))["parameters"]
    free = at == "best" ? Float64.(begin
        rs = load_results(rundir)
        rs[argmax(n -> rs[n]["J"], collect(keys(rs)))]["free"]
    end) : parse.(Float64, split(at, ","))
    setup = run_setup(ro)
    F = element_type(o["float"])
    backend = opt_backend(o["backend"])
    gather = opt_gather(o["gather"])
    energy = budget(Float64(ro["a0E"]))
    dir = run_dir(label, @__FILE__; prefix = "optim", work_package = "WP-068", dir = out,
                  params = merge(Dict{String,Any}(o), Dict{String,Any}("label" => label, "run" => basename(normpath(rundir)), "free" => free,
                                                                       "objective" => setup.objective, "kappa" => setup.kappa, "extend" => setup.extend)),
                  extra_inputs = [joinpath(@__DIR__, "trapped2d_opt.jl"), joinpath(@__DIR__, "problem.jl"), joinpath(@__DIR__, "trapped2d_case.jl"),
                                  joinpath(@__DIR__, "born2d_case.jl"), joinpath(REPO, "lib", "Wakefields", "src", "atomic.jl"), joinpath(REPO, "lib", "Wakefields", "src", "cases2d.jl")])
    x = trapped_point(F.(free))
    prob = trapped_problem(x...; energy = F(energy), ppl = o["ppl"], ny = o["ny"], float = F, extend = setup.extend)
    st, p, _ = start_of(prob; ppl = o["ppl"], ny = o["ny"], backend, gather)
    h = Float64.(value_of.(cellsize(st.grid)))
    n = ncells(st.grid)
    L = (n[1] * h[1], n[2] * h[2])
    nb = (n[1] ÷ 2, n[2] ÷ 4)
    n0 = free[4]
    dt = Float64(value_of(prob.dt))
    marks = unique(round.(Int, range(0, prob.nsteps; length = o["frames"])))
    frames = Any[]
    t0 = time()
    done = 0
    for k in marks
        k > done && (run!(st, p, k - done; first = done + 1); done = k)
        plasma = species_xy(st.species[1], h)
        born = species_xy(ionization_electrons(st), h)
        parts = try
            pr = trapped_parts(st; setup.read...)
            (; born = Float64(value_of(pr.born)), plasma = Float64(value_of(pr.plasma)))
        catch
            (; born = nothing, plasma = nothing)
        end
        fr = Dict{String,Any}("t" => k * dt, "step" => k, "shift" => window_shifts(st, k) * h[1], "J_born" => parts.born, "J_plasma" => parts.plasma,
                              "ne" => frame_bytes(density_map((plasma, born), L, nb) .* (h[1] * h[2] / n0), 0.0, 4.0),
                              "a" => frame_bytes(envelope_map(st, o["ppl"], nb), 0.0, 4.0),
                              "born" => pick(born, trues(length(born.x)), o["sample"], k),
                              "fast" => pick(plasma, plasma.ux .> 1.0, o["sample"], k + 1))
        push!(frames, fr)
        println("frame t = $(round(k * dt; digits = 1)): born $(length(born.x)), fast $(fr["fast"]["count"]), $(round(time() - t0; digits = 1)) s"); flush(stdout)
    end
    ionization_lost(st) == 0 || error("$(ionization_lost(st)) births found no free slot")
    a0 = a0_for_energy(free[1], free[2]; energy)
    fade = get(setup.read, :fade, TCASE.fade)
    res = Dict{String,Any}("label" => label, "free" => free, "a0" => a0, "objective" => setup.objective, "kappa" => setup.kappa, "extend" => setup.extend,
                           "L" => collect(L), "nb" => collect(nb), "dx" => h, "dt" => dt, "nsteps" => prob.nsteps, "fade" => collect(fade),
                           "rise" => collect(TCASE.rise), "window_start" => BCASE.start + 2π * setup.extend, "vacuum" => BCASE.vacuum, "ramp" => BCASE.ramp,
                           "section" => Dict(string(k) => v for (k, v) in pairs(BCASE.section)), "seconds" => time() - t0, "frames" => frames,
                           "summary" => Dict{String,Any}(string(k) => v for (k, v) in pairs(trapped_summary(st; setup.read...))))
    open(io -> JSON.print(io, res), joinpath(dir, "$label.json"), "w")
    println("wrote $(joinpath(dir, "$label.json")) in $(round(time() - t0; digits = 1)) s")
end

if length(FRAME_ARGS) >= 3
    o, rest = parse_options(FRAME_ARGS[4:end], FRAME_DEFAULTS)
    at = isempty(rest) ? "best" : rest[1]
    main_frames(FRAME_ARGS[1], FRAME_ARGS[2], FRAME_ARGS[3], at, o)
end
