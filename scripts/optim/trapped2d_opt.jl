# WP-068: the first optimizations of WP-065's trapped-electron objective (trapped2d_case.jl):
# L-BFGS-B over the four free parameters (duration, waist, focus, density) in `TRAPPED_BOX` at
# the energy budget of a₀ = 1.9, from a few starts, recording J, its gradient, a₀ and the dark
# current at every point; then a check of the best optimum on a finer grid and against Float64
# central differences.
#
#   julia +1.13 --project=scripts -t 2 scripts/optim/trapped2d_opt.jl opt [dir] [key=value ...]
#       every start of `STARTS` (or `starts=base,long`) into <dir>/<start>/, the points into
#       <dir>/points/; keys: ppl=32 ny=2 (Δx = λ₀/ppl, Δy = π/(4·ny)), float=32, backend=cuda
#       (or cpu), gather=compensated (or plain), maxiters=25, pgtol=<rel> (the stopping
#       tolerance on the projected gradient in z, relative to the largest component of the
#       base point's gradient; default 2⁻⁶), a0E=1.9 (the budget's a₀ at the base pulse),
#       cap=none (or cetal25, cetal35: a peak power cap, which at the fixed energy is a floor on
#       the duration that rises with the density, `cetal_cap`; the duration then lies between
#       the floor and duration_hi=12, `Coords`, and the window is longer behind by
#       `window_extend` wavelengths to emit the longest pulse), objective=any (or born: the
#       ionization's electrons alone; plasmacap: any origin with the plasma's own share capped
#       smoothly at kappa=0.05, `objective_value`), from=<run dir> (one start at that run's best
#       optimum instead of the starts, e.g. to re-optimize it on a finer grid)
#   julia +1.13 --project=scripts -t 2 scripts/optim/trapped2d_opt.jl check <dir> [ppl=48 ny=3 fdppl=32 fdny=2 power=13]
#       at the best optimum of <dir>'s starts: a plain run and one with duals at ppl×ny in the
#       optimization's element type, and a Float64 run with duals and Float64 central differences
#       at a relative step 2^-power in the four free directions at fdppl×fdny, into <dir>/check/
#   julia +1.13 --project=scripts -t 2 scripts/optim/trapped2d_opt.jl scan <dir> [n=11 shard=1/1]
#       plain runs on n×n grids for figures of J around the paths: the plane of each start's path
#       and every pair of free parameters over the box through the best optimum, into <dir>/scan/
#       (`shard=k/m` runs every m-th job, for m processes on one card)
#   julia +1.13 --project=scripts scripts/optim/trapped2d_opt.jl report <out dir> <run dir>
#       trapped2d_opt.png (paths) and trapped2d_opt_scan.png (J around them, when scanned), each
#       with its JSON sidecar, and tables.md into <out dir>
#
# Every point the optimizer asks for, for its value or for its gradient, is one run with duals
# in the four free directions, which gives both: the value and the gradient L-BFGS-B asks for
# separately at the same point (#440) come from that one run. Each point's record is a JSON file
# named by the point's bits in <dir>/points/, and a point with a record is not run again, so an
# optimization that was stopped replays its path from the records (L-BFGS-B is deterministic) and
# continues where it stopped; a start with a result.json is done.

isdefined(@__MODULE__, :run_dir) || include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
isdefined(@__MODULE__, :case_problem) || include(joinpath(@__DIR__, "problem.jl"))
isdefined(@__MODULE__, :TCASE) || include(joinpath(@__DIR__, "trapped2d_case.jl"))
using ParticleMeshElectromagnetic: CompensatedGather

"""
The starts: the base point and three others inside `TRAPPED_BOX`, as (duration, waist, focus,
density). At the budget of a₀ = 1.9 their a₀ are 1.9 (base), 1.50 (long: a longer, wider pulse
focused deeper, into a thinner plasma), 2.32 (short: shorter and narrower into a denser plasma,
below the plasma's own injection at a₀ ≈ 2.45 to 2.55 of WP-065) and 2.10 (narrow: a narrow
pulse focused deep into a medium density).
"""
const STARTS = (base = [6.0, 10.0, 80.0, 0.072], long = [8.0, 12.0, 100.0, 0.06], short = [5.0, 8.0, 70.0, 0.09], narrow = [7.0, 7.0, 104.0, 0.08])

"""
    PowerCap(c, p)

A cap on the pulse's peak power. At a fixed energy the peak power is W/τ_FWHM whatever the waist
(`pulse_energy` is linear in the duration), so the cap is a floor on the duration, here
τ ≥ c·n^(−p) for the density n (in n_c): [`duration_floor`](@ref).
"""
struct PowerCap
    c::Float64
    p::Float64
end

"The shortest duration (the field's 1/e half width) that `cap` allows at the density `n`."
duration_floor(cap::PowerCap, n) = cap.c * n^(-cap.p)

"""
    cetal_cap(; fwhm_fs = 25, density_cm3 = 4.2e18) -> PowerCap

CETAL-PW's peak power at its energy (1 PW at 25 J, a 25 fs transform limit), carried to this case
by τ/λ_p as the CETAL-scaled proposal of the optimization setup does: τ_FWHM ≥ (fwhm_fs/T_p)·λ_p,
T_p = 2π/ω_p at `density_cm3` (the middle of CETAL's jet, 3.2 to 5.2·10¹⁸ cm⁻³, Diplasu et al.
2021, a reference from memory, unverified: the paper has not been checked), ω_p = 5.641·10⁴ rad/s·√(n/cm⁻³). In the case's units (ω₀ = 1, λ_p = 2π/√n, intensity FWHM
= τ·√(2 ln 2)) that is τ ≥ c/√n with c = (fwhm_fs/T_p)·2π/√(2 ln 2), about 2.45 at 25 fs.
"""
function cetal_cap(; fwhm_fs = 25.0, density_cm3 = 4.2e18)
    Tp_fs = 2π / (5.641e4 * sqrt(density_cm3)) * 1.0e15
    return PowerCap(fwhm_fs / Tp_fs * 2π / sqrt(2 * log(2)), 0.5)
end

"""
    Coords(cap; duration_hi = 12)

The solver's coordinates z over the free parameters and their bounds. Without a cap (`nothing`),
z = x ./ scale over `TRAPPED_BOX`. With a [`PowerCap`](@ref), the duration lies between its floor
at the point's density and `duration_hi`, and the first coordinate is its place between the two,
(τ − τ_floor(n))/(duration_hi − τ_floor(n)) from 0 to 1; the others are as without. That region
is curved in (τ, n), which L-BFGS-B's bounds cannot hold in τ itself. [`free_of`](@ref) maps z to
the free parameters and carries duals through.
"""
struct Coords{C}
    cap::C
    duration_hi::Float64
    lo::Vector{Float64}
    hi::Vector{Float64}
end
function Coords(cap = nothing; duration_hi = 12.0)
    lo, hi = TRAPPED_BOX.lo ./ TRAPPED_BOX.scale, TRAPPED_BOX.hi ./ TRAPPED_BOX.scale
    cap === nothing && return Coords{Nothing}(nothing, TRAPPED_BOX.hi[1], lo, hi)
    duration_floor(cap, TRAPPED_BOX.lo[4]) < duration_hi ||
        throw(ArgumentError("the floor at the box's lowest density, $(duration_floor(cap, TRAPPED_BOX.lo[4])), is not below duration_hi = $duration_hi"))
    return Coords{PowerCap}(cap, Float64(duration_hi), [0.0; lo[2:end]], [1.0; hi[2:end]])
end

"The free parameters (duration, waist, focus, density) at the solver's coordinates `z`."
free_of(c::Coords{Nothing}, z) = collect(z) .* TRAPPED_BOX.scale
function free_of(c::Coords{PowerCap}, z)
    s = TRAPPED_BOX.scale
    n = z[4] * s[4]
    f = duration_floor(c.cap, n)
    return [f + z[1] * (c.duration_hi - f), z[2] * s[2], z[3] * s[3], n]
end

"The place of the duration `τ` between its floor at the density `n` and `duration_hi`, from 0 to 1."
duration_place(c::Coords{PowerCap}, τ, n) = (f = duration_floor(c.cap, n); (τ - f) / (c.duration_hi - f))

"The solver's coordinates of the free parameters `free`, the inverse of [`free_of`](@ref)."
z_of(c::Coords{Nothing}, free) = collect(Float64, free) ./ TRAPPED_BOX.scale
z_of(c::Coords{PowerCap}, free) = [duration_place(c, free[1], free[4]); collect(Float64, free[2:end]) ./ TRAPPED_BOX.scale[2:end]]

"The parameters as the figures show them: the free parameters, with the duration's place between its floor and `duration_hi` in place of the duration under a cap."
shown(c::Coords{Nothing}, free) = collect(Float64, free)
shown(c::Coords{PowerCap}, free) = [duration_place(c, free[1], free[4]); collect(Float64, free[2:end])]
shown_bounds(c::Coords{Nothing}) = (lo = collect(TRAPPED_BOX.lo), hi = collect(TRAPPED_BOX.hi))
shown_bounds(c::Coords{PowerCap}) = (lo = [0.0; TRAPPED_BOX.lo[2:end]], hi = [1.0; TRAPPED_BOX.hi[2:end]])
shown_name(c::Coords, i) = i == 1 && c isa Coords{PowerCap} ? "τ: floor→$(fmt_hi(c.duration_hi))" : TCASE.names[TCASE.free[i]]
fmt_hi(x) = isinteger(x) ? string(Int(x)) : string(x)

"The longest duration of the coordinates' region."
duration_max(c::Coords) = c.cap === nothing ? TRAPPED_BOX.hi[1] : c.duration_hi

"The gradient in the solver's coordinates at `z` from `g`, the gradient in the free parameters (the chain rule through [`free_of`](@ref))."
grad_z(c, z, g) = ForwardDiff.jacobian(u -> free_of(c, u), collect(Float64, z))' * collect(Float64, g)

"The coordinates of a run's options (or manifest parameters): `cap = none`, `cetal25` or `cetal35` (CETAL's 25 fs limit or its measured 35 fs), with `duration_hi`."
function coords_of(o)
    cap = get(o, "cap", "none")
    cap == "none" && return Coords()
    cap in ("cetal25", "cetal35") || error("the cap is none, cetal25 or cetal35, got $cap")
    return Coords(cetal_cap(; fwhm_fs = cap == "cetal25" ? 25.0 : 35.0); duration_hi = get(o, "duration_hi", 12.0))
end

"""
    window_extend(c) -> k

The whole wavelengths the case's window needs behind its 14 (born_problem's `extend`) for the
coordinates' longest pulse: the window may pass the antenna only six durations after the pulse's
centroid (`check_window_antennas`), so delay + 6τ must stay within the window's start, 108 + 2πk
(the delay at the base point's 54). Extending the window keeps the case's box, plasma and pulse
where they are in the lab frame (it only adds wake behind, T-976), with the objective's position
weight moved back by 2πk in window x ([`run_read`](@ref)). 0 for the box without a cap (τ ≤ 9).
"""
window_extend(c::Coords) = max(0, ceil(Int, (TCASE.x0[4] + 6 * duration_max(c) - BCASE.start) / (2π) - 1.0e-12))

"The read-out keywords of a run whose window is `k` wavelengths longer: the position weight `fade` moved back with the window's back."
run_read(k::Integer) = k == 0 ? (;) : (; fade = TCASE.fade .+ 2π * k)

"The coordinates a run directory's optimization used, from its manifest."
run_coords(dir) = coords_of(TOML.parsefile(joinpath(dir, "manifest.toml"))["parameters"])

"""
    starts_of(c) -> NamedTuple of starts (free parameters)

[`STARTS`](@ref) without a cap; under one, each start's waist, focus and density with its duration
moved to the same place between the floor and `duration_hi` as it had between the box's 4 and 9.
"""
starts_of(c::Coords{Nothing}) = STARTS
function starts_of(c::Coords{PowerCap})
    lo, hi = TRAPPED_BOX.lo[1], TRAPPED_BOX.hi[1]
    return map(x -> free_of(c, [(x[1] - lo) / (hi - lo); x[2:end] ./ TRAPPED_BOX.scale[2:end]]), STARTS)
end

"The backend named `cpu` or `cuda` (loaded on first use; the scripts environment has CUDACore)."
function opt_backend(name)
    name == "cpu" && return CPU()
    name == "cuda" || error("the backend is cpu or cuda, got $name")
    @eval Main using CUDACore: CUDACore, CUDABackend
    return Base.invokelatest(() -> Main.CUDABackend())
end

"The gather named `plain` (the grid's fields) or `compensated` (WP-067 at the laser's wavenumber)."
opt_gather(name) = name == "plain" ? nothing : name == "compensated" ? CompensatedGather() : error("the gather is plain or compensated, got $name")

"""
    run_point(free; ppl, ny, float, backend, gather, grad, energy, read, kw...) -> Dict

One run of the case at the four free parameters `free` (the delay at the base point's), on
`backend` in `float`, with `gather` ([`born_numerics`](@ref)): J, the gradient in the free
parameters (physical units) when `grad` (duals in the four free directions; J is the plain run's
bit for bit, T-912), a₀, the run's seconds and [`trapped_summary`](@ref), with its dark current.
`read` holds `fade` and `rise` for J and the summary; `objective` and `kappa` choose what J is
([`objective_value`](@ref)); the other keywords go to the case's problem.
"""
function run_point(free; ppl::Integer, ny::Integer, float::Type{<:AbstractFloat} = Float64, backend = CPU(), gather = nothing,
                   grad::Bool = true, energy = budget(), read = (;), objective::AbstractString = "any", kappa::Real = 0.05, kw...)
    F = float
    x = trapped_point(collect(Float64, free))
    n = length(TCASE.free)
    arg(i) = (k = findfirst(==(i), TCASE.free); grad && k !== nothing ? ForwardDiff.Dual{Nothing}(F(x[i]), ntuple(j -> F(j == k), n)) : F(x[i]))
    args = ntuple(arg, length(x))
    t = @elapsed st = trapped_state(args...; energy = F(energy), ppl, ny, float = F, backend, gather, kw...)
    J = objective_value(trapped_parts(st; read...), objective, kappa)
    out = Dict{String,Any}(string(k) => v for (k, v) in pairs(trapped_summary(st; read...)))
    out["objective"] = objective
    out["J"] = Float64(value_of(J))
    out["free"] = collect(Float64, free)
    out["x"] = collect(Float64, x)
    out["z"] = collect(Float64, free) ./ TRAPPED_BOX.scale
    out["a0"] = a0_for_energy(x[1], x[2]; energy = Float64(energy))
    out["gradient"] = grad ? collect(Float64, ForwardDiff.partials(J)) : nothing
    out["seconds"] = t
    out["eltype"] = string(F)
    out["ppl"], out["ny"] = ppl, ny
    out["gather"] = gather === nothing ? "plain" : "compensated"
    out["cells"] = collect(ncells(st.grid))
    return out
end

"A point's name: the bits of its free parameters in hexadecimal."
point_key(free) = join((string(reinterpret(UInt64, Float64(v)); base = 16, pad = 16) for v in free), "-")

"""
    PointStore(dir; run)

The records of the points in `dir`, read from its JSON files, and `run(free) -> Dict` that makes a
new one ([`run_point`](@ref) with duals); `made` counts the points this store ran.
"""
mutable struct PointStore
    dir::String
    run::Function
    records::Dict{String,Dict{String,Any}}
    made::Int
end
function PointStore(dir; run)
    mkpath(dir)
    records = Dict{String,Dict{String,Any}}()
    for f in readdir(dir)
        endswith(f, ".json") || continue
        r = JSON.parsefile(joinpath(dir, f))
        records[r["key"]] = r
    end
    return PointStore(dir, run, records, 0)
end

"The record of the point `free` in `store`, made (and written) if it has none."
function record!(store::PointStore, free)
    k = point_key(free)
    haskey(store.records, k) && return store.records[k]
    r = store.run(collect(Float64, free))
    r["key"] = k
    tmp = joinpath(store.dir, k * ".json.tmp")
    open(io -> JSON.print(io, r, 2), tmp, "w")
    mv(tmp, joinpath(store.dir, k * ".json"); force = true)
    store.records[k] = r
    store.made += 1
    return r
end

"""
    stored_objective(store, calls) -> J(free...)

The objective for [`case_problem`](@ref) from the store's records: J at the point, and for duals
J + Σ ∂J/∂x_i·(x_i − value(x_i)), the record's gradient carried by whatever duals the solver's
AD hands in. Each call appends to `calls` the point's key, its parameters, what was asked
(`value` or `gradient`) and whether the store had the record.
"""
function stored_objective(store::PointStore, calls::Vector)
    return function (x...)
        v = Float64[value_of(xi) for xi in x]
        had = haskey(store.records, point_key(v))
        r = record!(store, v)
        push!(calls, Dict{String,Any}("key" => r["key"], "free" => v, "asked" => eltype(x) <: ForwardDiff.Dual ? "gradient" : "value", "stored" => had))
        J, g = Float64(r["J"]), Float64.(r["gradient"])
        return eltype(x) <: ForwardDiff.Dual ? J + sum(g[i] * (x[i] - v[i]) for i in eachindex(g)) : J
    end
end

"The largest component of the projected gradient in z at `z` (a component pushing out of the box at its bound counts zero): what L-BFGS-B's `pgtol` bounds, for a maximum."
function projected_gradient(gz, z, lb, ub)
    p = map(eachindex(gz)) do i
        (z[i] <= lb[i] && gz[i] < 0) || (z[i] >= ub[i] && gz[i] > 0) ? 0.0 : gz[i]
    end
    return maximum(abs, p)
end

"""
    optimize_start(store, start, dir; maxiters, pgtol, coords = Coords()) -> Dict

L-BFGS-B over the bounds of `coords` (`TRAPPED_BOX` without a cap) in its coordinates z, from
`start` (the free parameters), with the store's objective; writes
`dir/result.json` (the solution, the record of every point the solver reported, the calls) and
returns it.
"""
function optimize_start(store::PointStore, start, dir; maxiters::Integer, pgtol::Real, coords = Coords())
    mkpath(dir)
    calls = Dict{String,Any}[]
    J = stored_objective(store, calls)
    prob = case_problem((z...) -> J(free_of(coords, collect(z))...), (lo = coords.lo, hi = coords.hi, scale = ones(length(coords.lo))), Tuple(z_of(coords, start)))
    rec = Record(prob, LBFGSB(); verbose = true)
    made = store.made
    t = @elapsed sol = solve(prob, LBFGSB(); callback = rec, maxiters, reltol = pgtol)
    Jopt = Float64(sol.objective)
    r = record!(store, free_of(coords, sol.u))
    out = Dict{String,Any}("start" => collect(Float64, start), "u" => collect(sol.u), "free" => free_of(coords, sol.u), "J" => Jopt,
                           "retcode" => string(sol.retcode), "iterations" => sol.stats.iterations, "fevals" => sol.stats.fevals,
                           "maxiters" => maxiters, "pgtol" => pgtol, "seconds" => t, "points_run" => store.made - made,
                           "key" => r["key"], "a0" => r["a0"], "projected_gradient" => projected_gradient(grad_z(coords, sol.u, r["gradient"]), sol.u, prob.lb, prob.ub),
                           "path" => [Dict{String,Any}("z" => u, "J" => j, "seconds" => s, "key" => point_key(free_of(coords, u))) for (u, j, s) in zip(rec.u, rec.objective, rec.seconds)],
                           "calls" => calls)
    open(io -> JSON.print(io, out, 2), joinpath(dir, "result.json"), "w")
    return out
end

"""
    objective_value(parts, objective, kappa) -> J

The objective from the two parts of WP-065's J ([`trapped_parts`](@ref)), in their element type:
- `"any"`: born + plasma, WP-065's J (every trapped electron of the first bucket, any origin);
- `"born"`: born alone, the electrons the ionization released (WP-064's label);
- `"plasmacap"`: born + κ·tanh(plasma/κ), any origin with the plasma's own injection counted
  in full while small and capped smoothly at `kappa`: it can add at most κ to J, so the
  optimizer gains nothing by driving the plasma deep into its own injection.
"""
function objective_value(parts, objective::AbstractString, kappa::Real)
    objective == "any" && return parts.born + parts.plasma
    objective == "born" && return parts.born
    if objective == "plasmacap"
        k = oftype(parts.plasma, kappa)
        return parts.born + k * tanh(parts.plasma / k)
    end
    throw(ArgumentError("the objective is any, born or plasmacap, got $objective"))
end

"`key=value` options from `args` over `defaults`, each parsed to its default's type; an unknown key is an error."
function parse_options(args, defaults)
    opts = Dict{String,Any}(string(k) => v for (k, v) in pairs(defaults))
    rest = String[]
    for a in args
        m = match(r"^(\w+)=(.*)$", a)
        m === nothing && (push!(rest, a); continue)
        k, v = m[1], m[2]
        haskey(opts, k) || error("unknown option $k; the options are $(join(sort(collect(keys(opts))), ", "))")
        d = opts[k]
        opts[k] = d isa Integer ? parse(Int, v) : d isa AbstractFloat ? parse(Float64, v) : v
    end
    return opts, rest
end

const OPT_DEFAULTS = (ppl = 32, ny = 2, float = 32, backend = "cuda", gather = "compensated", maxiters = 25, pgtol = 2.0^-6, a0E = 1.9, starts = join(keys(STARTS), ","),
                      cap = "none", duration_hi = 12.0, objective = "any", kappa = 0.05, from = "")

element_type(bits) = bits == 32 ? Float32 : bits == 64 ? Float64 : error("float is 32 or 64, got $bits")

"The window's extension and the read-out keywords of a run from its options (or manifest parameters): [`window_extend`](@ref), [`run_read`](@ref)."
run_geometry(o) = (k = window_extend(coords_of(o)); (; extend = k, read = run_read(k)))

"What every run of a run directory is run with besides its numerics: [`run_geometry`](@ref) and the objective (`any` where the options predate the choice)."
run_setup(o) = (; run_geometry(o)..., objective = string(get(o, "objective", "any")), kappa = Float64(get(o, "kappa", 0.05)))

"""
    start_points(o) -> NamedTuple of starts (free parameters)

The starts of a run's options: [`starts_of`](@ref) its coordinates, or, with `from = <run dir>`,
one start `from` at the best optimum of that run directory's starts.
"""
function start_points(o)
    from = get(o, "from", "")
    isempty(from) && return starts_of(coords_of(o))
    rs = load_results(from)
    isempty(rs) && error("no start in $from has a result")
    return (; from = Float64.(rs[argmax(n -> rs[n]["J"], collect(keys(rs)))]["free"]))
end

"The run's configuration from its options: what every point of a store is run with."
point_config(o) = (; ppl = o["ppl"], ny = o["ny"], float = element_type(o["float"]), backend = opt_backend(o["backend"]), gather = opt_gather(o["gather"]),
                    energy = budget(o["a0E"]), run_setup(o)...)

function run_manifest(name, dir, o; extra_params = Dict{String,Any}())
    params = merge(Dict{String,Any}(o), Dict{String,Any}("box" => Dict("lo" => TRAPPED_BOX.lo, "hi" => TRAPPED_BOX.hi, "scale" => TRAPPED_BOX.scale),
                                                         "trapped" => Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(TCASE)),
                                                         "start_points" => Dict(string(k) => v for (k, v) in pairs(start_points(o))),
                                                         "duration_floor" => coords_of(o).cap === nothing ? "none" :
                                                                             Dict("c" => coords_of(o).cap.c, "p" => coords_of(o).cap.p, "duration_hi" => o["duration_hi"]),
                                                         "window_extend" => window_extend(coords_of(o))), extra_params)
    cuda = o["backend"] == "cuda"
    return run_dir(name, @__FILE__; prefix = "optim", work_package = "WP-068", dir, params,
                   backend = cuda ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                   extra = cuda ? Dict{String,Any}("gpu" => Base.invokelatest(() -> Main.CUDACore.name(Main.CUDACore.device()))) : Dict{String,Any}(),
                   extra_inputs = [joinpath(@__DIR__, "problem.jl"), joinpath(@__DIR__, "trapped2d_case.jl"), joinpath(@__DIR__, "born2d_case.jl"),
                                   joinpath(@__DIR__, "..", "ionization", "atomic.jl")])
end

"""
    main_opt(dir, o)

Every start of `o["starts"]` in turn (or the one start of `from`, [`start_points`](@ref)), with
one store for all of them: the first start's point run first, whose gradient sets the stopping
tolerance, `o["pgtol"]` times its largest component in z.
"""
function main_opt(dir, o)
    cfg = point_config(o)
    d = run_manifest("trapped2d-opt", dir, o)
    store = PointStore(joinpath(d, "points"); run = free -> run_point(free; cfg..., grad = true))
    coords = coords_of(o)
    starts = start_points(o)
    names = isempty(o["from"]) ? split(o["starts"], ",") : ["from"]
    zb = z_of(coords, starts[Symbol(first(names))])
    base = record!(store, free_of(coords, zb))
    gb = grad_z(coords, zb, base["gradient"])
    pgtol = o["pgtol"] * maximum(abs, gb)
    println("base point: J = $(base["J"]) at $(base["free"]), gradient in z $gb, pgtol $pgtol"); flush(stdout)
    for name in names
        sdir = joinpath(d, name)
        isfile(joinpath(sdir, "result.json")) && (println("start $name: done"); continue)
        println("start $name from $(starts[Symbol(name)])"); flush(stdout)
        r = optimize_start(store, starts[Symbol(name)], sdir; maxiters = o["maxiters"], pgtol, coords)
        println("start $name: J = $(r["J"]) at $(r["free"]) (a₀ $(r["a0"])), $(r["retcode"]) after $(r["iterations"]) iterations, ",
                "$(r["points_run"]) points run in $(round(r["seconds"]; digits = 1)) s"); flush(stdout)
    end
    println("optimization done: $d")
    return d
end

"The results of a run directory's starts, by name."
load_results(dir) = Dict(name => JSON.parsefile(joinpath(dir, name, "result.json")) for name in readdir(dir) if isfile(joinpath(dir, name, "result.json")))

"""
    check_jobs(free, o) -> Vector of (id, keywords)

The check at the optimum `free`: plain and with duals at `o["ppl"]`×`o["ny"]` in the run's element
type, and at `o["fdppl"]`×`o["fdny"]` in Float64 a run with duals and central differences at
relative steps 2^-power in each free direction.
"""
function check_jobs(free, o)
    jobs = Tuple{String,Dict{Symbol,Any}}[]
    F = element_type(o["float"])
    push!(jobs, ("fine-plain", Dict{Symbol,Any}(:free => free, :ppl => o["ppl"], :ny => o["ny"], :float => F, :grad => false)))
    push!(jobs, ("fine-grad", Dict{Symbol,Any}(:free => free, :ppl => o["ppl"], :ny => o["ny"], :float => F, :grad => true)))
    push!(jobs, ("f64-grad", Dict{Symbol,Any}(:free => free, :ppl => o["fdppl"], :ny => o["fdny"], :float => Float64, :grad => true)))
    for i in eachindex(free), (s, sgn) in (("p", 1), ("m", -1))
        x = copy(free)
        x[i] += sgn * free[i] * 2.0^-o["power"]
        push!(jobs, ("fd-$(TCASE.names[TCASE.free[i]])-$s", Dict{Symbol,Any}(:free => x, :ppl => o["fdppl"], :ny => o["fdny"], :float => Float64, :grad => false,
                                                                               :direction => i, :sign => sgn, :h => free[i] * 2.0^-o["power"])))
    end
    return jobs
end

function main_check(dir, o)
    results = load_results(dir)
    isempty(results) && error("no start in $dir has a result")
    best = argmax(n -> results[n]["J"], collect(keys(results)))
    free = Float64.(results[best]["free"])
    println("checking the optimum of start $best: J = $(results[best]["J"]) at $free"); flush(stdout)
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))
    run_o = m["parameters"]
    oo = merge(Dict{String,Any}(run_o), o)
    be, ga, energy = opt_backend(oo["backend"]), opt_gather(oo["gather"]), budget(oo["a0E"])
    cdir = joinpath(dir, "check")
    mkpath(cdir)
    open(io -> TOML.print(io, Dict("start" => best, "free" => free, "ppl" => o["ppl"], "ny" => o["ny"], "fdppl" => o["fdppl"], "fdny" => o["fdny"], "power" => o["power"])),
         joinpath(cdir, "check.toml"), "w")
    for (id, j) in check_jobs(free, oo)
        f = joinpath(cdir, id * ".json")
        isfile(f) && continue
        extra = Dict{String,Any}(string(k) => j[k] for k in (:direction, :sign, :h) if haskey(j, k))
        r = run_point(j[:free]; ppl = j[:ppl], ny = j[:ny], float = j[:float], backend = be, gather = ga, grad = j[:grad], energy, run_setup(oo)...)
        merge!(r, extra, Dict{String,Any}("id" => id, "start" => best))
        open(io -> JSON.print(io, r, 2), f, "w")
        println(id, ": J = ", r["J"], " in ", round(r["seconds"]; digits = 1), " s"); flush(stdout)
    end
    println("check done: $cdir")
end

const CHECK_DEFAULTS = (ppl = 48, ny = 3, fdppl = 32, fdny = 2, power = 13)

vdot(a, b) = sum(a .* b)
vnorm(a) = sqrt(vdot(a, a))
tr_of(C) = sum(C[i, i] for i in axes(C, 1)) + eps()

"""
The leading unit eigenvector of the symmetric positive semi-definite matrix `C` orthogonal to the
unit vectors `against`, by power iteration from the column of `C` (projected off `against`) of
largest norm, or, where every such column vanishes, from the coordinate axis that `against` holds
least of; its largest component is positive.
"""
function leading_direction(C, against)
    proj(v) = (w = collect(Float64, v); for a in against; w .-= vdot(w, a) .* a; end; w)
    n = size(C, 1)
    cols = [proj(C[:, k]) for k in 1:n]
    k = argmax(vnorm.(cols))
    v = vnorm(cols[k]) > 1.0e-12 * tr_of(C) ? cols[k] : proj([Float64(i == argmin([sum(abs2(a[i]) for a in against; init = 0.0) for i in 1:n])) for i in 1:n])
    v ./= vnorm(v)
    for _ in 1:1000
        w = proj(C * v .+ 1.0e-3 * tr_of(C) .* v)                # the shift keeps a zero eigenvalue's direction from collapsing
        nw = vnorm(w)
        nw > 0 || break
        w ./= nw
        done = vnorm(w .- v) < 1.0e-13
        v = w
        done && break
    end
    return v[argmax(abs.(v))] < 0 ? -v : v
end

"""
    path_plane(zs, zopt) -> (; origin, e1, e2, s, t, residual)

The plane through `zopt` that holds most of the path `zs` (points in z): the two leading
directions of the path's offsets d from `zopt` (eigenvectors of Σ d·dᵀ, [`leading_direction`](@ref);
a path along one line gets a second direction orthogonal to it), each point's coordinates `s`,
`t` in it, and the largest distance of a point from the plane.
"""
function path_plane(zs, zopt)
    D = [z .- zopt for z in zs]
    C = sum(d * d' for d in D; init = zeros(length(zopt), length(zopt)))
    e1 = leading_direction(C, Vector{Float64}[])
    e2 = leading_direction(C, [e1])
    s = [vdot(z .- zopt, e1) for z in zs]
    t = [vdot(z .- zopt, e2) for z in zs]
    residual = maximum(vnorm(z .- zopt .- s[k] .* e1 .- t[k] .* e2) for (k, z) in enumerate(zs); init = 0.0)
    return (; origin = collect(zopt), e1, e2, s, t, residual)
end

"""
    scan_jobs(dir, o; coords = run_coords(dir)) -> Vector of (id, free, spec)

Plain runs for the figures of J around the paths: for every start, an `o["n"]`×`o["n"]` grid on
the plane of its path ([`path_plane`](@ref)) over the path's extent and a quarter more on each
side (points outside the box left out); and for every pair of free parameters, a grid over the
whole box (the bounds of the run's [`Coords`](@ref)) in the two with the other two at the best
start's optimum.
"""
function scan_jobs(dir, o; coords::Coords = run_coords(dir))
    results = load_results(dir)
    isempty(results) && error("no start in $dir has a result")
    n = o["n"]
    lb, ub = coords.lo, coords.hi
    jobs = Tuple{String,Vector{Float64},Dict{String,Any}}[]
    for name in sort(collect(keys(results)))
        r = results[name]
        zs = [Float64.(p["z"]) for p in r["path"]]
        pl = path_plane(zs, Float64.(r["u"]))
        function span(c)
            lo, hi = minimum(c; init = 0.0), maximum(c; init = 0.0)
            m = max(0.25 * (hi - lo), 0.05)
            return range(lo - m, hi + m; length = n)
        end
        S, Tt = span(pl.s), span(pl.t)
        for (i, a) in enumerate(S), (j, b) in enumerate(Tt)
            z = pl.origin .+ a .* pl.e1 .+ b .* pl.e2
            all(lb .<= z .<= ub) || continue
            push!(jobs, ("plane-$name-$i-$j", free_of(coords, z), Dict{String,Any}("kind" => "plane", "start" => name, "i" => i, "j" => j, "s" => a, "t" => b, "s_grid" => collect(S), "t_grid" => collect(Tt),
                                                                                  "origin" => pl.origin, "e1" => pl.e1, "e2" => pl.e2)))
        end
    end
    best = argmax(k -> results[k]["J"], collect(keys(results)))
    opt = Float64.(results[best]["u"])
    for a in 1:4, b in (a + 1):4, (i, va) in enumerate(range(lb[a], ub[a]; length = n)), (j, vb) in enumerate(range(lb[b], ub[b]; length = n))
        z = copy(opt)
        z[a], z[b] = va, vb
        push!(jobs, ("pair-$a$b-$i-$j", free_of(coords, z), Dict{String,Any}("kind" => "pair", "start" => best, "a" => a, "b" => b, "i" => i, "j" => j)))
    end
    return jobs
end

"""
    main_scan(dir, o)

The plain runs of [`scan_jobs`](@ref) into `dir/scan/<id>.json`, in the run's numerics; with
`shard = "k/m"` only every m-th job from the k-th (k from 1), so that m processes share the card.
A job with a JSON is not run again.
"""
function main_scan(dir, o)
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))["parameters"]
    be, ga, F, energy = opt_backend(m["backend"]), opt_gather(m["gather"]), element_type(m["float"]), budget(m["a0E"])
    k, of = parse.(Int, split(o["shard"], "/"))
    jobs = scan_jobs(dir, o)
    sdir = joinpath(dir, "scan")
    mkpath(sdir)
    mine = [j for (q, j) in enumerate(jobs) if mod(q - 1, of) == k - 1 && !isfile(joinpath(sdir, j[1] * ".json"))]
    println("scan: $(length(jobs)) jobs, $(length(mine)) to run in shard $k of $of"); flush(stdout)
    for (q, (id, free, spec)) in enumerate(mine)
        r = run_point(free; ppl = m["ppl"], ny = m["ny"], float = F, backend = be, gather = ga, grad = false, energy, run_setup(m)...)
        merge!(r, spec, Dict{String,Any}("id" => id))
        tmp = joinpath(sdir, id * ".json.tmp")
        open(io -> JSON.print(io, r, 2), tmp, "w")
        mv(tmp, joinpath(sdir, id * ".json"); force = true)
        (q % 20 == 0 || q == length(mine)) && (println("scan shard $k: $q of $(length(mine)), last J = $(r["J"]) in $(round(r["seconds"]; digits = 2)) s"); flush(stdout))
    end
end

const SCAN_DEFAULTS = (n = 11, shard = "1/1")

# the driver's commands run only when this file is the script, not when another driver includes it (pulse2d_opt.jl)
if abspath(PROGRAM_FILE) == (@__FILE__) && length(ARGS) >= 1 && ARGS[1] in ("opt", "check", "scan")
    # the backend loads at top level, so that the runs that follow see its methods (world age)
    ARGS[1] == "opt" && parse_options(ARGS[2:end], OPT_DEFAULTS)[1]["backend"] == "cuda" && opt_backend("cuda")
    if ARGS[1] in ("check", "scan")
        m = TOML.parsefile(joinpath(ARGS[2], "manifest.toml"))
        m["parameters"]["backend"] == "cuda" && opt_backend("cuda")
    end
end

if abspath(PROGRAM_FILE) != (@__FILE__)
elseif length(ARGS) >= 1 && ARGS[1] == "opt"
    o, rest = parse_options(ARGS[2:end], OPT_DEFAULTS)
    main_opt(isempty(rest) ? nothing : rest[1], o)
elseif length(ARGS) >= 2 && ARGS[1] == "check"
    o, _ = parse_options(ARGS[3:end], CHECK_DEFAULTS)
    main_check(ARGS[2], o)
elseif length(ARGS) >= 2 && ARGS[1] == "scan"
    o, _ = parse_options(ARGS[3:end], SCAN_DEFAULTS)
    main_scan(ARGS[2], o)
elseif length(ARGS) >= 3 && ARGS[1] == "report"
    include(joinpath(@__DIR__, "trapped2d_opt_report.jl"))
    opt_report(ARGS[2], ARGS[3])
elseif !isempty(ARGS)
    error("usage: trapped2d_opt.jl opt [dir] [key=value...] | check <dir> [key=value...] | scan <dir> [n=11 shard=k/m] | report <out dir> <run dir>")
end
