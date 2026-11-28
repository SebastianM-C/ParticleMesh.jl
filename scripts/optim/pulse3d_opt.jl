# WP-087, the 3D half: L-BFGS-B over the pulse's shape (pulse3d_case.jl) at a fixed plasma and a
# fixed laser energy, from several random starts, maximizing the 3D case's trapped-electron
# objective. The fourteen parameters are the focus, eleven Zernike modes of the wavefront (OSA
# j = 3, 5 to 14, waves rms), the GDD and the TOD, in `PULSE3_BOX`.
#
#   julia +1.13 --project=scripts -t 2 scripts/optim/pulse3d_opt.jl opt [dir] [key=value ...]
#       the reference point (the unaberrated, transform-limited pulse, `PCASE3.x0`) and then every
#       start of `starts=` into <dir>/<start>/, the points into <dir>/points/; keys: ppl=16 nt=1
#       (Δx = λ₀/ppl, Δy = Δz = π/(4·nt)), float=32, backend=cuda (or cpu), chunk=4 (partials per
#       run with duals: the gradient takes ⌈14/chunk⌉ of them), maxiters=20, pgtol=<rel> (the
#       stopping tolerance on the projected gradient in z, relative to the largest component of
#       the reference point's gradient; default 2⁻⁶), starts=r1,r2,r3,r4, seed=873,
#       spread=0.05 (the modes' start, uniform in ±spread waves)
#   julia +1.13 --project=scripts -t 2 scripts/optim/pulse3d_opt.jl check <dir> [steps=5,7 directions=3]
#       at the best optimum of <dir>'s starts and at the reference point: central differences of
#       plain runs on the optimization's numerics along the gradient's direction in z and along
#       `directions − 1` random unit directions, at the steps 2^-s in z for s in `steps`, into
#       <dir>/check/, against the recorded gradient's projection
#
# Every point the optimizer asks for is ⌈14/chunk⌉ runs with duals (the same plain values, the
# partials of `chunk` parameters each), recorded as one JSON file named by the point's bits
# (WP-068's `PointStore`): a stopped optimization replays its path from the records and continues.

isdefined(@__MODULE__, :PointStore) || include(joinpath(@__DIR__, "trapped2d_opt.jl"))
isdefined(@__MODULE__, :splitmix64) || include(joinpath(@__DIR__, "pulse2d_opt.jl"))
isdefined(@__MODULE__, :PCASE3) || include(joinpath(@__DIR__, "pulse3d_case.jl"))

"""
    Pulse3Coords()

The solver's coordinates over the fourteen parameters: z = x ./ `PULSE3_BOX.scale`, with the box's
bounds, so that WP-068's `optimize_start` runs on this case unchanged.
"""
struct Pulse3Coords
    lo::Vector{Float64}
    hi::Vector{Float64}
end
Pulse3Coords() = Pulse3Coords(PULSE3_BOX.lo ./ PULSE3_BOX.scale, PULSE3_BOX.hi ./ PULSE3_BOX.scale)
free_of(::Pulse3Coords, z) = collect(z) .* PULSE3_BOX.scale
z_of(::Pulse3Coords, free) = collect(Float64, free) ./ PULSE3_BOX.scale

"""
    random_start3(k; seed = 873, spread = 0.05, margin = 0.1) -> x

The k-th random start: the focus, the GDD and the TOD uniform over `PULSE3_BOX` with `margin` of
their width left free at each side, and every Zernike mode uniform in ±`spread` waves, from the
counter-based [`uniform01`](@ref), so the starts are the same on every machine. The modes start
small: eleven modes drawn over their whole box would be a wavefront of 0.19 waves rms on average
(a Strehl ratio near 0.25), where little is trapped; at ±0.05 it is about 0.1 waves, a Strehl ratio
near 0.7 (`questions/wp-087-pulse-shape-study.md` §7). Every mode is away from zero (with
probability one), so the modes odd under the mirror z → −z start with a gradient.
"""
function random_start3(k::Integer; seed::Integer = 873, spread::Real = 0.05, margin::Real = 0.1)
    lo, hi = PULSE3_BOX.lo, PULSE3_BOX.hi
    m = length(PCASE3.modes)
    u(i) = uniform01(seed, k, i)
    x = [lo[i] + (margin + (1 - 2margin) * u(i)) * (hi[i] - lo[i]) for i in eachindex(lo)]
    for i in 2:(1 + m)
        x[i] = spread * (2 * u(i) - 1)
    end
    return x
end

"The starts named `r<k>` in `names` (comma-separated), as a named tuple of [`random_start3`](@ref)s."
function pulse3_starts(names::AbstractString; seed::Integer = 873, spread::Real = 0.05)
    ks = split(names, ",")
    all(n -> occursin(r"^r\d+$", n), ks) || error("the starts are r1, r2, …; got $names")
    return NamedTuple{Tuple(Symbol.(ks))}(Tuple(random_start3(parse(Int, n[2:end]); seed, spread) for n in ks))
end

"The directions of the gradient in chunks of at most `chunk`, in order: `1:chunk`, `chunk+1:2chunk`, …"
gradient_chunks(n::Integer, chunk::Integer) = [i:min(i + chunk - 1, n) for i in 1:chunk:n]

"""
    pulse3_run_point(x; ppl, nt, float, backend, gather, grad, chunk, energy, read, kw...) -> Dict

One point of the case at the fourteen parameters `x` on `backend` in `float`: J (the 3D case's
`trapped_energy3`), its gradient in the fourteen parameters (physical units) when `grad`, from
one run with duals per chunk of `chunk` of the `directions` ([`gradient_chunks`](@ref); the
partials of the others are left zero, which only the tests use; J is the first run's
value, and `J_chunks` every run's, which agree to roundoff), or one plain run; a₀, the vacuum
focus's Strehl ratio, the share of the energy the antenna launches ([`launched_share3`](@ref)),
the envelope's centroid and duration, the seconds and [`trapped_summary`](@ref). `read` holds
`fade` and `rise`; the other keywords go to [`pulse3_problem`](@ref).
"""
function pulse3_run_point(x; ppl::Integer, nt::Integer = 1, float::Type{<:AbstractFloat} = Float32, backend = CPU(),
                          gather = CompensatedGather(; k = 1), grad::Bool = true, chunk::Integer = 4, directions = 1:pulse3_nparams(),
                          energy = budget3(PCASE3.a0_energy), read = (;), kw...)
    F = float
    n = length(x)
    n == pulse3_nparams() || throw(ArgumentError("pulse3_run_point takes the $(pulse3_nparams()) parameters, got $n"))
    dirs = collect(directions)
    runs = grad ? [dirs[c] for c in gradient_chunks(length(dirs), chunk)] : [Int[]]
    Js, gradient, seconds = Float64[], zeros(n), Float64[]
    summary, launched, cells, state_eltype = nothing, nothing, nothing, ""
    for c in runs
        arg(i) = grad ? ForwardDiff.Dual{Nothing}(F(x[i]), ntuple(j -> F(c[j] == i), length(c))) : F(x[i])
        xs = [arg(i) for i in 1:n]
        t = @elapsed begin
            st, p, prob = pulse3_start(xs; energy = F(energy), ppl, nt, float = F, backend, gather, kw...)
            launched === nothing && (launched = launched_share3(p.lasers[1]))
            run_to_end((st, p, prob))
            J = trapped_energy3(st; read...)
        end
        push!(Js, Float64(value_of(J)))
        grad && (gradient[c] .= Float64.(ForwardDiff.partials(J)))
        push!(seconds, t)
        summary === nothing && (summary = trapped_summary3(st; read...); state_eltype = string(eltype(st.ρ)))
        cells = collect(ncells(st.grid))
        st = nothing
        GC.gc()
    end
    shape = pulse3_shape(collect(Float64, x))
    out = Dict{String,Any}(string(k) => v for (k, v) in pairs(summary))
    out["J"] = Js[1]
    out["J_chunks"] = Js
    out["free"] = collect(Float64, x)
    out["z"] = collect(Float64, x) ./ PULSE3_BOX.scale
    out["a0"] = Float64(pulse3_a0(shape; energy = Float64(energy)))
    out["strehl"] = strehl3(shape)
    out["launched_box"], out["launched_interior"] = launched.box, launched.interior
    ce, he = PulseShapes.envelope_extent(shape.envelope)
    out["envelope_centroid"], out["envelope_duration"] = ce, he
    out["gradient"] = grad ? gradient : nothing
    out["chunk"] = grad ? chunk : 0
    out["directions"] = grad ? dirs : Int[]
    out["seconds"] = sum(seconds)
    out["seconds_runs"] = seconds
    out["eltype"] = string(F)
    out["state_eltype"] = state_eltype
    out["ppl"], out["nt"] = ppl, nt
    out["cells"] = cells
    return out
end

"""
    trapped_summary3(st; fade, rise) -> NamedTuple

[`trapped_summary`](@ref) of a 3D state: its sums over the cell's area times Δz, the cell's volume
(J, its parts and the weights and energies of the counted, window and dark electrons); the means,
the largest u_x and the particle count as they are.
"""
function trapped_summary3(st; fade = TCASE.fade, rise = TCASE.rise)
    s = trapped_summary(st; fade, rise)
    hz = Float64(value_of(cellsize(st.grid)[3]))
    scaled = (:J, :J_born, :J_plasma, :weight_born, :weight_plasma, :fast_born_window, :fast_plasma_window, :dark_weight, :dark_energy)
    return NamedTuple{keys(s)}(map(k -> k in scaled ? hz * s[k] : s[k], keys(s)))
end

const PULSE3_OPT_DEFAULTS = (ppl = 16, nt = 1, float = 32, backend = "cuda", chunk = 4, maxiters = 20, pgtol = 2.0^-6, starts = "r1,r2,r3,r4",
                             seed = 873, spread = 0.05)

"The run's configuration from its options (or manifest parameters): what every point of a store is run with."
pulse3_config(o) = (; ppl = o["ppl"], nt = o["nt"], float = element_type(o["float"]), backend = opt_backend(o["backend"]), chunk = o["chunk"])

nt_dict(t) = Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(t))

function pulse3_manifest(dir, o)
    starts = pulse3_starts(o["starts"]; seed = o["seed"], spread = o["spread"])
    params = merge(Dict{String,Any}(o), Dict{String,Any}("box" => Dict("lo" => PULSE3_BOX.lo, "hi" => PULSE3_BOX.hi, "scale" => PULSE3_BOX.scale),
                                                         "pulse" => nt_dict(PCASE3), "na" => pulse3_na(), "energy" => budget3(PCASE3.a0_energy),
                                                         "gather" => "CompensatedGather(k = 1)", "solver" => "Yee", "mode" => "Reference",
                                                         "trapped" => nt_dict(TCASE), "start_points" => Dict(string(k) => v for (k, v) in pairs(starts))))
    cuda = o["backend"] == "cuda"
    return run_dir("pulse3d-opt", @__FILE__; prefix = "optim", work_package = "WP-087", dir, params,
                   backend = cuda ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                   extra = cuda ? Dict{String,Any}("gpu" => Base.invokelatest(() -> Main.CUDACore.name(Main.CUDACore.device()))) : Dict{String,Any}(),
                   extra_inputs = [joinpath(@__DIR__, f) for f in ("problem.jl", "trapped2d_opt.jl", "pulse2d_opt.jl", "pulse3d_case.jl", "trapped3d_case.jl",
                                                                   "trapped2d_case.jl", "born2d_case.jl")] ∪
                                  [joinpath(@__DIR__, "..", "ionization", "atomic.jl")])
end

"""
    main_pulse3_opt(dir, o)

The reference point first (the unaberrated, transform-limited pulse), whose gradient sets the
stopping tolerance, `o["pgtol"]` times its largest component in z; then every start of
`o["starts"]` in turn, with one store for all of them.
"""
function main_pulse3_opt(dir, o)
    cfg = pulse3_config(o)
    d = pulse3_manifest(dir, o)
    store = PointStore(joinpath(d, "points"); run = x -> pulse3_run_point(x; cfg..., grad = true))
    coords = Pulse3Coords()
    ref = record!(store, collect(PCASE3.x0))
    gref = grad_z(coords, z_of(coords, PCASE3.x0), ref["gradient"])
    pgtol = o["pgtol"] * maximum(abs, gref)
    println("reference point: J = $(ref["J"]), gradient in z $gref, pgtol $pgtol"); flush(stdout)
    starts = pulse3_starts(o["starts"]; seed = o["seed"], spread = o["spread"])
    for name in keys(starts)
        sdir = joinpath(d, string(name))
        isfile(joinpath(sdir, "result.json")) && (println("start $name: done"); continue)
        println("start $name from $(starts[name])"); flush(stdout)
        r = optimize_start(store, starts[name], sdir; maxiters = o["maxiters"], pgtol, coords)
        println("start $name: J = $(r["J"]) at $(r["free"]), $(r["retcode"]) after $(r["iterations"]) iterations, ",
                "$(r["points_run"]) points run in $(round(r["seconds"]; digits = 1)) s"); flush(stdout)
    end
    println("optimization done: $d")
    return d
end

"A unit vector in the fourteen z directions from the counter (`seed`, `k`): normalized uniform numbers in [−1, 1)."
function random_direction(k::Integer; seed::Integer = 8731)
    v = [2 * uniform01(seed, k, i) - 1 for i in 1:pulse3_nparams()]
    return v ./ sqrt(sum(abs2, v))
end

"""
    pulse3_check_jobs(points, o) -> Vector of (id, Dict)

At each of `points` (name => (x, gradient in physical units)): plain runs at x ± 2^-s·d (in z) for
every s of `o["steps"]` along the unit direction d of the gradient in z and along
`o["directions"] − 1` random unit directions ([`random_direction`](@ref)), each with the recorded
gradient's projection dJ/dz·d to compare with the central difference.
"""
function pulse3_check_jobs(points, o)
    coords = Pulse3Coords()
    steps = parse.(Int, split(o["steps"], ","))
    jobs = Tuple{String,Dict{Symbol,Any}}[]
    for (name, (x, g)) in points
        z = z_of(coords, x)
        gz = grad_z(coords, z, g)
        dirs = [gz ./ sqrt(sum(abs2, gz))]
        append!(dirs, [random_direction(k) for k in 1:(o["directions"] - 1)])
        for (q, d) in enumerate(dirs), s in steps, (sfx, sgn) in (("p", 1), ("m", -1))
            h = 2.0^-s
            y = free_of(coords, z .+ sgn * h .* d)
            push!(jobs, ("$name-d$q-s$s-$sfx", Dict{Symbol,Any}(:x => y, :point => name, :direction => q, :step => h, :sign => sgn, :d => d,
                                                                  :projection => sum(gz .* d))))
        end
    end
    return jobs
end

function main_pulse3_check(dir, o)
    results = load_results(dir)
    isempty(results) && error("no start in $dir has a result")
    best = argmax(n -> results[n]["J"], collect(keys(results)))
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))["parameters"]
    cfg = pulse3_config(m)
    store = PointStore(joinpath(dir, "points"); run = x -> error("the check reads the optimization's records; the point $x has none"))
    cdir = joinpath(dir, "check")
    mkpath(cdir)
    xb = Float64.(results[best]["free"])
    points = ["best" => (xb, record!(store, xb)["gradient"]), "reference" => (collect(PCASE3.x0), record!(store, collect(PCASE3.x0))["gradient"])]
    open(io -> TOML.print(io, Dict("start" => best, "best" => xb, "steps" => o["steps"], "directions" => o["directions"])), joinpath(cdir, "check.toml"), "w")
    for (id, j) in pulse3_check_jobs(points, o)
        f = joinpath(cdir, id * ".json")
        isfile(f) && continue
        r = pulse3_run_point(j[:x]; cfg..., grad = false)
        merge!(r, Dict{String,Any}(string(k) => v for (k, v) in j if k != :x), Dict{String,Any}("id" => id, "start" => best))
        open(io -> JSON.print(io, r, 2), f, "w")
        println(id, ": J = ", r["J"], " in ", round(r["seconds"]; digits = 1), " s"); flush(stdout)
    end
    println("check done: $cdir")
end

"""
    pulse3_check_rows(cdir, point) -> Vector of Dict

The check's central differences at `point` ("best" or "reference"): per direction and step, the
central difference (J(+) − J(−))/2h in z, the recorded gradient's projection and their relative
difference.
"""
function pulse3_check_rows(cdir, point)
    recs = [JSON.parsefile(joinpath(cdir, f)) for f in readdir(cdir) if endswith(f, ".json")]
    recs = filter(r -> get(r, "point", "") == point, recs)
    rows = Dict{String,Any}[]
    for q in sort(unique(r["direction"] for r in recs)), h in sort(unique(r["step"] for r in recs if r["direction"] == q); rev = true)
        p = filter(r -> r["direction"] == q && r["step"] == h, recs)
        jp = only(r["J"] for r in p if r["sign"] == 1)
        jm = only(r["J"] for r in p if r["sign"] == -1)
        fd = (jp - jm) / 2h
        ad = only(unique(r["projection"] for r in p))
        push!(rows, Dict{String,Any}("direction" => q, "step" => h, "fd" => fd, "ad" => ad, "relative" => abs(fd - ad) / abs(ad)))
    end
    return rows
end

const PULSE3_CHECK_DEFAULTS = (steps = "5,7", directions = 3)

const PULSE3_USAGE = "usage: pulse3d_opt.jl opt [dir] [key=value...] | check <dir> [key=value...]"
abspath(PROGRAM_FILE) == (@__FILE__) && isempty(ARGS) && error(PULSE3_USAGE)

if abspath(PROGRAM_FILE) == (@__FILE__)
    # the backend loads at top level, so that the runs that follow see its methods (world age)
    ARGS[1] == "opt" && parse_options(ARGS[2:end], PULSE3_OPT_DEFAULTS)[1]["backend"] == "cuda" && opt_backend("cuda")
    ARGS[1] == "check" && TOML.parsefile(joinpath(ARGS[2], "manifest.toml"))["parameters"]["backend"] == "cuda" && opt_backend("cuda")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    if ARGS[1] == "opt"
        o, rest = parse_options(ARGS[2:end], PULSE3_OPT_DEFAULTS)
        main_pulse3_opt(isempty(rest) ? nothing : rest[1], o)
    elseif ARGS[1] == "check"
        o, _ = parse_options(ARGS[3:end], PULSE3_CHECK_DEFAULTS)
        main_pulse3_check(ARGS[2], o)
    else
        error(PULSE3_USAGE)
    end
end
