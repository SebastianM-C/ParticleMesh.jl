# WP-087, the 2D pilot: L-BFGS-B over the pulse's shape (pulse2d_case.jl) at a fixed plasma and a
# fixed laser energy, from several random starts, maximizing WP-065's trapped-electron objective.
# The five parameters are the focus, the Legendre modes P₃ and P₄ of the wavefront (waves), the
# GDD and the TOD, in `PULSE_BOX`.
#
#   julia +1.13 --project=scripts -t 2 scripts/optim/pulse2d_opt.jl opt [dir] [key=value ...]
#       the reference point (the unaberrated, transform-limited pulse, `PCASE.x0`) and then every
#       start of `starts=` into <dir>/<start>/, the points into <dir>/points/; keys: ppl=32 ny=2
#       (Δx = λ₀/ppl, Δy = π/(4·ny)), float=32, backend=cuda (or cpu), gather=compensated (or
#       plain), maxiters=25, pgtol=<rel> (the stopping tolerance on the projected gradient in z,
#       relative to the largest component of the reference point's gradient; default 2⁻⁶),
#       starts=r1,r2,r3,r4,r5 (random starts, `random_start`), seed=87
#   julia +1.13 --project=scripts -t 2 scripts/optim/pulse2d_opt.jl check <dir> [fdppl=32 fdny=2 power=13]
#       at the best optimum of <dir>'s starts and at the reference point: a Float64 run with duals
#       and Float64 central differences at a relative step 2^-power of each parameter's scale in
#       the five directions, at fdppl×fdny, into <dir>/check/
#   julia +1.13 --project=scripts scripts/optim/pulse2d_opt.jl report <out dir> <run dir>
#       the figures (pulse2d_paths.png, pulse2d_shapes.png), each with its JSON sidecar, and
#       tables.md, into <out dir> (pulse2d_opt_report.jl)
#
# Every point the optimizer asks for is one run with duals in the five directions, recorded as a
# JSON file named by the point's bits (WP-068's `PointStore`): a stopped optimization replays its
# path from the records and continues.

isdefined(@__MODULE__, :PointStore) || include(joinpath(@__DIR__, "trapped2d_opt.jl"))
isdefined(@__MODULE__, :PCASE) || include(joinpath(@__DIR__, "pulse2d_case.jl"))

"""
    PulseCoords()

The solver's coordinates over the five parameters: z = x ./ `PULSE_BOX.scale`, with the box's
bounds. [`free_of`](@ref) and [`z_of`](@ref) map between them, as WP-068's `Coords` do, so that its
`optimize_start` runs on this case unchanged.
"""
struct PulseCoords
    lo::Vector{Float64}
    hi::Vector{Float64}
end
PulseCoords() = PulseCoords(PULSE_BOX.lo ./ PULSE_BOX.scale, PULSE_BOX.hi ./ PULSE_BOX.scale)
free_of(::PulseCoords, z) = collect(z) .* PULSE_BOX.scale
z_of(::PulseCoords, free) = collect(Float64, free) ./ PULSE_BOX.scale

"SplitMix64's output function: a counter-based hash of a 64-bit word, the same on every machine and Julia version."
function splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

"A number in [0, 1) from the counter (`seed`, `k`, `i`): the top 53 bits of [`splitmix64`](@ref)."
uniform01(seed::Integer, k::Integer, i::Integer) =
    (splitmix64(splitmix64(splitmix64(UInt64(seed)) ⊻ UInt64(k)) ⊻ UInt64(i)) >> 11) * 2.0^-53

"""
    random_start(k; seed = 87, margin = 0.1) -> x

The k-th random start: every parameter uniform over `PULSE_BOX` with `margin` of its width left
free at each side, from the counter-based [`uniform01`](@ref), so the starts are the same on
every machine. Every mode is away from zero (with probability one), so the modes that are odd
under the mirror y → −y start with a gradient (`questions/wp-087-pulse-shape-study.md` §4).
"""
function random_start(k::Integer; seed::Integer = 87, margin::Real = 0.1)
    lo, hi = PULSE_BOX.lo, PULSE_BOX.hi
    return [lo[i] + (margin + (1 - 2margin) * uniform01(seed, k, i)) * (hi[i] - lo[i]) for i in eachindex(lo)]
end

"The starts named `r<k>` in `names` (comma-separated), as a named tuple of [`random_start`](@ref)s."
function pulse_starts(names::AbstractString; seed::Integer = 87)
    ks = split(names, ",")
    all(n -> occursin(r"^r\d+$", n), ks) || error("the starts are r1, r2, …; got $names")
    return NamedTuple{Tuple(Symbol.(ks))}(Tuple(random_start(parse(Int, n[2:end]); seed) for n in ks))
end

"""
    pulse_run_point(x; ppl, ny, float, backend, gather, grad, energy, read, kw...) -> Dict

One run of the case at the five parameters `x` on `backend` in `float` with `gather`: J (WP-065's
`trapped_energy`), its gradient in the five parameters (physical units) when `grad` (duals in
the five directions; J is the plain run's to roundoff, T-1185), a₀, the vacuum focus's Strehl
ratio, the share of the profile's power the antenna launches ([`launched_fraction`](@ref)), the
envelope's centroid and duration, the run's seconds and [`trapped_summary`](@ref). `read` holds
`fade` and `rise` for J and the summary; the other keywords go to the case's problem.
"""
function pulse_run_point(x; ppl::Integer, ny::Integer, float::Type{<:AbstractFloat} = Float64, backend = CPU(), gather = nothing,
                         grad::Bool = true, energy = budget(PCASE.a0_energy), read = (;), kw...)
    F = float
    n = length(x)
    arg(i) = grad ? ForwardDiff.Dual{Nothing}(F(x[i]), ntuple(j -> F(j == i), n)) : F(x[i])
    xs = [arg(i) for i in 1:n]
    t = @elapsed st = pulse_state(xs; energy = F(energy), ppl, ny, float = F, backend, gather, kw...)
    J = trapped_energy(st; read...)
    shape = pulse_shape(collect(Float64, x))
    prob = pulse_problem(collect(Float64, x); energy = Float64(energy), ppl, ny)
    out = Dict{String,Any}(string(k) => v for (k, v) in pairs(trapped_summary(st; read...)))
    out["J"] = Float64(value_of(J))
    out["free"] = collect(Float64, x)
    out["z"] = collect(Float64, x) ./ PULSE_BOX.scale
    out["a0"] = pulse_a0(shape; energy = Float64(energy))
    out["strehl"] = strehl(shape)
    lf = launched_fraction(prob)
    out["launched_box"], out["launched_interior"] = lf.box, lf.interior
    c, h = PulseShapes.envelope_extent(shape.envelope)
    out["envelope_centroid"], out["envelope_duration"] = c, h
    out["gradient"] = grad ? collect(Float64, ForwardDiff.partials(J)) : nothing
    out["seconds"] = t
    out["eltype"] = string(F)
    out["ppl"], out["ny"] = ppl, ny
    out["gather"] = gather === nothing ? "plain" : "compensated"
    out["cells"] = collect(ncells(st.grid))
    return out
end

const PULSE_OPT_DEFAULTS = (ppl = 32, ny = 2, float = 32, backend = "cuda", gather = "compensated", maxiters = 25, pgtol = 2.0^-6,
                            starts = "r1,r2,r3,r4,r5", seed = 87)

"The run's configuration from its options (or manifest parameters): what every point of a store is run with."
pulse_config(o) = (; ppl = o["ppl"], ny = o["ny"], float = element_type(o["float"]), backend = opt_backend(o["backend"]), gather = opt_gather(o["gather"]))

function pulse_manifest(dir, o)
    starts = pulse_starts(o["starts"]; seed = o["seed"])
    params = merge(Dict{String,Any}(o), Dict{String,Any}("box" => Dict("lo" => PULSE_BOX.lo, "hi" => PULSE_BOX.hi, "scale" => PULSE_BOX.scale),
                                                         "pulse" => Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(PCASE)),
                                                         "na" => pulse_na(), "energy" => budget(PCASE.a0_energy),
                                                         "trapped" => Dict{String,Any}(string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(TCASE)),
                                                         "start_points" => Dict(string(k) => v for (k, v) in pairs(starts))))
    cuda = o["backend"] == "cuda"
    return run_dir("pulse2d-opt", @__FILE__; prefix = "optim", work_package = "WP-087", dir, params,
                   backend = cuda ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                   extra = cuda ? Dict{String,Any}("gpu" => Base.invokelatest(() -> Main.CUDACore.name(Main.CUDACore.device()))) : Dict{String,Any}(),
                   extra_inputs = [joinpath(@__DIR__, f) for f in ("problem.jl", "trapped2d_opt.jl", "pulse2d_case.jl", "trapped2d_case.jl", "born2d_case.jl")] ∪
                                  [joinpath(@__DIR__, "..", "ionization", "atomic.jl")])
end

"""
    main_pulse_opt(dir, o)

The reference point first (the unaberrated, transform-limited pulse), whose gradient sets the
stopping tolerance, `o["pgtol"]` times its largest component in z; then every start of
`o["starts"]` in turn, with one store for all of them.
"""
function main_pulse_opt(dir, o)
    cfg = pulse_config(o)
    d = pulse_manifest(dir, o)
    store = PointStore(joinpath(d, "points"); run = x -> pulse_run_point(x; cfg..., grad = true))
    coords = PulseCoords()
    ref = record!(store, collect(PCASE.x0))
    gref = grad_z(coords, z_of(coords, PCASE.x0), ref["gradient"])
    pgtol = o["pgtol"] * maximum(abs, gref)
    println("reference point: J = $(ref["J"]), gradient in z $gref, pgtol $pgtol"); flush(stdout)
    starts = pulse_starts(o["starts"]; seed = o["seed"])
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

"""
    pulse_check_jobs(points, o) -> Vector of (id, keywords)

At each of `points` (name => parameters): a Float64 run with duals and Float64 central differences
in each direction at the step 2^-power of the parameter's scale (the modes and the spectral phase
are near zero, so a step relative to the value would vanish), at `o["fdppl"]`×`o["fdny"]`.
"""
function pulse_check_jobs(points, o)
    jobs = Tuple{String,Dict{Symbol,Any}}[]
    for (name, x) in points
        push!(jobs, ("$name-f64-grad", Dict{Symbol,Any}(:x => x, :grad => true)))
        for i in eachindex(x), (s, sgn) in (("p", 1), ("m", -1))
            h = PULSE_BOX.scale[i] * 2.0^-o["power"]
            y = copy(x)
            y[i] += sgn * h
            push!(jobs, ("$name-fd-$(PCASE.names[i])-$s", Dict{Symbol,Any}(:x => y, :grad => false, :direction => i, :sign => sgn, :h => h, :point => name)))
        end
    end
    return jobs
end

function main_pulse_check(dir, o)
    results = load_results(dir)
    isempty(results) && error("no start in $dir has a result")
    best = argmax(n -> results[n]["J"], collect(keys(results)))
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))["parameters"]
    cfg = pulse_config(m)
    cdir = joinpath(dir, "check")
    mkpath(cdir)
    points = ["best" => Float64.(results[best]["free"]), "reference" => collect(PCASE.x0)]
    open(io -> TOML.print(io, Dict("start" => best, "best" => points[1][2], "fdppl" => o["fdppl"], "fdny" => o["fdny"], "power" => o["power"])),
         joinpath(cdir, "check.toml"), "w")
    for (id, j) in pulse_check_jobs(points, o)
        f = joinpath(cdir, id * ".json")
        isfile(f) && continue
        r = pulse_run_point(j[:x]; ppl = o["fdppl"], ny = o["fdny"], float = Float64, backend = cfg.backend, gather = cfg.gather, grad = j[:grad])
        merge!(r, Dict{String,Any}(string(k) => v for (k, v) in j if k in (:direction, :sign, :h, :point)), Dict{String,Any}("id" => id, "start" => best))
        open(io -> JSON.print(io, r, 2), f, "w")
        println(id, ": J = ", r["J"], " in ", round(r["seconds"]; digits = 1), " s"); flush(stdout)
    end
    println("check done: $cdir")
end

const PULSE_CHECK_DEFAULTS = (fdppl = 32, fdny = 2, power = 13)

const PULSE_USAGE = "usage: pulse2d_opt.jl opt [dir] [key=value...] | check <dir> [key=value...] | report <out dir> <run dir>"
abspath(PROGRAM_FILE) == (@__FILE__) && isempty(ARGS) && error(PULSE_USAGE)

if abspath(PROGRAM_FILE) == (@__FILE__)
    # the backend loads at top level, so that the runs that follow see its methods (world age)
    ARGS[1] == "opt" && parse_options(ARGS[2:end], PULSE_OPT_DEFAULTS)[1]["backend"] == "cuda" && opt_backend("cuda")
    ARGS[1] == "check" && TOML.parsefile(joinpath(ARGS[2], "manifest.toml"))["parameters"]["backend"] == "cuda" && opt_backend("cuda")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    if ARGS[1] == "opt"
        o, rest = parse_options(ARGS[2:end], PULSE_OPT_DEFAULTS)
        main_pulse_opt(isempty(rest) ? nothing : rest[1], o)
    elseif ARGS[1] == "check"
        o, _ = parse_options(ARGS[3:end], PULSE_CHECK_DEFAULTS)
        main_pulse_check(ARGS[2], o)
    elseif ARGS[1] == "report"
        include(joinpath(@__DIR__, "pulse2d_opt_report.jl"))
        pulse_report(ARGS[2], ARGS[3])
    else
        error(PULSE_USAGE)
    end
end
