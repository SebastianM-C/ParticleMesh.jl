# Resolution scans of the cold cross-code cases (WP-031): is the common offset of a measured
# growth rate from linear theory discretization error?
#
#   julia +1.13 --project=scripts scripts/crosscode/convergence.jl prepare <scan.toml>
#   julia +1.13 --project=scripts scripts/crosscode/convergence.jl summarize scripts/runs/<scan>
#   julia +1.13 --project=scripts scripts/crosscode/convergence.jl report <summary dir> <out dir> [box-status.toml]
#   julia +1.13 --project=scripts scripts/crosscode/convergence.jl compare <reference summary> <other summary> <out dir>
#
# `prepare` reads a scan file (scripts/crosscode/scans/*.toml: a case and a list of points) and
# writes, under scripts/runs/<scan>/, one derived case file per point (summary/cases/<point>.toml,
# by `derive_case`), one Smilei run directory per point (smilei.jl's `prepare_run`) and the CPU
# box request (box-request.toml: a smilei step per point, then the summarize step; only the
# summary directory comes back; `category = "validation"` for the box's cost ledger). `summarize` reads each finished point's scalars and field
# dumps and writes summary/<point>.json: the series of the case's observables (the field energy
# and the complex mode coefficient), the wall time and the output hashes, so that the dumps
# need not leave the box. `report` evaluates the pre-registered window fit on every point, the
# same fit on the exact linear cold-fluid solution at the same sample times (theory.jl,
# `cold_ivp`), and writes results.json, the figures with their JSON sidecars and a manifest.
#
# A point changes the case in named ways only (T-412): `refine` multiplies the cells per axis
# and divides Δt by the same factor (the Courant number is kept); `dt_divisor` divides Δt
# further; the output cadence in steps is multiplied by both, so the output times stay those
# of the case; `amplitude_factor` scales every perturbation amplitude, `ppc_factor` the
# particles per cell per axis, `seed` replaces the random seed, `fields` the dumped fields.

module Convergence

using TOML
using JSON
using Statistics: mean

include(joinpath(@__DIR__, "smilei.jl"))
include(joinpath(@__DIR__, "read.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "theory.jl"))
using .SmileiDeck, .SmileiRead, .Observables, .Theory
using .SmileiDeck.RunManifest: write_manifest, file_hash, git_commit, stamp_interval

export derive_case, prepare_scan, summarize_scan, point_fits, local_rate, ivp_series, observable_keys

const CROSSCODE = @__DIR__
const SCRIPTS = dirname(CROSSCODE)
const ROOT = dirname(SCRIPTS)

as_rational(x::Integer) = x // 1
as_rational(x::Rational) = x
as_rational(x::AbstractFloat) = rationalize(x; tol = 1e-12)

function scaled_int(n, f, what)
    v = n * f
    denominator(v) == 1 || error("derive_case: $what $n × $f is not an integer")
    return Int(v)
end

"""
    derive_case(case; name, refine = 1, dt_divisor = 1, amplitude_factor = 1, ppc_factor = 1,
                seed = nothing, fields = nothing) -> Dict

A copy of a loaded case file with the changes of one scan point (file header). Cells per axis
× `refine`; Δt / (`refine` · `dt_divisor`); `scalars_every` and `fields_every` × (`refine` ·
`dt_divisor`), so every output time is a time of the case; perturbation amplitudes ×
`amplitude_factor`; particles per cell (per axis in 2D) × `ppc_factor`. Every product must be
an integer. The step count follows from the case's `t_end`. The result carries a `[derived]`
table with the source case's name and SHA-256 and the changes, and no `_path`.
"""
function derive_case(case; name, refine = 1, dt_divisor = 1, amplitude_factor = 1, ppc_factor = 1,
                     seed = nothing, fields = nothing)
    c = deepcopy(case)
    src = pop!(c, "_path", nothing)
    r, m = as_rational(refine), as_rational(dt_divisor)
    r > 0 && m > 0 || error("derive_case: refine and dt_divisor must be positive")
    two = get(c, "schema", 1) == 2
    g = c["grid"]
    g["cells"] = two ? [scaled_int(n, r, "cells") for n in g["cells"]] : scaled_int(g["cells"], r, "cells")
    c["time"]["dt"] = Float64(c["time"]["dt"] / (r * m))
    d = c["diagnostics"]
    for k in ("scalars_every", "fields_every")
        d[k] = scaled_int(d[k], r * m, k)
    end
    fields === nothing || (d["fields"] = collect(String, fields))
    pf = as_rational(ppc_factor)
    for s in c["species"]
        s["mobile"] || continue     # the immobile background keeps its load
        p = get(s, "perturbation", nothing)
        p === nothing || (p["amplitude"] = Float64(p["amplitude"] * amplitude_factor))
        s["particles_per_cell"] = two ? [scaled_int(n, pf, "particles per cell") for n in s["particles_per_cell"]] :
            scaled_int(s["particles_per_cell"], pf, "particles per cell")
    end
    seed === nothing || (c["numerics"]["random_seed"] = Int(seed))
    base = c["name"]
    c["name"] = "$base-$name"
    c["description"] = c["description"] * "; WP-031 scan point $name"
    c["derived"] = Dict{String,Any}("from" => base,
        "from_sha256" => src === nothing ? "unknown" : file_hash(src),
        "refine" => string(r), "dt_divisor" => string(m), "amplitude_factor" => Float64(amplitude_factor),
        "ppc_factor" => string(pf), "seed" => seed === nothing ? "case" : Int(seed),
        "fields" => fields === nothing ? "case" : collect(String, fields))
    return c
end

"The keyword arguments of `derive_case` for one point of a scan file."
point_kwargs(p) = (; name = p["name"], refine = get(p, "refine", 1), dt_divisor = get(p, "dt_divisor", 1),
    amplitude_factor = get(p, "amplitude_factor", 1), ppc_factor = get(p, "ppc_factor", 1),
    seed = get(p, "seed", nothing), fields = get(p, "fields", nothing))

"""
    prepare_scan(scan_path; runs = scripts/runs) -> dir

Write scripts/runs/<scan>/ for a scan file: summary/scan.toml (a copy), summary/cases/<point>.toml,
<point>/ (a Smilei run directory) and box-request.toml (`scripts/cpu-box.sh submit`).
"""
function prepare_scan(scan_path; runs = joinpath(SCRIPTS, "runs"))
    scan = TOML.parsefile(scan_path)
    name = scan["name"]
    dir = joinpath(runs, name)
    sumdir = joinpath(dir, "summary")
    mkpath(joinpath(sumdir, "cases"))
    cp(scan_path, joinpath(sumdir, "scan.toml"); force = true)
    case = load_case(scan["case"])
    names = String[]
    for p in scan["points"]
        c = derive_case(case; point_kwargs(p)...)
        cpath = joinpath(sumdir, "cases", p["name"] * ".toml")
        open(io -> TOML.print(io, c; sorted = true), cpath, "w")
        load_case(cpath)    # the derived file passes the generator's validation
        prepare_run(cpath; tag = p["name"], runs = dir)
        push!(names, p["name"])
    end
    rel(x) = relpath(x, ROOT)
    io = IOBuffer()
    println(io, "# CPU box request of the WP-031 scan $name (scripts/crosscode/convergence.jl prepare)")
    println(io, "timeout = ", get(scan, "timeout", 7200))
    println(io, "category = \"validation\"")
    println(io, "outputs = [\"", rel(sumdir), "\"]")
    for n in names
        println(io, "\n[[steps]]\nkind = \"smilei\"\ndir = \"", rel(joinpath(dir, n)), "\"")
    end
    println(io, "\n[[steps]]\nkind = \"julia\"\nargs = [\"--project=scripts\", \"scripts/crosscode/convergence.jl\", \"summarize\", \"",
        rel(dir), "\"]")
    write(joinpath(dir, "box-request.toml"), String(take!(io)))
    return dir
end

"The pre-registered rate observables of a cold case: (key, observable table) pairs, sorted."
observable_keys(case) = sort([(k, o) for (k, o) in case["observables"] if o["method"] == "exponential_window"]; by = first)

"""
    observable_series(case, sc, fl) -> Dict(key => (; t, U, c))

The series each rate observable fits, from Smilei's scalars `sc` and field dumps `fl`: the
field energy (two-stream `Uelm_Ex`; Weibel's in-plane magnetic energy) with `c = nothing`, or
|c|² of the complex mode coefficient c of the dumps (Weibel's B·ê; `mode_projection`). Schema 1
cases also get `mode`, |Ê_m|² of Ex from the dumps, the estimate analyze.jl records next to
the primary one.
"""
function observable_series(case, sc, fl)
    out = Dict{String,Any}()
    if get(case, "schema", 1) == 2
        lengths = Tuple(Float64.(case["grid"]["length"]))
        mode = Tuple(Int.(case["wavevector"]["mode"]))
        for (k, o) in observable_keys(case)
            if o["signal"] == "mode_projection"
                a, b = o["fields"]
                c = project_mode(fl.fields[a], fl.fields[b], fl.x[a], fl.x[b], mode, lengths; component = Symbol(o["component"]))
                out[k] = (; t = fl.time, U = abs2.(c), c)
            elseif o["signal"] == "scalar_sum"
                out[k] = (; t = sc.time, U = sum(sc.columns[x] for x in o["columns"]), c = nothing)
            else
                error("observable_series: signal $(o["signal"]) is not a rate signal of a cold case")
            end
        end
    else
        (k, o), = observable_keys(case)
        out[k] = (; t = sc.time, U = sc.columns[o["signal"]], c = nothing)
        m = only(unique(Int(s["perturbation"]["mode"]) for s in case["species"] if haskey(s, "perturbation")))
        c = mode_coefficient(fl.fields["Ex"], m)
        out["mode"] = (; t = fl.time, U = abs2.(c), c)
    end
    return out
end

"The line `# <key>: value` of a container.txt, or nothing."
function container_line(f, key)
    isfile(f) || return nothing
    for l in eachline(f)
        startswith(l, "# $key:") && return strip(l[length(key)+4:end])
    end
    return nothing
end

"""
    summarize_scan(dir) -> Vector{String}

For every point of scripts/runs/<scan>/ (the cases in summary/cases/), write summary/<point>.json
with the observable series (`observable_series`), the total energy, the step count, wall time,
thread count and CPU model from the run's container.txt, and the SHA-256 of its outputs; copy
the run's manifest.toml (the input hashes of prepare) to summary/<point>-manifest.toml. A point
whose run did not finish is listed in summary/missing.txt instead.
"""
function summarize_scan(dir)
    sumdir = joinpath(dir, "summary")
    points = sort([splitext(f)[1] for f in readdir(joinpath(sumdir, "cases")) if endswith(f, ".toml")])
    done, missing_ = String[], String[]
    for p in points
        rd = joinpath(dir, p)
        if !isfile(joinpath(rd, "scalars.txt")) || container_line(joinpath(rd, "container.txt"), "end") === nothing
            push!(missing_, p)
            continue
        end
        case = TOML.parsefile(joinpath(rd, "case.toml"))
        sc = read_scalars(rd)
        names = get(case, "schema", 1) == 2 ?
            unique(vcat(String[], (get(o, "fields", String[]) for (_, o) in observable_keys(case))...)) : ["Ex"]
        fl = read_fields(rd; names, timestep = case["time"]["dt"])
        series = observable_series(case, sc, fl)
        rec = Dict{String,Any}("point" => p, "case" => case["name"], "derived" => case["derived"],
            "dt" => case["time"]["dt"], "cells" => case["grid"]["cells"],
            "steps" => round(Int, case["time"]["t_end"] / case["time"]["dt"]),
            "wall_seconds" => stamp_interval(joinpath(rd, "container.txt"), "# start:", "# end:"),
            "threads" => (l = container_line(joinpath(rd, "container.txt"), "threads"); l === nothing ? nothing : strip(first(split(l, ',')))),
            "cpu" => container_line(joinpath(rd, "container.txt"), "cpu model"),
            "smilei" => container_line(joinpath(rd, "container.txt"), "smilei version"),
            "energy" => Dict("t" => sc.time, "Utot" => sc.columns["Utot"]),
            "outputs" => Dict(f => "sha256:" * file_hash(joinpath(rd, f))
                              for f in ("scalars.txt", "Fields0.h5", "smilei.log", "container.txt") if isfile(joinpath(rd, f))),
            "series" => Dict(k => Dict{String,Any}("t" => s.t, "U" => s.U,
                                 "c_re" => s.c === nothing ? nothing : real.(s.c),
                                 "c_im" => s.c === nothing ? nothing : imag.(s.c)) for (k, s) in series))
        open(io -> JSON.print(io, rec), joinpath(sumdir, p * ".json"), "w")
        cp(joinpath(rd, "manifest.toml"), joinpath(sumdir, p * "-manifest.toml"); force = true)
        push!(done, p)
    end
    write(joinpath(sumdir, "missing.txt"), join(missing_, "\n"))
    c = git_commit(ROOT)
    write_manifest(sumdir; extra = Dict{String,Any}("summarized" => done, "missing" => missing_,
        "summarize_commit" => c.commit, "summarize_dirty" => coalesce(c.dirty, "unknown"),
        "summarize_inputs" => Dict("scripts/crosscode/" * f => "sha256:" * file_hash(joinpath(CROSSCODE, f))
                                   for f in ("convergence.jl", "read.jl", "analysis.jl", "smilei.jl"))))
    return done
end

"""
    ivp_series(case, key, t) -> Vector{Float64}

The exact linear cold-fluid value of the series `key` fits (`observable_series`) at the times
`t`, from `cold_ivp(case)`: the box energy of the observed field (energy observables) or the
squared mode amplitude (`mode_projection` and the 1D `mode`), in the units of the run.
"""
function ivp_series(case, key, t)
    p = cold_ivp(case)
    a = ivp_solution(p.M, p.y0, t)[p.index, :]
    energy = key != "mode" && (get(case, "schema", 1) == 1 || case["observables"][key]["signal"] == "scalar_sum")
    return energy ? p.energy_scale .* abs2.(a) : abs2.(a)
end

"""
    local_rate(t, U; half_width = 1.0) -> (tc, γ)

The local growth rate of the amplitude, half the slope of a least-squares line through log U
over the samples within ± `half_width` of each time (the check of the WP-012 review), at the
times whose whole stencil lies inside the series.
"""
function local_rate(t, U; half_width = 1.0)
    tc, g = Float64[], Float64[]
    for i in eachindex(t)
        (t[i] - half_width < t[1] || t[i] + half_width > t[end]) && continue
        idx = findall(x -> abs(x - t[i]) <= half_width + 1e-12, t)
        all(>(0), U[idx]) || continue
        push!(tc, t[i]); push!(g, linear_fit(t[idx], log.(U[idx])).slope / 2)
    end
    return tc, g
end

"""
    point_fits(case, key, t, U; window) -> NamedTuple

For the series `key` (t, U) of a run: the pre-registered fit (`growth_from_window` with the
observable's window), and the same least-squares fit on the exact linear solution at the same
sample indices (`gamma_ivp`), the rate a run without discretization error and without
nonlinearity would report; the ratio U/U_ivp at the first and last sample of the window.
"""
function point_fits(case, key, t, U; window)
    r = growth_from_window(t, U; window)
    isfinite(r.gamma) || return (; r, gamma = NaN, gamma_ivp = NaN, idx = 1:0, ratio = (NaN, NaN))
    i1 = findfirst(==(first(r.t_fit)), t)
    i2 = findfirst(==(last(r.t_fit)), t)
    idx = i1:i2
    Uivp = ivp_series(case, key, t[idx])
    f = linear_fit(t[idx], log.(Uivp))
    return (; r, gamma = r.gamma, gamma_ivp = f.slope / 2, idx, ratio = (U[i1] / Uivp[1], U[i2] / Uivp[end]))
end

end

include(joinpath(@__DIR__, "convergence_report.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    using .Convergence
    cmd = get(ARGS, 1, "")
    if cmd == "prepare" && length(ARGS) == 2
        println(prepare_scan(ARGS[2]))
    elseif cmd == "summarize" && length(ARGS) == 2
        done = summarize_scan(ARGS[2])
        println("summarized ", length(done), " points: ", join(done, ", "))
    elseif cmd == "report" && length(ARGS) in (3, 4)
        ConvergenceReport.report(ARGS[2], ARGS[3]; box_status = get(ARGS, 4, nothing))
    elseif cmd == "compare" && length(ARGS) == 4
        ConvergenceReport.compare_codes(ARGS[2], ARGS[3], ARGS[4])
    else
        println(stderr, "usage: convergence.jl prepare <scan.toml> | summarize <scan dir> | report <summary dir> <out dir> [box-status.toml] | compare <reference summary> <other summary> <out dir>")
        exit(2)
    end
end
