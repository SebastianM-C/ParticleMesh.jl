# ParticleMesh's oblique Landau ensemble over the case's 32 seeds (WP-032), run on the CPU box in
# requests of a few seeds each and merged afterwards into one ensemble directory, evaluated with
# the pre-registered observable `damping_rate_mode_ensemble` by the shared ensemble code, as
# WP-023's four-seed run (scripts/em2d/crosscode.jl) and WP-025's Smilei and PIConGPU runs.
#
#   julia +1.13 --project=scripts -t 16 scripts/em2d/landau_ensemble.jl run <dir> <seed>... [--ppc a,b --cells a,b]
#   julia +1.13 --project=scripts scripts/em2d/landau_ensemble.jl merge <out dir> <request dir>...
#   julia +1.13 --project=scripts scripts/em2d/landau_ensemble.jl checks <ensemble dir> <smilei seed dir> <out dir>
#   julia +1.13 --project=scripts scripts/em2d/landau_ensemble.jl figure <ensemble dir> <out dir>
#
# `run` writes seed-<seed>/ of each named seed (one of the case's seeds) into the fixed directory
# <dir> (a box request names its outputs exactly), with the series, the observables, the standard
# plots and a manifest that also records the machine's boot id and boot time, so that a request
# that ran on a new box shows it. `merge` takes the request directories of all seeds (each fetched
# by scripts/cpu-box.sh into scripts/runs/box-<tag>/, with box-status.toml one level up), refuses
# a missing, repeated or foreign seed and requests whose inputs differ, and writes the ensemble
# into <out dir>: every seed's directory, seed-coefficients.json, observables.json and the
# ensemble plots (write_ensemble), and a manifest naming each request. `checks` records the
# post-hoc checks: the reruns of WP-023's four seeds against its committed observables and mode
# series, the box's Smilei seed against WP-025's run of it, and the ns per particle-step of both
# codes on the box. `figure` draws the three codes' per-seed and ensemble rates.
include(joinpath(@__DIR__, "crosscode.jl"))

const CASE_NAME = "landau-oblique-2d"
const WP023_RUN = joinpath(REPO, "reports", "wp-023-em-step-2d", "em2d-pm-landau-oblique-2d-20261102T163231")
const WP025_SMILEI = joinpath(REPO, "reports", "crosscode-picongpu", "smilei-landau-oblique-2d-431f4192")

"The machine a run is on: boot id and boot time from /proc (unknown elsewhere), CPU model, CPU and thread counts."
function host_record()
    rd(f) = try
        strip(read(f, String))
    catch
        "unknown"
    end
    btime = try
        m = match(r"^btime (\d+)"m, read("/proc/stat", String))
        m === nothing ? "unknown" : string(unix2datetime(parse(Int, m.captures[1]))) * "Z"
    catch
        "unknown"
    end
    return Dict{String,Any}("boot_id" => rd("/proc/sys/kernel/random/boot_id"), "boot_time" => btime,
        "cpu" => Sys.cpu_info()[1].model, "cpus" => Sys.CPU_THREADS, "threads" => Threads.nthreads())
end

"""
    run_seeds(dir, case_file, seeds; ppc = nothing, cells = nothing) -> dir

Run each of `seeds` (each one of the case's seeds) of the case in `case_file` into
`dir/seed-<seed>/` with `run_one` (Reference mode, Float64), and write a copy of the case file
`dir/case.toml` and `dir/manifest.toml`: the
commit, the SHA-256 of every input, the seeds, the host (`host_record`), and per seed the wall
time of the steps and its ns per particle-step. A seed whose `observables.json` exists is kept.
"""
function run_seeds(dir, case_file, seeds; ppc = nothing, cells = nothing)
    case = TOML.parsefile(case_file)
    all_seeds = case_seeds(case)
    bad = setdiff(seeds, all_seeds)
    isempty(bad) || error("run_seeds: seeds $bad are not seeds of the case")
    allunique(seeds) || error("run_seeds: a seed is named twice")
    mkpath(dir)
    cp(case_file, joinpath(dir, "case.toml"); force = true)
    script = @__FILE__
    files = sort(unique([input_files(script, nothing); case_file;
                         joinpath.(CROSSCODE, ["ensemble.jl", "ensemble_report.jl"]); joinpath(@__DIR__, "crosscode.jl")]))
    inputs = Dict{String,Any}(relpath(f, REPO) => "sha256:" * RunManifest.file_hash(f) for f in files)
    perf = Dict{String,Any}()
    record() = write_manifest(dir; root = REPO, extra = Dict{String,Any}("inputs" => inputs, "run_id" => basename(normpath(dir)),
        "work_package" => "WP-032", "julia" => string(VERSION), "threads" => Threads.nthreads(), "backend" => "KernelAbstractions CPU",
        "cpu" => Sys.cpu_info()[1].model, "host" => host_record(), "seeds" => collect(seeds), "per_seed" => perf,
        "case_sha256" => RunManifest.file_hash(joinpath(dir, "case.toml")),
        "parameters" => Dict{String,Any}("case" => case["name"], "element_type" => "Float64", "mode" => "Reference",
            "module" => "ParticleMeshElectromagnetic", "current" => "Esirkepov",
            "scaled" => ppc === nothing && cells === nothing ? "no" : "ppc = $ppc, cells = $cells")))
    record()
    for s in seeds
        sdir = joinpath(dir, "seed-$s")
        if isfile(joinpath(sdir, "observables.json"))
            println("seed $s: kept from an earlier run")
            continue
        end
        sim, out = run_one(sdir, case, s; ppc, cells)
        p = out["performance"]
        perf[string(s)] = Dict{String,Any}("wall_seconds" => p["wall_seconds"], "ns_per_particle_step" => p["ns_per_particle_step"],
            "particles" => p["particles"], "steps" => p["steps"], "finished" => string(now(UTC)) * "Z")
        record()
        println("seed $s: $(round(sim.wall; digits = 1)) s, $(round(p["ns_per_particle_step"]; digits = 2)) ns per particle-step, ",
                "damping_rate_mode = $(something(out["observables"]["damping_rate_mode"]["measured"], "none")); Gauss $(sim.checks.final.gauss)")
        flush(stdout)
        sim = nothing
        GC.gc()
    end
    return dir
end

"The complex mode coefficient E·k̂ of one seed and the per-run energy fit, from its `series.json`."
function seed_series(sdir, case)
    s = JSON.parsefile(joinpath(sdir, "series.json"))
    tC = Float64.(s["tC"])
    El = complex.(Float64.(s["El_re"]), Float64.(s["El_im"]))
    ro = case["observables"]["damping_rate_energy"]
    per_run = damping_from_peaks(Float64.(s["t"]), Float64.(s["WEx"]) .+ Float64.(s["WEy"]);
                                 fit_start = ro["fit_start"], floor_factor = ro["floor_factor"])
    return tC, El, per_run
end

"The box-status.toml next to a fetched request directory (scripts/runs/box-<tag>/<dir>), or nothing."
function box_status(rdir)
    f = joinpath(dirname(normpath(rdir)), "box-status.toml")
    return isfile(f) ? TOML.parsefile(f) : nothing
end

"""
    merge_requests(out, rdirs; case_file = the case file of landau-oblique-2d) -> result

Merge the request directories `rdirs` (each written by `run_seeds`) into the ensemble directory
`out`. Every request must have run `case_file` (its `case.toml` has the same SHA-256) with the
same inputs (the same SHA-256 per file) and parameters, every seed of the case must come from
exactly one request, and no request may hold a seed the case does not have; otherwise it throws
before writing anything. Writes `out/seed-<seed>/`
(copies), `out/case.toml`, `seed-coefficients.json`, the ensemble observable with its plots
(`write_ensemble`, the seeds in the case's order) and `manifest.toml` with each request (its
directory, seeds, host and box status) and the hash of every output.
"""
function merge_requests(out, rdirs; case_file = joinpath(CASES, "$CASE_NAME.toml"))
    case = TOML.parsefile(case_file)
    seeds = case_seeds(case)
    h = RunManifest.file_hash(case_file)
    for r in rdirs
        RunManifest.file_hash(joinpath(r, "case.toml")) == h || error("merge: $r ran a case file other than $case_file")
    end
    where_ = Dict{Int,String}()
    manifests = [TOML.parsefile(joinpath(r, "manifest.toml")) for r in rdirs]
    for (r, m) in zip(rdirs, manifests)
        for name in readdir(r)
            startswith(name, "seed-") && isdir(joinpath(r, name)) || continue
            s = parse(Int, name[6:end])
            s in seeds || error("merge: $r holds seed $s, which is not a seed of the case")
            haskey(where_, s) && error("merge: seed $s is in both $(where_[s]) and $r")
            isfile(joinpath(r, name, "series.json")) || error("merge: seed $s in $r did not finish (no series.json)")
            where_[s] = r
        end
    end
    missing_ = setdiff(seeds, keys(where_))
    isempty(missing_) || error("merge: no request holds seeds $missing_")
    ref = manifests[1]
    for (r, m) in zip(rdirs, manifests)
        m["inputs"] == ref["inputs"] || error("merge: the inputs of $r differ from those of $(rdirs[1]): " *
            join(sort([k for k in union(keys(m["inputs"]), keys(ref["inputs"])) if get(m["inputs"], k, nothing) != get(ref["inputs"], k, nothing)]), ", "))
        m["case_sha256"] == ref["case_sha256"] || error("merge: $r ran another case file")
        m["parameters"] == ref["parameters"] || error("merge: $r ran other parameters")
    end
    mkpath(out)
    cp(joinpath(rdirs[1], "case.toml"), joinpath(out, "case.toml"); force = true)
    coeffs, per_run, tC = Vector{ComplexF64}[], [], nothing
    walls, nsps, particles, steps, threads, cpus = Float64[], Float64[], Int[], Int[], Int[], String[]
    for s in seeds
        src = joinpath(where_[s], "seed-$s")
        cp(src, joinpath(out, "seed-$s"); force = true)
        t, El, pr = seed_series(src, case)
        tC === nothing ? (tC = t) : (t == tC || error("merge: seed $s has another field time grid"))
        push!(coeffs, El)
        push!(per_run, pr)
        p = JSON.parsefile(joinpath(src, "observables.json"))["performance"]
        push!(walls, p["wall_seconds"]); push!(nsps, p["ns_per_particle_step"])
        push!(particles, p["particles"]); push!(steps, p["steps"]); push!(threads, p["threads"])
        push!(cpus, TOML.parsefile(joinpath(where_[s], "manifest.toml"))["cpu"])
    end
    length(unique(particles)) == 1 && length(unique(steps)) == 1 || error("merge: the seeds ran different sizes")
    r = write_ensemble(out, case, tC, coeffs, seeds; code = "particlemesh", per_run,
        extra = Dict{String,Any}("wall_seconds_per_seed" => walls, "ns_per_particle_step" => nsps,
            "particles_per_seed" => first(particles), "steps" => first(steps),
            "particle_steps_per_seed" => first(particles) * first(steps), "threads" => unique(threads), "cpu" => unique(cpus)))
    write_seed_coefficients(out, tC, seeds, coeffs; observable = r["observable"])
    requests = [Dict{String,Any}("directory" => relpath(normpath(rd), REPO), "seeds" => m["seeds"], "host" => m["host"],
                    "commit" => m["commit"], "box_status" => something(box_status(rd), "none")) for (rd, m) in zip(rdirs, manifests)]
    outputs = Dict{String,Any}()
    for (root, _, fs) in walkdir(out), f in fs
        f == "manifest.toml" && continue
        p = joinpath(root, f)
        outputs[relpath(p, out)] = "sha256:" * RunManifest.file_hash(p)
    end
    write_manifest(out; root = REPO, extra = Dict{String,Any}("run_id" => basename(normpath(out)), "work_package" => "WP-032",
        "inputs" => ref["inputs"], "parameters" => ref["parameters"], "case_sha256" => ref["case_sha256"], "requests" => requests,
        "outputs" => outputs, "merged_by" => "scripts/em2d/landau_ensemble.jl merge",
        "observables" => Dict(k => r[k] for k in ("observable", "measured", "measured_stderr", "maxima_used", "ensemble_floor",
                                                    "observable_exists", "M") if r[k] !== nothing)))
    println("ParticleMesh ensemble of $(length(seeds)) seeds: γ = $(r["measured"]) ± $(r["measured_stderr"]) ($(r["maxima_used"]) maxima)")
    return r
end

"Largest |a − b| over the elements and the largest |b|, of two series (vectors of numbers)."
maxdiff(a, b) = (; max_abs_difference = maximum(abs.(Float64.(a) .- Float64.(b))), max_abs = maximum(abs.(Float64.(b))),
                   bitwise = Float64.(a) == Float64.(b))

"""
    checks(ens, smilei_seed, out) -> sidecar

Post-hoc checks, written to `out/checks.json`: (1) each of WP-023's four seeds, rerun in `ens`,
against WP-023's committed run: every per-seed observable and the mode series E·k̂ (bitwise or
the largest difference); (2) the box's Smilei run of one seed (`smilei_seed`, analysed by
analyze.jl) against WP-025's committed run of the same seed; (3) the ns per particle-step of
both codes on the box, normalized by the mobile electrons as WP-025's Smilei record is.
"""
function checks(ens, smilei_seed, out)
    mkpath(out)
    side = Dict{String,Any}("note" => "post-hoc checks, not thresholds (A7)")
    old_seeds = Int.(TOML.parsefile(joinpath(WP023_RUN, "case.toml"))["numerics"]["seeds"])
    rerun = Dict{String,Any}()
    for s in old_seeds
        a = JSON.parsefile(joinpath(ens, "seed-$s", "observables.json"))
        b = JSON.parsefile(joinpath(WP023_RUN, "seed-$s", "observables.json"))
        sa, sb = JSON.parsefile(joinpath(ens, "seed-$s", "series.json")), JSON.parsefile(joinpath(WP023_RUN, "seed-$s", "series.json"))
        obs = Dict{String,Any}(k => Dict("box" => a["observables"][k]["measured"], "wp023" => b["observables"][k]["measured"],
                                         "equal" => a["observables"][k]["measured"] == b["observables"][k]["measured"])
                               for k in keys(b["observables"]))
        rerun[string(s)] = Dict{String,Any}("observables" => obs,
            "El_re" => maxdiff(sa["El_re"], sb["El_re"]), "El_im" => maxdiff(sa["El_im"], sb["El_im"]),
            "WEx" => maxdiff(sa["WEx"], sb["WEx"]), "total" => maxdiff(sa["total"], sb["total"]),
            "gauss_relative_final" => Dict("box" => a["checks"]["gauss_relative_final"], "wp023" => b["checks"]["gauss_relative_final"]))
    end
    side["wp023_seeds_rerun"] = rerun
    s = basename(normpath(smilei_seed))
    a = JSON.parsefile(joinpath(smilei_seed, "observables.json"))
    b = JSON.parsefile(joinpath(WP025_SMILEI, s, "observables.json"))
    side["smilei_seed_rerun"] = Dict{String,Any}("seed" => s, "observables" => Dict{String,Any}(k => Dict("box" => get(get(a["observables"], k, Dict()), "measured", nothing),
            "wp025" => v["measured"], "equal" => get(get(a["observables"], k, Dict()), "measured", nothing) == v["measured"])
            for (k, v) in b["observables"] if v isa AbstractDict && haskey(v, "measured")))
    pm = JSON.parsefile(joinpath(ens, "observables.json"))
    sp = a["performance"]
    pm["particles_per_seed"] == sp["mobile_particles"] || error("checks: ParticleMesh and Smilei count other particles")
    pm_ns = Float64.(pm["ns_per_particle_step"])
    side["throughput"] = Dict{String,Any}(
        "normalization" => "wall time / (mobile electrons × steps); ParticleMesh times its steps (run!), Smilei's container.txt its whole run, initialization and output included",
        "electrons" => sp["mobile_particles"], "steps" => pm["steps"],
        "particlemesh" => Dict("ns_per_particle_step" => pm_ns, "min" => minimum(pm_ns), "median" => median(pm_ns), "max" => maximum(pm_ns),
                               "wall_seconds_per_seed" => pm["wall_seconds_per_seed"], "threads" => pm["threads"], "cpu" => pm["cpu"]),
        "smilei" => Dict("ns_per_particle_step" => sp["ns_per_mobile_particle_step"], "wall_seconds" => sp["wall_seconds"],
                         "steps" => sp["steps"], "hardware" => get(TOML.parsefile(joinpath(smilei_seed, "manifest.toml")), "hardware", nothing)),
        "ratio_particlemesh_median_over_smilei" => median(pm_ns) / sp["ns_per_mobile_particle_step"])
    open(io -> JSON.print(io, side, 2), joinpath(out, "checks.json"), "w")
    JSON.print(stdout, side, 2)
    println()
    return side
end

"""
    summary_figure(ens, out) -> sidecar

`out/three-codes-rates.png` with its sidecar: per code (ParticleMesh's ensemble `ens`, WP-025's
Smilei and PIConGPU FP64 runs of the same case file), each seed's γ at its ensemble's maxima
(points), the ensemble γ with its jackknife error (bar) and the theory value (line).
"""
function summary_figure(ens, out)
    mkpath(out)
    runs = ["particlemesh" => ens; collect(reference_ensembles(CASE_NAME, ens))]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xticks = (1:length(runs), first.(runs)), ylabel = "γ [ω_p]",
              title = "landau-oblique-2d: γ per seed at the ensemble's maxima, and the ensemble γ ± jackknife")
    side = Dict{String,Any}("plot" => "per-seed γ at the ensemble's maxima (points) and ensemble γ ± jackknife (bars) per code, theory line")
    theory = nothing
    for (i, (code, dir)) in enumerate(runs)
        r = JSON.parsefile(joinpath(dir, "observables.json"))
        theory = r["theory"]
        g = Float64[x for x in r["per_seed_at_ensemble_peaks"]["gammas"] if x !== nothing]
        scatter!(ax, i .+ 0.25 .* (range(-1, 1; length = length(g))), g; color = (:gray, 0.6), markersize = 6)
        errorbars!(ax, [i], [r["measured"]], [r["measured_stderr"]]; color = :black, whiskerwidth = 12, linewidth = 2)
        scatter!(ax, [i], [r["measured"]]; color = :red, markersize = 12)
        side[code] = Dict("run" => relpath(normpath(dir), REPO), "measured" => r["measured"], "measured_stderr" => r["measured_stderr"],
                          "M" => r["M"], "maxima_used" => r["maxima_used"], "per_seed_at_ensemble_peaks" => g)
    end
    hlines!(ax, [theory]; color = :blue, linestyle = :dash, label = "theory γ = $(round(theory; sigdigits = 5))")
    axislegend(ax; position = :rb)
    side["theory"] = theory
    save_plot(out, "three-codes-rates", fig, side)
    return side
end

function cli_flags(args)
    pos, opts = String[], Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            opts[args[i]] = args[i + 1]
            i += 2
        else
            push!(pos, args[i])
            i += 1
        end
    end
    return pos, opts
end

if abspath(PROGRAM_FILE) == @__FILE__()
    pos, opts = cli_flags(ARGS)
    cmd = pos[1]
    if cmd == "run"
        ppc = haskey(opts, "--ppc") ? parse_pair(opts["--ppc"]) : nothing
        cells = haskey(opts, "--cells") ? parse_pair(opts["--cells"]) : nothing
        run_seeds(abspath(pos[2]), joinpath(CASES, "$CASE_NAME.toml"), parse.(Int, pos[3:end]); ppc, cells)
    elseif cmd == "merge"
        merge_requests(abspath(pos[2]), abspath.(pos[3:end]))
    elseif cmd == "figure"
        summary_figure(abspath(pos[2]), abspath(pos[3]))
    elseif cmd == "checks"
        checks(abspath(pos[2]), abspath(pos[3]), abspath(pos[4]))
    else
        error("usage: run <dir> <seed>... [--ppc a,b --cells a,b] | merge <out dir> <request dir>... | checks <ensemble dir> <smilei seed dir> <out dir> | figure <ensemble dir> <out dir>")
    end
end
