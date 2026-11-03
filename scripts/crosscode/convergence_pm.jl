# ParticleMesh on the points of a 1D scan of scripts/crosscode/convergence.jl (WP-031, #226):
# the electromagnetic module (Reference mode, Float64, the settings of scripts/em1d/crosscode.jl,
# its start `MOMENTA_AT` included, recorded in each point and in the manifest)
# runs the derived case of each named point and writes <out>/<point>.json in the format of
# `summarize_scan`, so that `convergence.jl report` evaluates it and `compare_codes` sets it
# against the Smilei point of the same case.
#
#   julia +1.13 --project=scripts -t 16 scripts/crosscode/convergence_pm.jl <summary dir> <out dir> <point>...
#
# The cases are read from <summary dir>/cases/<point>.toml and copied to <out dir>/cases/.

include(joinpath(dirname(@__DIR__), "em1d", "crosscode.jl"))   # simulate(case), REPO, write_manifest

function run_points(sumdir, out, points)
    mkpath(joinpath(out, "cases"))
    cp(joinpath(sumdir, "scan.toml"), joinpath(out, "scan.toml"); force = true)
    for p in points
        src = joinpath(sumdir, "cases", p * ".toml")
        cp(src, joinpath(out, "cases", p * ".toml"); force = true)
        case = TOML.parsefile(src)
        sim = simulate(case)
        iE = 1:case["diagnostics"]["scalars_every"]:length(sim.total)    # the total energy at the scalar cadence
        rec = Dict{String,Any}("point" => p, "case" => case["name"], "derived" => case["derived"], "code" => "particlemesh",
            "momenta_at" => string(MOMENTA_AT), "dt" => sim.dt, "cells" => case["grid"]["cells"], "steps" => sim.nsteps, "wall_seconds" => sim.wall,
            "threads" => string(Threads.nthreads()), "cpu" => Sys.cpu_info()[1].model,
            "energy" => Dict("t" => (iE .- 1) .* sim.dt, "Utot" => sim.total[iE]),
            "series" => Dict(
                "growth_rate" => Dict{String,Any}("t" => sim.tU, "U" => sim.U, "c_re" => nothing, "c_im" => nothing),
                "mode" => Dict{String,Any}("t" => sim.tC, "U" => abs2.(sim.C), "c_re" => real.(sim.C), "c_im" => imag.(sim.C))))
        open(io -> JSON.print(io, rec), joinpath(out, p * ".json"), "w")
        println(p, ": ", sim.nsteps, " steps in ", round(sim.wall; digits = 1), " s")
    end
    # inputs relative to the repository, as scripts/em1d/common.jl records them
    inputs = Dict{String,Any}(relpath(f, REPO) => "sha256:" * RunManifest.file_hash(f) for f in input_files(@__FILE__, nothing))
    write_manifest(out; root = REPO,
        extra = Dict{String,Any}("inputs" => inputs, "points" => collect(points), "threads" => Threads.nthreads(),
            "cpu" => Sys.cpu_info()[1].model, "mode" => "Reference", "precision" => "Float64", "momenta_at" => string(MOMENTA_AT)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 3 || (println(stderr, "usage: convergence_pm.jl <summary dir> <out dir> <point>..."); exit(2))
    run_points(ARGS[1], ARGS[2], ARGS[3:end])
end
