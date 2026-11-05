# Tests of ParticleMesh's oblique Landau ensemble driver (WP-032, scripts/em2d/landau_ensemble.jl):
# seeds run in separate requests and merged give the ensemble of the same seeds run in one
# process, bit for bit, and the merge refuses an incomplete, repeated or mixed set of requests.
# Included by runtests.jl after convergence.jl.

# The driver includes the electromagnetic module and the cross-code modules: a module of its own.
module PMEns
include(joinpath(dirname(dirname(@__DIR__)), "em2d", "landau_ensemble.jl"))
end

"Equal JSON records: the same keys and strings, numbers equal to `rtol` (a record recomputed elsewhere may differ in the last digits)."
pm_close(a::AbstractDict, b::AbstractDict; rtol) = keys(a) == keys(b) && all(k -> pm_close(a[k], b[k]; rtol), keys(a))
pm_close(a::AbstractVector, b::AbstractVector; rtol) = length(a) == length(b) && all(i -> pm_close(a[i], b[i]; rtol), eachindex(a))
pm_close(a::Real, b::Real; rtol) = isapprox(a, b; rtol)
pm_close(a, b; rtol) = isequal(a, b)

"Rewrite the JSON file `f` after `change!` edits its parsed content."
function edit_json!(change!, f)
    d = PMEns.JSON.parsefile(f)
    change!(d)
    open(io -> PMEns.JSON.print(io, d), f, "w")
end

@testset "T-490 seeds run in separate requests merge into the ensemble of one run, and the merge refuses an incomplete or mixed set [A4, A5, WP-032]" begin
    mktempdir() do tmp
        # the case with its first four seeds, at 32 × 32 cells and 4 × 4 electrons per cell, with a
        # 20 % perturbation so that the ensemble fit and its jackknife find their maxima above the noise of
        # 16384 electrons (at 10 % the reference codes' start, momenta at −Δt/2 and the density at cell
        # centres, gave no jackknife error on these four seeds)
        src = TOML.parsefile(joinpath(PMEns.CASES, "landau-oblique-2d.toml"))
        seeds = Int.(src["numerics"]["seeds"][1:4])[[2, 1, 4, 3]]     # not ascending: a sort shows
        src["numerics"]["seeds"] = seeds
        src["species"][1]["perturbation"]["amplitude"] = 0.2
        case_file = joinpath(tmp, "case.toml")
        open(io -> TOML.print(io, src; sorted = true), case_file, "w")
        sz = (; ppc = (4, 4), cells = (32, 32))
        r1 = PMEns.run_seeds(joinpath(tmp, "r1"), case_file, seeds[[3, 1]]; sz...)
        r2 = PMEns.run_seeds(joinpath(tmp, "r2"), case_file, seeds[[2, 4]]; sz...)
        m1 = TOML.parsefile(joinpath(r1, "manifest.toml"))
        @test m1["seeds"] == seeds[[3, 1]] && sort(collect(keys(m1["per_seed"]))) == sort(string.(seeds[[3, 1]]))
        @test m1["host"]["threads"] == Threads.nthreads() && haskey(m1["host"], "boot_id")
        @test m1["case_sha256"] == PMEns.RunManifest.file_hash(case_file)
        # the merge, requests in either order, against the same four seeds simulated here
        out = joinpath(tmp, "merged")
        r = PMEns.merge_requests(out, [r2, r1]; case_file)
        case = TOML.parsefile(case_file)
        sims = [PMEns.simulate(case, s; sz...) for s in seeds]
        ro = case["observables"]["damping_rate_energy"]
        per_run = [PMEns.damping_from_peaks(x.tU, x.WEx .+ x.WEy; fit_start = ro["fit_start"], floor_factor = ro["floor_factor"]) for x in sims]
        direct = PMEns.EnsembleReport.ensemble_result(case, sims[1].tC, [x.El for x in sims], seeds; per_run)[1]
        @test r["seeds"] == seeds
        @test r["measured"] isa Float64 && r["measured_stderr"] isa Float64 && r["maxima_used"] >= 3
        for k in ("measured", "measured_stderr", "measured_frequency", "maxima_used", "ensemble_floor", "jackknife_gammas")
            @test isequal(r[k], direct[k])
        end
        @test isequal(r["per_seed_single_run_rule_on_energy"], direct["per_seed_single_run_rule_on_energy"])
        # the per-run energy series of each seed is Ex's plus Ey's energy, seed by seed
        for (i, s) in enumerate(seeds)
            pr = PMEns.seed_series(joinpath(out, "seed-$s"), case)[3]
            @test pr.floor == per_run[i].floor && pr.used == per_run[i].used
        end
        @test r["per_run_energy_floors"] == [x.floor for x in per_run]      # each seed's own fit, in the case's order
        # the throughput fields: each seed's own record, in the case's order, and its definition
        perf = [PMEns.JSON.parsefile(joinpath(out, "seed-$s", "observables.json"))["performance"] for s in seeds]
        @test r["ns_per_particle_step"] == [p["ns_per_particle_step"] for p in perf]
        @test all(p -> p["ns_per_particle_step"] ≈ 1e9 * p["wall_seconds"] / (p["particles"] * p["steps"]), perf)
        @test r["particle_steps_per_seed"] == perf[1]["particles"] * perf[1]["steps"] == 32 * 32 * 16 * sims[1].nsteps
        @test r["wall_seconds_per_seed"] == [p["wall_seconds"] for p in perf] && r["cpu"] == [Sys.cpu_info()[1].model]
        t, s_, c = PMEns.seed_coefficients(out)
        @test s_ == seeds && t == sims[1].tC
        @test all(reinterpret(UInt64, c[i]) == reinterpret(UInt64, sims[i].El) for i in 1:4)
        @test all(isdir(joinpath(out, "seed-$s")) for s in seeds)
        m = TOML.parsefile(joinpath(out, "manifest.toml"))
        @test length(m["requests"]) == 2 && m["inputs"] == m1["inputs"]
        @test haskey(m["outputs"], "seed-coefficients.json") && haskey(m["outputs"], "observables.json")
        # refusals: a missing seed, a repeated seed, a foreign seed, other inputs, another case file
        @test_throws ErrorException PMEns.merge_requests(joinpath(tmp, "x1"), [r1]; case_file)
        cp(r1, joinpath(tmp, "r1b"))
        @test_throws ErrorException PMEns.merge_requests(joinpath(tmp, "x2"), [r1, r2, joinpath(tmp, "r1b")]; case_file)
        @test_throws ErrorException PMEns.run_seeds(joinpath(tmp, "r3"), case_file, [seeds[1] + 1]; sz...)
        cp(r2, joinpath(tmp, "r2b"))
        mb = TOML.parsefile(joinpath(tmp, "r2b", "manifest.toml"))
        k = first(sort(collect(keys(mb["inputs"]))))
        mb["inputs"][k] = "sha256:" * "0"^64
        open(io -> TOML.print(io, mb; sorted = true), joinpath(tmp, "r2b", "manifest.toml"), "w")
        @test_throws ErrorException PMEns.merge_requests(joinpath(tmp, "x3"), [r1, joinpath(tmp, "r2b")]; case_file)
        other = joinpath(tmp, "other.toml")
        write(other, read(case_file, String) * "\n")
        @test_throws ErrorException PMEns.merge_requests(joinpath(tmp, "x4"), [r1, r2]; case_file = other)
        # a seed directory moved from one request to another: every seed once, but neither request's
        # directories are the seeds its manifest ran
        cp(r1, joinpath(tmp, "r1d")); cp(r2, joinpath(tmp, "r2d"))
        mv(joinpath(tmp, "r2d", "seed-$(seeds[4])"), joinpath(tmp, "r1d", "seed-$(seeds[4])"))
        @test_throws ErrorException PMEns.merge_requests(joinpath(tmp, "x5"), [joinpath(tmp, "r1d"), joinpath(tmp, "r2d")]; case_file)
        @test !any(isdir, joinpath.(tmp, ["x1", "x2", "x3", "x4", "x5"]))
        # a request never reuses a seed directory that exists (another commit or size could have written it)
        @test_throws ErrorException PMEns.run_seeds(r1, case_file, seeds[[1]]; sz...)
        @test_throws ErrorException PMEns.run_seeds(joinpath(tmp, "r4"), case_file, seeds[[1, 1]]; sz...)
        @test !isdir(joinpath(tmp, "r4"))
        # more refusals, each on a copy of r2 changed in one place, before anything is written
        function refused(change!, name)
            d = joinpath(tmp, "c-$name")
            cp(r2, d)
            change!(d)
            o = joinpath(tmp, "x-$name")
            ok = try
                PMEns.merge_requests(o, [r1, d]; case_file)
                false
            catch e
                e isa ErrorException
            end
            return ok && !ispath(o)
        end
        s2 = seeds[2]
        @test refused("foreign") do d                      # a seed of no case in a request's directory
            cp(joinpath(d, "seed-$s2"), joinpath(d, "seed-$(s2 + 7)"))
        end
        @test refused("unfinished") do d                   # a seed directory without its series
            rm(joinpath(d, "seed-$s2", "series.json"))
        end
        @test refused("parameters") do d                   # another size recorded in the manifest
            mb = TOML.parsefile(joinpath(d, "manifest.toml"))
            mb["parameters"]["scaled"] = "ppc = (8, 8), cells = (32, 32)"
            open(io -> TOML.print(io, mb; sorted = true), joinpath(d, "manifest.toml"), "w")
        end
        @test refused("grid") do d                         # another field time grid
            edit_json!(x -> (x["tC"][end] += 1.0), joinpath(d, "seed-$s2", "series.json"))
        end
        @test refused("size") do d                         # another particle count
            edit_json!(x -> (x["performance"]["particles"] += 1), joinpath(d, "seed-$s2", "observables.json"))
        end
        # provenance: the commit from box-status.toml one level up when the box could not read git,
        # and the request commit's parent as the code commit in either case
        head = readchomp(`git -C $(PMEns.REPO) rev-parse HEAD`)
        parent = try readchomp(`git -C $(PMEns.REPO) rev-parse HEAD^`) catch; "unknown" end
        box = mkpath(joinpath(tmp, "box-x"))
        cp(r1, joinpath(box, "r1"))
        write(joinpath(box, "box-status.toml"), "request_commit = \"$head\"\nok = true\n")
        mu = TOML.parsefile(joinpath(box, "r1", "manifest.toml"))
        mu["commit"] = "unknown"
        q = PMEns.request_record(joinpath(box, "r1"), mu)
        @test q["commit"] == head && q["commit_source"] == "box-status.toml" && q["code_commit"] == parent
        @test q["box_status"]["request_commit"] == head
        mu["commit"] = head
        q = PMEns.request_record(joinpath(box, "r1"), mu)
        @test q["commit"] == head && q["commit_source"] == "run manifest" && q["code_commit"] == parent
        q = PMEns.request_record(r1, m1)                    # not from the box: no box status, no code commit
        @test q["box_status"] == "none" && q["commit"] == m1["commit"] && !haskey(q, "code_commit")
    end
end

@testset "T-491 the three-code comparison takes the reference ensembles of the run's case file [A5, WP-032]" begin
    em2d = PMEns
    wp023 = joinpath(REPORTS, "wp-023-em-step-2d", "em2d-pm-landau-oblique-2d-20261102T163231")
    four = em2d.reference_ensembles("landau-oblique-2d", wp023)
    @test [first(p) for p in four] == ["smilei", "picongpu"]
    @test all(endswith(last(p), "-bc8fd6af") for p in four)
    mktempdir() do tmp
        cp(joinpath(em2d.CASES, "landau-oblique-2d.toml"), joinpath(tmp, "case.toml"))
        all32 = em2d.reference_ensembles("landau-oblique-2d", tmp)
        @test all(endswith(last(p), "-431f4192") for p in all32)
        # each reference ran the same case file, so its seeds are the run's
        h = em2d.RunManifest.file_hash(joinpath(tmp, "case.toml"))
        @test all(em2d.RunManifest.file_hash(joinpath(last(p), "case.toml")) == h for p in all32)
        @test all(isfile(joinpath(last(p), "seed-coefficients.json")) for p in all32)
        write(joinpath(tmp, "case.toml"), read(joinpath(tmp, "case.toml"), String) * "\n")
        @test em2d.reference_ensembles("landau-oblique-2d", tmp) == ()
        # compare itself, on copies of the committed runs: WP-023's four seeds against WP-017's
        # references, WP-032's 32 against WP-025's, the record as committed; a seeded run of an
        # unknown case file is refused instead of compared with nothing
        run32 = joinpath(REPORTS, "wp-032-pm-landau-ensemble", "em2d-pm-landau-oblique-2d-32")
        for (src, h) in ((wp023, "bc8fd6af"), (run32, "431f4192"))
            d = joinpath(tmp, basename(src))
            cp(src, d)
            rm(joinpath(d, "compare"); recursive = true)
            side = redirect_stdout(() -> em2d.compare(d), devnull)
            for code in ("smilei", "picongpu")
                @test endswith(side["$(code)_ensemble"]["reference_run"], "-$h")
                @test side["$(code)_ensemble"]["particlemesh"]["measured"] == em2d.JSON.parsefile(joinpath(src, "observables.json"))["measured"]
            end
            h == "431f4192" && @test pm_close(em2d.JSON.parsefile(joinpath(d, "compare", "landau-oblique-2d-comparison.json")),
                                               em2d.JSON.parsefile(joinpath(src, "compare", "landau-oblique-2d-comparison.json")); rtol = 1e-12)
        end
        d = joinpath(tmp, basename(run32))
        write(joinpath(d, "case.toml"), read(joinpath(d, "case.toml"), String) * "\n")
        @test_throws ErrorException redirect_stdout(() -> em2d.compare(d), devnull)
    end
end

@testset "T-492 the committed 32-seed ParticleMesh record is what the merge computes from the committed seed files [A4, A5, A7, WP-032]" begin
    run = joinpath(REPORTS, "wp-032-pm-landau-ensemble", "em2d-pm-landau-oblique-2d-32")
    case = TOML.parsefile(joinpath(run, "case.toml"))
    seeds = PMEns.case_seeds(case)
    @test length(seeds) == 32 && PMEns.RunManifest.file_hash(joinpath(run, "case.toml")) == PMEns.RunManifest.file_hash(joinpath(PMEns.CASES, "landau-oblique-2d.toml"))
    # every seed's series gives the committed coefficients, and the ensemble the committed record
    series = [PMEns.seed_series(joinpath(run, "seed-$s"), case) for s in seeds]
    t, s_, c = PMEns.seed_coefficients(run)
    @test s_ == seeds && all(x -> x[1] == t, series) && all(i -> c[i] == series[i][2], eachindex(seeds))
    r = PMEns.EnsembleReport.ensemble_result(case, t, c, seeds; per_run = last.(series))[1]
    ob = PMEns.JSON.parsefile(joinpath(run, "observables.json"))
    for k in ("measured", "measured_stderr", "measured_frequency", "maxima_used", "ensemble_floor")
        @test r[k] ≈ ob[k] rtol = 1e-12
    end
    @test ob["M"] == 32 && ob["maxima_used"] == 6
    # the comparison record holds the committed reference ensembles of the same case file
    cmp = PMEns.JSON.parsefile(joinpath(run, "compare", "landau-oblique-2d-comparison.json"))
    for (code, dir) in PMEns.reference_ensembles("landau-oblique-2d", run)
        ref = PMEns.JSON.parsefile(joinpath(dir, "observables.json"))
        e = cmp["$(code)_ensemble"]
        @test e[code]["measured"] == ref["measured"] && e["particlemesh"]["measured"] == ob["measured"]
        @test e["difference_in_units_of_combined_stderr"] ≈ (ob["measured"] - ref["measured"]) / hypot(ob["measured_stderr"], ref["measured_stderr"]) rtol = 1e-12
    end
    # the checks and the figure regenerate from the committed records
    smilei = joinpath(REPORTS, "wp-032-pm-landau-ensemble", "smilei-seed-20261031")
    mktempdir() do tmp
        ck = redirect_stdout(() -> PMEns.checks(run, smilei, tmp), devnull)
        @test pm_close(PMEns.JSON.parsefile(joinpath(tmp, "checks.json")), PMEns.JSON.parsefile(joinpath(run, "checks", "checks.json")); rtol = 1e-12)
        th = ck["throughput"]
        sm = PMEns.JSON.parsefile(joinpath(smilei, "observables.json"))["performance"]
        @test th["particlemesh"]["ns_per_particle_step"] == ob["ns_per_particle_step"]
        @test th["smilei"]["ns_per_particle_step"] ≈ 1e9 * sm["wall_seconds"] / (sm["mobile_particles"] * sm["steps"])
        @test th["ratio_particlemesh_median_over_smilei"] ≈ PMEns.median(ob["ns_per_particle_step"]) / th["smilei"]["ns_per_particle_step"]
        @test ob["particles_per_seed"] == sm["mobile_particles"] == 128^2 * 32^2
        @test all(v -> v["El_re"].bitwise && all(o -> o["equal"], values(v["observables"])), values(ck["wp023_seeds_rerun"]))
        @test all(o -> o["equal"], values(ck["smilei_seed_rerun"]["observables"]))
        # the rerun check compares with WP-023's run, not with itself: a changed series shows
        d = joinpath(tmp, "changed")
        mkpath(d)
        for f in ("observables.json", "case.toml")
            cp(joinpath(run, f), joinpath(d, f))
        end
        for s in Int.(TOML.parsefile(joinpath(PMEns.WP023_RUN, "case.toml"))["numerics"]["seeds"])
            cp(joinpath(run, "seed-$s"), joinpath(d, "seed-$s"))
        end
        edit_json!(x -> (x["El_re"][end] *= 2), joinpath(d, "seed-20261031", "series.json"))
        ck2 = redirect_stdout(() -> PMEns.checks(d, smilei, joinpath(tmp, "c2")), devnull)
        @test !ck2["wp023_seeds_rerun"]["20261031"]["El_re"].bitwise && ck2["wp023_seeds_rerun"]["20262031"]["El_re"].bitwise
        fig = PMEns.summary_figure(run, joinpath(tmp, "fig"))
        @test pm_close(fig, PMEns.JSON.parsefile(joinpath(run, "compare", "three-codes-rates.json")); rtol = 1e-12)
        @test fig["particlemesh"]["per_seed_at_ensemble_peaks"] == ob["per_seed_at_ensemble_peaks"]["gammas"]
        @test all(endswith(fig[c]["run"], "-431f4192") for c in ("smilei", "picongpu"))
    end
end
