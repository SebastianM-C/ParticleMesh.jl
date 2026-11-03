# Tests of ParticleMesh's oblique Landau ensemble driver (WP-032, scripts/em2d/landau_ensemble.jl):
# seeds run in separate requests and merged give the ensemble of the same seeds run in one
# process, bit for bit, and the merge refuses an incomplete, repeated or mixed set of requests.
# Included by runtests.jl after convergence.jl.

# The driver includes the electromagnetic module and the cross-code modules: a module of its own.
module PMEns
include(joinpath(dirname(dirname(@__DIR__)), "em2d", "landau_ensemble.jl"))
end

@testset "T-460 seeds run in separate requests merge into the ensemble of one run, and the merge refuses an incomplete or mixed set [A4, A5, WP-032]" begin
    mktempdir() do tmp
        # the case with its first four seeds, at 32 × 32 cells and 4 × 4 electrons per cell, with a
        # 10 % perturbation so that the ensemble fit finds its maxima above the noise of 16384 electrons
        src = TOML.parsefile(joinpath(PMEns.CASES, "landau-oblique-2d.toml"))
        seeds = Int.(src["numerics"]["seeds"][1:4])
        src["numerics"]["seeds"] = seeds
        src["species"][1]["perturbation"]["amplitude"] = 0.1
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
        @test isequal(r["per_seed_single_run_rule_on_energy"]["gammas"], direct["per_seed_single_run_rule_on_energy"]["gammas"])
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
        @test !any(isdir, joinpath.(tmp, ["x1", "x2", "x3", "x4"]))
    end
end

@testset "T-461 the three-code comparison takes the reference ensembles of the run's case file [A5, WP-032]" begin
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
    end
end
