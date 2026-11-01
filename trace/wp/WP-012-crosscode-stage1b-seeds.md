+++
id = "WP-012"
status = "done"
goals = ["A1", "A4", "A5", "A7"]
depends_on = ["WP-003", "WP-011"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
+++
# WP-012: cross-code stage 1b: Landau over several seeds in Smilei and ParticleMesh, independent review, proposed A5 values and noise floors

This closes the A5 gap early. The 1D set has no accepted expected values, and the Landau case as written cannot compare the codes, because the fits move by 20 % with the fit choices. Seeds were chosen over a velocity quiet start because seeds are case inputs and need no engine change; a coherent ensemble average and a linear α replace any fallback to a larger amplitude. The package brings measured numbers to the owner after the independent review that CLAUDE.md requires, and it sets how reviews are recorded for every later report. It proposes A5 expected values and noise floors only: the A7 agreement thresholds rest on the measured Smilei–PIConGPU spread, so they stay pending until cross-code stage 2, and no threshold is derived from ParticleMesh's own difference to Smilei.

## Scope

`scripts/crosscode/`, `scripts/es1d/crosscode.jl`, `reports/` and `questions/`, plus "proposed" entries in `lib/ParticleMeshCore/test/tolerances.toml`. It takes up WP-003's open item: neither code reached the pre-registered Landau observable. The route is several seeds in both codes, with no engine change. Seeds are case inputs (Smilei's `random_seed`, and Philox keys on the ParticleMesh side).

α stays at 1/100 so that the run stays linear: the stated estimate puts the bounce frequency √α·ω_p at about 0.1, below γ ≈ 0.153, while α = 1/20 would put it above. The report checks that estimate. Noise is reduced by averaging the complex mode-1 amplitude of Ex over seeds, which keeps the signal coherent and divides the noise energy by the number of seeds, and the per-seed spread is reported as well. The fit rules are pre-registered in the case file before any run.

Two-stream is cold and seed-independent: it gets the ParticleMesh column through the same harness and a resolution pair that separates the common bias from the code difference. The package also repairs the two WP-003 analysis defects the spike found. A fresh numerics-reviewer subagent then reviews the numbers.

What the package proposes, and what it leaves pending, is kept apart. It proposes A5 expected values (theory with the measured ensemble spread) and reports the Smilei seed-to-seed spread as the noise floor of each observable. It reports the Smilei–ParticleMesh difference in units of that spread as a measurement. It proposes no A7 agreement threshold: A7 bases those on the Smilei–PIConGPU spread, PIConGPU is stage 2, and a threshold built from ParticleMesh's own difference to Smilei would be circular.

This package owns the `seeds` field of the case schema. The velocity quiet start (inversion sampling on a per-cell velocity lattice, case key `velocity = "quiet"`) is recorded as a later engine-loader package.

## Deliverables

1. **Case file**: `scripts/crosscode/cases/landau.toml` gains an optional `seeds = [...]` list (schema 1 stays backward compatible: when the list is absent, the single `random_seed` applies), the ensemble observable `damping_rate_ensemble` (a fit of the peaks of |⟨Ê₁⟩_seeds| with the same `fit_start` and `floor_factor` rule), and the particle count per seed. All of it is fixed in a commit before any run, and `scripts/crosscode/README.md` documents the field. The seed count is chosen from the measured per-seed spread and stated, with at least five seeds.
2. **Generator and analysis**: `scripts/crosscode/smilei.jl` writes one namelist per seed. The ensemble reader and the per-seed and ensemble fits live in `scripts/crosscode/`, with the pre-registered rules unchanged and aggregation to a mean and standard deviation per code.
3. **Analysis repairs**: the manifest's `image.id` stores the bare image id, not the runner's header line, and `measured_from_mode` refuses or flags a fit with too few points in the window instead of reporting an unusable value.
4. **The ParticleMesh column**: `scripts/es1d/crosscode.jl` gains a seed loop over the same seeds as Philox keys, in Reference Float64, through the include path WP-011 set up, writing the same observables and sidecars.
5. **Runs**: Smilei for every Landau seed and for the two-stream case on the machine's runner (the Smilei CPU family and container image of WP-003; the request and its exact command go in the PR), and ParticleMesh on the CPU. Two-stream also runs as a resolution pair (128 and 256 cells). Every run directory has its manifest, `observables.json` and the standard plot set with JSON sidecars (F3).
6. **Report**: `reports/crosscode-smilei-1d/README.md` gains a stage 1b section: γ per seed, the mean and standard deviation per code, the ensemble γ with its standard error, the Smilei seed-to-seed spread as the noise floor, the Smilei–ParticleMesh difference in units of that spread (a measurement, not a threshold), the theory values from `scripts/crosscode/theory.jl`, and the bounce-frequency check for the chosen α.
7. **Independent review**: the numerics-reviewer role (`.claude/agents/numerics-reviewer.md`) runs as a fresh subagent on the report and the sidecars. Its findings are committed as `reports/crosscode-smilei-1d/review.md` and each one is answered in the report. The report header gets a "Review" row (reviewer role, model id, date, findings file), `reports/README.md` adds that row to its header list, and the spike report's header is updated if the review covers its numbers.
8. **Proposals**: `questions/wp-012-a5-a7-proposals.md` keeps the A5/A7 split explicit. Under A5 it lists the proposed expected values (theory with the measured ensemble spread) with their derivation. Under A7 it lists the measured Smilei noise floors and the Smilei–ParticleMesh difference in units of them, and states that the agreement thresholds are pending PIConGPU (stage 2), with no number proposed. Only the A5 expected values and the noise floors go into `lib/ParticleMeshCore/test/tolerances.toml`, as `status = "proposed"` entries for T-141 and T-142.

## Non-goals

- No engine or test-support source change: no velocity quiet start and no loader option. The only ParticleMesh-side edit is the seed loop in `scripts/es1d/crosscode.jl`.
- No edit of `docs/goals/`: A5 values are proposed through the report and `questions/`, and the owner accepts them by editing the goals.
- No A7 agreement threshold proposed, and none derived from the Smilei–ParticleMesh difference; that waits for the Smilei–PIConGPU spread (stage 2).
- No threshold asserted in any test beyond finiteness and sign; the proposals are inputs for the owner.
- No change to the pre-registered fit rules after a run. Post-hoc variants appear only as labelled sensitivity. No raising of α or of the particle count per seed above what the pre-registered rule states.
- No PIConGPU, no Weibel reference run (cross-code stage 2), no 2D case, no GPU run, no performance comparison with Smilei.
- No new container image unless the pinned Smilei image cannot take a seed; in that case a question file, not a rebuild from the sandbox.

## Acceptance

Test ids T-180 to T-189.

- T-180 the ensemble fit recovers a known damping rate from synthetic multi-seed signals (a coherent damped mode plus independent noise per seed) within its reported standard error, and averaging over M seeds lowers the fitted noise floor by the expected factor in M, within statistical error [A5, A7, WP-012].
- T-181 the namelist generator writes one namelist per seed with the seed in it and otherwise identical bytes, checked by hashing each namelist with the seed line removed, on the Landau case [A5, WP-003, WP-012].
- T-182 through the case-file path, the same seed gives a bitwise identical ParticleMesh load and identical observables on rerun, and different seeds give different velocity draws (Philox stream check) [A4, WP-012].
- T-183 the seed aggregation recovers the mean and spread of synthetic per-seed rates, and it refuses a seed whose fit fails the pre-registered criteria instead of dropping it silently [A7, WP-012].
- T-184 the manifest writer stores the bare image id from a runner header line [A5, WP-012].
- T-160 and T-161 still pass, and `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl` is green on the CPU.
- Both codes yield the pre-registered ensemble Landau observable (at least three maxima above `floor_factor` × the ensemble floor after `fit_start`), or the report states which code did not and why, with the numbers.
- The report carries the Review row and the committed review findings, each answered (fixed, or argued with numbers). `questions/wp-012-a5-a7-proposals.md` lists the A5 proposals with their derivation and the A7 measurements with the thresholds marked pending PIConGPU, and the PR description points to it. `build.jl --check` passes.

## Depends on

WP-003 (the Smilei container, case files, generator, reader and the stage 1 runs) and WP-011, because this package edits `scripts/es1d/crosscode.jl`, which WP-011 rewires to the moved electrostatic reference.

## Budgets

The A7 agreement thresholds are not proposed here: they wait for the measured Smilei–PIConGPU spread (stage 2). This package measures the Smilei seed-to-seed noise floor and the Smilei–ParticleMesh difference in units of it, and proposes A5 expected values. Compute: the total Smilei cost stays within about one WP-003 reference run (2.56·10⁷ electrons; one such seed took 379 s on 8 threads of an Intel Core i9-14900K), for example 8 seeds × 50 000 ppc. The ParticleMesh side runs at about 11 ns per particle-step at 16 threads. Wall time and particle-steps per seed are recorded for both codes, with the hardware named by CPU model. No GPU, and no throughput budget (A7's is on the GPU). Tier 1.
