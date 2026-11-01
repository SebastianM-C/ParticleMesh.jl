# Independent review: cross-code stage 1b (WP-012)

Reviewer: numerics-reviewer role, model claude-opus-5-5, 2026-11-01, fresh subagent. Paths are relative to the worktree root (`~/.julia/dev/ParticleMesh/.claude/worktrees/wp-012-seeds`, branch `worktree-wp-012-seeds`, compared against `origin/main`). Committed verbatim by the implementer; the answers are in `README.md`, "Review findings and answers".

**Scope.** I checked the "Stage 1b" section of `reports/crosscode-smilei-1d/README.md` and its header rows, `questions/wp-012-a5-a7-proposals.md`, and the `["T-141".wp012]` and `["T-142".wp012]` entries of `lib/ParticleMeshCore/test/tolerances.toml`. I compared every number against the sidecars and manifests in `reports/crosscode-smilei-1d/stage-1b/`, and read the code that produced them (`scripts/crosscode/{ensemble,ensemble_report,ensemble_bias,analysis,analyze,manifest}.jl`, `scripts/es1d/crosscode.jl`, the three case files, and T-180 to T-188).

What I ran, all on the CPU:
- `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl`: all testsets pass, T-010 to T-188.
- `scripts/trace/build.jl --check`: passes.
- My own snippets, which wrote nothing into the repository:
  - (a) recomputed both Landau ensembles from the raw data (Smilei `Fields0.h5` and ParticleMesh `series.json` in `scripts/runs/`);
  - (b) the leave-one-out maxima counts, and why each fit stops;
  - (c) the distribution of the jackknife standard error on the synthetic ensembles of `ensemble_bias.jl`, plus a replicate with independent random numbers;
  - (d) the two-stream growth rate against the fit window, and the local rate over time;
  - (e) a rerun of ParticleMesh seed 20263101 at 16 threads (full run) and of seed 20266101 at 1 thread (to t = 2), compared bit for bit with the committed series;
  - (f) a hash check of every manifest input and of the committed sidecar copies against `scripts/runs/`.

The pre-registration of the observable and its rules, the units, the Landau and two-stream numbers, and the determinism hold. I have no blocking finding. The major findings concern how uncertain the Landau numbers are stated to be, and one repository rule.

## Findings

### F1 (major): the ParticleMesh ensemble standard error 0.0068 is very likely understated, and the headline and derived ratios rest on it

- **Where:** `reports/crosscode-smilei-1d/README.md` l. 121 (bold "0.1407 ± 0.0068"), l. 144 ("0.66 combined standard errors"); `questions/wp-012-a5-a7-proposals.md` l. 14 ("1.9 of its jackknife standard error") and l. 28; `tolerances.toml` l. 95 (`ensemble_stderr.particlemesh = 0.0068`).
- **Claim:** ParticleMesh γ = 0.1407 ± 0.0068, with one jackknife standard error over 8 seeds.
- **Evidence:**
  - I re-ran the synthetic calibration at the ParticleMesh noise (400 realizations) and kept each realization's jackknife standard error. Their quantiles are 10 % 0.0079, median 0.0125, 90 % 0.020. P(se ≤ 0.0068) = 0.04, and 0.03 in an independent replicate.
  - Mechanism: in the ParticleMesh run every leave-one-out fit takes the same 5 maxima. In about 90 % of synthetic realizations the count changes between leave-one-out fits, and the same happens for Smilei, whose counts are 6, 6, 5, 5, 6, 6, 6, 5.
  - So the jackknife leaves out the discrete stop-rule variation for the ParticleMesh run and includes it for Smilei. That accounts for the factor of two the report notes (0.0144 against 0.0068), with no difference in the noise. The noise levels after `fit_start` are nearly equal: 2.50 × 10⁻⁹ and 2.33 × 10⁻⁹ per seed.
  - The report itself says the jackknife "is itself a noisy estimate" (l. 155), yet keeps 0.0068 in the bold headline, in the combined error and in the tolerances entry.
- **Resolution:** quote the synthetic spread (0.0126 for ParticleMesh, 0.0130 for Smilei) next to each jackknife standard error in the headline table. State that the ParticleMesh run's leave-one-out fits never change the number of maxima. Recompute or add the derived ratios with the synthetic spread: ParticleMesh − Smilei is 0.58 combined spreads (not 0.66), and ParticleMesh − theory is 1.0 (already in the report). Mark `ensemble_stderr.particlemesh` in the tolerances file as a probable underestimate.

### F2 (major): "which the measured spread covers" overstates agreement with theory, because both independent ensembles lie below it

- **Where:** `questions/wp-012-a5-a7-proposals.md` l. 14; `README.md` l. 155 ("Neither code's distance from the theory exceeds two synthetic spreads").
- **Claim:** the theory value γ = 0.15336 is covered by the measured spread.
- **Evidence:**
  - The two ensembles are independent (different codes and random number generators), and both are low: −15.1 % and −8.2 %.
  - Pooled with the synthetic spreads (0.0130, 0.0126), the mean is 0.1357 ± 0.0091, which is 1.95 spreads below theory. After removing the rule's synthetic bias (synthetic mean about 0.1507), it is 1.66 below.
  - Pooled with the jackknife errors as reported, it is 0.1388 ± 0.0062, which is 2.4 below.
  - The report's statement is true for each code taken alone. The pooled picture hints at a shared systematic that this stage does not quantify. Candidates, none computed:
    - finite-grid and quadratic-shape effects at Δx = 0.196 λ_D;
    - early trapping at ω_b/γ = 0.65 (see F7);
    - the relativistic correction (Smilei only);
    - the part of the rule's bias that the narrowband model does not capture (F4).
- **Resolution:** keep theory as the proposed expected value, but state the pooled deficit with its significance in place of "covers", and list the candidates as open. Do not use it to adjust anything (A7).

### F3 (major, repository rule, not numerics): a committed manifest contains absolute home paths with the machine user name

- **Where:** `reports/crosscode-smilei-1d/stage-1b/ensemble-bias-20261101T100023/manifest.toml` l. 10–12 (the `[inputs]` keys).
- **Evidence:** the keys are absolute paths under the home directory. `scripts/crosscode/ensemble_bias.jl` l. 208–209 passes `joinpath(@__DIR__, f)` to `write_manifest`, which stores them unchanged. The ParticleMesh manifests store repository-relative paths.
- **Resolution:** store paths relative to the repository root in `ensemble_bias.jl`, and regenerate or edit the committed manifest (the hashes stay valid). This is CLAUDE.md "Rules that are never relaxed" (A6).

### F4 (minor): the rule-bias calibration gives two non-independent numbers, and its model omits two features of the runs

- **Where:** `README.md` l. 146–155; `ensemble_bias.jl` l. 172; `tolerances.toml` l. 98.
- **Claim:** the rule's bias is −1.78 % ± 0.42 % (Smilei noise) and −1.64 % ± 0.41 % (ParticleMesh noise).
- **Evidence:**
  - Both calibrations draw realization r from `Xoshiro(20261101 + r)`, so they use the same random numbers at noise levels 7 % apart. They are one estimate, not two confirmations.
  - My replicate at the ParticleMesh noise with independent draws gives −0.72 % ± 0.40 % and coverage 0.71. That is 1.6 combined errors from the stated value, so the bias lies roughly between −0.7 % and −1.8 %.
  - The model also assumes stationary noise, while the measured per-seed noise changes between the after-`fit_start` average and the floor window: Smilei +36 % (2.50 → 3.39 × 10⁻⁹), ParticleMesh −19 % (2.33 → 1.88 × 10⁻⁹).
  - And the synthetic signal has no early non-modal transient.
  - The conclusion (the bias is small next to the spread) holds in every case.
- **Resolution:** say that the two rows share their random numbers, quote the bias as about −1 % to −2 %, or pool independent draws, and name the stationary-noise and no-transient assumptions next to the narrowband one (l. 199).

### F5 (minor): the two-stream offset attribution can be backed with data; its "noise floor" is a fit residual

- **Where:** `README.md` l. 170 ("the fit window ... is the remaining candidate, which this stage did not test"); `questions/wp-012-a5-a7-proposals.md` l. 12 ("it belongs to the case and its fit window ... not to a code", stated more firmly than the report); `tolerances.toml` l. 107 (`noise_floor_smilei.fit_stderr = 0.0009`).
- **Evidence (my post-hoc check):**
  - In all four runs the local growth rate (log U fitted over ±1/ω₀), relative to each code's own theory, is a decaying oscillation that is identical in both codes and at both resolutions. For the Smilei 128-cell run: t = 10: −52 %, 12: +22 %, 14: −5.4 %, 16: −0.5 %, 18: +0.4 %. It falls about 4× per 2/ω₀. ParticleMesh is the same to within 1 %.
  - Over t = 16 to 18 every run is within 0.5 % of its own theory.
  - Neighbouring windows of the same rule move γ: [10⁻³, 10⁻²] gives −5.0 %, [3·10⁻³, 3·10⁻²] gives +2.8 %, [10⁻², 3·10⁻²] gives +0.4 %.
  - This supports the attribution: a transient of the other roots excited by the initial perturbation, sampled by the window. It is common to the case, not to a code.
  - The 0.0009 slope error is therefore the residual of that systematic over 160 correlated points, not noise. The |Ê₁|² estimate, 3.8 % to 4.0 % below theory in both codes, is not discussed.
- **Resolution:** add this evidence as a labelled post-hoc sensitivity, or keep the proposal's wording as tentative as the report's. Rename `noise_floor_smilei.fit_stderr` to a fit-residual key, or state that the window systematic (several %) dominates it. Note the 2.5 % gap between the U and |Ê₁|² estimates.

### F6 (minor): the two-stream "seed spread 0.0" is inferred, not measured

- **Where:** `tolerances.toml` l. 107; `questions/wp-012-a5-a7-proposals.md` l. 12; `README.md` l. 170.
- **Evidence:** the rerun reproducing γ bit for bit shows that the run is deterministic, not that it is seed-independent. Seed independence follows from the namelist: `momentum_initialization = "cold"` and `position_initialization = "regular"` (I checked `smilei.py`). No second seed was run.
- **Resolution:** write "none by construction (cold, regular loading); not measured" in place of a number 0.0.

### F7 (minor): the bounce-frequency argument is right but stated more strongly than ω_b/γ = 0.65 allows

- **Where:** `README.md` l. 159.
- **Claim:** "The bounce frequency stays below γ, so the linear estimate holds over the fit."
- **Evidence:**
  - The arithmetic is correct: ω_b(0) = √(10 × 9.99 × 10⁻⁴) = 0.0999 = 0.65 γ; 0.059 = 0.39 γ at the first maximum used; 0.224 = 1.46 γ at α = 1/20.
  - But 0.65 is not ≪ 1.
  - Two stronger arguments are available. First, a standing wave is two travelling waves of amplitude E₁/2, so the trapping frequency is smaller by √2 and the check is conservative. Second, the accumulated bounce phase over the whole run is at most 2ω_b(0)/γ ≈ 1.3 rad, below one bounce (2π).
- **Resolution:** add both arguments, and soften "holds" to "the run stays short of one bounce period".

### F8 (minor): cost notes, one wrong and one unverified

- **Where:** `README.md` l. 183, l. 186.
- **Evidence:**
  - "(container start included)" is wrong. `run.sh` writes the start and end stamps inside the container around the `smilei` call (`smilei-two-stream-736a428f/run.sh`), so container start-up is excluded. The values (16.6 s and 47.7 s) are right.
  - The explanation of 684 s against 379 s ("eight start-ups and field dumps every 4 steps") is not measured. The dumps hold 65 doubles each, and the per-particle-step cost rose from 5.5 ns to 9.2–11.2 ns.
- **Resolution:** drop "container start included". Mark the explanation as unverified, or remove it.

### F9 (minor): the Smilei ensemble records no provenance for its analysis code

- **Where:** `stage-1b/smilei-landau-295ddf42/manifest.toml`; `scripts/es1d/common.jl` `run_dir` (the dirty scope).
- **Evidence:**
  - The Smilei manifest records the preparation commit `9d86048`. Its `observables.json`, however, carries fields added in `6d376ea` and `fc35be5` (`seed_scatter_post_hoc`, `maxima_accepted`), and no commit or hash of `analysis.jl`, `ensemble.jl` or `ensemble_report.jl` is recorded.
  - I recomputed the Smilei ensemble from the raw HDF5 with the HEAD code and got γ, the jackknife standard error and the maxima bit for bit, so the values are right; only the record is incomplete.
  - Separately, the ParticleMesh dirty flag does not cover `scripts/crosscode/`. This is mitigated by the input hashes, all 25 of which match HEAD in all three manifests.
- **Resolution:** have `analyze.jl` record the analysis commit and the hashes of its inputs in the Smilei manifest, as the ParticleMesh manifest does.

### F10 (minor, wording): broken header row and unfilled placeholders

- **Where:** `README.md` l. 12, l. 14, l. 205.
- **Evidence:** l. 12 has a doubled backtick, a parenthetical inside a code span, and "(clean)" right after "the Landau and 256-cell manifests say dirty". `{REVIEW}` and `{ANSWERS}` are still placeholders.
- **Resolution:** fix the markdown, and fill both placeholders.

### F11 (minor): T-180 tests the 1/M floor in a regime the runs are not in

- **Where:** `scripts/crosscode/test/runtests.jl`, T-180.
- **Evidence:**
  - The synthetic grid runs to t = 50, so the floor window (t ≥ 40) holds only noise. In the runs (t_end = 25) the floor is 2.0 (Smilei) and 4.6 (ParticleMesh) times its noise-only value, and the seed-mean floor ratio is 4.4 and 3.1 where M = 8.
  - The coverage check (≥ 0.5 within one standard error) is lenient.
  - `ensemble_bias.jl` covers the real regime, but it is a script, not a test.
- **Resolution:** add a T-180 case on the runs' time grid (t ≤ 25) that checks the coverage of the jackknife there, or state in the test that the 1/M check applies to a noise-only floor.

### F12 (minor): unexplained difference in late-time noise between the codes

- **Where:** `README.md` l. 157.
- **Evidence:** the per-seed noise in the floor window is 3.39 × 10⁻⁹ for Smilei and 1.88 × 10⁻⁹ for ParticleMesh (1.8×), while the after-`fit_start` averages agree within 7 %. So Smilei's noise grows over the run and ParticleMesh's falls. The report gives the numbers without comment. The difference matters because the floor sets where the stop rule cuts.
- **Resolution:** add a sentence. Optionally plot the seed-scatter noise against time for both codes as a labelled post-hoc diagnostic.

### F13 (minor): the open question's options partly depart from A7's text

- **Where:** `questions/wp-012-a5-a7-proposals.md` l. 32.
- **Evidence:** A7 says ParticleMesh "must agree with each of them to within that spread". Options (b), a multiple such as two, and (c), the combined standard error, would therefore be edits to A7, not readings of it. The file correctly proposes no threshold.
- **Resolution:** say that (a) is the current text of A7 and that (b) and (c) would be goal edits.

## Verified as correct

- **The numbers.** Every number in the Stage 1b section, the proposals and the tolerances entries matches its sidecar:
  - Landau: γ, jackknife and fit standard errors, maxima, fit ranges, relative differences, ω_r and its error, floors, |Ê₁(0)|².
  - Per seed: the γ table and mean ± std (0.1296 ± 0.0268, 0.1382 ± 0.0217), and the failed seeds and accepted counts of both single-run rules.
  - The difference between the codes: 0.0105, +8.0 %, 0.39, 0.73, 0.66.
  - Seed scatter: 3.1/2.9 and 4.2/2.4 × 10⁻¹⁰, ratios 2.0 and 4.6, signal-to-noise 39 and 38.9.
  - Bias table: −1.78 %/−1.64 %, ±0.42/0.41 %, sd 0.0130/0.0126, mean jackknife 0.0141/0.0135, coverage 0.64/0.635, −1.78 and −1.00.
  - Two-stream: all four rows, the +0.09 %/+0.01 % resolution changes, +0.74 %/+0.66 % against the +0.76 % between the theories, and the offsets in the tolerances file.
  - Cost: per-seed walls, sums 684 s and 785 s, ns per particle-step, particles, steps, particle-steps.
- **The committed copies.** The stage-1b sidecars are byte-identical to `scripts/runs/`. The case hashes (`295ddf42…`, `56d5d9ae…`, `736a428f…`) match the files at HEAD and, for `landau.toml`, at `9d86048`. All ParticleMesh input hashes match HEAD. The last commit `d582e4d` changes no code.
- **Reproducibility.** Recomputing both ensembles from the raw data with the HEAD code reproduces γ, the jackknife standard error and the maxima exactly.
- **Pre-registration.**
  - `9d86048` (09:00:42Z) fixed the seeds, the particle count, the cadence and every rule of `damping_rate_ensemble`, and added `two-stream-256.toml`. The first Smilei seed started at 09:01:06Z, and the Smilei two-stream runs at 09:12Z; their manifests record `9d86048`, clean.
  - Between `9d86048` and HEAD the fit code changed only in docstrings and messages; the seed-correlation diagnostic was dropped and that is disclosed.
  - The seed scatter, the bias calibration and the relaxed-fit table are labelled post hoc. The T-188 fix (field-cadence times) touches only the secondary |Ê₁|² estimate for two-stream and predates the ParticleMesh runs.
  - The seed count came from the budget, not from a measured spread, and that is disclosed.
- **The estimators.** The seed mean of the complex coefficient is coherent: the phases of ⟨Ê₁⟩ agree across seeds. The fit stops on the threshold, not on a rising maximum: the next maxima are 4.1 × 10⁻⁹ against 5.9 × 10⁻⁹ for Smilei, and 2.9 × 10⁻⁹ against 7.5 × 10⁻⁹ for ParticleMesh. The spacings of the maxima match the report, so no maximum is split. The jackknife is the standard leave-one-out formula. The jackknife's coverage at two standard errors is 0.93 to 0.94 in the synthetic. Reviewer's own sensitivity (post hoc, not for the record): `fit_start` 4 changes neither γ; `floor_factor` 5 moves Smilei to 0.1279; `fit_start` 2 moves Smilei to 0.1332 and ParticleMesh to 0.1491. These moves are within the spread, unlike stage 1's 20 %.
- **Units and normalization (A1).** n = n_c gives ω_p = ω₀; k = 10; the expected E₁ = α/k = 10⁻³. The measured |Ê₁(0)| = 9.988 × 10⁻⁴ (Smilei) and 9.992 × 10⁻⁴ (ParticleMesh) agree to 0.04 %. The ParticleMesh value matches the quadratic-shape and finite-difference factor (0.9992). The two-stream U(0) agrees to 0.02 %.
- **Smilei's Ex positions.** Smilei's Ex dump sits at (j + ½)Δx with a periodic duplicate, which `mode_coefficient` drops correctly. The phase of ⟨Ê₁⟩(0) is 0.048 rad, against kΔx/2 = 0.049. This also settles stage 1's open "dual-grid position" item: the offset is half a cell.
- **Determinism (A4).** My 16-thread rerun of ParticleMesh seed 20263101 reproduces U and Ê₁ of the committed run bit for bit over all 671 samples. A 1-thread run of seed 20266101 to t = 2 matches the committed series bit for bit, so the result is independent of the thread count over that prefix. T-182 passes. The Smilei two-stream rerun gives the same γ to all digits; its output file hashes differ, so "bit for bit" refers to γ.
- **A7.** No agreement threshold is proposed or derived from the ParticleMesh–Smilei difference, and the A5/A7 split is kept. The tolerance entries are valid TOML, are marked proposed, and are asserted by no test.
- **Tests.** `runtests.jl` is green, including T-180 to T-188, and `build.jl --check` passes. Issue #145 exists and matches its description.

## Not verified

- That Smilei seeds patch p with `random_seed + p` (`src/Patch/Patch.cpp`): the Smilei source is not reachable from the sandbox. Indirect support: both codes show similar mode-1 noise, which identical per-patch streams would have suppressed.
- That a first ParticleMesh Landau run at `6d376ea` gave the same series for all 8 seeds: that run directory is not present. I verified one seed fully and one seed's prefix against the committed run instead.
- That the uncommitted edit behind the dirty flag was only `tolerances.toml`. It is consistent with the flag turning dirty between the 128-cell and 256-cell runs, and with all input hashes matching HEAD, but it is not recorded.
- The machine-load note (load average about 10) and the cause of Smilei's slower ensemble.
- The size of the relativistic correction to the Landau root, and any finite-grid correction to γ: neither was computed by the package or by me.
