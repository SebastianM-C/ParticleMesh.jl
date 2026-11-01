# Independent review: cross-code stage 1b (WP-012)

Reviewer: numerics-reviewer role (`.claude/agents/numerics-reviewer.md`), fresh subagent, model claude-opus-5-5, 2026-11-01. Paths are relative to the worktree root (`~/.julia/dev/ParticleMesh/.claude/worktrees/wp-012-seeds`, branch `worktree-wp-012-seeds`, compared with `origin/main`). This review replaces the earlier `review.md`. That review checked a ParticleMesh column, from the electrostatic reference, which no longer exists. Committed verbatim by the implementer; the answers are in `README.md`, "Review findings and answers".

**Scope.** I checked these against the sidecars and manifests in `reports/crosscode-smilei-1d/stage-1b/`:
- the "Stage 1b" section of `reports/crosscode-smilei-1d/README.md` and its header rows;
- `questions/wp-012-a5-a7-proposals.md`;
- the `["T-214".wp012]` and `["T-215".wp012]` entries of `lib/ParticleMeshElectromagnetic/test/tolerances.toml`;
- the pinned seed in `lib/ParticleMeshElectromagnetic/test/physics.jl`;
- the Review row of `reports/spike-electrostatic-1d.md`.

I read the code that produced the numbers: `scripts/crosscode/{ensemble,ensemble_report,ensemble_bias,analysis,analyze,manifest}.jl`, `scripts/em1d/{crosscode,common}.jl`, the three case files, T-180 to T-188, and the module's loader (`load_quiet!`, `initialize!`, `problem_from_case`).

What I ran, all on the CPU, writing nothing into the repository:
- `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl`: every testset passes, T-010 to T-275 including T-180 to T-188 (43 s).
- `julia +1.13 --project=lib/ParticleMeshElectromagnetic -t 8 -e 'using Pkg; Pkg.test()'`: 18168 of 18168 pass. T-214 with the pinned seed gives γ = 0.1614265290944256, bit for bit the value recorded in `tolerances.toml`.
- `julia +1.13 --project=scripts scripts/trace/build.jl --check`: passes.
- My own snippets:
  - (a) recomputed both Landau ensembles from the raw data in `scripts/runs/` (Smilei `Fields0.h5`, ParticleMesh `seed-*/series.json`) with the HEAD code, including the leave-one-out fits and why each fit stops;
  - (b) checked every manifest input hash at HEAD and at the recorded commit;
  - (c) the local growth rate and window sensitivity of all four two-stream runs;
  - (d) an independent replicate of the synthetic calibration, keeping each realization's jackknife error;
  - (e) the same calibration with the synthetic mode amplitude matched to the measured maxima;
  - (f) a rerun of ParticleMesh seed 20263101 at the production particle count to t = 2 (214 steps) at 16 and at 3 threads, compared bit for bit with the committed series.

**Engine checklist.** The diff changes no kernel and no engine or module source: under `lib/` it changes only `test/physics.jl` (the seed pin) and `test/tolerances.toml`. So charge conservation, the half-step centring, the shape order and staggering, element-type genericity and the two deposition variants are as on `main`. The module suite covers them and passes, including T-212 (Gauss law, both modes, all three element types) and the determinism cases. The stop rule's branches on data values live in post-processing, not on a gradient path.

No finding is blocking: every run number I checked is right and reproducible. One major finding concerns the uncertainty that the report and the proposals attach to the Landau numbers.

## Findings

### F1 (major): the synthetic calibration overstates the signal by about 1.9 in energy, so its spread (0.012 to 0.013, "the better single number") and the significance of the pooled deficit are overstated

- **Where:**
  - `scripts/crosscode/ensemble_bias.jl` l. 36 (`A = sqrt(o["initial_energy"])`);
  - `README.md` l. 122, l. 131, l. 148 to 159, l. 204;
  - `questions/wp-012-a5-a7-proposals.md` l. 11 and l. 14;
  - `tolerances.toml` `["T-214".wp012]`: `ensemble_stderr.synthetic_spread_*`, `rule_bias_synthetic`, `pooled_deficit`.
- **Claim:** synthetic ensembles "with the theory's γ and ω_r, the measured |Ê₁(0)|" calibrate the rule. Their spread, 0.0128 (Smilei) and 0.0121 (ParticleMesh), is "the better single number", the rule's bias is "about −1 %", and the pooled ensembles lie "2.1 spreads below" the theory.
- **Evidence:**
  - The synthetic signal is |Ê₁(0)| e^{−γt} cos(ω_r t). It puts all of the initial perturbation into the damped mode.
  - In the runs, part of the initial perturbation is the non-modal (ballistic) response, which has phase-mixed away by fit_start. Take the first maximum used (Smilei t = 4.746, |⟨Ê₁⟩|² = 1.251 × 10⁻⁷; ParticleMesh t = 4.766, 1.226 × 10⁻⁷) and extrapolate it back with the theory γ. The modal amplitude comes out as 7.32 × 10⁻⁴ and 7.27 × 10⁻⁴, or 0.73 of |Ê₁(0)|, so 0.53 to 0.54 of it in energy.
  - At the measured noise, the synthetic ensembles therefore have about 1.9 times the signal-to-noise ratio of the runs at every maximum. The SNR at the anchoring maximum is 280 to 540, so the anchor's own noise is below 1 %.
  - The report names "no early non-modal transient" as an omission, but not this consequence of it.
  - My rerun of the calibration with the amplitude matched this way (400 realizations per code, independent draws, otherwise the script's model):

    | | Smilei noise | ParticleMesh noise |
    |---|---|---|
    | bias of the rule | −2.18 % ± 0.51 % | −2.44 % ± 0.50 % |
    | spread of γ | 0.0158 | 0.0155 |
    | mean jackknife error; coverage at one | 0.0194; 0.76 | 0.0183; 0.70 |
    | measured from theory, in spreads | −1.47 | −0.89 |

    That is 0.0158 and 0.0155 against the 0.0128 and 0.0121 recorded, and −2.2 % to −2.4 % against the −0.8 % and −1.25 % recorded.
  - With these spreads:
    - the pooled ensembles give 0.1350 ± 0.0111, which is 1.7 spreads below the theory; after removing a −2.3 % rule bias, 1.3;
    - ParticleMesh − Smilei is 0.42 combined spreads, not 0.53;
    - one combined standard error of the code difference is 0.022, or 16 % of γ.
  - Even the unmatched spreads give 0.0176 (13 %), so "cannot resolve a code difference below about 10 %" (l. 204) understates it.
- **Related inconsistencies:**
  - l. 159 removes "about −1.5 % of rule bias", while l. 157 says "about −1 %".
  - The replicate range "−0.7 % to −1.8 %" (l. 157, proposals l. 11, tolerances) mixes in two calibrations at the noise of the deleted electrostatic ParticleMesh run (the "first version" and the earlier review's).
  - My replicate of the script as committed agrees with the committed numbers: −0.98 % ± 0.40 % (ParticleMesh) and −1.06 % ± 0.41 % (Smilei), spreads 0.0123 and 0.0126.
- **Resolution:**
  - Anchor the synthetic amplitude on the measured maxima (for example the first maximum used, extrapolated with the theory γ), or add that variant as a labelled sensitivity next to the current one.
  - Quote the A5 spread as about 0.012 to 0.016 depending on the amplitude model, not as 0.012 to 0.013.
  - Restate the pooled deficit as 1.7 to 2.1 spreads (1.3 to 1.8 after the rule bias), and the code-difference resolution as about 13 % to 16 % at one standard error.
  - Make the bias statements consistent, and drop the calibrations of the deleted run from the replicate range.
  - The same caveat applies to T-180's run-grid coverage case, which sets the noise relative to the initial energy (`runtests.jl` l. 288 to 299). It remains a valid coverage check of the jackknife, but at a higher SNR than the runs had.
  - None of this changes a measured γ, and none of it is a threshold.

### F2 (minor): the report and the proposals cite findings of the replaced review

- **Where:**
  - `README.md` l. 131 ("review F1"), l. 157 ("the review finds 93 % to 94 % at two"), l. 178 ("The review's post-hoc check (`review.md`)");
  - `questions/wp-012-a5-a7-proposals.md` l. 12 ("`review.md`, F5").
- **Evidence:** this file replaces that review, so the finding numbers and its numbers no longer exist. I redid the checks those sentences rest on, on the current runs (see "Verified as correct"):
  - the ParticleMesh jackknife understatement: P(se ≤ 0.0069) = 0.058, and in 12 % of synthetic realizations the leave-one-out maxima count stays constant;
  - coverage at two jackknife errors: 0.915 (ParticleMesh noise) and 0.928 (Smilei noise);
  - the two-stream local-rate and window checks.
- **Resolution:** point these sentences at this review's "Verified as correct" items and update the 93 % to 94 % figure to 92 % to 93 %.

### F3 (minor): "a tenth of the fit's residual error" holds at 256 cells only

- **Where:** `README.md` l. 178; `questions/wp-012-a5-a7-proposals.md` l. 28.
- **Evidence:** ParticleMesh − Smilei is 0.00029 at 128 cells, a third of the residual slope error 0.0009. At 256 cells it is 0.00006, 0.07 of it.
- **Resolution:** "a third (128 cells) and a fourteenth (256 cells) of the fit's residual error".

### F4 (minor): the thermal loading differs between the codes, and the text reads as if it did not

- **Where:** `README.md` l. 113 ("thermal spread along x only, the 1D edge case with no transverse fields, as Smilei runs it") and the candidate list at l. 159.
- **Evidence:**
  - ParticleMesh draws v_x Gaussian at v_th and sets u = γv, with u_y = u_z = 0 (`load_quiet!`, `problem_from_case` with `transverse_thermal = false`; T-182 checks u_y = u_z = 0).
  - Smilei's namelist sets `momentum_initialization = "maxwell-juettner"`, `temperature = [0.0025]` (`seed-*/smilei.py`). Per Smilei's documentation a single value is isotropic, so its electrons carry thermal p_y and p_z, and with them transverse currents and fields in 1D.
  - The difference is O(T) = O(2.5 × 10⁻³) in the longitudinal response, far below the ensemble resolution, but it is a setup difference between the codes.
- **Resolution:** state it in the setup paragraph, and list it with the relativistic candidates at l. 159.

### F5 (minor): with `random_seed` gone from `landau.toml`, every other consumer silently draws with seed 1

- **Where:**
  - `scripts/crosscode/cases/landau.toml` (the `random_seed` line replaced by `seeds`);
  - `lib/ParticleMeshElectromagnetic/src/problem.jl` l. 114 (`getnum(case["numerics"], "random_seed", 1)`).
- **Evidence:**
  - `problem_from_case` does not read `seeds`. Callers that pass no seed now use seed 1 instead of 20261031: T-212 (`gauss.jl`), `allocations.jl`, `determinism_cases.jl`, the harness tests, `scripts/em1d/case.jl landau` and `scripts/em1d/timing.jl`.
  - `case.jl landau` without overrides also now runs 50 000 rather than 400 000 particles per cell, with field histories every 4 rather than 16 steps.
  - No recorded number moved: in my module run T-212 gives 1.14 and 1.31 as recorded, T-217 gives 2.11 × 10⁻⁶, and T-214, T-219 and T-105 pin 20261031.
- **Resolution:** WP-012 excludes module source changes, so open a `follow-up` issue: `problem_from_case` should take `first(seeds)` when the list is present, or refuse a case that has neither key. Optionally state the change in the PR description.

### F6 (minor): the calibration manifest does not bind the run sidecars it reads

- **Where:** `stage-1b/ensemble-bias-20261101T122122/manifest.toml`.
- **Evidence:** it hashes the three scripts and names the two runs. It does not hash their `observables.json` and `ensemble-series.json`, which supply the amplitude, the noise and the time grid. Today those inputs are committed and unchanged, but a later reanalysis of either run would silently change what this record refers to.
- **Resolution:** add the SHA-256 of the input sidecars to `[inputs]`.

### F7 (minor): per-seed spread and the seed-count arithmetic

- **Where:** `README.md` l. 146; `questions/wp-012-a5-a7-proposals.md` l. 24 and l. 34.
- **Evidence:**
  - The per-seed γ are fitted at the ensemble's maxima, so they share the ensemble's stop.
  - 0.027/√8 = 0.0095 is below both the Smilei jackknife (0.0144) and the synthetic spreads (0.013 to 0.016), so the 0.027 does not carry the ensemble error by 1/√M.
  - "Halving the ensemble standard error takes about four times the seeds" assumes 1/√M. More seeds also lower the floor and admit more maxima, so that is an estimate, not a measured scaling.
- **Resolution:** add one clause to each sentence saying so.

### F8 (minor, wording): header Review row and placeholder

- **Where:** `README.md` l. 14 ("no blocking finding", scope "stage 1b and the two-stream rerun") and l. 213 (`{ANSWERS}`).
- **Resolution:** update the row to this review (no blocking finding; major F1), and fill the answers.

## Verified as correct

**The numbers.** Every number in the Stage 1b section, the proposals and the two tolerance entries matches its sidecar:
- **Landau ensemble:** γ 0.13026 and 0.13956; jackknife errors 0.0144 and 0.0069; fit residual errors 0.0049 and 0.0105; maxima 6 and 5 with their ranges; −15.1 % and −9.0 %; ω_r 1.4046 (−0.78 %) and 1.4096 (−0.43 %); floors 5.9 and 8.2 × 10⁻¹⁰; |Ê₁(0)|² 9.976 and 9.984 × 10⁻⁷.
- **Per seed:** all 16 per-seed γ, 0.1296 ± 0.0268 and 0.1373 ± 0.0217, the single-run refusals (at most two maxima on U; mode failures 20265101 and 20267101 for Smilei, 20265101 and 20266101 for ParticleMesh).
- **Code difference:** +0.0093, +7.1 %, 0.35, 0.65, 0.58, 0.53.
- **Seed scatter:** 3.1/2.9 and 4.2/2.4 × 10⁻¹⁰, ratios 2.0 and 4.9, per-seed 2.5 → 3.4 and 2.3 → 1.9 × 10⁻⁹, SNR 39 and 40.
- **Bias table:** as committed.
- **Pooled values:** 0.1352 ± 0.0088, 2.1 and 2.5.
- **Two-stream:** all four rows, +0.09 % and +0.02 % on refinement, +0.08 % and +0.02 % between the codes, the mode estimates 2.5 % to 2.6 % below.
- **Cost:** per-seed walls, sums 684 s and 4138 s, ns per particle-step, 17 s and 48 s, 62 s and 186 s, 5.5 ns at stage 1.
- **Tolerance entries:** offsets −0.0150, −0.0141, −0.0142, −0.0139.

**Reproducibility from raw data.** Both ensembles recomputed with the HEAD code are bitwise equal to the committed `ensemble-series.json`, and give the same γ, jackknife error, residual error, ω_r and floor.
- The leave-one-out maxima counts are 6, 6, 5, 5, 6, 6, 6, 5 for Smilei and 5 in all eight for ParticleMesh, as the report says.
- Both fits stop on the threshold, not on a rising maximum: the next maxima are 4.13 × 10⁻⁹ against 5.89 × 10⁻⁹ (Smilei) and 3.02 × 10⁻⁹ against 8.17 × 10⁻⁹ (ParticleMesh).
- The spacings of the maxima used are 2.22 to 2.25 and 2.19 to 2.31 (π/ω_r = 2.22), so no maximum is split.
- The Smilei two-stream rerun's observables equal the stage 1 run's in every recorded digit, while its output hashes differ, as stated.

**Pre-registration and provenance.**
- `landau.toml` at `9d86048` (committed 09:00:42Z) hashes to `295ddf42…`, the same as at HEAD and in the runs. `two-stream-256.toml` at `9d86048` hashes to `56d5d9ae…`.
- The first Smilei seed ended at 09:02:42Z after a run of about 95 s, and the two-stream runs started at 09:12:34Z and 09:12:53Z. The Smilei manifests record `9d86048`, clean, and their analyses `4d11c1d`, clean.
- The ParticleMesh runs started 4 s (128 cells) to 4.5 minutes after `63795d2`.
- All 32 input hashes of each of the three ParticleMesh manifests match both `63795d2` and HEAD. Between `63795d2` and `c67c156` only `trace/wp/WP-012-crosscode-stage1b-seeds.md` changed, which bears out the explanation of the dirty flag.
- The fit rules in `landau.toml` are unchanged since `9d86048`. The seed scatter, the calibration and the two-stream window analysis are labelled post hoc. The seed count is disclosed as budget-set.

**Estimators.**
- The seed mean of the complex coefficient is coherent: every seed has the same Ê₁(0), because positions and weights do not depend on the seed. T-182 checks that too.
- The jackknife is the standard leave-one-out formula.
- Σ|Ê_s − ⟨Ê⟩|²/(M(M−1)) is the unbiased variance of the mean. The ln 2 factor is the median of an exponential, the noise-only distribution of |⟨Ê⟩|².
- My replicate of the committed calibration model: coverage at one jackknife error 0.70 for both codes, and at two 0.915 and 0.928. P(jackknife ≤ 0.0069) at the ParticleMesh noise is 0.058, so "a few per cent" holds. The leave-one-out maxima count stays constant in 12 % to 14 % of realizations.
- The jackknife of a discontinuous stop rule is unreliable for one run, as the report says. The ParticleMesh run is such a case.

**Units and normalization (A1).**
- n = n_c gives ω_p = ω₀, k = 10 and E₁ = α/k = 10⁻³.
- |Ê₁(0)| is 9.988 × 10⁻⁴ (Smilei) and 9.992 × 10⁻⁴ (ParticleMesh).
- The phase of ⟨Ê₁⟩(0) is π/2 exactly for ParticleMesh, the sign expected from Ex = −(α/k) sin kx for electrons of charge −1, taken at the field's own stagger. For Smilei it is 1.5229 = π/2 − kΔx/2, its dual-grid Ex taken at the point index.
- The two-stream U(0) = (δn/k)² L/4 = 6.84 × 10⁻⁹ matches both codes (6.8375 and 6.8363 × 10⁻⁹).
- The bounce check holds: ω_b = √(kE₁) with q/m = 1; 0.0999 = 0.65 γ; 0.059 = 0.39 γ at |⟨Ê₁⟩| = 3.5 × 10⁻⁴; 2ω_b(0)/γ = 1.30 rad; 0.224 = 1.46 γ at α = 1/20. The √2 argument for a standing wave is correct.
- The two-stream γ₀³ root 0.3508758 recomputes by hand from the biquadratic.

**Two-stream interpretation, on the current runs.** The local growth rate (log U fitted over ±1/ω₀), relative to theory at t = 8, 10, 12, 14, 16, 18, is a decaying oscillation that is the same in all four runs to within 1 % of the theory. Smilei at 128 cells: +138 %, −52 %, +22.5 %, −5.4 %, −0.5 %, +0.4 %. Over t = 16 to 18 every run is within 0.5 % of the theory. Other windows of the same rule move γ:

| window | change of γ, all four runs |
|---|---|
| [10⁻³, 10⁻²] | −5.0 % |
| [3·10⁻³, 3·10⁻²] | +2.8 % |
| [10⁻², 3·10⁻²] | +0.3 % |
| [3·10⁻⁴, 3·10⁻²] | −1.5 % |

That supports reading the common −1.4 % to −1.5 % as the fit window sampling a transient of the other roots, common to the case. With both codes relativistic and compared with the γ₀³ value, the resolution pair shows that the offset is not a resolution effect: refinement moves γ by 0.02 % to 0.09 %.

**Determinism (A4).**
- T-182 passes.
- ParticleMesh seed 20263101 at the production 50 000 particles per cell, rerun to t = 2 (214 steps, 54 samples), reproduces U and Ê₁ of the committed series bit for bit at 16 threads and at 3 threads.
- The Philox keys (seed, stream) are distinct for the eight seeds. `load_quiet!` refuses ppc ≥ 2¹⁶, so the counter layout does not overflow at 50 000.

**A7.**
- No agreement threshold is proposed, and none is derived from the ParticleMesh–Smilei difference.
- The open question names (a) as A7's text and (b) and (c) as goal edits.
- The tolerance entries are marked proposed, and no test reads them.

**Spike Review row.** It limits the review to Smilei's two-stream 0.34561 ± 0.00090. That value is covered here; I checked the −1.50 % and −2.25 % next to it.

## Not verified

- That Smilei seeds patch p with `random_seed + p` (`src/Patch/Patch.cpp`), and that a single `temperature` value is isotropic: the Smilei source is not reachable from the sandbox, and the second point rests on Smilei's documentation.
- Bitwise thread-count independence of a full-length ParticleMesh run: I checked a 214-step prefix of one seed at two thread counts, and not a whole run.
- The cause of the common Landau deficit. Neither the package nor I computed the relativistic correction to the root or the finite-grid and quadratic-shape corrections. My amplitude-matched calibration (F1) still uses the narrowband, stationary noise model, not the PIC noise.
- The machine-load note (load average about 5) and the cause of the slower Smilei ensemble (9.2 to 11.2 ns per particle-step against 5.5 ns).
