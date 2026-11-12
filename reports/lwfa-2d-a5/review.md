# Reviews of `reports/lwfa-2d-a5.md`

Four independent passes on 2026-11-12, each by a fresh numerics-reviewer subagent (`.claude/agents/numerics-reviewer.md`, claude-fable-5-2) that was given the files and no conclusions. Each recomputed the report's numbers with its own script from the `summary.json` files (and `maps.h5` where needed), not with `scripts/lwfa/lwfa.jl compare`. None edited anything. The commits named are commits of the pull request's branch.

## First pass, on `1080c9a` (the first runs, unmatched)

Verdict: not ready. The arithmetic reproduced (all 435 table cells, every key of `comparison.json` to 1e-13, 341 test assertions), and the findings were about what was compared.

- Blocking: the bunch-weight comparison was not like for like. PIConGPU's "weight above 20 mₑc²" held two electrons outside the bunch; with the bunch isolated by its energy group ParticleMesh was 0.23 % from PIConGPU, not 0.97 %.
- Blocking: "about seven macro-particles, the same handful" was contradicted by the histogram: the bunch occupies 36 to 40 bins, most of them lighter than half a plateau macro-particle.
- Major: ParticleMesh's 0.40 c/ω₀ centroid lag behind PIConGPU was reproduced to 0.0044 by a parameter-free model of an envelope along the curved wavefront against ParticleMesh's envelope in the axis time.
- Major: the Smilei–PIConGPU spread was Smilei's two launch offsets (1.979 rad of carrier, two cells), so it was not A7's spread "for equivalent inputs".
- Major: the wake statement depended on an unstated window and first output; the box maximum had no resolving power at the level counted (a crest sampled at 48 points per wavelength reads up to 0.214 % low).

What followed: the Smilei launch was derived from its source and corrected (sine carrier without Gouy phase, injection node two cells behind the boundary), `GaussianLaser` gained the opt-in `WavefrontTime` envelope, the bunch is isolated by its energy group, the amplitude is carrier-free, and all three codes were run again.

## Second pass, on `c9de0de` (the second runs)

Verdict: not ready as written. Every table reproduced; the one-pulse claim was confirmed as following from matched definitions (source lines checked for Smilei and PIConGPU, the namelist holding exactly the derived values), not from tuned offsets. The default `laser_field` was bit-identical to the source before the change in 200 000 random evaluations per element type.

- Blocking: "a fifth or less of the spread" was a third for the amplitude; the first runs' wake range started at 0.17 %, not 1.2 %.
- Blocking: §6 called the spread "two implementations of one scheme" while its main part, Smilei's amplitude 0.13 % to 0.22 % above the others, was unexplained with an untested launch-side candidate.
- Major: from t = 500, 51 % to 97 % of every pair's squared wake difference lay within 20 cells of the sawtooth jump at the back of the second bucket, at the edge of the two-wavelength window; shifting the window by 24 cells changed one value from 0.165 % to 0.028 %.
- Major: the pulse was compared on the axis only. The reviewer compared the 2D maps of E_y (whole map 0.070 % for ParticleMesh–PIConGPU at t = 300).
- Major: the explanation of PIConGPU's spectrum was an inference; a multiple of the spread rejects a mismatched launch only if the spread is the stored matched one.

What followed: Smilei's amplitude was derived as its Silver–Müller condition's plane-wave factor 2/(cos(kΔx/2) + cos(ω₀Δt/2)) = 1.001984 and corrected; the wake measure moved to an interior window of 1.5 λ_p; `compare` gained the map rows; T-709 checks the launched `WavefrontTime` pulse's arrival in the engine; Smilei was run again.

## Third pass, on `2ee80a5` (the third runs)

Verdict: not ready. 732 table cells reproduced to the last printed digit; the three Smilei corrections were confirmed as closed forms from cited source lines with nothing fitted; a 1D transcription of Smilei's update launches 1.001984074, equal to the formula to 1e-14.

- Blocking: a fourth launch term. Smilei's normal-incidence condition launches E_y with its transverse spectrum multiplied by (1 − k_y²/4): 0.064 % of amplitude at the focus and up to 0.32 mrad of phase. The closed form matched ParticleMesh–Smilei at 22 of 24 vacuum outputs to 0.003 % with no free parameter. The report had attributed half of it to the wrong pair.
- Blocking: "the Smilei–PIConGPU spread is no larger than ParticleMesh's distance to either code" held in 5 of 16 rows; the bunch's "1.4 %" was not its extreme; option 2 was wrongly called the form A7 accepted for the cold two-stream; one phase statement and one digit were wrong.
- Blocking: the library change's tests had not run on the merge with `main`.
- Major: PIConGPU's plasma `maps.h5` was on no branch; the review row named no findings file; the options were judged on the data they were set from.

What followed: the fourth correction (Smilei's waist √(w₀² − 1) with a₀ and the Gouy constant to match), a model of PIConGPU's launch from its source (modelled, not verified), the maps file committed with its hash, the electromagnetic suite run on the merge with GPU tests (116 794 of 116 794 on an RTX 4080 SUPER), and Smilei run a fourth time.

## Fourth pass, on `efdc9b4` (the fourth runs, the summary, §6)

Verdict: every recomputed number reproduces (the summary, the §4 tables, all 16 rows of §6.1 with counts and ratios, the before-and-after numbers of the fourth correction, the fits); not fit to be cited until the wording is corrected. No rerun needed.

- Blocking: the 0.0025 c/ω₀ centroid distance to PIConGPU was called its launch term. The fit covers amplitude and phase only; the centroid differences of the two vacuum sets are not one function of the distance from the focus, and are unexplained.
- Blocking: "0.06 % in the maps" is the all-rows value with the back 48 cells left out (0.07 % in the flank, 0.29 % in the far wing, 0.15 % with the back cells).
- Blocking: two sentences of §6.1 on the PIConGPU pairs were wrong in the plasma amplitude row and in the size of the phase excess; "the pair with equivalent inputs" should be "the pair whose launches are matched" (the loads at the ramp's foot follow different rules); "Smilei is the code apart" does not hold for the wake minimum.
- Major: the summary left out the largest ParticleMesh–Smilei electron differences (3.0 % above 1 mₑc², 2.1 % above 10) and the plasma amplitude (0.03 %); the "not shown" list was incomplete; a table row was mislabelled; the second choice overstated what its wake bound would catch; one inference was stated as fact.

What followed: those corrections are in the text as it stands; no number changed. The passes did not check: the figures by eye, PIConGPU's box-edge treatment, Smilei's and PIConGPU's determinism (no run was repeated), and convergence, for which no second resolution exists.

## What the four passes agree on

- The numbers in the report are what the committed summaries give.
- The launch corrections are derived from the codes' sources, not fitted.
- The comparison is agreement between three implementations of one scheme at one resolution. It is not a validation and not an A7 pass.
- The evidence rejects A7's sentence taken output by output for this case and does not yet support a threshold number.
