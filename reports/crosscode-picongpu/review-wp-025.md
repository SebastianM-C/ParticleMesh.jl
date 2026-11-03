# Review of WP-025: the 32-seed oblique 2D Landau ensembles

Reviewer: numerics-reviewer (`.claude/agents/numerics-reviewer.md`), fresh subagent, claude-opus-5-5, 2026-11-03. It reviewed the branch at `16b7032` (the report, the run records, `questions/wp-017-a7-thresholds.md`, the `["T-214".wp017]` entry, the code in `scripts/crosscode/` and T-390 to T-392). The answers are in `README.md`, "Review of WP-025".

## Verdict

The numbers and the main decision hold. Every headline number was recomputed from the committed JSON, mostly to the last digit. The decision follows from the data: at 32 seeds the codes differ by −0.0047 ± 0.0046 (1.01 combined jackknife errors), so the four-seed gap was noise. Two statements drawn from the sub-ensembles did not hold (B1, M1).

## Blocking

**B1.** "At four seeds the jackknife underestimates the measured spread by 1.15 (Smilei) and 1.36 (PIConGPU)" (README, the limitations, P2). The ratios are right, but they come from one consecutive split into eight groups, and a spread from eight values is uncertain by about ±27 %. Over 300 random splits the reviewer found median ratios of 0.96 (5 % to 95 %: 0.55 to 1.60) and 0.94 (0.52 to 1.47); a bootstrap gave 0.95 and 0.94; the synthetic study at M = 4 gives 1.00 to 1.09. No underestimate is shown. Fix: drop the claim, or give the distribution.

## Major

**M1.** "The mean four-seed γ (0.1465, 0.1443) lies below the 32-seed value in both codes, so the rule's result depends on the seed count" (README, P2). The gap is mostly the luck of the split. Over random splits the reviewer found a median of −0.0017 for Smilei (5 % to 95 %: −0.0037 to +0.0003), where the consecutive split's −0.0040 is at the tail, and −0.0006 for PIConGPU. The equal-seed-count requirement is still sound, but its evidence is the synthetic bias (−0.0009 to −0.0027 and −0.0001 to −0.0016 at M = 4, ≤ 3 × 10⁻⁴ at M = 32). Fix: cite that, and state the requirement as a precaution.

## Minor

1. Common-window table: Smilei k = 3 is 0.145149 (0.1451, not 0.1452) and k = 5 is 0.150446 (0.1504, not 0.1505).
2. "0.04 % below 10 × its floor": 4.69251 × 10⁻⁹ against 4.69503 × 10⁻⁹ is 0.054 %. "About one standard deviation" holds for the noise of the two-code difference (ratio 1.1); against one code's noise it is about 1.5.
3. "About 56 seeds, about 14 h": 56 scales the M = 4 synthetic spread as 1/√M, which falls slower than that. Scaling from M = 32 gives 62 and from M = 48 gives 65 (about 16 h). The range 0.0030 to 0.0042 pools both codes; Smilei's own is 0.0034 to 0.0042, so its 0.00329 sits just below it, explained by the lower per-seed noise at 32 seeds (4.7 × 10⁻¹⁰ against 5.8 × 10⁻¹⁰).
4. The 95 % interval is the normal approximation (−0.0138 to +0.0044); with t(31) or the bootstrap spread (0.00487) it is −0.014 to +0.005.
5. The stop is structural: at 32 seeds the floor is the damped mode itself, so the 10 × floor cut lands near the sixth maximum whatever M is. Smilei's leave-one-out fits split 17 to 15 between five and six maxima; every PIConGPU one takes six, so its jackknife misses the window shift; the resampled spreads still agree with the jackknife.
6. For the owner: two 2-error comparisons fail a correct code about one run in twelve; a ParticleMesh run near theory would be about 1.6 combined errors from PIConGPU and fail about a third of the time. P2 assumes the deficit below theory is shared.
7. PIConGPU's rerun of its first four seeds keeps γ (0.14047 against 0.14045) but its jackknife moves from 0.0096 to 0.0141. The attribution of the non-bitwise reruns to the atomic deposit is untested: the driver changed and the binary was rebuilt.
8. T-392 does not check `first_maxima_fits`' jackknife or the group `mean_jackknife_stderr`; a broken leave-one-out or a missing (M − 1)/M factor would pass.

## Checked, and not checked

Checked: every number in the new section, the changed lines of the question file and the tolerances entry; both 32-seed fits and jackknifes recomputed bitwise from `seed-coefficients.json`; two seeds' coefficients recomputed bitwise from the raw dumps; the common-window fits with independent code; provenance (the case hash, `ada4769` before the seed-count study and the first run, clean manifests, six runner jobs with matching image ids); the reruns of WP-017's seeds; the crosscode suite and the trace check; the run-script skip and budget logic.

Not checked: the 54 MB figure for all PNGs; the cause of PIConGPU's non-bitwise reruns; the physics of the shared deficit below theory; nothing was rerun on a GPU.
