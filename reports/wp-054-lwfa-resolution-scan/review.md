# Reviews of `reports/wp-054-lwfa-resolution-scan.md`

Two independent passes on 2026-11-12, each by a fresh numerics-reviewer subagent (`.claude/agents/numerics-reviewer.md`, claude-fable-5-2) that was given the files and no conclusions. Each recomputed with its own script from the run summaries and `maps.h5`, not with `scripts/lwfa/lwfa.jl compare-scan`. Neither edited anything. The commits named are commits of the pull request's branch.

## First pass, on `fb2015d`

Verdict: the data and every table reproduce (all of `scan.json` to 13 to 16 digits, the launch fits, the comparison with the Yee dispersion relation, 1455 test assertions); not fit to merge as written.

- Blocking: "ParticleMesh–Smilei does not converge" was stronger than the data. The wake distance falls at all ten outputs from λ₀/32 to λ₀/64, by 16 % to 51 %; that excludes a second-order difference and a first-order one towards zero, not a first-order part on a constant.
- Blocking: five of six extrapolated limits of the distances were orders of run maxima that sit at different outputs; per output they had no single order.
- Blocking: "most of the wake's change is the driver's place" failed a direct test. Shifting the finest lineout by the centroid lag makes the earliest outputs worse; the best rigid shift is about two thirds of the lag and leaves 1.5 % to 3.3 % (λ₀/32) and 0.5 % to 1.1 % (λ₀/48).
- Blocking: the bunch was called 3.5 to 12.9 macro-particles; those are full-density weights, carried by a few hundred light macro-particles, so Smilei's excess is several of the bunch's own particles and the inference to loading lost its premise.
- Blocking: "the distances to PIConGPU follow its launch term" is shown for the vacuum amplitude and phase only; the match with the Yee relation to 2.5 % holds for ParticleMesh and Smilei, not PIConGPU's first outputs.
- Blocking: the self "wake maximum" at t = 400 was the window's edge sample, not an extremum.
- Blocking: the scan's tests passed five wrong definitions (the smallest value over outputs for the largest, an absolute for a relative extremum distance, a different or no wake window, the edge fallback removed, the distance floor changed), and asserted the launch model's values as bands wide enough to admit a surface one cell off.
- Major: the code-to-code wake anchor differs across grids at one output, so a wake row is known to ±10 %; the summary's factors mixed two constructions; four sets of numbers were computed from formulas with no committed code; the launch model's sensitivity was not stated.
- Confirmed: the prediction of PIConGPU's launch term for λ₀/32 and λ₀/64 was committed (`a991f96`, 07:24:45Z) before PIConGPU's first compile of the scan started (07:36:46Z), and the model is unchanged since.

What followed (`16d943d`, `e2dff80`, `4f8dc38`): orders per output with count and range; no extremum at the window's edge; the four formula sets and the bunch's bins as tested keys of `scan.json`; assertions that fail each of the surviving mutants; the model's values asserted to 0.002; the claims reworded.

## Second pass, on `4f8dc38`

Verdict: the data and every table checked are right (1602 assertions; the new keys recomputed independently to 1e-8 or better; all 18 summary hashes match); eleven statements were wrong or stronger than the evidence and two definitions were not pinned by a test. Text and two assertions; no rerun.

- Two of the summary's factors used PIConGPU's distance to its own finest run over a pair PIConGPU is not in (81 to 1091 and 9.2 to 11.5 with each code against its own pairs).
- "A fourth grid would separate the two readings" does not hold at the rows' own uncertainty: a constant plus first order and a power law towards zero are 5 % apart at λ₀/96, inside the ±10 % of a wake row, and 11 % to 12 % apart at λ₀/128.
- Smilei–PIConGPU's wake at λ₀/48 already tracks ParticleMesh–Smilei from t = 700, not ParticleMesh–PIConGPU.
- λ₀/32 confirms the launch model's surface position sharply and does not test its absorber (1.72 to 1.77 over the variants, which holds all four fitted values).
- The bunch's particle count is an estimate from the lightest bins and must be called one; no run kept particles.
- The A7 section did not name its second option at a factor of one as A7's rule as it stands, by which the case fails on all three grids, nor that A7's premise of equivalent inputs is not met for Smilei–PIConGPU; its list of what no option can rest on was incomplete.
- Smaller: several universal statements with an exception (the wake maximum, the plasma phase at late outputs, PIConGPU's displacement before t = 700), a ratio read as an order from bins of 0.2, bounds and roundings.
- Unpinned: the share of the squared difference a displacement removes, and the highest bin's distance as an absolute difference.
- An observation left as a hypothesis: ParticleMesh–Smilei's third-order phase constant equals (k_Yee − 1)·2Δx to 2 % to 6 %, the phase the grid carries over Smilei's two ghost cells against the c the correction assumes.

What followed: the corrections are in the text as it stands, with the two assertions added; no number in `scan.json` changed. No third pass was run on the corrected text.

## Not checked by either pass

No simulation was re-run and no summary was regenerated from raw dumps; ParticleMesh on the CPU off λ₀/48; Float32 and the Fast mode; PIConGPU's and Smilei's sources; the figures by eye; the PIConGPU job times including compilation, which are from the session that ran them.

## What the passes agree on

- The numbers are what the committed summaries give.
- Each code's laser converges at second order and, in ParticleMesh and Smilei, follows the Yee dispersion relation; the three codes share that error, which at λ₀/48 is one to three orders of magnitude larger than the distance between codes.
- A code-to-code distance on this case says whether two implementations solve one discrete problem. It does not bound the error of the solution; only a bound across grids does.
- This is not a validation and not an A7 pass.
