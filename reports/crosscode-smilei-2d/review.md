# Review of the Smilei 2D report (WP-019)

| | |
|---|---|
| Reviewer | numerics-reviewer role (`.claude/agents/numerics-reviewer.md`), a fresh subagent with no part in the work, Claude Code (claude-opus-5-5) |
| Reviewed | commits `b9b0cba..5587366`: the report, both run directories, the case files, `theory.jl`, `analysis.jl`, `analyze.jl`, `read.jl`, the crosscode README and the A5 proposals; raw Smilei output re-read and every fitted number reproduced |
| Answers | by the implementing session; the fixes are in `dbd00e4` (code, tests, README), the re-analysis at `b80311d` and the report revision |

Each finding as the reviewer stated it, then the answer. The severities are the reviewer's.

## Blocking

**B1. Dual-axis positions one cell off.** `read.jl` placed point i of a dual axis at (i + ½)Δ, the openPMD reading of Smilei's `position = 0.5`. Smilei's dump starts at local index `oversize`, which is x = −Δ/2 on a dual axis (`DiagnosticFields2D.cpp`, `DiagnosticFields.cpp`), so the point is at (i − ½)Δ. The run data show it:
- Gauss's law at t = 0 has a residual of 8.8 × 10⁻⁴ with the old positions and 6.5 × 10⁻¹² with the corrected ones;
- ∇·B on the Weibel dumps is 1.6 × 10⁻⁵ against 1.1 × 10⁻¹⁶;
- the phases of Êx and Êy are off by k_xΔx and k_yΔy.

The pre-registered rates are unaffected, since the error is a constant phase, but the claim that each component is taken at its own Yee points was false, and T-274 and T-275 encoded the wrong convention.

*Answer: fixed.* I confirmed the Gauss residuals independently before changing anything.
- The 2D reader places dual points at gridGlobalOffset + (i − position)·gridSpacing (`read.jl`, with the source lines in its header). T-274 and T-275 follow that layout.
- `analyze.jl` records two checks that fail on real data if the convention is wrong. Gauss's law at t = 0 is now 6.5 × 10⁻¹⁰ of max |ρ| (Landau), and max |∇·B| over all outputs is 2.2 × 10⁻¹³ of max |B| (Weibel).
- The README documents the layout in "Smilei specifics" and in the reader schemas.
- After re-analysis the rates moved in the sixth digit only (Landau 0.160935 to 0.160930), and the report says so.
- The 1D reader has the same convention, but T-012 must pass unchanged under WP-019's acceptance, so it is left and filed as issue #161.

## Major

**M1. Smilei ran 3390 Landau steps, not 3391, and the "Smilei specifics" claim about the step count was false.** `pyinit.py` sets simulation_time = n·Δt from `number_of_timesteps`, and `Params.cpp` truncates simulation_time/timestep, which gives 3390 here. The science impact is nil (the last dump is at iteration 3376 either way), but the recorded steps and particle-steps were wrong.

*Answer: fixed.*
- The 2D generator now also passes simulation_time = (n + ½)Δt, which truncates to n, and T-271 checks Smilei's truncation for both cases. It also records that 3391·Δt/Δt truncates to 3390.
- `analyze.jl` takes the step count from `n_time` in `smilei.log`, and the manifests and report give 3390 steps and 5.69 × 10¹⁰ particle-steps (5.687 × 10¹⁰).
- The README's step-count bullet now states what Smilei does.
- The 1D generator is unchanged, since T-270 requires byte-identical namelists. Its two cases keep their counts (2681 and 5253 truncate to themselves, checked), and the README says the 1D path relies on that.
- The run was not repeated: no observable depends on the lost step.

**M2. The budget statement hid overruns.** In CPU time Landau used 3.8 times and Weibel 1.2 times the 3 030 thread-seconds of the 1D reference. The report named only Landau's CPU time, and its 4.6× per-step attribution to the dimension is confounded by 8 against 16 threads on a hybrid CPU.

*Answer: fixed in the report.*
- The Cost section states that both runs exceed the budget in CPU time, by how much, and that the cases were sized in particle-steps, below the bound.
- It also states that the per-step ratio mixes dimension, thread count and core type.
- The PR lists the budget as not met in CPU time.

**M3. "Maxwell–Jüttner momenta" is not what Smilei loads.** Below T = 0.1, `ParticleCreator::maxwellJuttner` samples the kinetic energy from the classical Maxwell–Boltzmann energy distribution. The measured Ukin(0) is 0.016 % above (3/2) nTL², where Maxwell–Jüttner would be 0.31 % above.

*Answer: fixed in the report and the README.* I checked the source lines. The case file's pre-registered `caveat` keeps its wording, and the report notes the correction. The 1D report has the same wording, which belongs to WP-003's report and is outside this package; this note records it.

## Minor

- **m1. 3.6 % should be 3.8 %** (relative to the wave amplitude, not to max |Ey| with the offset). *Fixed* in the report and the README.
- **m2. The `expectation` in `weibel-2d.toml` gives the large-|k| limit as 0.296; it is 0.2930.** *Noted in the report;* the case file stays as registered.
- **m3. The energy scalars exceed ½ΔxΔyΣF² over the distinct points by 131/128** at t = 0 of the Landau run. *Added* to the README and to the report's known limitations. Ratios are unaffected; absolute comparisons are not. The cause is not traced.
- **m4. T-273's agreement "at every |k|, v₀ and ω_p" compares two copies of one formula.** *Agreed;* the report now says it holds by construction and that the evidence is the derivation, which the reviewer re-derived term by term. The decoupled (E_e, B_z) polarization is not named in the header; left as is.
- **m5. "As expected from a single-mode noise level set by the total electron count" is not shown.** *Softened:* the report says the noise level was not compared between the runs.
- **m6. The Rho mode factor 0.9980 can be derived.** *Added:* sinc(0.975 kΔ/2) sinc³(kΔ/2) per axis gives 9.98019 × 10⁻³, equal to the measured value to the digits shown.
- **m7. `review.md` was referenced but missing, and a "see below" dangled.** *Fixed:* this file; the report cites issue #157.
- **m8. "So the cases use a cosine" read as a deliberate choice.** *Reworded:* the cosine is the schema's form.
- **m9. Provenance: nothing recorded the analysis code version.** *Fixed:* each manifest has an `analysis` entry with the commit (`b80311d`, clean tree) of the re-analysis.
- **m10. The snapshot colour range was not centred on zero.** *Fixed:* `colorrange = (−m, m)`.

## Areas the reviewer checked and found right

The reviewer found no fault in these:
- the pre-registration order (`310c2ee` before both runs, case files and `analysis.jl` unchanged afterwards);
- the fits against the registered rules and windows;
- every other number in the report and the questions file;
- the Yee ratio and the Courant step;
- the reading of the t = 0 ratio, and the k = 0 oscillation of the Ey mean at about π/ω_p;
- Smilei's `E_Add` against its source;
- the Weibel mode holding 99.99 % of the in-plane magnetic energy through the fit window;
- the orientation of the [2, 1] pattern.

Not verified by the reviewer:
- why the Ey ghost corner is empty (issue #157);
- the O(T) correction to the Landau root;
- whether WP-003's 1D runs lost a step (since checked: they did not);
- the 2D patch rule, `regular_number` and the `mean_velocity` claims against the source.
