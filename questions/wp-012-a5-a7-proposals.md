# WP-012: proposed A5 expected values, measured noise floors, A7 thresholds pending

Measured in WP-012 (report: `reports/crosscode-smilei-1d/README.md`, section "Stage 1b"; independent review: `reports/crosscode-smilei-1d/review.md`). The owner accepts a value by editing A5 or A7 (`docs/goals/`); until then nothing here binds, and no test asserts these numbers beyond finiteness and sign. The entries for T-141 and T-142 in `lib/ParticleMeshCore/test/tolerances.toml` (`[“T-141”.wp012]`, `[“T-142”.wp012]`) carry the same numbers with `status = "proposed"`.

## Under A5: proposed expected values

Rates are of the field amplitude, in units of ω_p, on the cross-code case files (`scripts/crosscode/cases/`).

| Case | Proposed expected value | Spread that goes with it | Derivation |
|---|---|---|---|
| Landau damping, `landau.toml` (kλ_D = 1/2, α = 1/100, 64 cells, 8 seeds × 50 000 electrons per cell, quadratic) | γ = 0.15336, ω_r = 1.41566 | one standard error of the ensemble γ: about 0.012 to 0.013, the spread of the rule's γ on synthetic ensembles at each run's measured noise; the runs' own jackknife errors are 0.014 (Smilei) and 0.007 (ParticleMesh, probably an underestimate: its leave-one-out fits never change their number of maxima) | Linear theory, the least-damped root of the Maxwellian dispersion relation (`scripts/crosscode/theory.jl`, T-160). The spread is measured on the pre-registered observable `damping_rate_ensemble`. The rule alone is biased by about −1 % at this noise (−0.7 % to −1.8 % across independent synthetic replicates; `stage-1b/ensemble-bias-20261101T101900/`), small next to the spread |
| Cold two-stream, `two-stream.toml` and `two-stream-256.toml` (k v₀ = √(3/8) ω_p, v₀ = 0.1 c) | γ = 0.35088 for a relativistic pusher, 0.35355 for a nonrelativistic one | no seed spread by construction (cold, regular loading; not measured with a second seed); the fit's residual standard error 0.0009, dominated by the window's systematic; a common offset of −1.4 % to −1.5 % from the theory of each code's equations, which halving Δx and Δt changes by less than 0.1 % in either code | Cold-fluid dispersion relation with the longitudinal mass γ₀³m (`theory.jl`, T-160). The offset is common to both codes and both resolutions, so it does not come from the resolution; the review's post-hoc check (`reports/crosscode-smilei-1d/review.md`, F5) points to a transient sampled by the fit window, not tested here by a script of this package. An expected value for a test of this case is the theory with that offset stated |

The Landau expected value is the theory, not a measurement. Both codes' ensemble γ lie below it: Smilei −15 % (1.8 synthetic spreads), ParticleMesh −8 % (1.05). Each alone is within two spreads, but the two ensembles are independent and both are low: pooled, 0.1358 ± 0.0088, 2.0 spreads below the theory. That hints at a systematic common to both codes which WP-012 did not quantify (candidates in the report, "Against the theory"); it is recorded, and nothing is adjusted for it.

## Under A7: measured noise floors and the Smilei–ParticleMesh difference

The agreement thresholds of A7 rest on the measured Smilei–PIConGPU spread. PIConGPU is cross-code stage 2 (WP-017), so **every agreement threshold is pending PIConGPU, and no number is proposed here**. None is derived from ParticleMesh's own difference to Smilei, which would be circular.

Measured, for the owner and for stage 2:

| | Landau (`damping_rate_ensemble`) | Two-stream (`growth_rate`) |
|---|---|---|
| Smilei seed-to-seed spread of γ (one seed, 50 000 electrons per cell; per-seed γ at the ensemble's maxima) | 0.027 | none by construction (cold, regular loading); not measured |
| Smilei ensemble standard error (jackknife, 8 seeds; synthetic spread 0.013) | 0.014 | none from seeds; fit residual 0.0009 |
| Smilei γ | 0.1303 | 0.34561 (128 cells), 0.34593 (256 cells) |
| ParticleMesh γ | 0.1407 | 0.34816 (128 cells), 0.34821 (256 cells) |
| ParticleMesh − Smilei | +0.0105 (+8.0 %): 0.39 Smilei seed spreads, 0.60 combined synthetic spreads (0.66 combined jackknife errors) | +0.74 % (128), +0.66 % (256); the theories of the two codes' equations (nonrelativistic and relativistic pusher) differ by +0.76 % |

## The open question

Which multiple of the Smilei–PIConGPU spread A7 should take as the agreement threshold, once stage 2 has measured that spread. Options: (a) one spread, which is A7's current text ("to within that spread"); (b) a stated multiple, for example two; (c) the combined standard error of the two ensembles. Options (b) and (c) would be edits of A7, not readings of it. Assumption taken here: none; the package records the measurements above and proposes no threshold.

A second, smaller one: the seed count (8) and the electrons per seed (50 000) were set from WP-012's compute budget (one 400 000-per-cell Smilei run) before any run, because no per-seed spread existed until these runs measured it. The measured Smilei spread (0.027 per seed, 0.014 for the ensemble) says that halving the ensemble standard error takes about four times the seeds. Assumption: the 8 × 50 000 setup stands for stage 1b; a later package that needs a tighter Landau number adds seeds or a velocity quiet start (recorded as a later engine-loader package).

Affected: A5, A7; WP-012, WP-017 (stage 2 measures the spread these thresholds wait for); T-141, T-142.
