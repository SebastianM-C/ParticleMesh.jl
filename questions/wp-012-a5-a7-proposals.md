# WP-012: proposed A5 expected values, measured noise floors, A7 thresholds pending

Measured in WP-012 (report: `reports/crosscode-smilei-1d/README.md`, section "Stage 1b"; independent review: `reports/crosscode-smilei-1d/review.md`). The owner accepts a value by editing A5 or A7 (`docs/goals/`); until then nothing here binds, and no test asserts these numbers beyond finiteness and sign. The entries for T-141 and T-142 in `lib/ParticleMeshCore/test/tolerances.toml` (`[“T-141”.wp012]`, `[“T-142”.wp012]`) carry the same numbers with `status = "proposed"`.

## Under A5: proposed expected values

Rates are of the field amplitude, in units of ω_p, on the cross-code case files (`scripts/crosscode/cases/`).

| Case | Proposed expected value | Spread that goes with it | Derivation |
|---|---|---|---|
| Landau damping, `landau.toml` (kλ_D = 1/2, α = 1/100, 64 cells, 8 seeds × 50 000 electrons per cell, quadratic) | γ = 0.15336, ω_r = 1.41566 | one standard error of the ensemble γ: 0.014 (Smilei, jackknife over 8 seeds), 0.007 (ParticleMesh); 0.013 for the rule on synthetic ensembles at the measured noise | Linear theory, the least-damped root of the Maxwellian dispersion relation (`scripts/crosscode/theory.jl`, T-160). The spread is measured: the jackknife over seeds of the pre-registered observable `damping_rate_ensemble`. The rule alone is biased by −1.7 % ± 0.4 % at this noise (synthetic calibration, `stage-1b/ensemble-bias-20261101T100023/`), small next to the spread |
| Cold two-stream, `two-stream.toml` and `two-stream-256.toml` (k v₀ = √(3/8) ω_p, v₀ = 0.1 c) | γ = 0.35088 for a relativistic pusher, 0.35355 for a nonrelativistic one | no seed spread (cold, regular loading: the rerun of the Smilei case reproduced γ bit for bit); fit standard error 0.0009; a common offset of −1.4 % to −1.5 % from the theory of each code's equations, which halving Δx and Δt changes by less than 0.1 % in either code | Cold-fluid dispersion relation with the longitudinal mass γ₀³m (`theory.jl`, T-160). The offset is common to both codes and both resolutions, so it belongs to the case and its fit window (U between 10⁻³ and 3·10⁻² of saturation), not to a code; an expected value for a test of this case is the theory with that offset stated |

The Landau expected value is the theory, not a measurement: both codes' ensemble γ lie below it (Smilei −15 %: 1.6 of its jackknife standard error, 1.8 of the synthetic spread; ParticleMesh −8 %: 1.9 of its jackknife standard error, 1.0 of the synthetic spread), which the measured spread covers.

## Under A7: measured noise floors and the Smilei–ParticleMesh difference

The agreement thresholds of A7 rest on the measured Smilei–PIConGPU spread. PIConGPU is cross-code stage 2 (WP-017), so **every agreement threshold is pending PIConGPU, and no number is proposed here**. None is derived from ParticleMesh's own difference to Smilei, which would be circular.

Measured, for the owner and for stage 2:

| | Landau (`damping_rate_ensemble`) | Two-stream (`growth_rate`) |
|---|---|---|
| Smilei seed-to-seed spread of γ (one seed, 50 000 electrons per cell; per-seed γ at the ensemble's maxima) | 0.027 | none: cold and seed-independent |
| Smilei ensemble standard error (jackknife, 8 seeds) | 0.014 | 0.0009 (fit standard error) |
| Smilei γ | 0.1303 | 0.34561 (128 cells), 0.34593 (256 cells) |
| ParticleMesh γ | 0.1407 | 0.34816 (128 cells), 0.34821 (256 cells) |
| ParticleMesh − Smilei | +0.0105 (+8.0 %): 0.39 Smilei seed spreads, 0.66 combined standard errors | +0.74 % (128), +0.66 % (256); the theories of the two codes' equations (nonrelativistic and relativistic pusher) differ by +0.76 % |

## The open question

Which multiple of the Smilei–PIConGPU spread A7 should take as the agreement threshold, once stage 2 has measured that spread. Options: (a) one spread; (b) a stated multiple, for example two; (c) the combined standard error of the two ensembles. Assumption taken here: none; the package records the measurements above and proposes no threshold.

A second, smaller one: the seed count (8) and the electrons per seed (50 000) were set from WP-012's compute budget (one 400 000-per-cell Smilei run) before any run, because no per-seed spread existed until these runs measured it. The measured Smilei spread (0.027 per seed, 0.014 for the ensemble) says that halving the ensemble standard error takes about four times the seeds. Assumption: the 8 × 50 000 setup stands for stage 1b; a later package that needs a tighter Landau number adds seeds or a velocity quiet start (recorded as a later engine-loader package).

Affected: A5, A7; WP-012, WP-017 (stage 2 measures the spread these thresholds wait for); T-141, T-142.
