# WP-017: proposed A7 agreement thresholds from the measured Smilei–PIConGPU spread

Work package: `trace/wp/WP-017-crosscode-picongpu.md`, deliverable 6. A7 bases the agreement thresholds of the benchmark set on the measured spread between Smilei and PIConGPU for equivalent inputs, and the owner accepts a value by editing A7. This file proposes values; until the owner accepts them, nothing binds. The entries `["T-214".wp017]` and `["T-215".wp017]` in `lib/ParticleMeshElectromagnetic/test/tolerances.toml` carry the same numbers with `status = "proposed"`; no test asserts them. The measurements are in `reports/crosscode-picongpu/README.md`.

## What was measured

PIConGPU 0.8.0 against Smilei v5.1, FP64 in both, on the matched cases (A5's matching list, `reports/crosscode-picongpu/README.md`, "Matching"):

| Observable | Smilei | PIConGPU | PIConGPU − Smilei | Statistical scale |
|---|---|---|---|---|
| two-stream `growth_rate`, 128 cells | 0.3456117 | 0.3456123 | +5.9 × 10⁻⁷ (1.7 × 10⁻⁶ relative) | none from seeds (cold, regular); fit residual 0.0009 |
| two-stream `growth_rate`, 256 cells | 0.3459258 | 0.3459258 | +4.4 × 10⁻⁸ (1.3 × 10⁻⁷) | as above |
| two-stream γ from \|Ê₁\|², 128 / 256 cells | 0.3369876 / 0.3370343 | 0.3369876 / 0.3370343 | 6.9 × 10⁻⁸ / −2.8 × 10⁻⁹ | — |
| Landau `damping_rate_ensemble` (8 seeds, quasi-1D in PIConGPU) | 0.1303 ± 0.0144 | 0.1360 ± 0.0131 | +0.0057: 0.29 combined jackknife errors | seed spread 0.027 (Smilei), 0.041 (PIConGPU) |
| oblique 2D `damping_rate_mode_ensemble` (4 seeds) | 0.1570 ± 0.0060 | 0.1405 ± 0.0096 | −0.0165: 1.46 combined jackknife errors | seed spread 0.013 (Smilei), 0.007 (PIConGPU) |
| oblique 2D `damping_rate_mode`, mean over 4 seeds | 0.1465 ± 0.0267 (std) | 0.1522 ± 0.0208 | +0.0057: 0.33 combined errors of the means | — |

PIConGPU FP32 against FP64 (the precision effect, A5): two-stream 1 × 10⁻⁶ to 3 × 10⁻⁶ relative in γ; Landau FP32_LANDAU_LINE.

Two kinds of observable, two kinds of spread:

- **Deterministic** (cold, regular loading: two-stream). The two codes agree to 10⁻⁶ relative, about a thousandth of the fit's own residual error. That is the spread of two implementations of one scheme with one initial-condition convention, not a statistical quantity.
- **Noise-limited** (thermal: both Landau cases). The code difference is consistent with zero at the seed counts used (0.3 and 1.5 combined standard errors). The spread A7 refers to cannot be separated from the seed noise here, so the noise sets the scale.

## Proposal

**P1, deterministic observables (two-stream).** ParticleMesh agrees with each reference code within **1 × 10⁻⁵ relative** in γ, with these conditions:

- The fit is taken over the same output samples in both runs: the window of the reference run, applied to ParticleMesh's series. The pre-registered window rule is anchored on the saturation peak and jumps by one output when a curve shifts slightly; a one-sample shift moves γ by 1 × 10⁻³ in any code (`reports/crosscode-picongpu/README.md`, "ParticleMesh against both"), a hundred times this threshold.
- ParticleMesh runs with the reference codes' initial-momentum convention (#226). With its A2 half-step kick, ParticleMesh differs by −1.0 × 10⁻⁴ over a common window and fails this threshold. Without the kick it agrees to six digits.

The value is about 5 times the larger measured difference (1.7 × 10⁻⁶), rounded up to a power of ten. It is roughly the size of the effect a cell-centred against per-particle weight sampling has on U (2 × 10⁻⁴ in energy, untested), so it may need relaxing to 5 × 10⁻⁵ if ParticleMesh keeps per-particle weights.

**P2, noise-limited observables (Landau 1D and 2D).** ParticleMesh's ensemble γ agrees with each reference code's within **2 combined jackknife standard errors**, √(σ²_PM + σ²_ref), on the same case and seed count. The measured Smilei–PIConGPU differences are 0.29 and 1.46 such errors, both inside it. This is option (b) of `questions/wp-012-a5-a7-proposals.md`. The measurement shows the code-to-code spread is not resolved by these ensembles, so a threshold in units of the measured difference would be a threshold in units of noise. With the cases as registered, 2 errors are 0.039 in 1D (8 seeds each) and 0.023 in 2D (4 seeds each); a tighter threshold needs more seeds.

**P3, precision.** No threshold is proposed: the FP32–FP64 difference is a property of the code under test, not of the reference. FP32_LANDAU_P3

## Not proposed

- A threshold for `energy_ratio` (t = 0): in PIConGPU it is the generator's background field, so the two-code difference measures Smilei's post-Poisson offset, not a spread.
- A threshold for `energy_ratio_mode`: the codes differ by 2.3 errors of the means over four seeds, each about 1.6 to 1.9 errors from the Yee value. It is noise-dominated and needs more seeds before a spread can be read from it.
- A throughput factor (A7's "within a stated factor of Smilei on the same card"): the codes ran on different hardware here, and A5's same-card comparison is a non-goal of WP-017.

## Open questions for the owner

1. Whether A7's "within that spread" should, for noise-limited observables, read as a multiple of the combined statistical error (P2). The measured spread is noise at any affordable seed count, and A7's text does not say what to do then.
2. Whether ParticleMesh's cross-code runs should use the reference codes' initial-momentum convention (#226), as P1 assumes, or whether A2's half-step centring should stay and the threshold should allow for it (about 1 × 10⁻⁴ on two-stream over a common window).

Assumption taken here: none is accepted; the numbers above are proposals.

Affected: A7; WP-012, WP-017; T-214, T-215.
