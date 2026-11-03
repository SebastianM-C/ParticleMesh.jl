# WP-031: what tolerance against linear theory the cold cases support (A7)

Work package: `trace/wp/WP-031-offset-convergence.md`; measurements: `reports/wp-031-offset-convergence.md`. A7 asks for agreement thresholds against the benchmark set, set by measurement and accepted by the owner editing A7. This file proposes; nothing binds until then.

**Question.** A test of a code's growth rate on the cold cases against linear theory: against which reference value, and within what tolerance?

## What was measured

Smilei v5.1 on the CPU box (AMD EPYC 9655): Δx at the case's Courant number, Δt at fixed Δx, the perturbation amplitude, the particles per cell and the seed varied. Each pre-registered fit is compared with the same fit, over the same samples, of the exact linear cold-fluid solution of the case's initial state (`cold_ivp`, `scripts/crosscode/theory.jl`). θ is the dispersion root; all numbers are relative to θ.

| | two-stream `growth_rate` | Weibel `growth_rate_mode` | Weibel `growth_rate_energy` |
|---|---|---|---|
| offset γ/θ − 1 at the case's resolution | −1.500 % | −1.084 % | −1.006 % |
| offset at the finest point | −1.407 % (1024 cells) | −1.067 % (512²) | −0.984 % (512²) |
| the transient (linear solution, same samples) | −0.961 % to −1.035 % (163 or 164 samples) | −1.021 % | −0.930 % |
| residual (γ − γ_lin)/θ, limit Δx, Δt → 0 | −0.446 % to −0.448 % | −0.041 % | −0.047 % |
| discretization part of the residual at the case's resolution, Smilei | −0.019 % | −0.023 % | −0.029 % |
| the same for ParticleMesh (physics breaker) | −0.028 % | about −0.03 % | about −0.03 % |
| order in Δx (Δt ∝ Δx); in Δt alone | 2; 1 | 2; about 1 | 2; 2 |

The offsets are not discretization error. They are two things:

- the transient of the other roots in the window, which the linear solution gives exactly;
- a residual at a fixed fraction of saturation, which grows with the window's upper end and does not depend on the resolution, the particles per cell or the seed (two-stream: amplitude × 1/10 and × 10 also leave it within 0.04 %). This is the onset of saturation, physics of the case, not of a code.

ParticleMesh reproduces both on two-stream from 64 to 1024 cells (its residual converges to −0.447 % over Smilei's samples) and on Weibel from 64² to 256². At the cases' resolutions the discretization error of a fitted rate is at most 3 × 10⁻⁴ θ in either code.

## Proposal

**P1. Reference value: the linear solution through the run's own samples, plus the measured residual of those samples.**

- For a run of a cold case, γ_ref = γ_lin + r_NL θ.
  - γ_lin is the fit of `cold_ivp`'s solution over the samples the run's window took (`point_fits`, `scripts/crosscode/convergence.jl`).
  - r_NL is the converged residual above: two-stream −0.447 %, Weibel −0.041 % (mode) and −0.047 % (energy).
- A code passes when **|γ − γ_ref| ≤ 1 × 10⁻³ θ**.
- The value is about three times the largest discretization part measured at the cases' resolutions (3 × 10⁻⁴ θ, ParticleMesh), rounded to a power of ten.
- These all lie inside it:
  - the spread of r_NL over amplitude, particles per cell and seed (±0.04 % on two-stream; within 0.02 % on Weibel's energy observable);
  - the Float32 roundoff of ParticleMesh (5 × 10⁻⁵ θ at 42 000 steps, physics breaker);
  - ParticleMesh's initial-momentum convention (#226: 10⁻⁴ θ at the case's Δt).
- A test of this kind catches a 0.1 % error in a code's growth rate, which a comparison with the root cannot (P2).

**P2, not recommended: the dispersion root with a tolerance wide enough for the offset.** Against θ alone the tolerance must cover the transient and the onset of saturation, at least 1.6 % (two-stream) and 1.2 % (Weibel) at these windows. The transient also moves by 0.07 % with one sample more or less in the window. Such a test passes a code whose rate is wrong by a per cent.

**Conditions of P1:**

- γ_lin uses the samples the run's window took, as WP-017's P1 asks for the two-code comparison.
- r_NL is a constant of those samples, not of the case. ParticleMesh's own two-stream window keeps 164 samples where Smilei's keeps 163, and its limit is −0.453 % instead of −0.446 %: 7 × 10⁻⁵ θ, inside the tolerance, but r_NL is to be quoted with the sample count.
- A change of a case, its cadence or its window rule needs a new scan (one run per point; the cases are cold and quiet-loaded).

## Not proposed

- A value for the 1D Landau case (#167). It is noise-limited at 8 % to 10 % per ensemble, so the scan cannot resolve discretization effects there (report, §4).
- A change of the cases' registered windows. A window further from saturation shrinks r_NL (−0.22 % at [10⁻³, 10⁻²] on two-stream, −0.02 % at [10⁻⁴, 3.3 × 10⁻³] on Weibel) but enlarges the transient. With γ_lin computed exactly the transient costs nothing, so the windows can stay.

## Open questions for the owner

1. Should A7's agreement with linear theory be read as P1, a reference value that includes the transient and the measured onset of saturation, rather than as agreement with the dispersion root?
2. May r_NL, measured with Smilei and reproduced by ParticleMesh, enter a reference value at all? The alternative is a one-sided tolerance around γ_lin: −0.6 % ≤ (γ − γ_lin)/θ ≤ +0.1 % on two-stream, −0.2 % to +0.1 % on Weibel.

Assumption taken here: none accepted; the numbers are proposals.

Affected: A5, A7; the cold cases `two-stream.toml` and `weibel-2d.toml`; WP-012, WP-017, WP-023.
